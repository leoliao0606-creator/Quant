"""The bar cache decides whether an experiment costs ten minutes or one second.

The important guarantees: a cached run must not open a TWS session at all, a
partial cache must fetch only what is missing, and a stored frame must come
back byte-identical - otherwise two experiments are not comparable, which is
the whole reason the cache exists.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from ibkr_ml.cache import (
    WHAT_TO_SHOW_MARKER,
    cache_age_days,
    cache_key,
    cache_kind,
    claim_cache_kind,
    drop_partial_session,
    fetch_frames,
    load_cached_frame,
    require_adjusted,
    save_cached_frame,
)


def sample_frame(rows=5, start_price=100.0):
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 09:30", periods=rows, freq="5min"),
            "open": [start_price + i for i in range(rows)],
            "high": [start_price + i + 0.5 for i in range(rows)],
            "low": [start_price + i - 0.5 for i in range(rows)],
            "close": [start_price + i + 0.2 for i in range(rows)],
            "volume": [1000.0 * (i + 1) for i in range(rows)],
        }
    )


class FakeBroker:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


class Recorder:
    """Records whether a session was opened and which symbols were downloaded."""

    def __init__(self):
        self.connect_calls = 0
        self.fetched: list[str] = []
        self.broker = FakeBroker()

    def connect(self):
        self.connect_calls += 1
        return self.broker

    def fetch_one(self, ib, symbol):
        self.fetched.append(symbol)
        return sample_frame(start_price=100.0 + len(self.fetched))


def run_fetch(recorder, symbols, cache_dir, refresh=False):
    return fetch_frames(
        symbols=symbols,
        duration="30 D",
        bar_size="5 mins",
        use_rth=True,
        max_duration_per_request=None,
        cache_dir=cache_dir,
        refresh_cache=refresh,
        connect=recorder.connect,
        fetch_one=recorder.fetch_one,
        what_to_show="TRADES",
    )


class TestCacheKey:
    def test_every_parameter_changes_the_key(self):
        base = cache_key("SPY", "30 D", "5 mins", True)
        assert base != cache_key("QQQ", "30 D", "5 mins", True)
        assert base != cache_key("SPY", "60 D", "5 mins", True)
        assert base != cache_key("SPY", "30 D", "1 hour", True)
        assert base != cache_key("SPY", "30 D", "5 mins", False)

    def test_the_key_is_filename_safe(self):
        key = cache_key("SPY", "360 D", "5 mins", True)
        assert " " not in key
        assert "/" not in key


class TestRoundTrip:
    def test_a_saved_frame_comes_back_unchanged(self, tmp_path):
        frame = sample_frame(rows=10)
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, frame,
                          what_to_show="TRADES")
        loaded, metadata = load_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True)

        pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
        assert metadata["symbol"] == "SPY"
        assert metadata["row_count"] == 10

    def test_a_missing_entry_reports_nothing(self, tmp_path):
        loaded, metadata = load_cached_frame(tmp_path, "NOPE", "30 D", "5 mins", True)
        assert loaded is None and metadata is None

    def test_different_parameters_do_not_collide(self, tmp_path):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, sample_frame(rows=3), what_to_show="TRADES")
        save_cached_frame(tmp_path, "SPY", "60 D", "5 mins", True, sample_frame(rows=7), what_to_show="TRADES")

        short, _ = load_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True)
        long, _ = load_cached_frame(tmp_path, "SPY", "60 D", "5 mins", True)
        assert len(short) == 3
        assert len(long) == 7

    def test_corrupt_metadata_does_not_hide_the_bars(self, tmp_path):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, sample_frame(), what_to_show="TRADES")
        meta_path = next(tmp_path.glob("*.json"))
        meta_path.write_text("{ not json", encoding="utf-8")

        loaded, metadata = load_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True)
        assert loaded is not None
        assert metadata == {}


class TestFetchFrames:
    def test_a_full_cache_never_opens_a_session(self, tmp_path):
        recorder = Recorder()
        run_fetch(recorder, ["SPY", "QQQ"], tmp_path)
        assert recorder.connect_calls == 1

        second = Recorder()
        frames = run_fetch(second, ["SPY", "QQQ"], tmp_path)
        assert second.connect_calls == 0
        assert second.fetched == []
        assert set(frames) == {"SPY", "QQQ"}

    def test_only_missing_symbols_are_downloaded(self, tmp_path):
        first = Recorder()
        run_fetch(first, ["SPY"], tmp_path)

        second = Recorder()
        run_fetch(second, ["SPY", "QQQ", "IWM"], tmp_path)
        assert second.fetched == ["QQQ", "IWM"]
        assert second.connect_calls == 1

    def test_refresh_redownloads_everything(self, tmp_path):
        run_fetch(Recorder(), ["SPY", "QQQ"], tmp_path)

        refreshed = Recorder()
        run_fetch(refreshed, ["SPY", "QQQ"], tmp_path, refresh=True)
        assert refreshed.fetched == ["SPY", "QQQ"]

    def test_the_session_is_closed_even_when_a_fetch_fails(self, tmp_path):
        recorder = Recorder()

        def failing_fetch(ib, symbol):
            raise RuntimeError("IBKR said no")

        with pytest.raises(RuntimeError, match="IBKR said no"):
            fetch_frames(
                symbols=["SPY"],
                duration="30 D",
                bar_size="5 mins",
                use_rth=True,
                max_duration_per_request=None,
                cache_dir=tmp_path,
                refresh_cache=False,
                connect=recorder.connect,
                fetch_one=failing_fetch,
                what_to_show="TRADES",
            )
        assert recorder.broker.disconnected

    def test_caching_can_be_switched_off(self, tmp_path):
        recorder = Recorder()
        fetch_frames(
            symbols=["SPY"],
            duration="30 D",
            bar_size="5 mins",
            use_rth=True,
            max_duration_per_request=None,
            cache_dir=None,
            refresh_cache=False,
            connect=recorder.connect,
            fetch_one=recorder.fetch_one,
            what_to_show="TRADES",
        )
        assert list(tmp_path.iterdir()) == []

    def test_the_caller_ordering_is_preserved(self, tmp_path):
        run_fetch(Recorder(), ["QQQ"], tmp_path)  # QQQ cached, others not
        frames = run_fetch(Recorder(), ["SPY", "QQQ", "IWM"], tmp_path)
        assert list(frames) == ["SPY", "QQQ", "IWM"]

    def test_a_stale_cache_is_used_and_flagged(self, tmp_path, capsys):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, sample_frame(), what_to_show="TRADES")
        meta_path = next(tmp_path.glob("*.json"))
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        metadata["fetched_at"] = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        meta_path.write_text(json.dumps(metadata), encoding="utf-8")

        recorder = Recorder()
        frames = run_fetch(recorder, ["SPY"], tmp_path)

        assert "SPY" in frames
        assert recorder.connect_calls == 0
        printed = capsys.readouterr().out
        assert "--refresh-cache" in printed


class TestCacheAge:
    def test_a_fresh_entry_is_about_zero_days_old(self):
        age = cache_age_days({"fetched_at": datetime.now(timezone.utc).isoformat()})
        assert age is not None and age < 0.01

    def test_an_old_entry_reports_its_age(self):
        stamp = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
        assert cache_age_days({"fetched_at": stamp}) == pytest.approx(5.0, abs=0.01)

    def test_missing_or_broken_metadata_reports_unknown(self):
        assert cache_age_days({}) is None
        assert cache_age_days(None) is None
        assert cache_age_days({"fetched_at": "not a date"}) is None

    def test_a_naive_timestamp_is_read_as_utc(self):
        stamp = (datetime.now(timezone.utc) - timedelta(days=2)).replace(tzinfo=None).isoformat()
        assert cache_age_days({"fetched_at": stamp}) == pytest.approx(2.0, abs=0.01)


class TestConnectivityBlips:
    """A network hiccup must not abort a download that can simply be repeated.

    IBKR reports a blip as 1100 (connectivity lost) followed seconds later by
    1102 (restored, data maintained). Treating 1100 alone as fatal killed a
    60-symbol download over a hiccup that had already fixed itself.
    """

    def error(self, *codes):
        from ibkr_ml.data import IBDataError

        return IBDataError(
            "boom", ib_errors=[{"code": c, "message": f"code {c}"} for c in codes]
        )

    def test_lost_then_restored_is_retryable(self):
        assert self.error(1100, 1102).is_retryable is True

    def test_lost_without_restore_is_not(self):
        assert self.error(1100).is_retryable is False

    def test_other_fatal_codes_are_unaffected(self):
        assert self.error(502).is_retryable is False
        assert self.error(504).is_retryable is False
        assert self.error(1300).is_retryable is False

    def test_an_ordinary_error_stays_retryable(self):
        assert self.error(162).is_retryable is True

    def test_a_session_conflict_stays_fatal_even_with_a_restore(self):
        from ibkr_ml.data import IBDataError

        error = IBDataError(
            "boom",
            ib_errors=[
                {"code": 162, "message": "already connected from a different IP address"},
                {"code": 1102, "message": "restored"},
            ],
        )
        assert error.is_retryable is False


class TestWhatToShowMarker:
    """A cache of traded prices has no dividends in it.

    Reading one by mistake reversed a cross-asset conclusion in this project
    once: AGG loses about 2.8 points a year without its distributions and GLD
    loses nothing, so the error does not shift results evenly, it favours
    whatever pays least. The check therefore stops the run.
    """

    def mark(self, tmp_path, kind):
        (tmp_path / WHAT_TO_SHOW_MARKER).write_text(kind + "\n")
        return tmp_path

    def test_the_kind_is_read_back_without_the_newline(self, tmp_path):
        self.mark(tmp_path, "ADJUSTED_LAST")
        assert cache_kind(tmp_path) == "ADJUSTED_LAST"

    def test_an_unmarked_directory_reads_as_empty(self, tmp_path):
        assert cache_kind(tmp_path) == ""

    def test_an_adjusted_directory_passes(self, tmp_path):
        require_adjusted(self.mark(tmp_path, "ADJUSTED_LAST"))

    def test_a_traded_price_directory_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            require_adjusted(self.mark(tmp_path, "TRADES"))
        assert "TRADES" in str(caught.value)

    def test_an_unmarked_directory_is_refused(self, tmp_path):
        # An old cache predates the marker, and its kind cannot be recovered
        # from the bars, so "unknown" has to fail rather than be assumed good.
        with pytest.raises(SystemExit) as caught:
            require_adjusted(tmp_path)
        assert WHAT_TO_SHOW_MARKER in str(caught.value)

    def test_the_purpose_is_named_in_the_message(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            require_adjusted(self.mark(tmp_path, "TRADES"), "跨资产趋势回测")
        assert "跨资产趋势回测" in str(caught.value)

    def test_a_string_path_works_as_well_as_a_path_object(self, tmp_path):
        require_adjusted(str(self.mark(tmp_path, "ADJUSTED_LAST")))


class TestClaimCacheKind:
    """The marker has to be written where the bars are, not beside them.

    fetch_assets.py wrote the marker and refused to mix two kinds, but only
    for downloads that went through fetch_assets.py. train_model.py does not:
    it calls fetch_frames, which calls save_cached_frame directly, and it
    passed no what_to_show at all, taking fetch_historical_frame's TRADES
    default into whatever --cache-dir named. So `train_model.py --cache-dir
    data_cache_adj` put unadjusted bars in the adjusted directory, left the
    marker saying ADJUSTED_LAST, and every later require_adjusted passed.
    """

    def test_an_unmarked_directory_gets_marked(self, tmp_path):
        claim_cache_kind(tmp_path, "ADJUSTED_LAST")
        assert cache_kind(tmp_path) == "ADJUSTED_LAST"

    def test_claiming_the_same_kind_again_is_fine(self, tmp_path):
        claim_cache_kind(tmp_path, "TRADES")
        claim_cache_kind(tmp_path, "TRADES")
        assert cache_kind(tmp_path) == "TRADES"

    def test_a_second_kind_is_refused(self, tmp_path):
        claim_cache_kind(tmp_path, "ADJUSTED_LAST")
        with pytest.raises(SystemExit) as caught:
            claim_cache_kind(tmp_path, "TRADES")
        assert "ADJUSTED_LAST" in str(caught.value)
        assert "TRADES" in str(caught.value)

    def test_the_refusal_leaves_the_marker_alone(self, tmp_path):
        claim_cache_kind(tmp_path, "ADJUSTED_LAST")
        with pytest.raises(SystemExit):
            claim_cache_kind(tmp_path, "TRADES")
        assert cache_kind(tmp_path) == "ADJUSTED_LAST"

    def test_an_empty_kind_is_refused(self, tmp_path):
        with pytest.raises(SystemExit):
            claim_cache_kind(tmp_path, "")
        assert cache_kind(tmp_path) == ""


class TestSaveDeclaresItsKind:
    def test_saving_marks_the_directory(self, tmp_path):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True,
                          sample_frame(), what_to_show="ADJUSTED_LAST")
        assert cache_kind(tmp_path) == "ADJUSTED_LAST"
        require_adjusted(tmp_path)          # now passes on its own evidence

    def test_the_kind_is_recorded_in_the_sidecar_too(self, tmp_path):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True,
                          sample_frame(), what_to_show="ADJUSTED_LAST")
        _, metadata = load_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True)
        assert metadata["what_to_show"] == "ADJUSTED_LAST"

    def test_traded_bars_cannot_be_written_into_an_adjusted_directory(self, tmp_path):
        """Now refused at the write, whichever caller got here."""
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True,
                          sample_frame(), what_to_show="ADJUSTED_LAST")
        with pytest.raises(SystemExit) as caught:
            save_cached_frame(tmp_path, "AGG", "30 D", "5 mins", True,
                              sample_frame(), what_to_show="TRADES")
        assert "2.8" in str(caught.value)

    def test_the_refused_write_leaves_no_bars_behind(self, tmp_path):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True,
                          sample_frame(), what_to_show="ADJUSTED_LAST")
        with pytest.raises(SystemExit):
            save_cached_frame(tmp_path, "AGG", "30 D", "5 mins", True,
                              sample_frame(), what_to_show="TRADES")
        frame, _ = load_cached_frame(tmp_path, "AGG", "30 D", "5 mins", True)
        assert frame is None

    def test_a_missing_kind_is_a_type_error_not_a_silent_default(self):
        """A default here would be the silent TRADES that caused this."""
        with pytest.raises(TypeError):
            save_cached_frame("unused", "SPY", "30 D", "5 mins", True, None)


class TestFetchFramesDeclaresItsKind:
    def test_fetched_bars_mark_the_directory(self, tmp_path):
        recorder = Recorder()
        fetch_frames(symbols=["SPY"], duration="30 D", bar_size="5 mins",
                     use_rth=True, max_duration_per_request=None,
                     cache_dir=tmp_path, refresh_cache=False,
                     connect=recorder.connect, fetch_one=recorder.fetch_one,
                     what_to_show="ADJUSTED_LAST")
        assert cache_kind(tmp_path) == "ADJUSTED_LAST"

    def test_the_train_model_path_cannot_pollute_an_adjusted_cache(self, tmp_path):
        """train_model.py:283's call, with the TRADES default it used to take."""
        claim_cache_kind(tmp_path, "ADJUSTED_LAST")
        recorder = Recorder()
        with pytest.raises(SystemExit) as caught:
            fetch_frames(symbols=["SPY"], duration="30 D", bar_size="5 mins",
                         use_rth=True, max_duration_per_request=None,
                         cache_dir=tmp_path, refresh_cache=False,
                         connect=recorder.connect,
                         fetch_one=recorder.fetch_one, what_to_show="TRADES")
        assert "ADJUSTED_LAST" in str(caught.value)

    def test_a_missing_kind_is_a_type_error(self, tmp_path):
        recorder = Recorder()
        with pytest.raises(TypeError):
            fetch_frames(symbols=["SPY"], duration="30 D", bar_size="5 mins",
                         use_rth=True, max_duration_per_request=None,
                         cache_dir=tmp_path, refresh_cache=False,
                         connect=recorder.connect, fetch_one=recorder.fetch_one)


def daily_frame(dates, volume=25_000_000.0):
    return pd.DataFrame({
        "timestamp": [pd.Timestamp(d, tz="America/New_York") for d in dates],
        "open": [100.0] * len(dates), "high": [101.0] * len(dates),
        "low": [99.0] * len(dates), "close": [100.5] * len(dates),
        "volume": [volume] * len(dates),
    })


NOON = datetime(2026, 9, 11, 12, 49, tzinfo=ZoneInfo("America/New_York"))
AFTER_CLOSE = datetime(2026, 9, 11, 16, 49, tzinfo=ZoneInfo("America/New_York"))


class TestDropPartialSession:
    """A bar for a session still trading is half a day recorded as a whole one.

    Measured on this repo's own cache: data_cache_adj was fetched at 12:49
    Eastern on 2026-09-11 and SPY's bar for that day carried 12.4M shares
    against 21.6M-28.8M on the five days before, range 2.78 against 3.47-6.58.
    Ten scripts read that cache and none of them dropped it.
    """

    def test_a_bar_for_a_session_still_trading_is_dropped(self):
        frame = daily_frame(["2026-09-10", "2026-09-11"])
        kept, dropped = drop_partial_session(frame, "1 day", NOON)
        assert dropped is True and len(kept) == 1

    def test_the_same_bar_after_the_close_is_kept(self):
        frame = daily_frame(["2026-09-10", "2026-09-11"])
        kept, dropped = drop_partial_session(frame, "1 day", AFTER_CLOSE)
        assert dropped is False and len(kept) == 2

    def test_a_bar_from_a_previous_session_is_always_kept(self):
        frame = daily_frame(["2026-09-09", "2026-09-10"])
        kept, dropped = drop_partial_session(frame, "1 day", NOON)
        assert dropped is False and len(kept) == 2

    def test_a_bar_stamped_in_the_future_is_dropped(self):
        frame = daily_frame(["2026-09-11", "2026-09-14"])
        kept, dropped = drop_partial_session(frame, "1 day", NOON)
        assert dropped is True and len(kept) == 1

    def test_intraday_bars_are_left_alone(self):
        """A five-minute bar has its own notion of partial, and no backtest
        in this project reads one."""
        frame = daily_frame(["2026-09-10", "2026-09-11"])
        kept, dropped = drop_partial_session(frame, "5 mins", NOON)
        assert dropped is False and len(kept) == 2

    def test_an_empty_frame_is_returned_unchanged(self):
        frame = daily_frame([])
        kept, dropped = drop_partial_session(frame, "1 day", NOON)
        assert dropped is False and len(kept) == 0

    def test_saving_at_noon_stores_only_finished_sessions(self, tmp_path):
        """The end-to-end case: what fetch_assets.py did on 2026-09-11."""
        frame = daily_frame(["2026-09-10", "2026-09-11"])
        import ibkr_ml.cache as cache_module

        real = cache_module.datetime

        class FrozenNoon(real):
            @classmethod
            def now(cls, tz=None):
                return NOON

        cache_module.datetime = FrozenNoon
        try:
            save_cached_frame(tmp_path, "SPY", "20 Y", "1 day", True, frame,
                              what_to_show="ADJUSTED_LAST")
        finally:
            cache_module.datetime = real
        stored, metadata = load_cached_frame(tmp_path, "SPY", "20 Y", "1 day", True)
        assert len(stored) == 1
        assert metadata["row_count"] == 1
        assert str(metadata["last_timestamp"])[:10] == "2026-09-10"
