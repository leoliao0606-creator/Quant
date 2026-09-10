"""The bar cache decides whether an experiment costs ten minutes or one second.

The important guarantees: a cached run must not open a TWS session at all, a
partial cache must fetch only what is missing, and a stored frame must come
back byte-identical - otherwise two experiments are not comparable, which is
the whole reason the cache exists.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from ibkr_ml.cache import (
    cache_age_days,
    cache_key,
    fetch_frames,
    load_cached_frame,
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
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, frame)
        loaded, metadata = load_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True)

        pd.testing.assert_frame_equal(loaded, frame, check_dtype=False)
        assert metadata["symbol"] == "SPY"
        assert metadata["row_count"] == 10

    def test_a_missing_entry_reports_nothing(self, tmp_path):
        loaded, metadata = load_cached_frame(tmp_path, "NOPE", "30 D", "5 mins", True)
        assert loaded is None and metadata is None

    def test_different_parameters_do_not_collide(self, tmp_path):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, sample_frame(rows=3))
        save_cached_frame(tmp_path, "SPY", "60 D", "5 mins", True, sample_frame(rows=7))

        short, _ = load_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True)
        long, _ = load_cached_frame(tmp_path, "SPY", "60 D", "5 mins", True)
        assert len(short) == 3
        assert len(long) == 7

    def test_corrupt_metadata_does_not_hide_the_bars(self, tmp_path):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, sample_frame())
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
        )
        assert list(tmp_path.iterdir()) == []

    def test_the_caller_ordering_is_preserved(self, tmp_path):
        run_fetch(Recorder(), ["QQQ"], tmp_path)  # QQQ cached, others not
        frames = run_fetch(Recorder(), ["SPY", "QQQ", "IWM"], tmp_path)
        assert list(frames) == ["SPY", "QQQ", "IWM"]

    def test_a_stale_cache_is_used_and_flagged(self, tmp_path, capsys):
        save_cached_frame(tmp_path, "SPY", "30 D", "5 mins", True, sample_frame())
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
