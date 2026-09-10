"""The log analyzer's job is to name a cause, so the naming is what is tested.

Each test writes a log that only a real run could have produced and asserts on
the tag the diagnosis picks, since that tag is the whole point of the tool.
"""

from __future__ import annotations

import json

import pytest

from analyze_logs import LogSummary, _diagnose, _load_events, _percentile


def write_log(tmp_path, events, name="paper_trade_2026-06-05.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return path


def cycle_event(decisions, phase="trading", equity=100000.0):
    return {
        "timestamp": "2026-06-05T10:00:00-04:00",
        "event_type": "cycle",
        "payload": {"phase": phase, "equity": equity, "decisions": decisions},
    }


def blocked_event(symbol, age_minutes, reason="stale_data", stale_after=15, zone="America/New_York"):
    return {
        "timestamp": "2026-06-05T10:00:00-04:00",
        "event_type": "bar_blocked",
        "payload": {
            "symbol": symbol,
            "reason": reason,
            "age_minutes": age_minutes,
            "stale_after_minutes": stale_after,
            "bar_timezone": zone,
        },
    }


def tags(diagnosis):
    return {line.split("]")[0] + "]" for line in diagnosis}


def summarize(tmp_path, events):
    path = write_log(tmp_path, events)
    return LogSummary(path, _load_events(path))


class TestDiagnosis:
    def test_cycles_without_orders_are_reported_as_blocked(self, tmp_path):
        events = [cycle_event([{"symbol": "AAA", "action": "HOLD", "reason": "stale_data"}])]
        assert "[BLOCKED]" in tags(_diagnose(summarize(tmp_path, events)))

    def test_a_consistently_late_feed_is_named_as_delayed(self, tmp_path):
        events = [blocked_event("AAA", age) for age in (18.0, 19.5, 20.1, 21.0)]
        diagnosis = _diagnose(summarize(tmp_path, events))
        assert "[DELAYED FEED]" in tags(diagnosis)

    def test_future_timestamps_are_named_as_a_timezone_fault(self, tmp_path):
        events = [blocked_event("AAA", age, reason="future_bar_timestamp") for age in (-240.0, -239.0)]
        diagnosis = _diagnose(summarize(tmp_path, events))
        assert "[TIMEZONE]" in tags(diagnosis)

    def test_a_very_old_bar_is_not_blamed_on_the_delayed_feed(self, tmp_path):
        events = [blocked_event("AAA", age) for age in (300.0, 320.0)]
        assert "[STALE]" in tags(_diagnose(summarize(tmp_path, events)))

    def test_fresh_bars_blocked_only_as_same_bar_point_at_polling(self, tmp_path):
        events = [blocked_event("AAA", 2.0, reason="same_bar") for _ in range(3)]
        diagnosis = tags(_diagnose(summarize(tmp_path, events)))
        assert "[FRESH]" in diagnosis
        assert "[POLLING]" in diagnosis

    def test_a_session_conflict_is_called_out(self, tmp_path):
        events = [
            {
                "timestamp": "2026-06-05T10:00:00-04:00",
                "event_type": "symbol_error",
                "payload": {
                    "symbol": "SPY",
                    "error_type": "IBDataError",
                    "error": "code 162: already connected from a different IP address",
                },
            }
        ]
        assert "[SESSION]" in tags(_diagnose(summarize(tmp_path, events)))

    def test_timeouts_are_called_out(self, tmp_path):
        events = [
            {
                "timestamp": "2026-06-05T10:00:00-04:00",
                "event_type": "cycle_error",
                "payload": {"error_type": "TimeoutError", "error": ""},
            }
        ]
        assert "[TIMEOUT]" in tags(_diagnose(summarize(tmp_path, events)))

    def test_a_run_that_traded_is_reported_as_traded(self, tmp_path):
        events = [
            cycle_event([{"symbol": "AAA", "action": "BUY", "reason": "model_entry"}]),
            {
                "timestamp": "2026-06-05T10:00:00-04:00",
                "event_type": "order_submitted",
                "payload": {"symbol": "AAA", "action": "BUY", "quantity_delta": 100, "reason": "model_entry"},
            },
        ]
        diagnosis = tags(_diagnose(summarize(tmp_path, events)))
        assert "[TRADED]" in diagnosis
        assert "[BLOCKED]" not in diagnosis

    def test_a_log_of_only_skips_is_reported_as_idle(self, tmp_path):
        events = [
            {
                "timestamp": "2026-06-05T18:00:00-04:00",
                "event_type": "cycle_skipped",
                "payload": {"reason": "after_close_buffer", "equity": 100000.0},
            }
        ]
        assert "[IDLE]" in tags(_diagnose(summarize(tmp_path, events)))

    def test_old_logs_without_measured_ages_ask_for_a_rerun(self, tmp_path):
        events = [cycle_event([{"symbol": "AAA", "action": "HOLD", "reason": "stale_data"}])]
        assert "[UPGRADE]" in tags(_diagnose(summarize(tmp_path, events)))


class TestSummary:
    def test_orders_and_decisions_are_counted(self, tmp_path):
        events = [
            cycle_event(
                [
                    {"symbol": "AAA", "action": "BUY", "reason": "model_entry"},
                    {"symbol": "BBB", "action": "HOLD", "reason": "no_change"},
                ]
            ),
            {
                "timestamp": "2026-06-05T10:00:00-04:00",
                "event_type": "order_submitted",
                "payload": {"symbol": "AAA", "action": "BUY", "quantity_delta": 100, "reason": "model_entry"},
            },
        ]
        summary = summarize(tmp_path, events)
        assert summary.total_decisions == 2
        assert len(summary.order_rows) == 1
        assert summary.dominant_reason[0] in {"model_entry", "no_change"}

    def test_age_statistics_are_computed_per_symbol(self, tmp_path):
        events = [blocked_event("AAA", 10.0), blocked_event("AAA", 20.0), blocked_event("BBB", 30.0)]
        summary = summarize(tmp_path, events)
        assert summary.age_stats["median"] == pytest.approx(20.0)
        assert summary.block_ages_by_symbol["BBB"] == [30.0]

    def test_a_malformed_line_is_skipped_rather_than_fatal(self, tmp_path, capsys):
        path = tmp_path / "paper_trade_2026-06-05.jsonl"
        path.write_text('{"event_type": "cycle", "payload": {}}\nnot json at all\n', encoding="utf-8")
        events = _load_events(path)
        assert len(events) == 1
        assert "not valid JSON" in capsys.readouterr().out


class TestPercentile:
    def test_endpoints_and_midpoint(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(values, 0.0) == 1.0
        assert _percentile(values, 0.5) == 3.0
        assert _percentile(values, 1.0) == 5.0

    def test_interpolates_between_samples(self):
        assert _percentile([0.0, 10.0], 0.5) == pytest.approx(5.0)

    def test_a_single_value_is_its_own_percentile(self):
        assert _percentile([7.0], 0.9) == 7.0


class TestUntransmittedOrders:
    """An order held inside TWS has to be named, not counted as a trade."""

    def untransmitted_event(self, symbol="AAA", status="PendingSubmit"):
        return {
            "timestamp": "2026-09-09T14:31:46-04:00",
            "event_type": "order_not_transmitted",
            "payload": {
                "symbol": symbol,
                "action": "SELL",
                "quantity_delta": 132,
                "order_status": status,
                "reason": "take_profit",
            },
        }

    def test_held_orders_are_named_with_the_tws_setting(self, tmp_path):
        events = [cycle_event([{"symbol": "AAA", "action": "SELL", "reason": "take_profit"}]),
                  self.untransmitted_event()]
        diagnosis = _diagnose(summarize(tmp_path, events))
        assert "[NOT TRANSMITTED]" in tags(diagnosis)
        joined = " ".join(diagnosis)
        assert "Bypass Order Precautions" in joined
        assert "PendingSubmit" in joined

    def test_held_orders_are_not_counted_as_trades(self, tmp_path):
        events = [cycle_event([{"symbol": "AAA", "action": "SELL", "reason": "take_profit"}]),
                  self.untransmitted_event()]
        summary = summarize(tmp_path, events)
        assert summary.order_rows == []
        assert len(summary.untransmitted_rows) == 1
        assert "[TRADED]" not in tags(_diagnose(summary))

    def test_every_affected_symbol_is_listed(self, tmp_path):
        events = [self.untransmitted_event("MSFT"), self.untransmitted_event("NVDA")]
        diagnosis = " ".join(_diagnose(summarize(tmp_path, events)))
        assert "MSFT" in diagnosis and "NVDA" in diagnosis

    def test_skipped_orders_are_collected(self, tmp_path):
        events = [
            {
                "timestamp": "2026-09-09T14:31:46-04:00",
                "event_type": "order_skipped",
                "payload": {"symbol": "AAA", "reason": "entry_order_already_working"},
            }
        ]
        summary = summarize(tmp_path, events)
        assert len(summary.skipped_order_rows) == 1
