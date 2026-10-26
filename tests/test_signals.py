"""Tests for the ChatGPT report inbox and its scoring.

The clock that matters is the paste, not anything the report says about
itself, and the entry is the first open after it. A report that could enter
at a price printed before it arrived would be scored on a move it never
had to predict - the same look-ahead that produced the retracted momentum
result in this project.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibkr_ml.signals import (entry_session, extract_reports, ingest,  # noqa: E402
                             information_coefficient, load_inbox, score_inbox,
                             score_item, summarize)

EASTERN = ZoneInfo("America/New_York")


def block(items, task="daily", **extra):
    payload = {"schema": 1, "task": task, "items": items, **extra}
    return f"一些正文。\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```\n结尾。"


GOOD = {"ticker": "googl", "action": "BUY", "score": 8.7, "horizon_days": 30,
        "entry_limit": 245.0, "stop": 228.0, "target": 275.0,
        "invalidation": "云业务增速跌破 25%", "sources": ["https://example.com/a"]}


def bars(days, opens, closes, adj=None):
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in days])
    adj = closes if adj is None else adj
    return pd.DataFrame({"open": opens, "high": closes, "low": closes, "close": closes,
                         "adjclose": adj, "dividend": 0.0, "split": 1.0}, index=index)


DAYS = ["2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"]


class TestExtract:
    def test_a_fenced_block_is_found_and_normalised(self):
        reports, errors = extract_reports(block([GOOD]))
        assert errors == []
        (item,) = reports[0]["items"]
        assert item["ticker"] == "GOOGL"
        assert item["action"] == "buy"
        assert item["horizon_days"] == 30

    def test_a_bad_item_is_dropped_and_named_but_the_report_is_kept(self):
        reports, errors = extract_reports(block([GOOD, {"ticker": "NVDA", "action": "moon"}]))
        assert len(reports[0]["items"]) == 1
        assert any("第 2 条" in e for e in errors)

    def test_an_empty_radar_report_is_a_record_that_it_ran(self):
        reports, errors = extract_reports(block([], task="radar"))
        assert reports[0]["items"] == [] and errors == []

    def test_class_shares_use_yahoos_spelling(self):
        reports, _ = extract_reports(block([{**GOOD, "ticker": "brk.b"}]))
        assert reports[0]["items"][0]["ticker"] == "BRK-B"

    def test_text_without_a_block_says_so(self):
        reports, errors = extract_reports("今天没有重大变化。")
        assert reports == [] and errors

    def test_a_score_out_of_range_is_refused(self):
        reports, errors = extract_reports(block([{**GOOD, "score": 85}]))
        assert reports[0]["items"] == [] and errors

    def test_a_report_pasted_into_the_wrong_box_is_refused(self):
        reports, errors = extract_reports(block([GOOD], task="radar"), expected_task="daily")
        assert reports == []
        assert "radar" in errors[0] and "daily" in errors[0]

    def test_a_block_without_a_task_takes_the_boxs(self):
        text = "```json\n" + json.dumps({"schema": 1, "items": [GOOD]}) + "\n```"
        reports, _ = extract_reports(text, expected_task="short")
        assert reports[0]["task"] == "short"

    def test_the_richer_fields_are_kept_and_unknown_reasons_become_other(self):
        item = {**GOOD, "price_seen": 250.0, "subscores": {"成长": 9, "估值": 6},
                "reasons": ["earnings", "vibes"], "materiality": 4, "thesis_change": "weaken",
                "event": {"type": "guidance", "time_et": "2026-10-26T07:00"},
                "catalyst": {"type": "earnings", "date": "2026-10-28"}}
        (report,), errors = extract_reports(block([item]))
        kept = report["items"][0]
        assert errors == []
        assert kept["subscores"] == {"成长": 9.0, "估值": 6.0}
        assert kept["reasons"] == ["earnings", "other"]
        assert kept["materiality"] == 4 and kept["thesis_change"] == "weaken"
        assert kept["catalyst"]["date"] == "2026-10-28"

    def test_an_unknown_thesis_change_drops_the_item(self):
        reports, errors = extract_reports(block([{**GOOD, "thesis_change": "maybe"}]))
        assert reports[0]["items"] == [] and errors


class TestIngest:
    def test_the_same_block_pasted_twice_is_one_report(self, tmp_path):
        received = datetime(2026, 10, 26, 8, 55, tzinfo=EASTERN)
        first = ingest(block([GOOD]), tmp_path, received)
        second = ingest(block([GOOD]), tmp_path, received)
        assert len(first["added"]) == 1
        assert second["added"] == [] and second["duplicates"] == 1
        assert len(load_inbox(tmp_path)) == 1

    def test_the_paste_time_is_the_clock_not_the_reports_claim(self, tmp_path):
        received = datetime(2026, 10, 26, 14, 0, tzinfo=EASTERN)
        ingest(block([GOOD], report_time_et="2026-10-26T09:00"), tmp_path, received)
        (record,) = load_inbox(tmp_path)
        assert record["received"] == "2026-10-26T14:00:00-04:00"
        assert record["report_time_et"] == "2026-10-26T09:00"

    def test_the_pasted_text_is_kept_whole(self, tmp_path):
        text = block([GOOD])
        ingest(text, tmp_path, datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN))
        (record,) = load_inbox(tmp_path)
        assert (tmp_path / "raw" / record["raw"]).read_text(encoding="utf-8") == text


class TestEntry:
    INDEX = pd.DatetimeIndex([pd.Timestamp(d) for d in DAYS])

    def test_before_the_open_enters_that_morning(self):
        received = datetime(2026, 10, 26, 8, 55, tzinfo=EASTERN)
        assert entry_session(self.INDEX, received) == pd.Timestamp("2026-10-26")

    def test_after_the_open_waits_for_the_next_session(self):
        received = datetime(2026, 10, 26, 9, 31, tzinfo=EASTERN)
        assert entry_session(self.INDEX, received) == pd.Timestamp("2026-10-27")

    def test_a_weekend_paste_enters_monday(self):
        index = pd.DatetimeIndex([pd.Timestamp("2026-10-23"), pd.Timestamp("2026-10-26")])
        received = datetime(2026, 10, 24, 12, 0, tzinfo=EASTERN)
        assert entry_session(index, received) == pd.Timestamp("2026-10-26")

    def test_no_session_yet_means_no_entry(self):
        received = datetime(2026, 10, 30, 10, 0, tzinfo=EASTERN)
        assert entry_session(self.INDEX, received) is None


class TestScore:
    STOCK = bars(DAYS, [100, 101, 102, 103, 104], [101, 102, 103, 104, 110])
    SPY = bars(DAYS, [500, 500, 500, 500, 500], [500, 501, 502, 503, 505])

    def item(self, **overrides):
        return {**GOOD, "ticker": "GOOGL", "action": "buy", **overrides}

    def test_return_runs_from_the_entry_open_against_spy_over_the_same_span(self):
        received = datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN)
        row = score_item(self.item(horizon_days=None), received, self.STOCK, self.SPY)
        assert row["entry_price"] == 100
        assert row["return"] == pytest.approx(0.10)
        assert row["spy_return"] == pytest.approx(0.01)
        assert row["signed_excess"] == pytest.approx(0.09)
        assert row["status"] == "进行中"

    def test_a_risk_call_is_right_when_the_name_trails_spy(self):
        received = datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN)
        row = score_item(self.item(action="risk", horizon_days=None), received,
                         self.STOCK, self.SPY)
        assert row["direction"] == -1 and row["signed_excess"] < 0

    def test_a_watch_keeps_its_return_but_takes_no_side(self):
        received = datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN)
        row = score_item(self.item(action="watch", horizon_days=None), received,
                         self.STOCK, self.SPY)
        assert row["excess"] == pytest.approx(0.09)
        assert row["signed_excess"] is None

    def test_the_horizon_stops_the_clock(self):
        received = datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN)
        row = score_item(self.item(horizon_days=2), received, self.STOCK, self.SPY)
        assert row["through"] == "2026-10-28"
        assert row["status"] == "已到期"
        assert row["return"] == pytest.approx(0.03)

    def test_a_dividend_after_entry_counts(self):
        # adjclose sits 2% under close before the ex-date: the open, scaled the
        # same way, has to be measured in the same units as the later prices.
        stock = bars(DAYS, [100] * 5, [100] * 5, adj=[98, 98, 100, 100, 100])
        received = datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN)
        row = score_item(self.item(horizon_days=None), received, stock, self.SPY)
        assert row["return"] == pytest.approx(100 / 98 - 1)

    def test_a_paste_after_the_last_open_is_still_waiting(self):
        received = datetime(2026, 10, 30, 10, 0, tzinfo=EASTERN)
        row = score_item(self.item(), received, self.STOCK, self.SPY)
        assert row["status"] == "等待开盘" and "return" not in row

    def test_checkpoints_are_the_closes_of_the_first_and_fifth_sessions(self):
        received = datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN)
        row = score_item(self.item(), received, self.STOCK, self.SPY)
        first, fifth = row["checkpoints"]["1"], row["checkpoints"]["5"]
        assert first["through"] == "2026-10-26"
        assert first["excess"] == pytest.approx(0.01 - 0.0)
        assert fifth["excess"] == pytest.approx(0.10 - 0.01)
        assert "20" not in row["checkpoints"]

    def test_the_price_the_report_saw_is_compared_with_the_last_close(self):
        received = datetime(2026, 10, 26, 10, 0, tzinfo=EASTERN)
        row = score_item(self.item(price_seen=99.0), received, self.STOCK, self.SPY)
        assert row["entry_session"] == "2026-10-27"
        assert row["price_gap"] == pytest.approx(99.0 / 101.0 - 1.0)

    def test_score_inbox_puts_the_newest_report_first(self):
        records = [{"key": "a", "received": "2026-10-26T08:00:00-04:00", "task": "daily",
                    "items": [self.item()]},
                   {"key": "b", "received": "2026-10-27T08:00:00-04:00", "task": "radar",
                    "items": [self.item(action="risk")]}]
        rows = score_inbox(records, lambda t: self.SPY if t == "SPY" else self.STOCK)
        assert [r["key"] for r in rows] == ["b", "a"]


def matured(key, score, excess, direction=1):
    return {"key": key, "score": score, "direction": direction,
            "action": "buy" if direction else "watch",
            "entry_session": "2026-10-26",
            "checkpoints": {"5": {"excess": excess, "signed": direction * excess}}}


class TestSummary:
    def test_a_score_that_orders_the_returns_has_a_rank_correlation_of_one(self):
        rows = [matured("a", s, s / 100) for s in (1, 3, 5, 7, 9)]
        assert information_coefficient(rows, 5)["mean"] == pytest.approx(1.0)

    def test_reports_with_too_few_scored_names_are_left_out(self):
        rows = [matured("a", s, s / 100) for s in (1, 3, 5)]
        assert information_coefficient(rows, 5)["reports"] == 0

    def test_each_report_is_ranked_on_its_own(self):
        rising = [matured("a", s, s / 100) for s in (1, 3, 5, 7, 9)]
        falling = [matured("b", s, -s / 100) for s in (1, 3, 5, 7, 9)]
        ic = information_coefficient(rising + falling, 5)
        assert ic["reports"] == 2 and ic["mean"] == pytest.approx(0.0)

    def test_watch_items_count_for_the_ic_but_not_for_the_hit_rate(self):
        rows = [matured("a", 8, 0.02), matured("a", 6, -0.01, direction=0)]
        summary = summarize(rows)
        assert summary["directional"] == 1
        assert summary["checkpoints"]["5"] == {"n": 1, "hits": 1, "mean": 0.02, "t": None}


from ibkr_ml.signals import (random_control, selected_against_rest,  # noqa: E402
                             simulate_plan)

MONDAY_EARLY = datetime(2026, 10, 26, 8, 0, tzinfo=EASTERN)


def ohlc(rows, start="2026-10-26", splits=None):
    days = pd.bdate_range(start, periods=len(rows))
    o, h, l, c = (list(column) for column in zip(*rows))
    frame = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "adjclose": c,
                          "dividend": 0.0, "split": 1.0}, index=days)
    for position, ratio in (splits or {}).items():
        frame.iloc[position, frame.columns.get_loc("split")] = ratio
    return frame


def spy(n):
    return ohlc([(500.0, 500.0, 500.0, 500.0)] * n)


def plan(rows, **levels):
    item = {"ticker": "X", "action": "buy", "horizon_days": None, "entry_limit": None,
            "stop": None, "target": None, **levels}
    return simulate_plan(item, MONDAY_EARLY, ohlc(rows), spy(len(rows)))


class TestPlan:
    def test_no_limit_fills_at_the_open_and_a_target_exits_at_the_target(self):
        result = plan([(100, 101, 99, 100), (100, 103, 99, 102), (102, 111, 101, 110)],
                      target=110.0)
        assert result["fill_price"] == 100 and result["filled_at"] == "开盘"
        assert result["status"] == "止盈" and result["exit_price"] == 110.0
        assert result["return"] == pytest.approx(0.10)

    def test_a_limit_touched_by_the_low_fills_at_the_limit(self):
        result = plan([(100, 101, 96, 98), (97, 98, 94, 95)], entry_limit=95.0)
        assert result["fill_session"] == "2026-10-27"
        assert result["fill_price"] == 95.0 and result["filled_at"] == "限价"

    def test_an_open_already_under_the_limit_fills_at_the_open(self):
        result = plan([(94, 96, 93, 95)], entry_limit=95.0)
        assert result["fill_price"] == 94 and result["filled_at"] == "开盘"

    def test_a_limit_never_reached_expires_after_five_sessions(self):
        rows = [(100, 101, 99, 100)] * 5
        assert plan(rows, entry_limit=80.0)["status"] == "未成交"
        assert plan(rows[:3], entry_limit=80.0)["status"] == "等待成交"

    def test_a_bar_through_both_stop_and_target_counts_as_the_stop(self):
        result = plan([(100, 101, 99, 100), (100, 120, 80, 100)], stop=90.0, target=110.0)
        assert result["status"] == "止损" and result["exit_price"] == 90.0

    def test_an_open_through_the_stop_exits_at_the_open_not_the_stop(self):
        result = plan([(100, 101, 99, 100), (85, 86, 84, 85)], stop=90.0)
        assert result["status"] == "止损" and result["exit_price"] == 85

    def test_a_limit_fill_with_the_stop_inside_the_same_bar_is_stopped(self):
        # The bar cannot say whether its low came before or after the fill.
        result = plan([(100, 115, 85, 100)], entry_limit=95.0, stop=90.0, target=110.0)
        assert result["filled_at"] == "限价"
        assert result["status"] == "止损" and result["exit_session"] == "2026-10-26"

    def test_the_horizon_exits_at_the_close_of_the_first_session_on_or_after_it(self):
        rows = [(100, 101, 99, 100), (100, 101, 99, 101), (101, 103, 100, 102), (102, 104, 101, 103)]
        result = plan(rows, horizon_days=2)
        assert result["status"] == "到期"
        assert result["exit_session"] == "2026-10-28" and result["exit_price"] == 102

    def test_a_stop_above_the_fill_is_ignored_with_a_note(self):
        result = plan([(100, 101, 99, 100)], stop=105.0)
        assert result["status"] == "持有中" and result["notes"]

    def test_levels_follow_a_split_while_held(self):
        rows = [(100, 101, 99, 100), (100, 101, 99, 100), (50, 51, 49, 50)]
        frame = ohlc(rows, splits={2: 2.0})
        item = {"ticker": "X", "action": "buy", "horizon_days": None, "entry_limit": None,
                "stop": 90.0, "target": None}
        result = simulate_plan(item, MONDAY_EARLY, frame, spy(3))
        assert result["status"] == "持有中"

    def test_a_short_side_call_has_no_plan(self):
        item = {"ticker": "X", "action": "risk"}
        assert simulate_plan(item, MONDAY_EARLY, ohlc([(1, 1, 1, 1)]), spy(1))["status"] == "不适用"


class TestControls:
    def test_chosen_names_are_measured_against_the_names_passed_over(self):
        rows = [{"key": "a", "action": "buy", "direction": 1,
                 "checkpoints": {"5": {"excess": 0.03, "signed": 0.03}}},
                {"key": "a", "action": "watch", "direction": 0,
                 "checkpoints": {"5": {"excess": -0.01, "signed": None}}},
                {"key": "b", "action": "buy", "direction": 1,
                 "checkpoints": {"5": {"excess": 0.02, "signed": 0.02}}}]
        result = selected_against_rest(rows, 5)
        assert result["n"] == 1 and result["mean"] == pytest.approx(0.04)

    def universe(self, gains):
        frames = {}
        for i, gain in enumerate(gains):
            frames[f"S{i}"] = ohlc([(100, 100, 100, 100)] * 4 + [(100, 100, 100, 100 * (1 + gain))])
        return frames

    def test_a_call_better_than_every_random_name_gets_the_smallest_p(self):
        universe = self.universe([g / 100 for g in range(-5, 5)])
        rows = [{"key": "a", "direction": 1, "entry_session": "2026-10-26",
                 "checkpoints": {"5": {"excess": 0.10, "signed": 0.10}}}]
        result = random_control(rows, universe, spy(5), 5, draws=999)
        assert result["n"] == 1 and result["pool"] == 10
        assert result["p"] == pytest.approx(1 / 1000)

    def test_a_name_not_trading_on_the_entry_day_is_never_drawn(self):
        universe = self.universe([0.0, 0.0])
        universe["LATE"] = ohlc([(100, 100, 100, 100)] * 4, start="2026-10-27")
        rows = [{"key": "a", "direction": 1, "entry_session": "2026-10-26",
                 "checkpoints": {"5": {"excess": 0.0, "signed": 0.0}}}]
        assert random_control(rows, universe, spy(5), 5, draws=99)["pool"] == 2


class TestSessionHorizon:
    STOCK = bars(DAYS + ["2026-11-02"], [100] * 6, [100, 101, 102, 103, 104, 110])
    SPY = bars(DAYS + ["2026-11-02"], [500] * 6, [500] * 6)

    def test_a_short_call_is_judged_at_the_fifth_session_close(self):
        item = {**GOOD, "action": "buy", "horizon_days": 7}
        rows = score_inbox([{"key": "a", "received": "2026-10-26T08:00:00-04:00",
                             "task": "short", "items": [item]}],
                           lambda t: self.SPY if t == "SPY" else self.STOCK)
        (row,) = rows
        assert row["through"] == "2026-10-30" and row["status"] == "已到期"
        assert row["return"] == pytest.approx(0.04)
        assert row["plan"]["exit_session"] == "2026-10-30"

    def test_a_daily_call_keeps_its_calendar_horizon(self):
        item = {**GOOD, "action": "buy", "horizon_days": 7, "entry_limit": None,
                "stop": None, "target": None}
        rows = score_inbox([{"key": "a", "received": "2026-10-26T08:00:00-04:00",
                             "task": "daily", "items": [item]}],
                           lambda t: self.SPY if t == "SPY" else self.STOCK)
        assert rows[0]["through"] == "2026-11-02"


class TestBatchPaste:
    def test_several_radar_outputs_pasted_at_once_are_separate_reports(self, tmp_path):
        first = block([{"ticker": "NVDA", "action": "risk", "score": 7}], task="radar")
        second = block([{"ticker": "SLB", "action": "opportunity", "score": 8}], task="radar")
        repeat = block([{"ticker": "NVDA", "action": "risk", "score": 7}], task="radar")
        text = "\n\n---\n\n".join([first, second, repeat])
        result = ingest(text, tmp_path, datetime(2026, 10, 27, 8, 50, tzinfo=EASTERN),
                        expected_task="radar")
        assert len(result["added"]) == 2 and result["duplicates"] == 1
        assert [r["items"][0]["ticker"] for r in load_inbox(tmp_path)] == ["NVDA", "SLB"]
