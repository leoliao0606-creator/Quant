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
