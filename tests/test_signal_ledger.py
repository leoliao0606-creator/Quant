"""Tests for the ledger that trades the ChatGPT reports by a fixed rule."""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_ledger import PRESETS, new_ledger, run_ledger  # noqa: E402

EASTERN = ZoneInfo("America/New_York")
DAYS = pd.bdate_range(end="2026-10-30", periods=20)
CREATED = datetime(2026, 10, 23, 17, 0, tzinfo=EASTERN)
NOW = datetime(2026, 10, 30, 18, 0, tzinfo=EASTERN)


def flat(price, drift=0.0):
    closes = [price * (1 + drift) ** i for i in range(len(DAYS))]
    return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                         "adjclose": closes, "dividend": 0.0, "split": 1.0}, index=DAYS)


def with_bar(frame, day, o, h, l, c):
    frame = frame.copy()
    frame.loc[pd.Timestamp(day), ["open", "high", "low", "close", "adjclose"]] = [o, h, l, c, c]
    return frame


def bars(**tickers):
    table = {"SPY": flat(500.0), "SGOV": flat(100.0, 0.0002), **tickers}
    return lambda ticker: table.get(ticker)


def ledger(task="daily", **overrides):
    params = {"task": task, **PRESETS[task], "cost_bps": 5.0, "capital": 100000.0, **overrides}
    return new_ledger("t", params, CREATED)


def report(received, items, task="daily", key=None):
    return {"key": key or received, "received": received, "task": task, "items": items}


def buy(ticker, **extra):
    return {"ticker": ticker, "action": "buy", "score": 9.0, "horizon_days": None,
            "entry_limit": None, "stop": None, "target": None, **extra}


MONDAY_EARLY = "2026-10-26T08:00:00-04:00"


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


class TestLedger:
    def test_a_buy_opens_at_the_entry_open_with_a_tenth_of_the_book(self):
        state, events = run_ledger(ledger(), [report(MONDAY_EARLY, [buy("AAA")])],
                                   bars(AAA=flat(50.0)), NOW)
        (fill,) = of_type(events, "fill")
        assert fill["session"] == "2026-10-26" and fill["price"] == 50.0
        assert fill["shares"] == 200
        assert state["positions"]["AAA"]["shares"] == 200
        for mark in of_type(events, "mark"):
            held = sum(p["value"] for p in mark["positions"].values())
            assert mark["equity"] == pytest.approx(mark["cash"] + held)

    def test_the_target_closes_the_position_at_the_target(self):
        prices = with_bar(flat(50.0), "2026-10-28", 50.0, 56.0, 49.0, 55.0)
        state, events = run_ledger(ledger(), [report(MONDAY_EARLY, [buy("AAA", target=55.0)])],
                                   bars(AAA=prices), NOW)
        sells = [e for e in of_type(events, "fill") if e["side"] == "SELL"]
        assert sells[0]["session"] == "2026-10-28" and sells[0]["price"] == 55.0
        assert sells[0]["reason"] == "止盈"
        assert "AAA" not in state["positions"]

    def test_a_later_risk_call_from_the_same_task_sells_at_its_open(self):
        prices = with_bar(flat(50.0), "2026-10-28", 47.0, 48.0, 46.0, 47.5)
        records = [report(MONDAY_EARLY, [buy("AAA")]),
                   report("2026-10-28T08:30:00-04:00",
                          [{"ticker": "AAA", "action": "risk", "score": 3.0}])]
        state, events = run_ledger(ledger(), records, bars(AAA=prices), NOW)
        sell = [e for e in of_type(events, "fill") if e["side"] == "SELL"][0]
        assert sell["session"] == "2026-10-28" and sell["price"] == 47.0
        assert "AAA" not in state["positions"]

    def test_a_full_book_skips_the_next_call_and_says_why(self):
        records = [report(MONDAY_EARLY, [buy("AAA"), {**buy("BBB"), "score": 8.6}])]
        state, events = run_ledger(ledger(max_positions=1), records,
                                   bars(AAA=flat(50.0), BBB=flat(20.0)), NOW)
        assert list(state["positions"]) == ["AAA"]
        (skip,) = of_type(events, "skip")
        assert skip["symbol"] == "BBB" and skip["reason"] == "满仓"

    def test_reports_pasted_before_the_ledger_existed_are_not_traded(self):
        # Pasted after Friday's close but before the ledger was opened at
        # 17:00: its entry is Monday's open, inside the ledger's life, so only
        # the creation time can keep it out.
        records = [report("2026-10-23T16:30:00-04:00", [buy("AAA")])]
        state, events = run_ledger(ledger(), records, bars(AAA=flat(50.0)), NOW)
        assert of_type(events, "fill") == [] and of_type(events, "skip") == []
        assert state["positions"] == {}

    def test_another_tasks_reports_are_not_traded(self):
        records = [report(MONDAY_EARLY, [buy("AAA")], task="radar")]
        _, events = run_ledger(ledger(), records, bars(AAA=flat(50.0)), NOW)
        assert of_type(events, "fill") == []

    def test_running_again_books_nothing_twice(self):
        records = [report(MONDAY_EARLY, [buy("AAA")])]
        prices = bars(AAA=flat(50.0))
        state, _ = run_ledger(ledger(), records, prices, NOW)
        again, events = run_ledger(state, records, prices, NOW)
        assert events == [] and again == state

    def test_an_unfilled_limit_is_booked_on_the_day_it_lapses(self):
        records = [report(MONDAY_EARLY, [buy("AAA", entry_limit=40.0)])]
        _, events = run_ledger(ledger(), records, bars(AAA=flat(50.0)), NOW)
        (skip,) = of_type(events, "skip")
        assert skip["reason"] == "限价未成交" and skip["session"] == "2026-10-30"

    def test_idle_cash_earns_the_cash_proxy(self):
        _, events = run_ledger(ledger(), [], bars(), NOW)
        first = of_type(events, "mark")[0]
        assert first["interest"] == pytest.approx(100000.0 * 0.0002)
