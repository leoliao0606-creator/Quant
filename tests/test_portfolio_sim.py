"""Tests for the broker-free ledger and the Yahoo parser behind it.

The first class is the one that matters. A signal that read the session it
was filled on produced a false +6.66% result in this project and passed
every control built on the same pipeline. Here the defence is the same one
test_portfolio_trade.py uses: change the fill session and the decision must
not move; change the session before it and the decision must.
"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibkr_ml.yahoo import parse_chart  # noqa: E402
from portfolio_sim import (CASH_PROXY, frozen_mismatch, new_state,  # noqa: E402
                           run_sessions)

EASTERN = ZoneInfo("America/New_York")
ALLOCATION = {"SPY": 1 / 3, "AGG": 1 / 3, "TIP": 1 / 3}
PARAMS = {"allocation": "thirds", "overlay": "expanding", "fixed_target": 0.16,
          "band": 0.03, "rebalance": 21, "min_trade": 0.005, "cost_bps": 5.0,
          "capital": 300000.0}
# The last session in every invented history is Friday 2026-10-23.
LAST_FRIDAY = pd.Timestamp("2026-10-23")
AFTER_LAST_CLOSE = datetime(2026, 10, 23, 18, 0, tzinfo=EASTERN)


def make_bars(days=420, seed=3):
    index = pd.bdate_range(end=LAST_FRIDAY, periods=days)
    rng = np.random.default_rng(seed)
    bars = {}
    for symbol in [*ALLOCATION, CASH_PROXY]:
        drift, vol = (0.00016, 0.0002) if symbol == CASH_PROXY else (0.0003, 0.008)
        prices = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, days))
        bars[symbol] = pd.DataFrame({"close": prices, "adjclose": prices,
                                     "dividend": 0.0, "split": 1.0}, index=index)
    return bars


def started(created, **overrides):
    return new_state("t", ALLOCATION, {**PARAMS, **overrides}, created)


def evening(day):
    return datetime(day.year, day.month, day.day, 17, 0, tzinfo=EASTERN)


def of_type(events, kind):
    return [e for e in events if e["type"] == kind]


def run_two_weeks(bars, **overrides):
    """Start the evening of Friday 10-09: decisions on 10-16 and 10-23."""
    state = started(evening(pd.Timestamp("2026-10-09")), **overrides)
    return run_sessions(state, bars, AFTER_LAST_CLOSE)


class TestNoLookAhead:
    def test_the_fill_session_cannot_move_the_decision(self):
        bars = make_bars()
        _, before = run_two_weeks(bars)
        bars["SPY"].loc[LAST_FRIDAY, "adjclose"] *= 1.30
        _, after = run_two_weeks(bars)
        first, second = of_type(before, "decision")[-1], of_type(after, "decision")[-1]
        assert first["data_through"] == "2026-10-22"
        assert second["trailing_vol"] == first["trailing_vol"]
        assert second["raw_scale"] == first["raw_scale"]

    def test_the_session_before_does_move_the_decision(self):
        bars = make_bars()
        _, before = run_two_weeks(bars)
        bars["SPY"].loc[pd.Timestamp("2026-10-22"), "adjclose"] *= 1.05
        _, after = run_two_weeks(bars)
        assert (of_type(after, "decision")[-1]["trailing_vol"]
                != of_type(before, "decision")[-1]["trailing_vol"])

    def test_sessions_that_closed_before_the_ledger_existed_are_not_traded(self):
        state, events = run_two_weeks(make_bars())
        marks = of_type(events, "mark")
        assert marks[0]["session"] == "2026-10-12"
        assert state["first_session"] == "2026-10-12"
        assert state["base_session"] == "2026-10-09"
        assert len(marks) == 10

    def test_a_ledger_opened_before_the_close_takes_that_session(self):
        created = datetime(2026, 10, 16, 10, 0, tzinfo=EASTERN)
        _, events = run_sessions(started(created), make_bars(), AFTER_LAST_CLOSE)
        assert of_type(events, "mark")[0]["session"] == "2026-10-16"
        assert of_type(events, "decision")[0]["data_through"] == "2026-10-15"

    def test_a_session_still_settling_is_left_for_the_next_run(self):
        state = started(evening(pd.Timestamp("2026-10-09")))
        settling = datetime(2026, 10, 23, 16, 10, tzinfo=EASTERN)
        state, events = run_sessions(state, make_bars(), settling)
        assert state["last_processed"] == "2026-10-22"
        assert not of_type(events, "decision")[1:]


class TestLedger:
    def test_only_fridays_decide(self):
        _, events = run_two_weeks(make_bars())
        days = [pd.Timestamp(e["session"]).day_name() for e in of_type(events, "decision")]
        assert days == ["Friday", "Friday"]

    def test_fills_are_at_the_close_and_pay_the_cost(self):
        bars = make_bars()
        _, events = run_two_weeks(bars)
        fills = [e for e in of_type(events, "fill") if e["session"] == "2026-10-16"]
        assert {f["symbol"] for f in fills} == set(ALLOCATION)
        for fill in fills:
            close = bars[fill["symbol"]].loc[pd.Timestamp("2026-10-16"), "close"]
            assert fill["price"] == pytest.approx(close)
            assert fill["cost"] == pytest.approx(fill["shares"] * close * 5e-4)

    def test_every_mark_adds_up(self):
        _, events = run_two_weeks(make_bars())
        for mark in of_type(events, "mark"):
            held = sum(row["shares"] * row["close"] for row in mark["positions"].values())
            assert mark["equity"] == pytest.approx(mark["cash"] + held)

    def test_cash_never_goes_negative_even_when_trading_is_dear(self):
        _, events = run_two_weeks(make_bars(), cost_bps=80.0)
        assert min(m["cash"] for m in of_type(events, "mark")) >= 0.0

    def test_idle_cash_earns_the_cash_proxy(self):
        bars = make_bars()
        _, events = run_two_weeks(bars)
        first = of_type(events, "mark")[0]
        sgov = bars[CASH_PROXY]["adjclose"]
        expected = 300000.0 * (sgov[pd.Timestamp("2026-10-12")]
                               / sgov[pd.Timestamp("2026-10-09")] - 1.0)
        assert first["interest"] == pytest.approx(expected)

    def test_running_again_processes_nothing_twice(self):
        bars = make_bars()
        state, _ = run_two_weeks(bars)
        again, events = run_sessions(state, bars, AFTER_LAST_CLOSE)
        assert events == []
        assert again == state

    def test_a_dividend_pays_the_shares_held_into_the_ex_date(self):
        bars = make_bars()
        bars["AGG"].loc[pd.Timestamp("2026-10-20"), "dividend"] = 0.5
        _, events = run_two_weeks(bars)
        bought = sum(f["shares"] for f in of_type(events, "fill")
                     if f["symbol"] == "AGG" and f["session"] == "2026-10-16")
        (dividend,) = of_type(events, "dividend")
        assert dividend["session"] == "2026-10-20"
        assert dividend["amount"] == pytest.approx(bought * 0.5)

    def test_a_split_multiplies_shares_and_pays_the_fraction_in_cash(self):
        bars = make_bars()
        split_day = pd.Timestamp("2026-10-20")
        bars["SPY"].loc[split_day, "split"] = 1.5
        bars["SPY"].loc[split_day:, "close"] /= 1.5
        _, events = run_two_weeks(bars)
        (split,) = of_type(events, "split")
        held = split["shares_before"]
        assert split["shares_after"] == int(held * 1.5)
        price = bars["SPY"].loc[split_day, "close"]
        assert split["cash_in_lieu"] == pytest.approx((held * 1.5 - int(held * 1.5)) * price)

    def test_changing_a_parameter_is_named(self):
        assert frozen_mismatch(PARAMS, {**PARAMS, "band": 0.05}) == ["band"]
        assert frozen_mismatch(PARAMS, dict(PARAMS)) == []


def chart(closes, adjcloses, days, dividends=(), splits=(), opens=None):
    stamps = [int(datetime(*d, 9, 30, tzinfo=EASTERN).timestamp()) for d in days]
    quote = {"close": closes} if opens is None else {"close": closes, "open": opens}
    return {"chart": {"error": None, "result": [{
        "timestamp": stamps,
        "indicators": {"quote": [quote], "adjclose": [{"adjclose": adjcloses}]},
        "events": {
            "dividends": {str(stamps[i]): {"amount": a, "date": stamps[i]} for i, a in dividends},
            "splits": {str(stamps[i]): {"date": stamps[i], "numerator": n, "denominator": d}
                       for i, n, d in splits}}}]}}


class TestParseChart:
    DAYS = [(2026, 10, 19), (2026, 10, 20), (2026, 10, 21), (2026, 10, 22)]

    def test_prices_and_dividends_before_a_split_are_restored_to_what_printed(self):
        # Yahoo shows the two sessions before a 2:1 split at half their price,
        # and the dividend paid between them at half its amount.
        payload = chart([50.0, 51.0, 51.0, 52.0], [49.0, 50.0, 51.0, 52.0], self.DAYS,
                        dividends=[(1, 0.25)], splits=[(2, 2, 1)],
                        opens=[49.5, 50.5, 50.8, 51.5])
        frame = parse_chart(payload)
        assert list(frame["close"]) == [100.0, 102.0, 51.0, 52.0]
        assert list(frame["open"]) == [99.0, 101.0, 50.8, 51.5]
        assert list(frame["dividend"]) == [0.0, 0.5, 0.0, 0.0]
        assert list(frame["split"]) == [1.0, 1.0, 2.0, 1.0]
        assert list(frame["adjclose"]) == [49.0, 50.0, 51.0, 52.0]

    def test_sessions_are_dated_in_new_york(self):
        frame = parse_chart(chart([1.0] * 4, [1.0] * 4, self.DAYS))
        assert [d.date().isoformat() for d in frame.index] == [
            "2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]

    def test_a_session_yahoo_has_not_filled_in_is_dropped(self):
        frame = parse_chart(chart([1.0, None, 1.0, 1.0], [1.0, None, 1.0, 1.0], self.DAYS))
        assert len(frame) == 3

    def test_an_error_response_is_refused(self):
        with pytest.raises(ValueError):
            parse_chart({"chart": {"error": {"code": "Not Found"}, "result": None}})
