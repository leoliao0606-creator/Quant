"""Tests for the allocation runner.

The ones that matter are in TestSessionBoundary. A look-ahead - a signal
read on the same day it starts earning - produced a false +6.66% momentum
result in this project, and it survived a permutation test, a cluster
bootstrap and a cost sweep before being caught. The defence here is
structural: the last completed session must change the size (otherwise the
rule is a session staler than the backtest, worth 0.10 of Sharpe), and a
session still trading must not be visible at all.
"""

import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conftest import (FakeContract, FakeIB, FakeOrder, FakeOrderStatus,
                      FakeTrade, SequencedOrderStatus)
from portfolio_trade import (CANCEL_REFUSED_CODES, MIN_HISTORY,
                             SESSION_CLOSE_HOUR, VOL_WINDOW, book_scale,
                             cancel_and_wait, describe_trades,
                             drop_incomplete_bar, load_overlay_state,
                             market_is_open, parse_allocation, parse_sessions,
                             plan_targets, save_overlay_state, sessions_since,
                             split_working_orders, step_scale)


EASTERN = ZoneInfo("America/New_York")


def calm_history(days=800, scale=0.004, seed=7):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=days)
    return pd.Series(rng.normal(0.0003, scale, days), index=index)


class TestParseAllocation:
    def test_a_preset_name_is_expanded(self):
        weights = parse_allocation("thirds")
        assert set(weights) == {"SPY", "AGG", "TIP"}
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_an_explicit_list_is_parsed(self):
        weights = parse_allocation("SPY:0.6,AGG:0.4")
        assert weights == {"SPY": 0.6, "AGG": 0.4}

    def test_lower_case_symbols_are_accepted(self):
        assert parse_allocation("spy:0.5,agg:0.5") == {"SPY": 0.5, "AGG": 0.5}

    def test_weights_that_do_not_sum_to_one_are_refused(self):
        with pytest.raises(SystemExit):
            parse_allocation("SPY:0.6,AGG:0.3")

    def test_a_missing_weight_is_refused(self):
        with pytest.raises(SystemExit):
            parse_allocation("SPY,AGG:0.5")

    def test_a_preset_is_copied_not_shared(self):
        first = parse_allocation("thirds")
        first["SPY"] = 0.9
        assert parse_allocation("thirds")["SPY"] != 0.9


class TestBookScale:
    def test_a_calm_book_is_held_in_full(self):
        # Plain noise leaves trailing and long-run volatility within a
        # fraction of a percent of each other, so which side of 1.0 the
        # ratio lands on is an accident of the seed. Damp the recent window
        # so the book is actually calmer than its own history.
        returns = calm_history()
        returns.iloc[-VOL_WINDOW:] *= 0.3
        scale, trailing, target = book_scale(returns, "expanding", 0.16)
        assert scale == pytest.approx(1.0)
        assert trailing > 0 and target > 0
        assert trailing < target

    def test_a_book_moving_more_than_usual_is_cut(self):
        returns = calm_history()
        returns.iloc[-VOL_WINDOW:] *= 6.0
        scale, trailing, target = book_scale(returns, "expanding", 0.16)
        assert scale < 0.5
        assert trailing > target

    def test_the_cut_is_the_ratio_of_target_to_trailing(self):
        returns = calm_history()
        returns.iloc[-VOL_WINDOW:] *= 4.0
        scale, trailing, target = book_scale(returns, "expanding", 0.16)
        assert scale == pytest.approx(min(target / trailing, 1.0))

    def test_the_scale_never_exceeds_one(self):
        returns = calm_history()
        returns.iloc[-VOL_WINDOW:] *= 0.01
        scale, _, _ = book_scale(returns, "expanding", 0.16)
        assert scale == pytest.approx(1.0)

    def test_the_last_completed_session_does_change_the_size(self):
        """The caller hands in completed sessions only, so the last one
        counts. Shifting it away is what made an earlier version size off
        stale data."""
        returns = calm_history()
        baseline, _, _ = book_scale(returns, "expanding", 0.16)
        shocked = returns.copy()
        shocked.iloc[-1] = -0.25
        after, _, _ = book_scale(shocked, "expanding", 0.16)
        assert after < baseline

    def test_a_session_beyond_the_end_cannot_reach_the_estimate(self):
        """The protection against look-ahead lives in drop_incomplete_bar,
        not in book_scale: what is never passed in cannot be read."""
        returns = calm_history()
        baseline, _, _ = book_scale(returns, "expanding", 0.16)
        extended = pd.concat([returns, pd.Series(
            [-0.25], index=[returns.index[-1] + pd.Timedelta(days=3)])])
        trimmed, dropped = drop_incomplete_bar(
            extended, datetime(2020, 1, 1, 18, tzinfo=EASTERN))
        assert dropped
        after, _, _ = book_scale(trimmed, "expanding", 0.16)
        assert after == pytest.approx(baseline)

    def test_the_fixed_target_uses_the_number_it_is_given(self):
        returns = calm_history()
        returns.iloc[-VOL_WINDOW:] *= 8.0
        scale, trailing, target = book_scale(returns, "fixed", 0.16)
        assert target == 0.16
        assert scale == pytest.approx(min(0.16 / trailing, 1.0))

    def test_a_history_too_short_for_the_expanding_target_is_refused(self):
        short = calm_history(days=MIN_HISTORY - 10)
        with pytest.raises(SystemExit):
            book_scale(short, "expanding", 0.16)

    def test_a_book_that_never_moves_is_refused_rather_than_divided_by_zero(self):
        flat = pd.Series(0.0, index=pd.bdate_range("2020-01-01", periods=800))
        with pytest.raises(SystemExit):
            book_scale(flat, "expanding", 0.16)


class TestSessionBoundary:
    """A bar for a session that is still trading is a few hours of data
    wearing a day's clothes. It reads calm, so it would inflate the size
    exactly when the market is moving."""

    def frame(self, last_day):
        index = pd.bdate_range(end=last_day, periods=30)
        return pd.DataFrame({"SPY": np.linspace(100.0, 130.0, 30)}, index=index)

    def test_a_bar_dated_today_is_dropped_before_the_close(self):
        prices = self.frame(pd.Timestamp("2026-09-11"))
        kept, dropped = drop_incomplete_bar(
            prices, datetime(2026, 9, 11, 13, 22, tzinfo=EASTERN))
        assert dropped
        assert kept.index[-1] == pd.Timestamp("2026-09-10")

    def test_a_bar_dated_today_is_kept_after_the_close(self):
        prices = self.frame(pd.Timestamp("2026-09-11"))
        kept, dropped = drop_incomplete_bar(
            prices, datetime(2026, 9, 11, SESSION_CLOSE_HOUR, 5, tzinfo=EASTERN))
        assert not dropped
        assert kept.index[-1] == pd.Timestamp("2026-09-11")

    def test_yesterdays_bar_is_always_kept(self):
        prices = self.frame(pd.Timestamp("2026-09-10"))
        kept, dropped = drop_incomplete_bar(
            prices, datetime(2026, 9, 11, 9, 31, tzinfo=EASTERN))
        assert not dropped
        assert len(kept) == len(prices)

    def test_a_bar_dated_in_the_future_is_dropped(self):
        prices = self.frame(pd.Timestamp("2026-09-14"))
        kept, dropped = drop_incomplete_bar(
            prices, datetime(2026, 9, 11, 17, 0, tzinfo=EASTERN))
        assert dropped

    def test_an_empty_frame_does_not_raise(self):
        empty = pd.DataFrame({"SPY": []}, index=pd.DatetimeIndex([]))
        kept, dropped = drop_incomplete_bar(
            empty, datetime(2026, 9, 11, 17, 0, tzinfo=EASTERN))
        assert not dropped
        assert len(kept) == 0


class TestWorkingOrders:
    """Shares in an unfilled order are not in the position count.

    This is the defect that placed the same order twice: a re-run before the
    fill saw the same gap. The guard has to see orders placed by any client
    id, because connect_ib raises the id by one whenever a connection attempt
    fails, and it has to leave the rest of the account alone, because the
    account holds positions this project did not put there.
    """

    allocation = {"SPY": 1 / 3, "AGG": 1 / 3, "TIP": 1 / 3}

    def trade(self, symbol, status="Submitted", sec_type="STK", action="BUY",
              quantity=100, filled=0.0):
        order_status = FakeOrderStatus(status)
        order_status.filled = filled
        return FakeTrade(FakeContract(symbol, sec_type),
                         FakeOrder(1, "MKT", action, quantity), order_status)

    def test_a_working_order_on_an_allocation_symbol_is_flagged(self):
        mine, others = split_working_orders([self.trade("SPY")],
                                            self.allocation)
        assert [t.contract.symbol for t in mine] == ["SPY"]
        assert others == []

    def test_a_working_order_on_anything_else_is_not_flagged(self):
        # One manual limit order on an unrelated holding must not stop the
        # weekly rebalance: it does not change the share count of SPY/AGG/TIP.
        mine, others = split_working_orders([self.trade("PLTR")],
                                            self.allocation)
        assert mine == []
        assert [t.contract.symbol for t in others] == ["PLTR"]

    def test_a_filled_order_is_not_working(self):
        mine, others = split_working_orders(
            [self.trade("SPY", "Filled", filled=100.0)], self.allocation)
        assert mine == [] and others == []

    def test_a_cancelled_order_is_not_working(self):
        for status in ("Cancelled", "ApiCancelled", "Inactive"):
            mine, others = split_working_orders(
                [self.trade("SPY", status)], self.allocation)
            assert mine == [], status

    def test_a_partially_filled_order_is_still_working(self):
        # The dangerous one: 40 of 100 shares are in the position count and
        # 60 are not, so the gap this run computes is real but the order that
        # closes it is already live.
        mine, _ = split_working_orders(
            [self.trade("SPY", "Submitted", filled=40.0)], self.allocation)
        assert len(mine) == 1

    def test_a_non_stock_order_on_an_allocation_symbol_is_flagged(self):
        mine, _ = split_working_orders(
            [self.trade("SPY", sec_type="OPT")], self.allocation)
        assert len(mine) == 1

    def test_both_groups_are_separated_in_one_pass(self):
        trades = [self.trade("SPY"), self.trade("PLTR"), self.trade("AGG"),
                  self.trade("VOO", "Filled")]
        mine, others = split_working_orders(trades, self.allocation)
        assert sorted(t.contract.symbol for t in mine) == ["AGG", "SPY"]
        assert [t.contract.symbol for t in others] == ["PLTR"]

    def test_the_description_names_what_has_to_be_acted_on(self):
        line = describe_trades([self.trade("SPY", action="SELL", quantity=130,
                                           filled=30.0)])
        for part in ("SPY", "STK", "SELL", "130", "Submitted", "30"):
            assert part in line, part

    def test_the_description_survives_a_status_without_a_filled_count(self):
        status = FakeOrderStatus("PreSubmitted")
        trade = FakeTrade(FakeContract("SPY"), FakeOrder(1, "MKT"), status)
        assert "SPY" in describe_trades([trade])


class TestTradingHours:
    """A market order sent outside the session is parked until the next open.

    Nothing in the runner waits that long, so it reads the order as unfilled,
    skips every buy by the rule that holds them back until the sells are
    done, and the book sits half rebalanced for a week. IBKR is asked rather
    than the clock, because holidays and early closes are not on the clock.
    """

    FRIDAY = "20260911:0930-20260911:1600;20260914:0930-20260914:1600"

    def at(self, hour, minute=0, day=11):
        return datetime(2026, 9, day, hour, minute, tzinfo=EASTERN)

    def test_the_middle_of_the_session_is_open(self):
        assert market_is_open(self.FRIDAY, "US/Eastern", self.at(15, 30))

    def test_fifteen_minutes_after_the_close_is_shut(self):
        # The cron this replaced ran here, at 16:15.
        assert market_is_open(self.FRIDAY, "US/Eastern", self.at(16, 15)) is False

    def test_the_open_is_inclusive_and_the_close_is_not(self):
        assert market_is_open(self.FRIDAY, "US/Eastern", self.at(9, 30))
        assert market_is_open(self.FRIDAY, "US/Eastern", self.at(16, 0)) is False

    def test_before_the_open_is_shut(self):
        assert market_is_open(self.FRIDAY, "US/Eastern", self.at(8, 0)) is False

    def test_a_holiday_is_shut_even_at_noon(self):
        # This is what a clock-only check would get wrong.
        spec = "20260911:CLOSED;20260914:0930-20260914:1600"
        assert market_is_open(spec, "US/Eastern", self.at(12, 0)) is False

    def test_an_early_close_is_respected(self):
        spec = "20260911:0930-20260911:1300"
        assert market_is_open(spec, "US/Eastern", self.at(14, 0)) is False
        assert market_is_open(spec, "US/Eastern", self.at(12, 0))

    def test_the_next_session_is_read_too(self):
        assert market_is_open(self.FRIDAY, "US/Eastern", self.at(10, 0, day=14))

    def test_an_empty_string_is_unknown_rather_than_shut(self):
        # Unknown must not block a rebalance; the caller prints and proceeds.
        assert market_is_open("", "US/Eastern", self.at(15, 30)) is None

    def test_an_unparseable_string_is_unknown(self):
        assert market_is_open("nonsense", "US/Eastern", self.at(15, 30)) is None

    def test_an_unknown_time_zone_falls_back_to_eastern(self):
        assert market_is_open(self.FRIDAY, "Mars/Olympus", self.at(15, 30))

    def test_every_window_is_parsed(self):
        sessions = parse_sessions(self.FRIDAY, "US/Eastern")
        assert len(sessions) == 2
        assert sessions[0][0].hour == 9 and sessions[0][1].hour == 16

    def test_closed_days_are_not_windows(self):
        assert parse_sessions("20260911:CLOSED", "US/Eastern") == []


class StepClock:
    """A clock that moves one second every time it is read."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        value = self.now
        self.now += 1.0
        return value


class TestCancelAndWait:
    """Measured against the paper account on 2026-09-14.

    Cancelling client id 62's order from client id 41 came back as
    "Error 10147, reqId 5: OrderId 5 that needs to be cancelled is not found."
    two milliseconds after the request, and the run then sat out the whole
    sixty second --fill-timeout because ib_insync files an incoming error
    under this connection's own client id (ib_insync/wrapper.py:1096) and so
    never wrote that answer onto the order, which stayed at PendingCancel.
    """

    ALLOCATION = {"SPY": 1.0}
    REFUSAL = (5, 10147, "OrderId 5 that needs to be cancelled is not found.")

    def working_trade(self, order_id=5, statuses=None):
        status = (FakeOrderStatus("Submitted") if statuses is None
                  else SequencedOrderStatus(statuses))
        return FakeTrade(FakeContract("SPY"), FakeOrder(order_id, "LMT"),
                         status)

    def test_the_measured_refusal_code_is_recognised(self):
        assert 10147 in CANCEL_REFUSED_CODES

    def test_a_refusal_ends_the_wait_without_sleeping(self):
        ib = FakeIB()
        ib.error_on_cancel = self.REFUSAL
        trade = self.working_trade()
        stuck, refusals = cancel_and_wait(ib, [trade], self.ALLOCATION, 60.0,
                                          clock=StepClock())
        assert len(stuck) == 1
        assert ib.slept == 0.0

    def test_the_refusal_carries_the_brokers_own_words(self):
        ib = FakeIB()
        ib.error_on_cancel = self.REFUSAL
        _, refusals = cancel_and_wait(ib, [self.working_trade()],
                                      self.ALLOCATION, 60.0, clock=StepClock())
        assert len(refusals) == 1
        assert "10147" in refusals[0]
        assert "not found" in refusals[0]

    def test_without_a_refusal_the_wait_runs(self):
        ib = FakeIB()
        trade = self.working_trade()
        stuck, refusals = cancel_and_wait(ib, [trade], self.ALLOCATION, 60.0,
                                          clock=StepClock())
        assert len(stuck) == 1
        assert refusals == []
        assert ib.slept > 0.0

    def test_an_order_that_goes_away_returns_nothing_stuck(self):
        ib = FakeIB()
        trade = self.working_trade(statuses=["Submitted", "Cancelled"])
        stuck, refusals = cancel_and_wait(ib, [trade], self.ALLOCATION, 60.0,
                                          clock=StepClock())
        assert stuck == []
        assert refusals == []

    def test_an_unrelated_error_code_does_not_end_the_wait(self):
        ib = FakeIB()
        # 10349 is the order-preset notice seen on the same account; it says
        # nothing about whether the cancel will be honoured.
        ib.error_on_cancel = (5, 10349, "Order TIF was set to DAY.")
        stuck, refusals = cancel_and_wait(ib, [self.working_trade()],
                                          self.ALLOCATION, 60.0,
                                          clock=StepClock())
        assert refusals == []
        assert ib.slept > 0.0

    def test_a_refusal_for_another_order_is_ignored(self):
        ib = FakeIB()
        ib.error_on_cancel = (99, 10147, "OrderId 99 ... is not found.")
        stuck, refusals = cancel_and_wait(ib, [self.working_trade(5)],
                                          self.ALLOCATION, 60.0,
                                          clock=StepClock())
        assert refusals == []
        assert ib.slept > 0.0

    def test_every_order_given_is_cancelled(self):
        ib = FakeIB()
        trades = [self.working_trade(5), self.working_trade(6)]
        cancel_and_wait(ib, trades, self.ALLOCATION, 60.0, clock=StepClock())
        assert sorted(o.orderId for o in ib.cancelled) == [5, 6]

    def test_the_error_handler_is_removed_afterwards(self):
        ib = FakeIB()
        ib.error_on_cancel = self.REFUSAL
        cancel_and_wait(ib, [self.working_trade()], self.ALLOCATION, 60.0,
                        clock=StepClock())
        assert ib.errorEvent.handlers == []

    def test_the_error_handler_is_removed_when_cancelling_raises(self):
        ib = FakeIB()
        ib.cancel_should_raise = True
        with pytest.raises(RuntimeError):
            cancel_and_wait(ib, [self.working_trade()], self.ALLOCATION, 60.0,
                            clock=StepClock())
        assert ib.errorEvent.handlers == []


THIRDS = {"SPY": 1 / 3, "AGG": 1 / 3, "TIP": 1 / 3}
PRICES = {"SPY": 750.0, "AGG": 100.0, "TIP": 100.0}


class TestStepScale:
    """The band is measured against the book scale, not a symbol's weight.

    portfolio_build.step_overlay compares `abs(value - current) > band` where
    value is the whole book's scale. The runner compared its band against one
    symbol's target weight instead, which in `thirds` is a third of the same
    number - a 3% band on a weight needing a 9% move in the scale to fire.
    """

    def test_the_first_run_adopts_whatever_it_computed(self):
        scale, moved = step_scale(0.82, None, 0.03)
        assert (scale, moved) == (0.82, True)

    def test_a_move_inside_the_band_keeps_the_old_scale(self):
        scale, moved = step_scale(0.82, 0.80, 0.03)
        assert (scale, moved) == (0.80, False)

    def test_a_move_past_the_band_adopts_the_new_scale(self):
        scale, moved = step_scale(0.86, 0.80, 0.03)
        assert (scale, moved) == (0.86, True)

    def test_the_boundary_is_strictly_greater_like_the_backtest(self):
        # portfolio_build.step_overlay line 175 uses > and not >=.
        scale, moved = step_scale(0.83, 0.80, 0.03)
        assert (scale, moved) == (0.80, False)

    def test_the_band_is_symmetric(self):
        assert step_scale(0.74, 0.80, 0.03) == (0.74, True)

    def test_a_third_of_a_move_would_not_have_fired_on_a_weight(self):
        """The bug this replaces, stated as arithmetic.

        A book going from 100% to 95% moves each third-weight from 33.3% to
        31.7%, a gap of 1.7%. The old test was `1.7% <= 3%`, so it held. The
        scale itself moved 5%, which is past the band that was measured.
        """
        weight_gap = abs(1 / 3 * 0.95 - 1 / 3 * 1.00)
        assert weight_gap < 0.03          # the old rule would not have traded
        assert step_scale(0.95, 1.00, 0.03) == (0.95, True)


class TestSessionsSince:
    def test_no_previous_date_reads_as_unknown(self):
        index = pd.bdate_range("2026-01-01", periods=10)
        assert sessions_since(index, None) is None

    def test_it_counts_bars_and_not_calendar_days(self):
        """21 has to mean a month of trading, which is what the backtest
        means: static_weights slices closes.index, so weekends and holidays
        are simply absent rather than counted."""
        index = pd.bdate_range("2026-01-01", periods=30)
        elapsed = sessions_since(index, date(2026, 1, 1))
        assert elapsed == 29
        assert (index[-1].date() - date(2026, 1, 1)).days > elapsed

    def test_the_day_itself_is_not_counted(self):
        index = pd.bdate_range("2026-01-01", periods=5)
        assert sessions_since(index, index[-1].date()) == 0

    def test_a_date_after_the_last_bar_counts_nothing(self):
        index = pd.bdate_range("2026-01-01", periods=5)
        assert sessions_since(index, date(2030, 1, 1)) == 0


class TestPlanTargets:
    def test_a_rebalance_restores_the_target_weights(self):
        plans = plan_targets(THIRDS, {"SPY": 0, "AGG": 0, "TIP": 0}, PRICES,
                             300000.0, 1.0, None, True, 0.0)
        wanted = {p["symbol"]: p["wanted"] for p in plans}
        assert wanted == {"SPY": 133, "AGG": 1000, "TIP": 1000}

    def test_a_rebalance_multiplies_the_target_by_the_scale(self):
        plans = plan_targets(THIRDS, {"SPY": 0, "AGG": 0, "TIP": 0}, PRICES,
                             300000.0, 0.5, None, True, 0.0)
        wanted = {p["symbol"]: p["wanted"] for p in plans}
        assert wanted == {"SPY": 66, "AGG": 500, "TIP": 500}

    def test_between_rebalances_price_drift_is_left_alone(self):
        """This is the half the runner did not have.

        The backtest lets the weights drift between rebalance dates and
        trades only on them (static_weights, line 136). A held book that has
        drifted is therefore correct, not something to correct.
        """
        held = {"SPY": 200, "AGG": 500, "TIP": 900}   # badly off target
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 0.80, 0.80,
                             False, 0.0)
        assert all(p["delta"] == 0 for p in plans)

    def test_between_rebalances_a_new_scale_moves_everything_in_proportion(self):
        held = {"SPY": 100, "AGG": 1000, "TIP": 1000}
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 0.50, 1.00,
                             False, 0.0)
        wanted = {p["symbol"]: p["wanted"] for p in plans}
        assert wanted == {"SPY": 50, "AGG": 500, "TIP": 500}

    def test_a_rising_scale_buys_between_rebalances(self):
        held = {"SPY": 50, "AGG": 500, "TIP": 500}
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 1.00, 0.50,
                             False, 0.0)
        assert {p["symbol"]: p["delta"] for p in plans} == {
            "SPY": 50, "AGG": 500, "TIP": 500}

    def test_without_a_previous_scale_nothing_moves_off_a_rebalance(self):
        """A first run that is somehow not a rebalance has no ratio to apply,
        and inventing one would resize the book off a number never adopted."""
        held = {"SPY": 100, "AGG": 1000, "TIP": 1000}
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 0.5, None,
                             False, 0.0)
        assert all(p["delta"] == 0 for p in plans)

    def test_min_trade_drops_a_difference_too_small_to_be_worth_a_commission(self):
        # One share of a 750 dollar stock against 300,000 is 0.25%.
        held = {"SPY": 132, "AGG": 1000, "TIP": 1000}
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 1.0, None,
                             True, 0.005)
        spy = next(p for p in plans if p["symbol"] == "SPY")
        assert spy["wanted"] == 133 and spy["skipped"] is True
        assert spy["delta"] == 0

    def test_min_trade_zero_follows_the_backtest_exactly(self):
        held = {"SPY": 132, "AGG": 1000, "TIP": 1000}
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 1.0, None,
                             True, 0.0)
        spy = next(p for p in plans if p["symbol"] == "SPY")
        assert spy["delta"] == 1 and spy["skipped"] is False

    def test_a_difference_above_min_trade_survives(self):
        held = {"SPY": 100, "AGG": 1000, "TIP": 1000}
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 1.0, None,
                             True, 0.005)
        spy = next(p for p in plans if p["symbol"] == "SPY")
        assert spy["delta"] == 33 and spy["skipped"] is False

    def test_the_reported_target_weight_is_what_is_actually_wanted(self):
        """Between rebalances the target is not share * scale, so printing
        that would describe a book the run is not building."""
        held = {"SPY": 100, "AGG": 1000, "TIP": 1000}
        plans = plan_targets(THIRDS, held, PRICES, 300000.0, 0.50, 1.00,
                             False, 0.0)
        spy = next(p for p in plans if p["symbol"] == "SPY")
        assert spy["target_weight"] == pytest.approx(50 * 750.0 / 300000.0)
        assert spy["target_weight"] != pytest.approx(1 / 3 * 0.50)


class TestOverlayState:
    def test_a_missing_file_is_a_fresh_start_not_an_error(self, tmp_path):
        assert load_overlay_state(tmp_path / "nope.json", "thirds") is None

    def test_a_corrupt_file_is_a_fresh_start_not_an_error(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("{not json", encoding="utf-8")
        assert load_overlay_state(path, "thirds") is None

    def test_a_state_from_another_allocation_is_ignored(self, tmp_path):
        """A scale measured on one allocation says nothing about another."""
        path = tmp_path / "state.json"
        save_overlay_state(path, "balanced", 0.8, "2026-09-14")
        assert load_overlay_state(path, "thirds") is None
        assert load_overlay_state(path, "balanced")["scale"] == 0.8

    def test_what_is_saved_is_what_is_read_back(self, tmp_path):
        path = tmp_path / "logs" / "state.json"
        save_overlay_state(path, "thirds", 0.83, "2026-09-14")
        state = load_overlay_state(path, "thirds")
        assert state["scale"] == 0.83
        assert state["last_rebalance"] == "2026-09-14"
        assert date.fromisoformat(state["last_rebalance"]) == date(2026, 9, 14)

    def test_saving_creates_the_directory(self, tmp_path):
        path = tmp_path / "a" / "b" / "state.json"
        save_overlay_state(path, "thirds", 1.0, None)
        assert json.loads(path.read_text(encoding="utf-8"))["scale"] == 1.0

    def test_a_json_file_that_is_not_an_object_is_a_fresh_start(self, tmp_path):
        path = tmp_path / "state.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        assert load_overlay_state(path, "thirds") is None
