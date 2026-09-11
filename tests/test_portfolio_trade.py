"""Tests for the allocation runner.

The ones that matter are in TestSessionBoundary. A look-ahead - a signal
read on the same day it starts earning - produced a false +6.66% momentum
result in this project, and it survived a permutation test, a cluster
bootstrap and a cost sweep before being caught. The defence here is
structural: the last completed session must change the size (otherwise the
rule is a session staler than the backtest, worth 0.10 of Sharpe), and a
session still trading must not be visible at all.
"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_trade import (MIN_HISTORY, SESSION_CLOSE_HOUR, VOL_WINDOW,
                             book_scale, drop_incomplete_bar, parse_allocation)


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
