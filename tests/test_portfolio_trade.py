"""Tests for the allocation runner.

The one that matters is test_todays_move_cannot_change_todays_size. A
look-ahead of exactly this shape - a signal read on the same day it starts
earning - produced a false +6.66% momentum result in this project, and it
survived a permutation test, a cluster bootstrap and a cost sweep before
being caught. The cheapest defence is a test that fires a large move into
the last bar and asserts the size does not react to it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from portfolio_trade import MIN_HISTORY, VOL_WINDOW, book_scale, parse_allocation


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
        scale, trailing, target = book_scale(calm_history(), "expanding", 0.16)
        assert scale == pytest.approx(1.0)
        assert trailing > 0 and target > 0

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

    def test_todays_move_cannot_change_todays_size(self):
        """The whole point. A crash today must not resize the book today."""
        returns = calm_history()
        baseline, _, _ = book_scale(returns, "expanding", 0.16)
        shocked = returns.copy()
        shocked.iloc[-1] = -0.25
        after, _, _ = book_scale(shocked, "expanding", 0.16)
        assert after == pytest.approx(baseline)

    def test_yesterdays_move_does_change_todays_size(self):
        """The other half: the rule has to react, one day later."""
        returns = calm_history()
        baseline, _, _ = book_scale(returns, "expanding", 0.16)
        shocked = returns.copy()
        shocked.iloc[-2] = -0.25
        after, _, _ = book_scale(shocked, "expanding", 0.16)
        assert after < baseline

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
