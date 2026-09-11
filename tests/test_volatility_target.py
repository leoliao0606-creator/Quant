"""The sizing rule must not see the day it is sizing.

weight = target / trailing_volatility is one line, and the one way it can
be wrong is to include today's return in the volatility it uses to decide
today's position. That is invisible in a backtest's summary and turns a
modest risk overlay into a machine that sells before the fall it already
knows about, so it gets an explicit test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volatility_target import apply_band, backtest, target_weight


def calm_then_wild(calm=60, wild=40, seed=5):
    rng = np.random.default_rng(seed)
    returns = np.concatenate([rng.normal(0, 0.002, calm), rng.normal(0, 0.02, wild)])
    return pd.Series(returns, index=pd.bdate_range("2020-01-01", periods=calm + wild))


class TestNoLookahead:
    def test_todays_return_cannot_change_todays_weight(self):
        returns = calm_then_wild()
        baseline = target_weight(returns)
        # A violent move on the last day. A rule that peeked would shrink
        # the position on that same day.
        shocked = returns.copy()
        shocked.iloc[-1] = -0.25
        assert target_weight(shocked).iloc[-1] == pytest.approx(baseline.iloc[-1])

    def test_the_weight_falls_after_volatility_rises_not_before(self):
        returns = calm_then_wild(calm=60, wild=40)
        weights = target_weight(returns, window=20)
        # Still sized off the calm stretch on the first wild day, and cut
        # once the wild days have entered the trailing window.
        assert weights.iloc[60] > weights.iloc[85]


class TestBounds:
    def test_the_cap_is_never_exceeded(self):
        # Near-zero volatility would ask for an enormous position.
        returns = pd.Series(np.full(300, 1e-6),
                            index=pd.bdate_range("2020-01-01", periods=300))
        returns.iloc[::50] = 1e-5
        assert target_weight(returns, cap=1.0).max() <= 1.0

    def test_a_higher_target_asks_for_a_larger_position(self):
        returns = calm_then_wild()
        low = target_weight(returns, target=0.10)
        high = target_weight(returns, target=0.30)
        assert (high >= low).all()
        assert high.mean() > low.mean()


class TestBand:
    def test_the_band_holds_a_position_between_trades(self):
        weights = pd.Series([0.50, 0.52, 0.54, 0.80],
                            index=pd.bdate_range("2020-01-01", periods=4))
        banded = apply_band(weights, band=0.10)
        assert list(banded) == [0.50, 0.50, 0.50, 0.80]

    def test_the_band_lowers_turnover_without_changing_the_shape(self):
        returns = calm_then_wild(calm=120, wild=120)
        raw = target_weight(returns)
        banded = apply_band(raw, band=0.05)
        assert backtest(returns, banded)["turnover"] < backtest(returns, raw)["turnover"]
        assert banded.mean() == pytest.approx(raw.mean(), abs=0.05)

    def test_zero_band_changes_nothing(self):
        raw = target_weight(calm_then_wild())
        assert (apply_band(raw, band=0.0) == raw).all()
