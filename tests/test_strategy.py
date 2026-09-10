"""Rules that decide what happens to one symbol on one bar.

This is the file that decides whether money is at risk, so every branch is
pinned here, including the order the branches are checked in.
"""

from __future__ import annotations

import pytest

from ibkr_ml.config import ModelConfig, RiskConfig
from ibkr_ml.strategy import _target_quantity, generate_trade_decision


def decide(**overrides):
    kwargs = {
        "symbol": "AAA",
        "probability_up": 0.50,
        "last_price": 100.0,
        "current_quantity": 0,
        "average_cost": 0.0,
        "equity": 100000.0,
        "model_config": ModelConfig(entry_probability=0.60, exit_probability=0.45),
        "risk_config": RiskConfig(),
        "daily_loss_limit_hit": False,
    }
    kwargs.update(overrides)
    return generate_trade_decision(**kwargs)


class TestEntries:
    def test_probability_above_entry_opens_a_position(self):
        decision = decide(probability_up=0.70)
        assert decision.action == "BUY"
        assert decision.reason == "model_entry"
        assert decision.target_quantity > 0

    def test_probability_below_entry_does_nothing(self):
        decision = decide(probability_up=0.55)
        assert decision.action == "HOLD"
        assert decision.reason == "no_change"

    def test_entry_is_suppressed_when_no_slot_is_available(self):
        decision = decide(probability_up=0.70, allow_new_position=False)
        assert decision.action == "HOLD"
        assert decision.reason == "entry_filtered"
        assert decision.target_quantity == 0

    def test_entry_too_small_to_size_becomes_a_hold(self):
        decision = decide(probability_up=0.70, equity=1.0)
        assert decision.action == "HOLD"
        assert decision.reason == "size_too_small"


class TestExits:
    def test_stop_loss_fires_below_the_threshold(self):
        decision = decide(current_quantity=100, average_cost=100.0, last_price=99.1)
        assert decision.action == "SELL"
        assert decision.reason == "stop_loss"
        assert decision.target_quantity == 0

    def test_stop_loss_does_not_fire_just_above_the_threshold(self):
        decision = decide(current_quantity=100, average_cost=100.0, last_price=99.3)
        assert decision.reason != "stop_loss"

    def test_take_profit_fires_above_the_threshold(self):
        decision = decide(current_quantity=100, average_cost=100.0, last_price=101.6)
        assert decision.action == "SELL"
        assert decision.reason == "take_profit"

    def test_probability_below_exit_closes_the_position(self):
        decision = decide(current_quantity=100, average_cost=100.0, probability_up=0.40)
        assert decision.action == "SELL"
        assert decision.reason == "model_exit"

    def test_position_is_held_between_the_thresholds(self):
        decision = decide(current_quantity=100, average_cost=100.0, probability_up=0.50)
        assert decision.action == "HOLD"
        assert decision.reason == "no_change"
        assert decision.target_quantity == 100


class TestForceFlat:
    def test_force_flat_closes_an_open_position(self):
        decision = decide(force_flat=True, current_quantity=100, average_cost=100.0)
        assert decision.action == "SELL"
        assert decision.reason == "session_close_flatten"
        assert decision.target_quantity == 0

    def test_force_flat_on_a_flat_book_does_nothing(self):
        decision = decide(force_flat=True, probability_up=0.99)
        assert decision.action == "HOLD"
        assert decision.reason == "session_close_already_flat"
        assert decision.target_quantity == 0

    def test_force_flat_beats_a_strong_entry_signal(self):
        decision = decide(force_flat=True, probability_up=0.99, current_quantity=0)
        assert decision.action != "BUY"

    def test_force_flat_still_closes_when_the_bar_is_untrusted(self):
        # A stale bar must never be able to carry a position overnight.
        decision = decide(
            force_flat=True,
            risk_exit_only=True,
            current_quantity=100,
            average_cost=100.0,
        )
        assert decision.action == "SELL"
        assert decision.reason == "session_close_flatten"


class TestRiskExitOnly:
    def test_untrusted_bar_blocks_a_model_entry(self):
        decision = decide(probability_up=0.99, risk_exit_only=True, blocked_reason="stale_data")
        assert decision.action == "HOLD"
        assert decision.reason == "stale_data"

    def test_untrusted_bar_still_honours_the_stop_loss(self):
        decision = decide(
            risk_exit_only=True,
            current_quantity=100,
            average_cost=100.0,
            last_price=99.1,
        )
        assert decision.action == "SELL"
        assert decision.reason == "stop_loss"

    def test_untrusted_bar_blocks_a_model_exit(self):
        decision = decide(
            risk_exit_only=True,
            current_quantity=100,
            average_cost=100.0,
            probability_up=0.10,
            blocked_reason="same_bar",
        )
        assert decision.action == "HOLD"
        assert decision.reason == "same_bar"


class TestDailyLossLimit:
    def test_limit_closes_an_open_position(self):
        decision = decide(daily_loss_limit_hit=True, current_quantity=100, average_cost=100.0)
        assert decision.action == "SELL"
        assert decision.reason == "daily_loss_limit"

    def test_limit_blocks_new_entries(self):
        decision = decide(daily_loss_limit_hit=True, probability_up=0.99)
        assert decision.action == "HOLD"
        assert decision.reason == "daily_loss_limit_halt"

    def test_stop_loss_is_checked_before_the_daily_limit(self):
        # Both would sell; the reason has to say which rule actually fired.
        decision = decide(
            daily_loss_limit_hit=True,
            current_quantity=100,
            average_cost=100.0,
            last_price=99.1,
        )
        assert decision.reason == "stop_loss"


class TestPositionSizing:
    def test_notional_cap_binds_at_the_shipped_defaults(self):
        # risk_per_trade only binds below max_position_fraction * stop_loss_pct
        # (0.20 * 0.008 = 0.0016). At the default 0.01 the notional cap always
        # wins, so every position is 20% of equity whatever risk_per_trade says.
        risk_config = RiskConfig()
        quantity = _target_quantity(100.0, 100000.0, risk_config)
        assert quantity == 200
        assert quantity * 100.0 == pytest.approx(100000.0 * risk_config.max_position_fraction)

    def test_risk_budget_binds_once_it_is_small_enough(self):
        risk_config = RiskConfig(risk_per_trade=0.0008)
        quantity = _target_quantity(100.0, 100000.0, risk_config)
        # 100000 * 0.0008 / (100 * 0.008) = 100 shares, below the 200 notional cap.
        assert quantity == 100

    def test_non_positive_price_sizes_to_nothing(self):
        assert _target_quantity(0.0, 100000.0, RiskConfig()) == 0
        assert _target_quantity(-5.0, 100000.0, RiskConfig()) == 0
