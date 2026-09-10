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


class TestConvictionScore:
    """Position size follows conviction, because realised return does.

    Measured on held-out data: the top decile of predictions returned 0.109%
    against 0.003% for the bottom, and the top 1% returned 0.96%. A fixed size
    charges the same cost against all of them, so the weak trades spend what the
    strong ones earn.
    """

    def test_below_the_entry_threshold_scores_zero(self):
        from ibkr_ml.strategy import conviction_score

        assert conviction_score(0.55, 0.60, 0.85) == 0
        assert conviction_score(0.599, 0.60, 0.85) == 0

    def test_exactly_at_the_threshold_scores_one(self):
        from ibkr_ml.strategy import conviction_score

        assert conviction_score(0.60, 0.60, 0.85) == 1

    def test_the_ceiling_scores_ten(self):
        from ibkr_ml.strategy import conviction_score

        assert conviction_score(0.85, 0.60, 0.85) == 10
        assert conviction_score(0.99, 0.60, 0.85) == 10  # capped, never above

    def test_the_score_rises_with_probability(self):
        from ibkr_ml.strategy import conviction_score

        scores = [conviction_score(p, 0.60, 0.85) for p in (0.62, 0.68, 0.74, 0.80, 0.85)]
        assert scores == sorted(scores)
        assert scores[0] < scores[-1]

    def test_a_ceiling_of_one_compresses_the_scale(self):
        """Why the ceiling must come from the model's observed range.

        A class-balanced gradient boosting model rarely prints above 0.85.
        Measuring headroom to 1.0 leaves the top of the scale unreachable, so
        the strongest signal in the data gets sized as if it were ordinary.
        """
        from ibkr_ml.strategy import conviction_score

        realistic = conviction_score(0.84, 0.62, 0.84)
        against_one = conviction_score(0.84, 0.62, 1.0)
        assert realistic == 10
        assert against_one < 7


class TestPositionScaling:
    def test_fixed_ignores_conviction(self):
        from ibkr_ml.strategy import position_scale

        assert position_scale(1, "fixed") == 1.0
        assert position_scale(10, "fixed") == 1.0

    def test_linear_tracks_conviction(self):
        from ibkr_ml.strategy import position_scale

        assert position_scale(1, "linear") == pytest.approx(0.1)
        assert position_scale(5, "linear") == pytest.approx(0.5)
        assert position_scale(10, "linear") == pytest.approx(1.0)

    def test_quadratic_concentrates_on_the_top(self):
        from ibkr_ml.strategy import position_scale

        assert position_scale(1, "quadratic") == pytest.approx(0.01)
        assert position_scale(5, "quadratic") == pytest.approx(0.25)
        assert position_scale(10, "quadratic") == pytest.approx(1.0)
        # A weak signal gets a tenth of what linear would give it.
        assert position_scale(3, "quadratic") < position_scale(3, "linear") / 3

    def test_zero_conviction_means_no_position(self):
        from ibkr_ml.strategy import position_scale

        for mode in ("fixed", "linear", "quadratic"):
            assert position_scale(0, mode) == 0.0

    def test_an_unknown_mode_falls_back_to_fixed(self):
        from ibkr_ml.strategy import position_scale

        assert position_scale(4, "something else") == 1.0


class TestConvictionDrivesSize:
    def decide_with(self, probability, sizing, entry=0.60, ceiling=0.85):
        return generate_trade_decision(
            symbol="AAA",
            probability_up=probability,
            last_price=100.0,
            current_quantity=0,
            average_cost=0.0,
            equity=1_000_000.0,
            model_config=ModelConfig(
                entry_probability=entry, exit_probability=0.45, probability_ceiling=ceiling
            ),
            risk_config=RiskConfig(position_sizing=sizing),
            daily_loss_limit_hit=False,
        )

    def test_a_stronger_signal_buys_more_under_linear_sizing(self):
        weak = self.decide_with(0.62, "linear")
        strong = self.decide_with(0.84, "linear")
        assert weak.action == "BUY" and strong.action == "BUY"
        assert strong.target_quantity > weak.target_quantity * 3

    def test_fixed_sizing_buys_the_same_regardless(self):
        weak = self.decide_with(0.62, "fixed")
        strong = self.decide_with(0.84, "fixed")
        assert weak.target_quantity == strong.target_quantity

    def test_quadratic_separates_them_further_than_linear(self):
        linear_ratio = (
            self.decide_with(0.84, "linear").target_quantity
            / max(self.decide_with(0.65, "linear").target_quantity, 1)
        )
        quadratic_ratio = (
            self.decide_with(0.84, "quadratic").target_quantity
            / max(self.decide_with(0.65, "quadratic").target_quantity, 1)
        )
        assert quadratic_ratio > linear_ratio

    def test_the_decision_carries_the_score_and_the_scale(self):
        decision = self.decide_with(0.80, "linear")
        assert 1 <= decision.conviction <= 10
        assert 0.0 < decision.position_scale <= 1.0

    def test_a_signal_below_the_threshold_is_not_bought(self):
        decision = self.decide_with(0.55, "linear")
        assert decision.action == "HOLD"
        assert decision.conviction == 0


class TestHoldingAndEntryTiming:
    """Two execution rules that cost 12x the annualised return when missing.

    Measured: positions closed after 6.9 bars against a 12-bar label horizon,
    and 44% of trades exited via the end-of-session flatten averaging 0.04%.
    """

    def decide(self, **over):
        kwargs = dict(
            symbol="AAA", probability_up=0.50, last_price=100.0,
            current_quantity=100, average_cost=100.0, equity=100000.0,
            model_config=ModelConfig(entry_probability=0.60, exit_probability=0.45),
            risk_config=RiskConfig(minimum_holding_bars=12, no_entry_within_bars_of_close=12),
            daily_loss_limit_hit=False,
        )
        kwargs.update(over)
        return generate_trade_decision(**kwargs)

    def test_the_model_cannot_close_before_the_horizon(self):
        decision = self.decide(probability_up=0.10, bars_held=3)
        assert decision.action == "HOLD"
        assert decision.reason == "minimum_holding"

    def test_it_can_close_once_the_horizon_has_elapsed(self):
        decision = self.decide(probability_up=0.10, bars_held=12)
        assert decision.action == "SELL"
        assert decision.reason == "model_exit"

    def test_the_stop_loss_still_fires_inside_the_holding_period(self):
        # The hold applies to the model changing its mind, never to protection.
        decision = self.decide(bars_held=1, last_price=90.0)
        assert decision.action == "SELL"
        assert decision.reason == "stop_loss"

    def test_the_flatten_still_fires_inside_the_holding_period(self):
        decision = self.decide(bars_held=1, force_flat=True)
        assert decision.action == "SELL"
        assert decision.reason == "session_close_flatten"

    def test_no_entry_close_to_the_session_end(self):
        decision = self.decide(probability_up=0.90, current_quantity=0, bars_to_close=5)
        assert decision.action == "HOLD"
        assert decision.reason == "too_close_to_session_end"

    def test_entry_allowed_with_room_left(self):
        decision = self.decide(probability_up=0.90, current_quantity=0, bars_to_close=30)
        assert decision.action == "BUY"

    def test_both_rules_are_off_by_default(self):
        decision = generate_trade_decision(
            symbol="AAA", probability_up=0.10, last_price=100.0,
            current_quantity=100, average_cost=100.0, equity=100000.0,
            model_config=ModelConfig(entry_probability=0.60, exit_probability=0.45),
            risk_config=RiskConfig(), daily_loss_limit_hit=False, bars_held=0,
        )
        assert decision.reason == "model_exit"
