"""The backtest must apply the same rules the live loop applies.

Each test here drives a hand-built price path so the expected outcome is known
in advance, which is the only way to tell "the rule fired" apart from "the
random data happened to end there".
"""

from __future__ import annotations

import pandas as pd
import pytest

from ibkr_ml.backtest import _PortfolioSimulator, simulate_probability_strategy
from ibkr_ml.config import ModelConfig, RiskConfig


def build_rows(prices_by_symbol, probabilities_by_symbol, start="2026-01-05 09:30"):
    rows = []
    for symbol, prices in prices_by_symbol.items():
        probabilities = probabilities_by_symbol[symbol]
        for index, (price, probability) in enumerate(zip(prices, probabilities)):
            rows.append(
                {
                    "timestamp": pd.Timestamp(start) + pd.Timedelta(minutes=5 * index),
                    "symbol": symbol,
                    "close": float(price),
                    "probability_up": float(probability),
                }
            )
    return pd.DataFrame(rows)


def run_simulator(prices, probabilities, risk_config=None, cost_bps=0.0,
                  max_active_positions=2, flatten_last=False, symbol="AAA"):
    """Drive one symbol through the simulator and return it for inspection."""
    simulator = _PortfolioSimulator(
        model_config=ModelConfig(entry_probability=0.60, exit_probability=0.40),
        risk_config=risk_config or RiskConfig(),
        transaction_cost_bps=cost_bps,
        starting_equity=100000.0,
        max_active_positions=max_active_positions,
    )
    last_index = len(prices) - 1
    for index, (price, probability) in enumerate(zip(prices, probabilities)):
        timestamp = pd.Timestamp("2026-01-05 09:30") + pd.Timedelta(minutes=5 * index)
        rows = pd.DataFrame(
            [{"timestamp": timestamp, "symbol": symbol, "close": price, "probability_up": probability}]
        )
        simulator.step(timestamp, rows, force_flat=flatten_last and index == last_index)
    return simulator


class TestRiskRulesReachTheBacktest:
    def test_stop_loss_closes_the_position(self):
        # Entry at 100, stop at 0.8%. The drop to 98.8 is 1.2% and must sell.
        simulator = run_simulator([100.0, 100.0, 98.8], [0.9, 0.9, 0.9])
        assert simulator.trade_count == 2
        assert simulator.positions["AAA"].quantity == 0

    def test_no_stop_means_the_position_survives_the_same_drop(self):
        # Same path, stop moved out of reach: the sell above was the stop, not
        # the model, which is what makes the first test meaningful.
        loose = RiskConfig(stop_loss_pct=0.50, take_profit_pct=0.50)
        simulator = run_simulator([100.0, 100.0, 98.8], [0.9, 0.9, 0.9], risk_config=loose)
        assert simulator.positions["AAA"].quantity > 0

    def test_take_profit_closes_the_position(self):
        simulator = run_simulator([100.0, 100.0, 101.6], [0.9, 0.9, 0.9])
        assert simulator.trade_count == 2
        assert simulator.positions["AAA"].quantity == 0

    def test_flatten_closes_the_book_on_the_last_bar(self):
        simulator = run_simulator([100.0, 100.1, 100.2], [0.9, 0.9, 0.9], flatten_last=True)
        assert simulator.positions["AAA"].quantity == 0

    def test_without_flatten_the_position_is_carried(self):
        simulator = run_simulator([100.0, 100.1, 100.2], [0.9, 0.9, 0.9], flatten_last=False)
        assert simulator.positions["AAA"].quantity > 0

    def test_daily_trade_cap_stops_further_entries(self):
        # Alternate above and below the thresholds so the strategy wants to
        # trade on every bar, then check it is cut off at the cap.
        capped = RiskConfig(max_daily_trade_count=4)
        prices = [100.0] * 20
        probabilities = [0.9, 0.1] * 10
        simulator = run_simulator(prices, probabilities, risk_config=capped)
        assert simulator.trade_count <= 4

    def test_uncapped_trading_produces_more_trades(self):
        uncapped = RiskConfig(max_daily_trade_count=None)
        prices = [100.0] * 20
        probabilities = [0.9, 0.1] * 10
        simulator = run_simulator(prices, probabilities, risk_config=uncapped)
        assert simulator.trade_count > 4

    def test_daily_loss_limit_halts_the_day(self):
        # Sized to the full account so a 5% fall in the only holding moves
        # equity by 5% as well, past the 2% daily limit, while the 10% stop
        # stays out of reach - otherwise the stop would take the credit.
        risk_config = RiskConfig(
            risk_per_trade=0.20,
            max_position_fraction=1.0,
            stop_loss_pct=0.10,
            take_profit_pct=0.50,
            max_daily_loss_pct=0.02,
        )
        simulator = run_simulator(
            [100.0, 95.0, 95.0, 95.0],
            [0.9, 0.9, 0.9, 0.9],
            risk_config=risk_config,
        )
        assert simulator.positions["AAA"].quantity == 0

    def test_no_daily_loss_limit_means_the_same_fall_is_held(self):
        risk_config = RiskConfig(
            risk_per_trade=0.20,
            max_position_fraction=1.0,
            stop_loss_pct=0.10,
            take_profit_pct=0.50,
            max_daily_loss_pct=0.99,
        )
        simulator = run_simulator(
            [100.0, 95.0, 95.0, 95.0],
            [0.9, 0.9, 0.9, 0.9],
            risk_config=risk_config,
        )
        assert simulator.positions["AAA"].quantity > 0


class TestPortfolioConstraints:
    def test_max_active_positions_is_respected(self):
        rows = build_rows(
            {"AAA": [100.0] * 4, "BBB": [100.0] * 4, "CCC": [100.0] * 4},
            {"AAA": [0.90] * 4, "BBB": [0.85] * 4, "CCC": [0.80] * 4},
        )
        result = simulate_probability_strategy(
            prediction_rows=rows,
            entry_probability=0.60,
            exit_probability=0.40,
            max_active_positions=2,
        )
        assert result["equity_curve"] is not None
        # Three symbols all qualify; only two slots exist.
        assert result["trade_count"] <= 4

    def test_highest_probability_symbol_gets_the_only_slot(self):
        rows = build_rows(
            {"AAA": [100.0] * 3, "BBB": [100.0] * 3},
            {"AAA": [0.70] * 3, "BBB": [0.95] * 3},
        )
        simulator = _PortfolioSimulator(
            model_config=ModelConfig(entry_probability=0.60, exit_probability=0.40),
            risk_config=RiskConfig(),
            transaction_cost_bps=0.0,
            starting_equity=100000.0,
            max_active_positions=1,
        )
        for timestamp, timestamp_rows in rows.groupby("timestamp", sort=True):
            simulator.step(timestamp, timestamp_rows, force_flat=False)
        assert simulator.positions["BBB"].quantity > 0
        assert simulator.positions.get("AAA", None) is None or simulator.positions["AAA"].quantity == 0


class TestCosts:
    def test_transaction_cost_reduces_equity(self):
        flat_prices = [100.0] * 6
        churn = [0.9, 0.1] * 3
        free = run_simulator(flat_prices, churn, cost_bps=0.0)
        charged = run_simulator(flat_prices, churn, cost_bps=50.0)
        assert charged._equity() < free._equity()

    def test_a_flat_market_with_no_cost_preserves_equity(self):
        simulator = run_simulator([100.0] * 6, [0.9, 0.1] * 3, cost_bps=0.0)
        assert simulator._equity() == pytest.approx(100000.0, rel=1e-9)


class TestResultShape:
    def test_missing_columns_are_rejected(self):
        rows = pd.DataFrame({"timestamp": [pd.Timestamp("2026-01-05")], "symbol": ["AAA"]})
        with pytest.raises(ValueError, match="Missing columns"):
            simulate_probability_strategy(rows, 0.6, 0.4)

    def test_empty_input_returns_a_zeroed_result(self):
        rows = pd.DataFrame(columns=["timestamp", "symbol", "close", "probability_up"])
        result = simulate_probability_strategy(rows, 0.6, 0.4)
        assert result["trade_count"] == 0
        assert result["total_return"] == 0.0

    def test_result_carries_every_reported_metric(self):
        rows = build_rows({"AAA": [100.0, 101.0, 102.0]}, {"AAA": [0.9, 0.9, 0.9]})
        result = simulate_probability_strategy(rows, 0.6, 0.4, transaction_cost_bps=5.0)
        for key in (
            "trade_count", "exposure", "total_return", "annualized_return",
            "annualized_volatility", "sharpe", "max_drawdown", "equity_curve",
        ):
            assert key in result
        assert len(result["equity_curve"]) == 3
