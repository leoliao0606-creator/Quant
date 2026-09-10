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


class TestThresholdActivityFloor:
    """A threshold that almost never fires is not a strategy.

    Staying flat cannot lose money, so an entry bar set high enough to suppress
    trading scores well on every risk-adjusted measure while doing nothing.
    Seen live: entry 0.69 traded 14 times in 54 days at 0.1% exposure, and its
    Sharpe of 0.53 cleared the deployment gate.
    """

    def rows_with_probabilities(self, probabilities, symbols=("AAA", "BBB"), n=300):
        """Prices that drift up, with probabilities drawn from a given range."""
        import numpy as np

        rng = np.random.default_rng(3)
        low, high = probabilities
        rows = []
        for symbol in symbols:
            price = 100.0
            for index in range(n):
                price *= 1.0 + rng.normal(0.0002, 0.001)
                rows.append(
                    {
                        "timestamp": pd.Timestamp("2026-01-05 09:30")
                        + pd.Timedelta(minutes=5 * index),
                        "symbol": symbol,
                        "close": price,
                        "probability_up": float(rng.uniform(low, high)),
                    }
                )
        return pd.DataFrame(rows)

    def select(self, rows, **overrides):
        from ibkr_ml.backtest import select_probability_thresholds

        kwargs = {
            "validation_rows": rows,
            "transaction_cost_bps": 1.0,
            "threshold_hysteresis": 0.06,
            "max_active_positions": 2,
        }
        kwargs.update(overrides)
        return select_probability_thresholds(**kwargs)

    def test_an_active_strategy_is_selected_and_marked_qualified(self):
        selection = self.select(self.rows_with_probabilities((0.40, 0.80)))
        assert selection["qualified"] is True
        assert selection["validation_backtest"]["trade_count"] >= 20
        assert selection["validation_backtest"]["exposure"] >= 0.005

    def test_too_little_data_to_judge_is_not_qualified(self, capsys):
        """Percentile search always finds a threshold that fires.

        Under the old absolute grid a series that never reached 0.45 produced
        no trades at all. A percentile is relative, so the only way to fail the
        floor now is genuinely too little activity to measure.
        """
        selection = self.select(self.rows_with_probabilities((0.40, 0.80), n=12))
        assert selection["qualified"] is False
        assert "no percentile reached" in selection["selection_note"]
        assert "no percentile reached" in capsys.readouterr().out

    def test_an_unqualified_selection_still_reports_usable_numbers(self):
        selection = self.select(self.rows_with_probabilities((0.40, 0.80), n=12))
        # The caller needs an entry/exit pair and a backtest either way; what
        # changes is that the result says not to trust them.
        assert 0.0 < selection["entry_probability"] < 1.0
        assert selection["validation_backtest"] is not None

    def test_the_floor_can_be_raised(self):
        rows = self.rows_with_probabilities((0.40, 0.80))
        assert self.select(rows, min_trade_count=1, min_exposure=0.0)["qualified"] is True
        assert self.select(rows, min_trade_count=100000)["qualified"] is False

    def test_exposure_and_trade_count_are_checked_separately(self):
        from ibkr_ml.backtest import _threshold_is_usable

        busy_but_flat = {"trade_count": 500, "exposure": 0.001}
        held_but_idle = {"trade_count": 3, "exposure": 0.90}
        healthy = {"trade_count": 100, "exposure": 0.30}

        assert "exposure" in _threshold_is_usable(busy_but_flat, 20, 0.05)
        assert "trades" in _threshold_is_usable(held_but_idle, 20, 0.05)
        assert _threshold_is_usable(healthy, 20, 0.05) is None


class TestPercentileThresholdSearch:
    """Thresholds are searched as percentiles of the model's own distribution.

    An absolute grid searches a different thing for each model, because
    probability scales differ: a class-balanced model that prints 0.84 at its
    most confident is not comparable with one that reaches 0.99.
    """

    def rows(self, low, high, n=400, seed=11):
        import numpy as np

        rng = np.random.default_rng(seed)
        out = []
        for symbol in ("AAA", "BBB"):
            price = 100.0
            for index in range(n):
                price *= 1.0 + rng.normal(0.0003, 0.001)
                out.append(
                    {
                        "timestamp": pd.Timestamp("2026-01-05 09:30")
                        + pd.Timedelta(minutes=5 * index),
                        "symbol": symbol,
                        "close": price,
                        "probability_up": float(rng.uniform(low, high)),
                    }
                )
        return pd.DataFrame(out)

    def select(self, rows, **overrides):
        from ibkr_ml.backtest import select_probability_thresholds

        kwargs = {
            "validation_rows": rows,
            "transaction_cost_bps": 1.0,
            "threshold_hysteresis": 0.06,
            "max_active_positions": 2,
        }
        kwargs.update(overrides)
        return select_probability_thresholds(**kwargs)

    def test_the_selection_reports_which_percentile_it_used(self):
        selection = self.select(self.rows(0.30, 0.90))
        assert 0.0 < selection["entry_percentile"] < 1.0
        assert selection["probability_ceiling"] > selection["entry_probability"]

    def test_the_entry_probability_is_that_percentile_of_the_data(self):
        rows = self.rows(0.30, 0.90)
        selection = self.select(rows)
        expected = rows["probability_up"].quantile(selection["entry_percentile"])
        assert selection["entry_probability"] == pytest.approx(expected)

    def test_two_models_on_different_scales_get_comparable_selectivity(self):
        """The same shape on a compressed scale must select the same fraction.

        This is what an absolute grid could not do: shifting a model's output
        into a narrower band changed which rows were selected, even though the
        ranking - the only thing that matters - was identical.
        """
        wide = self.rows(0.20, 0.95)
        narrow = wide.copy()
        narrow["probability_up"] = 0.40 + wide["probability_up"] * 0.30

        wide_selection = self.select(wide)
        narrow_selection = self.select(narrow)

        wide_share = (wide["probability_up"] >= wide_selection["entry_probability"]).mean()
        narrow_share = (narrow["probability_up"] >= narrow_selection["entry_probability"]).mean()
        assert wide_share == pytest.approx(narrow_share, abs=0.02)

    def test_the_candidate_grid_can_be_supplied(self):
        selection = self.select(self.rows(0.30, 0.90), candidate_percentiles=(0.50,))
        assert selection["entry_percentile"] == 0.50

    def test_the_ceiling_sits_inside_the_observed_range(self):
        rows = self.rows(0.30, 0.90)
        selection = self.select(rows)
        assert selection["probability_ceiling"] <= rows["probability_up"].max()
        assert selection["probability_ceiling"] > rows["probability_up"].median()


class TestPurgedSplits:
    """Labels read ahead, so rows next to a split boundary leak across it.

    The last horizon_bars rows before a boundary carry targets built from
    prices that land in the next split. Training on them lets the model see,
    through its own label, data it is about to be scored on - and validation
    is what selects the entry threshold.
    """

    def test_purge_scales_with_horizon_and_symbol_count(self):
        from ibkr_ml.modeling import _purge_rows

        assert _purge_rows(None, 0, 12, 69) == 828
        assert _purge_rows(None, 0, 3, 5) == 15
        assert _purge_rows(None, 0, 0, 69) == 0

    def test_training_shrinks_but_validation_does_not(self):
        from ibkr_ml.modeling import _split_indices

        train_purged, train_end, validation_end = _split_indices(
            row_count=10000, train_split=0.7, validation_split=0.15, purge=100
        )
        assert train_purged == train_end - 100
        assert train_end == 7000          # boundary itself is unmoved
        assert validation_end == 8500     # validation keeps every row

    def test_no_purge_leaves_the_split_untouched(self):
        from ibkr_ml.modeling import _split_indices

        train_purged, train_end, _ = _split_indices(
            row_count=10000, train_split=0.7, validation_split=0.15, purge=0
        )
        assert train_purged == train_end

    def test_an_oversized_purge_keeps_half_the_training_set(self, capsys):
        """A purge bigger than the training set is a configuration problem.

        It means the label horizon is long relative to the data. Silently
        training on one row would be worse than saying so.
        """
        from ibkr_ml.modeling import _split_indices

        train_purged, train_end, validation_end = _split_indices(
            row_count=1000, train_split=0.7, validation_split=0.15, purge=100000
        )
        assert train_purged == train_end // 2
        assert train_purged < validation_end
        assert "horizon is long relative to the data" in capsys.readouterr().out
