from __future__ import annotations

import argparse
from pathlib import Path

from ibkr_ml.backtest import simulate_probability_strategy
from ibkr_ml.modeling import load_model_bundle


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest a trained model on saved validation/test predictions.")
    parser.add_argument("--model-path", default="artifacts/gradient_boosting_model.joblib")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--entry-probability", type=float, default=None)
    parser.add_argument("--exit-probability", type=float, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=None)
    parser.add_argument("--max-active-positions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(Path(args.model_path))

    prediction_key = f"{args.split}_predictions"
    prediction_rows = bundle.get(prediction_key)
    if prediction_rows is None:
        raise RuntimeError(
            f"Model bundle at {args.model_path} does not contain saved {args.split} predictions. "
            "Retrain the model with the updated training pipeline first."
        )

    thresholds = bundle.get("thresholds", {})
    model_config = bundle.get("model_config", {})
    entry_probability = (
        args.entry_probability
        if args.entry_probability is not None
        else thresholds.get("entry_probability", model_config.get("entry_probability", 0.58))
    )
    exit_probability = (
        args.exit_probability
        if args.exit_probability is not None
        else thresholds.get("exit_probability", model_config.get("exit_probability", 0.48))
    )
    transaction_cost_bps = (
        args.transaction_cost_bps
        if args.transaction_cost_bps is not None
        else model_config.get("transaction_cost_bps", 1.0)
    )
    max_active_positions = (
        args.max_active_positions
        if args.max_active_positions is not None
        else model_config.get("max_active_positions", 2)
    )

    result = simulate_probability_strategy(
        prediction_rows=prediction_rows,
        entry_probability=float(entry_probability),
        exit_probability=float(exit_probability),
        transaction_cost_bps=float(transaction_cost_bps),
        max_active_positions=max_active_positions,
    )

    print(f"Split: {args.split}")
    print(f"Entry probability: {result['entry_probability']:.3f}")
    print(f"Exit probability: {result['exit_probability']:.3f}")
    print(f"Max active positions: {result['max_active_positions']}")
    print(f"Trade count: {result['trade_count']}")
    print(f"Exposure: {result['exposure']:.3f}")
    print(f"Total return: {result['total_return']:.4f}")
    print(f"Annualized return: {result['annualized_return']:.4f}")
    print(f"Annualized volatility: {result['annualized_volatility']:.4f}")
    print(f"Sharpe: {result['sharpe']:.4f}")
    print(f"Max drawdown: {result['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()
