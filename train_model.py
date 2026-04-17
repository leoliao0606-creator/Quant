from __future__ import annotations

import argparse
from pathlib import Path

from ibkr_ml.config import DEFAULT_SYMBOLS, IBKRConnectionConfig, MarketDataConfig, ModelConfig
from ibkr_ml.data import connect_ib, fetch_historical_frame
from ibkr_ml.modeling import train_model_from_frames


def parse_args():
    parser = argparse.ArgumentParser(description="Train a baseline ML model for IBKR paper trading.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--duration", default="90 D")
    parser.add_argument("--bar-size", default="5 mins")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=15)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=3.0)
    parser.add_argument("--client-id-step", type=int, default=1)
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--positive-return-threshold", type=float, default=0.001)
    parser.add_argument("--max-duration-per-request", default=None)
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--walk-forward-splits", type=int, default=3)
    parser.add_argument("--entry-probability", type=float, default=None)
    parser.add_argument("--exit-probability", type=float, default=None)
    parser.add_argument("--threshold-hysteresis", type=float, default=0.06)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--max-active-positions", type=int, default=2)
    parser.add_argument("--model-path", default="artifacts/gradient_boosting_model.joblib")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    connection_config = IBKRConnectionConfig(
        host=args.host,
        port=args.port,
        client_id=args.client_id,
        request_timeout=args.request_timeout,
        connect_retries=args.connect_retries,
        retry_delay_seconds=args.retry_delay_seconds,
        client_id_step=args.client_id_step,
    )
    market_config = MarketDataConfig(
        symbols=tuple(args.symbols),
        duration=args.duration,
        bar_size=args.bar_size,
        max_duration_per_request=args.max_duration_per_request,
    )
    model_config = ModelConfig(
        horizon_bars=args.horizon_bars,
        positive_return_threshold=args.positive_return_threshold,
        train_split=args.train_split,
        validation_split=args.validation_split,
        walk_forward_splits=args.walk_forward_splits,
        entry_probability=args.entry_probability,
        exit_probability=args.exit_probability,
        threshold_hysteresis=args.threshold_hysteresis,
        transaction_cost_bps=args.transaction_cost_bps,
        max_active_positions=args.max_active_positions,
        model_path=Path(args.model_path),
    )

    ib = connect_ib(connection_config)
    try:
        frames = {}
        for symbol in market_config.symbols:
            print(f"Fetching bars for {symbol}...")
            try:
                frames[symbol] = fetch_historical_frame(
                    ib=ib,
                    symbol=symbol,
                    duration=market_config.duration,
                    bar_size=market_config.bar_size,
                    use_rth=market_config.use_rth,
                    max_duration_per_request=market_config.max_duration_per_request,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to fetch historical bars for {symbol}: {exc}\n"
                    "If TWS shows 'connected from a different IP address', "
                    "fix the TWS/API session conflict first and rerun training."
                ) from exc
    finally:
        ib.disconnect()

    bundle = train_model_from_frames(frames, model_config)
    print(f"Saved model bundle to {model_config.model_path}")
    print("Selected thresholds:")
    for key, value in bundle["thresholds"].items():
        print(f"  {key}: {value}")
    print("Validation metrics:")
    for key, value in bundle["validation_metrics"].items():
        print(f"  {key}: {value}")
    print("Test metrics:")
    for key, value in bundle["test_metrics"].items():
        print(f"  {key}: {value}")
    print("Validation backtest:")
    for key, value in bundle["validation_backtest"].items():
        print(f"  {key}: {value}")
    print("Test backtest:")
    for key, value in bundle["test_backtest"].items():
        print(f"  {key}: {value}")
    print("Walk-forward summary:")
    for key, value in bundle["walk_forward_summary"].items():
        print(f"  {key}: {value}")
    print("Top feature importances:")
    for row in bundle["feature_importances"][:10]:
        print(f"  {row['feature']}: {row['importance']}")


if __name__ == "__main__":
    main()
