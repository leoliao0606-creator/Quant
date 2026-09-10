from __future__ import annotations

import argparse
from pathlib import Path

from ibkr_ml.config import (
    DEFAULT_SYMBOLS,
    IBKRConnectionConfig,
    MarketDataConfig,
    ModelConfig,
    RiskConfig,
)
from ibkr_ml.cache import fetch_frames
from ibkr_ml.data import connect_ib, fetch_historical_frame, resample_frame
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
    parser.add_argument(
        "--label-mode",
        choices=["absolute", "volatility_scaled", "direction"],
        default="absolute",
        help=(
            "How the forward return becomes a label. absolute: beat a fixed "
            "return. volatility_scaled: beat a multiple of current ATR%%, so a "
            "volatile stretch needs a bigger move. direction: only which way "
            "it went. The absolute form lets the model score well by "
            "predicting volatility instead of direction."
        ),
    )
    parser.add_argument("--positive-return-threshold", type=float, default=0.001)
    parser.add_argument(
        "--volatility-threshold-multiple",
        type=float,
        default=0.5,
        help="Multiple of ATR%% used as the label threshold in volatility_scaled mode.",
    )
    parser.add_argument("--max-duration-per-request", default=None)
    parser.add_argument(
        "--bar-timezone",
        default=None,
        help=(
            "IANA timezone of the bar timestamps TWS returns, e.g. UTC. Set "
            "this when TWS/IB Gateway runs on a machine that is not on US "
            "Eastern time, and pass the same value to paper_trade.py. The "
            "time-of-day feature reads this clock."
        ),
    )
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--walk-forward-splits", type=int, default=3)
    parser.add_argument("--entry-probability", type=float, default=None)
    parser.add_argument("--exit-probability", type=float, default=None)
    parser.add_argument("--threshold-hysteresis", type=float, default=0.06)
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=5.0,
        help=(
            "One-way cost in basis points applied on every position change: "
            "spread crossed by a market order, commission, and impact. "
            "Threshold selection reads this, so understating it picks entry "
            "thresholds that only look profitable."
        ),
    )
    parser.add_argument("--max-active-positions", type=int, default=2)
    parser.add_argument("--model-path", default="artifacts/gradient_boosting_model.joblib")
    parser.add_argument(
        "--cache-dir",
        default="data_cache",
        help=(
            "Directory for cached IBKR bars. A cached run needs no TWS "
            "session at all, and two experiments that read the same cache "
            "are comparable because they replay identical input. Pass an "
            "empty string to disable caching."
        ),
    )
    parser.add_argument(
        "--resample-to",
        default=None,
        help=(
            "Aggregate the fetched bars up to this bar size before training, "
            "e.g. '1 hour'. One cached download then serves several "
            "timescales, and experiments across timescales stay comparable "
            "because they are built from the same underlying prices. "
            "paper_trade.py must be given the same value."
        ),
    )
    parser.add_argument(
        "--market-symbol",
        default=None,
        help=(
            "Symbol to use as the broad-market reference, e.g. SPY. Adds "
            "excess return, beta, correlation and relative strength features. "
            "Without a reference the model only sees each stock's own history, "
            "which is why it ends up predicting volatility rather than direction."
        ),
    )
    parser.add_argument(
        "--sector-symbol",
        default=None,
        help="Symbol for the sector reference, e.g. XLK for technology.",
    )
    parser.add_argument(
        "--volatility-symbol",
        default=None,
        help="Symbol for the volatility-regime reference, e.g. VXX.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Re-download from IBKR and overwrite the cache.",
    )

    # The backtests run inside training replay the live decision rules, so
    # these have to match what paper_trade.py will be launched with. A model
    # tuned against a 0.8% stop behaves differently under a 2% one.
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-fraction", type=float, default=0.20)
    parser.add_argument("--max-daily-trade-count", type=int, default=12)
    parser.add_argument("--stop-loss-pct", type=float, default=0.008)
    parser.add_argument("--take-profit-pct", type=float, default=0.015)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.02)
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
        bar_timezone=args.bar_timezone,
    )
    model_config = ModelConfig(
        horizon_bars=args.horizon_bars,
        label_mode=args.label_mode,
        positive_return_threshold=args.positive_return_threshold,
        volatility_threshold_multiple=args.volatility_threshold_multiple,
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
    risk_config = RiskConfig(
        risk_per_trade=args.risk_per_trade,
        max_position_fraction=args.max_position_fraction,
        max_active_positions=args.max_active_positions,
        max_daily_trade_count=args.max_daily_trade_count,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_daily_loss_pct=args.max_daily_loss_pct,
    )

    def fetch_one(ib, symbol):
        try:
            return fetch_historical_frame(
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

    reference_symbols = {}
    if args.market_symbol:
        reference_symbols["mkt"] = args.market_symbol
    if args.sector_symbol:
        reference_symbols["sector"] = args.sector_symbol
    if args.volatility_symbol:
        reference_symbols["volx"] = args.volatility_symbol

    # A reference symbol may also be traded (SPY commonly is), so fetch the
    # union once. dict.fromkeys keeps the order and drops duplicates.
    wanted_symbols = list(dict.fromkeys([*market_config.symbols, *reference_symbols.values()]))

    frames = fetch_frames(
        symbols=wanted_symbols,
        duration=market_config.duration,
        bar_size=market_config.bar_size,
        use_rth=market_config.use_rth,
        max_duration_per_request=market_config.max_duration_per_request,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        refresh_cache=args.refresh_cache,
        connect=lambda: connect_ib(connection_config),
        fetch_one=fetch_one,
    )

    effective_bar_timezone = args.bar_timezone
    if args.resample_to:
        frames = {
            symbol: resample_frame(frame, args.resample_to, args.bar_timezone)
            for symbol, frame in frames.items()
        }
        # resample_frame returns US Eastern wall-clock timestamps, so declaring
        # a source zone again downstream would shift them a second time.
        effective_bar_timezone = None
        example = next(iter(frames.values()))
        print(
            f"Resampled {market_config.bar_size} bars to {args.resample_to}: "
            f"{len(example)} bars per symbol"
        )

    reference_frames = {key: frames[symbol] for key, symbol in reference_symbols.items()}
    training_frames = {symbol: frames[symbol] for symbol in market_config.symbols}
    if reference_symbols:
        print("Cross-asset references: " + ", ".join(
            f"{key}={symbol}" for key, symbol in reference_symbols.items()
        ))

    bundle = train_model_from_frames(
        training_frames,
        model_config,
        risk_config,
        bar_timezone=effective_bar_timezone,
        reference_frames=reference_frames or None,
        reference_symbols=reference_symbols,
        resample_to=args.resample_to,
    )
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
    print("Risk rules replayed in every backtest above:")
    print(f"  stop_loss_pct: {risk_config.stop_loss_pct}")
    print(f"  take_profit_pct: {risk_config.take_profit_pct}")
    print(f"  max_daily_trade_count: {risk_config.max_daily_trade_count}")
    print(f"  max_daily_loss_pct: {risk_config.max_daily_loss_pct}")
    print(f"  max_position_fraction: {risk_config.max_position_fraction}")
    print("Walk-forward summary:")
    for key, value in bundle["walk_forward_summary"].items():
        print(f"  {key}: {value}")
    print(f"Feature count: {len(bundle['feature_columns'])}")
    print("Top feature importances:")
    for row in bundle["feature_importances"][:10]:
        print(f"  {row['feature']}: {row['importance']}")


if __name__ == "__main__":
    main()
