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
from ibkr_ml.execution import IBKRPaperTrader


def parse_args():
    parser = argparse.ArgumentParser(description="Run the IBKR ML paper-trading baseline.")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--duration", default="5 D")
    parser.add_argument("--bar-size", default="5 mins")
    parser.add_argument("--stale-after-minutes", type=int, default=15)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=15)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--connect-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=3.0)
    parser.add_argument("--client-id-step", type=int, default=1)
    parser.add_argument("--model-path", default="artifacts/gradient_boosting_model.joblib")
    parser.add_argument("--entry-probability", type=float, default=None)
    parser.add_argument("--exit-probability", type=float, default=None)
    parser.add_argument("--risk-per-trade", type=float, default=0.01)
    parser.add_argument("--max-position-fraction", type=float, default=0.20)
    parser.add_argument("--max-active-positions", type=int, default=None)
    parser.add_argument("--max-daily-trade-count", type=int, default=12)
    parser.add_argument("--stop-loss-pct", type=float, default=0.008)
    parser.add_argument("--take-profit-pct", type=float, default=0.015)
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.02)
    parser.add_argument("--allow-unsafe-model", action="store_true")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        stale_after_minutes=args.stale_after_minutes,
    )
    model_config = ModelConfig(
        entry_probability=args.entry_probability,
        exit_probability=args.exit_probability,
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
        allow_unsafe_model=args.allow_unsafe_model,
        log_dir=Path(args.log_dir),
    )

    trader = IBKRPaperTrader(
        connection_config=connection_config,
        market_config=market_config,
        model_config=model_config,
        risk_config=risk_config,
    )

    if args.once:
        trader.run_once(dry_run=args.dry_run)
    else:
        trader.run_forever(interval_seconds=args.interval_seconds, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
