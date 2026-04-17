from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_SYMBOLS = ("SPY", "QQQ", "AAPL", "MSFT", "NVDA")


@dataclass(slots=True)
class IBKRConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 15
    request_timeout: float = 120.0
    connect_retries: int = 3
    retry_delay_seconds: float = 3.0
    client_id_step: int = 1


@dataclass(slots=True)
class MarketDataConfig:
    symbols: tuple[str, ...] = DEFAULT_SYMBOLS
    duration: str = "90 D"
    bar_size: str = "5 mins"
    use_rth: bool = True
    max_duration_per_request: str | None = None
    regular_session_only: bool = True
    stale_after_minutes: int = 15


@dataclass(slots=True)
class ModelConfig:
    horizon_bars: int = 3
    positive_return_threshold: float = 0.001
    train_split: float = 0.7
    validation_split: float = 0.15
    walk_forward_splits: int = 3
    entry_probability: float | None = None
    exit_probability: float | None = None
    threshold_hysteresis: float = 0.06
    transaction_cost_bps: float = 1.0
    max_active_positions: int = 2
    model_path: Path = Path("artifacts/gradient_boosting_model.joblib")


@dataclass(slots=True)
class RiskConfig:
    starting_capital: float = 100000.0
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.20
    max_active_positions: int | None = None
    max_daily_trade_count: int | None = 12
    stop_loss_pct: float = 0.008
    take_profit_pct: float = 0.015
    max_daily_loss_pct: float = 0.02
    allow_unsafe_model: bool = False
    log_dir: Path = Path("logs")
