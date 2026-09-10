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
    # Timezone of the naive bar timestamps returned by TWS/IB Gateway with
    # formatDate=1. Those timestamps carry the wall-clock time of the machine
    # running TWS, which is not necessarily US Eastern. Leave as None to keep
    # the historical behaviour of treating them as US Eastern.
    bar_timezone: str | None = None
    # A bar timestamp this many minutes in the future is treated as a data
    # fault instead of a fresh bar, so a timezone mistake cannot silently
    # disable the staleness guard.
    future_bar_tolerance_minutes: float = 1.0


@dataclass(slots=True)
class ModelConfig:
    horizon_bars: int = 3
    # How the forward return becomes a label. "absolute" compares it with
    # positive_return_threshold; "volatility_scaled" compares it with
    # volatility_threshold_multiple * ATR%, so a volatile stretch needs a
    # proportionally bigger move; "direction" only asks which way it went.
    # The absolute form lets a model score well by predicting volatility
    # instead of direction, which earns nothing.
    label_mode: str = "absolute"
    positive_return_threshold: float = 0.001
    volatility_threshold_multiple: float = 0.5
    train_split: float = 0.7
    validation_split: float = 0.15
    walk_forward_splits: int = 3
    entry_probability: float | None = None
    exit_probability: float | None = None
    threshold_hysteresis: float = 0.06
    # One-way cost in basis points (1 bps = 0.01%), charged on every position
    # change. The old default of 1.0 covered roughly the quoted spread and
    # nothing else. A market order actually pays: the full bid-ask spread it
    # crosses (about 0.2 bps on SPY, 1-2 on AAPL/MSFT, 3-5 on NVDA when it
    # moves), IBKR commission (USD 0.005 per share, USD 1.00 minimum, which is
    # 0.5 bps on a USD 20,000 order), and market impact. This matters more than
    # it looks: the label is a 0.1% move over three bars, so a round trip at
    # 5 bps each way already eats the entire edge the model is trained to find.
    transaction_cost_bps: float = 5.0
    max_active_positions: int = 2
    model_path: Path = Path("artifacts/gradient_boosting_model.joblib")


@dataclass(slots=True)
class RiskConfig:
    starting_capital: float = 100000.0
    # Note that risk_per_trade only binds when it is smaller than
    # max_position_fraction * stop_loss_pct (0.20 * 0.008 = 0.0016 by default).
    # Above that the notional cap always wins and every position is simply
    # max_position_fraction of equity, whatever this value says.
    risk_per_trade: float = 0.01
    max_position_fraction: float = 0.20
    max_active_positions: int | None = None
    max_daily_trade_count: int | None = 12
    stop_loss_pct: float = 0.008
    take_profit_pct: float = 0.015
    max_daily_loss_pct: float = 0.02
    allow_unsafe_model: bool = False
    # Send entries as a bracket: a market parent with a stop-loss and a
    # take-profit child that live at IBKR. The in-process stop in strategy.py
    # only fires while the loop is alive and only sees a bar close every cycle,
    # so it cannot protect a position against a crash, a lost connection, or a
    # move that happens between two cycles.
    use_bracket_orders: bool = True
    # Close every position before the session ends. The model's label is the
    # return three five-minute bars ahead, so a position carried overnight sits
    # far outside the horizon it was trained on and is exposed to the opening
    # gap, which no intraday stop can act on.
    flatten_before_close: bool = True
    # US Eastern wall clock. From this time until SESSION_END_ET the loop only
    # closes positions; it opens none. Leave enough room before the close for
    # several cycles, so one failed cycle does not carry the book overnight.
    flatten_time_et: str = "15:45"
    log_dir: Path = Path("logs")
