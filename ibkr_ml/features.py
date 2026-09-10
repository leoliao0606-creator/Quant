from __future__ import annotations


FEATURE_COLUMNS = [
    "gap_1",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "vol_12",
    "vol_24",
    "vol_ratio_12_24",
    "dist_ema_8",
    "dist_ema_21",
    "ema_spread_8_21",
    "rsi_14",
    "atr_14_pct",
    "range_pct",
    "close_location",
    "breakout_high_20",
    "breakout_low_20",
    "volume_z_20",
    "tod_sin",
    "tod_cos",
]


# Cross-asset features: what this stock is doing *relative to* the market, its
# sector, and the volatility regime. The single-asset block above can only
# describe a stock's own history, which is why a model trained on it alone
# settles on volatility - given a label like "will it move more than 0.1%",
# volatility is the most predictable thing available, and it says nothing about
# direction. Relative performance is where directional information lives.
MARKET_FEATURE_COLUMNS = [
    "mkt_ret_1",
    "mkt_ret_6",
    "mkt_ret_24",
    "excess_ret_1",
    "excess_ret_6",
    "excess_ret_24",
    "beta_60",
    "corr_60",
    "rel_strength_20",
    "mkt_vol_12",
    "vol_ratio_to_mkt",
    "mkt_dist_ema_21",
]

SECTOR_FEATURE_COLUMNS = [
    "sector_ret_6",
    "sector_excess_ret_6",
    "sector_rel_strength_20",
]

VOLATILITY_INDEX_FEATURE_COLUMNS = [
    "volx_ret_6",
    "volx_z_60",
]

# Keys accepted in the reference_frames mapping, in the order their columns are
# appended to the feature matrix.
REFERENCE_KINDS = (
    ("mkt", MARKET_FEATURE_COLUMNS),
    ("sector", SECTOR_FEATURE_COLUMNS),
    ("volx", VOLATILITY_INDEX_FEATURE_COLUMNS),
)


def cross_asset_feature_columns(reference_frames) -> list[str]:
    """Columns a given set of reference series produces.

    Returned in a fixed order so a model's feature matrix stays stable across
    runs that supply the same references.
    """
    if not reference_frames:
        return []

    columns: list[str] = []
    for key, kind_columns in REFERENCE_KINDS:
        if reference_frames.get(key) is not None:
            columns.extend(kind_columns)
    return columns


def feature_columns(reference_frames=None) -> list[str]:
    """The full ordered feature list for a given reference configuration."""
    return [*FEATURE_COLUMNS, *cross_asset_feature_columns(reference_frames)]


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'numpy'. Install requirements.txt first.") from exc
    return np


def _load_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'pandas'. Install requirements.txt first.") from exc
    return pd


# Everything that reads a clock - the time-of-day feature here, the daily
# grouping in the backtest - is expressed in this zone.
EASTERN_TIMEZONE = "America/New_York"

# Minutes from midnight to the 09:30 US equities open, and the length of a
# regular session in minutes.
SESSION_OPEN_MINUTES = 570
SESSION_LENGTH_MINUTES = 390


def parse_bar_timestamps(timestamps):
    """Parse bar timestamps into one datetime dtype, offsets and all.

    A year of US equity bars spans a daylight-saving change, so IBKR stamps the
    first part -04:00 and the rest -05:00. While the series stays inside pandas
    it is a single tz-aware column and the change is invisible. Once it goes
    through any text format - a CSV cache, a JSON log - the type is gone and
    what comes back is a mix of two offsets, which pandas refuses to parse into
    one dtype: "Tz-aware datetime.datetime cannot be converted to datetime64
    unless utc=True".

    Parsing through UTC is the way out. It is only a storage detail: the wall
    clock the caller wants is restored by converting to the target zone.
    """
    import warnings

    pd = _load_pandas()
    stamps = timestamps if hasattr(timestamps, "dt") else pd.Series(timestamps)

    parsed = None
    try:
        with warnings.catch_warnings():
            # The mixed-offset path warns here and is handled right below, so
            # the warning would only be noise.
            warnings.simplefilter("ignore", FutureWarning)
            parsed = pd.to_datetime(stamps)
    except (ValueError, TypeError):
        parsed = None

    # pandas does not reliably raise on mixed offsets: some versions raise,
    # others return an object-dtype Series of datetimes with a FutureWarning.
    # Check what actually came back rather than trusting the exception.
    if parsed is None or not pd.api.types.is_datetime64_any_dtype(parsed):
        parsed = pd.to_datetime(stamps, utc=True)
    return parsed


def to_eastern_naive(timestamps, bar_timezone: str | None = None):
    """Return bar timestamps as naive US Eastern wall-clock times.

    TWS stamps bars with the wall clock of the machine it runs on and drops the
    timezone, so one and the same bar is labelled 14:30 on a US Eastern host and
    19:30 on a UTC host. The time-of-day feature below reads that clock, so
    without this the same bar produces different features depending on where
    TWS happens to run - and a model trained on one machine then scores wrong
    on another, silently, with no metric anywhere revealing it.

    Passing None keeps the historical behaviour of treating naive stamps as
    already being US Eastern.
    """
    pd = _load_pandas()
    stamps = parse_bar_timestamps(timestamps)
    has_timezone = getattr(stamps.dt, "tz", None) is not None

    if bar_timezone is None:
        if has_timezone:
            return stamps.dt.tz_convert(EASTERN_TIMEZONE).dt.tz_localize(None)
        return stamps

    if not has_timezone:
        # ambiguous/nonexistent only bite on DST boundaries; resolving them
        # beats raising in the middle of a training run.
        stamps = stamps.dt.tz_localize(
            bar_timezone, ambiguous=False, nonexistent="shift_forward"
        )
    return stamps.dt.tz_convert(EASTERN_TIMEZONE).dt.tz_localize(None)


def _rsi(close, period: int = 14):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    relative_strength = avg_gain / avg_loss.replace(0.0, float("nan"))
    return 100.0 - (100.0 / (1.0 + relative_strength))


def _atr(high, low, close, period: int = 14):
    previous_close = close.shift(1)
    true_range = (high - low).to_frame("hl")
    true_range["hc"] = (high - previous_close).abs()
    true_range["lc"] = (low - previous_close).abs()
    return true_range.max(axis=1).rolling(period).mean()


def _align_reference(frame, reference_frame, bar_timezone: str | None, prefix: str):
    """Attach a reference series' close price, aligned without look-ahead.

    merge_asof with direction="backward" pairs each bar with the most recent
    reference bar at or before it. An ordinary join on the timestamp would drop
    rows whenever the two series disagree - a halted symbol, a late first
    print, an ETF that did not trade in that minute - and a forward-filling
    join would pair a bar with a reference bar that had not happened yet, which
    is look-ahead of exactly the kind that makes a backtest look good and a
    live run lose money.
    """
    pd = _load_pandas()

    reference = reference_frame[["timestamp", "close"]].copy()
    reference["timestamp"] = to_eastern_naive(reference["timestamp"], bar_timezone)
    reference = (
        reference.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .rename(columns={"close": f"{prefix}_close"})
        .reset_index(drop=True)
    )
    return pd.merge_asof(
        frame.sort_values("timestamp"),
        reference,
        on="timestamp",
        direction="backward",
    )


def _add_market_features(frame, np):
    """Performance relative to the broad market."""
    market_close = frame["mkt_close"]
    frame["mkt_ret_1"] = market_close.pct_change(1)
    frame["mkt_ret_6"] = market_close.pct_change(6)
    frame["mkt_ret_24"] = market_close.pct_change(24)

    # Excess return is the part of the move that is this stock's own, with the
    # market's move taken out. This is the directional signal the single-asset
    # feature set had no way to express.
    frame["excess_ret_1"] = frame["ret_1"] - frame["mkt_ret_1"]
    frame["excess_ret_6"] = frame["ret_6"] - frame["mkt_ret_6"]
    frame["excess_ret_24"] = frame["ret_24"] - frame["mkt_ret_24"]

    stock_return = frame["ret_1"]
    market_return = frame["mkt_ret_1"]
    market_variance = market_return.rolling(60).var()
    frame["beta_60"] = stock_return.rolling(60).cov(market_return) / market_variance.replace(0.0, np.nan)
    frame["corr_60"] = stock_return.rolling(60).corr(market_return)

    ratio = frame["close"] / market_close.replace(0.0, np.nan)
    frame["rel_strength_20"] = ratio / ratio.shift(20) - 1.0

    frame["mkt_vol_12"] = market_return.rolling(12).std()
    frame["vol_ratio_to_mkt"] = frame["vol_12"] / frame["mkt_vol_12"].replace(0.0, np.nan)

    market_ema_21 = market_close.ewm(span=21, adjust=False).mean()
    frame["mkt_dist_ema_21"] = market_close / market_ema_21 - 1.0
    return frame


def _add_sector_features(frame, np):
    """Performance relative to the stock's sector."""
    sector_close = frame["sector_close"]
    frame["sector_ret_6"] = sector_close.pct_change(6)
    frame["sector_excess_ret_6"] = frame["ret_6"] - frame["sector_ret_6"]

    ratio = frame["close"] / sector_close.replace(0.0, np.nan)
    frame["sector_rel_strength_20"] = ratio / ratio.shift(20) - 1.0
    return frame


def _add_volatility_index_features(frame, np):
    """Where the volatility regime sits, from a volatility proxy series."""
    volatility_close = frame["volx_close"]
    frame["volx_ret_6"] = volatility_close.pct_change(6)

    rolling_mean = volatility_close.rolling(60).mean()
    rolling_std = volatility_close.rolling(60).std()
    frame["volx_z_60"] = (volatility_close - rolling_mean) / rolling_std.replace(0.0, np.nan)
    return frame


def _add_cross_asset_features(frame, reference_frames, bar_timezone: str | None, np):
    builders = {
        "mkt": _add_market_features,
        "sector": _add_sector_features,
        "volx": _add_volatility_index_features,
    }
    for key, _ in REFERENCE_KINDS:
        reference_frame = reference_frames.get(key)
        if reference_frame is None:
            continue
        frame = _align_reference(frame, reference_frame, bar_timezone, key)
        frame = builders[key](frame, np)
    return frame


def build_feature_frame(price_frame, bar_timezone: str | None = None, reference_frames=None):
    """Build the feature matrix for one symbol.

    bar_timezone is the IANA zone the raw bar timestamps are expressed in, i.e.
    the timezone of the machine running TWS. Timestamps are converted to US
    Eastern here, so every downstream consumer reads the same clock regardless
    of where the data was fetched.

    reference_frames optionally supplies other series to measure this one
    against: "mkt" for the broad market, "sector" for the sector, "volx" for a
    volatility proxy. Each key that is present adds its block of cross-asset
    columns; see cross_asset_feature_columns.
    """
    np = _load_numpy()
    pd = _load_pandas()

    frame = price_frame.copy()
    frame["timestamp"] = to_eastern_naive(frame["timestamp"], bar_timezone)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    previous_close = frame["close"].shift(1)

    frame["gap_1"] = frame["open"] / previous_close - 1.0
    frame["ret_1"] = frame["close"].pct_change(1)
    frame["ret_3"] = frame["close"].pct_change(3)
    frame["ret_6"] = frame["close"].pct_change(6)
    frame["ret_12"] = frame["close"].pct_change(12)
    frame["ret_24"] = frame["close"].pct_change(24)

    frame["vol_12"] = frame["ret_1"].rolling(12).std()
    frame["vol_24"] = frame["ret_1"].rolling(24).std()
    frame["vol_ratio_12_24"] = frame["vol_12"] / frame["vol_24"].replace(0.0, np.nan)

    ema_8 = frame["close"].ewm(span=8, adjust=False).mean()
    ema_21 = frame["close"].ewm(span=21, adjust=False).mean()
    frame["dist_ema_8"] = (frame["close"] / ema_8) - 1.0
    frame["dist_ema_21"] = (frame["close"] / ema_21) - 1.0
    frame["ema_spread_8_21"] = ema_8 / ema_21 - 1.0

    frame["rsi_14"] = _rsi(frame["close"], period=14) / 100.0
    frame["atr_14_pct"] = _atr(frame["high"], frame["low"], frame["close"], period=14) / frame["close"]
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["close"]
    intrabar_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
    frame["close_location"] = (frame["close"] - frame["low"]) / intrabar_range - 0.5
    frame["breakout_high_20"] = frame["close"] / frame["high"].rolling(20).max() - 1.0
    frame["breakout_low_20"] = frame["close"] / frame["low"].rolling(20).min() - 1.0

    volume_mean = frame["volume"].rolling(20).mean()
    volume_std = frame["volume"].rolling(20).std()
    frame["volume_z_20"] = (frame["volume"] - volume_mean) / volume_std.replace(0.0, np.nan)

    # Safe to read the clock directly now that build_feature_frame has put the
    # timestamps in US Eastern.
    minutes_of_day = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    minutes_from_open = minutes_of_day - SESSION_OPEN_MINUTES
    clipped_minutes = minutes_from_open.clip(lower=0, upper=SESSION_LENGTH_MINUTES)
    frame["tod_sin"] = np.sin(2.0 * np.pi * clipped_minutes / SESSION_LENGTH_MINUTES)
    frame["tod_cos"] = np.cos(2.0 * np.pi * clipped_minutes / SESSION_LENGTH_MINUTES)

    if reference_frames:
        frame = _add_cross_asset_features(frame, reference_frames, bar_timezone, np)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


LABEL_MODES = ("absolute", "volatility_scaled", "direction")


def build_target(
    frame,
    label_mode: str,
    positive_return_threshold: float,
    volatility_threshold_multiple: float,
):
    """Turn the forward return into the binary label the model is fitted on.

    This choice matters more than it looks. Under an absolute threshold - "did
    it move more than 0.1% in three bars" - the cheapest way for a model to
    raise its AUC is to predict *when* the market moves a lot rather than
    *which way*: in a volatile stretch both directions clear 0.1% more often.
    A model trained that way ends up putting its weight on ATR and range, which
    is exactly what happened here (88% of the importance sat on volatility
    features) and it earns nothing, because trading needs direction.

    volatility_scaled removes that free lunch by scaling the bar with current
    volatility, so a volatile stretch needs a proportionally larger move to
    count as a positive. direction drops the size question altogether.
    """
    if label_mode == "direction":
        return (frame["future_return"] > 0.0).astype(int)
    if label_mode == "volatility_scaled":
        threshold = volatility_threshold_multiple * frame["atr_14_pct"]
        return (frame["future_return"] > threshold).astype(int)
    if label_mode == "absolute":
        return (frame["future_return"] > positive_return_threshold).astype(int)
    raise ValueError(
        f"Unknown label_mode {label_mode!r}. Use one of {', '.join(LABEL_MODES)}."
    )


def build_labeled_rows(
    symbol: str,
    price_frame,
    horizon_bars: int,
    positive_return_threshold: float,
    bar_timezone: str | None = None,
    reference_frames=None,
    label_mode: str = "absolute",
    volatility_threshold_multiple: float = 0.5,
):
    frame = build_feature_frame(price_frame, bar_timezone, reference_frames)
    frame["symbol"] = symbol
    frame["future_return"] = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
    frame["next_bar_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    frame["target"] = build_target(
        frame, label_mode, positive_return_threshold, volatility_threshold_multiple
    )

    columns = [
        "timestamp",
        "symbol",
        "close",
        *feature_columns(reference_frames),
        "future_return",
        "next_bar_return",
        "target",
    ]
    rows = frame[columns].dropna().reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"Not enough history to build features for symbol {symbol}.")
    return rows


def build_latest_feature_row(
    symbol: str,
    price_frame,
    bar_timezone: str | None = None,
    reference_frames=None,
):
    frame = build_feature_frame(price_frame, bar_timezone, reference_frames)
    frame["symbol"] = symbol

    wanted = ["timestamp", "symbol", "close", *feature_columns(reference_frames)]
    latest = frame[wanted].dropna().tail(1)
    if latest.empty:
        raise ValueError(f"Not enough history to score symbol {symbol}.")
    return latest.reset_index(drop=True)
