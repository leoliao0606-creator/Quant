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


def build_feature_frame(price_frame):
    np = _load_numpy()
    pd = _load_pandas()

    frame = price_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
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

    minutes_from_open = (frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute) - 570
    clipped_minutes = minutes_from_open.clip(lower=0, upper=390)
    frame["tod_sin"] = np.sin(2.0 * np.pi * clipped_minutes / 390.0)
    frame["tod_cos"] = np.cos(2.0 * np.pi * clipped_minutes / 390.0)

    frame = frame.replace([np.inf, -np.inf], np.nan)
    return frame


def build_labeled_rows(symbol: str, price_frame, horizon_bars: int, positive_return_threshold: float):
    frame = build_feature_frame(price_frame)
    frame["symbol"] = symbol
    frame["future_return"] = frame["close"].shift(-horizon_bars) / frame["close"] - 1.0
    frame["next_bar_return"] = frame["close"].shift(-1) / frame["close"] - 1.0
    frame["target"] = (frame["future_return"] > positive_return_threshold).astype(int)

    columns = [
        "timestamp",
        "symbol",
        "close",
        *FEATURE_COLUMNS,
        "future_return",
        "next_bar_return",
        "target",
    ]
    rows = frame[columns].dropna().reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"Not enough history to build features for symbol {symbol}.")
    return rows


def build_latest_feature_row(symbol: str, price_frame):
    frame = build_feature_frame(price_frame)
    frame["symbol"] = symbol

    latest = frame[["timestamp", "symbol", "close", *FEATURE_COLUMNS]].dropna().tail(1)
    if latest.empty:
        raise ValueError(f"Not enough history to score symbol {symbol}.")
    return latest.reset_index(drop=True)
