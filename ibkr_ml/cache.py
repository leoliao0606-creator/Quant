"""On-disk cache for the historical bars fetched from IBKR.

Pulling 360 days of five-minute bars for five symbols takes about ten minutes
and holds the TWS session the whole time. Every experiment that only changes a
label definition or a feature then pays that cost again, and - worse - works on
a slightly different window than the previous run, so two results cannot be
compared. Caching the frames makes an experiment cheap and makes runs
comparable, because they replay byte-identical input.

CSV rather than Parquet on purpose: no extra dependency to install on the
Raspberry Pi, and the files can be inspected with ordinary tools.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .features import parse_bar_timestamps


def _load_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'pandas'. Install requirements.txt first.") from exc
    return pd


def _slug(text: str) -> str:
    """Make a string safe for a filename without losing what it said."""
    return "".join(char if char.isalnum() else "_" for char in str(text)).strip("_")


def cache_key(symbol: str, duration: str, bar_size: str, use_rth: bool) -> str:
    """Identity of a cached pull.

    Everything that changes the returned bars belongs in the key, so a run that
    asks for different data can never silently read another run's file.
    """
    return f"{_slug(symbol)}__{_slug(duration)}__{_slug(bar_size)}__rth{int(bool(use_rth))}"


def _paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    return cache_dir / f"{key}.csv", cache_dir / f"{key}.json"


def load_cached_frame(cache_dir, symbol: str, duration: str, bar_size: str, use_rth: bool):
    """Return the cached frame and its metadata, or (None, None) when absent."""
    pd = _load_pandas()
    cache_dir = Path(cache_dir)
    data_path, meta_path = _paths(cache_dir, cache_key(symbol, duration, bar_size, use_rth))
    if not data_path.exists():
        return None, None

    # Parsed separately rather than via parse_dates: a cache written before
    # timestamps were normalised to UTC holds two different offsets, which
    # read_csv refuses outright.
    frame = pd.read_csv(data_path)
    frame["timestamp"] = parse_bar_timestamps(frame["timestamp"])
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt sidecar must not hide usable bars; the age simply
            # becomes unknown.
            metadata = {}
    return frame, metadata


def save_cached_frame(cache_dir, symbol: str, duration: str, bar_size: str, use_rth: bool, frame) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = _paths(cache_dir, cache_key(symbol, duration, bar_size, use_rth))

    # Store timezone-aware timestamps as UTC so the file has one offset. A
    # year of bars spans a daylight-saving change, and writing the local
    # offsets produces a CSV mixing -04:00 and -05:00 that pandas cannot read
    # back into a single dtype.
    stored = frame.copy()
    stamps = parse_bar_timestamps(stored["timestamp"])
    if getattr(stamps.dt, "tz", None) is not None:
        stored["timestamp"] = stamps.dt.tz_convert("UTC")
    stored.to_csv(data_path, index=False)
    metadata = {
        "symbol": symbol,
        "duration": duration,
        "bar_size": bar_size,
        "use_rth": bool(use_rth),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(frame)),
        "first_timestamp": str(frame["timestamp"].iloc[0]) if len(frame) else None,
        "last_timestamp": str(frame["timestamp"].iloc[-1]) if len(frame) else None,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data_path


def cache_age_days(metadata) -> float | None:
    """How old the cached pull is, in days, or None when it cannot be told."""
    if not metadata or not metadata.get("fetched_at"):
        return None
    try:
        fetched_at = datetime.fromisoformat(str(metadata["fetched_at"]))
    except ValueError:
        return None
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - fetched_at).total_seconds() / 86400.0


def fetch_frames(
    *,
    symbols,
    duration: str,
    bar_size: str,
    use_rth: bool,
    max_duration_per_request: str | None,
    cache_dir=None,
    refresh_cache: bool = False,
    connect,
    fetch_one,
    stale_after_days: float = 3.0,
):
    """Return {symbol: frame}, reading the cache and fetching only what is missing.

    ``connect`` is called lazily and only when at least one symbol actually has
    to be downloaded, so a fully cached run never opens a TWS session at all -
    which also means it cannot collide with a session held elsewhere.

    A cache older than ``stale_after_days`` is still used, with a warning: the
    bars in it are correct history, they just stop earlier than a fresh pull
    would. Deciding to refresh is left to the caller, because a deliberately
    frozen dataset is exactly what makes two experiments comparable.
    """
    frames = {}
    missing = []

    for symbol in symbols:
        if cache_dir is None or refresh_cache:
            missing.append(symbol)
            continue

        frame, metadata = load_cached_frame(cache_dir, symbol, duration, bar_size, use_rth)
        if frame is None or frame.empty:
            missing.append(symbol)
            continue

        age_days = cache_age_days(metadata)
        age_text = "unknown age" if age_days is None else f"{age_days:.1f} days old"
        print(f"Using cached bars for {symbol}: {len(frame)} rows, {age_text}")
        if age_days is not None and age_days > stale_after_days:
            print(
                f"  note: this cache is {age_days:.1f} days old, so it ends "
                f"{age_days:.1f} days before today. Pass --refresh-cache to re-download."
            )
        frames[symbol] = frame

    if not missing:
        return frames

    ib = connect()
    try:
        for symbol in missing:
            print(f"Fetching bars for {symbol}...")
            frame = fetch_one(ib, symbol)
            frames[symbol] = frame
            if cache_dir is not None:
                path = save_cached_frame(cache_dir, symbol, duration, bar_size, use_rth, frame)
                print(f"  cached {len(frame)} rows to {path}")
    finally:
        ib.disconnect()

    # Preserve the caller's ordering rather than "cached first, fetched after".
    return {symbol: frames[symbol] for symbol in symbols if symbol in frames}
