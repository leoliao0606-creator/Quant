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

from zoneinfo import ZoneInfo

from .features import parse_bar_timestamps, to_eastern_naive


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


# Nothing inside a cached CSV says whether dividends are in the prices, so the
# answer lives in one marker file per directory, written by save_cached_frame
# on every write. fetch_assets.py also checks it before downloading anything,
# to fail before holding a TWS session rather than after.
WHAT_TO_SHOW_MARKER = ".what_to_show"

# Regular trading ends at 16:00 Eastern. A daily bar stamped today, read
# before then, covers only part of its session.
SESSION_CLOSE_HOUR = 16
EASTERN = ZoneInfo("America/New_York")


def cache_kind(cache_dir) -> str:
    """Which `--what-to-show` the directory was fetched with, "" if unmarked."""
    path = Path(cache_dir) / WHAT_TO_SHOW_MARKER
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def require_adjusted(cache_dir, purpose: str = "") -> None:
    """Refuse to read a cache that silently left dividends out.

    Bars fetched as TRADES carry no distributions, which costs AGG about 2.8
    points a year and GLD nothing at all, so it does not move a result evenly
    - it tilts every comparison towards whatever pays least. That reversed a
    cross-asset conclusion in this project once already, so it stops the run
    rather than warning.

    This lives here, next to the reading, because a copy in one script is a
    check the other scripts do not have: trend_multi_asset.py changed its
    default to the adjusted directory and kept no check at all, so naming the
    unadjusted one on the command line still ran.
    """
    kind = cache_kind(cache_dir)
    if kind == "ADJUSTED_LAST":
        return
    detail = (f"标记文件写着 {kind!r}" if kind
              else f"缓存里没有 {WHAT_TO_SHOW_MARKER} 标记文件，无法确认怎么取的")
    tail = f"（{purpose}）" if purpose else ""
    raise SystemExit(
        f"{cache_dir} 不是分红调整过的数据{tail}：{detail}。\n"
        f"未复权的成交价不含分红，AGG 会少约 2.8 个百分点/年，GLD 一分不少，"
        f"所以它不是把结果整体拉低，而是系统性偏袒不分红的资产。\n"
        f"用 --cache-dir data_cache_adj，或先跑 "
        f"fetch_assets.py --what-to-show ADJUSTED_LAST 重新下载。")


def drop_partial_session(frame, bar_size: str, now=None):
    """Drop a daily bar for a session that is still trading.

    IBKR hands back a partial bar for the session in progress, and once it is
    written to the cache nothing distinguishes it from a finished day.
    Measured on this repo's own cache: data_cache_adj was fetched at
    2026-09-11T16:49 UTC, which is 12:49 Eastern, and SPY's bar for that day
    carried 12,365,241 shares against 21.6M to 28.8M on the five days before
    it, with a high-low range of 2.78 against 3.47 to 6.58. 12:49 is 51% of a
    6.5 hour session and the volume ratio was 49%, so that bar was half a day
    recorded as a whole one.

    Ten scripts read that cache and none of them dropped it. Doing it here, at
    the one place bars are written, is what keeps a backtest run at noon and
    the same backtest run after the close from disagreeing - and portfolio
    volatility read off a half-session looks calmer than the market is, which
    is the direction that sizes a book too large.

    Daily bars only. An intraday bar size has its own notion of a partial bar,
    and none of the backtests read one.
    """
    pd = _load_pandas()
    if bar_size != "1 day" or frame is None or len(frame) == 0:
        return frame, False
    now = now or datetime.now(EASTERN)
    last = to_eastern_naive(frame["timestamp"]).iloc[-1]
    if pd.isna(last):
        return frame, False
    unfinished = last.date() > now.date() or (
        last.date() == now.date() and now.hour < SESSION_CLOSE_HOUR)
    return (frame.iloc[:-1], True) if unfinished else (frame, False)


def claim_cache_kind(cache_dir, what_to_show: str) -> None:
    """Mark the directory with what it holds, or refuse to mix two kinds.

    require_adjusted reads this marker to decide whether a cache carries
    dividends. fetch_assets.py wrote it and refused to mix two kinds, but it
    did so on its own, before its own download, so the guard covered only the
    path that went through fetch_assets.py. train_model.py does not: it calls
    fetch_frames, which calls save_cached_frame directly, and it passed no
    what_to_show at all (train_model.py:256), taking fetch_historical_frame's
    TRADES default (ibkr_ml/data.py:584) into whatever --cache-dir named. So
    `train_model.py --cache-dir data_cache_adj` wrote unadjusted bars into the
    adjusted directory, left the marker saying ADJUSTED_LAST, and every later
    require_adjusted passed.

    Writing the marker here, at the one place bars reach the disk, makes the
    claim and the contents the same act. A directory holds one kind, and a
    run that would mix them stops instead - by whichever path it arrived.
    """
    if not what_to_show:
        raise SystemExit("写缓存必须说明 what_to_show，空值无法标记")
    existing = cache_kind(cache_dir)
    if not existing:
        (Path(cache_dir) / WHAT_TO_SHOW_MARKER).write_text(
            what_to_show + "\n", encoding="utf-8")
        return
    if existing != what_to_show:
        raise SystemExit(
            f"{cache_dir} 里已有的数据是 {existing} 取的，这次要写的是 "
            f"{what_to_show} 取的，两种不能放在同一个目录。\n"
            f"TRADES 是成交价，不含分红；ADJUSTED_LAST 含。混在一个目录之后，"
            f"没有任何办法分辨哪根K线是哪种，而 AGG 这类标的两者差约 2.8 "
            f"个百分点/年。\n"
            f"换一个 --cache-dir，或者先清空这个目录再重新下载。")


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


def save_cached_frame(cache_dir, symbol: str, duration: str, bar_size: str,
                      use_rth: bool, frame, *, what_to_show: str) -> Path:
    """Write bars to the cache, declaring what kind of bars they are.

    `what_to_show` is keyword-only and has no default on purpose: a default
    here would be exactly the silent TRADES that got into an adjusted
    directory in the first place. Every caller has to say what it fetched.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    claim_cache_kind(cache_dir, what_to_show)
    frame, dropped = drop_partial_session(frame, bar_size)
    if dropped:
        print(f"  {symbol}: 丢掉当日未收盘的半截K线，缓存只存完整交易日")
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
        "what_to_show": what_to_show,
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
    what_to_show: str,
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
                path = save_cached_frame(cache_dir, symbol, duration, bar_size,
                                        use_rth, frame,
                                        what_to_show=what_to_show)
                print(f"  cached {len(frame)} rows to {path}")
    finally:
        ib.disconnect()

    # Preserve the caller's ordering rather than "cached first, fetched after".
    return {symbol: frames[symbol] for symbol in symbols if symbol in frames}
