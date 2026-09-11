#!/usr/bin/env python3
"""Download daily bars for a list of index funds into the cache.

Written because an earlier download skipped any symbol whose cache file
already existed, and three of them existed but were short: TLT held only
2016 onward and IEF and SHY only 2017 onward, against the twenty years
every other fund had. A truncated file is worse than a missing one - it
looks complete to anything reading the cache - so this script reports what
it actually got and takes --force to replace a file it judges too short.

The funds here are the asset classes a trend-following book trades: equity
indices, government and corporate bonds across maturities, commodities,
currencies and property. None of them has a survivorship problem, since an
index fund that closes is replaced by its index, not deleted from history.

Use --what-to-show ADJUSTED_LAST for anything that compares one asset
class with another. The default, TRADES, is the traded price and leaves
dividends out, which understates SPY by 1.8% a year, AGG by 2.8% and gold
by nothing - enough to make a bond fund look like a twenty-year loss and
to hand gold an advantage it does not have.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ibkr_ml.cache import load_cached_frame, save_cached_frame
from ibkr_ml.config import IBKRConnectionConfig
from ibkr_ml.data import connect_ib, fetch_historical_frame

DEFAULT_SYMBOLS = (
    # long-dated Treasuries: the crisis hedge the cache is missing
    "TLT", "IEF", "SHY",
    # currencies: dollar, euro, yen
    "UUP", "FXE", "FXY",
    # commodities beyond the broad basket
    "USO", "DBA",
    # emerging-market debt and developed/emerging equity
    "EMB", "VEA", "VWO",
)


def first_bar(cache_dir, symbol, duration):
    frame, _ = load_cached_frame(cache_dir, symbol, duration, "1 day", True)
    if frame is None or frame.empty:
        return None, 0
    return frame["timestamp"].min(), len(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch daily bars into the cache.")
    parser.add_argument("--symbols", nargs="*", default=list(DEFAULT_SYMBOLS))
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--client-id", type=int, default=131)
    parser.add_argument("--min-bars", type=int, default=4500,
                        help="A cached file shorter than this counts as truncated.")
    parser.add_argument("--what-to-show", default="TRADES",
                        choices=("TRADES", "ADJUSTED_LAST"),
                        help="TRADES is the traded price and drops dividends; "
                             "ADJUSTED_LAST adds them back for total return.")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even a file that looks complete.")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()

    cache = Path(args.cache_dir)
    # Nothing in the cached CSV or its sidecar says whether dividends are in
    # the prices, and mixing the two kinds in one directory is silent and
    # unrecoverable. Stamp the directory, and refuse to mix.
    cache.mkdir(parents=True, exist_ok=True)
    marker = cache / ".what_to_show"
    existing = marker.read_text().strip() if marker.exists() else ""
    if existing and existing != args.what_to_show:
        raise SystemExit(
            f"{cache} 里已经是 {existing} 取的数据，现在要写 "
            f"{args.what_to_show}，两种混在一个目录里没法分辨。"
            f"换一个 --cache-dir。")
    marker.write_text(args.what_to_show + "\n")

    pending = []
    for symbol in args.symbols:
        start, rows = first_bar(cache, symbol, args.duration)
        if rows and not args.force and rows >= args.min_bars:
            print(f"{symbol}: 已有 {rows} 行，{start.date()} 起，跳过", flush=True)
            continue
        if rows:
            print(f"{symbol}: 已有 {rows} 行（{start.date()} 起），少于 "
                  f"{args.min_bars}，重新下载", flush=True)
        pending.append(symbol)
    if not pending:
        print("没有需要下载的标的")
        return

    ib = connect_ib(IBKRConnectionConfig(client_id=args.client_id,
                                         request_timeout=180.0,
                                         connect_retries=3,
                                         retry_delay_seconds=6.0))
    done, failed, started = 0, [], time.monotonic()
    try:
        for symbol in pending:
            for attempt in range(args.attempts):
                try:
                    frame = fetch_historical_frame(
                        ib=ib, symbol=symbol, duration=args.duration,
                        bar_size="1 day", use_rth=True,
                        max_duration_per_request=args.duration,
                        what_to_show=args.what_to_show)
                    save_cached_frame(cache, symbol, args.duration, "1 day", True, frame)
                    print(f"{symbol}: {len(frame)} 行  "
                          f"{frame['timestamp'].min().date()} 起", flush=True)
                    done += 1
                    break
                except Exception as exc:
                    if attempt == args.attempts - 1:
                        failed.append(symbol)
                        print(f"{symbol}: 失败 {str(exc)[:70]}", flush=True)
                    else:
                        time.sleep(3)
    finally:
        ib.disconnect()
    print(f"\n完成 {done}，失败 {len(failed)}: {failed}，"
          f"用时 {(time.monotonic() - started) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
