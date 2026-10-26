#!/usr/bin/env python3
"""Score the ChatGPT reports against random picks, once a day.

A report's calls beating SPY says little on its own: an equal-weight pick
of ordinary names drifts away from a cap-weighted index for reasons that
have nothing to do with skill. The control asks the sharper question -
would the same number of names, drawn at random from a fixed universe on
the same entry dates and signed the same way, have done as well? - which
is the rule every idea in this project has to clear.

The universe is the project's own: the companies in data_cache_adj, funds
removed, 253 names. They are fetched from Yahoo at most once every 20
hours, and only once there is at least one entered directional call, so
nothing is downloaded while the inbox is empty. The result goes to
logs/signals/control.json for sim_dashboard.py; run_sim_daily.sh runs this
after the close.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

from cross_section_audit import FUNDS
from portfolio_trade import EASTERN

# Funds in data_cache_adj that cross_section_audit.FUNDS does not list.
MORE_FUNDS = {"DBA", "EMB", "FXE", "FXY", "USO", "UUP", "VEA", "VWO"}
CONTROL_CHECKPOINTS = (1, 5, 20)
UNIVERSE_MAX_AGE = 20 * 3600
SIGNAL_MAX_AGE = 30 * 60


def universe_symbols(cache_dir):
    names = {p.name.split("__")[0]
             for p in Path(cache_dir).glob("*__20_Y__1_day__rth1.csv")}
    return sorted(names - FUNDS - MORE_FUNDS)


def main() -> None:
    from datetime import datetime

    from ibkr_ml.signals import TASKS, load_inbox, random_control, score_inbox
    from ibkr_ml.yahoo import cached_bars, chart_path

    parser = argparse.ArgumentParser(description="把 ChatGPT 报告和随机抽股比较")
    parser.add_argument("--signals-dir", default="logs/signals")
    parser.add_argument("--cache-dir", default="data_cache_yahoo")
    parser.add_argument("--universe-from", default="data_cache_adj",
                        help="股票池取这个目录里缓存过的公司（去掉基金）")
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args()

    now = datetime.now(EASTERN)
    signals_dir, cache = Path(args.signals_dir), Path(args.cache_dir)
    out_path = signals_dir / "control.json"
    records = load_inbox(signals_dir)
    payload = {"computed": now.isoformat(timespec="seconds"), "draws": args.draws,
               "universe": 0, "tasks": {}}
    if records:
        earliest = min(datetime.fromisoformat(r["received"]) for r in records)
        start = min(earliest - timedelta(days=10), now - timedelta(days=730))
        memo = {}

        def bars_for(ticker):
            if ticker not in memo:
                memo[ticker] = cached_bars(ticker, cache / "signals", start, SIGNAL_MAX_AGE)
            return memo[ticker]

        rows = score_inbox(records, bars_for)
        ready = [r for r in rows if r["direction"] and r.get("entry_session")]
        if ready:
            universe = {}
            for symbol in universe_symbols(args.universe_from):
                path = chart_path(cache / "universe", symbol)
                if not path.exists() or time.time() - path.stat().st_mtime > UNIVERSE_MAX_AGE:
                    time.sleep(0.15)
                frame = cached_bars(symbol, cache / "universe", start, UNIVERSE_MAX_AGE)
                if frame is not None and not frame.empty:
                    universe[symbol] = frame
            payload["universe"] = len(universe)
            benchmark = bars_for("SPY")
            for task in TASKS:
                mine = [r for r in rows if r["task"] == task]
                payload["tasks"][task] = {
                    str(k): random_control(mine, universe, benchmark, k, args.draws)
                    for k in CONTROL_CHECKPOINTS}
            print(f"随机对照：股票池 {len(universe)} 只，每个时间点抽 {args.draws} 次")
            for task, result in payload["tasks"].items():
                for k, row in result.items():
                    if row.get("n"):
                        print(f"  {task} 第 {k} 日：{row['n']} 条，平均方向超额 "
                              f"{row['observed']:+.2%}，随机中位数 {row['median']:+.2%}，"
                              f"p = {row['p']:.3f}")
        else:
            print("还没有已入场、带方向的建议，不做随机对照")
    else:
        print("还没有导入任何报告，不做随机对照")

    signals_dir.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".json.part")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                       encoding="utf-8")
    partial.replace(out_path)


if __name__ == "__main__":
    main()
