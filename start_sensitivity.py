#!/usr/bin/env python3
"""How much of the published Sharpe is an accident of where the data starts.

`portfolio_build.py` picks its rebalance dates as `closes.index[::rebalance]`
- counted from the first row of the data, not from a calendar - and
`step_overlay` counts its check days the same way with `i % every == 0`. So
the start date does not merely trim the sample. It decides which ~240 days of
the next twenty years are rebalance days, and which are overlay days.

That matters because `--duration "20 Y"` is a window relative to now. Every
re-download moves the first row, and every rebalance day of the whole history
moves with it. Re-fetching the cache on 2026-09-15 moved the start from
2006-09-18 to 2006-09-20 and changed all thirty lines of the 全期 table, on
price history that was otherwise identical to the last cent (5023 of 5024
shared closes unchanged).

This sweeps the start across one full rebalance period and reports the
spread, which is the honest error bar on any single published figure. It
reads the cache and writes nothing.

Measured 2026-09-15, thirds, 2006-2026, 5 bps a side, cash 1.5%:

    no overlay      Sharpe 0.628 - 0.639   spread 0.011
    with overlay    Sharpe 0.687 - 0.726   spread 0.039
    the gain        +0.052 - +0.093        median +0.080, positive 21/21

The conclusion survives - the overlay helps at every start - but a single
0.72 is reproducible only until the data is pulled again. Report the gain
and its spread instead.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把回测起点在一个再平衡周期内扫一遍，看结论有多稳。")
    parser.add_argument("--allocation", default="股债TIP 各三分",
                        help="portfolio_build.ALLOCATIONS 里的名字。")
    parser.add_argument("--cache-dir", default="data_cache_adj")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--overlay-every", type=int, default=5)
    parser.add_argument("--overlay-band", type=float, default=0.03)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--cash-rate", type=float, default=0.015)
    args = parser.parse_args()

    import numpy as np

    from ibkr_ml.cache import require_adjusted
    from portfolio_build import (ALLOCATIONS, load_prices, overlay_scale,
                                 score, static_weights, step_overlay)

    require_adjusted(args.cache_dir, "起点敏感度")
    if args.allocation not in ALLOCATIONS:
        raise SystemExit(f"没有这个配置: {args.allocation}。"
                         f"可选: {'、'.join(ALLOCATIONS)}")
    allocation = ALLOCATIONS[args.allocation]

    everything = load_prices(allocation, Path(args.cache_dir), args.duration,
                             None, args.end)
    # One full rebalance period of starts. Beyond that the pattern repeats,
    # because start + rebalance lands every rebalance day back where it was.
    starts = list(everything.index[:args.rebalance])

    print(f"\n{args.allocation}，{args.duration}，到 {args.end} 止，"
          f"现金 {args.cash_rate:.1%}，{args.cost_bps:.0f}bp/边")
    print(f"再平衡每 {args.rebalance} 个交易日，overlay 每 "
          f"{args.overlay_every} 个交易日看一次，带 "
          f"{args.overlay_band:.0%} 不动区间\n")
    print(f"{'起点':<12}{'偏移':>5}{'不做overlay':>13}{'做overlay':>12}"
          f"{'提升':>8}{'年化':>9}")

    plain_sharpes, overlay_sharpes = [], []
    for offset, start in enumerate(starts):
        closes = everything[everything.index >= start]
        returns = closes.pct_change(fill_method=None)
        weights, traded = static_weights(allocation, closes, args.rebalance)
        gross = (weights * returns).sum(axis=1)
        scaled = weights.mul(
            step_overlay(overlay_scale(gross, "expanding"),
                         args.overlay_every, args.overlay_band), axis=0)
        scaled_traded = scaled.diff().abs().sum(axis=1).fillna(
            scaled.abs().sum(axis=1))
        _, plain = score(weights, traded, returns, args.cost_bps, args.cash_rate)
        _, with_overlay = score(scaled, scaled_traded, returns, args.cost_bps,
                                args.cash_rate)
        plain_sharpes.append(plain["sharpe"])
        overlay_sharpes.append(with_overlay["sharpe"])
        print(f"{str(start.date()):<12}{offset:>5}{plain['sharpe']:>13.3f}"
              f"{with_overlay['sharpe']:>12.3f}"
              f"{with_overlay['sharpe'] - plain['sharpe']:>+8.3f}"
              f"{with_overlay['ann']:>9.2%}")

    plain = np.array(plain_sharpes)
    overlaid = np.array(overlay_sharpes)
    gain = overlaid - plain
    print(f"\n不做 overlay: {plain.min():.3f} ~ {plain.max():.3f}   "
          f"跨度 {np.ptp(plain):.3f}   中位数 {np.median(plain):.3f}")
    print(f"做   overlay: {overlaid.min():.3f} ~ {overlaid.max():.3f}   "
          f"跨度 {np.ptp(overlaid):.3f}   中位数 {np.median(overlaid):.3f}")
    print(f"overlay 的提升: {gain.min():+.3f} ~ {gain.max():+.3f}   "
          f"中位数 {np.median(gain):+.3f}   "
          f"为正的起点 {int((gain > 0).sum())}/{len(gain)}")
    print(f"\n读法：跨度就是任何单个数字的误差范围。不做 overlay 的跨度小，"
          f"说明价格数据本身稳定；做 overlay 的跨度大，是因为 overlay 的"
          f"检查日和再平衡日都是从数据第一行数出来的，起点一动就全动。"
          f"该报告的是提升和它的范围，不是某一次的单点值。")


if __name__ == "__main__":
    main()
