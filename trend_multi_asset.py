#!/usr/bin/env python3
"""Trend following across asset classes, which is the one standard approach
this project has not tried.

Two hundred and fifty-three stocks and twelve published cross-sectional
effects produced nothing, and six direction models failed their hold-outs.
What did survive is that risk is predictable: volatility targeting cut the
worst drawdown in 26 of 26 instrument-periods and again on a 253-stock
book. Trend following is the other half of the same claim - that an asset
which has been rising keeps rising slightly more often than not - and it
has one property nothing here has had, which is that its edge is supposed
to come from holding many weakly-correlated markets at once rather than
from any single one being predictable.

Trend timing on SPY alone was already rejected here: it helps only in
crises and is net negative over twenty years. That is the expected result
for one market. The open question is whether the same rule over equities,
bonds, commodities, currencies and property behaves differently, because
the diversification is the mechanism, not the signal.

No new parameters. The look-back is 252 days and the sizing is the
volatility target already fixed in volatility_target.py: 16% annual, 20-day
trailing window, capped at 1. Selected assets are equally weighted. Inverse-volatility weighting is the
textbook choice and is reported alongside, but it is wrong under this
project's binding constraint. Weighting by the inverse of volatility gives
a 3%-volatility bond fund six times the weight of an 18%-volatility equity
fund, which is correct only if the book can then be levered back up to the
risk it wants. Borrowing costs 6% at IBKR, which an earlier test showed
consumes the entire advantage, so the book cannot be levered, and the
weighting simply parks the money in the lowest-returning assets. The first
run of this script did exactly that: 94.6% invested, 7.1% volatility, and
+0.09% a year.

Monthly rebalancing and 5 bps a side, as everywhere else in this project.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ibkr_ml.cache import load_cached_frame, require_adjusted
from ibkr_ml.features import to_eastern_naive

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
TARGET_VOLATILITY = 0.16
VOL_WINDOW = 20

ASSET_CLASSES = {
    "股票": ("SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "VEA", "VWO"),
    "债券": ("AGG", "LQD", "HYG", "TIP", "EMB", "TLT", "IEF", "SHY"),
    "商品": ("GLD", "SLV", "DBC", "USO", "DBA"),
    "外汇": ("UUP", "FXE", "FXY"),
    "房地产": ("VNQ", "IYR"),
}


def load_prices(symbols, cache_dir, duration, start, end, min_bars):
    columns = {}
    for symbol in symbols:
        frame, _ = load_cached_frame(cache_dir, symbol, duration, "1 day", True)
        if frame is None:
            continue
        stamps = to_eastern_naive(frame["timestamp"]).dt.normalize()
        keep = stamps.notna()
        if start:
            keep &= stamps >= pd.Timestamp(start)
        if end:
            keep &= stamps < pd.Timestamp(end)
        if keep.sum() < min_bars:
            continue
        columns[symbol] = pd.Series(frame.loc[keep, "close"].to_numpy(float),
                                    index=stamps[keep].to_numpy())
    if not columns:
        raise SystemExit("缓存里没有符合条件的数据")
    return pd.DataFrame(columns).sort_index()


def inverse_volatility_weights(chosen, volatility):
    """Split the book across the chosen assets, inversely to their volatility.

    Without this a single volatile market sets the whole portfolio's risk.
    Dividing by volatility and renormalising gives each market a similar
    share of the risk rather than a similar share of the money, and it adds
    no parameter of its own.
    """
    inverse = chosen.div(volatility.where(volatility > 0))
    total = inverse.sum(axis=1)
    return inverse.div(total.where(total > 0), axis=0).fillna(0.0)


def build_book(closes, returns, available, rebalance_days, long_short,
               lookback, permutation=None, weighting="equal"):
    """Monthly trend decisions, held with drift until the next rebalance.

    The signal is read at the close of the day before a rebalance, so the
    position it creates never collects a return the signal already saw.
    """
    trailing = returns.rolling(VOL_WINDOW).std(ddof=1) * TRADING_DAYS ** 0.5
    momentum = (closes / closes.shift(lookback) - 1.0).where(available)
    filled = returns.fillna(0.0)

    target = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for i, day in enumerate(rebalance_days):
        if day not in closes.index:
            continue
        position = closes.index.get_loc(day)
        if position == 0:
            continue
        yesterday = closes.index[position - 1]
        signal = momentum.loc[yesterday]
        volatility = trailing.loc[yesterday]
        if permutation is not None:
            signal = pd.Series(permutation[i], index=closes.columns)
        direction = signal.gt(0).astype(float)
        if long_short:
            direction = direction - signal.lt(0).astype(float)
        usable = available.loc[yesterday] & volatility.notna() & signal.notna()
        chosen = (direction * usable.astype(float)).abs()
        if weighting == "inverse-vol":
            weights = inverse_volatility_weights(
                chosen.to_frame().T, volatility.to_frame().T).iloc[0]
        else:
            count = chosen.sum()
            weights = chosen / count if count > 0 else chosen
        target.loc[day] = (weights * np.sign(direction)).to_numpy()

    held = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    traded = pd.Series(0.0, index=closes.index)
    rebalance_set = set(rebalance_days)
    prior = filled.shift(1).fillna(0.0).to_numpy()
    positions = np.zeros(closes.shape[1])
    for i, day in enumerate(closes.index):
        positions = positions * (1.0 + prior[i])
        total = np.abs(positions).sum()
        if total > 0:
            positions = positions / total
        if day in rebalance_set:
            wanted = target.loc[day].to_numpy()
            if np.abs(wanted).sum() > 0:
                traded.iloc[i] = float(np.abs(wanted - positions).sum())
                positions = wanted
        held.iloc[i] = positions
    return held, traded


def apply_overlay(weights, returns):
    """Scale the whole book by the volatility target fixed elsewhere."""
    gross = (weights * returns).sum(axis=1)
    trailing = gross.rolling(VOL_WINDOW).std(ddof=1).shift(1) * TRADING_DAYS ** 0.5
    scale = (TARGET_VOLATILITY / trailing).clip(upper=1.0).fillna(0.0)
    return weights.mul(scale, axis=0)


def score(weights, traded, returns, cost_bps, cash_rate, borrow_rate=0.0):
    """Net of commission, of interest on idle cash, and of stock-borrow fees.

    A short position is not free: the shares have to be borrowed, and the
    lender charges for them. For liquid ETFs that runs a few tenths of a
    percent a year, and it is charged here on the short side only, because
    ignoring it would make a long-short book look better than it can be.
    """
    gross_exposure = weights.abs().sum(axis=1)
    short_exposure = weights.clip(upper=0.0).abs().sum(axis=1)
    idle = (1.0 - gross_exposure).clip(lower=0.0)
    daily_cash = cash_rate / TRADING_DAYS
    net = ((weights * returns).sum(axis=1) - traded * cost_bps / 10000.0
           + idle * daily_cash - short_exposure * borrow_rate / TRADING_DAYS
           ).dropna()
    excess = net - daily_cash
    equity = (1 + net).cumprod()
    return net, {
        "ann": float(equity.iloc[-1] ** (TRADING_DAYS / len(net)) - 1),
        "vol": float(net.std(ddof=1) * TRADING_DAYS ** 0.5),
        "sharpe": float(excess.mean() / excess.std(ddof=1) * TRADING_DAYS ** 0.5),
        "dd": float((equity / equity.cummax() - 1).min()),
        "exposure": float(gross_exposure.mean()),
        "turnover": float(traded.mean() * TRADING_DAYS),
    }


def static_book(closes, returns, available, allocation):
    """A fixed-weight benchmark, rebalanced monthly at the same cost."""
    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for symbol, share in allocation.items():
        if symbol in weights.columns:
            weights[symbol] = share * available[symbol].astype(float)
    total = weights.sum(axis=1)
    weights = weights.div(total.where(total > 0), axis=0).fillna(0.0)
    return weights, weights.diff().abs().sum(axis=1).fillna(1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-asset trend following.")
    parser.add_argument("--cache-dir", default="data_cache_adj",
                        help="必须是 ADJUSTED_LAST 取的数据。未复权价格不含分红，"
                             "对不同资产的影响不一样（AGG 约 2.8 个百分点/年，"
                             "GLD 为零），跨资产比较会被系统性扭曲。")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--start", default="2006-09-18")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--split", default="2017-01-01")
    parser.add_argument("--min-bars", type=int, default=400)
    parser.add_argument("--lookback", type=int, default=252)
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--weighting", choices=("equal", "inverse-vol"),
                        default="equal",
                        help="How to split the book across the chosen assets.")
    parser.add_argument("--long-short", action="store_true",
                        help="Short the falling markets as well. Off by default: "
                             "shorting an ETF costs borrow the backtest cannot see.")
    parser.add_argument("--borrow-rate", type=float, default=0.01,
                        help="Annual stock-borrow fee charged on short exposure.")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    # Only the default changed when this was last touched, so naming the
    # unadjusted directory on the command line still ran - which is the
    # mistake that reversed this script's own conclusion once.
    require_adjusted(args.cache_dir, "跨资产趋势回测")

    symbols = [s for group in ASSET_CLASSES.values() for s in group]
    closes = load_prices(symbols, args.cache_dir, args.duration, args.start,
                         args.end, args.min_bars)
    returns = closes.pct_change(fill_method=None)
    available = closes.notna()
    rebalance_days = list(closes.index[::args.rebalance])

    present = {group: [s for s in members if s in closes.columns]
               for group, members in ASSET_CLASSES.items()}
    print(f"{closes.index.min().date()} .. {closes.index.max().date()}   "
          f"{len(closes)} 天   {closes.shape[1]} 个标的")
    for group, members in present.items():
        first = {s: closes[s].first_valid_index().date() for s in members}
        late = [f"{s}({d})" for s, d in first.items() if str(d) > "2007-01-01"]
        print(f"  {group}: {len(members)} 个" + (f"，晚于 2007 上市: {'、'.join(late)}"
                                               if late else ""))

    trend, trend_traded = build_book(closes, returns, available, rebalance_days,
                                     args.long_short, args.lookback,
                                     weighting=args.weighting)
    sized = apply_overlay(trend, returns)
    other = "inverse-vol" if args.weighting == "equal" else "equal"
    alt, alt_traded = build_book(closes, returns, available, rebalance_days,
                                 args.long_short, args.lookback, weighting=other)
    equal, equal_traded = static_book(closes, returns, available,
                                      {s: 1.0 for s in closes.columns})
    sixty_forty, sf_traded = static_book(closes, returns, available,
                                         {"SPY": 0.6, "AGG": 0.4})
    spy_only, spy_traded = static_book(closes, returns, available, {"SPY": 1.0})

    books = {
        "SPY 买入持有": (spy_only, spy_traded),
        "60/40（SPY+AGG）": (sixty_forty, sf_traded),
        "全资产等权": (equal, equal_traded),
        "趋势跟踪": (trend, trend_traded),
        "趋势 + 波动率目标化": (sized, sized.diff().abs().sum(axis=1).fillna(
            sized.abs().sum(axis=1))),
        f"趋势（{other} 加权）": (alt, alt_traded),
    }
    periods = [(args.start, args.split, "2006-2016", 0.006),
               (args.split, args.end, "2017-2026", 0.022),
               (args.start, args.end, "全期", 0.015)]

    for start, end, title, cash in periods:
        mask = ((closes.index >= pd.Timestamp(start))
                & (closes.index < pd.Timestamp(end)))
        if mask.sum() < 200:
            continue
        print(f"\n===== {title}（现金利率 {cash:.1%}，成本 {args.cost_bps:g} bp/边）=====")
        for label, (w, t) in books.items():
            _, s = score(w[mask], t[mask], returns[mask], args.cost_bps, cash,
                         args.borrow_rate)
            print(f"  {label:<20} 年化 {s['ann']:+7.2%}  波动 {s['vol']:6.2%}  "
                  f"Sharpe {s['sharpe']:5.2f}  回撤 {s['dd']:+7.2%}  "
                  f"仓位 {s['exposure']:5.1%}  换手 {s['turnover']:4.1f}x")

    print(f"\n[逐年，现金利率 1.5%]")
    labels = list(books)
    print("  年份  " + "".join(f"{l[:12]:>13}" for l in labels))
    for year in range(closes.index.min().year, closes.index.max().year + 1):
        mask = ((closes.index >= pd.Timestamp(f"{year}-01-01"))
                & (closes.index < pd.Timestamp(f"{year + 1}-01-01")))
        if mask.sum() < 60:
            continue
        cells = ""
        for label in labels:
            w, t = books[label]
            net, _ = score(w[mask], t[mask], returns[mask], args.cost_bps, 0.015,
                           args.borrow_rate)
            cells += f"{(1 + net).prod() - 1:>+12.2%} "
        print(f"  {year}  {cells}")

    if args.draws:
        print(f"\n[随机对照：打乱趋势信号，保留调仓日与仓位规模，{args.draws} 次]")
        rng = np.random.default_rng(args.seed)
        real = score(sized, books["趋势 + 波动率目标化"][1], returns,
                     args.cost_bps, 0.015, args.borrow_rate)[1]["sharpe"]
        control = []
        for _ in range(args.draws):
            noise = [rng.standard_normal(closes.shape[1]) for _ in rebalance_days]
            w, t = build_book(closes, returns, available, rebalance_days,
                              args.long_short, args.lookback, permutation=noise,
                              weighting=args.weighting)
            w = apply_overlay(w, returns)
            control.append(score(w, w.diff().abs().sum(axis=1).fillna(
                w.abs().sum(axis=1)), returns, args.cost_bps, 0.015,
                args.borrow_rate)[1]["sharpe"])
        control = np.array(control)
        p = (int((control >= real).sum()) + 1) / (args.draws + 1)
        print(f"  真实 Sharpe {real:.2f}   随机中位数 {np.median(control):.2f}   "
              f"区间 [{control.min():.2f}, {control.max():.2f}]   p = {p:.4f}")


if __name__ == "__main__":
    main()
