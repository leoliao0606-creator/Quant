#!/usr/bin/env python3
"""Build a book from the two things in this project that survived testing.

Everything that tried to predict direction failed: six machine-learning
configurations against hold-outs, twelve published cross-sectional effects
against a pre-registered threshold, post-earnings drift, the overnight
effect, and cross-asset trend following in six configurations. Two things
did survive, and neither predicts anything.

  Volatility targeting. Holding less when an instrument has been moving
  more cut the worst drawdown in 26 of 26 instrument-periods, and again on
  a 253-stock portfolio, -49.8% to -31.9%. It forecasts risk, which is
  predictable, and says nothing about return, which is not.

  Diversification. Assets that do not move together have a combined
  volatility lower than the average of their parts. This is arithmetic,
  not a forecast, and it is the only free lunch in the subject.

The allocations tested here are textbook ones - 100% equity, 60/40, equal
thirds across stocks, bonds and gold - chosen because they are standard
rather than because they scored well. Picking the best-performing weights
out of this table would be fitting to twenty years of bond bull market and
a gold run, and is exactly what the rest of this project has spent its
time learning not to do.

### The expanding-volatility target

volatility_target.py aims at a fixed 16% a year, which is roughly the long
-run volatility of US equities and was deliberately set from outside the
data. That number does not transfer to a diversified book: a stock/bond/
gold third runs near 10%, so min(16%/vol, 1) is 1 on almost every day and
the rule does nothing.

The fix keeps the rule and replaces the constant with the book's own
long-run volatility, estimated from an expanding window of everything up
to the previous day. It reads "hold less when this book is moving more
than it usually does", needs no constant chosen by anyone, and cannot see
the future. The fixed and expanding forms are both reported on SPY so the
substitution can be checked rather than assumed.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ibkr_ml.cache import load_cached_frame
from ibkr_ml.features import to_eastern_naive

warnings.filterwarnings("ignore")

TRADING_DAYS = 252
VOL_WINDOW = 20
FIXED_TARGET = 0.16
MIN_HISTORY = 252

ALLOCATIONS = {
    "SPY 100%": {"SPY": 1.0},
    "60/40 股债": {"SPY": 0.6, "AGG": 0.4},
    "股债金 各三分": {"SPY": 1 / 3, "AGG": 1 / 3, "GLD": 1 / 3},
    "股债金房 各四分": {"SPY": 0.25, "AGG": 0.25, "GLD": 0.25, "VNQ": 0.25},
    "全球股债金": {"SPY": 0.25, "EFA": 0.15, "EEM": 0.10,
                "AGG": 0.25, "LQD": 0.10, "GLD": 0.15},
    "股债TIP 各三分": {"SPY": 1 / 3, "AGG": 1 / 3, "TIP": 1 / 3},
    "股40债30金15TIP15": {"SPY": 0.40, "AGG": 0.30, "GLD": 0.15, "TIP": 0.15},
}

# Gold has no cash flow, so its expected long-run real return is near zero.
# Its 10.2% a year over this window is a historical outlier - it was roughly
# flat in nominal terms from 1980 to 2000 - and any allocation leaning on it
# has to be reported against the possibility that it does not repeat.
GOLD_SCENARIOS = (None, 0.05, 0.025, 0.0)


def load_prices(symbols, cache_dir, duration, start, end):
    columns = {}
    for symbol in sorted(set(symbols)):
        frame, _ = load_cached_frame(cache_dir, symbol, duration, "1 day", True)
        if frame is None:
            raise SystemExit(f"缓存里没有 {symbol}")
        stamps = to_eastern_naive(frame["timestamp"]).dt.normalize()
        keep = stamps.notna()
        if start:
            keep &= stamps >= pd.Timestamp(start)
        if end:
            keep &= stamps < pd.Timestamp(end)
        columns[symbol] = pd.Series(frame.loc[keep, "close"].to_numpy(float),
                                    index=stamps[keep].to_numpy())
    return pd.DataFrame(columns).sort_index()


def static_weights(allocation, closes, rebalance):
    """Fixed shares, restored to target every `rebalance` days and drifting
    in between, which is what an account actually does."""
    available = closes.notna()
    filled = closes.pct_change(fill_method=None).fillna(0.0)
    target = pd.Series(0.0, index=closes.columns)
    for symbol, share in allocation.items():
        target[symbol] = share
    rebalance_days = set(closes.index[::rebalance])
    held = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    traded = pd.Series(0.0, index=closes.index)
    prior = filled.shift(1).fillna(0.0).to_numpy()
    positions = np.zeros(closes.shape[1])
    for i, day in enumerate(closes.index):
        positions = positions * (1.0 + prior[i])
        total = positions.sum()
        if total > 0:
            positions = positions / total
        if day in rebalance_days:
            live = target * available.loc[day].to_numpy()
            if live.sum() > 0:
                wanted = (live / live.sum()).to_numpy()
                traded.iloc[i] = float(np.abs(wanted - positions).sum())
                positions = wanted
        held.iloc[i] = positions
    return held, traded


def overlay_scale(gross, mode):
    """How much of the book to hold, from its own trailing volatility.

    "fixed" aims at 16% a year. "expanding" aims at the book's own average
    volatility so far, which needs no constant and adapts to whatever the
    book is; both use only days strictly before the one being sized.
    """
    trailing = gross.rolling(VOL_WINDOW).std(ddof=1).shift(1) * TRADING_DAYS ** 0.5
    if mode == "fixed":
        target = pd.Series(FIXED_TARGET, index=gross.index)
    else:
        target = (gross.expanding(MIN_HISTORY).std(ddof=1).shift(1)
                  * TRADING_DAYS ** 0.5)
    return (target / trailing).clip(upper=1.0).fillna(0.0)


def score(weights, traded, returns, cost_bps, cash_rate):
    invested = weights.sum(axis=1)
    idle = (1.0 - invested).clip(lower=0.0)
    daily_cash = cash_rate / TRADING_DAYS
    net = ((weights * returns).sum(axis=1) - traded * cost_bps / 10000.0
           + idle * daily_cash).dropna()
    excess = net - daily_cash
    equity = (1 + net).cumprod()
    downside = excess[excess < 0]
    return net, {
        "ann": float(equity.iloc[-1] ** (TRADING_DAYS / len(net)) - 1),
        "vol": float(net.std(ddof=1) * TRADING_DAYS ** 0.5),
        "sharpe": float(excess.mean() / excess.std(ddof=1) * TRADING_DAYS ** 0.5),
        "sortino": float(excess.mean() / downside.std(ddof=1) * TRADING_DAYS ** 0.5)
        if len(downside) > 2 else float("nan"),
        "dd": float((equity / equity.cummax() - 1).min()),
        "exposure": float(invested.mean()),
        "turnover": float(traded.mean() * TRADING_DAYS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Static allocations, with and "
                                                 "without the volatility overlay.")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--start", default="2006-09-18")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--split", default="2017-01-01")
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--sgov-rate", type=float, default=0.043,
                        help="What cash earns today, the bar to beat.")
    parser.add_argument("--yearly", action="store_true")
    parser.add_argument("--gold-sensitivity", action="store_true",
                        help="Re-score every allocation with gold's average "
                             "return replaced by 5%, 2.5% and 0%, keeping its "
                             "volatility and its correlations intact.")
    args = parser.parse_args()

    symbols = [s for weights in ALLOCATIONS.values() for s in weights]
    closes = load_prices(symbols, args.cache_dir, args.duration, args.start, args.end)
    returns = closes.pct_change(fill_method=None)

    books = {}
    for name, allocation in ALLOCATIONS.items():
        weights, traded = static_weights(allocation, closes, args.rebalance)
        books[name] = (weights, traded)
        gross = (weights * returns).sum(axis=1)
        for mode, tag in (("fixed", "固定16%"), ("expanding", "扩展窗口")):
            if name != "SPY 100%" and mode == "fixed":
                continue  # the fixed target is only meaningful on equities
            scaled = weights.mul(overlay_scale(gross, mode), axis=0)
            books[f"{name} + 目标化({tag})"] = (
                scaled, scaled.diff().abs().sum(axis=1).fillna(
                    scaled.abs().sum(axis=1)))

    periods = [(args.start, args.split, "2006-2016", 0.006),
               (args.split, args.end, "2017-2026", 0.022),
               (args.start, args.end, "全期", 0.015)]
    for start, end, title, cash in periods:
        mask = ((closes.index >= pd.Timestamp(start))
                & (closes.index < pd.Timestamp(end)))
        print(f"\n===== {title}（现金利率 {cash:.1%}，成本 {args.cost_bps:g} bp/边）=====")
        print(f"  {'组合':<28} {'年化':>8} {'波动':>7} {'Sharpe':>7} {'Sortino':>8} "
              f"{'回撤':>8} {'仓位':>6} {'换手':>6}")
        for label, (w, t) in books.items():
            _, s = score(w[mask], t[mask], returns[mask], args.cost_bps, cash)
            print(f"  {label:<28} {s['ann']:>+7.2%} {s['vol']:>7.2%} "
                  f"{s['sharpe']:>7.2f} {s['sortino']:>8.2f} {s['dd']:>+7.2%} "
                  f"{s['exposure']:>6.1%} {s['turnover']:>5.1f}x")

    print(f"\n[对照：SGOV 今天约 {args.sgov_rate:.1%}，零波动，零回撤]")

    if args.gold_sensitivity:
        print(f"\n[黄金收益敏感性：只改黄金的平均日收益，波动和相关性保持不变]")
        print(f"  黄金没有现金流，长期实际回报理论上接近零；这 20 年的 +10.2% "
              f"是历史异常（1980-2000 年名义上基本没涨）。")
        cash = 0.015 / TRADING_DAYS
        header = (f"  {'配置':<22} " + "".join(
            f"{'实际' if g is None else f'金{g:.1%}':>9}" for g in GOLD_SCENARIOS)
            + f"{'实际年化':>10}{'回撤':>9}")
        print(header)
        for name, allocation in ALLOCATIONS.items():
            cells = ""
            for scenario in GOLD_SCENARIOS:
                shifted = returns.copy()
                if scenario is not None and "GLD" in shifted.columns:
                    daily = (1 + scenario) ** (1 / TRADING_DAYS) - 1
                    shifted["GLD"] = (returns["GLD"] - returns["GLD"].mean() + daily)
                book = sum(shifted[k] * v for k, v in allocation.items()).dropna()
                excess = book - cash
                cells += (f"{excess.mean() / excess.std(ddof=1) * TRADING_DAYS ** 0.5:>9.2f}")
            book = sum(returns[k] * v for k, v in allocation.items()).dropna()
            equity = (1 + book).cumprod()
            print(f"  {name:<22} {cells}"
                  f"{equity.iloc[-1] ** (TRADING_DAYS / len(book)) - 1:>+9.2%}"
                  f"{(equity / equity.cummax() - 1).min():>+9.2%}")
        print("  不含黄金的配置四列相同，因为改动只作用于黄金。")

    if args.yearly:
        print(f"\n[逐年]")
        labels = list(books)
        print("  年份 " + "".join(f"{l[:14]:>16}" for l in labels))
        for year in range(closes.index.min().year, closes.index.max().year + 1):
            mask = ((closes.index >= pd.Timestamp(f"{year}-01-01"))
                    & (closes.index < pd.Timestamp(f"{year + 1}-01-01")))
            if mask.sum() < 60:
                continue
            cells = ""
            for label in labels:
                w, t = books[label]
                net, _ = score(w[mask], t[mask], returns[mask], args.cost_bps, 0.015)
                cells += f"{(1 + net).prod() - 1:>+15.2%} "
            print(f"  {year} {cells}")


if __name__ == "__main__":
    main()
