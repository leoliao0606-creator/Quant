#!/usr/bin/env python3
"""Cross-sectional signals, scored under three timing conventions.

An earlier run of this reported that twelve-month momentum earns +6.66% a
year over the basket on a 253-name universe, with a permutation p of
0.0099. That was wrong, and this script is what found it.

The mistake was in when the weights started earning. The signal was read on
day D and the position began collecting day D's return, so a signal that
contains day D's closing price -- `closes / closes.shift(252)`, or
`closes / closes.rolling(200).mean()` -- chose its names using a price that
had not printed yet when the trade was supposed to happen. That is
look-ahead bias: the backtest knows something the account would not have.

Three conventions are reported side by side so the size of the error is
visible rather than argued about:

  A as-run        what the first version did: signal on day D, weights on
                  day D, day D's return collected.
  B lagged        the signal is shifted one day, so day D's weights use
                  only what had printed by the close of D-1.
  C lagged+drift  as B, and the holdings are left to move with prices
                  between rebalances. A does something subtler than it
                  looks: holding a fixed 1/50 every day silently sells the
                  winners and buys the losers each morning, free of charge.

The random control is also repaired. It used to draw from every column,
including names that had not listed yet, whose price is NaN; those
positions contributed nothing, so the control was not fully invested and
was too easy to beat. It now draws only from names trading that day.

Alpha is measured against the equal-weight basket of the same universe, and
the t on alpha is reported twice: ordinary least squares, which assumes one
day's residual says nothing about the next, and Newey-West, which does not
make that assumption. Holdings are kept 21 days, so the residuals are
related and the OLS number is the more generous of the two.
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

# Funds, not companies: including them would mix a diversified basket into a
# single-stock ranking and let the signal "pick" the market itself.
FUNDS = {
    "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLV", "XLE", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLC", "VXX", "TLT", "IEF", "SHY", "AGG", "LQD",
    "HYG", "TIP", "GLD", "SLV", "DBC", "EFA", "EEM", "VNQ", "IYR", "BND",
}
TRADING_DAYS = 252
OVERLAY_TARGET = 0.16
OVERLAY_WINDOW = 20


def load_universe(cache_dir, duration, start, end, min_bars):
    """Every cached company with at least `min_bars` days inside the window."""
    names = sorted({p.name.split("__")[0]
                    for p in Path(cache_dir).glob(f"*__{duration.replace(' ', '_')}"
                                                  "__1_day__rth1.csv")} - FUNDS)
    columns = {}
    for symbol in names:
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
        raise SystemExit(f"{cache_dir} 里没有符合条件的日线数据")
    return pd.DataFrame(columns).sort_index()


def signal_table(closes, returns):
    return {
        "动量 12-1": closes.shift(21) / closes.shift(252) - 1.0,
        "动量 12-0": closes / closes.shift(252) - 1.0,
        "动量 6-0": closes / closes.shift(126) - 1.0,
        "短期反转": -(closes / closes.shift(21) - 1.0),
        "低波动": -returns.rolling(126).std(),
        "趋势 价格/200日均线": closes / closes.rolling(200).mean() - 1.0,
    }


def choose_names(signal, available, rebalance_days, top, lag):
    """Top `top` names on each rebalance date, from the signal `lag` days back.

    Carrying the previous selection forward when too few names have a value
    keeps the early history from starting with an empty book.
    """
    source = signal.shift(lag).where(available.shift(lag).fillna(False))
    picks, last = {}, None
    for day in rebalance_days:
        row = source.loc[day].dropna()
        last = list(row.nlargest(top).index) if len(row) >= top else last
        if last:
            picks[day] = last
    return picks


def reset_daily(picks, closes, rebalance_days, top):
    """Hold exactly 1/top of each name every day, as the first version did."""
    grid = pd.DataFrame(0.0, index=rebalance_days, columns=closes.columns)
    for day, names in picks.items():
        grid.loc[day, names] = 1.0 / top
    weights = grid.reindex(closes.index, method="ffill").fillna(0.0)
    traded = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    return weights, traded


def let_drift(picks, closes, filled, top):
    """Buy equal amounts on a rebalance date, then let prices move them.

    weights.loc[D] is what is held *during* day D, established at the close
    of D-1, so it grows with D-1's return. Growing it with D's own return
    would multiply each day's return by a weight already containing it,
    which lifts every signal regardless of merit -- the squared return is
    always positive.
    """
    prior = filled.shift(1).fillna(0.0).to_numpy()
    columns = list(closes.columns)
    index_of = {name: i for i, name in enumerate(columns)}
    out = np.zeros((len(closes), len(columns)))
    traded = np.zeros(len(closes))
    positions = np.zeros(len(columns))
    for i, day in enumerate(closes.index):
        positions = positions * (1.0 + prior[i])
        total = positions.sum()
        if total > 0:
            positions = positions / total
        if day in picks:
            target = np.zeros(len(columns))
            for name in picks[day]:
                target[index_of[name]] = 1.0 / top
            traded[i] = float(np.abs(target - positions).sum())
            positions = target
        out[i] = positions
    return (pd.DataFrame(out, index=closes.index, columns=columns),
            pd.Series(traded, index=closes.index))


def score(weights, traded, returns, cost_bps, cash_rate=0.0):
    idle = (1.0 - weights.sum(axis=1)).clip(lower=0.0)
    daily_cash = cash_rate / TRADING_DAYS
    net = ((weights * returns).sum(axis=1) - traded * cost_bps / 10000.0
           + idle * daily_cash).dropna()
    excess = net - daily_cash
    equity = (1 + net).cumprod()
    return net, {
        "ann": float(equity.iloc[-1] ** (TRADING_DAYS / len(net)) - 1),
        "vol": float(net.std(ddof=1) * TRADING_DAYS ** 0.5),
        "sharpe": float(excess.mean() / excess.std(ddof=1) * TRADING_DAYS ** 0.5),
        "dd": float((equity / equity.cummax() - 1).min()),
        "turnover": float(traded.mean() * TRADING_DAYS),
        "exposure": float(weights.sum(axis=1).mean()),
    }


def alpha_against(net, basket, lags=21):
    """Alpha and two t statistics from a regression on the basket.

    Newey-West widens the standard error using the residual's own
    autocorrelation out to `lags` days, which matters because a position is
    kept for 21 days and consecutive days are therefore not independent.
    """
    keep = net.notna() & basket.reindex(net.index).notna()
    y = net[keep].to_numpy()
    x = basket.reindex(net.index)[keep].to_numpy()
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    resid = y - X @ coef
    ols_t = coef[0] / (resid.std(ddof=2) / np.sqrt(n))
    moments = X * resid[:, None]
    S = moments.T @ moments
    for lag in range(1, lags + 1):
        G = moments[lag:].T @ moments[:-lag]
        S += (1.0 - lag / (lags + 1.0)) * (G + G.T)
    cov = xtx_inv @ S @ xtx_inv
    return (float(coef[0]) * TRADING_DAYS, float(coef[1]),
            float(ols_t), float(coef[0] / np.sqrt(cov[0, 0])))


def overlay_weights(gross, target=OVERLAY_TARGET, window=OVERLAY_WINDOW):
    """min(target / trailing volatility, 1), using only prior days."""
    trailing = gross.rolling(window).std(ddof=1).shift(1) * TRADING_DAYS ** 0.5
    return (target / trailing).clip(upper=1.0).fillna(0.0)


def run_timing_comparison(args, closes, returns, filled, available, basket,
                          rebalance_days, top):
    _, base = score(available.div(available.sum(axis=1), axis=0),
                    pd.Series(0.0, index=closes.index), returns, args.cost_bps)
    print(f"区间 {closes.index.min().date()} .. {closes.index.max().date()}   "
          f"{len(closes)} 天   {closes.shape[1]} 只   前 {top} 名   "
          f"每 {args.rebalance} 天调仓   成本 {args.cost_bps:g} bp/边")
    print(f"基准 等权篮子   年化 {base['ann']:+.2%}   Sharpe {base['sharpe']:.2f}   "
          f"回撤 {base['dd']:+.1%}\n")

    header = (f"{'信号':<20} {'口径':<14} {'年化':>8} {'Sharpe':>7} {'回撤':>7} "
              f"{'换手':>6} {'贝塔':>6} {'阿尔法':>8} {'t(OLS)':>7} {'t(NW)':>7}")
    print(header)
    print("-" * len(header))
    lagged_sharpe = {}
    for name, raw in signal_table(closes, returns).items():
        for tag, lag, drift in (("A 原版", 0, False),
                                ("B 滞后一天", 1, False),
                                ("C B+权重漂移", 1, True)):
            picks = choose_names(raw, available, rebalance_days, top, lag)
            weights, traded = (let_drift(picks, closes, filled, top) if drift
                               else reset_daily(picks, closes, rebalance_days, top))
            net, s = score(weights, traded, returns, args.cost_bps)
            alpha, beta, ols_t, nw_t = alpha_against(net, basket)
            if tag.startswith("B"):
                lagged_sharpe[name] = s["sharpe"]
            print(f"{name if tag.startswith('A') else '':<20} {tag:<14} "
                  f"{s['ann']:>+7.2%} {s['sharpe']:>7.2f} {s['dd']:>+6.1%} "
                  f"{s['turnover']:>5.1f}x {beta:>6.2f} {alpha:>+7.2%} "
                  f"{ols_t:>+7.2f} {nw_t:>+7.2f}")
        print()

    if not args.draws:
        return
    print(f"[随机对照：只从当天在市的名字里抽 {top} 只，{args.draws} 次，"
          f"与滞后口径比较]")
    rng = np.random.default_rng(args.seed)
    columns = np.array(closes.columns)
    listed = {day: columns[available.loc[day].to_numpy()] for day in rebalance_days}
    control = []
    for _ in range(args.draws):
        draw = {d: list(rng.choice(listed[d], size=min(top, len(listed[d])),
                                   replace=False)) for d in rebalance_days}
        weights, traded = reset_daily(draw, closes, rebalance_days, top)
        control.append(score(weights, traded, returns, args.cost_bps)[1]["sharpe"])
    control = np.array(control)
    print(f"  随机组合 Sharpe 中位数 {np.median(control):.2f}   "
          f"区间 [{control.min():.2f}, {control.max():.2f}]")
    for name, sharpe in lagged_sharpe.items():
        p = (int((control >= sharpe).sum()) + 1) / (args.draws + 1)
        print(f"  {name:<20} Sharpe {sharpe:.2f}   p = {p:.4f}")


def run_overlay(args, closes, returns, filled, available, rebalance_days, top):
    """Momentum picks the names, the volatility rule decides the size."""
    signal = closes / closes.shift(252) - 1.0
    picks = choose_names(signal, available, rebalance_days, top, args.lag)
    weights, traded = let_drift(picks, closes, filled, top)
    gross = (weights * returns).sum(axis=1)
    sized = weights.mul(overlay_weights(gross), axis=0)
    equal = available.div(available.sum(axis=1), axis=0)

    variants = {
        "等权篮子（基准）": (equal, equal.diff().abs().sum(axis=1).fillna(1.0)),
        "动量选股": (weights, traded),
        "动量 + 波动率目标化": (sized, sized.diff().abs().sum(axis=1).fillna(
            sized.abs().sum(axis=1))),
    }
    periods = [(args.start, "2017-01-01", "2006-2016", 0.006),
               ("2017-01-01", args.end, "2017-2026", 0.022),
               (args.start, args.end, "全期", 0.015)]
    print(f"信号滞后 {args.lag} 天   {closes.shape[1]} 只   前 {top} 名   "
          f"持仓随价格漂移，只在调仓日交易")
    for start, end, title, cash in periods:
        mask = ((closes.index >= pd.Timestamp(start))
                & (closes.index < pd.Timestamp(end)))
        if mask.sum() < 100:
            continue
        print(f"\n===== {title}（现金利率 {cash:.1%}）=====")
        for label, (w, t) in variants.items():
            _, s = score(w[mask], t[mask], returns[mask], args.cost_bps, cash)
            print(f"  {label:<22} 年化 {s['ann']:+7.2%}  波动 {s['vol']:6.2%}  "
                  f"Sharpe {s['sharpe']:5.2f}  回撤 {s['dd']:+7.2%}  "
                  f"仓位 {s['exposure']:5.1%}")

    print(f"\n[逐年]  {'年份':<6} {'等权篮子':>10} {'动量':>10} {'动量+目标化':>12}")
    years = range(closes.index.min().year, closes.index.max().year + 1)
    for year in years:
        mask = ((closes.index >= pd.Timestamp(f"{year}-01-01"))
                & (closes.index < pd.Timestamp(f"{year + 1}-01-01")))
        if mask.sum() < 60:
            continue
        cells = []
        for label, (w, t) in variants.items():
            net, _ = score(w[mask], t[mask], returns[mask], args.cost_bps, 0.015)
            cells.append(f"{(1 + net).prod() - 1:+9.2%}")
        print(f"        {year:<6} {cells[0]:>10} {cells[1]:>10} {cells[2]:>12}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score cross-sectional signals under three timing conventions.")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--start", default="2006-09-18")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--min-bars", type=int, default=400)
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--lag", type=int, default=1,
                        help="Days between reading the signal and holding the "
                             "position. 0 reproduces the biased original.")
    parser.add_argument("--overlay", action="store_true",
                        help="Momentum selection with the volatility target on "
                             "top, by period and by year.")
    args = parser.parse_args()

    closes = load_universe(args.cache_dir, args.duration, args.start, args.end,
                           args.min_bars)
    returns = closes.pct_change(fill_method=None)
    filled = returns.fillna(0.0)
    available = closes.notna()
    basket = (available.div(available.sum(axis=1), axis=0) * returns).sum(axis=1)
    top = max(int(closes.shape[1] * args.top_fraction), 10)
    rebalance_days = closes.index[::args.rebalance]

    if args.overlay:
        run_overlay(args, closes, returns, filled, available, rebalance_days, top)
    else:
        run_timing_comparison(args, closes, returns, filled, available, basket,
                              rebalance_days, top)


if __name__ == "__main__":
    main()
