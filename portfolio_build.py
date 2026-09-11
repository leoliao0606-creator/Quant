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


MARKER = ".what_to_show"


def check_adjusted(cache_dir):
    """Refuse to quote a number that silently left dividends out.

    Bars fetched as TRADES carry no distributions, which costs AGG about
    2.8 points a year and GLD nothing at all, so it does not shift a result
    evenly - it tilts every comparison towards whatever pays least. The
    cache records what it was fetched with in a marker file.
    """
    path = Path(cache_dir) / MARKER
    kind = path.read_text().strip() if path.exists() else ""
    if kind == "ADJUSTED_LAST":
        return
    detail = (f"标记文件写着 {kind!r}" if kind
              else f"缓存里没有 {MARKER} 标记文件，无法确认怎么取的")
    raise SystemExit(
        f"{cache_dir} 不是分红调整过的数据：{detail}。\n"
        f"未复权的成交价不含分红，AGG 会少约 2.8 个百分点/年，GLD 一分不少，"
        f"所以它不是把结果整体拉低，而是系统性偏袒不分红的资产。\n"
        f"用 --cache-dir data_cache_adj，或先跑 "
        f"fetch_assets.py --what-to-show ADJUSTED_LAST 重新下载。")


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
    frame = pd.DataFrame(columns).sort_index()

    # A symbol missing bars in the middle of its life is a gap in the feed,
    # not an absence from the market. AGG is missing 2007-08-31 to
    # 2007-10-16. Left alone, the rebalance sees it as unavailable, hands
    # its third to SPY and TIP, and hands it back when the data resumes: a
    # two-thirds round trip of turnover, charged at 5 bps, caused by
    # nothing that happened in the market, and sitting on the opening days
    # of the credit crisis. Carrying the last price forward makes those
    # days flat instead, and the real move across AGG's gap was +0.17%.
    for symbol in frame.columns:
        column = frame[symbol]
        live = column.notna()
        if not live.any():
            raise SystemExit(f"{symbol} 在这段区间里一根K线都没有")
        span = (frame.index >= live.idxmax()) & (frame.index <= live[::-1].idxmax())
        holes = int((span & column.isna()).sum())
        if holes:
            gap_days = frame.index[span & column.isna()]
            print(f"  提示: {symbol} 在上市期间缺 {holes} 个交易日"
                  f"（{gap_days.min().date()} 到 {gap_days.max().date()}），"
                  f"按最后价格顺延，不当作停牌")
            frame.loc[span, symbol] = column[span].ffill()
    return frame


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


def step_overlay(scale, every, band):
    """Only look every `every` sessions, and only move if it is worth moving.

    The backtest used to resize the book every single day while the runner
    was told to go monthly. Those are different strategies: over 2006-2026
    the daily version scores Sharpe 0.73 and the monthly one 0.62, against
    0.63 for not scaling at all. Whatever the runner can actually do has to
    be what gets measured here.
    """
    if every <= 1 and band <= 0:
        return scale
    held, current = [], 0.0
    for i, value in enumerate(scale.to_numpy()):
        if i % every == 0 and np.isfinite(value) and abs(value - current) > band:
            current = value
        held.append(current)
    return pd.Series(held, index=scale.index)


def score(weights, traded, returns, cost_bps, cash_rate):
    invested = weights.sum(axis=1)
    idle = (1.0 - invested).clip(lower=0.0)
    daily_cash = cash_rate / TRADING_DAYS
    net = ((weights * returns).sum(axis=1) - traded * cost_bps / 10000.0
           + idle * daily_cash).dropna()
    excess = net - daily_cash
    equity = (1 + net).cumprod()
    downside = excess[excess < 0]
    annual = float(equity.iloc[-1] ** (TRADING_DAYS / len(net)) - 1)
    return net, {
        "ann": annual,
        # The bar is cash, and cash paid 0.6% over 2006-2016 and 4.3% today.
        # Comparing a twenty-year average return against today's bill rate
        # compares two different things; the spread over the cash of the
        # day is the part that could carry forward.
        "excess": annual - cash_rate,
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
    parser.add_argument("--cache-dir", default="data_cache_adj",
                        help="必须是 ADJUSTED_LAST 取的数据，否则拒绝运行。")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--start", default="2006-09-18")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--split", default="2017-01-01")
    parser.add_argument("--rebalance", type=int, default=21,
                        help="多少个交易日把配置恢复到目标权重一次。")
    parser.add_argument("--overlay-every", type=int, default=5,
                        help="多少个交易日看一次波动率并调整整体仓位。"
                             "运行器实际做得到什么，这里就该填什么："
                             "每天 1、每周 5、每月 21。每周和每天几乎一样，"
                             "每月会把 Sharpe 从 0.73 打回 0.62。")
    parser.add_argument("--overlay-band", type=float, default=0.03,
                        help="整体仓位偏离超过这个比例才调整。")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--sgov-rate", type=float, default=0.043,
                        help="What cash earns today, the bar to beat.")
    parser.add_argument("--yearly", action="store_true")
    parser.add_argument("--gold-sensitivity", action="store_true",
                        help="Re-score every allocation with gold's average "
                             "return replaced by 5%, 2.5% and 0%, keeping its "
                             "volatility and its correlations intact.")
    args = parser.parse_args()

    check_adjusted(args.cache_dir)
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
            scaled = weights.mul(
                step_overlay(overlay_scale(gross, mode), args.overlay_every,
                             args.overlay_band), axis=0)
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
        print(f"  {'组合':<28} {'年化':>8} {'超额':>8} {'波动':>7} {'Sharpe':>7} "
              f"{'Sortino':>8} {'回撤':>8} {'仓位':>6} {'换手':>6}")
        for label, (w, t) in books.items():
            _, s = score(w[mask], t[mask], returns[mask], args.cost_bps, cash)
            print(f"  {label:<28} {s['ann']:>+7.2%} {s['excess']:>+7.2%} "
                  f"{s['vol']:>7.2%} {s['sharpe']:>7.2f} {s['sortino']:>8.2f} "
                  f"{s['dd']:>+7.2%} {s['exposure']:>6.1%} {s['turnover']:>5.1f}x")

    print(f"\n[对照：SGOV 今天约 {args.sgov_rate:.1%}，零波动，零回撤。")
    print(f" 要跟它比，看的是「超额」那一列，不是「年化」：现金在 2006-2016 只有")
    print(f" 0.6%、2017-2026 是 2.2%，拿二十年平均收益去减今天的 {args.sgov_rate:.1%}")
    print(f" 是拿两个时期的东西相减。能不能带到未来的是超额这个差，不是年化本身。]")

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
