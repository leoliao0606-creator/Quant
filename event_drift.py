#!/usr/bin/env python3
"""Buy the stocks that jumped on their earnings, and see whether it is real.

Two attempts to predict direction from price and volume failed on untouched
data. The common factor was the information, not the model, so this asks a
different question with the same bars: after a company reports and the
market reacts strongly upward, does the stock keep outperforming?

The two inputs that question needs are both recoverable from the bars:

  the date      an earnings day is a large move on heavy volume, and the
                events repeat about every 63 trading days. Requiring both
                conditions and a minimum spacing recovers a quarterly
                calendar without an earnings feed. It finds about 2.7
                events per symbol per year against a true 4, so it misses
                the quiet reports and catches some non-earnings news.
  the surprise  the market's own reaction on the day. The original
                statement of post-earnings-announcement drift is exactly
                that: the sign of the announcement-day return predicts what
                follows it.

Nothing here looks forward. The move and volume thresholds are trailing
medians; an event's percentile is computed against events already seen, so
the rank floor never uses a boundary drawn from the future; entry is the
close of the day *after* the event.

Three comparisons, because a backtest total can be met with no signal:

  the basket at the same average exposure - long-only in a rising market
    earns something whatever it holds
  a permutation - the same entry days and the same number of positions, on
    randomly chosen symbols, which isolates selection from timing
  a regression on the basket - beta is what leverage would have given,
    alpha is what is left

And one caveat no statistic here can remove: the default universe was
chosen in 2026 for size and liquidity, so every member is a company that
lasted. Pass --symbols with a wider list to reduce that.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_panel(symbols, cache_dir, duration, bar_size, start, end, minimum_rows=250):
    """Close and volume on one shared date index.

    Both come from the same loader deliberately. The daily cache stores a
    bare date with no timezone; reading it with utc=True treats that as UTC
    midnight and shifts the series back a day, which silently misaligned
    volume against price the first time this was written and produced zero
    detected events.
    """
    import pandas as pd

    from ibkr_ml.cache import load_cached_frame
    from ibkr_ml.features import to_eastern_naive

    closes, volumes = {}, {}
    for symbol in symbols:
        frame, _ = load_cached_frame(cache_dir, symbol, duration, bar_size, True)
        if frame is None:
            continue
        stamps = to_eastern_naive(frame["timestamp"]).dt.normalize()
        keep = (stamps >= start) & (stamps < end)
        if keep.sum() < minimum_rows:
            continue
        index = stamps[keep].to_numpy()
        closes[symbol] = pd.Series(frame.loc[keep, "close"].to_numpy(float), index=index)
        volumes[symbol] = pd.Series(frame.loc[keep, "volume"].to_numpy(float), index=index)
    if not closes:
        raise SystemExit("缓存里没有这段日期的日线数据")
    close_frame = pd.DataFrame(closes).sort_index()
    return close_frame, pd.DataFrame(volumes).sort_index().reindex(close_frame.index)


def find_events(closes, volumes, returns, basket, move_cut, volume_cut, spacing):
    """Large moves on heavy volume, thinned to at most one per quarter."""
    import pandas as pd

    move_ratio = returns.abs() / returns.abs().rolling(60).median()
    volume_ratio = volumes / volumes.rolling(60).median()
    candidate = (move_ratio > move_cut) & (volume_ratio > volume_cut)

    rows = []
    for symbol in closes.columns:
        last = None
        for day in candidate.index[candidate[symbol].fillna(False)]:
            position = closes.index.get_loc(day)
            if last is not None and position - last < spacing:
                continue
            last = position
            rows.append({"symbol": symbol, "day": day, "position": position,
                         "excess": returns[symbol].iloc[position] - basket.iloc[position]})
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)


def add_trailing_rank(events, warmup):
    """Each event's percentile among events that had already happened.

    Ranking against the whole sample would place today's threshold using
    events from years later. The first `warmup` events only build the
    reference and are never traded.
    """
    import numpy as np

    excess = events["excess"].to_numpy()
    events["rank"] = [
        float((excess[:i] < excess[i]).mean()) if i >= warmup else np.nan
        for i in range(len(excess))
    ]
    return events


def build_weights(entries, closes, hold, max_positions):
    """A daily weight matrix from (entry position, symbol) pairs."""
    import pandas as pd

    weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    slot = 1.0 / max_positions
    for entry, symbol in entries:
        if entry >= len(closes):
            continue
        column = weights.columns.get_loc(symbol)
        weights.iloc[entry:min(entry + hold, len(closes)), column] += slot
    # Two events on one symbol inside a holding period would otherwise stack
    # without limit; three slots is the cap.
    return weights.clip(upper=slot * 3)


def score(weights, returns, cost_bps):
    """Net-of-cost performance of a weight matrix."""
    gross = (weights * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    net = (gross - turnover * cost_bps / 10000.0).dropna()
    equity = (1 + net).cumprod()
    return {
        "net": net,
        "annualised": float(equity.iloc[-1] ** (252 / len(net)) - 1),
        "volatility": float(net.std(ddof=1) * 252 ** 0.5),
        "sharpe": float(net.mean() / net.std(ddof=1) * 252 ** 0.5),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "exposure": float(weights.sum(axis=1).mean()),
    }


def describe(result, label):
    print(f"  {label:<32} 年化 {result['annualised']:+7.2%}  "
          f"波动 {result['volatility']:6.2%}  Sharpe {result['sharpe']:5.2f}  "
          f"回撤 {result['max_drawdown']:+7.2%}  平均敞口 {result['exposure']:6.1%}")


def regress_on_basket(net, basket, weights):
    """Split the return into what beta explains and what it does not.

    Only days with a position open: the flat stretch before the first trade
    would drag beta toward zero and flatter the alpha.
    """
    import numpy as np

    active = weights.sum(axis=1).reindex(net.index) > 0
    y = net[active]
    x = basket.reindex(y.index)
    keep = y.notna() & x.notna()
    y, x = y[keep], x[keep]
    if len(y) < 60:
        return None
    beta = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    alpha_daily = float(y.mean() - beta * x.mean())
    residual = y - (alpha_daily + beta * x)
    standard_error = float(residual.std(ddof=2) / np.sqrt(len(y)))
    return {
        "beta": beta,
        "alpha": alpha_daily * 252,
        "t": alpha_daily / standard_error if standard_error else float("nan"),
        "information_ratio": alpha_daily / residual.std(ddof=1) * 252 ** 0.5,
        "days": len(y),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-earnings drift on cached daily bars.")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Defaults to every symbol with daily bars in the cache.")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--bar-size", default="1 day")
    parser.add_argument("--start", default="2006-09-18")
    parser.add_argument("--end", default="2017-01-01")
    parser.add_argument("--move-threshold", type=float, default=3.0,
                        help="Multiples of the trailing 60-day median absolute return.")
    parser.add_argument("--volume-threshold", type=float, default=1.8,
                        help="Multiples of the trailing 60-day median volume.")
    parser.add_argument("--spacing", type=int, default=40,
                        help="Minimum trading days between two events on one symbol.")
    parser.add_argument("--rank-floor", type=float, default=0.80,
                        help="Trade an event only if its reaction ranks this high among past events.")
    parser.add_argument("--hold", type=int, default=63,
                        help="Trading days to hold. 63 is one earnings quarter.")
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--warmup-events", type=int, default=200)
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--draws", type=int, default=200,
                        help="Permutation draws. 0 skips the control.")
    parser.add_argument("--exclude", nargs="*", default=(),
                        help="Symbols to drop from both the strategy and the control.")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    cache_dir = Path(args.cache_dir)
    tag = args.duration.replace(" ", "_")
    size = args.bar_size.replace(" ", "_")
    symbols = args.symbols or sorted(
        path.name.split("__")[0] for path in cache_dir.glob(f"*__{tag}__{size}__rth1.csv"))
    symbols = [s for s in symbols if s not in set(args.exclude)]

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    closes, volumes = load_panel(symbols, cache_dir, args.duration, args.bar_size, start, end)
    returns = closes.pct_change(fill_method=None)
    basket = returns.mean(axis=1)

    events = find_events(closes, volumes, returns, basket,
                         args.move_threshold, args.volume_threshold, args.spacing)
    events = add_trailing_rank(events, args.warmup_events)
    qualified = events.dropna(subset=["rank"]).query("rank >= @args.rank_floor")
    if qualified.empty:
        raise SystemExit("没有事件通过门槛")

    years = len(closes) / 252
    print(f"期间 {closes.index.min().date()} .. {closes.index.max().date()}   "
          f"{len(closes)} 天   {closes.shape[1]} 个标的")
    print(f"事件 {len(events):,} 个（每标的每年 {len(events)/closes.shape[1]/years:.2f} 次，"
          f"真实财报 4 次）；前 {args.warmup_events} 个仅建立阈值，"
          f"合格 {len(qualified):,} 个")

    entries = [(int(row.position) + 1, row.symbol) for row in qualified.itertuples()]
    weights = build_weights(entries, closes, args.hold, args.max_positions)
    actual = score(weights, returns, args.transaction_cost_bps)

    print(f"\n结果（成本 {args.transaction_cost_bps:g} bp/单边，持有 {args.hold} 天，"
          f"{args.max_positions} 个仓位槽）:")
    describe(actual, "事件漂移组合")
    available = closes.notna()
    equal = available.div(available.sum(axis=1), axis=0)
    describe(score(equal * actual["exposure"], returns, args.transaction_cost_bps),
             f"等权篮子，固定 {actual['exposure']:.1%} 敞口")
    describe(score(equal, returns, args.transaction_cost_bps), "等权篮子，满仓")

    fit = regress_on_basket(actual["net"], basket, weights)
    if fit:
        print(f"\n对篮子回归（{fit['days']} 个持仓日）:")
        print(f"  贝塔 {fit['beta']:.3f}   年化阿尔法 {fit['alpha']:+.2%}   "
              f"t 值 {fit['t']:+.2f}   信息比率 {fit['information_ratio']:+.2f}")

    if args.draws > 0:
        print(f"\n置换对照（同样的开仓日与笔数，随机换标的，{args.draws} 次）:")
        rng = np.random.default_rng(args.seed)
        pool = list(closes.columns)
        annualised, sharpes = [], []
        for _ in range(args.draws):
            shuffled = [(position, pool[rng.integers(len(pool))]) for position, _ in entries]
            drawn = score(build_weights(shuffled, closes, args.hold, args.max_positions),
                          returns, args.transaction_cost_bps)
            annualised.append(drawn["annualised"])
            sharpes.append(drawn["sharpe"])
        annualised, sharpes = np.array(annualised), np.array(sharpes)
        for name, values, observed in (("年化", annualised, actual["annualised"]),
                                       ("Sharpe", sharpes, actual["sharpe"])):
            beat = int((values >= observed).sum())
            fmt = (lambda v: f"{v:+.2%}") if name == "年化" else (lambda v: f"{v:.2f}")
            print(f"  {name:<7} 中位 {fmt(np.median(values))}   "
                  f"5%~95% [{fmt(np.percentile(values, 5))}, {fmt(np.percentile(values, 95))}]   "
                  f"p = {(beat + 1) / (args.draws + 1):.4f}（{beat}/{args.draws} 次不输于真实）")


if __name__ == "__main__":
    main()
