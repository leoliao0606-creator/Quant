#!/usr/bin/env python3
"""What the same basket returns without any model.

The universe was chosen in 2026 for liquidity, so running it back to 2006
selects companies that survived and grew. A long-only strategy on this basket
inherits that upward drift whether or not its model predicts anything, which
is why beating SGOV is not evidence here. This prints the drift itself, so a
strategy result can be read against it.

Sharpe is reported at full investment. Scaling a portfolio down toward cash
that earns nothing scales mean and standard deviation by the same factor, so
Sharpe does not move; only the return does, and that is reported separately
at whatever exposure the strategy actually ran.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_daily_closes(symbols, cache_dir, duration, start, end):
    """A close-price matrix, one column per symbol, dates as the index."""
    import pandas as pd

    from ibkr_ml.cache import load_cached_frame
    from ibkr_ml.features import to_eastern_naive

    series = {}
    for symbol in symbols:
        frame, _ = load_cached_frame(cache_dir, symbol, duration, "1 day", True)
        if frame is None:
            continue
        stamps = to_eastern_naive(frame["timestamp"])
        keep = (stamps >= start) & (stamps < end)
        if keep.sum() < 2:
            continue
        series[symbol] = pd.Series(
            frame.loc[keep, "close"].to_numpy(dtype=float),
            index=stamps[keep].dt.normalize().to_numpy(),
        )
    if not series:
        raise SystemExit("缓存里没有这段日期的日线数据")
    return pd.DataFrame(series).sort_index()


def summarise(daily_returns, label, exposure=1.0):
    """Annualised return, volatility, Sharpe and worst drawdown of one series."""
    scaled = daily_returns * exposure
    equity = (1.0 + scaled).cumprod()
    years = len(scaled) / 252.0
    annualised = float(equity.iloc[-1]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    volatility = float(scaled.std(ddof=1)) * (252 ** 0.5)
    sharpe = float(scaled.mean() / scaled.std(ddof=1) * (252 ** 0.5)) if scaled.std(ddof=1) > 0 else float("nan")
    drawdown = float((equity / equity.cummax() - 1.0).min())
    print(f"  {label:<34} 年化 {annualised:+7.2%}   波动 {volatility:6.2%}   "
          f"Sharpe {sharpe:5.2f}   最大回撤 {drawdown:+7.2%}")
    return {"annualized_return": annualised, "sharpe": sharpe, "max_drawdown": drawdown}


def main() -> None:
    parser = argparse.ArgumentParser(description="Buy-and-hold benchmark for the traded basket.")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Defaults to every symbol with daily bars in the cache.")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--exposure", type=float, default=1.0,
                        help="Average gross exposure of the strategy being compared.")
    args = parser.parse_args()

    import pandas as pd

    cache_dir = Path(args.cache_dir)
    if args.symbols:
        symbols = args.symbols
    else:
        symbols = sorted(
            path.name.split("__")[0]
            for path in cache_dir.glob(f"*__{args.duration.replace(' ', '_')}__1_day__rth1.csv")
        )

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    closes = load_daily_closes(symbols, cache_dir, args.duration, start, end)
    returns = closes.pct_change()

    print(f"期间 {closes.index.min().date()} .. {closes.index.max().date()}   "
          f"{len(closes)} 个交易日   {closes.shape[1]} 个标的")
    print()
    print("基准（满仓）:")
    # Equal weight rebalanced daily: the mean skips symbols that had not yet
    # listed, so early years are an average over whatever existed then.
    equal_weight = returns.mean(axis=1).dropna()
    summarise(equal_weight, "等权买入持有（每日再平衡）")
    if "SPY" in returns.columns:
        summarise(returns["SPY"].dropna(), "SPY 单独")

    if args.exposure != 1.0:
        print()
        print(f"按策略实际平均敞口 {args.exposure:.2%} 缩放后（现金按 0% 计）:")
        summarise(equal_weight, "等权买入持有", exposure=args.exposure)
        if "SPY" in returns.columns:
            summarise(returns["SPY"].dropna(), "SPY 单独", exposure=args.exposure)


if __name__ == "__main__":
    main()
