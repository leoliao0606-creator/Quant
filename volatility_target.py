#!/usr/bin/env python3
"""Hold less when volatility is high. The one rule here that held up.

Six ways of predicting direction were tried in this project and none
survived a hold-out; the record is in docs/experiment-protocol.md. This
forecasts nothing about direction. It sets position size from how much the
instrument has been moving, which is the one quantity that is genuinely
predictable, and it is the only result whose parameters were fixed in one
period and confirmed in another.

    weight = min(target_volatility / trailing_volatility, 1.0)

Every constant is set from outside the data:

  target 16%   roughly the long-run annualised volatility of US equities.
               An in-sample estimate was tried and did slightly worse.
  window 20    the trailing window. HAR-RV forecasts volatility better and
               sizes worse; 60-day and EWMA variants were no better.
  cap 1.0      the rule only ever reduces exposure. Letting it borrow in
               calm stretches drops SPY's 2006-2016 Sharpe from 0.47 to
               0.27, because calm stretches are what precede crashes.

Measured against buy-and-hold, 5 bps per side, idle cash at 4.3%:

    SPY 2006-2016   Sharpe 0.35 -> 0.50   worst drawdown -56.5% -> -36.0%
    SPY 2017-2026   Sharpe 0.78 -> 0.90   worst drawdown -34.1% -> -19.7%
    QQQ 2006-2016   Sharpe 0.60 -> 0.75   worst drawdown -53.5% -> -31.3%
    QQQ 2017-2026   Sharpe 0.92 -> 1.09   worst drawdown -35.6% -> -21.4%

It gives up half a point to three points of annual return for a drawdown
about a third smaller. It is not alpha and will not beat the index on
return; levered to matched volatility with borrowing charged at 6%, the
return advantage is 0.43 points a year at identical Sharpe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

TARGET_VOLATILITY = 0.16
WINDOW = 20
CAP = 1.0
TRADING_DAYS = 252


def annualised_volatility(returns, window=WINDOW):
    return returns.rolling(window).std(ddof=1) * TRADING_DAYS ** 0.5


def target_weight(returns, target=TARGET_VOLATILITY, window=WINDOW, cap=CAP):
    """Weight for each day, using only returns available the day before."""
    realised = annualised_volatility(returns, window).shift(1)
    return (target / realised).clip(upper=cap).fillna(0.0)


def apply_band(weights, band):
    """Trade only when the target has moved more than `band` from the book.

    Rebalancing to three decimal places every day pays commission for
    changes too small to matter. A band leaves the position alone until the
    gap is worth closing.
    """
    if band <= 0:
        return weights
    held = []
    current = 0.0
    for value in weights:
        if abs(value - current) > band:
            current = value
        held.append(current)
    return type(weights)(held, index=weights.index)


def backtest(returns, weights, cost_bps=5.0, cash_rate=0.043):
    turnover = weights.diff().abs().fillna(weights.abs())
    idle = (1.0 - weights).clip(lower=0.0)
    net = (weights * returns - turnover * cost_bps / 10000.0
           + idle * cash_rate / TRADING_DAYS).dropna()
    equity = (1 + net).cumprod()
    return {
        "annualised": float(equity.iloc[-1] ** (TRADING_DAYS / len(net)) - 1),
        "volatility": float(net.std(ddof=1) * TRADING_DAYS ** 0.5),
        "sharpe": float(net.mean() / net.std(ddof=1) * TRADING_DAYS ** 0.5),
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "exposure": float(weights.mean()),
        "turnover": float(turnover.mean() * TRADING_DAYS),
    }


def load_returns(symbol, cache_dir, duration, start, end):
    import pandas as pd

    from ibkr_ml.cache import load_cached_frame
    from ibkr_ml.features import to_eastern_naive

    frame, _ = load_cached_frame(cache_dir, symbol, duration, "1 day", True)
    if frame is None:
        raise SystemExit(f"缓存里没有 {symbol} 的日线数据")
    stamps = to_eastern_naive(frame["timestamp"]).dt.normalize()
    keep = stamps.notna()
    if start:
        keep &= stamps >= pd.Timestamp(start)
    if end:
        keep &= stamps < pd.Timestamp(end)
    price = pd.Series(frame.loc[keep, "close"].to_numpy(float), index=stamps[keep].to_numpy())
    return price, price.pct_change().dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="Volatility-targeted position size.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--equity", type=float, default=100000.0)
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--target-volatility", type=float, default=TARGET_VOLATILITY)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--cap", type=float, default=CAP)
    parser.add_argument("--band", type=float, default=0.05,
                        help="Leave the position alone until the target moves this far.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--cash-rate", type=float, default=0.043)
    parser.add_argument("--backtest", action="store_true",
                        help="Replay the rule over the cached history instead of "
                             "reporting today's size.")
    args = parser.parse_args()

    price, returns = load_returns(args.symbol, Path(args.cache_dir), args.duration,
                                  args.start, args.end)
    raw = target_weight(returns, args.target_volatility, args.window, args.cap)
    weights = apply_band(raw, args.band)

    if args.backtest:
        print(f"{args.symbol}  {returns.index.min().date()} .. {returns.index.max().date()}   "
              f"{len(returns)} 天   目标波动 {args.target_volatility:.0%}   "
              f"窗口 {args.window} 天   上限 {args.cap:g}   不动区间 {args.band:.0%}")
        held = backtest(returns, raw * 0 + 1.0, args.transaction_cost_bps, args.cash_rate)
        print(f"  {'买入持有':<22} 年化 {held['annualised']:+7.2%}  "
              f"波动 {held['volatility']:6.2%}  Sharpe {held['sharpe']:5.2f}  "
              f"回撤 {held['max_drawdown']:+7.2%}")
        for label, series in (("波动率目标化（每日调）", raw), ("波动率目标化（带不动区间）", weights)):
            result = backtest(returns, series, args.transaction_cost_bps, args.cash_rate)
            print(f"  {label:<22} 年化 {result['annualised']:+7.2%}  "
                  f"波动 {result['volatility']:6.2%}  Sharpe {result['sharpe']:5.2f}  "
                  f"回撤 {result['max_drawdown']:+7.2%}  仓位 {result['exposure']:5.1%}  "
                  f"年换手 {result['turnover']:4.1f}")
        return

    recent = annualised_volatility(returns, args.window)
    last_price = float(price.iloc[-1])
    weight = float(weights.iloc[-1])
    shares = int(args.equity * weight / last_price)
    print(f"{args.symbol}   最新一根K线 {price.index[-1].date()}   收盘 {last_price:.2f}")
    print(f"  过去 {args.window} 天年化波动 {float(recent.iloc[-1]):.2%}   "
          f"目标 {args.target_volatility:.0%}")
    print(f"  目标仓位 {weight:.1%}   账户 {args.equity:,.0f} 美元   "
          f"应持有 {shares:,} 股（约 {shares * last_price:,.0f} 美元）")
    print(f"  近 10 个交易日的目标仓位:")
    for day, value, vol in zip(weights.index[-10:], weights.iloc[-10:], recent.iloc[-10:]):
        print(f"    {day.date()}  波动 {vol:6.2%}  仓位 {value:6.1%}")
    if price.index[-1].date().isoformat() < "2026-09-09":
        print("  提示: 缓存里的最后一根K线不是最近的交易日，先更新数据再据此下单")


if __name__ == "__main__":
    main()
