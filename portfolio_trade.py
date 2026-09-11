#!/usr/bin/env python3
"""Hold a fixed allocation at its volatility target, against a paper account.

This is the only thing in this project with evidence behind it that is
worth running. Nothing here predicts anything. A fixed set of weights is
held, and the whole book is scaled down when it has been moving more than
it usually does. Both halves were tested: the allocation weights are
textbook, and the scaling rule cut the worst drawdown in 26 of 26
instrument-periods, on a 253-stock portfolio, and on every allocation in
portfolio_build.py.

What it returned over 2006-2026, monthly, net of 5 bps a side, with
dividends counted:

    stocks/bonds/TIPS in thirds   +5.50% a year, Sharpe 0.73, worst -15.0%
    40/30/15/15 with gold         +6.84% a year, Sharpe 0.77, worst -18.7%
    SPY for comparison           +11.03% a year, Sharpe 0.56, worst -55.4%

It does not beat the index on return and nothing found in this project
does. It beats the index on Sharpe, and it turns a 55% drawdown into a 15%
one. `thirds` is the honest default: every holding has a cash flow, so
none of it rests on an assumption about the price of gold. `balanced`
scored better but its weights were written after reading the sensitivity
table, which makes it a fitted answer.

Bars are fetched with ADJUSTED_LAST. Using the traded price would leave
dividends out, which understates a bond fund by nearly three points a year
and would distort both the volatility estimate and the drift between
rebalances.

Defaults to a dry run. Nothing reaches the market without --execute.

Meant to run once a month, after the close. The band exists so that a book
already close to its target is left alone; running it daily pays
commission for nothing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
TRADING_DAYS = 252
VOL_WINDOW = 20
MIN_HISTORY = 252

PRESETS = {
    "thirds": {"SPY": 1 / 3, "AGG": 1 / 3, "TIP": 1 / 3},
    "balanced": {"SPY": 0.40, "AGG": 0.30, "GLD": 0.15, "TIP": 0.15},
    "gold-thirds": {"SPY": 1 / 3, "AGG": 1 / 3, "GLD": 1 / 3},
    "sixty-forty": {"SPY": 0.6, "AGG": 0.4},
    "spy": {"SPY": 1.0},
}


def parse_allocation(text):
    """Either a preset name or "SPY:0.4,AGG:0.3,GLD:0.15,TIP:0.15"."""
    if text in PRESETS:
        return dict(PRESETS[text])
    weights = {}
    for part in text.split(","):
        symbol, _, share = part.partition(":")
        if not share:
            raise SystemExit(f"配置格式错误: {part!r}，应为 SYMBOL:权重")
        weights[symbol.strip().upper()] = float(share)
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise SystemExit(f"权重之和是 {total:.4f}，应为 1.0")
    return weights


def log_event(log_dir: Path, payload: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(EASTERN)
    path = log_dir / f"portfolio_{stamp:%Y-%m-%d}.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps({"timestamp": stamp.isoformat(), **payload}) + "\n")


def book_scale(portfolio_returns, mode, fixed_target):
    """How much of the allocation to hold today, from its own volatility.

    Both the trailing estimate and the long-run target end at yesterday, so
    today's size cannot depend on today's move. "expanding" uses the book's
    own average volatility so far and needs no constant; "fixed" aims at a
    number, which only makes sense for an all-equity book.
    """
    trailing = (portfolio_returns.rolling(VOL_WINDOW).std(ddof=1).shift(1)
                * TRADING_DAYS ** 0.5)
    if mode == "fixed":
        target = fixed_target
    else:
        target = float((portfolio_returns.expanding(MIN_HISTORY).std(ddof=1)
                        .shift(1) * TRADING_DAYS ** 0.5).iloc[-1])
    latest = float(trailing.iloc[-1])
    # Both of these come back NaN when the history is too short, and NaN
    # propagates silently: min(nan, 1.0) is nan, and the failure only
    # surfaces later as int(nan), where the message explains nothing.
    if not latest > 0:
        raise SystemExit(f"算不出组合近 {VOL_WINDOW} 天的波动率，"
                         f"只有 {len(portfolio_returns)} 天数据")
    if not target > 0:
        raise SystemExit(f"算不出长期目标波动率，扩展窗口需要至少 "
                         f"{MIN_HISTORY} 天，现有 {len(portfolio_returns)} 天")
    return min(target / latest, 1.0), latest, target


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hold a fixed allocation at its volatility target.")
    parser.add_argument("--allocation", default="thirds",
                        help="预设名 " + "/".join(PRESETS)
                             + "，或 SYMBOL:权重 的逗号列表")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=41)
    parser.add_argument("--overlay", choices=("expanding", "fixed", "none"),
                        default="expanding")
    parser.add_argument("--fixed-target", type=float, default=0.16)
    parser.add_argument("--band", type=float, default=0.03,
                        help="单个标的的仓位偏离超过这个比例才调整。")
    parser.add_argument("--max-stale-days", type=int, default=4,
                        help="最后一根K线超过这么多天就拒绝交易。")
    parser.add_argument("--duration", default="20 Y",
                        help="取多长的历史。扩展窗口目标需要足够长的历史。")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--execute", action="store_true",
                        help="真的下单。不加这个参数什么都不会送出去。")
    args = parser.parse_args()

    import pandas as pd

    from ibkr_ml.config import IBKRConnectionConfig
    from ibkr_ml.data import connect_ib, fetch_historical_frame, load_ib_components
    from ibkr_ml.features import to_eastern_naive

    allocation = parse_allocation(args.allocation)
    components = load_ib_components()
    ib = connect_ib(IBKRConnectionConfig(host=args.host, port=args.port,
                                         client_id=args.client_id,
                                         request_timeout=180.0,
                                         connect_retries=3,
                                         retry_delay_seconds=5.0))
    try:
        contracts, closes = {}, {}
        for symbol in allocation:
            contract = components.Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            contracts[symbol] = contract
            frame = fetch_historical_frame(
                ib=ib, symbol=symbol, duration=args.duration, bar_size="1 day",
                use_rth=True, max_duration_per_request=args.duration,
                what_to_show="ADJUSTED_LAST")
            stamps = to_eastern_naive(frame["timestamp"]).dt.normalize()
            closes[symbol] = pd.Series(frame["close"].to_numpy(float),
                                       index=stamps.to_numpy()).sort_index()

        prices = pd.DataFrame(closes).dropna()
        if len(prices) < MIN_HISTORY + VOL_WINDOW:
            raise SystemExit(f"只有 {len(prices)} 天的共同历史，不足以算长期目标波动")
        returns = prices.pct_change().dropna()
        portfolio = sum(returns[s] * w for s, w in allocation.items())

        last_day = prices.index[-1]
        age_days = (datetime.now(EASTERN).date() - last_day.date()).days
        if age_days > args.max_stale_days:
            raise SystemExit(f"最后一根K线是 {last_day.date()}，距今 {age_days} 天，"
                             f"超过 {args.max_stale_days} 天上限，已中止")

        if args.overlay == "none":
            scale, trailing, target = 1.0, float("nan"), float("nan")
        else:
            scale, trailing, target = book_scale(portfolio, args.overlay,
                                                 args.fixed_target)

        equity = 0.0
        for row in ib.accountSummary():
            if row.tag == "NetLiquidation":
                equity = float(row.value)
        if equity <= 0:
            raise SystemExit("取不到账户净值，无法计算仓位")

        held = {symbol: 0 for symbol in allocation}
        for item in ib.positions():
            symbol = getattr(item.contract, "symbol", "")
            if getattr(item.contract, "secType", "") == "STK" and symbol in held:
                held[symbol] = int(item.position)

        print(f"配置 {args.allocation}   最后一根K线 {last_day.date()}"
              f"（{age_days} 天前）   账户净值 {equity:,.0f} 美元")
        if args.overlay != "none":
            print(f"  组合近 {VOL_WINDOW} 天年化波动 {trailing:.2%}   "
                  f"长期目标 {target:.2%}   整体仓位 {scale:.1%}")
        print(f"  {'标的':<6} {'价格':>9} {'目标权重':>9} {'当前权重':>9} "
              f"{'应持':>8} {'现持':>8} {'差':>8}  动作")

        orders, decisions = [], []
        for symbol, share in allocation.items():
            price = float(prices[symbol].iloc[-1])
            want_weight = share * scale
            wanted = int(equity * want_weight / price)
            have = held[symbol]
            have_weight = have * price / equity
            delta = wanted - have
            gap = abs(want_weight - have_weight)
            if gap <= args.band:
                action = f"不动（差 {gap:.1%} ≤ {args.band:.0%}）"
            elif delta == 0:
                action = "不动（股数差 0）"
            else:
                action = f"{'买入' if delta > 0 else '卖出'} {abs(delta):,}"
                orders.append((symbol, delta))
            print(f"  {symbol:<6} {price:>9.2f} {want_weight:>9.1%} "
                  f"{have_weight:>9.1%} {wanted:>8,} {have:>8,} {delta:>+8,}  {action}")
            decisions.append({"symbol": symbol, "price": price,
                              "target_weight": want_weight,
                              "current_weight": have_weight, "wanted": wanted,
                              "held": have, "delta": delta, "gap": gap})

        payload = {"allocation": allocation, "bar_date": str(last_day.date()),
                   "equity": equity, "scale": scale, "trailing_vol": trailing,
                   "target_vol": target, "decisions": decisions,
                   "executed": False}

        if not orders:
            print("  所有标的都在不动区间内，不交易")
            payload["action"] = "hold"
        elif not args.execute:
            print(f"  演练模式：本应下 {len(orders)} 笔单。加 --execute 才会真的下单")
            payload["action"] = "dry_run"
        else:
            # Sells first: they free the cash the buys need, which matters on
            # an account without margin.
            results = []
            for symbol, delta in sorted(orders, key=lambda pair: pair[1]):
                order = components.MarketOrder("BUY" if delta > 0 else "SELL",
                                               abs(delta))
                trade = ib.placeOrder(contracts[symbol], order)
                ib.sleep(2)
                status = trade.orderStatus.status
                print(f"  已下单 {symbol}：{'买入' if delta > 0 else '卖出'} "
                      f"{abs(delta):,} 股，状态 {status}")
                results.append({"symbol": symbol, "quantity": abs(delta),
                                "side": "BUY" if delta > 0 else "SELL",
                                "status": status})
            payload.update({"action": "orders_sent", "executed": True,
                            "orders": results})

        log_event(Path(args.log_dir), payload)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
