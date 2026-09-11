#!/usr/bin/env python3
"""Run the volatility-targeted position against an IBKR paper account.

One instrument, one number: hold min(16% / trailing 20-day volatility, 1)
of the account in it, and leave the rest in cash. There is no forecast of
direction here and no model to load. The evidence for the rule, and for why
none of the direction models that came before it are running instead, is in
docs/experiment-protocol.md; the rule itself and its tests are in
volatility_target.py.

Defaults to a dry run. Nothing reaches the market without --execute.

Meant to be called once a day after the close, or in the last half hour of
the session when the day's bar is close to final. Running it more often
than that does nothing but pay commission: the weight moves with a 20-day
average and the 5% band exists to stop it trading on noise.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")


def log_event(log_dir: Path, payload: dict) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(EASTERN)
    path = log_dir / f"vol_target_{stamp:%Y-%m-%d}.jsonl"
    with path.open("a") as handle:
        handle.write(json.dumps({"timestamp": stamp.isoformat(), **payload}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebalance one ETF to its volatility target.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497)
    parser.add_argument("--client-id", type=int, default=31)
    parser.add_argument("--target-volatility", type=float, default=0.16)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--cap", type=float, default=1.0)
    parser.add_argument("--band", type=float, default=0.05,
                        help="Do nothing unless the target is this far from the book.")
    parser.add_argument("--max-stale-days", type=int, default=4,
                        help="Refuse to trade on a bar older than this many calendar days.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--execute", action="store_true",
                        help="Actually send the order. Without it nothing is placed.")
    args = parser.parse_args()

    import pandas as pd

    from ibkr_ml.config import IBKRConnectionConfig
    from ibkr_ml.data import connect_ib, fetch_historical_frame, load_ib_components
    from ibkr_ml.features import to_eastern_naive
    from volatility_target import annualised_volatility, target_weight

    components = load_ib_components()
    connection = IBKRConnectionConfig(host=args.host, port=args.port,
                                      client_id=args.client_id, request_timeout=120.0,
                                      connect_retries=3, retry_delay_seconds=5.0)
    ib = connect_ib(connection)
    try:
        contract = components.Stock(args.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)

        # A year of daily bars is far more than a 20-day window needs; the
        # extra is there so a few missing days cannot silently shorten it.
        frame = fetch_historical_frame(ib=ib, symbol=args.symbol, duration="1 Y",
                                       bar_size="1 day", use_rth=True,
                                       max_duration_per_request="1 Y")
        stamps = to_eastern_naive(frame["timestamp"]).dt.normalize()
        price = pd.Series(frame["close"].to_numpy(float), index=stamps.to_numpy()).sort_index()
        returns = price.pct_change().dropna()
        if len(returns) < args.window + 5:
            raise SystemExit(f"只取到 {len(returns)} 根K线，不足以算 {args.window} 天波动")

        last_day = price.index[-1]
        age_days = (datetime.now(EASTERN).date() - last_day.date()).days
        if age_days > args.max_stale_days:
            raise SystemExit(
                f"最后一根K线是 {last_day.date()}，距今 {age_days} 天，"
                f"超过 {args.max_stale_days} 天上限，已中止")

        weight = float(target_weight(returns, args.target_volatility,
                                     args.window, args.cap).iloc[-1])
        volatility = float(annualised_volatility(returns, args.window).iloc[-1])
        last_price = float(price.iloc[-1])

        account = {row.tag: row.value for row in ib.accountSummary()
                   if row.tag == "NetLiquidation"}
        equity = float(account.get("NetLiquidation", 0.0))
        if equity <= 0:
            raise SystemExit("取不到账户净值，无法计算仓位")

        held = 0
        for item in ib.positions():
            if getattr(item.contract, "secType", "") == "STK" \
                    and item.contract.symbol == args.symbol:
                held = int(item.position)

        wanted = int(equity * weight / last_price)
        current_weight = held * last_price / equity
        delta = wanted - held

        print(f"{args.symbol}   最后一根K线 {last_day.date()}（{age_days} 天前）   "
              f"收盘 {last_price:.2f}")
        print(f"  过去 {args.window} 天年化波动 {volatility:.2%}   目标 {args.target_volatility:.0%}")
        print(f"  目标仓位 {weight:.1%}   当前仓位 {current_weight:.1%}   "
              f"账户净值 {equity:,.0f} 美元")
        print(f"  应持有 {wanted:,} 股   现持有 {held:,} 股   差 {delta:+,} 股")

        # The band is checked in weight, not in shares: a hundred shares
        # means something different on a $700 ETF than on a $40 one.
        gap = abs(weight - current_weight)
        payload = {"symbol": args.symbol, "bar_date": str(last_day.date()),
                   "volatility": volatility, "target_weight": weight,
                   "current_weight": current_weight, "equity": equity,
                   "price": last_price, "held": held, "wanted": wanted,
                   "delta": delta, "gap": gap, "executed": False}

        if gap <= args.band:
            print(f"  仓位差 {gap:.1%} 未超过不动区间 {args.band:.0%}，不交易")
            payload["action"] = "hold_inside_band"
        elif delta == 0:
            print("  股数差为 0，不交易")
            payload["action"] = "hold_zero_delta"
        elif not args.execute:
            print(f"  演练模式：本应 {'买入' if delta > 0 else '卖出'} {abs(delta):,} 股。"
                  f"加 --execute 才会真的下单")
            payload["action"] = "dry_run"
        else:
            order = components.MarketOrder("BUY" if delta > 0 else "SELL", abs(delta))
            trade = ib.placeOrder(contract, order)
            ib.sleep(2)
            status = trade.orderStatus.status
            print(f"  已下单：{'买入' if delta > 0 else '卖出'} {abs(delta):,} 股，"
                  f"状态 {status}")
            payload.update({"action": "order_sent", "executed": True,
                            "side": "BUY" if delta > 0 else "SELL",
                            "quantity": abs(delta), "status": status})

        log_event(Path(args.log_dir), payload)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
