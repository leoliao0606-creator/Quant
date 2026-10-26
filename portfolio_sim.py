#!/usr/bin/env python3
"""Run the weekly allocation against a ledger kept here instead of a broker.

The account went to IBKR Lite some time after 2026-09-15 and Lite has no
API, so portfolio_trade.py can no longer reach a paper account. This keeps
the experiment going without one: the same decision functions, imported
from portfolio_trade.py so a rule cannot exist on one side only, applied to
cash and share counts held in logs/sim/<name>/state.json, with every
dividend, decision, fill and daily mark appended to ledger.jsonl.

The conventions are the runner's, not the backtest's:

- A decision is taken on Fridays only, the day run_weekly.sh trades. A
  Friday holiday skips the week, as the runner refuses a closed market.
- The decision on session D sees adjusted closes through D-1 and nothing of
  D. The runner trades at 15:30 with D's bar dropped; this fills at D's
  close, half an hour later, at the price that printed.
- Fills pay --cost-bps a side, 5 by default as in portfolio_build.py.
- The book is sized off its own equity before the trade, so gains compound
  as they do in the backtest. The runner sizes off a fixed --capital only
  because the paper account holds positions this project did not place.
- Cash earns SGOV's total return, the bar this project is trying to beat;
  portfolio_build.py credits idle cash at a cash rate for the same reason.
- Dividends are cash on the ex-date, splits multiply the share count.

Two things keep the record honest. A session is processed only if it closed
after the ledger was created, so a start date cannot be picked by looking at
how the following days went. And the parameters are frozen at creation:
changing the band or the cost is a different strategy, and a different
strategy gets a different --name.

What this cannot test is everything between deciding and being filled: a
rejected order, a partial fill, TWS holding an order behind a dialog, a
connection that drops. Those only happen at a broker, and the move to real
money has to test them there.

Run it once a day after the close; run_sim_daily.sh does. Missing days are
caught up in order on the next run, and a decision computed late is the
same decision, because it only ever reads data through the day before.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, time as time_of_day, timedelta
from pathlib import Path

from portfolio_trade import (EASTERN, MIN_HISTORY, VOL_WINDOW, book_scale,
                             drop_incomplete_bar, parse_allocation,
                             plan_targets, sessions_since, step_scale)

DECISION_WEEKDAY = 4
CASH_PROXY = "SGOV"
HISTORY_YEARS = 20
# Yahoo's daily bar keeps moving for a few minutes after 16:00 while the
# closing auction prints settle. A bar is taken as final half an hour later.
SETTLE_MINUTES = 30
# Changing any of these changes the strategy being recorded.
FROZEN_PARAMS = ("allocation", "overlay", "fixed_target", "band", "rebalance",
                 "min_trade", "cost_bps", "capital")


def session_close(day):
    return datetime.combine(day.date(), time_of_day(16, 0), EASTERN)


def new_state(name, allocation, params, created):
    return {"name": name, "allocation": allocation, "params": params,
            "created": created.isoformat(timespec="seconds"),
            "cash": float(params["capital"]),
            "shares": {symbol: 0 for symbol in allocation},
            "scale": None, "last_rebalance": None,
            "first_session": None, "last_processed": None}


def frozen_mismatch(stored, current):
    """Parameter names whose value differs from the ledger's start."""
    return [key for key in FROZEN_PARAMS if stored.get(key) != current.get(key)]


def align_bars(bars, symbols):
    """Close, adjusted close, dividend and split tables over common sessions."""
    import pandas as pd

    def table(column):
        return pd.DataFrame({s: bars[s][column] for s in symbols})

    close, adjusted = table("close"), table("adjclose")
    sessions = close.dropna().index.intersection(adjusted.dropna().index)
    return {"close": close.loc[sessions], "adjclose": adjusted.loc[sessions],
            "dividend": table("dividend").loc[sessions].fillna(0.0),
            "split": table("split").loc[sessions].fillna(1.0)}


def run_sessions(state, bars, now):
    """Process every completed session the ledger has not seen yet.

    `bars` maps each allocation symbol and CASH_PROXY to a frame from
    ibkr_ml.yahoo.parse_chart. Returns the new state and the ledger events,
    in the order they happened. Pure: it reads no clock but `now` and writes
    nothing, which is what lets the tests drive it with invented prices.
    """
    import pandas as pd

    state = json.loads(json.dumps(state))
    symbols = list(state["allocation"])
    tables = align_bars(bars, symbols + [CASH_PROXY])
    close, adjusted = tables["close"], tables["adjclose"]
    complete, _ = drop_incomplete_bar(close, now - timedelta(minutes=SETTLE_MINUTES))

    created = datetime.fromisoformat(state["created"])
    last = pd.Timestamp(state["last_processed"]) if state["last_processed"] else None
    pending = [day for day in complete.index
               if session_close(day) > created and (last is None or day > last)]

    events = []
    for day in pending:
        position = adjusted.index.get_loc(day)
        if position == 0:
            raise SystemExit(f"{day.date()} 之前没有价格，算不出当天的现金收益")
        price = {s: float(close.at[day, s]) for s in symbols}
        stamp = str(day.date())
        if state["first_session"] is None:
            state["first_session"] = stamp
            state["base_session"] = str(adjusted.index[position - 1].date())

        for symbol in symbols:
            ratio = float(tables["split"].at[day, symbol])
            held = state["shares"][symbol]
            if ratio != 1.0 and held:
                exact = held * ratio
                whole = math.floor(exact + 1e-9)
                cash_in_lieu = (exact - whole) * price[symbol]
                state["shares"][symbol] = whole
                state["cash"] += cash_in_lieu
                events.append({"type": "split", "session": stamp, "symbol": symbol,
                               "ratio": ratio, "shares_before": held,
                               "shares_after": whole, "cash_in_lieu": cash_in_lieu})
        for symbol in symbols:
            per_share = float(tables["dividend"].at[day, symbol])
            held = state["shares"][symbol]
            if per_share and held:
                state["cash"] += held * per_share
                events.append({"type": "dividend", "session": stamp, "symbol": symbol,
                               "shares": held, "per_share": per_share,
                               "amount": held * per_share})

        proxy_return = float(adjusted[CASH_PROXY].iloc[position]
                             / adjusted[CASH_PROXY].iloc[position - 1] - 1.0)
        interest = state["cash"] * proxy_return if state["cash"] > 0 else 0.0
        state["cash"] += interest

        if day.weekday() == DECISION_WEEKDAY:
            events.extend(_decide(state, adjusted, day, price))

        values = {s: state["shares"][s] * price[s] for s in symbols}
        equity = state["cash"] + sum(values.values())
        events.append({"type": "mark", "session": stamp, "cash": state["cash"],
                       "interest": interest, "equity": equity,
                       "positions": {s: {"shares": state["shares"][s],
                                         "close": price[s], "value": values[s]}
                                     for s in symbols}})
        state["last_processed"] = stamp
    return state, events


def _decide(state, adjusted, day, price):
    """One Friday: the runner's decision, then fills at the session's close."""
    params = state["params"]
    allocation = state["allocation"]
    symbols = list(allocation)
    history = adjusted.loc[:day, symbols].iloc[:-1]
    history = history[history.index > day - timedelta(days=round(365.25 * HISTORY_YEARS))]
    if len(history) < MIN_HISTORY + VOL_WINDOW:
        raise SystemExit(f"{day.date()} 之前只有 {len(history)} 天共同历史，"
                         f"不足以算长期目标波动")
    returns = history.pct_change().dropna()
    portfolio = sum(returns[s] * w for s, w in allocation.items())
    if params["overlay"] == "none":
        raw_scale, trailing, target = 1.0, float("nan"), float("nan")
    else:
        raw_scale, trailing, target = book_scale(portfolio, params["overlay"],
                                                 params["fixed_target"])
    previous_scale = state["scale"]
    scale, scale_moved = step_scale(raw_scale, previous_scale, params["band"])
    last_rebalance = (datetime.fromisoformat(state["last_rebalance"]).date()
                      if state["last_rebalance"] else None)
    elapsed = sessions_since(history.index, last_rebalance)
    rebalancing = elapsed is None or elapsed >= params["rebalance"]

    held = dict(state["shares"])
    equity = state["cash"] + sum(held[s] * price[s] for s in symbols)
    plans = plan_targets(allocation, held, price, equity, scale, previous_scale,
                         rebalancing, params["min_trade"])
    deltas = {plan["symbol"]: plan["delta"] for plan in plans}

    rate = params["cost_bps"] / 1e4
    cash_after = state["cash"] - sum(d * price[s] + abs(d) * price[s] * rate
                                     for s, d in deltas.items())
    # Whole shares and the cost on top can leave the buys a few dollars
    # short. Trim the largest buy until the cash covers it; the runner would
    # have borrowed instead, which this strategy was never tested with.
    while cash_after < -1e-9:
        buys = [s for s, d in deltas.items() if d > 0]
        if not buys:
            break
        symbol = max(buys, key=lambda s: deltas[s] * price[s])
        deltas[symbol] -= 1
        cash_after += price[symbol] * (1 + rate)

    stamp = str(day.date())
    events = [{"type": "decision", "session": stamp,
               "data_through": str(history.index[-1].date()),
               "raw_scale": raw_scale, "scale": scale, "previous_scale": previous_scale,
               "scale_moved": scale_moved, "rebalancing": rebalancing,
               "sessions_since_rebalance": elapsed, "trailing_vol": trailing,
               "target_vol": target, "equity_before": equity, "plans": plans}]
    for symbol, delta in sorted(deltas.items(), key=lambda item: item[1]):
        if not delta:
            continue
        cost = abs(delta) * price[symbol] * rate
        state["cash"] -= delta * price[symbol] + cost
        state["shares"][symbol] += delta
        events.append({"type": "fill", "session": stamp, "symbol": symbol,
                       "side": "BUY" if delta > 0 else "SELL", "shares": abs(delta),
                       "price": price[symbol], "cost": cost})
    state["scale"] = scale
    if rebalancing:
        state["last_rebalance"] = str(history.index[-1].date())
    return events


def read_marks(ledger_path):
    marks = []
    try:
        lines = Path(ledger_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return marks
    for line in lines:
        if line.strip():
            event = json.loads(line)
            if event.get("type") == "mark":
                marks.append(event)
    return marks


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    partial.replace(path)


def load_bars(symbols, cache_dir, offline):
    from ibkr_ml.yahoo import fetch_chart, load_chart, parse_chart

    bars, sources = {}, {}
    for symbol in symbols:
        payload, source = None, "Yahoo"
        if not offline:
            try:
                payload = fetch_chart(symbol, cache_dir)
            except Exception as exc:
                print(f"  {symbol}: 向 Yahoo 取数据失败（{str(exc)[:90]}），改用缓存")
        if payload is None:
            payload, source = load_chart(symbol, cache_dir), f"缓存 {cache_dir}"
            if payload is None:
                raise SystemExit(f"{symbol}: Yahoo 取不到，{cache_dir} 里也没有缓存")
        bars[symbol] = parse_chart(payload)
        sources[symbol] = source
    return bars, sources


def summarize(state, marks, bars, events):
    import pandas as pd

    if not marks:
        print("  还没有处理过任何交易日。第一个会是建立模拟之后收盘的那一天，"
              "第一次下单在那之后的第一个周五")
        return
    capital = state["params"]["capital"]
    last = marks[-1]
    path = pd.Series([capital] + [m["equity"] for m in marks])
    drawdown = float((path / path.cummax() - 1.0).min())
    print(f"  从 {state['first_session']} 到 {last['session']}，"
          f"共 {len(marks)} 个交易日")
    print(f"  净值 {last['equity']:,.2f} 美元（起始 {capital:,.0f}），"
          f"累计 {last['equity'] / capital - 1:+.2%}，最大回撤 {drawdown:.2%}")
    base = pd.Timestamp(state.get("base_session"))
    end = pd.Timestamp(last["session"])
    for symbol in ("SPY", CASH_PROXY):
        if symbol in bars and base in bars[symbol].index and end in bars[symbol].index:
            series = bars[symbol]["adjclose"]
            print(f"    同期 {symbol} 总收益 {series[end] / series[base] - 1:+.2%}")
    print(f"  {'标的':<6} {'股数':>8} {'收盘价':>10} {'市值':>12} {'占比':>7}")
    for symbol, row in last["positions"].items():
        print(f"  {symbol:<6} {row['shares']:>8,} {row['close']:>10.2f} "
              f"{row['value']:>12,.2f} {row['value'] / last['equity']:>7.1%}")
    print(f"  {'现金':<6} {'':>8} {'':>10} {last['cash']:>12,.2f} "
          f"{last['cash'] / last['equity']:>7.1%}")
    for event in events:
        if event["type"] == "decision":
            print(f"  {event['session']} 周五决策（用到 {event['data_through']} 为止的数据）："
                  f"整体仓位 {event['scale']:.1%}"
                  f"{'，再平衡到目标权重' if event['rebalancing'] else ''}")
        elif event["type"] == "fill":
            print(f"    {'买入' if event['side'] == 'BUY' else '卖出'} {event['symbol']} "
                  f"{event['shares']:,} 股 @ {event['price']:.2f}，成本 {event['cost']:.2f}")
        elif event["type"] in ("dividend", "split"):
            detail = (f"分红 {event['amount']:,.2f}" if event["type"] == "dividend"
                      else f"拆股 {event['ratio']:g} 比 1")
            print(f"    {event['session']} {event['symbol']} {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="不连券商，在本地账本上跑每周配置加波动率调节。")
    parser.add_argument("--allocation", default="thirds",
                        help="预设名或 SYMBOL:权重 列表，同 portfolio_trade.py")
    parser.add_argument("--name", default=None,
                        help="这个模拟的名字，决定账本目录。默认等于 --allocation。"
                             "换参数就是换策略，要换名字另开一个。")
    parser.add_argument("--capital", type=float, default=300000.0,
                        help="起始资金。只在建立模拟时使用")
    parser.add_argument("--overlay", choices=("expanding", "fixed", "none"),
                        default="expanding")
    parser.add_argument("--fixed-target", type=float, default=0.16)
    parser.add_argument("--band", type=float, default=0.03)
    parser.add_argument("--rebalance", type=int, default=21)
    parser.add_argument("--min-trade", type=float, default=0.005)
    parser.add_argument("--cost-bps", type=float, default=5.0,
                        help="每边成本，基点。回测默认 5")
    parser.add_argument("--state-dir", default="logs/sim")
    parser.add_argument("--cache-dir", default="data_cache_yahoo")
    parser.add_argument("--offline", action="store_true",
                        help="不联网，只用 --cache-dir 里上次存下的数据")
    parser.add_argument("--max-stale-days", type=int, default=4,
                        help="最新一根完整K线超过这么多天，就当数据源出了问题并报错")
    args = parser.parse_args()

    allocation = parse_allocation(args.allocation)
    name = args.name or args.allocation
    folder = Path(args.state_dir) / name
    state_path, ledger_path = folder / "state.json", folder / "ledger.jsonl"
    params = {"allocation": args.allocation, "overlay": args.overlay,
              "fixed_target": args.fixed_target, "band": args.band,
              "rebalance": args.rebalance, "min_trade": args.min_trade,
              "cost_bps": args.cost_bps, "capital": args.capital}
    now = datetime.now(EASTERN)

    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        changed = frozen_mismatch(state["params"], params)
        if changed:
            listing = "，".join(f"{k}: 开始时 {state['params'].get(k)!r}，现在 {params[k]!r}"
                               for k in changed)
            raise SystemExit(f"模拟 {name} 的参数和建立时不同：{listing}。\n"
                             f"换参数就是换策略，用 --name 另开一个模拟。")
    else:
        state = new_state(name, allocation, params, now)
        write_state(state_path, state)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "start", "written": now.isoformat(timespec="seconds"),
                                     "name": name, "allocation": allocation,
                                     "params": params}, ensure_ascii=False) + "\n")
        print(f"建立模拟 {name}：起始资金 {args.capital:,.0f} 美元，账本在 {folder}")

    bars, sources = load_bars(list(allocation) + [CASH_PROXY], args.cache_dir, args.offline)
    if any(s != "Yahoo" for s in sources.values()):
        print("  价格来源: " + "，".join(f"{k} {v}" for k, v in sources.items()))

    state, events = run_sessions(state, bars, now)
    if events:
        written = now.isoformat(timespec="seconds")
        with ledger_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps({"written": written, **event},
                                        ensure_ascii=False) + "\n")
    write_state(state_path, state)

    newest = min(frame.index[-1] for frame in bars.values())
    print(f"模拟 {name}   数据最新到 {newest.date()}   本次处理 "
          f"{sum(1 for e in events if e['type'] == 'mark')} 个交易日")
    summarize(state, read_marks(ledger_path), bars, events)

    age = (now.date() - newest.date()).days
    if age > args.max_stale_days:
        print(f"最新数据是 {newest.date()}，距今 {age} 天，超过 {args.max_stale_days} 天。"
              f"Yahoo 可能改了接口或拒绝了请求，查看 {args.cache_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
