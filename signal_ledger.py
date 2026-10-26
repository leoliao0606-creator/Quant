#!/usr/bin/env python3
"""Trade one ChatGPT task's reports by a fixed rule, on a local ledger.

    python signal_ledger.py --task daily      # ledger logs/sim/chatgpt-daily

Scoring each report against SPY says whether its calls were right. This
answers the question that matters for money: following them by a rule,
with a finite book, what would the account have done? The rule was fixed
before any report was pasted and is frozen into the ledger when it is
created:

- Long only. A long call (buy, opportunity) opens a position of
  `position_size` of the book's last marked equity, up to
  `max_positions` names; a call that finds the book full, the name already
  held, or the cash short is skipped and the reason recorded.
- Entry and exit are ibkr_ml.signals.simulate_plan's, so a position here
  is filled and closed exactly where the dashboard's "as planned" column
  says. With `use_levels` the report's limit, stop and target are honoured;
  without, the entry is the open and the exit is the horizon. The horizon
  is the report's horizon_days in calendar days, except for a task in
  signals.SESSION_HORIZONS: the short task holds exactly
  `default_sessions` sessions, because that is what it predicts.
- A later sell, avoid or risk call from the same task on a name the book
  holds closes it at the open of that call's entry session.
- Fills pay `cost_bps` a side; idle cash earns SGOV; dividends are cash on
  the ex-date and splits multiply the share count, as in portfolio_sim.py.
- Only reports pasted after the ledger was created count, and only
  sessions that closed after it are processed, so the record cannot start
  on a date chosen by hindsight.

Run it after the close; run_sim_daily.sh does, for all three tasks. Each
session reads only its own bar and earlier ones, so a catch-up run after
missed days books the same trades a punctual one would have.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

from portfolio_sim import CASH_PROXY, SETTLE_MINUTES, session_close
from portfolio_trade import EASTERN

BENCHMARK = "SPY"
PRESETS = {
    "daily": {"use_levels": True, "position_size": 0.10, "max_positions": 10,
              "default_sessions": 20},
    "radar": {"use_levels": False, "position_size": 0.10, "max_positions": 10,
              "default_sessions": 20},
    "short": {"use_levels": False, "position_size": 0.05, "max_positions": 20,
              "default_sessions": 5},
}
FROZEN_PARAMS = ("task", "use_levels", "position_size", "max_positions",
                 "default_sessions", "cost_bps", "capital")
SIGNAL_MAX_AGE = 30 * 60


def new_ledger(name, params, created):
    return {"kind": "signals", "name": name, "params": params,
            "created": created.isoformat(timespec="seconds"),
            "cash": float(params["capital"]), "positions": {}, "shares": {},
            "allocation": {}, "handled": {}, "scale": None, "last_rebalance": None,
            "first_session": None, "base_session": None, "last_processed": None,
            "last_equity": float(params["capital"])}


def item_id(record, index):
    return f"{record['key']}:{index}"


def run_ledger(state, records, bars_for, now):
    """Process every completed session the ledger has not seen yet.

    `records` is the whole inbox; `bars_for(ticker)` returns parsed Yahoo
    bars and must also answer for SPY and SGOV. Pure apart from those
    reads, which is what lets the tests drive it with invented prices.
    """
    from ibkr_ml.signals import (EXIT_ACTIONS, LONG_ACTIONS, SESSION_HORIZONS,
                                 entry_session, simulate_plan)

    state = json.loads(json.dumps(state))
    params = state["params"]
    use_item_horizon = params["task"] not in SESSION_HORIZONS
    created = datetime.fromisoformat(state["created"])
    benchmark, proxy = bars_for(BENCHMARK), bars_for(CASH_PROXY)
    if benchmark is None or proxy is None:
        raise SystemExit("取不到 SPY 或 SGOV 的价格，无法记账")
    calendar = benchmark.index.intersection(proxy.index)
    settled = now - timedelta(minutes=SETTLE_MINUTES)
    last = state["last_processed"]
    pending = [day for day in calendar
               if session_close(day) <= settled and session_close(day) > created
               and (last is None or str(day.date()) > last)]

    mine = [r for r in records if r["task"] == params["task"]
            and datetime.fromisoformat(r["received"]) > created]
    longs, exits = [], []
    for record in sorted(mine, key=lambda r: r["received"]):
        received = datetime.fromisoformat(record["received"])
        for index, item in enumerate(record["items"]):
            if item["action"] in LONG_ACTIONS:
                longs.append((item_id(record, index), item, received))
            elif item["action"] in EXIT_ACTIONS:
                exits.append((item, received))
    plans = {}

    def plan_for(key, item, received):
        if key not in plans:
            plans[key] = simulate_plan(item, received, bars_for(item["ticker"]), benchmark,
                                       use_levels=params["use_levels"],
                                       default_sessions=params["default_sessions"],
                                       use_item_horizon=use_item_horizon)
        return plans[key]

    rate = params["cost_bps"] / 1e4
    events = []

    def sell(ticker, price, stamp, reason):
        held = state["positions"].pop(ticker)
        cost = held["shares"] * price * rate
        state["cash"] += held["shares"] * price - cost
        events.append({"type": "fill", "session": stamp, "symbol": ticker, "side": "SELL",
                       "shares": held["shares"], "price": price, "cost": cost,
                       "reason": reason,
                       "return": price / held["fill_price"] - 1.0})

    for day in pending:
        stamp = str(day.date())
        position = calendar.get_loc(day)
        if position == 0:
            raise SystemExit(f"{stamp} 之前没有价格，算不出当天的现金收益")
        if state["first_session"] is None:
            state["first_session"] = stamp
            state["base_session"] = str(calendar[position - 1].date())

        for ticker, held in state["positions"].items():
            bars = bars_for(ticker)
            if bars is None or day not in bars.index:
                continue
            ratio = float(bars.at[day, "split"])
            if ratio != 1.0:
                exact = held["shares"] * ratio
                whole = math.floor(exact + 1e-9)
                state["cash"] += (exact - whole) * float(bars.at[day, "close"])
                events.append({"type": "split", "session": stamp, "symbol": ticker,
                               "ratio": ratio, "shares_before": held["shares"],
                               "shares_after": whole})
                held["shares"] = whole
                held["fill_price"] /= ratio
            per_share = float(bars.at[day, "dividend"])
            if per_share:
                state["cash"] += held["shares"] * per_share
                events.append({"type": "dividend", "session": stamp, "symbol": ticker,
                               "shares": held["shares"], "per_share": per_share,
                               "amount": held["shares"] * per_share})

        proxy_return = float(proxy["adjclose"].iloc[proxy.index.get_loc(day)]
                             / proxy["adjclose"].iloc[proxy.index.get_loc(calendar[position - 1])]
                             - 1.0)
        interest = state["cash"] * proxy_return if state["cash"] > 0 else 0.0
        state["cash"] += interest

        for item, received in exits:
            ticker = item["ticker"]
            bars = bars_for(ticker)
            if ticker not in state["positions"] or bars is None:
                continue
            if entry_session(bars.index, received) == day:
                opened = float(bars.at[day, "open"])
                if not math.isnan(opened):
                    sell(ticker, opened, stamp, f"报告{item['action']}信号")

        for ticker, held in list(state["positions"].items()):
            plan = plans.get(held["id"]) or plan_for(held["id"], held["item"],
                                                     datetime.fromisoformat(held["received"]))
            if plan.get("exit_session") == stamp:
                sell(ticker, plan["exit_price"], stamp, plan["status"])

        equity = state["last_equity"]
        for key, item, received in sorted(longs, key=lambda e: (e[2], -(e[1].get("score") or 0))):
            if key in state["handled"]:
                continue
            plan = plan_for(key, item, received)
            fill = plan.get("fill_session")
            if fill is None:
                # Only once the order has actually lapsed: the plan was worked
                # out on every bar there is, and a catch-up run must not book
                # the lapse before the day it happened.
                if plan["status"] == "未成交" and plan["expired_session"] <= stamp:
                    state["handled"][key] = "未成交"
                    events.append({"type": "skip", "session": stamp, "symbol": item["ticker"],
                                   "reason": "限价未成交"})
                continue
            if fill > stamp:
                continue
            ticker, price = item["ticker"], plan["fill_price"]
            reason = None
            if fill < stamp:
                reason = "错过成交日"
            elif ticker in state["positions"]:
                reason = "已持有"
            elif len(state["positions"]) >= params["max_positions"]:
                reason = "满仓"
            else:
                shares = math.floor(params["position_size"] * equity / price)
                shares = min(shares, math.floor(state["cash"] / (price * (1 + rate))))
                if shares <= 0:
                    reason = "现金不足"
            state["handled"][key] = reason or "成交"
            if reason:
                events.append({"type": "skip", "session": stamp, "symbol": ticker,
                               "reason": reason})
                continue
            cost = shares * price * rate
            state["cash"] -= shares * price + cost
            state["positions"][ticker] = {"shares": shares, "fill_price": price,
                                          "fill_session": stamp, "id": key, "item": item,
                                          "received": received.isoformat(timespec="seconds")}
            events.append({"type": "fill", "session": stamp, "symbol": ticker, "side": "BUY",
                           "shares": shares, "price": price, "cost": cost,
                           "reason": f"{item['action']}，评分 {item.get('score')}"})
            if plan.get("exit_session") == stamp:
                sell(ticker, plan["exit_price"], stamp, plan["status"])

        marks = {}
        for ticker, held in state["positions"].items():
            bars = bars_for(ticker)
            known = bars["close"][bars.index <= day] if bars is not None else []
            close = float(known.iloc[-1]) if len(known) else held["fill_price"]
            marks[ticker] = {"shares": held["shares"], "close": close,
                             "value": held["shares"] * close}
        equity = state["cash"] + sum(m["value"] for m in marks.values())
        state["last_equity"] = equity
        state["shares"] = {t: h["shares"] for t, h in state["positions"].items()}
        events.append({"type": "mark", "session": stamp, "cash": state["cash"],
                       "interest": interest, "equity": equity, "positions": marks})
        state["last_processed"] = stamp
    return state, events


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    partial.replace(path)


def main() -> None:
    from ibkr_ml.signals import load_inbox
    from ibkr_ml.yahoo import cached_bars

    parser = argparse.ArgumentParser(description="按固定规则交易某个 ChatGPT 任务的报告")
    parser.add_argument("--task", required=True, choices=sorted(PRESETS))
    parser.add_argument("--name", default=None, help="账本名，默认 chatgpt-<task>")
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--state-dir", default="logs/sim")
    parser.add_argument("--signals-dir", default="logs/signals")
    parser.add_argument("--cache-dir", default="data_cache_yahoo")
    args = parser.parse_args()

    name = args.name or f"chatgpt-{args.task}"
    params = {"task": args.task, **PRESETS[args.task], "cost_bps": args.cost_bps,
              "capital": args.capital}
    folder = Path(args.state_dir) / name
    state_path, ledger_path = folder / "state.json", folder / "ledger.jsonl"
    now = datetime.now(EASTERN)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        changed = [k for k in FROZEN_PARAMS if state["params"].get(k) != params.get(k)]
        if changed:
            listing = "，".join(f"{k}: 开始时 {state['params'].get(k)!r}，现在 {params[k]!r}"
                               for k in changed)
            raise SystemExit(f"账本 {name} 的规则和建立时不同：{listing}。\n"
                             f"换规则就是换策略，用 --name 另开一个账本。")
    else:
        state = new_ledger(name, params, now)
        write_state(state_path, state)
        with ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"type": "start", "written": now.isoformat(timespec="seconds"),
                                     "name": name, "params": params}, ensure_ascii=False) + "\n")
        print(f"建立账本 {name}：{params}")

    records = load_inbox(args.signals_dir)
    earliest = min((datetime.fromisoformat(r["received"]) for r in records), default=now)
    start = min(earliest - timedelta(days=10), now - timedelta(days=730))
    memo = {}

    def bars_for(ticker):
        if ticker not in memo:
            memo[ticker] = cached_bars(ticker, Path(args.cache_dir) / "signals", start,
                                       SIGNAL_MAX_AGE)
        return memo[ticker]

    state, events = run_ledger(state, records, bars_for, now)
    if events:
        written = now.isoformat(timespec="seconds")
        with ledger_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps({"written": written, **event},
                                        ensure_ascii=False, default=str) + "\n")
    write_state(state_path, state)

    capital = state["params"]["capital"]
    print(f"账本 {name}   记到 {state['last_processed'] or '尚未记账'}   "
          f"净值 {state['last_equity']:,.2f}（{state['last_equity'] / capital - 1:+.2%}）   "
          f"持仓 {len(state['positions'])}/{state['params']['max_positions']}")
    for event in events:
        if event["type"] == "fill":
            print(f"  {event['session']} {'买入' if event['side'] == 'BUY' else '卖出'} "
                  f"{event['symbol']} {event['shares']} 股 @ {event['price']:.2f}（{event['reason']}）")
        elif event["type"] == "skip":
            print(f"  {event['session']} 跳过 {event['symbol']}：{event['reason']}")


if __name__ == "__main__":
    main()
