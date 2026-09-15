#!/usr/bin/env python3
"""Hold a fixed allocation at its volatility target, against a paper account.

This is the only thing in this project with evidence behind it that is
worth running. Nothing here predicts anything. A fixed set of weights is
held, and the whole book is scaled down when it has been moving more than
it usually does. Both halves were tested: the allocation weights are
textbook, and the scaling rule cut the worst drawdown in 26 of 26
instrument-periods, on a 253-stock portfolio, and on every allocation in
portfolio_build.py.

What it returned over 2006-2026, net of 5 bps a side, with dividends
counted, for stocks/bonds/TIPS in thirds:

    allocation restored monthly, no scaling   +5.89%, Sharpe 0.63, -22.1%
    scaling checked every session             +5.53%, Sharpe 0.73, -14.9%
    scaling checked weekly                    +5.53%, Sharpe 0.72, -15.0%
    scaling checked monthly                   +5.16%, Sharpe 0.62, -16.7%
    SPY for comparison                       +11.03%, Sharpe 0.56, -55.4%

**Run this weekly.** An earlier version of this note quoted the daily
figure while telling the reader to run it monthly, and the two differ by
0.10 of Sharpe and 1.8 points of drawdown - at monthly the scaling is
worth nothing against simply holding the allocation (0.62 against 0.63)
and only the drawdown improves. The cliff sits between weekly and
fortnightly; weekly costs 0.01 of Sharpe against every session, and the
band matters hardly at all (3% and 10% are within 0.01 of each other).

Two clocks, not one, because that is what was measured. The allocation is
restored to its target weights every --rebalance sessions (21, a month) and
is left to drift with price in between; the book scale is looked at on every
run and only adopted when it has moved more than --band (3%). These are
portfolio_build.py's --rebalance and --overlay-band, and the results above
come from running both. An earlier version of this file ran neither: it
compared --band against a single symbol's target weight, which in `thirds`
is a third of the book, so the 3% band needed a 9% move in the scale before
it fired - three times slower than the rule that scored 0.72 - and it
restored the weights on every run rather than monthly. Remembering the last
scale and the last rebalance date is what --state-file is for; deleting that
file makes the next run a first run, which rebalances immediately.

The gold allocation `balanced` returned +6.84% at Sharpe 0.77 on the daily
convention; it has not been re-measured weekly, and its weights were
chosen after reading a sensitivity table, so treat both numbers as soft.

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

Run once a week, late in the session and not after it. Bars for a session
still in progress are dropped before anything is measured, so sizing uses
every completed session up to and including the last one - the same
convention the backtest uses, where a weight set from data through day D-1
collects day D's return.

It has to be inside the session because the orders are market orders, and
a market order sent after the close is not executed, it is parked until the
next open. Nothing here waits that long, so every order would read as
unfilled, the buys would be skipped by the rule that holds them back until
the sells are done, and the book would sit half rebalanced until somebody
noticed. The run refuses to send anything when IBKR says the market is
shut, which also covers holidays and early closes.

The account is assumed to hold nothing but this allocation. Positions in
anything else make the position sizes wrong, because they are computed
from net liquidation; the run stops and says so unless --capital states
which slice of the account this strategy is allowed to use.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

# Neither of these pulls in pandas at import time, so --help still works
# without it. SESSION_CLOSE_HOUR is shared rather than redefined: the cache
# drops a partial session on the way in and this file drops one on the way
# out, and two copies of "the close is 16:00" is one copy that can be missed.
from ibkr_ml.cache import SESSION_CLOSE_HOUR

EASTERN = ZoneInfo("America/New_York")
TRADING_DAYS = 252
VOL_WINDOW = 20
MIN_HISTORY = 252
# Anything else means the order is still working and its shares are not in
# the position count yet.
SETTLED_STATUS = frozenset({"Filled", "Cancelled", "ApiCancelled", "Inactive"})
# What IBKR says when it will not act on a cancel request. An orderId belongs
# to the client id that placed it, so a cancel sent under a different one names
# an order the broker cannot find. Measured against the paper account on
# 2026-09-14: cancelling client id 62's order from client id 41 came back as
# "Error 10147, reqId 5: OrderId 5 that needs to be cancelled is not found."
CANCEL_REFUSED_CODES = frozenset({135, 10147, 10148})

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


def drop_incomplete_bar(prices, now):
    """Remove a bar for a session that has not finished trading.

    IBKR hands back a partial bar for the session in progress. A few hours
    of trading counted as a day understates the range, so the estimate
    would read calmer than the market is. The caller needs the series to
    end at the last completed session; this is what guarantees it.
    """
    if len(prices) == 0:
        return prices, False
    last = prices.index[-1]
    if last.date() > now.date() or (last.date() == now.date()
                                    and now.hour < SESSION_CLOSE_HOUR):
        return prices.iloc[:-1], True
    return prices, False


def split_working_orders(trades, allocation):
    """Working orders that make the position count wrong, and the rest.

    Shares sitting in an unfilled order are not in the position count, so a
    second run sees the same gap and sends the same order a second time.
    That only happens for an order on a symbol this allocation trades; an
    order on anything else leaves the share arithmetic alone and must not
    stop the run, or one manual limit order elsewhere in the account blocks
    the weekly rebalance for a week.
    """
    working = [t for t in trades
               if getattr(t.orderStatus, "status", "") not in SETTLED_STATUS]
    mine = [t for t in working
            if getattr(t.contract, "symbol", "") in allocation]
    others = [t for t in working
              if getattr(t.contract, "symbol", "") not in allocation]
    return mine, others


def cancel_and_wait(ib, trades, allocation, timeout, clock=time.monotonic):
    """Cancel these orders; stop the moment IBKR says it will not.

    Returns the ones still working and whatever IBKR said about refusing.

    A refusal arrives as an error, not as a status change on the order, and
    ib_insync files an incoming error under (this connection's client id,
    reqId) at ib_insync/wrapper.py:1096. An error about an order some other
    client id placed therefore matches no order here, nothing writes to its
    status, and it sits at PendingCancel for as long as anyone waits. Measured
    on 2026-09-14: the refusal came back one millisecond after the request and
    the run then waited out the full sixty second --fill-timeout to learn
    nothing further. Reading the error stream ends it after one ib.sleep
    instead - a second, measured, not the millisecond the answer took to
    arrive, because the error only reaches this process while that sleep runs
    the event loop - and lets the message quote the broker rather than guess.
    """
    targets = {getattr(t.order, "orderId", None) for t in trades}
    refusals = []

    def note(reqId, errorCode, errorString, contract=None):
        if errorCode in CANCEL_REFUSED_CODES and reqId in targets:
            refusals.append(f"{errorCode} {errorString}")

    ib.errorEvent += note
    try:
        for trade in trades:
            ib.cancelOrder(trade.order)
        deadline = clock() + timeout
        while clock() < deadline:
            stuck, _ = split_working_orders(trades, allocation)
            if not stuck or refusals:
                break
            ib.sleep(1)
    finally:
        ib.errorEvent -= note
    stuck, _ = split_working_orders(trades, allocation)
    return stuck, refusals


def parse_sessions(hours_spec, tz_name):
    """IBKR's liquid-hours string, as a list of (start, end) datetimes.

    The string looks like "20260911:0930-20260911:1600;20260914:CLOSED", in
    the time zone the contract reports. Anything unparseable is skipped
    rather than guessed at, and an empty result means "IBKR did not say".
    """
    try:
        zone = ZoneInfo(tz_name)
    except Exception:
        # IBKR has reported several spellings over the years; the funds this
        # trades are all US-listed, so falling back to Eastern is right for
        # them and the caller treats an empty result as unknown anyway.
        zone = EASTERN
    sessions = []
    for piece in (hours_spec or "").split(";"):
        piece = piece.strip()
        if not piece or piece.upper().endswith("CLOSED"):
            continue
        start_text, _, end_text = piece.partition("-")
        try:
            start = datetime.strptime(start_text.strip(), "%Y%m%d:%H%M")
            end = datetime.strptime(end_text.strip(), "%Y%m%d:%H%M")
        except ValueError:
            continue
        sessions.append((start.replace(tzinfo=zone), end.replace(tzinfo=zone)))
    return sessions


def market_is_open(hours_spec, tz_name, now):
    """True, False, or None when IBKR did not say.

    A market order is not executed outside regular trading hours; it is held
    until the next open. That matters here because the runner refuses to send
    buys until the sells are done, so an order sent after the close leaves the
    book half rebalanced until somebody runs it again. Asking IBKR rather
    than reading the clock is what makes holidays and early closes right.
    """
    sessions = parse_sessions(hours_spec, tz_name)
    if not sessions:
        return None
    return any(start <= now < end for start, end in sessions)


def closed_symbols(ib, contracts, symbols, now):
    """Ask IBKR which of these are shut right now, and which it would not say.

    Reading the clock instead would get holidays and early closes wrong; the
    answer comes from the contract's own liquid hours.
    """
    closed, unknown = [], []
    for symbol in sorted(symbols):
        try:
            details = ib.reqContractDetails(contracts[symbol])
        except Exception as exc:
            unknown.append(f"{symbol}（问不到：{str(exc)[:60]}）")
            continue
        if not details:
            unknown.append(f"{symbol}（IBKR 没返回合约细节）")
            continue
        state = market_is_open(getattr(details[0], "liquidHours", ""),
                               getattr(details[0], "timeZoneId", ""), now)
        if state is None:
            unknown.append(f"{symbol}（交易时段字符串读不出来）")
        elif not state:
            closed.append(symbol)
    return closed, unknown


def describe_trades(trades):
    """One line naming every order, for a message that has to be acted on."""
    return "；".join(
        f"{getattr(t.contract, 'symbol', '?')}"
        f"[{getattr(t.contract, 'secType', '?')}] "
        f"{getattr(t.order, 'action', '?')} "
        f"{float(getattr(t.order, 'totalQuantity', 0.0)):,.0f} 股"
        f"（{getattr(t.orderStatus, 'status', '?')}，已成交 "
        f"{float(getattr(t.orderStatus, 'filled', 0.0)):,.0f}）"
        for t in trades)


def step_scale(raw, previous, band):
    """Adopt a new book scale only once it has moved more than `band`.

    This is the runner's half of portfolio_build.step_overlay (line 162),
    which is where the measured results come from. That function compares the
    band against the *book scale*. The runner used to compare its band
    against one symbol's weight instead, and in `thirds` a symbol carries a
    third of the book, so a 3% band on a weight is a 9% band on the scale -
    three times slower to react than anything that was measured. Slower is
    the direction that turns the weekly overlay back into the monthly one,
    which step_overlay's own docstring puts at 0.10 of Sharpe.

    Returns the scale to use and whether it moved. `previous` is None on the
    first run, when there is nothing to hold on to.
    """
    if previous is None:
        return raw, True
    if abs(raw - previous) > band:
        return raw, True
    return previous, False


def sessions_since(index, day):
    """Completed sessions in `index` after `day`, or None if `day` is None.

    Counts bars rather than calendar days, the same way
    portfolio_build.static_weights picks its rebalance dates off
    closes.index (line 126). Holidays and weekends are therefore not
    counted, which is what makes 21 mean a month of trading.
    """
    if day is None:
        return None
    return sum(1 for stamp in index if stamp.date() > day)


def plan_targets(allocation, held, price, capital, scale, previous_scale,
                 rebalancing, min_trade):
    """Target share counts, following the backtest's two separate clocks.

    portfolio_build.py runs two of them. static_weights (line 118) restores
    the allocation to its target weights every --rebalance sessions and lets
    it drift with price in between, with no band anywhere. step_overlay
    (line 162) resizes the whole book, and only when the scale has moved more
    than its band. The weights that were measured are the first multiplied by
    the second (line 250).

    So on a rebalance session the target is the allocation share times the
    scale. In between, the drift is left alone - the account's own positions
    are the drift - and only a change in the scale moves anything, applied to
    every holding in proportion.

    `min_trade` has no counterpart in the backtest. It drops a difference
    worth less than that fraction of capital, because an order for one or two
    shares pays a per-order cost that the backtest's flat 5 bps a side does
    not model. Pass 0 to follow the backtest exactly.
    """
    plans = []
    for symbol, share in allocation.items():
        have = held[symbol]
        if rebalancing:
            wanted = int(capital * share * scale / price[symbol])
        elif previous_scale and previous_scale > 0:
            # Same proportion of whatever is held, which is what multiplying
            # a drifted weight vector by a new scale comes out as.
            wanted = int(have * scale / previous_scale)
        else:
            wanted = have
        delta = wanted - have
        gap = abs(delta) * price[symbol] / capital
        skipped = bool(delta) and gap < min_trade
        plans.append({"symbol": symbol, "price": price[symbol],
                      "target_weight": wanted * price[symbol] / capital,
                      "current_weight": have * price[symbol] / capital,
                      "wanted": wanted, "held": have,
                      "delta": 0 if skipped else delta,
                      "gap": gap, "skipped": skipped})
    return plans


def load_overlay_state(path, allocation_name):
    """The last adopted scale and rebalance date, or None on a fresh start.

    A missing, unreadable or unparseable file is a fresh start rather than an
    error: the run can always rebuild both from the account itself, and
    refusing to trade because a bookkeeping file is corrupt would be worse
    than rebalancing one session early.
    """
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(state, dict) or state.get("allocation") != allocation_name:
        # A scale measured on one allocation says nothing about another, and
        # a rebalance date belongs to the weights that were restored.
        return None
    return state


def save_overlay_state(path, allocation_name, scale, last_rebalance):
    """Record what this run actually adopted, for the next one to read."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"allocation": allocation_name, "scale": scale,
         "last_rebalance": last_rebalance,
         "written": datetime.now(EASTERN).isoformat(timespec="seconds")},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def book_scale(portfolio_returns, mode, fixed_target):
    """How much of the allocation to hold, from the book's own volatility.

    The series passed in must end at the last completed session; use
    `drop_incomplete_bar` first. There is deliberately no shift here. The
    backtest sizes day D from everything through D-1 and then collects day
    D's return; running late in day D off data through D-1 is that same
    convention, and so is running after the close of D and holding from
    D+1. Shifting as well would size off a session older than anything that
    was tested, which over 2006-2026 is worth about 0.10 of Sharpe.

    "expanding" uses the book's own average volatility so far and needs no
    constant; "fixed" aims at a number, which only makes sense for an
    all-equity book.
    """
    trailing = (portfolio_returns.rolling(VOL_WINDOW).std(ddof=1)
                * TRADING_DAYS ** 0.5)
    if mode == "fixed":
        target = fixed_target
    else:
        target = float((portfolio_returns.expanding(MIN_HISTORY).std(ddof=1)
                        * TRADING_DAYS ** 0.5).iloc[-1])
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
                        help="整体仓位偏离超过这个比例才调整。这是对整个组合"
                             "的仓位倍数说的，不是对单个标的的权重说的，"
                             "和回测 portfolio_build.py --overlay-band 同义。")
    parser.add_argument("--rebalance", type=int, default=21,
                        help="隔多少个交易日把配置恢复到目标权重一次。"
                             "和回测 portfolio_build.py --rebalance 同义，"
                             "21 是一个月。中间的价格漂移不管。")
    parser.add_argument("--min-trade", type=float, default=0.005,
                        help="差额折算成本金的比例小于这个值就不下单。"
                             "回测里没有这一条，它挡掉的是一两股的零头单，"
                             "这种单在实盘要付每笔固定费用，而回测只按 5bps "
                             "比例算。填 0 就完全照回测来。")
    parser.add_argument("--state-file", default="logs/overlay_state.json",
                        help="记上次采用的整体仓位和上次再平衡日期。"
                             "删掉它下次运行就当首次运行：立刻再平衡。")
    parser.add_argument("--max-stale-days", type=int, default=4,
                        help="最后一根K线超过这么多天就拒绝交易。")
    parser.add_argument("--duration", default="20 Y",
                        help="取多长的历史。扩展窗口目标需要足够长的历史。")
    parser.add_argument("--prices", choices=("auto", "live", "cache"),
                        default="auto",
                        help="auto 先向 IBKR 要，要不到就用本地缓存；"
                             "live 只用 IBKR；cache 只用缓存。"
                             "缓存太旧仍然会被 --max-stale-days 拦下。")
    parser.add_argument("--cache-dir", default="data_cache_adj",
                        help="回退时读哪个缓存目录。必须是 ADJUSTED_LAST 取的。")
    parser.add_argument("--capital", type=float, default=None,
                        help="用多少钱做这个策略。不给就用账户全部净值，"
                             "但那样要求账户里没有配置之外的持仓。")
    parser.add_argument("--allow-foreign", action="store_true",
                        help="账户里有配置之外的持仓时仍按全部净值下单。"
                             "会造成融资杠杆，只在明知后果时使用。")
    parser.add_argument("--cancel-open", action="store_true",
                        help="下单前撤掉配置标的上的未成交挂单，别的标的一律不碰。"
                             "不加这个参数，配置标的上有挂单就中止。")
    parser.add_argument("--fill-timeout", type=float, default=60.0,
                        help="每笔单等待成交的秒数。卖单没成交就不会下买单。")
    parser.add_argument("--allow-closed", action="store_true",
                        help="常规交易时段之外也照发市价单。这些单会挂到下个"
                             "交易日开盘，本次的买单会被全部跳过。")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--execute", action="store_true",
                        help="真的下单。不加这个参数什么都不会送出去。")
    args = parser.parse_args()

    import pandas as pd

    from ibkr_ml.cache import load_cached_frame, require_adjusted
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
        # connect_ib steps the client id on a failed attempt, and which id a
        # run ends up with decides whose orders it is allowed to cancel.
        print(f"已连接 IBKR，client id {getattr(ib.client, 'clientId', '?')}")
        contracts, closes, sources = {}, {}, {}
        for symbol in allocation:
            contract = components.Stock(symbol, "SMART", "USD")
            ib.qualifyContracts(contract)
            contracts[symbol] = contract
            frame, source = None, ""
            if args.prices in ("auto", "live"):
                try:
                    frame = fetch_historical_frame(
                        ib=ib, symbol=symbol, duration=args.duration,
                        bar_size="1 day", use_rth=True,
                        max_duration_per_request=args.duration,
                        what_to_show="ADJUSTED_LAST")
                    source = "IBKR"
                except Exception as exc:
                    if args.prices == "live":
                        raise
                    # The trading session survives when the market-data
                    # subscription is handed to another login, so this fails
                    # while positions and orders still work. The staleness
                    # check below is what decides whether the cache is
                    # usable, so falling back is safe to do quietly here but
                    # never quietly overall.
                    print(f"  {symbol}: 向 IBKR 取历史失败（{str(exc)[:90]}），"
                          f"改用缓存")
            if frame is None:
                frame, _ = load_cached_frame(Path(args.cache_dir), symbol,
                                             args.duration, "1 day", True)
                source = f"缓存 {args.cache_dir}"
                if frame is None:
                    raise SystemExit(
                        f"IBKR 取不到 {symbol} 的历史，{args.cache_dir} 里也没有。"
                        f"先跑 fetch_assets.py --cache-dir {args.cache_dir} "
                        f"--what-to-show ADJUSTED_LAST {symbol}")
            sources[symbol] = source
            stamps = to_eastern_naive(frame["timestamp"]).dt.normalize()
            closes[symbol] = pd.Series(frame["close"].to_numpy(float),
                                       index=stamps.to_numpy()).sort_index()
        if any(s != "IBKR" for s in sources.values()):
            require_adjusted(args.cache_dir, "回退到缓存下单")
            print(f"  价格来源: " + "，".join(f"{k} {v}" for k, v in sources.items()))

        prices = pd.DataFrame(closes).dropna()
        now = datetime.now(EASTERN)
        # The same frame does two jobs. Sizing an order wants the price that
        # is trading now, so it takes the last bar even when that bar belongs
        # to a session still running: a partial daily bar's close is the
        # latest trade, which is exactly what is wanted here. The volatility
        # estimate must not see that bar at all, so it is dropped below.
        #
        # That only holds when the bars came from IBKR moments ago. On the
        # cache fallback the last bar is some previous session's close, up to
        # --max-stale-days old, and sizing off it misses every move since -
        # a symbol up 5% since that close gets bought 5% short of its target.
        # dropna() also aligns the symbols, so one lagging cached symbol pulls
        # every price back to its own last session. An earlier version of this
        # comment called the price "the one trading now" with no qualifier,
        # which was only ever true on the live path.
        price_day = prices.index[-1]
        price_age = (now.date() - price_day.date()).days
        live_price = {s: float(prices[s].iloc[-1]) for s in allocation}
        if price_age > 0:
            origin = "，".join(f"{k} {v}" for k, v in sources.items())
            print(f"  注意：下单价用的是 {price_day.date()} 的收盘价，"
                  f"距今 {price_age} 天，不是现价。这之后的涨跌不会反映在"
                  f"股数里。价格来源：{origin}")
        prices, dropped = drop_incomplete_bar(prices, now)
        if dropped:
            print(f"  {now:%Y-%m-%d %H:%M} 美东，当日尚未收盘，"
                  f"这根未完成的K线不计入波动率估计")
        if len(prices) < MIN_HISTORY + VOL_WINDOW:
            raise SystemExit(f"只有 {len(prices)} 天的共同历史，不足以算长期目标波动")
        returns = prices.pct_change().dropna()
        portfolio = sum(returns[s] * w for s, w in allocation.items())

        last_day = prices.index[-1]
        age_days = (now.date() - last_day.date()).days
        if age_days > args.max_stale_days:
            raise SystemExit(f"最后一根K线是 {last_day.date()}，距今 {age_days} 天，"
                             f"超过 {args.max_stale_days} 天上限，已中止")

        if args.overlay == "none":
            raw_scale, trailing, target = 1.0, float("nan"), float("nan")
        else:
            raw_scale, trailing, target = book_scale(portfolio, args.overlay,
                                                     args.fixed_target)

        # The backtest runs two clocks and the runner has to run both of them
        # or it is measuring a different strategy: the scale only moves once
        # it has moved more than the band (portfolio_build.step_overlay, line
        # 162), and the weights are only restored every --rebalance sessions
        # (portfolio_build.static_weights, line 118). Both have to remember
        # the previous run, which is the whole reason the state file exists.
        state = load_overlay_state(args.state_file, args.allocation)
        previous_scale = (state or {}).get("scale")
        last_rebalance = None
        if state and state.get("last_rebalance"):
            try:
                last_rebalance = date.fromisoformat(state["last_rebalance"])
            except (TypeError, ValueError):
                last_rebalance = None
        scale, scale_moved = step_scale(raw_scale, previous_scale, args.band)
        elapsed = sessions_since(prices.index, last_rebalance)
        rebalancing = elapsed is None or elapsed >= args.rebalance

        # ib_insync asks TWS for open orders once, as it connects
        # (ib_insync/ib.py:1762 calls reqOpenOrders), and that request returns
        # only what the connected client id placed. connect_ib raises the
        # client id by one on each failed attempt (ibkr_ml/data.py:274), so a
        # single retry hides the order a previous run left working - and those
        # shares are not in the position count either, which is exactly how
        # the same order goes out twice. reqAllOpenOrders is account-wide and
        # does not depend on which client id this run ended up with.
        try:
            ib.reqAllOpenOrders()
        except Exception as exc:
            raise SystemExit(
                f"取不到账户的全部挂单（{str(exc)[:90]}）。挂单里的股数不在持仓里，"
                f"看不到它们就可能把同一笔单再下一遍，已中止。")
        mine, others = split_working_orders(ib.openTrades(), allocation)
        if others:
            print(f"  配置之外有 {len(others)} 笔未成交挂单，本次不动它们："
                  f"{describe_trades(others)}")
        if mine:
            if args.cancel_open:
                # Only these. Cancelling every order the account has would
                # reach into orders this project did not place.
                print(f"  撤掉 {len(mine)} 笔配置标的上的未成交挂单："
                      f"{describe_trades(mine)}")
                stuck, refusals = cancel_and_wait(ib, mine, allocation,
                                                  args.fill_timeout)
                if stuck:
                    said = ("\nIBKR 的原话：" + "；".join(refusals)
                            if refusals else "")
                    raise SystemExit(
                        f"{len(stuck)} 笔挂单撤不掉：{describe_trades(stuck)}。"
                        f"{said}\n"
                        f"orderId 是按 client id 分配的，所以从别的 client id "
                        f"发的撤单请求，IBKR 找不到对应的单。用 --client-id "
                        f"指定当初下单用的编号重跑，或在 TWS 里手工撤。")
            else:
                raise SystemExit(
                    f"配置标的上有 {len(mine)} 笔未成交挂单："
                    f"{describe_trades(mine)}。\n"
                    f"这些单的股数还没进持仓，现在按持仓算差额会把同一笔"
                    f"再下一遍。先等它们成交，或加 --cancel-open 撤掉。")

        equity = 0.0
        for row in ib.accountSummary():
            if row.tag == "NetLiquidation":
                equity = float(row.value)
        if equity <= 0:
            raise SystemExit("取不到账户净值，无法计算仓位")

        held = {symbol: 0 for symbol in allocation}
        foreign = []
        for item in ib.portfolio():
            symbol = getattr(item.contract, "symbol", "")
            sec_type = getattr(item.contract, "secType", "")
            if sec_type == "STK" and symbol in held:
                held[symbol] = int(item.position)
            elif item.position:
                # Bonds, options and futures carry risk too; counting only
                # stocks would let an option book through the check below.
                label = symbol if sec_type == "STK" else f"{symbol}[{sec_type}]"
                foreign.append((label, int(item.position),
                                float(item.marketValue)))
        foreign.sort(key=lambda row: -abs(row[2]))
        foreign_value = sum(abs(value) for _, _, value in foreign)

        # Sizing off net liquidation while ignoring what the account already
        # holds is how a 1.05M account ends up with 1.78M of stock on margin.
        # Either the whole account is the book, or the caller states the
        # slice that is.
        # `if args.capital` would read 0 as "not given" and silently fall
        # back to the whole account while also skipping the check below.
        capital = equity if args.capital is None else args.capital
        if capital <= 0:
            raise SystemExit(f"--capital 是 {args.capital:,.2f}，必须大于 0。"
                             f"负数或零会算出做空或全空的目标仓位。")
        if capital > equity:
            raise SystemExit(f"--capital 是 {capital:,.0f} 美元，超过账户净值 "
                             f"{equity:,.0f} 美元，那要融资，本策略没测过杠杆。")
        if foreign and capital > max(equity - foreign_value, 0.0):
            print(f"  注意：配置外持仓 {foreign_value:,.0f} 美元加上本策略 "
                  f"{capital:,.0f} 美元，合计超过净值 {equity:,.0f} 美元，需要融资")
        if foreign and args.capital is None and not args.allow_foreign:
            listing = "，".join(f"{s} {q:,} 股 {v:,.0f} 美元"
                               for s, q, v in foreign[:8])
            raise SystemExit(
                f"账户里有 {len(foreign)} 个配置之外的持仓，合计 "
                f"{foreign_value:,.0f} 美元：{listing}。\n"
                f"按全部净值 {equity:,.0f} 下单会在这些持仓之上再建仓，"
                f"总持仓约 {equity + foreign_value:,.0f} 美元，"
                f"相当于 {(equity + foreign_value) / equity:.2f} 倍杠杆。\n"
                f"要么先清掉这些持仓，要么用 --capital 指定这个策略用多少钱"
                f"（可用现金约 {max(equity - foreign_value, 0):,.0f} 美元）。")

        print(f"配置 {args.allocation}   最后一根K线 {last_day.date()}"
              f"（{age_days} 天前）   账户净值 {equity:,.0f} 美元")
        if foreign:
            print(f"  配置之外还持有 {len(foreign)} 个标的，合计 "
                  f"{foreign_value:,.0f} 美元，本次不动它们：")
            for symbol, quantity, value in foreign:
                print(f"    {symbol:<6} {quantity:>10,} 股  {value:>12,.0f} 美元")
        if capital != equity:
            print(f"  本策略只用 {capital:,.0f} 美元，占净值 "
                  f"{capital / equity:.1%}")
        if args.overlay != "none":
            print(f"  组合近 {VOL_WINDOW} 天年化波动 {trailing:.2%}   "
                  f"长期目标 {target:.2%}   整体仓位 {scale:.1%}")
            if not scale_moved:
                print(f"    本次算出 {raw_scale:.1%}，与上次采用的 "
                      f"{previous_scale:.1%} 相差 "
                      f"{abs(raw_scale - previous_scale):.1%} ≤ "
                      f"{args.band:.0%}，沿用上次的")
            elif previous_scale is not None:
                print(f"    上次采用 {previous_scale:.1%}，相差 "
                      f"{abs(raw_scale - previous_scale):.1%} > "
                      f"{args.band:.0%}，改用本次算出的")
        if rebalancing:
            since = ("没有上次记录，按首次运行处理" if elapsed is None
                     else f"距上次再平衡 {elapsed} 个交易日 ≥ {args.rebalance}")
            print(f"  本次再平衡到目标权重（{since}）")
        else:
            print(f"  本次不再平衡（距上次再平衡 {elapsed} 个交易日 < "
                  f"{args.rebalance}），价格漂移不管，只跟随整体仓位的变化")
        print(f"  {'标的':<6} {'价格':>9} {'目标权重':>9} {'当前权重':>9} "
              f"{'应持':>8} {'现持':>8} {'差':>8}  动作")

        plans = plan_targets(allocation, held, live_price, capital, scale,
                             previous_scale, rebalancing, args.min_trade)
        orders, decisions = [], []
        for plan in plans:
            if plan["skipped"]:
                action = (f"不动（差 {plan['gap']:.2%} < "
                          f"{args.min_trade:.1%} 本金）")
            elif plan["delta"] == 0:
                action = "不动（股数差 0）"
            else:
                action = (f"{'买入' if plan['delta'] > 0 else '卖出'} "
                          f"{abs(plan['delta']):,}")
                orders.append((plan["symbol"], plan["delta"]))
            print(f"  {plan['symbol']:<6} {plan['price']:>9.2f} "
                  f"{plan['target_weight']:>9.1%} {plan['current_weight']:>9.1%} "
                  f"{plan['wanted']:>8,} {plan['held']:>8,} "
                  f"{plan['wanted'] - plan['held']:>+8,}  {action}")
            decisions.append(plan)

        payload = {"allocation": allocation, "bar_date": str(last_day.date()),
                   "equity": equity, "capital": capital, "scale": scale,
                   "price_day": str(price_day.date()), "price_age": price_age,
                   "price_sources": dict(sources),
                   "raw_scale": raw_scale, "previous_scale": previous_scale,
                   "scale_moved": scale_moved, "rebalancing": rebalancing,
                   "sessions_since_rebalance": elapsed,
                   "trailing_vol": trailing, "target_vol": target,
                   "foreign_positions": [{"symbol": s, "quantity": q,
                                          "market_value": v}
                                         for s, q, v in foreign],
                   "decisions": decisions, "executed": False}

        # A market order sent outside regular trading hours is not executed,
        # it is parked until the next open. Nothing below waits that long, so
        # every order would read as unfilled, the buys would be skipped on
        # purpose, and the book would sit half rebalanced for a week. The cron
        # in run_weekly.sh runs inside the session for this reason; this
        # refuses the case the cron cannot prevent, such as a manual run in
        # the evening. A dry run reports it rather than hiding it, since a
        # dry run that says "would place 3 orders" at midnight is a lie.
        checked_at = datetime.now(EASTERN)
        closed, unknown = ([], [])
        if orders:
            closed, unknown = closed_symbols(ib, contracts,
                                             {s for s, _ in orders}, checked_at)
            payload["market_closed"] = closed
            if unknown:
                print(f"  问不到交易时段，按开市处理：{'，'.join(unknown)}")

        if not orders:
            print("  所有标的都在不动区间内，不交易")
            payload["action"] = "hold"
        elif not args.execute:
            if closed:
                print(f"  {checked_at:%Y-%m-%d %H:%M} 美东，"
                      f"{'、'.join(closed)} 不在常规交易时段内，"
                      f"加 --execute 真跑会被拒绝")
            print(f"  演练模式：本应下 {len(orders)} 笔单。加 --execute 才会真的下单")
            payload["action"] = "dry_run"
        else:
            if closed and not args.allow_closed:
                payload["action"] = "market_closed"
                log_event(Path(args.log_dir), payload)
                raise SystemExit(
                    f"{checked_at:%Y-%m-%d %H:%M} 美东，"
                    f"{'、'.join(closed)} 不在常规交易时段内，没有下单。\n"
                    f"市价单在盘外不会成交，会挂到下个交易日开盘；本程序不等"
                    f"那么久，会把卖单判成未成交，于是买单全部跳过，账户停在"
                    f"半调仓状态。\n"
                    f"在交易时段内重跑，或加 --allow-closed 明知后果地照发。")
            if closed:
                print(f"  注意：{'、'.join(closed)} 不在交易时段内，"
                      f"--allow-closed 已指定，照发")

            # Sells first, and confirmed filled before the buys go out: they
            # free the cash the buys need. Sleeping two seconds and assuming
            # it worked is how a rejected sell turns into a margin loan.
            results = []
            for group, side_label in ((sorted(o for o in orders if o[1] < 0),
                                       "卖出"),
                                      (sorted(o for o in orders if o[1] > 0),
                                       "买入")):
                if not group:
                    continue
                if side_label == "买入" and any(not r["complete"]
                                              for r in results):
                    print("  有卖单没有完全成交，不下买单，"
                          "以免用融资买入。请稍后重跑")
                    payload["action"] = "sells_incomplete"
                    break
                live = []
                for symbol, delta in group:
                    order = components.MarketOrder(
                        "BUY" if delta > 0 else "SELL", abs(delta))
                    live.append((symbol, delta,
                                 ib.placeOrder(contracts[symbol], order)))
                deadline = time.monotonic() + args.fill_timeout
                while time.monotonic() < deadline:
                    if all(trade.isDone() for _, _, trade in live):
                        break
                    ib.sleep(1)
                for symbol, delta, trade in live:
                    status = trade.orderStatus.status
                    filled = float(trade.orderStatus.filled)
                    complete = status == "Filled" and filled == abs(delta)
                    note = "" if complete else f"  ← 未完全成交"
                    print(f"  {side_label} {symbol}：报 {abs(delta):,} 股，"
                          f"成交 {filled:,.0f} 股，状态 {status}{note}")
                    results.append({"symbol": symbol, "quantity": abs(delta),
                                    "side": "BUY" if delta > 0 else "SELL",
                                    "status": status, "filled": filled,
                                    "complete": complete})
            payload.update({"action": payload.get("action", "orders_sent"),
                            "executed": True, "orders": results})
            if any(not r["complete"] for r in results):
                print("  有单未完全成交，下次运行前先确认账户状态")

        # Only a run that actually reached the market may move these
        # markers forward. A dry run must not, or the next real run would
        # think a rebalance it never sent had already happened; and a partial
        # fill must not either, because the positions are not where the
        # markers would claim they are. Not writing is always safe: the next
        # run rebalances one session early, which costs one round of
        # commission and nothing else.
        settled = all(row["complete"] for row in payload.get("orders", []))
        if args.execute and payload["action"] in ("orders_sent", "hold") and settled:
            save_overlay_state(
                args.state_file, args.allocation, scale,
                str(last_day.date()) if rebalancing
                else (state or {}).get("last_rebalance"))

        log_event(Path(args.log_dir), payload)
    finally:
        try:
            ib.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
