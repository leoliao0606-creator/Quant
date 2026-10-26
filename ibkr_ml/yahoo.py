"""Daily bars from Yahoo Finance, for running without a broker connection.

The account went to IBKR Lite some time after 2026-09-15, and Lite has no
API, so nothing in this project can ask IBKR for a price any more. Yahoo's
chart endpoint needs no account. It is not a published API: it can change
shape or start refusing without notice, which is why every response is kept
on disk as it arrived and why the parser checks the shape instead of
trusting it.

Measured on 2026-10-25 against data_cache_adj (IBKR ADJUSTED_LAST): over
2024-01..2026-09 the annualised total return of SPY, AGG and TIP agrees to
0.001 of a point and the 99th percentile daily gap is 0.03 bp. The two
disagree in 2006-2013, where AGG's worst day is 2007-10-17 - the first
session after IBKR's 33-session hole in AGG.

Yahoo reports three things a ledger has to keep apart. `close` is adjusted
for splits but not dividends; `adjclose` for both; and dividend amounts are
split-adjusted too. Checked on NVDA's 10:1 split of 2024-06-10: Yahoo gives
the 2024-06-07 close as 120.888 against the 1,208.88 that printed, and the
2024-03-05 dividend as 0.004 against the 0.04 that was paid. A split after
a session therefore rewrites that session's numbers, and a ledger holding
pre-split share counts would value them at a tenth of their worth.
`parse_chart` undoes it, so `close` here is the price that printed that day
and `dividend` the cash paid per share then held.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
CHART_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
             "?period1={start}&period2={end}&interval=1d&events=div%2Csplit")
QUOTE_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
             "?range=1d&interval=5m")
# Without a browser-like agent the endpoint answers 429 to most requests.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64)"
# Far enough back that a 20-year expanding window always has its full span.
DEFAULT_START = datetime(2005, 1, 1, tzinfo=timezone.utc)


def chart_path(cache_dir, symbol) -> Path:
    return Path(cache_dir) / f"{symbol}.json"


def fetch_chart(symbol, cache_dir, start=DEFAULT_START, timeout=30.0):
    """Download one symbol's daily history and keep the response on disk.

    The file is replaced only after the new response has parsed, so a
    truncated download or an error page can never overwrite the last good
    copy that load_chart falls back to.
    """
    end = datetime.now(timezone.utc)
    url = CHART_URL.format(symbol=symbol, start=int(start.timestamp()),
                           end=int(end.timestamp()))
    body = _get(url, timeout)
    payload = json.loads(body)
    parse_chart(payload)
    path = chart_path(cache_dir, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_bytes(body)
    partial.replace(path)
    return payload


def load_chart(symbol, cache_dir):
    """The last response fetch_chart kept, or None."""
    try:
        return json.loads(chart_path(cache_dir, symbol).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _get(url, timeout):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def fetch_quote(symbol, timeout=15.0):
    """The latest regular-session price, for showing a value between closes."""
    return parse_quote(json.loads(_get(QUOTE_URL.format(symbol=symbol), timeout)))


def parse_quote(payload):
    """Price, its time, the previous close and today's regular session.

    Nothing here is booked. A quote is only ever an estimate of what the
    ledger will write at the close; the session bounds say whether the
    price is still moving.
    """
    chart = (payload or {}).get("chart") or {}
    results = chart.get("result") or []
    if chart.get("error") or not results:
        raise ValueError(f"Yahoo 没有返回报价：{chart.get('error')}")
    meta = results[0].get("meta") or {}
    try:
        price = float(meta["regularMarketPrice"])
        stamp = datetime.fromtimestamp(int(meta["regularMarketTime"]), EASTERN)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Yahoo 的报价格式变了，缺少 {exc}") from exc
    previous = meta.get("previousClose") or meta.get("chartPreviousClose")
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}

    def when(key):
        value = regular.get(key)
        return (datetime.fromtimestamp(int(value), EASTERN).isoformat(timespec="seconds")
                if value else None)

    return {"symbol": meta.get("symbol"), "price": price,
            "time": stamp.isoformat(timespec="seconds"),
            "previous_close": float(previous) if previous else None,
            "session_start": when("start"), "session_end": when("end")}


def _session_date(epoch):
    return datetime.fromtimestamp(int(epoch), EASTERN).date()


def parse_chart(payload):
    """One row per session: printed prices, adjusted close, cash, split.

    Columns:
      open, high, low, close  the prices that printed that session
      adjclose  Yahoo's dividend-and-split adjusted close, for returns
      dividend  cash per share held into that ex-date, as paid
      split     new shares per old share taking effect that session, else 1
    """
    import pandas as pd

    chart = (payload or {}).get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"Yahoo 返回错误：{chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo 的返回里没有 result")
    result = results[0]
    try:
        stamps = result["timestamp"]
        quote = result["indicators"]["quote"][0]
        closes = quote["close"]
        adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Yahoo 的返回格式变了，缺少 {exc}") from exc
    if not (len(stamps) == len(closes) == len(adjusted)):
        raise ValueError("Yahoo 返回的时间、收盘价、调整后收盘价长度不一致")

    days = pd.DatetimeIndex([pd.Timestamp(_session_date(t)) for t in stamps])
    columns = {}
    for name in ("open", "high", "low"):
        values = quote.get(name)
        # Optional: the ledger only needs the close, so a missing or short
        # column becomes gaps rather than a refusal.
        if not isinstance(values, list) or len(values) != len(stamps):
            values = [None] * len(stamps)
        columns[name] = pd.Series(values, index=days, dtype=float)
    columns["close"] = pd.Series(closes, index=days, dtype=float)
    columns["adjclose"] = pd.Series(adjusted, index=days, dtype=float)
    frame = pd.DataFrame(columns)
    # A session the endpoint has not filled in yet comes back as null.
    frame = (frame[~frame.index.duplicated(keep="last")]
             .dropna(subset=["close", "adjclose"]).sort_index())
    frame["dividend"] = 0.0
    frame["split"] = 1.0

    events = result.get("events") or {}
    for item in (events.get("splits") or {}).values():
        ratio = float(item["numerator"]) / float(item["denominator"])
        day = _next_session(frame.index, _session_date(item["date"]))
        if day is not None and ratio > 0:
            frame.loc[day, "split"] *= ratio
    for item in (events.get("dividends") or {}).values():
        day = _next_session(frame.index, _session_date(item["date"]))
        if day is not None:
            frame.loc[day, "dividend"] += float(item["amount"])

    # Every split rescales what came before it, so a session's printed price
    # is Yahoo's figure times every split ratio that took effect after it.
    later_splits = frame["split"][::-1].cumprod()[::-1].shift(-1).fillna(1.0)
    for name in ("open", "high", "low", "close", "dividend"):
        frame[name] = frame[name] * later_splits
    return frame


def _next_session(index, day):
    """The first session on or after `day`; events are stamped at the open."""
    import pandas as pd

    position = index.searchsorted(pd.Timestamp(day))
    return index[position] if position < len(index) else None
