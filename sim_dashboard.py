#!/usr/bin/env python3
"""A local page for the broker-free ledger and the ChatGPT signals.

    python sim_dashboard.py            # then open http://127.0.0.1:8765

Three things on one page, each refreshed on its own clock:

- a ledger, re-read every five minutes, so a close booked by
  run_sim_daily.sh shows up without a reload. Every folder under
  logs/sim with a state.json is offered: the weekly allocation that
  portfolio_sim.py keeps, and the rule-driven ledgers signal_ledger.py
  keeps for each ChatGPT task;
- an estimate of that book at Yahoo's latest prices, every minute. It is
  never written anywhere: the ledger books at the close and only there;
- the ChatGPT reports pasted into the page, scored against SPY, with the
  random control signals_control.py writes after the close.

Standard library only, because nothing else is installed in ibkr_trade and
a page needs nothing more. It listens on 127.0.0.1 so nothing off this
machine can reach it. The one write - pasting a report - also refuses any
request that is not JSON from this page's own address: a web page in the
same browser can send a form to localhost, and without that check it could
plant reports in the record.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from portfolio_sim import CASH_PROXY, DECISION_WEEKDAY
from portfolio_trade import EASTERN

ROOT = Path(__file__).resolve().parent
STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
QUOTE_SECONDS = 60
SIGNAL_BARS_SECONDS = 30 * 60
MAX_BODY = 256 * 1024
RECENT_EVENTS = 40


def read_ledger(path):
    events = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return events
    for line in lines:
        if line.strip():
            events.append(json.loads(line))
    return events


def next_decision_day(today, last_processed):
    """The next Friday the ledger has not booked yet. Holidays not known."""
    day = today
    if last_processed and date.fromisoformat(last_processed) >= day:
        day = date.fromisoformat(last_processed) + timedelta(days=1)
    while day.weekday() != DECISION_WEEKDAY:
        day += timedelta(days=1)
    return day


def _index_series(bars, symbol, dates):
    """Total return of `symbol` on each of `dates`, 100 on the first."""
    import pandas as pd

    frame = bars.get(symbol)
    if frame is None:
        return [None] * len(dates)
    series = frame["adjclose"]
    stamps = [pd.Timestamp(d) for d in dates]
    if stamps[0] not in series.index:
        return [None] * len(dates)
    base = float(series[stamps[0]])
    return [round(float(series[s]) / base * 100.0, 4) if s in series.index else None
            for s in stamps]


def build_state_payload(state, ledger, bars, today):
    """Everything the page draws from the ledger, as plain JSON."""
    if state is None:
        return {"exists": False}
    capital = float(state["params"]["capital"])
    marks = [e for e in ledger if e.get("type") == "mark"]
    payload = {"exists": True, "name": state["name"], "kind": state.get("kind", "allocation"),
               "allocation": state.get("allocation", {}),
               "positions_count": len(state.get("positions") or {}),
               "params": state["params"], "created": state["created"],
               "first_session": state.get("first_session"),
               "last_processed": state.get("last_processed"),
               "scale": state.get("scale"), "last_rebalance": state.get("last_rebalance"),
               "capital": capital, "cash": state["cash"], "shares": state["shares"],
               "next_decision": str(next_decision_day(today, state.get("last_processed"))),
               "data_latest": (str(min(f.index[-1] for f in bars.values()).date())
                               if bars else None)}
    events = [e for e in ledger if e.get("type") not in ("mark", "start")]
    payload["events"] = list(reversed(events[-RECENT_EVENTS:]))
    if not marks:
        payload.update({"series": None, "equity": capital, "positions": None})
        return payload

    dates = [state["base_session"]] + [m["session"] for m in marks]
    equity = [capital] + [m["equity"] for m in marks]
    peak, drawdown = equity[0], 0.0
    for value in equity:
        peak = max(peak, value)
        drawdown = min(drawdown, value / peak - 1.0)
    spy = _index_series(bars, "SPY", dates)
    cash_proxy = _index_series(bars, CASH_PROXY, dates)
    last = marks[-1]
    payload.update({
        "series": {"dates": dates,
                   "sim": [round(v / capital * 100.0, 4) for v in equity],
                   "spy": spy, "sgov": cash_proxy},
        "equity": last["equity"], "booked_session": last["session"],
        "total_return": last["equity"] / capital - 1.0,
        "spy_return": spy[-1] / 100.0 - 1.0 if spy[-1] is not None else None,
        "sgov_return": (cash_proxy[-1] / 100.0 - 1.0
                        if cash_proxy[-1] is not None else None),
        "max_drawdown": drawdown, "positions": last["positions"],
        "booked_cash": last["cash"]})
    return payload


def build_live_payload(state, quotes, now):
    """The book at the latest quotes. An estimate, never booked."""
    if state is None:
        return {"exists": False}
    rows, invested, open_now, stamps = [], 0.0, False, []
    for symbol, shares in state["shares"].items():
        quote = quotes.get(symbol) or {}
        price = quote.get("price")
        previous = quote.get("previous_close")
        value = shares * price if price is not None else None
        invested += value or 0.0
        if quote.get("time"):
            stamps.append(quote["time"])
        start, end = quote.get("session_start"), quote.get("session_end")
        if start and end and (datetime.fromisoformat(start) <= now
                              < datetime.fromisoformat(end)):
            open_now = True
        rows.append({"symbol": symbol, "shares": shares, "price": price,
                     "previous_close": previous, "value": value,
                     "day_change": (price / previous - 1.0
                                    if price is not None and previous else None),
                     "error": quote.get("error")})
    return {"exists": True, "market_open": open_now, "rows": rows,
            "cash": state["cash"], "equity": state["cash"] + invested,
            "quote_time": max(stamps) if stamps else None,
            "fetched": now.isoformat(timespec="seconds")}


class Dashboard:
    """Paths, caches and the one lock that guards them."""

    def __init__(self, args):
        self.state_root = Path(args.state_dir)
        self.default_name = args.name
        self.cache_dir = Path(args.cache_dir)
        self.signals_dir = Path(args.signals_dir)
        self.signal_cache = self.cache_dir / "signals"
        self.lock = threading.Lock()
        self.quotes = {}
        self.signal_rows, self.signals_key = None, None

    def now(self):
        return datetime.now(EASTERN)

    def ledgers(self):
        """Every ledger folder there is, the allocation first."""
        found = []
        for folder in sorted(self.state_root.glob("*/state.json")):
            try:
                state = json.loads(folder.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            found.append({"name": folder.parent.name,
                          "kind": state.get("kind", "allocation"),
                          "task": state.get("params", {}).get("task"),
                          "created": state.get("created"),
                          "last_processed": state.get("last_processed")})
        found.sort(key=lambda row: (row["kind"] != "allocation", row["name"]))
        return {"default": self.default_name, "ledgers": found}

    def load_state(self, name):
        # Only a name that is an existing folder: the name comes from the
        # query string and must not reach the filesystem as a path.
        names = {row["name"] for row in self.ledgers()["ledgers"]}
        if (name or self.default_name) not in names:
            return None
        path = self.state_root / (name or self.default_name) / "state.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def state(self, name=None):
        from ibkr_ml.yahoo import load_chart, parse_chart

        state = self.load_state(name)
        bars = {}
        if state is not None:
            for symbol in ("SPY", CASH_PROXY):
                payload = (load_chart(symbol, self.cache_dir)
                           or load_chart(symbol, self.signal_cache))
                if payload is not None:
                    bars[symbol] = parse_chart(payload)
        ledger = read_ledger(self.state_root / (name or self.default_name) / "ledger.jsonl") \
            if state is not None else []
        return build_state_payload(state, ledger, bars, self.now().date())

    def live(self, name=None):
        """Quotes for what the ledger holds, each fetched at most once a minute."""
        from ibkr_ml.yahoo import fetch_quote

        state = self.load_state(name)
        if state is None:
            return {"exists": False}
        quotes = {}
        with self.lock:
            for symbol in state["shares"]:
                fetched_at, quote = self.quotes.get(symbol, (0.0, None))
                if quote is None or time.monotonic() - fetched_at > QUOTE_SECONDS:
                    try:
                        quote = fetch_quote(symbol)
                    except Exception as exc:
                        quote = {"error": str(exc)[:120]}
                    self.quotes[symbol] = (time.monotonic(), quote)
                quotes[symbol] = quote
        return build_live_payload(state, quotes, self.now())

    def bars_for(self, ticker, start):
        """Bars for one signal ticker, fetched at most every half hour."""
        from ibkr_ml.yahoo import cached_bars

        return cached_bars(ticker, self.signal_cache, start, SIGNAL_BARS_SECONDS)

    def signals(self):
        """Scored rows and their summary, kept apart per task."""
        from ibkr_ml.signals import TASKS, load_inbox, score_inbox, summarize

        records = load_inbox(self.signals_dir)
        inbox = self.signals_dir / "inbox.jsonl"
        key = (inbox.stat().st_mtime if inbox.exists() else None,
               int(time.time() // SIGNAL_BARS_SECONDS))
        with self.lock:
            if self.signal_rows is not None and self.signals_key == key:
                return self.signal_rows
        # Two years back at least, so a newly pasted older report rarely
        # moves the start past what is already cached.
        earliest = min((datetime.fromisoformat(r["received"]) for r in records),
                       default=self.now())
        start = min(earliest - timedelta(days=10), self.now() - timedelta(days=730))
        cache = {}

        def bars_for(ticker):
            if ticker not in cache:
                cache[ticker] = self.bars_for(ticker, start)
            return cache[ticker]

        rows = score_inbox(records, bars_for) if records else []
        try:
            control = json.loads((self.signals_dir / "control.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            control = {}
        payload = {"tasks": {}, "control": {k: control.get(k)
                                            for k in ("computed", "universe", "draws")}}
        for task in TASKS:
            mine = [r for r in rows if r["task"] == task]
            payload["tasks"][task] = {
                "reports": sum(1 for r in records if r["task"] == task),
                "rows": mine, "summary": summarize(mine),
                "control": (control.get("tasks") or {}).get(task, {})}
        with self.lock:
            self.signal_rows, self.signals_key = payload, key
        return payload

    def paste(self, text, task):
        from ibkr_ml.signals import ingest

        with self.lock:
            result = ingest(text, self.signals_dir, self.now(), expected_task=task)
            self.signal_rows = None
        return {"added": len(result["added"]), "duplicates": result["duplicates"],
                "items": sum(len(r["items"]) for r in result["added"]),
                "errors": result["errors"]}


def make_handler(dashboard, allowed_hosts):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def host_ok(self):
            # A name that resolves to 127.0.0.1 can still carry a foreign Host
            # header; refusing it keeps another site's page from reading this.
            return self.headers.get("Host", "") in allowed_hosts

        def do_GET(self):
            if not self.host_ok():
                return self.send_json({"error": "host not allowed"}, 403)
            from urllib.parse import parse_qs, urlsplit

            parts = urlsplit(self.path)
            path = parts.path
            name = (parse_qs(parts.query).get("name") or [None])[0]
            routes = {"/api/state": lambda: dashboard.state(name),
                      "/api/live": lambda: dashboard.live(name),
                      "/api/ledgers": dashboard.ledgers,
                      "/api/signals": dashboard.signals}
            if path in routes:
                try:
                    return self.send_json(routes[path]())
                except Exception as exc:
                    return self.send_json({"error": str(exc)[:300]}, 500)
            if path in STATIC:
                name, kind = STATIC[path]
                body = (ROOT / "dashboard" / name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                return self.wfile.write(body)
            self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/signals" or not self.host_ok():
                return self.send_json({"error": "not allowed"}, 403)
            origin = self.headers.get("Origin")
            if origin is not None and origin != f"http://{self.headers.get('Host')}":
                return self.send_json({"error": "origin not allowed"}, 403)
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self.send_json({"error": "需要 application/json"}, 415)
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY:
                return self.send_json({"error": f"内容为空或超过 {MAX_BODY} 字节"}, 413)
            from ibkr_ml.signals import TASKS

            try:
                body = json.loads(self.rfile.read(length))
                text, task = body["text"], body["task"]
                if not isinstance(text, str):
                    raise TypeError("text 应为字符串")
                if task not in TASKS:
                    raise ValueError(f"task 应为 {'/'.join(TASKS)} 之一")
            except Exception as exc:
                return self.send_json({"error": f"请求格式不对：{exc}"}, 400)
            self.send_json(dashboard.paste(text, task))

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="本地网页：模拟账本和 ChatGPT 建议")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址。默认只有本机能访问")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--name", default="thirds", help="要显示的模拟，同 portfolio_sim.py --name")
    parser.add_argument("--state-dir", default="logs/sim")
    parser.add_argument("--cache-dir", default="data_cache_yahoo")
    parser.add_argument("--signals-dir", default="logs/signals")
    args = parser.parse_args()

    dashboard = Dashboard(args)
    allowed = {f"{args.host}:{args.port}", f"localhost:{args.port}",
               f"127.0.0.1:{args.port}"}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(dashboard, allowed))
    print(f"打开 http://{'127.0.0.1' if args.host == '0.0.0.0' else args.host}:{args.port}  "
          f"（Ctrl-C 停止）")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
