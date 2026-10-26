"""Tests for what the local page is told, and who may write to it."""

import argparse
import http.client
import json
import sys
import threading
from datetime import date, datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim_dashboard import (Dashboard, build_live_payload,  # noqa: E402
                           build_state_payload, make_handler, next_decision_day)

EASTERN = ZoneInfo("America/New_York")
STATE = {"name": "thirds", "allocation": {"SPY": 0.5, "AGG": 0.5},
         "params": {"capital": 1000.0}, "created": "2026-10-25T20:00:00-04:00",
         "cash": 100.0, "shares": {"SPY": 1, "AGG": 2}, "scale": 1.0,
         "last_rebalance": "2026-10-29", "first_session": "2026-10-26",
         "base_session": "2026-10-23", "last_processed": "2026-10-27"}


def mark(session, equity):
    return {"type": "mark", "session": session, "equity": equity, "cash": 100.0,
            "positions": {"SPY": {"shares": 1, "close": 500.0, "value": 500.0}}}


def adj(values):
    days = ["2026-10-23", "2026-10-26", "2026-10-27"]
    return pd.DataFrame({"adjclose": values}, index=pd.DatetimeIndex(days))


class TestNextDecision:
    def test_a_friday_not_yet_booked_is_today(self):
        assert next_decision_day(date(2026, 10, 30), "2026-10-29") == date(2026, 10, 30)

    def test_a_friday_already_booked_moves_to_the_next(self):
        assert next_decision_day(date(2026, 10, 30), "2026-10-30") == date(2026, 11, 6)

    def test_a_sunday_points_at_friday(self):
        assert next_decision_day(date(2026, 10, 25), None) == date(2026, 10, 30)


class TestStatePayload:
    def test_no_ledger_says_so(self):
        assert build_state_payload(None, [], {}, date(2026, 10, 25)) == {"exists": False}

    def test_before_the_first_mark_there_is_no_series(self):
        state = {**STATE, "first_session": None, "last_processed": None}
        payload = build_state_payload(state, [], {}, date(2026, 10, 25))
        assert payload["series"] is None and payload["equity"] == 1000.0

    def test_every_line_starts_at_100_on_the_base_session(self):
        ledger = [mark("2026-10-26", 1010.0), mark("2026-10-27", 990.0)]
        bars = {"SPY": adj([500.0, 505.0, 510.0]), "SGOV": adj([100.0, 100.01, 100.02])}
        payload = build_state_payload(STATE, ledger, bars, date(2026, 10, 27))
        series = payload["series"]
        assert series["dates"] == ["2026-10-23", "2026-10-26", "2026-10-27"]
        assert series["sim"] == [100.0, 101.0, 99.0]
        assert series["spy"] == [100.0, 101.0, 102.0]
        assert payload["spy_return"] == pytest.approx(0.02)
        assert payload["total_return"] == pytest.approx(-0.01)
        assert payload["max_drawdown"] == pytest.approx(990.0 / 1010.0 - 1.0)


class TestLivePayload:
    def test_the_estimate_is_cash_plus_shares_at_the_quotes(self):
        quotes = {"SPY": {"price": 510.0, "previous_close": 500.0,
                          "time": "2026-10-26T11:00:00-04:00",
                          "session_start": "2026-10-26T09:30:00-04:00",
                          "session_end": "2026-10-26T16:00:00-04:00"},
                  "AGG": {"price": 100.0, "previous_close": 100.0}}
        now = datetime(2026, 10, 26, 11, 1, tzinfo=EASTERN)
        payload = build_live_payload(STATE, quotes, now)
        assert payload["equity"] == pytest.approx(100.0 + 510.0 + 200.0)
        assert payload["market_open"] is True
        assert payload["rows"][0]["day_change"] == pytest.approx(0.02)

    def test_a_missing_quote_leaves_a_gap_not_a_zero_price(self):
        now = datetime(2026, 10, 26, 18, 0, tzinfo=EASTERN)
        payload = build_live_payload(STATE, {"SPY": {"error": "timeout"}}, now)
        assert payload["rows"][0]["price"] is None
        assert payload["market_open"] is False


REPORT = ("正文\n```json\n" + json.dumps({"schema": 1, "task": "radar", "items": [
    {"ticker": "NVDA", "action": "risk", "score": 7}]}) + "\n```")


@pytest.fixture
def server(tmp_path):
    """The real handler on a spare port, writing into a temporary folder."""
    args = argparse.Namespace(state_dir=str(tmp_path / "sim"), name="thirds",
                              cache_dir=str(tmp_path / "cache"),
                              signals_dir=str(tmp_path / "signals"))
    allowed = set()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Dashboard(args), allowed))
    port = httpd.server_address[1]
    allowed.update({f"127.0.0.1:{port}", f"localhost:{port}"})
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield port, tmp_path / "signals"
    httpd.shutdown()
    httpd.server_close()


def post(port, body, headers):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request("POST", "/api/signals", body=body.encode("utf-8"),
                       headers={"Host": f"127.0.0.1:{port}", **headers})
    response = connection.getresponse()
    return response.status, json.loads(response.read() or b"{}")


class TestWhoMayWrite:
    def test_the_page_itself_can_paste_a_report(self, server):
        port, folder = server
        status, result = post(port, json.dumps({"text": REPORT, "task": "radar"}),
                              {"Content-Type": "application/json",
                               "Origin": f"http://127.0.0.1:{port}"})
        assert status == 200 and result["added"] == 1
        assert (folder / "inbox.jsonl").exists()

    def test_a_form_post_from_another_site_is_refused(self, server):
        # A plain form can be sent cross-site without the browser asking
        # first; only a JSON body forces the preflight this server never grants.
        port, folder = server
        status, _ = post(port, "text=" + REPORT, {"Content-Type": "text/plain"})
        assert status == 415
        assert not (folder / "inbox.jsonl").exists()

    def test_a_json_post_with_a_foreign_origin_is_refused(self, server):
        port, folder = server
        status, _ = post(port, json.dumps({"text": REPORT, "task": "radar"}),
                         {"Content-Type": "application/json",
                          "Origin": "http://evil.example"})
        assert status == 403
        assert not (folder / "inbox.jsonl").exists()

    def test_a_radar_report_in_the_daily_box_is_not_filed(self, server):
        port, folder = server
        status, result = post(port, json.dumps({"text": REPORT, "task": "daily"}),
                              {"Content-Type": "application/json"})
        assert status == 200 and result["added"] == 0 and result["errors"]

    def test_a_foreign_host_header_cannot_even_read(self, server):
        port, _ = server
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request("GET", "/api/signals", headers={"Host": "evil.example"})
        assert connection.getresponse().status == 403


class TestSignalBarsCache:
    def chart_from(self, first_day):
        days = pd.bdate_range(first_day, periods=5)
        stamps = [int(datetime(d.year, d.month, d.day, 9, 30, tzinfo=EASTERN).timestamp())
                  for d in days]
        return {"chart": {"error": None, "result": [{
            "timestamp": stamps,
            "indicators": {"quote": [{"close": [1.0] * 5, "open": [1.0] * 5}],
                           "adjclose": [{"adjclose": [1.0] * 5}]}}]}}

    def test_a_fresh_cache_that_starts_too_late_is_fetched_again(self, tmp_path, monkeypatch):
        import ibkr_ml.yahoo as yahoo

        args = argparse.Namespace(state_dir=str(tmp_path), name="t",
                                  cache_dir=str(tmp_path / "cache"),
                                  signals_dir=str(tmp_path / "signals"))
        dashboard = Dashboard(args)
        cache = dashboard.signal_cache
        cache.mkdir(parents=True)
        (cache / "GOOGL.json").write_text(json.dumps(self.chart_from("2026-10-12")))
        fetched = []

        def fake_fetch(symbol, cache_dir, start=None, timeout=30.0):
            fetched.append(start)
            return self.chart_from("2026-10-05")

        monkeypatch.setattr(yahoo, "fetch_chart", fake_fetch)
        frame = dashboard.bars_for("GOOGL", datetime(2026, 10, 6, tzinfo=EASTERN))
        assert len(fetched) == 1
        assert frame.index[0] == pd.Timestamp("2026-10-05")

    def test_a_fresh_cache_that_covers_the_start_is_used_as_is(self, tmp_path, monkeypatch):
        import ibkr_ml.yahoo as yahoo

        args = argparse.Namespace(state_dir=str(tmp_path), name="t",
                                  cache_dir=str(tmp_path / "cache"),
                                  signals_dir=str(tmp_path / "signals"))
        dashboard = Dashboard(args)
        dashboard.signal_cache.mkdir(parents=True)
        (dashboard.signal_cache / "GOOGL.json").write_text(json.dumps(self.chart_from("2026-10-01")))
        monkeypatch.setattr(yahoo, "fetch_chart",
                            lambda *a, **k: pytest.fail("fetched a cache that already covers"))
        frame = dashboard.bars_for("GOOGL", datetime(2026, 10, 6, tzinfo=EASTERN))
        assert frame.index[0] == pd.Timestamp("2026-10-01")
