#!/usr/bin/env python3
"""Summarize paper-trading JSONL logs and explain why a run produced no orders.

The live loop writes one JSON object per line to logs/paper_trade_<date>.jsonl.
Reading those files by hand does not answer the question that actually matters
after an unattended session: did the strategy trade, and if not, what stopped
it. This script answers that from the logs alone.

It deliberately imports nothing outside the standard library, so it also runs
on a Raspberry Pi that only has the paper-trading virtualenv, or none at all.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


# Age thresholds used to turn a measured bar age into a named cause. A bar
# stamped in the future can only come from reading the timestamp in the wrong
# timezone; a bar consistently 15-25 minutes old matches IBKR's delayed feed.
FUTURE_BAR_MINUTES = -1.0
DELAYED_FEED_CEILING_MINUTES = 30.0


def _load_events(path: Path) -> list[dict]:
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            print(f"warning: {path}:{line_number} is not valid JSON, skipped ({exc.msg})")
    return events


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def _clock(timestamp_text: str) -> str:
    try:
        return datetime.fromisoformat(timestamp_text).strftime("%H:%M:%S")
    except (TypeError, ValueError):
        return str(timestamp_text)


class LogSummary:
    """Everything the diagnosis needs, collected in one pass over the events."""

    def __init__(self, path: Path, events: list[dict]) -> None:
        self.path = path
        self.events = events
        self.event_counts: Counter[str] = Counter()
        self.decision_counts: Counter[tuple[str, str]] = Counter()
        self.skip_reasons: Counter[str] = Counter()
        self.block_reasons: Counter[str] = Counter()
        self.order_rows: list[dict] = []
        self.untransmitted_rows: list[dict] = []
        self.skipped_order_rows: list[dict] = []
        self.error_rows: list[dict] = []
        self.block_ages: list[float] = []
        self.block_ages_by_symbol: defaultdict[str, list[float]] = defaultdict(list)
        self.stale_after_minutes: float | None = None
        self.bar_timezones: Counter[str] = Counter()
        self.equity_points: list[tuple[str, float]] = []
        self.first_timestamp: str | None = None
        self.last_timestamp: str | None = None
        self._collect()

    def _collect(self) -> None:
        for event in self.events:
            event_type = str(event.get("event_type", "unknown"))
            payload = event.get("payload") or {}
            timestamp = str(event.get("timestamp", ""))

            self.event_counts[event_type] += 1
            if self.first_timestamp is None:
                self.first_timestamp = timestamp
            self.last_timestamp = timestamp

            if event_type == "cycle":
                equity = payload.get("equity")
                if isinstance(equity, (int, float)):
                    self.equity_points.append((timestamp, float(equity)))
                for decision in payload.get("decisions", []):
                    action = str(decision.get("action", "?"))
                    reason = str(decision.get("reason", "?"))
                    self.decision_counts[(action, reason)] += 1

            elif event_type == "cycle_skipped":
                self.skip_reasons[str(payload.get("reason", "?"))] += 1
                equity = payload.get("equity")
                if isinstance(equity, (int, float)):
                    self.equity_points.append((timestamp, float(equity)))

            elif event_type == "order_submitted":
                self.order_rows.append({"timestamp": timestamp, **payload})

            elif event_type == "order_not_transmitted":
                self.untransmitted_rows.append({"timestamp": timestamp, **payload})

            elif event_type == "order_skipped":
                self.skipped_order_rows.append({"timestamp": timestamp, **payload})

            elif event_type == "bar_blocked":
                reason = str(payload.get("reason", "?"))
                self.block_reasons[reason] += 1
                age = payload.get("age_minutes")
                if isinstance(age, (int, float)):
                    self.block_ages.append(float(age))
                    self.block_ages_by_symbol[str(payload.get("symbol", "?"))].append(float(age))
                stale_after = payload.get("stale_after_minutes")
                if isinstance(stale_after, (int, float)):
                    self.stale_after_minutes = float(stale_after)
                bar_timezone = payload.get("bar_timezone")
                if bar_timezone:
                    self.bar_timezones[str(bar_timezone)] += 1

            elif event_type in {"symbol_error", "cycle_error"}:
                self.error_rows.append({"timestamp": timestamp, "event_type": event_type, **payload})

    @property
    def total_decisions(self) -> int:
        return sum(self.decision_counts.values())

    @property
    def dominant_reason(self) -> tuple[str, int] | None:
        by_reason: Counter[str] = Counter()
        for (_, reason), count in self.decision_counts.items():
            by_reason[reason] += count
        if not by_reason:
            return None
        return by_reason.most_common(1)[0]

    @property
    def age_stats(self) -> dict[str, float] | None:
        if not self.block_ages:
            return None
        ordered = sorted(self.block_ages)
        return {
            "min": ordered[0],
            "p25": _percentile(ordered, 0.25),
            "median": _percentile(ordered, 0.50),
            "p75": _percentile(ordered, 0.75),
            "max": ordered[-1],
        }


def _print_report(summary: LogSummary) -> None:
    print(f"=== {summary.path} ===")

    if summary.first_timestamp and summary.last_timestamp:
        print(f"Session   : {_clock(summary.first_timestamp)} -> {_clock(summary.last_timestamp)}")

    executed = summary.event_counts.get("cycle", 0)
    skipped = summary.event_counts.get("cycle_skipped", 0)
    skip_detail = ""
    if summary.skip_reasons:
        skip_detail = " (" + ", ".join(
            f"{reason}: {count}" for reason, count in summary.skip_reasons.most_common()
        ) + ")"
    print(f"Cycles    : {executed} executed, {skipped} skipped{skip_detail}")

    if summary.equity_points:
        first_equity = summary.equity_points[0][1]
        last_equity = summary.equity_points[-1][1]
        change = last_equity - first_equity
        change_pct = (change / first_equity * 100.0) if first_equity else 0.0
        print(f"Equity    : {first_equity:,.2f} -> {last_equity:,.2f} ({change:+,.2f}, {change_pct:+.2f}%)")

    print()
    print(f"Orders submitted: {len(summary.order_rows)}")
    for row in summary.order_rows:
        print(
            f"  {_clock(row.get('timestamp', ''))} {row.get('symbol', '?'):<6} "
            f"{row.get('action', '?'):<4} qty={row.get('quantity_delta', '?'):<6} "
            f"status={row.get('order_status', '?'):<12} reason={row.get('reason', '?')}"
        )

    if summary.untransmitted_rows:
        print(f"Orders held inside TWS: {len(summary.untransmitted_rows)}")
        for row in summary.untransmitted_rows:
            print(
                f"  {_clock(row.get('timestamp', ''))} {row.get('symbol', '?'):<6} "
                f"{row.get('action', '?'):<4} qty={row.get('quantity_delta', '?'):<6} "
                f"status={row.get('order_status', '?'):<12} reason={row.get('reason', '?')}"
            )

    if summary.skipped_order_rows:
        print(f"Orders skipped: {len(summary.skipped_order_rows)}")
        for row in summary.skipped_order_rows:
            print(
                f"  {_clock(row.get('timestamp', ''))} {row.get('symbol', '?'):<6} "
                f"reason={row.get('reason', '?')}"
            )

    print()
    if summary.total_decisions:
        print(f"Decisions ({summary.total_decisions} total):")
        for (action, reason), count in summary.decision_counts.most_common():
            share = count / summary.total_decisions * 100.0
            print(f"  {action:<5} {reason:<32} {count:>6}  {share:5.1f}%")
    else:
        print("Decisions : none recorded")

    print()
    if summary.block_reasons:
        blocked_total = sum(summary.block_reasons.values())
        print(f"Bar freshness ({blocked_total} bar_blocked events):")
        for reason, count in summary.block_reasons.most_common():
            print(f"  {reason:<28} {count:>6}")
        stats = summary.age_stats
        if stats:
            print(
                "  age_minutes: "
                f"min {stats['min']:.1f} | p25 {stats['p25']:.1f} | "
                f"median {stats['median']:.1f} | p75 {stats['p75']:.1f} | max {stats['max']:.1f}"
            )
            for symbol in sorted(summary.block_ages_by_symbol):
                ages = sorted(summary.block_ages_by_symbol[symbol])
                print(f"    {symbol:<6} median {_percentile(ages, 0.50):7.1f}  n={len(ages)}")
        if summary.stale_after_minutes is not None:
            print(f"  stale_after_minutes in use: {summary.stale_after_minutes:g}")
        if summary.bar_timezones:
            print(f"  bar_timezone in use: {', '.join(sorted(summary.bar_timezones))}")
    else:
        print("Bar freshness : no bar_blocked events in this log")

    print()
    if summary.error_rows:
        error_counts = Counter(
            (row["event_type"], str(row.get("error_type", "?"))) for row in summary.error_rows
        )
        print(f"Errors ({len(summary.error_rows)} total):")
        for (event_type, error_type), count in error_counts.most_common():
            example = next(
                row for row in summary.error_rows
                if row["event_type"] == event_type and str(row.get("error_type", "?")) == error_type
            )
            message = str(example.get("error", "")).strip() or "(empty message)"
            print(f"  {event_type}/{error_type} x{count}: {message[:110]}")
    else:
        print("Errors    : none")


def _diagnose(summary: LogSummary) -> list[str]:
    """Turn the collected numbers into named causes and concrete next steps."""
    lines: list[str] = []
    executed = summary.event_counts.get("cycle", 0)

    if executed == 0 and not summary.order_rows:
        lines.append(
            "[IDLE] No trading cycle ran. Every cycle was skipped or errored, so "
            "the strategy was never asked for a decision."
        )

    if summary.untransmitted_rows:
        symbols = sorted({str(row.get("symbol", "?")) for row in summary.untransmitted_rows})
        statuses = sorted({str(row.get("order_status", "?")) for row in summary.untransmitted_rows})
        lines.append(
            f"[NOT TRANSMITTED] {len(summary.untransmitted_rows)} orders reached TWS but never "
            f"reached IBKR (status {', '.join(statuses)}; symbols {', '.join(symbols)}). TWS is "
            "holding them, almost always behind the order precautions dialog. In TWS open "
            "Global Configuration - API - Precautions and tick 'Bypass Order Precautions for "
            "API Orders', then check no dialog is waiting on screen. Such orders are discarded "
            "when the client disconnects, so nothing was traded."
        )

    if executed > 0 and not summary.order_rows:
        dominant = summary.dominant_reason
        detail = ""
        if dominant:
            reason, count = dominant
            share = count / max(summary.total_decisions, 1) * 100.0
            detail = f" Dominant decision reason: {reason} ({count}/{summary.total_decisions}, {share:.1f}%)."
        lines.append(
            f"[BLOCKED] {executed} cycles ran and no order was ever submitted.{detail}"
        )

    stats = summary.age_stats
    if stats is not None:
        median = stats["median"]
        threshold = summary.stale_after_minutes if summary.stale_after_minutes is not None else 15.0

        if median < FUTURE_BAR_MINUTES:
            lines.append(
                f"[TIMEZONE] Median bar age is {median:.1f} min, i.e. bars are stamped in the "
                "future. The bar timestamps are being read in the wrong timezone. Set "
                "--bar-timezone to the IANA zone of the machine running TWS/IB Gateway "
                f"(currently reading them as {', '.join(sorted(summary.bar_timezones)) or 'US/Eastern'})."
            )
        elif median > DELAYED_FEED_CEILING_MINUTES:
            lines.append(
                f"[STALE] Median bar age is {median:.1f} min, far past the {threshold:g} min "
                "threshold and past what a delayed feed explains. Check that the market data "
                "subscription covers these symbols, that TWS is not in a reconnecting state, "
                "and that --bar-timezone matches the TWS machine."
            )
        elif median > threshold:
            lines.append(
                f"[DELAYED FEED] Median bar age is {median:.1f} min, just above the "
                f"{threshold:g} min threshold, and consistently positive. That is the signature "
                "of IBKR's delayed market data (15-20 min without a realtime subscription) "
                "rather than a timezone mistake. Either subscribe to realtime data for these "
                f"symbols, or raise --stale-after-minutes above {median:.0f} - but note a "
                "5-minute-bar model predicting 15 minutes ahead is already expired by then."
            )
        else:
            lines.append(
                f"[FRESH] Median bar age is {median:.1f} min, inside the {threshold:g} min "
                "threshold. Blocked bars here are mostly same_bar, meaning the loop polled "
                "faster than new bars arrive. Align --interval-seconds with the bar size."
            )

    if summary.block_reasons.get("same_bar") and not summary.block_reasons.get("stale_data"):
        lines.append(
            "[POLLING] Every block was same_bar: the loop is re-running before a new bar "
            "closes. Harmless, but it wastes IBKR requests."
        )

    if summary.event_counts.get("cycle") and not summary.block_reasons and summary.dominant_reason:
        reason, count = summary.dominant_reason
        if reason == "stale_data":
            lines.append(
                "[UPGRADE] Decisions say stale_data but this log has no bar_blocked events, "
                "so the measured age was never recorded. That code predates the current "
                "version. Re-run with the current code and analyze again to get age_minutes."
            )

    session_conflicts = [
        row for row in summary.error_rows
        if "different ip" in str(row.get("error", "")).lower()
    ]
    if session_conflicts:
        lines.append(
            f"[SESSION] {len(session_conflicts)} IBKR 162 'connected from a different IP address' "
            "errors. Another API client holds the TWS session. Close it, or give this process "
            "its own --client-id."
        )

    timeouts = [row for row in summary.error_rows if "timeout" in str(row.get("error_type", "")).lower()]
    if timeouts:
        lines.append(
            f"[TIMEOUT] {len(timeouts)} request timeouts. Raise --request-timeout, or reduce "
            "the number of symbols per cycle."
        )

    if summary.order_rows:
        actions = Counter(str(row.get("action", "?")) for row in summary.order_rows)
        reasons = Counter(str(row.get("reason", "?")) for row in summary.order_rows)
        lines.append(
            f"[TRADED] {len(summary.order_rows)} orders: "
            + ", ".join(f"{action} x{count}" for action, count in actions.most_common())
            + " | reasons: "
            + ", ".join(f"{reason} x{count}" for reason, count in reasons.most_common())
        )

    if not lines:
        lines.append("[OK] Nothing stood out in this log.")
    return lines


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize paper-trading JSONL logs and diagnose why a run produced no orders.",
    )
    parser.add_argument(
        "logs",
        nargs="*",
        help="Log files to analyze. Defaults to every paper_trade_*.jsonl under --log-dir.",
    )
    parser.add_argument("--log-dir", default="logs", help="Directory to scan when no files are given.")
    parser.add_argument("--last", type=int, default=None, help="Only analyze the N most recent files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.logs:
        paths = [Path(item) for item in args.logs]
    else:
        paths = sorted(Path(args.log_dir).glob("paper_trade_*.jsonl"))

    if not paths:
        print(f"No log files found. Looked in {args.log_dir}/paper_trade_*.jsonl")
        return

    if args.last is not None:
        paths = paths[-args.last:]

    all_diagnoses: list[tuple[Path, list[str]]] = []
    for path in paths:
        if not path.exists():
            print(f"warning: {path} does not exist, skipped")
            continue

        summary = LogSummary(path, _load_events(path))
        _print_report(summary)
        diagnosis = _diagnose(summary)
        all_diagnoses.append((path, diagnosis))
        print()
        print("DIAGNOSIS")
        for line in diagnosis:
            print(f"  {line}")
        print()

    if len(all_diagnoses) > 1:
        print("=" * 72)
        print("ACROSS ALL LOGS")
        for path, diagnosis in all_diagnoses:
            tags = " ".join(line.split("]")[0] + "]" for line in diagnosis)
            print(f"  {path.name}: {tags}")


if __name__ == "__main__":
    main()
