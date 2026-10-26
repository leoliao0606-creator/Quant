"""Reports from the ChatGPT scheduled tasks, kept and scored here.

Three tasks feed this module, each pasted into its own box:

- daily   the 9:00 screen, a scored candidate pool, 2 weeks to 3 months
- radar   the hourly opportunity-and-risk radar, silent unless something
          changes a decision
- short   a task written for measurement rather than reading: the same
          fixed list of names scored every morning for the next five
          sessions against SPY

OpenAI's terms forbid extracting ChatGPT's output programmatically, so
nothing here reads chatgpt.com. A report arrives as pasted text ending in a
fixed JSON block; the prose is kept as it arrived and only the block is
parsed.

A report written in real time cannot know the prices that followed it, so
an archive of them is a forward test with none of the look-ahead that
spoils LLM-scored historical news. That holds only while the archive is
honest:

- The clock is the moment the text was pasted, never a time the report
  claims for itself. Pasting late can only move the entry later.
- The entry is the first regular-session open at or after that moment.
- The same block pasted twice is one report, and a block pasted into the
  wrong task's box is refused rather than filed under the box.
- Every item is scored against SPY at fixed checkpoints - the close of
  the 1st, 5th, 20th and 60th session from the entry - so a fast call and
  a slow one are each judged where they claim to work. The sign is the
  report's: buy and opportunity are right when the name beats SPY; sell,
  avoid and risk when it trails; watch takes no side.
- Whether the score means anything is asked separately: within each
  report, do the higher-scored names do better? That is a rank
  correlation per report (an information coefficient), averaged over
  reports. It needs every candidate scored, not only the recommended ones.

Limit prices, stops and targets are stored but not simulated yet; the
score here is "bought at the open, held".
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, time as time_of_day, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
OPEN_TIME = time_of_day(9, 30)
TASKS = ("daily", "radar", "short")
# +1 means the report expects the name to beat SPY, -1 to trail it. "watch"
# is a candidate scored but not recommended: its return is kept so the score
# itself can be tested, and it takes no side in the hit count.
ACTIONS = {"buy": 1, "opportunity": 1, "sell": -1, "avoid": -1, "risk": -1,
           "watch": 0}
# A closed list, so that "which kind of reason worked" can be counted later.
# Free text cannot be counted; a tag outside the list is filed as "other".
REASONS = ("earnings", "guidance", "valuation", "ai_demand", "orders", "macro",
           "rates", "energy", "financing", "insider", "analyst", "flows",
           "technical", "other")
THESIS_CHANGES = ("new", "strengthen", "weaken", "invalidate")
CHECKPOINTS = (1, 5, 20, 60)
IC_CHECKPOINTS = (5, 20)
IC_MIN_ITEMS = 5
TICKER = re.compile(r"^[A-Z][A-Z0-9-]{0,9}$")
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)
BENCHMARK = "SPY"


def extract_reports(text, expected_task=None):
    """Every valid report block in a pasted text, and what was wrong."""
    candidates = FENCE.findall(text or "")
    if not candidates:
        stripped = (text or "").strip()
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            candidates = [stripped[start:end + 1]]
    reports, errors = [], []
    for index, block in enumerate(candidates, 1):
        try:
            payload = json.loads(block)
        except ValueError as exc:
            errors.append(f"第 {index} 个 JSON 块解析失败：{exc}")
            continue
        report, problems = validate_report(payload, expected_task)
        errors.extend(f"第 {index} 个 JSON 块：{p}" for p in problems)
        if report is not None:
            reports.append(report)
    if not candidates:
        errors.append("没有找到 JSON 块。报告末尾应附 ```json ... ``` 格式的代码块")
    return reports, errors


def _number(value, low=None, high=None):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"应为数字，实际是 {value!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{value} 超出范围 [{low}, {high}]")
    return float(value)


def _text(value, limit):
    return str(value).strip()[:limit] if value not in (None, "") else None


def validate_report(payload, expected_task=None):
    """A normalised report, or None, and the problems found.

    `expected_task` is the box the text was pasted into. A block that names
    a different task is refused: filing a radar alert as a daily pick would
    score it against the wrong horizon and mix two records that must stay
    apart. A block that names no task takes the box's.
    """
    if not isinstance(payload, dict):
        return None, ["顶层应为 JSON 对象"]
    if payload.get("schema") != 1:
        return None, [f"schema 应为 1，实际是 {payload.get('schema')!r}"]
    task = payload.get("task") or expected_task
    if task not in TASKS:
        return None, [f"task 应为 {'/'.join(TASKS)} 之一，实际是 {task!r}"]
    if expected_task and task != expected_task:
        return None, [f"这段报告写的是 task \"{task}\"，却粘贴在了 \"{expected_task}\" 的框里，"
                      f"没有导入。请粘贴到对应的标签页"]
    items = payload.get("items")
    if not isinstance(items, list):
        return None, ["items 应为列表"]

    problems, kept = [], []
    for position, raw in enumerate(items, 1):
        try:
            kept.append(_validate_item(raw))
        except ValueError as exc:
            problems.append(f"第 {position} 条被丢弃：{exc}")
    report = {"schema": 1, "task": task, "items": kept}
    claimed = _text(payload.get("report_time_et"), 40)
    if claimed:
        report["report_time_et"] = claimed
    return report, problems


def _validate_item(raw):
    if not isinstance(raw, dict):
        raise ValueError("应为 JSON 对象")
    ticker = str(raw.get("ticker", "")).strip().upper().replace(".", "-")
    if not TICKER.match(ticker):
        raise ValueError(f"ticker {raw.get('ticker')!r} 不是有效代码")
    action = str(raw.get("action", "")).strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"action {raw.get('action')!r} 应为 {'/'.join(ACTIONS)} 之一")
    horizon = raw.get("horizon_days")
    if horizon is not None:
        if isinstance(horizon, bool) or not isinstance(horizon, (int, float)) \
                or not 1 <= horizon <= 400:
            raise ValueError(f"horizon_days {horizon!r} 应为 1 到 400 的天数")
        horizon = int(horizon)
    sources = raw.get("sources") or []
    if not isinstance(sources, list):
        raise ValueError("sources 应为列表")
    item = {"ticker": ticker, "action": action,
            "score": _number(raw.get("score"), 0, 10),
            "horizon_days": horizon,
            "entry_limit": _number(raw.get("entry_limit"), 0),
            "stop": _number(raw.get("stop"), 0),
            "target": _number(raw.get("target"), 0),
            "price_seen": _number(raw.get("price_seen"), 0),
            "invalidation": str(raw.get("invalidation") or "")[:500],
            "sources": [str(s)[:500] for s in sources[:20]]}

    subscores = raw.get("subscores")
    if subscores is not None:
        if not isinstance(subscores, dict) or len(subscores) > 8:
            raise ValueError("subscores 应为最多 8 项的 {名称: 分数} 对象")
        item["subscores"] = {str(k)[:20]: _number(v, 0, 10) for k, v in subscores.items()}
    reasons = raw.get("reasons")
    if reasons is not None:
        if not isinstance(reasons, list):
            raise ValueError("reasons 应为列表")
        item["reasons"] = sorted({r if r in REASONS else "other"
                                  for r in (str(x).strip().lower() for x in reasons[:6])})
    catalyst = raw.get("catalyst")
    if isinstance(catalyst, dict):
        day = _text(catalyst.get("date"), 10)
        item["catalyst"] = {"type": _text(catalyst.get("type"), 40),
                            "date": day if day and DAY.match(day) else None}
    event = raw.get("event")
    if isinstance(event, dict):
        item["event"] = {"type": _text(event.get("type"), 40),
                         "time_et": _text(event.get("time_et"), 40)}
    materiality = raw.get("materiality")
    if materiality is not None:
        item["materiality"] = int(_number(materiality, 1, 5))
    change = raw.get("thesis_change")
    if change is not None:
        if change not in THESIS_CHANGES:
            raise ValueError(f"thesis_change {change!r} 应为 {'/'.join(THESIS_CHANGES)} 之一")
        item["thesis_change"] = change
    return item


def report_key(report):
    canonical = json.dumps(report, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def load_inbox(folder):
    records = []
    try:
        lines = (Path(folder) / "inbox.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        return records
    for line in lines:
        if line.strip():
            records.append(json.loads(line))
    return records


def ingest(text, folder, received, expected_task=None):
    """Store every new report in `text`; return what happened.

    The pasted text is kept whole under raw/, named by its hash, so the
    prose a block came with can always be read back next to the record.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    reports, errors = extract_reports(text, expected_task)
    known = {record["key"] for record in load_inbox(folder)}
    added, duplicates = [], 0
    raw_name = None
    if reports:
        raw_name = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] + ".txt"
        (folder / "raw").mkdir(parents=True, exist_ok=True)
        (folder / "raw" / raw_name).write_text(text, encoding="utf-8")
    with (folder / "inbox.jsonl").open("a", encoding="utf-8") as handle:
        for report in reports:
            key = report_key(report)
            if key in known:
                duplicates += 1
                continue
            known.add(key)
            record = {"key": key, "received": received.isoformat(timespec="seconds"),
                      "source": "paste", "raw": raw_name, **report}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            added.append(record)
    return {"added": added, "duplicates": duplicates, "errors": errors}


def entry_session(index, received):
    """The first session in `index` whose open is at or after `received`."""
    import pandas as pd

    position = index.searchsorted(pd.Timestamp(received.date()))
    for day in index[position:position + 2]:
        if datetime.combine(day.date(), OPEN_TIME, EASTERN) >= received:
            return day
    return None


def _adjusted_open(bars, day):
    """The open in adjusted-close units, so later dividends count."""
    opened = float(bars.at[day, "open"])
    return opened * float(bars.at[day, "adjclose"]) / float(bars.at[day, "close"])


def score_item(item, received, bars, benchmark):
    """Excess over SPY from the entry open, at each checkpoint and so far."""
    direction = ACTIONS[item["action"]]
    row = {"ticker": item["ticker"], "action": item["action"], "direction": direction,
           "score": item.get("score"), "horizon_days": item.get("horizon_days"),
           "entry_limit": item.get("entry_limit"), "stop": item.get("stop"),
           "target": item.get("target"), "price_seen": item.get("price_seen"),
           "materiality": item.get("materiality"), "reasons": item.get("reasons"),
           "event": item.get("event"), "thesis_change": item.get("thesis_change"),
           "subscores": item.get("subscores"), "status": "等待开盘", "checkpoints": {}}
    if bars is None or bars.empty:
        row["status"] = "没有价格数据"
        return row
    if benchmark is None or benchmark.empty:
        row["status"] = "没有 SPY 数据"
        return row
    entry = entry_session(bars.index, received)
    if entry is None or entry not in benchmark.index:
        return row
    if math.isnan(float(bars.at[entry, "open"])) or math.isnan(float(benchmark.at[entry, "open"])):
        row["status"] = "缺开盘价"
        return row

    span = bars.index[bars.index >= entry].intersection(benchmark.index)
    own_base, market_base = _adjusted_open(bars, entry), _adjusted_open(benchmark, entry)

    def excess_at(day):
        own = float(bars.at[day, "adjclose"]) / own_base - 1.0
        market = float(benchmark.at[day, "adjclose"]) / market_base - 1.0
        return own, market

    for count in CHECKPOINTS:
        if len(span) >= count:
            own, market = excess_at(span[count - 1])
            row["checkpoints"][str(count)] = {
                "through": str(span[count - 1].date()), "excess": own - market,
                "signed": direction * (own - market) if direction else None}

    horizon = item.get("horizon_days")
    expired = False
    if horizon:
        due = entry + timedelta(days=horizon)
        expired = span[-1] >= due
        span = span[span <= due]
    end = span[-1]
    own, market = excess_at(end)
    prior = bars.index[bars.index < entry]
    if item.get("price_seen") and len(prior):
        row["price_gap"] = item["price_seen"] / float(bars.at[prior[-1], "close"]) - 1.0
    row.update({"status": "已到期" if expired else "进行中",
                "entry_session": str(entry.date()),
                "entry_price": float(bars.at[entry, "open"]),
                "through": str(end.date()), "last_price": float(bars.at[end, "close"]),
                "return": own, "spy_return": market, "excess": own - market,
                "signed_excess": direction * (own - market) if direction else None})
    return row


def score_inbox(records, bars_for):
    """One row per item, newest report first. `bars_for(ticker)` gives bars."""
    benchmark = bars_for(BENCHMARK)
    rows = []
    for record in sorted(records, key=lambda r: r["received"], reverse=True):
        received = datetime.fromisoformat(record["received"])
        for item in record["items"]:
            row = score_item(item, received, bars_for(item["ticker"]), benchmark)
            row.update({"received": record["received"], "task": record["task"],
                        "key": record["key"]})
            rows.append(row)
    return rows


def _stats(values):
    """Count, hit rate, mean and t-statistic of signed excess returns."""
    n = len(values)
    if not n:
        return {"n": 0}
    mean = sum(values) / n
    out = {"n": n, "hits": sum(1 for v in values if v > 0), "mean": mean, "t": None}
    if n >= 2:
        spread = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        if spread > 0:
            out["t"] = mean / (spread / math.sqrt(n))
    return out


def information_coefficient(rows, checkpoint):
    """Mean within-report rank correlation of score and excess return.

    Only reports with at least IC_MIN_ITEMS matured, scored items count; a
    correlation over three names says nothing.
    """
    import pandas as pd

    by_report = {}
    for row in rows:
        mark = row["checkpoints"].get(str(checkpoint))
        if row.get("score") is not None and mark:
            by_report.setdefault(row["key"], []).append((row["score"], mark["excess"]))
    values = []
    for pairs in by_report.values():
        if len(pairs) < IC_MIN_ITEMS:
            continue
        frame = pd.DataFrame(pairs, columns=["score", "excess"])
        if frame["score"].nunique() < 2:
            continue
        value = frame["score"].rank().corr(frame["excess"].rank())
        if not math.isnan(value):
            values.append(value)
    stats = _stats(values)
    return {"reports": stats["n"], "mean": stats.get("mean"), "t": stats.get("t")}


def summarize(rows):
    """Per-checkpoint hit statistics for the directional items, and the IC."""
    directional = [r for r in rows if r["direction"]]
    summary = {"items": len(rows), "directional": len(directional),
               "entered": sum(1 for r in rows if r.get("entry_session")),
               "checkpoints": {}, "ic": {}}
    for count in CHECKPOINTS:
        values = [r["checkpoints"][str(count)]["signed"] for r in directional
                  if str(count) in r["checkpoints"]]
        summary["checkpoints"][str(count)] = _stats(values)
    for count in IC_CHECKPOINTS:
        summary["ic"][str(count)] = information_coefficient(rows, count)
    return summary
