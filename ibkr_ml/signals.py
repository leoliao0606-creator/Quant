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

Two scores are kept for a long call. "Bought at the open, held" says
whether the view was right. "As planned" (`simulate_plan`) says whether the
trade the report laid out - its limit, stop, target and horizon - would
have made money, filled and exited on daily bars by rules fixed before any
report arrived. A view can be right and its plan still lose: a limit that
never fills, a stop inside the noise.

Neither says whether the result is better than luck. Two controls do:
within a report, the names it chose against the ones it scored and passed
over; and against the same number of names drawn at random, on the same
entry dates, from a fixed universe (`random_control`).
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
LONG_ACTIONS = ("buy", "opportunity")
EXIT_ACTIONS = ("sell", "avoid", "risk")
# A limit order not filled within this many sessions is treated as cancelled.
LIMIT_VALID_SESSIONS = 5
# Holding period for a plan that names none.
DEFAULT_HOLD_SESSIONS = 20
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


def _factor(bars, day):
    """Adjusted over printed close: turns a printed price into adjclose units."""
    return float(bars.at[day, "adjclose"]) / float(bars.at[day, "close"])


def simulate_plan(item, received, bars, benchmark, use_levels=True,
                  default_sessions=DEFAULT_HOLD_SESSIONS):
    """The long trade a report laid out, filled and exited on daily bars.

    Fixed before any report arrived, and conservative wherever a daily bar
    cannot say which came first:

    - Entry. With no limit, the entry session's open. With a limit, the
      first of the next LIMIT_VALID_SESSIONS sessions that reaches it: at
      the open if the open is already at or below it, else at the limit if
      the low touches it. None of them: not filled.
    - Exit, from the next session on. An open through the stop or the
      target exits at that open; otherwise a low through the stop exits at
      the stop and a high through the target at the target; both on one bar
      counts as the stop. On the fill session itself, a fill at the open is
      checked the same way, and a fill at the limit only against the stop -
      the bar cannot say whether its low came before or after the fill, so
      the losing order is assumed.
    - Horizon. `horizon_days` calendar days after the fill, exiting at the
      close of the first session on or after it; without one,
      `default_sessions` sessions.
    - A stop at or above the fill, or a target at or below it, cannot have
      been meant and is ignored with a note.

    Returns the fill, the exit if there was one, and the return against SPY
    over the same span, both in adjusted terms so dividends count.
    """
    if item["action"] not in LONG_ACTIONS:
        return {"status": "不适用"}
    if bars is None or bars.empty or benchmark is None or benchmark.empty:
        return {"status": "没有价格数据"}
    entry = entry_session(bars.index, received)
    if entry is None or entry not in benchmark.index:
        return {"status": "等待开盘"}
    sessions = bars.index[bars.index >= entry].intersection(benchmark.index)
    limit = item.get("entry_limit") if use_levels else None
    stop = item.get("stop") if use_levels else None
    target = item.get("target") if use_levels else None

    fill_day = fill_price = None
    at_open = True
    for day in sessions[:LIMIT_VALID_SESSIONS]:
        opened, low = float(bars.at[day, "open"]), float(bars.at[day, "low"])
        if math.isnan(opened):
            return {"status": "缺开盘价"}
        if limit is None or opened <= limit:
            fill_day, fill_price = day, opened
            break
        if not math.isnan(low) and low <= limit:
            fill_day, fill_price, at_open = day, float(limit), False
            break
    if fill_day is None:
        if len(sessions) >= LIMIT_VALID_SESSIONS:
            return {"status": "未成交", "limit": limit,
                    "expired_session": str(sessions[LIMIT_VALID_SESSIONS - 1].date())}
        return {"status": "等待成交", "limit": limit}

    notes = []
    if stop is not None and stop >= fill_price:
        stop = None
        notes.append("止损价不低于成交价，已忽略")
    if target is not None and target <= fill_price:
        target = None
        notes.append("目标价不高于成交价，已忽略")
    horizon = item.get("horizon_days")
    due = fill_day + timedelta(days=horizon) if horizon else None

    held = sessions[sessions >= fill_day]
    exit_day = exit_price = reason = None
    for position, day in enumerate(held):
        ratio = float(bars.at[day, "split"]) if "split" in bars else 1.0
        if position > 0 and ratio != 1.0:
            # Levels were written in pre-split prices; the bars from here on
            # are post-split, so the levels move with them.
            stop = stop / ratio if stop is not None else None
            target = target / ratio if target is not None else None
        opened = float(bars.at[day, "open"])
        high, low = float(bars.at[day, "high"]), float(bars.at[day, "low"])
        through_stop = stop is not None and low <= stop
        through_target = target is not None and high >= target
        if position == 0 and not at_open:
            through_target = False
        if position > 0 and stop is not None and opened <= stop:
            exit_day, exit_price, reason = day, opened, "止损"
        elif position > 0 and target is not None and opened >= target:
            exit_day, exit_price, reason = day, opened, "止盈"
        elif through_stop:
            exit_day, exit_price, reason = day, float(stop), "止损"
        elif through_target:
            exit_day, exit_price, reason = day, float(target), "止盈"
        elif (due is not None and day >= due) or (due is None and default_sessions
                                                  and position + 1 >= default_sessions):
            exit_day, exit_price, reason = day, float(bars.at[day, "close"]), "到期"
        if exit_day is not None:
            break

    end = exit_day if exit_day is not None else held[-1]
    end_price = exit_price if exit_price is not None else float(bars.at[end, "close"])
    own = end_price * _factor(bars, end) / (fill_price * _factor(bars, fill_day)) - 1.0
    market = (float(benchmark.at[end, "adjclose"])
              / _adjusted_open(benchmark, fill_day) - 1.0)
    return {"status": reason or "持有中", "fill_session": str(fill_day.date()),
            "fill_price": fill_price, "filled_at": "开盘" if at_open else "限价",
            "exit_session": str(exit_day.date()) if exit_day is not None else None,
            "exit_price": exit_price, "through": str(end.date()),
            "return": own, "spy_return": market, "excess": own - market, "notes": notes}


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
    if item["action"] in LONG_ACTIONS:
        row["plan"] = simulate_plan(item, received, bars, benchmark)
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


def selected_against_rest(rows, checkpoint):
    """Within each report, chosen long calls minus the names passed over.

    The passed-over names are the report's "watch" items: scored by the
    same reader on the same morning and not chosen, which makes them the
    control nearest to the choice itself. Reports without both kinds do
    not count.
    """
    groups = {}
    for row in rows:
        mark = row["checkpoints"].get(str(checkpoint))
        if not mark:
            continue
        side = ("chosen" if row["action"] in LONG_ACTIONS
                else "rest" if row["action"] == "watch" else None)
        if side:
            groups.setdefault(row["key"], {"chosen": [], "rest": []})[side].append(mark["excess"])
    gaps = [sum(g["chosen"]) / len(g["chosen"]) - sum(g["rest"]) / len(g["rest"])
            for g in groups.values() if g["chosen"] and g["rest"]]
    return _stats(gaps)


def summarize(rows):
    """Hit statistics per checkpoint, the IC, the plan, and the inner control."""
    directional = [r for r in rows if r["direction"]]
    summary = {"items": len(rows), "directional": len(directional),
               "entered": sum(1 for r in rows if r.get("entry_session")),
               "checkpoints": {}, "ic": {}, "selected_vs_rest": {}}
    for count in CHECKPOINTS:
        values = [r["checkpoints"][str(count)]["signed"] for r in directional
                  if str(count) in r["checkpoints"]]
        summary["checkpoints"][str(count)] = _stats(values)
        summary["selected_vs_rest"][str(count)] = selected_against_rest(rows, count)
    for count in IC_CHECKPOINTS:
        summary["ic"][str(count)] = information_coefficient(rows, count)

    plans = [r["plan"] for r in rows if r.get("plan")]
    filled = [p for p in plans if p.get("fill_session")]
    summary["plan"] = {
        "long": len(plans), "filled": len(filled),
        "unfilled": sum(1 for p in plans if p["status"] == "未成交"),
        "closed": _stats([p["excess"] for p in filled if p.get("exit_session")]),
        "all": _stats([p["excess"] for p in filled])}
    return summary


def universe_excess(bars_by_symbol, benchmark, entry, checkpoint):
    """Excess over SPY, entry open to the checkpoint close, for every name.

    Only names that printed an open on the entry session and a close on the
    checkpoint session count - a name that was not trading that day cannot
    be drawn, or the control is handed a zero it did not earn.
    """
    span = benchmark.index[benchmark.index >= entry]
    if len(span) < checkpoint or entry not in benchmark.index:
        return []
    end = span[checkpoint - 1]
    market = float(benchmark.at[end, "adjclose"]) / _adjusted_open(benchmark, entry) - 1.0
    values = []
    for frame in bars_by_symbol.values():
        if frame is None or entry not in frame.index or end not in frame.index:
            continue
        if math.isnan(float(frame.at[entry, "open"])):
            continue
        values.append(float(frame.at[end, "adjclose"]) / _adjusted_open(frame, entry)
                      - 1.0 - market)
    return values


def random_control(rows, bars_by_symbol, benchmark, checkpoint, draws=2000, seed=0):
    """Where the reports' mean signed excess sits among random picks.

    Each directional item is replaced by a name drawn from the universe on
    its own entry date, signed the way the item was, and the mean taken;
    `draws` times. p is the share of random means at or above the observed
    one, counting the observed as one of them.
    """
    import numpy as np
    import pandas as pd

    items = [r for r in rows if r["direction"] and str(checkpoint) in r["checkpoints"]]
    pools, observed, directions, kept = {}, [], [], []
    for row in items:
        entry = row["entry_session"]
        if entry not in pools:
            pools[entry] = np.array(universe_excess(bars_by_symbol, benchmark,
                                                    pd.Timestamp(entry), checkpoint))
        if len(pools[entry]):
            kept.append(row)
            observed.append(row["checkpoints"][str(checkpoint)]["signed"])
            directions.append(row["direction"])
    if not kept:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    sums = np.zeros(draws)
    for row, direction in zip(kept, directions):
        pool = pools[row["entry_session"]]
        sums += direction * pool[rng.integers(len(pool), size=draws)]
    means = sums / len(kept)
    actual = float(np.mean(observed))
    return {"n": len(kept), "observed": actual,
            "median": float(np.median(means)),
            "low": float(np.quantile(means, 0.025)),
            "high": float(np.quantile(means, 0.975)),
            "p": float((np.sum(means >= actual) + 1) / (draws + 1)),
            "pool": int(np.mean([len(pools[r["entry_session"]]) for r in kept]))}
