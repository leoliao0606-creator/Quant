"use strict";
// Every string that comes from data goes in through textContent, never
// innerHTML: pasted reports are free text and may contain markup.

const SVG = "http://www.w3.org/2000/svg";
const SERIES = [
  { key: "sim", name: "模拟账户", n: 1 },
  { key: "spy", name: "SPY", n: 2 },
  { key: "sgov", name: "SGOV", n: 3 },
];
const TASK_INFO = {
  daily: {
    name: "每日研究", checkpoints: [5, 20, 60],
    about: "每天 9:00，2 周到 3 个月的波段。候选池里每一只都要打分，达到 8.5 分的才是买入建议，"
      + "其余写 watch（观察）。成绩记在入场后第 5、20、60 个交易日收盘。",
  },
  radar: {
    name: "机会与风险雷达", checkpoints: [1, 5, 20],
    about: "每小时，只在重大变化时报告。成绩记在入场后第 1、5、20 个交易日收盘，"
      + "看消息出来之后价格是否还在继续往报告说的方向走。",
  },
  short: {
    name: "短线评分", checkpoints: [1, 5],
    about: "每天开盘前，同一组固定股票全部打分，预测未来 5 个交易日相对 SPY 的表现。"
      + "要在 9:30 前粘贴，否则入场顺延一天。成绩记在第 1、5 个交易日收盘。",
  },
};
const ACTION = { buy: "买入", opportunity: "机会", sell: "卖出", avoid: "回避", risk: "风险", watch: "观察" };
const WEEKDAY = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];
const view = { state: null, live: null, signals: null, mode: "chart", task: "daily", panels: {},
  name: null, ledgers: [] };

const $ = (id) => document.getElementById(id);

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  }
  for (const child of children) {
    if (child == null) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

const money = (v) => v == null ? "—"
  : v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const whole = (v) => v == null ? "—" : v.toLocaleString("en-US");
const pct = (v, digits = 2) => v == null ? "—"
  : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v * 100).toFixed(digits)}%`;
const signClass = (v) => v == null || v === 0 ? "" : v > 0 ? "up" : "down";
const clock = (iso) => iso ? iso.slice(5, 16).replace("T", " ") : "—";
const weekday = (isoDate) => WEEKDAY[new Date(`${isoDate}T12:00:00`).getDay()];

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function ledgerLabel(ledger) {
  if (ledger.kind !== "signals") return `${ledger.name}（每周配置）`;
  const info = TASK_INFO[ledger.task];
  return `${ledger.name}（按规则交易「${info ? info.name : ledger.task}」）`;
}

async function loadLedgers() {
  const data = await getJSON("/api/ledgers");
  view.ledgers = data.ledgers;
  const names = data.ledgers.map((l) => l.name);
  let saved = null;
  try { saved = localStorage.getItem("ledger"); } catch (error) { /* private window */ }
  const asked = new URLSearchParams(location.search).get("ledger");
  view.name = [asked, saved, data.default].find((n) => names.includes(n)) || names[0] || null;
  const select = $("ledger");
  select.replaceChildren(...data.ledgers.map((l) => el("option", { value: l.name, text: ledgerLabel(l) })));
  if (view.name) select.value = view.name;
}

async function refresh(kind, ids) {
  const cards = ids.map($).filter(Boolean);
  cards.forEach((c) => c.classList.add("refreshing"));
  const scoped = (kind === "state" || kind === "live") && view.name;
  try {
    view[kind] = await getJSON(`/api/${kind}${scoped ? `?name=${encodeURIComponent(view.name)}` : ""}`);
  } catch (error) {
    console.warn(kind, error);
  } finally {
    cards.forEach((c) => c.classList.remove("refreshing"));
  }
  render();
}

const refreshState = () => refresh("state", ["chart-card", "events-card"]);
const refreshLive = () => refresh("live", ["hero", "positions-card"]);
const refreshSignals = () => refresh("signals", ["signals-card"]);

function render() {
  renderHeader();
  renderHero();
  renderTiles();
  renderChart();
  renderPositions();
  renderEvents();
  renderSignals();
}

function renderHeader() {
  const { state, live } = view;
  if (state && state.exists && state.kind === "signals") {
    const p = state.params;
    const info = TASK_INFO[p.task];
    $("tag").textContent = `按固定规则交易「${info ? info.name : p.task}」的看多建议：每笔净值的 `
      + `${(p.position_size * 100).toFixed(0)}%，最多 ${p.max_positions} 只，`
      + `${p.use_levels ? "按报告给的限价、止损、目标价成交" : "开盘买入、到期卖出"}，`
      + `后来的卖出/回避/风险信号在开盘卖出；起始 ${whole(state.capital)} 美元，每边 ${p.cost_bps} bp`;
    $("booked").textContent = state.last_processed
      ? `账本记到 ${state.last_processed}` : "账本尚未记账";
  } else if (state && state.exists) {
    const parts = Object.entries(state.allocation)
      .map(([s, w]) => `${s} ${(w * 100).toFixed(0)}%`).join(" · ");
    $("tag").textContent = `每周配置加波动率调节：${parts}，起始 ${whole(state.capital)} 美元`;
    $("booked").textContent = state.last_processed
      ? `账本记到 ${state.last_processed}` : "账本尚未记账";
  }
  const open = live && live.market_open;
  $("market").classList.toggle("open", Boolean(open));
  $("market-text").textContent = live ? (open ? "盘中" : "休市") : "读取中";
  $("quote-time").textContent = `报价 ${live ? clock(live.quote_time) : "—"}`;
}

function renderHero() {
  const { state, live } = view;
  if (!state) return;
  if (!state.exists) {
    $("hero-value").textContent = "—";
    $("hero-note").textContent = "还没有建立模拟。先运行 run_sim_daily.sh 或 portfolio_sim.py";
    return;
  }
  const holding = Object.values(state.shares).some((n) => n > 0);
  let equity = state.equity;
  let label = "账户净值";
  let note;
  if (live && live.exists && holding && live.market_open) {
    equity = live.equity;
    label = "账户净值（盘中估算）";
    note = `按 ${clock(live.quote_time)} 的报价估算，不写进账本。收盘后由账本正式记账`;
  } else if (state.booked_session) {
    note = `已记账，截至 ${state.booked_session} 收盘`;
  } else if (state.kind === "signals") {
    note = `尚未开始交易，全部是现金。只交易 ${clock(state.created)} 之后粘贴的报告`;
  } else {
    note = `尚未开始交易，全部是现金。第一次决策在 ${state.next_decision}（${weekday(state.next_decision)}）收盘`;
  }
  $("hero-label").textContent = label;
  $("hero-value").textContent = `$${money(equity)}`;
  const change = equity / state.capital - 1;
  const delta = $("hero-delta");
  delta.textContent = `${pct(change)}（${change >= 0 ? "+" : "−"}$${money(Math.abs(equity - state.capital))}）`;
  delta.className = `delta ${signClass(change)}`;
  $("hero-note").textContent = note;
}

function tile(label, value, hint, tone) {
  return el("div", { class: "tile" },
    el("div", { class: "label", text: label }),
    el("div", { class: `value ${tone || ""}`, text: value }),
    hint ? el("div", { class: "hint", text: hint }) : null);
}

function renderTiles() {
  const { state } = view;
  const box = $("tiles");
  if (!state || !state.exists) { box.replaceChildren(); return; }
  const span = state.first_session ? `${state.first_session} 起` : "尚未开始";
  const cashShare = state.positions
    ? state.booked_cash / state.equity : 1;
  if (state.kind === "signals") {
    box.replaceChildren(
      tile("累计收益", pct(state.total_return), span, signClass(state.total_return)),
      tile("同期 SPY", pct(state.spy_return), "含分红", signClass(state.spy_return)),
      tile("同期 SGOV", pct(state.sgov_return), "要跑赢的基准", signClass(state.sgov_return)),
      tile("最大回撤", pct(state.max_drawdown ?? null), "从最高点的最大跌幅",
        state.max_drawdown < 0 ? "down" : ""),
      tile("持仓", `${state.positions_count}/${state.params.max_positions}`,
        `现金占 ${(cashShare * 100).toFixed(1)}%`),
      tile("记到", state.last_processed ? state.last_processed.slice(5) : "—", "每个交易日收盘后记账"),
    );
    return;
  }
  box.replaceChildren(
    tile("累计收益", pct(state.total_return), span, signClass(state.total_return)),
    tile("同期 SPY", pct(state.spy_return), "含分红", signClass(state.spy_return)),
    tile("同期 SGOV", pct(state.sgov_return), "要跑赢的基准", signClass(state.sgov_return)),
    tile("最大回撤", pct(state.max_drawdown ?? null), "从最高点的最大跌幅",
      state.max_drawdown < 0 ? "down" : ""),
    tile("整体仓位", state.scale == null ? "—" : `${(state.scale * 100).toFixed(1)}%`,
      `现金占 ${(cashShare * 100).toFixed(1)}%`),
    tile("下一次决策", state.next_decision.slice(5), `${weekday(state.next_decision)}收盘后`),
  );
}

function niceStep(range, count) {
  const raw = range / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  return power * (unit <= 1 ? 1 : unit <= 2 ? 2 : unit <= 2.5 ? 2.5 : unit <= 5 ? 5 : 10);
}

function renderChart() {
  const { state } = view;
  const box = $("chart");
  const tableBox = $("chart-table");
  const series = state && state.series;
  box.hidden = view.mode !== "chart";
  tableBox.hidden = view.mode !== "table";
  $("show-chart").setAttribute("aria-pressed", String(view.mode === "chart"));
  $("show-table").setAttribute("aria-pressed", String(view.mode === "table"));
  if (!series || series.dates.length < 2) {
    const message = !(state && state.exists) ? "没有数据"
      : state.kind === "signals"
        ? "还没有记账的交易日。账本从建立之后收盘的那一天开始记账，有建议入场才会买入。"
        : `还没有记账的交易日。第一个是建立模拟之后收盘的那一天，第一次决策在 ${state.next_decision}。`;
    box.replaceChildren(el("div", { class: "empty", text: message }));
    tableBox.replaceChildren(el("div", { class: "empty", text: message }));
    return;
  }
  renderChartTable(series, tableBox);
  if (view.mode !== "chart") return;

  const n = series.dates.length;
  const width = Math.max(box.clientWidth, 300);
  const plotHeight = 260;
  const values = SERIES.flatMap((s) => series[s.key]).filter((v) => v != null);
  let low = Math.min(...values), high = Math.max(...values);
  const pad = Math.max((high - low) * 0.12, 0.25);
  low -= pad; high += pad;
  const step = niceStep(high - low, 5);
  const ticks = [];
  for (let t = Math.ceil(low / step) * step; t <= high + 1e-9; t += step) ticks.push(t);
  const decimals = step < 1 ? (step < 0.5 ? 2 : 1) : 0;

  const ends = SERIES.map((s) => {
    const list = series[s.key];
    for (let i = n - 1; i >= 0; i -= 1) if (list[i] != null) return { ...s, i, v: list[i] };
    return null;
  }).filter(Boolean);
  const left = 48, top = 12, bottomBand = 28;
  const y = (v) => top + (high - v) / (high - low) * plotHeight;
  // End labels only where they do not collide; nudging them apart would
  // detach them from their lines. The legend and tooltip carry the rest,
  // and the margin kept for labels is given back when none are drawn.
  const byHeight = [...ends].sort((a, b) => y(a.v) - y(b.v));
  const labelled = width < 520 ? [] : byHeight.filter((end, k) => {
    const above = k > 0 ? y(end.v) - y(byHeight[k - 1].v) : Infinity;
    const below = k < byHeight.length - 1 ? y(byHeight[k + 1].v) - y(end.v) : Infinity;
    return Math.min(above, below) >= 14;
  });
  const right = labelled.length ? 112 : 16;
  const plotWidth = width - left - right;
  const x = (i) => left + (n === 1 ? plotWidth / 2 : (i * plotWidth) / (n - 1));
  const height = top + plotHeight + bottomBand;

  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img",
    "aria-label": "净值走势：模拟账户、SPY、SGOV，起点等于 100" });
  for (const t of ticks) {
    root.append(svg("line", { class: "grid", x1: left, x2: left + plotWidth, y1: y(t), y2: y(t) }));
    const label = svg("text", { class: "tick", x: left - 8, y: y(t) + 4, "text-anchor": "end" });
    label.textContent = t.toFixed(decimals);
    root.append(label);
  }
  root.append(svg("line", { class: "base", x1: left, x2: left + plotWidth,
    y1: top + plotHeight, y2: top + plotHeight }));
  const labelCount = Math.min(n, Math.max(2, Math.floor(plotWidth / 90)));
  for (let k = 0; k < labelCount; k += 1) {
    const i = Math.round((k * (n - 1)) / Math.max(labelCount - 1, 1));
    const text = svg("text", { class: "tick", x: x(i), y: top + plotHeight + 18,
      "text-anchor": k === 0 ? "start" : k === labelCount - 1 ? "end" : "middle" });
    text.textContent = series.dates[i].slice(5);
    root.append(text);
  }
  for (const s of SERIES) {
    let d = "", pen = false;
    series[s.key].forEach((v, i) => {
      if (v == null) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`;
      pen = true;
    });
    if (d) root.append(svg("path", { class: `line l${s.n}`, d }));
  }
  const cross = svg("line", { class: "cross", y1: top, y2: top + plotHeight, visibility: "hidden" });
  root.append(cross);
  for (const end of ends) {
    root.append(svg("circle", { class: `dot d${end.n}`, cx: x(end.i), cy: y(end.v), r: 4 }));
  }
  for (const end of labelled) {
    const text = svg("text", { class: "end-label", x: x(end.i) + 10, y: y(end.v) + 4 });
    text.textContent = `${end.name} ${end.v.toFixed(decimals + 1)}`;
    root.append(text);
  }

  const tip = el("div", { class: "tooltip", hidden: "" });
  const hit = svg("rect", { class: "hit", x: left, y: top, width: plotWidth, height: plotHeight,
    tabindex: "0", "aria-label": "按左右方向键查看每个交易日的数值" });
  let current = n - 1;
  const show = (i) => {
    current = Math.max(0, Math.min(n - 1, i));
    cross.setAttribute("x1", x(current));
    cross.setAttribute("x2", x(current));
    cross.setAttribute("visibility", "visible");
    const rows = SERIES.map((s) => {
      const v = series[s.key][current];
      return el("div", { class: "row" },
        el("i", { class: `key k${s.n}` }),
        el("strong", { text: v == null ? "—" : v.toFixed(decimals + 1) }),
        el("span", { text: s.name }));
    });
    tip.replaceChildren(el("div", { class: "when",
      text: `${series.dates[current]}（${weekday(series.dates[current])}）` }), ...rows);
    tip.hidden = false;
    const scale = box.clientWidth / width;
    const px = x(current) * scale;
    tip.style.left = `${px + 170 > box.clientWidth ? px - 170 : px + 12}px`;
    tip.style.top = `${top}px`;
  };
  const hide = () => { tip.hidden = true; cross.setAttribute("visibility", "hidden"); };
  hit.addEventListener("pointermove", (event) => {
    const bounds = root.getBoundingClientRect();
    const px = (event.clientX - bounds.left) * (width / bounds.width);
    show(Math.round(((px - left) / plotWidth) * (n - 1)));
  });
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("focus", () => show(current));
  hit.addEventListener("blur", hide);
  hit.addEventListener("keydown", (event) => {
    if (event.key === "ArrowLeft") { show(current - 1); event.preventDefault(); }
    if (event.key === "ArrowRight") { show(current + 1); event.preventDefault(); }
  });
  root.append(hit);
  box.replaceChildren(root, tip);
}

function renderChartTable(series, box) {
  const head = el("tr", {}, el("th", { text: "日期" }),
    ...SERIES.map((s) => el("th", { class: "num", text: s.name })));
  const rows = series.dates.map((d, i) => el("tr", {}, el("td", { text: d }),
    ...SERIES.map((s) => el("td", { class: "num",
      text: series[s.key][i] == null ? "—" : series[s.key][i].toFixed(2) }))));
  box.replaceChildren(el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows.reverse())));
}

function renderPositions() {
  const { state, live } = view;
  const box = $("positions");
  if (!state || !state.exists) { box.replaceChildren(); return; }
  const liveRows = live && live.exists ? live.rows : [];
  const bySymbol = Object.fromEntries(liveRows.map((r) => [r.symbol, r]));
  const equity = live && live.exists ? live.equity : state.equity;
  const head = el("tr", {}, ...["标的", "股数", "现价", "最近一日", "市值", "占比"]
    .map((t, i) => el("th", { class: i ? "num" : "", text: t })));
  const rows = Object.keys(state.shares).map((symbol) => {
    const shares = state.shares[symbol];
    const row = bySymbol[symbol] || {};
    const booked = state.positions && state.positions[symbol];
    const price = row.price ?? (booked ? booked.close : null);
    const value = price == null ? null : shares * price;
    return el("tr", {},
      el("td", { text: symbol }),
      el("td", { class: "num", text: whole(shares) }),
      el("td", { class: "num", text: price == null ? "—" : price.toFixed(2) }),
      el("td", { class: `num ${signClass(row.day_change)}`, text: pct(row.day_change) }),
      el("td", { class: "num", text: money(value) }),
      el("td", { class: "num", text: value == null ? "—" : `${(value / equity * 100).toFixed(1)}%` }));
  });
  rows.push(el("tr", {}, el("td", { text: "现金" }), el("td"), el("td"), el("td"),
    el("td", { class: "num", text: money(state.cash) }),
    el("td", { class: "num", text: `${(state.cash / equity * 100).toFixed(1)}%` })));
  box.replaceChildren(el("table", {}, el("thead", {}, head), el("tbody", {}, ...rows)));
  $("positions-sub").textContent = live && live.exists
    ? `现价来自 Yahoo，${clock(live.quote_time)}；现金按账本` : "现价暂时取不到，显示账本里的收盘价";
}

function describe(event) {
  switch (event.type) {
    case "decision":
      return `周五决策：整体仓位 ${(event.scale * 100).toFixed(1)}%`
        + `${event.rebalancing ? "，再平衡到目标权重" : ""}（数据截至 ${event.data_through.slice(5)}）`;
    case "fill":
      return `${event.side === "BUY" ? "买入" : "卖出"} ${event.symbol} ${whole(event.shares)} 股 @ `
        + `${event.price.toFixed(2)}，成本 ${event.cost.toFixed(2)}`
        + (event.reason ? `（${event.reason}${event.return == null ? "" : `，${pct(event.return)}`}）` : "");
    case "skip":
      return `跳过 ${event.symbol}：${event.reason}`;
    case "dividend":
      return `${event.symbol} 分红 ${money(event.amount)}（${whole(event.shares)} 股 × ${event.per_share}）`;
    case "split":
      return `${event.symbol} 拆股 ${event.ratio} 比 1，${whole(event.shares_before)} → ${whole(event.shares_after)} 股`;
    default:
      return event.type;
  }
}

function renderEvents() {
  const { state } = view;
  const box = $("events");
  if (!state || !state.exists) { box.replaceChildren(); return; }
  if (!state.events.length) {
    box.replaceChildren(el("li", {}, el("span", { class: "when", text: "—" }),
      el("span", { text: `还没有事件。第一次决策在 ${state.next_decision}（${weekday(state.next_decision)}）` })));
    return;
  }
  box.replaceChildren(...state.events.map((e) => el("li", {},
    el("span", { class: "when", text: e.session }), el("span", { text: describe(e) }))));
}

function buildTaskPanels() {
  try { view.task = localStorage.getItem("task") || view.task; } catch (error) { /* private window */ }
  view.task = new URLSearchParams(location.search).get("task") || view.task;
  if (!TASK_INFO[view.task]) view.task = "daily";
  const tabs = $("task-tabs");
  const panels = $("task-panels");
  for (const [task, info] of Object.entries(TASK_INFO)) {
    const count = el("span", { class: "count" });
    const tab = el("button", { type: "button", role: "tab", id: `tab-${task}`,
      "aria-controls": `panel-${task}` }, info.name, count);
    tab.addEventListener("click", () => selectTask(task));
    const textarea = el("textarea", { id: `paste-${task}`,
      placeholder: `整段粘贴「${info.name}」的报告，包括末尾的 \`\`\`json ... \`\`\` 代码块` });
    const button = el("button", { type: "button", class: "primary", text: "导入" });
    const message = el("span", { class: "message", role: "status" });
    const summary = el("p", { class: "summary message" });
    const table = el("div", { class: "table-wrap" });
    const panel = el("div", { role: "tabpanel", id: `panel-${task}`, "aria-labelledby": `tab-${task}` },
      el("p", { class: "about", text: info.about }),
      el("label", { class: "sub", for: `paste-${task}`, text: "报告原文" }),
      textarea, el("div", { class: "actions" }, button, message), summary, table);
    button.addEventListener("click", () => importReport(task));
    tabs.append(tab);
    panels.append(panel);
    view.panels[task] = { tab, count, panel, textarea, button, message, summary, table };
  }
  selectTask(view.task);
}

function selectTask(task) {
  view.task = task;
  try { localStorage.setItem("task", task); } catch (error) { /* private window */ }
  for (const [name, parts] of Object.entries(view.panels)) {
    parts.tab.setAttribute("aria-selected", String(name === task));
    parts.panel.hidden = name !== task;
  }
}

function summaryLines(task, data) {
  const s = data.summary;
  if (!data.rows.length) return "还没有导入这个任务的报告。";
  const lines = [`${data.reports} 份报告，${s.items} 条记录，已入场 ${s.entered} 条，`
    + `其中有方向（不含"观察"）的 ${s.directional} 条。`];
  let enough = false;
  for (const k of TASK_INFO[task].checkpoints) {
    const c = s.checkpoints[k];
    if (!c || !c.n) { lines.push(`第 ${k} 个交易日：还没有到这一天的记录。`); continue; }
    enough = enough || c.n >= 30;
    lines.push(`第 ${k} 个交易日：方向对 ${c.hits}/${c.n}，平均方向超额 ${pct(c.mean)}`
      + `${c.t == null ? "" : `，t = ${c.t.toFixed(2)}`}。`);
  }
  for (const k of ["5", "20"]) {
    const ic = s.ic[k];
    if (!ic || !ic.reports) continue;
    enough = enough || ic.reports >= 20;
    lines.push(`分数与第 ${k} 日超额的秩相关：平均 ${ic.mean.toFixed(3)}（${ic.reports} 份报告`
      + `${ic.t == null ? "" : `，t = ${ic.t.toFixed(2)}`}）。`);
  }
  for (const k of TASK_INFO[task].checkpoints) {
    const gap = s.selected_vs_rest && s.selected_vs_rest[k];
    if (!gap || !gap.n) continue;
    lines.push(`第 ${k} 日，被选中的减去同一份报告里落选的：平均 ${pct(gap.mean)}（${gap.n} 份报告`
      + `${gap.t == null ? "" : `，t = ${gap.t.toFixed(2)}`}）。`);
  }
  const plan = s.plan;
  if (plan && plan.long) {
    lines.push(`按计划：看多 ${plan.long} 条，成交 ${plan.filled} 条，未成交 ${plan.unfilled} 条；`
      + `已平仓 ${plan.closed.n} 条${plan.closed.n ? `，平均超额 ${pct(plan.closed.mean)}` : ""}；`
      + `连同持有中的共 ${plan.all.n} 条${plan.all.n ? `，平均超额 ${pct(plan.all.mean)}` : ""}。`);
  }
  let controlled = false;
  for (const k of TASK_INFO[task].checkpoints) {
    const c = data.control && data.control[k];
    if (!c || !c.n) continue;
    controlled = true;
    lines.push(`第 ${k} 日随机对照：报告平均方向超额 ${pct(c.observed)}；同一天从约 ${c.pool} 只股票里`
      + `随机抽同样数量，中位数 ${pct(c.median)}，95% 范围 ${pct(c.low)} 到 ${pct(c.high)}，`
      + `p = ${c.p.toFixed(3)}（越小越不像运气）。`);
  }
  const computed = view.signals && view.signals.control && view.signals.control.computed;
  if (controlled && computed) lines.push(`随机对照每天收盘后计算一次，上次 ${clock(computed)}。`);
  if (!enough) lines.push("样本太少，还不能下结论。");
  return lines.join("\n");
}

function planText(row) {
  const p = row.plan;
  if (!p) return "—";
  if (!p.fill_session) return p.status;
  if (p.exit_session) return `${p.status} ${p.exit_session.slice(5)}`;
  return `持有中（${p.filled_at}成交）`;
}

function renderSignals() {
  const data = view.signals;
  if (!data || !data.tasks) return;
  for (const [task, info] of Object.entries(TASK_INFO)) {
    const parts = view.panels[task];
    const mine = data.tasks[task] || { rows: [], reports: 0, summary: {} };
    parts.count.textContent = mine.reports ? String(mine.reports) : "";
    parts.summary.textContent = summaryLines(task, mine);
    if (!mine.rows.length) { parts.table.replaceChildren(); continue; }
    const radar = task === "radar";
    const planned = task !== "short";
    const titles = ["收到", "代码", "方向", "评分", ...(radar ? ["重要性", "事件"] : []), "入场",
      ...info.checkpoints.map((k) => `${k}日超额`), "至今超额", "判断",
      ...(planned ? ["按计划", "计划超额"] : []), "价格偏离"];
    const numeric = new Set(titles.filter((t) => t.includes("超额") || t === "价格偏离"));
    const head = el("tr", {}, ...titles.map((t) => el("th", { class: numeric.has(t) ? "num" : "", text: t })));
    const body = mine.rows.map((r) => {
      const verdict = r.signed_excess == null ? "—" : r.signed_excess > 0 ? "方向对" : "方向错";
      const gap = r.price_gap;
      const flagged = gap != null && Math.abs(gap) > 0.02;
      return el("tr", {},
        el("td", { text: clock(r.received) }),
        el("td", { text: r.ticker }),
        el("td", { text: ACTION[r.action] || r.action }),
        el("td", { text: r.score == null ? "—" : String(r.score) }),
        ...(radar ? [el("td", { text: r.materiality == null ? "—" : `${r.materiality}/5` }),
          el("td", { text: (r.event && r.event.type) || "—" })] : []),
        el("td", { text: r.entry_session
          ? `${r.entry_session.slice(5)} @ ${r.entry_price.toFixed(2)}${r.status === "已到期" ? "（已到期）" : ""}`
          : r.status }),
        ...info.checkpoints.map((k) => {
          const mark = r.checkpoints[k];
          return el("td", { class: `num ${signClass(mark && mark.excess)}`, text: mark ? pct(mark.excess) : "—" });
        }),
        el("td", { class: `num ${signClass(r.excess)}`, text: pct(r.excess) }),
        el("td", { class: r.signed_excess == null ? "" : r.signed_excess > 0 ? "verdict-yes" : "verdict-no",
          text: verdict }),
        ...(planned ? [el("td", { text: planText(r) }),
          el("td", { class: `num ${signClass(r.plan && r.plan.excess)}`,
            text: r.plan && r.plan.excess != null ? pct(r.plan.excess) : "—" })] : []),
        el("td", { class: `num ${flagged ? "flag down" : ""}`, text: gap == null ? "—" : pct(gap, 1) }));
    });
    parts.table.replaceChildren(el("table", { class: "dense" }, el("thead", {}, head), el("tbody", {}, ...body)));
  }
}

async function importReport(task) {
  const { textarea, button, message } = view.panels[task];
  const text = textarea.value;
  if (!text.trim()) { message.textContent = "先粘贴报告原文"; return; }
  button.disabled = true;
  message.textContent = "导入中…";
  try {
    const response = await fetch("/api/signals", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, task }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || response.statusText);
    const lines = [`新增 ${result.added} 份报告、${result.items} 条记录`
      + (result.duplicates ? `，${result.duplicates} 份之前导入过，已跳过` : "")];
    if (result.errors.length) lines.push(...result.errors);
    message.textContent = lines.join("\n");
    if (result.added) textarea.value = "";
    await refreshSignals();
  } catch (error) {
    message.textContent = `导入失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
}

$("show-chart").addEventListener("click", () => { view.mode = "chart"; renderChart(); });
$("show-table").addEventListener("click", () => { view.mode = "table"; renderChart(); });
$("ledger").addEventListener("change", (event) => {
  view.name = event.target.value;
  try { localStorage.setItem("ledger", view.name); } catch (error) { /* private window */ }
  view.state = null;
  view.live = null;
  refreshState();
  refreshLive();
});
buildTaskPanels();
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderChart, 150); });
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) { refreshState(); refreshLive(); refreshSignals(); }
});

loadLedgers().catch((error) => console.warn("ledgers", error)).finally(() => {
  refreshState();
  refreshLive();
});
refreshSignals();
setInterval(refreshLive, 60 * 1000);
setInterval(refreshState, 5 * 60 * 1000);
setInterval(refreshSignals, 10 * 60 * 1000);
