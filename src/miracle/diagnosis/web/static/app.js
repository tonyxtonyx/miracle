"use strict";
// Read-only viewer for recorded diagnosis runs. No dependencies. All text is inserted with textContent (excerpts may contain markup).
const $ = (s, el = document) => el.querySelector(s);
const sid = (id) => (id || "").split(":").pop();
function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") e.className = v; else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat(Infinity)) if (k != null && k !== false) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}
const svgNS = "http://www.w3.org/2000/svg";
function s(tag, attrs, ...kids) {
  const e = document.createElementNS(svgNS, tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null) e.setAttribute(k, v);
  for (const k of kids.flat(Infinity)) if (k != null) e.append(k.nodeType ? k : document.createTextNode(String(k)));
  return e;
}

const EVIDENCE_TYPES = ["issue_statement", "code_excerpt", "test_result", "runtime_observation", "configuration", "documentation", "intervention_result"];
const EV_COLOR = {issue_statement: "#7b8790", code_excerpt: "#2f6fdd", test_result: "#d9822b", runtime_observation: "#16a085", configuration: "#8e6f3e", documentation: "#a0a46a", intervention_result: "#c2185b"};
const REL_COLOR = {supports: "var(--supports)", contradicts: "var(--contradicts)", explains: "var(--explains)", refines: "var(--refines)", alternative_to: "var(--alt)", depends_on: "var(--depends)", tests: "var(--tests)"};
const STATUS_COLOR = {proposed: "--proposed", investigating: "--investigating", supported: "--supported", weakened: "--weakened", refuted: "--refuted", inconclusive: "--inconclusive", selected: "--selected"};

const S = {runs: [], run: null, instances: [], instance: null, events: [], total: 0, step: null, g: null, art: null, calls: {}, sel: null, tab: "detail",
           filters: {refuted: true, evidence: true, probes: false, types: new Set(EVIDENCE_TYPES.filter((t) => t !== "issue_statement"))}, timer: null, play: null};

async function api(path) {
  const r = await fetch(path, {cache: "no-store"});
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function fail(e) { const el = $("#error"); el.hidden = false; el.textContent = String(e.message || e); }
function clearFail() { $("#error").hidden = true; }
const q = () => `run=${encodeURIComponent(S.run)}&instance=${encodeURIComponent(S.instance)}`;

// ---------- loading ----------
async function init() {
  try {
    const cfg = await api("/api/config");
    S.runs = await api("/api/runs");
    const sel = $("#run");
    sel.replaceChildren(...S.runs.map((r) => h("option", {value: r.name}, `${r.name} (${r.instances})`)));
    S.run = cfg.run && S.runs.some((r) => r.name === cfg.run) ? cfg.run : (S.runs[0] || {}).name;
    if (S.run) sel.value = S.run;
    sel.onchange = () => { S.run = sel.value; S.instance = null; loadRun(); };
    $("#instance").onchange = () => { S.instance = $("#instance").value; loadInstance(); };
    document.querySelectorAll("#tabs button").forEach((b) => b.onclick = () => { S.tab = b.dataset.tab; renderPanel(); });
    for (const id of ["refuted", "evidence", "probes"]) $("#f-" + id).onchange = (e) => { S.filters[id] = e.target.checked; renderGraph(); };
    $("#f-types").replaceChildren(...EVIDENCE_TYPES.map((t) => h("label", {title: t}, h("input", {type: "checkbox", checked: S.filters.types.has(t), onchange: (e) => { e.target.checked ? S.filters.types.add(t) : S.filters.types.delete(t); renderGraph(); }}), " ", t.replace("_", " "))));
    $("#legend").replaceChildren(...Object.keys(STATUS_COLOR).map((k) => h("span", {}, h("i", {style: `background:var(${STATUS_COLOR[k]})`}), k)),
      ...Object.keys(REL_COLOR).map((k) => h("span", {}, h("i", {style: `background:${REL_COLOR[k]}`}), k)), h("span", {}, "dashed evidence border = observed after a temporary change"));
    $("#slider").oninput = (e) => { stopPlay(); setStep(+e.target.value); };
    $("#latest").onclick = () => { stopPlay(); setStep(null); };
    $("#play").onclick = togglePlay;
    if (S.run) await loadRun(cfg.instance);
    else fail(new Error("no recorded diagnosis runs found"));
  } catch (e) { fail(e); }
}

async function loadRun(prefer) {
  S.instances = await api(`/api/instances?run=${encodeURIComponent(S.run)}`);
  const sel = $("#instance");
  sel.replaceChildren(...S.instances.map((i) => h("option", {value: i.id}, `${i.id}  ·  ${i.official ? (i.official.resolved ? "resolved" : i.official.status) : "no grade"}${i.blind_mean != null ? "  ·  process " + i.blind_mean : ""}`)));
  S.instance = prefer && S.instances.some((i) => i.id === prefer) ? prefer : (S.instances[0] || {}).id;
  if (S.instance) { sel.value = S.instance; await loadInstance(); }
}

async function loadInstance() {
  clearFail(); stopPlay();
  S.step = null; S.sel = null;
  try {
    [S.events, S.art, S.calls] = await Promise.all([api(`/api/events?${q()}`), api(`/api/artifacts?${q()}`),
      api(`/api/tool_calls?${q()}`).then((rows) => Object.fromEntries(rows.map((r) => [r.id, r])))]);
    S.total = S.events.length;
    await loadGraph();
    startPolling();
  } catch (e) { fail(e); }
}

async function loadGraph() {
  const r = await api(`/api/graph?${q()}${S.step == null ? "" : "&upto=" + S.step}`);
  S.g = r.graph; S.total = r.total_events;
  const sl = $("#slider"); sl.max = S.total; sl.value = S.step == null ? S.total : S.step;
  renderAll();
}

function setStep(n) { S.step = n; loadGraph().catch(fail); }

function startPolling() {
  clearInterval(S.timer);
  S.timer = setInterval(async () => {
    try {
      const m = await api(`/api/meta?${q()}`);
      $("#live").textContent = m.done ? "run complete" : "run in progress (polling)";
      if (m.events !== S.events.length) {
        S.events = await api(`/api/events?${q()}`);
        S.art = await api(`/api/artifacts?${q()}`);
        S.calls = Object.fromEntries((await api(`/api/tool_calls?${q()}`)).map((r) => [r.id, r]));
        if (S.step == null) await loadGraph(); else renderTimeline();
      }
      if (m.done && m.events === S.events.length) clearInterval(S.timer);
    } catch (e) { /* transient while files are being written */ }
  }, 3000);
}

function togglePlay() {
  if (S.play) return stopPlay();
  let n = 0; S.step = 0; $("#play").textContent = "⏸";
  const tick = () => { n += 1; if (n > S.total) return stopPlay(); S.step = n; loadGraph().catch(fail); };
  tick(); S.play = setInterval(tick, 700);
}
function stopPlay() { clearInterval(S.play); S.play = null; $("#play").textContent = "▶"; }

// ---------- rendering ----------
function renderAll() { renderHeader(); renderGraph(); renderPanel(); renderTimeline(); }

function renderHeader() {
  const g = S.g, b = g.budget || {};
  $("#phase").textContent = "phase: " + g.phase;
  $("#budget").textContent = [b.iteration ? `iteration ${b.iteration}/${b.max_iterations}` : "", b.llm_calls != null ? `${b.llm_calls} LLM calls` : "", b.tool_calls != null ? `${b.tool_calls} tool calls` : "",
    b.cost_usd != null ? `$${b.cost_usd}` : "", b.wall_s != null ? `${b.wall_s}s` : "", `${Object.keys(g.hypotheses).length} hypotheses · ${Object.keys(g.evidence).length} evidence · ${Object.keys(g.probes).length} probes`].filter(Boolean).join(" · ");
  const o = S.art.official;
  const el = $("#official"); el.textContent = o ? `official: ${o.resolved ? "resolved" : o.status}` : "official: not graded"; el.style.borderColor = o ? (o.resolved ? "var(--supported)" : "var(--refuted)") : "";
  $("#stepinfo").textContent = S.step == null ? `latest (event ${S.total})` : `event ${S.step} of ${S.total}`;
}

function depthOf(g, id, memo = {}) {
  if (id in memo) return memo[id];
  const ps = (g.hypotheses[id] || {parent_ids: []}).parent_ids.filter((p) => g.hypotheses[p]);
  return memo[id] = ps.length ? 1 + Math.max(...ps.map((p) => depthOf(g, p, memo))) : 0;
}

function renderGraph() {
  const g = S.g, svg = $("#graph");
  const W = 236, H = 76, GX = 28, GY = 70;
  const hyps = Object.values(g.hypotheses).filter((x) => S.filters.refuted || !(x.status === "refuted" || x.merged_into));
  const levels = [];
  const memo = {};
  hyps.forEach((x) => (levels[depthOf(g, x.id, memo)] = levels[depthOf(g, x.id, memo)] || []).push(x));
  const rowW = (l) => l.length * (W + GX) - GX;
  const maxW = Math.max(W, ...levels.filter(Boolean).map(rowW));
  const pos = {};
  levels.forEach((row, d) => (row || []).sort((a, b) => a.created_at_step - b.created_at_step).forEach((x, i) => { pos[x.id] = {x: 24 + (maxW - rowW(row)) / 2 + i * (W + GX), y: 18 + d * (H + GY), w: W, h: H, kind: "h"}; }));
  const bandY = 18 + levels.length * (H + GY) + 6;
  // evidence band
  const evs = S.filters.evidence ? Object.values(g.evidence).filter((e) => S.filters.types.has(e.type)) : [];
  const relOf = (id) => Object.values(g.relations).filter((r) => r.from_id === id && pos[r.to_id]);
  const want = evs.map((e, i) => { const rs = relOf(e.id); return {e, x: rs.length ? rs.reduce((a, r) => a + pos[r.to_id].x + W / 2, 0) / rs.length - 30 : i * 70}; }).sort((a, b) => a.x - b.x);
  let cursor = 12;
  const EW = 64, EH = 28;
  want.forEach((w) => { const x = Math.max(w.x, cursor); pos[w.e.id] = {x, y: bandY, w: EW, h: EH, kind: "e"}; cursor = x + EW + 8; });
  // probes band
  const prs = S.filters.probes ? Object.values(g.probes) : [];
  prs.forEach((p, i) => { pos[p.id] = {x: 12 + i * 70, y: bandY + 70, w: EW, h: EH, kind: "p"}; });
  const width = Math.max(maxW + 48, cursor + 24, prs.length * 70 + 24), height = bandY + (prs.length ? 130 : 70);
  svg.setAttribute("width", width); svg.setAttribute("height", height); svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.replaceChildren();
  const defs = s("defs");
  for (const [k, c] of Object.entries(REL_COLOR)) defs.append(s("marker", {id: "m-" + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse"}, s("path", {d: "M0,0 L10,5 L0,10 z", fill: c})));
  svg.append(defs);
  const center = (p) => ({x: p.x + p.w / 2, y: p.y + p.h / 2});
  for (const r of Object.values(g.relations)) {
    const a = pos[r.from_id], b = pos[r.to_id];
    if (!a || !b) continue;
    const ca = center(a), cb = center(b);
    let d;
    if (r.type === "alternative_to") { const mid = (ca.x + cb.x) / 2, lift = Math.min(60, Math.abs(ca.x - cb.x) / 3 + 18); d = `M${ca.x},${a.y} C${ca.x},${a.y - lift} ${cb.x},${b.y - lift} ${cb.x},${b.y}`; }
    else if (a.kind === "h" && b.kind === "h") { const parentBelow = a.y > b.y; d = parentBelow ? `M${ca.x},${a.y} C${ca.x},${a.y - 30} ${cb.x},${b.y + b.h + 30} ${cb.x},${b.y + b.h}` : `M${ca.x},${a.y + a.h} C${ca.x},${a.y + a.h + 30} ${cb.x},${b.y - 30} ${cb.x},${b.y}`; }
    else { d = `M${ca.x},${a.y} C${ca.x},${a.y - 50} ${cb.x},${b.y + b.h + 50} ${cb.x},${b.y + b.h}`; }
    const edge = s("path", {d, fill: "none", stroke: REL_COLOR[r.type], "stroke-width": 1 + (r.strength || 2) * 0.7, "stroke-dasharray": r.type === "alternative_to" ? "6 4" : r.type === "depends_on" ? "2 3" : null, "marker-end": `url(#m-${r.type})`,
      opacity: S.sel && S.sel.id !== r.from_id && S.sel.id !== r.to_id ? 0.28 : 0.9}, s("title", {}, `${sid(r.from_id)} ${r.type} ${sid(r.to_id)} (strength ${r.strength}): ${r.rationale}`));
    svg.append(edge);
  }
  for (const x of hyps) {
    const p = pos[x.id];
    const g1 = s("g", {class: `node st-${x.status}${x.status === "refuted" || x.merged_into ? " refuted" : ""}${S.sel && S.sel.id === x.id ? " sel" : ""}`, transform: `translate(${p.x},${p.y})`, tabindex: 0}, s("title", {}, `${sid(x.id)} [${x.status}] ${x.claim}`));
    g1.append(s("rect", {width: p.w, height: p.h, rx: 8}), s("text", {x: 10, y: 17, "font-weight": 600}, `${sid(x.id)} · ${x.type.replace("_", " ")}${x.merged_into ? " (merged)" : ""}`),
      s("text", {x: p.w - 10, y: 17, "text-anchor": "end", fill: `var(${STATUS_COLOR[x.status]})`, "font-weight": 600}, x.status + (x.support_level !== "none" ? "/" + x.support_level : "")));
    wrap(x.claim, 36, 3).forEach((ln, i) => g1.append(s("text", {class: "claim", x: 10, y: 34 + i * 14}, ln)));
    g1.onclick = () => select("hypothesis", x.id);
    svg.append(g1);
  }
  for (const e of evs) {
    const p = pos[e.id], post = e.observation_state !== "original";
    const g1 = s("g", {class: `node ev-node${post ? " post" : ""}${S.sel && S.sel.id === e.id ? " sel" : ""}`, transform: `translate(${p.x},${p.y})`}, s("title", {}, `${sid(e.id)} [${e.type}${post ? ", after temporary change" : ""}] ${e.summary}`));
    g1.append(s("rect", {width: p.w, height: p.h, rx: 6, stroke: EV_COLOR[e.type]}), s("text", {x: p.w / 2, y: 18, "text-anchor": "middle"}, sid(e.id)));
    g1.onclick = () => select("evidence", e.id);
    svg.append(g1);
  }
  for (const pr of prs) {
    const p = pos[pr.id];
    const g1 = s("g", {class: `node ev-node${S.sel && S.sel.id === pr.id ? " sel" : ""}`, transform: `translate(${p.x},${p.y})`}, s("title", {}, `${sid(pr.id)} ${pr.purpose}: ${pr.planned_action}`));
    g1.append(s("rect", {width: p.w, height: p.h, rx: 14, stroke: "var(--tests)"}), s("text", {x: p.w / 2, y: 18, "text-anchor": "middle"}, sid(pr.id) + " " + pr.status.slice(0, 4)));
    g1.onclick = () => select("probe", pr.id);
    svg.append(g1);
  }
  if (!hyps.length) svg.append(s("text", {x: 24, y: 40}, "No hypotheses at this step yet."));
}

function wrap(text, width, maxLines) {
  const words = (text || "").split(/\s+/), lines = [""];
  for (const w of words) { if ((lines[lines.length - 1] + " " + w).trim().length > width) lines.push(w); else lines[lines.length - 1] = (lines[lines.length - 1] + " " + w).trim(); }
  if (lines.length > maxLines) { lines.length = maxLines; lines[maxLines - 1] = lines[maxLines - 1].slice(0, width - 1) + "…"; }
  return lines;
}

function select(kind, id) { S.sel = {kind, id}; S.tab = "detail"; renderGraph(); renderPanel(); }
const ref = (kind, id) => h("a", {class: "ref", onclick: () => select(kind, id)}, sid(id));
const kindOf = (id) => ({H: "hypothesis", E: "evidence", P: "probe", B: "branch"}[sid(id).match(/^[A-Z]+/)[0]] || "hypothesis");
const refAny = (id) => ["H", "E", "P"].includes(sid(id).match(/^[A-Z]+/)[0]) ? ref(kindOf(id), id) : h("span", {}, sid(id));
const pill = (st) => h("span", {class: "pill " + st}, st);
const list = (items) => items.length ? h("ul", {}, items.map((x) => h("li", {}, x))) : h("span", {class: "muted"}, "none");

function renderPanel() {
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === S.tab));
  const p = $("#panel");
  p.replaceChildren(({detail: panelDetail, diagnosis: panelDiagnosis, judges: panelJudges, controller: panelController})[S.tab]());
}

function panelDetail() {
  const g = S.g, sel = S.sel;
  if (!sel) return h("div", {}, h("h2", {}, "Select a node"), h("p", {class: "muted"}, "Click a hypothesis or an evidence item. Use the slider below to replay how the graph evolved."),
    S.art.repo_map ? h("details", {}, h("summary", {}, "Repository map shown to the controller"), h("pre", {}, S.art.repo_map)) : null);
  if (sel.kind === "hypothesis") return hypDetail(g.hypotheses[sel.id]);
  if (sel.kind === "evidence") return evDetail(g.evidence[sel.id]);
  if (sel.kind === "probe") return probeDetail(g.probes[sel.id]);
  return h("div", {}, "unknown selection");
}

function hypDetail(x) {
  if (!x) return h("p", {class: "muted"}, "This hypothesis does not exist yet at the selected step.");
  const g = S.g, rels = Object.values(g.relations);
  const ev = rels.filter((r) => r.to_id === x.id && g.evidence[r.from_id]);
  return h("div", {}, h("h2", {}, sid(x.id), " ", pill(x.status), " ", h("span", {class: "muted"}, x.support_level !== "none" ? "support: " + x.support_level : "")),
    h("p", {}, x.claim), h("div", {class: "kv"}, h("b", {}, "type"), x.type.replace("_", " "), h("b", {}, "created"), `event ${x.created_at_step} by ${x.created_by}`, h("b", {}, "updated"), `event ${x.last_updated_at_step}`,
      h("b", {}, "refines"), x.parent_ids.length ? x.parent_ids.map((p) => [ref("hypothesis", p), " "]) : "—", x.merged_into ? [h("b", {}, "merged into"), ref("hypothesis", x.merged_into)] : null,
      h("b", {}, "alternative to"), rels.filter((r) => r.type === "alternative_to" && (r.from_id === x.id || r.to_id === x.id)).map((r) => [ref("hypothesis", r.from_id === x.id ? r.to_id : r.from_id), " "])),
    h("h3", {}, "Predictions"), list(x.predictions), h("h3", {}, "Falsification criteria"),
    h("details", {open: true}, h("summary", {}, `${x.falsification_criteria.length} criteria`), list(x.falsification_criteria.map((c) => c.text))),
    h("h3", {}, "Unresolved questions"), list(x.unresolved_questions),
    h("h3", {}, "Evidence"), ev.length ? h("table", {}, ev.map((r) => { const e = g.evidence[r.from_id]; return h("tr", {}, h("td", {}, ref("evidence", e.id)), h("td", {style: `color:${REL_COLOR[r.type]};font-weight:600`}, r.type, ` (${r.strength})`), h("td", {}, e.summary), h("td", {class: "muted"}, e.observation_state !== "original" ? "post-change" : "")); })) : h("span", {class: "muted"}, "none"),
    h("h3", {}, "Status history"), x.history.length ? h("table", {}, h("tr", {}, h("th", {}, "event"), h("th", {}, "change"), h("th", {}, "cause")),
      x.history.map((hi) => h("tr", {}, h("td", {}, hi.seq), h("td", {}, `${hi.old} → ${hi.new}`), h("td", {}, (hi.cause.evidence_ids || []).map((e) => [ref("evidence", e), " "]), hi.cause.decision ? h("div", {class: "muted"}, hi.cause.decision) : null)))) : h("span", {class: "muted"}, "no changes yet"),
    h("h3", {}, "Probes"), list(Object.values(g.probes).filter((p) => p.target_hypothesis_ids.includes(x.id)).map((p) => [ref("probe", p.id), ` ${p.purpose} [${p.status}] `, p.planned_action])));
}

function evDetail(e) {
  if (!e) return h("p", {class: "muted"}, "This evidence does not exist yet at the selected step.");
  const g = S.g, call = e.source_tool_call_id ? S.calls[e.source_tool_call_id] : null, post = e.observation_state !== "original";
  const rels = Object.values(g.relations).filter((r) => r.from_id === e.id);
  const isCmd = !!e.command;
  return h("div", {}, h("h2", {}, sid(e.id), " ", h("span", {class: "badge", style: `border-color:${EV_COLOR[e.type]}`}, e.type.replace("_", " ")), " ",
    h("span", {class: "badge", style: post ? "border-color:#c2185b;color:#c2185b" : ""}, post ? `after temporary change ${sid(e.intervention_id) || ""}` : "original repository")),
    h("p", {}, e.summary),
    h("div", {class: "kv"}, h("b", {}, "recorded"), `event ${e.created_at_step} by ${e.created_by}`, e.path ? [h("b", {}, "location"), `${e.path}${e.start_line ? `:${e.start_line}-${e.end_line}` : ""}`] : null,
      e.source_tool_call_id ? [h("b", {}, "tool call"), sid(e.source_tool_call_id)] : null, isCmd ? [h("b", {}, "command"), h("code", {}, e.command)] : null,
      e.exit_code != null ? [h("b", {}, "exit code"), String(e.exit_code)] : null, e.duration_s != null ? [h("b", {}, "duration"), e.duration_s + " s"] : null,
      e.intervention_id ? [h("b", {}, "intervention"), sid(e.intervention_id)] : null, e.branch_id ? [h("b", {}, "branch"), sid(e.branch_id)] : null,
      e.probe_id ? [h("b", {}, "probe"), ref("probe", e.probe_id)] : null,
      h("b", {}, "repo state"), e.repo_state ? h("span", {class: "hash"}, `tree ${(e.repo_state.base_tree || "").slice(0, 12)}${e.repo_state.base_head ? " · head " + e.repo_state.base_head.slice(0, 10) : ""}`) : "—",
      h("b", {}, "excerpt hash"), h("span", {class: "hash"}, e.excerpt_hash || "—")),
    h("h3", {}, isCmd ? "Output (exact excerpt)" : "Exact excerpt"),
    h("pre", {}, e.start_line ? (e.excerpt || "").split("\n").map((ln, i) => String(e.start_line + i).padStart(5) + "  " + ln).join("\n") : (e.excerpt || "")),
    call && call.output && call.output !== e.excerpt ? h("details", {}, h("summary", {}, "Full recorded tool output"), h("pre", {}, call.output)) : null,
    h("h3", {}, "Relations"), rels.length ? h("table", {}, rels.map((r) => h("tr", {}, h("td", {style: `color:${REL_COLOR[r.type]};font-weight:600`}, r.type, ` (${r.strength})`), h("td", {}, ref("hypothesis", r.to_id)), h("td", {}, r.rationale)))) : h("span", {class: "muted"}, "not related to any hypothesis"));
}

function probeDetail(p) {
  if (!p) return h("p", {class: "muted"}, "This probe does not exist yet at the selected step.");
  return h("div", {}, h("h2", {}, sid(p.id), " ", h("span", {class: "badge"}, p.purpose), " ", h("span", {class: "badge"}, p.status)), h("p", {}, p.planned_action),
    h("h3", {}, "Targets"), list(p.target_hypothesis_ids.map((t) => ref("hypothesis", t))), h("h3", {}, "Expected outcomes"),
    h("table", {}, p.expected_outcomes.map((o) => h("tr", {}, h("td", {}, ref("hypothesis", o.hypothesis_id)), h("td", {}, o.expect)))), h("h3", {}, "Actual evidence"), list(p.actual_evidence_ids.map((e) => ref("evidence", e))));
}

function panelDiagnosis() {
  const g = S.g, d = g.diagnosis;
  if (!d) return h("div", {}, h("h2", {}, "Diagnosis"), h("p", {class: "muted"}, "No diagnosis has been committed at this step."));
  return h("div", {}, h("h2", {}, "Committed diagnosis ", h("span", {class: "badge"}, d.status), " ", h("span", {class: "muted"}, `event ${d.committed_at_step}`)),
    h("p", {}, d.causal_explanation), h("div", {class: "kv"}, h("b", {}, "selected"), d.selected_hypothesis_ids.length ? d.selected_hypothesis_ids.map((x) => [ref("hypothesis", x), " "]) : "none (inconclusive)",
      d.leading_hypothesis_ids ? [h("b", {}, "leading"), d.leading_hypothesis_ids.map((x) => [ref("hypothesis", x), " "])] : null, h("b", {}, "components"), d.affected_components.join(", ") || "—"),
    h("h3", {}, "Issue requirements explained"), list(d.requirements_explained), h("h3", {}, "Strongest evidence"), list(d.strongest_evidence_ids.map((e) => [ref("evidence", e), " ", (g.evidence[e] || {}).summary])),
    h("h3", {}, "Rejected alternatives"), list(d.rejected_alternatives.map((r) => [ref("hypothesis", r.hypothesis_id), ": ", r.reason])),
    h("h3", {}, "Unresolved risks"), list(d.unresolved_risks), h("h3", {}, "Recommended implementation direction"), h("p", {}, d.implementation_direction),
    S.art.implementation ? h("div", {}, h("h3", {}, "Implementation stage"), h("div", {class: "kv"}, h("b", {}, "stop"), String(S.art.implementation.stop_reason), h("b", {}, "patch"), (S.art.implementation.patch_bytes || 0) + " bytes", h("b", {}, "files"), (S.art.implementation.modified_files || []).join(", ") || "—")) : null,
    h("h3", {}, "Reset verification"), h("table", {}, g.resets.map((r) => h("tr", {}, h("td", {}, r.seq), h("td", {}, r.context), h("td", {style: `color:var(${r.ok ? "--supported" : "--refuted"})`}, r.ok ? "verified clean" : "FAILED")))));
}

function scoreRows(scores) {
  return h("table", {}, h("tr", {}, h("th", {}, "dimension"), h("th", {}, "score"), h("th", {}, "justification")),
    Object.entries(scores).map(([k, v]) => h("tr", {}, h("td", {}, k.replace(/_/g, " ")), h("td", {class: "score s" + v.score}, v.score + "/4"), h("td", {}, v.justification, " ", (v.cited_ids || []).map((c) => [refAny(c), " "])))));
}

function panelJudges() {
  const j = S.art.judge || {}, p = [];
  const off = S.art.official;
  p.push(h("h2", {}, "Judges"), h("p", {class: "muted"}, "Official outcome, blind process judge and oracle judge are kept separate; none is folded into another."),
    h("div", {class: "kv"}, h("b", {}, "official"), off ? `${off.status}${off.f2p_total != null ? ` (target tests ${off.f2p_passed}/${off.f2p_total}, regression ${off.p2p_passed}/${off.p2p_total})` : ""}` : "not graded"));
  if (j.blind) { const r = j.blind.result; p.push(h("h3", {}, "Blind process judge ", j.blind.valid ? "" : h("span", {class: "badge"}, "citations invalid")), scoreRows(r.scores),
    h("h3", {}, "Major diagnostic errors"), list((r.major_diagnostic_errors || []).map((f) => [f.text, " ", f.cited_ids.map((c) => [refAny(c), " "])])),
    h("h3", {}, "Unsupported conclusions"), list((r.unsupported_conclusions || []).map((f) => [f.text, " ", f.cited_ids.map((c) => [refAny(c), " "])])),
    h("h3", {}, "Missed hypotheses"), list(r.missed_hypotheses || []), h("h3", {}, "Strongest part"), h("p", {}, r.strongest_part.text, " ", r.strongest_part.cited_ids.map((c) => [refAny(c), " "])),
    h("div", {class: "kv"}, h("b", {}, "earliest supported"), r.earliest_supported_event_seq != null ? h("a", {class: "ref", onclick: () => { stopPlay(); setStep(r.earliest_supported_event_seq); }}, "event " + r.earliest_supported_event_seq) : "—",
      h("b", {}, "impl. followed"), `${r.implementation_followed_diagnosis.value}: ${r.implementation_followed_diagnosis.justification}`)); }
  else p.push(h("h3", {}, "Blind process judge"), h("p", {class: "muted"}, "not run"));
  if (j.reference) p.push(h("h3", {}, "Independent reference account (written before the comparison)"), h("details", {}, h("summary", {}, "show"), h("pre", {}, JSON.stringify(j.reference.reference, null, 2))));
  if (j.oracle) { const r = j.oracle.result; p.push(h("h3", {}, "Oracle judge"), scoreRows(r.scores),
    h("div", {class: "kv"}, h("b", {}, "hindsight"), `${r.hindsight_leakage.level}: ${r.hindsight_leakage.text}`, h("b", {}, "earliest sufficient"), r.earliest_sufficient_event_seq != null ? h("a", {class: "ref", onclick: () => { stopPlay(); setStep(r.earliest_sufficient_event_seq); }}, "event " + r.earliest_sufficient_event_seq) : "—",
      h("b", {}, "after that"), r.investigation_after_sufficient || "—"),
    h("h3", {}, "Accepted despite contradiction"), list((r.accepted_despite_contradiction || []).map((f) => [f.text, " ", f.cited_ids.map((c) => [refAny(c), " "])])),
    h("h3", {}, "Major diagnostic errors"), list((r.major_diagnostic_errors || []).map((f) => [f.text, " ", f.cited_ids.map((c) => [refAny(c), " "])]))); }
  else p.push(h("h3", {}, "Oracle judge"), h("p", {class: "muted"}, "not run"));
  return h("div", {}, p);
}

function panelController() {
  const g = S.g, b = g.budget || {};
  return h("div", {}, h("h2", {}, "Controller state"), h("div", {class: "kv"}, h("b", {}, "phase"), g.phase, h("b", {}, "iteration"), b.iteration ? `${b.iteration} / ${b.max_iterations}` : "—",
    h("b", {}, "last budget"), b.stage || "—", h("b", {}, "tokens"), b.prompt_tokens != null ? `${b.prompt_tokens} in / ${b.completion_tokens} out` : "—", h("b", {}, "cost"), b.cost_usd != null ? "$" + b.cost_usd : "—",
    h("b", {}, "interventions"), String(Object.keys(g.interventions).length), h("b", {}, "syntheses"), String(Object.keys(g.syntheses).length)),
    h("h3", {}, "Controller decisions"), g.decisions.length ? h("table", {}, g.decisions.map((d) => h("tr", {}, h("td", {}, d.seq), h("td", {}, d.actor), h("td", {}, d.kind, d.hypothesis_id ? [" → ", ref("hypothesis", d.hypothesis_id)] : "", d.rationale ? h("div", {class: "muted"}, d.rationale) : null)))) : h("span", {class: "muted"}, "none yet"),
    h("h3", {}, "Branches"), g.branches && Object.keys(g.branches).length ? h("table", {}, Object.values(g.branches).map((br) => h("tr", {}, h("td", {}, sid(br.id)), h("td", {}, ref("hypothesis", br.hypothesis_id)), h("td", {}, br.status, br.report ? " · " + br.report.outcome : ""), h("td", {class: "muted"}, br.report ? br.report.summary : "")))) : h("span", {class: "muted"}, "none yet"),
    h("h3", {}, "Interventions (temporary changes)"), Object.keys(g.interventions).length ? h("table", {}, Object.values(g.interventions).map((i) => h("tr", {}, h("td", {}, sid(i.id)), h("td", {}, sid(i.branch_id)), h("td", {}, i.description), h("td", {}, i.reset_verified ? "reverted + verified" : "open")))) : h("span", {class: "muted"}, "none"),
    h("h3", {}, "Syntheses"), Object.keys(g.syntheses).length ? list(Object.values(g.syntheses).map((sy) => [sid(sy.id), ": ", sy.reasons.join("; "), sy.result ? h("div", {class: "muted"}, sy.result) : null])) : h("span", {class: "muted"}, "none"));
}

function describe(ev) {
  const d = ev.data, i = sid;
  switch (ev.type) {
    case "hypothesis_created": return `+ ${i(d.id)} ${d.claim}`;
    case "hypothesis_status_updated": return `${i(d.id)} → ${d.new_status}${(d.cause.evidence_ids || []).length ? " (evidence " + d.cause.evidence_ids.map(i).join(", ") + ")" : ""}`;
    case "evidence_attached": return `+ ${i(d.id)} [${d.type}${d.observation_state !== "original" ? ", post-change" : ""}] ${d.summary}`;
    case "relation_created": return `${i(d.from_id)} ${d.type} ${i(d.to_id)}`;
    case "probe_created": return `+ ${i(d.id)} ${d.purpose}: ${d.planned_action}`;
    case "probe_updated": return `${i(d.id)} ${d.status || ""} ${d.add_evidence_ids && d.add_evidence_ids.length ? "+evidence" : ""}`;
    case "falsification_criterion_defined": return `+ criterion for ${i(d.hypothesis_id)}`;
    case "branch_started": return `branch ${i(d.id)} on ${i(d.hypothesis_id)}`;
    case "branch_ended": return `branch ${i(d.id)} ended (${d.stop_reason}): ${(d.report || {}).outcome || ""}`;
    case "intervention_started": return `temporary change ${i(d.id)}: ${d.description}`;
    case "intervention_ended": return `temporary change ${i(d.id)} reverted${d.reset_verified ? " + verified" : ""}`;
    case "reset_verified": return `reset check [${d.context}]: ${d.ok ? "clean" : "FAILED"}`;
    case "synthesis_requested": return `synthesis requested: ${d.reasons.join("; ")}`;
    case "synthesis_completed": return `synthesis completed`;
    case "controller_decision": return `${d.kind}${d.hypothesis_id ? " " + i(d.hypothesis_id) : ""}`;
    case "phase_changed": return `phase → ${d.phase}`;
    case "budget_updated": return `budget ${d.stage || ""} ${d.llm_calls != null ? d.llm_calls + " calls" : ""}`;
    case "diagnosis_committed": return `DIAGNOSIS ${d.status}: ${d.causal_explanation.slice(0, 120)}`;
    case "operation_rejected": return `rejected ${d.op}: ${d.error}`;
    case "hypotheses_merged": return `merged ${i(d.absorbed_id)} into ${i(d.kept_id)}`;
    default: return ev.type;
  }
}

function renderTimeline() {
  const cur = S.step == null ? S.events.length : S.step;
  $("#stepinfo").textContent = S.step == null ? `latest (event ${S.total})` : `event ${S.step} of ${S.total}`;
  const t = $("#timeline");
  t.replaceChildren(...S.events.map((ev) => h("div", {class: `ev${ev.seq > cur ? " future" : ""}${ev.seq === cur ? " current" : ""}${ev.type === "operation_rejected" ? " rej" : ""}`, onclick: () => { stopPlay(); setStep(ev.seq); }},
    h("span", {class: "n"}, ev.seq), h("span", {class: "a"}, ev.actor), h("span", {class: "t"}, describe(ev)))));
  const c = t.querySelector(".current");
  if (c) c.scrollIntoView({block: "nearest"});
}

init();
