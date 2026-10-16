"""Context builders: what each role is allowed to see. The controller and synthesis get graph summaries (never raw traces); an investigator
gets exactly one hypothesis branch plus compact views of its competitors. Pure functions of the graph state."""
from __future__ import annotations

from .graph import ACTIVE_STATUSES, GraphState, short_id

S = short_id


def _clip(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def evidence_line(e: dict, rels: list[dict] | None = None, state: GraphState | None = None) -> str:
    where = e["path"] + (f":{e['start_line']}-{e['end_line']}" if e.get("start_line") else "") if e.get("path") else (e.get("command") or e["type"])
    obs = "" if e.get("observation_state", "original") == "original" else f" [AFTER TEMP CHANGE {S(e['intervention_id'])}]" if e.get("intervention_id") else " [post-intervention]"
    link = ""
    if rels:
        link = " -> " + ", ".join(f"{r['type']} {S(r['to_id'])}(s{r['strength']})" for r in rels)
    exit_ = f" exit={e['exit_code']}" if e.get("exit_code") is not None else ""
    return f"{S(e['id'])} [{e['type']}{exit_}] {_clip(where, 70)}{obs}: {_clip(e['summary'], 160)}{link}"


def hypothesis_block(st: GraphState, h: dict, *, detail: bool) -> str:
    hid = h["id"]
    rin = st.relations_of(hid, direction="in")
    sup = [S(r["from_id"]) for r in rin if r["type"] == "supports"]
    con = [S(r["from_id"]) for r in rin if r["type"] == "contradicts"]
    alts = [S(r["to_id"] if r["from_id"] == hid else r["from_id"]) for r in st.relations_of(hid) if r["type"] == "alternative_to"]
    parents = [S(p) for p in h["parent_ids"]]
    head = (f"{S(hid)} [{h['status']}{'/' + h['support_level'] if h['support_level'] != 'none' else ''}] ({h['type']}) {_clip(h['claim'], 240)}"
            + (f"  <- refines {', '.join(parents)}" if parents else "") + (f"  alternative_to {', '.join(alts)}" if alts else "")
            + (f"  [merged into {S(h['merged_into'])}]" if h.get("merged_into") else ""))
    lines = [head, f"    evidence: supports {sup or '-'}; contradicts {con or '-'}"]
    if detail:
        lines.append("    predictions: " + " | ".join(_clip(p, 120) for p in h["predictions"]))
        lines.append("    falsified if: " + " | ".join(_clip(c["text"], 120) for c in h["falsification_criteria"]))
        if h["unresolved_questions"]:
            lines.append("    unresolved: " + " | ".join(_clip(q, 100) for q in h["unresolved_questions"]))
    return "\n".join(lines)


def graph_summary(st: GraphState, *, detail: bool = True, include_refuted_detail: bool = False) -> str:
    out = ["Hypotheses:"]
    for h in st.hypotheses.values():
        out.append(hypothesis_block(st, h, detail=detail and (h["status"] != "refuted" or include_refuted_detail)))
    probes = [p for p in st.probes.values()]
    if probes:
        out.append("Probes:")
        for p in probes:
            out.append(f"  {S(p['id'])} [{p['status']}] {p['purpose']} on {', '.join(S(t) for t in p['target_hypothesis_ids'])}: {_clip(p['planned_action'], 140)}"
                       + (f"  evidence: {', '.join(S(e) for e in p['actual_evidence_ids'])}" if p["actual_evidence_ids"] else ""))
    ev = [e for e in st.evidence.values() if e["type"] != "issue_statement"]
    if ev:
        out.append("Evidence (id [type] where: summary -> relations):")
        for e in ev[-40:]:
            out.append("  " + evidence_line(e, [r for r in st.relations_of(e["id"], direction="out")], st))
    unexplained = [o for b in st.branches.values() for o in ((b.get("report") or {}).get("unexplained_observations") or [])]
    if unexplained:
        out.append("Unexplained observations reported by investigators:")
        out += [f"  - {_clip(o, 200)}" for o in unexplained[-6:]]
    return "\n".join(out)


def branch_report_text(st: GraphState, bid: str) -> str:
    b = st.branches[bid]
    rep = b.get("report") or {}
    lines = [f"Investigator outcome: {rep.get('outcome', 'no report')} (stopped: {b.get('stop_reason')}).", f"Summary: {_clip(rep.get('summary', '(none)'), 700)}"]
    mine = [e for e in st.evidence.values() if e.get("branch_id") == bid]
    if mine:
        lines.append("Evidence recorded:")
        lines += ["  " + evidence_line(e, st.relations_of(e["id"], direction="out"), st) for e in mine]
    for key, label in (("unexplained_observations", "Unexplained observations"), ("unresolved_questions", "Unresolved questions"),
                       ("suggested_hypotheses", "Suggested new hypotheses")):
        if rep.get(key):
            lines.append(f"{label}: " + " | ".join(_clip(x if isinstance(x, str) else str(x), 200) for x in rep[key]))
    if st.interventions and any(i["branch_id"] == bid for i in st.interventions.values()):
        ivs = [S(i["id"]) for i in st.interventions.values() if i["branch_id"] == bid]
        lines.append(f"Temporary changes were made during this branch ({', '.join(ivs)}) and have been reverted; the repository was verified clean afterwards.")
    return "\n".join(lines)


def chain_text(st: GraphState, h: dict) -> str:
    chain, cur = [], h
    while cur["parent_ids"]:
        cur = st.hypotheses[cur["parent_ids"][0]]
        chain.append(f"{S(cur['id'])} [{cur['status']}] {_clip(cur['claim'], 160)}")
    return ("\nRefinement chain (this hypothesis refines):\n" + "\n".join("  " + c for c in chain) + "\n") if chain else ""


def branch_context(st: GraphState, issue: str, hid: str, probe_ids: list[str], steps: int, template: str) -> tuple[str, set[str]]:
    """The ONLY view an investigator gets. Returns (user message, hypothesis ids it may relate evidence to)."""
    h = st.hypotheses[hid]
    ev = [e for e in st.evidence_for(hid)]
    ev_lines = []
    for e in ev:
        rel = [r for r in st.relations_of(e["id"], direction="out") if r["to_id"] == hid]
        ev_lines.append("  " + evidence_line(e, rel, st))
    competing = [x for x in st.active() if x["id"] != hid]
    visible = {hid} | {x["id"] for x in competing}
    comp = "\n".join(f"  {S(x['id'])} [{x['status']}] {_clip(x['claim'], 140)}" for x in competing) or "  (none)"
    probes = []
    for pid in probe_ids:
        p = st.probes.get(pid)
        if p:
            outs = "; ".join(f"if {S(o['hypothesis_id'])} is true: {_clip(o['expect'], 100)}" for o in p["expected_outcomes"])
            probes.append(f"  {S(pid)} ({p['purpose']}): {_clip(p['planned_action'], 220)}  Expected -> {outs}")
    msg = template.format(
        issue=issue, hid=S(hid), htype=h["type"], status=h["status"], claim=h["claim"],
        predictions="\n".join(f"  - {p}" for p in h["predictions"]), criteria="\n".join(f"  - {c['text']}" for c in h["falsification_criteria"]),
        chain=chain_text(st, h), evidence="\n".join(ev_lines) or "  (none yet)", competing=comp,
        probes="\n".join(probes) or "  (none: design your own test of the predictions and falsification criteria)",
        questions="\n".join(f"  - {q}" for q in h["unresolved_questions"]) or "  (none)", steps=steps)
    return msg, visible
