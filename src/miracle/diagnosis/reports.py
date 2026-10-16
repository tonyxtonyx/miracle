"""Per-task report and cross-task aggregate. The official outcome, the blind process judge and the oracle judge are reported side by side and never
combined; their relationship is analysed afterwards. Cost 'before / after the diagnosis was sufficiently supported' uses the budget snapshots the
engine writes into the event log (granularity: after each branch and each review)."""
from __future__ import annotations

import json
import statistics as st
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .graph import read_events, short_id
from .judging.schema import BLIND_DIMS, ORACLE_DIMS

SNAP_KEYS = ("prompt_tokens", "completion_tokens", "llm_calls", "tool_calls", "cost_usd", "wall_s")


def _json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def snapshots(events: list[dict]) -> list[tuple[int, dict]]:
    return [(e["seq"], {k: e["data"].get(k, 0) for k in SNAP_KEYS}) for e in events if e["type"] == "budget_updated" and "llm_calls" in e["data"]]


def snapshot_at(snaps: list[tuple[int, dict]], seq: int | None) -> dict:
    best = {k: 0 for k in SNAP_KEYS}
    for s, snap in snaps:
        if seq is not None and s <= seq:
            best = snap
    return best


def _diff(a: dict, b: dict) -> dict:
    return {k: round((a.get(k) or 0) - (b.get(k) or 0), 4) for k in SNAP_KEYS}


def task_report(inst_dir: Path, instance_id: str, official: dict | None) -> dict:
    d = Path(inst_dir)
    events = read_events(d / "diagnosis" / "events.jsonl")
    graph = _json(d / "diagnosis" / "graph.json") or {}
    usage = _json(d / "diagnosis" / "usage.json") or {}
    impl = _json(d / "implementation" / "result.json") or {}
    blind = (_json(d / "judge" / "blind.json") or {})
    oracle = (_json(d / "judge" / "oracle.json") or {})
    br, orr = blind.get("result") or {}, oracle.get("result") or {}
    hyps = list((graph.get("hypotheses") or {}).values())
    children = Counter(p for h in hyps for p in h["parent_ids"])
    snaps = snapshots(events)
    end = snaps[-1][1] if snaps else {k: 0 for k in SNAP_KEYS}
    suff = {}
    for who, seq in (("oracle", orr.get("earliest_sufficient_event_seq")), ("blind", br.get("earliest_supported_event_seq"))):
        if seq is not None:
            at = snapshot_at(snaps, seq)
            suff[who] = {"event_seq": seq, "cumulative_at_that_point": at, "spent_after": _diff(end, at),
                         "share_after": {k: round(((end.get(k) or 0) - (at.get(k) or 0)) / (end.get(k) or 1), 3) for k in ("llm_calls", "tool_calls", "cost_usd", "wall_s")}}
    mean = lambda scores: round(st.mean(v["score"] for v in scores.values()), 2) if scores else None
    judge_cost = round(sum((j.get("judge", {}).get("cost_usd") or 0) for j in (blind, oracle, _json(d / "judge" / "reference_diagnosis.json") or {})), 4)
    ev_types = Counter(e["type"] for e in (graph.get("evidence") or {}).values())
    rel_types = Counter(r["type"] for r in (graph.get("relations") or {}).values())
    rep = {
        "instance_id": instance_id, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "official": {k: (official or {}).get(k) for k in ("status", "resolved", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total")} if official else None,
        "diagnosis": {"status": (graph.get("diagnosis") or {}).get("status"), "selected": [short_id(x) for x in (graph.get("diagnosis") or {}).get("selected_hypothesis_ids", [])],
                      "causal_explanation": (graph.get("diagnosis") or {}).get("causal_explanation"), "events": len(events)},
        "graph": {**(graph.get("stats") or {}), "evidence_by_type": dict(ev_types), "relations_by_type": dict(rel_types),
                  "probes_by_purpose": dict(Counter(p["purpose"] for p in (graph.get("probes") or {}).values())),
                  "probes_by_status": dict(Counter(p["status"] for p in (graph.get("probes") or {}).values())),
                  "avg_children": round(sum(children.values()) / max(len(hyps), 1), 2), "max_children": max(children.values(), default=0),
                  "post_intervention_evidence": sum(1 for e in (graph.get("evidence") or {}).values() if e.get("observation_state") != "original"),
                  "interventions": len(graph.get("interventions") or {}), "syntheses": len(graph.get("syntheses") or {}),
                  "rejected_operations": usage.get("rejected_operations", 0), "resets_all_verified": all(r["ok"] for r in graph.get("resets", [])) if graph.get("resets") else None},
        "cost": {"diagnosis_total": usage.get("total"), "diagnosis_by_role": usage.get("by_role"), "at_diagnosis_end": end, "implementation": impl.get("usage"),
                 "judges_usd": judge_cost},
        "sufficiency": suff,
        "judges": {"blind": {"present": bool(br), "valid": blind.get("valid"), "mean_score": mean(br.get("scores")), "scores": {k: v["score"] for k, v in (br.get("scores") or {}).items()},
                             "major_errors": len(br.get("major_diagnostic_errors") or []), "unsupported": len(br.get("unsupported_conclusions") or []),
                             "missed_hypotheses": br.get("missed_hypotheses") or [], "impl_followed": (br.get("implementation_followed_diagnosis") or {}).get("value"),
                             "major_error_texts": [f["text"] for f in (br.get("major_diagnostic_errors") or [])]},
                   "oracle": {"present": bool(orr), "valid": oracle.get("valid"), "mean_score": mean(orr.get("scores")), "scores": {k: v["score"] for k, v in (orr.get("scores") or {}).items()},
                              "hindsight_leakage": (orr.get("hindsight_leakage") or {}).get("level"), "accepted_despite_contradiction": len(orr.get("accepted_despite_contradiction") or []),
                              "major_errors": len(orr.get("major_diagnostic_errors") or []), "major_error_texts": [f["text"] for f in (orr.get("major_diagnostic_errors") or [])],
                              "impl_followed": (orr.get("implementation_followed_diagnosis") or {}).get("value")}},
        "implementation": {"stop_reason": impl.get("stop_reason"), "patch_bytes": impl.get("patch_bytes"), "modified_files": impl.get("modified_files")},
    }
    (d / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    (d / "report.md").write_text(render_task_md(rep))
    return rep


def render_task_md(r: dict) -> str:
    o, dg, g, c = r["official"], r["diagnosis"], r["graph"], r["cost"]
    L = [f"# {r['instance_id']}", "", "## Outcomes (kept separate)",
         f"- **Official SWE-bench result:** {('resolved' if o['resolved'] else o['status']) if o else 'not graded'}" + (f" (target tests {o['f2p_passed']}/{o['f2p_total']}, regression {o['p2p_passed']}/{o['p2p_total']})" if o and o.get('f2p_total') is not None else ""),
         f"- **Blind process judge:** mean {r['judges']['blind']['mean_score']}/4" if r["judges"]["blind"]["present"] else "- **Blind process judge:** not run",
         f"- **Oracle judge:** mean {r['judges']['oracle']['mean_score']}/4, causal correctness {r['judges']['oracle']['scores'].get('causal_correctness')}/4" if r["judges"]["oracle"]["present"] else "- **Oracle judge:** not run",
         "", "## Diagnosis", f"Status: **{dg['status']}**; selected {dg['selected'] or 'none'}; {dg['events']} events.", "", dg["causal_explanation"] or "", "",
         "## Graph", f"{g.get('hypotheses', 0)} hypotheses (max depth {g.get('max_depth', 0)}, avg children {g['avg_children']}), {g.get('evidence', 0)} evidence "
         f"({g['evidence_by_type']}), {g.get('relations', 0)} relations {g['relations_by_type']}, {g.get('probes', 0)} probes {g['probes_by_purpose']}, {g.get('branches', 0)} branches, "
         f"{g['interventions']} temporary changes ({g['post_intervention_evidence']} post-intervention evidence), {g['syntheses']} syntheses, {g['rejected_operations']} rejected operations. "
         f"Status counts: {g.get('by_status')}. All resets verified: {g['resets_all_verified']}.", "", "## Cost"]
    if c.get("diagnosis_total"):
        t = c["diagnosis_total"]
        L.append(f"Diagnosis: {t['llm_calls']} LLM calls, {t['tool_calls']} tool calls, {t['prompt_tokens']} + {t['completion_tokens']} tokens, ${t['cost_usd']}, {t['wall_s']} s.")
    if c.get("implementation"):
        i = c["implementation"]
        L.append(f"Implementation: {i['llm_calls']} LLM calls, {i['tool_calls']} tool calls, {i['prompt_tokens']} + {i['completion_tokens']} tokens, ${i['cost_usd']}.")
    for who, s in r["sufficiency"].items():
        a = s["spent_after"]
        L.append(f"After the diagnosis was already sufficiently supported ({who} judge, event {s['event_seq']}): {a['llm_calls']:.0f} LLM calls, {a['tool_calls']:.0f} tool calls, ${a['cost_usd']}, {a['wall_s']} s "
                 f"({s['share_after']['cost_usd']:.0%} of diagnosis cost).")
    L += ["", "## Judges"]
    for who in ("blind", "oracle"):
        j = r["judges"][who]
        if j["present"]:
            L.append(f"**{who}** (citations valid: {j['valid']}): " + ", ".join(f"{k} {v}" for k, v in j["scores"].items()))
            for t in j["major_error_texts"]:
                L.append(f"- error: {t}")
    L += ["", "## Implementation", f"{r['implementation']}"]
    return "\n".join(L) + "\n"


# ---- aggregate ---------------------------------------------------------------------------------------------------------
def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = st.mean(ra), st.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return round(num / den, 3) if den else None


def _stats(xs):
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "mean": round(st.mean(xs), 2), "median": st.median(xs), "min": min(xs), "max": max(xs)} if xs else {"n": 0}


def aggregate(run_dir: Path, instance_ids: list[str]) -> dict:
    reps = []
    for iid in instance_ids:
        r = _json(Path(run_dir) / "agent" / iid / "report.json")
        if r:
            reps.append(r)
    dims = {"blind": {d: _stats([r["judges"]["blind"]["scores"].get(d) for r in reps if r["judges"]["blind"]["present"]]) for d in BLIND_DIMS},
            "oracle": {d: _stats([r["judges"]["oracle"]["scores"].get(d) for r in reps if r["judges"]["oracle"]["present"]]) for d in ORACLE_DIMS}}
    pairs = [(r["official"]["resolved"], r["judges"]["blind"]["mean_score"], r["judges"]["oracle"]["scores"].get("causal_correctness"), r["instance_id"]) for r in reps
             if r.get("official") and r["judges"]["blind"]["present"] and r["judges"]["oracle"]["present"]]
    by_res = {str(k): {"n": sum(1 for p in pairs if p[0] == k), "blind_mean": _stats([p[1] for p in pairs if p[0] == k]).get("mean"),
                       "oracle_causal_mean": _stats([p[2] for p in pairs if p[0] == k]).get("mean")} for k in (True, False)}
    # judge agreement: blind process quality vs oracle causal correctness (different constructs; reported as association + explicit disagreements)
    disagreements = [{"instance_id": p[3], "blind_mean": p[1], "oracle_causal_correctness": p[2], "official_resolved": p[0]}
                     for p in pairs if p[1] is not None and p[2] is not None and ((p[1] >= 3 and p[2] <= 1) or (p[1] <= 1.5 and p[2] >= 3))]
    seq_pairs = [(r["sufficiency"]["blind"]["event_seq"], r["sufficiency"]["oracle"]["event_seq"]) for r in reps if "blind" in r["sufficiency"] and "oracle" in r["sufficiency"]]
    impl = Counter((r["judges"]["blind"]["impl_followed"], r["judges"]["oracle"]["impl_followed"]) for r in reps if r["judges"]["blind"]["present"] and r["judges"]["oracle"]["present"])
    g = lambda k: [r["graph"].get(k) for r in reps]
    after = lambda who, key: [r["sufficiency"][who]["spent_after"][key] for r in reps if who in r["sufficiency"]]
    cost = lambda f: [f(r) for r in reps]
    common_failures = Counter()
    for r in reps:
        common_failures.update(["diagnosis inconclusive"] if r["diagnosis"]["status"] == "inconclusive" else [])
        common_failures.update(["hypothesis accepted despite contradictory evidence (oracle)"] * (r["judges"]["oracle"].get("accepted_despite_contradiction") or 0))
        common_failures.update([f"hindsight leakage: {r['judges']['oracle']['hindsight_leakage']}"] if r["judges"]["oracle"].get("hindsight_leakage") in ("suspected", "clear") else [])
        common_failures.update(["post-intervention evidence mistaken for original observation (none recorded)"] * 0)
        common_failures.update(["rejected structured operations (model format errors)"] if r["graph"].get("rejected_operations") else [])
        common_failures.update([f"missed hypothesis ({len(r['judges']['blind']['missed_hypotheses'])} flagged by blind judge)"] if r["judges"]["blind"].get("missed_hypotheses") else [])
    return {"tasks": [r["instance_id"] for r in reps], "n_tasks": len(reps), "judge_scores": dims,
            "graph": {"hypotheses": _stats(g("hypotheses")), "evidence": _stats(g("evidence")), "relations": _stats(g("relations")), "probes": _stats(g("probes")),
                      "branches": _stats(g("branches")), "max_depth": _stats(g("max_depth")), "avg_children": _stats(g("avg_children")), "syntheses": _stats(g("syntheses")),
                      "interventions": _stats(g("interventions")), "rejected_operations": _stats(g("rejected_operations"))},
            "resources_before_commit": {k: _stats(cost(lambda r, k=k: (r["cost"]["at_diagnosis_end"] or {}).get(k))) for k in ("llm_calls", "tool_calls", "prompt_tokens", "completion_tokens", "cost_usd", "wall_s")},
            "resources_after_sufficient_support": {who: {k: _stats(after(who, k)) for k in ("llm_calls", "tool_calls", "cost_usd", "wall_s")} for who in ("oracle", "blind")},
            "official_vs_diagnosis_quality": {"by_resolution": by_res, "spearman_blind_mean_vs_oracle_causal": spearman([p[1] for p in pairs if p[1] is not None and p[2] is not None], [p[2] for p in pairs if p[1] is not None and p[2] is not None]),
                                              "spearman_oracle_causal_vs_resolved": spearman([p[2] for p in pairs if p[2] is not None], [1.0 if p[0] else 0.0 for p in pairs if p[2] is not None]),
                                              "per_task": [{"instance_id": p[3], "resolved": p[0], "blind_mean": p[1], "oracle_causal": p[2]} for p in pairs]},
            "judge_agreement": {"disagreements": disagreements, "earliest_sufficient_seq_pairs_blind_oracle": seq_pairs,
                                "implementation_followed_diagnosis_blind_vs_oracle": {f"{a}|{b}": n for (a, b), n in impl.items()}},
            "common_diagnostic_failures": dict(common_failures.most_common()),
            "judge_error_texts": {"blind": [t for r in reps for t in r["judges"]["blind"]["major_error_texts"]][:30], "oracle": [t for r in reps for t in r["judges"]["oracle"]["major_error_texts"]][:30]},
            "judges_cost_usd": round(sum((r["cost"].get("judges_usd") or 0) for r in reps), 4)}


def render_aggregate_md(a: dict) -> str:
    L = [f"# Aggregate report ({a['n_tasks']} tasks)", "", "Official outcome, blind process judge and oracle judge are separate measures; see the last sections for their relationship.", "", "## Judge scores by dimension (0-4)",
         "| dimension | blind: mean (n) | oracle: mean (n) |", "|---|---|---|"]
    for d in sorted(set(a["judge_scores"]["blind"]) | set(a["judge_scores"]["oracle"])):
        b, o = a["judge_scores"]["blind"].get(d, {"n": 0}), a["judge_scores"]["oracle"].get(d, {"n": 0})
        L.append(f"| {d} | {b.get('mean', '-')} ({b['n']}) | {o.get('mean', '-')} ({o['n']}) |")
    L += ["", "## Graph statistics (per task: mean / median / max)", "| quantity | mean | median | max |", "|---|---|---|---|"]
    L += [f"| {k} | {v.get('mean', '-')} | {v.get('median', '-')} | {v.get('max', '-')} |" for k, v in a["graph"].items()]
    L += ["", "## Resources before the diagnosis was committed (per task)", "| resource | mean | median | max |", "|---|---|---|---|"]
    L += [f"| {k} | {v.get('mean', '-')} | {v.get('median', '-')} | {v.get('max', '-')} |" for k, v in a["resources_before_commit"].items()]
    L += ["", "## Resources spent AFTER the diagnosis was already sufficiently supported (per task)", "| judge | resource | mean | median | max | n |", "|---|---|---|---|---|---|"]
    for who, d in a["resources_after_sufficient_support"].items():
        L += [f"| {who} | {k} | {v.get('mean', '-')} | {v.get('median', '-')} | {v.get('max', '-')} | {v['n']} |" for k, v in d.items()]
    o = a["official_vs_diagnosis_quality"]
    L += ["", "## Official resolution vs diagnosis quality", "| official resolved | tasks | blind mean | oracle causal mean |", "|---|---|---|---|"]
    L += [f"| {k} | {v['n']} | {v['blind_mean']} | {v['oracle_causal_mean']} |" for k, v in o["by_resolution"].items()]
    L += [f"", f"Spearman (blind mean vs oracle causal correctness): {o['spearman_blind_mean_vs_oracle_causal']}; (oracle causal vs resolved): {o['spearman_oracle_causal_vs_resolved']} (None = fewer than 3 tasks or no variance).", "",
          "## Judge agreement", f"Disagreements (blind process high but oracle causal low, or the reverse): {a['judge_agreement']['disagreements'] or 'none'}",
          f"Earliest-sufficient event (blind, oracle) pairs: {a['judge_agreement']['earliest_sufficient_seq_pairs_blind_oracle']}", f"Implementation followed diagnosis (blind|oracle): {a['judge_agreement']['implementation_followed_diagnosis_blind_vs_oracle']}",
          "", "## Common diagnostic failures", *(f"- {k}: {v}" for k, v in a["common_diagnostic_failures"].items()), "", "### Major errors named by the judges"]
    L += [f"- blind: {t}" for t in a["judge_error_texts"]["blind"]] + [f"- oracle: {t}" for t in a["judge_error_texts"]["oracle"]]
    L += ["", f"Judge cost: ${a['judges_cost_usd']}"]
    return "\n".join(L) + "\n"
