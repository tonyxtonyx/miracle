"""Global-synthesis triggers: a pure function of the graph state and a few counters kept by the engine."""
from __future__ import annotations

from .config import DiagnosisConfig
from .graph import GraphState, short_id

VIABLE = ("proposed", "investigating", "supported")


def synthesis_triggers(st: GraphState, cfg: DiagnosisConfig, *, stalled: int, repeated_actions: bool, last_report: dict | None) -> list[str]:
    """Reasons a dedicated synthesis step is required now (empty list = none)."""
    reasons: list[str] = []
    active = st.active()
    viable = [h for h in active if h["status"] in VIABLE]
    if st.hypotheses and not viable:
        reasons.append("all active hypotheses are refuted, weakened or inconclusive: none is viable under the support threshold")
    if len(viable) >= 2:
        ids = {h["id"] for h in viable}
        if not any(p["status"] == "planned" and len(set(p["target_hypothesis_ids"]) & ids) >= 2 for p in st.probes.values()):
            reasons.append(f"no planned probe discriminates between the remaining hypotheses ({', '.join(short_id(i) for i in sorted(ids))})")
    unexplained = (last_report or {}).get("unexplained_observations") or []
    if unexplained:
        reasons.append("evidence is unexplained by every current hypothesis: " + "; ".join(u[:80] for u in unexplained[:2]))
    for h in viable:
        if h["status"] == "supported":
            sup_at = max((x["seq"] for x in h["history"] if x["new"] == "supported"), default=0)
            late = [e for e in st.evidence_for(h["id"], "contradicts") if e["created_at_step"] > sup_at]
            if late:
                reasons.append(f"new evidence ({', '.join(short_id(e['id']) for e in late)}) contradicts the leading hypothesis {short_id(h['id'])}")
    if repeated_actions:
        reasons.append("the investigation repeated equivalent actions")
    if stalled >= cfg.stall_iterations:
        reasons.append(f"no progress for {stalled} iterations (no new relations or status changes)")
    size = len(st.hypotheses) + len(st.evidence) + len(st.probes)
    if size >= cfg.max_graph_size:
        reasons.append(f"the graph reached its size limit ({size} nodes)")
    if st.hypotheses and max(st.depth(h) for h in st.hypotheses) >= cfg.max_depth and len(viable) > 1:
        reasons.append("the refinement depth limit was reached with several hypotheses still open")
    return reasons
