"""The diagnosis -> implementation boundary. The implementation stage receives ONLY: the issue (added by the caller), the committed diagnosis,
selected supporting evidence, rejected alternatives with reasons, unresolved risks, and the compact repository map. No graph, no traces."""
from __future__ import annotations

from .context import _clip
from .graph import GraphState, short_id

S = short_id


def _evidence_plain(e: dict) -> str:
    """Evidence for the implementation agent: what was observed and where, without graph IDs or investigation bookkeeping."""
    where = (e["path"] + (f" lines {e['start_line']}-{e['end_line']}" if e.get("start_line") else "")) if e.get("path") else (f"command `{_clip(e['command'], 100)}`" if e.get("command") else e["type"])
    exit_ = f" (exit code {e['exit_code']})" if e.get("exit_code") is not None else ""
    temp = " [observed with a temporary debug change that has since been reverted]" if e.get("observation_state") == "post_intervention" else ""
    return f"{where}{exit_}{temp}: {_clip(e['summary'], 220)}"


def build_handoff(st: GraphState, repo_map: str) -> str:
    d = st.diagnosis
    if d is None:
        return "No diagnosis was produced.\n\n## Repository map\n" + repo_map
    L = [f"## Diagnosis ({d['status']})", d["causal_explanation"]]
    if d["status"] == "inconclusive":
        lead = [st.hypotheses[h]["claim"] for h in d.get("leading_hypothesis_ids", []) if h in st.hypotheses]
        L.append("The investigation did NOT establish a cause. Leading, unverified hypotheses: " + (" | ".join(lead) or "none"))
    if d["affected_components"]:
        L.append("Affected components: " + ", ".join(d["affected_components"]))
    if d["requirements_explained"]:
        L.append("Requirements of the issue this explains:\n" + "\n".join(f"- {r}" for r in d["requirements_explained"]))
    L.append("Recommended direction: " + d["implementation_direction"])
    ev = [st.evidence[e] for e in d["strongest_evidence_ids"] if e in st.evidence]
    if ev:
        L.append("## Supporting evidence")
        for e in ev:
            ex = (e.get("excerpt") or "")[:900]
            L.append(f"{_evidence_plain(e)}\n```\n{ex}\n```")
    if d["rejected_alternatives"]:
        L.append("## Alternatives that were rejected")
        L += [f"- {_clip(st.hypotheses[r['hypothesis_id']]['claim'], 200)} -- rejected because: {r['reason']}"
              for r in d["rejected_alternatives"] if r["hypothesis_id"] in st.hypotheses]
    if d["unresolved_risks"]:
        L.append("## Unresolved risks\n" + "\n".join(f"- {r}" for r in d["unresolved_risks"]))
    L.append("## Repository map\n" + repo_map)
    return "\n\n".join(L)
