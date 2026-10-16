"""Input builders = the data boundaries, enforced by function signatures.

  blind judge     <- CandidateMaterial only (issue, the agent's graph/trace/diagnosis/patch). It has no parameter through which gold data,
                     the official result or hints could enter.
  reference       <- the instance record + base-repo excerpts. It has no parameter for any candidate artefact (written BEFORE the comparison).
  oracle compare  <- instance + the PERSISTED reference + official result + CandidateMaterial.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..context import _clip
from ..graph import GraphState, read_events, replay, short_id

S = short_id
MAX_RENDER = 150_000


@dataclass(frozen=True)
class CandidateMaterial:
    """Everything the solving agent produced or could see. Nothing from the dataset's hidden fields is representable here."""
    issue: str
    state: GraphState
    event_count: int
    tool_calls: list[dict]
    implementation_patch: str

    @staticmethod
    def from_run_dir(instance_dir: Path, issue: str) -> "CandidateMaterial":
        d = Path(instance_dir)
        events = read_events(d / "diagnosis" / "events.jsonl")
        st = replay(events[0]["data"]["id"].rsplit(":", 1)[0] if events and ":" in events[0]["data"].get("id", "") else _iid(events), events)
        calls = []
        tc = d / "diagnosis" / "tool_calls.jsonl"
        if tc.exists():
            calls = [json.loads(l) for l in tc.read_text().splitlines() if l.strip()]
        st._tool_call_ids = {c["id"] for c in calls}
        patch = (d / "implementation" / "patch.diff").read_text() if (d / "implementation" / "patch.diff").exists() else ""
        return CandidateMaterial(issue, st, len(events), calls, patch)


def _iid(events: list[dict]) -> str:
    for e in events:
        for v in e["data"].values():
            if isinstance(v, str) and ":" in v and re.match(r"^[\w.\-]+:[A-Z]+\d+$", v):
                return v.rsplit(":", 1)[0]
    return "instance"


def render_candidate(m: CandidateMaterial) -> str:
    st = m.state
    L = ["# Reasoning graph (final state)", "", "## Hypotheses"]
    for h in st.hypotheses.values():
        L.append(f"{S(h['id'])} [{h['status']}, support {h['support_level']}] ({h['type']}) created at event {h['created_at_step']} by {h['created_by']}"
                 + (f", merged into {S(h['merged_into'])}" if h.get("merged_into") else "") + f"\n  claim: {h['claim']}")
        L.append("  predictions: " + " | ".join(h["predictions"]))
        L.append("  falsification criteria: " + " | ".join(c["text"] for c in h["falsification_criteria"]))
        if h["unresolved_questions"]:
            L.append("  unresolved questions: " + " | ".join(h["unresolved_questions"]))
        for x in h["history"]:
            c = x["cause"]
            L.append(f"  event {x['seq']}: {x['old']} -> {x['new']} by {x['actor']}; evidence {[S(e) for e in c.get('evidence_ids', [])] or '-'}; decision: {_clip(c.get('decision', ''), 200) or '-'}")
    L += ["", "## Probes"]
    for p in st.probes.values():
        L.append(f"{S(p['id'])} [{p['status']}] {p['purpose']} targets {[S(t) for t in p['target_hypothesis_ids']]}: {p['planned_action']}")
        L.append("  expected: " + " | ".join(f"{S(o['hypothesis_id'])}: {o['expect']}" for o in p["expected_outcomes"]) + f"  actual evidence: {[S(e) for e in p['actual_evidence_ids']] or '-'}")
    L += ["", "## Evidence"]
    for e in st.evidence.values():
        where = (f"{e['path']}:{e['start_line']}-{e['end_line']}" if e.get("start_line") else e.get("path")) or (f"`{e['command']}` exit={e['exit_code']} {e['duration_s']}s" if e.get("command") else e["type"])
        L.append(f"{S(e['id'])} [{e['type']}] event {e['created_at_step']} by {e['created_by']} | {where} | observation: {e['observation_state']}"
                 + (f" ({S(e['intervention_id'])})" if e.get("intervention_id") else "") + (f" | source call {S(e['source_tool_call_id'])}" if e.get("source_tool_call_id") else "")
                 + f" | hash {str(e.get('excerpt_hash'))[:10]}\n  summary: {e['summary']}\n  excerpt: {_clip(e.get('excerpt') or '', 700)}")
    L += ["", "## Relations"]
    for r in st.relations.values():
        L.append(f"{S(r['id'])} {S(r['from_id'])} --{r['type']}(strength {r['strength']})--> {S(r['to_id'])}: {_clip(r['rationale'], 220)}")
    L += ["", "## Branches and interventions"]
    for b in st.branches.values():
        rep = b.get("report") or {}
        L.append(f"{S(b['id'])} on {S(b['hypothesis_id'])}: outcome {rep.get('outcome')} stop {b.get('stop_reason')} usage {b.get('usage')}\n  summary: {_clip(rep.get('summary', ''), 500)}")
    for i in st.interventions.values():
        L.append(f"{S(i['id'])} in {S(i['branch_id'])}: {i['description']} (reset verified: {i['reset_verified']})")
    for r in st.resets:
        L.append(f"reset check at event {r['seq']} [{r['context']}]: verified clean = {r['ok']}")
    L += ["", "## Controller decisions and syntheses"]
    L += [f"event {d['seq']} {d['actor']}: {d['kind']} {_clip(d.get('rationale', ''), 160)}" for d in st.decisions]
    for s_ in st.syntheses.values():
        L.append(f"{S(s_['id'])}: reasons {s_['reasons']}; result {_clip(str(s_.get('result')), 300)}")
    dg = st.diagnosis
    L += ["", "## Committed diagnosis"]
    if dg:
        L += [f"{S(dg['id'])} status: {dg['status']} (event {dg['committed_at_step']})", "selected: " + str([S(h) for h in dg['selected_hypothesis_ids']]),
              "causal explanation: " + dg["causal_explanation"], "affected components: " + str(dg["affected_components"]),
              "requirements explained: " + str(dg["requirements_explained"]), "strongest evidence: " + str([S(e) for e in dg["strongest_evidence_ids"]]),
              "rejected alternatives: " + "; ".join(f"{S(r['hypothesis_id'])}: {r['reason']}" for r in dg["rejected_alternatives"]),
              "unresolved risks: " + str(dg["unresolved_risks"]), "implementation direction: " + dg["implementation_direction"]]
    L += ["", "## Timeline (event number, actor, type)"]
    L += [f"{d['seq']} {d['actor']}: {d['kind']}" for d in st.decisions]
    L += ["", "## Tool trace (investigator tool calls)"]
    for c in m.tool_calls:
        L.append(f"{S(c['id'])} [{S(c['branch_id'] or '-')}{', ' + S(c['intervention_id']) if c.get('intervention_id') else ''}] {c['name']}({_clip(json.dumps(c.get('arguments'), default=str), 200)}) "
                 f"exit={c.get('exit_code')} -> {_clip(c.get('output', ''), 500)}")
    text = "\n".join(L)
    return text if len(text) <= MAX_RENDER else text[:MAX_RENDER] + "\n...[truncated]"


def blind_user_prompt(m: CandidateMaterial) -> str:
    patch = f"\n\n# The agent's final patch (for the 'implementation_followed_diagnosis' field only)\n```diff\n{m.implementation_patch[:12000]}\n```" if m.implementation_patch else ""
    return f"# Issue\n{m.issue}\n\n{render_candidate(m)}{patch}\n\nEvent numbers run 1..{m.event_count}."


def gold_hunks(patch: str) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    cur = None
    for line in patch.splitlines():
        m = re.match(r"diff --git a/(\S+) b/", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, [])
        h = re.match(r"@@ -(\d+)(?:,(\d+))? \+", line)
        if h and cur:
            out[cur].append((int(h.group(1)), int(h.group(1)) + int(h.group(2) or 1)))
    return out


def base_excerpts(repo_dir: Path, gold_patch: str, context: int = 60, max_file_lines: int = 400) -> dict[str, str]:
    """Pre-fix code of the files the gold patch touches, with line numbers (the oracle's view of 'the base repository')."""
    res = {}
    for path, hunks in gold_hunks(gold_patch).items():
        f = Path(repo_dir) / path
        if not f.is_file():
            continue
        lines = f.read_text(errors="replace").splitlines()
        spans = [(1, len(lines))] if len(lines) <= max_file_lines else [(max(1, a - context), min(len(lines), b + context)) for a, b in hunks]
        res[path] = "\n...\n".join("\n".join(f"{i:>5}  {lines[i - 1]}" for i in range(a, b + 1)) for a, b in spans)
    return res


def reference_user_prompt(instance: dict, excerpts: dict[str, str]) -> str:
    code = "\n\n".join(f"## {p} (before the fix)\n```\n{t}\n```" for p, t in excerpts.items()) or "(no base-code excerpts available)"
    return (f"# Issue\n{instance['problem_statement']}\n\n# Repository code before the fix\n{code}\n\n# Reference fix (gold patch)\n```diff\n{instance['patch']}\n```\n\n"
            f"# Tests that define success\n```diff\n{instance.get('test_patch') or '(none)'}\n```\nFAIL_TO_PASS (fail before, must pass after): {instance['FAIL_TO_PASS']}\n"
            f"PASS_TO_PASS (must keep passing): {instance['PASS_TO_PASS'][:40]}")


def oracle_user_prompt(instance: dict, reference: dict, official: dict, m: CandidateMaterial) -> str:
    off = json.dumps({k: official.get(k) for k in ("status", "resolved", "f2p_passed", "f2p_total", "p2p_passed", "p2p_total")}, default=str)
    return (f"# Issue\n{instance['problem_statement']}\n\n# Reference causal account (written independently, before the agent's work was shown)\n{json.dumps(reference, indent=1)}\n\n"
            f"# Gold patch\n```diff\n{instance['patch']}\n```\nFAIL_TO_PASS: {instance['FAIL_TO_PASS']}\n\n# Official grading result of the agent's final patch\n{off}\n\n"
            f"---\n# THE AGENT'S WORK\n\n{render_candidate(m)}\n\n# The agent's final patch\n```diff\n{m.implementation_patch[:12000]}\n```\n\nEvent numbers run 1..{m.event_count}.")
