"""Judge output schemas (JSON Schema, also used for structured outputs) and validators. Citations are checked against the graph:
a finding that cites nothing, or an id that does not exist, is rejected and the judge is asked once to repair it."""
from __future__ import annotations

import re

from ..graph import GraphState

BLIND_DIMS = ["hypothesis_quality", "hypothesis_diversity", "falsifiability", "probe_discrimination", "diagnostic_depth", "evidence_quality",
              "evidence_provenance", "belief_update_consistency", "contradictory_evidence_handling", "debugging_quality",
              "uncertainty_calibration", "audit_clarity"]
ORACLE_DIMS = ["causal_correctness", "causal_localization", "mechanism_correctness", "requirement_coverage", "explains_failing_behavior",
               "important_alternatives_tested", "supports_correct_implementation"]
ID_RE = re.compile(r"^(?:[\w.\-]+:)?(H|E|P|R|B|T|IV|D|S)\d+$")


def _cited(desc="Identifiers that exist in the material, e.g. H2, E5, P1."):
    return {"type": "array", "items": {"type": "string"}, "description": desc}


def _scored(dims):
    one = {"type": "object", "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 4}, "justification": {"type": "string"},
                                            "cited_ids": _cited()}, "required": ["score", "justification", "cited_ids"], "additionalProperties": False}
    return {"type": "object", "properties": {d: one for d in dims}, "required": list(dims), "additionalProperties": False}


_finding = {"type": "object", "properties": {"text": {"type": "string"}, "cited_ids": _cited()}, "required": ["text", "cited_ids"], "additionalProperties": False}
_findings = {"type": "array", "items": _finding}
_implfollow = {"type": "object", "properties": {"value": {"type": "string", "enum": ["yes", "partial", "no", "not_applicable"]}, "justification": {"type": "string"}},
               "required": ["value", "justification"], "additionalProperties": False}

BLIND_SCHEMA = {"type": "object", "properties": {
    "scores": _scored(BLIND_DIMS), "major_diagnostic_errors": _findings, "unsupported_conclusions": _findings,
    "missed_hypotheses": {"type": "array", "items": {"type": "string"}}, "strongest_part": _finding,
    "earliest_supported_event_seq": {"type": ["integer", "null"]}, "implementation_followed_diagnosis": _implfollow,
    "overall_assessment": {"type": "string"}},
    "required": ["scores", "major_diagnostic_errors", "unsupported_conclusions", "missed_hypotheses", "strongest_part", "earliest_supported_event_seq",
                 "implementation_followed_diagnosis", "overall_assessment"], "additionalProperties": False}

_strs = {"type": "array", "items": {"type": "string"}}
REFERENCE_SCHEMA = {"type": "object", "properties": {
    "causal_mechanism": {"type": "string"}, "key_locations": _strs, "issue_requirements": _strs, "facts_established": _strs,
    "inferences_from_gold_patch": _strs, "uncertainties": _strs, "alternative_valid_explanations": _strs,
    "what_a_correct_implementation_must_do": {"type": "string"}},
    "required": ["causal_mechanism", "key_locations", "issue_requirements", "facts_established", "inferences_from_gold_patch", "uncertainties",
                 "alternative_valid_explanations", "what_a_correct_implementation_must_do"], "additionalProperties": False}

ORACLE_SCHEMA = {"type": "object", "properties": {
    "scores": _scored(ORACLE_DIMS), "accepted_despite_contradiction": _findings, "major_diagnostic_errors": _findings, "missed_hypotheses": _findings,
    "hindsight_leakage": {"type": "object", "properties": {"level": {"type": "string", "enum": ["none", "suspected", "clear"]}, "text": {"type": "string"},
                                                         "cited_ids": _cited()}, "required": ["level", "text", "cited_ids"], "additionalProperties": False},
    "earliest_sufficient_event_seq": {"type": ["integer", "null"]}, "investigation_after_sufficient": {"type": "string"},
    "implementation_followed_diagnosis": _implfollow, "overall_assessment": {"type": "string"}},
    "required": ["scores", "accepted_despite_contradiction", "major_diagnostic_errors", "missed_hypotheses", "hindsight_leakage",
                 "earliest_sufficient_event_seq", "investigation_after_sufficient", "implementation_followed_diagnosis", "overall_assessment"], "additionalProperties": False}


def _exists(st: GraphState, ref: str, event_count: int) -> str | None:
    """Full id if `ref` names an object in the graph, else None."""
    if not ID_RE.match(ref or ""):
        return None
    full = st.resolve(ref) if ":" not in ref else ref
    kind = st.kind_of(full)
    table = {"hypothesis": st.hypotheses, "evidence": st.evidence, "relation": st.relations, "probe": st.probes, "branch": st.branches,
             "intervention": st.interventions, "synthesis": st.syntheses}
    if kind == "diagnosis":
        return full if st.diagnosis and st.diagnosis["id"] == full else None
    if kind == "tool_call":
        return full if getattr(st, "_tool_call_ids", None) is None or full in st._tool_call_ids else None
    return full if kind in table and full in table[kind] else None


def validate_and_normalize(obj: dict, st: GraphState, kind: str, event_count: int) -> tuple[dict, list[str]]:
    """-> (object with citations normalised to full ids, problems). `kind` in blind|oracle|reference."""
    problems: list[str] = []
    out = obj

    def norm(ids, where, must=True):
        res = []
        for i in ids or []:
            full = _exists(st, str(i), event_count)
            if full is None:
                problems.append(f"{where}: cited id {i!r} does not exist in the graph")
            else:
                res.append(full)
        if must and not res:
            problems.append(f"{where}: must cite at least one existing id")
        return res

    if kind == "reference":
        return out, problems
    dims = BLIND_DIMS if kind == "blind" else ORACLE_DIMS
    for d in dims:
        s = (out.get("scores") or {}).get(d)
        if not isinstance(s, dict) or not isinstance(s.get("score"), int) or not 0 <= s["score"] <= 4:
            problems.append(f"scores.{d}: score must be an integer 0-4")
            continue
        s["cited_ids"] = norm(s.get("cited_ids"), f"scores.{d}", must=s["score"] <= 2 and d not in ("causal_correctness",))
    lists = ("major_diagnostic_errors", "unsupported_conclusions") if kind == "blind" else ("accepted_despite_contradiction", "major_diagnostic_errors", "missed_hypotheses")
    for key in lists:
        for n, f in enumerate(out.get(key) or []):
            f["cited_ids"] = norm(f.get("cited_ids"), f"{key}[{n}]", must=key != "missed_hypotheses")
    if kind == "blind":
        sp = out.get("strongest_part") or {}
        sp["cited_ids"] = norm(sp.get("cited_ids"), "strongest_part")
        seq = out.get("earliest_supported_event_seq")
        if seq is not None and not (isinstance(seq, int) and 1 <= seq <= event_count):
            problems.append(f"earliest_supported_event_seq must be null or an event number in 1..{event_count}")
    else:
        h = out.get("hindsight_leakage") or {}
        h["cited_ids"] = norm(h.get("cited_ids"), "hindsight_leakage", must=h.get("level") in ("suspected", "clear"))
        seq = out.get("earliest_sufficient_event_seq")
        if seq is not None and not (isinstance(seq, int) and 1 <= seq <= event_count):
            problems.append(f"earliest_sufficient_event_seq must be null or an event number in 1..{event_count}")
    return out, problems
