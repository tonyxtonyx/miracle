"""A deterministic stand-in judge for PLUMBING checks only (toy runs, UI development). Its scores are constants and say nothing about quality;
every justification states that. Outputs satisfy the schemas and cite ids that exist in the prompt, so the whole pipeline can be exercised offline."""
from __future__ import annotations

import re

from .schema import BLIND_DIMS, ORACLE_DIMS

NOTE = "MOCK JUDGE: constant score for plumbing checks, not a judgement"


def deterministic_mock(name: str, system: str, user: str, schema: dict) -> dict:
    hid = (re.findall(r"^(H\d+) \[", user, re.M) or ["H1"])[0]
    eid = (re.findall(r"^(E\d+) \[", user, re.M) or ["E1"])[0]
    pid = (re.findall(r"^(P\d+) \[", user, re.M) or ["P1"])[0]
    n = int((re.search(r"Event numbers run 1\.\.(\d+)", user) or [0, 1])[1]) if name != "reference" else 1
    if name == "reference":
        return {"causal_mechanism": NOTE, "key_locations": [], "issue_requirements": [], "facts_established": [], "inferences_from_gold_patch": [],
                "uncertainties": [NOTE], "alternative_valid_explanations": [], "what_a_correct_implementation_must_do": NOTE}
    dims = BLIND_DIMS if name == "blind" else ORACLE_DIMS
    scores = {d: {"score": 2, "justification": NOTE, "cited_ids": [hid]} for d in dims}
    if name == "blind":
        return {"scores": scores, "major_diagnostic_errors": [], "unsupported_conclusions": [], "missed_hypotheses": [], "strongest_part": {"text": NOTE, "cited_ids": [pid]},
                "earliest_supported_event_seq": max(1, n // 2), "implementation_followed_diagnosis": {"value": "not_applicable", "justification": NOTE}, "overall_assessment": NOTE}
    return {"scores": scores, "accepted_despite_contradiction": [], "major_diagnostic_errors": [], "missed_hypotheses": [],
            "hindsight_leakage": {"level": "none", "text": NOTE, "cited_ids": []}, "earliest_sufficient_event_seq": max(1, n // 2),
            "investigation_after_sufficient": NOTE, "implementation_followed_diagnosis": {"value": "not_applicable", "justification": NOTE}, "overall_assessment": NOTE}
