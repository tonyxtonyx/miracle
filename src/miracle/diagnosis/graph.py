"""The reasoning graph: entities, vocabularies, and a state derived by replaying an append-only event log.

`GraphState.apply(event)` is a pure function of the event (timestamps live in the event, never generated here), so
`replay(events, upto=n)` reproduces the graph exactly at any step: the web UI's replay and the judges rely on that.
Nothing is ever deleted: refuted hypotheses and contradicting evidence stay in the state (and in the log).
"""
from __future__ import annotations

import json
import re
from collections import Counter

HYP_TYPES = ("root_cause", "mechanism", "implementation_gap", "location")
HYP_STATUSES = ("proposed", "investigating", "supported", "weakened", "refuted", "inconclusive", "selected")
ACTIVE_STATUSES = ("proposed", "investigating", "supported", "weakened", "inconclusive")
SUPPORT_LEVELS = ("none", "low", "medium", "high")
EVIDENCE_TYPES = ("issue_statement", "code_excerpt", "test_result", "runtime_observation", "configuration",
                  "documentation", "intervention_result")
RELATION_TYPES = ("supports", "contradicts", "tests", "refines", "depends_on", "alternative_to", "explains")
PROBE_PURPOSES = ("falsify", "discriminate", "confirm_prediction", "reproduce")
PROBE_STATUSES = ("planned", "running", "completed", "abandoned")
# which (from-kind, to-kind) pairs each relation type may connect
RELATION_KINDS = {
    "supports": ("evidence", "hypothesis"), "contradicts": ("evidence", "hypothesis"), "explains": ("evidence", "hypothesis"),
    "tests": ("probe", "hypothesis"),
    "refines": ("hypothesis", "hypothesis"), "depends_on": ("hypothesis", "hypothesis"), "alternative_to": ("hypothesis", "hypothesis"),
}
KIND_PREFIX = {"H": "hypothesis", "E": "evidence", "R": "relation", "P": "probe", "IV": "intervention", "B": "branch",
               "D": "diagnosis", "S": "synthesis", "T": "tool_call"}


class GraphError(ValueError):
    """A rejected operation. The message is written for the model that issued it (it is returned as a tool error)."""


def short_id(full: str) -> str:
    return full.rsplit(":", 1)[-1]


class GraphState:
    def __init__(self, instance_id: str):
        self.iid = instance_id
        self.hypotheses: dict[str, dict] = {}
        self.evidence: dict[str, dict] = {}
        self.relations: dict[str, dict] = {}
        self.probes: dict[str, dict] = {}
        self.branches: dict[str, dict] = {}
        self.interventions: dict[str, dict] = {}
        self.syntheses: dict[str, dict] = {}
        self.diagnosis: dict | None = None
        self.decisions: list[dict] = []
        self.resets: list[dict] = []
        self.budget: dict = {}
        self.phase: str = "init"
        self.counters: Counter = Counter()
        self.seq = 0

    # ---- identity ----------------------------------------------------------------------------------------------
    def full_id(self, prefix: str, n: int) -> str:
        return f"{self.iid}:{prefix}{n}"

    def next_id(self, prefix: str) -> str:
        return self.full_id(prefix, self.counters[prefix] + 1)

    def resolve(self, ref: str) -> str:
        """Accept a short ID (`H3`) or a full one; return the full ID (existence is checked by the caller)."""
        ref = (ref or "").strip()
        if not ref:
            raise GraphError("empty reference")
        return ref if ":" in ref else f"{self.iid}:{ref.upper()}"

    def kind_of(self, full: str) -> str | None:
        m = re.match(r"^([A-Z]+)\d+$", short_id(full))
        return KIND_PREFIX.get(m.group(1)) if m else None

    def lookup(self, full: str) -> tuple[str, dict] | None:
        for kind, table in (("hypothesis", self.hypotheses), ("evidence", self.evidence), ("probe", self.probes)):
            if full in table:
                return kind, table[full]
        return None

    def _bump(self, full: str):
        m = re.match(r"^([A-Z]+)(\d+)$", short_id(full))
        if m:
            self.counters[m.group(1)] = max(self.counters[m.group(1)], int(m.group(2)))

    # ---- derived views -----------------------------------------------------------------------------------------
    def relations_of(self, node: str, *, direction: str = "any", rtype: str | None = None) -> list[dict]:
        out = []
        for r in self.relations.values():
            hit = (r["from_id"] == node and direction in ("any", "out")) or (r["to_id"] == node and direction in ("any", "in"))
            if hit and (rtype is None or r["type"] == rtype):
                out.append(r)
        return out

    def evidence_for(self, hid: str, rtype: str | None = None) -> list[dict]:
        return [self.evidence[r["from_id"]] for r in self.relations_of(hid, direction="in", rtype=rtype)
                if r["from_id"] in self.evidence]

    def active(self) -> list[dict]:
        return [h for h in self.hypotheses.values() if h["status"] in ACTIVE_STATUSES and not h.get("merged_into")]

    def depth(self, hid: str) -> int:
        parents = self.hypotheses[hid]["parent_ids"]
        return 0 if not parents else 1 + max(self.depth(p) for p in parents if p in self.hypotheses)

    # ---- event application ----------------------------------------------------------------------------------------
    def apply(self, ev: dict) -> None:
        d, seq = ev["data"], ev["seq"]
        if seq != self.seq + 1:
            raise GraphError(f"event sequence gap: expected {self.seq + 1}, got {seq}")
        self.seq = seq
        getattr(self, f"_on_{ev['type']}", self._on_unknown)(ev, d)

    def _on_unknown(self, ev, d):       # forward compatibility: unknown events are kept in the log, ignored by the state
        pass

    def _on_hypothesis_created(self, ev, d):
        self._bump(d["id"])
        self.hypotheses[d["id"]] = {
            "id": d["id"], "claim": d["claim"], "type": d["type"], "parent_ids": list(d.get("parent_ids", [])),
            "status": "proposed", "predictions": list(d["predictions"]),
            "falsification_criteria": [{"id": f"{d['id']}:F{i + 1}", "text": t} for i, t in enumerate(d["falsification_criteria"])],
            "support_level": "none", "created_at_step": ev["seq"], "last_updated_at_step": ev["seq"],
            "unresolved_questions": list(d.get("unresolved_questions", [])), "history": [], "created_by": ev["actor"],
            "rationale": d.get("rationale", ""), "merged_into": None}
        for p in d.get("parent_ids", []):
            self._link(d["id"], p, "refines", f"{short_id(d['id'])} refines {short_id(p)}", 2, ev, d.get("relation_ids", {}).get(p))

    def _link(self, frm, to, rtype, rationale, strength, ev, rid):
        if rid:
            self._bump(rid)
            self.relations[rid] = {"id": rid, "from_id": frm, "to_id": to, "type": rtype, "rationale": rationale,
                                   "strength": strength, "created_at_step": ev["seq"], "created_by": ev["actor"]}

    def _on_falsification_criterion_defined(self, ev, d):
        h = self.hypotheses[d["hypothesis_id"]]
        h["falsification_criteria"].append({"id": d["id"], "text": d["text"]})
        h["last_updated_at_step"] = ev["seq"]

    def _on_prediction_added(self, ev, d):
        h = self.hypotheses[d["hypothesis_id"]]
        h["predictions"].append(d["text"])
        h["last_updated_at_step"] = ev["seq"]

    def _on_probe_created(self, ev, d):
        self._bump(d["id"])
        self.probes[d["id"]] = {"id": d["id"], "target_hypothesis_ids": list(d["target_hypothesis_ids"]), "purpose": d["purpose"],
                                "planned_action": d["planned_action"], "expected_outcomes": d.get("expected_outcomes", []),
                                "actual_evidence_ids": [], "status": "planned", "metadata": {}, "created_at_step": ev["seq"],
                                "created_by": ev["actor"]}
        for hid in d["target_hypothesis_ids"]:
            self._link(d["id"], hid, "tests", f"{short_id(d['id'])} tests {short_id(hid)}", 2, ev, d["relation_ids"].get(hid))

    def _on_probe_updated(self, ev, d):
        p = self.probes[d["id"]]
        if "status" in d:
            p["status"] = d["status"]
        for e in d.get("add_evidence_ids", []):
            if e not in p["actual_evidence_ids"]:
                p["actual_evidence_ids"].append(e)
        p["metadata"].update(d.get("metadata", {}))

    def _on_evidence_attached(self, ev, d):
        self._bump(d["id"])
        self.evidence[d["id"]] = {**d, "created_at_step": ev["seq"], "created_by": ev["actor"]}

    def _on_relation_created(self, ev, d):
        self._bump(d["id"])
        self.relations[d["id"]] = {**d, "created_at_step": ev["seq"], "created_by": ev["actor"]}

    def _on_hypothesis_status_updated(self, ev, d):
        h = self.hypotheses[d["id"]]
        h["history"].append({"seq": ev["seq"], "old": h["status"], "new": d["new_status"], "cause": d["cause"],
                             "support_level": d.get("support_level"), "actor": ev["actor"]})
        h["status"] = d["new_status"]
        if d.get("support_level"):
            h["support_level"] = d["support_level"]
        h["last_updated_at_step"] = ev["seq"]

    def _on_hypothesis_questions_updated(self, ev, d):
        h = self.hypotheses[d["id"]]
        h["unresolved_questions"] = list(d["unresolved_questions"])
        h["last_updated_at_step"] = ev["seq"]

    def _on_hypotheses_merged(self, ev, d):
        h = self.hypotheses[d["absorbed_id"]]
        h["merged_into"] = d["kept_id"]
        h["history"].append({"seq": ev["seq"], "old": h["status"], "new": "inconclusive", "cause": {"decision": d["rationale"]},
                             "support_level": None, "actor": ev["actor"]})
        h["status"] = "inconclusive"
        h["last_updated_at_step"] = ev["seq"]

    def _on_branch_started(self, ev, d):
        self._bump(d["id"])
        self.branches[d["id"]] = {**d, "status": "running", "started_at_step": ev["seq"], "ended_at_step": None, "report": None}

    def _on_branch_ended(self, ev, d):
        b = self.branches[d["id"]]
        b.update(status="ended", ended_at_step=ev["seq"], report=d.get("report"), usage=d.get("usage"), stop_reason=d.get("stop_reason"))

    def _on_intervention_started(self, ev, d):
        self._bump(d["id"])
        self.interventions[d["id"]] = {**d, "started_at_step": ev["seq"], "ended_at_step": None, "reset_verified": None}

    def _on_intervention_ended(self, ev, d):
        self.interventions[d["id"]].update(ended_at_step=ev["seq"], reset_verified=d.get("reset_verified"), residual=d.get("residual"))

    def _on_reset_verified(self, ev, d):
        self.resets.append({**d, "seq": ev["seq"]})

    def _on_synthesis_requested(self, ev, d):
        self._bump(d["id"])
        self.syntheses[d["id"]] = {"id": d["id"], "reasons": d["reasons"], "requested_at_step": ev["seq"], "result": None}

    def _on_synthesis_completed(self, ev, d):
        self.syntheses[d["id"]]["result"] = d.get("summary")
        self.syntheses[d["id"]]["completed_at_step"] = ev["seq"]

    def _on_controller_decision(self, ev, d):
        self.decisions.append({**d, "seq": ev["seq"], "actor": ev["actor"]})

    def _on_budget_updated(self, ev, d):
        self.budget.update(d)

    def _on_phase_changed(self, ev, d):
        self.phase = d["phase"]

    def _on_diagnosis_committed(self, ev, d):
        self._bump(d["id"])
        self.diagnosis = {**d, "committed_at_step": ev["seq"]}
        for hid in d.get("selected_hypothesis_ids", []):
            h = self.hypotheses[hid]
            h["history"].append({"seq": ev["seq"], "old": h["status"], "new": "selected", "cause": {"decision": "diagnosis committed"},
                                 "support_level": None, "actor": ev["actor"]})
            h["status"] = "selected"
            h["last_updated_at_step"] = ev["seq"]

    # ---- serialisation -------------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"instance_id": self.iid, "seq": self.seq, "phase": self.phase, "hypotheses": self.hypotheses, "evidence": self.evidence,
                "relations": self.relations, "probes": self.probes, "branches": self.branches, "interventions": self.interventions,
                "syntheses": self.syntheses, "diagnosis": self.diagnosis, "decisions": self.decisions, "resets": self.resets,
                "budget": self.budget,
                "stats": {"hypotheses": len(self.hypotheses), "evidence": len(self.evidence), "relations": len(self.relations),
                          "probes": len(self.probes), "branches": len(self.branches),
                          "max_depth": max((self.depth(h) for h in self.hypotheses), default=0),
                          "by_status": dict(Counter(h["status"] for h in self.hypotheses.values()))}}


def replay(instance_id: str, events: list[dict], upto: int | None = None) -> GraphState:
    """State after the first `upto` events (all if None). Deterministic."""
    st = GraphState(instance_id)
    for ev in events[: (len(events) if upto is None else upto)]:
        st.apply(ev)
    return st


def read_events(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
