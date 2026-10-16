"""Validated graph operations. The ONLY way graph state changes: every call is checked, then appended to the event log and applied.

Provenance cannot be fabricated here: code excerpts are read by the runtime (`source.read_excerpt`), tool-derived evidence must point at a
recorded tool call and quote it verbatim, issue evidence must quote the issue verbatim. Statuses change only with a recorded cause.
Rejections are returned as `GraphError` (written for the model) and also logged as `operation_rejected` events for the judges.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .config import DiagnosisConfig
from .graph import (ACTIVE_STATUSES, EVIDENCE_TYPES, HYP_STATUSES, HYP_TYPES, PROBE_PURPOSES, PROBE_STATUSES, RELATION_KINDS,
                    RELATION_TYPES, SUPPORT_LEVELS, GraphError, GraphState, short_id)

ROLES = {
    "controller": {"create_hypothesis", "define_falsification_criterion", "create_probe", "create_relation", "update_hypothesis_status",
                   "set_unresolved_questions", "merge_hypotheses", "request_global_synthesis", "commit_diagnosis", "select_branch",
                   "attach_evidence"},
    "synthesis": {"create_hypothesis", "define_falsification_criterion", "create_probe", "create_relation", "update_hypothesis_status",
                  "set_unresolved_questions", "merge_hypotheses"},
    "investigator": {"attach_evidence", "create_relation"},
    "runtime": {"*"},
}
TRANSITIONS = {
    "proposed": {"investigating", "refuted", "weakened", "inconclusive", "supported"},
    "investigating": {"supported", "weakened", "refuted", "inconclusive", "proposed"},
    "supported": {"weakened", "refuted", "investigating", "inconclusive"},
    "weakened": {"investigating", "refuted", "supported", "inconclusive"},
    "inconclusive": {"investigating", "refuted", "weakened", "supported"},
    "refuted": {"investigating"},                     # reopen: needs evidence newer than the refutation
    "selected": set(),
}
LEVEL_FOR = {"supported": "medium", "weakened": "low", "refuted": "none"}


class EvidenceSource(Protocol):
    def read_excerpt(self, path: str, start: int, end: int) -> dict:
        """-> {"path", "start_line", "end_line", "excerpt", "repo_state"}; raises GraphError for bad paths/ranges."""
    def tool_call(self, call_id: str) -> dict | None:
        """-> the recorded tool call {id, branch_id, name, command, exit_code, duration_s, output, intervention_id, ...} or None."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def as_list(v, name: str) -> list:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, (list, tuple)):
        return list(v)
    raise GraphError(f"{name} must be a list of strings")


class GraphOps:
    def __init__(self, state: GraphState, log_path: Path | None, issue_text: str, cfg: DiagnosisConfig,
                 source: EvidenceSource | None = None, clock=time.time):
        self.state, self.issue, self.cfg, self.source, self.clock = state, issue_text, cfg, source, clock
        self.log_path = Path(log_path) if log_path else None
        self.visible: dict[str, set[str]] = {}          # actor -> hypothesis ids it may relate evidence to
        self.rejections = 0
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("")

    # ---- plumbing --------------------------------------------------------------------------------------------------
    def emit(self, etype: str, actor: str, data: dict) -> dict:
        ev = {"seq": self.state.seq + 1, "type": etype, "actor": actor,
              "ts": datetime.fromtimestamp(self.clock(), timezone.utc).isoformat(timespec="seconds"), "data": data}
        self.state.apply(ev)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    def _alloc(self, prefix: str, n: int = 1) -> list[str]:
        base = self.state.counters[prefix]
        return [self.state.full_id(prefix, base + i + 1) for i in range(n)]

    def reject(self, op: str, actor: str, msg: str):
        self.rejections += 1
        self.emit("operation_rejected", actor, {"op": op, "error": msg})
        err = GraphError(msg)
        err.logged = True                      # the tool layer must not log it a second time
        raise err

    def _guard(self, op: str, actor: str):
        role = actor.split(":")[0]
        allowed = ROLES.get(role, set())
        if "*" not in allowed and op not in allowed:
            self.reject(op, actor, f"{role} is not allowed to {op}")

    def ref(self, ref: str, kind: str | None = None) -> str:
        full = self.state.resolve(ref)
        found = self.state.lookup(full)
        if found is None:
            raise GraphError(f"unknown id {short_id(full)!r}")
        if kind and found[0] != kind:
            raise GraphError(f"{short_id(full)} is a {found[0]}, expected a {kind}")
        return full

    def _text(self, v, name: str, minimum: int | None = None, maximum: int = 1500) -> str:
        v = (v or "").strip() if isinstance(v, str) else ""
        m = self.cfg.min_text_chars if minimum is None else minimum
        if len(v) < m:
            raise GraphError(f"{name} is required and must be at least {m} characters (be specific)")
        return v[:maximum]

    def _wrap(self, op: str, actor: str, fn):
        self._guard(op, actor)
        try:
            return fn()
        except GraphError as e:
            if str(e).startswith(f"{actor.split(':')[0]} is not allowed"):
                raise
            self.reject(op, actor, str(e))

    # ---- hypotheses -------------------------------------------------------------------------------------------------
    def create_hypothesis(self, actor, claim, type, predictions, falsification_criteria, parent_ids=(), unresolved_questions=(), rationale=""):
        def run():
            st = self.state
            claim_ = self._text(claim, "claim", 15, 600)
            if type not in HYP_TYPES:
                raise GraphError(f"type must be one of {', '.join(HYP_TYPES)}")
            preds = [self._text(p, "each prediction", 8, 400) for p in as_list(predictions, "predictions")]
            crit = [self._text(c, "each falsification criterion", 8, 400) for c in as_list(falsification_criteria, "falsification_criteria")]
            if not preds:
                raise GraphError("give at least one observable prediction: what would you see if this hypothesis is true?")
            if not crit:
                raise GraphError("give at least one falsification criterion: what observation would prove this hypothesis wrong?")
            parents = [self.ref(p, "hypothesis") for p in as_list(parent_ids, "parent_ids")]
            if len(st.hypotheses) >= self.cfg.max_hypotheses:
                raise GraphError(f"hypothesis limit ({self.cfg.max_hypotheses}) reached: refine or merge existing hypotheses instead")
            if parents and 1 + max(st.depth(p) for p in parents) > self.cfg.max_depth:
                raise GraphError(f"refinement depth limit ({self.cfg.max_depth}) reached")
            for h in st.active():
                if difflib.SequenceMatcher(None, h["claim"].lower(), claim_.lower()).ratio() >= self.cfg.duplicate_similarity:
                    raise GraphError(f"this overlaps {short_id(h['id'])} ({h['claim'][:80]!r}): refine it (parent_ids) or state a distinct claim")
            hid, = self._alloc("H")
            rids = dict(zip(parents, self._alloc("R", len(parents))))
            self.emit("hypothesis_created", actor, {"id": hid, "claim": claim_, "type": type, "parent_ids": parents, "predictions": preds,
                                                    "falsification_criteria": crit, "unresolved_questions": as_list(unresolved_questions, "unresolved_questions"),
                                                    "rationale": (rationale or "")[:600], "relation_ids": rids})
            return {"id": short_id(hid)}
        return self._wrap("create_hypothesis", actor, run)

    def define_falsification_criterion(self, actor, hypothesis_id, text):
        def run():
            hid = self.ref(hypothesis_id, "hypothesis")
            t = self._text(text, "falsification criterion", 8, 400)
            n = len(self.state.hypotheses[hid]["falsification_criteria"]) + 1
            fid = f"{hid}:F{n}"
            self.emit("falsification_criterion_defined", actor, {"hypothesis_id": hid, "id": fid, "text": t})
            return {"id": short_id(fid)}
        return self._wrap("define_falsification_criterion", actor, run)

    def set_unresolved_questions(self, actor, hypothesis_id, questions):
        def run():
            hid = self.ref(hypothesis_id, "hypothesis")
            self.emit("hypothesis_questions_updated", actor, {"id": hid, "unresolved_questions": [str(q)[:400] for q in as_list(questions, "questions")]})
            return {"id": short_id(hid)}
        return self._wrap("set_unresolved_questions", actor, run)

    def merge_hypotheses(self, actor, kept_id, absorbed_id, rationale):
        def run():
            k, a = self.ref(kept_id, "hypothesis"), self.ref(absorbed_id, "hypothesis")
            if k == a:
                raise GraphError("cannot merge a hypothesis into itself")
            for x in (k, a):
                if self.state.hypotheses[x]["status"] not in ACTIVE_STATUSES:
                    raise GraphError(f"{short_id(x)} is {self.state.hypotheses[x]['status']}; only active hypotheses can be merged")
            self.emit("hypotheses_merged", actor, {"kept_id": k, "absorbed_id": a, "rationale": self._text(rationale, "rationale", 15)})
            return {"kept": short_id(k), "absorbed": short_id(a)}
        return self._wrap("merge_hypotheses", actor, run)

    # ---- probes ---------------------------------------------------------------------------------------------------
    def create_probe(self, actor, target_hypothesis_ids, purpose, planned_action, expected_outcomes):
        def run():
            targets = list(dict.fromkeys(self.ref(t, "hypothesis") for t in as_list(target_hypothesis_ids, "target_hypothesis_ids")))
            if not targets:
                raise GraphError("a probe needs at least one target hypothesis")
            if purpose not in PROBE_PURPOSES:
                raise GraphError(f"purpose must be one of {', '.join(PROBE_PURPOSES)}")
            if purpose == "discriminate" and len(targets) < 2:
                raise GraphError("a 'discriminate' probe must target at least two competing hypotheses")
            action = self._text(planned_action, "planned_action", 15, 800)
            outs = []
            for o in as_list(expected_outcomes, "expected_outcomes"):
                if not isinstance(o, dict):
                    raise GraphError("expected_outcomes must be a list of {hypothesis_id, expect}")
                hid = self.ref(o.get("hypothesis_id", ""), "hypothesis")
                outs.append({"hypothesis_id": hid, "expect": self._text(o.get("expect"), "expect", 8, 400)})
            covered = {o["hypothesis_id"] for o in outs}
            if covered != set(targets):
                raise GraphError("give an expected outcome for EACH target hypothesis (what you would observe if that hypothesis is true)")
            if purpose == "discriminate" and len({o["expect"].lower() for o in outs}) < 2:
                raise GraphError("a discriminating probe needs different expected outcomes for different hypotheses")
            pid, = self._alloc("P")
            self.emit("probe_created", actor, {"id": pid, "target_hypothesis_ids": targets, "purpose": purpose, "planned_action": action,
                                               "expected_outcomes": outs, "relation_ids": dict(zip(targets, self._alloc("R", len(targets))))})
            return {"id": short_id(pid)}
        return self._wrap("create_probe", actor, run)

    def update_probe(self, actor, probe_id, status=None, add_evidence_ids=(), metadata=None):
        pid = self.ref(probe_id, "probe")
        if status and status not in PROBE_STATUSES:
            raise GraphError(f"status must be one of {', '.join(PROBE_STATUSES)}")
        data = {"id": pid, "add_evidence_ids": list(add_evidence_ids), "metadata": metadata or {}}
        if status:
            data["status"] = status
        self.emit("probe_updated", actor, data)

    # ---- evidence -------------------------------------------------------------------------------------------------
    def attach_evidence(self, actor, type, summary, path=None, start_line=None, end_line=None, source_tool_call_id=None,
                        quote=None, probe_id=None, branch_id=None):
        def run():
            if type not in EVIDENCE_TYPES:
                raise GraphError(f"type must be one of {', '.join(EVIDENCE_TYPES)}")
            summ = self._text(summary, "summary", 10, 500)
            eid, = self._alloc("E")
            data = {"id": eid, "type": type, "summary": summ, "source_tool_call_id": None, "repo_state": None, "path": None,
                    "start_line": None, "end_line": None, "excerpt": None, "excerpt_hash": None, "command": None, "exit_code": None,
                    "duration_s": None, "output_excerpt": None, "intervention_id": None, "observation_state": "original",
                    "branch_id": branch_id, "probe_id": None}
            if type == "issue_statement":
                q = self._text(quote, "quote", 8, 2000)
                pos = self.issue.find(q)
                if pos < 0:
                    raise GraphError("the quote is not an exact substring of the issue; copy the text verbatim")
                data.update(excerpt=q, excerpt_hash=sha(q), start_line=None, end_line=None, char_start=pos, char_end=pos + len(q),
                            repo_state={"kind": "issue"})
            elif type in ("code_excerpt", "configuration", "documentation"):
                if not path:
                    raise GraphError("path, start_line and end_line are required: the runtime reads the excerpt from the repository for you")
                try:
                    s_, e_ = int(start_line), int(end_line)
                except (TypeError, ValueError):
                    raise GraphError("start_line and end_line must be integers")
                if self.source is None:
                    raise GraphError("no repository is attached")
                got = self.source.read_excerpt(path, s_, e_)
                st_ = got["repo_state"]
                data.update(path=got["path"], start_line=got["start_line"], end_line=got["end_line"], excerpt=got["excerpt"],
                            excerpt_hash=sha(got["excerpt"]), repo_state=st_, intervention_id=st_.get("intervention_id"),
                            observation_state="post_intervention" if st_.get("intervention_id") else "original")
            else:
                if not source_tool_call_id:
                    raise GraphError("source_tool_call_id is required: cite the tool call (for example T7) that produced this observation")
                call = self.source.tool_call(source_tool_call_id) if self.source else None
                if call is None:
                    raise GraphError(f"unknown tool call {source_tool_call_id!r}")
                if branch_id and call.get("branch_id") != branch_id:
                    raise GraphError("that tool call belongs to a different investigation branch")
                if type == "test_result" and call["name"] != "run_command":
                    raise GraphError("a test_result must cite a run_command call")
                if type == "intervention_result" and not call.get("intervention_id"):
                    raise GraphError("that call ran on the original repository (no temporary change was open); use runtime_observation")
                out = call.get("output", "")
                if quote:
                    if quote not in out:
                        raise GraphError("the quote is not an exact substring of that tool call's output")
                    ex = quote
                else:
                    ex = out if len(out) <= 1500 else out[:700] + "\n...[truncated]...\n" + out[-700:]
                data.update(source_tool_call_id=call["id"], repo_state=call.get("repo_state"), command=call.get("command"),
                            exit_code=call.get("exit_code"), duration_s=call.get("duration_s"), output_excerpt=ex, excerpt=ex,
                            excerpt_hash=sha(ex), intervention_id=call.get("intervention_id"),
                            observation_state="post_intervention" if call.get("intervention_id") else "original")
                if call["name"] in ("read_file", "search_code") and call.get("path"):
                    data["path"] = call["path"]
            if probe_id:
                data["probe_id"] = self.ref(probe_id, "probe")
            self.emit("evidence_attached", actor, data)
            if data["probe_id"]:
                self.emit("probe_updated", actor, {"id": data["probe_id"], "add_evidence_ids": [eid], "metadata": {}})
            return {"id": short_id(eid)}
        return self._wrap("attach_evidence", actor, run)

    # ---- relations ---------------------------------------------------------------------------------------------------
    def create_relation(self, actor, from_id, to_id, type, rationale, strength=2):
        def run():
            st = self.state
            if type not in RELATION_TYPES:
                raise GraphError(f"relation type must be one of {', '.join(RELATION_TYPES)}")
            a, b = self.ref(from_id), self.ref(to_id)
            ka, kb = st.lookup(a)[0], st.lookup(b)[0]
            if (ka, kb) != RELATION_KINDS[type]:
                raise GraphError(f"'{type}' links a {RELATION_KINDS[type][0]} to a {RELATION_KINDS[type][1]}, not a {ka} to a {kb}")
            if a == b:
                raise GraphError("a node cannot be related to itself")
            try:
                s_ = int(strength)
            except (TypeError, ValueError):
                s_ = 2
            if s_ not in (1, 2, 3):
                raise GraphError("strength must be 1 (weak), 2 (moderate) or 3 (strong)")
            why = self._text(rationale, "rationale", 12, 600)
            role = actor.split(":")[0]
            if role == "investigator":
                if ka != "evidence" or st.evidence[a].get("created_by") != actor:
                    raise GraphError("you can only relate evidence you recorded yourself")
                if b not in self.visible.get(actor, set()):
                    raise GraphError(f"{short_id(b)} is not part of your investigation context")
            for r in st.relations.values():
                if r["type"] == type and ((r["from_id"], r["to_id"]) == (a, b) or (type == "alternative_to" and (r["from_id"], r["to_id"]) == (b, a))):
                    raise GraphError("that relation already exists")
                opposite = {"supports": "contradicts", "contradicts": "supports"}.get(type)
                if opposite and r["type"] == opposite and (r["from_id"], r["to_id"]) == (a, b):
                    raise GraphError(f"{short_id(a)} already {opposite} {short_id(b)} ({short_id(r['id'])}); the same evidence cannot both support and contradict one hypothesis. "
                                     f"If {short_id(b)} duplicates another hypothesis, merge them with merge_hypotheses instead of refuting it.")
            if type == "refines" and st.depth(a) + 1 > self.cfg.max_depth:
                raise GraphError(f"refinement depth limit ({self.cfg.max_depth}) reached")
            if type == "refines" and self._reaches(b, a):
                raise GraphError("that refinement would create a cycle")
            rid, = self._alloc("R")
            self.emit("relation_created", actor, {"id": rid, "from_id": a, "to_id": b, "type": type, "rationale": why, "strength": s_})
            return {"id": short_id(rid)}
        return self._wrap("create_relation", actor, run)

    def _reaches(self, frm: str, target: str) -> bool:
        seen, stack = set(), [frm]
        while stack:
            n = stack.pop()
            if n == target:
                return True
            if n in seen or n not in self.state.hypotheses:
                continue
            seen.add(n)
            stack.extend(self.state.hypotheses[n]["parent_ids"])
        return False

    # ---- status ---------------------------------------------------------------------------------------------------
    def update_hypothesis_status(self, actor, hypothesis_id, new_status, cause_evidence_ids=(), rationale="", support_level=None):
        def run():
            st = self.state
            hid = self.ref(hypothesis_id, "hypothesis")
            h = st.hypotheses[hid]
            if new_status not in HYP_STATUSES or new_status == "selected":
                raise GraphError("status must be one of proposed, investigating, supported, weakened, refuted, inconclusive ('selected' is set only by commit_diagnosis)")
            if new_status not in TRANSITIONS[h["status"]]:
                raise GraphError(f"cannot move {short_id(hid)} from {h['status']} to {new_status}")
            ev_ids = [self.ref(e, "evidence") for e in as_list(cause_evidence_ids, "cause_evidence_ids")]
            why = (rationale or "").strip()
            linked = lambda rtype: [e for e in ev_ids if any(r["from_id"] == e and r["to_id"] == hid and r["type"] == rtype for r in st.relations.values())]
            if new_status == "refuted":
                con = linked("contradicts")
                strong = [e for e in con if any(r["from_id"] == e and r["to_id"] == hid and r["type"] == "contradicts" and r["strength"] >= 2 for r in st.relations.values())]
                if not (strong or len(con) >= 2):
                    raise GraphError("refuting needs cause_evidence_ids with a 'contradicts' relation to this hypothesis (strength >= 2, or two pieces of evidence); relate the evidence first")
                if h["status"] == "refuted":
                    raise GraphError("already refuted")
            elif new_status == "weakened":
                if not linked("contradicts"):
                    raise GraphError("weakening needs cause_evidence_ids with a 'contradicts' relation to this hypothesis")
            elif new_status == "supported":
                sup = [e for e in linked("supports") if st.evidence[e]["type"] != "issue_statement"]
                if not sup:
                    raise GraphError("'supported' needs non-issue cause_evidence_ids with a 'supports' relation to this hypothesis")
            else:
                if len(why) < 10:
                    raise GraphError("give a rationale (>= 10 characters) for this decision")
                if h["status"] == "refuted" and new_status == "investigating":
                    last = max(x["seq"] for x in h["history"] if x["new"] == "refuted")
                    if not any(st.evidence[e]["created_at_step"] > last for e in ev_ids):
                        raise GraphError("reopening a refuted hypothesis needs cause_evidence_ids recorded AFTER the refutation")
            level = support_level or LEVEL_FOR.get(new_status)
            if level and level not in SUPPORT_LEVELS:
                raise GraphError(f"support_level must be one of {', '.join(SUPPORT_LEVELS)}")
            self.emit("hypothesis_status_updated", actor, {"id": hid, "new_status": new_status, "support_level": level,
                                                          "cause": {"evidence_ids": ev_ids, "decision": why[:600]}})
            return {"id": short_id(hid), "status": new_status}
        return self._wrap("update_hypothesis_status", actor, run)

    # ---- synthesis / branches / bookkeeping (runtime-driven) ----------------------------------------------------------------
    def request_global_synthesis(self, actor, reasons):
        def run():
            rs = [self._text(r, "reason", 6, 300) for r in as_list(reasons, "reasons")]
            if not rs:
                raise GraphError("give at least one reason")
            sid, = self._alloc("S")
            self.emit("synthesis_requested", actor, {"id": sid, "reasons": rs})
            return {"id": short_id(sid)}
        return self._wrap("request_global_synthesis", actor, run)

    def complete_synthesis(self, sid, summary):
        self.emit("synthesis_completed", "runtime", {"id": self.ref(sid) if ":" not in sid else sid, "summary": summary})

    def decide(self, actor, kind, rationale="", **data):
        self.emit("controller_decision", actor, {"kind": kind, "rationale": rationale[:800], **data})

    def start_branch(self, hypothesis_id, probe_ids, budget):
        bid, = self._alloc("B")
        self.emit("branch_started", "runtime", {"id": bid, "hypothesis_id": hypothesis_id, "probe_ids": list(probe_ids), "budget": budget})
        return bid

    def end_branch(self, bid, report, usage, stop_reason):
        self.emit("branch_ended", "runtime", {"id": bid, "report": report, "usage": usage, "stop_reason": stop_reason})

    def start_intervention(self, branch_id, description, baseline):
        iid, = self._alloc("IV")
        self.emit("intervention_started", "runtime", {"id": iid, "branch_id": branch_id, "description": description[:300], "baseline": baseline})
        return iid

    def end_intervention(self, iid, reset_verified, residual=None):
        self.emit("intervention_ended", "runtime", {"id": iid, "reset_verified": reset_verified, "residual": residual})

    def record_reset(self, context, ok, details):
        self.emit("reset_verified", "runtime", {"context": context, "ok": ok, "details": details})

    def set_phase(self, phase):
        self.emit("phase_changed", "runtime", {"phase": phase})

    def set_budget(self, **b):
        self.emit("budget_updated", "runtime", b)

    # ---- diagnosis ------------------------------------------------------------------------------------------------
    def commit_diagnosis(self, actor, selected_hypothesis_ids, causal_explanation, affected_components, requirements_explained,
                         strongest_evidence_ids, rejected_alternatives, unresolved_risks, implementation_direction):
        def run():
            st = self.state
            if st.diagnosis:
                raise GraphError("a diagnosis is already committed")
            sel = list(dict.fromkeys(self.ref(h, "hypothesis") for h in as_list(selected_hypothesis_ids, "selected_hypothesis_ids")))
            if not sel:
                raise GraphError("select at least one hypothesis")
            for h in sel:
                if st.hypotheses[h]["status"] != "supported":
                    raise GraphError(f"{short_id(h)} is {st.hypotheses[h]['status']}: only a 'supported' hypothesis can be selected (update its status with evidence first)")
                con = st.evidence_for(h, "contradicts")
                if any(r["strength"] >= 3 for r in st.relations_of(h, direction="in", rtype="contradicts")):
                    raise GraphError(f"{short_id(h)} has strong contradicting evidence; resolve it before committing")
            expl = self._text(causal_explanation, "causal_explanation", 40, 3000)
            comps = [self._text(c, "each affected component", 3, 200) for c in as_list(affected_components, "affected_components")]
            reqs = [self._text(r, "each requirement", 8, 400) for r in as_list(requirements_explained, "requirements_explained")]
            if not comps or not reqs:
                raise GraphError("affected_components and requirements_explained must not be empty")
            ev = [self.ref(e, "evidence") for e in as_list(strongest_evidence_ids, "strongest_evidence_ids")]
            if not ev:
                raise GraphError("cite the strongest supporting evidence (strongest_evidence_ids)")
            if not any(st.evidence[e]["type"] != "issue_statement" and any(r["from_id"] == e and r["to_id"] in sel and r["type"] in ("supports", "explains")
                                                                       for r in st.relations.values()) for e in ev):
                raise GraphError("strongest_evidence_ids must include non-issue evidence linked (supports/explains) to a selected hypothesis")
            rej, listed = [], set()
            for item in as_list(rejected_alternatives, "rejected_alternatives"):
                if not isinstance(item, dict):
                    raise GraphError("rejected_alternatives must be a list of {hypothesis_id, reason}")
                hid = self.ref(item.get("hypothesis_id", ""), "hypothesis")
                if hid in sel:
                    raise GraphError(f"{short_id(hid)} is selected, it cannot also be rejected")
                rej.append({"hypothesis_id": hid, "reason": self._text(item.get("reason"), "reason", 15, 600)})
                listed.add(hid)
            missing = [short_id(h["id"]) for h in st.active() if h["id"] not in sel and h["id"] not in listed]
            if missing:
                raise GraphError(f"every other active hypothesis must be explicitly dispositioned in rejected_alternatives (or refuted first): {', '.join(missing)}")
            risks = [self._text(r, "each risk", 6, 500) for r in as_list(unresolved_risks, "unresolved_risks")]
            contra = [e for h in sel for e in st.evidence_for(h, "contradicts")]
            if contra and not risks:
                raise GraphError("a selected hypothesis has contradicting evidence: state it under unresolved_risks")
            did, = self._alloc("D")
            self.emit("diagnosis_committed", actor, {
                "id": did, "status": "committed", "selected_hypothesis_ids": sel, "causal_explanation": expl, "affected_components": comps,
                "requirements_explained": reqs, "strongest_evidence_ids": ev, "rejected_alternatives": rej, "unresolved_risks": risks,
                "implementation_direction": self._text(implementation_direction, "implementation_direction", 15, 1500)})
            return {"id": short_id(did)}
        return self._wrap("commit_diagnosis", actor, run)

    def commit_inconclusive(self, best_ids, reason: str, implementation_direction: str = ""):
        """Runtime path when the budget ends without a valid commit: an honest, clearly-labelled inconclusive diagnosis."""
        st = self.state
        sel = [h for h in best_ids if h in st.hypotheses]
        did, = self._alloc("D")
        self.emit("diagnosis_committed", "runtime", {
            "id": did, "status": "inconclusive", "selected_hypothesis_ids": [], "leading_hypothesis_ids": sel,
            "causal_explanation": f"INCONCLUSIVE: {reason}", "affected_components": [], "requirements_explained": [],
            "strongest_evidence_ids": [], "rejected_alternatives": [], "unresolved_risks": [reason],
            "implementation_direction": implementation_direction or "No validated direction; treat the leading hypotheses as unverified."})
        return {"id": short_id(did)}
