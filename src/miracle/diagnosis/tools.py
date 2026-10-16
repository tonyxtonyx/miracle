"""LLM-facing tools. Controller/synthesis tools are thin, validated wrappers over GraphOps (no repository access). Investigator tools are the
repository tools (the S0 `EnvToolBox`, wrapped to keep a ledger and manage interventions) plus a small evidence facade."""
from __future__ import annotations

import json
from collections import Counter

from ..agents.env_tools import EnvToolBox, _I, _S, _fn
from . import prompts as P
from .config import DiagnosisConfig
from .graph import EVIDENCE_TYPES, HYP_TYPES, PROBE_PURPOSES, GraphError, short_id
from .ops import GraphOps
from .sandbox import WRITE_TOOLS, InvestigationSandbox


def _A(desc, item=None):
    return {"type": "array", "items": item or {"type": "string"}, "description": desc}


def _coerce(args: dict, spec: dict) -> dict:
    """Small models often pass JSON-in-a-string or a bare string for a list: be tolerant on shape, strict on content."""
    props = spec["function"]["parameters"]["properties"]
    out = dict(args)
    for k, v in args.items():
        t = (props.get(k) or {}).get("type")
        if t == "array" and isinstance(v, str):
            try:
                parsed = json.loads(v)
                out[k] = parsed if isinstance(parsed, list) else [parsed]
            except ValueError:
                out[k] = [v] if v.strip() else []
        elif t == "integer" and isinstance(v, str) and v.strip().lstrip("-").isdigit():
            out[k] = int(v)
    return out


H_PROPS = {
    "claim": _S("The hypothesis: a specific causal claim (name functions/values)."),
    "type": {"type": "string", "enum": list(HYP_TYPES), "description": "root_cause | mechanism | implementation_gap | location"},
    "predictions": _A("Observable predictions: what you would see if the hypothesis is true."),
    "falsification_criteria": _A("Observations that would prove the hypothesis wrong."),
    "parent_ids": _A("IDs of hypotheses this one refines (optional)."),
    "unresolved_questions": _A("Open questions (optional)."),
    "rationale": _S("Why this hypothesis is plausible (optional)."),
}
CONTROLLER_SPECS = {
    "create_hypothesis": _fn("create_hypothesis", "Create a hypothesis. Needs predictions AND falsification criteria.", H_PROPS,
                             ["claim", "type", "predictions", "falsification_criteria"]),
    "define_falsification_criterion": _fn("define_falsification_criterion", "Add a falsification criterion to a hypothesis.",
                                          {"hypothesis_id": _S("e.g. H2"), "text": _S("An observation that would prove it wrong.")}, ["hypothesis_id", "text"]),
    "create_probe": _fn("create_probe", "Plan a probe (an experiment). A 'discriminate' probe targets >=2 hypotheses with DIFFERENT expected outcomes.", {
        "target_hypothesis_ids": _A("Hypothesis IDs the probe tests."), "purpose": {"type": "string", "enum": list(PROBE_PURPOSES)},
        "planned_action": _S("What to do (commands/files/inputs)."),
        "expected_outcomes": _A("One item per target: {hypothesis_id, expect} - what you would observe if THAT hypothesis is true.", {"type": "object"})},
        ["target_hypothesis_ids", "purpose", "planned_action", "expected_outcomes"]),
    "create_relation": _fn("create_relation", "Link nodes: evidence->hypothesis (supports|contradicts|explains), hypothesis->hypothesis (refines|depends_on|alternative_to).", {
        "from_id": _S("Source ID"), "to_id": _S("Target ID"), "type": {"type": "string", "enum": ["supports", "contradicts", "explains", "refines", "depends_on", "alternative_to"]},
        "rationale": _S("Why."), "strength": _I("1 weak, 2 moderate, 3 strong")}, ["from_id", "to_id", "type", "rationale"]),
    "update_hypothesis_status": _fn("update_hypothesis_status", "Change a hypothesis status. refuted/weakened need cause_evidence_ids linked with 'contradicts'; supported needs non-issue evidence linked with 'supports'.", {
        "hypothesis_id": _S("e.g. H1"), "new_status": {"type": "string", "enum": ["proposed", "investigating", "supported", "weakened", "refuted", "inconclusive"]},
        "cause_evidence_ids": _A("Evidence IDs that justify the change."), "rationale": _S("Why (required for investigating/inconclusive)."),
        "support_level": {"type": "string", "enum": ["none", "low", "medium", "high"]}}, ["hypothesis_id", "new_status"]),
    "set_unresolved_questions": _fn("set_unresolved_questions", "Replace the open questions of a hypothesis.", {"hypothesis_id": _S("ID"), "questions": _A("Questions")}, ["hypothesis_id", "questions"]),
    "merge_hypotheses": _fn("merge_hypotheses", "Merge a duplicate hypothesis into another.", {"kept_id": _S("ID"), "absorbed_id": _S("ID"), "rationale": _S("Why they are the same.")},
                            ["kept_id", "absorbed_id", "rationale"]),
    "done_planning": _fn("done_planning", "Finish the initial hypothesis/probe planning.", {}, []),
    "select_branch": _fn("select_branch", "Choose the hypothesis to investigate next.", {"hypothesis_id": _S("ID"), "probe_ids": _A("Probes to run (optional)."), "rationale": _S("Why this one.")}, ["hypothesis_id"]),
    "request_global_synthesis": _fn("request_global_synthesis", "Ask for a global synthesis over the whole graph.", {"reasons": _A("Why.")}, ["reasons"]),
    "commit_diagnosis": _fn("commit_diagnosis", "Commit the diagnosis. Selected hypotheses must be 'supported'; every other active hypothesis must be dispositioned in rejected_alternatives.", {
        "selected_hypothesis_ids": _A("Supported hypothesis IDs."), "causal_explanation": _S("The causal mechanism / implementation gap, and why the current code fails the issue."),
        "affected_components": _A("Files/functions/symbols involved."), "requirements_explained": _A("Which parts of the issue this explains."),
        "strongest_evidence_ids": _A("Evidence IDs."), "rejected_alternatives": _A("Items {hypothesis_id, reason}.", {"type": "object"}),
        "unresolved_risks": _A("Remaining uncertainty."), "implementation_direction": _S("What a fix should do.")},
        ["selected_hypothesis_ids", "causal_explanation", "affected_components", "requirements_explained", "strongest_evidence_ids", "rejected_alternatives", "implementation_direction"]),
    "finish_synthesis": _fn("finish_synthesis", "Finish the synthesis.", {"summary": _S("What you changed and why."), "inconclusive": {"type": "boolean"}}, ["summary"]),
}
PHASE_TOOLS = {
    "hypothesize": ["create_hypothesis", "define_falsification_criterion", "create_probe", "create_relation", "done_planning"],
    "review": ["update_hypothesis_status", "create_hypothesis", "define_falsification_criterion", "create_probe", "create_relation", "set_unresolved_questions",
               "merge_hypotheses", "select_branch", "request_global_synthesis", "commit_diagnosis"],
    "synthesis": ["create_hypothesis", "define_falsification_criterion", "create_probe", "create_relation", "update_hypothesis_status", "merge_hypotheses",
                  "set_unresolved_questions", "finish_synthesis"],
}


class ControllerTools:
    """Toolset for the controller (phases `hypothesize` and `review`) and for synthesis. No repository access."""

    def __init__(self, ops: GraphOps, cfg: DiagnosisConfig, phase: str):
        self.ops, self.cfg, self.phase = ops, cfg, phase
        self.actor = "synthesis" if phase == "synthesis" else "controller"
        self.specs = [CONTROLLER_SPECS[n] for n in PHASE_TOOLS[phase]]
        self.terminal: dict | None = None

    def call(self, name: str, args: dict) -> tuple:
        spec = CONTROLLER_SPECS.get(name)
        if spec is None or name not in PHASE_TOOLS[self.phase]:
            return f"ERROR: unknown tool {name!r}. Available: {', '.join(PHASE_TOOLS[self.phase])}", True
        args = _coerce(args, spec)
        try:
            out = self._dispatch(name, args)
            return (out[0] + self._hint(), *out[1:]) if self.terminal is None else out
        except GraphError as e:
            self._log_rejection(name, e)
            return f"ERROR: {e}", True
        except TypeError as e:
            self._log_rejection(name, e)
            return f"ERROR: bad arguments for {name}: {e}", True

    def _log_rejection(self, name: str, e: Exception) -> None:
        if not getattr(e, "logged", False):                     # rejections raised by the tool layer itself (caps, protocol rules, bad arguments)
            self.ops.rejections += 1
            self.ops.emit("operation_rejected", self.actor, {"op": name, "error": str(e)})

    def _hint(self) -> str:
        st = self.ops.state
        if self.phase == "hypothesize":
            return "\n" + P.HINT_PLANNING.format(n=len(st.hypotheses), cap=self.cfg.max_initial_hypotheses, p=len(st.probes), min_h=self.cfg.min_initial_hypotheses)
        return "\n" + (P.HINT_REVIEW if self.phase == "review" else P.HINT_SYNTHESIS)

    def _dispatch(self, name: str, a: dict) -> tuple:
        ops, st = self.ops, self.ops.state
        if name == "create_hypothesis" and self.phase == "hypothesize" and len(st.hypotheses) >= self.cfg.max_initial_hypotheses:
            raise GraphError(f"you already have {len(st.hypotheses)} hypotheses, the limit for the planning phase. Plan a discriminating probe with create_probe "
                             f"(if you have none) and call done_planning. More hypotheses can be added later, guided by evidence.")
        if name == "done_planning":
            if len(st.hypotheses) < self.cfg.min_initial_hypotheses:
                raise GraphError(f"create at least {self.cfg.min_initial_hypotheses} distinct hypotheses first (you have {len(st.hypotheses)})")
            if not any(p["purpose"] in ("discriminate", "falsify") for p in st.probes.values()):
                raise GraphError("plan at least one 'discriminate' or 'falsify' probe first")
            self.terminal = {"kind": "done_planning"}
            return "OK: planning complete", False
        if name == "select_branch":
            hid = ops.ref(a.get("hypothesis_id", ""), "hypothesis")
            if st.hypotheses[hid]["status"] not in ("proposed", "investigating", "weakened", "inconclusive", "supported"):
                raise GraphError(f"{short_id(hid)} is {st.hypotheses[hid]['status']}; choose an active hypothesis")
            probes = [ops.ref(p, "probe") for p in (a.get("probe_ids") or [])]
            self.terminal = {"kind": "select_branch", "hypothesis_id": hid, "probe_ids": probes, "rationale": a.get("rationale", "")}
            return f"OK: will investigate {short_id(hid)}", False
        if name == "finish_synthesis":
            self.terminal = {"kind": "finish_synthesis", "summary": a.get("summary", ""), "inconclusive": bool(a.get("inconclusive"))}
            return "OK: synthesis finished", False
        if name == "request_global_synthesis":
            r = ops.request_global_synthesis(self.actor, a.get("reasons"))
            self.terminal = {"kind": "request_synthesis", "reasons": a.get("reasons")}
            return json.dumps(r), False
        if name == "commit_diagnosis":
            r = ops.commit_diagnosis(self.actor, **a)
            self.terminal = {"kind": "commit"}
            return json.dumps(r), False
        r = getattr(ops, name)(self.actor, **a)
        return json.dumps(r), False

    def on_step_end(self, i: int, step: dict, max_steps: int):
        return self.terminal["kind"] if self.terminal else None


INVESTIGATOR_GRAPH_SPECS = {
    "record_evidence": _fn("record_evidence", "Record an evidence item. For code/config/docs: path + start_line + end_line (the runtime quotes the file). For command results: source_tool_call_id (e.g. T7) and optionally an exact quote from its output.", {
        "type": {"type": "string", "enum": [e for e in EVIDENCE_TYPES if e != "issue_statement"]},
        "summary": _S("What this evidence shows (one or two sentences)."), "path": _S("File path relative to the repo root."),
        "start_line": _I("First line."), "end_line": _I("Last line."), "source_tool_call_id": _S("Tool call that produced the observation, e.g. T7."),
        "quote": _S("Exact substring of that call's output (optional)."), "probe_id": _S("Probe this evidence belongs to (optional).")}, ["type", "summary"]),
    "record_relation": _fn("record_relation", "Relate evidence you recorded to a hypothesis.", {
        "evidence_id": _S("e.g. E4"), "hypothesis_id": _S("e.g. H1"), "relation": {"type": "string", "enum": ["supports", "contradicts", "explains"]},
        "rationale": _S("Why this evidence supports/contradicts/explains it."), "strength": _I("1 weak, 2 moderate, 3 strong")},
        ["evidence_id", "hypothesis_id", "relation", "rationale"]),
    "reset_changes": _fn("reset_changes", "Revert all your temporary edits now (also done automatically when you finish).", {}, []),
    "finish_investigation": _fn("finish_investigation", "Finish. Say what the evidence shows about the hypothesis.", {
        "summary": _S("What you found."), "outcome": {"type": "string", "enum": ["supports", "contradicts", "mixed", "inconclusive"]},
        "unresolved_questions": _A("What remains unknown."), "unexplained_observations": _A("Observations that the hypotheses do not explain."),
        "suggested_hypotheses": _A("Alternative explanations worth testing.")}, ["summary", "outcome"]),
}


class InvestigatorTools:
    """Repository tools wrapped with a ledger/intervention manager, plus the evidence facade. One instance per branch."""

    def __init__(self, env, sandbox: InvestigationSandbox, ops: GraphOps, bid: str, hid: str, cfg: DiagnosisConfig, max_steps: int):
        self.env_tools = EnvToolBox(env)
        self.sandbox, self.ops, self.bid, self.hid, self.cfg, self.max_steps = sandbox, ops, bid, hid, cfg, max_steps
        self.actor = f"investigator:{short_id(bid)}"
        self.specs = list(self.env_tools.specs) + list(INVESTIGATOR_GRAPH_SPECS.values())
        self.report: dict | None = None
        self.sigs: Counter = Counter()
        self.repeat_flag = False
        self.step = 0
        self.evidence_ids: list[str] = []

    def _budget(self) -> str:
        return f"[budget: {max(self.max_steps - self.step - 1, 0)} tool-call rounds left]"

    def call(self, name: str, args: dict) -> tuple:
        if name in INVESTIGATOR_GRAPH_SPECS:
            try:
                return self._graph_call(name, _coerce(args, INVESTIGATOR_GRAPH_SPECS[name]))
            except GraphError as e:
                if not getattr(e, "logged", False):
                    self.ops.rejections += 1
                    self.ops.emit("operation_rejected", self.actor, {"op": name, "error": str(e)})
                return f"ERROR: {e}", True
            except TypeError as e:
                self.ops.rejections += 1
                self.ops.emit("operation_rejected", self.actor, {"op": name, "error": f"bad arguments: {e}"})
                return f"ERROR: bad arguments for {name}: {e}", True
        sb = self.sandbox
        sig = (name, json.dumps(args, sort_keys=True, default=str))
        self.sigs[sig] += 1
        if self.sigs[sig] >= self.cfg.repeated_action_limit and name in ("run_command", "search_code", "read_file", "list_files"):
            self.repeat_flag = True
        iv_before = sb.open_intervention
        res = self.env_tools.call(name, args)
        text, is_err = res[0], res[1]
        meta = dict(res[2]) if len(res) > 2 and res[2] else {}
        iv = iv_before
        if name in WRITE_TOOLS and not is_err:
            iv = sb.open_if_needed(f"{name} {args.get('path', '')}")
        elif name == "run_command" and iv_before is None and sb.detect_implicit_changes(f"command modified the tree: {str(args.get('command', ''))[:120]}"):
            iv = sb.open_intervention
        rec = sb.record_call(name, args, text, is_err, meta, iv)
        meta["call_id"] = short_id(rec["id"])
        tag = f"[{short_id(rec['id'])}] " + (f"(temporary change {short_id(iv)} active) " if iv else "")
        return f"{tag}{text}\n{self._budget()}", is_err, meta

    def _graph_call(self, name: str, a: dict) -> tuple:
        ops, st = self.ops, self.ops.state
        if name == "record_evidence":
            r = ops.attach_evidence(self.actor, a.get("type"), a.get("summary"), path=a.get("path"), start_line=a.get("start_line"),
                                    end_line=a.get("end_line"), source_tool_call_id=a.get("source_tool_call_id"), quote=a.get("quote"),
                                    probe_id=a.get("probe_id"), branch_id=self.bid)
            self.evidence_ids.append(r["id"])
            e = st.evidence[st.resolve(r["id"])]
            note = f" (observed after temporary change {short_id(e['intervention_id'])})" if e.get("intervention_id") else ""
            return f"OK: recorded {r['id']}{note}. Now relate it with record_relation.", False
        if name == "record_relation":
            r = ops.create_relation(self.actor, a.get("evidence_id", ""), a.get("hypothesis_id", ""), a.get("relation"), a.get("rationale"), a.get("strength", 2))
            return f"OK: {r['id']}", False
        if name == "reset_changes":
            self.sandbox.reset_changes(f"requested by {self.actor}")
            return "OK: repository restored and verified clean", False
        # finish_investigation
        outcome = a.get("outcome")
        if outcome not in ("supports", "contradicts", "mixed", "inconclusive"):
            raise GraphError("outcome must be supports, contradicts, mixed or inconclusive")
        summary = (a.get("summary") or "").strip()
        if len(summary) < 20:
            raise GraphError("give a summary of at least 20 characters")
        if outcome in ("supports", "contradicts", "mixed") and not self.evidence_ids:
            raise GraphError(f"you claim the outcome is '{outcome}' but recorded no evidence: use record_evidence and record_relation first")
        self.report = {"outcome": outcome, "summary": summary[:1500], "evidence_ids": [st.resolve(e) for e in self.evidence_ids],
                       "unresolved_questions": [str(x)[:300] for x in (a.get("unresolved_questions") or [])],
                       "unexplained_observations": [str(x)[:300] for x in (a.get("unexplained_observations") or [])],
                       "suggested_hypotheses": [str(x)[:300] for x in (a.get("suggested_hypotheses") or [])]}
        return "OK: investigation finished", False

    def on_step_end(self, i: int, step: dict, max_steps: int):
        self.step = i
        if self.report is not None:
            return "finished"
        if self.repeat_flag:
            return "repeated_actions"
        return None
