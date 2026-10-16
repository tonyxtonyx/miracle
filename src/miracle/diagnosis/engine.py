"""The diagnosis state machine. Python owns the control flow, budgets, isolation and bookkeeping; the LLMs make the reasoning decisions.

ORIENT -> HYPOTHESIZE -> loop[ SELECT -> INVESTIGATE -> RESET+VERIFY -> REVIEW (-> SYNTHESIS) ] -> COMMIT -> final RESET+VERIFY.
Controller, synthesis and every investigator branch are separate LLM calls with separate contexts (same underlying model).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from ..environment import Environment
from ..errors import InfraError
from ..llm.client import ChatResult, LLMClient
from ..llm.loop import run_tool_loop
from ..runtime import Usage
from . import prompts as P
from .config import DiagnosisConfig
from .context import branch_context, branch_report_text, graph_summary
from .graph import GraphState, short_id
from .ops import GraphOps
from .repo_map import build_repo_map
from .sandbox import InvestigationSandbox
from .synthesis import synthesis_triggers
from .tools import ControllerTools, InvestigatorTools

MAX_SYNTHESES = 3


class BoundClient(LLMClient):
    """Injects per-role call options (max_tokens) without changing the shared client."""

    def __init__(self, inner: LLMClient, **overrides):
        self.inner, self.overrides, self.config = inner, overrides, inner.config

    def chat(self, messages, tools=None, **o) -> ChatResult:
        return self.inner.chat(messages, tools, **{**self.overrides, **o})


class DiagnosisEngine:
    def __init__(self, client: LLMClient, env: Environment, instance_id: str, issue: str, cfg: DiagnosisConfig, out_dir: Path, clock=time.time):
        self.cfg, self.env, self.issue, self.out = cfg, env, issue, Path(out_dir)
        self.ctl = BoundClient(client, max_tokens=cfg.controller_max_tokens)
        self.inv = BoundClient(client, max_tokens=cfg.investigator_max_tokens)
        self.state = GraphState(instance_id)
        self.ops = GraphOps(self.state, self.out / "events.jsonl", issue, cfg, clock=clock)
        self.sandbox = InvestigationSandbox(env, self.ops, cfg.max_excerpt_lines)
        self.usage = {r: Usage() for r in ("controller", "investigator", "synthesis")}
        self.wall = {r: 0.0 for r in self.usage}
        self.controller_trace: list[dict] = []
        self.repo_map = ""
        self.t_start = time.monotonic()
        self._stalled = 0
        self._repeated = False
        self._last_report: dict | None = None
        self._syntheses = 0
        self.issue_evidence: str | None = None

    # ---- accounting ------------------------------------------------------------------------------------------------
    def _add(self, role: str, usage: Usage, wall: float):
        u = self.usage[role]
        u.prompt_tokens += usage.prompt_tokens
        u.completion_tokens += usage.completion_tokens
        u.llm_calls += usage.llm_calls
        u.tool_calls += usage.tool_calls
        u.cost_usd = round((u.cost_usd or 0) + (usage.cost_usd or 0), 8) if (u.cost_usd is not None or usage.cost_usd is not None) else None
        self.wall[role] += wall

    def snapshot(self) -> dict:
        tot = lambda f: sum((getattr(u, f) or 0) for u in self.usage.values())
        return {"prompt_tokens": tot("prompt_tokens"), "completion_tokens": tot("completion_tokens"), "llm_calls": tot("llm_calls"),
                "tool_calls": tot("tool_calls"), "cost_usd": round(tot("cost_usd"), 6), "wall_s": round(time.monotonic() - self.t_start, 1)}

    def total_usage(self) -> Usage:
        s = self.snapshot()
        return Usage(prompt_tokens=s["prompt_tokens"], completion_tokens=s["completion_tokens"], llm_calls=s["llm_calls"],
                     tool_calls=s["tool_calls"], cost_usd=s["cost_usd"])

    # ---- LLM phases ----------------------------------------------------------------------------------------------------
    def _controller_call(self, phase: str, user: str, max_steps: int, label: str) -> dict | None:
        tools = ControllerTools(self.ops, self.cfg, phase)
        messages = [{"role": "system", "content": P.CONTROLLER_SYSTEM}, {"role": "user", "content": user}]
        t0 = time.monotonic()
        trace, usage = run_tool_loop(self.ctl, messages, tools, max_steps, {"phase": phase, "label": label}, on_step_end=tools.on_step_end)
        self._add("synthesis" if phase == "synthesis" else "controller", usage, time.monotonic() - t0)
        self.controller_trace.append({"phase": phase, "label": label, "stop_reason": trace["stop_reason"], "terminal": tools.terminal, "trace": trace})
        (self.out / "controller_trace.json").write_text(json.dumps(self.controller_trace, indent=1, ensure_ascii=False, default=str))
        if trace["stop_reason"] == "llm_error":
            raise InfraError(f"model provider failed during the {phase} phase: {trace['error']}")
        return tools.terminal

    def _hypothesize(self):
        st, cfg = self.state, self.cfg
        user = P.HYPOTHESIZE_USER.format(issue=self.issue, repo_map=self.repo_map, min_h=cfg.min_initial_hypotheses)
        term = self._controller_call("hypothesize", user, cfg.hypothesize_max_steps, "initial")
        if not term or len(st.hypotheses) < cfg.min_initial_hypotheses:
            user2 = (user + f"\n\nNOTE: the graph currently contains {len(st.hypotheses)} hypotheses, not enough to proceed:\n" + graph_summary(st)
                     + "\nAdd the missing distinct hypotheses and at least one discriminating probe, then call done_planning.")
            self._controller_call("hypothesize", user2, cfg.hypothesize_max_steps, "retry")

    def _synthesis(self, reasons: list[str]) -> bool:
        """Returns True if the synthesis declared the investigation inconclusive."""
        self._syntheses += 1
        sid = self.ops.state.syntheses and next(iter(reversed(self.ops.state.syntheses)), None)
        user = P.SYNTHESIS_USER.format(issue=self.issue, graph=graph_summary(self.state, include_refuted_detail=True),
                                       reasons="\n".join(f"- {r}" for r in reasons))
        term = self._controller_call("synthesis", user, self.cfg.synthesis_max_steps, f"synthesis {self._syntheses}")
        if sid:
            self.ops.complete_synthesis(sid, (term or {}).get("summary", "no summary"))
        return bool(term and term.get("kind") == "finish_synthesis" and term.get("inconclusive"))

    # ---- branches -------------------------------------------------------------------------------------------------------
    def _run_branch(self, hid: str, probe_ids: list[str]) -> str:
        ops, st, sb, cfg = self.ops, self.state, self.sandbox, self.cfg
        if st.hypotheses[hid]["status"] in ("proposed", "weakened", "inconclusive", "supported"):
            ops.update_hypothesis_status("runtime", hid, "investigating", [], "selected for investigation by the controller")
        bid = ops.start_branch(hid, probe_ids, {"max_steps": cfg.branch_max_steps})
        for p in probe_ids:
            ops.update_probe("runtime", p, status="running")
        sb.begin_branch(bid)
        actor = f"investigator:{short_id(bid)}"
        msg, visible = branch_context(st, self.issue, hid, probe_ids, cfg.branch_max_steps, P.BRANCH_USER)
        ops.visible[actor] = visible
        tools = InvestigatorTools(self.env, sb, ops, bid, hid, cfg, cfg.branch_max_steps)
        messages = [{"role": "system", "content": P.INVESTIGATOR_SYSTEM}, {"role": "user", "content": msg}]
        t0 = time.monotonic()
        trace = usage = None
        try:
            trace, usage = run_tool_loop(self.inv, messages, tools, cfg.branch_max_steps, {"branch": bid, "hypothesis": hid},
                                         on_step_end=tools.on_step_end)
        finally:
            reset = sb.end_branch(bid)                     # ALWAYS restore + verify, even if the loop crashed
        stop = trace["stop_reason"]
        self._add("investigator", usage, time.monotonic() - t0)
        report = tools.report or {"outcome": "inconclusive", "auto": True, "evidence_ids": [st.resolve(e) for e in tools.evidence_ids],
                                  "summary": f"The investigator stopped ({stop}) without a final report; {len(tools.evidence_ids)} evidence item(s) were recorded.",
                                  "unresolved_questions": [], "unexplained_observations": [], "suggested_hypotheses": []}
        self._repeated = tools.repeat_flag
        self._last_report = report
        for p in probe_ids:
            ops.update_probe("runtime", p, status="completed" if tools.evidence_ids else "abandoned")
        ops.end_branch(bid, report, {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                                     "llm_calls": usage.llm_calls, "tool_calls": usage.tool_calls, "cost_usd": usage.cost_usd}, stop)
        inv_dir = self.out / "investigators"
        inv_dir.mkdir(parents=True, exist_ok=True)
        (inv_dir / f"{short_id(bid)}.json").write_text(json.dumps({
            "branch": bid, "hypothesis": hid, "probe_ids": probe_ids, "visible_hypotheses": sorted(visible), "report": report, "reset": reset,
            "evidence_ids": [st.resolve(e) for e in tools.evidence_ids], "stop_reason": stop,
            "tool_call_ids": [c["id"] for c in sb.calls.values() if c["branch_id"] == bid], "trace": trace}, indent=1, ensure_ascii=False, default=str))
        if stop == "llm_error":
            raise InfraError(f"model provider failed during investigation {bid}: {trace['error']}")
        return bid

    # ---- selection / progress ---------------------------------------------------------------------------------------------
    def _default_selection(self) -> dict | None:
        st = self.state
        cand = [h for h in st.active() if h["status"] in ("proposed", "investigating", "weakened", "inconclusive", "supported")]
        if not cand:
            return None
        runs = lambda hid: sum(1 for b in st.branches.values() if b["hypothesis_id"] == hid)
        cand.sort(key=lambda h: (runs(h["id"]), h["created_at_step"]))
        h = cand[0]
        probes = [p["id"] for p in st.probes.values() if p["status"] == "planned" and h["id"] in p["target_hypothesis_ids"]]
        return {"kind": "select_branch", "hypothesis_id": h["id"], "probe_ids": probes[:2], "rationale": "default selection: least-investigated active hypothesis"}

    def _signature(self) -> tuple:
        st = self.state
        return (len(st.hypotheses), len(st.relations), len(st.evidence), tuple(sorted((h["id"], h["status"]) for h in st.hypotheses.values())))

    # ---- main --------------------------------------------------------------------------------------------------------------
    def run(self) -> GraphState:
        ops, st, cfg = self.ops, self.state, self.cfg
        try:
            ops.set_phase("orient")
            e, = [ops.attach_evidence("runtime", "issue_statement", "The issue statement as given (first 2000 characters).", quote=self.issue[:2000])["id"]]
            self.issue_evidence = e
            self.sandbox.establish_baseline()
            self.repo_map = build_repo_map(self.env, self.issue)
            (self.out / "repo_map.txt").write_text(self.repo_map)
            ops.set_phase("hypothesize")
            self._hypothesize()
            nxt = None
            for it in range(1, cfg.max_iterations + 1):
                ops.set_budget(iteration=it, max_iterations=cfg.max_iterations, **self.snapshot())
                if st.diagnosis:
                    break
                sel = nxt or self._default_selection()
                if sel is None:
                    break
                ops.set_phase("investigate")
                ops.decide("runtime" if nxt is None else "controller", "select_branch", sel.get("rationale", ""), hypothesis_id=sel["hypothesis_id"], probe_ids=sel["probe_ids"], iteration=it)
                before = self._signature()
                bid = self._run_branch(sel["hypothesis_id"], sel["probe_ids"])
                ops.set_budget(stage=f"after branch {it}", **self.snapshot())
                ops.set_phase("review")
                term = self._controller_call("review", P.REVIEW_USER.format(
                    issue=self.issue, graph=graph_summary(st), branch_hypothesis=short_id(sel["hypothesis_id"]),
                    report=branch_report_text(st, bid), iteration=it, max_iterations=cfg.max_iterations), cfg.review_max_steps, f"review {it}")
                ops.set_budget(stage=f"after review {it}", **self.snapshot())
                kind = (term or {}).get("kind")
                self._stalled = self._stalled + 1 if self._signature() == before else 0
                if st.diagnosis:
                    break
                nxt = term if kind == "select_branch" else None
                reasons = list((term or {}).get("reasons") or []) if kind == "request_synthesis" else []
                auto = synthesis_triggers(st, cfg, stalled=self._stalled, repeated_actions=self._repeated, last_report=self._last_report)
                if (reasons or auto) and self._syntheses < MAX_SYNTHESES:
                    if not reasons:                                           # triggered by the runtime: record the request
                        ops.request_global_synthesis("runtime", auto)
                    ops.set_phase("synthesis")
                    if self._synthesis(reasons + [r for r in auto if r not in reasons]):
                        break
                    self._stalled, nxt = 0, None
            if not st.diagnosis:
                self._force_inconclusive()
            ops.set_phase("committed")
            ops.set_budget(stage="diagnosis end", **self.snapshot())
        finally:
            try:
                self.sandbox.final_reset()
            finally:
                self._write()
        return st

    def _force_inconclusive(self):
        st = self.state
        rank = {"high": 3, "medium": 2, "low": 1, "none": 0}
        lead = sorted([h for h in st.active()], key=lambda h: (h["status"] == "supported", rank[h["support_level"]], -h["created_at_step"]), reverse=True)
        reason = ("the diagnosis budget was exhausted without a validated commit" if st.hypotheses else "no hypothesis could be formed")
        self.ops.commit_inconclusive([h["id"] for h in lead[:2]], reason)

    def _write(self):
        out = self.out
        out.mkdir(parents=True, exist_ok=True)
        (out / "graph.json").write_text(json.dumps(self.state.to_dict(), indent=1, ensure_ascii=False, default=str))
        (out / "diagnosis.json").write_text(json.dumps(self.state.diagnosis, indent=1, ensure_ascii=False, default=str))
        (out / "controller_trace.json").write_text(json.dumps(self.controller_trace, indent=1, ensure_ascii=False, default=str))
        self.sandbox.write_artifacts(out)
        (out / "usage.json").write_text(json.dumps({"by_role": {r: {**u.to_dict(), "wall_s": round(self.wall[r], 1)} for r, u in self.usage.items()},
                                                     "total": self.snapshot(), "rejected_operations": self.ops.rejections}, indent=2))
