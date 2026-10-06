"""S2 = S1 + External Termination Controller (COMPLETION_CHECK).

The agent loop, model, system prompt, tools and 50-step budget are exactly S1's. The only addition is an
external controller that, when candidate evidence appears (a source edit followed by a successful test run;
see termination.py), pauses the loop and makes ONE small, bounded LLM call that sees only the issue, the
current diff and the relevant test results, and answers:

    COMPLETE                          -> the run is terminated (stop_reason = "completion_check")
    INCOMPLETE: <unresolved requirement> -> recorded, and the normal loop continues unchanged

The controller is a pure terminator: it never injects anything into the agent's conversation (no feedback, no
plan, no reflection). Anything it is unsure about (unparseable / truncated reply, API error) means "continue":
it never terminates on uncertainty. Check calls are not agent steps (they do not consume the 50-step budget);
their tokens/cost are counted in the run totals and also reported separately in `metadata["controller"]`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from ..llm.client import LLMError
from .s1 import S1Runtime
from .termination import CandidateTracker, is_test_path

CHECK_PROMPT = """You are reviewing a proposed code change for a GitHub issue. Decide whether the change already resolves the issue.

## Issue
{issue}

## Modified files
{files}

## Current diff
{diff}

## Test results (run after the last source modification)
{tests}

Are there any concrete requirements stated in the issue that this diff does not yet satisfy? Judge only against what the issue explicitly asks for. Do not suggest extra tests, documentation or hypothetical edge cases that the issue does not ask for.

Reply with exactly one line, either:
COMPLETE
or
INCOMPLETE: <the specific unresolved requirement>"""

MAX_DIFF_CHARS = 12_000
MAX_TEST_CHARS = 2_000
MAX_TESTS_SHOWN = 3


def parse_decision(text: str | None, finish_reason: str | None = None) -> tuple[str, str]:
    """-> (COMPLETE | INCOMPLETE | INVALID, reason). Only an unambiguous COMPLETE ends a run."""
    if finish_reason == "length":
        return "INVALID", "check reply truncated"
    lines = [l.strip() for l in (text or "").strip().splitlines() if l.strip()]
    if not lines:
        return "INVALID", "empty reply"
    m = re.match(r"^\W*(INCOMPLETE|COMPLETE)\b\s*:?\s*(.*)$", lines[0], re.I | re.S)
    if not m:
        return "INVALID", f"unrecognised reply: {lines[0][:120]}"
    if m.group(1).upper() == "INCOMPLETE":
        reason = " ".join([m.group(2)] + lines[1:]).strip().lstrip(" :-–—").strip()
        return "INCOMPLETE", reason[:600] or "(no requirement given)"
    return "COMPLETE", ""


def diff_files(diff: str) -> list[tuple[str, bool]]:
    """[(path, is_new_file)] from a unified diff."""
    out = []
    for block in re.split(r"(?m)^(?=diff --git )", diff):
        m = re.match(r"diff --git a/(\S+) b/", block)
        if m:
            out.append((m.group(1), bool(re.search(r"(?m)^new file mode", block))))
    return out


def has_source_modification(diff: str) -> bool:
    """The diff really changes an existing, non-test file (guards against reverted/never-applied edits)."""
    return any(not new and not is_test_path(p) for p, new in diff_files(diff))


def build_check_prompt(issue: str, diff: str, tests: list[dict]) -> str:
    files = "\n".join(f"- {p}" for p, _ in diff_files(diff)) or "(none)"
    shown = (diff if len(diff) <= MAX_DIFF_CHARS else diff[:MAX_DIFF_CHARS] + f"\n... [diff truncated, {len(diff) - MAX_DIFF_CHARS} more chars]")
    blocks = []
    for t in tests[-MAX_TESTS_SHOWN:]:
        out = t["output"].strip()
        out = out if len(out) <= MAX_TEST_CHARS else "...[earlier output omitted]\n" + out[-MAX_TEST_CHARS:]
        blocks.append(f"$ {t['command']}\n{out}")
    return CHECK_PROMPT.format(issue=issue.strip(), files=files, diff=shown.strip(), tests="\n\n".join(blocks) or "(none)")


@dataclass
class CompletionController:
    client: object
    env: object
    issue: str
    max_tokens: int = 300
    max_checks: int = 5
    tracker: CandidateTracker = field(default_factory=CandidateTracker)
    checks: list = field(default_factory=list)
    terminated_step: int | None = None
    _last_sha: str | None = None

    def after_step(self, i: int, step: dict, max_steps: int) -> str | None:
        for tr in step["tool_results"]:
            self.tracker.observe(i, tr)
        if i >= max_steps or len(self.checks) >= self.max_checks:   # nothing left to save / check budget spent
            return None
        trigger = self.tracker.new_candidate(i)
        if trigger is None:
            return None
        diff = self.env.diff()
        if not has_source_modification(diff):
            return None
        sha = hashlib.sha256(diff.encode()).hexdigest()
        if sha == self._last_sha:                                   # same code as the last check: nothing new to judge
            return None
        self._last_sha = sha
        rec = self._check(i, trigger, diff, sha)
        self.checks.append(rec)
        if rec["decision"] == "COMPLETE":
            self.terminated_step = i
            return "completion_check"
        return None

    def _check(self, i: int, trigger: dict, diff: str, sha: str) -> dict:
        prompt = build_check_prompt(self.issue, diff, self.tracker.tests_since_last_edit())
        rec = {"check": len(self.checks) + 1, "step": i, "trigger": trigger, "diff_sha256": sha,
               "diff_bytes": len(diff.encode()), "modified_files": [p for p, _ in diff_files(diff)], "prompt": prompt}
        try:
            r = self.client.chat([{"role": "user", "content": prompt}], None, max_tokens=self.max_tokens,
                                 temperature=0.0, reasoning_effort="none")
        except LLMError as e:
            rec.update(decision="ERROR", reason=str(e)[:300], raw_response=None, finish_reason=None,
                       prompt_tokens=0, completion_tokens=0, cost_usd=None, latency_s=0.0)
            return rec
        decision, reason = parse_decision(r.content, r.finish_reason)
        rec.update(decision=decision, reason=reason, raw_response=r.content, finish_reason=r.finish_reason,
                   prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
                   cost_usd=r.usage.cost_usd, latency_s=r.latency_s, attempts=r.attempts)
        return rec

    def finalize(self, trace: dict, usage) -> dict:
        """Record controller activity in the trace and fold check cost into the run's usage totals."""
        agent = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
                 "llm_calls": usage.llm_calls, "cost_usd": usage.cost_usd}
        pt = sum(c["prompt_tokens"] for c in self.checks)
        ct = sum(c["completion_tokens"] for c in self.checks)
        known = [c["cost_usd"] for c in self.checks if c["cost_usd"] is not None]
        cost = round(sum(known), 8) if known else None
        usage.prompt_tokens += pt
        usage.completion_tokens += ct
        usage.llm_calls += len(self.checks)
        if cost is not None:
            usage.cost_usd = round((usage.cost_usd or 0.0) + cost, 8)
        dec = [c["decision"] for c in self.checks]
        summary = {"name": "completion_check", "max_checks": self.max_checks, "max_tokens": self.max_tokens,
                   "checks": len(self.checks), "complete": dec.count("COMPLETE"), "incomplete": dec.count("INCOMPLETE"),
                   "invalid": dec.count("INVALID"), "errors": dec.count("ERROR"),
                   "terminated_by_controller": self.terminated_step is not None, "termination_step": self.terminated_step,
                   "check_steps": [c["step"] for c in self.checks],
                   "usage": {"prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost,
                             "latency_s": round(sum(c["latency_s"] for c in self.checks), 4)},
                   "agent_usage_excluding_checks": agent}
        trace["controller"] = {**summary, "records": self.checks}
        trace["totals"]["controller_checks"] = len(self.checks)
        return summary


class S2Runtime(S1Runtime):
    def __init__(self, *args, check_max_tokens: int = 300, max_checks: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_max_tokens, self.max_checks = check_max_tokens, max_checks
        self.name = "s2-" + (self.client.config.model or "model").split("/")[-1].lower()

    def describe(self) -> dict:
        d = super().describe()
        d.update({"class": "S2Runtime", "strategy": "S2: S1 + external termination controller (COMPLETION_CHECK)",
                  "controller": {"trigger": "source edit (edit_file, non-test path) then successful test command",
                                 "check_max_tokens": self.check_max_tokens, "max_checks": self.max_checks,
                                 "check_prompt": CHECK_PROMPT, "on_incomplete": "record only; agent context unchanged"}})
        return d

    def make_controller(self, task):
        return CompletionController(self.client, task.env, task.problem_statement,
                                    max_tokens=self.check_max_tokens, max_checks=self.max_checks)
