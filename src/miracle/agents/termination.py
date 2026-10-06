"""When might the agent already be done? One predicate, used live (S2's controller) and offline (analysis of any trace).

Candidate evidence (deliberately simple, no task-specific logic):
  1. a source file was modified: a successful `edit_file` on a non-test path (edit_file only modifies existing files);
  2. a test command (pytest, runtests.py, ...) ran after that modification;
  3. that test command succeeded: exit code 0, not timed out, and tests actually ran with no failures/errors.

The tracker is a pure function of the trace's tool results, so applying it to the recorded S0/S1 traces gives the
counterfactual "what would the controller have seen" without re-running anything. A candidate is *evidence*, not proof:
the agent may legitimately keep working after it (that is exactly what the completion check is for).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .env_tools import parse_test_summary

TEST_PATH_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH_RE.search(path or ""))


def run_succeeded(meta: dict, output: str) -> bool:
    if meta.get("exit_code") != 0 or meta.get("timed_out"):
        return False
    s = parse_test_summary(output)
    if s.get("ran") == 0:                      # e.g. "Ran 0 tests": exits 0 but proves nothing
        return False
    c = s.get("counts")
    if c is not None and (c.get("passed", 0) == 0 or c.get("failed", 0) or c.get("error", 0)):
        return False
    return True                                 # unrecognised runner output: trust exit code 0


@dataclass
class CandidateTracker:
    seq: int = 0
    last_src_edit_seq: int | None = None
    last_src_edit_step: int | None = None
    source_files: dict = field(default_factory=dict)       # path -> last step edited
    _tests: list = field(default_factory=list)              # successful test events since the last source edit

    def observe(self, step: int, tr: dict) -> None:
        """Feed one tool result (trace schema) in execution order."""
        self.seq += 1
        if tr["name"] == "edit_file" and not tr["is_error"]:
            path = (tr.get("meta") or {}).get("path") or (tr.get("arguments") or {}).get("path") or ""
            if path and not is_test_path(path):
                self.last_src_edit_seq, self.last_src_edit_step = self.seq, step
                self.source_files[path] = step
                self._tests = []                            # an edit invalidates earlier test evidence
        elif tr["name"] == "run_command" and (tr.get("meta") or {}).get("is_test"):
            if self.last_src_edit_seq is not None and run_succeeded(tr["meta"], tr.get("output", "")):
                self._tests.append({"step": step, "seq": self.seq, "command": tr["meta"]["command"],
                                    "exit_code": tr["meta"]["exit_code"], "output": tr.get("output", ""),
                                    "summary": parse_test_summary(tr.get("output", ""))})

    def new_candidate(self, step: int) -> dict | None:
        """Evidence if a valid successful test (after the last source edit) happened in this very step."""
        if not self._tests or self._tests[-1]["step"] != step:
            return None
        return {"rule": "source_edit_then_successful_test", "step": step,
                "last_source_edit_step": self.last_src_edit_step, "modified_source_files": sorted(self.source_files),
                "test_command": self._tests[-1]["command"], "test_exit_code": self._tests[-1]["exit_code"],
                "test_summary": self._tests[-1]["summary"]}

    def tests_since_last_edit(self) -> list[dict]:
        return list(self._tests)


def candidate_analysis(trace: dict) -> dict:
    """Derived, descriptive: where did candidate evidence first appear, and what happened after it?"""
    tr, cands = CandidateTracker(), []
    for s in trace["steps"]:
        for r in s["tool_results"]:
            tr.observe(s["index"], r)
        if tr.new_candidate(s["index"]):
            cands.append(s["index"])
    n = len(trace["steps"])
    u = lambda steps, k: sum(((x.get("usage") or {}).get(k) or 0) for x in steps)
    first = cands[0] if cands else None
    after = [s for s in trace["steps"] if first is not None and s["index"] > first]
    return {"first_candidate_step": first, "candidate_steps": cands, "total_steps": n,
            "steps_after_first_candidate": len(after) if first is not None else None,
            "prompt_tokens_after": u(after, "prompt_tokens") if first is not None else None,
            "completion_tokens_after": u(after, "completion_tokens") if first is not None else None,
            "cost_after": round(u(after, "cost_usd"), 6) if first is not None else None,
            "cost_total": round(u(trace["steps"], "cost_usd"), 6)}
