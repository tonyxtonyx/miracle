"""S0: the simplest software-engineering agent. Qwen3.5-9B + tools, no reasoning architecture.

Fixed on purpose: one minimal system instruction, native function calling, temperature 0,
reasoning_effort none, at most 50 steps, no planning / reflection / critic / retries / context
management. Everything it does is in the trace; nothing here should be tuned from failures.
Later strategies are compared against this baseline.
"""
from __future__ import annotations

import re

from ..llm.client import LLMClient, OpenAICompatClient
from ..llm.config import LLMConfig
from ..llm.loop import run_tool_loop
from ..runtime import AgentResult, AgentRuntime, Task
from .env_tools import TOOL_NAMES, EnvToolBox

SYSTEM_PROMPT = ("You are a software engineering agent. Solve the given issue using the available repository tools. "
                 "Inspect the repository, make the necessary changes, and verify your solution with tests when possible.")


class S0Runtime(AgentRuntime):
    environment = "container"

    def __init__(self, client: LLMClient | None = None, max_steps: int = 50, system_prompt: str = SYSTEM_PROMPT,
                 **llm_settings):
        llm_settings.setdefault("model", "Qwen/Qwen3.5-9B")
        llm_settings.setdefault("reasoning_effort", "none")
        llm_settings.setdefault("temperature", 0.0)
        llm_settings.setdefault("max_tokens", 4096)   # room for a whole-file write_file call
        self.client = client or OpenAICompatClient(LLMConfig.from_env(**llm_settings))
        self.max_steps, self.system_prompt = max_steps, system_prompt
        self.name = "s0-" + (self.client.config.model or "model").split("/")[-1].lower()

    def describe(self) -> dict:
        return {"name": self.name, "class": "S0Runtime", "strategy": "S0: single ReAct-style tool loop, no architecture",
                "max_steps": self.max_steps, "system_prompt": self.system_prompt,
                "llm": self.client.config.public(), "tools": TOOL_NAMES}

    def solve(self, task: Task) -> AgentResult:
        env = task.env
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task.problem_statement}]
        trace, usage = run_tool_loop(self.client, messages, EnvToolBox(env), self.max_steps,
                                     {"instance_id": task.instance_id, "runtime": self.describe(),
                                      "task": task.problem_statement})
        patch = env.diff()           # always extracted, whatever the stop reason: partial work is data too
        files = env.changed_files()
        commands = command_log(trace)
        trace["final_diff"], trace["modified_files"] = patch, files
        meta = {"final_answer": trace["final_answer"], "stop_reason": trace["stop_reason"],
                "modified_files": files, "commands": commands, "tests": [c for c in commands if c["is_test"]],
                "stats": trajectory_stats(trace), "trace": trace}
        return AgentResult(patch, usage, meta)


# ---- post-hoc extraction (pure functions of the trace; unit-tested) -------------------------------

def parse_test_summary(output: str) -> dict:
    """Best-effort pass/fail extraction from common runners; always keeps the raw tail."""
    tail = [l for l in output.strip().splitlines() if l.strip()][-3:]
    d: dict = {"tail": "\n".join(tail)[-400:]}
    m = re.findall(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)", output)
    if m:
        d["counts"] = {}
        for n, k in m:
            d["counts"][k.rstrip("s") if k.startswith("error") else k] = d["counts"].get(k, 0) + int(n)
    m = re.search(r"Ran (\d+) tests? in", output)
    if m:
        d["ran"] = int(m.group(1))
        d["ok"] = bool(re.search(r"^OK\b", output, re.M))
    return d


def command_log(trace: dict) -> list[dict]:
    out = []
    for s in trace["steps"]:
        for r in s["tool_results"]:
            if r["name"] == "run_command" and r.get("meta"):
                m = dict(r["meta"])
                m["step"] = s["index"]
                if m["is_test"]:
                    m["summary"] = parse_test_summary(r["output"])
                out.append(m)
    return out


def trajectory_stats(trace: dict) -> dict:
    """Descriptive counters for failure-mode analysis. Descriptive only -- nothing acts on them."""
    calls = [(s["index"], r) for s in trace["steps"] for r in s["tool_results"]]
    by_tool: dict[str, int] = {}
    for _, r in calls:
        by_tool[r["name"]] = by_tool.get(r["name"], 0) + 1
    edits = [(i, r) for i, r in calls if r["name"] in ("edit_file", "write_file") and not r["is_error"]]
    tests = [(i, r) for i, r in calls if r["name"] == "run_command" and (r.get("meta") or {}).get("is_test")]
    first_edit = edits[0][0] if edits else None
    sig = [(r["name"], r["raw_arguments"]) for _, r in calls]
    finishes: dict[str, int] = {}
    for s in trace["steps"]:
        k = s["response"]["finish_reason"] or "none"
        finishes[k] = finishes.get(k, 0) + 1
    return {
        "steps": len(trace["steps"]), "tool_calls": len(calls), "calls_by_tool": by_tool,
        "tool_errors": sum(r["is_error"] for _, r in calls),
        "invalid_arguments": sum(not r["arguments_valid"] for _, r in calls),
        "edit_errors": sum(r["is_error"] for _, r in calls if r["name"] == "edit_file"),
        "successful_edits": len(edits), "first_edit_step": first_edit,
        "tests_run": len(tests),
        "tested_after_last_edit": bool(edits and any(i >= edits[-1][0] for i, _ in tests)),
        "repeated_identical_calls": sum(1 for a, b in zip(sig, sig[1:]) if a == b),
        "finish_reasons": finishes,
        "empty_steps": sum(1 for s in trace["steps"] if not s["response"]["tool_calls"] and not (s["response"]["content"] or "").strip()),
        "stop_reason": trace["stop_reason"],
    }
