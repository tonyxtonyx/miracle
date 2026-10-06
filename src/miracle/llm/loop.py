"""Minimal tool-use loop: messages -> model -> tool calls -> results -> model -> ... -> answer.

Deliberately no strategy: no planning, reflection, retries-on-bad-answers, or prompt tuning. Native
function calling only. `run_tool_loop` is shared by every runtime built on it; `ToolLoopRuntime` is the
read-only toy-repo probe runtime.

Trace (schema v2; `length_nudge` flags and `totals.length_nudges` are additive fields): per step the raw response, content, reasoning text, finish_reason, tool calls, usage,
latency, retry attempts, `n_messages_sent` (the request was `messages[:n_messages_sent]` of the final
`messages` list, so every request is reconstructible without storing quadratic copies), and tool results.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Protocol

from ..runtime import AgentResult, AgentRuntime, Task, Usage
from .client import LLMClient, LLMError, OpenAICompatClient
from .config import LLMConfig
from .tools import ToolBox

TRACE_SCHEMA = 2


class ToolSet(Protocol):
    specs: list[dict]

    def call(self, name: str, args: dict) -> tuple:
        """(output, is_error) or (output, is_error, meta_dict)."""


def run_tool_loop(client: LLMClient, messages: list[dict], toolset: ToolSet, max_steps: int,
                  header: dict | None = None, length_nudge: str | None = None) -> tuple[dict, Usage]:
    """`length_nudge`: if set, a response with finish_reason == "length" (cut off by the output-token cap) is
    never treated as the end of the work. The truncated response stays in the conversation and this text is
    appended as a user message so the model continues. Each such continuation is a normal step and counts
    against `max_steps`. If None (S0 behaviour), a truncated reply without tool calls ends the run as "length"."""
    trace = {"schema": TRACE_SCHEMA, **(header or {}),
             "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "steps": [], "final_answer": None, "stop_reason": None, "error": None}
    t_start = time.monotonic()
    usage, costs = Usage(), []
    nudges = 0
    for i in range(1, max_steps + 1):
        n_sent = len(messages)
        try:
            r = client.chat(messages, toolset.specs)
        except LLMError as e:
            trace["stop_reason"], trace["error"] = "llm_error", f"{e}"
            break
        usage.llm_calls += 1
        usage.prompt_tokens += r.usage.prompt_tokens
        usage.completion_tokens += r.usage.completion_tokens
        costs.append(r.usage.cost_usd)
        step = {"index": i, "n_messages_sent": n_sent,
                "response": {"content": r.content, "reasoning": r.reasoning, "tool_calls": r.tool_calls,
                             "finish_reason": r.finish_reason, "model": r.model},
                "usage": {"prompt_tokens": r.usage.prompt_tokens, "completion_tokens": r.usage.completion_tokens,
                          "reasoning_tokens": r.usage.reasoning_tokens, "cost_usd": r.usage.cost_usd},
                "latency_s": r.latency_s, "attempts": r.attempts,
                "request_params": {k: v for k, v in r.request.items() if k not in ("messages", "tools")},
                "raw": r.raw, "tool_results": []}
        trace["steps"].append(step)

        truncated = r.finish_reason == "length"
        # a nudge sent after the final step could never be answered, so it is only added while budget remains
        nudge = bool(length_nudge) and truncated and i < max_steps

        if not r.tool_calls:
            messages.append({"role": "assistant", "content": r.content or ""})
            if nudge:
                messages.append({"role": "user", "content": length_nudge})
                step["length_nudge"] = True
                nudges += 1
                continue
            trace["final_answer"] = r.content
            trace["stop_reason"] = "length" if truncated else "final_answer"
            break

        messages.append({"role": "assistant", "content": r.content or "", "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}} for c in r.tool_calls]})
        for c in r.tool_calls:
            usage.tool_calls += 1
            t0 = time.monotonic()
            meta = None
            try:
                args = json.loads(c["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                res = toolset.call(c["name"], args)
                out, is_err = res[0], res[1]
                meta = res[2] if len(res) > 2 else None
                parsed = True
            except ValueError as e:
                out, is_err, parsed, args = f"ERROR: invalid JSON arguments: {e}", True, False, None
            tr = {"tool_call_id": c["id"], "name": c["name"], "arguments": args, "raw_arguments": c["arguments"],
                  "arguments_valid": parsed, "output": out, "is_error": is_err,
                  "duration_s": round(time.monotonic() - t0, 4)}
            if meta is not None:
                tr["meta"] = meta
            step["tool_results"].append(tr)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        if nudge:   # truncated mid tool call: every call has its result above; now ask the model to continue
            messages.append({"role": "user", "content": length_nudge})
            step["length_nudge"] = True
            nudges += 1
    else:
        trace["stop_reason"] = "max_steps"

    known = [c for c in costs if c is not None]
    usage.cost_usd = round(sum(known), 8) if known else None
    trace["messages"] = messages
    trace["totals"] = {**usage.to_dict(), "steps": len(trace["steps"]), "length_nudges": nudges,
                       "llm_latency_s": round(sum(s["latency_s"] for s in trace["steps"]), 4),
                       "wall_s": round(time.monotonic() - t_start, 4)}
    return trace, usage


class ToolLoopRuntime(AgentRuntime):
    def __init__(self, client: LLMClient | None = None, max_steps: int = 8,
                 system_prompt: str | None = None, **llm_settings):
        """llm_settings: any LLMConfig field (model, temperature, max_tokens, base_url, ...).
        DeepInfra guidance is to avoid system messages with tool calling, so default is none."""
        self.client = client or OpenAICompatClient(LLMConfig.from_env(**llm_settings))
        self.max_steps, self.system_prompt = max_steps, system_prompt
        self.name = "toolloop-" + (self.client.config.model or "model").split("/")[-1].lower()

    def describe(self) -> dict:
        return {"name": self.name, "class": "ToolLoopRuntime", "max_steps": self.max_steps,
                "system_prompt": self.system_prompt, "llm": self.client.config.public(),
                "tools": [s["function"]["name"] for s in ToolBox(".").specs]}

    def solve(self, task: Task) -> AgentResult:
        messages = ([{"role": "system", "content": self.system_prompt}] if self.system_prompt else [])
        messages.append({"role": "user", "content": task.problem_statement})
        trace, usage = run_tool_loop(self.client, messages, ToolBox(task.workspace.path), self.max_steps,
                                     {"instance_id": task.instance_id, "runtime": self.describe(),
                                      "task": task.problem_statement})
        return AgentResult("", usage, {"final_answer": trace["final_answer"],
                                       "stop_reason": trace["stop_reason"], "trace": trace})
