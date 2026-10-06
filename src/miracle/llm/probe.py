"""Capability probe: five tiny cases against the toy repo, each with an automatic check.

Checks are intentionally simple substring/trace heuristics, not a quality benchmark. They answer:
does the model follow an instruction, call tools with valid arguments, chain dependent calls,
recover from a tool error, and stop with an answer?
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from ..runtime import Task
from ..workspace import Workspace
from .loop import ToolLoopRuntime

TOYREPO = Path(__file__).parent / "toyrepo"


def _calls(trace) -> list[dict]:
    return [r for s in trace["steps"] for r in s["tool_results"]]


def _ans(trace) -> str:
    return (trace["final_answer"] or "").lower()


def _chk_plain(t):
    a = re.sub(r"[^a-z]", "", _ans(t))
    return (a == "ok" and not _calls(t)), "answered exactly OK without tools" if a == "ok" and not _calls(t) \
        else f"answer={t['final_answer']!r}, tool_calls={len(_calls(t))}"


def _chk_list(t):
    ok = any(c["name"] == "list_files" for c in _calls(t)) and all(f in _ans(t) for f in ("pricing", "stock", "config"))
    return ok, "listed dir and named the 3 modules" if ok else f"answer={t['final_answer']!r}"


def _chk_read(t):
    ok = bool(_calls(t)) and "42" in _ans(t)
    return ok, "used a tool and answered 42" if ok else f"answer={t['final_answer']!r}"


def _chk_chain(t):
    c = _calls(t)
    ok = len(c) >= 2 and len({s["index"] for s in t["steps"] if s["tool_results"]}) >= 2 \
        and "tax_rate" in _ans(t) and "0.0825" in _ans(t)
    return ok, "chained >=2 dependent tool steps; found TAX_RATE=0.0825" if ok \
        else f"tool_calls={len(c)}, answer={t['final_answer']!r}"


def _chk_recover(t):
    c = _calls(t)
    err = [i for i, x in enumerate(c) if x["is_error"]]
    recovered = bool(err) and any(not x["is_error"] for x in c[err[0] + 1:])
    ok = recovered and "discount" in _ans(t)
    return ok, "hit a tool error, recovered, answered" if ok \
        else f"errors={len(err)}, recovered={recovered}, answer={t['final_answer']!r}"


@dataclass(frozen=True)
class Case:
    id: str
    prompt: str
    check: Callable[[dict], tuple[bool, str]]


CASES = [
    Case("plain", "Reply with exactly the single word OK and nothing else.", _chk_plain),
    Case("list", "List the Python modules inside the inventory package directory and give their names.", _chk_list),
    Case("read", "Look at the code: what is the value of MAX_ITEMS_PER_ORDER?", _chk_read),
    Case("chain", "In this repository, the function final_price uses a constant that is defined in a different file. "
                  "What is that constant's name and value?", _chk_chain),
    Case("recover", "The function apply_discount is defined in inventory/discounts.py. Read that file and explain in one sentence what it does.", _chk_recover),
]


def run_probe(runtime: ToolLoopRuntime, case_ids: list[str] | None, out_dir: Path, repo: Path = TOYREPO, log=print) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in [c for c in CASES if not case_ids or c.id in case_ids]:
        task = Task(f"probe-{case.id}", "toyrepo", "n/a", case.prompt, Workspace(repo, "n/a"))
        res = runtime.solve(task)
        trace = res.metadata["trace"]
        passed, why = case.check(trace)
        (out_dir / f"{case.id}.trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False))
        row = {"case": case.id, "passed": passed, "detail": why, "stop_reason": trace["stop_reason"],
               "error": trace["error"], **trace["totals"]}
        results.append(row)
        log(f"{'PASS' if passed else 'FAIL'}  {case.id:<8} steps={row['steps']} tools={row['tool_calls']} "
            f"in={row['prompt_tokens']} out={row['completion_tokens']} llm={row['llm_latency_s']}s "
            f"cost={row['cost_usd']}  -- {why}")
    summary = {"model": runtime.client.config.model, "passed": sum(r["passed"] for r in results), "total": len(results),
               "prompt_tokens": sum(r["prompt_tokens"] for r in results),
               "completion_tokens": sum(r["completion_tokens"] for r in results),
               "cost_usd": (sum(r["cost_usd"] or 0 for r in results) if any(r["cost_usd"] is not None for r in results) else None),
               "llm_latency_s": round(sum(r["llm_latency_s"] for r in results), 3),
               "runtime": runtime.describe(), "cases": results,
               "finished_at": datetime.now().isoformat(timespec="seconds")}
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
