"""Experiment statistics as Markdown, computed from a run's recorded files (results.jsonl + traces).

Pure function of files on disk: rerunning on the same run always gives the same report, so numbers in
READMEs can be regenerated instead of hand-typed. Descriptive only.
"""
from __future__ import annotations

import json
import math
import statistics as st
from collections import Counter

from .dataset import read_jsonl
from .runner import Run


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def _med(xs):
    return st.median(xs) if xs else None


def _fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def _trace(run: Run, iid: str):
    p = run.agent_dir(iid) / "trace.json"
    return json.loads(p.read_text()) if p.exists() else None


def markdown_report(name: str) -> str:
    run = Run(name)
    rows = read_jsonl(run.results_path)
    man = run.manifest()
    n = len(rows)
    status = Counter(r["eval"]["status"] for r in rows)
    k = status.get("resolved", 0)
    lo, hi = wilson(k, n)
    out: list[str] = []
    w = out.append

    # ---- headline ----
    w(f"**{k}/{n} resolved ({k / n:.0%}; 95% Wilson CI {lo:.0%}–{hi:.0%})**  ")
    w("Outcomes: " + ", ".join(f"{v} {s}" for s, v in sorted(status.items(), key=lambda x: -x[1])) + ".\n")

    def group(key):
        g: dict[str, list] = {}
        for r in rows:
            g.setdefault(r[key] or "?", []).append(r)
        return g
    for title, key in (("By difficulty", "difficulty"), ("By repository", "repo")):
        w(f"| {title} | resolved | instances |\n|---|---|---|")
        for gk, gr in sorted(group(key).items()):
            w(f"| {gk} | {sum(r['eval']['resolved'] for r in gr)} | {len(gr)} |")
        w("")

    # ---- cost / tokens / time ----
    ag = [r["agent"] for r in rows if r["agent"]]
    u = lambda a, f: (a.get("usage") or {}).get(f) or 0
    pt, ct = sum(u(a, "prompt_tokens") for a in ag), sum(u(a, "completion_tokens") for a in ag)
    costs = [u(a, "cost_usd") for a in ag]
    lat = [a["latency_s"] for a in ag]
    traces = {r["instance_id"]: _trace(run, r["instance_id"]) for r in rows}
    llm_s = [t["totals"]["llm_latency_s"] for t in traces.values() if t]
    tool_s = [sum(x["duration_s"] for s in t["steps"] for x in s["tool_results"]) for t in traces.values() if t]
    calls = sum(u(a, "llm_calls") for a in ag)
    w("| Resource | Total | Mean / task | Median / task |\n|---|---|---|---|")
    w(f"| Prompt tokens | {pt:,} | {pt / n:,.0f} | {_fmt(_med([u(a, 'prompt_tokens') for a in ag]), 0)} |")
    w(f"| Completion tokens | {ct:,} | {ct / n:,.0f} | {_fmt(_med([u(a, 'completion_tokens') for a in ag]), 0)} |")
    w(f"| Inference cost (USD, provider-reported) | ${sum(costs):.3f} | ${sum(costs) / n:.3f} | ${_fmt(_med(costs), 3)} |")
    w(f"| Agent wall time (s) | {sum(lat):,.0f} | {sum(lat) / n:,.0f} | {_fmt(_med(lat), 0)} |")
    if llm_s:
        w(f"| ↳ in LLM calls (s) | {sum(llm_s):,.0f} | {sum(llm_s) / len(llm_s):,.0f} | {_fmt(_med(llm_s), 0)} |")
        w(f"| ↳ in tool execution (s) | {sum(tool_s):,.0f} | {sum(tool_s) / len(tool_s):,.0f} | {_fmt(_med(tool_s), 0)} |")
    w(f"| Official-eval test runtime (s) | {sum(r['eval'].get('test_runtime_s') or 0 for r in rows):,.0f} | | |")
    w("")
    w(f"- Cost per resolved task: **{'$%.3f' % (sum(costs) / k) if k else 'n/a'}**; LLM calls: {calls}; "
      f"prompt:completion token ratio {pt / max(ct, 1):.0f}:1 (context is re-sent every step).")
    if llm_s and calls:
        w(f"- Mean LLM latency {sum(llm_s) / calls:.1f}s/call; mean output {ct / calls:.0f} tokens/call.\n")

    # ---- trajectory ----
    stats = [(r["instance_id"], (r["agent"] or {}).get("metadata", {}).get("stats")) for r in rows]
    stats = [(i, s) for i, s in stats if s]
    if stats:
        stops = Counter(s["stop_reason"] for _, s in stats)
        steps = [s["steps"] for _, s in stats]
        by_tool = Counter()
        for _, s in stats:
            by_tool.update(s["calls_by_tool"])
        tot_calls = sum(by_tool.values())
        first = [s["first_edit_step"] for _, s in stats if s["first_edit_step"]]
        w("**Trajectories**\n")
        w(f"- Stop reasons: " + ", ".join(f"{v} {s}" for s, v in stops.most_common()) +
          f". Steps mean {st.mean(steps):.1f}, median {_med(steps):.0f}.")
        w(f"- Tool calls: {tot_calls} total (" + ", ".join(f"{t} {c}" for t, c in by_tool.most_common()) + ").")
        w(f"- Tool errors {sum(s['tool_errors'] for _, s in stats)} ({sum(s['tool_errors'] for _, s in stats) / max(tot_calls, 1):.0%}), "
          f"invalid arguments {sum(s['invalid_arguments'] for _, s in stats)}, "
          f"failed edits {sum(s['edit_errors'] for _, s in stats)}.")
        w(f"- Runs with a source edit: {len(first)}/{len(stats)}; first edit at step median {_fmt(_med(first), 0)}. "
          f"Test commands per run: median {_fmt(_med([s['tests_run'] for _, s in stats]), 0)} "
          f"(0 in {sum(s['tests_run'] == 0 for _, s in stats)} runs); "
          f"tested after last edit in {sum(s['tested_after_last_edit'] for _, s in stats)}/{len(stats)}.\n")

    # ---- per instance ----
    w("| instance | difficulty | outcome | F2P | P2P | steps | stop | first→last edit | tests | prompt tok | out tok | cost | time |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        a, e = r["agent"] or {}, r["eval"]
        s = (a.get("metadata") or {}).get("stats") or {}
        t = traces.get(r["instance_id"])
        src = [x["index"] for x in (t["steps"] if t else []) for y in x["tool_results"]
               if y["name"] == "edit_file" and not y["is_error"]]
        edits = f"{min(src)}→{max(src)}" if src else "none"
        f2p = f"{e['f2p_passed']}/{e['f2p_total']}" if e.get("f2p_total") is not None else "-"
        p2p = f"{e['p2p_passed']}/{e['p2p_total']}" if e.get("p2p_total") is not None else "-"
        w(f"| {r['instance_id']} | {r['difficulty']} | {e['status']} | {f2p} | {p2p} | {s.get('steps', '-')} | "
          f"{s.get('stop_reason', '-')} | {edits} | {s.get('tests_run', '-')} | {u(a, 'prompt_tokens'):,} | "
          f"{u(a, 'completion_tokens'):,} | ${u(a, 'cost_usd'):.3f} | {a.get('latency_s', 0):.0f}s |")
    w("")
    w(f"_Run `{man['name']}`; dataset `{man['dataset']['name']}` @ `{(man['dataset']['revision'] or '')[:12]}` "
      f"(sha256 `{man['dataset']['sha256'][:12]}`); runtime `{man['runtime']['name']}`; swebench {man['env']['swebench']}; "
      f"{man['env']['machine']} host, Docker {man['env']['docker_server']}._")
    return "\n".join(out)
