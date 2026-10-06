"""`miracle llm ...` -- poke the model, list models, run the capability probe."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

from .. import paths
from .client import LLMError, OpenAICompatClient
from .config import DEFAULT_BASE_URL, LLMConfig, MissingApiKey
from .loop import ToolLoopRuntime
from .probe import TOYREPO, run_probe


def _settings(a) -> dict:
    return {"model": a.model, "base_url": a.base_url, "temperature": a.temperature,
            "max_tokens": a.max_tokens, "reasoning_effort": a.reasoning_effort}


def _llm_dir(kind: str) -> Path:
    d = paths.runs_dir() / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_chat(a):
    client = OpenAICompatClient(LLMConfig.from_env(**_settings(a)))
    messages = ([{"role": "system", "content": a.system}] if a.system else []) + [{"role": "user", "content": a.prompt}]
    r = client.chat(messages)
    rec = {"ts": datetime.now().isoformat(timespec="seconds"), "config": client.config.public(), **r.to_dict()}
    with open(_llm_dir("llm-chat") / "chat-log.jsonl", "a") as f:     # append-only audit/cost log
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if a.raw:
        print(json.dumps(r.raw, indent=2, ensure_ascii=False))
        print("-" * 60)
    u = r.usage
    print(f"model:          {r.model} (requested {client.config.model})")
    print(f"latency:        {r.latency_s}s  (wall incl. retries {r.total_wall_s}s, attempts {r.attempts})")
    print(f"tokens:         in={u.prompt_tokens} out={u.completion_tokens} total={u.total_tokens}"
          + (f" (reasoning={u.reasoning_tokens})" if u.reasoning_tokens is not None else ""))
    print(f"cost:           {'$%.8f' % u.cost_usd if u.cost_usd is not None else 'not reported'}")
    print(f"finish_reason:  {r.finish_reason}")
    if r.reasoning:
        print(f"reasoning:      {r.reasoning[:500]}{'...' if len(r.reasoning) > 500 else ''}")
    print(f"\n{r.content}")


def cmd_models(a):
    cfg = LLMConfig.from_env(require_key=False, base_url=a.base_url)
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
    r = httpx.get(cfg.base_url.rstrip("/") + "/models", headers=headers, timeout=30)
    r.raise_for_status()
    for m in sorted(r.json()["data"], key=lambda m: m["id"]):
        meta = m.get("metadata") or {}
        if a.grep.lower() in m["id"].lower() and "chat" in (meta.get("tags") or ["chat"]):
            p = meta.get("pricing") or {}
            print(f"{m['id']:<50} ctx={meta.get('context_length')}  $/1M in={p.get('input_tokens')} out={p.get('output_tokens')}"
                  f"  {','.join(meta.get('tags') or [])}")


def cmd_probe(a):
    rt = ToolLoopRuntime(max_steps=a.max_steps, system_prompt=a.system, **_settings(a))
    out = _llm_dir("llm-probe") / f"{datetime.now():%Y%m%d-%H%M%S}-{rt.name}"
    s = run_probe(rt, [c for c in (a.cases or "").split(",") if c], out)
    print(f"\n{s['passed']}/{s['total']} passed | in={s['prompt_tokens']} out={s['completion_tokens']} "
          f"cost={s['cost_usd']} llm_latency={s['llm_latency_s']}s\ntraces: {out}")


def cmd_agent(a):
    from ..runtime import Task
    from ..workspace import Workspace
    rt = ToolLoopRuntime(max_steps=a.max_steps, system_prompt=a.system, **_settings(a))
    repo = Path(a.repo).resolve()
    res = rt.solve(Task("adhoc", repo.name, "n/a", a.task, Workspace(repo, "n/a")))
    t = res.metadata["trace"]
    out = _llm_dir("llm-agent") / f"{datetime.now():%Y%m%d-%H%M%S}.trace.json"
    out.write_text(json.dumps(t, indent=2, ensure_ascii=False))
    for s in t["steps"]:
        for r in s["tool_results"]:
            print(f"[step {s['index']}] {r['name']}({r['arguments']}) -> {'ERROR ' if r['is_error'] else ''}{r['output'][:80]!r}")
    print(f"\nstop_reason={t['stop_reason']}  totals={t['totals']}\nanswer: {t['final_answer']}\ntrace: {out}")


def add_parser(sub):
    p = sub.add_parser("llm", help="model client, capability probe").add_subparsers(dest="llm_cmd", required=True)

    def common(x):
        x.add_argument("--model")
        x.add_argument("--base-url")
        x.add_argument("--temperature", type=float)
        x.add_argument("--max-tokens", type=int)
        x.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high"])
        x.add_argument("--system")

    c = p.add_parser("chat", help="send one prompt; show raw response, tokens, latency, cost")
    c.add_argument("prompt")
    c.add_argument("--raw", action="store_true", help="print the untouched JSON response")
    common(c)
    c.set_defaults(fn=cmd_chat)

    m = p.add_parser("models", help="list chat models (public endpoint; no key needed)")
    m.add_argument("--grep", default="")
    m.add_argument("--base-url")
    m.set_defaults(fn=cmd_models)

    pr = p.add_parser("probe", help="run the tool-use capability cases against the toy repo")
    pr.add_argument("--cases", help="comma-separated subset: plain,list,read,chain,recover")
    pr.add_argument("--max-steps", type=int, default=8)
    common(pr)
    pr.set_defaults(fn=cmd_probe)

    ag = p.add_parser("agent", help="run the tool loop on an ad-hoc task over any directory")
    ag.add_argument("task")
    ag.add_argument("--repo", default=str(TOYREPO))
    ag.add_argument("--max-steps", type=int, default=8)
    common(ag)
    ag.set_defaults(fn=cmd_agent)


def guarded(fn, a):
    try:
        fn(a)
    except MissingApiKey as e:
        sys.exit(f"error: {e}")
    except LLMError as e:
        sys.exit(f"LLM request failed after {e.attempts} attempt(s): {e}")
