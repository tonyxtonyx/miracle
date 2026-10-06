import json

import httpx
import pytest

from miracle.llm.client import ChatResult, LLMClient, LLMError, LLMUsage, OpenAICompatClient
from miracle.llm.config import LLMConfig, MissingApiKey
from miracle.llm.loop import ToolLoopRuntime
from miracle.llm.probe import CASES, TOYREPO, run_probe
from miracle.llm.tools import ToolBox
from miracle.registry import build_runtime
from miracle.runtime import Task
from miracle.workspace import Workspace


# ---------- config ----------
def test_config_precedence_and_key_hygiene(monkeypatch, home):
    for k in ("DEEPINFRA_API_KEY", "MIRACLE_MODEL", "MIRACLE_MAX_TOKENS", "MIRACLE_TEMPERATURE"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(MissingApiKey):
        LLMConfig.from_env()
    home.mkdir(parents=True)
    (home / ".env").write_text("DEEPINFRA_API_KEY=sk-from-dotenv\nMIRACLE_MAX_TOKENS=111 # c\nMIRACLE_MODEL=a/b\n")
    monkeypatch.setenv("MIRACLE_MODEL", "env/model")           # real env beats .env
    cfg = LLMConfig.from_env(temperature=0.7)                  # explicit override beats both
    assert (cfg.api_key, cfg.max_tokens, cfg.model, cfg.temperature) == ("sk-from-dotenv", 111, "env/model", 0.7)
    assert "sk-from-dotenv" not in repr(cfg) and "sk-from-dotenv" not in json.dumps(cfg.public())
    assert LLMConfig.from_env(model=None).model == "env/model"  # None override = not set
    with pytest.raises(TypeError):
        LLMConfig.from_env(bogus=1)


# ---------- client over a faked HTTP layer ----------
def _ok(content="hi", calls=None, cost=0.0000268, model="Qwen/Qwen3.5-9B"):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for n, a in calls]
    return {"model": model, "choices": [{"message": {**msg, "reasoning_content": "thinking..."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "estimated_cost": cost,
                      "completion_tokens_details": {"reasoning_tokens": 3}}}


def _client(handler, **cfg):
    sleeps = []
    c = OpenAICompatClient(LLMConfig(api_key="sk-secret-123", **cfg), transport=httpx.MockTransport(handler), sleep=sleeps.append)
    return c, sleeps


def test_client_request_and_parse():
    seen = {}

    def h(req):
        seen.update(url=str(req.url), auth=req.headers["authorization"], body=json.loads(req.content))
        return httpx.Response(200, json=_ok())
    c, _ = _client(h, reasoning_effort="low", temperature=0.2, max_tokens=99)
    r = c.chat([{"role": "user", "content": "x"}], tools=[{"type": "function"}])
    assert seen["url"] == "https://api.deepinfra.com/v1/openai/chat/completions" and seen["auth"] == "Bearer sk-secret-123"
    b = seen["body"]
    assert (b["model"], b["temperature"], b["max_tokens"], b["reasoning_effort"], b["tool_choice"]) == \
        ("Qwen/Qwen3.5-9B", 0.2, 99, "low", "auto")
    assert (r.content, r.reasoning, r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.reasoning_tokens,
            r.usage.cost_usd, r.attempts) == ("hi", "thinking...", 10, 5, 3, 0.0000268, 1)
    assert "sk-secret-123" not in json.dumps(r.request) and r.raw["model"] == "Qwen/Qwen3.5-9B"


def test_retries_429_and_5xx_then_succeeds_and_honours_retry_after():
    codes = iter([429, 503])

    def h(req):
        try:
            return httpx.Response(next(codes), headers={"retry-after": "2"}, text="busy")
        except StopIteration:
            return httpx.Response(200, json=_ok())
    c, sleeps = _client(h)
    r = c.chat([{"role": "user", "content": "x"}])
    assert r.attempts == 3 and sleeps == [2.0, 2.0]


def test_no_retry_on_400_and_gives_up_after_max_retries():
    n = {"c": 0}

    def bad(req):
        n["c"] += 1
        return httpx.Response(400, text="bad model")
    c, _ = _client(bad)
    with pytest.raises(LLMError) as e:
        c.chat([])
    assert n["c"] == 1 and e.value.status == 400

    n["c"] = 0
    c, _ = _client(lambda r: (n.__setitem__("c", n["c"] + 1), httpx.Response(500, text="x"))[1], max_retries=2)
    with pytest.raises(LLMError) as e:
        c.chat([])
    assert n["c"] == 3 and e.value.attempts == 3


def test_network_error_is_retried():
    calls = {"n": 0}

    def h(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=_ok())
    c, _ = _client(h)
    assert c.chat([]).attempts == 2


# ---------- tools ----------
def test_toolbox_sandbox_and_behaviour(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.py").write_text("Hello = 1\n")
    (tmp_path / "secret.txt").write_text("s")
    tb = ToolBox(tmp_path / "a")
    assert tb.call("list_files", {"path": "."}) == ("x.py", False)
    assert tb.call("read_file", {"path": "x.py"}) == ("Hello = 1\n", False)
    assert tb.call("search_code", {"query": "hello"}) == ("x.py:1: Hello = 1", False)
    for bad in ("../secret.txt", "/etc/passwd", "../../a/../secret.txt"):
        out, err = tb.call("read_file", {"path": bad})
        assert err and "escapes" in out or "no such file" in out
    assert "escapes" in tb.call("list_files", {"path": ".."})[0]
    assert tb.call("read_file", {"path": "nope.py"})[1] and tb.call("nope", {})[1]
    assert tb.call("read_file", {})[1]                                  # missing arg -> error, not exception
    assert tb.call("search_code", {"query": "zzz"}) == ("(no matches)", False)


# ---------- scripted model ----------
class Scripted(LLMClient):
    """Replays a list of (content, [(tool, args)]) turns."""
    def __init__(self, turns, **kw):
        self.config = LLMConfig(api_key="sk-secret-123", model="Qwen/Qwen3.5-9B")
        self.turns, self.calls = list(turns), []

    def chat(self, messages, tools=None, **o):
        self.calls.append(json.loads(json.dumps(messages)))
        turn = self.turns.pop(0)
        content, tcs = turn[0], turn[1]
        forced_finish = turn[2] if len(turn) > 2 else None      # optional explicit finish_reason
        if isinstance(content, Exception):
            raise content
        calls = [{"id": f"id{len(self.calls)}_{i}", "name": n,
                  "arguments": a if isinstance(a, str) else json.dumps(a)} for i, (n, a) in enumerate(tcs)]
        return ChatResult(content, calls, forced_finish or ("tool_calls" if calls else "stop"), "m", LLMUsage(10, 5, 15, None, 0.001),
                          0.1, 0.1, 1, raw={"x": 1}, request={"model": "m", "temperature": 0, "messages": [], "tools": []})


def _task(repo=TOYREPO, prompt="q"):
    return Task("t", "r", "b", prompt, Workspace(repo, "n/a"))


def test_loop_multistep_trace_and_usage():
    c = Scripted([(None, [("search_code", {"query": "final_price"})]),
                  (None, [("read_file", {"path": "inventory/pricing.py"}), ("read_file", {"path": "nope"})]),
                  ("done", [])])
    res = ToolLoopRuntime(client=c).solve(_task())
    t = res.metadata["trace"]
    assert (t["stop_reason"], t["final_answer"], res.patch) == ("final_answer", "done", "")
    assert [len(s["tool_results"]) for s in t["steps"]] == [1, 2, 0]
    assert t["steps"][1]["tool_results"][1]["is_error"] and not t["steps"][1]["tool_results"][0]["is_error"]
    assert (res.usage.llm_calls, res.usage.tool_calls, res.usage.prompt_tokens, res.usage.cost_usd) == (3, 3, 30, 0.003)
    # tool results are fed back with matching ids, in OpenAI message format
    last = c.calls[2]
    assert [m["role"] for m in last] == ["user", "assistant", "tool", "assistant", "tool", "tool"]
    assert last[2]["tool_call_id"] == last[1]["tool_calls"][0]["id"]
    json.dumps(t)                                                      # trace is JSON-serialisable


def test_loop_failure_modes():
    t = ToolLoopRuntime(client=Scripted([(None, [("read_file", "{not json")]), ("ok", [])])).solve(_task()).metadata["trace"]
    r = t["steps"][0]["tool_results"][0]
    assert not r["arguments_valid"] and r["is_error"] and t["stop_reason"] == "final_answer"

    t = ToolLoopRuntime(client=Scripted([(None, [("list_files", {"path": "."})])] * 3), max_steps=3).solve(_task()).metadata["trace"]
    assert t["stop_reason"] == "max_steps" and len(t["steps"]) == 3

    t = ToolLoopRuntime(client=Scripted([(LLMError("HTTP 500", 500), [])])).solve(_task()).metadata["trace"]
    assert t["stop_reason"] == "llm_error" and "500" in t["error"] and t["steps"] == []


def test_runtime_is_a_normal_agent_runtime(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-secret-123")
    rt = build_runtime("toolloop", {"model": "Qwen/Qwen3.5-9B", "max_steps": 4}, [])
    d = rt.describe()
    assert d["name"] == "toolloop-qwen3.5-9b" and d["max_steps"] == 4 and "sk-secret-123" not in json.dumps(d)
    assert d["llm"]["api_key_set"] is True and d["system_prompt"] is None


# ---------- probe: an oracle model must pass every case (validates toy repo + checks) ----------
def test_probe_oracle_passes_all_cases(tmp_path):
    turns = [
        ("OK", []),
        (None, [("list_files", {"path": "inventory"})]), ("Modules: __init__, config, pricing, stock", []),
        (None, [("search_code", {"query": "MAX_ITEMS_PER_ORDER"})]), ("It is 42.", []),
        (None, [("search_code", {"query": "def final_price"})]), (None, [("search_code", {"query": "TAX_RATE"})]),
        ("final_price uses TAX_RATE = 0.0825 from config.py", []),
        (None, [("read_file", {"path": "inventory/discounts.py"})]), (None, [("list_files", {"path": "inventory"})]),
        (None, [("read_file", {"path": "inventory/pricing.py"})]), ("apply_discount reduces the price by a percent discount.", []),
    ]
    s = run_probe(ToolLoopRuntime(client=Scripted(turns)), None, tmp_path, log=lambda *_: None)
    assert [(r["case"], r["passed"]) for r in s["cases"]] == [(c.id, True) for c in CASES], s["cases"]
    assert (tmp_path / "chain.trace.json").exists() and (tmp_path / "report.json").exists()


def test_probe_checks_reject_wrong_behaviour(tmp_path):
    turns = [("Sure! OK.", []), ("I think pricing.py", []), ("42", []), ("TAX_RATE=0.0825", []), ("it discounts", [])]
    s = run_probe(ToolLoopRuntime(client=Scripted(turns)), None, tmp_path, log=lambda *_: None)
    assert [r["passed"] for r in s["cases"]] == [False, False, False, False, False]
