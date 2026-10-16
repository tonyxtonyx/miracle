"""Judge backends behind one interface. Every call is an isolated, fresh conversation.

  AnthropicJudge  official `anthropic` SDK; claude-opus-5-5, adaptive thinking, structured output (json_schema), server-side refusal fallback
                  ("default") with an automatic retry without it if the fallback beta is rejected. Needs ANTHROPIC_API_KEY (or `ant auth login`).
  ClaudeCLIJudge  `claude -p` headless in a scratch directory, no tools, no session persistence. Needs a logged-in Claude Code CLI.
  FileJudge       writes the prompt to disk and reads the response from disk (lets a human or an external session act as the judge).
  MockJudge       a callable, for tests.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL = "claude-opus-5-5"
PRICE = {"claude-opus-5-5": (4.0, 20.0)}          # USD per 1M input/output tokens


class JudgeError(RuntimeError):
    pass


class JudgePending(JudgeError):
    """FileJudge: the prompt was written, waiting for a response file."""
    def __init__(self, path: Path):
        super().__init__(f"waiting for the judge response: {path}")
        self.path = path


@dataclass
class JudgeResult:
    data: dict
    raw_text: str
    backend: str
    model: str
    usage: dict = field(default_factory=dict)
    cost_usd: float | None = None
    latency_s: float = 0.0
    meta: dict = field(default_factory=dict)


def extract_json(text: str) -> dict:
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", t, re.S)
    if m:
        t = m.group(1)
    elif not t.startswith("{"):
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            t = t[a:b + 1]
    try:
        obj = json.loads(t)
    except ValueError as e:
        raise JudgeError(f"the judge did not return valid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise JudgeError("the judge returned JSON that is not an object")
    return obj


class JudgeClient(ABC):
    backend = "abstract"
    model = "unknown"

    @abstractmethod
    def complete(self, system: str, user: str, schema: dict, name: str) -> JudgeResult: ...


class AnthropicJudge(JudgeClient):
    backend = "anthropic-sdk"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 24000, effort: str = "high", fallbacks: bool = True, client=None):
        from ...llm.config import load_dotenv
        load_dotenv()
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.max_tokens, self.effort, self.fallbacks = client, model, max_tokens, effort, fallbacks

    def _request(self, system, user, schema, use_fallback):
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=system, messages=[{"role": "user", "content": user}],
                  thinking={"type": "adaptive"}, output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}})
        if use_fallback:
            return self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        return self.client.messages.stream(**kw)

    def complete(self, system, user, schema, name) -> JudgeResult:
        t0 = time.monotonic()
        try:
            try:
                with self._request(system, user, schema, self.fallbacks) as stream:
                    msg = stream.get_final_message()
            except Exception as e:
                if not self.fallbacks or "fallback" not in str(e).lower():
                    raise
                with self._request(system, user, schema, False) as stream:           # the fallback beta was rejected: retry plain
                    msg = stream.get_final_message()
        except JudgeError:
            raise
        except Exception as e:
            raise JudgeError(f"{type(e).__name__}: {e}") from e
        if getattr(msg, "stop_reason", None) == "refusal":
            raise JudgeError(f"the judge model refused ({getattr(msg, 'stop_details', None)})")
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        u = getattr(msg, "usage", None)
        usage = {"input_tokens": getattr(u, "input_tokens", 0), "output_tokens": getattr(u, "output_tokens", 0)}
        pin, pout = PRICE.get(self.model, (None, None))
        cost = round(usage["input_tokens"] * pin / 1e6 + usage["output_tokens"] * pout / 1e6, 6) if pin else None
        if getattr(msg, "stop_reason", None) == "max_tokens":
            raise JudgeError("the judge reply was cut off by max_tokens")
        return JudgeResult(extract_json(text), text, self.backend, self.model, usage, cost, round(time.monotonic() - t0, 2),
                           {"request_id": getattr(msg, "_request_id", None), "stop_reason": getattr(msg, "stop_reason", None)})


class ClaudeCLIJudge(JudgeClient):
    backend = "claude-cli"

    def __init__(self, model: str = "opus", executable: str = "claude", effort: str = "high", timeout: int = 1800):
        self.model, self.exe, self.effort, self.timeout = model, executable, effort, timeout

    def complete(self, system, user, schema, name) -> JudgeResult:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="judge.") as scratch:               # empty cwd: no project files, no memory
            cmd = [self.exe, "-p", user, "--system-prompt", system, "--model", self.model, "--tools", "", "--no-session-persistence",
                   "--output-format", "json", "--json-schema", json.dumps(schema), "--effort", self.effort, "--setting-sources", ""]
            try:
                r = subprocess.run(cmd, cwd=scratch, capture_output=True, text=True, timeout=self.timeout)
            except (OSError, subprocess.TimeoutExpired) as e:
                raise JudgeError(f"claude CLI failed to run: {e}") from e
        try:
            out = json.loads(r.stdout)
        except ValueError:
            raise JudgeError(f"claude CLI returned non-JSON output: {r.stdout[:200]!r} {r.stderr[:200]!r}")
        if out.get("is_error"):
            raise JudgeError(f"claude CLI error: {out.get('result')}")
        data = out.get("structured_output") or extract_json(out.get("result", ""))
        return JudgeResult(data, out.get("result") or json.dumps(data), self.backend, self.model, out.get("usage") or {}, out.get("total_cost_usd"),
                           round(time.monotonic() - t0, 2), {"session_id": out.get("session_id")})


class FileJudge(JudgeClient):
    """Prompt out, response in: `<dir>/<name>.prompt.json` is written; the judge's JSON answer is read from `<dir>/<name>.response.json`."""
    backend = "file-exchange"
    model = "external"

    def __init__(self, directory: Path):
        self.dir = Path(directory)

    def complete(self, system, user, schema, name) -> JudgeResult:
        self.dir.mkdir(parents=True, exist_ok=True)
        resp, prompt = self.dir / f"{name}.response.json", self.dir / f"{name}.prompt.json"
        if not resp.exists():
            prompt.write_text(json.dumps({"system": system, "user": user, "schema": schema}, indent=1))
            raise JudgePending(resp)
        text = resp.read_text()
        return JudgeResult(extract_json(text), text, self.backend, "external", {}, None, 0.0, {"response_file": str(resp)})


class MockJudge(JudgeClient):
    backend = "mock"
    model = "mock"

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def complete(self, system, user, schema, name) -> JudgeResult:
        self.calls.append({"name": name, "system": system, "user": user})
        data = self.fn(name, system, user, schema)
        return JudgeResult(data, json.dumps(data), self.backend, "mock", {}, 0.0, 0.0)


def make_judge(backend: str, directory: Path | None = None, model: str | None = None) -> JudgeClient:
    if backend == "auto":
        from ...llm.config import load_dotenv
        load_dotenv()
        backend = "api" if os.environ.get("ANTHROPIC_API_KEY") else "cli"
    if backend == "api":
        return AnthropicJudge(model or DEFAULT_MODEL)
    if backend == "cli":
        return ClaudeCLIJudge(model or "opus")
    if backend == "files":
        return FileJudge((directory or Path(".")) / "pending")
    raise ValueError(f"unknown judge backend {backend!r} (api | cli | files | auto)")
