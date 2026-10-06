"""Provider-neutral chat client. `LLMClient` is the seam: a runtime depends only on it, so
swapping DeepInfra for another OpenAI-compatible provider (or a local server) is a config change,
and a non-OpenAI provider is one new subclass."""
from __future__ import annotations

import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

import httpx

from .config import LLMConfig


class LLMError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, body: str | None = None, attempts: int = 1):
        super().__init__(msg)
        self.status, self.body, self.attempts = status, body, attempts


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int | None = None    # if the provider reports it (usage.completion_tokens_details)
    cost_usd: float | None = None          # provider-reported usage.estimated_cost, else None


@dataclass
class ChatResult:
    content: str | None
    tool_calls: list[dict]                 # [{"id","name","arguments"(raw str)}]
    finish_reason: str | None
    model: str | None                      # model the provider says it served
    usage: LLMUsage
    latency_s: float                       # final successful HTTP attempt
    total_wall_s: float                    # including retries/backoff
    attempts: int
    reasoning: str | None = None           # reasoning text if the provider returns it separately
    raw: dict = field(default_factory=dict)        # untouched response body
    request: dict = field(default_factory=dict)    # exact JSON payload sent (no credentials)

    def to_dict(self) -> dict:
        return asdict(self)


class LLMClient(ABC):
    config: LLMConfig

    @abstractmethod
    def chat(self, messages: list[dict], tools: list[dict] | None = None, **overrides) -> ChatResult: ...


class OpenAICompatClient(LLMClient):
    def __init__(self, config: LLMConfig, transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.config = config
        self._http = httpx.Client(timeout=config.timeout_s, transport=transport)
        self._sleep = sleep

    def _payload(self, messages, tools, o) -> dict:
        c = self.config
        p = {"model": o.get("model", c.model), "messages": messages,
             "temperature": o.get("temperature", c.temperature),
             "max_tokens": o.get("max_tokens", c.max_tokens)}
        effort = o.get("reasoning_effort", c.reasoning_effort)
        if effort:
            p["reasoning_effort"] = effort
        if tools:
            p["tools"], p["tool_choice"] = tools, o.get("tool_choice", "auto")
        return p

    def chat(self, messages, tools=None, **overrides) -> ChatResult:
        payload = self._payload(messages, tools, overrides)
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"}
        t_all = time.monotonic()
        last: LLMError | None = None
        for attempt in range(1, self.config.max_retries + 2):
            t0 = time.monotonic()
            try:
                r = self._http.post(url, headers=headers, json=payload)
            except httpx.HTTPError as e:
                last = LLMError(f"network error: {type(e).__name__}: {e}", attempts=attempt)
                retry_after = None
            else:
                if r.status_code == 200:
                    return self._parse(r.json(), payload, time.monotonic() - t0, time.monotonic() - t_all, attempt)
                last = LLMError(f"HTTP {r.status_code}: {r.text[:500]}", r.status_code, r.text, attempt)
                if r.status_code != 429 and r.status_code < 500:
                    raise last                       # 4xx other than 429: our request is wrong, don't retry
                retry_after = _retry_after(r)
            if attempt > self.config.max_retries:
                raise last
            self._sleep(retry_after if retry_after is not None else min(2 ** (attempt - 1), 20) + random.random() * 0.25)
        raise last  # pragma: no cover

    @staticmethod
    def _parse(body: dict, payload: dict, latency: float, wall: float, attempts: int) -> ChatResult:
        ch = (body.get("choices") or [{}])[0]
        msg = ch.get("message") or {}
        u = body.get("usage") or {}
        details = u.get("completion_tokens_details") or {}
        calls = [{"id": tc.get("id"), "name": (tc.get("function") or {}).get("name"),
                  "arguments": (tc.get("function") or {}).get("arguments", "")}
                 for tc in msg.get("tool_calls") or []]
        return ChatResult(
            content=msg.get("content"), tool_calls=calls, finish_reason=ch.get("finish_reason"),
            model=body.get("model"),
            usage=LLMUsage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0), u.get("total_tokens", 0),
                           details.get("reasoning_tokens"), u.get("estimated_cost")),
            latency_s=round(latency, 4), total_wall_s=round(wall, 4), attempts=attempts,
            reasoning=msg.get("reasoning_content") or msg.get("reasoning"),
            raw=body, request=payload)


def _retry_after(r: httpx.Response) -> float | None:
    try:
        return min(float(r.headers["retry-after"]), 60.0)
    except (KeyError, ValueError):
        return None


def make_client(**overrides) -> OpenAICompatClient:
    return OpenAICompatClient(LLMConfig.from_env(**overrides))
