"""LLM settings, all externalised: environment variables (optionally via a gitignored .env),
overridable per call. The API key is never stored in traces or manifests (`public()`)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .. import paths

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"      # nearest to 8B currently on DeepInfra (no Qwen3-8B listed)
DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"


class MissingApiKey(RuntimeError):
    pass


def load_dotenv(path: Path | None = None) -> None:
    """Minimal KEY=VALUE loader. Never overrides variables already in the environment."""
    path = path or paths.root() / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.split(" #", 1)[0].strip().strip("'\"")
        if v:
            os.environ.setdefault(k.strip(), v)


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.0
    max_tokens: int = 2048
    reasoning_effort: str | None = None    # none|low|medium|high; None = provider default
    timeout_s: float = 120.0
    max_retries: int = 3                   # on 429 / 5xx / network errors
    api_key_env: str = "DEEPINFRA_API_KEY"
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls, require_key: bool = True, **overrides) -> "LLMConfig":
        """Precedence: explicit overrides (non-None) > environment/.env > defaults."""
        load_dotenv()
        env = os.environ
        key_env = overrides.get("api_key_env") or cls.api_key_env
        base = cls(
            model=env.get("MIRACLE_MODEL", DEFAULT_MODEL),
            base_url=env.get("DEEPINFRA_BASE_URL", DEFAULT_BASE_URL),
            temperature=float(env.get("MIRACLE_TEMPERATURE", 0.0)),
            max_tokens=int(env.get("MIRACLE_MAX_TOKENS", 2048)),
            reasoning_effort=env.get("MIRACLE_REASONING_EFFORT") or None,
            api_key_env=key_env,
            api_key=env.get(key_env, ""),
        )
        known = {f.name for f in fields(cls)}
        bad = set(overrides) - known
        if bad:
            raise TypeError(f"unknown LLM settings: {sorted(bad)}")
        cfg = replace(base, **{k: v for k, v in overrides.items() if v is not None and k != "api_key"})
        if overrides.get("api_key"):
            cfg = replace(cfg, api_key=overrides["api_key"])
        if require_key and not cfg.api_key:
            raise MissingApiKey(f"{cfg.api_key_env} is not set (export it or put it in {paths.root() / '.env'})")
        return cfg

    def public(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "api_key"}
        d["api_key_set"] = bool(self.api_key)
        return d
