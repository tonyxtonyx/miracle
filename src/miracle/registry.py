"""Resolve a --runtime spec to an AgentRuntime instance.

  gold | empty | broken | noop   built-in mocks
  some.module:ClassName        any AgentRuntime subclass; constructed as ClassName(**config)
"""
from __future__ import annotations

import importlib

from .mock import BrokenPatchRuntime, EmptyRuntime, GoldRuntime, NoopEditRuntime
from .runtime import AgentRuntime


def build_runtime(spec: str, config: dict, instances: list[dict]) -> AgentRuntime:
    if spec == "gold":
        return GoldRuntime({i["instance_id"]: i["patch"] for i in instances})
    if spec == "empty":
        return EmptyRuntime()
    if spec == "noop":
        return NoopEditRuntime()
    if spec == "broken":
        return BrokenPatchRuntime()
    if spec == "s0":
        spec = "miracle.agents.s0:S0Runtime"
    if spec == "toolloop":
        spec = "miracle.llm.loop:ToolLoopRuntime"
    mod, _, cls = spec.partition(":")
    if not cls:
        raise ValueError(f"unknown runtime {spec!r}; use gold|empty|broken|noop or module:Class")
    rt = getattr(importlib.import_module(mod), cls)(**config)
    if not isinstance(rt, AgentRuntime):
        raise TypeError(f"{spec} is not an AgentRuntime")
    return rt
