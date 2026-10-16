"""S3: causal, falsification-oriented diagnosis, then a deliberately simple implementation stage (the unchanged S1 loop)."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ..agents.s0 import SYSTEM_PROMPT
from ..agents.s1 import S1Runtime
from ..llm.client import LLMClient, OpenAICompatClient
from ..llm.config import LLMConfig
from ..runtime import AgentResult, AgentRuntime, Task, Usage
from .config import DiagnosisConfig
from .engine import DiagnosisEngine
from .freeze import FROZEN_PATH, verify_frozen
from .handoff import build_handoff
from .prompts import IMPLEMENTATION_PREFIX, prompt_hashes, prompt_texts


class S3Runtime(AgentRuntime):
    environment = "container"

    def __init__(self, client: LLMClient | None = None, cfg: DiagnosisConfig | None = None, **llm_settings):
        self.cfg = cfg or DiagnosisConfig()
        llm_settings.setdefault("model", self.cfg.model)
        llm_settings.setdefault("temperature", self.cfg.temperature)
        llm_settings.setdefault("reasoning_effort", self.cfg.reasoning_effort)
        self.client = client or OpenAICompatClient(LLMConfig.from_env(**llm_settings))
        self.name = "s3-" + (self.client.config.model or "model").split("/")[-1].lower()

    def describe(self) -> dict:
        ok, problems = verify_frozen() if FROZEN_PATH.exists() else (False, ["not frozen"])
        return {"name": self.name, "class": "S3Runtime",
                "strategy": "S3: controller + branch investigators over a typed reasoning graph; falsification-oriented diagnosis; simple S1 implementation",
                "config": self.cfg.to_dict(), "llm": self.client.config.public(), "prompt_hashes": prompt_hashes(), "prompts": prompt_texts(),
                "implementation_system_prompt": SYSTEM_PROMPT, "frozen": {"file_present": FROZEN_PATH.exists(), "matches": ok, "problems": problems}}

    def solve(self, task: Task) -> AgentResult:
        root = Path(task.artifact_dir) if task.artifact_dir else Path(tempfile.mkdtemp(prefix="s3."))
        engine = DiagnosisEngine(self.client, task.env, task.instance_id, task.problem_statement, self.cfg, root / "diagnosis")
        state = engine.run()                                      # ends with a verified reset of the repository
        handoff = build_handoff(state, engine.repo_map)
        impl_dir = root / "implementation"
        impl_dir.mkdir(parents=True, exist_ok=True)
        (impl_dir / "handoff.md").write_text(handoff)
        impl_task = Task(task.instance_id, task.repo, task.base_commit,
                         task.problem_statement + "\n\n---\n" + IMPLEMENTATION_PREFIX.format(handoff=handoff), env=task.env)
        res = S1Runtime(client=self.client, max_steps=self.cfg.implementation_max_steps).solve(impl_task)
        meta = dict(res.metadata)
        trace = meta.get("trace")
        (impl_dir / "patch.diff").write_text(res.patch)
        if trace is not None:
            (impl_dir / "trace.json").write_text(json.dumps(trace, indent=1, ensure_ascii=False, default=str))
        (impl_dir / "result.json").write_text(json.dumps({
            "patch_bytes": len(res.patch.encode()), "usage": res.usage.to_dict(), "stop_reason": meta.get("stop_reason"),
            "modified_files": meta.get("modified_files"), "stats": meta.get("stats"), "tests": meta.get("tests"),
            "final_answer": meta.get("final_answer")}, indent=2, default=str))
        diag_usage = engine.total_usage()
        usage = Usage(prompt_tokens=diag_usage.prompt_tokens + res.usage.prompt_tokens, completion_tokens=diag_usage.completion_tokens + res.usage.completion_tokens,
                      llm_calls=diag_usage.llm_calls + res.usage.llm_calls, tool_calls=diag_usage.tool_calls + res.usage.tool_calls,
                      cost_usd=round((diag_usage.cost_usd or 0) + (res.usage.cost_usd or 0), 8))
        d = state.diagnosis or {}
        meta["diagnosis"] = {"status": d.get("status"), "selected": [short for short in d.get("selected_hypothesis_ids", [])],
                             "stats": state.to_dict()["stats"], "events": state.seq, "usage": engine.snapshot(), "rejected_operations": engine.ops.rejections,
                             "syntheses": len(state.syntheses), "branches": len(state.branches), "interventions": len(state.interventions),
                             "resets_verified": all(r["ok"] for r in state.resets)}
        return AgentResult(res.patch, usage, meta)
