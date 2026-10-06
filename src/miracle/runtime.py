"""The Agent Runtime interface -- the only thing a future agent has to implement.

This module deliberately knows nothing about SWE-bench, Docker, or evaluation.
A runtime receives a `Task` (issue text + a repository workspace) and returns a
patch plus whatever it measured about itself. It is never shown the gold patch,
the test patch, or the FAIL_TO_PASS / PASS_TO_PASS lists.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field

from .environment import Environment
from .workspace import Workspace


@dataclass(frozen=True)
class Task:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    workspace: Workspace | None = None   # host source checkout (runtimes with environment="workspace")
    env: Environment | None = None       # executable environment (runtimes with environment="container")

    @staticmethod
    def from_instance(instance: dict, workspace: Workspace | None = None,
                      env: Environment | None = None) -> "Task":
        # Explicit whitelist: nothing else from the dataset record can leak to the agent.
        return Task(instance["instance_id"], instance["repo"], instance["base_commit"],
                    instance["problem_statement"], workspace, env)


@dataclass
class Usage:
    """Standard accounting fields. Runtimes fill what applies; the rest stay 0/None."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    tool_calls: int = 0
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentResult:
    patch: str                                    # unified diff against base_commit; "" = gave up
    usage: Usage = field(default_factory=Usage)
    metadata: dict = field(default_factory=dict)  # free-form: strategy, model(s), budget, trace path, ...


class AgentRuntime(ABC):
    #: short label; becomes `model_name_or_path` in predictions and names harness log dirs
    name: str = "runtime"
    #: what the runner must provide in the Task: "workspace" (source-only host checkout, cheap) or
    #: "container" (executable environment with the task's test setup; needed to run tests)
    environment: str = "workspace"

    def describe(self) -> dict:
        """JSON-serialisable config (model, strategy, budgets...), stored in the run manifest."""
        return {"name": self.name}

    @abstractmethod
    def solve(self, task: Task) -> AgentResult: ...
