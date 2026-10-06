"""Trivial runtimes that exercise every pipeline outcome without any model."""
from __future__ import annotations

from .runtime import AgentResult, AgentRuntime, Task
from .workspace import git


class GoldRuntime(AgentRuntime):
    """Applies the dataset's reference patch inside the workspace, then returns the
    workspace diff. Expected outcome: resolved. Validates workspace -> diff -> evaluator."""
    name = "mock-gold"

    def __init__(self, patches: dict[str, str]):
        self._patches = patches

    def solve(self, task: Task) -> AgentResult:
        p = task.workspace.path / ".gold.patch"
        p.write_text(self._patches[task.instance_id])
        try:
            git(task.workspace.path, "apply", p.name)
        finally:
            p.unlink(missing_ok=True)
        return AgentResult(task.workspace.diff(), metadata={"note": "reference patch"})


class EmptyRuntime(AgentRuntime):
    """Never produces a patch. Expected outcome: empty_patch."""
    name = "mock-empty"

    def solve(self, task: Task) -> AgentResult:
        return AgentResult("")


class BrokenPatchRuntime(AgentRuntime):
    """Returns a patch that cannot apply. Expected outcome: patch_apply_failed."""
    name = "mock-broken"

    def solve(self, task: Task) -> AgentResult:
        bad = ("diff --git a/__no_such_file__.py b/__no_such_file__.py\n"
               "--- a/__no_such_file__.py\n+++ b/__no_such_file__.py\n"
               "@@ -1,1 +1,1 @@\n-nothing\n+something\n")
        return AgentResult(bad)


class NoopEditRuntime(AgentRuntime):
    """Appends a comment to one source file: a valid patch that fixes nothing.
    Expected outcome: unresolved (patch applies, tests run, target tests still fail)."""
    name = "mock-noop"

    def solve(self, task: Task) -> AgentResult:
        files = git(task.workspace.path, "ls-files", "*.py").splitlines()
        target = task.workspace.path / sorted(files)[0]
        target.write_text(target.read_text() + "\n# miracle noop edit\n")
        return AgentResult(task.workspace.diff())
