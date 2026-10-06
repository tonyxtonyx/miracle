"""Execution environment interface for agents that need to *run* code (tests, scripts).

Like runtime.py this knows nothing about SWE-bench or Docker. An environment is a repository
checkout plus a way to execute commands in it. `diff()` is the canonical way to turn the agent's
edits into a prediction patch: it is relative to the state at environment start.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ExecResult:
    exit_code: int
    output: str            # stdout + stderr interleaved
    timed_out: bool
    duration_s: float


class Environment(ABC):
    root: str = "/testbed"   # repository root inside the environment

    @abstractmethod
    def exec(self, command: str, timeout: int = 120) -> ExecResult:
        """Run a shell command at the repo root in a fresh shell (no state kept between calls)."""

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        """Absolute path. Raises FileNotFoundError / IsADirectoryError / OSError."""

    @abstractmethod
    def write_bytes(self, path: str, data: bytes) -> None:
        """Absolute path; parent directories are created."""

    @abstractmethod
    def diff(self) -> str:
        """Unified diff of all text-file changes since the environment started ('' if none)."""

    @abstractmethod
    def changed_files(self) -> list[str]:
        """Repo-relative paths changed since the environment started."""
