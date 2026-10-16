"""An Environment backed by a plain directory on the host (a git repo). Used for toy tasks and tests; same contract as DockerEnvironment:
commands run in a fresh shell at the repo root, `diff()` is the patch relative to the baseline snapshot, `reset_to_baseline()` restores it."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .environment import Environment, ExecResult
from .gitstate import GitBaseline, stage_script
from .workspace import _GIT_ENV


_PREFIX = "hostenv."
_MARKER = ".miracle-hostenv"


class HostEnvironment(GitBaseline, Environment):
    def __init__(self, host_dir: Path, cleanup_dir: Path | None = None):
        """`cleanup_dir`: a directory THIS CLASS created (see from_directory) that close() may delete. Anything else is never deleted."""
        self.host = Path(host_dir).resolve()
        self.root = self._git_root = str(self.host)
        self._cleanup = Path(cleanup_dir).resolve() if cleanup_dir else None
        if not (self.host / ".git").exists():
            self._git("init", "-q")
        self._git("add", "-A")
        if subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=self.host, env=_GIT_ENV, capture_output=True).returncode != 0:
            self._git("commit", "-q", "-m", "baseline", "--allow-empty")
        self.base_head = self._git("rev-parse", "HEAD").strip()
        self._git("add", "-A")
        self._base_tree = self._git("write-tree").strip()

    @classmethod
    def from_directory(cls, src: Path) -> "HostEnvironment":
        base = Path(tempfile.mkdtemp(prefix=_PREFIX))
        (base / _MARKER).write_text("created by miracle HostEnvironment; safe to delete\n")
        work = base / "repo"
        shutil.copytree(src, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
        return cls(work, cleanup_dir=base)

    def close(self) -> None:
        d = self._cleanup
        # triple guard: we made it (marker file), it carries our prefix, and it is the direct parent of our repo directory
        if d and d.is_dir() and d.name.startswith(_PREFIX) and (d / _MARKER).is_file() and self.host.parent == d:
            shutil.rmtree(d, ignore_errors=True)
        self._cleanup = None

    def _git(self, *a: str) -> str:
        r = subprocess.run(["git", *a], cwd=self.host, env=_GIT_ENV, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()}")
        return r.stdout

    def _env(self) -> dict:
        e = dict(_GIT_ENV)
        e["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{e.get('PATH', '')}"      # `python` resolves to the project's interpreter
        e.pop("GIT_CONFIG_GLOBAL", None)
        return e

    def _sh(self, script: str, timeout: int) -> tuple[int, str]:
        r = subprocess.run(["bash", "-c", script], cwd=self.host, env=self._env(), capture_output=True, timeout=timeout)
        return r.returncode, r.stdout.decode("utf-8", "replace")

    def exec(self, command: str, timeout: int = 120) -> ExecResult:
        t0 = time.monotonic()
        p = subprocess.Popen(["bash", "-c", command], cwd=self.host, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
            code, timed_out = p.returncode, False
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            code, timed_out = 124, True
        return ExecResult(code, out.decode("utf-8", "replace"), timed_out, round(time.monotonic() - t0, 3))

    def _path(self, path: str) -> Path:
        return Path(path) if path.startswith("/") else self.host / path

    def read_bytes(self, path: str) -> bytes:
        p = self._path(path)
        if p.is_dir():
            raise IsADirectoryError(path)
        if not p.exists():
            raise FileNotFoundError(path)
        return p.read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def diff(self) -> str:
        code, out = self._sh(f"{stage_script(self.root, self._base_tree)}; git diff --cached --no-renames --no-color --no-ext-diff {self._base_tree}", 120)
        return out if not out or out.endswith("\n") else out + "\n"

    def changed_files(self) -> list[str]:
        code, out = self._sh(f"{stage_script(self.root, self._base_tree)}; git diff --cached --no-renames --name-only {self._base_tree}", 120)
        return [l for l in out.splitlines() if l]
