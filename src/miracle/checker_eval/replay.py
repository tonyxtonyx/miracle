"""Reconstruct the repository state at any step of a recorded trajectory, offline, with no Docker.

A trajectory's file modifications are (almost always) its successful `edit_file` / `write_file` calls, which are
deterministic functions of the file contents. Replaying them on a pristine checkout of `base_commit` therefore
reproduces the agent's working tree at any step. Anything done through `run_command` (sed -i, git apply, scripts
that write files) is invisible to the replay, so reconstruction is only trusted for a run if replaying the WHOLE
trajectory reproduces the recorded final patch exactly (`fidelity`). Diffs are extracted with the production
staging script (`docker_env.stage_script`), with git's abbreviated-hash length calibrated to the recorded patches
so reconstructed diffs are byte-identical to what the agent's container produced.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..agents.env_tools import EnvToolBox
from ..docker_env import stage_script
from ..environment import Environment, ExecResult
from ..workspace import _GIT_ENV

DIFF_START = re.compile(r"(?m)^diff --git ")


def normalize_patch(text: str) -> str:
    """Drop anything before the first `diff --git` (e.g. git warning text that leaked into early S0 patches)."""
    m = DIFF_START.search(text or "")
    return text[m.start():] if m else ""


def abbrev_len(patch: str) -> int | None:
    m = re.search(r"(?m)^index (\w+)\.\.(\w+)", patch or "")
    return len(m.group(1)) if m else None


class ReplayEnv(Environment):
    """Host-directory environment that answers to the container's paths (`/testbed/...`)."""
    root = "/testbed"

    def __init__(self, host_dir: Path, abbrev: int | None):
        self.host = Path(host_dir)
        if abbrev:
            self._git("config", "core.abbrev", str(abbrev))
        self._git("add", "-A")
        self._base_tree = self._git("write-tree").strip()

    def _git(self, *a: str) -> str:
        r = subprocess.run(["git", *a], cwd=self.host, env=_GIT_ENV, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()}")
        return r.stdout

    def _h(self, path: str) -> Path:
        return self.host / path[len(self.root):].lstrip("/") if path.startswith(self.root) else Path(path)

    def exec(self, command: str, timeout: int = 120) -> ExecResult:
        raise NotImplementedError("replay environments cannot run commands")

    def read_bytes(self, path: str) -> bytes:
        p = self._h(path)
        if p.is_dir():
            raise IsADirectoryError(path)
        if not p.exists():
            raise FileNotFoundError(path)
        return p.read_bytes()

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._h(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def _script(self) -> str:
        return stage_script(str(self.host), self._base_tree)

    def diff(self) -> str:
        r = subprocess.run(["bash", "-c", f"{self._script()}; git diff --cached --no-renames --no-color --no-ext-diff {self._base_tree}"],
                           env=_GIT_ENV, capture_output=True)
        out = r.stdout.decode("utf-8", "replace")
        return out if not out or out.endswith("\n") else out + "\n"

    def changed_files(self) -> list[str]:
        r = subprocess.run(["bash", "-c", f"{self._script()}; git diff --cached --no-renames --name-only {self._base_tree}"],
                           env=_GIT_ENV, capture_output=True)
        return [l for l in r.stdout.decode("utf-8", "replace").splitlines() if l]


def replay_trace(pristine: Path, trace: dict, at_steps: set[int], abbrev: int | None) -> dict:
    """Replay edit_file/write_file calls on a copy of `pristine`.
    -> {"diffs": {step: diff after that step}, "final": diff after the last step, "divergences": [...]}"""
    work = Path(tempfile.mkdtemp(prefix="replay."))
    try:
        host = work / "repo"
        shutil.copytree(pristine, host, symlinks=True)
        env = ReplayEnv(host, abbrev)
        tb, diffs, divergences = EnvToolBox(env), {}, []
        for s in trace["steps"]:
            for r in s["tool_results"]:
                if r["name"] in ("edit_file", "write_file") and not r["is_error"] and r.get("arguments"):
                    out, err = tb.call(r["name"], r["arguments"])[:2]
                    if err:                         # an edit that succeeded originally must succeed in replay
                        divergences.append({"step": s["index"], "tool": r["name"], "error": out[:200]})
            if s["index"] in at_steps:
                diffs[s["index"]] = env.diff()
        return {"diffs": diffs, "final": env.diff(), "divergences": divergences}
    finally:
        shutil.rmtree(work, ignore_errors=True)
