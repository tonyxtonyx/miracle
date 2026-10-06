"""Per-task repository workspaces at `base_commit`.

A workspace is a standalone, single-commit git repo: no remote, no history, no
future commits. That matters for experiment validity -- a full clone checked out at
base_commit still contains the upstream fix in its object database, reachable via
`git log --all`, which an agent could find.

Workspaces are source-only. They do not contain the task's Python environment; the
official evaluation images own that. (Future: expose execution inside the instance
image through this same Workspace object.)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import paths

REPO_URL = "https://github.com/{repo}.git"
_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "miracle", "GIT_AUTHOR_EMAIL": "miracle@localhost",
            "GIT_COMMITTER_NAME": "miracle", "GIT_COMMITTER_EMAIL": "miracle@localhost"}


def git(cwd: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, env=_GIT_ENV, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {cwd}: {r.stderr.strip()}")
    return r.stdout


@dataclass(frozen=True)
class Workspace:
    path: Path
    base_commit: str

    def diff(self) -> str:
        """Everything changed relative to base_commit (incl. new files) as a unified diff.

        This is the canonical way for a runtime to turn its edits into a prediction patch.
        """
        git(self.path, "add", "-A")
        out = git(self.path, "diff", "--cached", "--no-color", "--no-ext-diff", self.base_commit)
        return out if not out or out.endswith("\n") else out + "\n"

    def reset(self) -> None:
        git(self.path, "reset", "--hard", self.base_commit)
        git(self.path, "clean", "-fdx")


def _fetch_pristine(instance: dict, dest: Path, repo_url_template: str) -> None:
    sha = instance["base_commit"]
    tmp = Path(tempfile.mkdtemp(prefix=dest.name + ".", dir=dest.parent))
    try:
        git(tmp, "init", "-q")
        git(tmp, "remote", "add", "origin", repo_url_template.format(repo=instance["repo"]))
        git(tmp, "fetch", "-q", "--depth", "1", "origin", sha)       # one commit, no history
        git(tmp, "checkout", "-q", "--detach", sha)
        git(tmp, "remote", "remove", "origin")
        (tmp / ".git" / "FETCH_HEAD").unlink(missing_ok=True)
        head = git(tmp, "rev-parse", "HEAD").strip()
        if head != sha:
            raise RuntimeError(f"{instance['instance_id']}: checked out {head}, expected {sha}")
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def pristine(instance: dict, repo_url_template: str = REPO_URL) -> Path:
    """Cached untouched checkout; fetched from the network at most once per instance."""
    dest = paths.cache_dir() / "pristine" / instance["instance_id"]
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        _fetch_pristine(instance, dest, repo_url_template)
    return dest


def create_workspace(instance: dict, dest: Path, repo_url_template: str = REPO_URL) -> Workspace:
    """Fresh, writable copy of the pristine checkout at `dest` (replaced if present)."""
    src = pristine(instance, repo_url_template)
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest, symlinks=True)
    return Workspace(dest, instance["base_commit"])
