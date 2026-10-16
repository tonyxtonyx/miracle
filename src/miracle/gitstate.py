"""Baseline snapshot / reset / dirty-state checks shared by every git-backed environment.

Contract: after `reset_to_baseline()` there is no PATCH-VISIBLE difference from the baseline tree (tracked changes, staged changes, new
non-ignored files, moved HEAD). Ignored files (bytecode, caches) are not part of the contract; they cannot enter a patch.
Dirty state is computed with the production staging script (`docker_env.stage_script`), so "clean" means "the extracted patch would be empty".
"""
from __future__ import annotations

import re


# never part of a patch: bytecode / tool caches an agent's test runs create
_EXCLUDES = ["':(exclude)**/__pycache__/**'", "':(exclude)*.pyc'", "':(exclude).pytest_cache/**'",
             "':(exclude).hypothesis/**'", "':(exclude)**/*.egg-info/**'"]
_PATHSPEC = " ".join(["."] + _EXCLUDES)


def stage_script(root: str, base_tree: str) -> str:
    """Stage everything, then un-stage binary files (a text patch can't carry them).
    Shared by DockerEnvironment and the offline replay so both extract patches identically."""
    # `git add` exits 1 (while still staging everything) when a gitignored path such as .pytest_cache exists;
    # its status must not gate the binary filter below, so it is deliberately not chained with &&.
    return (f"cd {root} && {{ git add -A -- {_PATHSPEC} >/dev/null 2>&1 || true; }}; "
            f"git diff --cached --no-renames --numstat {base_tree} | "
            "awk -F'\\t' '$1==\"-\" && $2==\"-\" {print $3}' | while IFS= read -r f; do git reset -q -- \"$f\"; done")


_IGNORED_UNTRACKED = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.hypothesis)(/|$)|\.pyc$|\.egg-info(/|$)")


class GitBaseline:
    """Mixin. Needs: `_sh(script, timeout) -> (exit_code, text)`, `_git_root` (path used by scripts), `_base_tree`, `base_head`."""
    _git_root: str
    _base_tree: str | None = None
    base_head: str | None = None

    def reset_to_baseline(self) -> None:
        script = (f"cd {self._git_root} && git reset -q --hard {self.base_head} && git read-tree --reset -u {self._base_tree} "
                  f"&& git clean -fdq")
        code, out = self._sh(script, 120)
        if code != 0:
            raise RuntimeError(f"reset failed: {out.strip()[:300]}")

    def dirty_state(self) -> dict:
        script = (f"{stage_script(self._git_root, self._base_tree)}; cd {self._git_root}; "
                  f"echo '@@HEAD'; git rev-parse HEAD; echo '@@STAGED'; git diff --cached --no-renames --name-only {self._base_tree}; "
                  f"echo '@@UNSTAGED'; git diff --no-renames --name-only; echo '@@UNTRACKED'; git ls-files --others --exclude-standard")
        code, out = self._sh(script, 120)
        sections: dict[str, list[str]] = {}
        cur = None
        for line in out.splitlines():
            if line.startswith("@@"):
                cur = line[2:]
                sections[cur] = []
            elif cur and line.strip():
                sections[cur].append(line.strip())
        head = (sections.get("HEAD") or [""])[0]
        changed = sorted(set(sections.get("STAGED", [])) | set(sections.get("UNSTAGED", [])))
        untracked = sorted(f for f in sections.get("UNTRACKED", []) if not _IGNORED_UNTRACKED.search(f))
        head_ok = head == self.base_head
        return {"clean": not changed and not untracked and head_ok, "changed_files": changed, "untracked": untracked,
                "head": head, "head_unchanged": head_ok, "base_tree": self._base_tree}
