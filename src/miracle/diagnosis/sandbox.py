"""Branch isolation: a verified-clean baseline before every investigation branch, a verified reset after it.

Lifecycle: verify clean -> branch runs (the first write opens an INTERVENTION with an ID; evidence from tool calls made while one is open is
tagged with it and `observation_state=post_intervention`) -> end: close interventions, reset, VERIFY. A failed verification raises InfraError
(never scored as an agent failure). The sandbox also owns the tool-call ledger: evidence may only cite calls recorded here.
"""
from __future__ import annotations

import json
import posixpath
import time
from pathlib import Path

from ..environment import Environment
from ..errors import InfraError
from .graph import GraphError
from .ops import GraphOps

WRITE_TOOLS = ("edit_file", "write_file")


class InvestigationSandbox:
    def __init__(self, env: Environment, ops: GraphOps, max_excerpt_lines: int = 80):
        self.env, self.ops, self.max_lines = env, ops, max_excerpt_lines
        self.calls: dict[str, dict] = {}
        self.open_intervention: str | None = None
        self.branch_id: str | None = None
        self.resets: list[dict] = []
        self._n = 0
        self.ops.source = self                      # the sandbox is the evidence source for the ops layer

    # ---- EvidenceSource -----------------------------------------------------------------------------------------------
    def repo_state(self) -> dict:
        return {"base_tree": getattr(self.env, "_base_tree", None), "base_head": getattr(self.env, "base_head", None),
                "intervention_id": self.open_intervention, "branch_id": self.branch_id}

    def read_excerpt(self, path: str, start: int, end: int) -> dict:
        root = self.env.root.rstrip("/")
        p = (path or "").strip()
        full = posixpath.normpath(p if p.startswith("/") else posixpath.join(root, p))
        if full != root and not full.startswith(root + "/"):
            raise GraphError(f"no such file inside the repository: {path}")
        if "/.git/" in full + "/":
            raise GraphError("the .git directory cannot be cited")
        try:
            text = self.env.read_bytes(full).decode("utf-8", "surrogateescape")
        except (FileNotFoundError, IsADirectoryError, OSError):
            raise GraphError(f"no such file inside the repository: {path}")
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if not (1 <= start <= end <= len(lines)):
            raise GraphError(f"invalid line range {start}-{end} (the file has {len(lines)} lines)")
        if end - start + 1 > self.max_lines:
            raise GraphError(f"cite at most {self.max_lines} lines at a time")
        return {"path": posixpath.relpath(full, root), "start_line": start, "end_line": end,
                "excerpt": "\n".join(lines[start - 1:end]), "repo_state": self.repo_state()}

    def tool_call(self, call_id: str) -> dict | None:
        full = call_id if ":" in call_id else f"{self.ops.state.iid}:{call_id.upper()}"
        return self.calls.get(full)

    # ---- ledger -------------------------------------------------------------------------------------------------------
    def record_call(self, name: str, args: dict, output: str, is_error: bool, meta: dict | None, intervention_id: str | None) -> dict:
        self._n += 1
        cid = f"{self.ops.state.iid}:T{self._n}"
        meta = meta or {}
        rec = {"id": cid, "branch_id": self.branch_id, "name": name, "arguments": args, "output": output, "is_error": is_error,
               "command": meta.get("command"), "exit_code": meta.get("exit_code"), "duration_s": meta.get("duration_s"),
               "timed_out": meta.get("timed_out"), "path": meta.get("path") or (args or {}).get("path"),
               "intervention_id": intervention_id, "repo_state": self.repo_state(), "ts": time.time()}
        self.calls[cid] = rec
        return rec

    def open_if_needed(self, description: str) -> str:
        if self.open_intervention is None:
            self.open_intervention = self.ops.start_intervention(self.branch_id, description, getattr(self.env, "_base_tree", None))
        return self.open_intervention

    def detect_implicit_changes(self, description: str) -> bool:
        """After a command ran on the original repository: did it modify the tree on its own? If so, open an intervention retroactively."""
        if self.open_intervention is not None:
            return False
        if not self.env.dirty_state()["clean"]:
            self.open_if_needed(description)
            return True
        return False

    # ---- lifecycle ------------------------------------------------------------------------------------------------------
    def _verify(self, context: str, *, expect_dirty_reset: bool = False) -> dict:
        d = self.env.dirty_state()
        rec = {"context": context, "clean": d["clean"], "changed_files": d["changed_files"], "untracked": d["untracked"],
               "head_unchanged": d["head_unchanged"], "base_tree": d.get("base_tree")}
        return rec

    def _reset_and_verify(self, context: str) -> dict:
        before = self.env.dirty_state()
        self.env.reset_to_baseline()
        after = self.env.dirty_state()
        rec = {"context": context, "was_dirty": not before["clean"], "residual_before_reset": before["changed_files"] + before["untracked"],
               "verified_clean": after["clean"], "residual_after_reset": after["changed_files"] + after["untracked"],
               "head_unchanged": after["head_unchanged"], "base_tree": after.get("base_tree"), "intervention_id": self.open_intervention}
        self.resets.append(rec)
        self.ops.record_reset(context, rec["verified_clean"], rec)
        if not rec["verified_clean"]:
            raise InfraError(f"reset verification failed ({context}): residual {rec['residual_after_reset'][:5]}")
        return rec

    def establish_baseline(self) -> dict:
        d = self.env.dirty_state()
        rec = {"context": "baseline", "was_dirty": not d["clean"], "verified_clean": d["clean"], "residual_before_reset": d["changed_files"] + d["untracked"],
               "residual_after_reset": [], "head_unchanged": d["head_unchanged"], "base_tree": d.get("base_tree")}
        if not d["clean"]:
            return self._reset_and_verify("baseline")
        self.resets.append(rec)
        self.ops.record_reset("baseline", True, rec)
        return rec

    def begin_branch(self, branch_id: str) -> None:
        if self.open_intervention:
            raise InfraError("an intervention is still open from a previous branch")
        d = self.env.dirty_state()
        if not d["clean"]:                           # a previous branch leaked: repair, record, and verify
            self._reset_and_verify(f"before {branch_id}")
        else:
            rec = {"context": f"before {branch_id}", "was_dirty": False, "verified_clean": True, "residual_before_reset": [], "residual_after_reset": [],
                   "head_unchanged": True, "base_tree": d.get("base_tree")}
            self.resets.append(rec)
            self.ops.record_reset(rec["context"], True, rec)
        self.branch_id = branch_id

    def reset_changes(self, context: str) -> dict:
        """Close the open intervention (if any), restore the baseline and verify it."""
        iv = self.open_intervention
        rec = self._reset_and_verify(context)
        if iv:
            self.ops.end_intervention(iv, True, rec["residual_before_reset"])
            self.open_intervention = None
        return rec

    def end_branch(self, branch_id: str) -> dict:
        d = self.env.dirty_state()
        if not d["clean"] and self.open_intervention is None:
            self.open_if_needed("residual changes at branch end")          # make the leak visible in the log before repairing it
        rec = self.reset_changes(f"after {branch_id}")
        self.branch_id = None
        return rec

    def final_reset(self) -> dict:
        self.branch_id = None
        return self.reset_changes("after diagnosis (before implementation)")

    def write_artifacts(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "reset_verification.json").write_text(json.dumps(self.resets, indent=2))
        with open(directory / "tool_calls.jsonl", "w") as f:
            for c in self.calls.values():
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
