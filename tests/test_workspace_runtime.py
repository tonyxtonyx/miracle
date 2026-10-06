import dataclasses
import subprocess

from miracle.mock import EmptyRuntime, NoopEditRuntime
from miracle.runtime import Task
from miracle.workspace import create_workspace, git


def test_workspace_has_no_future_history_and_diff_roundtrips(upstream, home):
    tmpl, inst = upstream
    ws = create_workspace(inst, home / "ws", tmpl)
    assert git(ws.path, "rev-parse", "HEAD").strip() == inst["base_commit"]
    assert (ws.path / "a.py").read_text() == "x = 1\n"
    assert git(ws.path, "rev-list", "--all", "--count").strip() == "1"      # the fix commit is absent
    assert git(ws.path, "remote").strip() == ""                               # nothing to fetch from
    assert ws.diff() == ""
    (ws.path / "a.py").write_text("x = 3\n")
    (ws.path / "new.py").write_text("y = 1\n")
    d = ws.diff()
    assert "a/a.py" in d and "b/new.py" in d and d.endswith("\n")
    ws.reset()
    assert ws.diff() == ""


def test_workspaces_are_independent_copies(upstream, home):
    tmpl, inst = upstream
    a = create_workspace(inst, home / "a", tmpl)
    (a.path / "a.py").write_text("dirty\n")
    b = create_workspace(inst, home / "b", tmpl)
    assert b.diff() == ""


def test_task_exposes_only_whitelisted_fields(upstream, home):
    tmpl, inst = upstream
    task = Task.from_instance(inst, create_workspace(inst, home / "w", tmpl))
    assert {f.name for f in dataclasses.fields(task)} == {
        "instance_id", "repo", "base_commit", "problem_statement", "workspace", "env"}
    assert "SECRET" not in repr(task)


def test_mock_runtimes(upstream, home):
    tmpl, inst = upstream
    task = Task.from_instance(inst, create_workspace(inst, home / "w", tmpl))
    assert EmptyRuntime().solve(task).patch == ""
    p = NoopEditRuntime().solve(task).patch
    # the patch must apply cleanly to a pristine tree
    ws2 = create_workspace(inst, home / "w2", tmpl)
    (ws2.path / "p.diff").write_text(p)
    subprocess.run(["git", "apply", "--check", "p.diff"], cwd=ws2.path, check=True)
