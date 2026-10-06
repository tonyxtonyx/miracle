import json
import subprocess

import pytest

from miracle.agents.env_tools import EnvToolBox
from miracle.agents.s0 import S0Runtime, SYSTEM_PROMPT, command_log, parse_test_summary, trajectory_stats
from miracle.environment import Environment, ExecResult
from miracle.runtime import Task
from test_llm import Scripted


class LocalEnv(Environment):
    """Test double: a real git repo in a temp dir, real bash. Same contract as DockerEnvironment."""
    def __init__(self, root):
        self.root = str(root)
        g = lambda *a: subprocess.run(["git", *a], cwd=root, check=True, capture_output=True,
                                      env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                                           "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"]})
        (root / "pkg").mkdir()
        (root / "pkg" / "a.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 1\n")
        (root / "README.md").write_text("hello\n")
        g("init", "-q"); g("add", "-A"); g("commit", "-qm", "base")
        self.g = g

    def exec(self, command, timeout=120):
        try:
            r = subprocess.run(["bash", "-c", command], cwd=self.root, capture_output=True, text=True, timeout=timeout)
            return ExecResult(r.returncode, r.stdout + r.stderr, False, 0.01)
        except subprocess.TimeoutExpired:
            return ExecResult(124, "", True, float(timeout))

    def read_bytes(self, path):
        import pathlib
        p = pathlib.Path(path)
        if p.is_dir():
            raise IsADirectoryError(path)
        if not p.exists():
            raise FileNotFoundError(path)
        return p.read_bytes()

    def write_bytes(self, path, data):
        import pathlib
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_bytes(data)

    def diff(self):
        self.g("add", "-A")
        return subprocess.run(["git", "diff", "--cached", "HEAD"], cwd=self.root, capture_output=True, text=True).stdout

    def changed_files(self):
        self.g("add", "-A")
        return subprocess.run(["git", "diff", "--cached", "--name-only", "HEAD"], cwd=self.root, capture_output=True,
                              text=True).stdout.split()


@pytest.fixture
def env(tmp_path):
    return LocalEnv(tmp_path)


def test_toolbox_read_list_search(env):
    tb = EnvToolBox(env)
    assert tb.call("list_files", {"path": "."})[0].split() == [".git/", "README.md", "pkg/"]
    out, err = tb.call("read_file", {"path": "pkg/a.py"})
    assert not err and "     2\t    return 1" in out and "showing lines" not in out
    out, _ = tb.call("read_file", {"path": "pkg/a.py", "start_line": 5, "end_line": 6})
    assert out.splitlines()[0].startswith("     5\t") and "[showing lines 5-6 of 6]" in out
    assert "pkg/a.py:2:" in tb.call("search_code", {"query": "RETURN 1"})[0]
    assert tb.call("search_code", {"query": "zzzz"})[0] == "(no matches)"
    assert tb.call("search_code", {"query": "it's"})[0] == "(no matches)"       # quoting is safe


def test_toolbox_sandbox_and_errors(env):
    tb = EnvToolBox(env)
    for bad in ("../x", "/etc/passwd", "pkg/../../x"):
        assert "outside the repository" in tb.call("read_file", {"path": bad})[0]
    assert "not accessible" in tb.call("read_file", {"path": ".git/config"})[0]
    assert "no such file" in tb.call("read_file", {"path": "nope.py"})[0]
    assert "is a directory" in tb.call("read_file", {"path": "pkg"})[0]
    assert tb.call("read_file", {})[1] and tb.call("bogus", {})[1] and tb.call("git_diff", {"x": 1})[1]


def test_edit_semantics_and_diff(env):
    tb = EnvToolBox(env)
    assert tb.call("git_diff", {}) == ("(no changes yet)", False)
    out, err = tb.call("edit_file", {"path": "pkg/a.py", "old_str": "return 1", "new_str": "return 2"})
    assert err and "occurs 2 times" in out                                            # ambiguous -> refuse
    assert "not found" in tb.call("edit_file", {"path": "pkg/a.py", "old_str": "nothere", "new_str": "x"})[0]
    out, err, meta = tb.call("edit_file", {"path": "pkg/a.py", "old_str": "def f():\n    return 1",
                                           "new_str": "def f():\n    return 42"})
    assert not err and meta == {"path": "pkg/a.py"} and "return 42" in out
    tb.call("write_file", {"path": "new/dir/x.py", "content": "y = 1\n"})
    d = tb.call("git_diff", {})[0]
    assert "+    return 42" in d and "new/dir/x.py" in d
    assert sorted(env.changed_files()) == ["new/dir/x.py", "pkg/a.py"]
    assert (env.root and open(env.root + "/pkg/a.py").read().count("return 1") == 1)   # only the targeted one changed


def test_run_command_meta_and_timeout(env):
    tb = EnvToolBox(env)
    out, err, meta = tb.call("run_command", {"command": "echo hi; exit 3"})
    assert err and out.startswith("exit_code: 3\nhi") and meta["exit_code"] == 3 and not meta["is_test"]
    assert tb.call("run_command", {"command": "python -m pytest -x tests/"})[2]["is_test"]
    assert tb.call("run_command", {"command": "./tests/runtests.py foo"})[2]["is_test"]
    out, err, meta = tb.call("run_command", {"command": "sleep 5", "timeout": 1})
    assert meta["timed_out"] and "timed out" in out
    big = tb.call("run_command", {"command": "python3 -c \"print('x'*50000)\""})[0]
    assert len(big) < 13000 and "omitted" in big


def test_parse_test_summary():
    py = "....\n=== 2 failed, 5 passed, 1 skipped in 3.2s ==="
    assert parse_test_summary(py)["counts"] == {"failed": 2, "passed": 5, "skipped": 1}
    dj = "Ran 22 tests in 0.1s\n\nOK\n"
    assert parse_test_summary(dj)["ran"] == 22 and parse_test_summary(dj)["ok"]
    assert not parse_test_summary("Ran 3 tests in 0.1s\n\nFAILED (failures=1)\n")["ok"]
    assert "counts" not in parse_test_summary("nothing recognisable")


def _task(env, text="fix it"):
    return Task("i-1", "r", "b", text, env=env)


def test_s0_defaults_are_the_specified_baseline(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-secret-123")
    rt = S0Runtime()
    d = rt.describe()
    assert rt.environment == "container" and d["max_steps"] == 50 and d["system_prompt"] == SYSTEM_PROMPT
    assert (d["llm"]["model"], d["llm"]["temperature"], d["llm"]["reasoning_effort"]) == ("Qwen/Qwen3.5-9B", 0.0, "none")
    assert d["name"] == "s0-qwen3.5-9b" and "sk-secret-123" not in json.dumps(d)


def test_s0_trajectory_end_to_end_with_scripted_model(env):
    turns = [(None, [("search_code", {"query": "return 1"})]),
             (None, [("edit_file", {"path": "pkg/a.py", "old_str": "def f():\n    return 1", "new_str": "def f():\n    return 2"})]),
             (None, [("edit_file", {"path": "pkg/a.py", "old_str": "nothere", "new_str": "x"})]),
             (None, [("run_command", {"command": "python -m pytest pkg"})]),
             (None, [("git_diff", {})]),
             ("Fixed f.", [])]
    c = Scripted(turns)
    res = S0Runtime(client=c).solve(_task(env))
    m = res.metadata
    assert "def f():\n+    return 2" in res.patch.replace("-    return 1\n", "") or "+    return 2" in res.patch
    assert m["modified_files"] == ["pkg/a.py"] and m["stop_reason"] == "final_answer" and m["final_answer"] == "Fixed f."
    assert len(m["tests"]) == 1 and m["tests"][0]["step"] == 4 and m["tests"][0]["is_test"]
    st = m["stats"]
    assert (st["steps"], st["tool_calls"], st["successful_edits"], st["edit_errors"], st["first_edit_step"],
            st["tests_run"], st["tested_after_last_edit"]) == (6, 5, 1, 1, 2, 1, True)
    assert st["calls_by_tool"]["edit_file"] == 2
    t = m["trace"]
    assert t["final_diff"] == res.patch and t["schema"] == 2 and t["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [s["n_messages_sent"] for s in t["steps"]] == [2, 4, 6, 8, 10, 12]            # every request reconstructible
    assert c.calls[0][0]["content"] == SYSTEM_PROMPT and c.calls[0][1]["content"] == "fix it"
    assert res.usage.llm_calls == 6 and res.usage.tool_calls == 5


def test_s0_keeps_partial_patch_when_budget_runs_out(env):
    turns = [(None, [("write_file", {"path": "partial.py", "content": "x = 1\n"})])] * 3
    res = S0Runtime(client=Scripted(turns), max_steps=3).solve(_task(env))
    assert res.metadata["stop_reason"] == "max_steps" and "partial.py" in res.patch
    assert res.metadata["stats"]["tests_run"] == 0 and not res.metadata["stats"]["tested_after_last_edit"]


def test_trajectory_stats_repetition_and_finish_reasons():
    step = lambda i, tr, fr="tool_calls", content="x": {"index": i, "response": {"finish_reason": fr, "content": content, "tool_calls": tr},
                                                       "tool_results": tr}
    r = lambda: {"name": "list_files", "raw_arguments": "{}", "is_error": False, "arguments_valid": True}
    trace = {"steps": [step(1, [r()]), step(2, [r()]), step(3, [r()]), step(4, [], "length", "")], "stop_reason": "length"}
    st = trajectory_stats(trace)
    assert st["repeated_identical_calls"] == 2 and st["finish_reasons"] == {"tool_calls": 3, "length": 1}
    assert st["empty_steps"] == 1 and command_log(trace) == []


def test_infra_failure_is_not_scored_as_agent_error(upstream, home, monkeypatch):
    """A broken environment must abort the run and leave no result, never become an agent failure."""
    import miracle.evaluator as ev
    from miracle.dataset import DatasetRef
    from miracle.runner import InfraError, Run
    _, inst = upstream
    inst = {**inst, "image": "swebench/sweb.eval.x86_64.x_1776_y-1:latest"}
    monkeypatch.setattr(ev, "docker_image_id", lambda i: None)
    monkeypatch.setattr(ev, "ensure_images", lambda imgs, log=print: {imgs[0]: "no space left on device"})
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-secret-123")
    rt = S0Runtime()
    run = Run("r")
    run.create(DatasetRef("d", "rev", "sha"), [inst], rt, None)
    with pytest.raises(InfraError, match="no space left"):
        run.generate(rt)
    assert not (run.agent_dir(inst["instance_id"]) / "result.json").exists()


def test_export_scrubs_local_paths_and_skips_instances(home, tmp_path, monkeypatch):
    from pathlib import Path
    from miracle.export import export_run, scrub
    run = home / "runs" / "r"
    (run / "agent" / "i").mkdir(parents=True)
    (run / "manifest.json").write_text(f'{{"cmd": "{home}/runs/r/eval", "home": "{Path.home()}/x"}}')
    (run / "instances.jsonl").write_text("big")
    (run / "agent" / "i" / "patch.diff").write_text("diff")
    dest = tmp_path / "out"
    assert export_run("r", dest) == 2
    m = (dest / "manifest.json").read_text()
    assert str(home) not in m and "<project>/runs/r/eval" in m and "~/x" in m
    assert not (dest / "instances.jsonl").exists()
