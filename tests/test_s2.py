import json

import pytest

from miracle.agents.s0 import S0Runtime
from miracle.agents.s1 import S1Runtime
from miracle.agents.s2 import (CHECK_PROMPT, S2Runtime, build_check_prompt, has_source_modification, parse_decision)
from miracle.agents.termination import CandidateTracker, candidate_analysis, is_test_path, run_succeeded
from miracle.llm.client import LLMError
from test_llm import Scripted
from test_s0 import _task, env  # noqa: F401

ISSUE = "f() should return 2, not 1"

# scripted agent actions ------------------------------------------------------------------------------------------
READ = (None, [("read_file", {"path": "pkg/a.py"})])
EDIT_F = (None, [("edit_file", {"path": "pkg/a.py", "old_str": "def f():\n    return 1", "new_str": "def f():\n    return 2"})])
EDIT_G = (None, [("edit_file", {"path": "pkg/a.py", "old_str": "def g():\n    return 1", "new_str": "def g():\n    return 2"})])
PASS = (None, [("run_command", {"command": "echo '=== 3 passed in 0.01s ===' # pytest"})])
FAIL = (None, [("run_command", {"command": "echo '=== 1 failed, 2 passed ===' ; exit 1 # pytest"})])
ZERO = (None, [("run_command", {"command": "echo 'Ran 0 tests in 0.000s' # pytest"})])
DONE = ("all done", [])


def run(env, turns, max_steps=50, **kw):
    c = Scripted(turns)
    res = S2Runtime(client=c, max_steps=max_steps, **kw).solve(_task(env, ISSUE))
    return res, c


def check_calls(c):
    return [i for i, t in enumerate(c.tools_seen) if t is None]          # check calls pass no tools


def agent_calls(c):
    return [i for i, t in enumerate(c.tools_seen) if t is not None]


# ---- pure pieces ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize("text,fr,expect", [
    ("COMPLETE", None, ("COMPLETE", "")),
    ("  complete.\n", None, ("COMPLETE", "")),
    ("COMPLETE\n\nThe diff fixes f.", None, ("COMPLETE", "")),
    ("INCOMPLETE: g() is not updated", None, ("INCOMPLETE", "g() is not updated")),
    ("incomplete - missing docs", None, ("INCOMPLETE", "missing docs")),
    ("INCOMPLETE:\nline two", None, ("INCOMPLETE", "line two")),
    ("Completely unrelated", None, ("INVALID", None)),            # must not be read as COMPLETE
    ("I think it looks fine", None, ("INVALID", None)),
    ("", None, ("INVALID", "empty reply")),
    (None, None, ("INVALID", "empty reply")),
    ("COMPLETE", "length", ("INVALID", "check reply truncated")),  # truncated => never trusted
])
def test_parse_decision(text, fr, expect):
    d, reason = parse_decision(text, fr)
    assert d == expect[0] and (expect[1] is None or reason == expect[1])


def test_test_path_and_success_predicates():
    for p in ("tests/a.py", "pkg/tests/x.py", "test_a.py", "pkg/test_a.py", "a_test.py", "conftest.py", "testing/x.py"):
        assert is_test_path(p), p
    for p in ("pkg/a.py", "contest.py", "latest.py", "sympy/core/attest.py"):
        assert not is_test_path(p), p
    ok = lambda out, **m: run_succeeded({"exit_code": 0, "timed_out": False, **m}, out)
    assert ok("=== 5 passed in 1s ===") and ok("Ran 3 tests in 0.1s\n\nOK") and ok("no recognisable runner output")
    assert not ok("=== 1 failed, 4 passed ===") and not ok("Ran 0 tests in 0.0s\n\nOK") and not ok("=== 2 errors ===")
    assert not ok("=== 5 passed ===", exit_code=1) and not ok("=== 5 passed ===", timed_out=True) and not ok("0 passed")


def test_has_source_modification():
    mod = "diff --git a/pkg/a.py b/pkg/a.py\nindex 1..2 100644\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-x\n+y\n"
    new = "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+z\n"
    tst = mod.replace("pkg/a.py", "tests/test_a.py")
    assert has_source_modification(mod) and has_source_modification(new + mod)
    assert not has_source_modification("") and not has_source_modification(new) and not has_source_modification(tst)


def test_check_prompt_contains_only_the_allowed_inputs():
    diff = "diff --git a/pkg/a.py b/pkg/a.py\n--- a/pkg/a.py\n+++ b/pkg/a.py\n@@ -1 +1 @@\n-x\n+y\n"
    p = build_check_prompt(ISSUE, diff, [{"command": "pytest -q", "output": "exit_code: 0\n3 passed"}])
    assert ISSUE in p and "pkg/a.py" in p and "+y" in p and "$ pytest -q" in p and "3 passed" in p
    assert "COMPLETE" in p and "INCOMPLETE: <the specific unresolved requirement>" in p
    long = build_check_prompt(ISSUE, "x" * 50_000, [{"command": "t", "output": "y" * 50_000}])
    assert len(long) < 20_000 and "diff truncated" in long and "earlier output omitted" in long


def _tr(name, err=False, **kw):
    return {"name": name, "is_error": err, "arguments": kw.pop("arguments", {}), **kw}


def _test_tr(code=0, out="exit_code: 0\n=== 2 passed ==="):
    return _tr("run_command", meta={"command": "pytest x", "is_test": True, "exit_code": code, "timed_out": False}, output=out)


def test_tracker_requires_source_edit_then_successful_test_in_that_order():
    t = CandidateTracker()
    t.observe(1, _test_tr())                                            # test before any edit: no evidence
    assert t.new_candidate(1) is None
    t.observe(2, _tr("edit_file", arguments={"path": "tests/test_a.py"}, meta={"path": "tests/test_a.py"}))
    t.observe(2, _test_tr())                                            # editing only a test file is not a source edit
    assert t.new_candidate(2) is None
    t.observe(3, _tr("write_file", arguments={"path": "scratch.py"}))
    t.observe(3, _test_tr())                                            # scratch file creation is not a source edit
    assert t.new_candidate(3) is None
    t.observe(4, _tr("edit_file", arguments={"path": "pkg/a.py"}, meta={"path": "pkg/a.py"}))
    t.observe(4, _test_tr(code=1, out="exit_code: 1\n=== 1 failed ==="))
    assert t.new_candidate(4) is None                                   # failing test
    t.observe(5, _test_tr())
    c = t.new_candidate(5)
    assert c and c["last_source_edit_step"] == 4 and c["modified_source_files"] == ["pkg/a.py"] and c["step"] == 5
    assert t.new_candidate(6) is None                                   # only fires in the step the test ran
    t.observe(7, _test_tr()); t.observe(7, _tr("edit_file", arguments={"path": "pkg/a.py"}, meta={"path": "pkg/a.py"}))
    assert t.new_candidate(7) is None                                   # edit after the test (same step) invalidates it


def test_candidate_analysis_offline():
    mk = lambda i, results, cost: {"index": i, "tool_results": results, "usage": {"prompt_tokens": 100 * i, "completion_tokens": 10, "cost_usd": cost}}
    ed = _tr("edit_file", arguments={"path": "pkg/a.py"}, meta={"path": "pkg/a.py"})
    trace = {"steps": [mk(1, [], 1.0), mk(2, [ed], 1.0), mk(3, [_test_tr()], 1.0), mk(4, [], 2.0), mk(5, [], 3.0)]}
    a = candidate_analysis(trace)
    assert a["first_candidate_step"] == 3 and a["candidate_steps"] == [3] and a["total_steps"] == 5
    assert a["steps_after_first_candidate"] == 2 and a["cost_after"] == 5.0 and a["cost_total"] == 8.0
    assert a["prompt_tokens_after"] == 900 and a["completion_tokens_after"] == 20
    n = candidate_analysis({"steps": [mk(1, [], 1.0)]})
    assert n["first_candidate_step"] is None and n["steps_after_first_candidate"] is None and n["cost_after"] is None


# ---- controller flows ----------------------------------------------------------------------------------------
def test_complete_terminates_the_run_and_accounts_for_the_check(env):
    res, c = run(env, [READ, EDIT_F, PASS, ("COMPLETE", [])])
    t, m = res.metadata["trace"], res.metadata
    assert t["stop_reason"] == "completion_check" and len(t["steps"]) == 3 and t["final_answer"] is None
    assert "return 2" in res.patch                                           # the patch is still the final diff
    rec = t["controller"]["records"]
    assert len(rec) == 1 and rec[0]["decision"] == "COMPLETE" and rec[0]["step"] == 3
    assert rec[0]["trigger"]["test_command"].endswith("# pytest") and rec[0]["trigger"]["last_source_edit_step"] == 2
    assert rec[0]["trigger"]["modified_source_files"] == ["pkg/a.py"] and rec[0]["modified_files"] == ["pkg/a.py"]
    ctl = m["controller"]
    assert ctl["terminated_by_controller"] and ctl["termination_step"] == 3 and ctl["checks"] == 1 and ctl["check_steps"] == [3]
    # accounting: Scripted bills 10 prompt / 5 completion / $0.001 per call
    assert (res.usage.llm_calls, res.usage.prompt_tokens, res.usage.completion_tokens) == (4, 40, 20)
    assert ctl["usage"]["prompt_tokens"] == 10 and ctl["agent_usage_excluding_checks"]["prompt_tokens"] == 30
    assert res.usage.cost_usd == pytest.approx(0.004) and ctl["usage"]["cost_usd"] == pytest.approx(0.001)
    assert m["stats"]["completion_checks"] == 1 and m["stats"]["steps"] == 3 and m["stats"]["stop_reason"] == "completion_check"
    assert m["stats"]["candidate"]["first_candidate_step"] == 3


def test_check_call_sees_only_issue_diff_and_tests_and_no_tools(env):
    res, c = run(env, [READ, EDIT_F, PASS, ("COMPLETE", [])])
    i = check_calls(c)
    assert len(i) == 1
    msgs = c.calls[i[0]]
    assert len(msgs) == 1 and msgs[0]["role"] == "user"                      # no system prompt, no agent history
    body = msgs[0]["content"]
    assert ISSUE in body and "+    return 2" in body and "3 passed" in body and "$ echo" in body
    assert "read_file" not in body and "You are a software engineering agent" not in body
    assert c.opts_seen[i[0]] == {"max_tokens": 300, "temperature": 0.0, "reasoning_effort": "none"}


def test_incomplete_is_recorded_and_the_agent_never_sees_it(env):
    res, c = run(env, [READ, EDIT_F, PASS, ("INCOMPLETE: g() still returns 1", []), DONE])
    t = res.metadata["trace"]
    assert t["stop_reason"] == "final_answer" and t["final_answer"] == "all done" and len(t["steps"]) == 4
    r = t["controller"]["records"][0]
    assert r["decision"] == "INCOMPLETE" and r["reason"] == "g() still returns 1"
    assert not t["controller"]["terminated_by_controller"]
    for i in agent_calls(c):                                                 # nothing from the controller reaches the agent
        blob = json.dumps(c.calls[i])
        assert "INCOMPLETE" not in blob and "unresolved" not in blob and "g() still returns 1" not in blob
    assert [m["role"] for m in t["messages"]].count("user") == 1             # only the original task message


def test_same_diff_is_not_rechecked_but_a_changed_diff_is(env):
    turns = [READ, EDIT_F, PASS, ("INCOMPLETE: more", []),
             PASS,                                                          # same code, tested again: no new check
             EDIT_G, PASS, ("COMPLETE", [])]                                # code changed: checked again
    res, c = run(env, turns)
    ctl = res.metadata["controller"]
    assert ctl["checks"] == 2 and ctl["check_steps"] == [3, 6]
    assert [r["decision"] for r in res.metadata["trace"]["controller"]["records"]] == ["INCOMPLETE", "COMPLETE"]
    assert res.metadata["trace"]["stop_reason"] == "completion_check"


@pytest.mark.parametrize("reply,finish,decision", [
    ("looks fine to me", None, "INVALID"), ("COMPLETE", "length", "INVALID"), ("", None, "INVALID"),
    (LLMError("HTTP 500", 500), None, "ERROR")])
def test_uncertain_check_never_terminates(env, reply, finish, decision):
    res, _ = run(env, [READ, EDIT_F, PASS, (reply, [], finish) if finish else (reply, []), DONE])
    t = res.metadata["trace"]
    assert t["controller"]["records"][0]["decision"] == decision
    assert t["stop_reason"] == "final_answer" and not t["controller"]["terminated_by_controller"]


@pytest.mark.parametrize("name,turns", [
    ("failing test", [READ, EDIT_F, FAIL, DONE]),
    ("zero tests ran", [READ, EDIT_F, ZERO, DONE]),
    ("test before edit", [READ, PASS, EDIT_F, DONE]),
    ("no test at all", [READ, EDIT_F, DONE]),
    ("scratch file only", [(None, [("write_file", {"path": "scratch.py", "content": "x=1\n"})]), PASS, DONE]),
    ("test file only", [(None, [("write_file", {"path": "tests/test_x.py", "content": "a = 1\n"})]),
                        (None, [("edit_file", {"path": "tests/test_x.py", "old_str": "a = 1", "new_str": "a = 2"})]), PASS, DONE]),
])
def test_no_check_without_candidate_evidence(env, name, turns):
    res, c = run(env, turns)
    assert res.metadata["controller"]["checks"] == 0 and check_calls(c) == [], name
    assert res.metadata["trace"]["stop_reason"] == "final_answer"


def test_edit_reverted_before_the_test_means_no_check(env):
    undo = (None, [("run_command", {"command": "git checkout -- pkg/a.py"})])
    res, c = run(env, [READ, EDIT_F, undo, PASS, DONE])
    assert res.metadata["controller"]["checks"] == 0 and check_calls(c) == []     # diff has no source modification


def test_max_checks_bounds_the_controller(env):
    turns = [READ, EDIT_F, PASS, ("INCOMPLETE: a", []), EDIT_G, PASS, DONE]
    res, c = run(env, turns, max_checks=1)
    assert res.metadata["controller"]["checks"] == 1 and len(check_calls(c)) == 1


def test_no_check_on_the_last_step_there_is_nothing_left_to_save(env):
    res, c = run(env, [READ, EDIT_F, PASS], max_steps=3)
    assert res.metadata["controller"]["checks"] == 0 and check_calls(c) == []
    assert res.metadata["trace"]["stop_reason"] == "max_steps"


def test_checks_do_not_consume_the_agent_step_budget(env):
    res, _ = run(env, [READ, EDIT_F, PASS, ("INCOMPLETE: x", []), DONE], max_steps=4)
    assert len(res.metadata["trace"]["steps"]) == 4 and res.metadata["trace"]["stop_reason"] == "final_answer"


# ---- the experimental contrast is only the controller -----------------------------------------------------------
def test_s2_is_s1_plus_only_the_controller(monkeypatch):
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-secret-123")
    s1, s2 = S1Runtime(), S2Runtime()
    assert S0Runtime().make_controller(None) is None and s1.make_controller(None) is None
    d1, d2 = s1.describe(), s2.describe()
    assert s2.name == "s2-qwen3.5-9b" and d2["class"] == "S2Runtime"
    assert all(d1[k] == d2[k] for k in ("max_steps", "system_prompt", "llm", "tools", "length_nudge"))
    assert d2["controller"]["check_prompt"] == CHECK_PROMPT and d2["controller"]["max_checks"] == 5
    assert "controller" not in d1 and "sk-secret-123" not in json.dumps(d2)


def test_without_candidates_s2_behaves_exactly_like_s1(env, tmp_path_factory):
    from test_s0 import LocalEnv
    turns = [READ, (None, [("list_files", {"path": "."})]), DONE]
    r1 = S1Runtime(client=Scripted(list(turns))).solve(_task(env))
    r2 = S2Runtime(client=Scripted(list(turns))).solve(_task(LocalEnv(tmp_path_factory.mktemp("e2"))))
    strip = lambda t: [(s["response"]["tool_calls"] and s["response"]["tool_calls"][0]["name"], s["response"]["content"]) for s in t["steps"]]
    assert strip(r1.metadata["trace"]) == strip(r2.metadata["trace"])
    assert (r1.usage.llm_calls, r1.usage.prompt_tokens) == (r2.usage.llm_calls, r2.usage.prompt_tokens)
    assert r2.metadata["controller"]["checks"] == 0


def test_provider_failure_aborts_the_instance_and_is_never_scored(upstream, home, monkeypatch):
    """A provider outage (stop_reason llm_error) must not become a graded agent result."""
    from miracle.dataset import DatasetRef
    from miracle.runner import InfraError, Run
    from miracle.runtime import AgentResult, Usage
    _, inst = upstream
    monkeypatch.setenv("DEEPINFRA_API_KEY", "sk-secret-123")
    rt = S2Runtime()
    run = Run("r")
    run.create(DatasetRef("d", "rev", "sha"), [inst], rt, None)
    failed = AgentResult("diff --git a/x b/x\n", Usage(), {"stop_reason": "llm_error", "trace": {"error": "network error: ReadTimeout"}})
    monkeypatch.setattr(Run, "_solve", lambda self, *a, **k: (failed, 0.1))
    with pytest.raises(InfraError, match="provider failed"):
        run.generate(rt)
    assert not (run.agent_dir(inst["instance_id"]) / "result.json").exists()           # nothing to score or skip on resume
    arch = list((run.dir / "infra_failures").iterdir())
    assert len(arch) == 1 and (arch[0] / "result.json").exists() and (arch[0] / "trace.json").exists()
