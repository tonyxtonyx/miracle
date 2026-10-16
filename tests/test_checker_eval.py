import subprocess

import pytest

from miracle.checker_eval.dataset import candidates_in_trace
from miracle.checker_eval.evaluate import confusion, subsets
from miracle.checker_eval.replay import ReplayEnv, abbrev_len, normalize_patch, replay_trace
from miracle.stats import wilson


# ---- metrics ----------------------------------------------------------------------------------------------------
def test_confusion_matrix_and_the_primary_safety_metric():
    pairs = ([("RESOLVED", "COMPLETE")] * 6 + [("UNRESOLVED", "COMPLETE")] * 2          # TP=6 FP=2
             + [("RESOLVED", "INCOMPLETE")] * 1 + [("UNRESOLVED", "INCOMPLETE")] * 3    # FN=1 TN=3
             + [("UNRESOLVED", "INVALID")])                                             # counts as not-COMPLETE (TN)
    c = confusion(pairs)
    assert (c["TP"], c["FP"], c["FN"], c["TN"], c["n"]) == (6, 2, 1, 4, 13)
    assert c["complete_precision"] == pytest.approx(6 / 8) and c["complete_recall"] == pytest.approx(6 / 7)
    assert c["false_complete_rate"] == pytest.approx(2 / 6) and c["base_rate_resolved"] == pytest.approx(7 / 13)
    assert c["invalid_or_error"] == 1
    lo, hi = c["precision_ci95"]
    assert lo < 0.75 < hi and (lo, hi) == tuple(round(x, 3) for x in wilson(6, 8))


def test_confusion_handles_empty_denominators():
    c = confusion([("RESOLVED", "COMPLETE")] * 3)                 # no negatives: false-COMPLETE rate undefined
    assert c["complete_precision"] == 1.0 and c["false_complete_rate"] is None
    c = confusion([("UNRESOLVED", "INCOMPLETE")] * 3)             # never says COMPLETE: precision undefined
    assert c["complete_precision"] is None and c["complete_recall"] is None and c["false_complete_rate"] == 0.0
    assert confusion([])["n"] == 0


def test_subsets_exclude_unknown_labels_and_split_kinds():
    mk = lambda k, lab, el: {"kind": k, "label": lab, "online_eligible": el, "sample_id": f"{k}{lab}{el}"}
    S = [mk("final", "RESOLVED", True), mk("final", "UNRESOLVED", False), mk("intermediate", "UNKNOWN", True),
         mk("intermediate", "UNRESOLVED", True), mk("intermediate", "PENDING", True)]
    sub = subsets(S)
    assert len(sub["all labelled"]) == 3                               # UNKNOWN / PENDING never enter precision/recall
    assert len(sub["online_eligible (primary)"]) == 2 and len(sub["intermediate"]) == 1
    assert len(sub["final, not eligible (stress test)"]) == 1


# ---- patch helpers ---------------------------------------------------------------------------------------------
def test_normalize_patch_and_abbrev():
    p = "The following paths are ignored:\n.pytest_cache\ndiff --git a/a b/a\nindex 3a1b2c3d4e..4f5a6b7c8d 100644\n"
    assert normalize_patch(p).startswith("diff --git a/a b/a") and "ignored" not in normalize_patch(p)
    assert normalize_patch("") == "" and normalize_patch("no diff here") == ""
    assert abbrev_len(p) == 10 and abbrev_len("diff --git a/x b/x\n") is None


# ---- candidates from a trace (same tracker as the live controller) --------------------------------------------
def _res(name, err=False, **kw):
    return {"name": name, "is_error": err, "arguments": kw.pop("arguments", {}), **kw}


def _t(code=0, out="exit_code: 0\n=== 2 passed ==="):
    return _res("run_command", meta={"command": "pytest x", "is_test": True, "exit_code": code, "timed_out": False}, output=out)


def test_candidates_in_trace_and_end_state_tests():
    ed = _res("edit_file", arguments={"path": "pkg/a.py"}, meta={"path": "pkg/a.py"})
    trace = {"steps": [{"index": 1, "tool_results": [ed, _t()]}, {"index": 2, "tool_results": []},
                       {"index": 3, "tool_results": [_t()]}, {"index": 4, "tool_results": [ed]}]}
    cands, end = candidates_in_trace(trace)
    assert [c["step"] for c in cands] == [1, 3] and len(cands[0]["tests"]) == 1 and len(cands[1]["tests"]) == 2
    assert end == []                                                    # the step-4 edit invalidated earlier test evidence


# ---- replay on a real git repo -----------------------------------------------------------------------------------
@pytest.fixture
def pristine(tmp_path):
    repo = tmp_path / "pristine"
    repo.mkdir()
    sh = lambda c: subprocess.run(c, shell=True, cwd=repo, check=True, capture_output=True, text=True).stdout
    sh("git init -q && git config user.email t@t && git config user.name t")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "a.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 1\n")
    sh("git add -A && git commit -qm base")
    return repo


def _edit(old, new, path="pkg/a.py"):
    return _res("edit_file", arguments={"path": path, "old_str": old, "new_str": new}, meta={"path": path})


def test_replay_reconstructs_each_step_and_matches_a_direct_edit(pristine, tmp_path):
    trace = {"steps": [
        {"index": 1, "tool_results": [_edit("def f():\n    return 1", "def f():\n    return 2")]},
        {"index": 2, "tool_results": [_res("write_file", arguments={"path": "scratch.py", "content": "x = 1\n"})]},
        {"index": 3, "tool_results": [_edit("def g():\n    return 1", "def g():\n    return 2")]},
        {"index": 4, "tool_results": [_edit("nothere", "x", ), ]}]}
    trace["steps"][3]["tool_results"][0]["is_error"] = True              # a failed edit is skipped, as originally
    rep = replay_trace(pristine, trace, {1, 3}, abbrev=10)
    d1, d3 = rep["diffs"][1], rep["diffs"][3]
    assert d1.count("+    return 2") == 1 and "scratch.py" not in d1                  # state after step 1 only
    assert "scratch.py" in d3 and d3.count("+    return 2") == 2                      # steps 2 and 3 applied
    assert rep["final"] == d3 and rep["divergences"] == []
    import re
    assert re.search(r"(?m)^index \w{10}\.\.\w{10} ", d1)                              # abbrev calibrated to the recorded length
    # byte-identical to what the production extraction yields for the same final tree
    direct = tmp_path / "direct"
    import shutil
    shutil.copytree(pristine, direct)
    env = ReplayEnv(direct, 10)
    (direct / "pkg" / "a.py").write_text("def f():\n    return 2\n\n\ndef g():\n    return 2\n")
    (direct / "scratch.py").write_text("x = 1\n")
    assert env.diff() == rep["final"]


def test_replay_flags_an_edit_that_cannot_be_reproduced(pristine):
    ok_in_original = _edit("def f():\n    return 99", "def f():\n    return 2")      # recorded as success, but absent here
    rep = replay_trace(pristine, {"steps": [{"index": 1, "tool_results": [ok_in_original]}]}, {1}, abbrev=7)
    assert rep["divergences"] and rep["divergences"][0]["step"] == 1


def test_replay_ignores_binary_and_pycache_like_production(pristine):
    trace = {"steps": [{"index": 1, "tool_results": [_edit("def f():\n    return 1", "def f():\n    return 3")]}]}
    rep = replay_trace(pristine, trace, {1}, abbrev=None)
    assert "pkg/a.py" in rep["final"] and "Binary files" not in rep["final"]
