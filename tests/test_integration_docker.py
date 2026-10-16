"""Full pipeline against the real official harness. `pytest -m docker` (pulls ~3GB, needs network)."""
import json

import pytest

from miracle.dataset import load_snapshot
from miracle.registry import build_runtime
from miracle.runner import Run
from miracle.selection import SubsetSpec, select

pytestmark = pytest.mark.docker
IID = "django__django-11099"


@pytest.mark.parametrize("rt,expected", [("gold", "resolved"), ("noop", "unresolved"),
                                        ("broken", "patch_apply_failed"), ("empty", "empty_patch")])
def test_pipeline(rt, expected):
    ref, all_ = load_snapshot()
    ids = select(all_, SubsetSpec(ids=[IID]))
    inst = [i for i in all_ if i["instance_id"] in ids]
    runtime = build_runtime(rt, {}, all_)
    run = Run(f"t-{rt}")
    run.create(ref, inst, runtime, None)
    run.generate(runtime)
    run.evaluate(workers=1)
    assert run.collect()["status_counts"] == {expected: 1}


def test_scripted_s0_applying_gold_patch_through_agent_tools_resolves():
    """Container env + toolbox + patch extraction + official harness, with no real LLM."""
    from miracle.agents.s0 import S0Runtime
    from test_llm import Scripted
    ref, all_ = load_snapshot()
    inst = [i for i in all_ if i["instance_id"] == IID]
    gold = inst[0]["patch"]
    turns = [(None, [("list_files", {"path": "."})]),
             (None, [("write_file", {"path": ".gold.patch", "content": gold})]),
             (None, [("run_command", {"command": "git apply .gold.patch && rm .gold.patch && git status --short"})]),
             (None, [("run_command", {"command": "python tests/runtests.py auth_tests.test_validators -v 0"})]),
             (None, [("git_diff", {})]),
             ("done", [])]
    rt = S0Runtime(client=Scripted(turns))
    run = Run("t-s0")
    run.create(ref, inst, rt, None)
    run.generate(rt)
    res = json.loads((run.agent_dir(IID) / "result.json").read_text())
    assert res["status"] == "ok" and res["metadata"]["modified_files"] == ["django/contrib/auth/validators.py"]
    assert res["metadata"]["tests"] and (run.agent_dir(IID) / "trace.json").exists()
    run.evaluate(workers=1, only=[IID])
    assert run.collect()["status_counts"] == {"resolved": 1}


def test_scripted_s2_terminates_early_on_complete_and_the_patch_still_resolves():
    """Real container + real test runner + official harness; the model is scripted (agent and checker)."""
    from miracle.agents.s2 import S2Runtime
    from test_llm import Scripted
    ref, all_ = load_snapshot()
    inst = [i for i in all_ if i["instance_id"] == IID]
    P = "django/contrib/auth/validators.py"
    fix = lambda cls: (None, [("edit_file", {"path": P,
        "old_str": f"class {cls}(validators.RegexValidator):\n    regex = r'^[\\w.@+-]+$'",
        "new_str": f"class {cls}(validators.RegexValidator):\n    regex = r'\\A[\\w.@+-]+\\Z'"})])
    turns = [(None, [("list_files", {"path": "."})]), fix("ASCIIUsernameValidator"), fix("UnicodeUsernameValidator"),
             (None, [("run_command", {"command": "python tests/runtests.py auth_tests.test_validators -v 0"})]),
             ("COMPLETE", []),
             ("never reached", [])]
    rt = S2Runtime(client=Scripted(turns))
    run = Run("t-s2")
    run.create(ref, inst, rt, None)
    run.generate(rt)
    res = json.loads((run.agent_dir(IID) / "result.json").read_text())
    c = res["metadata"]["controller"]
    assert res["status"] == "ok" and res["metadata"]["stats"]["stop_reason"] == "completion_check"
    assert c["checks"] == 1 and c["complete"] == 1 and c["termination_step"] == 4 and res["metadata"]["stats"]["steps"] == 4
    assert res["metadata"]["modified_files"] == [P]
    run.evaluate(workers=1, only=[IID])
    assert run.collect()["status_counts"] == {"resolved": 1}


def test_container_reset_and_dirty_state_on_a_real_instance_image():
    """The branch-isolation contract inside the real SWE-bench container (its own git, root user, conda env)."""
    from miracle import evaluator
    from miracle.diagnosis.config import DiagnosisConfig
    from miracle.diagnosis.graph import GraphState
    from miracle.diagnosis.ops import GraphOps
    from miracle.diagnosis.sandbox import InvestigationSandbox
    from miracle.docker_env import DockerEnvironment
    ref, all_ = load_snapshot()
    img = next(i for i in all_ if i["instance_id"] == IID)["image"]
    assert not evaluator.ensure_images([img])
    with DockerEnvironment(img, "miracle.t-reset") as env:
        base_diff = env.diff()
        assert env.dirty_state()["clean"] and base_diff == ""
        f = "/testbed/django/contrib/auth/validators.py"
        orig = env.read_bytes(f)
        env.write_bytes(f, orig + b"\n# TEMP INSTRUMENTATION\n")                                   # tracked file edited
        env.write_bytes("/testbed/probe_script.py", b"print('probe')\n")                           # new file
        env.exec("rm README.rst && printf '\\x00\\x01bin' > /testbed/blob.bin")                    # deletion + new binary
        env.exec("git add -A && git commit -qm agent-commit")                                      # HEAD moved
        d = env.dirty_state()
        assert not d["clean"] and not d["head_unchanged"] and {"django/contrib/auth/validators.py", "README.rst", "probe_script.py"} <= set(d["changed_files"] + d["untracked"])
        env.reset_to_baseline()
        d = env.dirty_state()
        assert d["clean"], d
        assert env.read_bytes(f) == orig and env.diff() == "" and env.exec("test -f README.rst && echo yes").output.strip().endswith("yes")
        # the sandbox lifecycle on top of it: interventions are opened by writes, tagged, and reset+verified at the end
        st = GraphState(IID)
        ops = GraphOps(st, None, "issue text", DiagnosisConfig())
        sb = InvestigationSandbox(env, ops)
        sb.establish_baseline()
        sb.begin_branch(ops.start_branch(f"{IID}:H1", [], {}))
        sb.open_if_needed("write probe")
        env.write_bytes("/testbed/another_probe.py", b"x = 1\n")
        rec = sb.end_branch(f"{IID}:B1")
        assert rec["was_dirty"] and rec["verified_clean"] and env.dirty_state()["clean"]
        assert all(r["ok"] for r in st.resets) and [r["context"] for r in st.resets][0] == "baseline"


def test_s3_through_the_real_runner_container_grading_report_and_ui():
    """Run.generate -> S3Runtime in a real instance container -> official grading -> task report -> UI check. The model is scripted."""
    from miracle.diagnosis.reports import task_report
    from miracle.diagnosis.orchestrate import ui_check
    from miracle.diagnosis.runtime import S3Runtime
    from test_diagnosis_engine import T
    from test_llm import Scripted
    ref, all_ = load_snapshot()
    inst = [i for i in all_ if i["instance_id"] == IID]
    V = "django/contrib/auth/validators.py"
    hyp = lambda claim, typ, pred, crit: ("create_hypothesis", dict(claim=claim, type=typ, predictions=[pred], falsification_criteria=[crit]))
    turns = [
        T(hyp("the username regex ends with $, which in Python also matches before a trailing newline, so 'abc\\n' is accepted", "root_cause",
              "re.match of the validator regex accepts a username with a trailing newline", "if a trailing newline is rejected by the regex this hypothesis is false"),
          hyp("the validator strips whitespace from the input before matching, which hides the trailing newline", "mechanism",
              "validating 'abc\\n' passes only because the input is stripped first", "if the regex alone accepts 'abc\\n' the stripping explanation is unnecessary")),
        T(("create_relation", dict(from_id="H1", to_id="H2", type="alternative_to", rationale="different reasons for the same accepted input")),
          ("create_probe", dict(target_hypothesis_ids=["H1", "H2"], purpose="discriminate", planned_action="match the bare regex against 'abc\\n', then call the validator",
                                expected_outcomes=[dict(hypothesis_id="H1", expect="the bare regex already matches 'abc\\n'"), dict(hypothesis_id="H2", expect="the bare regex rejects 'abc\\n'")]))),
        T(("done_planning", {})),
        # branch 1 (H1)
        T(("read_file", dict(path=V, start_line=1, end_line=25))),
        T(("record_evidence", dict(type="code_excerpt", summary="both validators use the regex ^[\\w.@+-]+$", path=V, start_line=8, end_line=10)),
          ("record_relation", dict(evidence_id="E2", hypothesis_id="H1", relation="supports", rationale="the regex is anchored with $, which tolerates a trailing newline", strength=3))),
        T(("run_command", dict(command="python -c \"import re; print(bool(re.match(r'^[\\w.@+-]+$', 'abc\\n')))\""))),
        T(("record_evidence", dict(type="runtime_observation", summary="the bare regex matches a username with a trailing newline", source_tool_call_id="T2", quote="True")),
          ("record_relation", dict(evidence_id="E3", hypothesis_id="H1", relation="supports", rationale="no stripping is involved and the regex still matches", strength=3)),
          ("record_relation", dict(evidence_id="E3", hypothesis_id="H2", relation="contradicts", rationale="the bare regex matches without any stripping step", strength=2))),
        T(("finish_investigation", dict(summary="The bare regex accepts 'abc\\n'; no stripping is needed to explain the behaviour.", outcome="supports"))),
        T(("update_hypothesis_status", dict(hypothesis_id="H1", new_status="supported", cause_evidence_ids=["E2", "E3"])),
          ("update_hypothesis_status", dict(hypothesis_id="H2", new_status="refuted", cause_evidence_ids=["E3"])),
          ("commit_diagnosis", dict(selected_hypothesis_ids=["H1"], causal_explanation="The username validators use a regex ending in $, which in Python also matches before a trailing newline, so usernames ending in a newline are accepted.",
                                    affected_components=["django.contrib.auth.validators.ASCIIUsernameValidator", "UnicodeUsernameValidator"], requirements_explained=["usernames with a trailing newline are accepted"],
                                    strongest_evidence_ids=["E2", "E3"], rejected_alternatives=[dict(hypothesis_id="H2", reason="the bare regex accepts the newline with no stripping (E3)")],
                                    unresolved_risks=[], implementation_direction="anchor the regex with \\A and \\Z instead of ^ and $"))),
        # implementation stage (S1 loop)
        T(("edit_file", dict(path=V, old_str="class ASCIIUsernameValidator(validators.RegexValidator):\n    regex = r'^[\\w.@+-]+$'",
                             new_str="class ASCIIUsernameValidator(validators.RegexValidator):\n    regex = r'\\A[\\w.@+-]+\\Z'"))),
        T(("edit_file", dict(path=V, old_str="class UnicodeUsernameValidator(validators.RegexValidator):\n    regex = r'^[\\w.@+-]+$'",
                             new_str="class UnicodeUsernameValidator(validators.RegexValidator):\n    regex = r'\\A[\\w.@+-]+\\Z'"))),
        T(("run_command", dict(command="python tests/runtests.py auth_tests.test_validators -v 0"))),
        ("Anchored both username regexes with \\A and \\Z.", []),
    ]
    rt = S3Runtime(client=Scripted(turns))
    run = Run("t-s3")
    run.create(ref, inst, rt, None)
    run.generate(rt)
    a = run.agent_dir(IID)
    res = json.loads((a / "result.json").read_text())
    assert res["status"] == "ok", res.get("error")
    for f in ("diagnosis/events.jsonl", "diagnosis/graph.json", "diagnosis/diagnosis.json", "diagnosis/reset_verification.json", "diagnosis/tool_calls.jsonl",
              "implementation/patch.diff", "implementation/result.json", "trace.json", "patch.diff"):
        assert (a / f).exists(), f
    st = json.loads((a / "diagnosis" / "graph.json").read_text())
    assert st["diagnosis"]["status"] == "committed" and all(r["ok"] for r in st["resets"]) and len(st["resets"]) >= 3
    assert "TEMP" not in (a / "patch.diff").read_text() and res["metadata"]["diagnosis"]["resets_verified"]
    run.evaluate(workers=1, only=[IID])
    s = run.collect()
    assert s["status_counts"] == {"resolved": 1}
    row = json.loads(run.results_path.read_text().splitlines()[0])
    rep = task_report(a, IID, row["eval"])
    assert rep["official"]["resolved"] is True and rep["diagnosis"]["status"] == "committed"
    ui = ui_check(run.dir.parent, run.name, IID)
    assert ui["ok"], ui
