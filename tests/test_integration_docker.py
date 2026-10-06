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
