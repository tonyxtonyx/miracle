"""Run the full architecture on the toy task (real model, host environment). Used to check plumbing and to tune prompts WITHOUT touching SWE-bench tasks."""
from __future__ import annotations

import json
from pathlib import Path

from .. import paths
from ..host_env import HostEnvironment
from ..runtime import Task
from .judging.clients import JudgeClient, MockJudge
from .judging.mockjudge import deterministic_mock
from .judging.pipeline import judge_instance, toy_repo_dir
from .orchestrate import ui_check
from .reports import aggregate, render_aggregate_md, task_report
from .runtime import S3Runtime
from .toytask import INSTANCE_ID, TEST_COMMAND, instance_record, write_repo
import tempfile


def run_toy(name: str, runtime: S3Runtime | None = None, judge: JudgeClient | None = None, log=print) -> dict:
    rt = runtime or S3Runtime()
    run_dir = paths.runs_dir() / name
    inst_dir = run_dir / "agent" / INSTANCE_ID
    env = HostEnvironment.from_directory(write_repo(Path(tempfile.mkdtemp(prefix="toysrc."))))
    try:
        res = rt.solve(Task.from_instance(instance_record(), env=env, artifact_dir=inst_dir))
        t = env.exec(TEST_COMMAND.replace("| tail -n 5", ""))               # toy stand-in for the official harness: apply nothing more, run the tests on the final tree
        official = {"status": "resolved" if t.exit_code == 0 else "unresolved", "resolved": t.exit_code == 0, "f2p_passed": int(t.exit_code == 0), "f2p_total": 1,
                    "p2p_passed": None, "p2p_total": None, "note": "toy task: pass/fail of the toy test-suite on the final tree (not the SWE-bench harness)"}
        (run_dir / "manifest.json").write_text(json.dumps({"name": name, "toy": True, "runtime": rt.describe(), "instance_ids": [INSTANCE_ID]}, indent=2, default=str))
        (run_dir / "results.jsonl").write_text(json.dumps({"run": name, "instance_id": INSTANCE_ID, "eval": official}) + "\n")
        (inst_dir / "patch.diff").write_text(res.patch)
        (inst_dir / "result.json").write_text(json.dumps({"usage": res.usage.to_dict(), "metadata": {k: v for k, v in res.metadata.items() if k != "trace"}}, indent=2, default=str))
        if judge is not None:
            judge_instance(inst_dir, instance_record(), official, judge, toy_repo_dir, log=log)
        rep = task_report(inst_dir, INSTANCE_ID, official)
        agg = aggregate(run_dir, [INSTANCE_ID])
        (run_dir / "aggregate_report.json").write_text(json.dumps(agg, indent=2))
        (run_dir / "aggregate_report.md").write_text(render_aggregate_md(agg))
        ui = ui_check(paths.runs_dir(), name, INSTANCE_ID)
        return {"report": rep, "ui": ui, "official": official}
    finally:
        env.close()


def mock_judge() -> MockJudge:
    return MockJudge(deterministic_mock)
