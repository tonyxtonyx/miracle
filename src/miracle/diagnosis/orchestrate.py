"""Per-task workflow over an ORDERED task list:
   1 create artefacts  2 diagnosis (controller + investigators)  3 persist graph/log  4 commit  5 reset+verify  6 implementation
   7 official grading  8 blind judge  9 independent reference  10 oracle comparison  11 task report  12 verify the web UI can load and replay.
Tasks are processed strictly in the given order; none is skipped or replaced. An infrastructure failure aborts the run (existing policy: the instance
leaves no result and `--resume` retries it); a completed run lists every task with its status."""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from .. import paths
from ..dataset import DatasetRef, load_snapshot
from ..errors import InfraError
from ..runner import Run
from .freeze import verify_frozen
from .judging.clients import JudgeClient, JudgeError, JudgePending
from .judging.pipeline import judge_instance, toy_repo_dir
from .reports import aggregate, render_aggregate_md, task_report
from .runtime import S3Runtime


def load_task_list(path: Path) -> list[str]:
    """One instance id per line ('#' comments and blanks ignored). Order is preserved; duplicates are an error (no silent dedupe)."""
    ids = [l.split("#")[0].strip() for l in Path(path).read_text().splitlines()]
    ids = [i for i in ids if i]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate instance ids in the task list: {dup}")
    if not ids:
        raise ValueError("the task list is empty")
    return ids


def ui_check(runs_dir: Path, run: str, instance: str) -> dict:
    """Step 12: can the web UI load this task and replay it? Exercises the real HTTP server and the replay endpoint at start, middle and end."""
    from .web.server import start_in_thread
    srv, base = start_in_thread(runs_dir, default_run=run, default_instance=instance)
    q = f"run={run}&instance={instance}"
    try:
        get = lambda p: json.loads(urllib.request.urlopen(base + p, timeout=20).read())
        events = get(f"/api/events?{q}")
        n = len(events)
        steps = sorted({0, 1, n // 2, n})
        graphs = {s: get(f"/api/graph?{q}&upto={s}")["graph"] for s in steps}
        mono = all(len(graphs[a]["hypotheses"]) <= len(graphs[b]["hypotheses"]) for a, b in zip(steps, steps[1:]))
        art = get(f"/api/artifacts?{q}")
        page = urllib.request.urlopen(base + "/", timeout=20).read()
        res = {"ok": bool(n and mono and b"Diagnosis graph" in page), "events": n, "steps_checked": steps, "graph_grows_monotonically": mono,
               "diagnosis_loaded": art["diagnosis"] is not None, "judges_loaded": {k: bool(v) for k, v in art["judge"].items()}}
    except Exception as e:                                               # noqa: BLE001 - a failed check is data, not a crash
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        srv.shutdown()
    p = Path(runs_dir) / run / "agent" / instance / "ui_check.json"
    p.write_text(json.dumps(res, indent=2))
    return res


def process_task(run: Run, runtime: S3Runtime, iid: str, inst: dict, judge: JudgeClient | None, repo_dir_fn, log=print, workers: int = 1, timeout: int = 1800) -> dict:
    status = {"instance_id": iid}
    run.generate(runtime, ids=[iid], log=log)                       # steps 1-6 (diagnosis, reset, handoff, implementation)
    run.evaluate(workers=workers, timeout=timeout, rm_images=True, only=[iid], log=log)     # step 7: official grading
    row = next((r for r in map(json.loads, run.results_path.read_text().splitlines()) if r["instance_id"] == iid), {})
    official = row.get("eval") or {}
    status["official"] = {k: official.get(k) for k in ("status", "resolved")}
    inst_dir = run.agent_dir(iid)
    if not (inst_dir / "diagnosis" / "events.jsonl").exists():
        status["error"] = "no diagnosis artefacts (agent error?)"
        return status
    if judge is not None:
        judge_instance(inst_dir, inst, official, judge, repo_dir_fn, log=log)       # steps 8-10
        status["judged"] = True
    task_report(inst_dir, iid, official)                                              # step 11
    status["ui"] = ui_check(paths.runs_dir(), run.name, iid)                          # step 12
    return status



def run_task_list(name: str, task_ids: list[str], judge: JudgeClient | None, *, resume: bool = False, require_frozen: bool = True,
                  workers: int = 1, timeout: int = 1800, log=print) -> dict:
    if require_frozen:
        ok, problems = verify_frozen()
        if not ok:
            raise RuntimeError("prompts/config are not frozen or differ from FROZEN.json: " + "; ".join(problems))
    ref, instances = load_snapshot()
    by_id = {i["instance_id"]: i for i in instances}
    missing = [i for i in task_ids if i not in by_id]
    if missing:
        raise KeyError(f"instance ids not in the dataset: {missing}")
    ordered = [by_id[i] for i in task_ids]                                            # the list's order, verbatim
    rt = S3Runtime()
    run = Run(name)
    run.create(ref, ordered, rt, "diagnosis-task-list", resume=resume)
    statuses = []
    from ..workspace import pristine
    for n, iid in enumerate(task_ids, 1):
        log(f"=== [{n}/{len(task_ids)}] {iid}")
        try:
            statuses.append(process_task(run, rt, iid, by_id[iid], judge, lambda inst=by_id[iid]: pristine(inst), log, workers, timeout))
        except (InfraError, JudgePending, JudgeError) as e:
            statuses.append({"instance_id": iid, "stopped": f"{type(e).__name__}: {e}"})
            (run.dir / "run_status.json").write_text(json.dumps(statuses, indent=2))
            log(f"STOPPED at {iid}: {e}\n(no task was skipped; fix the cause and re-run with --resume)")
            raise
        (run.dir / "run_status.json").write_text(json.dumps(statuses, indent=2))
    agg = aggregate(run.dir, task_ids)
    (run.dir / "aggregate_report.json").write_text(json.dumps(agg, indent=2))
    (run.dir / "aggregate_report.md").write_text(render_aggregate_md(agg))
    return {"statuses": statuses, "aggregate": agg}
