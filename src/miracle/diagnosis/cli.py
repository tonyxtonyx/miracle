"""`miracle diagnosis ...` and `miracle graph serve ...`"""
from __future__ import annotations

import json
from pathlib import Path

from .. import paths


def cmd_freeze(a):
    from .freeze import FROZEN_PATH, write_frozen
    m = write_frozen()
    print(f"wrote {FROZEN_PATH}\n" + json.dumps({k: v for k, v in m.items() if k != "config"}, indent=2))


def cmd_verify(a):
    from .freeze import verify_frozen
    ok, problems = verify_frozen()
    print("frozen and unchanged" if ok else "NOT OK:\n  " + "\n  ".join(problems))
    raise SystemExit(0 if ok else 1)


def _judge(a, directory: Path | None = None):
    from .judging.clients import make_judge
    if a.judge == "none":
        return None
    if a.judge == "mock":
        from .toy import mock_judge
        return mock_judge()
    return make_judge(a.judge, directory, getattr(a, "judge_model", None))


def cmd_run(a):
    from .orchestrate import load_task_list, run_task_list
    from .judging.clients import JudgeError, JudgePending
    ids = load_task_list(Path(a.tasks_file))
    print(f"{len(ids)} tasks, processed in this exact order: {', '.join(ids[:5])}{' ...' if len(ids) > 5 else ''}")
    try:
        out = run_task_list(a.name, ids, _judge(a, paths.runs_dir() / a.name / "judge_exchange"), resume=a.resume, require_frozen=not a.allow_unfrozen, workers=a.workers, timeout=a.timeout)
    except JudgePending as e:
        raise SystemExit(f"waiting for judge responses (file exchange): {e.path}\nWrite the JSON answer there and re-run with --resume.")
    except JudgeError as e:
        raise SystemExit(f"judge failure: {e}")
    print((paths.runs_dir() / a.name / "aggregate_report.md").read_text())


def cmd_judge(a):
    from .orchestrate import ui_check
    from ..dataset import load_snapshot
    from ..runner import Run
    from ..workspace import pristine
    from .judging.pipeline import judge_instance
    from .reports import task_report
    run = Run(a.run)
    _, instances = load_snapshot()
    by_id = {i["instance_id"]: i for i in instances}
    ids = [a.instance] if a.instance else run.manifest()["instance_ids"]
    rows = {r["instance_id"]: r for r in map(json.loads, run.results_path.read_text().splitlines())} if run.results_path.exists() else {}
    judge = _judge(a, run.dir / "judge_exchange")
    for iid in ids:
        official = (rows.get(iid) or {}).get("eval") or {}
        print(f"judging {iid}")
        judge_instance(run.agent_dir(iid), by_id[iid], official, judge, lambda inst=by_id[iid]: pristine(inst), force=a.force)
        task_report(run.agent_dir(iid), iid, official)


def cmd_report(a):
    from ..runner import Run
    from .reports import aggregate, render_aggregate_md, task_report
    run = Run(a.run)
    rows = {r["instance_id"]: r for r in map(json.loads, run.results_path.read_text().splitlines())} if run.results_path.exists() else {}
    ids = run.manifest()["instance_ids"]
    for iid in ids:
        if (run.agent_dir(iid) / "diagnosis" / "events.jsonl").exists():
            task_report(run.agent_dir(iid), iid, (rows.get(iid) or {}).get("eval"))
    agg = aggregate(run.dir, ids)
    (run.dir / "aggregate_report.json").write_text(json.dumps(agg, indent=2))
    (run.dir / "aggregate_report.md").write_text(render_aggregate_md(agg))
    print((run.dir / "aggregate_report.md").read_text())


def cmd_toy(a):
    from .toy import run_toy
    out = run_toy(a.name, judge=_judge(a, paths.runs_dir() / a.name / "judge_exchange"))
    r = out["report"]
    print(json.dumps({"official": out["official"], "diagnosis": r["diagnosis"]["status"], "graph": {k: r["graph"].get(k) for k in ("hypotheses", "evidence", "relations", "probes", "branches", "interventions", "rejected_operations")},
                      "ui_ok": out["ui"]["ok"]}, indent=2))
    print(f"artefacts: {paths.runs_dir() / a.name}\nview: miracle graph serve {a.name}")


def cmd_serve(a):
    from .web.server import serve
    serve(paths.runs_dir(), a.run, a.instance, a.port, a.open)


def add_parser(sub):
    d = sub.add_parser("diagnosis", help="S3: causal, falsification-oriented diagnosis").add_subparsers(dest="d_cmd", required=True)
    f = d.add_parser("freeze", help="record prompt/config hashes in FROZEN.json (do this BEFORE processing the task list)")
    f.set_defaults(fn=cmd_freeze)
    v = d.add_parser("verify-frozen", help="exit 1 if prompts/config changed since the freeze")
    v.set_defaults(fn=cmd_verify)

    def judge_flags(x):
        x.add_argument("--judge", default="auto", choices=["auto", "api", "cli", "files", "mock", "none"], help="judge backend (auto: API if ANTHROPIC_API_KEY is set, else the claude CLI)")
        x.add_argument("--judge-model")
    r = d.add_parser("run", help="process an ORDERED task list: diagnosis -> implementation -> official grading -> judges -> report -> UI check")
    r.add_argument("name")
    r.add_argument("--tasks-file", required=True, help="one SWE-bench Verified instance id per line, in processing order")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--allow-unfrozen", action="store_true", help="skip the FROZEN.json check (do not use for the real list)")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--timeout", type=int, default=1800)
    judge_flags(r)
    r.set_defaults(fn=cmd_run)
    j = d.add_parser("judge", help="(re)run the judges on an existing run")
    j.add_argument("run")
    j.add_argument("--instance")
    j.add_argument("--force", action="store_true")
    judge_flags(j)
    j.set_defaults(fn=cmd_judge)
    rp = d.add_parser("report", help="rebuild per-task reports and the aggregate report")
    rp.add_argument("run")
    rp.set_defaults(fn=cmd_report)
    t = d.add_parser("toy", help="run the full architecture on the toy task (real model, host environment)")
    t.add_argument("--name", default="diag-toy")
    judge_flags(t)
    t.set_defaults(judge="none")
    t.set_defaults(fn=cmd_toy)

    g = sub.add_parser("graph", help="read-only web view of recorded diagnosis graphs").add_subparsers(dest="g_cmd", required=True)
    s = g.add_parser("serve", help="serve the UI for recorded runs")
    s.add_argument("run", nargs="?")
    s.add_argument("--instance")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--open", action="store_true", help="open the browser")
    s.set_defaults(fn=cmd_serve)
