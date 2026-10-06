from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from .dataset import DATASET_ID, load_snapshot
from .registry import build_runtime
from .runner import Run, compare, trajectory_table
from .export import export_run
from .stats import markdown_report
from .selection import Subset, SubsetSpec, select


def _csv(v: str | None) -> list[str]:
    return [x for x in (v or "").split(",") if x]


def cmd_subset_create(a):
    ref, instances = load_snapshot(DATASET_ID, a.revision)
    spec = SubsetSpec(ids=_csv(a.ids), repos=_csv(a.repo), difficulties=_csv(a.difficulty),
                      exclude=_csv(a.exclude), n=a.n, seed=a.seed)
    ids = select(instances, spec)
    path = Subset(a.name, ref, spec, ids).save()
    print(f"{len(ids)} instances -> {path}  (dataset {ref.revision[:12]}, sha256 {ref.sha256[:12]})")


def cmd_subset_show(a):
    s = Subset.load(a.name)
    print(json.dumps({"name": s.name, "dataset": s.dataset.to_dict(), "n": len(s.instance_ids)}, indent=2))
    print("\n".join(s.instance_ids))


def cmd_run(a):
    subset = Subset.load(a.subset) if a.subset else None
    ref, all_instances = load_snapshot(DATASET_ID, subset.dataset.revision if subset else a.revision)
    if subset and ref.sha256 != subset.dataset.sha256:
        sys.exit("dataset snapshot differs from the one this subset was created against")
    ids = subset.instance_ids if subset else select(all_instances, SubsetSpec(ids=_csv(a.ids)))
    by_id = {i["instance_id"]: i for i in all_instances}
    instances = [by_id[i] for i in ids]
    runtime = build_runtime(a.runtime, json.loads(a.runtime_config), all_instances)
    name = a.name or f"{runtime.name}-{a.subset or 'adhoc'}-{datetime.now():%Y%m%d-%H%M%S}"
    run = Run(name)
    run.create(ref, instances, runtime, a.subset, resume=a.resume)
    if a.pipeline:
        run.run_pipeline(runtime, a.workers, a.timeout, rm_images=a.rm_images, min_free_gb=a.min_free_gb)
    else:
        run.generate(runtime, keep_workspaces=a.keep_workspaces)
        if not a.skip_eval:
            run.evaluate(a.workers, a.timeout, fresh=a.fresh, rm_images=a.rm_images)
    _print_summary(run)


def cmd_evaluate(a):
    run = Run(a.run)
    run.evaluate(a.workers, a.timeout, fresh=a.fresh, rm_images=a.rm_images)
    _print_summary(run)


def cmd_report(a):
    run = Run(a.run)
    run.collect()
    _print_summary(run)


def cmd_table(a):
    print(trajectory_table(a.run))


def cmd_stats(a):
    print(markdown_report(a.run))


def cmd_export(a):
    from pathlib import Path
    n = export_run(a.run, Path(a.dest))
    print(f"exported {n} files from run {a.run} to {a.dest} (local paths scrubbed)")


def cmd_compare(a):
    print(compare(a.runs))


def _print_summary(run: Run):
    print(f"\nrun: {run.dir}")
    print(json.dumps(json.loads(run.summary_path.read_text()), indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(prog="miracle", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("subset", help="create/show named instance subsets").add_subparsers(dest="sub", required=True)
    c = sp.add_parser("create")
    c.add_argument("name")
    c.add_argument("--n", type=int)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--ids", help="comma-separated instance ids")
    c.add_argument("--repo", help="comma-separated, e.g. django/django,sympy/sympy")
    c.add_argument("--difficulty", help="comma-separated, e.g. '<15 min fix'")
    c.add_argument("--exclude", help="comma-separated instance ids")
    c.add_argument("--revision", help="HF dataset revision (default: current main)")
    c.set_defaults(fn=cmd_subset_create)
    s = sp.add_parser("show")
    s.add_argument("name")
    s.set_defaults(fn=cmd_subset_show)

    r = sub.add_parser("run", help="generate patches with a runtime, then evaluate them")
    r.add_argument("--runtime", required=True, help="gold|empty|broken|noop|s0|toolloop|module:Class")
    r.add_argument("--runtime-config", default="{}", help="JSON kwargs for module:Class")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--subset")
    g.add_argument("--ids", help="comma-separated instance ids")
    r.add_argument("--revision", help="HF dataset revision for --ids runs")
    r.add_argument("--name")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--skip-eval", action="store_true")
    r.add_argument("--keep-workspaces", action="store_true")
    r.add_argument("--pipeline", action="store_true",
                   help="one instance at a time: agent -> official eval -> remove its image (bounded disk use)")
    r.add_argument("--min-free-gb", type=float, default=5.0, help="--pipeline: stop if free disk drops below this")
    for x in (r,):
        _eval_flags(x)
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("evaluate", help="(re)run the official harness on a run's predictions")
    e.add_argument("run")
    _eval_flags(e)
    e.set_defaults(fn=cmd_evaluate)

    rp = sub.add_parser("report", help="rebuild results.jsonl/summary.json from files on disk (no Docker)")
    rp.add_argument("run")
    rp.set_defaults(fn=cmd_report)

    tp = sub.add_parser("table", help="per-instance outcome next to trajectory statistics")
    tp.add_argument("run")
    tp.set_defaults(fn=cmd_table)

    sp2 = sub.add_parser("stats", help="Markdown statistics report for a run (outcomes, cost, time, trajectories)")
    sp2.add_argument("run")
    sp2.set_defaults(fn=cmd_stats)

    ex = sub.add_parser("export", help="copy a run's evidence into a committable directory (paths scrubbed)")
    ex.add_argument("run")
    ex.add_argument("dest")
    ex.set_defaults(fn=cmd_export)

    cp = sub.add_parser("compare", help="compare runs on their common instances")
    cp.add_argument("runs", nargs="+")
    cp.set_defaults(fn=cmd_compare)

    from .llm.cli import add_parser, guarded
    add_parser(sub)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)    # progress must be visible when redirected to a log
    guarded(a.fn, a)


def _eval_flags(p):
    p.add_argument("--workers", "-j", type=int, default=2, help="parallel evaluations (default 2)")
    p.add_argument("--timeout", type=int, default=1800, help="per-instance test timeout, seconds")
    p.add_argument("--fresh", action="store_true", help="discard cached harness results and re-evaluate")
    p.add_argument("--rm-images", action="store_true",
                   help="remove instance images this run pulled (they are large; harness never removes them)")


if __name__ == "__main__":
    main()
