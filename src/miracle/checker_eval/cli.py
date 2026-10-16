"""`miracle checker-eval build|grade|run|report` -- offline evaluation of the S2 completion checker."""
from __future__ import annotations

import json
from pathlib import Path

from .. import paths

DEFAULT_RUNS = ["s0-dev10", "s1-dev10", "s2-dev10"]


def work_dir(name: str) -> Path:
    return paths.runs_dir() / name


def cmd_build(a):
    from .dataset import build
    s = build(a.runs, work_dir(a.name))
    print(json.dumps({k: v for k, v in s.items() if k != "run_info"}, indent=2))


def cmd_grade(a):
    from .grade import grade_pending
    print(json.dumps(grade_pending(work_dir(a.name), a.runs, a.timeout), indent=2))


def cmd_run(a):
    from .evaluate import run_checks
    n = run_checks(work_dir(a.name), a.runs, a.repeats, a.workers)
    print(f"{n} checker calls made")


def cmd_report(a):
    from .evaluate import report
    r = report(work_dir(a.name))
    print((work_dir(a.name) / "report.md").read_text())
    print("\n" + json.dumps({k: r[k] for k in ("stability", "cost", "decision_counts", "label_counts")}, indent=2))


def add_parser(sub):
    p = sub.add_parser("checker-eval", help="offline evaluation of the completion checker").add_subparsers(dest="ce_cmd", required=True)

    def common(x):
        x.add_argument("--name", default="checker-eval", help="working directory name under runs/")
    b = p.add_parser("build", help="build the labelled sample set from recorded runs (replays trajectories offline)")
    b.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    common(b)
    b.set_defaults(fn=cmd_build)
    g = p.add_parser("grade", help="label PENDING intermediate candidates with the official harness")
    g.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    g.add_argument("--timeout", type=int, default=1800)
    common(g)
    g.set_defaults(fn=cmd_grade)
    r = p.add_parser("run", help="run the existing checker on every sample (repeated)")
    r.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--workers", type=int, default=4)
    common(r)
    r.set_defaults(fn=cmd_run)
    rp = p.add_parser("report", help="confusion matrix, precision/recall, stability, cost")
    common(rp)
    rp.set_defaults(fn=cmd_report)
