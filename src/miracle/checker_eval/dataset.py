"""Build the labelled evaluation set for the S2 completion checker from recorded S0/S1/S2 runs.

Sample kinds (each sample = what the checker would receive online: issue, diff, relevant test output):
  final         the run's final patch (non-empty), with the tests that ran after its last source edit.
                Label = the official SWE-bench grade of that run. `online_eligible` says whether the online
                controller could actually have invoked the checker on it (candidate evidence + source change).
  intermediate  the diff at a step where candidate evidence appeared (source edit + successful test) and the
                diff differs from the run's final patch, reconstructed by replay (see replay.py).

Labels (conservative; anything not established reliably is UNKNOWN or PENDING-then-UNKNOWN):
  final         RESOLVED / UNRESOLVED from the official grade; UNKNOWN if the grade is an eval error/timeout.
  intermediate  RESOLVED / UNRESOLVED if its diff is byte-identical to a graded final patch of the same instance,
                otherwise graded with the official harness (grade.py); UNKNOWN if that grading fails; and an
                intermediate candidate from a run whose replay does not reproduce the final patch is not built at all.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..agents.s2 import build_check_prompt, has_source_modification
from ..agents.termination import CandidateTracker
from ..dataset import read_jsonl, write_jsonl
from ..runner import Run
from ..workspace import pristine
from .replay import abbrev_len, normalize_patch, replay_trace

LABELLED = ("RESOLVED", "UNRESOLVED")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _label_from_grade(row: dict) -> tuple[str, str]:
    st = row["eval"]["status"]
    if st == "resolved":
        return "RESOLVED", "official-run-grade"
    if st == "unresolved":
        return "UNRESOLVED", "official-run-grade"
    return "UNKNOWN", f"official-run-grade:{st}"


def candidates_in_trace(trace: dict) -> tuple[list[dict], list[dict]]:
    """-> (candidates at their steps, tests_since_last_edit at the end). Same tracker the live controller uses."""
    tr, cands = CandidateTracker(), []
    for s in trace["steps"]:
        for r in s["tool_results"]:
            tr.observe(s["index"], r)
        ev = tr.new_candidate(s["index"])
        if ev:
            cands.append({"step": s["index"], "trigger": ev, "tests": tr.tests_since_last_edit()})
    return cands, tr.tests_since_last_edit()


def _tests_view(tests: list[dict]) -> list[dict]:
    return [{"step": t["step"], "command": t["command"], "exit_code": t["exit_code"], "output": t["output"]} for t in tests]


def build(run_names: list[str], out: Path, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    runs = {n: Run(n) for n in run_names}
    inst: dict[str, dict] = {}
    rows: dict[tuple, dict] = {}
    patches: dict[tuple, str] = {}
    for n, run in runs.items():
        inst.update(run.instances())
        for r in read_jsonl(run.results_path):
            rows[(n, r["instance_id"])] = r
            patches[(n, r["instance_id"])] = normalize_patch((run.agent_dir(r["instance_id"]) / "patch.diff").read_text())
    abbrev = {}
    for (n, iid), p in patches.items():
        abbrev.setdefault(iid, abbrev_len(p))
        abbrev[iid] = abbrev[iid] or abbrev_len(p)

    samples, run_info = [], {}
    for (n, iid), row in rows.items():
        run, final = runs[n], patches[(n, iid)]
        trace = json.loads((run.agent_dir(iid) / "trace.json").read_text())
        cands, end_tests = candidates_in_trace(trace)
        online = [{"step": c["step"], "decision": c["decision"], "diff_sha256": c["diff_sha256"], "prompt": c["prompt"]}
                  for c in ((trace.get("controller") or {}).get("records") or [])]
        info = {"candidate_steps": [c["step"] for c in cands], "fidelity": None, "divergences": [], "final_empty": not final.strip()}
        recon = None
        if cands:
            try:
                rep = replay_trace(pristine(inst[iid]), trace, {c["step"] for c in cands}, abbrev.get(iid))
                recon = rep
                info["divergences"] = rep["divergences"]
                info["fidelity"] = (rep["final"] == final) and not rep["divergences"]
            except Exception as e:              # e.g. network error fetching the pristine checkout
                info["fidelity"], info["error"] = False, f"{type(e).__name__}: {e}"[:200]
        run_info[f"{n}:{iid}"] = info
        issue = inst[iid]["problem_statement"]

        def mk(kind, step, diff, tests, trigger):
            return {"sample_id": f"{n}:{iid}:{kind}" + (f"@{step}" if kind == "intermediate" else ""),
                    "run": n, "instance_id": iid, "kind": kind, "step": step, "diff": diff, "diff_sha256": _sha(diff),
                    "tests": _tests_view(tests), "trigger": trigger, "trigger_held": bool(tests),
                    "online_gate": has_source_modification(diff), "online_eligible": bool(tests) and has_source_modification(diff),
                    "prompt": build_check_prompt(issue, diff, tests), "run_status": row["eval"]["status"]}

        if final.strip():
            s = mk("final", None, final, end_tests, None)
            s["label"], s["label_source"] = _label_from_grade(row)
            s["online_checks"] = online
            samples.append(s)
        if info["fidelity"]:
            seen = {_sha(final)}
            for c in cands:
                d = recon["diffs"].get(c["step"], "")
                if not d or _sha(d) in seen or not has_source_modification(d):
                    continue                    # same code as an earlier candidate / the final patch / no source change
                seen.add(_sha(d))
                s = mk("intermediate", c["step"], d, c["tests"], c["trigger"])
                s["label"], s["label_source"] = "PENDING", None
                samples.append(s)
        log(f"{n}:{iid}: candidates {info['candidate_steps']} fidelity={info['fidelity']}")

    # cross-run label reuse: an intermediate diff identical to a graded final patch of the same instance
    graded = {}
    for s in samples:
        if s["kind"] == "final" and s["label"] in LABELLED:
            graded.setdefault((s["instance_id"], s["diff_sha256"]), (s["label"], s["sample_id"]))
    for s in samples:
        if s["kind"] == "intermediate" and (s["instance_id"], s["diff_sha256"]) in graded:
            lab, src = graded[(s["instance_id"], s["diff_sha256"])]
            s["label"], s["label_source"] = lab, f"equals-final-patch:{src}"
    write_jsonl(out / "samples.jsonl", samples)
    summary = summarize(samples, run_info)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def summarize(samples: list[dict], run_info: dict) -> dict:
    from collections import Counter
    c = Counter((s["kind"], s["label"]) for s in samples)
    return {"samples": len(samples), "by_kind_label": {f"{k}/{l}": v for (k, l), v in sorted(c.items())},
            "online_eligible": sum(s["online_eligible"] for s in samples),
            "runs_with_candidates": sum(bool(i["candidate_steps"]) for i in run_info.values()),
            "runs_fidelity_true": sum(i["fidelity"] is True for i in run_info.values()),
            "runs_fidelity_false": [k for k, i in run_info.items() if i["fidelity"] is False],
            "run_info": run_info}
