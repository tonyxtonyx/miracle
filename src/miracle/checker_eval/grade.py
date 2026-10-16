"""Ground truth for intermediate candidates: grade each distinct diff with the OFFICIAL SWE-bench harness.

One harness call per diff (the harness keys predictions by instance id), grouped by instance so each image is
pulled once and removed afterwards (unless it was already on the machine). Anything that is not a clean
resolved/unresolved verdict (apply failure, timeout, infra error) becomes UNKNOWN: excluded from precision/recall.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .. import evaluator
from ..dataset import read_jsonl, write_jsonl
from ..predictions import write_predictions
from ..runner import Run


def grade_pending(work: Path, run_names: list[str], timeout: int = 1800, log=print) -> dict:
    samples = read_jsonl(work / "samples.jsonl")
    inst = {}
    for n in run_names:
        inst.update(Run(n).instances())
    write_jsonl(work / "instances.jsonl", list(inst.values()))
    pending = [s for s in samples if s["label"] == "PENDING"]
    by_inst: dict[str, dict[str, list]] = {}
    for s in pending:
        by_inst.setdefault(s["instance_id"], {}).setdefault(s["diff_sha256"], []).append(s)
    gdir = work / "grading"
    results = {}
    reason = evaluator.docker_available()
    for iid, diffs in by_inst.items():
        image = inst[iid]["image"]
        was_there = evaluator.docker_image_id(image) is not None
        failed = {} if reason else evaluator.ensure_images([image], log)
        for n, (sha, group) in enumerate(diffs.items()):
            run_id = f"cke-{iid}-{n}"
            if reason or failed:
                verdict = ("UNKNOWN", f"grading unavailable: {reason or failed}")
            else:
                pred = {"instance_id": iid, "model_name_or_path": "ckpt", "model_patch": group[0]["diff"]}
                gdir.mkdir(parents=True, exist_ok=True)
                pp = gdir / f"{run_id}.predictions.jsonl"
                write_predictions(pp, [pred])
                log(f"grading {group[0]['sample_id']} (+{len(group) - 1} identical) with the official harness...")
                evaluator.run_harness(work / "instances.jsonl", pp, gdir, run_id, 1, timeout)
                ev = evaluator.collect(work, gdir, run_id, {iid: pred}, {iid: inst[iid]}, None, None)[iid]
                verdict = {"resolved": ("RESOLVED", "harness-grade"), "unresolved": ("UNRESOLVED", "harness-grade")}.get(
                    ev.status, ("UNKNOWN", f"harness-grade:{ev.status}"))
                verdict = (verdict[0], verdict[1], {"f2p": [ev.f2p_passed, ev.f2p_total], "p2p": [ev.p2p_passed, ev.p2p_total],
                                                    "status": ev.status, "log_dir": ev.log_dir})
            for s in group:
                s["label"], s["label_source"] = verdict[0], verdict[1]
                if len(verdict) > 2:
                    s["grade"] = verdict[2]
            results[group[0]["sample_id"]] = verdict[0]
        if not was_there and not failed:
            subprocess.run(["docker", "rmi", image], capture_output=True)
    write_jsonl(work / "samples.jsonl", samples)
    return results
