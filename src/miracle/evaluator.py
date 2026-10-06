"""Thin wrapper around the official SWE-bench harness.

We do not reimplement any grading. We (1) run `swebench.harness.run_evaluation` as a
subprocess with a per-run working directory (the harness writes `logs/` relative to cwd,
so this keeps every experiment's logs self-contained), and (2) read back the artefacts it
wrote -- report.json, test_output.txt, run_instance.log, the run summary -- into a status.

Status taxonomy (harness only distinguishes resolved / unresolved / error):
  resolved           harness report.json says resolved
  unresolved         patch applied, tests ran, FAIL_TO_PASS/PASS_TO_PASS not all satisfied
  empty_patch        runtime returned no patch (harness skips these, so no logs exist)
  patch_apply_failed every `git apply`/`patch` strategy failed
  timeout            test run exceeded the per-instance timeout
  eval_error         harness produced no report for another reason (see run_instance.log)
  not_evaluated      no evaluation artefacts at all (harness crashed or never reached it)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from swebench.harness.constants import APPLY_PATCH_FAIL

STATUSES = ("resolved", "unresolved", "empty_patch", "patch_apply_failed", "timeout",
            "eval_error", "not_evaluated")


@dataclass
class InstanceEval:
    instance_id: str
    status: str
    resolved: bool
    test_runtime_s: float | None = None
    f2p_passed: int | None = None
    f2p_total: int | None = None
    p2p_passed: int | None = None
    p2p_total: int | None = None
    infra_failure: str | None = None       # harness' advisory classification of environment breakage
    infra_tier: str | None = None
    image: str | None = None
    image_id: str | None = None            # docker image id actually used (reproducibility)
    log_dir: str | None = None             # relative to the run directory

    def to_dict(self) -> dict:
        return asdict(self)


def docker_available() -> str | None:
    """Return None if Docker works, else a human-readable reason."""
    try:
        r = subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"docker not usable: {e}"
    return None if r.returncode == 0 else f"docker daemon not reachable: {r.stderr.strip()[:200]}"


def docker_image_id(image: str) -> str | None:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                       capture_output=True, text=True)
    return r.stdout.strip() or None if r.returncode == 0 else None


def platform_of(image: str) -> str | None:
    # published names encode the arch: sweb.eval.<arch>.<repo>_1776_<name>
    for arch, plat in (("x86_64", "linux/amd64"), ("arm64", "linux/arm64")):
        if f".{arch}." in image:
            return plat
    return None


def ensure_images(images: list[str], log=print) -> dict[str, str]:
    """Pull any missing instance images with an explicit platform. Returns {image: reason} for failures.

    The harness pulls through docker-py with the daemon's default platform, which on an
    arm64 host (Apple Silicon) fails with "no matching manifest" because the official
    images are published for x86_64 only. Pre-pulling with --platform makes them run under
    emulation. Slower than native, but it is the official image and grading is identical.
    """
    failed: dict[str, str] = {}
    for img in images:
        if docker_image_id(img):
            continue
        cmd = ["docker", "pull", "-q"] + (["--platform", platform_of(img)] if platform_of(img) else []) + [img]
        log(f"pulling {img} ...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            failed[img] = (r.stderr.strip().splitlines() or ["unknown error"])[-1][:300]
            log(f"  pull failed: {failed[img]}")
    return failed


def run_harness(instances_path: Path, predictions_path: Path, eval_dir: Path, eval_run_id: str,
                workers: int, timeout: int) -> dict:
    """Run the official harness. Returns {returncode, wall_s, command, summary_path}."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
           "--dataset_name", str(instances_path.resolve()),   # frozen local copy, not the live HF dataset
           "--predictions_path", str(predictions_path.resolve()),
           "--run_id", eval_run_id, "--max_workers", str(workers),
           "--timeout", str(timeout), "--report_dir", "."]
    t0 = time.monotonic()
    with open(eval_dir / "harness_stdout.log", "w") as out:
        rc = subprocess.run(cmd, cwd=eval_dir, stdout=out, stderr=subprocess.STDOUT).returncode
    summaries = sorted(eval_dir.glob(f"*.{eval_run_id}.json"))
    return {"returncode": rc, "wall_s": round(time.monotonic() - t0, 2), "command": cmd,
            "summary_path": str(summaries[0].relative_to(eval_dir)) if summaries else None}


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def collect(run_dir: Path, eval_dir: Path, eval_run_id: str, predictions: dict[str, dict],
            instances: dict[str, dict], summary_paths: list[str] | str | None,
            image_ids: dict[str, str] | None = None) -> dict[str, InstanceEval]:
    """Turn harness artefacts into one InstanceEval per predicted instance.
    `summary_paths`: every harness summary of this run (one per evaluate call), merged."""
    if isinstance(summary_paths, str):
        summary_paths = [summary_paths]
    reasons, envfail, ambiguous = {}, set(), set()
    for sp in summary_paths or []:
        summary = _load(eval_dir / sp) or {}
        reasons.update(summary.get("failure_reasons", {}))
        envfail |= set(summary.get("infra_failure_ids", []))
        ambiguous |= set(summary.get("ambiguous_failure_ids", []))
    out: dict[str, InstanceEval] = {}
    for iid, pred in predictions.items():
        image = instances[iid].get("image")
        if not (pred.get("model_patch") or "").strip():
            out[iid] = InstanceEval(iid, "empty_patch", False)
            continue
        ldir = (eval_dir / "logs" / "run_evaluation" / eval_run_id
                / pred["model_name_or_path"].replace("/", "__") / iid)
        ev = InstanceEval(iid, "not_evaluated", False, image=image,
                          image_id=(image_ids or {}).get(iid) or (docker_image_id(image) if image else None))
        if ldir.is_dir():
            ev.log_dir = str(ldir.relative_to(run_dir))
            log = (ldir / "run_instance.log").read_text(errors="replace") if (ldir / "run_instance.log").exists() else ""
            m = re.search(r"Test runtime: ([\d_.]+) seconds", log)
            ev.test_runtime_s = float(m.group(1).replace("_", "")) if m else None
            report = _load(ldir / "report.json")
            if report and iid in report:
                r = report[iid]
                ev.resolved = bool(r.get("resolved"))
                ev.status = "resolved" if ev.resolved else "unresolved"
                ts = r.get("tests_status") or {}
                for key, tag in (("FAIL_TO_PASS", "f2p"), ("PASS_TO_PASS", "p2p")):
                    d = ts.get(key) or {}
                    ok, bad = len(d.get("success", [])), len(d.get("failure", []))
                    setattr(ev, f"{tag}_passed", ok)
                    setattr(ev, f"{tag}_total", ok + bad)
            elif APPLY_PATCH_FAIL in log:
                ev.status = "patch_apply_failed"
            elif "Timeout error" in ((ldir / "test_output.txt").read_text(errors="replace")
                                      if (ldir / "test_output.txt").exists() else ""):
                ev.status = "timeout"
            else:
                ev.status = "eval_error"
        if iid in reasons:
            ev.infra_failure = reasons[iid]
            ev.infra_tier = "environment" if iid in envfail else "ambiguous" if iid in ambiguous else None
        out[iid] = ev
    return out
