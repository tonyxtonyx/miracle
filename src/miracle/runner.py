"""Experiment orchestration: SWE-bench instance -> workspace -> runtime -> patch -> evaluator -> result.

Run directory (everything needed to understand / re-grade the run lives here):

  manifest.json        what was run: dataset revision+sha, instance ids, runtime config, versions
  instances.jsonl      frozen official records of the selected instances (fed to the harness)
  agent/<iid>/         result.json (OUR metadata: usage, latency, strategy...) + patch.diff
  predictions.jsonl    official SWE-bench format only
  eval/                cwd of the official harness: logs/run_evaluation/..., summary json, stdout
  results.jsonl        one joined row per instance (agent metadata + evaluation outcome)
  summary.json         aggregate metrics

Agent metadata and harness output are never mixed in the same file until results.jsonl,
which is derived and can be rebuilt at any time with `collect`.
"""
from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from . import __version__, evaluator, paths
from .dataset import DatasetRef, read_jsonl, sha256_file, write_jsonl
from .predictions import write_predictions
from .docker_env import DockerEnvironment, DockerError
from .runtime import AgentRuntime, Task
from .workspace import REPO_URL, create_workspace


class InfraError(RuntimeError):
    """The environment broke (image pull, container start, disk). Never recorded as an agent result:
    scoring infrastructure failures as agent failures would corrupt the baseline."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _read_json(p: Path):
    return json.loads(p.read_text())


def _env_info() -> dict:
    d = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"],
                       capture_output=True, text=True)
    return {"miracle": __version__, "swebench": version("swebench"),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "machine": platform.machine(),
            "docker_server": d.stdout.strip() or None}


class Run:
    def __init__(self, name: str):
        self.name = name
        self.dir = paths.runs_dir() / name

    # -- paths ---------------------------------------------------------------
    manifest_path = property(lambda s: s.dir / "manifest.json")
    instances_path = property(lambda s: s.dir / "instances.jsonl")
    predictions_path = property(lambda s: s.dir / "predictions.jsonl")
    eval_dir = property(lambda s: s.dir / "eval")
    results_path = property(lambda s: s.dir / "results.jsonl")
    summary_path = property(lambda s: s.dir / "summary.json")

    def agent_dir(self, iid: str) -> Path:
        return self.dir / "agent" / iid

    def instances(self) -> dict[str, dict]:
        return {i["instance_id"]: i for i in read_jsonl(self.instances_path)}

    def manifest(self) -> dict:
        return _read_json(self.manifest_path)

    # -- creation ------------------------------------------------------------
    def create(self, dataset: DatasetRef, instances: list[dict], runtime: AgentRuntime,
               subset: str | None, resume: bool = False) -> None:
        ids = [i["instance_id"] for i in instances]
        if self.dir.exists():
            if not resume:
                raise FileExistsError(f"{self.dir} exists (use --resume to continue it)")
            m = self.manifest()
            if (m["instance_ids"] != ids or m["dataset"] != dataset.to_dict()
                    or m["runtime"] != runtime.describe()):
                raise ValueError("--resume: instances, dataset or runtime config differ from this run's manifest")
            return
        self.dir.mkdir(parents=True)
        write_jsonl(self.instances_path, instances)
        _write_json(self.manifest_path, {
            "name": self.name, "created_at": _now(), "subset": subset,
            "dataset": dataset.to_dict(), "instances_sha256": sha256_file(self.instances_path),
            "instance_ids": ids, "runtime": runtime.describe(), "env": _env_info(),
            "evaluation": [],   # appended by every `evaluate` call
            # images already on this machine before the run: --rm-images never removes these
            "images_preexisting": [i["instance_id"] for i in instances
                                   if i.get("image") and evaluator.docker_image_id(i["image"])],
            "image_ids": {},    # instance_id -> docker image id actually used (reproducibility)
        })

    # -- stage 1: agent ------------------------------------------------------
    def _record_image_id(self, iid: str, image: str) -> None:
        m = self.manifest()
        img_id = evaluator.docker_image_id(image)
        if img_id and m["image_ids"].get(iid) != img_id:
            m["image_ids"][iid] = img_id
            _write_json(self.manifest_path, m)

    def _solve(self, runtime: AgentRuntime, inst: dict, ws_path: Path, repo_url_template: str, log):
        """Provision what the runtime asked for, run it. Returns (AgentResult, setup_seconds)."""
        t0 = time.monotonic()
        if runtime.environment == "container":
            failed = evaluator.ensure_images([inst["image"]], log)
            if failed:
                raise InfraError(f"could not pull {inst['image']}: {failed[inst['image']]}")
            self._record_image_id(inst["instance_id"], inst["image"])
            try:
                env = DockerEnvironment(inst["image"], f"miracle.{self.name}.{inst['instance_id']}")
                env.start()
            except DockerError as e:
                env.close()
                raise InfraError(f"container for {inst['instance_id']} failed to start: {e}") from e
            try:
                setup = time.monotonic() - t0
                return runtime.solve(Task.from_instance(inst, env=env)), setup
            finally:
                env.close()
        ws = create_workspace(inst, ws_path, repo_url_template)
        return runtime.solve(Task.from_instance(inst, ws)), time.monotonic() - t0

    def generate(self, runtime: AgentRuntime, ids: list[str] | None = None, keep_workspaces: bool = False,
                 repo_url_template: str = REPO_URL, log=print) -> None:
        instances = {i: v for i, v in self.instances().items() if not ids or i in ids}
        for n, (iid, inst) in enumerate(instances.items(), 1):
            out = self.agent_dir(iid)
            if (out / "result.json").exists():
                log(f"[{n}/{len(instances)}] {iid}: already generated, skipping")
                continue
            out.mkdir(parents=True, exist_ok=True)
            ws_path = self.dir / "workspaces" / iid
            rec = {"instance_id": iid, "started_at": _now()}
            patch = ""
            t1 = time.monotonic()
            try:
                res, setup_s = self._solve(runtime, inst, ws_path, repo_url_template, log)
                rec["setup_s"] = round(setup_s, 3)
                patch = res.patch or ""
                meta = dict(res.metadata)
                trace = meta.pop("trace", None)    # large; kept in its own file
                if trace is not None:
                    _write_json(out / "trace.json", trace)
                    meta["trace_path"] = f"agent/{iid}/trace.json"
                rec.update(status="ok", usage=res.usage.to_dict(), metadata=meta)
            except InfraError:
                shutil.rmtree(out, ignore_errors=True)   # leave no result: --resume will retry this instance
                raise
            except Exception:
                rec.update(status="agent_error", usage={}, metadata={}, error=traceback.format_exc())
            # wall time of setup + agent, measured by us (not self-reported)
            rec["latency_s"] = round(time.monotonic() - t1 - rec.get("setup_s", 0), 3)
            rec["patch_bytes"] = len(patch.encode())
            (out / "patch.diff").write_text(patch)
            _write_json(out / "result.json", rec)
            if not keep_workspaces:
                shutil.rmtree(ws_path, ignore_errors=True)
            log(f"[{n}/{len(instances)}] {iid}: {rec['status']}, patch {rec['patch_bytes']}B, {rec['latency_s']}s")
        if not keep_workspaces:
            shutil.rmtree(self.dir / "workspaces", ignore_errors=True)

    def _predictions(self, runtime_name: str) -> list[dict]:
        rows = []
        for iid in self.manifest()["instance_ids"]:
            p = self.agent_dir(iid) / "patch.diff"
            if p.exists():   # instances never generated simply have no prediction
                rows.append({"instance_id": iid, "model_name_or_path": runtime_name,
                             "model_patch": p.read_text()})
        return rows

    # -- stage 2: official evaluation ---------------------------------------
    def _has_report(self, runtime_name: str, iid: str) -> bool:
        return (self.eval_dir / "logs" / "run_evaluation" / self.name / runtime_name.replace("/", "__")
                / iid / "report.json").exists()

    def evaluate(self, workers: int = 4, timeout: int = 1800, fresh: bool = False, rm_images: bool = False,
                 only: list[str] | None = None, log=print) -> None:
        """Run the official harness on this run's patches (or just `only`). Already-graded instances
        are skipped (the harness caches by run_id); `fresh` discards that cache."""
        reason = evaluator.docker_available()
        if reason:
            raise RuntimeError(reason)
        m = self.manifest()
        name = m["runtime"]["name"]
        rows = self._predictions(name)
        write_predictions(self.predictions_path, rows)      # the full official artefact
        if fresh and self.eval_dir.exists():
            shutil.rmtree(self.eval_dir)
        instances = self.instances()
        want = [r for r in rows if (not only or r["instance_id"] in only) and r["model_patch"].strip()]
        todo = [r for r in want if not self._has_report(name, r["instance_id"])]
        k = len(m["evaluation"])
        info = {"started_at": _now(), "workers": workers, "timeout_s": timeout, "fresh": fresh,
                "instances": [r["instance_id"] for r in (rows if not only else [r for r in rows if r["instance_id"] in only])],
                "n_to_run": len(todo)}
        if todo:
            images = sorted({instances[r["instance_id"]]["image"] for r in todo})
            info["image_pull_failures"] = evaluator.ensure_images(images, log)
            for r in todo:
                self._record_image_id(r["instance_id"], instances[r["instance_id"]]["image"])
            log(f"running official harness on {len(todo)} patch(es)...")
            sub = self.eval_dir / "predictions.jsonl"       # only what this call grades
            self.eval_dir.mkdir(parents=True, exist_ok=True)
            write_predictions(sub, todo)
            h = evaluator.run_harness(self.instances_path, sub, self.eval_dir, self.name, workers, timeout)
            if h["summary_path"]:                           # the harness overwrites its summary each call
                kept = f"summary-{k}.json"
                (self.eval_dir / h["summary_path"]).rename(self.eval_dir / kept)
                h["summary_path"] = kept
            info.update(h)
            log(f"harness exited {h['returncode']} in {h['wall_s']}s (log: {self.eval_dir}/harness_stdout.log)")
        else:
            info["note"] = "nothing to run (empty patches or already graded)"
        m = self.manifest()
        m["evaluation"].append(info)
        _write_json(self.manifest_path, m)
        self.collect(log=log)
        if rm_images:
            for iid in (only or list(instances)):
                if iid not in m["images_preexisting"] and instances[iid].get("image"):
                    subprocess.run(["docker", "rmi", instances[iid]["image"]], capture_output=True)

    def run_pipeline(self, runtime: AgentRuntime, workers: int = 1, timeout: int = 1800,
                     rm_images: bool = True, min_free_gb: float = 5.0, log=print) -> None:
        """One instance at a time: agent -> official evaluation -> free its image. Bounded disk use."""
        for iid in self.manifest()["instance_ids"]:
            free = shutil.disk_usage(paths.root()).free / 1e9
            if free < min_free_gb:
                raise RuntimeError(f"only {free:.1f} GB free (< {min_free_gb}); stopping before {iid}. "
                                   "Free disk space and rerun with --resume.")
            log(f"=== {iid}  (free disk {free:.1f} GB)")
            self.generate(runtime, ids=[iid], log=log)
            self.evaluate(workers, timeout, rm_images=rm_images, only=[iid], log=log)

    # -- stage 3: join & summarise (pure function of files on disk) ----------
    def collect(self, log=print) -> dict:
        m = self.manifest()
        instances = self.instances()
        preds = {r["instance_id"]: r for r in self._predictions(m["runtime"]["name"])}
        summaries = [e["summary_path"] for e in m["evaluation"] if e.get("summary_path")]
        evals = evaluator.collect(self.dir, self.eval_dir, self.name, preds, instances, summaries,
                                  m.get("image_ids")) if preds else {}
        rows = []
        for iid in m["instance_ids"]:
            inst = instances[iid]
            ap = self.agent_dir(iid) / "result.json"
            agent = _read_json(ap) if ap.exists() else None
            ev = evals.get(iid)
            rows.append({"run": self.name, "instance_id": iid, "repo": inst["repo"],
                         "difficulty": inst.get("difficulty"), "runtime": m["runtime"],
                         "agent": agent,
                         "eval": ev.to_dict() if ev else {"status": "not_evaluated", "resolved": False}})
        write_jsonl(self.results_path, rows)
        s = summarize(rows, m)
        _write_json(self.summary_path, s)
        return s


def summarize(rows: list[dict], manifest: dict | None = None) -> dict:
    total = len(rows)
    status = Counter(r["eval"]["status"] for r in rows)
    resolved = status.get("resolved", 0)
    agents = [r["agent"] for r in rows if r["agent"]]
    usage = lambda k: sum((a.get("usage") or {}).get(k) or 0 for a in agents)
    costs = [(a.get("usage") or {}).get("cost_usd") for a in agents]
    cost = sum(c for c in costs if c is not None) if any(c is not None for c in costs) else None
    lat = [a["latency_s"] for a in agents]
    return {
        "run": rows[0]["run"] if rows else None,
        "runtime": rows[0]["runtime"] if rows else None,
        "total": total, "resolved": resolved,
        "resolve_rate": round(resolved / total, 4) if total else None,   # denominator = all selected
        "status_counts": dict(sorted(status.items())),
        "agent_errors": sum(a["status"] != "ok" for a in agents),
        "infra_failures": sum(1 for r in rows if r["eval"].get("infra_failure")),
        "prompt_tokens": usage("prompt_tokens"), "completion_tokens": usage("completion_tokens"),
        "llm_calls": usage("llm_calls"), "tool_calls": usage("tool_calls"),
        "cost_usd": cost,
        "cost_per_resolved_usd": round(cost / resolved, 4) if cost is not None and resolved else None,
        "agent_latency_total_s": round(sum(lat), 2), "agent_latency_mean_s": round(sum(lat) / len(lat), 2) if lat else None,
        "eval_test_runtime_total_s": round(sum(r["eval"].get("test_runtime_s") or 0 for r in rows), 2),
        "dataset": (manifest or {}).get("dataset"),
    }


def compare(names: list[str]) -> str:
    """Side-by-side table over the instances common to all runs."""
    data = {n: {r["instance_id"]: r for r in read_jsonl(Run(n).results_path)} for n in names}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    lines = [f"{len(common)} common instances (of {', '.join(f'{n}:{len(d)}' for n, d in data.items())})", ""]
    hdr = f"{'run':<28}{'runtime':<18}{'resolved':>10}{'rate':>8}{'tokens':>12}{'cost$':>9}{'$/resolved':>12}{'latency_s':>11}"
    lines += [hdr, "-" * len(hdr)]
    for n, d in data.items():
        s = summarize([d[i] for i in common])
        c = f"{s['cost_usd']:.4f}" if s["cost_usd"] is not None else "-"
        cpr = f"{s['cost_per_resolved_usd']:.4f}" if s["cost_per_resolved_usd"] is not None else "-"
        lines.append(f"{n:<28}{s['runtime']['name']:<18}{s['resolved']:>6}/{s['total']:<3}{s['resolve_rate'] or 0:>8.1%}"
                     f"{s['prompt_tokens'] + s['completion_tokens']:>12}{c:>9}{cpr:>12}{s['agent_latency_total_s']:>11}")
    if len(names) >= 2:
        solved = {n: {i for i in common if data[n][i]["eval"]["resolved"]} for n in names}
        both = set.intersection(*solved.values())
        lines += ["", f"resolved by all: {len(both)}"]
        for n in names:
            only = solved[n] - set.union(*(solved[o] for o in names if o != n))
            lines.append(f"resolved only by {n}: {len(only)}" + (f"  {sorted(only)}" if only else ""))
    return "\n".join(lines)


def trajectory_table(name: str) -> str:
    """One row per instance: outcome next to what the trajectory looked like. Descriptive only."""
    rows = read_jsonl(Run(name).results_path)
    hdr = (f"{'instance':<30}{'eval':<19}{'f2p':>6}{'p2p':>9}{'steps':>6}{'stop':>13}{'edits':>6}{'tests':>6}"
           f"{'t_after':>8}{'errs':>5}{'files':>6}{'in_tok':>9}{'out':>6}{'cost$':>8}{'agent_s':>8}")
    out = [hdr, "-" * len(hdr)]
    for r in rows:
        a, e = r["agent"] or {}, r["eval"]
        st = (a.get("metadata") or {}).get("stats") or {}
        u = a.get("usage") or {}
        f2p = f"{e['f2p_passed']}/{e['f2p_total']}" if e.get("f2p_total") is not None else "-"
        p2p = f"{e['p2p_passed']}/{e['p2p_total']}" if e.get("p2p_total") is not None else "-"
        out.append(f"{r['instance_id']:<30}{e['status']:<19}{f2p:>6}{p2p:>9}{st.get('steps', '-'):>6}"
                   f"{st.get('stop_reason', a.get('status', '-')):>13}{st.get('successful_edits', '-'):>6}"
                   f"{st.get('tests_run', '-'):>6}{str(st.get('tested_after_last_edit', '-'))[:1]:>8}"
                   f"{st.get('tool_errors', '-'):>5}{len((a.get('metadata') or {}).get('modified_files') or []):>6}"
                   f"{u.get('prompt_tokens', 0):>9}{u.get('completion_tokens', 0):>6}"
                   f"{(u.get('cost_usd') or 0):>8.4f}{a.get('latency_s', 0):>8.0f}")
    return "\n".join(out)
