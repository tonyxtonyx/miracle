"""Read-only data access for the web UI: everything is derived from the recorded artefacts (event log, ledger, judge outputs, reports)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..graph import read_events, replay

NAME_RE = re.compile(r"^[\w.\-]+$")


class NotFound(Exception):
    pass


def _safe(name: str) -> str:
    if not name or not NAME_RE.match(name) or name in (".", ".."):
        raise NotFound(f"bad name {name!r}")
    return name


def _json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def instance_dir(runs_dir: Path, run: str, instance: str) -> Path:
    d = Path(runs_dir) / _safe(run) / "agent" / _safe(instance)
    if not (d / "diagnosis" / "events.jsonl").exists():
        raise NotFound(f"no diagnosis recorded for {run}/{instance}")
    return d


def official_result(runs_dir: Path, run: str, instance: str) -> dict | None:
    p = Path(runs_dir) / _safe(run) / "results.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["instance_id"] == instance:
                    return r.get("eval")
    return None


def list_runs(runs_dir: Path) -> list[dict]:
    out = []
    for d in sorted(Path(runs_dir).iterdir()) if Path(runs_dir).exists() else []:
        if (d / "agent").is_dir():
            n = [x for x in (d / "agent").iterdir() if (x / "diagnosis" / "events.jsonl").exists()]
            if n:
                out.append({"name": d.name, "instances": len(n)})
    return out


def list_instances(runs_dir: Path, run: str) -> list[dict]:
    base = Path(runs_dir) / _safe(run) / "agent"
    rows = []
    for d in sorted(base.iterdir()) if base.exists() else []:
        ev = d / "diagnosis" / "events.jsonl"
        if not ev.exists():
            continue
        events = read_events(ev)
        dg = _json(d / "diagnosis" / "diagnosis.json")
        blind = (_json(d / "judge" / "blind.json") or {}).get("result", {}).get("scores")
        orc = (_json(d / "judge" / "oracle.json") or {}).get("result", {}).get("scores")
        off = official_result(runs_dir, run, d.name)
        rows.append({"id": d.name, "events": len(events), "phase": next((e["data"]["phase"] for e in reversed(events) if e["type"] == "phase_changed"), "init"),
                     "diagnosis_status": (dg or {}).get("status"), "official": {k: (off or {}).get(k) for k in ("status", "resolved")} if off else None,
                     "blind_mean": round(sum(v["score"] for v in blind.values()) / len(blind), 2) if blind else None,
                     "causal_correctness": orc["causal_correctness"]["score"] if orc and "causal_correctness" in orc else None})
    return rows


def meta(runs_dir: Path, run: str, instance: str) -> dict:
    d = instance_dir(runs_dir, run, instance)
    p = d / "diagnosis" / "events.jsonl"
    n = sum(1 for l in p.read_text().splitlines() if l.strip())
    return {"events": n, "mtime": p.stat().st_mtime, "done": (d / "diagnosis" / "graph.json").exists()}


def graph_at(runs_dir: Path, run: str, instance: str, upto: int | None) -> dict:
    d = instance_dir(runs_dir, run, instance)
    events = read_events(d / "diagnosis" / "events.jsonl")
    iid = _iid(events, instance)
    n = len(events) if upto is None else max(0, min(upto, len(events)))
    return {"total_events": len(events), "upto": n, "graph": replay(iid, events, n).to_dict()}


def _iid(events: list[dict], fallback: str) -> str:
    for e in events:
        v = e["data"].get("id")
        if isinstance(v, str) and ":" in v:
            return v.rsplit(":", 1)[0]
    return fallback


def events(runs_dir: Path, run: str, instance: str) -> list[dict]:
    return read_events(instance_dir(runs_dir, run, instance) / "diagnosis" / "events.jsonl")


def tool_calls(runs_dir: Path, run: str, instance: str) -> list[dict]:
    p = instance_dir(runs_dir, run, instance) / "diagnosis" / "tool_calls.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
    for r in rows:
        r["output"] = (r.get("output") or "")[:6000]
    return rows


def artifacts(runs_dir: Path, run: str, instance: str) -> dict:
    d = instance_dir(runs_dir, run, instance)
    j = d / "judge"
    return {"diagnosis": _json(d / "diagnosis" / "diagnosis.json"), "usage": _json(d / "diagnosis" / "usage.json"),
            "resets": _json(d / "diagnosis" / "reset_verification.json", []), "report": _json(d / "report.json"),
            "implementation": _json(d / "implementation" / "result.json"), "official": official_result(runs_dir, run, instance),
            "judge": {k: _json(j / f"{k}.json") for k in ("blind", "oracle")} | {"reference": _json(j / "reference_diagnosis.json")},
            "repo_map": (d / "diagnosis" / "repo_map.txt").read_text() if (d / "diagnosis" / "repo_map.txt").exists() else None}
