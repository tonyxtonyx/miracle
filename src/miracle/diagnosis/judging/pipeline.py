"""Blind process judge -> independent reference -> oracle comparison, for one instance. Resumable (existing outputs are kept unless force)."""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from . import prompts as JP
from .clients import JudgeClient, JudgeError, JudgeResult
from .inputs import CandidateMaterial, base_excerpts, blind_user_prompt, oracle_user_prompt, reference_user_prompt
from .schema import BLIND_SCHEMA, ORACLE_SCHEMA, REFERENCE_SCHEMA, validate_and_normalize


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def _meta(r: JudgeResult) -> dict:
    return {"backend": r.backend, "model": r.model, "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": r.latency_s, **r.meta}


def call_validated(judge: JudgeClient, system: str, user: str, schema: dict, name: str, kind: str, material: CandidateMaterial | None,
                   raw_dir: Path) -> tuple[dict, dict]:
    """-> (result object with normalised citations, record {valid, problems, attempts, judge meta}). One repair round for invalid citations."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    attempts, prompt, problems, obj, res = [], user, [], {}, None
    for attempt in (1, 2):
        res = judge.complete(system, prompt, schema, name)
        (raw_dir / f"{name}.attempt{attempt}.json").write_text(json.dumps({"response_text": res.raw_text, "meta": _meta(res)}, indent=1))
        attempts.append(_meta(res))
        obj = res.data
        if material is None:
            problems = []
            break
        obj, problems = validate_and_normalize(obj, material.state, kind, material.event_count)
        if not problems:
            break
        prompt = (user + "\n\n# Your previous answer had these problems. Return the full corrected JSON.\n" + "\n".join(f"- {p}" for p in problems)
                  + "\n\nPrevious answer:\n" + json.dumps(res.data)[:20000])
    return obj, {"valid": not problems, "problems": problems, "attempts": attempts, "input_sha256": _sha(user), "system_sha256": _sha(system),
                 "judge": attempts[-1]}


def judge_instance(inst_dir: Path, instance: dict, official: dict, judge: JudgeClient, repo_dir_for_excerpts, log=print, force: bool = False) -> dict:
    """`instance` is the full dataset record (judges only). `repo_dir_for_excerpts()` returns a checkout of the base commit."""
    d = Path(inst_dir) / "judge"
    d.mkdir(parents=True, exist_ok=True)
    raw = d / "raw"
    material = CandidateMaterial.from_run_dir(inst_dir, instance["problem_statement"])      # <- only the issue string crosses from the instance
    out = {}

    # B. blind process judge: no gold, no official result
    if force or not (d / "blind.json").exists():
        user = blind_user_prompt(material)
        obj, rec = call_validated(judge, JP.BLIND_SYSTEM, user, BLIND_SCHEMA, "blind", "blind", material, raw)
        (d / "blind.json").write_text(json.dumps({"created_at": _now(), **rec, "result": obj}, indent=2, ensure_ascii=False))
        log(f"  blind judge done (valid={rec['valid']})")
    out["blind"] = json.loads((d / "blind.json").read_text())

    # C1. reference account: built from the instance alone, BEFORE the candidate graph is shown to anyone
    if force or not (d / "reference_diagnosis.json").exists():
        excerpts = base_excerpts(repo_dir_for_excerpts(), instance["patch"])
        user = reference_user_prompt(instance, excerpts)
        obj, rec = call_validated(judge, JP.REFERENCE_SYSTEM, user, REFERENCE_SCHEMA, "reference", "reference", None, raw)
        (d / "reference_diagnosis.json").write_text(json.dumps({"persisted_at": _now(), "reference_sha256": _sha(json.dumps(obj, sort_keys=True)), **rec,
                                                                "reference": obj}, indent=2, ensure_ascii=False))
        log("  reference account persisted")
    ref_rec = json.loads((d / "reference_diagnosis.json").read_text())                    # always re-read from disk

    # C2. oracle comparison: only now is the candidate shown next to the persisted reference
    if force or not (d / "oracle.json").exists():
        built_at = _now()
        assert ref_rec["persisted_at"] <= built_at, "the reference must be persisted before the oracle prompt is built"
        user = oracle_user_prompt(instance, ref_rec["reference"], official, material)
        obj, rec = call_validated(judge, JP.ORACLE_SYSTEM, user, ORACLE_SCHEMA, "oracle", "oracle", material, raw)
        (d / "oracle.json").write_text(json.dumps({"created_at": _now(), "oracle_prompt_built_at": built_at, "reference_persisted_at": ref_rec["persisted_at"],
                                                   "reference_sha256": ref_rec["reference_sha256"], "official_result_shown": {k: official.get(k) for k in ("status", "resolved")},
                                                   **rec, "result": obj}, indent=2, ensure_ascii=False))
        log(f"  oracle judge done (valid={rec['valid']})")
    out["oracle"] = json.loads((d / "oracle.json").read_text())
    out["reference"] = ref_rec
    return out


def toy_repo_dir() -> Path:
    from ..toytask import write_repo
    return write_repo(Path(tempfile.mkdtemp(prefix="toyrepo.")))
