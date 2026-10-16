"""Freezing: prompts, configuration and judge rubrics are hashed and recorded in FROZEN.json BEFORE the target list is processed.
The pipeline refuses to run if anything differs, so prompts cannot be tuned against the target tasks by accident."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from .config import DiagnosisConfig
from .prompts import prompt_hashes

FROZEN_PATH = Path(__file__).with_name("FROZEN.json")


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def current_manifest() -> dict:
    m = {"diagnosis_prompts": prompt_hashes(), "diagnosis_config_sha256": _sha(DiagnosisConfig().to_dict())}
    try:
        from .judging.prompts import judge_prompt_hashes
        m["judge_prompts"] = judge_prompt_hashes()
    except ImportError:
        m["judge_prompts"] = {}
    return m


def git_commit() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True)
    return r.stdout.strip() or None


def write_frozen() -> dict:
    m = {**current_manifest(), "git_commit_at_freeze": git_commit(), "config": DiagnosisConfig().to_dict()}
    FROZEN_PATH.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
    return m


def verify_frozen() -> tuple[bool, list[str]]:
    """(ok, problems). Not frozen yet -> (False, ['not frozen'])."""
    if not FROZEN_PATH.exists():
        return False, ["FROZEN.json is missing: run `miracle diagnosis freeze` first"]
    frozen, now = json.loads(FROZEN_PATH.read_text()), current_manifest()
    problems = []
    for key in ("diagnosis_prompts", "judge_prompts"):
        for name in sorted(set(frozen.get(key, {})) | set(now[key])):
            if frozen.get(key, {}).get(name) != now[key].get(name):
                problems.append(f"{key}.{name} changed since the freeze")
    if frozen.get("diagnosis_config_sha256") != now["diagnosis_config_sha256"]:
        problems.append("DiagnosisConfig changed since the freeze")
    return not problems, problems
