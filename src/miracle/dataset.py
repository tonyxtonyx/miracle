"""Load SWE-bench Verified as frozen, content-addressed snapshots.

Instances stay in the official record format (we never reshape them), so the
official harness can consume them directly. A snapshot is the Hugging Face
revision plus the sha256 of the exact JSONL we derived from it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from . import paths

DATASET_ID = "SWE-bench/SWE-bench_Verified"


@dataclass(frozen=True)
class DatasetRef:
    name: str
    revision: str | None
    sha256: str

    def to_dict(self) -> dict:
        return asdict(self)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve_revision(name: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(name, revision=revision).sha
    except Exception:
        return None  # offline; caller falls back to the local cache


def snapshot_path(name: str, revision: str) -> Path:
    return paths.data_dir() / f"{name.replace('/', '__')}@{revision[:12]}.jsonl"


def load_snapshot(name: str = DATASET_ID, revision: str | None = None) -> tuple[DatasetRef, list[dict]]:
    """Return (ref, instances). Downloads once per revision, then reads the local JSONL.

    revision=None means "current main" if reachable, else the newest cached snapshot.
    Pass an explicit revision (a sha from a previous manifest) to reproduce a run exactly.
    """
    resolved = _resolve_revision(name, revision) or revision
    path = snapshot_path(name, resolved) if resolved else None
    if path is None or not path.exists():
        if resolved is None:
            cached = sorted(paths.data_dir().glob(f"{name.replace('/', '__')}@*.jsonl"),
                            key=lambda p: p.stat().st_mtime)
            if not cached:
                raise RuntimeError(f"cannot reach Hugging Face and no cached snapshot of {name}")
            path = cached[-1]
            resolved = path.stem.split("@", 1)[1]
        else:
            from datasets import load_dataset

            ds = load_dataset(name, split="test", revision=resolved)
            write_jsonl(path, [dict(r) for r in ds])
    return DatasetRef(name, resolved, sha256_file(path)), read_jsonl(path)
