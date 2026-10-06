"""Official SWE-bench prediction format: JSONL of
{"instance_id", "model_name_or_path", "model_patch"}. Nothing else goes in this file;
our own experiment metadata is stored separately (see runner.py)."""
from __future__ import annotations

import json
from pathlib import Path


def write_predictions(path: Path, rows: list[dict]) -> None:
    seen = set()
    for r in rows:
        if r["instance_id"] in seen:
            raise ValueError(f"duplicate prediction for {r['instance_id']}")
        seen.add(r["instance_id"])
        if not isinstance(r["model_patch"], str):
            raise TypeError(f"{r['instance_id']}: model_patch must be a str")
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps({"instance_id": r["instance_id"],
                                "model_name_or_path": r["model_name_or_path"],
                                "model_patch": r["model_patch"]}) + "\n")
