"""Deterministic subset selection and named, versioned subset files."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import paths
from .dataset import DatasetRef


@dataclass
class SubsetSpec:
    """How a subset was chosen. Filters apply first, then `n`/`seed` sampling."""
    ids: list[str] = field(default_factory=list)       # explicit ids (bypass sampling)
    repos: list[str] = field(default_factory=list)     # e.g. ["django/django"]
    difficulties: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    n: int | None = None
    seed: int = 0


def select(instances: list[dict], spec: SubsetSpec) -> list[str]:
    """Pure function of (instances, spec): same inputs always give the same ordered ids."""
    by_id = {i["instance_id"]: i for i in instances}
    if spec.ids:
        missing = [i for i in spec.ids if i not in by_id]
        if missing:
            raise KeyError(f"unknown instance ids: {missing}")
        pool = list(dict.fromkeys(spec.ids))
    else:
        pool = sorted(by_id)
    pool = [i for i in pool if i not in set(spec.exclude)]
    if spec.repos:
        pool = [i for i in pool if by_id[i]["repo"] in spec.repos]
    if spec.difficulties:
        pool = [i for i in pool if by_id[i].get("difficulty") in spec.difficulties]
    if spec.n is not None and spec.n < len(pool):
        pool = sorted(random.Random(spec.seed).sample(sorted(pool), spec.n))
    return pool


@dataclass
class Subset:
    name: str
    dataset: DatasetRef
    spec: SubsetSpec
    instance_ids: list[str]

    def save(self) -> Path:
        p = paths.subsets_dir() / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "name": self.name, "dataset": self.dataset.to_dict(),
            "spec": asdict(self.spec), "instance_ids": self.instance_ids,
        }, indent=2) + "\n")
        return p

    @staticmethod
    def load(name: str) -> "Subset":
        d = json.loads((paths.subsets_dir() / f"{name}.json").read_text())
        return Subset(d["name"], DatasetRef(**d["dataset"]), SubsetSpec(**d["spec"]), d["instance_ids"])
