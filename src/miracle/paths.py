"""Filesystem layout. Everything lives under one root so a checkout is self-contained."""
from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("MIRACLE_HOME", Path.cwd())).resolve()


def data_dir() -> Path:      # frozen dataset snapshots
    return root() / "data"


def subsets_dir() -> Path:   # named, versioned instance subsets
    return root() / "subsets"


def cache_dir() -> Path:     # pristine repo checkouts (rebuildable)
    return root() / "cache"


def runs_dir() -> Path:      # one directory per experiment
    return root() / "runs"
