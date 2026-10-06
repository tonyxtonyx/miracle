"""Copy a run's evidence (not its heavy/rebuildable parts) into a committable directory.

Included: manifest, summary, results, predictions, per-instance agent result/patch/trace, and the
official harness' reports/logs. Excluded: instances.jsonl (re-derivable from the pinned dataset revision).
Local absolute paths are replaced with `<project>` so personal directory names don't end up in a public repo.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import paths

TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".diff", ".sh", ".md"}


def scrub(text: str, project_root: Path | None = None) -> str:
    root = str(project_root or paths.root())
    home = str(Path.home())
    return text.replace(root, "<project>").replace(home, "~")


def export_run(name: str, dest: Path) -> int:
    src = paths.runs_dir() / name
    n = 0
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if not f.is_file() or rel.name == "instances.jsonl" or "__pycache__" in rel.parts:
            continue
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if f.suffix in TEXT_SUFFIXES:
            out.write_text(scrub(f.read_text(errors="replace")))
        else:
            shutil.copy2(f, out)
        n += 1
    return n
