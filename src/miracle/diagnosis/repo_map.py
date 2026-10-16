"""ORIENT: a compact, deterministic repository map (no LLM, no free investigation): directory layout with file counts, key top-level files,
test locations, and where identifiers mentioned in the issue occur."""
from __future__ import annotations

import re
from collections import Counter

from ..environment import Environment

MAX_CHARS = 3800
_STOP = {"the", "this", "that", "with", "when", "from", "have", "should", "would", "could", "return", "returns", "error", "issue", "the", "none",
         "true", "false", "python", "import", "class", "def", "self", "test", "tests", "name", "value", "values", "using", "used"}


def issue_terms(issue: str, limit: int = 12) -> list[str]:
    cands: list[str] = re.findall(r"`([^`\n]{2,60})`", issue)
    cands += re.findall(r"\b[\w./-]+\.(?:py|js|ts|rst|txt|cfg|toml|ini|c|h|go|rs|java)\b", issue)
    cands += re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b", issue)            # CamelCase
    cands += re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", issue)                         # snake_case
    cands += re.findall(r"\b[a-z_]+\.[a-z_][\w.]*\b", issue)                         # dotted
    out, seen = [], set()
    for c in cands:
        c = c.strip().strip("()").strip(".,:;'\"")
        k = c.lower()
        if len(c) >= 3 and k not in seen and k not in _STOP and not c.isdigit() and " " not in c:
            seen.add(k)
            out.append(c)
    return out[:limit]


def build_repo_map(env: Environment, issue: str) -> str:
    r = env.exec("git ls-files | head -n 20000", timeout=60)
    files = [f for f in r.output.splitlines() if f.strip()] if r.exit_code == 0 else []
    lines = [f"{len(files)} tracked files."]
    top = Counter(f.split("/")[0] if "/" in f else "." for f in files)
    two = Counter("/".join(f.split("/")[:2]) for f in files if f.count("/") >= 1)
    lines.append("Top-level layout (entries: file count):")
    for name, n in top.most_common(18):
        subs = [(k, v) for k, v in two.most_common() if k.startswith(name + "/")][:4]
        lines.append(f"  {name}{'/' if name != '.' else ''}: {n}" + ("  [" + ", ".join(f"{k.split('/', 1)[1]}:{v}" for k, v in subs) + "]" if subs else ""))
    key = [f for f in files if "/" not in f and re.match(r"(?i)(readme|setup\.py|setup\.cfg|pyproject\.toml|tox\.ini|pytest\.ini|manage\.py|makefile|package\.json)", f)]
    if key:
        lines.append("Key files: " + ", ".join(key[:8]))
    tests = sorted({"/".join(f.split("/")[:2]) for f in files if re.search(r"(^|/)tests?(/|_)|(^|/)test_", f)})[:8]
    if tests:
        lines.append("Test locations: " + ", ".join(tests))
    terms = issue_terms(issue)
    if terms:
        lines.append("Where terms from the issue occur (git grep, file counts):")
        for t in terms:
            q = "'" + t.replace("'", "'\\''") + "'"
            g = env.exec(f"git grep -l -F -i -e {q} | head -n 6", timeout=60)
            hits = [h for h in g.output.splitlines() if h.strip()]
            if hits:
                lines.append(f"  {t}: " + ", ".join(hits[:5]) + (" ..." if len(hits) > 5 else ""))
    text = "\n".join(lines)
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n...[truncated]"
