"""Read-only repository tools exposed to a model, sandboxed to one root directory.

Tools never raise: failures come back as "ERROR: ..." strings so the model can see and
recover from them (and the trace records `is_error`)."""
from __future__ import annotations

import json
from pathlib import Path

MAX_READ_CHARS = 20_000
MAX_MATCHES = 50
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", ".venv"}

SPECS = [
    {"type": "function", "function": {
        "name": "list_files", "description": "List the entries of a directory (non-recursive). Directories end with '/'.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory relative to the repository root. Use '.' for the root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a text file and return its contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path relative to the repository root."}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_code", "description": "Case-insensitive substring search across all text files. Returns 'path:line: text' matches.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Text to search for."}},
            "required": ["query"]}}},
]


class ToolError(Exception):
    pass


class ToolBox:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    @property
    def specs(self) -> list[dict]:
        return SPECS

    def _resolve(self, rel: str) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"path escapes the repository: {rel!r}")
        return p

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Returns (output, is_error)."""
        fn = {"list_files": self.list_files, "read_file": self.read_file, "search_code": self.search_code}.get(name)
        if fn is None:
            return f"ERROR: unknown tool {name!r}. Available: list_files, read_file, search_code", True
        try:
            return fn(**args), False
        except ToolError as e:
            return f"ERROR: {e}", True
        except TypeError as e:           # missing / unexpected argument
            return f"ERROR: bad arguments for {name}: {e}", True

    def list_files(self, path: str = ".") -> str:
        p = self._resolve(path)
        if not p.is_dir():
            raise ToolError(f"not a directory: {path}")
        entries = sorted(e for e in p.iterdir() if e.name not in SKIP_DIRS)
        return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty directory)"

    def read_file(self, path: str) -> str:
        p = self._resolve(path)
        if not p.is_file():
            raise ToolError(f"no such file: {path}")
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            raise ToolError(f"not a text file: {path}")
        if len(text) > MAX_READ_CHARS:
            return text[:MAX_READ_CHARS] + f"\n... [truncated, {len(text) - MAX_READ_CHARS} more characters]"
        return text

    def search_code(self, query: str) -> str:
        if not query:
            raise ToolError("empty query")
        q, out = query.lower(), []
        for f in sorted(self.root.rglob("*")):
            if not f.is_file() or SKIP_DIRS & set(f.relative_to(self.root).parts):
                continue
            try:
                lines = f.read_text().splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            for i, line in enumerate(lines, 1):
                if q in line.lower():
                    out.append(f"{f.relative_to(self.root)}:{i}: {line.strip()}")
                    if len(out) >= MAX_MATCHES:
                        return "\n".join(out) + f"\n... [stopped at {MAX_MATCHES} matches]"
        return "\n".join(out) or "(no matches)"
