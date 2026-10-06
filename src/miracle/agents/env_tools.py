"""Repository tools backed by an `Environment`: inspect, search, edit, run commands, view diff.

Same contract as llm.tools: tools never raise; failures return "ERROR: ..." so the model sees them.
Outputs are length-capped (head+tail for commands) -- there is no smarter context management in S0.
"""
from __future__ import annotations

import posixpath
import re

from ..environment import Environment

MAX_READ_LINES = 300
MAX_OUT_CHARS = 15_000
MAX_CMD_CHARS = 12_000
MAX_MATCHES = 60
MAX_LINE_CHARS = 300
DEFAULT_CMD_TIMEOUT = 120
MAX_CMD_TIMEOUT = 600

TEST_CMD_RE = re.compile(
    r"\b(pytest|py\.test|runtests\.py|tox|nosetests|manage\.py\s+test|unittest|bin/test|setup\.py\s+test)\b")


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required}}}


def _S(d):
    return {"type": "string", "description": d}


def _I(d):
    return {"type": "integer", "description": d}


SPECS = [
    _fn("list_files", "List the entries of a directory (non-recursive). Directories end with '/'.",
        {"path": _S("Directory relative to the repository root; '.' for the root.")}, ["path"]),
    _fn("read_file", f"Read a text file with line numbers. Shows at most {MAX_READ_LINES} lines per call; "
                     "use start_line/end_line for more.",
        {"path": _S("File path relative to the repository root."),
         "start_line": _I("First line to show (1-based, optional)."),
         "end_line": _I("Last line to show (inclusive, optional).")}, ["path"]),
    _fn("search_code", "Case-insensitive literal text search over the repository's files (git grep). "
                       "Returns 'path:line: text'.",
        {"query": _S("Text to search for."),
         "path": _S("Optional subdirectory or file to limit the search.")}, ["query"]),
    _fn("edit_file", "Replace one exact occurrence of old_str with new_str in a file. old_str must match exactly "
                     "(including whitespace/indentation) and occur exactly once; include surrounding lines to "
                     "make it unique.",
        {"path": _S("File path relative to the repository root."),
         "old_str": _S("Exact text to replace."), "new_str": _S("Replacement text.")},
        ["path", "old_str", "new_str"]),
    _fn("write_file", "Create a new file, or completely overwrite an existing one, with the given content.",
        {"path": _S("File path relative to the repository root."), "content": _S("Full file content.")},
        ["path", "content"]),
    _fn("run_command", "Run a shell command at the repository root in the task's Python environment (e.g. run "
                       "tests or a script). Each call is a fresh shell: `cd` and environment variables do not "
                       "persist between calls.",
        {"command": _S("Shell command."),
         "timeout": _I(f"Seconds before the command is killed (default {DEFAULT_CMD_TIMEOUT}, "
                       f"max {MAX_CMD_TIMEOUT}).")}, ["command"]),
    _fn("git_diff", "Show all changes you have made to the repository so far, as a unified diff.", {}, []),
]
TOOL_NAMES = [s["function"]["name"] for s in SPECS]


class ToolError(Exception):
    pass


def _trunc(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


def _head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit // 3, limit - limit // 3
    return text[:head] + f"\n... [{len(text) - limit} chars omitted] ...\n" + text[-tail:]


def _sh(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


class EnvToolBox:
    specs = SPECS

    def __init__(self, env: Environment):
        self.env = env

    # -- dispatch ------------------------------------------------------------------
    def call(self, name: str, args: dict) -> tuple:
        fn = getattr(self, f"t_{name}", None) if name in TOOL_NAMES else None
        if fn is None:
            return f"ERROR: unknown tool {name!r}. Available: " + ", ".join(TOOL_NAMES), True
        try:
            res = fn(**args)
            return res if isinstance(res, tuple) else (res, False)
        except ToolError as e:
            return f"ERROR: {e}", True
        except TypeError as e:
            return f"ERROR: bad arguments for {name}: {e}", True

    # -- helpers -------------------------------------------------------------------
    def _abs(self, path: str) -> str:
        root = self.env.root
        p = (path or ".").strip()
        full = posixpath.normpath(p if p.startswith("/") else posixpath.join(root, p))
        if full != root and not full.startswith(root + "/"):
            raise ToolError(f"path is outside the repository: {path!r}")
        if "/.git/" in full + "/":
            raise ToolError("the .git directory is not accessible with this tool")
        return full

    def _rel(self, full: str) -> str:
        return posixpath.relpath(full, self.env.root)

    def _read_text(self, path: str) -> tuple[str, str]:
        full = self._abs(path)
        try:
            data = self.env.read_bytes(full)
        except FileNotFoundError:
            raise ToolError(f"no such file: {path}")
        except IsADirectoryError:
            raise ToolError(f"{path} is a directory; use list_files")
        except OSError as e:
            raise ToolError(f"cannot read {path}: {e}")
        if b"\x00" in data[:8000]:
            raise ToolError(f"{path} looks like a binary file")
        return full, data.decode("utf-8", "surrogateescape")

    # -- tools ---------------------------------------------------------------------
    def t_list_files(self, path: str = ".") -> str:
        full = self._abs(path)
        r = self.env.exec(f"ls -1Ap -- {_sh(full)}", timeout=30)
        if r.exit_code != 0:
            raise ToolError(r.output.strip() or f"cannot list {path}")
        names = [n for n in r.output.splitlines() if n not in ("./", "../")]
        if not names:
            return "(empty directory)"
        extra = f"\n... [{len(names) - 200} more entries]" if len(names) > 200 else ""
        return "\n".join(names[:200]) + extra

    def t_read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        _, text = self._read_text(path)
        lines = text.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        total = len(lines)
        lo = max(1, int(start_line)) if start_line else 1
        if total and lo > total:
            raise ToolError(f"start_line {lo} is past the end of the file ({total} lines)")
        hi = min(total, int(end_line)) if end_line else total
        hi = min(hi, lo + MAX_READ_LINES - 1)
        body = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(lo, hi + 1))
        note = f"\n[showing lines {lo}-{hi} of {total}]" if (lo > 1 or hi < total) else ""
        return _trunc(body, MAX_OUT_CHARS) + note

    def t_search_code(self, query: str, path: str = ".") -> str:
        if not query:
            raise ToolError("empty query")
        target = self._rel(self._abs(path))
        r = self.env.exec(f"git grep -n -I -F -i --untracked -e {_sh(query)} -- {_sh(target)} "
                          f"| head -n {MAX_MATCHES + 1}", timeout=60)
        lines = [l[:MAX_LINE_CHARS] for l in r.output.splitlines()]
        if not lines:
            return "(no matches)"
        more = (f"\n... [stopped at {MAX_MATCHES} matches; narrow the query or path]"
                if len(lines) > MAX_MATCHES else "")
        return "\n".join(lines[:MAX_MATCHES]) + more

    def t_edit_file(self, path: str, old_str: str, new_str: str) -> tuple:
        full, text = self._read_text(path)
        if old_str == "":
            raise ToolError("old_str is empty")
        n = text.count(old_str)
        if n == 0:
            raise ToolError("old_str was not found in the file. It must match exactly, including whitespace and "
                            "indentation. Use read_file to see the current contents.")
        if n > 1:
            raise ToolError(f"old_str occurs {n} times; include more surrounding lines so it matches exactly once.")
        pos = text.index(old_str)
        new_text = text[:pos] + new_str + text[pos + len(old_str):]
        self.env.write_bytes(full, new_text.encode("utf-8", "surrogateescape"))
        start = text[:pos].count("\n") + 1
        lines = new_text.split("\n")
        end = start + new_str.count("\n")
        lo, hi = max(1, start - 2), min(len(lines), end + 2)
        snippet = "\n".join(f"{i:>6}\t{lines[i - 1]}" for i in range(lo, hi + 1))
        return f"OK: edited {self._rel(full)}\n{_trunc(snippet, 3000)}", False, {"path": self._rel(full)}

    def t_write_file(self, path: str, content: str) -> tuple:
        full = self._abs(path)
        self.env.write_bytes(full, content.encode("utf-8", "surrogateescape"))
        return f"OK: wrote {len(content)} characters to {self._rel(full)}", False, {"path": self._rel(full)}

    def t_run_command(self, command: str, timeout: int = DEFAULT_CMD_TIMEOUT) -> tuple:
        t = max(1, min(int(timeout), MAX_CMD_TIMEOUT))
        r = self.env.exec(command, timeout=t)
        meta = {"command": command, "exit_code": r.exit_code, "timed_out": r.timed_out,
                "duration_s": r.duration_s, "is_test": bool(TEST_CMD_RE.search(command))}
        head = f"[timed out after {t}s and was killed]\n" if r.timed_out else ""
        return f"{head}exit_code: {r.exit_code}\n{_head_tail(r.output, MAX_CMD_CHARS)}", r.exit_code != 0, meta

    def t_git_diff(self) -> str:
        d = self.env.diff()
        return _trunc(d, MAX_OUT_CHARS) if d else "(no changes yet)"
