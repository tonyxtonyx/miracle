"""Environment backed by a container of the official SWE-bench instance image.

The agent works inside the exact image the official evaluation later uses (same Python env,
same installed repo, compiled extensions), so "tests pass for the agent" means the same thing
as in grading. Only the docker CLI is used (no SDK) to keep behaviour easy to reproduce by hand.
"""
from __future__ import annotations

import subprocess
import time

from .environment import Environment, ExecResult
from .evaluator import platform_of

CONDA_PRELUDE = (
    "source /opt/miniconda3/bin/activate >/dev/null 2>&1; conda activate testbed; "
    "export LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8; cd {root}"
)
# never part of a patch: bytecode / tool caches an agent's test runs create
_EXCLUDES = ["':(exclude)**/__pycache__/**'", "':(exclude)*.pyc'", "':(exclude).pytest_cache/**'",
             "':(exclude).hypothesis/**'", "':(exclude)**/*.egg-info/**'"]
_PATHSPEC = " ".join(["."] + _EXCLUDES)


class DockerError(RuntimeError):
    pass


class DockerEnvironment(Environment):
    def __init__(self, image: str, name: str, network: str = "none", prelude: str = CONDA_PRELUDE,
                 root: str = "/testbed"):
        self.image, self.name, self.network, self.root = image, name, network, root
        self._prelude = prelude.format(root=root)
        self._base_tree: str | None = None
        self.base_head: str | None = None
        self._started = False

    # -- lifecycle -------------------------------------------------------------
    def __enter__(self) -> "DockerEnvironment":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def start(self) -> None:
        plat = platform_of(self.image)
        cmd = ["docker", "run", "-d", "--name", self.name, "--network", self.network,
               "--cap-add", "SYS_ADMIN"] + (["--platform", plat] if plat else []) + \
              [self.image, "tail", "-f", "/dev/null"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise DockerError(f"docker run failed: {r.stderr.strip()[:300]}")
        self._started = True
        # same locale setup the official eval script performs; baseline snapshot of the tree
        setup = (f"cd {self.root} && git config --global --add safe.directory {self.root} && "
                 "{ sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen; } >/dev/null 2>&1; "
                 f"git rev-parse HEAD && {{ git add -A -- {_PATHSPEC} >/dev/null 2>&1 || true; }} && git write-tree")
        r = self._raw(["bash", "-c", setup], timeout=300)
        lines = r.stdout.decode().split()
        if r.returncode != 0 or len(lines) < 2:
            raise DockerError(f"environment setup failed: {r.stdout.decode()[-300:]}")
        self.base_head, self._base_tree = lines[-2], lines[-1]

    def close(self) -> None:
        if self._started:
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self._started = False

    # -- primitives --------------------------------------------------------------
    def _raw(self, argv: list[str], timeout: float, input: bytes | None = None,
             merge_stderr: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", "exec", *(["-i"] if input is not None else []), "-w", self.root,
                               self.name, *argv], input=input, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT if merge_stderr else subprocess.DEVNULL, timeout=timeout)

    def exec(self, command: str, timeout: int = 120) -> ExecResult:
        t0 = time.monotonic()
        try:
            r = self._raw(["timeout", "-k", "5", str(timeout), "bash", "-c", f"{self._prelude}\n{command}"],
                          timeout=timeout + 30)
            out, code = r.stdout.decode("utf-8", "replace"), r.returncode
            timed_out = code in (124, 137)
        except subprocess.TimeoutExpired as e:
            out, code, timed_out = (e.stdout or b"").decode("utf-8", "replace"), 124, True
        return ExecResult(code, out, timed_out, round(time.monotonic() - t0, 3))

    def read_bytes(self, path: str) -> bytes:
        r = subprocess.run(["docker", "exec", "-w", self.root, self.name, "cat", "--", path],
                           capture_output=True)
        if r.returncode != 0:
            msg = r.stderr.decode("utf-8", "replace").strip()
            if "No such file" in msg:
                raise FileNotFoundError(path)
            if "Is a directory" in msg:
                raise IsADirectoryError(path)
            raise OSError(msg)
        return r.stdout

    def write_bytes(self, path: str, data: bytes) -> None:
        r = self._raw(["bash", "-c", 'mkdir -p "$(dirname "$1")" && cat > "$1"', "_", path], timeout=60, input=data)
        if r.returncode != 0:
            raise OSError(r.stdout.decode("utf-8", "replace").strip())

    # -- patch extraction ---------------------------------------------------------
    def _stage(self) -> str:
        """Stage everything, then un-stage binary files (a text patch can't carry them)."""
        # `git add` exits 1 (while still staging everything) when a gitignored path such as .pytest_cache exists;
        # its status must not gate the binary filter below, so it is deliberately not chained with &&.
        return (f"cd {self.root} && {{ git add -A -- {_PATHSPEC} >/dev/null 2>&1 || true; }}; "
                f"git diff --cached --no-renames --numstat {self._base_tree} | "
                "awk -F'\\t' '$1==\"-\" && $2==\"-\" {print $3}' | while IFS= read -r f; do git reset -q -- \"$f\"; done")

    def diff(self) -> str:
        # stdout only: git writes warnings (e.g. "paths are ignored") to stderr and they must not enter the patch
        r = self._raw(["bash", "-c", f"{self._stage()}; git diff --cached --no-renames --no-color "
                                      f"--no-ext-diff {self._base_tree}"], timeout=120, merge_stderr=False)
        out = r.stdout.decode("utf-8", "replace")
        return out if not out or out.endswith("\n") else out + "\n"

    def changed_files(self) -> list[str]:
        r = self._raw(["bash", "-c", f"{self._stage()}; git diff --cached --no-renames --name-only {self._base_tree}"],
                      timeout=120, merge_stderr=False)
        return [l for l in r.stdout.decode("utf-8", "replace").splitlines() if l]
