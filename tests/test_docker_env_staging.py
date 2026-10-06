"""The patch-extraction shell script of DockerEnvironment, run locally against a real git repo.

Regression: with a gitignored `.pytest_cache` present, `git add -A -- . ':(exclude).pytest_cache/**'` exits 1.
Chaining the binary-unstaging step with `&&` silently skipped it, so binary files (e.g. PNGs written by an
agent's scratch scripts) leaked into patches as unappliable "Binary files ... differ" entries.
"""
import base64
import subprocess

from miracle.docker_env import DockerEnvironment

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def _repo(tmp_path, pytest_cache: bool):
    sh = lambda c: subprocess.run(c, shell=True, cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()
    sh("git init -q && git config user.email t@t && git config user.name t")
    (tmp_path / ".gitignore").write_text(".pytest_cache/\n")
    (tmp_path / "a.py").write_text("x = 1\n")
    sh("git add -A && git commit -qm base")
    tree = sh("git write-tree")
    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "scratch.py").write_text("y = 1\n")
    (tmp_path / "plot.png").write_bytes(PNG)
    if pytest_cache:
        (tmp_path / ".pytest_cache").mkdir()
        (tmp_path / ".pytest_cache" / "f").write_text("x")
    env = DockerEnvironment("img", "n")
    env.root, env._base_tree = str(tmp_path), tree
    return env, sh


def _staged(env, sh):
    return sh(f"{env._stage()}; git diff --cached --no-renames --name-only {env._base_tree}").split()


def test_binaries_are_dropped_even_when_git_add_exits_nonzero_because_of_pytest_cache(tmp_path):
    env, sh = _repo(tmp_path, pytest_cache=True)
    rc = subprocess.run(f"git add -A -- . ':(exclude).pytest_cache/**' >/dev/null 2>&1", shell=True, cwd=tmp_path).returncode
    assert rc == 1, "precondition: this git version exits 1 here (the trigger)"
    subprocess.run("git reset -q", shell=True, cwd=tmp_path)
    assert sorted(_staged(env, sh)) == ["a.py", "scratch.py"]          # plot.png excluded, .pytest_cache excluded


def test_binaries_are_dropped_in_the_ordinary_case(tmp_path):
    env, sh = _repo(tmp_path, pytest_cache=False)
    assert sorted(_staged(env, sh)) == ["a.py", "scratch.py"]
