import subprocess
import pytest


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MIRACLE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def upstream(tmp_path):
    """A local 'GitHub': <root>/org/proj with two commits. Returns (url_template, instance)."""
    repo = tmp_path / "up" / "org" / "proj"
    repo.mkdir(parents=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "PATH": __import__("os").environ["PATH"]}
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, env=env, check=True, capture_output=True, text=True).stdout.strip()
    run("init", "-q", "-b", "main")
    (repo / "a.py").write_text("x = 1\n")
    run("add", "-A"); run("commit", "-qm", "base")
    base = run("rev-parse", "HEAD")
    (repo / "a.py").write_text("x = 2  # the upstream fix\n")
    run("commit", "-qam", "fix")
    inst = {"instance_id": "org__proj-1", "repo": "org/proj", "base_commit": base,
            "problem_statement": "x should be 2", "patch": "SECRET", "test_patch": "SECRET",
            "FAIL_TO_PASS": ["SECRET"], "PASS_TO_PASS": ["SECRET"], "image": "img", "difficulty": "<15 min fix"}
    return f"file://{tmp_path}/up/{{repo}}", inst
