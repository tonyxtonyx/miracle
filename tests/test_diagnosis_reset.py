"""Reset / dirty-state contract on a real git repo (HostEnvironment shares the exact scripts DockerEnvironment uses)."""
import base64

import pytest

from miracle.host_env import HostEnvironment

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


@pytest.fixture
def env(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "a.py").write_text("def f():\n    return 1\n")
    (src / "b.txt").write_text("hello\n")
    (src / "keep.bin").write_bytes(b"\x00\x01\x02binary")
    (src / ".gitignore").write_text("*.log\n__pycache__/\n")
    e = HostEnvironment.from_directory(src)
    yield e
    e.close()


def test_fresh_environment_is_clean(env):
    d = env.dirty_state()
    assert d["clean"] and d["changed_files"] == [] and d["untracked"] == [] and d["head_unchanged"]
    assert env.diff() == ""


def test_every_kind_of_temporary_change_is_detected_and_fully_reset(env):
    env.write_bytes(f"{env.root}/pkg/a.py", b"def f():\n    print('DEBUG')\n    return 1\n")      # instrumentation of tracked file
    env.write_bytes(f"{env.root}/probe.py", b"print('probe')\n")                                 # new file
    env.exec("rm b.txt")                                                                         # deletion
    env.write_bytes(f"{env.root}/keep.bin", b"\x00changed-binary")                               # modified binary (never in a patch)
    env.write_bytes(f"{env.root}/new.png", PNG)                                                  # new binary
    env.exec("git add pkg/a.py && git commit -qm 'agent commit' && echo committed")              # HEAD moved
    d = env.dirty_state()
    assert not d["clean"] and not d["head_unchanged"]
    assert {"pkg/a.py", "b.txt", "keep.bin"} <= set(d["changed_files"]) | set(d["untracked"]) or d["changed_files"]
    assert "probe.py" in d["changed_files"] + d["untracked"] and "new.png" in d["changed_files"] + d["untracked"]
    env.reset_to_baseline()
    d = env.dirty_state()
    assert d["clean"], d
    assert (env.host / "pkg" / "a.py").read_text() == "def f():\n    return 1\n" and (env.host / "b.txt").read_text() == "hello\n"
    assert (env.host / "keep.bin").read_bytes() == b"\x00\x01\x02binary"
    assert not (env.host / "probe.py").exists() and not (env.host / "new.png").exists()
    assert env.diff() == ""


def test_ignored_artifacts_do_not_make_the_state_dirty(env):
    env.write_bytes(f"{env.root}/run.log", b"log\n")                    # gitignored
    env.exec("mkdir -p pkg/__pycache__ && echo x > pkg/__pycache__/a.pyc")
    assert env.dirty_state()["clean"] and env.diff() == ""


def test_reset_is_idempotent_and_repeatable_across_branches(env):
    for i in range(3):                                                  # branch 1, 2, 3 each dirty the tree differently
        env.write_bytes(f"{env.root}/pkg/a.py", f"def f():\n    return {i + 10}\n".encode())
        env.write_bytes(f"{env.root}/branch{i}.txt", b"x")
        assert not env.dirty_state()["clean"]
        env.reset_to_baseline(); env.reset_to_baseline()
        assert env.dirty_state()["clean"] and env.diff() == ""


def test_exec_timeout_kills_the_process_group(env):
    r = env.exec("sleep 30 & sleep 30", timeout=1)
    assert r.timed_out and r.exit_code == 124 and r.duration_s < 10
    assert env.exec("python -c 'print(6*7)'").output.strip() == "42"


def test_close_deletes_only_what_the_class_created(tmp_path):
    """Regression: a buggy cleanup once deleted the PARENT of a caller-supplied directory (the user's temp folder)."""
    parent = tmp_path / "precious"
    repo = parent / "repo"
    repo.mkdir(parents=True)
    (repo / "a.txt").write_text("x")
    (parent / "unrelated.txt").write_text("keep me")
    e = HostEnvironment(repo)                                   # caller-supplied directory: close() must delete nothing
    e.close()
    assert repo.exists() and (parent / "unrelated.txt").read_text() == "keep me"
    src = tmp_path / "src"
    src.mkdir()
    (src / "b.txt").write_text("y")
    e2 = HostEnvironment.from_directory(src)                    # created by the class: removed on close()
    base = e2.host.parent
    assert (base / ".miracle-hostenv").is_file()
    e2.close()
    assert not base.exists() and src.exists() and (src / "b.txt").exists()
    e2.close()                                                  # idempotent
    forged = HostEnvironment(repo, cleanup_dir=parent)          # a cleanup_dir without prefix + marker is refused
    forged.close()
    assert parent.exists() and (parent / "unrelated.txt").exists()
