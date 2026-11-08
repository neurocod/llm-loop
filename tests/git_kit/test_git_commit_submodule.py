"""commit.py -C against a real submodule of a linked pool worktree."""

import json
import subprocess
from pathlib import Path

import pytest

from kit_support import checkout_with_submodule, kit_env, out as git, run_kit
from llm_agents.git import commit
from llm_agents.git.commit_identity import ENV, write_approval


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    return checkout_with_submodule(tmp_path, "lib/sub")


def run_commit(slot, sub, message, *paths, env=None):
    return run_kit("commit", "-C", str(sub), "-F", str(message), *paths, cwd=slot, env=env)


def test_submodule_commit_uses_its_root_and_preserves_other_staged_work(checkout, tmp_path):
    slot, sub, _ = checkout
    parent_head = git(slot, "rev-parse", "HEAD")
    (slot / "work.txt").write_text("parent\n", encoding="utf-8")
    (sub / "work.txt").write_text("submodule\n", encoding="utf-8")
    (sub / "other.txt").write_text("staged\n", encoding="utf-8")
    git(sub, "add", "other.txt")
    message = tmp_path / "message.txt"
    message.write_text("Commit inside the submodule\n", encoding="utf-8")
    result = run_commit(slot, sub, message, "work.txt")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not message.exists()
    assert git(sub, "show", "HEAD:work.txt") == "submodule"
    assert git(sub, "diff", "--cached", "--name-only") == "other.txt"
    assert git(slot, "rev-parse", "HEAD") == parent_head
    assert git(slot, "diff", "--cached", "--name-only") == ""


def test_submodule_commit_checks_its_own_gitlinks(checkout, tmp_path):
    slot, sub, source = checkout
    # The parent's declaration of this path must not cover the submodule's.
    git(sub, "clone", str(source), "lib/sub")
    before = git(sub, "rev-parse", "HEAD")
    message = tmp_path / "message.txt"
    message.write_text("Must refuse an undeclared gitlink\n", encoding="utf-8")
    result = run_commit(slot, sub, message, "lib/sub")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "absent from .gitmodules" in result.stderr
    assert git(sub, "rev-parse", "HEAD") == before
    assert message.exists()


def _store_env(slot, directory):
    return kit_env(**{ENV: json.dumps({"workspace": str(slot), "directory": str(directory)})})


@pytest.mark.parametrize("mismatch", [None, "admin", "owner"])
def test_commit_rechecks_approved_identity(checkout, tmp_path, mismatch):
    slot, sub, _ = checkout
    directory = Path(git(slot, "rev-parse", "--absolute-git-dir")) / "approvals"
    directory.mkdir()
    message = tmp_path / "message.txt"
    message.write_text("Identity check\n", encoding="utf-8")
    args = ["-C", str(sub), "-F", str(message), "work.txt"]
    identity = {"cwd": str(sub), "root": str(sub),
                "admin": git(sub, "rev-parse", "--absolute-git-dir"),
                "owner": str(slot)}
    if mismatch:
        identity[mismatch] = str(tmp_path / "another")
    write_approval(directory, slot, args, identity)
    (sub / "work.txt").write_text("changed\n", encoding="utf-8")
    result = run_kit("commit", *args, cwd=slot, env=_store_env(slot, directory))
    if mismatch:
        assert result.returncode == 1, result.stdout + result.stderr
        assert f"approved {mismatch} changed" in result.stderr
        assert git(sub, "diff", "--cached", "--name-only") == ""
        assert git(sub, "show", "HEAD:work.txt") == "initial"
        assert message.exists()
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert git(sub, "show", "HEAD:work.txt") == "changed"


def test_worker_commit_without_approval_fails_closed(checkout, tmp_path):
    slot, sub, _ = checkout
    (sub / "work.txt").write_text("not approved\n", encoding="utf-8")
    message = tmp_path / "message.txt"
    message.write_text("No approval\n", encoding="utf-8")
    missing = Path(git(slot, "rev-parse", "--absolute-git-dir")) / "missing"
    result = run_commit(slot, sub, message, "work.txt", env=_store_env(slot, missing))
    assert result.returncode == 1
    assert "no readable approval" in result.stderr


@pytest.mark.parametrize("location", ["scratch", "main", "other_slot"])
def test_commit_refuses_approval_store_outside_own_admin(checkout, tmp_path, location):
    slot, sub, _ = checkout
    main = tmp_path / "main"
    if location == "other_slot":
        other = tmp_path / "main-wt2"
        git(main, "worktree", "add", "--detach", str(other))
        parent = Path(git(other, "rev-parse", "--absolute-git-dir"))
    elif location == "main":
        parent = main / ".git"
    else:
        parent = tmp_path
    directory = parent / "approvals"
    directory.mkdir()
    message = tmp_path / "message.txt"
    message.write_text("Untrusted store\n", encoding="utf-8")
    args = ["-C", str(sub), "-F", str(message), "work.txt"]
    write_approval(directory, slot, args, {
        "cwd": str(sub), "root": str(sub), "owner": str(slot),
        "admin": git(sub, "rev-parse", "--absolute-git-dir"),
    })
    (sub / "work.txt").write_text("changed\n", encoding="utf-8")
    result = run_kit("commit", *args, cwd=slot, env=_store_env(slot, directory))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "approval store is not under this slot's Git admin directory" in result.stderr
    assert git(sub, "show", "HEAD:work.txt") == "initial"
    assert git(sub, "diff", "--cached", "--name-only") == ""
    assert message.exists()


def test_guarded_writes_pin_git_directory(monkeypatch, tmp_path):
    seen = []

    def run(args, **kwargs):
        seen.append(args)
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(commit, "git", run)
    identity = {"admin": str(tmp_path / "approved.git"), "root": str(tmp_path / "tree")}
    commit.run_indexing(["add", "--", "work.txt"], identity)
    assert seen == [["--git-dir", identity["admin"], "--work-tree", identity["root"],
                     "add", "--", "work.txt"]]
