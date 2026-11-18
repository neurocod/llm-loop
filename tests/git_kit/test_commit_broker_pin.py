"""The commit broker's pinned git (`commit_broker._FORCED` through
`gitlib.pin`): each forced setting and the `GIT_*` strip, by its effect.

A repository the worker writes may name in its own config a program git
starts on a commit's way (an fsmonitor, an external diff, an editor), and
the environment may redirect git (`GIT_DIR`). Each test plants one of them,
runs git through the pin, and asserts the planted program did not run or
the redirect did not take; its control shows that the same fixture does run
it with git unpinned."""

from pathlib import Path

import pytest

from llm_agents import gitlib
from llm_agents.git import commit_broker


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository with one commit and a modified file, git unpinned (the
    test pins it), and `marker` a file a planted program appends to."""
    monkeypatch.setattr(gitlib, "_pinned", None)
    # An inherited editor would mask the repository's in the control.
    monkeypatch.delenv("GIT_EDITOR", raising=False)
    work = tmp_path / "repo"
    work.mkdir()
    for args in (["init", "--quiet"], ["config", "user.name", "t"],
                 ["config", "user.email", "t@example.invalid"]):
        gitlib.git(args, cwd=work)
    (work / "a.txt").write_text("one\n", encoding="utf-8")
    gitlib.git(["add", "a.txt"], cwd=work)
    gitlib.git(["commit", "--quiet", "-m", "initial"], cwd=work)
    (work / "a.txt").write_text("two\n", encoding="utf-8")
    return work, tmp_path / "marker.txt"


def _program(tmp_path: Path, marker: Path) -> str:
    """A shell script that records its start, as a config value git runs."""
    script = tmp_path / "planted.sh"
    script.write_text(f"#!/bin/sh\necho ran >> '{marker.as_posix()}'\n",
                      encoding="utf-8", newline="\n")
    script.chmod(0o755)
    return script.as_posix()


def _pin(tmp_path: Path) -> None:
    hooks = tmp_path / "no-hooks"
    hooks.mkdir(exist_ok=True)
    commit_broker._pin(commit_broker.Grant(Path("git"), hooks, hooks), hooks)


def test_the_repository_s_fsmonitor_does_not_run(repo, tmp_path):
    work, marker = repo
    gitlib.git(["config", "core.fsmonitor", _program(tmp_path, marker)], cwd=work)
    gitlib.git(["status", "--porcelain"], cwd=work)
    assert marker.exists()  # the control
    marker.unlink()
    _pin(tmp_path)
    gitlib.git(["status", "--porcelain"], cwd=work)
    assert not marker.exists()


def test_the_repository_s_external_diff_does_not_run(repo, tmp_path):
    work, marker = repo
    gitlib.git(["config", "diff.external", _program(tmp_path, marker)], cwd=work)
    gitlib.git(["diff"], cwd=work)
    assert marker.exists()  # the control
    marker.unlink()
    _pin(tmp_path)
    # `diff.external=` does not switch the external diff off: git tries to
    # start the empty program and dies (exit 128, measured 2026-11-18, Git
    # for Windows 2.x) — closed, not open, which is what is pinned here.
    gitlib.git(["diff"], cwd=work, check=False)
    assert not marker.exists()


def test_the_repository_s_editor_is_not_the_editor(repo, tmp_path):
    """`git var GIT_EDITOR` is what a commit that opens an editor starts."""
    work, marker = repo
    planted = _program(tmp_path, marker)
    gitlib.git(["config", "core.editor", planted], cwd=work)
    assert gitlib.git(["var", "GIT_EDITOR"], cwd=work).stdout.strip() == planted
    _pin(tmp_path)
    assert gitlib.git(["var", "GIT_EDITOR"], cwd=work).stdout.strip() == "true"


def test_an_inherited_git_dir_does_not_redirect_the_pinned_git(repo, tmp_path, monkeypatch):
    work, _ = repo
    other = tmp_path / "other"
    other.mkdir()
    gitlib.git(["init", "--quiet"], cwd=other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    here = Path(gitlib.git(["rev-parse", "--absolute-git-dir"], cwd=work).stdout.strip())
    assert here == (other / ".git").resolve()  # the control
    _pin(tmp_path)
    here = Path(gitlib.git(["rev-parse", "--absolute-git-dir"], cwd=work).stdout.strip())
    assert here == (work / ".git").resolve()
