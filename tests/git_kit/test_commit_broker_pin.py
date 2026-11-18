"""The pinned git (`gitlib.Pinned` through `gitlib.pin`, as the commit
broker sets it): each forced setting, the `GIT_*` strip and the binding to
a named repository, by its effect.

A repository the worker writes may name in its own config a program git
starts on a commit's way (an fsmonitor, an external diff, a clean filter,
an editor), a `.git` the worker writes may point discovery at such a
repository, and the environment may redirect git (`GIT_DIR`). Each test
plants one of them, runs git through the pin, and asserts the planted
program did not run or the redirect did not take; its control shows that
the same fixture does run it with git unpinned."""

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
    script.write_text(f"#!/bin/sh\necho ran >> '{marker.as_posix()}'\ncat\n",
                      encoding="utf-8", newline="\n")
    script.chmod(0o755)
    return script.as_posix()


def _pin(tmp_path: Path, repository=None) -> None:
    hooks = tmp_path / "no-hooks"
    hooks.mkdir(exist_ok=True)
    commit_broker._pin(commit_broker.Grant(Path("git"), hooks, hooks), hooks, repository)


def _hijacked(work: Path, tmp_path: Path, marker: Path) -> gitlib.Repository:
    """`work`'s own repository, moved where the worker cannot write (the
    returned binding), and in its place a `.git` file pointing discovery at
    a worker repository whose config names `marker`'s program as fsmonitor,
    external diff and clean filter for every file (`.gitattributes`)."""
    trusted = tmp_path / "trusted.git"
    (work / ".git").rename(trusted)
    planted = tmp_path / "planted.git"
    gitlib.git(["init", "--quiet", "--bare", str(planted)], cwd=tmp_path)
    program = _program(tmp_path, marker)
    for key, value in (("core.bare", "false"), ("core.fsmonitor", program),
                       ("diff.external", program), ("filter.evil.clean", program)):
        gitlib.git(["config", "--file", str(planted / "config"), key, value], cwd=tmp_path)
    (work / ".git").write_text(f"gitdir: {planted}\n", encoding="utf-8")
    (work / ".gitattributes").write_text("* filter=evil\n", encoding="utf-8")
    return gitlib.Repository(trusted, work)


def test_the_repository_s_fsmonitor_does_not_run(repo, tmp_path):
    work, marker = repo
    gitlib.git(["config", "core.fsmonitor", _program(tmp_path, marker)], cwd=work)
    gitlib.git(["status", "--porcelain"], cwd=work)
    assert marker.exists()  # the control
    marker.unlink()
    _pin(tmp_path)
    gitlib.git(["status", "--porcelain"], cwd=work)
    assert not marker.exists()


@pytest.mark.parametrize("question", [
    ["status", "--porcelain"],
    ["ls-files", "-co", "--exclude-standard", "-z"],
    ["diff"],
    ["add", "a.txt"],
])
def test_a_bound_git_never_discovers_the_worker_s_git_file(repo, tmp_path, question):
    """Item 2432, review A/C: a `.git` under a writable root (a submodule's)
    is the worker's to repoint; bound, git reads the trusted repository."""
    work, marker = repo
    trusted = _hijacked(work, tmp_path, marker)
    gitlib.git(question, cwd=work, check=False)
    assert marker.exists()  # the control: discovery runs the planted programs
    marker.unlink()
    _pin(tmp_path, trusted)
    done = gitlib.git(question, cwd=work, check=False)
    assert done.returncode == 0, done.stderr
    assert not marker.exists()
    where = gitlib.git(["rev-parse", "--absolute-git-dir"], cwd=work).stdout.strip()
    assert Path(where).resolve() == trusted.git_dir.resolve()


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
