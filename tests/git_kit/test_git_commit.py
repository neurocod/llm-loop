"""The git kit's explicit-path commit helper, run as `python -m llm_agents.git.commit`."""

import subprocess
import sys
from pathlib import Path

import pytest

from kit_support import git as _git, kit_env, repo as _repo, run_kit
from llm_agents.git import commit
from llm_agents.git.commit_identity import ENV


@pytest.fixture(autouse=True)
def temporary_repository_environment(monkeypatch):
    """These isolated repositories do not belong to the invoking worker."""
    monkeypatch.delenv(ENV, raising=False)


def _run_commit(repo: Path, message_file: Path, *paths: str) -> subprocess.CompletedProcess:
    return run_kit("commit", "-F", str(message_file), *(paths or ("work.txt",)), cwd=repo)


def test_file_message_is_removed_and_reported_after_success(tmp_path):
    repo = _repo(tmp_path)
    (repo / "work.txt").write_text("first\n", encoding="utf-8")
    message_file = repo / "message.txt"
    message_file.write_text("Commit the work\n", encoding="utf-8")

    result = _run_commit(repo, message_file)

    assert result.returncode == 0, result.stderr
    assert not message_file.exists()
    assert f"Removed commit message file: {message_file}" in result.stdout
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "Commit the work"


def test_the_bare_script_says_it_runs_without_host_rules(tmp_path):
    repo = _repo(tmp_path)
    (repo / "work.txt").write_text("first\n", encoding="utf-8")
    result = run_kit("commit", "-m", "Work", "work.txt", cwd=repo)
    assert result.returncode == 0, result.stderr
    assert "no host rules given" in result.stderr


def test_file_message_is_preserved_when_commit_fails(tmp_path):
    repo = _repo(tmp_path)
    (repo / "work.txt").write_text("already committed\n", encoding="utf-8")
    _git(repo, "add", "work.txt")
    _git(repo, "commit", "--quiet", "-m", "Initial commit")
    message_file = repo / "message.txt"
    message_file.write_text("Nothing changed\n", encoding="utf-8")

    result = _run_commit(repo, message_file)

    assert result.returncode != 0
    assert message_file.read_text(encoding="utf-8") == "Nothing changed\n"
    assert "Removed commit message file:" not in result.stdout


def test_replaced_message_file_is_preserved_after_success(tmp_path):
    repo = _repo(tmp_path)
    (repo / "work.txt").write_text("first\n", encoding="utf-8")
    message_file = repo / "message.txt"
    message_file.write_text("First agent\n", encoding="utf-8")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "rm -f message.txt\n"
        "printf 'Second agent\\n' > message.txt\n",
        encoding="utf-8",
    )
    hook.chmod(0o755)

    result = _run_commit(repo, message_file)

    assert result.returncode == 0, result.stderr
    assert not message_file.exists()
    quarantine = next(repo.glob(".message.txt.commitpy-*.tmp"))
    assert quarantine.read_text(encoding="utf-8") == "Second agent\n"
    assert f"moved without deletion to {quarantine}" in result.stderr
    assert "Removed commit message file:" not in result.stdout
    assert _git(repo, "log", "-1", "--format=%s").stdout.strip() == "First agent"


def test_message_file_cannot_be_an_exact_commit_path(tmp_path):
    repo = _repo(tmp_path)
    message_file = repo / "message.txt"
    message_file.write_text("Do not commit me\n", encoding="utf-8")

    result = _run_commit(repo, message_file, "message.txt")

    assert result.returncode != 0
    assert message_file.exists()
    assert "covered by the commit pathspecs" in result.stderr
    assert _git(repo, "status", "--short").stdout == "?? message.txt\n"


def test_message_file_cannot_be_inside_an_absolute_committed_directory(tmp_path):
    repo = _repo(tmp_path)
    folder = repo / "product"
    folder.mkdir()
    (folder / "work.txt").write_text("first\n", encoding="utf-8")
    message_file = folder / "message.txt"
    message_file.write_text("Do not commit me\n", encoding="utf-8")

    result = _run_commit(repo, message_file, str(folder))

    assert result.returncode != 0
    assert message_file.exists()
    assert "covered by the commit pathspecs" in result.stderr
    assert _git(repo, "status", "--short").stdout == "?? product/\n"


def test_tracked_message_file_cannot_be_committed(tmp_path):
    repo = _repo(tmp_path)
    message_file = repo / "message.txt"
    message_file.write_text("Initial message\n", encoding="utf-8")
    _git(repo, "add", "message.txt")
    _git(repo, "commit", "--quiet", "-m", "Track the message file")
    message_file.write_text("Updated message\n", encoding="utf-8")
    (repo / "work.txt").write_text("first\n", encoding="utf-8")

    result = _run_commit(repo, message_file, "message.txt", "work.txt")

    assert result.returncode != 0
    assert message_file.read_text(encoding="utf-8") == "Updated message\n"
    assert "covered by the commit pathspecs" in result.stderr


def test_a_tracked_message_file_outside_the_pathspecs_is_never_deleted(tmp_path):
    """commit.py deletes its -F source after the commit: a tracked one
    (`-F CLAUDE.md work.txt`) would lose the file and its uncommitted edits.
    Refused before anything is staged, in the repository committed in and in
    another one (`-C <slot> -F <main tree file>`)."""
    repo = _repo(tmp_path)
    notes = repo / "NOTES.md"
    notes.write_text("committed\n", encoding="utf-8")
    _git(repo, "add", "NOTES.md")
    _git(repo, "commit", "--quiet", "-m", "Track the notes")
    notes.write_text("committed\nuncommitted edit\n", encoding="utf-8")
    (repo / "work.txt").write_text("first\n", encoding="utf-8")

    result = _run_commit(repo, notes, "work.txt")

    assert result.returncode != 0
    assert "is tracked by git" in result.stderr
    assert notes.read_text(encoding="utf-8") == "committed\nuncommitted edit\n"
    assert _git(repo, "diff", "--cached", "--name-only").stdout == ""

    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "--quiet")
    (other / "work.txt").write_text("first\n", encoding="utf-8")
    elsewhere = run_kit("commit", "-C", str(other), "-F", str(notes), "work.txt",
                        cwd=tmp_path)
    assert elsewhere.returncode != 0
    assert "is tracked by git" in elsewhere.stderr
    assert notes.exists()


def test_a_nested_repository_is_caught_from_a_subdirectory(tmp_path):
    """check_gitlinks reads the pathspecs from the directory they were typed
    in: from `sub/`, `nested` once matched nothing at the root and the stray
    repository went in as a gitlink unchecked."""
    repo = _repo(tmp_path)
    nested = repo / "sub" / "nested"
    nested.mkdir(parents=True)
    _git(nested, "init", "--quiet")
    (nested / "x.txt").write_text("x\n", encoding="utf-8")
    _git(nested, "add", "x.txt")
    _git(nested, "commit", "--quiet", "-m", "nested")
    message_file = tmp_path / "message.txt"
    message_file.write_text("Add a nested repository\n", encoding="utf-8")

    result = _run_commit(repo / "sub", message_file, "nested")

    assert result.returncode != 0
    assert "sub/nested: recorded as a submodule" in result.stderr
    assert _git(repo, "rev-list", "--all").stdout == ""


def test_a_denied_index_lock_is_reported_at_once(monkeypatch, capsys):
    """A sandboxed run may not create index.lock at all: no contention to wait
    out, so it fails on the first try and says why."""
    denied = subprocess.CompletedProcess(
        ["git", "add"], 128, "",
        "fatal: Unable to create 'C:/r/.git/worktrees/wt1/index.lock': "
        "Permission denied\n")
    calls = []
    monkeypatch.setattr(commit, "git", lambda args, **_: calls.append(args) or denied)
    monkeypatch.setattr(commit.time, "sleep", lambda _: pytest.fail("waited"))

    assert commit.run_indexing(["add", "--", "a"]).returncode == 128

    assert len(calls) == 1
    assert "not escalated / sandboxed?" in capsys.readouterr().err


def test_a_held_index_lock_is_waited_out(monkeypatch, capsys):
    held = subprocess.CompletedProcess(
        ["git", "add"], 128, "",
        "fatal: Unable to create 'C:/r/.git/index.lock': File exists.\n\n"
        "Another git process seems to be running in this repository\n")
    done = subprocess.CompletedProcess(["git", "add"], 0, "", "")
    answers = [held, held, done]
    monkeypatch.setattr(commit, "git", lambda args, **_: answers.pop(0))
    monkeypatch.setattr(commit.time, "sleep", lambda _: None)
    # The deadline is latched at import, maybe long before this test runs.
    monkeypatch.setattr(commit, "_LOCK_DEADLINE", commit.time.monotonic() + 60)

    assert commit.run_indexing(["add", "--", "a"]).returncode == 0

    assert answers == []
    assert "not escalated" not in capsys.readouterr().err


def test_ignored_message_inside_directory_is_still_rejected(tmp_path):
    repo = _repo(tmp_path)
    (repo / ".gitignore").write_text("product/message.txt\n", encoding="utf-8")
    folder = repo / "product"
    folder.mkdir()
    (folder / "work.dat").write_text("first\n", encoding="utf-8")
    message_file = folder / "message.txt"
    message_file.write_text("Commit the product\n", encoding="utf-8")

    result = _run_commit(repo, message_file, "product")

    assert result.returncode != 0
    assert message_file.read_text(encoding="utf-8") == "Commit the product\n"
    assert "covered by the commit pathspecs" in result.stderr
    assert _git(repo, "status", "--short").stdout == "?? .gitignore\n?? product/\n"


def test_message_alias_cannot_evade_directory_overlap(tmp_path):
    repo = _repo(tmp_path)
    product = repo / "product"
    product.mkdir()
    message_file = product / "message.txt"
    message_file.write_text("Do not commit me\n", encoding="utf-8")
    alias = repo / "product-alias"
    try:
        alias.symlink_to(product, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    result = _run_commit(repo, alias / "message.txt", "product")

    assert result.returncode != 0
    assert message_file.exists()
    assert "covered by the commit pathspecs" in result.stderr


def test_invalid_utf8_replacement_is_preserved_after_success(tmp_path):
    repo = _repo(tmp_path)
    (repo / "work.txt").write_text("first\n", encoding="utf-8")
    message_file = repo / "message.txt"
    message_file.write_text("First agent\n", encoding="utf-8")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nprintf '\\377' > message.txt\n", encoding="utf-8")
    hook.chmod(0o755)

    result = _run_commit(repo, message_file)

    assert result.returncode == 0, result.stderr
    assert not message_file.exists()
    quarantine = next(repo.glob(".message.txt.commitpy-*.tmp"))
    assert quarantine.read_bytes() == b"\xff"
    assert f"moved without deletion to {quarantine}" in result.stderr
    assert "Traceback" not in result.stderr


def test_replacement_created_after_quarantine_is_never_deleted(
    tmp_path, monkeypatch, capsys
):
    message_file = tmp_path / "message.txt"
    expected = b"First agent\n"
    message_file.write_bytes(expected)
    with message_file.open("rb") as handle:
        identity = commit.file_identity(handle)
    real_replace = commit.os.replace

    def replace_then_recreate(source, target):
        real_replace(source, target)
        Path(source).write_bytes(b"Second agent\n")

    monkeypatch.setattr(commit.os, "replace", replace_then_recreate)

    commit.remove_committed_message_file(message_file, expected, identity)

    assert message_file.read_bytes() == b"Second agent\n"
    output = capsys.readouterr()
    assert "another file now occupies its path" in output.err
    assert "Removed commit message file:" not in output.out


# --------------------------------------------------------------- host rules

def _main(monkeypatch, tmp_path, host, *argv):
    """`commit.main(host)` in-process with `argv`; the cwd is restored."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["commit.py", *argv])
    return commit.main(host)


def test_a_host_blocks_its_names_and_the_empty_host_none(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "stop").write_text("", encoding="utf-8")
    host = commit.CommitHost(blocked_names=frozenset({"stop"}))
    assert _main(monkeypatch, tmp_path, host, "-C", str(repo), "-m", "x", "stop") == 1
    assert "refusing to commit sentinel file(s): stop" in capsys.readouterr().err
    assert _git(repo, "rev-list", "--all").stdout == ""
    assert _main(monkeypatch, tmp_path, commit.EMPTY, "-C", str(repo), "-m", "x", "stop") == 0


def test_a_host_check_is_asked_before_staging_and_before_committing(
        tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "work.txt").write_text("first\n", encoding="utf-8")
    asked = []

    def check(paths):
        asked.append((list(paths), Path.cwd().resolve()))
        return ["held by the host"] if len(asked) == 2 else []

    host = commit.CommitHost(checks=(check,))
    assert _main(monkeypatch, tmp_path, host, "-C", str(repo), "-m", "x", "work.txt") == 1
    assert asked == [(["work.txt"], repo.resolve())] * 2
    err = capsys.readouterr().err
    assert "error: held by the host" in err and "the paths stay staged" in err
    assert _git(repo, "rev-list", "--all").stdout == ""


def test_the_cwd_help_is_the_host_s():
    host = commit.CommitHost(cwd_help="commit in a pool slot")
    assert "commit in a pool slot" in commit.build_parser(host).format_help()


def test_the_parser_takes_exactly_the_option_table():
    """`OPTION_FLAGS` is what a command-line judge allows; a parser option
    missing from it would be a flag the judge refuses or never sees."""
    parser = commit.build_parser(commit.EMPTY)
    spelled = {flag for action in parser._actions for flag in action.option_strings}
    assert spelled - {"-h", "--help"} == {
        flag for flags in commit.OPTION_FLAGS.values() for flag in flags}
