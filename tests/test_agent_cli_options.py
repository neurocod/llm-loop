"""Agent utilities accept documented option names, never implicit prefixes."""

import importlib
import sys

import pytest

from llm_agents.git import commit, worktree


@pytest.mark.parametrize("module, argv", [
    ("claude.find_session", ["text", "--proj", "repo"]),
    ("git.branch_overlap", ["ours", "theirs", "--jso"]),
    ("git.union_conflicts", ["file", "--dry-ru"]),
    ("git.check_gitlinks", ["--hel"]),
])
def test_utility_refuses_option_prefix(module, argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module, *argv])
    with pytest.raises(SystemExit) as error:
        importlib.import_module("llm_agents." + module).main()
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_commit_refuses_prefix_and_keeps_documented_spellings(capsys):
    parser = commit.build_parser(commit.HOST)
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--fil", "message.txt", "src"])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert parser.parse_args(["-F", "message.txt", "-C", "repo", "src"]).paths == ["src"]


@pytest.mark.parametrize("command, abbreviated, full, value", [
    ("new", "--no-p", "--no-push", []),
    ("new", "--no-i", "--no-install", []),
    ("new", "--sub", "--submodules", ["all"]),
    ("land", "--pus", "--push", []),
    ("equalize", "--di", "--dir", ["wt1"]),
    ("next", "--di", "--dir", ["wt1"]),
    ("release", "--di", "--dir", ["wt1"]),
    ("drop", "--di", "--dir", ["wt1"]),
])
def test_worktree_refuses_prefix_and_accepts_full_option(
        command, abbreviated, full, value, capsys):
    parser = worktree.build_parser()
    option = [abbreviated + "=" + value[0]] if value else [abbreviated]
    with pytest.raises(SystemExit) as error:
        parser.parse_args([command, *option])
    assert error.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err
    assert parser.parse_args([command, full, *value]).command == command
