"""examples/host_rules.py is what a downstream project copies, so it must keep
importing and its values must stay the kit's own types: a field renamed in
`PoolHost` or `CommitHost` fails here instead of in the first adopter's copy."""

import argparse
import importlib.util
from pathlib import Path

import pytest

from llm_agents.git import commit, worktree
from llm_agents.host_policy import HostPolicy

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "host_rules.py"


@pytest.fixture(scope="module")
def rules():
    spec = importlib.util.spec_from_file_location("host_rules_example", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_example_names_a_host_for_each_main(rules):
    assert isinstance(rules.COMMIT, commit.CommitHost)
    assert isinstance(rules.POOL, worktree.PoolHost)
    assert isinstance(rules.POLICY, HostPolicy)


def test_the_pool_spells_out_the_neutral_defaults(rules):
    neutral = worktree.PoolHost()
    assert (rules.POOL.main_branch, rules.POOL.remote, rules.POOL.push_by_default) \
        == (neutral.main_branch, neutral.remote, neutral.push_by_default)


def test_the_pool_parser_builds_with_the_example_host(rules, monkeypatch):
    """`build_parser` reads the installed host: the share command appears
    because the example shares a directory, and `--push` is listed (not
    suppressed) because the example does not push by default."""
    monkeypatch.setattr(worktree, "HOST", rules.POOL)
    parser = worktree.build_parser()
    assert parser.parse_args([rules.POOL.share_command]).func is worktree.cmd_share
    assert parser.parse_args(["land", "wt1", "--push"]).push
    land_help = next(action for action in parser._subparsers._actions
                     if isinstance(action, argparse._SubParsersAction)
                     ).choices["land"].format_help()
    assert "--push" in land_help and "--no-push" not in land_help
    assert f"push {rules.POOL.main_branch} to {rules.POOL.remote}" in parser.format_help()


def test_the_commit_parser_carries_the_example_help(rules):
    assert "pool slot" in commit.build_parser(rules.COMMIT).format_help()


def test_the_example_check_refuses_generated_output_only(rules):
    assert rules.no_generated_output(["src/a.py", "docs/b.md"]) == []
    assert rules.no_generated_output(["build/a.o"]) == [
        "build/a.o is generated output; commit its source instead"]
    assert rules.no_generated_output(["build\\x\\a.o"])
