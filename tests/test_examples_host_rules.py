"""examples/host_rules.py is what a downstream project copies, so it must keep
importing and its values must stay the kit's own types: a field renamed in
`PoolHost` or `CommitHost` fails here instead of in the first adopter's copy.

Two ways in. The in-process import below gets `llm_agents` from pytest's own
path and checks the VALUES; `downstream` copies the example into the layout
its docstring prescribes (`tools/host_rules.py` beside `tools/llm-loop/src`)
and imports it there in a child interpreter with no PYTHONPATH, which is the
only way to check the `sys.path` line itself — in here it is dead code.
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from llm_agents.git import commit, worktree
from llm_agents.host_policy import HostPolicy, ReviewProfile

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "host_rules.py"
ENTRY_POINTS = ("worktree_main", "commit_main", "codex_agent_main",
                "claude_agent_main")


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
    assert isinstance(rules.POLICY.review, ReviewProfile), \
        "without `review` the Claude bridge's review profile refuses to start"


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


# --- the layout the docstring prescribes, in a child interpreter --------------

# Run with the downstream root as argv[1]. Imports the copy, then runs each
# entry function under `--help`: argparse exits 0 after printing usage, which
# proves the function reaches its `main` with the host installed and that the
# host's values survive that `main`'s parser. Everything printed by the mains
# is swallowed so the JSON report is the child's only stdout.
PROBE = r"""
import contextlib, importlib.util, io, json, sys
root = sys.argv[1]
spec = importlib.util.spec_from_file_location("host_rules", root + "/tools/host_rules.py")
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)
import llm_agents
report = {"llm_agents": llm_agents.__file__, "root": str(rules.ROOT),
          "review": vars(rules.POLICY.review), "help": {}}
for name in %r:
    sys.argv = [name, "--help"]
    usage = io.StringIO()
    try:
        with contextlib.redirect_stdout(usage), contextlib.redirect_stderr(io.StringIO()):
            getattr(rules, name)()
        code = "returned"
    except SystemExit as exc:
        code = exc.code
    report["help"][name] = [code, usage.getvalue()[:5]]
print(json.dumps(report))
""" % (ENTRY_POINTS,)


@pytest.fixture(scope="module")
def downstream(tmp_path_factory):
    """A project laid out as the example's docstring says, with a copy of this
    checkout's `src/` where the submodule would be."""
    root = tmp_path_factory.mktemp("downstream")
    (root / "tools").mkdir()
    shutil.copy(EXAMPLE, root / "tools" / "host_rules.py")
    shutil.copytree(REPO / "src", root / "tools" / "llm-loop" / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHONPATH", "PYTHONSAFEPATH")}
    # -E -s: no PYTHONPATH, no user site — the copy is reachable only through
    # the example's own sys.path line. cwd elsewhere, so `''` on sys.path is
    # not the project either.
    completed = subprocess.run(
        [sys.executable, "-E", "-s", "-c", PROBE, str(root)],
        cwd=str(root.parent), env=env, capture_output=True, text=True,
        encoding="utf-8", timeout=120)
    assert completed.returncode == 0, completed.stderr
    return root, json.loads(completed.stdout.strip().splitlines()[-1])


def test_the_copy_imports_the_submodule_beside_it(downstream):
    root, report = downstream
    imported = Path(report["llm_agents"]).resolve()
    assert (root / "tools" / "llm-loop" / "src").resolve() in imported.parents, \
        f"the sys.path line did not resolve to the vendored copy: {imported}"
    assert Path(report["root"]).resolve() == root.resolve()


def test_the_copy_names_the_review_gate_under_its_root(downstream):
    root, report = downstream
    review = report["review"]
    assert review["gate"] == "tools/llm-loop/claude-plugin/ask-user-gate"
    assert review["submodule"] == "tools/llm-loop"
    assert review["settings"]


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_each_entry_point_reaches_its_main(downstream, name):
    _root, report = downstream
    code, usage = report["help"][name]
    assert (code, usage) == (0, "usage"), (
        f"{name}() under --help: exit {code!r}, printed {usage!r}")
