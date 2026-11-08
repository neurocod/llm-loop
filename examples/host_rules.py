"""
Example host rules: what a downstream project hands the git kit and the agent bridges.

`llm_agents.git` and the bridges know no project. Everything that names one —
the sentinel file never committed, the submodules a new slot gets, the main
branch and the remote, the paths no agent writes, the gate hook a review
reuses — is a value of a module like this one, and the project's own entry
points (four small scripts, sketched at the bottom) hand it to each `main`.
Nothing is read from here on its own: a kit script run without a host runs
with the empty host and says so on stderr.

Where to put it: `tools/host_rules.py`, with the submodule at
`tools/llm-loop` — the layout every path below assumes, and the one
`tests/test_examples_host_rules.py` imports a copy under. The entry scripts
beside it import it by name (`from host_rules import worktree_main`), since a
script's own directory is on `sys.path`; one run as a package directory
(`python tools/codex_agent`) puts `tools/` there first. Kept elsewhere, the
file needs `ROOT` changed and nothing else: the submodule path and every
project path derive from it.

Keep the fields you need and drop the rest. Every field's default is the
empty host's answer; the fields and their meaning are documented on
`PoolHost`, `CommitHost`, `HostPolicy` and `ReviewProfile` themselves, not
repeated here.
"""

import subprocess
import sys
from pathlib import Path

# The checkout these rules belong to: this file is tools/host_rules.py.
ROOT = Path(__file__).resolve().parents[1]

# Make the vendored submodule importable. First, because the imports below
# resolve against it; derived from ROOT, so a moved copy changes one line.
sys.path.insert(0, str(ROOT / "tools" / "llm-loop" / "src"))

from llm_agents.git.commit import CommitHost
from llm_agents.git.worktree import PoolHost, SharedDir
from llm_agents.host_policy import GIT_STORE, HostPolicy, ReviewProfile


# --- the commit helper: python -m llm_agents.git.commit ---------------------

def no_generated_output(paths: list) -> list:
    """A `CommitHost.checks` entry: called with the pathspecs of one commit,
    returns the problems that refuse it ([] lets it through)."""
    return [f"{path} is generated output; commit its source instead"
            for path in paths if path.replace("\\", "/").startswith("build/")]


COMMIT = CommitHost(
    # The loop's stop sentinel is created and removed by runs, never history.
    blocked_names=frozenset({"stop"}),
    checks=(no_generated_output,),
    cwd_help="run in this worktree (paths are then relative to it); the way to "
             "commit inside a pool slot (python tools/worktree.py) from elsewhere",
)


# --- the worktree pool: python -m llm_agents.git.worktree -------------------

def install_dependencies(slot: Path, admin: Path) -> None:
    """`PoolHost.install`: the project's dependency step for one slot. `admin`
    is the slot's git admin directory (`.git/worktrees/<slot>`), a place for a
    stamp of what was installed last so an unchanged lock file skips the step."""
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "-r", "requirements.txt"], cwd=slot, check=True)


POOL = PoolHost(
    command="python tools/worktree.py",        # how messages spell the command
    anchor=ROOT,                               # a checkout the main one is found from
    default_submodules=("tools/llm-loop",),    # what `new` initializes
    shared=(SharedDir("assets/generated", what="generated assets",
                      items="generated files"),),   # filled by the main checkout only
    install=install_dependencies,
    new_notes=("note: the dev database is shared with the main checkout",),
    # The neutral defaults, spelled out: slots are cut from and land into
    # `main`, pushes go to `origin`, and nothing pushes unless asked (--push).
    main_branch="main",
    remote="origin",
    push_by_default=False,
)


# --- the agent bridges: python -m llm_agents.codex / llm_agents.claude ------

POLICY = HostPolicy(
    # Path runs (whole components) no agent writes, sandboxed or not.
    protected_parts=GIT_STORE + (("docs", "decisions"),),
    worker_lines=("Run the gates from the repository root.",),
    commit_script="tools/commit.py",
    pool_script="tools/worktree.py",
    # What the Claude bridge's `review` profile reuses, and refuses to start
    # without: the ask-user-gate hook the project already runs. `settings` is
    # the project settings file whose PreToolUse entry names the compiled gate
    # (`<gate>/hooks/ask_user_gate.exe`, built by `<gate>/cpp/build.py`); the
    # vendored plugin directory is the gate here. `submodule` lets the
    # reviewer read this one submodule's commits with `git -C` templates.
    review=ReviewProfile(
        settings=".claude/settings.json",
        gate="tools/llm-loop/claude-plugin/ask-user-gate",
        submodule="tools/llm-loop",
    ),
)
# A worker's command allowlist (`HostPolicy.commands`) is the one table worth
# a module of its own; the core's own test host,
# tests/codex_agent/core_policy.py, binds every judge the core ships to a
# small table and is the fuller example.


# --- entry points -----------------------------------------------------------
# In a project each is a script of its own (tools/worktree.py, tools/commit.py,
# tools/codex_agent/__main__.py, tools/claude_agent/__main__.py) ending in
# `sys.exit(<function>())`; they are functions here so this file imports whole.
# `entry` is the path the bridge's printed hints name (`python tools/codex_agent
# logs <name>`).

def worktree_main() -> int:
    from llm_agents.git import worktree
    return worktree.main(POOL)


def commit_main() -> int:
    from llm_agents.git import commit
    return commit.main(COMMIT)


def codex_agent_main() -> int:
    from llm_agents.codex import cli
    return cli.main(policy=POLICY, root=ROOT, entry=ROOT / "tools" / "codex_agent")


def claude_agent_main() -> int:
    from llm_agents.claude import cli
    return cli.main(policy=POLICY, root=ROOT, entry=ROOT / "tools" / "claude_agent")
