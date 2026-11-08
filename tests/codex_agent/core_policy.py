"""The core tests' own host: generic judges bound to generic tables.

Every core test runs under `CORE_POLICY` (`_core_host` in tests/conftest.py) unless it asks for
`EMPTY`; nothing here names a project. The tables are small on purpose: one
entry per generic judge the core ships (pytest, try_patch, an editing script,
the slot commit, npm/npx/node), so a judge's own rules are pinned here and
travel with the code.
"""

from functools import partial

from llm_agents.codex.command_arguments import _args_refusal
from llm_agents.codex.command_commit import _commit
from llm_agents.codex.command_dispatch import Shape, argv_refusal
from llm_agents.codex.command_node import NpmScript, _node, _npm, _npx
from llm_agents.codex.command_python import Entrypoint, _checks, _edits, _python, _try_patch_refusal
from llm_agents.host_policy import GIT_STORE, HostPolicy, Reserved, WorkerCommands

PYTHON_TABLE = "the core test entrypoints"
NPM_TABLE = "the core test npm scripts"

PYTHON_ENTRIES = {
    "bin/try_patch.py": Entrypoint("edits its --file(s), runs one worker command",
                                   partial(_try_patch_refusal, judge_argv=argv_refusal)),
    "bin/edit.py": Entrypoint("edits the files it names", _edits),
    "bin/read.py": Entrypoint("reads", _args_refusal),
    "bin/gen.py": Entrypoint("compares its generated output", _checks("--stdout")),
    "tools/git/commit.py": Entrypoint("commits a pool slot's named paths", _commit),
}
PYTHON_DECLINED = {"bin/agents/": "starts agents", "bin/*launch_*.py": "launches"}
NPM_SCRIPTS = {"test": NpmScript("the unit tests"), "check:": NpmScript("a checker")}

COMMANDS = WorkerCommands(
    shapes=(
        Shape(("tool",), "`tool <paths>`", _args_refusal),
        Shape(("npm",), "`npm test`, `npm run check:*`",
              partial(_npm, judge_argv=argv_refusal)),
        Shape(("npx",), "`npx vitest run …`", partial(_npx, judge_argv=argv_refusal)),
        Shape(("node",), "`node <script>`", partial(_node, judge_argv=argv_refusal)),
        Shape(("python", "python3", "py"), "`python -m pytest …`, `python <entry>`",
              partial(_python, entries=PYTHON_ENTRIES, declined=PYTHON_DECLINED,
                      table=PYTHON_TABLE)),
    ),
    shapes_name="the core test shapes",
    npm_scripts=NPM_SCRIPTS,
    npm_name=NPM_TABLE,
    reserved=Reserved(text=lambda record: "anything the core test host forgot"),
)

def core_main(argv):
    """`cli.main` the way a host entry point calls it, with this suite's
    host: a call without one runs under EMPTY (`host_policy.installed`)."""
    from pathlib import Path
    from llm_agents.codex import cli
    return cli.main(argv, policy=CORE_POLICY, root=Path.cwd())


CORE_POLICY = HostPolicy(
    commands=COMMANDS,
    protected_parts=GIT_STORE + (("vault",),),
    worker_lines=("Core test host line.",),
    commit_briefing=lambda where, model: (
        f"commit with tools/git/commit.py, the message file in {where}; "
        f"end it with `Co-Authored-By: Codex {model or '<your model>'}`."),
)
