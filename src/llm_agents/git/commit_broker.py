"""The commit broker: a sandboxed worker's one way to commit in its pool slot.

A linked worktree keeps its index in the main checkout's `.git`, outside the
writable roots of a sandboxed agent, so its commit has to run outside the
sandbox — with the user's full rights. The commit script itself
(`llm_agents.git.commit`) is the kit's; what a plain `python
tools/git/commit.py` from the worker's tree would also run is the worker's
own: that script and the kit under the slot's `src/` (it may edit both
before asking), the hooks under `core.hooksPath` (MultiSkill's is the
tracked `.githooks/` of the tree the worker writes; measured 2026-11-18:
pre-commit, prepare-commit-msg, commit-msg, post-commit, post-index-change
and reference-transaction all ran from there), and `git` resolved by name.

So the bridge grants THIS file, of the BRIDGE'S OWN checkout (outside every
writable root; started as `codex.command_brokers` requires), and hands it
through `ENV` what it must not take from the worker (`Grant`). Run by path
under `-I`, no package is on `sys.path`: the file puts the `src/` folder it
lies in first — the bridge's, since only its literal path is granted — and
runs the kit's `commit.main` with:

- git pinned for the whole process (`gitlib.pin`): the absolute git the
  bridge resolved (`rebase.trusted_git`), `_FORCED` over the repository's
  own configuration, no `GIT_*` from the environment;
- the hooks the grant names: the host's folder in its MAIN checkout
  (`HostPolicy.commit_hooks`: the house rules a commit must pass) for the
  slot's own tree, an empty folder for a commit inside a submodule (which
  runs no hooks of its own in a pool slot either);
- the host's `CommitHost` from its main checkout (`HostPolicy.commit_host`),
  never the slot's copy of it;
- an approval ticket (`commit_identity`) for exactly this invocation, which
  the bridge's judge wrote when it granted the command: no ticket, no
  commit.

Residual: the slot's `.git` FILE is the worker's to rewrite, and the kit's
read-only questions (status, ls-files, diff --cached) discover the
repository through it. Its writes do not: `run_indexing` passes the
approved `--git-dir`/`--work-tree`, and `approved_identity_problems`
refuses a slot whose discovery no longer names them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# The bridge's facts, as JSON: {"git", "hooks", "no_hooks", "host"}.
ENV = "LLM_AGENTS_COMMIT_BROKER"

# Configuration forced on every git of the broker, after (and so over) the
# repository's own: the grant's hooks folder (`hooks` is substituted), and
# the settings that would start a program of the repository's choosing
# during a commit — an fsmonitor, an external diff, an editor.
_FORCED = ("core.hooksPath={hooks}", "core.fsmonitor=false", "diff.external=",
           "core.editor=true")


@dataclass(frozen=True)
class Grant:
    """The bridge's facts for one worker: the absolute git, the hooks folder
    for a commit in the slot's own tree, an empty one for a submodule's, and
    the host's commit rules as `<absolute file>:<attribute>` ("" for none)."""
    git: Path
    hooks: Path
    no_hooks: Path
    host: str = ""

    def to_env(self) -> str:
        return json.dumps({"git": str(self.git), "hooks": str(self.hooks),
                           "no_hooks": str(self.no_hooks), "host": self.host})

    @staticmethod
    def from_env(text: str) -> "Grant":
        data = json.loads(text)
        return Grant(Path(data["git"]), Path(data["hooks"]),
                     Path(data["no_hooks"]), data.get("host", ""))


def _pin(grant: Grant, hooks: Path) -> None:
    from llm_agents import gitlib
    gitlib.pin(grant.git, tuple(setting.format(hooks=hooks) for setting in _FORCED))


def _load_host(spec: str):
    """The host's `CommitHost` named by `spec` (`<file>:<attribute>`), or the
    kit's `EMPTY` for ""."""
    from llm_agents.git import commit
    if not spec:
        return commit.EMPTY
    file, _, attribute = spec.rpartition(":")
    module_spec = importlib.util.spec_from_file_location("_commit_broker_host", file)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load the host's commit rules from {file}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, attribute)


def main(argv: Optional[list[str]] = None) -> int:
    """Commit as `commit.py <argv>` would, under the bridge's grant."""
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    argv = sys.argv[1:]
    text = os.environ.get(ENV)
    if not text:
        print(f"commit broker: {ENV} is not set; this broker runs only as a "
              f"command the Codex bridge granted", file=sys.stderr)
        return 2
    try:
        grant = Grant.from_env(text)
    except (ValueError, KeyError, TypeError) as error:
        print(f"commit broker: unreadable {ENV}: {error}", file=sys.stderr)
        return 2
    # No hooks until the ticket says whose tree this is; rev-parse runs none.
    _pin(grant, grant.no_hooks)
    from llm_agents.git import commit
    from llm_agents.git.commit_identity import approved_invocation
    try:
        expected = approved_invocation(argv)
    except (OSError, ValueError, KeyError) as error:
        print(f"commit broker: {error}", file=sys.stderr)
        return 1
    if expected is None:
        print("commit broker: no approval store: this broker commits only what "
              "the bridge's judge approved", file=sys.stderr)
        return 2
    _pin(grant, grant.no_hooks if expected.get("owner") else grant.hooks)
    return commit.main(_load_host(grant.host))


if __name__ == "__main__":
    # `-I` put no script folder on sys.path: this file's own src/ goes first.
    sys.path.insert(0, str(Path(__file__).parents[2]))
    sys.exit(main())
