"""Bind a worker's commit to the checkout approved by its wrapper.

The app-server approval reply cannot replace argv or set a per-command env.
Instead it inherits the location of a private approval store in the slot's
Git admin directory. The judge writes there before accepting; commit.py reads
the exact invocation's resolved cwd and identity there before any Git write.
The store is outside the worker's writable roots and removed with the server.

`write_approval` is the judge's half (`llm_agents.codex.command_commit`),
`approved_invocation` the commit script's (`llm_agents.git.commit`, which
calls it before its first git write and refuses on a ValueError). Both halves
live with the kit's commit script, which is the `commit.py` its messages name.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from ..gitlib import Repository
from .slots import registered_admin


ENV = "CODEX_COMMIT_APPROVAL_STORE"


def ticket_path(directory: Path, cwd: Path, args: list[str]) -> Path:
    key = json.dumps([os.path.normcase(str(cwd)), args], ensure_ascii=True)
    return directory / (hashlib.sha256(key.encode()).hexdigest() + ".json")


def write_approval(directory: Path, cwd: Path, args: list[str], identity: dict) -> None:
    ticket_path(directory, cwd, args).write_text(json.dumps(identity), encoding="utf-8")


def approved_invocation(args: list[str]) -> Optional[dict]:
    setting = os.environ.get(ENV)
    if not setting:
        return None
    config = json.loads(setting)
    workspace = Path(config["workspace"]).resolve()
    directory = Path(config["directory"]).resolve()
    # The admin directory the MAIN clone registers for the slot, never the
    # one the slot's `.git` file names: that file and a discovered
    # repository's config are the worker's to plant (`gitlib.Repository`).
    admin, why = registered_admin(workspace)
    # The inherited workspace must contain this invocation, and only its
    # linked-worktree admin directory can hold approvals outside worker writes.
    if (admin is None or not Path.cwd().resolve().is_relative_to(workspace)
            or directory == admin or not directory.is_relative_to(admin)):
        raise ValueError(f"commit.py: approval store is not under this slot's Git "
                         f"admin directory{f' ({why})' if why else ''}")
    path = ticket_path(directory, Path.cwd(), args)
    try:
        identity = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("commit.py: no readable approval for this invocation") from exc
    # The judge wrote both; a ticket naming a repository outside the slot's
    # own admin tree is not one it wrote.
    ticket_admin = Path(identity["admin"])
    if ticket_admin != admin and not ticket_admin.is_relative_to(admin / "modules"):
        raise ValueError("commit.py: the approval names a repository outside this slot")
    return identity


def bound_repository(identity: dict) -> Repository:
    """The repository an approved invocation commits in: the ticket's
    admin directory and work tree, named to git explicitly."""
    return Repository(Path(identity["admin"]), Path(identity["root"]))
