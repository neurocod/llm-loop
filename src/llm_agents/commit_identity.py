"""Bind a worker's commit to the checkout approved by its wrapper.

The app-server approval reply cannot replace argv or set a per-command env.
Instead it inherits the location of a private approval store in the slot's
Git admin directory. The judge writes there before accepting; commit.py reads
the exact invocation's resolved cwd and identity there before any Git write.
The store is outside the worker's writable roots and removed with the server.

`write_approval` is the judge's half (`llm_agents.codex.command_commit`),
`approved_invocation` the commit script's: a host's commit command calls it
before its first git write and refuses on a ValueError.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional


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
    result = subprocess.run(
        ["git", "-C", str(workspace), "rev-parse", "--absolute-git-dir"],
        capture_output=True, text=True, encoding="utf-8",
    )
    admin = Path(result.stdout.strip()).resolve()
    # The inherited workspace must contain this invocation, and only its
    # linked-worktree admin directory can hold approvals outside worker writes.
    if (result.returncode != 0 or not Path.cwd().resolve().is_relative_to(workspace)
            or admin.parent.name != "worktrees" or admin.parent.parent.name != ".git"
            or directory == admin or not directory.is_relative_to(admin)):
        raise ValueError("commit.py: approval store is not under this slot's Git admin directory")
    path = ticket_path(directory, Path.cwd(), args)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("commit.py: no readable approval for this invocation") from exc
