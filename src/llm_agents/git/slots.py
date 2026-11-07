"""The worktree pool's naming rule, in one place for the pool and its judges.

A slot is a linked worktree that sits BESIDE the main checkout as
`<main>-wtN` on branch `wtN` (`worktree.py`). The Codex bridge's commit judge
(`llm_agents.codex.command_commit`) grants a worker's commit only inside such
a slot, so both read the rule here: a pool that renamed its slots alone would
leave every worker refused, or worse, a judge that kept an older rule.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Between the main checkout's directory name and the slot number.
SLOT_INFIX = "-wt"
# A slot's branch and short name: `wt1`, `wt2`, ...
SLOT_NAME_RE = re.compile(r"^wt(\d+)$")


def slot_name(number: int) -> str:
    return f"wt{number}"


def slot_dir_name(main_name: str, number: int) -> str:
    """The directory of slot `number` beside a main checkout named `main_name`."""
    return f"{main_name}{SLOT_INFIX}{number}"


def slot_number(main_name: str, dir_name: str) -> Optional[int]:
    """The number of the slot directory `dir_name`, or None when it is none of
    `main_name`'s slots. Letter case is folded: Windows paths come back from
    git in either case."""
    match = re.fullmatch(rf"{re.escape(main_name)}{SLOT_INFIX}(\d+)", dir_name,
                         re.IGNORECASE)
    return int(match.group(1)) if match else None


def git_admin(workspace: Path) -> Optional[Path]:
    """The resolved admin directory named by a checkout's .git file."""
    marker = workspace / ".git"
    try:
        first = marker.read_text(encoding="utf-8").strip().splitlines()[0]
    except (OSError, ValueError, IndexError):
        return None
    key, _, value = first.partition(":")
    if key.strip() != "gitdir" or not value.strip():
        return None
    return (workspace / value.strip()).resolve()


def pool_slot(workspace: Path) -> bool:
    """A linked `<main>-wtN` worktree, not the main tree or a submodule.

    Read off the `.git` file alone, without running git: a linked worktree's
    admin directory is `<main>/.git/worktrees/<name>`, a submodule's lies
    under `modules/`. The worker identity stays the slot even when `-C`
    selects its submodule.
    """
    admin = git_admin(workspace)
    if admin is None:
        return False
    if (admin.parent.name.casefold() != "worktrees"
            or admin.parent.parent.name.casefold() != ".git"):
        return False
    main = admin.parent.parent.parent.name
    return slot_number(main, workspace.name) is not None
