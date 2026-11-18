"""The worktree pool's naming rule, in one place for the pool and its judges.

A slot is a linked worktree that sits BESIDE the main checkout as
`<main>-wtN` on branch `wtN` (`worktree.py`). The Codex bridge's commit judge
(`llm_agents.codex.command_commit`) grants a worker's commit only inside such
a slot, so both read the rule here: a pool that renamed its slots alone would
leave every worker refused, or worse, a judge that kept an older rule.
"""

from __future__ import annotations

import os
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


def _same(a: os.PathLike, b: os.PathLike) -> bool:
    """Equal spellings, case and separators folded, `..` dropped — nothing
    resolved: a link along a worker-written path must not make it equal."""
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def registered_admin(workspace: Path) -> tuple[Optional[Path], str]:
    """The admin directory git registered for the pool slot `workspace`, or
    None and why — read from the MAIN clone beside it, never from
    `<workspace>/.git` (the worker's file: `git_admin` reads it).

    The slot `<parent>/<main>-wtN` names its main clone `<parent>/<main>`;
    git keeps every linked worktree's admin directory as
    `<main>/.git/worktrees/<id>` with a `gitdir` file naming the worktree's
    `.git` (measured 2026-11-18: `<id>` is the slot's directory name, as
    `git worktree add` picks it). The one whose `gitdir` names
    `<workspace>/.git` is the slot's; `commondir` must lead back to
    `<main>/.git`."""
    main_name, infix, number = workspace.name.rpartition(SLOT_INFIX)
    if not infix or not main_name or not number.isdigit():
        return None, f"{workspace} is not named `<main>{SLOT_INFIX}N`"
    common = workspace.parent / main_name / ".git"
    worktrees = common / "worktrees"
    if not worktrees.is_dir():
        return None, f"{common} registers no linked worktrees"
    marker = workspace / ".git"
    for admin in sorted(worktrees.iterdir()):
        try:
            gitdir = (admin / "gitdir").read_text(encoding="utf-8").strip()
            commondir = (admin / "commondir").read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not _same(gitdir, marker):
            continue
        if not _same(admin / commondir, common):
            return None, f"{admin} does not belong to {common}"
        return admin.resolve(), ""
    return None, f"{common} registers no worktree at {workspace}"


def registered_submodule(slot_admin: Path, slot: Path, path: str,
                         name: str) -> tuple[Optional[Path], str]:
    """The admin directory of the slot's submodule `name` checked out at
    `path`, or None and why — `<slot_admin>/modules/<name>` (where `git
    submodule` keeps a worktree's submodules, measured 2026-11-18), never
    the `<slot>/<path>/.git` file the worker may rewrite. Both words come
    from the worker's `.gitmodules`, so the answer must stay inside the
    trusted `modules/` and its own `config` (which the worker cannot write)
    must name `<slot>/<path>` as its work tree."""
    modules = slot_admin / "modules"
    admin = modules / name
    if not name or os.path.isabs(name) or not _same(
            os.path.commonpath([os.path.normpath(admin), os.path.normpath(modules)]),
            modules) or _same(admin, modules):
        return None, f"submodule name {name!r} leaves {modules}"
    worktree = _config_value(admin / "config", "core", "worktree")
    if worktree is None:
        return None, f"{admin} is not an initialized submodule repository"
    if not _same(admin / worktree, slot / path):
        return None, f"{admin} names {worktree} as its work tree, not {slot / path}"
    return admin, ""


def _config_value(config: Path, section: str, key: str) -> Optional[str]:
    """`<section>.<key>` of a git config file git itself wrote (a submodule
    admin's `config`: plain `[core]` lines, no includes), or None — read
    without git, which would look for a repository of its own first."""
    try:
        lines = config.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    current = ""
    for line in lines:
        text = line.strip()
        if text.startswith("["):
            current = text.strip("[]").strip().casefold()
        elif current == section and "=" in text:
            name, _, value = text.partition("=")
            if name.strip().casefold() == key:
                return value.strip().strip('"')
    return None


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
