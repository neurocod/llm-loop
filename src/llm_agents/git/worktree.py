#!/usr/bin/env python3
"""A standing git worktree ("slot") so a second writer can work beside the main checkout.

The unit of occupancy is a TASK, not an agent: the slot hosts the whole pipeline
of one task (implement -> review -> fixes -> refactor pass) and is handed back
once the task's commits are on the main branch. Slots are siblings of the main checkout
(`<repo>-wt1`, ...) on branches `wt1`, ... (`slots.py` holds the naming rule);
created ONCE and kept, because a dependency install and a build configure cost
minutes.

The price to plan for: every slot is a separate writer on the SAME machine, so
slots running tests at once share the CPU, and wall-clock budgets measured on a
quiet machine -- test timeouts, a gate's wall budget -- can be overrun by
contention alone (measured 2026-09-27: 4-5 concurrent agents slowed one child
pytest from 57 s to 400-800 s). A red timeout from a slot whose neighbour was
also running a gate is not yet a defect: rerun it alone before judging.

Subcommands (each has --help): status, new, equalize, the host's share command
(when it shares directories), land, push, next, release, drop, and the two
Claude Code hooks hook-create / hook-remove.

The hooks are wired in .claude/settings.json as WorktreeCreate/WorktreeRemove.
With them, an agent spawned with `isolation: "worktree"` and a `claude
--worktree <name>` session land in the slot instead of a bare checkout under
.claude/worktrees/ -- pinned there by the harness, so "the agent was never
told which directory it works in" cannot happen. Measured 2026-09-06: a `-p
--worktree` run and an `isolation: "worktree"` subagent both landed in the slot
the hook printed; the hook input carries `name`, `cwd` = main checkout,
`session_id`; a non-zero exit aborts the creation (the harness does not start
the agent); WorktreeRemove did not fire when the subagent finished; the harness
did not object to the slot being locked before it received the path.

What a project adds -- how its users spell this command, the submodules a new
slot gets, directories only the main checkout fills, its dependency install,
its notes and help texts, its main branch and remote and whether it pushes
on its own -- is a `PoolHost` handed to `main` by the project's
own entry point. Run directly, the script uses `EMPTY` and says so on stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import filecmp
import functools
import json
import os
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from ..gitlib import (Fail, PUSH_TIMEOUT_S, SKIP_SUBMODULES, Pinned, Repository, git,
                      trusted_git)
from . import live_workers
from .slots import (SLOT_NAME_RE, git_admin, registered_admin, registered_submodule,
                    slot_dir_name, slot_name, slot_number)

# `terminal_prompt=False`: every command here may run from a hook with nobody at
# the keyboard, so git must never ask for credentials (the automatic push in
# `push_main_branch`); `failure="exit"` turns a failed or timed-out command into a
# one-line `error:` instead of a traceback. Both knobs are explained on
# `gitlib.git`.
git = functools.partial(git, errors="replace", terminal_prompt=False, failure="exit")


@dataclass(frozen=True)
class SharedDir:
    """A directory the main checkout holds, linked into every slot at the
    same relative path, so a slot sees the main checkout's files -- and their
    later additions -- without copying them, and what a slot writes there
    lands in the main checkout for every slot. `what` names it in messages
    ("shared <what> linked"), `items` names its contents ("keeping local
    <items>")."""
    path: str
    what: str = "directory"
    items: str = "files"


def _no_install(slot: Path, admin: Path) -> None:
    return None


@dataclass(frozen=True)
class PoolHost:
    """One project's rules for the pool; every default is the empty host's
    answer. Compared by identity: `EMPTY` is tested for with `is`."""
    # How messages spell this command ("land them first (<command> land wt1)").
    command: str = "python -m llm_agents.git.worktree"
    # A checkout of the project, which the main checkout is found from; None:
    # the current directory.
    anchor: Optional[Path] = None
    # Submodules `new` initializes unless asked for all of them.
    default_submodules: tuple = ()
    shared: tuple = ()          # SharedDir
    share_command: str = "share-dirs"
    # The project's dependency step, (slot, slot's git admin dir); `new`,
    # `equalize` and the create hook run it unless --no-install.
    install: Callable[[Path, Path], None] = _no_install
    # Lines `new` prints after creating a slot.
    new_notes: tuple = ()
    # Help texts that replace the defaults below, by key.
    help: Mapping[str, str] = field(default_factory=dict)
    # The branch slots are cut from and land into, and the remote it is pushed
    # to. A fixed name rather than the remote's default branch read from
    # `refs/remotes/<remote>/HEAD`: that ref exists only in a clone (not after
    # `git init` + `remote add`) and goes stale when the remote renames its
    # branch, so a detected name would differ between two checkouts of one
    # project. `main` is the name the hosting services start a repository on;
    # a project on another name says so here, and a wrong one fails on the
    # first git call that names it rather than landing somewhere else.
    main_branch: str = "main"
    remote: str = "origin"
    # Whether the commands that move the main branch (`new`, `equalize`,
    # `land`, the create hook) push it afterwards; `--push`/`--no-push`
    # override it per call. Off unless the host opts in: a first `land` in
    # somebody else's repository must not publish anything.
    push_by_default: bool = False


EMPTY = PoolHost()
# The host every function here reads; `main` installs the one it is given.
HOST: PoolHost = EMPTY

HELP = {
    "--no-install": "skip the host's install step",
    "new": "create the slot (worktree + submodules + install step)",
    "--submodules": "'all' to initialize every declared submodule, not only the host's default ones",
    "share": "link a slot's missing shared directories to the main checkout; "
             "default slot: the current worktree",
    "drop": "remove a slot and its branch (refuses unlanded or dirty work)",
    "drop shared": "safely detach shared directories, remove a slot and its branch "
                   "(refuses unlanded or dirty work)",
}


def help_text(key: str) -> str:
    return HOST.help.get(key, HELP[key])


SLOT_RE = SLOT_NAME_RE
# Pool state lives under the repository's own .git dir, never inside the slot: an
# untracked state file there would make the slot look dirty to `is_clean`.
NEXT_FILE = "wt-pool-next"          # .git/<NEXT_FILE>: which slot the next create hook hands out
LOCK_REASON_PREFIX = "worktree.py:"


def norm(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


@functools.lru_cache(maxsize=None)
def _main_dir_from(anchor: Path) -> Path:
    common = git(["rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=anchor).stdout.strip()
    return Path(common).parent


def main_dir() -> Path:
    """The main checkout, whichever worktree (or copy of this script) we run from."""
    return _main_dir_from((HOST.anchor or Path.cwd()).resolve())


def git_dir() -> Path:
    return main_dir() / ".git"


@dataclass
class Slot:
    number: int
    path: Path
    branch: str          # "" when detached
    head: str
    locked: bool
    lock_reason: str = ""

    @property
    def name(self) -> str:
        return slot_name(self.number)

    @property
    def detached(self) -> bool:
        return not self.branch

    def admin_dir(self) -> Path:
        # Read off the main clone, not asked of git in the slot (`slot_top`).
        admin, why = registered_admin(self.path)
        if admin is None:
            raise Fail(f"{self.name}: {why}")
        return admin


def slot_path(number: int) -> Path:
    return main_dir().parent / slot_dir_name(main_dir().name, number)


# Slots live OUTSIDE the main checkout, so Claude Code treats every file there as
# foreign until the directory is listed in `permissions.additionalDirectories`:
# an agent pinned to an unlisted slot stops on a permission prompt at its first
# edit, and with nobody at the keyboard the whole wave stalls on it (measured
# 2026-10-09, after `new` made a fourth slot). The project file is where the
# slots are usually listed; the local and user files count too, since a
# listing there works just as well.
PERMISSION_FILES = (Path(".claude/settings.json"), Path(".claude/settings.local.json"))


def permitted_dirs() -> list[str]:
    files = [main_dir() / f for f in PERMISSION_FILES] + [Path.home() / ".claude" / "settings.json"]
    dirs: list[str] = []
    for file in files:
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        permissions = data.get("permissions") if isinstance(data, dict) else None
        listed = permissions.get("additionalDirectories") if isinstance(permissions, dict) else None
        dirs += [norm(d) for d in listed if isinstance(d, str)] if isinstance(listed, list) else []
    return dirs


def is_permitted(path: Path) -> bool:
    target = norm(path)
    return any(target == d or target.startswith(d.rstrip("\\/") + os.sep) for d in permitted_dirs())


def permission_warning(name: str, path: Path) -> str:
    return (f"warning: {name} ({path}) is not in permissions.additionalDirectories of "
            f"{PERMISSION_FILES[0].as_posix()}: an agent pinned there stops on a permission prompt "
            f"at its first edit, and an unattended session stalls. Add {json.dumps(str(path))} to that list.")


def list_slots() -> list[Slot]:
    """Pool slots as git knows them: worktrees named <repo>-wtN (any letter case)."""
    out = git(["worktree", "list", "--porcelain"], cwd=main_dir()).stdout
    slots: list[Slot] = []
    path = branch = head = reason = ""
    locked = False
    main_name = main_dir().name

    def flush() -> None:
        nonlocal path, branch, head, locked, reason
        if path:
            number = slot_number(main_name, Path(path).name)
            if number is not None:
                slots.append(Slot(number, Path(path), branch, head, locked, reason))
        path = branch = head = reason = ""
        locked = False

    for line in out.splitlines():
        if line.startswith("worktree "):
            flush()
            path = line[len("worktree "):]
        elif line.startswith("HEAD "):
            head = line[len("HEAD "):]
        elif line.startswith("branch refs/heads/"):
            branch = line[len("branch refs/heads/"):]
        elif line == "locked" or line.startswith("locked "):
            locked = True
            reason = line[len("locked "):] if line.startswith("locked ") else ""
    flush()
    return sorted(slots, key=lambda s: s.number)


def find_slot(spec: str) -> Slot:
    """Resolve `wt1`, `1`, a directory path, or '.' (the current worktree)."""
    slots = list_slots()
    m = SLOT_RE.match(spec)
    if m or spec.isdigit():
        number = int(m.group(1) if m else spec)
        for s in slots:
            if s.number == number:
                return s
        raise Fail(f"no slot wt{number}; existing: {', '.join(s.name for s in slots) or 'none'}")
    target = Path(spec)
    if not target.is_dir():
        raise Fail(f"{spec} is not a slot name (wtN) nor a directory")
    # By path, not by `git rev-parse --show-toplevel` there: git asked from a
    # folder of a slot discovers the nearest `.git`, which in a submodule (or
    # any folder the worker plants one in) is the worker's to point at a
    # repository of its own (`slot_submodule`).
    for s in slots:
        if is_inside(target, s.path):
            return s
    if is_inside(target, main_dir()):
        raise Fail(f"{target} is in the main checkout ({HOST.main_branch}), not a pool slot.\n"
                   + slots_hint(slots))
    raise Fail(f"{target} is in no pool slot ({main_dir().name}-wtN)")


def slots_hint(slots: list[Slot]) -> str:
    if not slots:
        return f"No slot exists yet: {HOST.command} new"
    lines = ["Slots:"]
    for s in slots:
        lines.append(f"  {s.name}  {s.path}  {describe(s)}")
    lines.append("Pass one with --dir <path or wtN>")
    return "\n".join(lines)


# ------------------------------------------------------------------ state

def is_clean(path: Path, top: Optional[Pinned] = None) -> bool:
    # Edits or stray files INSIDE a submodule are that submodule's business, not
    # the slot's work; a gitlink pointing elsewhere than the main branch's is still reported.
    # `top`: a slot's bound git (`slot_top`); without it, git by discovery --
    # the main checkout only. `.gitattributes` is the worker's and may name
    # any filter driver that system, global or the main clone's config
    # defines (`lfs` is one: `slot_top` names its endpoint), and a failing
    # filter fails the status -- read as dirty, never as clean. `=dirty`
    # starts no git in a submodule (it reads the HEAD its `.git` names
    # in-process; `gitlib.SKIP_SUBMODULES`) and on the command line beats an
    # `ignore` in the worker's `.gitmodules`.
    return status_entries(path, top) == []


def status_entries(path: Path, top: Optional[Pinned] = None) -> Optional[list[tuple[str, str]]]:
    """(XY, path) per `git status --porcelain` entry, as `is_clean` reads the
    tree; None when the status itself fails (dirty, never clean)."""
    proc = git(["status", "--porcelain", "-z", "--ignore-submodules=dirty"], cwd=path, check=False,
               pinned=top)
    if proc.returncode != 0:
        return None
    entries: list[tuple[str, str]] = []
    fields = iter(proc.stdout.split("\0"))
    for item in fields:
        if not item:
            continue
        entries.append((item[:2], item[3:]))
        if item[0] in "RC":
            next(fields, None)      # a rename's or copy's source path
    return entries


def gitlink_drift_hold(slot_path: Path, sub: str, pinned: Pinned) -> tuple[str, str, str]:
    """(head, wanted, why) for the slot's submodule `sub` whose checkout is not
    the commit the slot's HEAD records: '' as `why` when `git submodule update`
    may move it there with nothing lost, else the reason the slot stays held.

    The case it exists for (2699, measured): a land bumped a gitlink, the
    orchestrator rebased the slot onto it, and the slot's clone stayed on the
    old commit -- ` M <sub>` in the slot's status, so the create hook refused
    it as dirty ("no free slot") with two slots really free. The move is safe
    when the clone's tracked and untracked files are clean and its HEAD is an
    ancestor of the gitlink. Any commit past the gitlink holds the slot, even
    one a remote-tracking branch has: such a ref proves the commit reachable
    here, not published (a fetch from a local path or a stale ref counts as
    one), so whether it is a worker's unrecorded submodule work (e.g. on a
    branch like `wt1-2689`) is the orchestrator's call. A HEAD that diverged
    from the gitlink holds it too. Ancestry is read in the slot's clone, or
    else in the main checkout's when only that one has the gitlink (landed but
    not fetched yet); a gitlink in neither holds the slot. Nothing is fetched
    here: `slot_state` only looks."""
    wanted = gitlink("HEAD", sub, slot=slot_path)
    head = bound_git(pinned, ["rev-parse", "HEAD"], check=False).stdout.strip()
    if not wanted or not head:
        return head, wanted, "its HEAD or the gitlink the slot records cannot be read"
    status = bound_git(pinned, ["status", "--porcelain", SKIP_SUBMODULES], check=False)
    if status.returncode != 0:
        # An error prints nothing on stdout: read as output only, it was clean.
        return head, wanted, f"its status cannot be read ({one_line(status.stderr) or f'exit {status.returncode}'})"
    if status.stdout.strip():
        return head, wanted, "it has uncommitted changes or untracked files"
    branch = bound_git(pinned, ["symbolic-ref", "-q", "--short", "HEAD"], check=False).stdout.strip()
    at = f"{head[:9]}" + (f" (branch {branch})" if branch else "")
    main_clone = main_dir() / sub
    diverged = f"its HEAD {at} has diverged from the gitlink {wanted[:9]} (side-branch work?)"
    if has_commit(slot_path / sub, wanted, pinned):
        def is_ancestor(a: str, b: str) -> bool:
            return bound_git(pinned, ["merge-base", "--is-ancestor", a, b], check=False).returncode == 0
    elif (main_clone / ".git").exists() and has_commit(main_clone, wanted):
        # A clone holding `wanted` holds its ancestors, so a `head` it lacks is
        # none of them; nor a descendant, which the slot's clone would make it hold.
        if not has_commit(main_clone, head):
            return head, wanted, diverged

        def is_ancestor(a: str, b: str) -> bool:
            return git(["merge-base", "--is-ancestor", a, b], cwd=main_clone, check=False).returncode == 0
    else:
        return head, wanted, (f"{wanted[:9]} is in neither its clone nor the main checkout's, "
                              f"so whether the move loses work cannot be told")
    if is_ancestor(head, wanted):
        return head, wanted, ""
    if not is_ancestor(wanted, head):
        return head, wanted, diverged
    past = bound_git(pinned, ["rev-list", "--count", "HEAD", "--not", wanted], check=False).stdout.strip()
    return head, wanted, (f"its HEAD {at} carries {past or 'uncounted'} commit(s) past the gitlink "
                          f"{wanted[:9]} (unrecorded work? record or drop it)")


def unlanded_branch(branch: str) -> list[str]:
    """`git cherry -v <main> <branch>` '+' lines: commits whose PATCH is not on
    the main branch.

    Patch identity, not ancestry: a branch that reached the main branch through
    a rebase or squash has different SHAs but no '+' lines, and may be reset;
    an ancestry check would refuse it forever.
    """
    out = git(["cherry", "-v", HOST.main_branch, branch], cwd=main_dir()).stdout
    return [line[2:] for line in out.splitlines() if line.startswith("+ ")]


@dataclass
class SlotState:
    exists: bool
    detached: bool
    locked: bool
    lock_reason: str
    clean: bool
    unlanded: list[str]
    behind: int
    head: str
    # "<sub>: <why>" per initialized submodule no git here may be bound to
    # (`slot_submodule`): the commands that would move it refuse the slot.
    foreign: list[str] = field(default_factory=list)
    # One line per live-worker record naming the slot (`live_workers`): no
    # command that acts on the tree runs there unless told IGNORE_LIVE_FLAG.
    live: list[str] = field(default_factory=list)
    # {sub: (head, gitlink)} per submodule whose checkout lags the gitlink
    # while nothing else in the tree differs and the move loses nothing
    # (`gitlink_drift_hold`): not dirt; `equalize`, the create hook and `land`
    # of an unclaimed slot update it (`cmd_land`). A drift that may not be
    # followed is dirt, its reason in `held`.
    stale: dict[str, tuple[str, str]] = field(default_factory=dict)
    held: list[str] = field(default_factory=list)

    @property
    def free(self) -> bool:
        return (self.exists and not self.detached and not self.locked and self.clean
                and not self.unlanded and not self.foreign and not self.live)

    def describe(self) -> str:
        if not self.exists:
            return "MISSING (directory gone; `git worktree prune`, then `new`)"
        if self.detached:
            return f"DETACHED at {self.head[:9]} (rebase in progress? finish or abort it)"
        parts = []
        if self.locked:
            parts.append(f"locked ({self.lock_reason or 'no reason'})")
        if not self.clean:
            parts.append(f"dirty ({'; '.join(self.held)})" if self.held else "dirty")
        if self.unlanded:
            parts.append(f"{len(self.unlanded)} unlanded commit(s)")
        if self.behind:
            parts.append(f"{self.behind} behind {HOST.main_branch}")
        if self.foreign:
            parts.append(f"git not the slot's ({'; '.join(self.foreign)})")
        if self.live:
            parts.append(f"worker alive ({'; '.join(self.live)})")
        verdict = ", ".join(parts) if parts else "free"
        if self.stale:
            verdict += " (" + "; ".join(f"{sub} at {head[:9]}, gitlink {wanted[:9]}: updated on hand-out"
                                        for sub, (head, wanted) in self.stale.items()) + ")"
        return verdict


def slot_state(slot: Slot) -> SlotState:
    exists = slot.path.is_dir()
    usable = exists and not slot.detached
    behind = 0
    if usable:
        behind = int(git(["rev-list", "--count", f"{slot.branch}..{HOST.main_branch}"],
                         cwd=main_dir()).stdout.strip() or 0)
    foreign = []
    top: Optional[Pinned] = None
    bound: dict[str, Pinned] = {}
    if exists:
        top, why = slot_top(slot.path)
        if top is None:
            foreign.append(f"the slot itself: {why}")
        for sub in initialized_submodules(slot.path):
            pinned, why = slot_submodule(slot.path, sub)
            if why:
                foreign.append(f"{sub}: {why}")
            elif pinned is not None:
                bound[sub] = pinned
    clean, stale, held = True, {}, []
    if exists:
        entries = status_entries(slot.path, top) if top is not None else None
        clean = entries == []
        # Only a worktree-side gitlink change of a bound submodule may be a
        # drift to follow; any other entry (a staged gitlink included) is work.
        if entries and all(xy == " M" and path in bound for xy, path in entries):
            for _, sub in entries:
                head, wanted, why = gitlink_drift_hold(slot.path, sub, bound[sub])
                if why:
                    held.append(f"{sub}: {why}")
                else:
                    stale[sub] = (head, wanted)
            clean = not held
            if not clean:
                stale = {}
    return SlotState(
        exists=exists, detached=slot.detached, locked=slot.locked, lock_reason=slot.lock_reason,
        clean=clean,
        unlanded=unlanded_branch(slot.branch) if usable else [],
        behind=behind, head=slot.head, foreign=foreign,
        live=live_worker_lines(slot) if exists else [], stale=stale, held=held)


def describe(slot: Slot) -> str:
    return slot_state(slot).describe()


def refuse_unless_idle(slot: Slot, state: SlotState | None = None) -> None:
    """Refuse a slot that is busy or holds work: `equalize` and `drop` both need this.

    Every pool operation on a slot (`equalize`, `drop`, `next` and the create
    hook handing it out, `land`, `release`, `new`'s install) assumes the
    slot's worker has ENDED: the checks here and in `slot_state` look at the
    tree once, and the git that follows acts on it later -- a live worker
    could swap `<slot>/<sub>` for a junction between `slot_submodule`'s check
    and a bound checkout, or turn `<sub>/.git` into a directory before
    `drop`'s `submodule deinit`. For a worker the Codex bridge runs, and for
    another pool command on the slot, that is enforced, not assumed: each of
    those commands holds `worker_excluded` for its whole run. A worker no
    bridge records (a Claude agent, a shell) leaves only the lock (`claim`)."""
    state = state or slot_state(slot)
    if state.detached:
        raise Fail(f"{slot.name} is {state.describe()}")
    refuse_foreign(slot, state)
    if state.locked:
        raise Fail(f"{slot.name} is locked ({state.lock_reason or 'no reason'}): an agent may still be "
                   f"running there. If it is gone: {HOST.command} release {slot.name}")
    if not state.clean:
        proc = top_git(slot.path, ["status", "--short", "--ignore-submodules=dirty"], check=False)
        status = (proc.stdout + proc.stderr).rstrip()
        held = "".join(f"\n{why}" for why in state.held)
        raise Fail(f"{slot.name} has uncommitted changes; commit or discard them first:\n{status}{held}")
    if state.unlanded:
        raise unlanded_refusal(slot.name, state.unlanded, slot.path)


IGNORE_LIVE_FLAG = "--ignore-live-worker"


def live_worker_lines(slot: Slot) -> list[str]:
    """One line per live-worker record naming the slot; a records folder
    that cannot be listed is one line too, so the slot is not free."""
    try:
        return [worker.describe() for worker in live_workers.live_workers(slot.path)]
    except OSError as error:
        return [f"live-worker records unreadable ({error})"]


def worker_excluded(slot: Slot, ignore: bool = False, *,
                    wait_s: Optional[float] = None) -> contextlib.ExitStack:
    """What a command that acts on the slot's tree holds for its whole run,
    taken before its first look at the slot:

    - the slot's operation lock (`live_workers.slot_operation`), so no other
      pool command acts on the slot meanwhile -- waited for up to `wait_s`
      (default `live_workers.OPERATION_WAIT_S`), then refused naming the
      holder;
    - the pool's side of `live_workers.pool_guard`, so no worker turn can
      start in the slot meanwhile, after checking that none runs there now (a
      turn's guard, or a record whose job still has processes). `ignore`
      (`IGNORE_LIVE_FLAG`) skips this part only, for a worker known to be hung
      or not the slot's: the flag is about workers, never about another pool
      command."""
    held = contextlib.ExitStack()
    what = " ".join(["worktree", *sys.argv[1:]])
    try:
        operation = live_workers.slot_operation(slot.path, what, wait_s=wait_s)
    except live_workers.Busy as busy:
        raise Fail(f"{slot.name}: {busy}. Run this again once it is done; a holder that hangs "
                   f"is stopped by hand (its lock dies with it)") from busy
    held.enter_context(operation)
    if ignore:
        return held
    remedy = (f"Stop it first: the pool's checks assume the slot's worker has ended. If it "
              f"is hung or not this slot's: rerun with {IGNORE_LIVE_FLAG}")
    with held.pop_all() as undo:
        try:
            guard = live_workers.pool_guard(slot.path)
        except live_workers.Busy as busy:
            raise Fail(f"{slot.name}: a worker is running there ({busy}). {remedy}") from busy
        undo.enter_context(guard)
        live = live_worker_lines(slot)
        if live:
            raise Fail(f"{slot.name}: a worker is still alive there ({'; '.join(live)}). {remedy}")
        return undo.pop_all()


def excluding_worker(default: str | None = None):
    """Run a slot command as `command(args, slot)` inside `worker_excluded`
    for the slot it names (`slot_named`, `default`)."""
    def wrap(command):
        @functools.wraps(command)
        def run(args: argparse.Namespace) -> int:
            slot = find_slot(slot_named(args, default=default))
            with worker_excluded(slot, args.ignore_live_worker):
                return command(args, slot)
        return run
    return wrap


def refuse_foreign(slot: Slot, state: SlotState | None = None) -> None:
    """Refuse a slot some git of which cannot be bound (`SlotState.foreign`).
    No `--force`: nothing here may run in it, so the slot is repaired by hand."""
    state = state or slot_state(slot)
    if state.foreign:
        raise Fail(f"{slot.name}: a git in it is not the one the slot registers "
                   f"({'; '.join(state.foreign)}); no git here runs there. Inspect the "
                   f".git named by hand before anything else does")


def unlanded_refusal(name: str, pending: list[str], path: Path) -> Fail:
    listing = "\n".join(pending)
    main = HOST.main_branch
    return Fail(
        f"{name} has {len(pending)} commit(s) whose patch is not on {main}:\n{listing}\n"
        f"Land them first ({HOST.command} land {name}), or if {main} already carries their "
        f"content in another form, reset by hand: git -C {path} reset --hard {main}")


# ------------------------------------------------------------------ claims

def claim(slot: Slot, reason: str) -> bool:
    """Take the worktree lock atomically; False if somebody holds it already.

    The harness does not lock a hook-provided worktree and WorktreeRemove does
    not fire, so without this claim two agents spawned in one message would both
    be handed the same "free" slot, and the second one's reset + install step
    would run inside the first one's working directory. `land` and `release`
    unlock.
    """
    proc = git(["worktree", "lock", "--reason", f"{LOCK_REASON_PREFIX}{reason}", str(slot.path)],
               cwd=main_dir(), check=False)
    return proc.returncode == 0


def unclaim(slot: Slot) -> None:
    git(["worktree", "unlock", str(slot.path)], cwd=main_dir(), check=False)


def init_submodules(slot: Slot, which: list[str]) -> None:
    """A worktree cannot share the main checkout's submodule clones (each gets
    its own gitdir under .git/worktrees/<slot>/modules/), so clone with
    `--reference` to the main copy for speed and `--dissociate` so a `gc` in
    the main checkout cannot prune objects the slot still needs."""
    # The fresh clones below are git's own `submodule update --init` in a slot
    # `new` has just made, before any worker had it; an initialized one may
    # be older than that and is moved bound (`update_submodules`).
    update_submodules(slot, [sub for sub in which if (slot.path / sub / ".git").exists()])
    for sub in which:
        if (slot.path / sub / ".git").exists():
            continue
        args = ["submodule", "update", "--init"]
        if (main_dir() / sub / ".git").exists():
            print(f"initializing submodule {sub} (objects from the main checkout's clone, dissociated)")
            args += ["--reference", str(main_dir() / sub), "--dissociate"]
        else:
            print(f"initializing submodule {sub} (not initialized in the main checkout: full clone)")
        top_git(slot.path, [*args, "--", sub])


def all_submodules() -> list[str]:
    """Paths of the submodules .gitmodules declares; [] when there are none.

    `-z` prints each entry as `key\\nvalue\\0`, so a path with spaces comes back
    whole (splitting the plain `key value` line on whitespace cut it). Exit 1 is
    git's "no such key", i.e. no submodules -- not an error.
    """
    return list(declared_submodules().values())


def declared_submodules() -> dict[str, str]:
    """{name: path} of the submodules the MAIN checkout's .gitmodules declares
    (never a slot's: the worker writes that one)."""
    if not (main_dir() / ".gitmodules").is_file():
        return {}
    proc = git(["config", "-z", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
               cwd=main_dir(), check=False)
    if proc.returncode == 1:
        return {}
    if proc.returncode != 0:
        raise Fail(f"cannot read submodule paths from .gitmodules:\n{proc.stdout}{proc.stderr}".rstrip())
    pairs = (entry.partition("\n") for entry in proc.stdout.split("\0") if entry)
    return {key[len("submodule."):-len(".path")]: path for key, _, path in pairs}


def initialized_submodules(path: Path) -> list[str]:
    return [sub for sub in all_submodules() if (path / sub / ".git").exists()]


# The hooks folder of every bound git here (`slot_top`, `slot_submodule`):
# empty, and under the main clone's .git, where no slot's worker writes.
NO_HOOKS_DIR = "wt-pool-no-hooks"


def no_hooks() -> Path:
    hooks = git_dir() / NO_HOOKS_DIR
    hooks.mkdir(exist_ok=True)
    return hooks


# Git LFS off, for a repository whose trusted counterpart names no LFS
# endpoint (`lfs_settings`): an empty driver is no driver to git, and a
# missing one must not fail the command.
LFS_OFF = ("filter.lfs.process=", "filter.lfs.smudge=", "filter.lfs.clean=",
           "filter.lfs.required=false")


@functools.lru_cache(maxsize=None)
def _lfs_endpoint(trusted: str) -> str:
    proc = git(["lfs", "env"], cwd=trusted, check=False)
    if proc.returncode != 0:
        return ""
    line = next((line for line in proc.stdout.splitlines() if line.startswith("Endpoint=")), "")
    return line[len("Endpoint="):].split(" (auth=")[0].strip()


def lfs_settings(trusted: Optional[Path]) -> tuple[str, ...]:
    """The Git LFS settings of a bound git whose work tree mirrors `trusted`
    (the main checkout, or its clone of the same submodule; None: none).

    System and global config define `filter.lfs.*` (`required=true`), so a
    worker's `.gitattributes` makes `status`, `reset` and `checkout` run
    git-lfs, which reads the tree's `.lfsconfig` -- the worker's, choosing
    the URL git-lfs downloads a missing object from and asks credentials
    for. A command-line `lfs.url` beats every `.lfsconfig` key that names an
    endpoint (measured 2026-11-18, git-lfs 3.7.1), so it is forced to the
    endpoint `trusted` resolves. LFS is not simply switched off: the first host
    tracks 507 files through it, and git-lfs's own `--skip` smudge writes a
    pointer even for an object the shared `lfs/objects` holds (measured the
    same day) -- a reset would turn every one of them into a pointer. Off
    (`LFS_OFF`) only where `trusted` resolves no endpoint (git-lfs missing,
    or no such counterpart)."""
    endpoint = _lfs_endpoint(str(trusted)) if trusted is not None else ""
    return (f"lfs.url={endpoint}",) if endpoint else LFS_OFF


def slot_top(slot_path: Path) -> tuple[Optional[Pinned], str]:
    """The git to run in the slot's own (top-level) tree, or None and why.

    Not git by discovery there, although a sandbox write-protects the slot's
    `.git` and the admin directory it names: the main clone's
    `core.hooksPath` is `.githooks`, RELATIVE, so a discovered git in the
    slot runs `<slot>/.githooks/*` -- the worker's files, tracked or not
    (review of 2442 ran an untracked `reference-transaction` on `reset
    --hard` and a `post-index-change` on `status`'s index refresh). So the
    same shape as `slot_submodule`: the admin directory the main clone
    registers for the slot, named to git, the absolute git of this
    process's PATH, `gitlib.FORCED`, the empty hooks folder, and
    `lfs_settings` of the main checkout."""
    admin, why = registered_admin(slot_path)
    if admin is None:
        return None, why
    executable, why = trusted_git((slot_path.resolve(),))
    if executable is None:
        return None, why
    return Pinned(executable, no_hooks(), Repository(admin, slot_path),
                  lfs_settings(main_dir())), ""


def top_git(slot_path: Path, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """`git` in a slot's top-level tree as `slot_top` binds it; a Fail when it cannot be bound."""
    pinned, why = slot_top(slot_path)
    if pinned is None:
        raise Fail(f"no git runs in {slot_path}: {why}")
    return git(args, cwd=slot_path, pinned=pinned, **kwargs)


def slot_submodule(slot_path: Path, sub: str) -> tuple[Optional[Pinned], str]:
    """The git to run in the slot's initialized submodule `sub`, or None and why.

    Never git by discovery there: `<slot>/<sub>/.git` is the worker's to
    point at a repository of its own (a sandbox protects only the slot's
    top `.git`, `gitlib.Repository`), whose config names an fsmonitor or a
    clean filter for `status`, a hook for `fetch`/`checkout`, an
    `uploadpack` for its `origin` -- all run with the orchestrator's rights,
    outside any sandbox. So the submodule's repository is the admin
    directory the MAIN clone registers for the slot
    (`slots.registered_submodule`, the name from the main checkout's
    .gitmodules), named to git explicitly, run by the absolute git of this
    process's PATH (`gitlib.trusted_git`) with `gitlib.FORCED`, an empty
    hooks folder and `lfs_settings` of the main checkout's clone of `sub`.
    The `.git` file is read only to refuse: one naming
    another repository leaves the slot's own git and this one disagreeing
    about what the submodule is. A link along `<slot>/<sub>` would carry a
    checkout out of the slot, so it is a refusal too."""
    name = next((n for n, path in declared_submodules().items() if path == sub), None)
    if name is None:
        return None, "not declared in the main checkout's .gitmodules"
    folder = slot_path / sub
    while norm(folder) != norm(slot_path) and folder != folder.parent:
        if is_link(folder):
            return None, f"{folder} is a link"
        folder = folder.parent
    slot_admin, why = registered_admin(slot_path)
    if slot_admin is None:
        return None, why
    admin, why = registered_submodule(slot_admin, slot_path, sub, name)
    if admin is None:
        return None, why
    named = git_admin(slot_path / sub)
    if named is None or norm(named) != norm(admin):
        return None, f"its .git names {named}, the slot registers {admin}"
    executable, why = trusted_git((slot_path.resolve(),))
    if executable is None:
        return None, why
    trusted = main_dir() / sub
    return Pinned(executable, no_hooks(), Repository(admin, slot_path / sub),
                  lfs_settings(trusted if (trusted / ".git").exists() else None)), ""


def bound_git(pinned: Pinned, args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """`git` as `pinned` (`slot_submodule`), in its work tree."""
    return git(args, cwd=pinned.repository.work_tree, pinned=pinned, **kwargs)


def bound_submodules(slot: Slot, subs: list[str], *, strict: bool) -> dict[str, Pinned]:
    """{sub: its bound git} for each of `subs` (`slot_submodule`); one that
    cannot be bound is a Fail when `strict`, else a warning, and is left out."""
    bound: dict[str, Pinned] = {}
    for sub in subs:
        pinned, why = slot_submodule(slot.path, sub)
        if pinned is not None:
            bound[sub] = pinned
            continue
        message = f"{slot.name}'s {sub} is left where it is: {why}; no git here runs in it"
        if strict:
            raise Fail(message)
        print(f"warning: {message}", file=sys.stderr)
    return bound


# ------------------------------------------------------------ main branch

def push_main_branch() -> bool:
    """Push the main branch to the host's remote; a failure is reported and
    the local result stands.

    Whether a command calls this on its own is the host's choice
    (`PoolHost.push_by_default`): a pool that pushes automatically keeps the
    remote current without a manual step, and every such command takes
    `--no-push`.
    """
    main, remote = HOST.main_branch, HOST.remote
    try:
        proc = git(["push", remote, main], cwd=main_dir(), check=False, timeout=PUSH_TIMEOUT_S)
    except Fail:
        print(f"warning: push of {main} timed out after {PUSH_TIMEOUT_S} s (the local result stands)",
              file=sys.stderr)
        return False
    text = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        print(f"warning: push of {main} failed (the local result stands):\n{text}", file=sys.stderr)
        return False
    last = text.splitlines()[-1] if text else ""
    print(f"pushed {main} to {remote}: {last}" if last else f"pushed {main} to {remote}")
    return True


def wants_push(args: argparse.Namespace) -> bool:
    """`--push` or `--no-push` if the call gave one, else the host's default."""
    asked, refused = getattr(args, "push", False), getattr(args, "no_push", False)
    if asked and refused:
        raise Fail("both --push and --no-push given")
    return asked or (HOST.push_by_default and not refused)


def finish(args: argparse.Namespace) -> int:
    """Exit code of a command whose last step is the push: 2 when the push failed."""
    if wants_push(args):
        return 0 if push_main_branch() else 2
    if getattr(args, "no_push", False):
        print("push skipped (--no-push)")
    else:
        print(f"push skipped (this pool does not push by default; --push pushes "
              f"{HOST.main_branch} to {HOST.remote})")
    return 0


def main_branch() -> str:
    return git(["symbolic-ref", "--quiet", "--short", "HEAD"], cwd=main_dir(), check=False).stdout.strip()


# ------------------------------------------------------------ shared dirs

def is_directory_junction(path: Path) -> bool:
    """Detect Windows junctions without requiring Python 3.12's Path.is_junction."""
    if os.name != "nt":
        return False
    try:
        return path.lstat().st_reparse_tag == stat.IO_REPARSE_TAG_MOUNT_POINT
    except FileNotFoundError:
        return False


def share_dir(shared: SharedDir, path: Path, *, keep_local: bool = False) -> None:
    """Expose the main checkout's `shared` directory at the slot's same path.

    Link the directory the main checkout fills, not its parent: a tracked file
    beside it (a `.gitignore`) stays a regular tracked file in the slot, while
    both a dev server and direct filesystem readers see additions without
    copying. A link elsewhere is never overwritten, nor is a local directory
    holding anything the main checkout lacks. Remove linked slots through
    cmd_drop: direct Git removal follows Windows junctions. Creating an empty
    source directory lets fresh clones prepare slots before the main checkout
    has filled it.

    `keep_local` (equalize, the create hook) keeps a local directory that
    holds a file the main checkout lacks or holds differently, and replaces
    one that holds nothing else -- every file a byte-identical twin of the
    main checkout's -- by the link, since the link then loses nothing. Kept
    unconditionally until 2026-11-26, a slot made before its host shared the
    directory stayed on its first copy for ever (measured: a slot's two-month
    old copy of model files lacked 123 the main checkout made since, and its
    dev server answered those with index.html).

    The replacement deletes nothing: the copy is renamed aside
    (`<dir>.pre-share-<stamp>`, same parent, so an atomic rename that Windows
    refuses while a file in it is open), the link takes its name, and one
    message line names the renamed copy, says whether a read-only second
    comparison still finds every file a twin, and leaves deleting it to a
    human. Deleting it automatically was dropped on 2026-11-26: every
    compare-then-unlink walk raced a writer (a file rewritten between the
    comparison and the unlink, a folder swapped for a junction between the
    reparse-point check and the scan), and a stale copy costs only disk. The
    main checkout's directory is prepared before the rename; a link that
    cannot be made renames the copy back when nothing took its name. A main
    checkout directory resolving into the slot's copy (or the copy into it)
    is refused: every file would be its own twin.
    """
    relative = Path(shared.path)
    source = main_dir() / relative
    target = path / relative
    if norm(path) == norm(main_dir()):
        raise Fail(f"cannot link {shared.what} in the main checkout to itself")
    linked = target.is_symlink() or is_directory_junction(target)
    if keep_local and not linked and target.is_dir():
        advice = (f"move it away and run `{HOST.command} {HOST.share_command} {path}` to "
                  f"share them")
        if is_reparse_point(target):
            print(f"keeping local {shared.items}: {target} is a reparse point, not plain "
                  f"files; {advice}")
            return
        if source.exists() and (is_inside(source, target) or is_inside(target, source)):
            raise Fail(f"{source} resolves to {os.path.realpath(source)}, which overlaps the "
                       f"slot's {target}: comparing them would find every file its own twin; "
                       f"fix the main checkout's {shared.what} by hand")
        own = local_only_files(target, source)
        if own:
            print(f"keeping local {shared.items}: {target} holds {len(own)} entries the main "
                  f"checkout lacks or holds differently (first: {own[0]}), so it does not see "
                  f"the main checkout's {shared.items}; {advice}")
            return
        try:
            source.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise Fail(f"cannot create {source} ({error}); {target} is left as it was") from error
        aside = target.with_name(f"{target.name}.pre-share-{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            os.rename(target, aside)
        except OSError as error:
            print(f"keeping local {shared.items}: {target} could not be moved aside "
                  f"({error}); {advice}")
            return
        _replace_aside(shared, source, target, aside)
        return
    source.mkdir(parents=True, exist_ok=True)
    if linked:
        if norm(target) != norm(source):
            raise Fail(f"{target} already links elsewhere; expected {source}")
        print(f"shared {shared.what} already linked: {target} -> {source}")
        return
    if target.exists():
        raise Fail(f"{target} already exists; move the local {shared.items} away before linking "
                   f"shared {shared.what}")
    _link(source, target)
    print(f"shared {shared.what} linked: {target} -> {source}")


def _replace_aside(shared: SharedDir, source: Path, target: Path, aside: Path) -> None:
    """Link `target` to `source` once the slot's copy has been renamed to
    `aside`, and name the copy; it is never deleted from here. Every failure
    from here on names where the copy is."""
    try:
        _link(source, target)
    except OSError as error:
        failed = f"cannot link {target} -> {source} ({error})"
        if os.path.lexists(target):
            raise Fail(f"{failed}; the local {shared.items} are kept in {aside}, and {target} "
                       f"is taken by something else since") from error
        try:
            os.rename(aside, target)
        except OSError as back:
            raise Fail(f"{failed}, nor could {aside} be renamed back ({back}); the local "
                       f"{shared.items} are kept in {aside}") from error
        raise Fail(f"{failed}; the local {shared.items} were renamed back to {target}") from error
    try:
        left = local_only_files(aside, source)
    except Exception as error:
        raise Fail(f"{target} now links to {source}, but comparing the local {shared.items} "
                   f"kept in {aside} failed ({error}); look before deleting it by hand") from error
    if left:
        print(f"local {shared.items} moved aside to {aside}: {len(left)} entries there have no "
              f"byte-identical twin in {source} (first: {left[0]}; changed after the first "
              f"comparison); look before deleting it by hand")
    else:
        print(f"local {shared.items} moved aside to {aside}: every file there has a "
              f"byte-identical twin in {source}; it can be deleted by hand")
    print(f"shared {shared.what} linked: {target} -> {source}")


def _link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # Unlike a Windows symlink, a directory junction needs no administrator
        # privileges or Developer Mode. Both paths are absolute for this API.
        import _winapi
        _winapi.CreateJunction(str(source.resolve()), str(target.absolute()))
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)


def is_reparse_point(path: Path) -> bool:
    """Anything that is not plain data: a symlink or junction, and on Windows
    every other reparse point too (a cloud-sync placeholder, a WSL link, a
    deduplicated file), whose bytes or target the copy does not own."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _own(item: Path, twin: Path) -> bool:
    """Whether the copy's `item` is its own, not a byte-identical twin of the
    main checkout's `twin`. The same file reached by both paths (a main
    checkout directory that links back into the slot) is own: its "twin" is
    itself, and calling it a twin would invite deleting the only copy."""
    try:
        return not (twin.is_file() and not os.path.samefile(item, twin)
                    and filecmp.cmp(item, twin, shallow=False))
    except OSError:
        return True


def _walk_copy(local: Path, main: Path) -> list[str]:
    """Relative paths of what `local` holds of its own; "" stands for `local`
    itself when it cannot be read."""
    own: list[str] = []
    try:
        entries = list(os.scandir(local))
    except OSError:
        return [""]
    for entry in entries:
        item = Path(entry.path)
        name = entry.name
        if is_reparse_point(item):
            own.append(name)
        elif entry.is_dir(follow_symlinks=False):
            own += [f"{name}/{inner}" for inner in _walk_copy(item, main / name)]
        elif _own(item, main / name):
            own.append(name)
    return own


def local_only_files(local: Path, main: Path) -> list[str]:
    """What under `local` the main checkout's `main` lacks, as sorted paths
    relative to `local`: a file with no byte-identical twin at the same
    relative path (`_own`), and any reparse point (whatever it points at or
    stands for, it is not a plain copy). Read-only, and walked by hand
    without entering a reparse point (`os.walk` before Python 3.12 descends
    into a junction); an unreadable folder counts as one entry ("./" for
    `local` itself), since it may hide either."""
    return sorted(entry or "./" for entry in _walk_copy(local, main))


def share_dirs(path: Path, *, keep_local: bool = False) -> None:
    for shared in HOST.shared:
        share_dir(shared, path, keep_local=keep_local)


def is_link(path: Path) -> bool:
    return path.is_symlink() or is_directory_junction(path)


def is_inside(path: str | Path, root: str | Path) -> bool:
    target, top = norm(path), norm(root)
    return target == top or target.startswith(top.rstrip("\\/") + os.sep)


def outside_links(root: Path) -> list[tuple[Path, str]]:
    """Every symlink or junction under `root` whose target lies outside it, as
    (link, target).

    Walked without following any link, so what a link points at is never
    listed, let alone entered. A directory that cannot be read may hide a link,
    so it is a refusal, not a skip.
    """
    found: list[tuple[Path, str]] = []
    folders = [root]
    while folders:
        folder = folders.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError as error:
            raise Fail(f"cannot look for links in {folder} ({error}); removing the slot could follow "
                       f"one out of it, so remove {root} by hand")
        for entry in entries:
            path = Path(entry.path)
            if is_link(path):
                target = os.path.realpath(path)
                if not is_inside(target, root):
                    found.append((path, target))
            elif entry.is_dir(follow_symlinks=False):
                folders.append(path)
    return found


def _remove_link(link: Path) -> None:
    """The link itself, never its target: unlink a symlink (a POSIX one, or a
    Windows file symlink), rmdir a junction or a Windows directory symlink."""
    try:
        link.unlink()
    except OSError:
        os.rmdir(link)


def detach_outside_links(path: Path) -> None:
    """Remove every link that leads out of the slot before anything removes it recursively.

    Git for Windows follows directory junctions during `worktree remove
    --force` and `submodule deinit -f` (measured 2026-10-05,
    test_removing_a_worktree_keeps_shared_files): left in place, a junction
    to the main checkout's directory takes that directory's files with the
    slot. Every link is detached, not only the host's `shared` ones: a link
    the host does not list -- one made by hand, by a dependency installer, or
    by a run that knew another host -- costs the same, and the empty host
    lists none at all. One that cannot be detached is a refusal naming it.
    Links pointing inside the slot go with the slot, so they are left to Git.
    """
    listed = {norm(path / Path(shared.path)) for shared in HOST.shared}
    for link, target in outside_links(path):
        try:
            _remove_link(link)
        except OSError as error:
            raise Fail(f"cannot detach {link} -> {target} ({error}); removing the slot would follow "
                       f"it and delete files of {target}. Remove the link by hand, then drop again")
        if norm(link) not in listed:
            print(f"detached {link} -> {target} (a link out of the slot)")


# ------------------------------------------------------------- submodules

def gitlink(rev: str, sub: str, slot: Path | None = None) -> str:
    """The submodule commit `rev` records at `sub` in the slot `slot` (its
    bound git, `slot_top`) or else the main checkout; '' if `rev` has no
    gitlink there."""
    args = ["ls-tree", rev, "--", sub]
    proc = (top_git(slot, args, check=False) if slot is not None
            else git(args, cwd=main_dir(), check=False))
    out = proc.stdout.split()
    return out[2] if len(out) >= 3 and out[1] == "commit" else ""


def has_commit(repo: Path, sha: str, pinned: Optional[Pinned] = None) -> bool:
    """`pinned`: the bound git of a slot's submodule (`slot_submodule`);
    without it, git by discovery in `repo` -- the main checkout's trees only."""
    return git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo, check=False,
               pinned=pinned).returncode == 0


def one_line(text: str) -> str:
    """Git's multi-line output folded into one line of a warning."""
    return " ".join(text.split())


def fetch_commit(repo: Path, sha: str, sources: list[str],
                 failures: Optional[list[str]] = None, pinned: Optional[Pinned] = None) -> bool:
    """Bring `sha` into `repo` from the first source that has it; True when present.

    The slot's own clone goes first: a gitlink bumped in a slot points at a
    commit that may exist nowhere else yet (the submodule push is a separate
    step). A failing or timed-out fetch is only a miss -- the land already
    happened, so nothing here may abort it -- but not a silent one: each is
    appended to `failures` as "<source>: <why>" for the caller's warning.
    `pinned` as for `has_commit`; a source that is a slot's submodule is its
    registered admin directory, never `<slot>/<sub>` (whose `.git` git would
    follow to start `upload-pack` there).
    """
    for source in sources:
        if has_commit(repo, sha, pinned):
            return True
        try:
            proc = git(["fetch", "--quiet", "--no-tags", source, sha], cwd=repo, check=False,
                       timeout=PUSH_TIMEOUT_S, pinned=pinned)
            why = (one_line(proc.stderr) or f"exit {proc.returncode}") if proc.returncode else ""
        except Fail as error:      # the timeout
            why = one_line(str(error))
        if why and failures is not None:
            failures.append(f"{source}: {why}")
    return has_commit(repo, sha, pinned)


def fetch_unpushed_gitlinks(path: Path, bound: Mapping[str, Pinned]) -> None:
    """Bring each gitlink commit `path`'s HEAD records into its submodule clone
    from the MAIN checkout's clone, where `git submodule update` cannot get it.

    A landed submodule commit stays local until somebody pushes it, and the
    main checkout's clone is where `land` put it (`sync_main_submodules`). A
    slot's clone fetches from the submodule's origin only, so after
    `equalize` (or a rebase onto such a main branch) `git submodule update` failed
    with "not our ref" and left the slot showing `M <sub>` (measured
    2026-11-07; the fix by hand was a fetch from the main clone, then the
    update). A commit the clone already has is left alone, and one the main
    clone lacks too is left to the clone's own origin, which `submodule
    update` and `follow_gitlinks` fetch from. A fetch that fails although the
    main clone has the commit is one warning line naming the submodule, the
    commit and the commands that fix it by hand. `bound`: {sub: its bound
    git} (`bound_submodules`), the submodules to look at.
    """
    for sub, pinned in bound.items():
        repo = path / sub
        source = main_dir() / sub
        if norm(path) == norm(main_dir()):
            continue
        sha = gitlink("HEAD", sub, slot=path)
        if (not sha or has_commit(repo, sha, pinned) or not (source / ".git").exists()
                or not has_commit(source, sha)):
            continue
        failures: list[str] = []
        if fetch_commit(repo, sha, [str(source)], failures, pinned):
            print(f"{sub}: fetched {sha[:9]} from the main checkout's clone (not on its origin yet)")
            continue
        why = "; ".join(failures) or "the fetch reported success, yet the commit is missing"
        print(f"warning: {sub}: cannot fetch {sha[:9]} from the main checkout's clone ({why}). "
              f'Fix: git -C "{repo}" fetch "{source}" {sha}, then '
              f'git -C "{path}" submodule update -- "{sub}"', file=sys.stderr)


def follow_gitlinks(slot: Slot) -> None:
    """Move a slot's clean submodules FORWARD to the gitlinks its HEAD records.

    For a slot that keeps its work (`next`, an assigned create hook): after a
    rebase onto a main branch that bumped a submodule, the slot's clone still
    stands on the old commit. Only a clean clone whose HEAD is an ancestor of
    the recorded gitlink moves, so commits made in the slot's clone and not
    yet recorded are never left behind; anything else is named and kept.

    Never fatal: `next` has recorded its assignment by now, and the create
    hook has claimed the slot -- an abort there consumed the assignment, kept
    the claim and never started the agent. Every miss (a commit nowhere to be
    fetched, a move git refuses, e.g. over an untracked file the gitlink's
    commit adds) is a warning naming the submodule and the command.
    """
    bound = bound_submodules(slot, initialized_submodules(slot.path), strict=False)
    fetch_unpushed_gitlinks(slot.path, bound)
    for sub, pinned in bound.items():
        repo = slot.path / sub
        wanted = gitlink("HEAD", sub, slot=slot.path)
        head = bound_git(pinned, ["rev-parse", "HEAD"], check=False).stdout.strip()
        if not wanted or head == wanted:
            continue

        def warn(why: str) -> None:
            print(f"warning: {slot.name}'s {sub} stays at {head[:9]} although its HEAD records "
                  f"{wanted[:9]}: {why}. When that is resolved: "
                  f'git -C "{slot.path}" submodule update -- "{sub}"', file=sys.stderr)

        # Still missing after the main clone's turn: the clone's own origin may
        # have it, which is where `submodule update` would have looked.
        failures: list[str] = []
        if not fetch_commit(repo, wanted, ["origin"], failures, pinned):
            warn(f"{wanted[:9]} is not in its clone, and origin does not give it either "
                 f"({'; '.join(failures) or 'no error'})")
            continue
        dirty = bound_git(pinned, ["status", "--porcelain", "--untracked-files=no", SKIP_SUBMODULES],
                          check=False).stdout.strip()
        forward = bound_git(pinned, ["merge-base", "--is-ancestor", head, wanted],
                            check=False).returncode == 0
        if dirty or not forward:
            warn("it has uncommitted changes" if dirty else "its HEAD is not an ancestor of the gitlink")
            continue
        why = check_out_gitlink(pinned, wanted)
        if why:
            warn(why)
            continue
        print(f"{slot.name}'s {sub}: {head[:9]} -> {wanted[:9]} (follows the slot's gitlink)")


def check_out_gitlink(pinned: Pinned, wanted: str) -> str:
    """What `git submodule update` does to one initialized submodule
    (a detached checkout of the gitlink), with the bound git
    (`slot_submodule`) instead of the one `submodule update` would find
    through the submodule's `.git`; '' when done, else why not.

    `--no-overwrite-ignore`: an ordinary checkout silently replaces an IGNORED
    file the clone holds at a path the gitlink's commit tracks (a worker's
    local config, a build output), so git refuses instead and the refusal is
    the returned reason. `--no-recurse-submodules`: a configured
    `submodule.recurse` must not move the clone's own nested repositories."""
    moved = bound_git(pinned, ["checkout", "--quiet", "--no-overwrite-ignore", "--no-recurse-submodules",
                               "--detach", wanted], check=False)
    return "" if moved.returncode == 0 else (
        f"the checkout of the gitlink failed: {one_line(moved.stdout + moved.stderr)}")


def update_submodules(slot: Slot, subs: list[str], *, strict: bool = True) -> None:
    """Each initialized submodule of `subs` := the gitlink the slot's HEAD
    records (`git submodule update` without its discovery: `slot_submodule`),
    the commit fetched from the main clone or else from the clone's origin.
    `strict=False`: a miss is a warning, not an abort."""
    bound = bound_submodules(slot, subs, strict=strict)
    fetch_unpushed_gitlinks(slot.path, bound)
    for sub, pinned in bound.items():
        wanted = gitlink("HEAD", sub, slot=slot.path)
        head = bound_git(pinned, ["rev-parse", "HEAD"], check=False).stdout.strip()
        if not wanted or head == wanted:
            continue
        failures: list[str] = []
        why = "" if fetch_commit(slot.path / sub, wanted, ["origin"], failures, pinned) else (
            f"{wanted[:9]} is not in its clone, and origin does not give it either "
            f"({'; '.join(failures) or 'no error'})")
        why = why or check_out_gitlink(pinned, wanted)
        if not why:
            continue
        message = (f"{slot.name}'s {sub} was not updated: {why}. When that is resolved: "
                   f'git -C "{slot.path}" submodule update -- "{sub}"')
        if strict:
            raise Fail(message)
        print(f"warning: {message}", file=sys.stderr)


def reset_to_main_branch(slot: Slot, install: bool, *, strict: bool = True) -> None:
    """The slot's branch := the main branch, dependencies refreshed. Caller has
    checked it is safe and holds `worker_excluded` for the slot.

    `strict=False` (the remove hook, which has unclaimed the slot already): a
    failed submodule update is a warning, not an abort. The create hook
    resets strictly and takes another slot instead (`_choose_slot`).
    """
    # Bound (`slot_top`); no recursion into submodules whatever a
    # configuration says (`update_submodules`).
    top_git(slot.path, ["reset", "--hard", "--no-recurse-submodules", HOST.main_branch])
    update_submodules(slot, initialized_submodules(slot.path), strict=strict)
    # Keep legacy slots with their own local copies. Missing directories and
    # existing shared links use the main checkout, even with --no-install.
    share_dirs(slot.path, keep_local=True)
    print(f"{slot.name} is at {top_git(slot.path, ['rev-parse', '--short', 'HEAD']).stdout.strip()} "
          f"(= {HOST.main_branch})")
    if install:
        HOST.install(slot.path, slot.admin_dir())


def sync_main_submodules(old_main: str, slot: Slot) -> None:
    """After a land, move the main checkout's submodules to the gitlinks the
    main branch now records.

    A plain `merge --ff-only` leaves them where they were, so the main
    checkout -- where most commits happen -- went on running the new commit
    helper against an old submodule (a review found that pair let a killed
    mutation run's change into a commit). A blind `git submodule update` is no
    fix: the main checkout's submodule may legitimately stand on another
    session's unrecorded commits (2026-09-28: a submodule at 5f87561, gitlink
    bc75807). So a submodule only ever moves FORWARD: its tracked files are
    clean and its HEAD is an ancestor of the new gitlink, so every commit it
    stood on stays on the new one's history. A gitlink the land moved back or
    sideways is not followed even from the old gitlink itself -- that would
    leave the old commit reachable only through the reflog. A HEAD on a branch
    (a main checkout's submodule usually sits on `main`) is fast-forwarded with
    the branch, so the next manual commit there lands on the branch and not
    on a detached HEAD; a detached HEAD stays detached. Untracked files the
    move would overwrite make git refuse it (warning); ignored ones are
    overwritten, as by any git checkout. Anything else is left alone with a
    warning naming the reason and the command. Submodules whose gitlink the
    land did not move are not looked at at all.

    Nested submodules are not synced (one level, like `reset_to_main_branch`):
    when the move shifts the gitlink of an initialized nested submodule, a
    warning names it -- one `diff-tree` per moved submodule, and only on a
    bump, so the check is cheaper than finding out from a stale nested tree.
    """
    new_main = git(["rev-parse", HOST.main_branch], cwd=main_dir()).stdout.strip()
    for sub in all_submodules():
        repo = main_dir() / sub
        if not (repo / ".git").exists():
            continue
        old, new = gitlink(old_main, sub), gitlink(new_main, sub)
        if not new or old == new:
            continue
        head = git(["rev-parse", "HEAD"], cwd=repo, check=False).stdout.strip()
        if head == new:
            continue
        on_branch = git(["symbolic-ref", "-q", "HEAD"], cwd=repo, check=False).returncode == 0
        move = ["merge", "--ff-only", new] if on_branch else ["checkout", "--detach", new]
        command = f'git -C "{repo}" {" ".join(move)}'

        def warn(reason: str) -> None:
            print(f"warning: {sub} in the main checkout stays at {head[:9]} although "
                  f"{HOST.main_branch} now records {new[:9]}: {reason}.\n"
                  f"When that is resolved: {command}", file=sys.stderr)

        dirty = git(["status", "--porcelain", "--untracked-files=no"], cwd=repo, check=False).stdout.strip()
        if dirty:
            warn("it has uncommitted changes")
            continue
        # The main checkout's own submodule is asked by discovery (its `.git`
        # is no worker's); the slot's is a source only through its
        # registered admin directory (`fetch_commit`).
        sources: list[str] = []
        failures: list[str] = []
        if (slot.path / sub / ".git").exists():
            pinned, why = slot_submodule(slot.path, sub)
            if pinned is not None:
                sources.append(str(pinned.repository.git_dir))
            else:
                failures.append(f"{slot.name}: not asked, {why}")
        if not fetch_commit(repo, new, [*sources, "origin"], failures):
            warn(f"{new[:9]} is neither in its clone nor fetchable from {slot.name} or origin"
                 + (f" ({'; '.join(failures)})" if failures else ""))
            continue
        ff_ok = git(["merge-base", "--is-ancestor", head, new], cwd=repo, check=False).returncode == 0
        if not ff_ok:
            if head == old:
                warn(f"the land moved the gitlink back or sideways from {old[:9]}; following it would "
                     f"leave {head[:9]} reachable only through the reflog")
            elif git(["merge-base", "--is-ancestor", new, head], cwd=repo, check=False).returncode == 0:
                warn("its HEAD is ahead of the new gitlink (unrecorded work of another session?)")
            else:
                warn("its HEAD has diverged from the new gitlink (unrecorded work of another session?)")
            continue
        moved = git([move[0], "--quiet", *move[1:]], cwd=repo, check=False)
        if moved.returncode != 0:
            warn(f"{move[0]} failed:\n{moved.stdout}{moved.stderr}".rstrip())
            continue
        print(f"{sub}: {head[:9]} -> {new[:9]} (main checkout follows {HOST.main_branch}'s gitlink)")
        warn_nested_gitlinks(sub, repo, head, new)


def warn_nested_gitlinks(sub: str, repo: Path, before: str, after: str) -> None:
    """Name each initialized nested submodule whose gitlink moved between `before` and `after`."""
    out = git(["diff-tree", "-r", "-z", "--raw", before, after], cwd=repo, check=False).stdout
    fields = out.split("\0")
    # -z --raw without rename detection: `:<mode> <mode> <sha> <sha> <status>\0<path>\0` per entry.
    for meta, path in zip(fields[0::2], fields[1::2]):
        modes = meta.split()
        if len(modes) >= 2 and modes[1] == "160000" and (repo / path / ".git").exists():
            print(f"warning: {sub}/{path} (a nested submodule) stays where it was although {sub} "
                  f"now records another commit for it; nested submodules are not synced.\n"
                  f'To follow: git -C "{repo}" submodule update -- "{path}"', file=sys.stderr)


# ----------------------------------------------------------------- commands

def cmd_status(args: argparse.Namespace) -> int:
    slots = list_slots()
    main = HOST.main_branch
    unpushed = git(["rev-list", "--count", f"{HOST.remote}/{main}..{main}"], cwd=main_dir(),
                   check=False).stdout.strip()
    head = main_branch() or "DETACHED"
    print(f"main: {main_dir()}  {head} {git(['rev-parse', '--short', 'HEAD'], cwd=main_dir()).stdout.strip()}"
          f"  unpushed {main}: {unpushed or '?'}")
    if not slots:
        print(f"no slot ({HOST.command} new)")
    for s in slots:
        unlisted = "" if is_permitted(s.path) else ", NOT in additionalDirectories"
        print(f"{s.name:<5} {str(s.path):<32} {s.branch or '(detached)':<10} {describe(s)}{unlisted}")
    nxt = git_dir() / NEXT_FILE
    if nxt.is_file():
        print(f"next WorktreeCreate -> {nxt.read_text().strip()}")
    return 0


def cmd_new(args: argparse.Namespace) -> int:
    slots = list_slots()
    number = args.number or (max((s.number for s in slots), default=0) + 1)
    if any(s.number == number for s in slots):
        raise Fail(f"wt{number} already exists")
    path = slot_path(number)
    branch = slot_name(number)
    slot = Slot(number, path, branch, "", False)
    # Excluded before the path is even looked at, and held through creation,
    # submodules, shared links and install: a worker allowed to write a
    # parent of the new slot, or a turn started once `worktree add` made the
    # folder, would otherwise write a tree the steps below treat as fresh.
    with worker_excluded(slot):
        return _new_slot(args, slot)


def _new_slot(args: argparse.Namespace, slot: Slot) -> int:
    path, branch = slot.path, slot.branch
    if path.exists():
        raise Fail(f"{path} exists but is not a registered worktree; move it away first")
    if git(["rev-parse", "--verify", "--quiet", branch], cwd=main_dir(), check=False).returncode == 0:
        pending = unlanded_branch(branch)
        if pending:
            raise unlanded_refusal(branch, pending, main_dir())
        print(f"branch {branch} exists with nothing unlanded; reusing it at {HOST.main_branch}")
        git(["branch", "-f", branch, HOST.main_branch], cwd=main_dir())
        git(["worktree", "add", str(path), branch], cwd=main_dir())
    else:
        git(["worktree", "add", "-b", branch, str(path), HOST.main_branch], cwd=main_dir())
    print(f"created {slot.name} at {path}")
    share_dirs(path)
    init_submodules(slot, all_submodules() if args.submodules == "all" else list(HOST.default_submodules))
    if not args.no_install:
        HOST.install(path, slot.admin_dir())
    code = finish(args)
    for note in HOST.new_notes:
        print(note)
    if not is_permitted(path):
        print(permission_warning(slot.name, path), file=sys.stderr)
    return code


def slot_named(args: argparse.Namespace, *, default: str | None = None) -> str:
    """The slot a command was given, by either spelling: positional (`land wt1`)
    or `--dir wt1`.

    Both exist because callers reach for both: `equalize` grew `--dir` first
    (its default is the current worktree, so the name is optional there), `land`
    took a positional, and `land --dir wt1` typed by habit was refused with
    "unrecognized arguments" three times in two sessions (2026-09-13, decided:
    accept both). One name given both ways must agree; two different names is
    an error, not a silent pick.
    """
    positional = getattr(args, "slot", None)
    option = getattr(args, "dir", None)
    if positional and option and positional != option:
        raise Fail(f"two different slots named: {positional!r} and --dir {option!r}")
    given = positional or option or default
    if not given:
        raise Fail(f"name the slot: `{args.command} wt1` or `{args.command} --dir wt1`")
    return given


@excluding_worker(default=".")
def cmd_equalize(args: argparse.Namespace, slot: Slot) -> int:
    refuse_unless_idle(slot)
    reset_to_main_branch(slot, install=not args.no_install)
    return finish(args)


def cmd_share(args: argparse.Namespace) -> int:
    slot = find_slot(slot_named(args, default="."))
    share_dirs(slot.path)
    return 0


@excluding_worker()
def cmd_land(args: argparse.Namespace, slot: Slot) -> int:
    state = slot_state(slot)
    # The slot's own lagging clones (`SlotState.stale`) move only when it is
    # unclaimed: a claim is an agent that may still run there (a Claude agent
    # registers no live worker), and its checkout must not move under it. A
    # claimed slot keeps the lag until the create hook or `equalize` hands it out.
    stale = [] if state.locked else list(state.stale)
    if state.detached:
        raise Fail(f"{slot.name} is {state.describe()}")
    if not state.clean:
        held = f" ({'; '.join(state.held)})" if state.held else ""
        print(f"warning: {slot.name} has uncommitted changes{held}; they stay behind, only commits land",
              file=sys.stderr)
    if not state.unlanded:
        print(f"{slot.name} has nothing to land")
        update_submodules(slot, stale, strict=False)
        unclaim(slot)
        return finish(args)
    main = HOST.main_branch
    current = main_branch()
    if current != main:
        raise Fail(f"the main checkout has {current or 'a detached HEAD'} checked out, not {main}; "
                   f"`land` fast-forwards the checked-out branch and would land into the wrong one")
    old_main = git(["rev-parse", main], cwd=main_dir()).stdout.strip()
    merge = git(["merge", "--ff-only", slot.branch], cwd=main_dir(), check=False)
    if merge.returncode != 0:
        detail = f"{merge.stdout}{merge.stderr}".rstrip()
        if not is_clean(main_dir()):
            raise Fail(f"{main} cannot fast-forward to {slot.branch}:\n{detail}\n"
                       f"The main checkout has uncommitted changes that this land would overwrite; "
                       f"commit or stash them there first, then land again.")
        raise Fail(f"{main} cannot fast-forward to {slot.branch}:\n{detail}\n"
                   f"{main} moved on. Rebase the slot first: git -C {slot.path} rebase {main} "
                   f"(then re-run the gates there), and land again.")
    print(f"{main} -> {git(['rev-parse', '--short', main], cwd=main_dir()).stdout.strip()} "
          f"({len(state.unlanded)} commit(s) from {slot.name})")
    try:
        sync_main_submodules(old_main, slot)
    except (Fail, OSError, subprocess.SubprocessError) as error:
        # The main branch has landed: a sync failure must not keep the slot
        # claimed or skip the push.
        print(f"warning: the main checkout's submodules were not synced after the land: {error}",
              file=sys.stderr)
    # Judged before the merge, which moved neither the slot's HEAD nor its submodules.
    update_submodules(slot, stale, strict=False)
    unclaim(slot)
    return finish(args)


def cmd_push(args: argparse.Namespace) -> int:
    return 0 if push_main_branch() else 2


@excluding_worker()
def cmd_next(args: argparse.Namespace, slot: Slot) -> int:
    refuse_foreign(slot)
    # The override travels with the assignment, so the create hook asks
    # again unless this call was told not to (`choose_slot`).
    (git_dir() / NEXT_FILE).write_text(
        f"{slot.name} {IGNORE_LIVE_FLAG}" if args.ignore_live_worker else slot.name)
    follow_gitlinks(slot)
    print(f"next WorktreeCreate -> {slot.name} ({describe(slot)})")
    return 0


@excluding_worker()
def cmd_release(args: argparse.Namespace, slot: Slot) -> int:
    if not slot.locked:
        print(f"{slot.name} is not locked")
        return 0
    unclaim(slot)
    print(f"{slot.name} unlocked (was: {slot.lock_reason or 'no reason'})")
    return 0


@excluding_worker()
def cmd_drop(args: argparse.Namespace, slot: Slot) -> int:
    if slot.path.is_dir():
        refuse_unless_idle(slot)
        # Before deinit too: it empties a submodule's tree the same way.
        detach_outside_links(slot.path)
        # git's own `deinit` follows each submodule's `.git`; `refuse_unless_idle`
        # has just checked every one names the admin the slot registers, and
        # its worker has ended (the assumption named there).
        for sub in initialized_submodules(slot.path):
            top_git(slot.path, ["submodule", "deinit", "-f", "--", sub])
        # --force: git refuses a worktree whose index holds a gitlink even after
        # deinit (measured 2026-09-06); the checks above are what make it safe.
        git(["worktree", "remove", "--force", str(slot.path)], cwd=main_dir())
    else:
        git(["worktree", "prune"], cwd=main_dir())
    if slot.branch:
        git(["branch", "-D", slot.branch], cwd=main_dir())
    print(f"dropped {slot.name}")
    return 0


# -------------------------------------------------------------------- hooks

def hook_input() -> dict:
    raw = sys.stdin.read()
    try:
        return json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        print(f"worktree hook: stdin is not JSON: {raw[:200]!r}", file=sys.stderr)
        return {}


def take_next_assignment() -> str:
    """Atomically consume `.git/wt-pool-next`; '' if nobody set it."""
    nxt = git_dir() / NEXT_FILE
    claimed = nxt.with_name(f"{NEXT_FILE}.{os.getpid()}")
    try:
        os.replace(nxt, claimed)
    except FileNotFoundError:
        return ""
    wanted = claimed.read_text().strip()
    claimed.unlink(missing_ok=True)
    return wanted


def choose_slot(name: str, reason: str,
                hold: Optional[contextlib.ExitStack] = None) -> tuple[Slot, bool, str]:
    """Pick and claim the slot for a create hook: (slot, reset, how); `reset`:
    the slot was reset to the main branch (still to be installed).

    Three policies in order: an explicit `next` assignment (the reviewer/fixer
    of the task that holds the slot is meant to land in it, unlanded commits
    and all, so no reset and the existing claim stands); a slot named by the
    request; the first free slot, reset before it is picked. Claiming is what
    makes concurrent hooks safe.
    A slot missing from `additionalDirectories` (`is_permitted`) is never the
    first free one -- handing it out would park the agent on a permission prompt
    -- and an assignment or a request by name gets it with a warning, because
    somebody chose it on purpose. A slot with a git that cannot be bound
    (`refuse_foreign`) is handed out by no policy: the hook fails, and the
    harness starts no agent. A slot a worker runs in (`worker_excluded`) is
    never free, and an assignment is refused for it too unless its `next` was
    told `IGNORE_LIVE_FLAG`. The exclusion is entered into `hold`, which the
    caller keeps until the slot is ready (without one, until this returns).
    """
    with contextlib.ExitStack() as own:
        return _choose_slot(name, reason, hold if hold is not None else own)


def _choose_slot(name: str, reason: str, hold: contextlib.ExitStack) -> tuple[Slot, bool, str]:
    slots = list_slots()
    wanted, _, override = take_next_assignment().partition(" ")
    if wanted:
        slot = next((s for s in slots if s.name == wanted), None)
        if slot is None:
            print(f"worktree hook: `next` named {wanted}, which does not exist; ignoring", file=sys.stderr)
        else:
            # Excluded first, then inspected: a worker that ran between an
            # earlier look and the guard would have left the look stale.
            hold.enter_context(worker_excluded(slot, ignore=override == IGNORE_LIVE_FLAG))
            # `next` refused it already, but the slot may have changed since.
            refuse_foreign(slot)
            if not slot.locked:
                claim(slot, reason)
            if not is_permitted(slot.path):
                print(permission_warning(slot.name, slot.path), file=sys.stderr)
            return slot, False, f"assigned; {describe(slot)}"
    candidates = [s for s in slots if s.name == name] if SLOT_RE.match(name) else []
    candidates += [s for s in slots if s not in candidates]
    states: list[str] = []
    for slot in candidates:
        if not is_permitted(slot.path):
            if slot.name == name:
                print(permission_warning(slot.name, slot.path), file=sys.stderr)
            else:
                states.append(f"{slot.name}: not in additionalDirectories")
                continue
        with contextlib.ExitStack() as candidate:
            # Excluded before the verdict, which is then read under the guard
            # and held to the end of the hook: a verdict read first could
            # describe a tree a worker or another pool command changed before
            # the guard was taken. Not waited for: a slot another command is
            # acting on is not free now, and the next candidate may be.
            try:
                candidate.enter_context(worker_excluded(slot, wait_s=0))
            except Fail as busy:
                states.append(f"{slot.name}: {busy}")
                continue
            state = slot_state(slot)
            states.append(f"{slot.name}: {state.describe()}")
            if state.free and claim(slot, reason):
                # Reset here, strict, not after the pick: a submodule move git
                # refuses (an ignored file the gitlink tracks, a commit nowhere
                # to fetch) leaves the slot showing ` M <sub>`, as dirty as a
                # lagging clone that cannot follow (`SlotState.stale`), so it
                # is not handed out and the next candidate is tried. The
                # reset itself loses nothing: a free slot has nothing unlanded.
                try:
                    reset_to_main_branch(slot, install=False)
                except Fail as refused:
                    unclaim(slot)
                    why = one_line(str(refused)).removeprefix("error: ")
                    states[-1] = f"{slot.name}: dirty ({why})"
                    continue
                except BaseException:
                    unclaim(slot)
                    raise
                hold.enter_context(candidate.pop_all())
                return slot, True, "by name" if slot.name == name else "first free"
    unlisted = any(not is_permitted(s.path) for s in candidates)
    remedy = (f" A slot skipped as not in additionalDirectories is free once its path is added to "
              f"{PERMISSION_FILES[0].as_posix()} (or hand it out on purpose: `next <slot>`)." if unlisted else "")
    raise Fail(f"worktree hook: no free slot ({'; '.join(states) or 'no slot exists'}). "
               f"`land` the finished task, `release` a stale lock, or `new`.{remedy}")


def cmd_hook_create(args: argparse.Namespace) -> int:
    data = hook_input()
    name = str(data.get("name", ""))
    session = str(data.get("session_id", ""))[:8]
    # Everything but the path goes to stderr: the harness reads stdout as the directory.
    with contextlib.redirect_stdout(sys.stderr), contextlib.ExitStack() as hold:
        slot, reset, how = choose_slot(name, f"{name or 'unnamed'} session {session or '?'}", hold)
        if reset:
            HOST.install(slot.path, slot.admin_dir())
        else:
            follow_gitlinks(slot)
            share_dirs(slot.path, keep_local=True)
        print(f"worktree hook: {name or '(unnamed)'} -> {slot.name} ({how})")
        if HOST.push_by_default:
            push_main_branch()
    print(str(slot.path), file=sys.__stdout__)
    return 0


def cmd_hook_remove(args: argparse.Namespace) -> int:
    """Not observed to fire for a subagent (2026-09-06); the payload's field
    names are unknown, so the slot is matched by VALUE, not by a guessed key."""
    data = hook_input()
    print(f"worktree hook remove: {json.dumps(data)}", file=sys.stderr)
    values = [v for v in data.values() if isinstance(v, str) and Path(v).is_dir()]
    slot = next((s for s in list_slots() if any(norm(s.path) == norm(v) for v in values)), None)
    if slot is None:
        print("worktree hook remove: not a pool slot, nothing to do", file=sys.stderr)
        return 0
    with contextlib.redirect_stdout(sys.stderr), contextlib.ExitStack() as hold:
        # Excluded first, then inspected, then reset, all under one guard.
        try:
            hold.enter_context(worker_excluded(slot))
        except Fail as busy:
            print(f"worktree hook remove: {busy}; keeping it as is")
            return 0
        state = slot_state(slot)
        if state.detached or not state.clean or state.unlanded or state.foreign or state.live:
            print(f"worktree hook remove: {slot.name} is {state.describe()}; keeping it as is")
            return 0
        unclaim(slot)
        reset_to_main_branch(slot, install=False, strict=False)
        print(f"worktree hook remove: {slot.name} returned to the pool")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    main, remote = HOST.main_branch, HOST.remote
    # Both flags are always accepted (`wants_push`), so a script written for
    # one host runs on another; only the one that changes something here is
    # listed in the help.
    pushing = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    pushing.add_argument("--no-push", action="store_true",
                         help=f"do not push {main} afterwards" if HOST.push_by_default
                         else argparse.SUPPRESS)
    pushing.add_argument("--push", action="store_true",
                         help=argparse.SUPPRESS if HOST.push_by_default
                         else f"push {main} to {remote} afterwards")
    installing = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    installing.add_argument("--no-install", action="store_true", help=help_text("--no-install"))
    # Every command that names a slot takes it both ways — see `slot_named`.
    naming = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    naming.add_argument("slot", nargs="?", help="wtN, N, or a slot directory")
    naming.add_argument("--dir", help="the same slot, as an option (either spelling works)")
    overriding = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
    overriding.add_argument(IGNORE_LIVE_FLAG, action="store_true",
                            help="act although a live-worker record names the slot "
                                 "(its worker is hung or not this slot's)")

    sub.add_parser("status", allow_abbrev=False, help=f"slot state and unpushed {main}").set_defaults(func=cmd_status)

    p = sub.add_parser("new", allow_abbrev=False, parents=[pushing, installing], help=help_text("new"))
    p.add_argument("number", nargs="?", type=int)
    p.add_argument("--submodules", choices=["default", "all"], default="default",
                   help=help_text("--submodules"))
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("equalize", allow_abbrev=False, parents=[pushing, installing, naming, overriding],
                       help=f"slot branch := {main} (refuses dirty, unlanded or locked slots); "
                            "default slot: the current worktree")
    p.set_defaults(func=cmd_equalize)

    if HOST.shared:
        p = sub.add_parser(HOST.share_command, allow_abbrev=False, parents=[naming], help=help_text("share"))
        p.set_defaults(func=cmd_share)

    p = sub.add_parser("land", allow_abbrev=False, parents=[pushing, naming, overriding],
                       help=f"{main} := slot branch (ff-only), unlock the slot"
                            + (", push" if HOST.push_by_default else ""))
    p.set_defaults(func=cmd_land)

    sub.add_parser("push", allow_abbrev=False, help=f"push {main} to {remote}").set_defaults(func=cmd_push)

    p = sub.add_parser("next", allow_abbrev=False, parents=[naming, overriding], help="which slot the next WorktreeCreate hook hands out")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("release", allow_abbrev=False, parents=[naming, overriding], help="drop a slot's lock (the agent that held it is gone)")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("drop", allow_abbrev=False, parents=[naming, overriding],
                       help=help_text("drop shared" if HOST.shared else "drop"))
    p.set_defaults(func=cmd_drop)

    sub.add_parser("hook-create", allow_abbrev=False, help="Claude Code WorktreeCreate hook").set_defaults(func=cmd_hook_create)
    sub.add_parser("hook-remove", allow_abbrev=False, help="Claude Code WorktreeRemove hook").set_defaults(func=cmd_hook_remove)
    return ap


def _utf8_streams() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(host: Optional[PoolHost] = None) -> int:
    global HOST
    if host is not None:
        HOST = host
    _utf8_streams()
    args = build_parser().parse_args()
    wants_push(args)  # both flags: refused before anything moves
    return args.func(args)


if __name__ == "__main__":
    sys.stderr.write("worktree.py: no host rules given: no shared directories, no install "
                     "step, no default submodules\n")
    sys.exit(main())
