#!/usr/bin/env python3
"""A standing git worktree ("slot") so a second writer can work beside the main checkout.

The unit of occupancy is a TASK, not an agent: the slot hosts the whole pipeline
of one task (implement -> review -> fixes -> refactor pass) and is handed back
once the task's commits are on master. Slots are siblings of the main checkout
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
its notes and help texts -- is a `PoolHost` handed to `main` by the project's
own entry point. Run directly, the script uses `EMPTY` and says so on stderr.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Optional

from ..gitlib import Fail, PUSH_TIMEOUT_S, git
from .slots import SLOT_NAME_RE, slot_dir_name, slot_name, slot_number

# `terminal_prompt=False`: every command here may run from a hook with nobody at
# the keyboard, so git must never ask for credentials (the automatic push in
# `push_master`); `failure="exit"` turns a failed or timed-out command into a
# one-line `error:` instead of a traceback. Both knobs are explained on
# `gitlib.git`.
git = functools.partial(git, errors="replace", terminal_prompt=False, failure="exit")


@dataclass(frozen=True)
class SharedDir:
    """A directory only the main checkout fills, linked into every slot at the
    same relative path, so a slot sees the main checkout's files -- and their
    later additions -- without copying them. `what` names it in messages
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
        out = git(["rev-parse", "--path-format=absolute", "--git-dir"], cwd=self.path).stdout.strip()
        return Path(out)


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
    top = git(["rev-parse", "--show-toplevel"], cwd=target).stdout.strip()
    for s in slots:
        if norm(s.path) == norm(top):
            return s
    if norm(top) == norm(main_dir()):
        raise Fail(f"{top} is the main checkout (master), not a pool slot.\n" + slots_hint(slots))
    raise Fail(f"{top} is a worktree but not a pool slot ({main_dir().name}-wtN)")


def slots_hint(slots: list[Slot]) -> str:
    if not slots:
        return f"No slot exists yet: {HOST.command} new"
    lines = ["Slots:"]
    for s in slots:
        lines.append(f"  {s.name}  {s.path}  {describe(s)}")
    lines.append("Pass one with --dir <path or wtN>")
    return "\n".join(lines)


# ------------------------------------------------------------------ state

def is_clean(path: Path) -> bool:
    # Edits or stray files INSIDE a submodule are that submodule's business, not
    # the slot's work; a gitlink pointing elsewhere than master's is still reported.
    out = git(["status", "--porcelain", "--ignore-submodules=dirty"], cwd=path).stdout
    return out.strip() == ""


def unlanded_branch(branch: str) -> list[str]:
    """`git cherry -v master <branch>` '+' lines: commits whose PATCH is not on master.

    Patch identity, not ancestry: a branch that reached master through a rebase
    or squash has different SHAs but no '+' lines, and may be reset; an
    ancestry check would refuse it forever.
    """
    out = git(["cherry", "-v", "master", branch], cwd=main_dir()).stdout
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

    @property
    def free(self) -> bool:
        return self.exists and not self.detached and not self.locked and self.clean and not self.unlanded

    def describe(self) -> str:
        if not self.exists:
            return "MISSING (directory gone; `git worktree prune`, then `new`)"
        if self.detached:
            return f"DETACHED at {self.head[:9]} (rebase in progress? finish or abort it)"
        parts = []
        if self.locked:
            parts.append(f"locked ({self.lock_reason or 'no reason'})")
        if not self.clean:
            parts.append("dirty")
        if self.unlanded:
            parts.append(f"{len(self.unlanded)} unlanded commit(s)")
        if self.behind:
            parts.append(f"{self.behind} behind master")
        return ", ".join(parts) if parts else "free"


def slot_state(slot: Slot) -> SlotState:
    exists = slot.path.is_dir()
    usable = exists and not slot.detached
    behind = 0
    if usable:
        behind = int(git(["rev-list", "--count", f"{slot.branch}..master"], cwd=main_dir()).stdout.strip() or 0)
    return SlotState(
        exists=exists, detached=slot.detached, locked=slot.locked, lock_reason=slot.lock_reason,
        clean=is_clean(slot.path) if exists else True,
        unlanded=unlanded_branch(slot.branch) if usable else [],
        behind=behind, head=slot.head)


def describe(slot: Slot) -> str:
    return slot_state(slot).describe()


def refuse_unless_idle(slot: Slot, state: SlotState | None = None) -> None:
    """Refuse a slot that is busy or holds work: `equalize` and `drop` both need this."""
    state = state or slot_state(slot)
    if state.detached:
        raise Fail(f"{slot.name} is {state.describe()}")
    if state.locked:
        raise Fail(f"{slot.name} is locked ({state.lock_reason or 'no reason'}): an agent may still be "
                   f"running there. If it is gone: {HOST.command} release {slot.name}")
    if not state.clean:
        status = git(["status", "--short", "--ignore-submodules=dirty"], cwd=slot.path).stdout.rstrip()
        raise Fail(f"{slot.name} has uncommitted changes; commit or discard them first:\n{status}")
    if state.unlanded:
        raise unlanded_refusal(slot.name, state.unlanded, slot.path)


def unlanded_refusal(name: str, pending: list[str], path: Path) -> Fail:
    listing = "\n".join(pending)
    return Fail(
        f"{name} has {len(pending)} commit(s) whose patch is not on master:\n{listing}\n"
        f"Land them first ({HOST.command} land {name}), or if master already carries their "
        f"content in another form, reset by hand: git -C {path} reset --hard master")


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
    for sub in which:
        if (slot.path / sub / ".git").exists():
            fetch_unpushed_gitlinks(slot.path, [sub])
            git(["submodule", "update", "--", sub], cwd=slot.path)
            continue
        args = ["submodule", "update", "--init"]
        if (main_dir() / sub / ".git").exists():
            print(f"initializing submodule {sub} (objects from the main checkout's clone, dissociated)")
            args += ["--reference", str(main_dir() / sub), "--dissociate"]
        else:
            print(f"initializing submodule {sub} (not initialized in the main checkout: full clone)")
        git([*args, "--", sub], cwd=slot.path)


def all_submodules() -> list[str]:
    """Paths of the submodules .gitmodules declares; [] when there are none.

    `-z` prints each entry as `key\\nvalue\\0`, so a path with spaces comes back
    whole (splitting the plain `key value` line on whitespace cut it). Exit 1 is
    git's "no such key", i.e. no submodules -- not an error.
    """
    if not (main_dir() / ".gitmodules").is_file():
        return []
    proc = git(["config", "-z", "--file", ".gitmodules", "--get-regexp", r"^submodule\..*\.path$"],
               cwd=main_dir(), check=False)
    if proc.returncode == 1:
        return []
    if proc.returncode != 0:
        raise Fail(f"cannot read submodule paths from .gitmodules:\n{proc.stdout}{proc.stderr}".rstrip())
    return [entry.partition("\n")[2] for entry in proc.stdout.split("\0") if entry]


# ------------------------------------------------------------------- master

def push_master() -> bool:
    """Push master to origin; a failure is reported and the local result stands.

    Pushing is automatic, not a manual step (the decision this pool was built
    under, 2026-09-06), so every command that touches master pushes unless
    told `--no-push`.
    """
    try:
        proc = git(["push", "origin", "master"], cwd=main_dir(), check=False, timeout=PUSH_TIMEOUT_S)
    except Fail:
        print(f"warning: push of master timed out after {PUSH_TIMEOUT_S} s (the local result stands)",
              file=sys.stderr)
        return False
    text = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        print(f"warning: push of master failed (the local result stands):\n{text}", file=sys.stderr)
        return False
    last = text.splitlines()[-1] if text else ""
    print(f"pushed master to origin: {last}" if last else "pushed master to origin")
    return True


def finish(no_push: bool) -> int:
    """Exit code of a command whose last step is the push: 2 when the push failed."""
    if no_push:
        print("push skipped (--no-push)")
        return 0
    return 0 if push_master() else 2


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
    copying. An existing local directory or a link elsewhere is never
    overwritten. Remove linked slots through cmd_drop: direct Git removal
    follows Windows junctions. Creating an empty source directory lets fresh
    clones prepare slots before the main checkout has filled it.
    """
    relative = Path(shared.path)
    source = main_dir() / relative
    target = path / relative
    if norm(path) == norm(main_dir()):
        raise Fail(f"cannot link {shared.what} in the main checkout to itself")
    linked = target.is_symlink() or is_directory_junction(target)
    if keep_local and not linked and target.is_dir():
        print(f"keeping local {shared.items}: {target}")
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
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # Unlike a Windows symlink, a directory junction needs no administrator
        # privileges or Developer Mode. Both paths are absolute for this API.
        import _winapi
        _winapi.CreateJunction(str(source.resolve()), str(target.absolute()))
    else:
        target.symlink_to(source.resolve(), target_is_directory=True)
    print(f"shared {shared.what} linked: {target} -> {source}")


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

def gitlink(rev: str, sub: str, repo: Path | None = None) -> str:
    """The submodule commit `rev` records at `sub` in `repo` (default: the main
    checkout); '' if `rev` has no gitlink there."""
    out = git(["ls-tree", rev, "--", sub], cwd=repo or main_dir(), check=False).stdout.split()
    return out[2] if len(out) >= 3 and out[1] == "commit" else ""


def has_commit(repo: Path, sha: str) -> bool:
    return git(["cat-file", "-e", f"{sha}^{{commit}}"], cwd=repo, check=False).returncode == 0


def one_line(text: str) -> str:
    """Git's multi-line output folded into one line of a warning."""
    return " ".join(text.split())


def fetch_commit(repo: Path, sha: str, sources: list[str],
                 failures: Optional[list[str]] = None) -> bool:
    """Bring `sha` into `repo` from the first source that has it; True when present.

    The slot's own clone goes first: a gitlink bumped in a slot points at a
    commit that may exist nowhere else yet (the submodule push is a separate
    step). A failing or timed-out fetch is only a miss -- the land already
    happened, so nothing here may abort it -- but not a silent one: each is
    appended to `failures` as "<source>: <why>" for the caller's warning.
    """
    for source in sources:
        if has_commit(repo, sha):
            return True
        try:
            proc = git(["fetch", "--quiet", "--no-tags", source, sha], cwd=repo, check=False,
                       timeout=PUSH_TIMEOUT_S)
            why = (one_line(proc.stderr) or f"exit {proc.returncode}") if proc.returncode else ""
        except Fail as error:      # the timeout
            why = one_line(str(error))
        if why and failures is not None:
            failures.append(f"{source}: {why}")
    return has_commit(repo, sha)


def fetch_unpushed_gitlinks(path: Path, subs: list[str]) -> None:
    """Bring each gitlink commit `path`'s HEAD records into its submodule clone
    from the MAIN checkout's clone, where `git submodule update` cannot get it.

    A landed submodule commit stays local until somebody pushes it, and the
    main checkout's clone is where `land` put it (`sync_main_submodules`). A
    slot's clone fetches from the submodule's origin only, so after
    `equalize` (or a rebase onto such a master) `git submodule update` failed
    with "not our ref" and left the slot showing `M <sub>` (measured
    2026-11-07; the fix by hand was a fetch from the main clone, then the
    update). A commit the clone already has is left alone, and one the main
    clone lacks too is left to the clone's own origin, which `submodule
    update` and `follow_gitlinks` fetch from. A fetch that fails although the
    main clone has the commit is one warning line naming the submodule, the
    commit and the commands that fix it by hand.
    """
    for sub in subs:
        repo = path / sub
        source = main_dir() / sub
        if norm(path) == norm(main_dir()) or not (repo / ".git").exists():
            continue
        sha = gitlink("HEAD", sub, repo=path)
        if not sha or has_commit(repo, sha) or not (source / ".git").exists() or not has_commit(source, sha):
            continue
        failures: list[str] = []
        if fetch_commit(repo, sha, [str(source)], failures):
            print(f"{sub}: fetched {sha[:9]} from the main checkout's clone (not on its origin yet)")
            continue
        why = "; ".join(failures) or "the fetch reported success, yet the commit is missing"
        print(f"warning: {sub}: cannot fetch {sha[:9]} from the main checkout's clone ({why}). "
              f'Fix: git -C "{repo}" fetch "{source}" {sha}, then '
              f'git -C "{path}" submodule update -- "{sub}"', file=sys.stderr)


def follow_gitlinks(slot: Slot) -> None:
    """Move a slot's clean submodules FORWARD to the gitlinks its HEAD records.

    For a slot that keeps its work (`next`, an assigned create hook): after a
    rebase onto a master that bumped a submodule, the slot's clone still
    stands on the old commit. Only a clean clone whose HEAD is an ancestor of
    the recorded gitlink moves, so commits made in the slot's clone and not
    yet recorded are never left behind; anything else is named and kept.

    Never fatal: `next` has recorded its assignment by now, and the create
    hook has claimed the slot -- an abort there consumed the assignment, kept
    the claim and never started the agent. Every miss (a commit nowhere to be
    fetched, a move git refuses, e.g. over an untracked file the gitlink's
    commit adds) is a warning naming the submodule and the command.
    """
    initialized = [sub for sub in all_submodules() if (slot.path / sub / ".git").exists()]
    fetch_unpushed_gitlinks(slot.path, initialized)
    for sub in initialized:
        repo = slot.path / sub
        wanted = gitlink("HEAD", sub, repo=slot.path)
        head = git(["rev-parse", "HEAD"], cwd=repo, check=False).stdout.strip()
        if not wanted or head == wanted:
            continue

        def warn(why: str) -> None:
            print(f"warning: {slot.name}'s {sub} stays at {head[:9]} although its HEAD records "
                  f"{wanted[:9]}: {why}. When that is resolved: "
                  f'git -C "{slot.path}" submodule update -- "{sub}"', file=sys.stderr)

        # Still missing after the main clone's turn: the clone's own origin may
        # have it, which is where `submodule update` would have looked.
        failures: list[str] = []
        if not fetch_commit(repo, wanted, ["origin"], failures):
            warn(f"{wanted[:9]} is not in its clone, and origin does not give it either "
                 f"({'; '.join(failures) or 'no error'})")
            continue
        dirty = git(["status", "--porcelain", "--untracked-files=no"], cwd=repo, check=False).stdout.strip()
        forward = git(["merge-base", "--is-ancestor", head, wanted], cwd=repo, check=False).returncode == 0
        if dirty or not forward:
            warn("it has uncommitted changes" if dirty else "its HEAD is not an ancestor of the gitlink")
            continue
        moved = git(["submodule", "update", "--", sub], cwd=slot.path, check=False)
        if moved.returncode != 0:
            warn(f"submodule update failed: {one_line(moved.stdout + moved.stderr)}")
            continue
        print(f"{slot.name}'s {sub}: {head[:9]} -> {wanted[:9]} (follows the slot's gitlink)")


def reset_to_master(slot: Slot, install: bool, *, strict: bool = True) -> None:
    """The slot's branch := master, dependencies refreshed. Caller has checked it is safe.

    `strict=False` (the hooks): a failed submodule update is a warning, not an
    abort, for the reason `follow_gitlinks` gives.
    """
    git(["reset", "--hard", "master"], cwd=slot.path)
    initialized = [sub for sub in all_submodules() if (slot.path / sub / ".git").exists()]
    if initialized:
        fetch_unpushed_gitlinks(slot.path, initialized)
        updated = git(["submodule", "update", "--", *initialized], cwd=slot.path, check=strict)
        if updated.returncode != 0:
            print(f"warning: {slot.name}'s submodules were not updated: "
                  f"{one_line(updated.stdout + updated.stderr)}. When that is resolved: "
                  f'git -C "{slot.path}" submodule update', file=sys.stderr)
    # Keep legacy slots with their own local copies. Missing directories and
    # existing shared links use the main checkout, even with --no-install.
    share_dirs(slot.path, keep_local=True)
    print(f"{slot.name} is at {git(['rev-parse', '--short', 'HEAD'], cwd=slot.path).stdout.strip()} (= master)")
    if install:
        HOST.install(slot.path, slot.admin_dir())


def sync_main_submodules(old_master: str, slot: Slot) -> None:
    """After a land, move the main checkout's submodules to the gitlinks master now records.

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

    Nested submodules are not synced (one level, like `reset_to_master`):
    when the move shifts the gitlink of an initialized nested submodule, a
    warning names it -- one `diff-tree` per moved submodule, and only on a
    bump, so the check is cheaper than finding out from a stale nested tree.
    """
    new_master = git(["rev-parse", "master"], cwd=main_dir()).stdout.strip()
    for sub in all_submodules():
        repo = main_dir() / sub
        if not (repo / ".git").exists():
            continue
        old, new = gitlink(old_master, sub), gitlink(new_master, sub)
        if not new or old == new:
            continue
        head = git(["rev-parse", "HEAD"], cwd=repo, check=False).stdout.strip()
        if head == new:
            continue
        on_branch = git(["symbolic-ref", "-q", "HEAD"], cwd=repo, check=False).returncode == 0
        move = ["merge", "--ff-only", new] if on_branch else ["checkout", "--detach", new]
        command = f'git -C "{repo}" {" ".join(move)}'

        def warn(reason: str) -> None:
            print(f"warning: {sub} in the main checkout stays at {head[:9]} although master now "
                  f"records {new[:9]}: {reason}.\nWhen that is resolved: {command}", file=sys.stderr)

        dirty = git(["status", "--porcelain", "--untracked-files=no"], cwd=repo, check=False).stdout.strip()
        if dirty:
            warn("it has uncommitted changes")
            continue
        sources = [str(slot.path / sub)] if (slot.path / sub / ".git").exists() else []
        failures: list[str] = []
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
        print(f"{sub}: {head[:9]} -> {new[:9]} (main checkout follows master's gitlink)")
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
    unpushed = git(["rev-list", "--count", "origin/master..master"], cwd=main_dir(), check=False).stdout.strip()
    head = main_branch() or "DETACHED"
    print(f"main: {main_dir()}  {head} {git(['rev-parse', '--short', 'HEAD'], cwd=main_dir()).stdout.strip()}"
          f"  unpushed master: {unpushed or '?'}")
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
    if path.exists():
        raise Fail(f"{path} exists but is not a registered worktree; move it away first")
    branch = slot_name(number)
    if git(["rev-parse", "--verify", "--quiet", branch], cwd=main_dir(), check=False).returncode == 0:
        pending = unlanded_branch(branch)
        if pending:
            raise unlanded_refusal(branch, pending, main_dir())
        print(f"branch {branch} exists with nothing unlanded; reusing it at master")
        git(["branch", "-f", branch, "master"], cwd=main_dir())
        git(["worktree", "add", str(path), branch], cwd=main_dir())
    else:
        git(["worktree", "add", "-b", branch, str(path), "master"], cwd=main_dir())
    slot = Slot(number, path, branch, "", False)
    print(f"created {slot.name} at {path}")
    share_dirs(path)
    init_submodules(slot, all_submodules() if args.submodules == "all" else list(HOST.default_submodules))
    if not args.no_install:
        HOST.install(path, slot.admin_dir())
    code = finish(args.no_push)
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


def cmd_equalize(args: argparse.Namespace) -> int:
    slot = find_slot(slot_named(args, default="."))
    refuse_unless_idle(slot)
    reset_to_master(slot, install=not args.no_install)
    return finish(args.no_push)


def cmd_share(args: argparse.Namespace) -> int:
    slot = find_slot(slot_named(args, default="."))
    share_dirs(slot.path)
    return 0


def cmd_land(args: argparse.Namespace) -> int:
    slot = find_slot(slot_named(args))
    state = slot_state(slot)
    if state.detached:
        raise Fail(f"{slot.name} is {state.describe()}")
    if not state.clean:
        print(f"warning: {slot.name} has uncommitted changes; they stay behind, only commits land",
              file=sys.stderr)
    if not state.unlanded:
        print(f"{slot.name} has nothing to land")
        unclaim(slot)
        return finish(args.no_push)
    current = main_branch()
    if current != "master":
        raise Fail(f"the main checkout has {current or 'a detached HEAD'} checked out, not master; "
                   f"`land` fast-forwards the checked-out branch and would land into the wrong one")
    old_master = git(["rev-parse", "master"], cwd=main_dir()).stdout.strip()
    merge = git(["merge", "--ff-only", slot.branch], cwd=main_dir(), check=False)
    if merge.returncode != 0:
        detail = f"{merge.stdout}{merge.stderr}".rstrip()
        if not is_clean(main_dir()):
            raise Fail(f"master cannot fast-forward to {slot.branch}:\n{detail}\n"
                       f"The main checkout has uncommitted changes that this land would overwrite; "
                       f"commit or stash them there first, then land again.")
        raise Fail(f"master cannot fast-forward to {slot.branch}:\n{detail}\n"
                   f"master moved on. Rebase the slot first: git -C {slot.path} rebase master "
                   f"(then re-run the gates there), and land again.")
    print(f"master -> {git(['rev-parse', '--short', 'master'], cwd=main_dir()).stdout.strip()} "
          f"({len(state.unlanded)} commit(s) from {slot.name})")
    try:
        sync_main_submodules(old_master, slot)
    except (Fail, OSError, subprocess.SubprocessError) as error:
        # Master has landed: a sync failure must not keep the slot claimed or skip the push.
        print(f"warning: the main checkout's submodules were not synced after the land: {error}",
              file=sys.stderr)
    unclaim(slot)
    return finish(args.no_push)


def cmd_push(args: argparse.Namespace) -> int:
    return finish(no_push=False)


def cmd_next(args: argparse.Namespace) -> int:
    slot = find_slot(slot_named(args))
    (git_dir() / NEXT_FILE).write_text(slot.name)
    follow_gitlinks(slot)
    print(f"next WorktreeCreate -> {slot.name} ({describe(slot)})")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    slot = find_slot(slot_named(args))
    if not slot.locked:
        print(f"{slot.name} is not locked")
        return 0
    unclaim(slot)
    print(f"{slot.name} unlocked (was: {slot.lock_reason or 'no reason'})")
    return 0


def cmd_drop(args: argparse.Namespace) -> int:
    slot = find_slot(slot_named(args))
    if slot.path.is_dir():
        refuse_unless_idle(slot)
        # Before deinit too: it empties a submodule's tree the same way.
        detach_outside_links(slot.path)
        for sub in all_submodules():
            if (slot.path / sub / ".git").exists():
                git(["submodule", "deinit", "-f", "--", sub], cwd=slot.path)
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


def choose_slot(name: str, reason: str) -> tuple[Slot, bool, str]:
    """Pick and claim the slot for a create hook: (slot, needs_reset, how).

    Three policies in order: an explicit `next` assignment (the reviewer/fixer
    of the task that holds the slot is meant to land in it, unlanded commits
    and all, so no reset and the existing claim stands); a slot named by the
    request; the first free slot. Claiming is what makes concurrent hooks safe.
    A slot missing from `additionalDirectories` (`is_permitted`) is never the
    first free one -- handing it out would park the agent on a permission prompt
    -- and an assignment or a request by name gets it with a warning, because
    somebody chose it on purpose.
    """
    slots = list_slots()
    wanted = take_next_assignment()
    if wanted:
        slot = next((s for s in slots if s.name == wanted), None)
        if slot is None:
            print(f"worktree hook: `next` named {wanted}, which does not exist; ignoring", file=sys.stderr)
        else:
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
        state = slot_state(slot)
        states.append(f"{slot.name}: {state.describe()}")
        if state.free and claim(slot, reason):
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
    with contextlib.redirect_stdout(sys.stderr):
        slot, needs_reset, how = choose_slot(name, f"{name or 'unnamed'} session {session or '?'}")
        if needs_reset:
            # A free slot may still lag behind master; bring it up before the agent starts.
            reset_to_master(slot, install=True, strict=False)
        else:
            follow_gitlinks(slot)
            share_dirs(slot.path, keep_local=True)
        print(f"worktree hook: {name or '(unnamed)'} -> {slot.name} ({how})")
        push_master()
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
    with contextlib.redirect_stdout(sys.stderr):
        state = slot_state(slot)
        if state.detached or not state.clean or state.unlanded:
            print(f"worktree hook remove: {slot.name} is {state.describe()}; keeping it as is")
            return 0
        unclaim(slot)
        reset_to_master(slot, install=False, strict=False)
        print(f"worktree hook remove: {slot.name} returned to the pool")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    pushing = argparse.ArgumentParser(add_help=False)
    pushing.add_argument("--no-push", action="store_true", help="do not push master afterwards")
    installing = argparse.ArgumentParser(add_help=False)
    installing.add_argument("--no-install", action="store_true", help=help_text("--no-install"))
    # Every command that names a slot takes it both ways — see `slot_named`.
    naming = argparse.ArgumentParser(add_help=False)
    naming.add_argument("slot", nargs="?", help="wtN, N, or a slot directory")
    naming.add_argument("--dir", help="the same slot, as an option (either spelling works)")

    sub.add_parser("status", help="slot state and unpushed master").set_defaults(func=cmd_status)

    p = sub.add_parser("new", parents=[pushing, installing], help=help_text("new"))
    p.add_argument("number", nargs="?", type=int)
    p.add_argument("--submodules", choices=["default", "all"], default="default",
                   help=help_text("--submodules"))
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("equalize", parents=[pushing, installing, naming],
                       help="slot branch := master (refuses dirty, unlanded or locked slots); "
                            "default slot: the current worktree")
    p.set_defaults(func=cmd_equalize)

    if HOST.shared:
        p = sub.add_parser(HOST.share_command, parents=[naming], help=help_text("share"))
        p.set_defaults(func=cmd_share)

    p = sub.add_parser("land", parents=[pushing, naming],
                       help="master := slot branch (ff-only), unlock the slot, push")
    p.set_defaults(func=cmd_land)

    sub.add_parser("push", help="push master to origin").set_defaults(func=cmd_push)

    p = sub.add_parser("next", parents=[naming], help="which slot the next WorktreeCreate hook hands out")
    p.set_defaults(func=cmd_next)

    p = sub.add_parser("release", parents=[naming], help="drop a slot's lock (the agent that held it is gone)")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("drop", parents=[naming],
                       help=help_text("drop shared" if HOST.shared else "drop"))
    p.set_defaults(func=cmd_drop)

    sub.add_parser("hook-create", help="Claude Code WorktreeCreate hook").set_defaults(func=cmd_hook_create)
    sub.add_parser("hook-remove", help="Claude Code WorktreeRemove hook").set_defaults(func=cmd_hook_remove)
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
    return args.func(args)


if __name__ == "__main__":
    sys.stderr.write("worktree.py: no host rules given: no shared directories, no install "
                     "step, no default submodules\n")
    sys.exit(main())
