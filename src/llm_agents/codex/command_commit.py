"""The commit policy for a worker in a linked pool slot.

The worker commits through the commit broker (`llm_agents.git.commit_broker`,
granted as `python -I <the bridge's own copy> <commit.py's arguments>`,
`COMMIT`), never through a commit script of its own tree: that file, the
kit under its `src/` and the hooks of its `.githooks/` are the worker's to
edit before it asks (the broker's docstring). The commit script is the git
kit's (`llm_agents.git.commit`); the slot rule and the option table are the
kit's own, read here rather than copied.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Optional
from ..git import commit_broker
from ..git.commit import OPTION_FLAGS
from ..git.commit_identity import write_approval
from ..git.slots import git_admin, pool_slot, registered_submodule
from ..gitlib import Pinned, Repository
from .command_arguments import (
    CommandContext,
    _Option,
    _arg_path_refusal,
    _parse_options,
    _value_refusal,
)
from .command_brokers import Broker
from .command_paths import (
    PathRefused,
    protected_part,
    resolve_path,
    within,
)

# The broker file the worker names, as the bridge imported it.
BROKER = Path(commit_broker.__file__).resolve()

# --- committing from a pool slot -------------------------------------------


def _own_root_refusal(value: str, cwd: Path, ctx: CommandContext,
                      identity: Optional[dict] = None) -> str:
    """Allow the slot root or an initialized, directly declared submodule.

    Declaration alone is insufficient: a copied .git file can point at the
    main checkout's index. Require the slot's modules admin tree and git's
    own superproject answer as well; nested repositories are never targets.
    """
    why = _arg_path_refusal(value, cwd, ctx)
    try:
        root = resolve_path(value, cwd)[1]
    except PathRefused as refused:
        return str(refused)
    if not within(root, (ctx.workspace,)):
        if any(pool_slot(folder) for folder in (root, *root.parents)):
            reason = "another slot's submodule or tree"
        else:
            admin = ctx.slot_admin
            reason = ("the main checkout's submodule or tree"
                      if admin and within(root, (admin.parent.parent.parent,))
                      else "a path outside your own slot")
        return f"commit.py -C {value}: {reason}; {why or 'not your own tree'}"
    if why:
        return why
    repository, why = _tree_repository(root, ctx)
    if repository is None:
        return f"commit.py -C {value}: {why}"
    if identity is not None:
        identity.update(root=str(repository.work_tree), admin=str(repository.git_dir),
                        owner="" if root == ctx.workspace else str(ctx.workspace))
    return ""


def _tree_repository(root: Path, ctx: CommandContext) -> tuple[Optional[Repository], str]:
    """The repository a commit in `root` (resolved, inside the slot) writes,
    or None and why: the slot's own (`ctx.slot_admin`, which the bridge read
    from the main clone's worktree registration) or a directly declared,
    initialized submodule's (`slots.registered_submodule`). Never what a
    `.git` under the slot names: a submodule's is writable to the worker
    (`gitlib.Repository`), and git asked from there would run the
    configuration it points at."""
    if ctx.slot_admin is None:
        return None, "the bridge found no admin directory registered for your slot"
    slot = Repository(ctx.slot_admin, ctx.workspace)
    if root == ctx.workspace:
        return slot, ""
    if _nested_repo(root.parent, ctx.workspace) is not None:
        return None, "a nested submodule is not allowed"
    declared = _submodules(slot, ctx)
    if declared is None:
        return None, "git cannot read the slot's .gitmodules"
    # Compared as spelled: a link along `<slot>/<path>` is not that path.
    named = next((entry for entry in declared
                  if os.path.normcase(os.path.normpath(ctx.workspace / entry[1]))
                  == os.path.normcase(str(root))), None)
    if named is None:
        return None, (f"not a declared submodule; commits only in your own tree "
                      f"{ctx.workspace} or its declared submodules")
    admin, why = registered_submodule(ctx.slot_admin, ctx.workspace, named[1], named[0])
    if admin is None:
        return None, f"submodule git directory is not owned by your slot ({why})"
    # Read only to refuse, never to trust: a `.git` naming another
    # repository leaves the worker's own git and the commit disagreeing.
    named_admin = git_admin(root)
    if named_admin is None or named_admin != admin:
        return None, (f"submodule git directory is not owned by your slot (its "
                      f".git names {named_admin}, the slot registers {admin})")
    return Repository(admin, root), ""


def _message_file_refusal(value: str, cwd: Path, ctx: CommandContext) -> str:
    """commit.py's `-F`: a DELETE as well as a read — commit.py removes the
    message file after a successful commit — so it must lie in a writable
    root OUTSIDE the tree (the scratch root, the operator's `--writable`
    roots). In the tree it may be a tracked file (`-F CLAUDE.md`, review
    2026-11-03: lost with its uncommitted edits; commit.py refuses a tracked
    one too), elsewhere in %TEMP% it is nobody's to delete. Resolved
    against the caller's cwd, as commit.py reads it before its `-C`."""
    why = _value_refusal(value, cwd, ctx, is_path=True)
    if why:
        return why
    resolved = resolve_path(value, cwd)[1]
    outside = tuple(root for root in ctx.roots if root != ctx.workspace)
    if within(resolved, (ctx.workspace,)) or not within(resolved, outside):
        return (f"commit.py -F {value}: the message file goes in your scratch "
                f"directory or a --writable directory outside the tree, since "
                f"commit.py deletes it after the commit")
    return ""


# commit.py's own options (the kit's `OPTION_FLAGS`, which its parser is built
# from), spelled in full: the kit's parser refuses `--fi`/`--mess`/`--cw`
# (`allow_abbrev=False`), so the judge does too, and a
# glued `-Fx` / `-F=x` (argparse reads `=x`) is refused where an option is
# read (`_parse_options`). An option the kit adds without a judge here is a
# KeyError at import, not a flag let through unjudged.
_COMMIT_JUDGES = {
    "file": _Option(takes_value=True, check=_message_file_refusal),
    "message": _Option(takes_value=True, text=True),
    "cwd": _Option(takes_value=True, check=_own_root_refusal)}
_COMMIT_OPTIONS = {flag: _COMMIT_JUDGES[dest]
                   for dest, flags in OPTION_FLAGS.items() for flag in flags}
_CWD_FLAGS = OPTION_FLAGS["cwd"]


def _nested_repo(run: Path, workspace: Path) -> Optional[Path]:
    """The folder between `run` and `workspace` (inclusive of `run`) that
    holds its own `.git` — a submodule checkout, where git would commit
    into the submodule's repository — or None."""
    for folder in (run, *run.parents):
        if folder == workspace:
            return None
        if (folder / ".git").exists():
            return folder
    return None


# A read-only git question of the judge (a folder's contents, .gitmodules):
# 0.03-0.06 s on the largest folders of the tree it was measured on
# (2026-11-03). Bound, so a hung git declines instead of holding the
# turn; far over any measured answer.
_GIT_TIMEOUT_S = 30.0


def _git_names(repository: Repository, *args: str, ctx: CommandContext,
               none_found: int = 0) -> Optional[list[str]]:
    """The NUL-separated records `git <args>` prints in `repository`
    (`-z`), [] when it exits `none_found` (git config's "no such key"), or
    None when git fails or the bridge pinned no git (the caller declines: an
    unanswered question is no "nothing there").

    The commit broker's pinned invocation (`gitlib.Pinned`: the bridge's
    git, never `git` by name, which on Windows is looked up in the bridge's
    cwd; forced settings; no `GIT_*`) bound to `repository`, so the
    question never runs a configuration the worker wrote (item 2432)."""
    if ctx.commit is None:
        return None
    pinned = Pinned(ctx.commit.git, ctx.commit.no_hooks, repository)
    try:
        proc = subprocess.run(pinned.argv(list(args)), cwd=repository.work_tree,
                              env=Pinned.environment(), capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=_GIT_TIMEOUT_S, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == none_found != 0:
        return []
    if proc.returncode != 0:
        return None
    return [name for name in proc.stdout.split("\0") if name]


def _submodules(repository: Repository, ctx: CommandContext) -> Optional[list[tuple[str, str]]]:
    """(name, path) of each `submodule.<name>.path` in the tree's
    `.gitmodules`, or None when git cannot read it. Read as a plain file
    (`--file`, includes off), with git bound to `repository`."""
    gitmodules = repository.work_tree / ".gitmodules"
    if not gitmodules.is_file():
        return []
    records = _git_names(repository, "config", "--file", str(gitmodules), "-z",
                         "--get-regexp", r"^submodule\..*\.path$", ctx=ctx,
                         none_found=1)
    if records is None:
        return None
    return [(key[len("submodule."):-len(".path")], path)
            for key, _, path in (record.partition("\n") for record in records)]


def _folded_parts(relative: str) -> tuple[str, ...]:
    return tuple(part.rstrip(" .").casefold()
                 for part in Path(relative).parts)


def _commit_path_refusal(raw: str, run: Path, ctx: CommandContext,
                         submodules: list[str], repository: Repository) -> str:
    """One pathspec of commit.py: a plain path inside the worker's TREE
    (the scratch root is writable but not committed), outside
    the host's `protected_parts`, not the tree root, not a submodule, and nothing
    below it protected. Git expands a pathspec's wildcards (`[id].ts` also
    matches `i.ts`) and `:` magic (`:/` is the whole repository), so both
    are refused. A folder is allowed below the root: commit.py documents
    it, and a slot holds only its worker's changes — `.` is `git add -A`,
    which sweeps up whatever else the tree holds.

    A path that is, holds or lies in a `.gitmodules` path is refused: `git
    add tools` stages the submodule's pointer as checked out, which after a
    rebase without `git submodule update` is the OLD one (review
    2026-11-03); the orchestrator moves pointers. A folder is refused when
    anything git would commit under it — tracked, untracked or staged as
    deleted — has a protected component anywhere: `app` with an
    `app/.git/x` in it is as protected as `.git` itself
    (review 2026-11-03 granted it by matching at the root only)."""
    if raw.startswith(":"):
        return f"{raw}: git pathspec magic (`:/` is the whole repository)"
    why = _arg_path_refusal(raw, run, ctx)
    if why:
        return why
    resolved = resolve_path(raw, run)[1]
    if not within(resolved, (ctx.workspace,)):
        return (f"{raw} is not in your tree {ctx.workspace}: commit.py "
                f"commits the tree's files")
    relative = resolved.relative_to(ctx.workspace).as_posix()
    parts = _folded_parts(relative)
    if not parts:
        return (f"{raw} is the whole tree (`git add -A`): name the files or "
                f"a folder below the root")
    for submodule in submodules:
        held = _folded_parts(submodule)
        width = min(len(held), len(parts))
        if held[:width] == parts[:width]:
            return (f"{raw} is, holds or lies in the submodule {submodule}: "
                    f"commit.py would stage its pointer as checked out (stale "
                    f"after a rebase without `git submodule update`); the "
                    f"orchestrator moves submodule pointers")
    if resolved.is_file():
        return ""
    spec = f":(literal){relative}"
    below = _git_names(repository, "ls-files", "-z", "--full-name", "-co",
                       "--exclude-standard", "--", spec, ctx=ctx)
    gone = _git_names(repository, "ls-tree", "-r", "-z", "--name-only",
                      "HEAD", "--", spec, ctx=ctx)
    if below is None or gone is None:
        return (f"{raw}: git cannot list what the folder holds, so the judge "
                f"cannot tell whether it holds a protected path")
    for name in (*below, *gone):
        protected = protected_part(Path(name), ctx.policy.protected_parts)
        if protected:
            return (f"{raw} holds {name}, under {protected}/, which agents "
                    f"never write; name the files")
    return ""


def _commit(args: list[str], cwd: Path, ctx: CommandContext) -> str:
    """commit.py's arguments from a worker, as the commit broker (`COMMIT`)
    takes them: only in a pool slot (`pool_slot`) with the bridge's grant in
    place (`CommandContext.commit`), run in the worker's own tree or with
    `-C` selecting a directly declared submodule of that slot, options of
    `_COMMIT_OPTIONS` before the paths, every path by
    `_commit_path_refusal`. Granted, it writes the invocation's approval
    ticket, without which the broker commits nothing."""
    if not pool_slot(ctx.workspace):
        policy = ctx.policy
        return (f"the commit broker: a worker commits only in a pool slot "
                f"(`<repo>-wtN`, {policy.pool_script}); the main tree's master "
                f"is the orchestrator's")
    if ctx.commit is None:
        return f"the commit broker: {ctx.commit_refusal or 'not granted here'}"
    if within(BROKER, ctx.roots):
        return (f"the commit broker {BROKER} lies in your writable roots: the "
                f"bridge must run from a checkout the worker cannot edit")
    parsed, paths, why = _parse_options(
        args, _COMMIT_OPTIONS, "commit.py", "its own, spelled in full",
        cwd, ctx, remainder=True, reject_option_value=True)
    if why:
        return why
    if not paths:
        return "commit.py with no paths"
    stray = next((path for path in paths if path.startswith("-")), None)
    if stray is not None:
        return f"commit.py {stray}: options go before the paths"
    # Repeated uses of one spelling have argparse's last-value semantics.
    # Mixed spellings lose their order in _parse_options, so refuse them.
    targets = [value for key, value in parsed.items() if key in _CWD_FLAGS]
    if all(flag in parsed for flag in _CWD_FLAGS):
        return "commit.py: use only one spelling of -C/--cwd"
    run = resolve_path(targets[-1], cwd)[1] if targets else cwd
    nested = _nested_repo(run, ctx.workspace)
    if nested is not None and not targets:
        return (f"commit.py in {nested}: a repository nested in your tree "
                f"(a submodule); use -C {nested} from {ctx.workspace}")
    tree = run if targets else ctx.workspace
    # The tree's repository first: every git question below is asked of it.
    identity = {}
    why = _own_root_refusal(str(tree), cwd, ctx, identity)
    if why:
        return why
    repository = Repository(Path(identity["admin"]), Path(identity["root"]))
    tree_ctx = replace(ctx, workspace=tree)
    declared = _submodules(repository, ctx)
    if declared is None:
        return "commit.py: git cannot read the tree's .gitmodules"
    submodules = [path for _, path in declared]
    for raw in paths:
        why = _commit_path_refusal(raw, run, tree_ctx, submodules, repository)
        if why:
            return why
    if ctx.commit_approvals is not None:
        identity["cwd"] = str(run)
        write_approval(ctx.commit_approvals, cwd, args, identity)
    return ""


COMMIT = Broker(BROKER, "the commit broker", _commit)
