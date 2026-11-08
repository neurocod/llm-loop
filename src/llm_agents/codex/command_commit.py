"""The commit.py policy for a worker in a linked pool slot.

The commit script is the git kit's (`llm_agents.git.commit`), run through the
host's stand-in, which the host names (`HostPolicy.commit_script`); the slot
rule and the option table are the kit's own, read here rather than copied.
"""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Optional
from ..git.commit import OPTION_FLAGS
from ..git.commit_identity import write_approval
from ..git.slots import git_admin as _git_admin, pool_slot
from .command_arguments import (
    CommandContext,
    _Option,
    _arg_path_refusal,
    _parse_options,
    _value_refusal,
)
from .command_paths import (
    PathRefused,
    protected_part,
    resolve_path,
    within,
)

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
            admin = _git_admin(ctx.workspace)
            reason = ("the main checkout's submodule or tree"
                      if admin and within(root, (admin.parent.parent.parent,))
                      else "a path outside your own slot")
        return f"commit.py -C {value}: {reason}; {why or 'not your own tree'}"
    if why:
        return why
    if root == ctx.workspace:
        if identity is not None:
            identity.update(root=str(root), admin=str(_git_admin(root)), owner="")
        return ""
    if _nested_repo(root.parent, ctx.workspace) is not None:
        return f"commit.py -C {value}: a nested submodule is not allowed"
    declared = _submodule_paths(ctx.workspace)
    if declared is None:
        return "commit.py: git cannot read the slot's .gitmodules"
    if root not in [(ctx.workspace / path).resolve() for path in declared]:
        return (f"commit.py -C {value}: not a declared submodule; commits only "
                f"in your own tree {ctx.workspace} or its declared submodules")
    admin = _git_admin(root)
    slot_admin = _git_admin(ctx.workspace)
    if admin is None or slot_admin is None or not within(admin, (slot_admin / "modules",)):
        return f"commit.py -C {value}: submodule git directory is not owned by your slot"
    toplevel = _git_names(root, "rev-parse", "--show-toplevel")
    if toplevel is None or len(toplevel) != 1 or Path(toplevel[0].strip()).resolve() != root:
        return f"commit.py -C {value}: submodule checkout root does not match -C"
    owner = _git_names(root, "rev-parse", "--show-superproject-working-tree")
    if owner is None or len(owner) != 1 or Path(owner[0].strip()).resolve() != ctx.workspace:
        return f"commit.py -C {value}: not an initialized submodule of your slot"
    if identity is not None:
        identity.update(root=str(root), admin=str(admin), owner=str(ctx.workspace))
    return ""


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


def _git_names(workspace: Path, *args: str,
               none_found: int = 0) -> Optional[list[str]]:
    """The NUL-separated records `git <args>` prints in `workspace` (`-z`),
    [] when it exits `none_found` (git config's "no such key"), or None when
    git fails (the caller declines: an unanswered question is no "nothing
    there")."""
    try:
        proc = subprocess.run(["git", *args], cwd=workspace, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=_GIT_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == none_found != 0:
        return []
    if proc.returncode != 0:
        return None
    return [name for name in proc.stdout.split("\0") if name]


def _submodule_paths(workspace: Path) -> Optional[list[str]]:
    """The `path =` entries of the tree's `.gitmodules`, or None when git
    cannot read it."""
    if not (workspace / ".gitmodules").is_file():
        return []
    records = _git_names(workspace, "config", "--file", ".gitmodules", "-z",
                         "--get-regexp", r"^submodule\..*\.path$", none_found=1)
    return None if records is None else [
        record.partition("\n")[2] for record in records]


def _folded_parts(relative: str) -> tuple[str, ...]:
    return tuple(part.rstrip(" .").casefold()
                 for part in Path(relative).parts)


def _commit_path_refusal(raw: str, run: Path, ctx: CommandContext,
                         submodules: list[str]) -> str:
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
    below = _git_names(ctx.workspace, "ls-files", "-z", "--full-name", "-co",
                       "--exclude-standard", "--", spec)
    gone = _git_names(ctx.workspace, "ls-tree", "-r", "-z", "--name-only",
                      "HEAD", "--", spec)
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
    """The host's commit script (`HostPolicy.commit_script`) from a worker:
    only in a pool slot (`pool_slot`), run in the worker's own tree or with
    `-C` selecting a directly declared submodule of that slot, options of
    `_COMMIT_OPTIONS` before the paths, every path by
    `_commit_path_refusal`."""
    if not pool_slot(ctx.workspace):
        policy = ctx.policy
        return (f"{policy.commit_script}: a worker commits only in a pool slot "
                f"(`<repo>-wtN`, {policy.pool_script}); the main tree's master "
                f"is the orchestrator's")
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
    tree_ctx = replace(ctx, workspace=tree)
    submodules = _submodule_paths(tree)
    if submodules is None:
        return "commit.py: git cannot read the tree's .gitmodules"
    for raw in paths:
        why = _commit_path_refusal(raw, run, tree_ctx, submodules)
        if why:
            return why
    if ctx.commit_approvals is not None:
        identity = {}
        why = _own_root_refusal(str(tree), cwd, ctx, identity)
        if why:
            return why
        identity["cwd"] = str(run)
        write_approval(ctx.commit_approvals, cwd, args, identity)
    return ""
