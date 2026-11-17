"""The git policy for a worker in a linked pool slot: rebasing its own slot.

A slot's index, HEAD and refs live under the main checkout's `.git`
(`.git/worktrees/<slot>`, `.git/objects`, `.git/refs`), outside every
writable root, so `git rebase` from the sandbox fails with "Permission
denied". `_git` grants the few shapes a worker needs to bring its own slot
up to the pool's main branch (`HostPolicy.main_branch`) and to resolve a
conflict on the way; everything else git can do stays declined.

The escalated command runs with the user's full rights; the judge reads it,
it does not rewrite it. What it cannot put on the command line it puts in
the environment the bridge hands every escalated command
(`sandbox.escalation_judge`: `GIT_EDITOR=true`, so `--continue` never waits
for an editor nobody can see).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..git.slots import git_admin, pool_slot
from .command_arguments import CommandContext, _arg_path_refusal
from .command_commit import _commit_path_refusal, _git_names, _nested_repo, _submodule_paths
from .command_paths import PathRefused, resolve_path, within

# The environment every escalated command of a slot worker gets
# (`sandbox.escalation_judge`): `rebase --continue` opens the editor on the
# commit message of a resolved commit, and with no terminal behind the
# command an editor blocks the turn until its timeout. `true` keeps the
# message as it was (git for Windows runs the editor through its sh).
REBASE_ENVIRONMENT = {"GIT_EDITOR": "true"}

# The one-word steps of a rebase in progress.
_STEPS = frozenset({"--continue", "--abort", "--skip"})


def _rebasing(slot: Path) -> bool:
    """Whether a rebase is in progress in `slot`: git keeps its state in
    `rebase-merge/` (the merge backend) or `rebase-apply/` (the apply one)
    in the checkout's own admin directory."""
    admin = git_admin(slot)
    return admin is not None and any(
        (admin / name).is_dir() for name in ("rebase-merge", "rebase-apply"))


def _target_refusal(value: str, cwd: Path, ctx: CommandContext) -> str:
    """`-C` must name the slot's own root. Its submodules are refused: their
    main branch is not the pool's, a slot's submodule checkout sits on a
    detached gitlink commit, and the gitlink is moved by the orchestrator
    after the submodule lands — a rebased submodule would only make the
    pointer the slot records disagree with what it holds."""
    try:
        root = resolve_path(value, cwd)[1]
    except PathRefused as refused:
        return str(refused)
    if root == ctx.workspace:
        return ""
    if within(root, (ctx.workspace,)):
        if _nested_repo(root, ctx.workspace) is not None:
            return (f"git -C {value}: a submodule; the rebase grant covers your "
                    f"slot's own tree only (the orchestrator moves submodules)")
        return f"git -C {value}: name your slot's root {ctx.workspace}"
    if any(pool_slot(folder) for folder in (root, *root.parents)):
        return f"git -C {value}: another slot's tree"
    admin = git_admin(ctx.workspace)
    if admin is not None and within(root, (admin.parent.parent.parent,)):
        return f"git -C {value}: the main checkout"
    return (f"git -C {value}: a path outside your own slot; "
            f"{_arg_path_refusal(value, cwd, ctx) or 'not your own tree'}")


def _rebase_refusal(args: list[str], ctx: CommandContext) -> str:
    """`rebase <main branch>` or one step of a rebase in progress, nothing
    more. Refused on purpose: `-i`/`--interactive`/`--exec`/`-x` — the
    granted command runs OUTSIDE the sandbox, and an interactive todo
    (`exec` lines, a GIT_SEQUENCE_EDITOR) or `--exec` runs arbitrary commands
    there that the judge never read; with no terminal an interactive rebase
    either blocks on the editor or degenerates to a plain one. `--onto`, an
    upstream other than the main branch, and `--autostash` (it rewrites the
    tree's uncommitted work outside the sandbox) likewise, as is every
    strategy option (`-X`, `-s`): a merge rule nobody chose is a silent
    resolution of the conflict the worker should see."""
    main = ctx.policy.main_branch
    if len(args) == 1 and args[0] in _STEPS:
        return ""
    if len(args) == 1 and args[0] == main:
        status = _git_names(ctx.workspace, "status", "--porcelain=v1", "-z",
                            "--untracked-files=no", "--ignore-submodules=all")
        if status is None:
            return "git rebase: git cannot read the slot's status"
        if status:
            return (f"git rebase {main} with uncommitted changes in "
                    f"{status[0][3:]}: commit them first (no autostash)")
        return ""
    spelled = " ".join(args) or "(nothing)"
    return (f"git rebase {spelled}: only `rebase {main}` (the pool's main "
            f"branch) and `rebase --continue|--abort|--skip`, nothing else "
            f"(no -i, --onto, --exec, --autostash, strategy options)")


def _unmerged(slot: Path) -> Optional[set[str]]:
    """The slot's unmerged paths, case-folded relative POSIX names (`git
    ls-files -u`: one record per conflict stage, `<mode> <sha> <stage>\\t<path>`),
    or None when git cannot say."""
    records = _git_names(slot, "ls-files", "-u", "-z", "--full-name")
    if records is None:
        return None
    return {record.partition("\t")[2].casefold() for record in records}


# The options each resolving command may carry: `rm --cached` keeps the
# file on disk, which loses nothing; `-r`, `-f`, `-A`, `-u` and the rest
# stage or delete what no conflict named.
_RESOLVE_OPTIONS = {"add": frozenset(), "rm": frozenset({"--cached"})}


def _resolve_refusal(command: str, args: list[str], run: Path,
                     ctx: CommandContext) -> str:
    """`git add|rm <paths>` resolves a conflict (`rm` the modify/delete
    kind), so only while a rebase is in progress, and every path must be
    one git lists as unmerged in the slot (`_unmerged`): outside a conflict
    neither command stages or deletes anything. An unmerged path is a FILE,
    so a folder, the whole tree and a wildcard never match one; the paths
    are also judged as commit.py's (`_commit_path_refusal`: in the tree, no
    submodule, nothing protected)."""
    if not _rebasing(ctx.workspace):
        return (f"git {command}: only to resolve a conflict while a rebase is "
                f"in progress in your slot; commit through the commit script")
    allowed = _RESOLVE_OPTIONS[command]
    at = 0
    while at < len(args) and args[at].startswith("-") and args[at] != "--":
        if args[at] not in allowed:
            return (f"git {command} {args[at]}: no options but "
                    f"{', '.join(sorted(allowed)) or 'none'}; name the "
                    f"conflicting files")
        at += 1
    paths = args[at + 1:] if args[at:at + 1] == ["--"] else args[at:]
    if not paths:
        return f"git {command} with no paths"
    stray = next((path for path in paths if path.startswith("-")), None)
    if stray is not None and args[at:at + 1] != ["--"]:
        return f"git {command} {stray}: options go before the paths"
    submodules = _submodule_paths(ctx.workspace)
    unmerged = _unmerged(ctx.workspace)
    if submodules is None or unmerged is None:
        return f"git {command}: git cannot read the slot's index"
    for raw in paths:
        why = _commit_path_refusal(raw, run, ctx, submodules)
        if why:
            return f"git {command} {why}"
        relative = resolve_path(raw, run)[1].relative_to(ctx.workspace).as_posix()
        if relative.casefold() not in unmerged:
            return (f"git {command} {raw}: not an unmerged path of the rebase "
                    f"(`git ls-files -u` lists the conflict's files)")
    return ""


# Read-only questions a worker asks while resolving, escalated as well so it
# need not tell which of them the sandbox happens to allow: the options are
# an allowlist, since `diff --output` writes a file and `--ext-diff` runs a
# program.
_READ_OPTIONS = {
    "status": frozenset({"--short", "-s", "--porcelain", "--branch", "-b"}),
    "diff": frozenset({"--name-only", "--name-status", "--diff-filter=U",
                       "--stat", "--cached", "--staged", "--check"}),
    "ls-files": frozenset({"-u", "--unmerged"}),
}


def _read_refusal(command: str, args: list[str], run: Path,
                  ctx: CommandContext) -> str:
    allowed = _READ_OPTIONS[command]
    paths = False
    for arg in args:
        if arg == "--" and not paths:
            paths = True
            continue
        if not paths and arg.startswith("-"):
            if arg not in allowed:
                return (f"git {command} {arg}: only "
                        f"{', '.join(sorted(allowed))}")
            continue
        why = _arg_path_refusal(arg, run, ctx)
        if why:
            return f"git {command} {why}"
        if not within(resolve_path(arg, run)[1], (ctx.workspace,)):
            return f"git {command} {arg}: not in your slot"
    return ""


def _git(args: list[str], cwd: Path, ctx: CommandContext) -> str:
    """`git [-C <own slot>] rebase <main>|--continue|--abort|--skip`,
    `git [-C <own slot>] add|rm <unmerged paths>` during a rebase
    (`_resolve_refusal`) and `status`/`diff`/`ls-files -u` with a few
    read-only options (`_READ_OPTIONS`), from a worker in a
    pool slot (`pool_slot`) on a host that names its main branch
    (`HostPolicy.main_branch`). No other global option: `-c` would let the
    command set `core.editor`, `sequence.editor` or an alias the judge
    never read (the same hole as `-i`, `_rebase_refusal`). Without `-C` the
    cwd's repository is the slot's only when no submodule lies between."""
    # Outside the grant the refusal keeps the dispatcher's words for a program
    # of no shape (`argv_refusal`): logged requests are replayed across changes.
    if not pool_slot(ctx.workspace):
        return ("`git` is not a worker command shape (a worker rebases only "
                f"its own pool slot: `<repo>-wtN`, {ctx.policy.pool_script})")
    if not ctx.policy.main_branch:
        return ("`git` is not a worker command shape (the host names no main "
                "branch to rebase onto)")
    words = args[1:]
    target = None
    if words[:1] == ["-C"]:
        if len(words) < 2:
            return "git -C with no value"
        target, words = words[1], words[2:]
        why = _target_refusal(target, cwd, ctx)
        if why:
            return why
    elif _nested_repo(cwd, ctx.workspace) is not None:
        return (f"git in {_nested_repo(cwd, ctx.workspace)}: a submodule; "
                f"the rebase grant covers your slot's own tree only")
    if not words:
        return "git with no command"
    command, rest = words[0], words[1:]
    if command.startswith("-"):
        return (f"git {command}: no global options but `-C <your slot>` "
                f"(`-c` could set an editor or alias the judge never read)")
    run = ctx.workspace if target is not None else cwd
    if command == "rebase":
        return _rebase_refusal(rest, ctx)
    if command in _RESOLVE_OPTIONS:
        return _resolve_refusal(command, rest, run, ctx)
    if command in _READ_OPTIONS:
        return _read_refusal(command, rest, run, ctx)
    return (f"git {command}: a worker escalates only `git rebase` of its own "
            f"slot, `git add`/`git rm` of a conflict's files during it, and "
            f"`git status`/`diff`/`ls-files -u`")
