"""The rebase broker: a sandboxed worker's one way to rebase its own pool slot.

A linked worktree keeps its index, HEAD and refs in the main checkout's
`.git`, outside the writable roots of a sandboxed agent, so `git rebase`
from the sandbox fails with "Permission denied". The Codex bridge does not
grant `git` itself outside the sandbox: an escalated command runs with the
user's full rights, and a plain `git rebase` there would run whatever the
worker could plant on its way — hooks (`core.hooksPath` names a folder of
the tree the worker writes), a `git.cmd` earlier in the search order,
configuration-driven autostash, `updateRefs`, external diff and textconv
drivers, or an interactive todo's `exec` lines. Instead the bridge grants
this file, run from the BRIDGE'S OWN checkout (outside every writable root,
so the worker cannot edit what runs), and hands it the facts it must not
take from the worker through `ENV`: the slot, the admin directory the main
clone registers for it (`slots.registered_admin`), an absolute git the
bridge resolved from its own PATH, an empty hooks folder it owns, the main
branch.

Every git this broker runs is `_pinned(grant)` (`gitlib.Pinned`): that git,
hooks off, the configuration that widens a rebase forced off, a sanitized
environment, and the slot's repository named explicitly — never found
through the slot's `.git` (`gitlib.Repository`). Every request is checked
by `refusal` — by the bridge's judge when it grants the command, and here
again right before git runs.

Run by path (`python -I <it> <op> ...`): `__main__` puts the `src/` folder
the file lies in — the bridge's own — first on `sys.path`.

Operations: `start` (rebase onto `refs/heads/<main>`), `continue`, `skip`,
`abort`, `add <paths>` / `rm [--cached] <paths>` (only the conflict's
unmerged files), `status`, `diff [--name-only|--stat|--cached]`, `unmerged`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if __name__ == "__main__":
    # `-I` put no script folder on sys.path: this file's own src/ goes first.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from llm_agents.gitlib import Pinned, Repository  # noqa: E402

# The bridge's facts, as JSON: {"workspace", "admin", "git", "hooks", "main"}.
ENV = "LLM_AGENTS_REBASE_BROKER"

# Written by `start` into git's rebase-merge/ (the slot's admin directory,
# outside the worker's roots; git removes the folder when the rebase ends):
# `continue`/`skip` resume only a rebase this broker started, onto the
# commit it recorded. Git writes `interactive` there for a plain rebase too
# (measured 2026-11-18, git 2.53), so that file proves nothing.
MARKER = "llm-agents-broker"

# Configuration forced on every git run here over `gitlib.FORCED`: the
# settings that let a plain rebase do more than replay the slot's commits —
# `updateRefs` moves other branches, `autoStash` rewrites uncommitted work.
# `merge` backend: the only one whose state `_resumable` reads.
_FORCED = ("rebase.updateRefs=false", "rebase.autoStash=false", "rebase.backend=merge")

# A git question or step of a fixture-sized slot takes well under a second
# (measured 2026-11-18); a rebase of a long branch takes longer, and a hung
# git must still give the turn back.
_TIMEOUT_S = 600.0

_STEPS = ("continue", "skip", "abort")
_DIFF_OPTIONS = frozenset({"--name-only", "--name-status", "--stat", "--cached",
                           "--staged", "--check"})
OPERATIONS = ("start", *_STEPS, "add", "rm", "status", "diff", "unmerged")


@dataclass(frozen=True)
class Grant:
    """The bridge's facts for one worker: its slot (resolved), the slot's
    registered admin directory, the absolute git, the empty hooks folder,
    the pool's main branch."""
    workspace: Path
    admin: Path
    git: Path
    hooks: Path
    main: str

    def to_env(self) -> str:
        return json.dumps({"workspace": str(self.workspace), "admin": str(self.admin),
                           "git": str(self.git), "hooks": str(self.hooks),
                           "main": self.main})

    @staticmethod
    def from_env(text: str) -> "Grant":
        data = json.loads(text)
        return Grant(Path(data["workspace"]), Path(data["admin"]), Path(data["git"]),
                     Path(data["hooks"]), data["main"])


def _pinned(grant: Grant) -> Pinned:
    return Pinned(grant.git, grant.hooks, Repository(grant.admin, grant.workspace),
                  _FORCED)


def _run(grant: Grant, *args: str, capture: bool = True) -> subprocess.CompletedProcess:
    # cwd the slot root: the pathspecs `add`/`rm` take are root-relative.
    return subprocess.run(_pinned(grant).argv(["--literal-pathspecs", *args]),
                          cwd=grant.workspace, env=Pinned.environment(),
                          capture_output=capture, text=True, encoding="utf-8",
                          errors="replace", timeout=_TIMEOUT_S,
                          stdin=subprocess.DEVNULL)


def _ask(grant: Grant, *args: str) -> Optional[str]:
    """A read-only git answer, or None when git fails."""
    try:
        done = _run(grant, *args)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode == 0 else None


def _state(grant: Grant) -> Optional[Path]:
    """git's rebase-merge/ of the slot when a rebase is in progress there."""
    admin = grant.admin
    for name in ("rebase-merge", "rebase-apply"):
        if (admin / name).is_dir():
            return admin / name
    return None


def _resumable(grant: Grant, state: Path) -> str:
    """Why `continue`/`skip` may not resume the rebase in `state`, or "":
    this broker started it (`MARKER`, naming the commit it rebased onto,
    still `onto`), and what remains to do is `pick` lines only — an `exec`,
    `break` or any other command would run outside the judge's reading."""
    try:
        marked = (state / MARKER).read_text(encoding="utf-8").strip()
        onto = (state / "onto").read_text(encoding="utf-8").strip()
        todo = (state / "git-rebase-todo").read_text(encoding="utf-8")
    except OSError:
        return "the rebase in progress was not started by this broker"
    if not marked or marked != onto:
        return "the rebase in progress was not started by this broker"
    for line in todo.splitlines():
        word = line.strip().split(" ", 1)[0]
        if word and not word.startswith("#") and word not in ("pick", "p"):
            return f"the rebase's todo holds `{line.strip()}`: only picks resume here"
    return ""


def _folded(name: str) -> str:
    return name.rstrip(" .").casefold()


def _unmerged(grant: Grant) -> Optional[dict[str, set[str]]]:
    """{index pathname: its stage modes} of the slot's unmerged entries."""
    text = _ask(grant, "ls-files", "-u", "-z", "--full-name")
    if text is None:
        return None
    entries: dict[str, set[str]] = {}
    for record in filter(None, text.split("\0")):
        meta, _, name = record.partition("\t")
        entries.setdefault(name, set()).add(meta.split(" ", 1)[0])
    return entries


def _path_refusal(grant: Grant, raw: str, unmerged: dict[str, set[str]],
                  ignorecase: bool, deleting: bool) -> str:
    """One path of `add`/`rm`: the LITERAL root-relative index name of an
    unmerged entry (`/` or `\\` separators, no `.`/`..` parts, nothing
    resolved — a link's target is not its name), not a gitlink (mode
    160000: a submodule, whatever `.gitmodules` says), and on disk no link
    or junction anywhere along it, no nested repository, and a regular file
    or (for `rm`) nothing at all — a folder would stage what it holds."""
    name = raw.replace("\\", "/")
    parts = name.split("/")
    if not name or name.startswith("/") or ":" in name or any(
            part in ("", ".", "..") for part in parts):
        return f"{raw}: name the file as `unmerged` prints it, relative to the slot root"
    if any(char in name for char in "*?["):
        return f"{raw}: a wildcard"
    if any(_folded(part) == ".git" for part in parts):
        return f"{raw}: git's own store"
    match = name if name in unmerged else next(
        (entry for entry in unmerged
         if ignorecase and entry.casefold() == name.casefold()), None)
    if match is None:
        return f"{raw}: not an unmerged path of the rebase (see `unmerged`)"
    if "160000" in unmerged[match]:
        return f"{raw}: a submodule's pointer; the orchestrator moves submodules"
    folder = grant.workspace
    for part in parts:
        folder = folder / part
        if folder.is_symlink() or getattr(os.path, "isjunction", lambda _: False)(folder):
            return f"{raw}: {folder} is a link or junction"
        if folder != grant.workspace / name and (folder / ".git").exists():
            return f"{raw}: {folder} is a nested repository"
    target = grant.workspace / name
    if target.exists() and not target.is_file():
        return f"{raw}: not a regular file on disk"
    if not target.exists() and not deleting:
        return f"{raw}: not on disk; `rm` it to take the deletion"
    return ""


def refusal(grant: Grant, op: str, args: list[str]) -> str:
    """Why the broker must not run `op args` for `grant`, or "": the bridge's
    judge asks it to grant the escalation, the broker again before git runs."""
    if op not in OPERATIONS:
        return f"rebase broker: no operation {op!r} (one of {', '.join(OPERATIONS)})"
    if op == "start":
        if args:
            return "rebase broker start takes nothing: it rebases onto " \
                   f"refs/heads/{grant.main}, plainly"
        if _state(grant) is not None:
            return "rebase broker start: a rebase is already in progress"
        if _ask(grant, "rev-parse", "--verify", "--quiet",
                f"refs/heads/{grant.main}^{{commit}}") is None:
            return f"rebase broker start: no branch refs/heads/{grant.main}"
        status = _ask(grant, "status", "--porcelain=v1", "-z",
                      "--untracked-files=no", "--ignore-submodules=all")
        if status is None:
            return "rebase broker start: git cannot read the slot's status"
        if status.strip("\0"):
            first = status.split("\0")[0][3:]
            return (f"rebase broker start with uncommitted changes in {first}: "
                    f"commit them first (no autostash)")
        return ""
    if op in _STEPS:
        if args:
            return f"rebase broker {op} takes nothing"
        state = _state(grant)
        if state is None:
            return f"rebase broker {op}: no rebase in progress"
        return "" if op == "abort" else _resumable(grant, state)
    if op in ("add", "rm"):
        state = _state(grant)
        if state is None:
            return f"rebase broker {op}: only to resolve a conflict of a rebase in progress"
        why = _resumable(grant, state)
        if why:
            return f"rebase broker {op}: {why}"
        paths = args[1:] if op == "rm" and args[:1] == ["--cached"] else args
        if not paths:
            return f"rebase broker {op} with no paths"
        unmerged = _unmerged(grant)
        if unmerged is None:
            return f"rebase broker {op}: git cannot list the unmerged paths"
        ignorecase = (_ask(grant, "config", "--bool", "core.ignorecase") or "").strip() == "true"
        for raw in paths:
            why = _path_refusal(grant, raw, unmerged, ignorecase, op == "rm")
            if why:
                return f"rebase broker {op} {why}"
        return ""
    if op == "diff":
        stray = next((arg for arg in args if arg not in _DIFF_OPTIONS), None)
        return (f"rebase broker diff {stray}: only {', '.join(sorted(_DIFF_OPTIONS))}"
                if stray else "")
    return f"rebase broker {op} takes nothing" if args else ""


def _index_names(grant: Grant, paths: list[str]) -> list[str]:
    """The index's own spelling of each validated path (case may differ
    under core.ignorecase)."""
    unmerged = _unmerged(grant) or {}
    folded = {name.casefold(): name for name in unmerged}
    return [name if (name := raw.replace("\\", "/")) in unmerged
            else folded.get(name.casefold(), name) for raw in paths]


def execute(grant: Grant, op: str, args: list[str]) -> int:
    """Run a request `refusal` passed: git's exit status."""
    if op == "start":
        onto = (_ask(grant, "rev-parse", f"refs/heads/{grant.main}^{{commit}}") or "").strip()
        done = _run(grant, "rebase", "--no-autostash", f"refs/heads/{grant.main}",
                    capture=False)
        state = _state(grant)
        if state is not None and onto:
            (state / MARKER).write_text(onto + "\n", encoding="utf-8")
        return done.returncode
    if op in _STEPS:
        return _run(grant, "rebase", f"--{op}", capture=False).returncode
    if op == "add":
        return _run(grant, "add", "--", *_index_names(grant, args), capture=False).returncode
    if op == "rm":
        cached = ["--cached"] if args[:1] == ["--cached"] else []
        paths = args[len(cached):]
        return _run(grant, "rm", *cached, "--", *_index_names(grant, paths),
                    capture=False).returncode
    if op == "status":
        return _run(grant, "status", "--short", "--branch", capture=False).returncode
    if op == "diff":
        return _run(grant, "diff", "--no-ext-diff", "--no-textconv", *args,
                    capture=False).returncode
    return _run(grant, "ls-files", "-u", "--full-name", capture=False).returncode


def main(argv: Optional[list[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    text = os.environ.get(ENV)
    if not text:
        print(f"rebase broker: {ENV} is not set; this broker runs only as a "
              f"command the Codex bridge granted", file=sys.stderr)
        return 2
    try:
        grant = Grant.from_env(text)
    except (ValueError, KeyError, TypeError) as error:
        print(f"rebase broker: unreadable {ENV}: {error}", file=sys.stderr)
        return 2
    if not argv:
        print(f"rebase broker: name an operation ({', '.join(OPERATIONS)})",
              file=sys.stderr)
        return 2
    op, args = argv[0], argv[1:]
    why = refusal(grant, op, args)
    if why:
        print(why, file=sys.stderr)
        return 1
    return execute(grant, op, args)


if __name__ == "__main__":
    sys.exit(main())
