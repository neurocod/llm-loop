#!/usr/bin/env python3
"""One-shot git commit helper for repositories with several agents at work.

Wraps the pitfalls that otherwise take several trial-and-error steps:
 - PowerShell 5.1 mangles embedded quotes in native args, so the message is
   taken from a UTF-8 file (-F) or repeatable -m paragraphs, never re-quoted.
 - `git commit` normally snapshots the WHOLE index, silently including
   unrelated changes someone staged earlier; this script always commits by
   explicit pathspec, so only the listed paths land in the commit and any
   other staged work stays staged.
 - `git add -A` style sweeps are impossible: paths are explicit, and the
   names a host blocks (`CommitHost.blocked_names`) are refused outright.
 - Several agents committing at once (a parallel loop) would otherwise lose
   most of their commits to `.git/index.lock`; the two index-taking commands
   wait the lock out instead of failing.

Usage:
  python -m llm_agents.git.commit -F message.txt path1 path2 ...
  python -m llm_agents.git.commit -m "subject line" -m "body paragraph" path1 ...

The message file/paragraphs are used verbatim (the repo's commit-msg hook may
still trim trailers). After a successful commit, the source file passed with
``-F`` is removed and that cleanup is reported; a failed commit leaves it in
place for a retry, and a message file git tracks is refused outright. Prints
the resulting commit and the remaining working tree state so the caller sees
what was left uncommitted.

What a project adds -- names it refuses, extra checks before staging and
before committing, its wording of `-C` -- is a `CommitHost` handed to `main`
by the project's own entry point. Run directly, the script uses `EMPTY` and
says so on stderr.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from . import check_gitlinks
from .commit_identity import approved_invocation
from ..gitlib import git

git = partial(git, check=False, capture_output=False)


@dataclass(frozen=True)
class CommitHost:
    """One project's rules for the commit helper; every default is the empty
    host's answer. Compared by identity: `EMPTY` is tested for with `is`."""
    # Base names refused as commit paths (a loop's stop sentinel, say).
    blocked_names: frozenset = frozenset()
    # Problems that refuse the commit: each called with the pathspecs in the
    # tree being committed (the cwd), before staging and again right before
    # the commit, since a staging that waited out index.lock leaves time for
    # the checked state to change.
    checks: tuple = ()
    # The help text of -C, where a project names its worktree pool.
    cwd_help: str = "run in this worktree (paths are then relative to it)"


EMPTY = CommitHost()
# The host `main` runs with when it is given none.
HOST: CommitHost = EMPTY

# The one table of this script's options (dest -> spellings): `main` builds its
# parser from it, and a judge of the command line (the Codex bridge's
# `command_commit`) reads it, so neither keeps a copy of the other's list.
OPTION_FLAGS = {
    "file": ("-F", "--file"),
    "message": ("-m", "--message"),
    "cwd": ("-C", "--cwd"),
}

# `git add` and `git commit` both take .git/index.lock, and git does not wait for
# it: a second process fails outright. That happens routinely when several agents
# commit their own paths at once (a parallel model-generation loop), so the two
# index-taking commands retry instead of failing. Measured: 6 simultaneous
# commits, 4 of them lost to the lock before this.
#
# The budget must stay UNDER the agent's own tool timeout (120 s for the Codex
# CLI), and it is the whole process's budget rather than each command's — at 120 s
# per command a contended `add` plus `commit` could block for four minutes. Both
# halves of that mattered: a helper that outlives the timeout looks hung, and a
# hunt for the hung process is what twice ended with the agent issuing
# Stop-Process on the long-lived python it found — its own loop driver, killing
# the whole run (2026-08-20 13:02, 2026-08-21 02:44). Returning a diagnosed
# failure inside the window is what keeps this a retry instead of a manhunt.
LOCK_WAIT_S = 90.0
# Contention is a lock file that EXISTS: git's lockfile.c says "File exists"
# and "Another git process seems to be running" for any held .lock. A lock git
# may not CREATE (`Permission denied` / `Access is denied`: a sandboxed Codex
# worker, whose slot index lives under the main repository's .git/worktrees/)
# is no contention — waiting it out for the whole budget and then blaming
# "another agent" sent the caller to re-run the same refused command (review
# 2026-11-03), so it is reported at once.
LOCK_SIGNATURES = ("Another git process", "File exists")
LOCK_DENIED = ("Permission denied", "Access is denied")


def _lock_denied(stderr: str) -> bool:
    """Whether git failed because it may not create a lock file at all."""
    return ".lock" in stderr and any(sig in stderr for sig in LOCK_DENIED)

# One deadline for the invocation, latched at import: this helper is started for
# exactly one commit and exits after it.
_LOCK_DEADLINE = time.monotonic() + LOCK_WAIT_S


def run_indexing(args: list[str], identity: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run an index-taking git command, retrying while the index lock is held.

    Only the lock is retried — a real failure (bad pathspec, rejecting hook,
    nothing to commit) returns on the first try, as it should. Output is captured
    so it can be inspected, then replayed to the caller's streams unchanged.
    """
    # Do not rediscover the index after the identity check. Even a later
    # junction swap must never redirect a write into another repository.
    git_args = (["--git-dir", identity["admin"], "--work-tree", identity["root"], *args]
                if identity is not None else args)
    delay = 0.2
    while True:
        proc = git(git_args, capture_output=True)
        stderr = proc.stderr or ""
        if proc.returncode != 0 and _lock_denied(stderr):
            print(f"error: `git {args[0]}` cannot write the index: creating its "
                  f".lock file was denied, so nothing was committed. That is "
                  f"not contention: this process may not write the repository "
                  f"(not escalated / sandboxed?). Run it outside the sandbox.",
                  file=sys.stderr)
            break
        blocked = (proc.returncode != 0 and ".lock" in stderr
                   and any(sig in stderr for sig in LOCK_SIGNATURES))
        if not blocked:
            break
        if time.monotonic() >= _LOCK_DEADLINE:
            # Say what this is, because the alternative reading — "the helper is
            # stuck, find it and kill it" — has already cost two runs.
            print(f"error: another process held .git/index.lock for the whole "
                  f"{LOCK_WAIT_S:.0f}s budget, so `git {args[0]}` never ran and "
                  f"nothing was committed. That is contention, not a hang: "
                  f"another agent is committing. Re-run this same command in a "
                  f"minute. Do not kill any process over it.", file=sys.stderr)
            break
        # Jitter, or two waiters wake together and collide again.
        time.sleep(delay + random.uniform(0, delay))
        delay = min(delay * 2, 3.0)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc


def is_staged_deletion(path: str) -> bool:
    """True if `path` names something already removed via `git rm`.

    Such a path is in neither the working tree nor the index, so `git add`
    rejects it as an unmatched pathspec -- while `git commit -- <path>` still
    records the deletion happily. Matching is delegated to git, so directory
    pathspecs and Windows backslashes work as they do everywhere else.
    """
    staged = git(["diff", "--cached", "--name-only", "--diff-filter=D", "--", path],
                 capture_output=True)
    return staged.returncode == 0 and bool(staged.stdout.strip())


def toplevel() -> Optional[Path]:
    """The resolved root of the working tree commit.py runs in."""
    proc = git(["rev-parse", "--show-toplevel"], capture_output=True)
    return Path(proc.stdout.strip()).resolve() if proc.returncode == 0 else None


def message_matches_pathspecs(message_file: Path, paths: list[str]) -> bool:
    """Whether git would include the ``-F`` source in the requested paths."""
    root = toplevel()
    if root is None:
        return False
    relatives = []
    for candidate in (message_file.absolute(), message_file.resolve()):
        try:
            relatives.append(candidate.relative_to(root).as_posix())
        except ValueError:
            pass
    if not relatives:
        return False

    message_candidates = {message_file.absolute(), message_file.resolve()}
    for pathspec in paths:
        # Git magic/globs are answered by ls-files below. Plain paths also get a
        # physical comparison so junction/alias spellings cannot evade it.
        if pathspec.startswith(":") or any(c in pathspec for c in "*?["):
            continue
        candidate = Path(pathspec)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        path_candidates = {candidate.absolute(), candidate.resolve()}
        for path_candidate in path_candidates:
            for message_candidate in message_candidates:
                if (message_candidate == path_candidate
                        or path_candidate in message_candidate.parents):
                    return True

    matched = git(
        ["ls-files", "-co", "--full-name", "-z", "--", *paths],
        capture_output=True,
    )
    if matched.returncode != 0:
        return False
    wanted = {os.path.normcase(relative) for relative in relatives}
    return any(os.path.normcase(name) in wanted
               for name in matched.stdout.split("\0") if name)


def tracked_message_file(message_file: Path) -> Optional[Path]:
    """The spelling of the ``-F`` source that a git repository tracks, or None.

    This helper DELETES the message file after a successful commit, so a tracked
    one (`-F CLAUDE.md`) would lose the file and every uncommitted edit in it
    (review 2026-11-03). Asked of the repository holding the file, not the one
    committed in: `-C <slot> -F <main tree file>` deletes just the same. Both
    spellings are asked, as in `message_matches_pathspecs`, so a junction alias
    of a tracked file is caught too; a file in no repository, or in `.git`
    itself, is untracked.
    """
    for candidate in dict.fromkeys((message_file.absolute(),
                                    message_file.resolve())):
        listed = git(["ls-files", "-z", "--", f":(literal){candidate.name}"],
                     cwd=candidate.parent, capture_output=True)
        if listed.returncode == 0 and listed.stdout.strip("\0"):
            return candidate
    return None


def host_problems(host: CommitHost, paths: list[str]) -> list[str]:
    """What the host's checks say against committing `paths` now."""
    problems: list[str] = []
    for check in host.checks:
        problems += check(paths)
    return problems


def refused(problems: list[str], aftermath: str) -> bool:
    """Print a check's problems and what state they left; True if any."""
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        print(f"error: {aftermath}", file=sys.stderr)
    return bool(problems)


def file_identity(handle) -> tuple:
    """Stable-enough identity for detecting a replaced or rewritten message."""
    stat = os.fstat(handle.fileno())
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def remove_committed_message_file(
    message_file: Path, expected_bytes: bytes, expected_identity: tuple
) -> None:
    """Remove only the exact ``-F`` source read before the successful commit.

    Moving the current path to a unique quarantine name is atomic. Verification
    and deletion then happen through that private name, so a replacement created
    at the original path can never be deleted in a compare-to-unlink race.
    """
    quarantine = message_file.with_name(
        f".{message_file.name}.commitpy-{os.getpid()}-{time.time_ns()}.tmp")
    try:
        os.replace(message_file, quarantine)
    except OSError as exc:
        print(f"warning: commit succeeded but could not remove commit message "
              f"file {message_file}: {exc}", file=sys.stderr)
        return

    try:
        with quarantine.open("rb") as handle:
            current_bytes = handle.read()
            current_identity = file_identity(handle)
    except OSError as exc:
        state = (f"it remains at {quarantine}"
                 if quarantine.exists() else
                 "the quarantined path is no longer present")
        print(f"warning: commit succeeded but could not verify the quarantined "
              f"commit message file; {state}: {exc}", file=sys.stderr)
        return

    if (current_identity != expected_identity
            or current_bytes != expected_bytes):
        print(f"warning: commit succeeded but commit message file changed while "
              f"git was running; moved without deletion to {quarantine}",
              file=sys.stderr)
        return

    try:
        quarantine.unlink()
    except OSError as exc:
        print(f"warning: commit succeeded but could not remove commit message "
              f"file; it remains at {quarantine}: {exc}", file=sys.stderr)
        return
    if message_file.exists():
        print(f"warning: removed the original commit message file, but another "
              f"file now occupies its path and was preserved: {message_file}",
              file=sys.stderr)
    else:
        print(f"Removed commit message file: {message_file}")


def approved_identity_problems(expected: Optional[dict]) -> list[str]:
    """Match checkout identity at approval and after chdir, before Git writes.

    Git can still re-resolve the work tree during its run, after our last
    recheck. That residual rename window is accepted: this boundary protects
    against accidents and confused workers, not malicious workers racing Git.
    """
    if expected is None:
        return []
    for field, flag in (("root", "--show-toplevel"),
                        ("admin", "--absolute-git-dir"),
                        ("owner", "--show-superproject-working-tree")):
        result = git(["rev-parse", flag], capture_output=True)
        actual = result.stdout.strip()
        wanted = expected[field]
        # Compare captured strings, never re-resolve the approved path: that
        # would follow the very junction replacement this check must detect.
        if result.returncode != 0 or os.path.normcase(os.path.normpath(actual)) != os.path.normcase(os.path.normpath(wanted)):
            return [f"commit.py: approved {field} changed; refusing to write Git"]
    return []


def build_parser(host: CommitHost) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Commit exactly the given paths with a quoting-safe message.",
    )
    parser.add_argument(
        *OPTION_FLAGS["file"], metavar="MSGFILE",
        help="UTF-8 file holding the full commit message",
    )
    parser.add_argument(
        *OPTION_FLAGS["message"], action="append", default=[], metavar="TEXT",
        help="message paragraph; repeat for more paragraphs (alternative to -F)",
    )
    parser.add_argument(*OPTION_FLAGS["cwd"], metavar="DIR", help=host.cwd_help)
    parser.add_argument(
        "paths", nargs="+", metavar="PATH",
        help="files to commit (explicit list; directories allowed)",
    )
    return parser


def _utf8_streams() -> None:
    """Replay git's output as UTF-8 whatever the console claims.

    Windows hands a cp1252 stdout, and git's summary line quotes the SUBJECT: a
    commit message in any non-Latin-1 script then raised UnicodeEncodeError
    from the replay in `run_indexing` -- AFTER the commit was already made. The
    traceback reads exactly like a failed commit, so the caller's next move is
    to run it again (measured: twice in one session, by a caller who then had
    to check `git log` to find out it had worked). errors="replace" because a
    mangled echo of a commit that exists beats a clean traceback about one
    that does. In `main`, not at import: an importer that only reads the
    option table keeps its own streams.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main(host: Optional[CommitHost] = None) -> int:
    host = HOST if host is None else host
    _utf8_streams()
    parser = build_parser(host)
    options = parser.parse_args()
    try:
        expected = approved_invocation(sys.argv[1:])
    except (OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if expected is not None:
        # Execute the wrapper's canonical path, not the raw -C spelling.
        options.cwd = expected["cwd"]
    if options.cwd:
        # The message file is resolved where the caller typed it, BEFORE the
        # move: a relative -F must not be looked up (or, worse, found) in the slot.
        if options.file:
            options.file = str(Path(options.file).resolve())
        os.chdir(options.cwd)

    if refused(approved_identity_problems(expected), "nothing was committed or staged."):
        return 1

    if bool(options.file) == bool(options.message):
        parser.error("provide a message via exactly one of -F or -m")
    message_file = Path(options.file) if options.file else None
    if message_file:
        with message_file.open("rb") as handle:
            message_bytes = handle.read()
            message_identity = file_identity(handle)
        message = message_bytes.decode("utf-8")
    else:
        message = "\n\n".join(options.message)
        message_bytes = None
        message_identity = None
    if not message.strip():
        parser.error("commit message is empty")

    if message_file and message_matches_pathspecs(message_file, options.paths):
        print(f"error: commit message file is covered by the commit pathspecs: "
              f"{message_file}", file=sys.stderr)
        return 1
    tracked = message_file and tracked_message_file(message_file)
    if tracked:
        print(f"error: commit message file {tracked} is tracked by git, and this "
              f"helper deletes its -F file after the commit; write the message "
              f"to a scratch file instead. Nothing was committed or staged.",
              file=sys.stderr)
        return 1

    blocked = [p for p in options.paths if Path(p).name in host.blocked_names]
    if blocked:
        print(f"error: refusing to commit sentinel file(s): {', '.join(blocked)}",
              file=sys.stderr)
        return 1

    if refused(host_problems(host, options.paths),
               "nothing was committed or staged."):
        return 1

    # Stage only the given paths (also stages deletions of tracked files);
    # a typo'd path fails right here instead of after a half-made commit.
    # Paths already removed with `git rm` are skipped: their deletion is
    # staged, so there is nothing left for `add` to match. A typo still fails,
    # since it matches no staged deletion either.
    to_add = [p for p in options.paths
              if Path(p).exists() or not is_staged_deletion(p)]
    if to_add:
        if refused(approved_identity_problems(expected), "nothing was staged."):
            return 1
        add = run_indexing(["add", "--", *to_add], expected)
        if add.returncode != 0:
            return add.returncode

    # After staging, before committing: `git add` turns any directory holding
    # its own .git into a GITLINK without writing .gitmodules, and one such
    # entry makes every `git submodule` subcommand abort repo-wide (see
    # check_gitlinks.py beside this file). Scoped to this commit's own
    # pathspecs so a defect somebody else left in the index does not block an
    # unrelated commit.
    if refused(check_gitlinks.check(options.paths, cwd=os.getcwd()),
               "nothing was committed (the paths above stay staged)."):
        return 1

    # Again, because `add` may have waited out index.lock for a minute and the
    # state a host check reads may have changed meanwhile. `commit -- paths`
    # re-reads the working tree, so this is the last look that counts.
    if refused(host_problems(host, options.paths),
               "nothing was committed (the paths stay staged; a retry "
               "re-stages them from the working tree)."):
        return 1

    # Pathspec commit: only the listed paths are snapshotted, whatever else
    # happens to sit in the index stays staged for its own future commit.
    with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", delete=False) as handle:
        handle.write(message)
        message_path = handle.name
    try:
        if refused(approved_identity_problems(expected), "nothing was committed."):
            return 1
        commit = run_indexing(["commit", "-F", message_path, "--", *options.paths], expected)
    finally:
        Path(message_path).unlink(missing_ok=True)
    if commit.returncode != 0:
        return commit.returncode

    if message_file:
        remove_committed_message_file(
            message_file, message_bytes, message_identity)

    git(["log", "-1", "--oneline", "--stat"])
    status = git(["status", "--short"], capture_output=True)
    if status.stdout.strip():
        print("\nleft uncommitted (as intended):")
        print(status.stdout.rstrip())
    return 0


if __name__ == "__main__":
    sys.stderr.write("commit.py: no host rules given: no host checks run and no "
                     "names are blocked\n")
    sys.exit(main())
