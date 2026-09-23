#!/usr/bin/env python3
"""Mutate a file, run a command, put the file back -- in ONE shell call.

The thing this replaces is a chain: sed -i the constant, run the test, sed -i
it back. That chain has three problems, and only the first is about typing.

 1. It is a chain, so the permission layer cannot match an allow-rule against
    it and the run parks on a dialog until a human notices. One call to one
    script is one rule.
 2. The restore is just the next link. If the command crashes, hangs and is
    killed, or the run is interrupted, the restore never happens and the file
    is left mutated -- silently, because the shell reports the failure of the
    command, not the state of your tree. Here the restore is a `finally` and
    runs on exceptions and Ctrl-C alike, and the restored bytes are compared
    against the original.
 3. `sed -i` with a pattern that matches nothing edits nothing and exits 0, so
    the test runs against UNMUTATED code and passes. That reads as "the pin
    holds" when nothing was ever pinned. replace_in_file refuses to write
    unless the match count is what you said, so a wrong pattern fails here
    instead of lying there.

--expect-fail is the point of the usual case: verifying that a pin actually
fails without its fix. It inverts the exit code, so a test that passes with
the fix removed -- a pin that pins nothing -- comes back as a failure of THIS
command rather than a line of output somebody has to notice.

Runs from anywhere. Use --cwd for the command's directory instead of `cd X &&`
(on Windows a cd-compound also defeats the permission classifier).

DO NOT PIPE THIS INTO `Select-Object -First N`. That cmdlet stops the upstream
process the moment it has its N objects, and stopping it is stopping THIS one:
the `finally` never runs and the file is left mutated -- the exact silent
half-state the script exists to prevent (paid for once, 2026-08-20). The
restore contract only covers deaths this process survives long enough to
handle. `-Last N` reads the stream to the end and is safe; so is redirecting to
a file and reading that. Same caution for any consumer that closes the pipe
early (`| head` under a shell that turns SIGPIPE into a kill).

Examples are spelled with the BARE script name here and in --help, never with a
directory. There is no one true directory to name: this file ships as a plugin
(`bin/`), and the repository it grew up in reaches it through a stand-in under
`tools/`, so either spelling is a path that does not exist for half the readers
-- the failure the gate's own `tool_path()` exists to avoid. The caller already
knows how they spelled it; argparse says the same by printing `%(prog)s`.

Usage:
  python try_patch.py --file webgame/src/sim/player.ts \\
      --old 'const RISING_SPEED = 1e-4;' --new 'const RISING_SPEED = 0;' \\
      --cwd webgame --expect-fail \\
      -- npx vitest run test/player.rising.test.ts

  # several edits in one go: repeat --file/--old/--new as a triple
  python try_patch.py --file a.ts --old X --new Y \\
                      --file b.ts --old P --new Q -- npm test

  # several edits of the SAME file are fine too: each is applied on top of the
  # previous one, and the file is restored to its pre-run bytes exactly once
  python try_patch.py --file a.ts --old X --new Y \\
                      --file a.ts --old P --new Q -- npm test

  python try_patch.py --selftest   # pins the restore contract, no repo

  python try_patch.py --recover    # undo what a KILLED run left (see JOURNAL_DIR_NAME)

Exit codes of its own: 2 bad edit, 3 restore failed, 4 refused by the journal.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from replace_in_file import (  # noqa: E402  (needs the path line above)
    EditError, apply_replacement, changed_lines, is_uniform_crlf, parse_count,
    read_text, write_text,
)


class Triple(argparse.Action):
    """Collect --file/--old/--new into ordered edit triples.

    argparse has no notion of "these three flags belong together", so each is
    appended to its own list and zipped at the end; a caller who passes an
    unequal number of them gets told, rather than getting the wrong pairing.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        key = option_string.lstrip("-")
        getattr(namespace, "_order").setdefault(key, []).append(values)


def collect_edits(namespace) -> list[tuple[Path, str, str]]:
    order = namespace._order
    files = order.get("file", [])
    olds = order.get("old", [])
    news = order.get("new", [])
    if not files:
        raise EditError("no --file given")
    if not (len(files) == len(olds) == len(news)):
        raise EditError(
            f"--file/--old/--new must come in triples "
            f"(got {len(files)} file, {len(olds)} old, {len(news)} new)")
    return [(Path(f), o, n) for f, o, n in zip(files, olds, news)]


@dataclass
class Touched:
    """Everything needed to put ONE file back, however many edits hit it.

    `original` is the file as it was before this run and is snapshotted the
    FIRST time a path is edited -- never reassigned. That "never" is the whole
    class: with several --file/--old/--new triples aimed at the SAME file, each
    later edit necessarily reads a text that already carries the earlier
    mutations, so keeping a per-EDIT "before" makes the last restore write a
    half-mutated text over the good one. This is not hypothetical -- it shipped:
    three edits of one file printed three "restored" lines and left two of the
    three mutations in the working tree, silently disabling two guards. Hence
    one entry per file, keyed by the resolved path so `a.ts` and `./a.ts` are
    the same file, and hence the selftest below.

    `expected` is what we last wrote, i.e. what the file should still hold when
    the command is done; a mismatch means the command rewrote the file itself
    and is reported before those changes are discarded.
    """

    display: Path   # the path as the caller spelled it, for messages
    original: str
    expected: str


def main() -> int:
    # Looked for before argparse, and only ahead of the bare `--`, so that a
    # command containing the word (`-- grep --selftest`) is still a command.
    argv = sys.argv[1:]
    head = argv[:argv.index("--")] if "--" in argv else argv
    if "--selftest" in head:
        return selftest()

    # `prog` is left to argparse: it is the basename of argv[0], so the help
    # names the script the way the caller reached it (the stand-in sets argv[0]
    # to the real file). Examples below reuse it via %(prog)s rather than
    # hard-coding a directory -- see the header for why there is no right one.
    parser = argparse.ArgumentParser(
        description="Apply edits, run a command, always restore the files.",
        epilog="examples:\n"
               "  python %(prog)s --file src/player.ts \\\n"
               "      --old 'const SPEED = 1e-4;' --new 'const SPEED = 0;' \\\n"
               "      --cwd webgame --expect-fail -- npx vitest run "
               "test/player.test.ts\n"
               "  python %(prog)s --file a.ts --old X --new Y \\\n"
               "                      --file b.ts --old P --new Q -- npm test\n"
               "  python %(prog)s --selftest   # pins the restore contract\n",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", action=Triple, metavar="FILE",
                        help="file to edit (repeatable, with --old/--new)")
    parser.add_argument("--old", action=Triple, metavar="TEXT")
    parser.add_argument("--new", action=Triple, metavar="TEXT")
    parser.add_argument("--regex", action="store_true",
                        help="treat every --old as a regex (MULTILINE)")
    parser.add_argument("--count", type=parse_count, default=1, metavar="N",
                        help="occurrences required per edit, or 'any'")
    parser.add_argument("--cwd", type=Path, default=None, metavar="DIR",
                        help="run the command here (instead of `cd DIR &&`)")
    parser.add_argument("--expect-fail", action="store_true",
                        help="succeed only if the command FAILS (pin check)")
    parser.add_argument("--keep", action="store_true",
                        help="leave the edits in place (debugging this script)")
    parser.add_argument("--selftest", action="store_true",
                        help="run this script's own restore tests and exit")
    parser.add_argument("--recover", action="store_true",
                        help="undo the mutations of killed runs in this tree "
                             "and exit (every run also does it first)")
    parser.add_argument("command", nargs=argparse.REMAINDER, metavar="-- CMD",
                        help="the command to run, after a bare --")
    parser.set_defaults(_order={})
    options = parser.parse_args()

    if options.recover:
        folder = journal_dir(Path.cwd() / "_")
        return 0 if preflight(folder, set()) else 4

    command = options.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("no command given; put it after a bare --")

    try:
        edits = collect_edits(options)
    except EditError as exc:
        print(f"try_patch: {exc}", file=sys.stderr)
        return 2

    # Claim, then scan (RunJournal.claim says why), and both before any original
    # is snapshotted: a killed run's mutation recovered AFTER this run read the
    # file would become this run's "original".
    journal = RunJournal(command, options.cwd)
    try:
        journal.claim([path for path, _, _ in edits])
        targets = {path.resolve() for path, _, _ in edits}
        for folder in journal.folders():
            if not preflight(folder, targets, journal.own_entries()):
                journal.finish(remove=True)
                return 4
    except EditError as exc:
        print(f"try_patch: {exc}", file=sys.stderr)
        journal.finish(remove=True)
        return 2
    except BaseException:
        journal.finish(remove=True)
        raise

    # Every edit is worked out in memory first, and each file is written ONCE
    # with its final text. Written edit by edit, a death between two edits of
    # one file left an intermediate text that was neither the journalled
    # original nor the journalled mutation, so recovery refused it; and a bad
    # pattern now fails before anything is written at all.
    # Keyed by resolved path, so repeated edits of one file share one snapshot
    # of its pre-run bytes; insertion-ordered, so restore reports files in the
    # order the caller named them.
    plans: dict[Path, Touched] = {}
    try:
        for path, old, new in edits:
            key = path.resolve()
            plan = plans.get(key)
            if plan is None:
                original, _ = read_text(path)
                plan = plans[key] = Touched(path, original, original)
            # `before` already carries any earlier edit of the same file, which
            # is what makes stacked edits compose.
            before = plan.expected
            after, hits = apply_replacement(before, old, new, options.regex,
                                            options.count, is_uniform_crlf(before))
            plan.expected = after
            print(f"try_patch: {path}: {hits} occurrence(s) mutated")
            for line in changed_lines(before, after, limit=4):
                print(line)
    except EditError as exc:
        print(f"try_patch: {exc}", file=sys.stderr)
        journal.finish(remove=True)
        return 2
    except BaseException:
        journal.finish(remove=True)
        raise

    # Apply every file before running anything: a half-applied mutation would
    # test a state that neither branch of the comparison describes.
    touched: dict[Path, Touched] = {}
    try:
        for key, plan in plans.items():
            # Journalled, then registered, then written. A death after the
            # journal line leaves an entry for a file still at its original,
            # which recovery shrugs off; a Ctrl-C after registering restores a
            # file that may or may not have been written -- both fine. Any other
            # order leaves a mutation that one of the two does not know about.
            journal.record(plan.display, plan.original, plan.expected)
            touched[key] = plan
            write_text(plan.display, plan.expected)
            invalidate_bytecode(plan.display)
            if os.environ.get(SELFTEST_DIE_ENV) == str(len(touched)):
                # The selftest's stand-in for TaskStop: gone without a `finally`
                # after that many files were written.
                os._exit(SELFTEST_DIE_CODE)
    except EditError as exc:
        print(f"try_patch: {exc}", file=sys.stderr)
        journal.finish(remove=restore(touched))
        return 2
    except BaseException:
        # Ctrl-C between two files would otherwise leave the earlier ones
        # mutated: the `finally` below only covers the command.
        journal.finish(remove=restore(touched))
        raise

    status = 1
    started = False
    restored = True
    try:
        print(f"try_patch: running {' '.join(command)}", flush=True)
        status = subprocess.call([resolve_program(command[0]), *command[1:]],
                                 cwd=options.cwd, shell=False)
        started = True
    except KeyboardInterrupt:
        print("try_patch: interrupted", file=sys.stderr)
        status = 130
    except OSError as exc:
        print(f"try_patch: cannot run the command: {exc}", file=sys.stderr)
        status = 127
    finally:
        if options.keep:
            # Asked for, so not a mutation to undo behind the caller's back.
            journal.finish(remove=True)
            print("try_patch: --keep, files left mutated")
        else:
            restored = restore(touched)
            # A failed restore keeps its entry: the next run gets to judge it.
            journal.finish(remove=restored)

    if not restored:
        # A failed restore outranks whatever the command reported: the tree is
        # wrong, and that is the thing the caller must act on.
        return 3
    if options.expect_fail:
        # A command that never STARTED is not a pin that fired. Windows makes
        # this the likely outcome rather than an exotic one: `npx`, `npm` and
        # friends are .cmd shims, and shell=False cannot launch them, so
        # `--expect-fail -- npx vitest run` used to report "failed as expected"
        # having run no test at all. That reads as a green mutation table --
        # every row of it -- which is worse than no table, and it has already
        # been believed once (2026-08-18, slice U2). Say it instead.
        if not started:
            print("try_patch: the command never RAN, so it proves nothing about "
                  "the pin. On Windows a .cmd shim (npx, npm, tsc) needs its "
                  "real name -- npx.cmd -- because the command is spawned "
                  "without a shell.", file=sys.stderr)
            return 127
        if status == 0:
            print("try_patch: the command PASSED without the fix -- the pin "
                  "does not pin anything", file=sys.stderr)
            return 1
        print(f"try_patch: the command failed as expected (exit {status})")
        return 0
    return status


def resolve_program(name: str) -> str:
    """Turn a bare command name into something CreateProcess can launch.

    The command is spawned WITHOUT a shell (a shell would re-parse the caller's
    quoting), and on Windows that means the name is not resolved through
    PATHEXT: `npx`, `npm`, `tsx`, `vitest` are .cmd shims, so the obvious
    spelling raised WinError 2 and -- before the guard above -- read as a pin
    that fired. Resolving here lets the natural spelling work instead of
    demanding `npx.cmd` from every caller.

    Two traps this walks around, both measured rather than assumed:

    - since 3.12 `shutil.which` can answer with an EXTENSIONLESS file
      (cpython#109590), and `node_modules/.bin` is full of exactly those --
      bash shims CreateProcess cannot start. So a PATHEXT match is preferred
      over which()'s first answer, and only then does its answer stand;
    - once the name resolves to a .cmd, CreateProcess implicitly starts
      cmd.exe, whose escaping rules are not the ones Python quoted for
      (CVE-2024-24576, "BatBadBut"). Harmless here because the arguments are
      the caller's own command line -- but do NOT grow this into passing file
      contents or anything else untrusted through it.

    Anything unresolvable is handed back untouched: the OSError that follows is
    a better message than a guess, and non-Windows needs none of this.
    """
    if os.name != "nt" or os.path.dirname(name):
        return name
    exts = [ext for ext in os.environ.get("PATHEXT", "").split(os.pathsep) if ext]
    for ext in exts:
        found = shutil.which(name + ext)
        if found:
            return found
    return shutil.which(name) or name


def invalidate_bytecode(path: Path) -> None:
    """Drop the cached bytecode of a Python file we just rewrote.

    CPython validates a `.pyc` against the source's mtime IN WHOLE SECONDS plus
    its size. A mutation of the same length -- `= 1` for `= 2`, the commonest
    shape there is -- applied and undone inside one second leaves both fields
    unchanged, so the interpreter happily reuses bytecode for the other text.
    It goes wrong in both directions, and both are silent:

    - the `.pyc` written DURING the mutated run outlives the restore, and the
      NEXT run imports the mutant. Measured: a restored tree failed
      `test_textwidth` on two width pins, and a full run reported 17 failures
      where the mutation had caused 2;
    - the `.pyc` compiled BEFORE the run can survive the mutation, so the
      command tests unmutated code and `--expect-fail` reports "the pin does
      not pin anything" about a pin that was never challenged -- the very lie
      this script exists to prevent.

    Deleting the cache entry is what makes the outcome independent of the
    clock. It is done after mutating and again after restoring, because either
    write can be the one that collides.
    """
    if path.suffix != ".py":
        return
    stem = path.stem
    caches = [*(path.parent / "__pycache__").glob(f"{stem}.*.pyc"),
              *path.parent.glob(f"{stem}.pyc")]
    for cached in caches:
        try:
            cached.unlink()
        except OSError:
            # A held or read-only .pyc is not worth failing the run over: the
            # command below still gets the source we wrote, and the stale entry
            # is the next run's problem, which this line at least keeps rare.
            pass


def restore(touched: "dict[Path, Touched]") -> bool:
    """Put every touched file back. Returns False if any file is left wrong.

    One pass per FILE, never per edit: writing a file more than once here is
    how the old version undid its own restore (see Touched). The write is
    followed by a read-back compared byte for byte, because a write that lands
    on a read-only or externally-held file is exactly the case where a silent
    "restored" line would be a lie.

    A file that no longer holds the mutation we wrote was edited by the command
    itself -- a formatter, a codegen step, a test that rewrites a snapshot. The
    restore below discards that work, which is the right default (the caller
    asked for a temporary mutation, not a commit), but doing it silently would
    lose an edit somebody may have wanted. So it is named.
    """
    ok = True
    for entry in touched.values():
        path = entry.display
        try:
            current, _ = read_text(path)
            # Still the original: a Ctrl-C landed between registering and writing.
            if current not in (entry.expected, entry.original):
                print(f"try_patch: {path} was changed by the command; "
                      "those changes are being discarded with the mutation",
                      file=sys.stderr)
            write_text(path, entry.original)
            invalidate_bytecode(path)
            check, _ = read_text(path)
            if check != entry.original:
                ok = False
                print(f"try_patch: RESTORE FAILED for {path}", file=sys.stderr)
            else:
                print(f"try_patch: restored {path}")
        except EditError as exc:
            ok = False
            print(f"try_patch: RESTORE FAILED for {path}: {exc}",
                  file=sys.stderr)
    return ok


# --------------------------------------------------------------------------
# journal
# --------------------------------------------------------------------------

JOURNAL_DIR_NAME = "TRY_PATCH_MUTATED_FILES_NOT_RESTORED"
"""Where a run records its mutations until they are verifiably restored.

The `finally` covers every death this process lives through; this covers the
rest. A run killed outright -- TaskStop, a tool timeout, a pipe closed early, a
dead agent session -- never reaches the restore, and what it leaves looks like
real code (2026-09-23: `V.check(true, ...)` sat in a probe until a VM suite went
red). So before its first write a run journals the original and the mutated
text, and deletes the entry -- and the directory with its last entry -- after
the restore is read back.

The directory lives in the working-tree root (the nearest `.git`), untracked and
NOT ignored, on purpose: `git status` is what any agent runs first, whatever its
harness, and there this name explains the diff printed beside it. Under `.git/`
only this script would ever look.

A live run holds its entry locked, and the lock -- not the pid, which Windows
reuses and which `os.kill(pid, 0)` would deliver CTRL_C to -- is what tells a
live run from a killed one. A killed run's entry is replayed by the next run in
the same tree, or by --recover: a file that still holds exactly the mutated
text gets its original back; one edited since is refused, because nothing can
tell the mutation from the later edit.

Read from outside as well: `unrestored_mutations` in `tools/git/commit.py` (the
repository this grew up in) loads this file from whatever commit its submodule
sits at, so `JOURNAL_DIR_NAME`, `scan_journal(folder, recover=False)` and the
`Finding` fields it reads are an interface: change them together with commit.py,
in the landing that bumps the submodule.
"""

# Test-only: set to N, die after writing the N-th file, the way TaskStop kills
# -- no `finally`. "1" on a one-file run is "after mutating".
SELFTEST_DIE_ENV = "TRY_PATCH_SELFTEST_DIE_AFTER_MUTATION"
SELFTEST_DIE_CODE = 86

# The lock sits on one byte far past EOF, because Windows locks are mandatory:
# a lock over the content would stop every other process from READING the entry.
LOCK_OFFSET = 1 << 30

# An entry with no complete line is one whose owner died before its first line
# landed -- before any mutation, since every line precedes the write it
# describes -- so it can go. Unless it is this young: then it may be a run
# between creating the file and locking it (POSIX would let us unlink it from
# under that run; Windows would not).
UNREADABLE_GRACE_S = 60.0

# Scans hold each entry locked for a moment too (a neighbour's preflight,
# commit.py), so a refused lock is retried this long before the entry is called
# live. A live run holds its lock for the whole run.
SCAN_LOCK_PATIENCE_S = 0.5


def journal_dir(path: Path) -> Path:
    """The journal directory of the working tree `path` belongs to."""
    here = path.resolve().parent
    for folder in (here, *here.parents):
        # A file, not a directory, in a worktree or a submodule.
        if (folder / ".git").exists():
            return folder / JOURNAL_DIR_NAME
    return here / JOURNAL_DIR_NAME


def _try_lock(handle) -> bool:
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


def _remove_entry(entry: Path) -> None:
    # Windows refuses to delete a file another process has open, and a
    # neighbour's scan (another run, commit.py) holds each entry for a moment.
    # Giving up is safe: an entry whose files all hold their originals is
    # cleared by the next scan, so this must never fail a run that restored.
    for attempt in range(20):
        try:
            entry.unlink(missing_ok=True)
            break
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    else:
        return
    try:
        # Only succeeds once the last entry is gone; a neighbour's entry keeps it.
        entry.parent.rmdir()
    except OSError:
        pass


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@dataclass
class _OpenEntry:
    path: Path
    handle: object
    state: dict


def _append_state(handle, state: dict, path: Path) -> None:
    """Append one whole-state line, all of it, and fsync before returning.

    Everything downstream trusts that a returned call means a complete line on
    disk: the mutation is written next. An unbuffered write may take only part
    of the buffer, and a torn line would leave the previous state standing --
    one that does not name the mutation about to land.
    """
    data = memoryview(json.dumps(state, ensure_ascii=False).encode("utf-8") + b"\n")
    try:
        # The lock seek left the position far past EOF: seek back to the end.
        handle.seek(0, os.SEEK_END)
        while data:
            written = handle.write(data)
            if not written:
                raise OSError("the write made no progress")
            data = data[written:]
        os.fsync(handle.fileno())
    except OSError as exc:
        raise EditError(f"cannot write the journal entry {path}: {exc}") from exc


def _settled(state: dict) -> dict:
    """`state` with nothing left to undo: the last line of a finished entry.

    Written under the lock before it is released, by the owner and by a
    recovery alike, so a reader that opened the entry earlier and gets the lock
    later finds nothing to do -- rather than undoing a mutation somebody has
    made since (POSIX lets a deleted entry stay readable through an old handle).
    """
    return {**state, "targets": [], "files": []}


class RunJournal:
    """This run's entries: one per working tree its files belong to.

    An entry is JSON Lines, append-only, every line the WHOLE state, and the
    last complete line is the truth. Appending is what makes a death at any
    moment safe: a torn line leaves the previous state standing, and each state
    is written before the mutation it describes. A rewrite in place had an
    instant -- between truncate and write -- when the entry said nothing while a
    mutation sat on disk.
    """

    def __init__(self, command: list[str], cwd: "Path | None"):
        self.command = command
        self.cwd = str((cwd or Path.cwd()).resolve())
        self.entries: dict[Path, _OpenEntry] = {}

    def claim(self, targets: "list[Path]") -> None:
        """Name every target in a locked entry, BEFORE the neighbours are scanned.

        Scanning first let two runs started together both find the journal
        empty and mutate one file (reproduced: 30 pairs of 30) -- each then
        tested without its own mutation, and the tree ended clean, so nothing
        said so. Claimed first, such a pair sees each other and both refuse:
        loud, and a retry settles it.
        """
        for path in targets:
            folder = journal_dir(path)
            entry = self.entries.get(folder) or self._open(folder, path)
            key = str(path.resolve())
            if key not in entry.state["targets"]:
                entry.state["targets"].append(key)
        for entry in self.entries.values():
            self._append(entry)

    def folders(self) -> "list[Path]":
        return list(self.entries)

    def own_entries(self) -> "set[Path]":
        return {entry.path for entry in self.entries.values()}

    def record(self, path: Path, original: str, mutated: str) -> None:
        entry = self.entries[journal_dir(path)]  # claimed in claim()
        key = str(path.resolve())
        for item in entry.state["files"]:
            if item["path"] == key:
                item["mutated"] = mutated
                break
        else:
            entry.state["files"].append(
                {"path": key, "original": original, "mutated": mutated})
        self._append(entry)

    def _append(self, entry: _OpenEntry) -> None:
        _append_state(entry.handle, entry.state, entry.path)

    def _open(self, folder: Path, first: Path) -> _OpenEntry:
        # Named after the first victim, so `git status -uall` already says which
        # file is in trouble.
        for attempt in range(100):
            path = folder / f"{first.name}.{os.getpid()}.{attempt}.jsonl"
            try:
                folder.mkdir(exist_ok=True)
                handle = open(path, "x+b", buffering=0)
            except FileExistsError:
                continue
            except OSError:
                # A neighbour's last entry took the folder away between mkdir
                # and open (Windows answers PermissionError while it is pending
                # delete): make it again.
                time.sleep(0.02 * (attempt + 1))
                continue
            if _try_lock(handle):
                break
            # A scan opened it between our open and our lock. Leave it the
            # empty file (it is debris to the next scan) and take another name.
            handle.close()
            _unlink_quietly(path)
        else:
            raise EditError(f"cannot create a journal entry in {folder}")
        state = {
            "what_this_is": (
                "A try_patch run is mutating `targets` and has not restored "
                "`files` yet; each line is the whole state, the last complete "
                "line counts. If its pid is still running, wait for it. If it "
                "was killed, `python try_patch.py --recover` run inside this "
                "tree puts back every file that still holds exactly `mutated` "
                "and deletes this entry. A file edited since is left alone: "
                "repair it by hand against `original`, then delete this entry."),
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cwd": self.cwd,
            "command": self.command,
            "targets": [],
            "files": [],
        }
        entry = _OpenEntry(path, handle, state)
        self.entries[folder] = entry
        return entry

    def finish(self, remove: bool) -> None:
        """Release every entry; `remove` when the tree is known good.

        Known good -- restored, or left mutated because --keep asked for it --
        is written into the entry BEFORE the lock goes. Otherwise a scan landing
        between unlock and unlink would take it for a killed run's entry and
        "recover" the very mutation --keep was asked to keep.
        """
        for entry in self.entries.values():
            if remove:
                try:
                    _append_state(entry.handle, _settled(entry.state), entry.path)
                except EditError as exc:
                    # Nothing better to do on the way out than say so: the
                    # entry still names mutations that are no longer there.
                    print(f"try_patch: {exc}; a neighbour scanning before the "
                          f"entry is gone may undo a --keep", file=sys.stderr)
            _unlock(entry.handle)
            entry.handle.close()
            if remove:
                _remove_entry(entry.path)
        self.entries.clear()


@dataclass
class Finding:
    """One journal entry as another process sees it."""

    entry: Path
    live: bool = False
    # Why the entry could not even be opened; its run is then assumed live.
    error: "str | None" = None
    pid: "int | None" = None
    command: str = "?"
    # Files that may carry this run's mutation right now: for a live run all it
    # claimed, for a dead one those not back to their original.
    pending: list[Path] = field(default_factory=list)
    recovered: list[Path] = field(default_factory=list)
    # (file, why) for each dead-run file recovery could not put back.
    unresolved: "list[tuple[Path, str]]" = field(default_factory=list)


def scan_journal(folder: Path, recover: bool,
                 skip: "set[Path]" = frozenset()) -> list[Finding]:
    """Read every entry in `folder` but `skip`; with `recover`, undo dead ones.

    Changes nothing without `recover`, which is how `commit.py` asks.
    """
    findings: list[Finding] = []
    if not folder.is_dir():
        return findings
    for path in sorted(folder.glob("*.jsonl")):
        if path in skip:
            continue
        try:
            # Writable only to recover: that settles the entry (see _settled).
            handle = open(path, "r+b" if recover else "rb", buffering=0)
        except FileNotFoundError:
            continue  # finished and deleted between the glob and the open
        except OSError as exc:
            findings.append(Finding(path, live=True, error=str(exc)))
            continue
        finding = Finding(path)
        remove = False
        with handle:
            finding.live = not _lock_patiently(handle)
            handle.seek(0)
            state = _last_state(handle.read())
            if state is not None:
                finding.pid = state.get("pid")
                finding.command = " ".join(map(str, state.get("command") or ["?"]))
            if finding.live:
                if state is not None:
                    claimed = [*state["targets"], *(i["path"] for i in state["files"])]
                    finding.pending = [Path(p) for p in dict.fromkeys(claimed)]
            elif state is None:
                remove = recover and _older_than(path, UNREADABLE_GRACE_S)
            else:
                for item in state["files"]:
                    _settle_file(item, finding, recover)
                remove = recover and not finding.pending
                if remove and state["files"]:
                    try:
                        _append_state(handle, _settled(state), path)
                    except EditError:
                        # The files are back; a second recovery through an old
                        # handle would find them equal to `original` and skip.
                        pass
            if not finding.live:
                _unlock(handle)
        if remove:
            _remove_entry(path)
        findings.append(finding)
    return findings


def _lock_patiently(handle) -> bool:
    deadline = time.monotonic() + SCAN_LOCK_PATIENCE_S
    while not _try_lock(handle):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _last_state(blob: bytes) -> "dict | None":
    """The last complete, well-formed line of an entry; None if there is none.

    Complete means newline-terminated: the unterminated tail is dropped even
    when it parses, since only the newline -- written last -- says the whole
    line landed, and a parsing tail proves nothing about the write.
    """
    for line in reversed(blob.split(b"\n")[:-1]):
        try:
            state = json.loads(line.decode("utf-8"))
            if all(isinstance(p, str) for p in state["targets"]) and all(
                    isinstance(i[k], str) for i in state["files"]
                    for k in ("path", "original", "mutated")):
                return state
        except (UnicodeDecodeError, ValueError, KeyError, TypeError):
            continue  # a torn last line: the one before it stands
    return None


def _older_than(path: Path, seconds: float) -> bool:
    try:
        return time.time() - path.stat().st_mtime > seconds
    except OSError:
        return False


def _settle_file(item: dict, finding: Finding, recover: bool) -> None:
    """Judge one file of a dead run; with `recover`, put it back if that is safe."""
    victim = Path(item["path"])
    original = item["original"].encode("utf-8")
    mutated = item["mutated"].encode("utf-8")
    try:
        current = victim.read_bytes()
    except OSError as exc:
        finding.pending.append(victim)
        finding.unresolved.append((victim, f"cannot read it ({exc})"))
        return
    if current == original:
        return  # restored after all, or killed before the mutation landed
    finding.pending.append(victim)
    if current != mutated:
        finding.unresolved.append((
            victim, "holds neither its original nor the mutation -- edited "
                    "after the run died, so the mutation cannot be told from "
                    "the edit"))
        return
    if not recover:
        return
    try:
        victim.write_bytes(original)
    except OSError as exc:
        finding.unresolved.append((victim, f"cannot write it ({exc})"))
        return
    invalidate_bytecode(victim)
    if victim.read_bytes() != original:
        finding.unresolved.append((victim, "the write did not read back"))
        return
    finding.pending.remove(victim)
    finding.recovered.append(victim)


def preflight(folder: Path, targets: "set[Path]",
              own: "set[Path]" = frozenset()) -> bool:
    """Undo killed runs in `folder`; False when this run must not go ahead.

    Refused only over this run's own `targets`: a live run claiming one (each
    run would snapshot the other's mutation as its original, and a restore
    would put a mutation back), or a killed run's mutation on one that cannot
    be undone. Anything else is reported and let be -- `git status` and
    commit.py keep showing it -- so one stuck file does not stop every run in
    the tree. No targets means --recover: anything left undone is a refusal,
    and a clean or live-only journal is said out loud.
    """
    recovering = not targets
    go = True
    findings = scan_journal(folder, recover=True, skip=own)
    for finding in findings:
        who = f"pid {finding.pid}: {finding.command}"
        for victim in finding.recovered:
            print(f"try_patch: RECOVERED {victim}: a try_patch run ({who}) was "
                  f"killed before its restore; its mutation is undone",
                  file=sys.stderr)
        if finding.error:
            print(f"try_patch: cannot open journal entry {finding.entry} "
                  f"({finding.error}); what it holds is unknown, going ahead",
                  file=sys.stderr)
            continue
        clash = [p for p in finding.pending if p.resolve() in targets]
        if finding.live:
            for victim in clash:
                go = False
                print(f"try_patch: {victim} is claimed by a live try_patch run "
                      f"({who}); two runs on one file restore each other's "
                      f"mutations -- wait for it", file=sys.stderr)
            if recovering:
                print(f"try_patch: live run ({who}), left alone: {finding.entry}")
            continue
        for victim, problem in finding.unresolved:
            targeted = victim.resolve() in targets
            if recovering or targeted:
                go = False
            verdict = ("left as it is" if recovering else
                       "refusing, it is a target of this run" if targeted else
                       "not a target of this run, going ahead")
            print(f"try_patch: a killed run ({who}) left a mutation that cannot "
                  f"be undone automatically -- {victim}: {problem}. Repair the "
                  f"file against `original` in {finding.entry}, then delete "
                  f"that entry ({verdict}).", file=sys.stderr)
    if recovering and not findings:
        print(f"try_patch: nothing to recover in {folder}")
    return go


# --------------------------------------------------------------------------
# selftest
#
# There is no test runner under tools/, and this script must keep working when
# the repo it edits does not build, so its regression lives in the script:
# `python try_patch.py --selftest`, no arguments, no fixtures, nothing touched
# outside a temp directory.
#
# Every case runs THIS FILE as a subprocess. That is deliberate: the defect
# these pin -- several edits of one file restoring the file to a half-mutated
# state -- lived in the seam between collecting edits and restoring them, and
# an in-process call to a helper would have stepped straight over it. What is
# asserted is the only thing a caller can see: the exit code, and the bytes on
# disk afterwards.
# --------------------------------------------------------------------------

GUARDS = b"guardA = true;\nguardB = true;\nguardC = true;\n"

# `pass` and a non-zero exit, spelled with this interpreter so the selftest
# needs nothing on PATH.
CMD_OK = ["--", sys.executable, "-c", "pass"]
CMD_FAIL = ["--", sys.executable, "-c", "raise SystemExit(7)"]
CMD_COUNT_FALSE = [
    "--", sys.executable, "-c",
    "import pathlib, sys;"
    "sys.exit(0 if pathlib.Path('victim.cpp').read_text().count('false') == 3"
    " else 9)",
]
CMD_REWRITE = [
    "--", sys.executable, "-c",
    "import pathlib; pathlib.Path('victim.cpp').write_text('the command '"
    "'was here\\n')",
]


class SelftestFailure(Exception):
    pass


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SelftestFailure(message)


def _run(work: Path, *args: str,
         env: "dict[str, str] | None" = None) -> "subprocess.CompletedProcess[str]":
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), *args],
                          cwd=work, capture_output=True, text=True, env=env)


def _victim(work: Path, body: bytes = GUARDS) -> Path:
    path = work / "victim.cpp"
    path.write_bytes(body)
    return path


def _expect_bytes(path: Path, expected: bytes, result) -> None:
    actual = path.read_bytes()
    _expect(actual == expected,
            f"file not restored: {actual!r} != {expected!r}\n"
            f"--- exit {result.returncode}\n{result.stdout}{result.stderr}")


def _flip(name: str) -> list[str]:
    return ["--file", "victim.cpp",
            "--old", f"{name} = true;", "--new", f"{name} = false;"]


def _case_stacked_edits_of_one_file(work: Path) -> None:
    """The incident: three edits of one file left two of them behind."""
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *_flip("guardC"),
                  *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, GUARDS, result)
    _expect_no_journal(work)


def _case_stacked_edits_all_reach_the_command(work: Path) -> None:
    """...and the restore must not be bought by dropping the later edits."""
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *_flip("guardC"),
                  *CMD_COUNT_FALSE)
    _expect(result.returncode == 0,
            f"the command did not see all 3 mutations (exit "
            f"{result.returncode})\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_same_file_spelled_two_ways(work: Path) -> None:
    victim = _victim(work)
    result = _run(work,
                  "--file", "victim.cpp",
                  "--old", "guardA = true;", "--new", "guardA = false;",
                  "--file", str(victim.resolve()),
                  "--old", "guardB = true;", "--new", "guardB = false;",
                  *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_failing_command_restores_and_reports(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *CMD_FAIL)
    _expect(result.returncode == 7,
            f"command exit not passed through: {result.returncode}")
    _expect_bytes(victim, GUARDS, result)


def _case_expect_fail_accepts_a_failing_command(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, "--expect-fail", *_flip("guardA"), *_flip("guardB"),
                  *CMD_FAIL)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_expect_fail_rejects_a_passing_command(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, "--expect-fail", *_flip("guardA"), *_flip("guardB"),
                  *CMD_OK)
    _expect(result.returncode == 1,
            f"a pin that pins nothing must fail: exit {result.returncode}")
    _expect_bytes(victim, GUARDS, result)


def _case_command_rewriting_the_file_is_reported(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"), *CMD_REWRITE)
    _expect("was changed by the command" in result.stderr,
            f"silent discard of the command's own edit\n{result.stderr}")
    _expect(result.stderr.count("was changed by the command") == 1,
            f"one file, one report expected\n{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_missing_pattern_rolls_the_earlier_edits_back(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"),
                  "--file", "victim.cpp",
                  "--old", "guardZ = true;", "--new", "guardZ = false;",
                  *CMD_OK)
    _expect(result.returncode == 2,
            f"a pattern that matches nothing must be loud: {result.returncode}")
    _expect("try_patch: running" not in result.stdout,
            "the command ran on a half-applied mutation")
    _expect_bytes(victim, GUARDS, result)


def _case_crlf_survives_stacked_edits(work: Path) -> None:
    body = GUARDS.replace(b"\n", b"\r\n")
    victim = _victim(work, body)
    result = _run(work, *_flip("guardA"), *_flip("guardC"), *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, body, result)


def _case_unrunnable_command_restores(work: Path) -> None:
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), *_flip("guardB"),
                  "--", "no-such-binary-6f2a1c", "--please")
    _expect(result.returncode == 127,
            f"exit {result.returncode}: {result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_expect_fail_rejects_an_unrunnable_command(work: Path) -> None:
    """The mutation table's own trap: a command that never ran is not a pin.

    Windows makes it the likely spelling error rather than an exotic one --
    npx/npm/tsc are .cmd shims and cannot be spawned without a shell -- so
    accepting "did not start" as "failed as expected" turns a whole table of
    mutations green without running one test.
    """
    victim = _victim(work)
    result = _run(work, "--expect-fail", *_flip("guardA"),
                  "--", "no-such-binary-6f2a1c", "--please")
    _expect(result.returncode == 127,
            f"an unrunnable command must not pass as a fired pin: exit "
            f"{result.returncode}\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_pathext_shim_runs_by_its_bare_name(work: Path) -> None:
    """`-- npx vitest run` must work, and must not pick the bash shim.

    Windows only, because only there is the problem: the command is spawned
    without a shell, so a bare `npx` is unresolvable, and `node_modules/.bin`
    additionally offers an extensionless bash shim that CreateProcess cannot
    start. The fixture is that pair -- `tool.cmd` beside `tool` -- and the exit
    code says which one ran.
    """
    if os.name != "nt":
        return
    binaries = work / "bin"
    binaries.mkdir(exist_ok=True)
    (binaries / "trypatchtool.cmd").write_text("@exit /b 3\n", encoding="ascii")
    (binaries / "trypatchtool").write_text("#!/bin/sh\nexit 4\n", encoding="ascii")
    env = dict(os.environ, PATH=f"{binaries}{os.pathsep}{os.environ.get('PATH', '')}")
    victim = _victim(work)
    result = _run(work, *_flip("guardA"), "--", "trypatchtool", env=env)
    _expect(result.returncode == 3,
            f"the .cmd shim did not run by its bare name: exit "
            f"{result.returncode}\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)


def _case_a_python_edit_leaves_no_stale_bytecode(work: Path) -> None:
    """A same-length mutation must not survive in `__pycache__`.

    The fixture is the collision itself: a module compiled before the run, a
    mutation that changes no byte count, and a whole run inside one second — the
    two fields CPython validates a `.pyc` against. What is asserted is the
    mechanism rather than the race: no cache entry for the edited file outlives
    the run, so no later interpreter can be handed the mutant (nor, on the way
    in, the pre-mutation text). See `invalidate_bytecode`.
    """
    victim = work / "victim.py"
    victim.write_bytes(b"VALUE = 1\n")
    reader = ("import sys; sys.path.insert(0, '.'); import victim; "
              "sys.exit(0 if victim.VALUE == {} else 9)")
    warm = subprocess.run([sys.executable, "-c", reader.format(1)],
                          cwd=work, capture_output=True, text=True)
    _expect(warm.returncode == 0, f"fixture broken: {warm.stderr}")
    cache = work / "__pycache__"
    _expect(any(cache.glob("victim.*.pyc")), "fixture broken: nothing cached")

    result = _run(work, "--file", "victim.py",
                  "--old", "VALUE = 1", "--new", "VALUE = 2",
                  "--", sys.executable, "-c", reader.format(2))

    _expect(result.returncode == 0,
            f"the command did not see the mutation (exit {result.returncode})"
            f"\n{result.stdout}{result.stderr}")
    _expect(not any(cache.glob("victim.*.pyc")),
            "bytecode for the mutated source outlived the restore; the next "
            "run would import the mutant")
    _expect_bytes(victim, b"VALUE = 1\n", result)
    after = subprocess.run([sys.executable, "-c", reader.format(1)],
                           cwd=work, capture_output=True, text=True)
    _expect(after.returncode == 0,
            f"a later run still saw the mutant (exit {after.returncode})"
            f"\n{after.stdout}{after.stderr}")


def _case_keep_leaves_every_mutation(work: Path) -> None:
    """--keep is the escape hatch; it must keep ALL edits, not the last one."""
    victim = _victim(work)
    result = _run(work, "--keep", *_flip("guardA"), *_flip("guardB"), *CMD_OK)
    _expect(result.returncode == 0, f"exit {result.returncode}: {result.stderr}")
    _expect_bytes(victim, b"guardA = false;\nguardB = false;\nguardC = true;\n",
                  result)
    _expect_no_journal(work)


def _expect_no_journal(work: Path) -> None:
    folder = work / JOURNAL_DIR_NAME
    if folder.exists():
        raise SelftestFailure(f"the journal outlived the run: "
                              f"{sorted(p.name for p in folder.iterdir())}")


def _kill_mid_run(work: Path, *edits: str, after_files: int = 1) -> None:
    """A run that dies after writing `after_files` files: no `finally`."""
    env = dict(os.environ, **{SELFTEST_DIE_ENV: str(after_files)})
    killed = _run(work, *edits, *CMD_OK, env=env)
    _expect(killed.returncode == SELFTEST_DIE_CODE,
            f"fixture broken: exit {killed.returncode}\n{killed.stderr}")
    _expect(any((work / JOURNAL_DIR_NAME).glob("*.jsonl")),
            "a killed run left no journal entry")


def _case_killed_run_is_undone_by_the_next(work: Path) -> None:
    """The incident: a killed sweep's mutation looked like real code."""
    victim = _victim(work)
    _kill_mid_run(work, *_flip("guardA"), *_flip("guardB"))
    _expect(victim.read_bytes() != GUARDS, "fixture broken: nothing mutated")
    # The next run must read the RECOVERED text as its original, or its own
    # restore would put the dead run's mutation back.
    result = _run(work, *_flip("guardC"), *CMD_COUNT_FALSE)
    _expect(result.returncode == 9,
            f"the next run saw the dead run's mutations (exit "
            f"{result.returncode})\n{result.stdout}{result.stderr}")
    _expect(result.stderr.count("RECOVERED") == 1, f"not reported:\n{result.stderr}")
    _expect_bytes(victim, GUARDS, result)
    _expect_no_journal(work)


def _case_killed_run_edited_since_is_refused(work: Path) -> None:
    victim = _victim(work)
    _kill_mid_run(work, *_flip("guardA"))
    edited = victim.read_bytes() + b"edited = true;\n"
    victim.write_bytes(edited)
    result = _run(work, *_flip("guardC"), *CMD_OK)
    _expect(result.returncode == 4,
            f"exit {result.returncode}\n{result.stdout}{result.stderr}")
    _expect("try_patch: running" not in result.stdout,
            "ran on top of an unrecoverable mutation")
    _expect_bytes(victim, edited, result)
    _expect(any((work / JOURNAL_DIR_NAME).glob("*.jsonl")),
            "the evidence was deleted with nothing undone")


def _case_recover_flag_undoes_a_killed_run(work: Path) -> None:
    victim = _victim(work)
    _kill_mid_run(work, *_flip("guardA"))
    result = _run(work, "--recover")
    _expect(result.returncode == 0,
            f"exit {result.returncode}\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)
    _expect_no_journal(work)


def _case_live_run_blocks_a_second_run_of_the_same_file(work: Path) -> None:
    """Also pins the lock: a live entry must not be taken for a dead one."""
    victim = _victim(work)
    wait_for_go = [
        "--", sys.executable, "-c",
        "import pathlib, time\n"
        "for _ in range(1200):\n"
        "    if pathlib.Path('go').exists(): break\n"
        "    time.sleep(0.05)\n",
    ]
    first = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *_flip("guardA"),
         *wait_for_go],
        cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        for _ in range(600):
            if victim.read_bytes() != GUARDS:
                break
            time.sleep(0.05)
        second = _run(work, *_flip("guardB"), *CMD_OK)
        _expect(second.returncode == 4,
                f"a live run's file was mutated again (exit "
                f"{second.returncode})\n{second.stdout}{second.stderr}")
        _expect("RECOVERED" not in second.stderr,
                f"a LIVE run was recovered from under itself\n{second.stderr}")
    finally:
        (work / "go").write_text("")
        out, err = first.communicate(timeout=60)
    _expect(first.returncode == 0, f"first run: exit {first.returncode}\n{out}{err}")
    _expect(victim.read_bytes() == GUARDS, f"not restored: {victim.read_bytes()!r}")
    _expect_no_journal(work)


def _case_a_claim_blocks_before_any_mutation(work: Path) -> None:
    """Two runs started together: the neighbour has claimed, not yet mutated.

    Scanning before claiming let both such runs through (30 pairs of 30, each
    testing without its own mutation). The neighbour here is this process,
    holding a claimed entry the way a live run does between claim and write.
    """
    victim = _victim(work)
    folder = work / JOURNAL_DIR_NAME
    folder.mkdir()
    entry = folder / "victim.cpp.1.0.jsonl"
    with open(entry, "x+b", buffering=0) as handle:
        _expect(_try_lock(handle), "fixture broken: cannot lock")
        handle.seek(0)
        handle.write(json.dumps({"pid": 1, "command": ["neighbour"],
                                 "targets": [str(victim.resolve())],
                                 "files": []}).encode("utf-8") + b"\n")
        result = _run(work, *_flip("guardA"), *CMD_OK)
        _unlock(handle)
    _expect(result.returncode == 4,
            f"a claimed file was mutated (exit {result.returncode})\n"
            f"{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)
    _expect(sorted(p.name for p in folder.iterdir()) == [entry.name],
            "the refused run left its own entry behind")


def _case_a_stuck_file_does_not_block_other_files(work: Path) -> None:
    """One unrecoverable leftover must not stop every run in the tree."""
    victim = _victim(work)
    stuck = work / "stuck.cpp"
    stuck.write_bytes(b"flag = on;\n")
    _kill_mid_run(work, "--file", "stuck.cpp", "--old", "on", "--new", "off")
    stuck.write_bytes(stuck.read_bytes() + b"edited = true;\n")
    result = _run(work, *_flip("guardA"), *CMD_OK)
    _expect(result.returncode == 0,
            f"exit {result.returncode}\n{result.stdout}{result.stderr}")
    _expect("not a target of this run" in result.stderr,
            f"the stuck file was not reported\n{result.stderr}")
    _expect_bytes(victim, GUARDS, result)
    _expect(any((work / JOURNAL_DIR_NAME).glob("*.jsonl")),
            "the stuck file's entry was dropped")


def _case_killed_between_two_files(work: Path) -> None:
    """Dead after the first file: it is undone, the second was never touched."""
    victim = _victim(work)
    other = work / "other.cpp"
    other.write_bytes(GUARDS)
    _kill_mid_run(work, *_flip("guardA"),
                  "--file", "other.cpp", "--old", "guardB = true;",
                  "--new", "guardB = false;", after_files=1)
    _expect(victim.read_bytes() != GUARDS, "fixture broken: nothing mutated")
    _expect(other.read_bytes() == GUARDS, "fixture broken: died too late")
    result = _run(work, "--recover")
    _expect(result.returncode == 0,
            f"exit {result.returncode}\n{result.stdout}{result.stderr}")
    _expect_bytes(victim, GUARDS, result)
    _expect_bytes(other, GUARDS, result)
    _expect_no_journal(work)


def _case_an_unterminated_line_does_not_count(work: Path) -> None:
    """Only the newline says a line landed, even when the tail parses."""
    full = {"targets": ["a"], "files": [
        {"path": "a", "original": "x", "mutated": "y"}]}
    blob = (json.dumps(full) + "\n" + json.dumps(_settled(full))).encode("utf-8")
    state = _last_state(blob)
    _expect(state is not None and state["files"],
            f"an unterminated tail was taken for the state: {state!r}")
    _expect(_last_state(b"") is None, "an empty entry has a state")


def _case_a_finished_entry_is_never_recovered(work: Path) -> None:
    """--keep: a scan landing between unlock and unlink must find nothing.

    In-process, with the unlink held back, so the window stays open.
    """
    victim = _victim(work)
    mutated = GUARDS.replace(b"guardA = true", b"guardA = false").decode()
    journal = RunJournal(["test"], work)
    journal.claim([victim])
    journal.record(victim, GUARDS.decode(), mutated)
    victim.write_bytes(mutated.encode())
    real_remove = globals()["_remove_entry"]
    globals()["_remove_entry"] = lambda path: None
    try:
        journal.finish(remove=True)
    finally:
        globals()["_remove_entry"] = real_remove
    scan_journal(journal_dir(victim), recover=True)
    _expect(victim.read_bytes() == mutated.encode(),
            "a finished entry was replayed and undid a --keep")


SELFTEST_CASES = (
    _case_stacked_edits_of_one_file,
    _case_stacked_edits_all_reach_the_command,
    _case_same_file_spelled_two_ways,
    _case_failing_command_restores_and_reports,
    _case_expect_fail_accepts_a_failing_command,
    _case_expect_fail_rejects_a_passing_command,
    _case_expect_fail_rejects_an_unrunnable_command,
    _case_pathext_shim_runs_by_its_bare_name,
    _case_command_rewriting_the_file_is_reported,
    _case_missing_pattern_rolls_the_earlier_edits_back,
    _case_crlf_survives_stacked_edits,
    _case_a_python_edit_leaves_no_stale_bytecode,
    _case_unrunnable_command_restores,
    _case_keep_leaves_every_mutation,
    _case_killed_run_is_undone_by_the_next,
    _case_killed_run_edited_since_is_refused,
    _case_recover_flag_undoes_a_killed_run,
    _case_live_run_blocks_a_second_run_of_the_same_file,
    _case_a_claim_blocks_before_any_mutation,
    _case_a_stuck_file_does_not_block_other_files,
    _case_killed_between_two_files,
    _case_an_unterminated_line_does_not_count,
    _case_a_finished_entry_is_never_recovered,
)


def selftest() -> int:
    failed = 0
    with tempfile.TemporaryDirectory(prefix="try_patch_selftest_") as tmp:
        for case in SELFTEST_CASES:
            name = case.__name__[len("_case_"):]
            work = Path(tmp) / name
            work.mkdir()
            # Makes `work` the tree root, so the journal lands inside it.
            (work / ".git").mkdir()
            try:
                case(work)
            except SelftestFailure as exc:
                failed += 1
                print(f"FAIL {name}\n     {exc}")
            else:
                print(f"ok   {name}")
    total = len(SELFTEST_CASES)
    if failed:
        print(f"try_patch selftest: {failed}/{total} FAILED", file=sys.stderr)
        return 1
    print(f"try_patch selftest: {total} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
