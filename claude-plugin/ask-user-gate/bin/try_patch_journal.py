"""try_patch's journal: what a run mutated, until it is verifiably restored.

A module of its own, standard library only, because it is read from outside:
`unrestored_mutations` in `tools/git/commit.py` (the repository this grew up
in) refuses to commit a file a try_patch run still holds, and imports THIS file
from whatever commit its submodule sits at -- not try_patch.py, whose CLI,
`replace_in_file` import and path setup it has no use for. So
`JOURNAL_DIR_NAME`, `scan_journal(folder, recover=False)` and the `Finding`
fields it reads (`who` among them) are an interface: change them together with
commit.py, in the landing that bumps the submodule.

The selftest stays in try_patch.py.
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

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
"""

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


class JournalError(Exception):
    """The journal cannot be written, so the run must not mutate anything.

    Its own class rather than replace_in_file's EditError, which would pull
    that module -- and the path setup it needs -- into every reader.
    """


def journal_dir(path: Path) -> Path:
    """The journal directory of the working tree `path` belongs to."""
    here = path.resolve().parent
    for folder in (here, *here.parents):
        # A file, not a directory, in a worktree or a submodule.
        if (folder / ".git").exists():
            return folder / JOURNAL_DIR_NAME
    return here / JOURNAL_DIR_NAME


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
      try_patch exists to prevent.

    Deleting the cache entry is what makes the outcome independent of the
    clock. It is done after mutating and again after restoring -- a recovery
    here included -- because any of those writes can be the one that collides.
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
            # command still gets the source just written, and the stale entry
            # is the next run's problem, which this line at least keeps rare.
            pass


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
        raise JournalError(f"cannot write the journal entry {path}: {exc}") from exc


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
            raise JournalError(f"cannot create a journal entry in {folder}")
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
                except JournalError as exc:
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

    @property
    def who(self) -> str:
        """The run, as every message about it names it."""
        return f"pid {self.pid}: {self.command}"


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
                    except JournalError:
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
        who = finding.who
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
