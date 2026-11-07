"""What one agent leaves on disk: its event log and its last answer.

The per-agent half of the package's state. The other half — names, thread ids
and the lock they share — is `registry.py`, and the two stay apart on purpose:
the registry's locking and this layout change for different reasons, so a new
layout (a directory per turn, rotation, compressed old logs) is a change to this
file alone rather than to every command that writes or reads an artifact.

The file names live here and nowhere else. `turn.run_turn` writes both files and
four `cli` commands read or move them; spelled out at each use they were an
unwritten contract between two modules, and renaming either literal made
`result`/`logs` go quiet — the file they look for is simply not there, which is
also what a fresh agent looks like. Tests go through the same object for the
same reason: a test that writes `last.md` by hand stops guarding the name.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from .events import TurnTrace, iter_log, last_turn_trace
from .paths import state_dir

EVENTS_FILE = "events.jsonl"
ANSWER_FILE = "last.md"


class AgentStore:
    """The artifacts of the agent `name` in the current state directory.

    Holds no state of its own: every path is resolved on use, so a store made
    before `CODEX_AGENTS_DIR` changes follows the change, as the free functions
    it replaced did.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def dir(self) -> Path:
        return state_dir() / self.name

    @property
    def events_path(self) -> Path:
        """The raw JSON-RPC stream of every turn this agent has run."""
        return self.dir / EVENTS_FILE

    @property
    def answer_path(self) -> Path:
        """The agent's most recent stored answer — what `result` prints."""
        return self.dir / ANSWER_FILE

    # --- the event log ----------------------------------------------------

    @contextmanager
    def event_log(self) -> Iterator[Callable[[dict], None]]:
        """Open the log for appending and yield the function that appends one
        message, flushed at once so `status` and `logs` see a running turn.

        Held open for a whole turn by `run_turn` rather than reopened per
        event: a turn writes thousands of lines.
        """
        path = self.events_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _terminate_last_line(path)
        with path.open("a", encoding="utf-8") as handle:
            def append(message: dict) -> None:
                handle.write(json.dumps(message, ensure_ascii=False) + "\n")
                handle.flush()
            yield append

    def append_events(self, messages: Iterable[dict]) -> None:
        """Append `messages` in one opening of the log."""
        with self.event_log() as append:
            for message in messages:
                append(message)

    def has_events(self) -> bool:
        return self.events_path.exists()

    def events(self) -> Iterator[dict]:
        """Every parseable message of the log, oldest first — `events.iter_log`."""
        return iter_log(self.events_path)

    def last_turn(self) -> Optional[TurnTrace]:
        """The trace of the last turn, or None without a log or a turn."""
        return last_turn_trace(self.events_path)

    def events_silent_s(self) -> float:
        """Seconds since the log was last written. The log is appended on every
        event, so this is how long the turn writing it has said nothing."""
        return time.time() - self.events_path.stat().st_mtime

    # --- the answer -------------------------------------------------------

    def write_answer(self, text: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.answer_path.write_text(text, encoding="utf-8")

    def read_answer(self) -> Optional[str]:
        """The stored answer, or None when the agent has never produced one."""
        try:
            return self.answer_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

    # --- a reused name ----------------------------------------------------

    def archive(self) -> None:
        """Move the displaced agent's log and answer aside, keeping both.

        Not deletion: the previous run's findings are often the reason a new one
        is being started. Not appending either — `logs` and `result` would then
        mix two unrelated threads under one name and read as if a single agent
        had done all of it.
        """
        currents = (self.events_path, self.answer_path)
        stamp = free_archive_stamp(
            lambda s: [_archived_name(c, s) for c in currents])
        for current in currents:
            if current.exists():
                current.replace(_archived_name(current, stamp))

    def archived(self) -> list[Path]:
        """The files earlier runs under this name were moved to, oldest first
        within each kind, events before answers."""
        if not self.dir.exists():
            return []
        found = []
        for current in (self.events_path, self.answer_path):
            prefix, suffix = f"{current.stem}.", current.suffix
            found += sorted(
                self.dir.glob(f"{prefix}*{suffix}"),
                key=lambda p: archive_stamp_order(p.name[len(prefix):-len(suffix)]))
        return found


def _archived_name(current: Path, stamp: str) -> Path:
    return current.with_name(f"{current.stem}.{stamp}{current.suffix}")


def free_archive_stamp(targets: Callable[[str], Iterable[Path]]) -> str:
    """The second-resolution stamp of now, or `<stamp>-2`, `-3`, … when any of
    the paths `targets(stamp)` already exists.

    Two archives of one name in one second are real: a `spawn` whose turn fails
    at once and a scripted retry. `Path.replace` overwrites a file target, so a
    shared stamp silently swapped the previous agent's log — the one the
    archive exists to keep — for the failed run's; onto an existing directory
    it raises instead. All targets of one archive share the suffix, so a log
    and its answer keep pairing by name.
    """
    base = time.strftime("%Y%m%d-%H%M%S")
    stamp, n = base, 1
    while any(path.exists() for path in targets(stamp)):
        n += 1
        stamp = f"{base}-{n}"
    return stamp


def archive_stamp_order(stamp: str) -> tuple[str, int]:
    """Sort key for a `free_archive_stamp` stamp. Plain string order puts
    `<s>-2` before `<s>` (`-` sorts below the `.` that follows a bare stamp)
    and `-10` before `-2`."""
    base, sep, n = stamp.rpartition("-")
    if sep and len(base) == 15 and n.isdigit():
        return base, int(n)
    return stamp, 1


def _terminate_last_line(log_path: Path) -> None:
    """A writer killed mid-line leaves the log without a final newline; the
    next turn's opening record would then be glued onto that fragment, and
    `events.iter_log` would drop the new turn boundary together with the
    fragment — `status` keeping the old trace (review 2026-09-13)."""
    try:
        with log_path.open("rb+") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                return
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
    except FileNotFoundError:
        return
