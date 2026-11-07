"""The state: which name points at which Codex thread, on disk.

Separate from the wire because it outlives it — the registry is what makes a
turn's result findable after the process that produced it is gone, and it is the
one file every parallel agent writes.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .paths import state_dir

REGISTRY_NAME = "registry.json"


@dataclass
class AgentRecord:
    name: str
    thread_id: str = ""
    cwd: str = ""
    model: str = ""
    sandbox: str = ""
    created: str = ""
    last_turn: str = ""
    turns: int = 0
    # When a turn was last STARTED on this agent, completed or not; `last_turn`
    # is the last one that completed. `status` filters on this one, so an agent
    # whose resumed turn failed a minute ago is not listed as idle for weeks.
    touched: str = ""
    # Regex of commands the agent may run OUTSIDE its sandbox when it asks —
    # empty means never (see `sandbox.escalation_judge`). Part of the record so a
    # `send` keeps the same rule as the `spawn`.
    escalate: str = ""
    # Directories OUTSIDE the workspace the agent may also write to, resolved
    # absolute (`spawn --writable`, see `sandbox.turn_sandbox_policy`). In the
    # record for the same reason as `escalate`: a `send` must resume with the
    # rights the `spawn` granted, and `status` must print what the agent has.
    writable: list[str] = field(default_factory=list)
    # Persist the briefing and answer contract across `send`; empty for old records.
    role: str = ""
    # Gate commands the wrapper runs after every turn (`spawn
    # --gates-from-prompt`, `gates.prompt_gates`); in the record so a `send`,
    # whose prompt has no `## Gates`, reruns the spawn's gates.
    gates: list[str] = field(default_factory=list)
    # Whether a worker may escalate what the host reserves for online tasks
    # (`spawn --allow-online-probes`, `CommandContext.allow_online_probes`).
    # Field and key keep their first host's word: records on disk carry
    # `allowOnlineProbes`, and the format does not change with the host.
    allow_online_probes: bool = False

    @property
    def effective_sandbox(self) -> str:
        """What this agent will actually be allowed to do.

        A record written before the field existed has none recorded, and falls
        back to `read-only` — the one setting that is both safe and known to
        work here (see the `--sandbox` flag for why the middle setting is not
        the fallback). One property rather than a default in two places: `status`
        used to print its own guess and told the operator the agent could edit
        files while its turns in fact ran read-only and refused every write.
        Reading it back out of the app-server payload fixed that but tied the
        status line to a protocol key.
        """
        return self.sandbox or "read-only"

    def to_json(self) -> dict:
        return {
            "threadId": self.thread_id, "cwd": self.cwd, "model": self.model,
            "sandbox": self.sandbox, "created": self.created,
            "lastTurn": self.last_turn, "turns": self.turns,
            "touched": self.touched, "escalate": self.escalate,
            "writable": list(self.writable), "role": self.role,
            "gates": list(self.gates),
            "allowOnlineProbes": self.allow_online_probes,
        }

    @staticmethod
    def from_json(name: str, raw: dict) -> "AgentRecord":
        return AgentRecord(
            name=name,
            thread_id=str(raw.get("threadId") or ""),
            cwd=str(raw.get("cwd") or ""),
            model=str(raw.get("model") or ""),
            sandbox=str(raw.get("sandbox") or ""),
            created=str(raw.get("created") or ""),
            last_turn=str(raw.get("lastTurn") or ""),
            turns=int(raw.get("turns") or 0),
            touched=str(raw.get("touched") or ""),
            escalate=str(raw.get("escalate") or ""),
            writable=[str(root) for root in (raw.get("writable") or [])],
            role=str(raw.get("role") or ""),
            gates=[str(command) for command in (raw.get("gates") or [])],
            allow_online_probes=bool(raw.get("allowOnlineProbes")),
        )


# How long a writer waits for the registry lock. Long enough that a machine
# briefly busy with several finishing agents still gets its turn, short enough
# that a stale lock (a killed process) does not hang the caller: past it the
# lock is taken anyway, because losing one registry entry is a smaller loss than
# a wedged command.
LOCK_WAIT_S = 10.0

# How many times a waiter may break a lock it believes to be stale before it
# gives up on locking altogether. More than one because the break can lose a
# race (another waiter takes the freed lock first, and this one must wait it
# out); bounded because "break, wait, break again" against a live holder would
# otherwise never return.
LOCK_BREAK_ATTEMPTS = 3

# How long a rename retries while another process holds the target open. On
# Windows `os.replace` is not merely non-atomic under a concurrent reader — it
# FAILS, with `PermissionError: WinError 5`, and an unhandled one here costs the
# caller the answer a finished turn already paid for (the record is never saved
# and the text is never printed). Readers hold the file for microseconds, so a
# short retry covers it.
REPLACE_RETRY_S = 5.0


class RegistryError(Exception):
    """The registry exists but cannot be used — never treated as "empty".

    Deliberately NOT a RuntimeError: `run_turn` catches RuntimeError to turn a
    protocol failure into a reported turn outcome, and a registry problem
    reaching that handler would be announced as "the turn failed" — with the
    thread id thrown away, which is the one thing that must survive.
    """


def _replace_with_retry(tmp: Path, target: Path) -> None:
    deadline = time.monotonic() + REPLACE_RETRY_S
    while True:
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


class Registry:
    """name -> thread id, on disk.

    Read-modify-write on every change rather than held open, because separate
    invocations (several background agents at once) are the normal case; the
    file is the only thing they share.

    Which is exactly why the write half takes a lock: running several agents in
    parallel is the point of this script, they all finish by updating THIS file,
    and an unlocked read-modify-write drops whichever record was read before the
    other process wrote. The window is milliseconds and it will be hit, since
    background agents are launched together and tend to finish together.
    """

    def __init__(self) -> None:
        self.path = state_dir() / REGISTRY_NAME
        self.lock_path = self.path.with_suffix(".lock")

    @contextmanager
    def _locked(self):
        """Hold an exclusive lock around one read-modify-write.

        A lock left behind by a killed process is BROKEN rather than waited on
        forever — and breaking it means deleting it, not proceeding without it.
        An earlier version just gave up waiting and wrote unlocked, which turned
        one hard kill (`taskkill /F`, which this script itself uses) into
        permanently disabled mutual exclusion: the stale file stayed, so every
        later write in every later invocation waited LOCK_WAIT_S and then raced.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + LOCK_WAIT_S
        breaks_left = LOCK_BREAK_ATTEMPTS
        handle = None
        while handle is None:
            try:
                handle = os.open(str(self.lock_path),
                                 os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() < deadline:
                    time.sleep(0.05)
                    continue
                if breaks_left <= 0:
                    break  # heavily contended: write unlocked rather than never
                breaks_left -= 1
                try:
                    os.unlink(str(self.lock_path))  # break it, then take it
                except OSError:
                    pass  # someone broke it first — the retry below settles it
                # Back to waiting rather than straight to a break: if another
                # process took the freed lock, we must WAIT for it, not write
                # alongside it. Giving up here left the lock file in place and
                # wrote unlocked, which is the permanently-disabled mutual
                # exclusion this whole path exists to prevent.
                deadline = time.monotonic() + LOCK_WAIT_S
            except OSError:
                break  # unlockable filesystem: the write is still better than nothing
        try:
            yield
        finally:
            if handle is not None:
                try:
                    os.close(handle)
                    os.unlink(str(self.lock_path))
                except OSError:
                    pass

    def load(self) -> dict[str, AgentRecord]:
        """Read the registry, or raise if it exists but cannot be read.

        "Missing" and "unreadable" must not be the same answer. They were, and
        the consequence was silent mass deletion: a single failed read returned
        `{}`, the caller added its one record to that, and wrote back a registry
        containing only itself — every other named agent gone, with no message.
        """
        try:
            text = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise RegistryError(f"cannot read {self.path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except ValueError as exc:
            raise RegistryError(
                f"{self.path} is not valid JSON ({exc}); fix or delete it") from exc
        if not isinstance(raw, dict):
            raise RegistryError(f"{self.path} does not hold an object")
        bad = sorted(name for name, value in raw.items()
                     if not isinstance(value, dict))
        if bad:
            # Skipping them would be the same silent deletion this function was
            # hardened against, one level down: the entry vanishes on the next
            # write, and nothing ever said so.
            raise RegistryError(
                f"{self.path} has malformed entries ({', '.join(bad)}); "
                f"fix or remove them")
        return {name: AgentRecord.from_json(name, value)
                for name, value in raw.items()}

    def get(self, name: str) -> Optional[AgentRecord]:
        return self.load().get(name)

    def _write(self, agents: dict[str, AgentRecord]) -> None:
        payload = {name: rec.to_json() for name, rec in sorted(agents.items())}
        # Per-process temp name: two writers sharing one `registry.tmp` collide
        # on Windows with `PermissionError: WinError 32` while the other still
        # holds it (measured — that is what an unlocked run failed with, on top
        # of losing records). The lock above already serialises writers; this
        # keeps a stale lock from turning into a torn write.
        tmp = self.path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _replace_with_retry(tmp, self.path)

    def put(self, record: AgentRecord) -> None:
        with self._locked():
            agents = self.load()
            agents[record.name] = record
            self._write(agents)

    def reserve(self, record: AgentRecord, *, replace: bool) -> str:
        """Claim a name and store the record in ONE locked step.

        `spawn` used to check for the name, run a turn that takes minutes, and
        only then write — so two spawns of the same name both passed the check,
        both ran, and the loser's thread ended up with no registry entry at all
        while both turns appended to one log. Claiming the name up front also
        means a crash mid-turn leaves a resumable agent instead of an orphaned
        Codex thread.

        Windows paths are case-insensitive while dict keys are not, so `review`
        and `Review` would be separate records sharing one state directory and
        one `last.md`; the clash is refused here rather than discovered later.
        """
        with self._locked():
            agents = self.load()
            folded = record.name.casefold()
            clash = next((name for name in agents
                          if name.casefold() == folded and name != record.name), None)
            if clash:
                raise SystemExit(
                    f"agent {clash!r} already exists and would share "
                    f"{record.name!r}'s files (names differ only in case)")
            existing = agents.get(record.name)
            if existing is not None and not replace:
                raise SystemExit(
                    f"agent {record.name!r} already exists "
                    f"(thread {existing.thread_id or '?'}) and --keep was given. "
                    f"Use `send {record.name} ...` to continue it.")
            agents[record.name] = record
            self._write(agents)
        return existing.thread_id if existing is not None else ""

    def drop(self, name: str) -> bool:
        with self._locked():
            agents = self.load()
            if name not in agents:
                return False
            del agents[name]
            self._write(agents)
        return True
