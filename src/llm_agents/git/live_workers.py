"""Which agent processes are alive, and which directories they may write.

The pool (`worktree.py`) acts on a slot's tree on the assumption that the
slot's worker has ENDED: its checks read the tree once and the git or install
step that follows acts on it later, so a worker still running could plant a
file or a link in between. A worktree lock says a task holds the slot, not
that a process is running there. This module is the other half: a bridge that
runs a worker (the Codex bridge's `turn.run_turn`) writes a record here for as
long as the worker runs, and the pool refuses a slot that a live record
names.

A record is one JSON file per wrapper process in `live_dir()`: the agent's
name, the directories its sandbox is rooted at or may write (`roots`), and the
processes whose life is the worker's (`processes`: the wrapper itself and each
server it starts), each as pid + OS creation time. The creation time is what
makes a pid safe to trust on Windows, which reuses pids within seconds: a pid
alive with another creation time is a different process, and the record is
stale. A record none of whose processes is alive is stale too (a wrapper
killed with `taskkill /F` never removes its file); stale records never block
and are pruned when read.

The folder is outside every checkout and outside the temp folders on purpose:
a sandboxed worker may write its workspace and the temp folders
(`codex.sandbox`), and a record it could delete is no record. Standard library
only, like the rest of the kit.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

LIVE_DIR_ENV = "LLM_AGENTS_LIVE_DIR"
RECORD_SUFFIX = ".live.json"


def live_dir() -> Path:
    """`LLM_AGENTS_LIVE_DIR`, else a per-user folder outside any checkout:
    `%LOCALAPPDATA%\\llm_agents\\live-workers` on Windows,
    `$XDG_STATE_HOME/llm_agents/live-workers` (`~/.local/state`) elsewhere.
    Read per call, so a test's override is obeyed."""
    override = os.environ.get(LIVE_DIR_ENV)
    if override:
        return Path(override)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "llm_agents" / "live-workers"


# ------------------------------------------------------------------ processes

# OpenProcess fails with this for a pid no process has; any other failure
# (ERROR_ACCESS_DENIED for another user's or a protected process) means the
# process exists, which is the answer that matters here.
_ERROR_INVALID_PARAMETER = 87
_STILL_ACTIVE = 259
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _kernel32():
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    return kernel


def process_identity(pid: int) -> Optional[int]:
    """The OS creation time of `pid` as an integer that never changes while
    the process lives -- FILETIME ticks on Windows, the boot-relative
    `starttime` of `/proc/<pid>/stat` on Linux -- or None when the process is
    gone, cannot be asked, or the platform keeps no such time (`same_process`
    then trusts the pid alone).

    The same idea as `llm_loop.exitlog.pid_alive`, compared exactly rather
    than within seconds: both sides here are read from the OS, never from a
    clock."""
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel = _kernel32()
        handle = kernel.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
                return None
            times = [wintypes.FILETIME() for _ in range(4)]
            if not kernel.GetProcessTimes(handle, *(ctypes.byref(t) for t in times)):
                return None
            return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
        finally:
            kernel.CloseHandle(handle)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    # The command name (field 2) may hold spaces and parentheses: count from
    # the last ')'. `starttime` is field 22, the 20th after it.
    fields = stat[stat.rfind(")") + 2:].split()
    return int(fields[19]) if len(fields) > 19 and fields[19].isdigit() else None


def same_process(pid: int, identity: Optional[int]) -> bool:
    """Whether `pid` still names the process recorded with `identity`. A
    process this user may not open (another user's) counts as alive: a
    record must never read as stale because it could not be checked."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        current = process_identity(pid)
        if current is not None:
            return identity is None or current == identity
        kernel = _kernel32()
        handle = kernel.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel.CloseHandle(handle)
            return False    # opened, yet no identity: it has exited
        return ctypes.get_last_error() != _ERROR_INVALID_PARAMETER
    try:
        os.kill(pid, 0)     # POSIX: signal 0 is only a probe (never on Windows)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    current = process_identity(pid)
    return identity is None or current is None or current == identity


# ------------------------------------------------------------------- records

@dataclass
class LiveWorker:
    """One record as read back."""
    path: Path
    name: str = ""
    roots: list[str] = field(default_factory=list)
    processes: list[tuple[int, Optional[int]]] = field(default_factory=list)
    since: str = ""
    # Why the record could not be read; such a record is never stale.
    error: str = ""

    @property
    def alive(self) -> bool:
        return bool(self.error) or any(same_process(pid, ident) for pid, ident in self.processes)

    def touches(self, directory: Path) -> bool:
        """Whether a root lies in `directory` or contains it: a worker rooted
        at the slot, in one of its folders, or allowed to write a parent of it."""
        return any(_inside(Path(root), directory) or _inside(directory, Path(root))
                   for root in self.roots)

    def describe(self) -> str:
        if self.error:
            return f"unreadable record {self.path} ({self.error})"
        pids = ", ".join(str(pid) for pid, _ in self.processes)
        return f"{self.name or '?'} (pid {pids}, since {self.since or '?'}; record {self.path})"


def _inside(path: Path, root: Path) -> bool:
    def key(p: Path) -> str:
        return os.path.normcase(os.path.abspath(p))
    inner, outer = key(path), key(root).rstrip("\\/")
    return inner == outer or inner.startswith(outer + os.sep)


def _read(path: Path) -> LiveWorker:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return LiveWorker(
            path=path, name=str(raw.get("name") or ""),
            roots=[str(root) for root in raw.get("roots") or []],
            processes=[(int(p["pid"]), None if p.get("identity") is None else int(p["identity"]))
                       for p in raw.get("processes") or []],
            since=str(raw.get("since") or ""))
    except FileNotFoundError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return LiveWorker(path=path, error=f"{type(error).__name__}: {error}")


def live_workers(directory: Path, *, prune: bool = True) -> list[LiveWorker]:
    """The live records naming `directory` (`LiveWorker.touches`), plus every
    record that cannot be read (it might name it). Stale records are deleted
    when `prune`; a record removed while this reads it is skipped."""
    folder = live_dir()
    try:
        entries = sorted(folder.glob(f"*{RECORD_SUFFIX}"))
    except OSError:
        return []
    found = []
    for path in entries:
        try:
            worker = _read(path)
        except FileNotFoundError:
            continue
        if not worker.alive:
            if prune:
                path.unlink(missing_ok=True)
            continue
        if worker.error or worker.touches(directory):
            found.append(worker)
    return found


class LiveRecord:
    """The record a bridge holds while its worker runs (`recorded`)."""

    def __init__(self, name: str, roots: list[Path]) -> None:
        self.path = live_dir() / f"{os.getpid()}-{time.time_ns()}{RECORD_SUFFIX}"
        self.data = {
            "name": name,
            "roots": [os.path.abspath(root) for root in roots],
            "processes": [{"pid": os.getpid(), "identity": process_identity(os.getpid())}],
            "since": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def add_process(self, pid: int) -> None:
        """Count `pid` (a server the wrapper started) as the worker's too: a
        wrapper killed alone leaves its child tree running, and the record
        then stays live while that child does."""
        self.data["processes"].append({"pid": pid, "identity": process_identity(pid)})
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


@contextmanager
def recorded(name: str, roots: list[Path]) -> Iterator[LiveRecord]:
    """Hold a live record for the block. A record that cannot be written stops
    the block before it starts: a worker the pool cannot see is the case this
    module exists to prevent."""
    record = LiveRecord(name, roots)
    try:
        record._write()
    except OSError as error:
        raise SystemExit(f"cannot write the live-worker record {record.path}: {error}; "
                         f"the worktree pool would not see this worker") from error
    try:
        yield record
    finally:
        record.remove()
