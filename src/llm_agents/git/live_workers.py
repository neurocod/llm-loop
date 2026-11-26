"""Keep the worktree pool and a running agent worker off each other's slot.

The pool (`worktree.py`) acts on a slot's tree on the assumption that the
slot's worker has ENDED: its checks read the tree once and the git or install
step that follows acts on it later, so a worker still running could plant a
file or a link in between. A worktree lock says a task holds the slot, not
that a process is running there. This module turns that assumption into an
exclusion between two parties: a bridge running a worker (the Codex bridge's
`turn.run_turn`) and a pool command acting on a slot.

Three parts, all in `live_dir()`:

- **Locks** (`worker_guard`, `pool_guard`): the bridge holds a guard on its
  worker's roots for the whole turn, a pool command holds one on its slot
  for the whole operation, and whichever comes second fails at once
  (`Busy`). Neither side waits. The guard is what closes check-then-act: a
  turn cannot register between the pool's check and its git or install,
  because the check IS the lock the pool keeps holding. An OS lock dies with
  its process, so a killed wrapper never leaves one behind.
- **A job** (`WorkerJob`, Windows): every server the bridge starts runs in a
  kill-on-close job object without breakaway, so the worker's whole process
  tree ends with the wrapper however the wrapper ends -- a `taskkill /F`
  included -- and the pool can ask the job whether anything in it still runs.
- **Records** (`recorded`): what the guard and the job cannot show -- who
  holds a slot (`status`), and a tree whose wrapper is gone while its job
  still has processes (termination was not confirmed). A record file is never
  rewritten (`LiveRecord.add_process` writes a new one), so pruning a stale
  file can never delete a newer one. A record none of whose processes lives
  (pid + OS creation time: Windows reuses pids within seconds) and whose job
  is empty or gone is stale, never blocks, and is pruned when read; one that
  cannot be read blocks.

Beside them, one lock between pool commands (`slot_operation`): the guards
let any number of pool commands share a slot, and this one lets only one act
there at a time.

The folder is outside every checkout and outside the temp folders a
sandboxed worker may write (`codex.sandbox.sandbox_temp_dirs`): a record or
lock the worker could delete is none. `refuse_overlap` holds the bridge to
that. Every path compared here is canonical (`canonical`: links, junctions
and 8.3 names resolved), so an alias of a slot is the slot. Standard library
only, like the rest of the kit.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from llm_loop import win_job

LIVE_DIR_ENV = "LLM_AGENTS_LIVE_DIR"
RECORD_SUFFIX = ".live.json"
LOCKS_DIR = "locks"


class Busy(Exception):
    """The other party holds a guard on an overlapping tree."""


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


def _plain_dos(path: str) -> str:
    """`path` without Windows' extended-length prefix: `\\\\?\\C:\\x` is `C:\\x`
    and `\\\\?\\UNC\\host\\share` is `\\\\host\\share`. `realpath` keeps a prefix
    it was given (measured 2026-11-18: one checkout read back both as
    `c:\\work\\repo-wt1` and as `\\\\?\\c:\\work\\repo-wt1`), so two
    spellings of one folder would hash to two lock keys."""
    if os.name != "nt":
        return path
    spelled = path.replace("/", "\\")
    if spelled[:8].upper() == "\\\\?\\UNC\\":
        return "\\\\" + spelled[8:]
    if spelled.startswith("\\\\?\\") or spelled.startswith("\\\\.\\"):
        return spelled[4:]
    return path


def canonical(path: str | os.PathLike) -> str:
    """The one spelling of `path` both sides compare: absolute, without an
    extended-length prefix (`_plain_dos`), every link, junction and 8.3 short
    name resolved (`realpath` asks the file system for the final path of what
    exists), case-folded where the file system folds."""
    plain = _plain_dos(os.path.abspath(_plain_dos(os.fspath(path))))
    return os.path.normcase(_plain_dos(os.path.realpath(plain)))


def is_inside(path: str, root: str) -> bool:
    """For two `canonical` paths."""
    outer = root.rstrip("\\/")
    return path == outer or path.startswith(outer + os.sep)


def _ancestors(path: str) -> list[str]:
    """`path` and every folder above it, canonical spellings."""
    chain = [path]
    while True:
        parent = os.path.dirname(chain[-1])
        if parent == chain[-1]:
            return chain
        chain.append(parent)


def refuse_overlap(writable: Iterable[Path]) -> None:
    """Refuse a worker that could write `live_dir()`: a root inside it or
    above it (`LLM_AGENTS_LIVE_DIR` pointed into a workspace, a `--writable`
    granting a parent). Its records and locks would be its own to delete."""
    registry = canonical(live_dir())
    for root in writable:
        mine = canonical(root)
        if is_inside(registry, mine) or is_inside(mine, registry):
            raise SystemExit(f"the worker could write {root}, which overlaps the live-worker "
                             f"folder {live_dir()} ({LIVE_DIR_ENV}): its records and locks would "
                             f"be the worker's to delete. Move one of them")


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
    kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenJobObjectW.restype = wintypes.HANDLE
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                 wintypes.DWORD, ctypes.c_void_p]
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    kernel.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.DWORD, ctypes.c_void_p]
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


# ----------------------------------------------------------------------- jobs

_JOB_OBJECT_QUERY = 0x0004
_JOB_BASIC_ACCOUNTING = 1


def _accounting_struct():
    import ctypes
    from ctypes import wintypes

    class Accounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64),
                    ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD),
                    ("TotalTerminatedProcesses", wintypes.DWORD)]
    return Accounting


def _active_in(kernel, handle) -> Optional[int]:
    import ctypes

    info = _accounting_struct()()
    if not kernel.QueryInformationJobObject(handle, _JOB_BASIC_ACCOUNTING, ctypes.byref(info),
                                            ctypes.sizeof(info), None):
        return None
    return int(info.ActiveProcesses)


def job_active_processes(name: str) -> Optional[int]:
    """How many processes run in the job object `name`: 0 once it is gone
    (no handle and no process keeps it), None when it exists but cannot be
    asked -- which a caller must read as "maybe running"."""
    if os.name != "nt" or not name:
        return 0
    import ctypes

    kernel = _kernel32()
    handle = kernel.OpenJobObjectW(_JOB_OBJECT_QUERY, False, name)
    if not handle:
        return 0 if ctypes.get_last_error() == 2 else None    # ERROR_FILE_NOT_FOUND
    try:
        return _active_in(kernel, handle)
    finally:
        kernel.CloseHandle(handle)


class WorkerJob:
    """A named kill-on-close job object WITHOUT breakaway holding a worker's
    process tree (`adopt`): every descendant starts inside it and none may
    leave, so closing the last handle -- `close`, or Windows when this
    process ends in any way -- ends the whole tree. Off Windows a no-op whose
    `name` is "" (`create` returns one either way)."""

    def __init__(self, name: str = "", handle: int = 0) -> None:
        self.name = name
        self._handle = handle

    @classmethod
    def create(cls) -> "WorkerJob":
        """OSError when Windows will not make one: a worker that cannot be
        contained is not started (the caller's to report)."""
        if os.name != "nt":
            return cls()
        name = f"Local\\llm-agents-worker-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        return cls(name, win_job.create(name, win_job.KILL_ON_JOB_CLOSE))

    @property
    def creationflags(self) -> int:
        """What the child is started with, so that `adopt` puts it in the job
        before it can start a process of its own."""
        return win_job.CREATE_SUSPENDED if self._handle else 0

    def adopt(self, child) -> None:
        """Put a `Popen` started with `creationflags` into the job and resume
        it (`win_job.adopt`); a child that cannot be put there is killed and
        the error raised."""
        if self._handle:
            win_job.adopt(self._handle, child)

    def active(self) -> Optional[int]:
        return _active_in(_kernel32(), self._handle) if self._handle else 0

    def end(self, wait_s: float, sleep: Callable[[float], None] = time.sleep) -> bool:
        """Terminate every process in the job and wait up to `wait_s` for the
        count to reach 0; True once it has. Termination is asynchronous: a
        process ends after its pending I/O is cancelled."""
        if not self._handle:
            return True
        _kernel32().TerminateJobObject(self._handle, 1)
        deadline = time.monotonic() + wait_s
        while True:
            if self.active() == 0:
                return True
            if time.monotonic() >= deadline:
                return False
            sleep(0.05)

    def close(self) -> None:
        if self._handle:
            _kernel32().CloseHandle(self._handle)
            self._handle = 0


# ---------------------------------------------------------------------- locks

# How long a probe of the other side's lock is retried before the other side
# counts as present. A probe collides only with another probe of the same
# side, held for the microseconds between two system calls; a present party
# holds its lock for a whole turn or operation. A second covers a probe
# preempted under a loaded machine many times over.
PROBE_S = 1.0
_SIDES = ("worker", "pool")


class _FileLock:
    """A whole-file OS lock, shared or exclusive, never waited for. On
    Windows `LockFileEx` (byte 0), elsewhere `flock`: both belong to the open
    handle, so two handles in one process exclude each other as two
    processes would, and both are released by the kernel when the process
    ends."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # Binary: `slot_operation` writes its holder line through this fd.
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o600)
        self._held = False

    def try_lock(self, exclusive: bool) -> bool:
        if os.name == "nt":
            import ctypes
            import msvcrt

            overlapped = (ctypes.c_byte * 32)()     # OVERLAPPED, offset 0
            flags = 0x1 | (0x2 if exclusive else 0)  # FAIL_IMMEDIATELY | EXCLUSIVE
            self._held = bool(_kernel32().LockFileEx(msvcrt.get_osfhandle(self._fd), flags, 0,
                                                      1, 0, overlapped))
            return self._held
        import fcntl

        try:
            fcntl.flock(self._fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except OSError:
            return False
        self._held = True
        return True

    def unlock(self) -> None:
        if not self._held:
            return
        self._held = False
        if os.name == "nt":
            import ctypes
            import msvcrt

            _kernel32().UnlockFileEx(msvcrt.get_osfhandle(self._fd), 0, 1, 0,
                                     (ctypes.c_byte * 32)())
        else:
            import fcntl
            fcntl.flock(self._fd, fcntl.LOCK_UN)

    def close(self) -> None:
        self.unlock()
        os.close(self._fd)


def _lock_path(kind: str, path: str, side: str) -> Path:
    key = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
    return live_dir() / LOCKS_DIR / f"{key}.{kind}.{side}"


class Guard:
    """Locks one side holds; `release` (or the process ending) frees them."""

    def __init__(self) -> None:
        self._locks: list[_FileLock] = []

    def release(self) -> None:
        while self._locks:
            self._locks.pop().close()

    def __enter__(self) -> "Guard":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _guard(side: str, keys: set[tuple[str, str]], what: str,
           sleep: Callable[[float], None] = time.sleep) -> Guard:
    """Hold `side`'s lock on each key, then make sure the other side holds
    none of them; `Busy` (nothing held) otherwise.

    Each key has a lock file per side. A party holds its own side's file
    SHARED -- parties of one side never exclude each other: two workers in
    one tree, two pool commands on two slots -- and then PROBES the other
    side's file EXCLUSIVELY, letting go at once. Hold-then-probe on both
    sides means two parties starting together cannot both pass: whichever
    probes second finds the first one's hold. A probe fails briefly when it
    meets another probe of the same side, hence the retry (`PROBE_S`)."""
    guard = Guard()
    other = _SIDES[1 - _SIDES.index(side)]
    try:
        for kind, path in sorted(keys):
            own = _FileLock(_lock_path(kind, path, side))
            guard._locks.append(own)
            deadline = time.monotonic() + PROBE_S
            while not own.try_lock(exclusive=False):
                if time.monotonic() >= deadline:
                    raise Busy(f"{what}: {own.path} stays locked by a probe")
                sleep(0.01)
            probe = _FileLock(_lock_path(kind, path, other))
            try:
                while not probe.try_lock(exclusive=True):
                    if time.monotonic() >= deadline:
                        raise Busy(f"{what}: a {other} holds {path}")
                    sleep(0.01)
            finally:
                probe.close()
    except BaseException:
        guard.release()
        raise
    return guard


def worker_guard(roots: Iterable[Path]) -> Guard:
    """The bridge's guard for a worker rooted at `roots` (its cwd and every
    root it may write): for each root, its own `exact` key and the `tree` key
    of it and every folder above it. A pool command on a slot at or above a
    root meets the `tree` key; one on a slot inside a root meets `exact`."""
    keys: set[tuple[str, str]] = set()
    for root in roots:
        path = canonical(root)
        keys.add(("exact", path))
        keys.update(("tree", folder) for folder in _ancestors(path))
    return _guard("worker", keys, "the worker")


def pool_guard(slot: Path) -> Guard:
    """A pool command's guard on `slot`: the slot's `tree` key (any worker
    rooted in it) and the `exact` key of every folder above it (a worker
    rooted at a parent may write the slot)."""
    path = canonical(slot)
    keys = {("tree", path)} | {("exact", folder) for folder in _ancestors(path)[1:]}
    return _guard("pool", keys, f"the pool command on {slot}")


# How long `slot_operation` waits for another pool command on the slot before
# refusing. Most pool commands on a slot take well under a second (`release`,
# `next`, a `land` with nothing to land); one that installs takes minutes, and
# no wait worth sitting through covers it -- the refusal names the holder
# instead. Read per call, so a test may shorten it.
OPERATION_WAIT_S = 5.0


class SlotOperation:
    """An exclusive OS lock on one slot that a pool command holds from its
    first look at the slot to its last act there (`slot_operation`). Byte 0 is
    the lock; the bytes after it name the holder for a refused command to
    print -- outside the locked range, so Windows lets another handle read
    them."""

    def __init__(self, lock: _FileLock) -> None:
        self._lock = lock

    def release(self) -> None:
        if self._lock is None:
            return
        lock, self._lock = self._lock, None
        try:
            os.ftruncate(lock._fd, 0)
        except OSError:
            pass
        lock.close()

    def __enter__(self) -> "SlotOperation":
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _holder(path: Path) -> str:
    """The holder line, only when the process it names is still that process
    (`same_process`): the line is advisory -- written after the lock is taken,
    and left behind by a holder killed before it cleared it -- so a line whose
    process has ended describes a previous holder, not this one."""
    unknown = "holder unknown"
    try:
        with open(path, "rb") as file:
            file.seek(1)
            text = file.read(4096).decode("utf-8", "replace").strip()
    except OSError as error:
        return f"{unknown} (the holder line cannot be read: {error})"
    if not text:
        return f"{unknown} (it has not named itself yet)"
    try:
        named = json.loads(text)
        pid, identity = int(named["pid"]), named.get("identity")
        what, since = str(named.get("what", "")), str(named.get("since", ""))
    except (ValueError, TypeError, KeyError) as error:
        return f"{unknown} (the holder line does not parse: {error})"
    if not same_process(pid, identity if isinstance(identity, int) else None):
        return f"{unknown} (the line names pid {pid}, which has ended: a previous holder's)"
    return f"pid {pid} ({what or 'a pool command'}) since {since or 'an unknown time'}"


def slot_operation(slot: Path, what: str, *, wait_s: Optional[float] = None,
                   sleep: Callable[[float], None] = time.sleep) -> SlotOperation:
    """Hold the slot's operation lock exclusively, waiting up to `wait_s`
    (default `OPERATION_WAIT_S`) for another holder to let go; `Busy` naming
    that holder otherwise.

    What it closes that `pool_guard` does not: pool commands hold the pool's
    side SHARED, so two of them on one slot -- a `land` while the create hook
    stands between its inspection, claim, submodule update and reset -- would
    each act on a verdict the other had made stale. A worker no bridge records
    (a Claude agent) takes neither lock: the worktree lock (`claim`) is its
    side. An OS lock dies with its process, so a killed command leaves no
    stale lock to recover, and `what` (the command, for the holder line) is
    only ever read while the lock is held."""
    lock = _FileLock(_lock_path("slot", canonical(slot), "operation"))
    deadline = time.monotonic() + (OPERATION_WAIT_S if wait_s is None else wait_s)
    while not lock.try_lock(exclusive=True):
        if time.monotonic() >= deadline:
            holder = _holder(lock.path)
            lock.close()
            raise Busy(f"another pool command is acting on {slot}: {holder}")
        sleep(0.05)
    line = json.dumps({"pid": os.getpid(), "identity": process_identity(os.getpid()),
                       "what": what or "a pool command",
                       "since": time.strftime("%Y-%m-%d %H:%M:%S")}).encode("utf-8")
    try:
        os.ftruncate(lock._fd, 0)
        os.lseek(lock._fd, 0, os.SEEK_SET)
        os.write(lock._fd, b"\n" + line)
    except OSError:
        pass    # the holder line is a courtesy to the refused; the lock is held
    return SlotOperation(lock)


# -------------------------------------------------------------------- records

@dataclass
class LiveWorker:
    """One record file as read back."""
    path: Path
    name: str = ""
    roots: list[str] = field(default_factory=list)
    processes: list[tuple[int, Optional[int]]] = field(default_factory=list)
    job: str = ""
    since: str = ""
    # Why the record could not be read; such a record is never stale.
    error: str = ""

    @property
    def alive(self) -> bool:
        if self.error or any(same_process(pid, ident) for pid, ident in self.processes):
            return True
        active = job_active_processes(self.job)
        return active is None or active > 0

    def touches(self, directory: str) -> bool:
        """Whether a root lies in the canonical `directory` or contains it: a
        worker rooted at the slot, in one of its folders, or allowed to write
        a parent of it."""
        return any(is_inside(root, directory) or is_inside(directory, root)
                   for root in self.roots)

    def describe(self) -> str:
        if self.error:
            return f"unreadable record {self.path} ({self.error})"
        pids = ", ".join(str(pid) for pid, _ in self.processes)
        job = f", job {self.job}" if self.job else ""
        return f"{self.name or '?'} (pid {pids}{job}, since {self.since or '?'}; record {self.path})"


def _read(path: Path) -> LiveWorker:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return LiveWorker(
            path=path, name=str(raw.get("name") or ""),
            roots=[canonical(root) for root in raw.get("roots") or []],
            processes=[(int(p["pid"]), None if p.get("identity") is None else int(p["identity"]))
                       for p in raw.get("processes") or []],
            job=str(raw.get("job") or ""), since=str(raw.get("since") or ""))
    except FileNotFoundError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        return LiveWorker(path=path, error=f"{type(error).__name__}: {error}")


def live_workers(directory: Path, *, prune: bool = True) -> list[LiveWorker]:
    """The live records naming `directory` (`LiveWorker.touches`), plus every
    record that cannot be read (it might name it). Stale records are deleted
    when `prune` -- safe without a lock, since no record file is ever
    rewritten. No folder: no records. A folder that cannot be listed raises
    (OSError): "could not look" must not read as "nobody there"."""
    folder = live_dir()
    if not folder.exists():
        return []
    entries = sorted(path for path in folder.iterdir() if path.name.endswith(RECORD_SUFFIX))
    target = canonical(directory)
    found = []
    for path in entries:
        try:
            worker = _read(path)
        except FileNotFoundError:
            continue    # removed by its owner, or pruned by another reader
        if not worker.alive:
            if prune:
                path.unlink(missing_ok=True)
            continue
        if worker.error or worker.touches(target):
            found.append(worker)
    return found


class LiveRecord:
    """The record files a bridge holds while its worker runs (`recorded`).
    Each is written once, complete, and never rewritten."""

    def __init__(self, name: str, roots: list[Path], job: str = "") -> None:
        self._base = f"{os.getpid()}-{time.time_ns()}"
        self._fields = {"name": name, "roots": [canonical(root) for root in roots], "job": job,
                        "since": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.paths: list[Path] = []

    def _write(self, pid: int) -> None:
        path = live_dir() / f"{self._base}-{len(self.paths)}{RECORD_SUFFIX}"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        data = {**self._fields, "processes": [{"pid": pid, "identity": process_identity(pid)}]}
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        self.paths.append(path)

    def add_process(self, pid: int) -> None:
        """Count `pid` (a server the wrapper started) as the worker's too, in
        a file of its own: off Windows there is no job to hold the server's
        tree, and a wrapper killed alone leaves it running."""
        self._write(pid)

    def remove(self) -> None:
        for path in self.paths:
            path.unlink(missing_ok=True)


@contextmanager
def recorded(name: str, roots: list[Path], *, job: Optional[WorkerJob] = None,
             ended: Callable[[], bool] = lambda: True) -> Iterator[LiveRecord]:
    """Hold the worker's guard and its record for the block.

    Refused (SystemExit) before the block starts when a pool command holds
    one of the roots, or when the record cannot be written: a worker the pool
    cannot see is the case this module exists to prevent. On the way out the
    record is removed only once `ended()` confirms the worker's processes are
    gone (`WorkerJob.end`); otherwise it stays, naming the job, and keeps
    refusing the slot until the job is empty."""
    try:
        guard = worker_guard(roots)
    except Busy as busy:
        raise SystemExit(f"a worktree pool command is acting on this worker's tree ({busy}); "
                         f"start the worker again once it is done") from busy
    try:
        record = LiveRecord(name, roots, job.name if job is not None else "")
        try:
            record._write(os.getpid())
        except OSError as error:
            raise SystemExit(f"cannot write a live-worker record in {live_dir()}: {error}; "
                             f"the worktree pool would not see this worker") from error
        try:
            yield record
        finally:
            if ended():
                record.remove()
    finally:
        guard.release()
