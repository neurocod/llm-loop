"""Process-lifetime exclusion keyed by the invoking script's physical path."""

import atexit
import errno
import hashlib
import os
from pathlib import Path
import sys
import time
from typing import NamedTuple


# The operator requested a half-second polling cadence; this is not a timeout.
POLL_SECONDS = 0.5
LOCK_DIR = Path.home() / ".llm-loop" / "script-locks"
_launches = {}
_launch_decision = None


class LaunchDecision(NamedTuple):
    """The startup choice retained for the whole process and its run record."""

    mode: str
    path: str
    conflict: str
    wait_seconds: float

    @property
    def held_at_start(self) -> bool:
        return self.mode != "independent"

    def record(self) -> dict:
        return {"mode": self.mode, "path": self.path,
                "conflict": self.conflict,
                "wait_seconds": round(self.wait_seconds, 1),
                "held_at_start": self.held_at_start}

    def summary(self) -> str:
        detail = f"; initial conflict: {self.conflict}" if self.conflict else ""
        waited = (f"; waited {self.wait_seconds:.1f} s"
                  if self.mode == "waited" else "")
        return (f"script lock: {self.mode} (held at startup="
                f"{self.held_at_start}); pid {os.getpid()}; "
                f"path: {self.path}{detail}{waited}")


class ScriptLock:
    """An OS lock, not a sentinel: a killed owner cannot leave it locked.

    Keep the file after closing it. Unlinking would let a waiter lock the old
    inode while a newcomer locks a replacement at the same path. The first
    byte is lockable even in an empty file on Windows and POSIX, so opening
    never has to rewrite bytes another process already locked.
    """

    def __init__(self, script: str):
        self.script = os.path.normcase(os.path.realpath(script))
        digest = hashlib.sha256(os.fsencode(self.script)).hexdigest()
        self.path = LOCK_DIR / (digest + ".lock")
        self._file = None

    def acquire(self) -> bool:
        if self._file is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                return False
            raise
        except BaseException:
            handle.close()
            raise
        self._file = handle
        return True

    def close(self) -> None:
        if self._file is not None:
            # Closing the descriptor releases the OS lock on both platforms.
            self._file.close()
            self._file = None


def ensure_script_lock(*, app_name: str = None,
                       project_dir: str = None) -> LaunchDecision:
    """Ask once on contention, before logging, terminal input or work starts.

    The OS lock's identity comes from argv[0], never the driver class, app label,
    project root or library location. Resolve it before a runner can change its
    cwd. A live run record for this app/project also counts as contention: a
    process started with older engine code may not hold this OS lock. A batching
    wrapper calls several runners; retain its choice and lock until PROCESS
    exit, including the gaps between those calls. Dry-run callers skip this
    function because they only preview work; --help exits before it.

    Independent means bypassing the mutex for this invocation: it neither
    releases the owner's lock nor becomes an owner after that process exits.
    EOF refuses to run without an explicit decision, including redirected stdin.
    """
    # One process has one invoking script. Check the retained decision BEFORE
    # resolving argv[0] again: a batching wrapper may have changed its cwd.
    global _launch_decision
    if _launches:
        return _launch_decision
    lock = ScriptLock(sys.argv[0])
    if project_dir is not None:
        project_dir = os.path.normcase(os.path.abspath(project_dir))

    def live_runs() -> list[int]:
        if app_name is None or project_dir is None:
            return []
        from . import console, exitlog
        project = os.path.basename(os.path.normpath(project_dir))
        return exitlog.live_run_pids(
            app_name, console.LOG_DIR, project, script=lock.script,
            project_dir=project_dir)

    try:
        locked = lock.acquire()
        peers = live_runs()
        conflict = "; ".join(filter(None, (
            "OS lock held" if not locked else "",
            f"live run pid {', '.join(map(str, peers))}" if peers else "")))
        if not locked or peers:
            detail = (f" (live run pid {', '.join(map(str, peers))})"
                      if peers else "")
            print(f"Another instance of this script is running: "
                  f"{lock.script}{detail}", flush=True)
            while True:
                try:
                    choice = input(
                        "[e] Exit, [w] Wait, [i] Run independently: ").strip().lower()
                except EOFError:
                    print("No choice received; exiting.", file=sys.stderr)
                    raise SystemExit(1)
                if choice in ("e", "exit", "1"):
                    raise SystemExit(0)
                if choice in ("i", "independent", "3"):
                    print("Starting independently without the script lock.", flush=True)
                    lock.close()
                    decision = LaunchDecision(
                        "independent", str(lock.path), conflict, 0.0)
                    _launches[lock.script] = None
                    _launch_decision = decision
                    return decision
                if choice in ("w", "wait", "2"):
                    print("Waiting for the script lock (checking every 0.5 s); "
                          "Ctrl+C cancels.", flush=True)
                    started = time.monotonic()
                    while not lock.acquire() or live_runs():
                        time.sleep(POLL_SECONDS)
                    decision = LaunchDecision(
                        "waited", str(lock.path), conflict,
                        time.monotonic() - started)
                    break
                print("Choose e, w or i.", flush=True)
        else:
            decision = LaunchDecision("acquired", str(lock.path), "", 0.0)
        _launches[lock.script] = lock
        _launch_decision = decision
        atexit.register(lock.close)
        return decision
    except KeyboardInterrupt:
        lock.close()
        print("\nScript launch cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except BaseException:
        lock.close()
        raise
