"""Process-lifetime exclusion keyed by the invoking script's physical path."""

import atexit
import errno
import hashlib
import os
from pathlib import Path
import sys
import time


# The operator requested a half-second polling cadence; this is not a timeout.
POLL_SECONDS = 0.5
LOCK_DIR = Path.home() / ".llm-loop" / "script-locks"
_launches = {}


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


def ensure_script_lock() -> None:
    """Ask once on contention, before logging, terminal input or work starts.

    Identity comes from argv[0], never the driver class, app label, project root
    or library location. Resolve it before a runner can change its cwd. A
    batching wrapper calls several runners; retain its choice and lock until
    PROCESS exit, including the gaps between those calls. Dry-run callers skip
    this function because they only preview work; --help exits before it.

    Independent means bypassing the mutex for this invocation: it neither
    releases the owner's lock nor becomes an owner after that process exits.
    EOF refuses to run without an explicit decision, including redirected stdin.
    """
    lock = ScriptLock(sys.argv[0])
    if lock.script in _launches:
        return
    try:
        if not lock.acquire():
            print(f"Another instance of this script is running: {lock.script}",
                  flush=True)
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
                    _launches[lock.script] = None
                    return
                if choice in ("w", "wait", "2"):
                    print("Waiting for the script lock (checking every 0.5 s); "
                          "Ctrl+C cancels.", flush=True)
                    while not lock.acquire():
                        time.sleep(POLL_SECONDS)
                    break
                print("Choose e, w or i.", flush=True)
        _launches[lock.script] = lock
        atexit.register(lock.close)
    except KeyboardInterrupt:
        lock.close()
        print("\nScript launch cancelled.", file=sys.stderr)
        raise SystemExit(130)
    except BaseException:
        lock.close()
        raise
