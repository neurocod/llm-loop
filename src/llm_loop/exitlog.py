"""Why the run ended — including the endings the process cannot report itself.

A loop that dies leaves the mirror log ending mid-line, and nothing in it says
whether that was a crash, a stop request, or a kill from outside. Measured, not
theorised: two runGenerateModels runs (2026-08-20 13:02, 2026-08-21 02:44) ended
exactly like that, and the cause turned out to be the run's OWN agent, which
mistook the long-lived `python.exe` parent for a hung `commit.py` helper and
issued `Stop-Process` on it. Nothing in the 11 MB log named that; the log simply
stopped.

Two halves, because no single mechanism covers every ending:

* what the process CAN observe — the loop's own stop reason, ``sys.exit``, an
  unhandled exception, Ctrl+C, a delivered signal — is printed as one
  ``=== run ended: … ===`` line through the tee, so it lands in the mirror log;
* what it CANNOT — ``TerminateProcess`` (PowerShell ``Stop-Process``,
  ``taskkill /F``), an OOM kill, a power cut — leaves no line *by definition*:
  the process is not running any more, so no handler of its own can write one.
  Instead every run drops a small record file beside the log and removes it on
  the way out. A record still on disk whose owner is gone IS the report, and the
  next run prints it.

The record is a diagnostic, never a dependency: every write is best-effort, and
failing to write one must not cost an iteration.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, List, Optional

# How often the watchdog thread refreshes `alive_at`. The record's job is to
# answer "when did the process stop existing?", and only a clock the run keeps
# ticking can answer it: one iteration can run for a quarter of an hour, so a
# heartbeat written at iteration boundaries alone would place the death
# anywhere inside that window. 30 s is one small file write per half-minute.
HEARTBEAT_SECONDS = 30

RECORD_SUFFIX = ".run.json"

# Windows constants for the liveness probe (see pid_alive).
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_STILL_ACTIVE = 259
# A pid may be reused within seconds on Windows, and a recycled pid reads as
# "the owner is alive", which would silently swallow the very report this module
# exists for. So the probe also compares process creation times, and only a
# match this close counts as the same process.
_START_MATCH_SECONDS = 5.0

_record: Optional["RunRecord"] = None
_lock = threading.Lock()


# --- liveness ----------------------------------------------------------------


def _win_process_start(handle) -> Optional[float]:
    """Unix epoch seconds when the process behind `handle` was created."""
    import ctypes
    import ctypes.wintypes as wt

    creation, exit_t, kernel, user = (wt.FILETIME() for _ in range(4))
    ok = ctypes.windll.kernel32.GetProcessTimes(
        handle, ctypes.byref(creation), ctypes.byref(exit_t),
        ctypes.byref(kernel), ctypes.byref(user))
    if not ok:
        return None
    ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
    # FILETIME counts 100 ns intervals since 1601-01-01.
    return ticks / 1e7 - 11644473600.0


def pid_alive(pid: int, started: Optional[float] = None) -> bool:
    """Whether `pid` still names the process that wrote the record.

    `started` is the record's own start time; when the platform can report a
    process creation time, a pid whose process started at a different moment is
    a recycled pid and counts as gone.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        # NEVER os.kill(pid, 0) here. On Windows os.kill does not send a signal:
        # it opens the process and calls TerminateProcess with `sig` as the exit
        # code, so the liveness probe would kill the very run it asks about.
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False        # gone — or not ours to look at, same answer here
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            if code.value != _STILL_ACTIVE:
                return False
            if started is not None:
                actual = _win_process_start(handle)
                if actual is not None:
                    return abs(actual - started) <= _START_MATCH_SECONDS
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)         # POSIX: signal 0 really is only a probe
    except ProcessLookupError:
        return False
    except PermissionError:
        return True             # alive, just not ours to signal
    return True


# --- formatting --------------------------------------------------------------


def _fmt_moment(ts: Optional[float]) -> str:
    if not ts:
        return "unknown"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def describe_exception(exc_type, exc) -> str:
    """The reason string for an exception that escaped the wrapper."""
    if exc_type is KeyboardInterrupt:
        return "interrupted from the keyboard (Ctrl+C)"
    if exc_type is SystemExit:
        code = getattr(exc, "code", None)
        if code in (None, 0):
            return "sys.exit()"
        if isinstance(code, int):
            return f"sys.exit({code})"
        return f"sys.exit: {code}"
    text = str(exc).strip().splitlines()
    detail = f": {text[0]}" if text else ""
    return f"unhandled {exc_type.__name__}{detail}"


# --- the record --------------------------------------------------------------


class RunRecord:
    """One process's "I am still here" file, removed when it ends properly."""

    def __init__(self, path: Path, fields: dict, echo: Callable[[str], None]):
        self._path = path
        self._fields = fields
        self._echo = echo
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._finished = False
        self._reason: Optional[str] = None
        self._write()
        self._beat = threading.Thread(
            target=self._heartbeat, name="exitlog-heartbeat", daemon=True)
        self._beat.start()

    @property
    def path(self) -> Path:
        return self._path

    def _write(self) -> None:
        with self._lock:
            self._fields["alive_at"] = time.time()
            payload = json.dumps(self._fields, ensure_ascii=False, indent=1)
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError:
            pass    # a diagnostic must never be a reason for the run to fail

    def _heartbeat(self) -> None:
        while not self._done.wait(HEARTBEAT_SECONDS):
            self._write()

    def note(self, **fields) -> None:
        """Record what the run is doing now, so a kill can be placed in the work."""
        with self._lock:
            self._fields.update(fields)
        self._write()

    def set_reason(self, reason: str, **fields) -> None:
        """Remember why this run is ending; the line itself is printed at exit."""
        self._reason = reason
        if fields:
            self.note(**fields)

    def finish(self, reason: Optional[str] = None) -> None:
        """Print the closing line and drop the record. Safe to call twice."""
        with self._lock:
            if self._finished:
                return
            self._finished = True
        self._done.set()
        reason = reason or self._reason or "process exit (reason not recorded)"
        started = self._fields.get("started") or time.time()
        parts = [f"=== run ended: {reason}"]
        iterations = self._fields.get("iterations")
        if iterations is not None:
            parts.append(f"{iterations} iteration(s), "
                         f"{self._fields.get('completed', 0)} completed")
        parts.append(_fmt_elapsed(time.time() - started))
        try:
            self._echo(" · ".join(parts) + " ===")
        except Exception:       # a closed/broken stream at exit is not our problem
            pass
        try:
            os.remove(self._path)
        except OSError:
            pass


def _stem(app_name: str, project: str) -> str:
    return f"{app_name}-{project}"


def record_path(log_dir: Path, app_name: str, project: str, pid: int) -> Path:
    """One record per PROCESS, not per app: two runs of the same wrapper are
    routine here (a sequential run, a parallel run, the grow-kit pass), and a
    shared file name would have each clearing the other's report."""
    return Path(log_dir) / f"{_stem(app_name, project)}.{pid}{RECORD_SUFFIX}"


def report_orphans(app_name: str, log_dir: Path, project: str,
                   echo: Callable[[str], None] = print) -> List[dict]:
    """Print (and clear) the records of runs that never wrote an ending.

    Returns them, so a caller can act on more than the printed lines. Records
    belonging to a process that is still running are left alone and reported as
    live — that is also how concurrent runs of the same wrapper become visible,
    which matters because they contend for `.git/index.lock`.
    """
    log_dir = Path(log_dir)
    orphans: List[dict] = []
    try:
        candidates = sorted(log_dir.glob(f"{_stem(app_name, project)}.*{RECORD_SUFFIX}"))
    except OSError:
        return orphans
    for path in candidates:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            try:
                os.remove(path)     # unreadable record: nothing to learn from it
            except OSError:
                pass
            continue
        pid = int(data.get("pid") or 0)
        if pid == os.getpid():
            continue
        if pid_alive(pid, data.get("started")):
            echo(f"  · another {app_name} run is live (pid {pid}, started "
                 f"{_fmt_moment(data.get('started'))}) — they share this log, "
                 f"and they contend for the git index")
            continue
        orphans.append(data)
        echo(f"  ⚠ the previous run left no exit record: it did not stop itself "
             f"— it was killed from outside (Stop-Process / taskkill / a "
             f"reboot), or the process died without unwinding.")
        echo(f"      pid {pid}, started {_fmt_moment(data.get('started'))}, "
             f"last alive {_fmt_moment(data.get('alive_at'))}")
        if data.get("phase"):
            echo(f"      last known work: {data['phase']}")
        if data.get("argv"):
            echo(f"      command: {data['argv']}")
        try:
            os.remove(path)
        except OSError:
            pass
    return orphans


def begin(app_name: str, log_dir: Path, project: str, argv=None,
          echo: Callable[[str], None] = print) -> Optional[RunRecord]:
    """Report previous runs that vanished, then start this run's own record.

    Idempotent per process: a wrapper that calls several runners in one
    invocation (see the periodic mode) keeps one record for the whole process,
    because "why did the script terminate?" is a question about the process.
    """
    global _record
    with _lock:
        if _record is not None:
            return _record
    log_dir = Path(log_dir)
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    report_orphans(app_name, log_dir, project, echo=echo)
    if argv is None:
        argv = " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:])
    fields = {
        "pid": os.getpid(),
        "app": app_name,
        "project": project,
        "argv": argv,
        "started": _own_start_time(),
    }
    record = RunRecord(record_path(log_dir, app_name, project, os.getpid()),
                       fields, echo)
    with _lock:
        _record = record
    atexit.register(record.finish)
    _install_hooks()
    return record


def _own_start_time() -> float:
    """This process's creation time where the OS knows it, else "now".

    Taken from the OS rather than the clock so the recycled-pid check in
    pid_alive compares like with like — and so the reported start is the
    process's, not the moment logging happened to be set up.
    """
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(
            _PROCESS_QUERY_LIMITED_INFORMATION, False, os.getpid())
        if handle:
            try:
                actual = _win_process_start(handle)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
            if actual is not None:
                return actual
    return time.time()


# --- the endings the process can see ------------------------------------------

_hooks_installed = False
# Signals worth naming. SIGINT is deliberately absent: its default already
# raises KeyboardInterrupt, which the excepthook below reports with a better
# name. On Windows SIGTERM is never actually delivered — TerminateProcess is not
# a signal — which is exactly why the record file exists.
_WATCHED_SIGNALS = ("SIGTERM", "SIGHUP", "SIGBREAK")


def _install_hooks() -> None:
    global _hooks_installed
    if _hooks_installed:
        return
    _hooks_installed = True

    previous_hook = sys.excepthook

    def excepthook(exc_type, exc, tb):
        set_reason(describe_exception(exc_type, exc))
        previous_hook(exc_type, exc, tb)

    sys.excepthook = excepthook

    if threading.current_thread() is not threading.main_thread():
        return              # only the main thread may install signal handlers
    import signal

    for name in _WATCHED_SIGNALS:
        number = getattr(signal, name, None)
        if number is None:
            continue
        try:
            previous = signal.getsignal(number)
            signal.signal(number, _make_signal_handler(name, number, previous))
        except (ValueError, OSError, RuntimeError):
            continue


def _make_signal_handler(name: str, number: int, previous):
    def handler(signum, frame):
        # Named and flushed BEFORE the signal resumes its journey: whatever runs
        # next may end the process without unwinding anything of ours.
        set_reason(f"terminated by {name}")
        finish()
        import signal

        if callable(previous):
            previous(signum, frame)
            return
        # Never swallow a termination signal: re-raise it under its original
        # disposition so the process still dies the way it was told to.
        signal.signal(number, previous if previous is not None else signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    return handler


# --- module-level conveniences ------------------------------------------------


def current() -> Optional[RunRecord]:
    with _lock:
        return _record


def note(**fields) -> None:
    record = current()
    if record is not None:
        record.note(**fields)


def set_reason(reason: str, **fields) -> None:
    record = current()
    if record is not None:
        record.set_reason(reason, **fields)


def finish(reason: Optional[str] = None) -> None:
    record = current()
    if record is not None:
        record.finish(reason)


@contextmanager
def guard():
    """Name the endings Python does not route through `sys.excepthook`.

    `sys.exit(...)` raises SystemExit, which the interpreter handles itself
    without consulting the excepthook — so a wrapper that bails out with a
    message ("error: --grow-kit models nothing…") would leave the closing line
    saying only "reason not recorded". Wrap a wrapper's main body in this and
    every ending it can reach is named.
    """
    try:
        yield
    except BaseException as exc:        # SystemExit and KeyboardInterrupt too
        set_reason(describe_exception(type(exc), exc))
        raise
