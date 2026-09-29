"""A run that ends must say why — and a run that is killed must be reported.

The incident these pin: two long runs ended mid-command with 11 MB of mirror log
and nothing in it about the ending. Both had been killed from outside, which is
precisely the case no handler inside the dying process can log. So the closing
line covers everything the process can see, and the leftover record covers the
one thing it cannot.
"""

import collections
import json
import os
import sys
import threading

import pytest

from llm_loop import console, cyclecore, exitlog, parallel

from _runfixtures import (MemListDriver, NoWorkDriver, isolated_run, par_args,
                          seq_args)


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    """Every test gets its own log dir and its own fresh record."""
    with isolated_run(monkeypatch, tmp_path):
        yield


def _run_once(tmp_path):
    cyclecore.run_loop(NoWorkDriver(), seq_args(tmp_path, no_statusline=True),
                       app_name="pytest-exit", wait_on_start=False)


def test_a_clean_run_names_its_ending_in_the_log(tmp_path):
    _run_once(tmp_path)
    exitlog.finish()

    written = console.log_file_path("pytest-exit").read_text(
        encoding="utf-8", errors="replace")
    assert "=== run ended: no more work in the queue" in written


def test_a_finished_run_leaves_no_record_behind(tmp_path):
    """The record's only meaning is "this run never got to end", so a run that
    did end must not leave one — else every launch reports a phantom kill."""
    _run_once(tmp_path)
    record = exitlog.current().path
    assert record.exists()          # while the run is live it is the liveness

    exitlog.finish()
    assert not record.exists()


def test_a_record_whose_owner_is_gone_is_reported_and_cleared(
        tmp_path, monkeypatch, capsys):
    """The whole point: the kill that logs nothing is named by the NEXT run."""
    logs = console.LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    dead = exitlog.record_path(logs, "pytest-exit", "proj", 424242)
    dead.write_text(json.dumps({
        "pid": 424242, "app": "pytest-exit", "project": "proj",
        "argv": "runGenerateModels.py --codex --random",
        "started": 1000.0, "alive_at": 2000.0,
        "phase": "iteration 10 — electric-guitar-hollow-body.md",
    }), encoding="utf-8")
    monkeypatch.setattr(exitlog, "pid_alive", lambda pid, started=None: False)

    orphans = exitlog.report_orphans("pytest-exit", logs, "proj")

    out = capsys.readouterr().out
    assert len(orphans) == 1
    assert "killed from outside" in out
    assert "pid 424242" in out
    assert "electric-guitar-hollow-body.md" in out       # what it was working on
    assert "--codex --random" in out                     # and how it was launched
    assert not dead.exists()        # cleared, so the next run does not re-report it


def test_a_record_whose_owner_still_runs_is_left_alone(
        tmp_path, monkeypatch, capsys):
    """Two runs of one wrapper is routine here. Reporting a live sibling as a
    corpse — and deleting its record — would break the report for real kills."""
    logs = console.LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    live = exitlog.record_path(logs, "pytest-exit", "proj", 424242)
    live.write_text(json.dumps({"pid": 424242, "started": 1000.0}),
                    encoding="utf-8")
    monkeypatch.setattr(exitlog, "pid_alive", lambda pid, started=None: True)

    orphans = exitlog.report_orphans("pytest-exit", logs, "proj")

    assert orphans == []
    assert live.exists()
    assert "is live (pid 424242" in capsys.readouterr().out


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_the_report_of_a_vanished_run_lands_in_the_log(
        tmp_path, monkeypatch, runner):
    """The ORDER inside the shared prologue: the tee goes up BEFORE exitlog.begin.

    `begin` is what prints the previous run's missing-ending report, and the
    whole point of that report is to explain a mirror log that stops mid-line.
    Printed before the tee exists it goes to a terminal nobody is reading any
    more, and the log it explains never gets it — which is exactly the log the
    next reader opens. Nothing pinned that order before this: the closing line
    is written at exit, long after the tee is up either way, so swapping the two
    left the whole suite green.

    Asserted against the FILE, never against capsys: the report reaches the
    screen whichever order the two are in, so a screen-based assertion cannot
    tell them apart. Both runners, because both open a run through the one
    prologue and the pin has to fail if either stops doing so.
    """
    logs = console.LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    project = os.path.basename(str(tmp_path))
    app_name = "pytest-exit"
    # A mirror handler another test left on this name would send this run's
    # output to THAT test's file — which reads exactly like the defect below
    # and is not it. `isolated_run` closes every one on the way in.
    dead = exitlog.record_path(logs, app_name, project, 424242)
    dead.write_text(json.dumps({
        "pid": 424242, "app": app_name, "project": project,
        "argv": "runGenerateModels.py --codex --random",
        "started": 1000.0, "alive_at": 2000.0,
        "phase": "iteration 10 — electric-guitar-hollow-body.md",
    }), encoding="utf-8")
    monkeypatch.setattr(exitlog, "pid_alive", lambda pid, started=None: False)

    if runner == "sequential":
        _run_once(tmp_path)
    else:
        monkeypatch.setattr(parallel, "run_job",
                            lambda job_id, command, mailbox=None: (0, None, None))
        parallel.run_parallel(MemListDriver(["products/only.md"]),
                              par_args(tmp_path, jobs=1, no_statusline=True),
                              app_name=app_name, wait_on_start=False)
    exitlog.finish()

    written = console.log_file_path(app_name).read_text(
        encoding="utf-8", errors="replace")
    assert "killed from outside" in written, (
        "the report of the previous run that vanished never reached the mirror "
        "log — exitlog.begin ran before the tee was up")
    assert "electric-guitar-hollow-body.md" in written


def test_the_liveness_probe_does_not_kill_what_it_asks_about(monkeypatch):
    """On Windows `os.kill(pid, 0)` is TerminateProcess with exit code 0 — the
    probe would kill the run it was asked about. This pins that it is not used."""
    def forbidden(*args, **kwargs):
        raise AssertionError("os.kill must never be part of a liveness probe "
                             "on Windows")

    if os.name == "nt":
        monkeypatch.setattr(os, "kill", forbidden)
    assert exitlog.pid_alive(os.getpid()) is True
    assert exitlog.pid_alive(424242, started=1000.0) is False


def test_a_recycled_pid_does_not_mask_a_kill():
    """A pid is reused within seconds on Windows. Our own pid with someone
    else's start time must read as gone, or the report is silently swallowed."""
    if os.name != "nt":
        pytest.skip("only Windows reports a process creation time here")
    assert exitlog.pid_alive(os.getpid(), started=1000.0) is False


def test_sys_exit_is_named_although_no_excepthook_sees_it(tmp_path, capsys):
    _run_once(tmp_path)
    with pytest.raises(SystemExit):
        with exitlog.guard():
            sys.exit("error: --grow-kit models nothing")
    exitlog.finish()

    assert "=== run ended: sys.exit: error: --grow-kit models nothing" \
        in capsys.readouterr().out


# Upper bound on every wait of the write-in-flight pins; only a broken staging
# comes near it.
HELD_S = 10.0


class _HeldReplaceOs:
    """Stands in for `exitlog.os`: the heartbeat's first `replace` waits.

    A replacement MODULE for one importer (patching `os.replace` itself would
    hold every thread in the process); everything else is forwarded.
    `replaced` names the thread of every replace, held or not, as it starts;
    `completed` as it returns; `removed` the thread of every remove.
    """

    def __init__(self):
        self.in_replace = threading.Event()
        self.release = threading.Event()
        self.replaced = []
        self.completed = []
        self.removed = []

    def __getattr__(self, name):
        return getattr(os, name)

    def remove(self, path):
        self.removed.append(threading.current_thread().name)
        os.remove(path)

    def replace(self, src, dst):
        self.replaced.append(threading.current_thread().name)
        if (threading.current_thread().name == "exitlog-heartbeat"
                and not self.in_replace.is_set()):
            self.in_replace.set()
            self.release.wait(timeout=HELD_S)
        os.replace(src, dst)
        self.completed.append(threading.current_thread().name)


class _WatchedFileLock:
    """Stands in for a record's `_file_lock`, saying who has ASKED for it.

    `asked[name]` is set the moment the thread `name` starts to acquire —
    before it waits — which is what lets a pin prove that `finish` is waiting
    on a write rather than hope a head start was long enough.
    """

    def __init__(self, real):
        self._real = real
        self.asked = collections.defaultdict(threading.Event)

    def acquire(self, blocking=True, timeout=-1):
        self.asked[threading.current_thread().name].set()
        return self._real.acquire(blocking, timeout)

    def release(self):
        self._real.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()


def _record(tmp_path, monkeypatch, *, heartbeat=HELD_S * 100):
    """A RunRecord of its own under tmp_path; no heartbeat unless asked for."""
    monkeypatch.setattr(exitlog, "HEARTBEAT_SECONDS", heartbeat)
    path = tmp_path / f"pytest-exit{exitlog.RECORD_SUFFIX}"
    return exitlog.RunRecord(path, {"pid": os.getpid()}, echo=lambda line: None)


def test_a_write_in_flight_does_not_outlive_finish(tmp_path, monkeypatch):
    """A clean exit must not leave its record behind — not even by a heartbeat.

    The record on disk MEANS "this run never got to end", and the next launch
    reports it as a kill from outside. `finish` used to remove it while a write
    begun a moment earlier was still under way; that write then put it back,
    and a run that ended properly was reported as killed. Staged, not raced: the
    heartbeat's write is held between its temp file and its replace, and it is
    let go only once `finish` is provably waiting for the file lock — or has
    already returned without waiting, which is the defect.

    Two claims, because the write also cleans up behind itself once it sees
    the record finished (see `exitlog.RunRecord._write`): that cleanup alone
    leaves no file here, but it runs only if the process lives on — `finish`
    runs from `atexit`, and a process exiting the moment it returns kills the
    daemon heartbeat mid-write. So `finish` must also have WAITED for the write.
    """
    held = _HeldReplaceOs()
    monkeypatch.setattr(exitlog, "os", held)
    record = _record(tmp_path, monkeypatch, heartbeat=0.01)
    watched = _WatchedFileLock(record._file_lock)
    record._file_lock = watched
    returned = threading.Event()
    completed_at_return = []

    def finish():
        record.finish()
        completed_at_return.append(list(held.completed))
        returned.set()

    finisher = threading.Thread(target=finish, name="finisher", daemon=True)
    try:
        assert held.in_replace.wait(timeout=HELD_S), "the heartbeat never wrote"
        finisher.start()
        asked = watched.asked["finisher"]
        for _ in range(int(HELD_S / 0.01)):
            if asked.is_set() or returned.is_set():
                break
            returned.wait(0.01)
        assert asked.is_set() or returned.is_set(), "finish never ran"
    finally:
        held.release.set()
    finisher.join(timeout=HELD_S)
    record._beat.join(timeout=HELD_S)

    assert not finisher.is_alive(), "finish never returned"
    assert not record._beat.is_alive(), "the heartbeat never stopped"
    assert ("exitlog-heartbeat" in completed_at_return[0]), (
        "finish returned while the heartbeat's write was still in flight — a "
        "process exiting right then leaves that write's record behind")
    assert not record.path.exists(), (
        "a finished run's record is on disk again — the next launch will report "
        "this clean exit as a run killed from outside")


def test_a_finished_record_is_never_written_again(tmp_path, monkeypatch):
    """After `finish`, a worker's `note` must not touch the file at all.

    Cleaning up behind such a write (see the pins below) would leave no file
    either, so the pin watches the writes themselves: a record written after
    its run ended is a record a kill between the write and its cleanup leaves
    behind as a false report.
    """
    record = _record(tmp_path, monkeypatch)
    watched = _HeldReplaceOs()
    monkeypatch.setattr(exitlog, "os", watched)

    record.finish()
    record.note(phase="after the end")

    assert watched.replaced == [], (
        f"a finished record was written again: {watched.replaced}")
    assert not record.path.exists()


class _FinishingReplaceOs(_HeldReplaceOs):
    """`replace` runs `finish` first, on the writing thread — a signal handler.

    A termination signal's handler runs `finish` on the main thread wherever
    that thread happens to be, the middle of its own record write included (the
    file lock is reentrant for exactly that). `then` is what the previous
    handler does next: return, and the interrupted write resumes; or raise
    `SystemExit`, and it never does.
    """

    def __init__(self, record, then=None):
        super().__init__()
        self.record = record
        self.then = then

    def replace(self, src, dst):
        record, self.record = self.record, None
        if record is not None:
            record.finish()
            if self.then is not None:
                raise self.then
        return super().replace(src, dst)


@pytest.mark.parametrize("then", [None, SystemExit(143)],
                         ids=["handler-returns", "handler-raises-SystemExit"])
def test_finish_in_the_middle_of_a_write_leaves_nothing_behind(
        tmp_path, monkeypatch, then):
    """A signal's `finish` inside the main thread's own write: no record, no temp.

    The write it interrupted resumes after it (the previous handler returned)
    and its replace puts the record back — reported by the next launch as a
    kill of a run that ended itself — or never resumes (the previous handler
    raised) and leaves its temp file.
    """
    record = _record(tmp_path, monkeypatch)
    monkeypatch.setattr(exitlog, "os", _FinishingReplaceOs(record, then))
    tmp = record.path.with_name(record.path.name + ".tmp")

    if then is None:
        record.note(phase="interrupted by a signal")
    else:
        with pytest.raises(SystemExit):
            record.note(phase="interrupted by a signal")

    assert not record.path.exists(), (
        "the write a signal's finish interrupted put the record back")
    assert not tmp.exists(), "the interrupted write left its temp file behind"


# How long the stalled-write pin gives `finish` (FINISH_WAIT_S patched to 0.05
# there) to return — the deadline under test, kept apart from HELD_S, after
# which the held write lets itself go only so a broken staging cannot hang the
# suite. A giving-up `finish` took 0.062 s median, 0.0628 / 0.0633 s at worst
# (two runs of 20, measured 2026-09-29); thirty times that, because only the
# failing case waits it out.
FINISH_DEADLINE_S = 2.0


def test_finish_does_not_wait_for_a_stalled_write_forever(tmp_path, monkeypatch):
    """A stalled disk must not hold the process's ending for as long as it likes.

    `finish` runs from `atexit` and from a termination signal's handler, and
    the heartbeat holds the file lock across its write. Past FINISH_WAIT_S
    `finish` returns without touching the stalled directory — no closing line
    (the tee writes it into the mirror log there), no removal of the record —
    and the write, once it returns, removes what it wrote; the next `finish`
    prints the line, once.

    What is asked of the first `finish` is that it returned while the write was
    STILL held: a `finish` that waits without bound returns too, the moment the
    held write lets itself go.
    """
    monkeypatch.setattr(exitlog, "FINISH_WAIT_S", 0.05)
    held = _HeldReplaceOs()
    monkeypatch.setattr(exitlog, "os", held)
    record = _record(tmp_path, monkeypatch, heartbeat=0.01)
    lines = []
    record._echo = lines.append
    at_return = []

    def finish():
        record.finish()
        at_return.append((held.release.is_set(), list(held.removed), list(lines)))

    finisher = threading.Thread(target=finish, name="finisher", daemon=True)
    try:
        assert held.in_replace.wait(timeout=HELD_S), "the heartbeat never wrote"
        finisher.start()
        finisher.join(timeout=FINISH_DEADLINE_S)
        in_time = not finisher.is_alive()
    finally:
        held.release.set()
    finisher.join(timeout=HELD_S)
    record._beat.join(timeout=HELD_S)

    assert in_time, "finish waited for the stalled write past its bound"
    assert at_return == [(False, [], [])], (
        "finish returned only after the stalled write was let go, or wrote to "
        "the stalled directory on its way out — (write released, removed by, "
        f"lines printed) at its return: {at_return}")
    assert not record._beat.is_alive(), "the heartbeat never stopped"
    assert not record.path.exists(), (
        "the stalled write put the record back after finish gave up on it")
    record.finish()
    record.finish()
    assert len(lines) == 1 and "=== run ended" in lines[0], (
        f"the closing line was lost or printed twice: {lines}")


class _SignalInsideSet:
    """Stands in for a record's `_done`: a signal's `finish` lands inside `set`.

    `threading.Event.set` holds the Event's own lock, which is not reentrant,
    while it notifies; a termination signal's handler runs `finish` on the
    main thread right there. The lock here is a plain one too, taken with a
    bound instead of for ever: `deadlocked` is whether a `set` found it held
    by the thread already inside one — the real Event hangs there.
    """

    def __init__(self, record):
        self._real = record._done       # the one the heartbeat waits on
        self._lock = threading.Lock()
        self._record = record
        self.deadlocked = False

    def set(self):
        if not self._lock.acquire(timeout=REENTRY_BOUND_S):
            self.deadlocked = True
            return
        try:
            self._real.set()
            record, self._record = self._record, None
            if record is not None:
                record.finish()         # the handler, mid-notification
        finally:
            self._lock.release()

    def wait(self, timeout=None):
        return self._real.wait(timeout)

    def is_set(self):
        return self._real.is_set()


# How long `_SignalInsideSet` lets a reentrant `set` wait before calling it a
# deadlock. Only the failing case waits it out: a healthy re-entry never asks.
REENTRY_BOUND_S = 1.0


def test_a_signal_s_finish_inside_the_heartbeat_s_wakeup_does_not_deadlock(
        tmp_path, monkeypatch):
    """A finish re-entered during `_done.set()` must not set it again."""
    record = _record(tmp_path, monkeypatch)
    lines = []
    record._echo = lines.append
    done = _SignalInsideSet(record)
    record._done = done

    record.finish()

    assert not done.deadlocked, (
        "the re-entered finish set the heartbeat's Event again from inside "
        "its own set — on the real Event that waits for ever")
    assert done.is_set(), "the heartbeat was never told the record ended"
    assert len(lines) == 1 and "=== run ended" in lines[0], (
        f"the closing line was lost or printed twice: {lines}")
    assert not record.path.exists()


def test_an_older_snapshot_never_overwrites_a_newer_one(tmp_path, monkeypatch):
    """The payload is built before the file lock is taken; the older one loses.

    Staged: the heartbeat's payload is built and its thread is held asking for
    the file lock, while a `note` built after it writes first. Let go, the
    heartbeat's older fields must not replace the note's.
    """
    record = _record(tmp_path, monkeypatch)
    watched = _WatchedFileLock(record._file_lock)
    record._file_lock = watched
    real = watched._real

    real.acquire()      # the test thread holds the file: every other writer waits
    try:
        stale = threading.Thread(target=record.note, kwargs={"phase": "older"},
                                 name="stale-writer", daemon=True)
        stale.start()
        assert watched.asked["stale-writer"].wait(timeout=HELD_S), \
            "the older writer never reached the file lock"
        record.note(phase="newer")      # reentrant: written while it waits
    finally:
        real.release()
    stale.join(timeout=HELD_S)

    assert not stale.is_alive()
    written = json.loads(record.path.read_text(encoding="utf-8"))
    record.finish()
    assert written["phase"] == "newer", (
        "a snapshot built before the newest write replaced it on disk")


def test_an_unhandled_exception_is_named():
    assert exitlog.describe_exception(KeyboardInterrupt, KeyboardInterrupt()) \
        == "interrupted from the keyboard (Ctrl+C)"
    assert exitlog.describe_exception(ValueError, ValueError("bad seed\nmore")) \
        == "unhandled ValueError: bad seed"
