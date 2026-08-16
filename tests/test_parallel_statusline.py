"""The parallel runner's half of the pinned status line (wave 3).

Two things are pinned here, and only the first is cosmetic:

  * one Job per worker, each fed by the worker that owns it — the same data feed
    the sequential loop performs on its single Job, so the rows come out of the
    same renderer with no "parallel" branch anywhere in it; and
  * the stop grace. A stop request closes new claims but never kills work in
    flight, and while a job is still running the sentinel disappearing REOPENS
    the claims and the run carries on — which is what makes a mis-pressed `s`
    recoverable. Two things must NOT get that grace: a sentinel this run did not
    write (nobody is at the terminal to press `s` again) and a --max-runs
    boundary (a cap is not a request).
"""

import io
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_loop import cyclecore, parallel
from claude_loop import statusline as sl
from claude_loop.drivers import ListFileDriver


class _MemDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real provider)."""

    target_suffix = ".out.md"
    pick_order = "list"          # deterministic: the tests name the first item

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self._lock = threading.Lock()

    def prompt(self, source, target):
        return f"do {os.path.basename(source)}"

    def model(self):
        return "opus"

    def add(self, item):
        with self._lock:
            self._items.append(item)

    def pending_lines(self):
        with self._lock:
            return list(self._items)

    def strike(self, line):
        with self._lock:
            if line in self._items:
                self._items.remove(line)
                return True
            return False


class _LiveTerminal(sl.Terminal):
    """Active from reserve() on, with no screen behind it.

    The grace is interactive-only (`app.enabled`), so a NullTerminal — which is
    what a test process without a tty otherwise gets — cannot express the very
    behaviour under test.
    """

    def __init__(self):
        super().__init__(stream=io.StringIO())
        self._on = False

    @property
    def active(self):
        return self._on

    def size(self):
        return (120, 30)

    def reserve(self, rows):
        self._on = True
        return True

    def paint(self, lines, *, reassert=False):
        return True

    def release(self):
        self._on = False


def _args(project_dir, jobs, *, max_runs=None, dry_run=False,
          no_statusline=False):
    ns = type("NS", (), {})()
    ns.jobs = jobs
    ns.max = max_runs
    ns.dry_run = dry_run
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.ignore_usage = True
    ns.no_statusline = no_statusline
    return ns


def _live_statusline(monkeypatch, made, *, live=True):
    """Make run_parallel build a status line the test can drive, and capture it."""
    real_app_class = sl.StatusApp        # captured before the patch below

    def _app(**kwargs):
        enabled = kwargs.pop("enabled", True)
        terminal = _LiveTerminal() if (live and enabled) else sl.NullTerminal()
        app = real_app_class(terminal=terminal,
                             input_source=sl.NullInputSource(), refresh=60,
                             **kwargs)
        made["app"] = app
        made.setdefault("enabled", enabled)
        return app

    monkeypatch.setattr(sl, "StatusApp", _app)

    real_shared_class = parallel.Shared

    def _shared(driver, max_items):
        shared = real_shared_class(driver, max_items)
        made["shared"] = shared
        return shared

    monkeypatch.setattr(parallel, "Shared", _shared)
    return made


def _run_in_thread(driver, args, done=None):
    done = done or threading.Event()
    result = {}

    def go():
        try:
            result["value"] = parallel.run_parallel(
                driver, args, app_name="pytest-parallel-statusline")
        except SystemExit:
            pass
        finally:
            done.set()

    thread = threading.Thread(target=go, daemon=True)
    thread.start()
    return done, result


# --- one Job per worker, fed by the worker that owns it -------------------------


def test_a_job_row_per_worker_carries_its_item_and_model(tmp_path, monkeypatch):
    """-j 3 pins three rows, and each one shows what ITS worker is running."""
    made = _live_statusline(monkeypatch, {})
    seen = {}
    running = threading.Barrier(3, timeout=5)

    def record_job(job_id, command):
        frozen = made["app"].status.snapshot()
        seen[job_id] = [j for j in frozen.jobs if j.job_id == job_id][0]
        running.wait()          # hold all three at once: three rows, three items
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", record_job)
    driver = _MemDriver([f"products/f{i}.md" for i in range(3)])
    previous = cyclecore.project_dir()
    try:
        parallel.run_parallel(driver, _args(str(tmp_path), jobs=3),
                              app_name="pytest-parallel-statusline")
    finally:
        cyclecore.set_project_root(previous)

    app = made["app"]
    assert [j.job_id for j in app.status.jobs] == [1, 2, 3]
    assert sorted(seen) == [1, 2, 3]
    for job_id, job in seen.items():
        assert job.running and job.started_at > 0
        assert job.item.endswith(".md") and job.model == "opus"
        assert job.iteration == 1        # its first file, counted on the claim
        assert job.prompt.startswith("do ")
    # Distinct items: a worker must feed its OWN row, not a shared one.
    assert len({job.item for job in seen.values()}) == 3
    # And every row is idle again once the run is over.
    assert all(not job.running and job.started_at == 0
               for job in app.status.jobs)
    assert app.status.iteration == 3     # the run's own counter, not a job's


def test_the_summary_row_shows_the_run_counters(tmp_path, monkeypatch):
    """--max-runs caps FILES here; one claimed file is one iteration of work."""
    made = _live_statusline(monkeypatch, {})
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd: (0, 0.0, 0.01))
    driver = _MemDriver([f"products/f{i}.md" for i in range(5)])
    previous = cyclecore.project_dir()
    try:
        parallel.run_parallel(driver, _args(str(tmp_path), jobs=2, max_runs=2),
                              app_name="pytest-parallel-statusline")
    finally:
        cyclecore.set_project_root(previous)

    app = made["app"]
    assert app.status.max_iterations == 2
    assert app.status.iteration == 2
    assert ("max-runs", "2") in app.status.script_limits
    summary = app.render(width=200)[1]
    assert "iter 2/2" in summary and "2 jobs" in summary


def test_no_statusline_reaches_the_parallel_runner(tmp_path, monkeypatch):
    """The flag the periodic wrapper forwards must actually disable the area."""
    made = _live_statusline(monkeypatch, {})
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd: (0, 0.0, 0.01))
    previous = cyclecore.project_dir()
    try:
        parallel.run_parallel(_MemDriver(["products/a.md"]),
                              _args(str(tmp_path), jobs=1, no_statusline=True),
                              app_name="pytest-parallel-statusline")
    finally:
        cyclecore.set_project_root(previous)

    assert made["enabled"] is False
    assert not made["app"].enabled


# --- the stop grace -------------------------------------------------------------


def test_cancelling_the_stop_reopens_claims_while_a_job_runs(tmp_path, monkeypatch):
    """`s` twice, with a job still moving, must leave no trace on the run."""
    made = _live_statusline(monkeypatch, {})
    driver = _MemDriver(["products/a.md"])
    in_flight = threading.Event()
    release = threading.Event()

    def block_the_first_job(job_id, command):
        if not in_flight.is_set():
            in_flight.set()
            assert release.wait(5), "the test never released the in-flight job"
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", block_the_first_job)
    previous = cyclecore.project_dir()
    try:
        done, result = _run_in_thread(driver, _args(str(tmp_path), jobs=2))
        assert in_flight.wait(5), "no worker ever started a file"

        app = made["app"]
        app.request_stop()                       # the `s` key, pressed here
        shared = made["shared"]
        assert shared.stop_requested.wait(5), "the request never closed claims"
        assert not done.wait(0.5), "the run ended while a job was in flight"

        app.cancel_stop()                        # `s` again, mind changed
        driver.add("products/b.md")              # provable only if claims reopen
        release.set()
        assert done.wait(10), "the run did not carry on after the cancel"
    finally:
        release.set()
        cyclecore.set_project_root(previous)

    assert driver.pending_lines() == [], "claims never reopened"
    assert result["value"].reason is not cyclecore.RunStopReason.STOP_FILE
    assert result["value"].completed == 2
    assert not (tmp_path / "stop").exists()


def test_an_interactive_stop_left_alone_ends_the_run(tmp_path, monkeypatch):
    """The other half of the toggle: not cancelled means stopped, in-flight work
    finished, nothing new claimed."""
    made = _live_statusline(monkeypatch, {})
    monkeypatch.setattr(cyclecore, "STOP_GRACE_SECONDS", 0.05)
    driver = _MemDriver(["products/a.md", "products/b.md"])
    in_flight = threading.Event()
    release = threading.Event()

    def block_the_first_job(job_id, command):
        if not in_flight.is_set():
            in_flight.set()
            assert release.wait(5), "the test never released the in-flight job"
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", block_the_first_job)
    previous = cyclecore.project_dir()
    try:
        with cyclecore.stop_file_lifecycle():
            done, result = _run_in_thread(driver, _args(str(tmp_path), jobs=1))
            assert in_flight.wait(5)
            made["app"].request_stop()
            assert not done.wait(0.3), "in-flight work was cut short"
            release.set()
            assert done.wait(10), "the stop request was never acted on"
            assert (tmp_path / "stop").exists(), "sentinel dropped before exit"
    finally:
        release.set()
        cyclecore.set_project_root(previous)

    assert result["value"].reason is cyclecore.RunStopReason.STOP_FILE
    assert driver.pending_lines() == ["products/b.md"]   # nothing new claimed


def test_a_sentinel_this_run_did_not_write_is_not_reopenable(tmp_path, monkeypatch):
    """An external `touch stop` gets no grace even with the status line live —
    nobody is sitting at this terminal to take it back.

    Pinned on the promptness, which is where the two paths differ: the run must
    latch the stop while a job is STILL in flight, instead of holding the request
    open (and cancellable) until the last one drains.
    """
    made = _live_statusline(monkeypatch, {})
    driver = _MemDriver(["products/a.md"])
    in_flight = threading.Event()
    release = threading.Event()

    def block_the_first_job(job_id, command):
        if not in_flight.is_set():
            in_flight.set()
            assert release.wait(5)
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", block_the_first_job)
    previous = cyclecore.project_dir()
    try:
        with cyclecore.stop_file_lifecycle():
            done, result = _run_in_thread(driver, _args(str(tmp_path), jobs=2))
            assert in_flight.wait(5)
            (tmp_path / "stop").write_text("", encoding="utf-8")   # not the `s` key
            driver.add("products/b.md")     # offered only after the sentinel
            assert made["shared"].stop.wait(5), \
                "an external sentinel waited for the run to drain"
            release.set()
            assert done.wait(10), "an external sentinel did not stop the run"
    finally:
        release.set()
        cyclecore.set_project_root(previous)

    assert made["app"].enabled is False   # released with the run, but it WAS live
    assert result["value"].reason is cyclecore.RunStopReason.STOP_FILE
    assert driver.pending_lines() == ["products/b.md"]   # nothing new claimed


def test_a_max_items_stop_is_never_reopened():
    """The cap and the request both close claims; only the request is undoable."""
    driver = _MemDriver(["a", "b"])
    shared = parallel.Shared(driver, max_items=1)

    assert shared.claim() in ("a", "b")
    assert shared.claim() is None and shared.claims_closed.is_set()
    assert not shared.reopen_claims(), "the cap looked like a cancellable request"

    # Even after a stop request comes and goes, the cap keeps claims shut.
    shared.request_stop(1)
    assert shared.reopen_claims()
    assert shared.claim() is None
    assert shared.stop_reason is cyclecore.RunStopReason.LIMIT_REACHED


def test_only_one_worker_owns_the_pending_request():
    """N workers counting down together would each latch on their own deadline."""
    shared = parallel.Shared(_MemDriver(["a"]), max_items=None)

    assert shared.request_stop(2) == (True, True)     # owner, and the first pass
    assert shared.request_stop(3) == (False, False)
    assert shared.request_stop(2) == (True, False)    # still ours, no second log
    assert shared.reopen_claims() and shared.stop_owner is None


# --- periodic runs alternate the two runners --------------------------------------


class _RecordingTerminal(_LiveTerminal):
    """A live terminal that logs when its region opens and closes."""

    def __init__(self, log):
        super().__init__()
        self.log = log

    def reserve(self, rows):
        self.log.append(("reserve", id(self)))
        return super().reserve(rows)

    def release(self):
        if self._on:
            self.log.append(("release", id(self)))
        super().release()


def test_a_periodic_batch_never_stacks_two_status_areas(tmp_path, monkeypatch):
    """--grow-kit-periodically alternates parallel batches with sequential
    sweeps; two regions pinned at once would fight over the same screen rows."""
    from claude_loop.cyclecore import Driver

    log = []
    real_app_class = sl.StatusApp

    def _app(**kwargs):
        kwargs.pop("enabled", None)
        return real_app_class(terminal=_RecordingTerminal(log),
                              input_source=sl.NullInputSource(), refresh=60,
                              **kwargs)

    monkeypatch.setattr(sl, "StatusApp", _app)
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd: (0, 0.0, 0.01))
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: None)

    class _NoWork(Driver):
        def next_command(self):
            return None

    seq_args = type("NS", (), {})()
    for name, value in dict(max=1, dry_run=False, raw=False, start_in=None,
                            max_strike=None, git_push="none", cost=False,
                            no_statusline=False,
                            project_dir=str(tmp_path)).items():
        setattr(seq_args, name, value)

    previous = cyclecore.project_dir()
    try:
        for _batch in range(2):
            parallel.run_parallel(
                _MemDriver(["products/a.md"]), _args(str(tmp_path), jobs=2),
                app_name="pytest-parallel-statusline", setup_logging=False,
                wait_on_start=False)
            cyclecore.run_loop(_NoWork(), seq_args,
                               app_name="pytest-parallel-statusline",
                               setup_logging=False, wait_on_start=False)
    finally:
        cyclecore.set_project_root(previous)

    assert [event for event, _ in log] == ["reserve", "release"] * 4
    # Each region is closed by the run that opened it, and every run gets a new
    # one — no region survives its runner to be inherited by the next phase.
    pairs = list(zip(log[::2], log[1::2]))
    assert all(opened == closed for (_r, opened), (_x, closed) in pairs)
    assert len({opened for (_r, opened) in log[::2]}) == 4


# --- the dry run prints job 1's prompt ------------------------------------------


def test_a_parallel_dry_run_prints_the_prompt_block_once(tmp_path, capsys):
    """The joined `-p …` argv line hides the prompt; the block is what shows it."""
    driver = _MemDriver(["products/a.md", "products/b.md"])
    previous = cyclecore.project_dir()
    streams = (sys.stdout, sys.stderr)
    try:
        parallel.run_parallel(driver, _args(str(tmp_path), jobs=2, dry_run=True),
                              app_name="pytest-parallel-statusline")
    finally:
        cyclecore.set_project_root(previous)
        sys.stdout, sys.stderr = streams

    out = capsys.readouterr().out
    assert "DRY-RUN:" in out                       # unchanged, tests match on it
    assert out.count("prompt · job 1") == 1        # once, not per file
    assert "prompt · job 1 · a.md · 7 chars" in out
    assert out.index("DRY-RUN:") < out.index("prompt · job 1")
