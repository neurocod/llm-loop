"""The `stop` sentinel, who may claim it, and what the `s` key does instead.

A stop file is a one-shot signal: the run that honours it keeps it through its
complete application cleanup, then removes it immediately before exit so the
next launch starts clean. That makes *who* claims and removes it a correctness
question. A `--dry-run` previews the commands a run would build, and it is
routinely used while a real loop is running — which is exactly when a stop is
pending. A dry run that consumed the sentinel would silently cancel someone
else's stop request, and the loop it was meant to halt would keep going.

So: the already-running owner claims it at an iteration boundary, a new real
launch waits for it to disappear, and a dry run reports it without touching it.
The owner removes it only at its outermost lifecycle exit. Both runners are
covered, since either entry point could regress independently.

The status line's `s` key is the other half of the story and is deliberately NOT
this file: it sets an in-process flag, so one of several loops sharing a project
root can be stopped on its own (see cyclecore.StopSource).
"""

import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

from llm_loop import cyclecore, parallel, statusline
from llm_loop.cyclecore import ClaudeCommand, Driver
from llm_loop.drivers import ListFileDriver
from llm_loop.limits import LimitPolicy, SessionLimit
from llm_loop.usage import Usage, UsageReading


@pytest.fixture(autouse=True)
def _restore_streams():
    """Both runners tee sys.stdout/stderr into their log and never put them back;
    undo that so one test's tee does not follow the next one."""
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


class _OneShotDriver(Driver):
    """Hands out a single command, then reports the work exhausted."""

    def __init__(self):
        self.served = 0
        self.limit_policy = _StubPolicy()

    def next_command(self):
        if self.served:
            return None
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")


class _StopAfterOneDriver(_OneShotDriver):
    """Creates the sentinel after its first completed provider call."""

    def __init__(self, stop: Path):
        super().__init__()
        self.stop = stop

    def on_success(self, returncode):
        self.stop.write_text("", encoding="utf-8")


class _PressStopAfterOneDriver(_OneShotDriver):
    """Presses `s` on the captured status line after its first provider call.

    `also_touch` additionally writes the sentinel, for the case where both
    channels are up at once.
    """

    def __init__(self, captured: dict, also_touch: Optional[Path] = None):
        super().__init__()
        self.captured = captured
        self.also_touch = also_touch

    def next_command(self):
        # Unlike the one-shot base, keep offering work: the run must end because
        # of the key press, not because the queue ran dry.
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")

    def on_success(self, returncode):
        self.captured["app"].request_stop()
        if self.also_touch is not None:
            self.also_touch.write_text("", encoding="utf-8")


class _StubPolicy:
    """A LimitPolicy that never reads the usage report and never pauses."""

    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="",
                       cache_value=True, should_stop=None):
        return False, session_start


class _MemListDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real claude)."""

    target_suffix = ".ru.md"

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self.limit_policy = _StubPolicy()

    def prompt(self, source, target):
        return "do the thing"

    def pending_lines(self):
        return list(self._items)

    def strike(self, line):
        if line in self._items:
            self._items.remove(line)
            return True
        return False


def _seq_args(project_dir, *, dry_run):
    ns = type("NS", (), {})()
    ns.max = None
    ns.dry_run = dry_run
    ns.raw = False
    ns.start_in = None
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.cost = False
    return ns


def _par_args(project_dir, *, dry_run):
    ns = type("NS", (), {})()
    ns.jobs = 2
    ns.max = None
    ns.dry_run = dry_run
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.ignore_usage = True
    return ns


def _stop_file(project_dir: Path) -> Path:
    path = project_dir / "stop"
    path.write_text("", encoding="utf-8")
    return path


def _await_output(capsys, needle: str, timeout: float = 10.0) -> str:
    """Block until `needle` has been printed, and return everything seen so far.

    The handshake the launch-waits-for-a-stop-file tests need. Both start a
    runner on a background thread and then have to establish that it is being
    held at the startup gate before removing the sentinel - and "sleep a bit,
    then assume it got there" is not that. It cost a CI failure on the slowest
    matrix cell (windows/3.9): thread startup alone outran the 50 ms window, so
    the sentinel was deleted before the runner ever looked for it. The launch
    then sailed through a gate that was already open, the run completed exactly
    as the test demanded, and the only visible symptom was a missing message.
    A wait that is too short does not report "too short" - it reports whatever
    the race happened to produce.

    Waiting on the printed line is what makes it airtight rather than merely
    longer: the gate prints strictly after it has confirmed the sentinel and
    strictly before it starts polling, so seeing that line means the runner is
    inside the wait. Output is accumulated across polls because readouterr()
    drains what it returns.
    """
    seen = ""
    deadline = time.monotonic() + timeout
    while True:
        seen += capsys.readouterr().out
        if needle in seen or time.monotonic() >= deadline:
            return seen
        time.sleep(0.01)


# -- sequential runner ---------------------------------------------------------

def test_dry_run_leaves_the_stop_file(tmp_path, capsys):
    """A previewing run reports the sentinel and leaves it for the real run."""
    stop = _stop_file(tmp_path)
    driver = _OneShotDriver()
    cyclecore.run_loop(driver, _seq_args(str(tmp_path), dry_run=True),
                       app_name="pytest-stop")
    assert stop.exists(), "dry run consumed the stop file"
    out = capsys.readouterr().out
    assert "Stop file present" in out
    # It reports the sentinel *and* still previews — a pending stop must not
    # make -d useless, which is the whole reason it does not simply break here.
    assert "DRY-RUN:" in out
    assert driver.served == 1


def test_real_launch_waits_for_an_existing_stop_file(tmp_path, monkeypatch, capsys):
    """A new launch leaves another run's pending stop request alone."""
    monkeypatch.setattr(cyclecore, "STOP_POLL_SECONDS", 0.01)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)
    stop = _stop_file(tmp_path)
    driver = _OneShotDriver()
    done = threading.Event()

    def go():
        try:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path), dry_run=False),
                               app_name="pytest-stop")
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    output = _await_output(capsys, "Stop file present at startup")
    assert "Stop file present at startup" in output, "launch never reached the gate"
    assert not done.wait(0.05), "launch ignored an existing stop file"
    stop.unlink()
    assert done.wait(10), "launch did not continue after the stop file was removed"
    assert not stop.exists(), "a real run left the stop file behind"
    output += capsys.readouterr().out
    assert "Stop file removed" in output
    assert driver.served == 1


def test_dry_run_reports_the_stop_file_once(tmp_path, capsys):
    """--max-runs keeps the dry run looping; the note must not repeat per pass."""
    _stop_file(tmp_path)
    args = _seq_args(str(tmp_path), dry_run=True)
    args.max = 3
    cyclecore.run_loop(_OneShotDriver(), args, app_name="pytest-stop")
    assert capsys.readouterr().out.count("Stop file present") == 1


def test_sequential_stop_file_lives_until_outer_application_exit(
    tmp_path, monkeypatch, capsys
):
    stop = tmp_path / "stop"
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)

    with cyclecore.stop_file_lifecycle():
        result = cyclecore.run_loop(
            _StopAfterOneDriver(stop),
            _seq_args(str(tmp_path), dry_run=False),
            app_name="pytest-stop")
        assert result.reason == cyclecore.RunStopReason.STOP_FILE
        assert stop.exists(), "runner removed the mutex before application cleanup"

    assert not stop.exists(), "application exit left the stop mutex behind"
    out = capsys.readouterr().out
    assert out.index("remains in place") < out.index("removed on application exit")


def test_the_s_key_stops_the_sequential_run_without_writing_a_sentinel(
    tmp_path, monkeypatch
):
    """A key press must not become a project-wide brake: the neighbouring loop in
    the same root sees no sentinel, and this one still ends on the request."""
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)
    captured = {}
    real_app_class = statusline.StatusApp

    def _app(**kwargs):
        kwargs["enabled"] = False   # no terminal in pytest; the flag is the point
        captured["app"] = real_app_class(**kwargs)
        return captured["app"]

    monkeypatch.setattr(statusline, "StatusApp", _app)

    with cyclecore.stop_file_lifecycle():
        result = cyclecore.run_loop(_PressStopAfterOneDriver(captured),
                                    _seq_args(str(tmp_path), dry_run=False),
                                    app_name="pytest-stop")

    assert result.reason == cyclecore.RunStopReason.STOP_KEY
    assert result.completed == 1, "the key press did not stop at the boundary"
    assert not (tmp_path / "stop").exists(), "the s key wrote a stop file"


def test_a_sentinel_present_at_stop_time_is_consumed_even_if_s_was_pressed(
    tmp_path, monkeypatch
):
    """The chaining handshake outranks the key press.

    A script that writes the sentinel to end run A and start run B behind it
    (run B waits in wait_for_stop_file_clear, which has NO timeout) must get its
    file consumed even when the user also pressed `s` — reporting the key press
    and walking past the file would strand run B forever.
    """
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)
    stop = tmp_path / "stop"
    captured = {}
    real_app_class = statusline.StatusApp

    def _app(**kwargs):
        kwargs["enabled"] = False
        captured["app"] = real_app_class(**kwargs)
        return captured["app"]

    monkeypatch.setattr(statusline, "StatusApp", _app)

    with cyclecore.stop_file_lifecycle():
        result = cyclecore.run_loop(
            _PressStopAfterOneDriver(captured, also_touch=stop),
            _seq_args(str(tmp_path), dry_run=False), app_name="pytest-stop")
        assert stop.exists(), "consumed before the application finished cleanup"

    assert result.reason == cyclecore.RunStopReason.STOP_FILE
    assert not stop.exists(), "the sentinel outlived the run it stopped"


# -- parallel runner -----------------------------------------------------------

def test_parallel_dry_run_leaves_the_stop_file(tmp_path, capsys):
    """The workers are what remove the sentinel, and a dry run starts none."""
    stop = _stop_file(tmp_path)
    parallel.run_parallel(_MemListDriver(["products/a.md"]),
                          _par_args(str(tmp_path), dry_run=True),
                          app_name="pytest-stop-parallel")
    assert stop.exists(), "parallel dry run consumed the stop file"
    out = capsys.readouterr().out
    assert "stop file present" in out.lower()
    assert "DRY-RUN" in out


def test_parallel_launch_waits_for_an_existing_stop_file(tmp_path, monkeypatch, capsys):
    """A parallel launch also waits rather than stealing a pending stop."""
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))
    monkeypatch.setattr(cyclecore, "STOP_POLL_SECONDS", 0.01)
    stop = _stop_file(tmp_path)
    done = threading.Event()

    def go():
        try:
            args = _par_args(str(tmp_path), dry_run=False)
            args.max = 1
            parallel.run_parallel(_MemListDriver(["products/a.md"]), args,
                                  app_name="pytest-stop-parallel")
        except SystemExit:
            pass
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    output = _await_output(capsys, "Stop file present at startup")
    assert "Stop file present at startup" in output, "launch never reached the gate"
    assert not done.wait(0.05), "parallel launch ignored an existing stop file"
    stop.unlink()
    assert done.wait(10), "run_parallel did not continue after the stop file was removed"
    assert not stop.exists(), "parallel run left the stop file behind"


def test_parallel_stop_file_lives_until_outer_application_exit(
    tmp_path, monkeypatch
):
    stop = tmp_path / "stop"
    calls = 0

    def stop_after_first_job(job_id, command, mailbox=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            stop.write_text("", encoding="utf-8")
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", stop_after_first_job)
    args = _par_args(str(tmp_path), dry_run=False)
    args.jobs = 1

    with cyclecore.stop_file_lifecycle():
        result = parallel.run_parallel(
            _MemListDriver(["products/a.md", "products/b.md"]), args,
            app_name="pytest-stop-parallel")
        assert result.reason == cyclecore.RunStopReason.STOP_FILE
        assert stop.exists(), "workers removed the mutex before application cleanup"

    assert not stop.exists(), "application exit left the stop mutex behind"


def test_parallel_stop_file_is_reported_once_by_competing_workers(
    tmp_path, monkeypatch, capsys
):
    stop = tmp_path / "stop"
    stop_created = threading.Event()
    jobs_ready = threading.Barrier(4)
    checks_ready = threading.Barrier(4)
    checked_threads = set()
    checked_lock = threading.Lock()
    real_exists = os.path.exists

    def finish_jobs_together(job_id, command, mailbox=None):
        jobs_ready.wait(timeout=5)
        if job_id == 1:
            stop.write_text("", encoding="utf-8")
            stop_created.set()
        assert stop_created.wait(5)
        return 0, 0.0, 0.01

    def coordinate_first_worker_checks(path):
        thread_name = threading.current_thread().name
        if (path == str(stop) and stop_created.is_set()
                and thread_name.startswith("job")):
            with checked_lock:
                first = thread_name not in checked_threads
                checked_threads.add(thread_name)
            if first:
                checks_ready.wait(timeout=5)
        return real_exists(path)

    monkeypatch.setattr(parallel, "run_job", finish_jobs_together)
    monkeypatch.setattr(parallel.os.path, "exists", coordinate_first_worker_checks)
    args = _par_args(str(tmp_path), dry_run=False)
    args.jobs = 4

    result = parallel.run_parallel(
        _MemListDriver([f"products/{i}.md" for i in range(8)]), args,
        app_name="pytest-stop-parallel")

    assert result.reason == cyclecore.RunStopReason.STOP_FILE
    # The WHOLE sentence, and it is the one the sequential runner prints too
    # (both tails are cyclecore.commit_stop). Counting a short fragment is how
    # the two copies drifted into two wordings without a test noticing.
    # Whitespace is collapsed first because a job row goes through rich, which
    # wraps it at the console width - measured: this sentence lands on two rows
    # at 80 columns, exactly as the "no new files will be claimed" line beside
    # it already did.
    out = " ".join(capsys.readouterr().out.split())
    assert out.count("Stop file detected — stopping; it remains in place "
                     "until the application exits.") == 1


# The announcing worker's thread really does die on the refused write — that is
# the unchanged half of the cost (a `worker` has no try/except, and had none
# before the two tails were merged either), and pytest reports the dead thread as
# a warning. Ignored here rather than left to a future `-W error` to turn into a
# red test about the very thing this test arranges.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_parallel_stop_file_latches_even_when_the_console_write_fails(
    tmp_path, monkeypatch
):
    """A refused announcement costs the line, never the reason.

    The stop tail runs on the one worker that won the latch, and its console
    write can fail for reasons that have nothing to do with the run: a closed
    pipe (`llm-loop … | head`), a code page that cannot spell the em dash, rich
    itself. Written before the transition, that exception unwinds the winner
    with nothing latched — no reason, no `stop` event — the remaining workers
    find the run still going and die on the same line in turn, and `run_parallel`
    falls back to `shared.stop_reason or NO_WORK`. A run that obeyed a stop FILE
    then reports that it simply ran out of work: the sentinel is consumed, the
    exit log names the wrong cause, and a wrapper reading the reason to decide
    whether to start the next phase starts it.

    So the pin is on the REASON, not on the output — the whole point is that the
    output failed. `print_markup` is where a real console write lands
    (`_emit_markup` re-reads it per call for exactly this kind of pin).
    """
    stop = tmp_path / "stop"
    calls = 0
    real_print_markup = parallel.print_markup

    def stop_after_first_job(job_id, command, mailbox=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            stop.write_text("", encoding="utf-8")
        return 0, 0.0, 0.01

    def refuse_the_stop_line(plain, markup):
        if "Stop file detected" in plain:
            raise BrokenPipeError("stdout closed under the stop announcement")
        real_print_markup(plain, markup)

    monkeypatch.setattr(parallel, "run_job", stop_after_first_job)
    monkeypatch.setattr(parallel, "print_markup", refuse_the_stop_line)
    args = _par_args(str(tmp_path), dry_run=False)
    args.jobs = 1

    with cyclecore.stop_file_lifecycle():
        result = parallel.run_parallel(
            _MemListDriver(["products/a.md", "products/b.md"]), args,
            app_name="pytest-stop-parallel")
        assert result.reason == cyclecore.RunStopReason.STOP_FILE, (
            "a failed write to the console changed why the run ended")
        assert stop.exists(), "workers removed the mutex before application cleanup"

    assert not stop.exists(), "application exit left the stop mutex behind"


# -- asking a run that is parked on the usage wall to stop -----------------------
#
# The pause on the account's budget is the longest hold in the engine: hours,
# with every worker inside it and nothing else running. A hold that does not
# watch the stop channels therefore takes the brakes with it — there is nobody
# left to notice `s` or the sentinel until the window resets, and from outside
# the run is indistinguishable from a hung one.


class _PeggedSource:
    """A usage source stuck over any sane ceiling, with a distant reset."""

    def __init__(self, percent: float = 99.0, resets_in: float = 3600.0):
        reading = UsageReading(percent, time.time() + resets_in)
        self.usage = Usage(session=reading, week_all=reading,
                           week_sonnet=reading, summary_lines=[])

    def get_usage(self, cache_value=True):
        return self.usage

    def invalidate(self):
        pass


class _NeverEndingDriver(Driver):
    """Always has work, and gates on a real (flat 80%) session limit."""

    limit_policy = LimitPolicy([SessionLimit(80)])

    def next_command(self):
        return ClaudeCommand("do the thing", "", "the-thing")


def _capture_disabled_app(monkeypatch) -> dict:
    """Capture the run's StatusApp, screenless (pytest has no terminal).

    Disabled is what the `s` key needs here anyway: the flag it sets is
    in-process, and a screenless app skips the cancel countdown, which is only
    offered to somebody who can see the row it counts down on.
    """
    captured = {}
    real_app_class = statusline.StatusApp

    def _app(**kwargs):
        kwargs["enabled"] = False
        captured["app"] = real_app_class(**kwargs)
        return captured["app"]

    monkeypatch.setattr(statusline, "StatusApp", _app)
    return captured


def cancel_it(app=None, **kwargs):
    """Stand-in for `confirm_stop_request` when the user presses `s` again.

    The countdown itself is pinned elsewhere; what these tests are about is what
    a runner does with a request that turns out to be withdrawn.
    """
    app.cancel_stop()
    return False


class _Held:
    """A runner started on a thread, so the test can press `s` while it holds.

    Every test below needs the same three moments — start it, catch it inside
    the hold (by the line it prints, never by a sleep), then let it finish — and
    each of them is a race written the same way three times if it is written at
    all.
    """

    def __init__(self, call):
        self.result = None
        self.done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(call,),
                                        daemon=True)
        self._thread.start()

    def _run(self, call):
        try:
            self.result = call()
        finally:
            self.done.set()

    def wait_until_held(self, capsys, needle: str = "Over usage limit") -> None:
        assert needle in _await_output(capsys, needle), \
            f"the run never reached the hold ({needle!r} never printed)"

    def wait_for_exit(self, why: str) -> None:
        assert self.done.wait(10), why


def test_the_s_key_ends_a_sequential_run_parked_on_the_usage_limit(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)
    monkeypatch.setattr(cyclecore, "usage_source_for",
                        lambda provider: _PeggedSource())
    captured = _capture_disabled_app(monkeypatch)

    held = _Held(lambda: cyclecore.run_loop(
        _NeverEndingDriver(), _seq_args(str(tmp_path), dry_run=False),
        app_name="pytest-stop"))
    held.wait_until_held(capsys)
    captured["app"].request_stop()

    held.wait_for_exit("the s key was ignored while the run sat on the limit")
    assert held.result.reason == cyclecore.RunStopReason.STOP_KEY
    assert not (tmp_path / "stop").exists(), "the s key wrote a stop file"


def test_the_s_key_ends_a_parallel_fleet_parked_on_the_usage_limit(
    tmp_path, monkeypatch, capsys
):
    """The reported symptom: with the budget spent and every job paused, `s`
    did nothing at all — the one moment the key is most likely to be pressed."""
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))
    monkeypatch.setattr(parallel, "usage_source_for",
                        lambda provider: _PeggedSource())
    captured = _capture_disabled_app(monkeypatch)
    driver = _MemListDriver(["products/a.md", "products/b.md"])
    driver.limit_policy = LimitPolicy([SessionLimit(80)])
    args = _par_args(str(tmp_path), dry_run=False)
    args.ignore_usage = False

    held = _Held(lambda: parallel.run_parallel(
        driver, args, app_name="pytest-stop-parallel"))
    held.wait_until_held(capsys)
    captured["app"].request_stop()

    held.wait_for_exit("the s key was ignored while the fleet sat on the limit")
    assert held.result.reason == cyclecore.RunStopReason.STOP_KEY
    # Nothing ran, so nothing may be struck: a file claimed and handed back is
    # still pending work for the next run.
    assert driver.pending_lines() == ["products/a.md", "products/b.md"]


def test_cancelling_the_stop_keeps_the_file_a_parked_worker_had_claimed(
    tmp_path, monkeypatch, capsys
):
    """"A mis-pressed `s` costs nothing but the files not claimed in between."

    The worker that wakes out of the usage hold is holding a claim, and a claim
    closed by --max-runs (or a drained queue) cannot be made a second time. Hand
    it back to go and read the request, and a withdrawn request has silently
    eaten a whole item of the run's budget: claims are shut, so nobody ever
    picks it up again.
    """
    ran = []
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, cmd, mailbox=None: (
                            ran.append(cmd.label), (0, 0.0, 0.01))[1])
    monkeypatch.setattr(parallel, "usage_source_for",
                        lambda provider: _PeggedSource())
    monkeypatch.setattr(cyclecore, "confirm_stop_request", cancel_it)
    captured = _capture_disabled_app(monkeypatch)
    driver = _MemListDriver(["products/a.md", "products/b.md"])
    driver.limit_policy = LimitPolicy([SessionLimit(80)])
    args = _par_args(str(tmp_path), dry_run=False)
    args.ignore_usage = False
    args.max = 1               # claims shut the moment the one claim is made

    held = _Held(lambda: parallel.run_parallel(
        driver, args, app_name="pytest-stop-parallel"))
    held.wait_until_held(capsys)
    captured["app"].request_stop()

    held.wait_for_exit("the withdrawn request did not release the worker")
    assert len(ran) == 1, f"the claimed file was dropped by the cancel: {ran}"
    assert held.result.completed == 1
    assert len(driver.pending_lines()) == 1, "the finished file was not struck"


def test_a_withdrawn_request_does_not_reset_the_consecutive_error_brake(
    tmp_path, monkeypatch
):
    """The five-errors-in-a-row brake is the only thing that ends a run whose
    provider is simply broken. Pressing `s` and taking it back must not hand
    that provider a fresh set of five."""
    calls = []

    def always_fails(cmd, raw, partial, prompt="", mailbox=None):
        calls.append(1)
        if len(calls) == 2:
            captured["app"].request_stop()
        return 1

    monkeypatch.setattr(cyclecore, "run_claude_streaming", always_fails)
    monkeypatch.setattr(cyclecore, "confirm_stop_request", cancel_it)
    captured = _capture_disabled_app(monkeypatch)

    with pytest.raises(SystemExit):
        cyclecore.run_loop(_PressStopAfterOneDriver(captured),
                           _seq_args(str(tmp_path), dry_run=False),
                           app_name="pytest-stop")

    assert len(calls) == 5, \
        f"the brake was reset by the withdrawn request: {len(calls)} attempts"


def test_the_post_refusal_wait_answers_a_stop_request_too():
    """The other hold that outlasts an iteration: a run refused on the wire
    parks until that window refreshes, which is hours — and `s` has to reach it
    there as well, not only in the proactive usage gate."""
    started = time.time()

    # The real horizon is a window reset, hours out. Ninety seconds is the same
    # shape and keeps a REGRESSION bounded: a wait that ignores the request has
    # to sit through a full sleep slice before it can return, so the assertion
    # below fails in a minute and a half instead of hanging out the hour.
    left_early = cyclecore.wait_until(started + 90, reason="parked for a test",
                                      should_stop=lambda: True)

    assert left_early is True
    assert time.time() - started < 5, "the wait ran its clock instead of stopping"


def test_stop_file_path_follows_the_project_root(tmp_path):
    """The sentinel is resolved against the chosen root, not the cwd — otherwise
    a -C run would watch the wrong file (and the tests would pass by accident)."""
    previous = cyclecore.project_dir()
    try:
        cyclecore.set_project_root(str(tmp_path))
        assert cyclecore.STOP_FILE == os.path.join(str(tmp_path), "stop")
    finally:
        cyclecore.set_project_root(previous)
