"""Termination guarantees for the parallel runner's worker loop.

Regression coverage for the fix where a worker could keep running (or block in
the session-limit gate) after the list had drained to empty. The worker must:

  * claim work *before* touching the usage gate, so an empty/drained queue stops
    it via claim() -> stop and it never enters the gate with nothing to do;
  * exit promptly once the queue is drained, even with far more workers than
    files (the spare workers must not spin forever);
  * give its claim back if it dies holding one, so the queue can still read as
    drained — a worker killed by an exception latches no stop, and a line stuck
    in `in_progress` keeps every surviving worker in the back-off for ever; and
  * take its provider child with it, so a run that ends leaves nothing of itself
    still running in the terminal.

The last group is time-bounded on purpose: without the fix the run does not end
at all, so an unbounded pin would look like a wedged CI rather than a failure.
"""

import json
import os
import subprocess
import sys
import threading
import time

import pytest

from llm_loop import parallel, stopchannel
from llm_loop.drivers import ListFileDriver
from llm_loop.stopchannel import RunStopReason


class _MemDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real claude)."""

    target_suffix = ".ru.md"

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self._lock = threading.Lock()

    def prompt(self, source, target):
        return "translate"

    def model(self):
        return ""

    def pending_lines(self):
        with self._lock:
            return list(self._items)

    def strike(self, line):
        with self._lock:
            if line in self._items:
                self._items.remove(line)
                return True
            return False


def _args(project_dir, jobs, *, ignore_usage=True, max_runs=None):
    ns = type("NS", (), {})()
    ns.jobs = jobs
    ns.max = max_runs
    ns.dry_run = False
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.ignore_usage = ignore_usage
    return ns


def _run_and_wait(driver, args, timeout=10.0, result_box=None):
    """Run run_parallel on a thread; return True iff it returned within timeout.

    `result_box` is appended the RunResult for the tests that assert on the
    run's own bookkeeping (attempted/completed) rather than only on the list.
    """
    done = threading.Event()

    def go():
        outcome = None
        try:
            outcome = parallel.run_parallel(driver, args,
                                            app_name="pytest-parallel")
        except SystemExit:
            pass
        except BaseException as exc:
            # Recorded, not merely raised on a doomed thread: the pins that read
            # `result_box` silence the thread-death warning (they kill workers on
            # purpose), so a run_parallel that blew up would reach the test as a
            # bare IndexError with its cause nowhere on screen. See `_result`.
            outcome = exc
        finally:
            if result_box is not None and outcome is not None:
                result_box.append(outcome)
            done.set()

    threading.Thread(target=go, daemon=True).start()
    return done.wait(timeout)


def _result(result_box):
    """The RunResult out of `_run_and_wait`'s box, or say what happened instead."""
    assert result_box, "run_parallel returned nothing and raised nothing"
    outcome = result_box[0]
    assert not isinstance(outcome, BaseException), \
        f"run_parallel itself raised: {outcome!r}"
    return outcome


def test_drain_terminates_with_more_workers_than_files(tmp_path, monkeypatch):
    """20 workers, a few files: everyone must exit once the list drains."""
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))
    driver = _MemDriver([f"products/f{i}.md" for i in range(3)])
    assert _run_and_wait(driver, _args(str(tmp_path), jobs=20)), \
        "run_parallel did not terminate after the list drained"
    assert driver.pending_lines() == []


def test_worker_claims_before_touching_the_usage_gate(tmp_path, monkeypatch):
    """On an empty/drained queue no worker may enter the session-limit gate.

    The gate can block (over budget) with no regard for the stop flag, so a
    worker that reached it with nothing to do would wedge and hang the run. With
    an empty list every worker must stop at claim(), so the policy's
    check_and_wait is never called.
    """
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))
    # A source object is truthy so the gate branch is taken *if reached*.
    monkeypatch.setattr(parallel, "usage_source_for", lambda provider: object())

    gate_calls = []

    class SpyPolicy:
        def log_snapshot(self, *a, **k):
            pass

        def check_and_wait(self, source, session_start, note="",
                           cache_value=True, should_stop=None):
            gate_calls.append(1)
            return False, session_start

    # Non-empty list that drains immediately: run_parallel starts workers, they
    # drain the single item, then must exit — the spare workers never gate.
    driver = _MemDriver(["products/only.md"])
    driver.limit_policy = SpyPolicy()
    assert _run_and_wait(driver, _args(str(tmp_path), jobs=20, ignore_usage=False))
    assert driver.pending_lines() == []
    # At most one gate call (the single worker that claimed the one item); the 19
    # spare workers claimed nothing and so never touched the gate.
    assert len(gate_calls) <= 1, f"idle workers entered the usage gate: {gate_calls}"


def test_a_paused_fleet_claims_nothing_until_it_is_let_go(tmp_path, monkeypatch):
    """`p` holds the workers BEFORE the claim, so a paused run holds no file.

    Pausing after the claim would park a line in `in_progress` for the length of
    the hold — where a stop latched meanwhile releases it — and the point of the
    key is that it costs the run nothing.
    """
    paused = threading.Event()
    paused.set()
    started = threading.Event()

    def watched_job(job_id, command, mailbox=None):
        started.set()
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", watched_job)
    monkeypatch.setattr(stopchannel, "pause_requested",
                        lambda app=None: paused.is_set())
    driver = _MemDriver(["products/a.md", "products/b.md"])
    done = threading.Event()

    def go():
        try:
            parallel.run_parallel(driver, _args(str(tmp_path), jobs=3),
                                  app_name="pytest-parallel")
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()

    assert not started.wait(1.0), "a paused fleet started a file anyway"
    assert len(driver.pending_lines()) == 2, "a paused fleet struck a line"
    paused.clear()
    assert done.wait(10), "the released run did not terminate"
    assert driver.pending_lines() == []


def test_a_paused_fleet_still_ends_when_the_queue_drains(tmp_path, monkeypatch):
    """A hold is about not STARTING work, so a run with none left must end.

    Nothing but `claim()` discovers a drained queue, and a held worker does not
    claim — so a fleet that waited for a boundary that will never come never set
    `stop`, and run_parallel's join() never returned. `p` hung the whole run.
    """
    paused = threading.Event()

    def pause_while_it_runs(job_id, command, mailbox=None):
        paused.set()            # `p`, pressed while the last file is running
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", pause_while_it_runs)
    monkeypatch.setattr(stopchannel, "pause_requested",
                        lambda app=None: paused.is_set())
    driver = _MemDriver(["products/only.md"])

    assert _run_and_wait(driver, _args(str(tmp_path), jobs=3)), \
        "a paused fleet with an empty queue never terminated"
    assert paused.is_set(), "the pause was released — the test proved nothing"
    assert driver.pending_lines() == []


def test_a_paused_fleet_still_ends_at_the_item_cap(tmp_path, monkeypatch):
    """The other half: --max-runs is knowable without asking the queue, and the
    sequential loop already ends on it rather than holding — so must this one."""
    paused = threading.Event()

    def pause_while_it_runs(job_id, command, mailbox=None):
        paused.set()
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", pause_while_it_runs)
    monkeypatch.setattr(stopchannel, "pause_requested",
                        lambda app=None: paused.is_set())
    driver = _MemDriver(["products/a.md", "products/b.md"])

    assert _run_and_wait(driver, _args(str(tmp_path), jobs=3, max_runs=1)), \
        "a paused fleet at its item cap never terminated"
    assert paused.is_set(), "the pause was released — the test proved nothing"
    assert len(driver.pending_lines()) == 1


def test_release_returns_claim_to_queue():
    """release() drops the line from in_progress and undoes its --max reservation."""
    driver = _MemDriver(["a", "b"])
    shared = parallel.Shared(driver, max_items=5)
    line = shared.claim()
    assert line in ("a", "b")
    assert shared.claimed == 1 and line in shared.in_progress
    shared.release(line)
    assert shared.claimed == 0 and line not in shared.in_progress
    # The line is still pending (never struck), so a re-run picks it up.
    assert set(driver.pending_lines()) == {"a", "b"}


# How long the "a dying worker must not hang the fleet" pins wait for
# run_parallel to return. Generous next to a healthy run (which ends as soon as
# the last thread joins — well under a second here) and short next to the defect
# it pins, where the run never ends at all: measured at 2 workers / 4 files, the
# survivor was still spinning in the claim back-off 25 s in. The bound is what
# keeps a red pin a failure instead of a wedged CI.
HANG_TIMEOUT_S = 15.0

# The two pins below kill worker threads on purpose, and pytest reports every
# thread that dies of an exception. Here that IS the fixture, so the warning is
# silenced per-test rather than globally — a worker dying anywhere else in the
# suite is still news.
_EXPECTED_THREAD_DEATH = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning")


@_EXPECTED_THREAD_DEATH
def test_a_worker_dying_mid_turn_does_not_hang_the_run(tmp_path, monkeypatch):
    """An exception inside `run_job` must not strand the claim it was holding.

    Nothing about a dying worker sets `stop`, so the line it abandons in
    `in_progress` makes `_exhausted` (`not pending and not in_progress`) false
    for ever: `claim()` answers None, and every surviving worker sits in the
    two-second back-off until the run is killed by hand.

    The counters are the other half of the pin, and they are what distinguishes
    the two ways of giving a claim back (see `Shared.abandon`). A turn that had
    already started is recorded as a FAILED attempt, not released: `attempts` is
    the only thing that can ever park a poison line in `failed`, so a released
    one would go on killing worker after worker.
    """
    poison = "products/poison.md"
    good = [f"products/f{i}.md" for i in range(3)]

    def exploding_job(job_id, command, mailbox=None):
        if command.label == os.path.basename(poison):
            raise BrokenPipeError("the provider CLI died mid-stream")
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", exploding_job)
    driver = _MemDriver(good + [poison])
    # List order, so the poison line is only ever reached after the healthy
    # ones: with the default random pick the same run could kill both workers
    # early and the counts below would depend on the draw.
    driver.pick_order = "list"

    box = []
    assert _run_and_wait(driver, _args(str(tmp_path), jobs=2),
                         timeout=HANG_TIMEOUT_S, result_box=box), \
        "a worker died holding a claim and run_parallel never returned"
    assert driver.pending_lines() == [poison], \
        "the healthy files did not all get struck"
    result = _result(box)
    assert result.completed == 3
    # Two workers, each killed once by the poison file, plus the three healthy
    # claims. A claim released instead of failed would leave this at 3 and the
    # line at zero attempts — retried for ever, one dead worker at a time.
    assert result.attempted == 5, \
        f"a started-then-crashed turn was not counted: {result}"
    # And the run says so. Nobody reached a verdict here — every ending writes
    # `stop_reason` before its worker can leave — so reporting NO_WORK would
    # claim the queue drained while a file is still sitting in it.
    assert result.reason is RunStopReason.WORKERS_DIED, \
        f"a fleet that died reported the wrong reason: {result}"


@_EXPECTED_THREAD_DEATH
def test_a_worker_dying_in_the_usage_gate_gives_the_file_back(
    tmp_path, monkeypatch
):
    """The gate is inside the guarded region too — and its claim goes back WHOLE.

    The gate does network I/O and prints, so it fails for the same reasons the
    console does. Nothing has been attempted at that point, so the line must
    return to the queue verbatim and the run must still complete it: releasing
    it also undoes the --max-runs reservation, which is why `attempted` counts
    the one real claim rather than both.
    """
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))
    monkeypatch.setattr(parallel, "usage_source_for", lambda provider: object())

    calls = []

    class ExplodesOncePolicy:
        def log_snapshot(self, *a, **k):
            pass

        def check_and_wait(self, source, session_start, note="",
                           cache_value=True, should_stop=None):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("usage endpoint blew up")
            return False, session_start

    driver = _MemDriver(["products/only.md"])
    driver.limit_policy = ExplodesOncePolicy()

    box = []
    assert _run_and_wait(driver, _args(str(tmp_path), jobs=2,
                                       ignore_usage=False),
                         timeout=HANG_TIMEOUT_S, result_box=box), \
        "a worker died in the usage gate and run_parallel never returned"
    assert len(calls) >= 2, "the second worker never reached the gate"
    assert driver.pending_lines() == [], \
        "the released file was not picked up and finished by another worker"
    result = _result(box)
    assert result.attempted == 1, \
        f"the abandoned claim's --max reservation was not undone: {result}"


# The fake provider writes one event and then simply stays alive. Long enough
# that "still running when the run ended" cannot be a race with a child about to
# exit by itself — the question this pin asks is whether anything ENDED it.
_ORPHAN_LIFETIME_S = 120

# The tool name whose line the console refuses to print. `LineWriter.tool` puts
# the name in the plain text, so this is what lets the stub writer fail on the
# ONE line that run_job emits with a child running, and stay quiet for the
# worker's own `▶`/verdict lines around it.
_KILLS_THE_WRITER = "boom"

_FAKE_PROVIDER_SRC = "import sys, time\nsys.stdout.write(%r)\nsys.stdout.flush()\ntime.sleep(%d)\n" % (
    json.dumps({"type": "assistant",
                "message": {"content": [{"type": "tool_use",
                                         "name": _KILLS_THE_WRITER,
                                         "input": {}}]}}) + "\n",
    _ORPHAN_LIFETIME_S,
)

# How long a child gets to be dead once run_parallel has returned. The reaping is
# synchronous inside run_job, so a healthy run has already collected it before
# the run ends and this bound is never approached; it is here so the FAILING
# case reports an orphan instead of blocking for `_ORPHAN_LIFETIME_S`.
REAP_WAIT_S = 5.0


def _outlived_the_run(proc, timeout=REAP_WAIT_S) -> bool:
    """Is `proc` still running `timeout` after the run that started it ended?

    Always leaves the child dead: a pin that fails must not also leak the very
    process it is complaining about into the rest of the suite.
    """
    try:
        proc.wait(timeout=timeout)
        return False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=REAP_WAIT_S)
        return True


@_EXPECTED_THREAD_DEATH
def test_a_dying_job_does_not_leave_the_provider_running(tmp_path, monkeypatch):
    """A `run_job` that raises must not walk away from a live child process.

    `proc.wait()` at the bottom of run_job is its only reaping exit, and the
    console write behind every `out.*` line sits above it — a real
    BrokenPipeError there (the same one the fleet pin above is built on) used to
    unwind straight past the child. The provider CLI then kept running: unreaped,
    still holding the stdout it inherited, printing over whatever the terminal
    did next.

    A real subprocess rather than a stub, because the defect is exactly the
    thing a stub does not have — an OS process that outlives the function that
    started it.
    """
    children = []

    def fake_provider(argv, provider, prompt, project_dir):
        proc = subprocess.Popen(
            [sys.executable, "-c", _FAKE_PROVIDER_SRC],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1)
        children.append(proc)
        return proc

    def console_that_dies(plain, markup):
        if _KILLS_THE_WRITER in plain:
            raise BrokenPipeError("the terminal went away mid-line")

    monkeypatch.setattr(parallel, "start_agent_process", fake_provider)
    monkeypatch.setattr(parallel, "print_markup", console_that_dies)

    driver = _MemDriver(["products/only.md"])
    assert _run_and_wait(driver, _args(str(tmp_path), jobs=1),
                         timeout=HANG_TIMEOUT_S), \
        "the worker died holding a claim and run_parallel never returned"
    assert children, "the fake provider was never started — the pin proved nothing"
    assert not _outlived_the_run(children[0]), \
        "run_job unwound past a live provider child: orphaned, not reaped"


def test_max_runs_closes_claims_without_cancelling_in_flight_work(
    tmp_path, monkeypatch
):
    """The batch cap must let every already-claimed item finish."""
    started = 0
    started_lock = threading.Lock()
    all_started = threading.Event()
    release = threading.Event()

    def blocked_job(job_id, command, mailbox=None):
        nonlocal started
        with started_lock:
            started += 1
            if started == 3:
                all_started.set()
        assert release.wait(5), "test did not release the in-flight jobs"
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", blocked_job)
    driver = _MemDriver([f"products/f{i}.md" for i in range(6)])
    done = threading.Event()

    def go():
        try:
            parallel.run_parallel(
                driver, _args(str(tmp_path), jobs=6, max_runs=3),
                app_name="pytest-parallel")
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    assert all_started.wait(5), "fewer than three capped jobs reached execution"
    release.set()
    assert done.wait(5), "capped parallel run did not terminate"
    assert len(driver.pending_lines()) == 3
