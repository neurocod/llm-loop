"""What a list driver owns about its items, and how both runners honour it.

Five things a wrapper used to build outside the engine (products/runResearch.py
had all five): an item KEY that two lines can share (`item_key`), the line a
finished command served (`AgentCommand.item`), a verdict that can refuse a turn
which exited 0 (`settle` -> `Settled.REFUSED`), an ending that says "a human is
needed" with an exit code (`NeedsHuman`), and the one public writer of the list
(`replace_line`).
"""

import threading
import time

import pytest

from llm_loop import cyclecore, exitlog, parallel, stopchannel
from llm_loop.agentwork import NeedsHuman
from llm_loop.drivers import AttemptLedger, ListFileDriver, Settled
from llm_loop.stopchannel import RunStopReason

from _runfixtures import (MemListDriver, StubPolicy, isolated_run, par_args,
                          seq_args)

MAX_ATTEMPTS = AttemptLedger.MAX_ATTEMPTS


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
@pytest.mark.parametrize("after_turn", [False, True])
def test_polled_human_stops_before_the_next_claim(
        tmp_path, monkeypatch, capsys, reasons, runner, after_turn):
    calls = []

    class Driver(MemListDriver):
        def needs_human(self):
            if not after_turn or calls:
                return NeedsHuman("poll blocker", exit_code=3)

    driver = Driver(["a", "b"])
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *a, **kw: calls.append(1) or 0)
    monkeypatch.setattr(parallel, "run_job",
                        lambda *a, **kw: (calls.append(1) or 0, 0, 0))
    with pytest.raises(SystemExit) as stopped:
        if runner == "parallel":
            parallel.run_parallel(driver, par_args(
                tmp_path, jobs=1, no_statusline=True), wait_on_start=False)
        else:
            cyclecore.run_loop(driver, _seq(tmp_path), wait_on_start=False)
    assert stopped.value.code == 3
    assert len(calls) == int(after_turn)
    assert any("the driver needs a human (exit 3): poll blocker" == r
               for r in reasons)
    assert capsys.readouterr().out.count("the driver needs a human:") == 1


@pytest.mark.parametrize("method", ["claim", "exhausted"])
def test_queue_scan_blocker_wins_over_exhaustion(method):
    class Driver(MemListDriver):
        scanned = False

        def pending_lines(self):
            self.scanned = True
            return []

        def needs_human(self):
            if self.scanned:
                return NeedsHuman("found while scanning")

    latched = []

    class ObservedShared(parallel.Shared):
        def __setattr__(self, name, value):
            if name == "stop_reason":
                latched.append(value)
            super().__setattr__(name, value)

    shared = ObservedShared(Driver([]), type("S", (), {"max_runs": None})())
    getattr(shared, method)()
    # A late poll must not conceal NO_WORK having already closed claims.
    shared.request_driver_handback(shared.driver.needs_human())
    assert RunStopReason.NO_WORK not in latched
    assert shared.stop_reason is RunStopReason.NEEDS_HUMAN
    assert shared.handback.human.reason == "found while scanning"


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
@pytest.mark.parametrize("source", ["hook", "poll"])
def test_human_ending_is_announced_once(tmp_path, monkeypatch, capsys, runner, source):
    class Driver(MemListDriver):
        blocked = False

        def item_finished(self, command, returncode):
            self.blocked = True
            if source == "hook":
                return NeedsHuman("inspect remaining work")

        def needs_human(self):
            if source == "poll" and self.blocked:
                return NeedsHuman("inspect remaining work")

    driver = Driver(["a", "b"])
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **kw: 0)
    monkeypatch.setattr(parallel, "run_job", _ok_job)
    with pytest.raises(SystemExit):
        if runner == "parallel":
            parallel.run_parallel(driver, par_args(tmp_path, jobs=1, no_statusline=True),
                                  wait_on_start=False)
        else:
            cyclecore.run_loop(driver, _seq(tmp_path), wait_on_start=False)
    output = capsys.readouterr().out
    assert len([line for line in output.splitlines() if "✗" in line]) == 1
    assert output.count("still pending") == 1
    assert "\n\n1 item(s) still pending" in output
    assert "Processed " not in output


def test_empty_fleet_returns_before_opening_run_resources(tmp_path, monkeypatch, capsys):
    from llm_loop import runlifecycle

    def forbidden(*args, **kwargs):
        pytest.fail("an empty fleet opened real-run resources")

    monkeypatch.setattr(stopchannel, "wait_for_stop_file_clear", forbidden)
    monkeypatch.setattr(cyclecore, "wait_before_start", forbidden)
    monkeypatch.setattr(runlifecycle, "open_status", forbidden)
    monkeypatch.setattr(runlifecycle, "open_usage", forbidden)
    monkeypatch.setattr(runlifecycle, "close_run", forbidden)
    result = parallel.run_parallel(MemListDriver([]), par_args(
        tmp_path, jobs=2, start_in="1h", ignore_usage=False, no_statusline=True))
    assert result.reason is RunStopReason.NO_WORK
    assert result.attempted == 0
    assert "Nothing pending" in capsys.readouterr().out


@pytest.mark.parametrize("runner", ["sequential", "parallel", "parallel-interrupt"])
@pytest.mark.parametrize("code", [0, 3])
def test_final_poll_latches_human_after_the_region_closes(
        tmp_path, monkeypatch, capsys, reasons, runner, code):
    from llm_loop import ctrlc, runlifecycle

    class Driver(MemListDriver):
        report_ready = False

        def needs_human(self):
            if self.report_ready:
                if runner == "parallel-interrupt":
                    ctrlc.current().press()
                return NeedsHuman("late blocker", exit_code=code)

    driver = Driver(["a"])
    release = runlifecycle.RunBoundary.release_held

    def release_held(boundary):
        release(boundary)
        driver.report_ready = True

    monkeypatch.setattr(runlifecycle.RunBoundary, "release_held", release_held)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **kw: 0)
    monkeypatch.setattr(parallel, "run_job", _ok_job)

    def run():
        if runner.startswith("parallel"):
            return parallel.run_parallel(
                driver, par_args(tmp_path, jobs=1, no_statusline=True),
                wait_on_start=False)
        return cyclecore.run_loop(driver, _seq(tmp_path), wait_on_start=False)

    if code or runner == "parallel-interrupt":
        with pytest.raises(SystemExit) as stopped, ctrlc.captured():
            run()
        if runner == "parallel-interrupt":
            assert stopped.value.code == 130
            assert ctrlc.INTERRUPTED_REASON in reasons
        else:
            assert stopped.value.code == code
            assert f"the driver needs a human (exit {code}): late blocker" in reasons
    else:
        with ctrlc.captured():
            assert run().reason is RunStopReason.NEEDS_HUMAN
        assert "the driver needs a human" in reasons
    assert capsys.readouterr().out.count("the driver needs a human:") == (
        0 if runner == "parallel-interrupt" else 1)


def test_handback_between_worker_check_and_admission_returns_claim(tmp_path, monkeypatch):
    at_admission = threading.Event()
    latched = threading.Event()
    calls = []
    shared_seen = []
    original_start = parallel.Shared.start_turn
    original_handback = parallel.Shared.request_driver_handback

    def start(self, line):
        if line == "a":
            shared_seen.append(self)
            at_admission.set()
            # Twice the existing runner helper's 20 s watchdog, no sleeps.
            assert latched.wait(40), "the second worker never latched handback"
        return original_start(self, line)

    def handback(self, answer):
        result = original_handback(self, answer)
        if isinstance(answer, NeedsHuman):
            latched.set()
        return result

    class Driver(MemListDriver):
        def item_started(self, command):
            if command.item == "b":
                assert at_admission.wait(40), "the first worker never reached admission"
                return NeedsHuman("admission closed")

    monkeypatch.setattr(parallel.Shared, "start_turn", start)
    monkeypatch.setattr(parallel.Shared, "request_driver_handback", handback)
    monkeypatch.setattr(parallel, "run_job", lambda job_id, command, mailbox=None:
                        (calls.append(command.item) or 0, 0, 0))
    driver = Driver(["a", "b"])
    outcome = _run_parallel(driver, par_args(tmp_path, jobs=2, no_statusline=True),
                            timeout=40)
    assert isinstance(outcome, SystemExit) and outcome.code == 1
    assert calls == ["b"]
    shared = shared_seen[0]
    assert shared.started == shared.claimed == 1
    assert not shared.in_progress and not shared.running
    assert driver.pending_lines() == ["a"]


def test_polled_human_after_failure_bypasses_quota_wait(tmp_path, monkeypatch):
    from llm_loop import runlifecycle
    from _runfixtures import StubSource

    class Driver(MemListDriver):
        blocked = False

        def item_finished(self, command, returncode):
            self.blocked = True

        def needs_human(self):
            if self.blocked:
                return NeedsHuman("report after failure", exit_code=3)

    driver = Driver(["a"])

    class Policy(StubPolicy):
        def check_and_wait(self, source, session_start, **kwargs):
            assert not driver.blocked, "a human handback entered the quota wait"
            return False, session_start

    driver.limit_policy = Policy()
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: StubSource())
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **kw: 1)
    with pytest.raises(SystemExit) as stopped:
        cyclecore.run_loop(driver, seq_args(tmp_path, no_statusline=True),
                           wait_on_start=False)
    assert stopped.value.code == 3


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    with isolated_run(monkeypatch, tmp_path):
        yield


def _ok_job(job_id, command, mailbox=None):
    return 0, 0.0, 0.01


def _run_parallel(driver, args, timeout=20.0):
    """run_parallel on a thread: its RunResult, or the SystemExit it left by."""
    box = []
    done = threading.Event()

    def go():
        try:
            box.append(parallel.run_parallel(driver, args,
                                             app_name="pytest-ownership"))
        except BaseException as exc:        # SystemExit is an answer here
            box.append(exc)
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    assert done.wait(timeout), "run_parallel did not return"
    assert box, "run_parallel returned nothing and raised nothing"
    return box[0]


def _seq(tmp_path):
    # A finite cap turns the usage machinery off (see run_loop).
    return seq_args(tmp_path, max=10, no_statusline=True)


class _KeyedDriver(MemListDriver):
    """Lines `<key>#<n>`: the part before `#` is the item."""

    def item_key(self, line):
        return line.split("#", 1)[0]


# --- 1. the item key ----------------------------------------------------------

def test_two_lines_with_one_key_never_run_at_once_in_the_fleet(
        tmp_path, monkeypatch):
    """Two workers, two spellings of one item: the second waits for the first.

    Without the key the fleet holds lines, so both are claimed at once and
    two agents write the same item side by side.
    """
    inside = []
    overlap = []
    lock = threading.Lock()

    def job(job_id, command, mailbox=None):
        with lock:
            inside.append(command.label)
            if len(inside) > 1:
                overlap.append(tuple(inside))
        # Long enough for the idle worker to try its claim meanwhile.
        time.sleep(0.3)
        with lock:
            inside.remove(command.label)
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", job)
    driver = _KeyedDriver(["prod/a#1", "prod/a#2"])

    result = _run_parallel(driver, par_args(tmp_path, jobs=2,
                                            no_statusline=True))

    assert result.reason is RunStopReason.NO_WORK
    assert not overlap, f"one item ran twice at once: {overlap}"
    assert driver.pending_lines() == []


def test_a_held_key_blocks_its_twin_until_the_item_settles():
    """The claim itself, under `Shared.lock`: no twin while a key is held."""
    driver = _KeyedDriver(["k#1", "k#2", "other"])
    shared = parallel.Shared(driver, type("S", (), {"max_runs": None})())

    first = shared.claim()
    second = shared.claim()
    assert {first, second} == {"k#1", "other"} or {first, second} == {
        "k#2", "other"}
    assert shared.claim() is None, "the twin of a held key was claimed"
    shared.finish(first if first.startswith("k") else second, True)
    assert shared.claim() is not None


def test_failed_attempts_are_counted_per_key(tmp_path, monkeypatch):
    """Two lines of one failing item share ONE budget, not one each."""
    calls = []
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None:
                        (calls.append(command.label), (1, 0.0, 0.01))[1])
    driver = _KeyedDriver(["prod/a#1", "prod/a#2"])

    result = _run_parallel(driver, par_args(tmp_path, jobs=1,
                                            no_statusline=True))

    assert result.reason is RunStopReason.NO_WORK
    assert len(calls) == MAX_ATTEMPTS, calls


class _DriftingKeyDriver(MemListDriver):
    """An `item_key` whose answer changes after the claim (`drift`)."""

    def __init__(self, items):
        super().__init__(items)
        self.salt = 0

    def item_key(self, line):
        return (line, self.salt)


@pytest.mark.parametrize("ending", ["settled", "released", "failed"])
def test_the_key_taken_at_claim_is_the_one_given_back(ending):
    """Every step after the claim uses the claim's key: a recomputed one that
    differs would leave the claimed key in `in_progress` (and `running`) for
    ever, so the run never reads as drained."""
    driver = _DriftingKeyDriver(["products/a.md"])
    shared = parallel.Shared(driver, type("S", (), {"max_runs": None})())

    line = shared.claim()
    driver.salt += 1                       # the key drifts mid-item
    shared.start_turn(line)
    if ending == "settled":
        shared.finish(line, True, settling=True)
        shared.settle(line, None)
    elif ending == "released":
        shared.release(line)
    else:
        shared.finish(line, False)

    assert shared.in_progress == set(), "the claimed key was never given back"
    assert shared.running == set()
    if ending == "failed":
        assert driver.ledger.attempts(("products/a.md", 0)) == 1, \
            "the failure was counted against a key nobody claimed"
        assert driver.ledger.attempts(driver.item_key(line)) == 0


@pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_status_row_that_throws_on_finish_does_not_hang_the_run(
        tmp_path, monkeypatch):
    """`job.finish` sits between the guard and `shared.finish`: a raise there
    must still record the outcome and give the key up."""
    from llm_loop import statusline
    monkeypatch.setattr(parallel, "run_job", _ok_job)
    real_finish = statusline.JobHandle.finish
    thrown = []

    def finish(self):
        if not thrown:
            thrown.append(1)
            raise RuntimeError("the status line broke")
        real_finish(self)

    monkeypatch.setattr(statusline.JobHandle, "finish", finish)
    driver = MemListDriver(["products/only.md"])

    result = _run_parallel(driver, par_args(tmp_path, jobs=2))

    assert thrown, "the throwing finish was never reached"
    assert driver.pending_lines() == [], "the finished turn was not recorded"
    assert result.completed == 1


# --- 2. the line a command served ---------------------------------------------

class _RecordingDriver(MemListDriver):
    """Records what `item_finished` was told; builds commands that never set
    `item` themselves, as an override of `command_for` would not."""

    def __init__(self, items):
        super().__init__(items)
        self.items_seen = []

    def item_finished(self, command, returncode):
        with self._lock:
            self.items_seen.append(command.item)
        return None


def test_the_fleet_tells_item_finished_which_line_finished(
        tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "run_job", _ok_job)
    lines = ["products/x.md", "products/y.md"]
    driver = _RecordingDriver(lines)

    _run_parallel(driver, par_args(tmp_path, jobs=2, no_statusline=True))

    assert sorted(driver.items_seen) == lines


def test_the_sequential_loop_tells_item_finished_which_line_finished(
        tmp_path, monkeypatch):
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)
    lines = ["products/x.md", "products/y.md"]
    driver = _RecordingDriver(lines)

    cyclecore.run_loop(driver, _seq(tmp_path), app_name="pytest-ownership",
                       wait_on_start=False)

    assert driver.items_seen == lines


# --- 3. a turn the driver refuses ---------------------------------------------

class _RefusingDriver(MemListDriver):
    """Refuses every turn of `refused`, as a driver whose output check fails."""

    def __init__(self, items, refused):
        super().__init__(items)
        self.refused = refused

    def settle(self, line):
        if line == self.refused:
            return Settled.REFUSED
        return super().settle(line)


def test_a_refused_turn_is_a_failed_attempt_in_the_fleet(tmp_path, monkeypatch):
    """Exit 0 and refused: retried, parked at MAX_ATTEMPTS, never "done"."""
    calls = []
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None:
                        (calls.append(command.label), (0, 0.0, 0.01))[1])
    driver = _RefusingDriver(["products/bad.md"], "products/bad.md")

    result = _run_parallel(driver, par_args(tmp_path, jobs=1,
                                            no_statusline=True))

    assert len(calls) == MAX_ATTEMPTS, calls
    assert result.completed == 0, "a refused turn was counted as done"
    assert driver.pending_lines() == ["products/bad.md"]


def test_a_line_already_gone_is_done_not_retried(tmp_path, monkeypatch):
    """GONE — edited out by hand during its turn — is not a refusal."""
    calls = []
    driver = MemListDriver(["products/a.md"])

    def job(job_id, command, mailbox=None):
        calls.append(command.label)
        driver.strike("products/a.md")      # the hand edit
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", job)

    result = _run_parallel(driver, par_args(tmp_path, jobs=1,
                                            no_statusline=True))

    assert calls == ["a.md"]
    assert result.completed == 1


def test_a_refused_turn_is_parked_in_the_sequential_loop(tmp_path, monkeypatch):
    """The same budget in the other runner: the loop ends, it does not spin."""
    calls = []
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: calls.append(1) or 0)
    driver = _RefusingDriver(["products/bad.md", "products/ok.md"],
                             "products/bad.md")

    result = cyclecore.run_loop(driver, _seq(tmp_path),
                                app_name="pytest-ownership",
                                wait_on_start=False)

    assert result.reason is RunStopReason.NO_WORK
    assert len(calls) == MAX_ATTEMPTS + 1
    assert driver.pending_lines() == ["products/bad.md"]
    # The fleet's count (see its twin above): only ok.md is done.
    assert result.completed == 1, "a refused turn was counted as done"


def test_the_sequential_refusal_budget_is_per_runner_call(
        tmp_path, monkeypatch):
    """A wrapper that reuses its driver gets a fresh budget each call, as the
    fleet's fresh `Shared` gives it one per `run_parallel`."""
    calls = []
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: calls.append(1) or 0)
    driver = _RefusingDriver(["products/bad.md"], "products/bad.md")

    for _ in range(2):
        cyclecore.run_loop(driver, _seq(tmp_path),
                           app_name="pytest-ownership", wait_on_start=False)

    assert len(calls) == 2 * MAX_ATTEMPTS, \
        "the second call found the item parked by the first"


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_success_resets_the_budget_for_a_twin(tmp_path, monkeypatch, runner):
    """Each twin needs two refusals then success; neither inherits failures."""
    class Driver(_KeyedDriver):
        def __init__(self):
            super().__init__(["k#1", "k#2"])
            self.calls = {}

        def settle(self, line):
            self.calls[line] = self.calls.get(line, 0) + 1
            if self.calls[line] < MAX_ATTEMPTS:
                return Settled.REFUSED
            return super().settle(line)

    monkeypatch.setattr(parallel, "run_job", _ok_job)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **kw: 0)
    driver = Driver()
    if runner == "parallel":
        result = _run_parallel(driver, par_args(tmp_path, jobs=1, no_statusline=True))
    else:
        result = cyclecore.run_loop(driver, _seq(tmp_path), wait_on_start=False)
    assert result.completed == 2
    assert driver.calls == {"k#1": MAX_ATTEMPTS, "k#2": MAX_ATTEMPTS}
    assert driver.pending_lines() == []
    assert driver.ledger.parked_lines() == []


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_ledger_and_parked_summary_are_per_runner_call(
        tmp_path, monkeypatch, capsys, runner):
    monkeypatch.setattr(parallel, "run_job", _ok_job)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **kw: 0)
    driver = _RefusingDriver(["bad.md"], "bad.md")
    for _ in range(2):
        previous = driver.ledger
        if runner == "parallel":
            result = _run_parallel(driver, par_args(tmp_path, jobs=1, no_statusline=True))
        else:
            result = cyclecore.run_loop(driver, _seq(tmp_path), wait_on_start=False)
        assert driver.ledger is not previous
        assert result.attempted == MAX_ATTEMPTS
        assert result.completed == 0
        assert driver.ledger.parked_lines() == ["bad.md"]
        summary = f"1 file(s) parked after {MAX_ATTEMPTS} failed attempts:"
        output = capsys.readouterr().out
        assert output.count(summary) == 1
        assert output.count("1 item(s) still pending") == 1
        assert output.count("still pending") == 1
        assert output.index("1 item(s) still pending") < output.index(summary)
        if runner == "parallel":
            assert output.index("Processed 0 file(s)") < output.index("still pending")


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_parked_summary_survives_a_failed_closing_report(
        tmp_path, monkeypatch, capsys, runner):
    class Driver(_RefusingDriver):
        def final_summary(self):
            raise RuntimeError("closing report failed")

    driver = Driver(["bad.md"], "bad.md")
    monkeypatch.setattr(parallel, "run_job", _ok_job)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **kw: 0)
    with pytest.raises(RuntimeError, match="closing report failed"):
        if runner == "parallel":
            parallel.run_parallel(driver, par_args(tmp_path, jobs=1, no_statusline=True),
                                  wait_on_start=False)
        else:
            cyclecore.run_loop(driver, _seq(tmp_path), wait_on_start=False)
    output = capsys.readouterr().out
    assert output.count(f"1 file(s) parked after {MAX_ATTEMPTS} failed attempts:") == 1


@pytest.mark.parametrize("ending, code", [("errors", 7), ("human", 3),
                                         ("interrupt", 130)])
def test_parked_summary_survives_an_abnormal_sequential_ending(
        tmp_path, monkeypatch, capsys, ending, code):
    class Driver(_RefusingDriver):
        def item_finished(self, command, returncode):
            if command.item == "b.md" and ending == "human":
                return NeedsHuman("inspect B", exit_code=code)

    driver = Driver(["a.md", "b.md"], "a.md")
    calls = []

    def run(*args, **kwargs):
        line = driver._current_line
        calls.append(line)
        if line == "b.md":
            if ending == "interrupt":
                raise KeyboardInterrupt
            if ending == "errors":
                return code
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", run)
    with pytest.raises(SystemExit) as stopped:
        cyclecore.run_loop(driver, _seq(tmp_path), wait_on_start=False)
    assert stopped.value.code == code
    assert calls == ["a.md"] * MAX_ATTEMPTS + ["b.md"] * (5 if ending == "errors" else 1)
    assert driver.ledger.parked_lines() == ["a.md"]
    output = capsys.readouterr().out
    assert output.count(f"1 file(s) parked after {MAX_ATTEMPTS} failed attempts:") == 1
    assert "\n\n  ⚠ 1 file(s) parked" in output


# --- 4. "a human is needed" ---------------------------------------------------

@pytest.fixture
def reasons(monkeypatch):
    """Every reason the run records for its exit line."""
    seen = []
    real = exitlog.set_reason

    def record(reason, **fields):
        seen.append(reason)
        real(reason, **fields)

    monkeypatch.setattr(exitlog, "set_reason", record)
    return seen


def _last_item_answers(answer):
    """A one-item driver whose `item_finished` answers `answer` only once a
    second worker has read the list AFTER the item's turn ended — the window
    in which that worker used to find the queue drained and latch NO_WORK,
    which then refused the hook's hand-back."""
    hook_running = threading.Event()
    polled = threading.Event()

    class Driver(MemListDriver):
        def pending_lines(self):
            if hook_running.is_set():
                polled.set()
            return super().pending_lines()

        def item_finished(self, command, returncode):
            hook_running.set()
            # The idle worker backs off 2 s between claims.
            assert polled.wait(10), "the second worker never polled"
            time.sleep(0.05)
            return answer

    return Driver(["products/only.md"])


def test_a_pause_on_the_last_item_is_not_read_as_a_drained_queue(
        tmp_path, monkeypatch):
    """The hook's answer is part of the item's completion, not a race."""
    monkeypatch.setattr(parallel, "run_job", _ok_job)

    result = _run_parallel(_last_item_answers("a request was filed"),
                           par_args(tmp_path, jobs=2, no_statusline=True))

    assert result.reason is RunStopReason.DRIVER_PAUSE


def test_the_fleet_ends_on_needs_human_not_on_a_drained_queue(
        tmp_path, monkeypatch, reasons):
    """The same window, a human asked for: its exit code and its reason."""
    monkeypatch.setattr(parallel, "run_job", _ok_job)

    outcome = _run_parallel(
        _last_item_answers(NeedsHuman("the search tool is down", exit_code=3)),
        par_args(tmp_path, jobs=2, no_statusline=True))

    assert isinstance(outcome, SystemExit), f"the run returned {outcome!r}"
    assert outcome.code == 3
    assert any(r.startswith("the driver needs a human (exit 3)")
               for r in reasons), reasons


def test_the_sequential_loop_exits_with_the_human_code(
        tmp_path, monkeypatch, capsys, reasons):
    """The run stops at its next boundary, prints its summary, exits 3."""
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)

    class Driver(MemListDriver):
        def item_finished(self, command, returncode):
            return NeedsHuman("the search tool is down", exit_code=3)

    driver = Driver(["products/a.md", "products/b.md"])
    with pytest.raises(SystemExit) as exit_info:
        cyclecore.run_loop(driver, _seq(tmp_path), app_name="pytest-ownership",
                           wait_on_start=False)

    assert exit_info.value.code == 3
    assert driver.pending_lines() == ["products/b.md"]
    assert any(r.startswith("the driver needs a human (exit 3)")
               for r in reasons), reasons
    assert "1 item(s) still pending" in capsys.readouterr().out


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_needs_human_with_exit_0_is_returned(tmp_path, monkeypatch, runner):
    monkeypatch.setattr(parallel, "run_job", _ok_job)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)

    class Driver(MemListDriver):
        def item_finished(self, command, returncode):
            return NeedsHuman("look at this", exit_code=0)

    driver = Driver(["products/a.md", "products/b.md"])
    if runner == "parallel":
        result = _run_parallel(driver, par_args(tmp_path, jobs=1,
                                                no_statusline=True))
    else:
        result = cyclecore.run_loop(driver, _seq(tmp_path),
                                    app_name="pytest-ownership",
                                    wait_on_start=False)
    assert result.reason is RunStopReason.NEEDS_HUMAN


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_needs_human_from_item_started_ends_the_run(
        tmp_path, monkeypatch, runner, reasons):
    """The start hook's answer: this turn still runs (it is claimed), nothing
    new starts, and the run exits with the human's code."""
    monkeypatch.setattr(parallel, "run_job", _ok_job)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)

    class Driver(MemListDriver):
        def item_started(self, command):
            return NeedsHuman("the licence server is down", exit_code=3)

    driver = Driver(["products/a.md", "products/b.md"])
    if runner == "parallel":
        outcome = _run_parallel(driver, par_args(tmp_path, jobs=1,
                                                 no_statusline=True))
    else:
        try:
            outcome = cyclecore.run_loop(driver, _seq(tmp_path),
                                         app_name="pytest-ownership",
                                         wait_on_start=False)
        except SystemExit as exc:
            outcome = exc

    assert isinstance(outcome, SystemExit), f"the run returned {outcome!r}"
    assert outcome.code == 3
    assert driver.pending_lines() == ["products/b.md"]
    assert any(r.startswith("the driver needs a human (exit 3)")
               for r in reasons), reasons


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_needs_human_outranks_the_cap_it_lands_on(tmp_path, monkeypatch,
                                                  runner):
    """The last item the cap allows asks for a human: the run reads
    NEEDS_HUMAN, not LIMIT_REACHED. In the fleet the cap is latched FIRST: the
    second worker reads it while the first one's turn runs. (A stop latched
    around it goes through the same `_handback_locked` / end-of-loop door;
    staging one per runner needs a stop file timed between two hooks, so it
    stays with the unit pins.)"""
    capped = threading.Event()

    def job(job_id, command, mailbox=None):
        # The idle worker backs off 2 s between claims; it reads the cap on
        # its first one, which runs beside this turn.
        capped.wait(5)
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", job)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)

    class Driver(MemListDriver):
        def item_finished(self, command, returncode):
            return NeedsHuman("look at this", exit_code=0)

    driver = Driver(["products/a.md", "products/b.md"])
    if runner == "parallel":
        real_exhausted = parallel.Shared._exhausted

        def exhausted(self, pending):
            answer = real_exhausted(self, pending)
            if self.stop_reason is RunStopReason.LIMIT_REACHED:
                capped.set()
            return answer

        monkeypatch.setattr(parallel.Shared, "_exhausted", exhausted)
        result = _run_parallel(driver, par_args(tmp_path, jobs=2, max=1,
                                                no_statusline=True))
        assert capped.is_set(), "the cap was never latched before the human"
    else:
        result = cyclecore.run_loop(
            driver, seq_args(tmp_path, max=1, no_statusline=True),
            app_name="pytest-ownership", wait_on_start=False)

    assert result.reason is RunStopReason.NEEDS_HUMAN
    assert driver.pending_lines() == ["products/b.md"]


def test_a_human_outranks_a_pause_latched_before_it():
    handback = stopchannel.DriverHandback()
    assert handback.latch("pause first") is True
    assert handback.latch(NeedsHuman("stuck")) is True
    assert handback.latch(NeedsHuman("stuck again")) is False
    assert handback.human == NeedsHuman("stuck")
    assert handback.reason == "pause first"


def test_the_fleet_relabels_a_cap_for_a_human():
    """A human is not "too late" for an ending already latched."""
    driver = MemListDriver(["products/only.md"])
    shared = parallel.Shared(driver, type("S", (), {"max_runs": None})())
    shared.stop_reason = RunStopReason.LIMIT_REACHED
    shared.claims_closed.set()

    assert shared.request_driver_handback(NeedsHuman("stuck")) is True
    assert shared.stop_reason is RunStopReason.NEEDS_HUMAN


# --- 5. the one writer of the list --------------------------------------------

class _FileDriver(ListFileDriver):
    def __init__(self, path):
        super().__init__()
        self.list_file = str(path)

    def prompt(self, source, target):
        return "go"


def test_replace_line_edits_removes_and_reports_a_missing_line(tmp_path):
    path = tmp_path / "list.md"
    path.write_text("﻿a.md\nb.md\nc.md\n", encoding="utf-8")
    driver = _FileDriver(path)

    assert driver.replace_line("b.md", "# done b.md") is True
    assert driver.strike("a.md") is True
    assert driver.replace_line("zzz.md", "x") is False
    assert driver.strike("a.md") is False
    assert path.read_text(encoding="utf-8") == "# done b.md\nc.md\n"
    assert driver.settle("c.md") is Settled.STRUCK
    assert driver.settle("c.md") is Settled.GONE
    assert path.read_text(encoding="utf-8") == "# done b.md\n"


def test_a_fleet_on_a_real_list_file_with_hooks_rewriting_it(
        tmp_path, monkeypatch):
    """The real `pending_lines` / `replace_line` / `_list_lock` under a fleet:
    workers strike their lines (under `Shared.lock`) while every
    `item_finished` rewrites a comment line from its worker thread, outside
    that lock. A read beside a truncating rewrite would lose lines — a work
    line run twice or never, or a comment left unrewritten."""
    count = 24
    work = [f"products/p{i:02}.md" for i in range(count)]
    path = tmp_path / "list.md"
    path.write_text("".join(f"{w}\n# {w}: pending\n" for w in work),
                    encoding="utf-8")
    ran = []
    lock = threading.Lock()

    def job(job_id, command, mailbox=None):
        with lock:
            ran.append(command.item)
        time.sleep(0.01)
        return 0, 0.0, 0.01

    class Driver(_FileDriver):
        pick_order = "list"

        def item_finished(self, command, returncode):
            assert self.replace_line(f"# {command.item}: pending",
                                     f"# {command.item}: done"), command.item
            return None

    monkeypatch.setattr(parallel, "run_job", job)
    driver = Driver(path)
    driver.limit_policy = StubPolicy()

    result = _run_parallel(driver, par_args(tmp_path, jobs=6,
                                            no_statusline=True), timeout=60.0)

    assert sorted(ran) == work, "a line ran twice or never"
    assert result.completed == count
    assert path.read_text(encoding="utf-8") == "".join(
        f"# {w}: done\n" for w in work)
