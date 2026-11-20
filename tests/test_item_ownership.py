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
from llm_loop.drivers import MAX_ATTEMPTS, ListFileDriver, Settled
from llm_loop.stopchannel import RunStopReason

from _runfixtures import MemListDriver, isolated_run, par_args, seq_args


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
