"""The per-item hooks, and the ending they can ask for.

`Driver.item_started` / `Driver.item_finished` exist so a host wrapper can watch
the world AROUND a run — a folder the agents themselves write into, a lock, a
budget — at the two boundaries of one unit of work, and ask for control back
without killing anything. What the pins here are about is the difference between
that ending and the two a driver already had: nothing may be cancelled, nothing
may be lost from the queue, and the reason a runner reports has to say "ask me
again" rather than "the queue is empty" (which is what a wrapper reads to decide
whether to start another runner call at all).

The parallel half is the one with teeth: a fleet must wind DOWN — every turn in
flight runs to its end, and only new claims are refused — which is the same
close `--max-runs` uses and deliberately not the `stop` flag.
"""

import threading

import pytest

from llm_loop import cyclecore, parallel, stopchannel
from llm_loop.agentwork import ClaudeCommand, Driver
from llm_loop.drivers import ListFileDriver
from llm_loop.stopchannel import RunStopReason


class _StubPolicy:
    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="",
                       cache_value=True, should_stop=None):
        return False, session_start


class _HookedListDriver(ListFileDriver):
    """An in-memory queue whose hooks are supplied per test."""

    target_suffix = ".out.md"

    def __init__(self, items, *, on_started=None, on_finished=None):
        super().__init__()
        self.pick_order = "list"
        self._items = list(items)
        self._lock = threading.Lock()
        self._on_started = on_started
        self._on_finished = on_finished
        self.started = []
        self.finished = []
        self.limit_policy = _StubPolicy()

    def prompt(self, source, target):
        return "do it"

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

    def item_started(self, command):
        with self._lock:
            self.started.append(command.label)
        return self._on_started(command) if self._on_started else None

    def item_finished(self, command, returncode):
        with self._lock:
            self.finished.append((command.label, returncode))
        return (self._on_finished(command, returncode)
                if self._on_finished else None)


def _parallel_args(project_dir, jobs):
    ns = type("NS", (), {})()
    ns.jobs = jobs
    ns.max = None
    ns.dry_run = False
    ns.git_push = "none"
    ns.project_dir = str(project_dir)
    ns.ignore_usage = True
    ns.no_statusline = True
    return ns


def _seq_args(project_dir, max_runs=5):
    ns = type("NS", (), {})()
    # A finite cap turns the usage machinery off (see run_loop): these pins are
    # about the hooks, not about the quota gate.
    ns.max = max_runs
    ns.dry_run = False
    ns.raw = False
    ns.start_in = None
    ns.git_push = "none"
    ns.project_dir = str(project_dir)
    ns.cost = False
    ns.no_statusline = True
    return ns


def _run_parallel(driver, args, timeout=10.0):
    """run_parallel on a thread, so a run that never ends fails as a timeout."""
    box = []
    done = threading.Event()

    def go():
        try:
            box.append(parallel.run_parallel(driver, args,
                                             app_name="pytest-hooks"))
        except BaseException as exc:      # reported, not swallowed on a thread
            box.append(exc)
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    assert done.wait(timeout), "run_parallel did not return after a driver pause"
    assert box, "run_parallel returned nothing and raised nothing"
    assert not isinstance(box[0], BaseException), f"run_parallel raised: {box[0]!r}"
    return box[0]


def test_every_stop_reason_has_a_sentence_for_the_ending_line():
    """The `=== run ended: … ===` line is the one place a reader looks.

    A new reason that nobody spells out there falls back to its wire value, and
    the log then explains an ending with an enum literal.
    """
    missing = [reason for reason in RunStopReason
               if reason not in stopchannel.STOP_REASON_TEXT]
    assert not missing, f"no ending text for {missing}"


def test_a_finished_item_can_end_the_parallel_run_without_losing_the_queue(
    tmp_path, monkeypatch
):
    """The end-of-item hook stops the fleet claiming; the rest stays pending."""
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))
    driver = _HookedListDriver(
        [f"products/f{i}.md" for i in range(4)],
        on_finished=lambda command, rc: "a request was filed")

    result = _run_parallel(driver, _parallel_args(tmp_path, jobs=1))

    assert result.reason is RunStopReason.DRIVER_PAUSE
    # Exactly one item ran: the hook fired at its end, and nothing was claimed
    # after that.
    assert driver.finished == [("f0.md", 0)]
    assert driver.pending_lines() == [f"products/f{i}.md" for i in (1, 2, 3)]


def test_a_pause_asked_for_at_the_start_still_lets_that_item_finish(
    tmp_path, monkeypatch
):
    """`item_started` cannot cancel the turn it is announcing.

    The item is claimed and the provider is about to be launched, so a pause
    here means "this is the last one" — never a claim handed back unprocessed,
    which would cost the run a file it had already paid for.
    """
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, cmd, mailbox=None: (0, 0.0, 0.01))
    driver = _HookedListDriver(
        [f"products/f{i}.md" for i in range(3)],
        on_started=lambda command: "the kit needs promoting")

    result = _run_parallel(driver, _parallel_args(tmp_path, jobs=1))

    assert result.reason is RunStopReason.DRIVER_PAUSE
    assert driver.finished == [("f0.md", 0)], "the started item was cancelled"
    assert driver.pending_lines() == ["products/f1.md", "products/f2.md"]
    assert result.completed == 1


def test_a_pause_lets_every_turn_already_in_flight_run_to_its_end(
    tmp_path, monkeypatch
):
    """Three workers, one pause: none of the three turns may be cut short.

    The whole point of closing claims rather than setting `stop`: a fleet winds
    down, it is not killed. Staged so all three are inside `run_job` before the
    hook fires, which is the only arrangement where "was anything cancelled?" is
    a question with a visible answer.
    """
    all_started = threading.Event()
    release = threading.Event()
    inside = []
    inside_lock = threading.Lock()
    completed = []

    def blocked_job(job_id, command, mailbox=None):
        with inside_lock:
            inside.append(command.label)
            if len(inside) == 3:
                all_started.set()
        assert release.wait(5), "the test never released the in-flight turns"
        completed.append(command.label)
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", blocked_job)
    driver = _HookedListDriver(
        [f"products/f{i}.md" for i in range(6)],
        on_finished=lambda command, rc: "a request was filed")
    args = _parallel_args(tmp_path, jobs=3)

    box = []
    done = threading.Event()

    def go():
        try:
            box.append(parallel.run_parallel(driver, args,
                                             app_name="pytest-hooks"))
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    assert all_started.wait(5), "fewer than three turns reached the provider"
    release.set()
    assert done.wait(10), "the paused fleet did not return"

    assert box[0].reason is RunStopReason.DRIVER_PAUSE
    assert sorted(completed) == sorted(inside), "a turn in flight was cut short"
    assert len(driver.pending_lines()) == 3


def test_an_ending_already_latched_is_not_relabelled_by_a_late_pause():
    """`--max-runs` and a drained queue are final; a pause must not rename them.

    A wrapper reads the reason to decide whether to start another runner call,
    so a cap reported as DRIVER_PAUSE would restart a run the user had bounded.
    """
    driver = _HookedListDriver(["products/only.md"])
    shared = parallel.Shared(driver, type("S", (), {"max_runs": None})())
    shared.stop_reason = RunStopReason.LIMIT_REACHED
    shared.claims_closed.set()

    assert shared.request_driver_pause("too late") is False
    assert shared.stop_reason is RunStopReason.LIMIT_REACHED
    assert shared.pause_reason is None


class _SequentialHookDriver(Driver):
    """Hands out N identical commands; its hooks record and can ask to pause."""

    def __init__(self, items, *, pause_after=None):
        self._items = list(items)
        self._pause_after = pause_after
        self.started = []
        self.finished = []
        self.limit_policy = _StubPolicy()

    def next_command(self):
        if not self._items:
            return None
        return ClaudeCommand("do it", "", self._items.pop(0))

    def item_started(self, command):
        self.started.append(command.label)
        return None

    def item_finished(self, command, returncode):
        self.finished.append((command.label, returncode))
        if command.label == self._pause_after:
            return "a request was filed"
        return None


def test_the_sequential_loop_stops_at_its_boundary_when_a_hook_asks(
    tmp_path, monkeypatch
):
    """One runner call ends; the items it never reached are still the driver's.

    Acted on at the loop head rather than where it is latched, so the paths that
    a failed iteration takes (retry, rate-limit wait) come back to one check.
    """
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)
    driver = _SequentialHookDriver(["a", "b", "c"], pause_after="b")

    result = cyclecore.run_loop(driver, _seq_args(tmp_path),
                                app_name="pytest-hooks", wait_on_start=False)

    assert result.reason is RunStopReason.DRIVER_PAUSE
    assert driver.started == ["a", "b"]
    assert driver.finished == [("a", 0), ("b", 0)]
    assert driver._items == ["c"], "the unreached item was consumed anyway"


def test_hooks_that_ask_for_nothing_leave_the_run_exactly_as_it_was(
    tmp_path, monkeypatch
):
    """The default hooks return None, so an unaware driver cannot be paused."""
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)
    driver = _SequentialHookDriver(["a", "b"])

    result = cyclecore.run_loop(driver, _seq_args(tmp_path),
                                app_name="pytest-hooks", wait_on_start=False)

    assert result.reason is RunStopReason.NO_WORK
    assert driver.finished == [("a", 0), ("b", 0)]


@pytest.mark.parametrize("hook", ["item_started", "item_finished"])
def test_the_base_driver_answers_none_to_both_hooks(hook):
    """The contract's default: a driver that never heard of them is unaffected."""
    command = ClaudeCommand("do it", "", "label")
    call = getattr(Driver(), hook)
    assert (call(command) if hook == "item_started" else call(command, 0)) is None
