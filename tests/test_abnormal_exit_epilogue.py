"""The three endings that `sys.exit` still get the epilogue every ending gets.

A run has four ways to end and only one of them RETURNS. The other three —
the driver stopping the run with an exit code, five provider errors in a row, and
Ctrl+C in the parallel runner — used to write `exitlog.set_reason` and leave, so
the endings with the most to explain were the ones that left the least behind:
no exit push (an operator's commits sat local until some later run happened to
push them), no closing usage snapshot, and no report of the notes nobody
delivered — which are, on a run that died of provider errors, the likeliest
explanation of what went wrong.

Each pin asserts all three steps at once, because the failure they guard is "one
of them was dropped", not "the exit stopped working": a run that exits 130 with
nothing behind it is exactly what the code did before.

The exit push is watched by replacing `runlifecycle.final_git_push`, NOT by
counting `git push` calls through a fake git. Measured while writing this file:
the sequential loop pushes at the top of every pass, so a run that failed five
times had already pushed five times, and "a push happened" was true whether the
exit push ran or not — the pin passed with the whole epilogue deleted. What these
pins are about is the exit push specifically, so that is the call they watch.
Whether it really runs git, and in which repository, is `test_git_push`'s — and
these runs are launched `--git-push none` so that stays true of them, which it
was not at first (see `_seq_args`).
"""

import logging
import sys
import threading

import pytest

from llm_loop import (console, cyclecore, exitlog, operator, parallel,
                      projectroot, runlifecycle)
from llm_loop.agentwork import ClaudeCommand, Driver, LoopStop
from llm_loop.drivers import ListFileDriver

# What the operator typed and never got delivered. One string, asserted by
# identity, so a run that printed SOME note would not satisfy a pin about THIS
# one.
NOTE = "please look at the third file"


class _StubPolicy:
    """A LimitPolicy that never reads the usage report and never pauses.

    `log_snapshot` RECORDS instead of doing nothing: the closing snapshot is one
    of the three steps these pins are about, and a stub that swallowed it would
    leave that third unpinned while looking pinned.
    """

    def __init__(self):
        self.snapshots = []

    def describe(self):
        return "stub"

    def log_snapshot(self, source, label, cache_value=True):
        self.snapshots.append(label)

    def check_and_wait(self, source, session_start, note="",
                       cache_value=True, should_stop=None):
        return False, session_start


class _AlwaysWorkDriver(Driver):
    """Hands out the same command forever — the loop has to decide when to stop."""

    def __init__(self):
        self.limit_policy = _StubPolicy()

    def next_command(self):
        return ClaudeCommand("do the thing", "", "the-thing")


class _StoppingDriver(Driver):
    """Raises LoopStop with an exit code, the way a bad state file does."""

    def __init__(self):
        self.limit_policy = _StubPolicy()

    def next_command(self):
        raise LoopStop("state file says: error\nsecond line", exit_code=3)


class _OneItemListDriver(ListFileDriver):
    """A one-item in-memory queue: enough for the parallel runner to open a run."""

    target_suffix = ".out.md"

    def __init__(self):
        super().__init__()
        self._items = ["products/only.md"]
        self._lock = threading.Lock()
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


def _seq_args(project_dir):
    ns = type("NS", (), {})()
    ns.max = None
    ns.dry_run = False
    ns.raw = False
    ns.start_in = None
    # NONE, and that is not laziness. `exit_pushes` replaces the EXIT push only;
    # the sequential loop's per-pass `maybe_git_push` stays real, so under
    # `after_new_commits` these pins ran `git push` as a subprocess six times
    # against a pytest tmp_path (measured). Harmless there and a real push the
    # day a tmp dir sits inside a repo with an upstream. The policy is not what
    # is being pinned — `exit_pushes` records the call and its project whatever
    # the policy says, and which repository a real push goes to is
    # `test_git_push`'s question.
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.cost = False
    ns.no_statusline = True
    return ns


def _par_args(project_dir):
    ns = _seq_args(project_dir)
    # One worker keeps the closing report focused on one staged mailbox. Parallel
    # runs expose a MailboxSet at every width so `+` can add addresses in place.
    ns.jobs = 1
    # And a usage source, so the closing snapshot has something to be taken from:
    # `--ignore-usage` leaves `source` None and `close_run` correctly skips the
    # snapshot, which would leave another third of the pin measuring nothing.
    ns.ignore_usage = False
    return ns


class _StubSource:
    """Stands in for a UsageSource without an endpoint behind it.

    Only the two calls a closing run makes are answered; the status area is a
    Null object under `--no-statusline`, so nothing else reaches for it.
    """

    def get_usage(self):
        return None

    def invalidate(self):
        pass


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    """Own log dir, own exit record, and the process globals put back after."""
    monkeypatch.setattr(console, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(exitlog, "_record", None)
    root = projectroot.project_dir()
    streams = sys.stdout, sys.stderr
    yield
    exitlog.finish()
    sys.stdout, sys.stderr = streams
    projectroot.set_project_root(root)
    for name in ("pytest-abnormal",):
        logger = logging.getLogger(f"runCycle.{name}")
        for handler in list(logger.handlers):
            handler.close()
        logger.handlers = []


@pytest.fixture
def exit_pushes(monkeypatch):
    """Records every call to the EXIT push, and stops it reaching git.

    Replaced on `runlifecycle`, not on `gitpush`: the epilogue imported the name,
    so that is the binding its call resolves — patching the owner would leave the
    real push running and the recorder empty.
    """
    calls = []
    monkeypatch.setattr(runlifecycle, "final_git_push",
                        lambda policy, project_dir: calls.append(
                            (policy, project_dir)))
    return calls


@pytest.fixture
def loaded_mailbox(monkeypatch):
    """Every run in this file gets a mailbox holding one undelivered note.

    Put there by replacing the constructor, because the mailbox is the RUN's —
    made inside `run_loop` / `run_parallel` and never handed to the caller — so a
    test that wants to know what happens to a note nobody delivered has no other
    way to stage one.
    """
    box = operator.Mailbox()
    box.submit(NOTE)
    monkeypatch.setattr(operator, "Mailbox", lambda: box)
    return box


def _assert_closed_down(pushes, policy, capsys, project_dir, *, snapshot, reason):
    """The three steps of the epilogue, plus the ending's own record."""
    out = capsys.readouterr().out
    assert [where for _policy, where in pushes] == [project_dir], (
        f"the exit push did not run once against the run's own project: {pushes}")
    assert policy.snapshots[-1] == snapshot, (
        f"the closing usage snapshot is missing or mislabelled: "
        f"{policy.snapshots}")
    assert "undelivered operator note" in out, (
        "the run exited holding a note and never said so")
    assert NOTE in out
    exitlog.finish()
    assert reason in capsys.readouterr().out


def test_five_provider_errors_in_a_row_still_close_the_run_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox):
    """The ending most in need of a post-mortem used to leave the least behind."""
    calls = []

    def fails(*args, **kwargs):
        # A note typed while THIS turn runs. Notes 1..4 are spliced into the next
        # iteration's prompt, so only the fifth is still in the mailbox when the
        # brake trips — which is exactly the note an operator loses.
        calls.append(1)
        loaded_mailbox.submit(NOTE)
        return 7

    monkeypatch.setattr(cyclecore, "run_claude_streaming", fails)
    driver = _AlwaysWorkDriver()

    with pytest.raises(SystemExit) as exit_info:
        cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)

    assert exit_info.value.code == 7, "the provider's exit code must survive"
    assert len(calls) == 5, "the brake is five errors in a row"
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (provider errors in a row)",
        reason="5 provider errors in a row (last exit code 7)")


def test_a_driver_that_stops_the_run_still_closes_it_down(
        tmp_path, capsys, exit_pushes, loaded_mailbox):
    """`LoopStop(exit_code=…)` is a run ending badly, not a run skipping the end."""
    driver = _StoppingDriver()

    with pytest.raises(SystemExit) as exit_info:
        cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)

    assert exit_info.value.code == 3
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (driver stopped the run)",
        # The FIRST line of the driver's message, so a multi-line diagnosis does
        # not turn the one-line ending into a paragraph.
        reason="the driver stopped the run (exit 3): state file says: error")


def test_ctrl_c_in_the_parallel_runner_still_closes_the_run_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox):
    """Interrupting a fleet must not strand its commits or its mailbox.

    The interrupt is staged at `join_workers`, which is where a real Ctrl+C
    arrives: that call is where a parallel run spends all of its time.
    """
    def interrupt(threads):
        # The real join first, so the queue is drained and the workers are gone
        # before the note is typed: a note submitted while a worker is still
        # claiming would race that worker's splice, and the pin would flake on
        # which of the two got there first.
        for t in threads:
            t.join()
        loaded_mailbox.submit(NOTE)
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel, "join_workers", interrupt)
    monkeypatch.setattr(parallel, "usage_source_for", lambda provider: _StubSource())
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    driver = _OneItemListDriver()

    with pytest.raises(SystemExit) as exit_info:
        parallel.run_parallel(driver, _par_args(str(tmp_path)),
                              app_name="pytest-abnormal", wait_on_start=False)

    assert exit_info.value.code == 130
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (interrupted)",
        reason="interrupted by the operator (Ctrl+C)")


def test_the_two_doors_of_the_epilogue_run_the_same_housekeeping():
    """`end_run` must not grow a step `close_run` does not have.

    The three pins above watch the abnormal door; this watches that the normal
    one still goes through the same body, so the two cannot drift into doing
    different amounts of housekeeping. Checked on the SOURCE, because a runtime
    check would need a fourth staged run to say anything the pins above do not.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(runlifecycle.end_run))
    called = {node.func.id for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

    assert "close_run" in called, (
        "end_run stopped delegating to close_run — the normal ending and the "
        "three sys.exit endings are doing different housekeeping again")
