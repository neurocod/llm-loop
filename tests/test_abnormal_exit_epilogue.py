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

import threading

import pytest

from llm_loop import cyclecore, exitlog, operator, parallel, runlifecycle
from llm_loop.agentwork import ClaudeCommand, Driver, LoopStop
from llm_loop.drivers import StateFileDriver

from _runfixtures import (MemListDriver, OneShotDriver, StubPolicy, StubSource,
                          isolated_run, par_args, record_exit_pushes, seq_args)

# What the operator typed and never got delivered. One string, asserted by
# identity, so a run that printed SOME note would not satisfy a pin about THIS
# one.
NOTE = "please look at the third file"


class _AlwaysWorkDriver(Driver):
    """Hands out the same command forever — the loop has to decide when to stop."""

    def __init__(self):
        self.limit_policy = StubPolicy()

    def next_command(self):
        return ClaudeCommand("do the thing", "", "the-thing")


class _StoppingDriver(Driver):
    """Raises LoopStop with an exit code, the way a bad state file does."""

    def __init__(self, commands=0):
        self.limit_policy = StubPolicy()
        self.commands = commands

    def next_command(self):
        if self.commands:
            self.commands -= 1
            return ClaudeCommand("do the thing")
        raise LoopStop("state file says: error\nsecond line", exit_code=3)


def _seq_args(project_dir):
    # `git_push` stays the fixtures' "none", and that is not laziness.
    # `exit_pushes` replaces the EXIT push only; the sequential loop's per-pass
    # `maybe_git_push` stays real, so under `after_new_commits` these pins ran
    # `git push` as a subprocess six times against a pytest tmp_path (measured).
    # The policy is not what is being pinned — `exit_pushes` records the call and
    # its project whatever the policy says, and which repository a real push goes
    # to is `test_git_push`'s question.
    return seq_args(project_dir, no_statusline=True)


def _par_args(project_dir):
    # One worker keeps the closing report focused on one staged mailbox. Parallel
    # runs expose a MailboxSet at every width so `+` can add addresses in place.
    # And a usage source, so the closing snapshot has something to be taken from:
    # `--ignore-usage` leaves `usage` unopened and `close_run` correctly skips
    # the snapshot, which would leave another third of the pin measuring nothing.
    return par_args(project_dir, jobs=1, ignore_usage=False, no_statusline=True)


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    with isolated_run(monkeypatch, tmp_path):
        yield


@pytest.fixture
def exit_pushes(monkeypatch):
    return record_exit_pushes(monkeypatch)


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
    """The three steps of the epilogue, plus the ending's own record.

    Returns what the run printed up to the record's closing line.
    """
    out = capsys.readouterr().out
    assert [where for _policy, where in pushes] == [project_dir], (
        f"the exit push did not run once against the run's own project: {pushes}")
    if snapshot is None:
        assert policy.snapshots == [], "no provider was selected before the stop"
    else:
        assert policy.snapshots[-1] == snapshot, (
            f"the closing usage snapshot is missing or mislabelled: "
            f"{policy.snapshots}")
    assert "undelivered operator note" in out, (
        "the run exited holding a note and never said so")
    assert NOTE in out
    exitlog.finish()
    assert reason in capsys.readouterr().out
    return out


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
        snapshot="at end (claude: provider errors in a row)",
        reason="5 provider errors in a row (last exit code 7)")


@pytest.mark.parametrize("commands", [0, 1])
def test_a_driver_that_stops_the_run_still_closes_it_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, commands):
    """`LoopStop(exit_code=…)` is a run ending badly, not a run skipping the end."""
    driver = _StoppingDriver(commands)

    def succeeds(*args, **kwargs):
        loaded_mailbox.submit(NOTE)
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", succeeds)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)

    with pytest.raises(SystemExit) as exit_info:
        cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)

    assert exit_info.value.code == 3
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (claude: driver stopped the run)" if commands else None,
        # The FIRST line of the driver's message, so a multi-line diagnosis does
        # not turn the one-line ending into a paragraph.
        reason="the driver stopped the run (exit 3): state file says: error")


def test_invalid_state_model_still_closes_the_run_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox):
    class InvalidSelection(StateFileDriver):
        def first_line(self):
            return "Current state: implementation"

        def model(self):
            return "claud/opus"

        def prompt(self):
            return "work"

    driver = InvalidSelection()
    driver.limit_policy = StubPolicy()
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda p: pytest.fail("opened account for invalid selector"))
    with pytest.raises(SystemExit) as stopped:
        cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)
    assert stopped.value.code == 1
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path), snapshot=None,
        reason="invalid model selection")


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
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: StubSource())
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    driver = MemListDriver(["products/only.md"])

    with pytest.raises(SystemExit) as exit_info:
        parallel.run_parallel(driver, _par_args(str(tmp_path)),
                              app_name="pytest-abnormal", wait_on_start=False)

    assert exit_info.value.code == 130
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (parallel claude: interrupted)",
        reason="interrupted by the operator (Ctrl+C)")


@pytest.mark.parametrize("where", ["start", "stop", "join and stop"])
def test_ctrl_c_in_the_status_region_s_own_waits_still_closes_the_run_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, where):
    """The status region's start and teardown wait too, and Ctrl+C lands there.

    `StatusApp.stop` waits for the services, the key reader and the painter's
    last frame; an interrupt in it unwound `run_parallel` past both of its
    epilogues — no exit push, no snapshot, no notes. `start` waits for the
    first frame, before any worker exists. Either is the interrupt's one
    ending: exit push once, snapshot, notes, 130 — and a Ctrl+C in the
    teardown after one in the join is still that one ending, heard once.
    """
    method = "start" if where == "start" else "stop"
    real = getattr(parallel.statusline.StatusApp, method)

    def interrupted(app):
        real(app)
        # Typed here, where no worker is left to splice it (see above).
        loaded_mailbox.submit(NOTE)
        raise KeyboardInterrupt

    def interrupt(threads):
        for t in threads:
            t.join()
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel.statusline.StatusApp, method, interrupted)
    if where == "join and stop":
        monkeypatch.setattr(parallel, "join_workers", interrupt)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: StubSource())
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    driver = MemListDriver(["products/only.md"])

    try:
        with pytest.raises(SystemExit) as exit_info:
            parallel.run_parallel(driver, _par_args(str(tmp_path)),
                                  app_name="pytest-abnormal",
                                  wait_on_start=False)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl+C unwound the run past its epilogue")

    assert exit_info.value.code == 130
    out = _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (parallel claude: interrupted)",
        reason="interrupted by the operator (Ctrl+C)")
    assert out.count(parallel.INTERRUPT_ANNOUNCEMENT.strip()) == 1, (
        f"the interrupt was not heard exactly once:\n{out}")


# Upper bound on every wait of the second-Ctrl+C pin below; only a broken
# staging comes near it.
HELD_S = 10.0


@pytest.mark.parametrize("cut_closes", [1, 2],
                         ids=["second-ctrl-c", "second-and-third-ctrl-c"])
def test_a_ctrl_c_in_the_pool_s_close_still_stops_the_workers(
        tmp_path, monkeypatch, capsys, exit_pushes, cut_closes):
    """A hearing cut short in `threads.close` must not leave the workers claiming.

    `close` waits for the pool's lock, which a worker holds across its claim,
    and that wait is where a second Ctrl+C lands. The hearing used to mark the
    interrupt heard before it stopped the workers, so the boundary outside
    skipped it, and the run did its housekeeping and left while its workers
    went on claiming the queue. Staged: the first Ctrl+C lands in the join
    while a worker is inside its turn, the next `cut_closes` in the pool's
    close; the worker is let go only once the run has left. With a third
    Ctrl+C the boundary's own close is cut short too, and only a stop set
    BEFORE the close keeps the workers from claiming.
    """
    working = threading.Event()
    release = threading.Event()
    turns = []

    def held_first_turn(job_id, command, mailbox=None):
        turns.append(threading.current_thread())
        if len(turns) == 1:
            working.set()
            release.wait(timeout=HELD_S)
        return 0, None, None

    def ctrl_c_once_a_worker_works(threads):
        working.wait(timeout=HELD_S)
        raise KeyboardInterrupt

    pools = []
    real_close = parallel.WorkerPool.close

    def ctrl_c_in_the_first_closes(pool):
        pools.append(pool)
        if len(pools) <= cut_closes:
            raise KeyboardInterrupt     # a further Ctrl+C, in the lock wait
        real_close(pool)

    housekeeping = []
    real_close_run = runlifecycle.close_run

    def counted_close_run(*args, **kwargs):
        housekeeping.append(kwargs.get("ending"))
        return real_close_run(*args, **kwargs)

    monkeypatch.setattr(parallel, "run_job", held_first_turn)
    monkeypatch.setattr(parallel, "join_workers", ctrl_c_once_a_worker_works)
    monkeypatch.setattr(parallel.WorkerPool, "close", ctrl_c_in_the_first_closes)
    monkeypatch.setattr(runlifecycle, "close_run", counted_close_run)
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())
    driver = MemListDriver([f"products/item{i}.md" for i in range(5)])

    try:
        with pytest.raises(SystemExit) as exit_info:
            parallel.run_parallel(driver, _par_args(str(tmp_path)),
                                  app_name="pytest-abnormal",
                                  wait_on_start=False)
    finally:
        release.set()
    assert turns, "no worker ever started its turn — nothing staged"
    turns[0].join(timeout=HELD_S)

    assert not turns[0].is_alive(), "the held worker never finished"
    assert len(turns) == 1, (
        f"the workers went on claiming after the run left: {len(turns)} turns")
    if cut_closes == 1:
        assert pools and pools[0].grow() is None, (
            "the pool still takes workers — the cut-short close was never "
            "completed")
    assert housekeeping == ["interrupted"], (
        f"the interrupt's housekeeping did not run exactly once: {housekeeping}")
    assert exit_info.value.code == 130
    assert exit_pushes, "the interrupted run left without its exit push"
    out = capsys.readouterr().out
    assert out.count(parallel.INTERRUPT_ANNOUNCEMENT.strip()) == 1, (
        f"the interrupt was not announced exactly once:\n{out}")
    exitlog.finish()


def test_ctrl_c_during_a_normal_ending_s_exit_push_exits_130(
        tmp_path, monkeypatch, capsys, loaded_mailbox):
    """Ctrl+C in `end_run`'s exit push ends the run the way Ctrl+C does elsewhere.

    `close_run` gives up the push, keeps the rest, and raises the interrupt on;
    `end_run` used to let it out as a bare KeyboardInterrupt, the one door of
    either runner that did not record the interrupt and exit 130.
    """
    pushes = []

    def ctrl_c_in_the_push(policy, project_dir, abort=None):
        pushes.append((policy, project_dir))
        raise KeyboardInterrupt

    def succeeds(*args, **kwargs):
        loaded_mailbox.submit(NOTE)
        return 0

    monkeypatch.setattr(runlifecycle, "final_git_push", ctrl_c_in_the_push)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "run_claude_streaming", succeeds)
    # One command, then the work is over: the run RETURNS, through `end_run`.
    driver = OneShotDriver()

    try:
        with pytest.raises(SystemExit) as exit_info:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl+C left end_run as a bare KeyboardInterrupt")

    assert exit_info.value.code == 130
    _assert_closed_down(
        pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (claude)",
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
