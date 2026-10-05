"""The endings that do not return still get the epilogue every ending gets.

Only one way to end a run RETURNS. The others — the driver stopping the run
with an exit code, five provider errors in a row, Ctrl+C wherever it lands, an
exit or an exception nobody wrote an ending for — used to write
`exitlog.set_reason` and leave, or not even that, so the endings with the most
to explain were the ones that left the least behind: no exit push (an
operator's commits sat local until some later run happened to push them), no
closing usage snapshot, and no report of the notes nobody delivered — which
are, on a run that died of provider errors, the likeliest explanation of what
went wrong. Each runner now holds one boundary (`runlifecycle.RunBoundary`)
from its first `open_usage` to `close_run`, and every pin below stages an
ending inside it.

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

import _thread
import contextlib
import threading
import time

import pytest

from llm_loop import (ctrlc, cyclecore, exitlog, gitpush, limits, operator,
                      ownership, parallel, projectroot, providers,
                      runlifecycle, statusline, stopchannel, streamrender,
                      usage)
from llm_loop.agentwork import ClaudeCommand, Driver, LoopStop
from llm_loop.drivers import StateFileDriver
from llm_loop.limits import LimitPolicy, SessionLimit
from llm_loop.usage import RateLimitEvent

from _runfixtures import (NOTE, HeldCountGit, MemListDriver, OneShotDriver,
                          StubPolicy, StubSource, isolated_run, par_args,
                          record_exit_pushes, seq_args, stage_undelivered_note)


def press_ctrl_c():
    """One Ctrl+C, as the run's SIGINT handler counts it (`ctrlc.captured`).

    Inside a run Ctrl+C raises nothing, so a pin stages it where it lands by
    pressing the run's Interrupt from that spot, and lets the staged call go
    on — the way the real handler returns into whatever the run was doing.
    """
    ctrlc.current().press()


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
    """Every run in this file gets a mailbox holding one undelivered note
    (`stage_undelivered_note`, typed before the run)."""
    return stage_undelivered_note(monkeypatch)


def _assert_closed_down(pushes, policy, capsys, project_dir, *, snapshot, reason,
                        exit_pushed=True):
    """The three steps of the epilogue, plus the ending's own record.

    `pushes` are the exit pushes, as `(policy, project_dir)` pairs. With
    `exit_pushed` False the ending gave its push up and `pushes` must be empty —
    any record of pushes will do then, a fake git's `pushes` included.
    `reason` None is a pin that drove `close_run` itself, outside any run: the
    record and its reason are the door's (`end_run`, `exit_run`), not
    `close_run`'s, so there is none to read.

    Returns what the run printed, up to the record's closing line, as
    `capsys.readouterr()` returned it (`.out`, `.err`).
    """
    captured = capsys.readouterr()
    if exit_pushed:
        assert [where for _policy, where in pushes] == [project_dir], (
            f"the exit push did not run once against the run's own project: "
            f"{pushes}")
    else:
        assert pushes == [], f"the given-up exit push still pushed: {pushes}"
    if snapshot is None:
        assert policy.snapshots == [], "no provider was selected before the stop"
    else:
        assert policy.snapshots[-1] == snapshot, (
            f"the closing usage snapshot is missing or mislabelled: "
            f"{policy.snapshots}")
    assert "undelivered operator note" in captured.out, (
        "the run exited holding a note and never said so")
    assert NOTE in captured.out
    if reason is not None:
        exitlog.finish()
        assert reason in capsys.readouterr().out
    return captured


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
    is heard: that call is where a parallel run spends all of its time.
    """
    def interrupt(threads):
        # The real join first, so the queue is drained and the workers are gone
        # before the note is typed: a note submitted while a worker is still
        # claiming would race that worker's splice, and the pin would flake on
        # which of the two got there first.
        for t in threads:
            t.join()
        loaded_mailbox.submit(NOTE)
        press_ctrl_c()

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
    """The status region's start and teardown wait too, and Ctrl+C comes there.

    `StatusApp.stop` waits for the services, the key reader and the painter's
    last frame; an interrupt in it unwound `run_parallel` past both of its
    epilogues — no exit push, no snapshot, no notes. `start` waits for the
    first frame, before any worker exists, and a press there starts no worker.
    Either is the interrupt's one ending: exit push once, snapshot, notes,
    130 — and a Ctrl+C in the teardown after one in the join is still that
    one ending, heard once; being a press past the one that began it, it
    gives up the exit push (`RunBoundary.interrupt`).
    """
    method = "start" if where == "start" else "stop"
    real = getattr(parallel.statusline.StatusApp, method)

    def interrupted(app):
        real(app)
        # Typed here, where no worker is left to splice it (see above): after
        # `stop` the fleet is gone, and a press in `start` starts none.
        loaded_mailbox.submit(NOTE)
        press_ctrl_c()

    def interrupt(threads):
        for t in threads:
            t.join()
        press_ctrl_c()

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
    if where == "join and stop":
        assert exit_pushes == [], "a second press did not give up the push"
        # Stands in for the push `_assert_closed_down` asks after.
        exit_pushes.append((None, str(tmp_path)))
    out = _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (parallel claude: interrupted)",
        reason="interrupted by the operator (Ctrl+C)").out
    assert out.count(parallel.INTERRUPT_ANNOUNCEMENT.strip()) == 1, (
        f"the interrupt was not heard exactly once:\n{out}")
    assert (runlifecycle.ABANDONED_PUSH in out) == (where == "join and stop")


# Upper bound on every wait of the second-Ctrl+C pin below and of the pin
# about Ctrl+C while the exit push is waited for; only a broken staging comes
# near it.
HELD_S = 10.0


@pytest.mark.parametrize("second", [False, True],
                         ids=["ctrl-c", "second-ctrl-c-gives-up-the-join"])
def test_a_ctrl_c_while_a_worker_works_stops_the_fleet(
        tmp_path, monkeypatch, capsys, exit_pushes, second):
    """Ctrl+C with a worker inside its turn: no further claim, one ending.

    A hearing that a second Ctrl+C cut short in the pool's close used to leave
    the workers claiming the queue while the run did its housekeeping and
    left. A press cuts nothing short now, but the scenario stays: the first
    Ctrl+C is heard in the join while a worker is inside its turn, and the run
    waits for that turn a bounded time (INTERRUPT_JOIN_TIMEOUT_S, shortened
    here) — or, with a second Ctrl+C during that wait, not at all, and then
    not the exit push either: the second press is past the one the ending
    began at. The worker is let go only once the run has left, and it must
    claim nothing after its turn.
    """
    monkeypatch.setattr(parallel, "INTERRUPT_JOIN_TIMEOUT_S",
                        HELD_S if second else 0.2)
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
        press_ctrl_c()

    # The second press lands where the fleet's join takes its budget: after
    # `_Interrupt.hear` has read its mark, before the join waits — the one
    # moment a press is "during the join", staged there rather than on a timer
    # that may fire before the join or after it.
    pressed_in_join = []
    real_budget = runlifecycle.StopBudget

    def budget_then_press(seconds):
        budget = real_budget(seconds)
        if second and seconds == HELD_S and not pressed_in_join:
            pressed_in_join.append(ctrlc.current().presses)
            press_ctrl_c()
        return budget

    monkeypatch.setattr(runlifecycle, "StopBudget", budget_then_press)

    pools = []
    real_close = parallel.WorkerPool.close

    def recorded_close(pool):
        pools.append(pool)
        real_close(pool)

    housekeeping = []
    real_close_run = runlifecycle.close_run

    def counted_close_run(*args, **kwargs):
        housekeeping.append(kwargs.get("ending"))
        return real_close_run(*args, **kwargs)

    monkeypatch.setattr(parallel, "run_job", held_first_turn)
    monkeypatch.setattr(parallel, "join_workers", ctrl_c_once_a_worker_works)
    monkeypatch.setattr(parallel.WorkerPool, "close", recorded_close)
    monkeypatch.setattr(runlifecycle, "close_run", counted_close_run)
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())
    driver = MemListDriver([f"products/item{i}.md" for i in range(5)])

    started = time.monotonic()
    try:
        with pytest.raises(SystemExit) as exit_info:
            parallel.run_parallel(driver, _par_args(str(tmp_path)),
                                  app_name="pytest-abnormal",
                                  wait_on_start=False)
    finally:
        elapsed = time.monotonic() - started
        release.set()
    assert turns, "no worker ever started its turn — nothing staged"
    turns[0].join(timeout=HELD_S)

    assert elapsed < HELD_S / 2, (
        f"the run sat out the held turn: {elapsed:.1f} s")
    assert not turns[0].is_alive(), "the held worker never finished"
    assert len(turns) == 1, (
        f"the workers went on claiming after the run left: {len(turns)} turns")
    assert pools and pools[0].grow() is None, "the pool still takes workers"
    assert housekeeping == ["interrupted"], (
        f"the interrupt's housekeeping did not run exactly once: {housekeeping}")
    assert exit_info.value.code == 130
    out = capsys.readouterr().out
    if second:
        assert pressed_in_join == [1], "the second press was never staged"
        assert exit_pushes == [], (
            "a second press during the join still waited for the exit push")
        assert runlifecycle.ABANDONED_PUSH in out
    else:
        assert exit_pushes, "the interrupted run left without its exit push"
    assert out.count(parallel.INTERRUPT_ANNOUNCEMENT.strip()) == 1, (
        f"the interrupt was not announced exactly once:\n{out}")
    exitlog.finish()


def test_ctrl_c_during_a_normal_ending_s_exit_push_exits_130(
        tmp_path, monkeypatch, capsys, loaded_mailbox):
    """Ctrl+C in `end_run`'s exit push ends the run the way Ctrl+C does elsewhere.

    `close_run` gives up the push and keeps the rest; `end_run` used to let
    the interrupt out as a bare KeyboardInterrupt, the one door of either
    runner that did not record the interrupt and exit 130. A RETURN would be
    worse: it tells a wrapper to start its next phase.
    """
    pushes = []

    def ctrl_c_in_the_push(policy, project_dir, abort=None):
        pushes.append((policy, project_dir))
        press_ctrl_c()

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


def test_an_exit_push_that_raises_does_not_cost_a_parallel_run_its_ending(
        tmp_path, monkeypatch, capsys):
    """On the pusher, a failed exit push is reported, and the run still returns.

    Made on the main thread it unwound `close_run` and `end_run` with it: no
    closing usage snapshot, no report of undelivered notes, no recorded
    reason — a push failing is not what ended the run, and it must not be what
    its record says. So the run here HAS all three to lose: a usage source (its
    closing snapshot), a note nobody delivered, and a reason to record.

    The failure is reported whole — with its traceback, on stderr, which the
    tee carries to the mirror log — and not through the owner's one-line
    report, which drops a failure worded like one it already reported.
    """
    pushes = []

    def exit_push_fails(policy, project_dir, abort=None):
        pushes.append((policy, project_dir))
        raise OSError("staged: the repository is not reachable")

    monkeypatch.setattr(runlifecycle, "final_git_push", exit_push_fails)
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    stage_undelivered_note(monkeypatch, after_workers=True)
    driver = MemListDriver(["products/only.md"])

    result = parallel.run_parallel(driver, _par_args(str(tmp_path)),
                                   app_name="pytest-abnormal",
                                   wait_on_start=False)

    assert result.reason == stopchannel.RunStopReason.NO_WORK
    err = _assert_closed_down(
        pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (parallel claude)",
        reason="=== run ended: no more work in the queue").err
    assert "the exit push failed" in err
    assert "Traceback" in err, "the failed exit push lost its traceback"
    assert "OSError: staged: the repository is not reachable" in err


# How long the pin below lets the main thread settle into its wait before it
# interrupts it, and afterwards lets a late interrupt land. Only the failing
# cases depend on it (an unbounded wait, a close_run that does not wait).
CTRL_C_SETTLE_S = 0.1


def test_ctrl_c_while_the_exit_push_is_waited_for_gives_up_the_push_only(
        tmp_path, monkeypatch, capsys, loaded_mailbox):
    """Ctrl+C during the exit push abandons the push, not the run's ending.

    The wait for the pusher is where a parallel run sits when its operator
    loses patience with a slow `git push`. Unguarded, the interrupt unwound
    `close_run` there — no closing snapshot, no report of the notes nobody
    delivered — while the exit push went on as a daemon, free to start a
    `git push` nobody wanted any more. Pinned: the snapshot and the notes
    still happen, the push starts no further git, and nothing is raised —
    inside a run Ctrl+C is the run's `ctrlc.Interrupt`, and what the press
    does to the ending is the door's (`end_run`, `exit_run`).

    The Ctrl+C is `_thread.interrupt_main` under `ctrlc.captured`, i.e. the
    run's real SIGINT handler. It — measured 2026-09-29 on 3.13 and 3.14
    alike — does not wake an unbounded lock wait, just as a real Ctrl+C does
    not up to 3.13. So a close_run that waited on the pusher in one unbounded
    wait hears it only after the push, on every version — and here the push
    is held until the interrupt has been handled.

    The real `final_git_push` runs, over a fake git whose count is held: what
    is pinned of the push is that it started no `git push`, so this pin, alone
    in this file, counts git calls rather than recording the exit push.
    """
    fake = HeldCountGit(hold_timeout_s=HELD_S)
    monkeypatch.setattr(gitpush, "subprocess", fake)
    projectroot.set_project_root(str(tmp_path))
    policy = StubPolicy()
    usage = runlifecycle.RunUsage(StubSource(), policy, "parallel claude")
    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=False, progress=None,
        settings=runlifecycle.RunSettings(
            git_push=gitpush.GitPushPolicy.AFTER_NEW_COMMITS),
        registry=None, status_enabled=False)
    pusher = ownership.OwnerThread("pusher").start()

    def ctrl_c_once_the_exit_push_runs():
        if fake.counting.wait(timeout=HELD_S):
            # Settled into its wait first: an interrupt landing before the
            # wait starts would pass an unbounded wait too.
            time.sleep(CTRL_C_SETTLE_S)
            _thread.interrupt_main()

    interrupter = threading.Thread(target=ctrl_c_once_the_exit_push_runs,
                                   daemon=True)
    raised = []
    try:
        # A press is in the block, so it leaves as the interrupt
        # (`ctrlc.captured`); what is pinned is that nothing inside it raised.
        with pytest.raises(SystemExit), ctrlc.captured() as interrupt:
            interrupter.start()
            try:
                runlifecycle.close_run(ctx, usages=[usage],
                                       ending="interrupted",
                                       mailbox=loaded_mailbox, pusher=pusher)
                returned_after = interrupt.presses
                # A close_run that did not wait returns before the Ctrl+C is
                # sent; it is then heard here, still inside the capture.
                interrupter.join(timeout=HELD_S)
                time.sleep(CTRL_C_SETTLE_S)
            except KeyboardInterrupt:
                raised.append("a bare KeyboardInterrupt")
    finally:
        fake.release.set()

    assert raised == [], "Ctrl+C inside a run was raised, not pressed"
    assert returned_after == 1, (
        f"close_run returned before the Ctrl+C was heard: {returned_after}")
    assert pusher.close(timeout=HELD_S), \
        "the abandoned exit push never finished"
    out = _assert_closed_down(
        fake.pushes, policy, capsys, str(tmp_path),
        snapshot="at end (parallel claude: interrupted)", reason=None,
        exit_pushed=False).out
    assert len(policy.snapshots) == 1, (
        f"close_run took more than its closing snapshot: {policy.snapshots}")
    assert "the exit push is abandoned" in out
    assert "final git push on exit" not in out, \
        "the abandoned exit push still announced a push it was not going to make"


@pytest.mark.parametrize("ending", ["exit", "exception"])
def test_what_the_boundary_holds_is_released_once_last_first(capsys, ending):
    """`RunBoundary.on_ending` is the run's destructor list.

    Released as the ending begins — whichever ending — in reverse order of
    registration, once however many doors the unwinding passes; a release
    that raises costs the ones under it nothing and is reported, never put in
    the ending's place.
    """
    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=True, progress=None,
        settings=runlifecycle.RunSettings(), registry=None,
        status_enabled=False)
    boundary = runlifecycle.RunBoundary(ctx, counts=lambda: (0, 0))
    released = []

    def broken():
        released.append("broken")
        raise RuntimeError("staged: a release broke")

    boundary.on_ending(lambda: released.append("first"))
    boundary.on_ending(broken)
    boundary.on_ending(lambda: released.append("last"))
    staged = ValueError("staged: the run broke")

    with pytest.raises(SystemExit if ending == "exit" else ValueError) as left:
        with boundary:
            if ending == "exit":
                boundary.exit(3, ending="staged", reason="staged exit")
            raise staged

    if ending == "exit":
        assert left.value.code == 3, "a failing release replaced the exit"
    else:
        assert left.value is staged
    assert released == ["last", "broken", "first"]
    assert "staged: a release broke" in capsys.readouterr().err


class _CountingProgress:
    def record_counts(self, iterations, completed):
        return dict(iterations=iterations, completed=completed)


class _ClosingPusher:
    """Records the boundary's last word on its pusher (`close_run` is stubbed
    below, so the boundary's own close is the only one)."""

    def __init__(self, log):
        self._log = log

    def close(self, timeout=None, final=None):
        self._log.append(f"pusher closed (timeout={timeout})")
        return True


# What each ending leaves in the log, in order: `region` and `console` are
# what the runner holds (`RunBoundary.hold`, the console first, as the parallel
# runner holds them), `housekeeping` is `close_run`.
_UNWOUND_FIRST = ["region", "console", "housekeeping"]
_DOOR_UNDER_THEM = ["housekeeping", "region", "console"]


@pytest.mark.parametrize("ending, expected, code", [
    ("normal", _UNWOUND_FIRST, None),
    ("exception", _UNWOUND_FIRST, None),
    ("keyboard interrupt", _UNWOUND_FIRST, ctrlc.EXIT_CODE),
    ("unwritten exit", _UNWOUND_FIRST, 2),
    ("exit door", _DOOR_UNDER_THEM, 3),
    ("interrupt door", _DOOR_UNDER_THEM, ctrlc.EXIT_CODE),
])
def test_the_boundary_orders_what_it_holds_against_the_housekeeping(
        monkeypatch, ending, expected, code):
    """The order of the two halves of an ending is the boundary's, not the
    runner's (`RunBoundary.hold`).

    An ending that unwinds the block — and the normal one, whose runner
    releases them where its work ends — releases the region and the console
    before the housekeeping prints; a door taken inside them (the sequential
    loop's) closes the run down under them, and they go as its exit leaves.
    Each is released once, last held first, and the pusher gets the
    boundary's last word on every ending.
    """
    log = []
    monkeypatch.setattr(runlifecycle, "close_run",
                        lambda ctx, **kw: log.append("housekeeping"))
    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=True,
        progress=_CountingProgress(),
        settings=runlifecycle.RunSettings(), registry=None,
        status_enabled=False)
    boundary = runlifecycle.RunBoundary(ctx, counts=lambda: (0, 0),
                                        pusher=_ClosingPusher(log))

    @contextlib.contextmanager
    def held(name):
        yield
        log.append(name)

    staged = ValueError("staged: the run broke")

    def run():
        with boundary:
            boundary.hold(held("console"))
            boundary.hold(held("region"))
            if ending == "normal":
                boundary.release_held()
                return boundary.end(stopchannel.RunResult(
                    stopchannel.RunStopReason.NO_WORK))
            if ending == "exception":
                raise staged
            if ending == "keyboard interrupt":
                raise KeyboardInterrupt
            if ending == "unwritten exit":
                raise SystemExit(2)
            if ending == "exit door":
                boundary.exit(3, ending="staged", reason="staged exit")
            boundary.interrupt()

    if code is None and ending == "normal":
        run()
    elif code is None:
        with pytest.raises(ValueError) as left:
            run()
        assert left.value is staged
    else:
        with pytest.raises(SystemExit) as left:
            run()
        assert left.value.code == code
    assert log == expected + ["pusher closed (timeout=0.5)"]


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_a_runner_releases_its_region_before_it_reports_and_closes_down(
        tmp_path, monkeypatch, exit_pushes, runner):
    """What each runner hands its boundary, and where it releases it.

    The sequential loop: its region, released before the driver's closing
    line. The parallel runner: its console window outside its region, so the
    region stops while the console owner still writes what it says, and the
    console is closed after it — both before the housekeeping.
    """
    log = []
    stop = statusline.StatusApp.stop

    def logged_stop(app):
        log.append("region")
        return stop(app)

    monkeypatch.setattr(statusline.StatusApp, "stop", logged_stop)
    real_close_run = runlifecycle.close_run

    def logged_close_run(*args, **kwargs):
        log.append("housekeeping")
        return real_close_run(*args, **kwargs)

    monkeypatch.setattr(runlifecycle, "close_run", logged_close_run)
    if runner == "sequential":
        class _ReportingDriver(OneShotDriver):
            def final_summary(self):
                log.append("report")
                return None

        monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **k: 0)
        monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
        cyclecore.run_loop(_ReportingDriver(), _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)
        assert log == ["region", "report", "housekeeping"]
        return
    close_console = parallel._close_console

    def logged_close_console():
        log.append("console")
        close_console()

    monkeypatch.setattr(parallel, "_close_console", logged_close_console)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    parallel.run_parallel(MemListDriver(["products/only.md"]),
                          par_args(str(tmp_path), jobs=1, no_statusline=True),
                          app_name="pytest-abnormal", wait_on_start=False)
    assert log == ["region", "console", "housekeeping"]


def test_the_two_doors_of_the_epilogue_run_the_same_housekeeping():
    """`end_run` must not grow a step `close_run` does not have.

    The three pins above watch the abnormal door; this watches that the normal
    one still goes through the same body, so the two cannot drift into doing
    different amounts of housekeeping. Checked on the SOURCE, because a runtime
    check would need a fourth staged run to say anything the pins above do not.
    """
    import ast
    import inspect

    def calls_close_run(statement) -> bool:
        return (isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == "close_run")

    def reaches_close_run(body) -> bool:
        """Is `close_run` called on the straight path through `body` — not
        under a branch, and not after a statement that leaves (a `return`, a
        `raise`, a call to `sys.exit` or to a door that exits)? Anywhere in
        the source was not enough: a call made unreachable passed."""
        for statement in body:
            if calls_close_run(statement):
                return True
            if isinstance(statement, ast.Try) and reaches_close_run(
                    statement.body):
                return True
            if isinstance(statement, (ast.Return, ast.Raise)):
                return False
            if (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and (ast.unparse(statement.value.func) == "sys.exit"
                         or ast.unparse(statement.value.func).startswith(
                             "_exit_"))):
                return False
        return False

    for door in (runlifecycle.end_run, runlifecycle.exit_run):
        function = ast.parse(inspect.getsource(door)).body[0]

        assert reaches_close_run(function.body), (
            f"{door.__name__} stopped delegating to close_run on every path — "
            f"the normal ending and the sys.exit endings are doing different "
            f"housekeeping again")


# --- the run's one boundary: endings no door of the runner wrote -------------


def _count_close_runs(monkeypatch) -> list:
    """The `ending` of every `close_run` call, so a pin can say "exactly once"."""
    endings = []
    real_close_run = runlifecycle.close_run

    def counted_close_run(*args, **kwargs):
        endings.append(kwargs.get("ending"))
        return real_close_run(*args, **kwargs)

    monkeypatch.setattr(runlifecycle, "close_run", counted_close_run)
    return endings


class _InterruptedStream:
    """A provider process whose stream Ctrl+C cuts mid-turn.

    Handed to the REAL `streamrender.run_agent_streaming`, so the pin covers
    what that function does with the interrupt, not what a stub does. The note
    is typed while the turn runs (see the five-errors pin for why then). The
    line after the press is one the turn must not render: a press is acted on
    at the stream's next item.
    """

    stdin = None

    def __init__(self, mailbox):
        self._mailbox = mailbox

    @property
    def stdout(self):
        self._mailbox.submit(NOTE)
        press_ctrl_c()
        yield "printed after the press\n"
        yield "never reached\n"  # pragma: no cover

    def wait(self):  # pragma: no cover — an interrupted turn never waits
        return 0


class _OverTheCeiling:
    """A usage source whose session is over any ceiling the pin sets."""

    def get_usage(self, cache_value=True):
        return usage.parse_usage({"five_hour": {"utilization": 90.0}})

    def invalidate(self):
        pass


class _HoldingPolicy(LimitPolicy):
    """The real quota hold, its snapshots recorded the way StubPolicy's are."""

    def __init__(self):
        super().__init__([SessionLimit(5)])
        self.snapshots = []

    def log_snapshot(self, source, label="", cache_value=True):
        self.snapshots.append(label)


class _SecondCommandDriver(Driver):
    """One command, then `on_second` from the second `next_command` — raised
    when an exception, called when callable (and None returned: no more
    work); `on_summary` likewise raised from `final_summary`."""

    def __init__(self, on_second=None, on_summary=None, policy=None):
        self.limit_policy = policy or StubPolicy()
        self.served = 0
        self.on_second = on_second
        self.on_summary = on_summary

    def next_command(self):
        self.served += 1
        if self.served == 1:
            return ClaudeCommand("do the thing", "", "the-thing")
        if isinstance(self.on_second, BaseException):
            raise self.on_second
        if callable(self.on_second):
            self.on_second()
        return None

    def final_summary(self):
        if isinstance(self.on_summary, BaseException):
            raise self.on_summary
        return None


@pytest.mark.parametrize("wait", ["stop file", "start-in"])
def test_ctrl_c_in_a_wait_before_the_boundary_records_the_interrupt_only(
        tmp_path, monkeypatch, capsys, exit_pushes, wait):
    """The waits ahead of the sequential run's boundary leave on Ctrl+C with
    the interrupt's reason in the exit record (`ctrlc.leave_wait`) — it read
    "reason not recorded", the prologue having opened the record — and with
    nothing else of an ending: no housekeeping, no exit push, and no pusher
    ever started, so none is left to release (`RunBoundary`)."""
    endings = _count_close_runs(monkeypatch)
    sentinel = tmp_path / "stop-sentinel"
    if wait == "stop file":
        sentinel.write_text("")
    monkeypatch.setattr(stopchannel, "stop_file_path", lambda: str(sentinel))

    def ctrl_c(seconds, should_stop=None):
        press_ctrl_c()
        return True

    monkeypatch.setattr(stopchannel, "sleep_unless", ctrl_c)
    started = []
    start = ownership.OwnerThread.start

    def recording_start(owner, *args, **kwargs):
        started.append(owner.name)
        return start(owner, *args, **kwargs)

    monkeypatch.setattr(ownership.OwnerThread, "start", recording_start)
    args = _seq_args(str(tmp_path))
    if wait == "start-in":
        args.start_in = "1m"

    with pytest.raises(SystemExit) as exit_info:
        cyclecore.run_loop(_AlwaysWorkDriver(), args, app_name="pytest-abnormal",
                           wait_on_start=(wait == "stop file"))

    assert exit_info.value.code == ctrlc.EXIT_CODE
    assert endings == [] and exit_pushes == []
    assert "pusher" not in started, f"a pusher started before the boundary: {started}"
    assert capsys.readouterr().out.count(ctrlc.WAIT_INTERRUPTED_LINE) == 1
    exitlog.finish()
    assert runlifecycle.INTERRUPTED_REASON in capsys.readouterr().out


@pytest.mark.parametrize("where", ["turn", "quota hold", "refusal wait",
                                   "driver"])
def test_ctrl_c_anywhere_in_a_sequential_run_still_closes_it_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, where):
    """Ctrl+C is the commonest way a sequential run ends, and it left nothing.

    The turn, the quota hold and the wait after a refusal each answered it
    with a `sys.exit(130)` of their own, straight past `close_run`: no exit
    push, no `at end` line, no report of the notes. Each is staged at the real
    site, so a site that exits on its own again is read here as an "exit 130"
    ending, not as the interrupt's. The turn's CLI is ended by the press itself
    (its `on_press` hook), not by whatever the run does next.
    """
    endings = _count_close_runs(monkeypatch)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)

    def succeeds(*args, **kwargs):
        loaded_mailbox.submit(NOTE)
        return 0

    def ctrl_c(seconds, should_stop=None):
        press_ctrl_c()
        return True

    asked_to_end = []
    driver = _SecondCommandDriver()
    if where == "turn":
        monkeypatch.setattr(streamrender, "start_agent_process",
                            lambda *a: _InterruptedStream(loaded_mailbox))
        monkeypatch.setattr(providers, "ask_agent_process_to_end",
                            asked_to_end.append)
        monkeypatch.setattr(streamrender, "reap_agent_process", lambda proc: None)
    elif where == "quota hold":
        # Before the first turn, so the note staged by the fixture is still
        # queued: the hold comes before the prompt takes the notes.
        driver = _SecondCommandDriver(policy=_HoldingPolicy())
        monkeypatch.setattr(runlifecycle, "usage_source_for",
                            lambda p: _OverTheCeiling())
        monkeypatch.setattr(limits, "sleep_unless", ctrl_c)
    elif where == "refusal wait":
        driver = _AlwaysWorkDriver()
        monkeypatch.setattr(cyclecore, "run_claude_streaming", succeeds)
        monkeypatch.setattr(
            cyclecore, "last_rate_limit_event",
            lambda: RateLimitEvent("rejected", "five_hour", time.time() + 3600))
        monkeypatch.setattr(stopchannel, "sleep_unless", ctrl_c)
    else:
        driver = _SecondCommandDriver(on_second=press_ctrl_c)
        monkeypatch.setattr(cyclecore, "run_claude_streaming", succeeds)

    try:
        with pytest.raises(SystemExit) as exit_info:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl+C left the run as a bare KeyboardInterrupt")

    assert exit_info.value.code == 130
    assert endings == ["interrupted"], (
        f"the run was not closed down exactly once, as interrupted: {endings}")
    if where == "turn":
        assert len(asked_to_end) == 1, "the press did not end the turn's CLI"
    out = _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (claude: interrupted)",
        reason=runlifecycle.INTERRUPTED_REASON).out
    assert "printed after the press" not in out, (
        "the turn went on rendering the stream after Ctrl+C")


@pytest.mark.parametrize("where", ["driver", "final summary"])
def test_an_exception_out_of_a_sequential_run_still_closes_it_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, where):
    """An exception is an ending too: closed down once, then raised on as is."""
    endings = _count_close_runs(monkeypatch)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)

    def succeeds(*args, **kwargs):
        loaded_mailbox.submit(NOTE)
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", succeeds)
    staged = RuntimeError("staged: the state file vanished")
    driver = (_SecondCommandDriver(on_second=staged) if where == "driver"
              else _SecondCommandDriver(on_summary=staged))

    with pytest.raises(RuntimeError) as raised:
        cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)

    assert raised.value is staged, "the exception was not the one let go on"
    assert endings == ["unhandled RuntimeError"], (
        f"the run was not closed down exactly once: {endings}")
    out = capsys.readouterr().out
    assert [where for _policy, where in exit_pushes] == [str(tmp_path)]
    assert driver.limit_policy.snapshots[-1] == \
        "at end (claude: unhandled RuntimeError)"
    assert NOTE in out and "undelivered operator note" in out


def test_a_provider_that_is_not_installed_still_closes_the_run_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox):
    """`streamrender`'s exit 2 keeps its code and gets the epilogue."""
    endings = _count_close_runs(monkeypatch)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())

    def not_installed(*args):
        loaded_mailbox.submit(NOTE)
        raise FileNotFoundError("claude")

    monkeypatch.setattr(streamrender, "start_agent_process", not_installed)
    driver = _SecondCommandDriver()

    with pytest.raises(SystemExit) as exit_info:
        cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)

    assert exit_info.value.code == 2
    assert endings == ["exit 2"], f"not closed down exactly once: {endings}"
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (claude: exit 2)", reason="sys.exit(2)")


@pytest.mark.parametrize("door", ["driver stop", "provider errors"])
def test_ctrl_c_in_an_exiting_door_s_push_keeps_the_door_s_exit(
        tmp_path, monkeypatch, capsys, loaded_mailbox, door):
    """Ctrl+C in an exiting door's push gives up the push, not the ending.

    The doors used to let the interrupt out bare — neither 130 nor the
    driver's code, and no record of the interrupt — and then turned the
    ending into the interrupt's, 130 over exit 3. The ending is chosen
    before the housekeeping now and kept: the press gives up the exit push,
    the snapshot and the notes still happen, and the run leaves with the
    door's code and the reason that explains it.
    """
    pushes = []
    stuck = threading.Event()

    def ctrl_c_in_the_push(policy, project_dir, abort=None):
        pushes.append((policy, project_dir))
        press_ctrl_c()
        stuck.wait(timeout=STUCK_PUSH_S)   # given up by the press, not done

    def turn(*args, **kwargs):
        loaded_mailbox.submit(NOTE)
        return 0 if door == "driver stop" else 7

    monkeypatch.setattr(runlifecycle, "final_git_push", ctrl_c_in_the_push)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", turn)
    if door == "driver stop":
        driver = _StoppingDriver(commands=1)
        ending = "driver stopped the run"
    else:
        driver = _AlwaysWorkDriver()
        ending = "provider errors in a row"

    started = time.monotonic()
    try:
        with pytest.raises(SystemExit) as exit_info:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl+C left the door as a bare KeyboardInterrupt")
    finally:
        elapsed = time.monotonic() - started
        stuck.set()

    assert elapsed < STUCK_PUSH_S / 2, (
        f"the press did not give up the exit push: {elapsed:.1f} s")
    assert exit_info.value.code == (3 if door == "driver stop" else 7), (
        "the press rewrote the door's exit")
    out = _assert_closed_down(
        pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot=f"at end (claude: {ending})",
        reason=("the driver stopped the run (exit 3)" if door == "driver stop"
                else "5 provider errors in a row (last exit code 7)")).out
    assert "the exit push is abandoned" in out


@pytest.mark.parametrize("staged", ["press", KeyboardInterrupt, RuntimeError],
                         ids=["ctrl-c", "raised-keyboard-interrupt",
                              "exception"])
def test_an_ending_in_the_parallel_preparation_still_closes_the_run_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, staged):
    """Between `open_usage` and the status region the fleet had no boundary.

    Staged where the worker pool is built: the usage is open, the mailboxes
    exist, the pusher has not started. A press there starts no worker; a
    KeyboardInterrupt raised there — what SIGINT still is for a runner off
    the main thread (`ctrlc.captured`) — is the same ending.
    """
    endings = _count_close_runs(monkeypatch)
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())
    real_pool = parallel.WorkerPool

    def pool_fails(*args, **kwargs):
        if staged == "press":
            press_ctrl_c()
            return real_pool(*args, **kwargs)
        raise staged("staged: the pool could not be built")

    monkeypatch.setattr(parallel, "WorkerPool", pool_fails)
    driver = MemListDriver(["products/only.md"])
    args = _par_args(str(tmp_path))

    if staged != RuntimeError:
        try:
            with pytest.raises(SystemExit) as exit_info:
                parallel.run_parallel(driver, args, app_name="pytest-abnormal",
                                      wait_on_start=False)
        except KeyboardInterrupt:
            pytest.fail("the Ctrl+C unwound the run past its epilogue")
        assert exit_info.value.code == 130
        assert endings == ["interrupted"], f"not closed once: {endings}"
        _assert_closed_down(
            exit_pushes, driver.limit_policy, capsys, str(tmp_path),
            snapshot="at end (parallel claude: interrupted)",
            reason=runlifecycle.INTERRUPTED_REASON)
        return

    with pytest.raises(RuntimeError, match="staged"):
        parallel.run_parallel(driver, args, app_name="pytest-abnormal",
                              wait_on_start=False)
    assert endings == ["unhandled RuntimeError"], f"not closed once: {endings}"
    out = capsys.readouterr().out
    assert [where for _policy, where in exit_pushes] == [str(tmp_path)]
    assert driver.limit_policy.snapshots[-1] == \
        "at end (parallel claude: unhandled RuntimeError)"
    assert NOTE in out and "undelivered operator note" in out


# --- second failures inside the ending ----------------------------------------

# How long the stuck exit push below would hold the run if nothing bounded the
# wait: far past the shortened deadline and the whole bounded run (1.01 and
# 1.03 s sequential — `run_loop`'s 0.5 s owner close included — 0.51 s
# parallel, measured 2026-10-03), so a run that waited it out is told apart by
# its time alone.
STUCK_PUSH_S = 30.0


def _noting_turn(mailbox):
    """A turn that succeeds with a note typed while it runs (see the
    five-errors pin for why then)."""
    def turn(*args, **kwargs):
        mailbox.submit(NOTE)
        return 0
    return turn


def _seq_run_raising(driver, project_dir):
    return lambda: cyclecore.run_loop(driver, _seq_args(project_dir),
                                      app_name="pytest-abnormal",
                                      wait_on_start=False)


@pytest.mark.parametrize("runner", ["sequential", "parallel",
                                    "parallel preparation"])
def test_an_exception_does_not_wait_out_a_stuck_exit_push(
        tmp_path, monkeypatch, capsys, loaded_mailbox, runner):
    """An exception's ending waits for its exit push a bounded time, then leaves.

    Nobody may be there to press Ctrl+C: a batch run unwinding an exception
    sat out the push in flight and the whole exit push, silently, before its
    traceback. Staged with an exit push that never comes back on its own; the
    run must still raise its own exception within the (shortened) deadline,
    say why it paused, abandon the push so no further git starts, and keep
    the snapshot and the notes.

    "parallel preparation" fails where the worker pool is built: the run has
    made its pusher and not started it, and an owner that never started runs
    its `final` inline — the stuck push on the main thread, where no deadline
    was being kept.
    """
    monkeypatch.setattr(runlifecycle, "UNWIND_PUSH_DEADLINE_S", 0.5)
    stuck = threading.Event()
    aborts = []

    def stuck_push(policy, project_dir, abort=None):
        aborts.append(abort)
        stuck.wait(timeout=STUCK_PUSH_S)

    monkeypatch.setattr(runlifecycle, "final_git_push", stuck_push)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    staged = RuntimeError("staged: the driver broke")
    if runner == "sequential":
        monkeypatch.setattr(cyclecore, "run_claude_streaming",
                            _noting_turn(loaded_mailbox))
        driver = _SecondCommandDriver(on_second=staged)
        run = _seq_run_raising(driver, str(tmp_path))
        snapshot = "at end (claude: unhandled RuntimeError)"
    elif runner == "parallel preparation":
        def pool_fails(*args, **kwargs):
            raise staged

        monkeypatch.setattr(parallel, "WorkerPool", pool_fails)
        driver = MemListDriver(["products/only.md"])
        args = _par_args(str(tmp_path))
        run = lambda: parallel.run_parallel(driver, args,  # noqa: E731
                                            app_name="pytest-abnormal",
                                            wait_on_start=False)
        snapshot = "at end (parallel claude: unhandled RuntimeError)"
    else:
        def broken_join(threads):
            for t in threads:
                t.join()
            loaded_mailbox.submit(NOTE)     # no worker left to splice it
            raise staged

        monkeypatch.setattr(parallel, "join_workers", broken_join)
        monkeypatch.setattr(parallel, "run_job",
                            lambda job_id, command, mailbox=None: (0, None, None))
        driver = MemListDriver(["products/only.md"])
        args = _par_args(str(tmp_path))
        run = lambda: parallel.run_parallel(driver, args,  # noqa: E731
                                            app_name="pytest-abnormal",
                                            wait_on_start=False)
        snapshot = "at end (parallel claude: unhandled RuntimeError)"

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError) as raised:
            run()
    finally:
        elapsed = time.monotonic() - started
        stuck.set()

    assert raised.value is staged, "the exception was not the one let go on"
    assert elapsed < STUCK_PUSH_S / 2, (
        f"the ending waited out the stuck exit push: {elapsed:.1f} s")
    assert aborts and aborts[0] is not None and aborts[0].is_set(), (
        "the abandoned exit push may still start git")
    assert driver.limit_policy.snapshots[-1] == snapshot
    captured = capsys.readouterr()
    assert "waiting at most 0.5 s for the exit push" in captured.out, (
        "the run went quiet without saying why")
    assert "did not finish within 0.5 s" in captured.err
    assert NOTE in captured.out and "undelivered operator note" in captured.out


@pytest.mark.parametrize("door", ["driver stop", "provider errors"])
def test_an_exiting_door_does_not_wait_out_a_stuck_exit_push(
        tmp_path, monkeypatch, capsys, loaded_mailbox, door):
    """The runner's own exiting doors are bounded the way an exception is.

    The driver stopping the run and five provider errors in a row are what
    ends a run left going overnight, with nobody there to press Ctrl+C: they
    waited for a stuck exit push without limit. Their code and reason stay.
    """
    monkeypatch.setattr(runlifecycle, "UNWIND_PUSH_DEADLINE_S", 0.5)
    stuck = threading.Event()
    aborts = []

    def stuck_push(policy, project_dir, abort=None):
        aborts.append(abort)
        stuck.wait(timeout=STUCK_PUSH_S)

    monkeypatch.setattr(runlifecycle, "final_git_push", stuck_push)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    def turn(*args, **kwargs):
        loaded_mailbox.submit(NOTE)     # see the five-errors pin for why here
        return 0 if door == "driver stop" else 7

    monkeypatch.setattr(cyclecore, "run_claude_streaming", turn)
    driver = (_StoppingDriver(commands=1) if door == "driver stop"
              else _AlwaysWorkDriver())

    started = time.monotonic()
    try:
        with pytest.raises(SystemExit) as exit_info:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    finally:
        elapsed = time.monotonic() - started
        stuck.set()

    assert elapsed < STUCK_PUSH_S / 2, (
        f"the door waited out the stuck exit push: {elapsed:.1f} s")
    assert exit_info.value.code == (3 if door == "driver stop" else 7)
    assert aborts and aborts[0] is not None and aborts[0].is_set(), (
        "the abandoned exit push may still start git")
    captured = capsys.readouterr()
    assert "did not finish within 0.5 s" in captured.err
    assert NOTE in captured.out and "undelivered operator note" in captured.out


def test_a_spent_stop_budget_starts_no_exit_push(capsys, monkeypatch):
    """A StopBudget spent before the push begins: no git starts at all."""
    pushes = []
    monkeypatch.setattr(runlifecycle, "final_git_push",
                        lambda *args, **kwargs: pushes.append(args))
    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=False, progress=None,
        settings=runlifecycle.RunSettings(), registry=None,
        status_enabled=False)

    runlifecycle.close_run(ctx, usages=[],
                           budget=runlifecycle.StopBudget(0))

    assert pushes == [], "a push started on a spent budget"
    assert "did not finish within 0 s" in capsys.readouterr().err


# The fleet's join budget in the pin below, and how many workers it holds:
# waited per worker the run would take JOIN_BUDGET_S x HELD_WORKERS (6 s);
# one budget for the fleet takes JOIN_BUDGET_S plus the run's own overhead
# (0.51 s for a whole parallel run, measured 2026-10-03), so 4.5 s tells the
# two apart with room either side.
JOIN_BUDGET_S = 2.0
HELD_WORKERS = 3


def test_an_interrupted_fleet_waits_for_its_workers_one_budget_in_all(
        tmp_path, monkeypatch, capsys, exit_pushes):
    """INTERRUPT_JOIN_TIMEOUT_S is the fleet's, not each worker's."""
    monkeypatch.setattr(parallel, "INTERRUPT_JOIN_TIMEOUT_S", JOIN_BUDGET_S)
    release = threading.Event()
    working = []
    workers = []
    all_working = threading.Event()

    def held_turn(job_id, command, mailbox=None):
        working.append(job_id)
        workers.append(threading.current_thread())
        if len(working) == HELD_WORKERS:
            all_working.set()
        release.wait(timeout=HELD_S)
        return 0, None, None

    def ctrl_c_once_all_work(threads):
        all_working.wait(timeout=HELD_S)
        press_ctrl_c()

    monkeypatch.setattr(parallel, "run_job", held_turn)
    monkeypatch.setattr(parallel, "join_workers", ctrl_c_once_all_work)
    driver = MemListDriver([f"products/item{i}.md" for i in range(5)])
    args = par_args(str(tmp_path), jobs=HELD_WORKERS, no_statusline=True)

    started = time.monotonic()
    try:
        with pytest.raises(SystemExit) as exit_info:
            parallel.run_parallel(driver, args, app_name="pytest-abnormal",
                                  wait_on_start=False)
    finally:
        elapsed = time.monotonic() - started
        release.set()
        # Joined here, not left to finish in a later test: a worker let go
        # prints its "✓" into whatever test runs next.
        for t in workers:
            t.join(timeout=HELD_S)

    assert len(working) == HELD_WORKERS, f"not all held: {working}"
    assert not any(t.is_alive() for t in workers), "a held worker never left"
    assert exit_info.value.code == 130
    # Both ways: no wait at all passes the upper bound too, and the workers
    # are held past the budget, so the fleet waits it out exactly once.
    assert elapsed >= JOIN_BUDGET_S, (
        f"the fleet's join took {elapsed:.1f} s — it never waited")
    assert elapsed < 4.5, (
        f"the fleet's join took {elapsed:.1f} s — a budget per worker?")


def test_an_exception_out_of_the_parallel_region_stops_the_workers(
        tmp_path, monkeypatch, capsys, exit_pushes):
    """An exception out of the region must not leave the fleet claiming.

    Only Ctrl+C used to stop the workers: an exception out of the region went
    to the boundary with `shared.stop` unset and the pool open, so the workers
    went on claiming the queue and starting agents through the exit push, and
    died with the process mid-turn. Staged: the join fails while a worker is
    inside its turn, and the worker is let go only once the run has left.
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

    def join_fails_once_a_worker_works(threads):
        working.wait(timeout=HELD_S)
        raise RuntimeError("staged: the join broke")

    monkeypatch.setattr(parallel, "run_job", held_first_turn)
    monkeypatch.setattr(parallel, "join_workers", join_fails_once_a_worker_works)
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())
    driver = MemListDriver([f"products/item{i}.md" for i in range(5)])

    try:
        with pytest.raises(RuntimeError, match="staged"):
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
    assert exit_pushes, "the run left without its exit push"


def test_a_claim_stuck_in_the_pool_s_lock_does_not_hold_an_exception_s_ending(
        tmp_path, monkeypatch, capsys, exit_pushes):
    """Stopping the fleet must not wait for the pool's lock.

    A worker holds that lock across its whole claim, the driver's read of the
    list file included, and the pool's close used to take it: a claim stuck
    in that I/O held an exception's ending in `stop_workers`, before the exit
    push's deadline had even begun. Staged: the lock is held, on another
    thread, as long as STUCK_PUSH_S at most, and the region then raises.
    """
    pools = []
    real_pool = parallel.WorkerPool

    def recording_pool(*args, **kwargs):
        pools.append(real_pool(*args, **kwargs))
        return pools[-1]

    held = threading.Event()
    release = threading.Event()

    def hold_the_pool_lock():
        with pools[0]._lock:
            held.set()
            release.wait(timeout=STUCK_PUSH_S)

    def join_then_fail_under_a_held_claim(threads):
        for t in threads:
            t.join()
        threading.Thread(target=hold_the_pool_lock, daemon=True).start()
        assert held.wait(timeout=HELD_S)
        raise RuntimeError("staged: the join broke")

    monkeypatch.setattr(parallel, "WorkerPool", recording_pool)
    monkeypatch.setattr(parallel, "join_workers",
                        join_then_fail_under_a_held_claim)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())

    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="staged"):
            parallel.run_parallel(MemListDriver(["products/only.md"]),
                                  _par_args(str(tmp_path)),
                                  app_name="pytest-abnormal",
                                  wait_on_start=False)
    finally:
        elapsed = time.monotonic() - started
        release.set()

    assert held.is_set(), "the pool's lock was never held — nothing staged"
    assert elapsed < STUCK_PUSH_S / 2, (
        f"the ending waited for the pool's lock: {elapsed:.1f} s")
    assert pools[0].grow() is None, "the pool still takes workers"
    assert exit_pushes, "the run left without its exit push"


class _ClosingSource(StubSource):
    """A source that keeps something running between reads, and says if closed."""

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FailingOpenPolicy(StubPolicy):
    """The opening snapshot raises `staged` — or calls it, when it is not an
    exception; every snapshot is recorded."""

    def __init__(self, staged):
        super().__init__()
        self.staged = staged

    def log_snapshot(self, source, label="", cache_value=True):
        super().log_snapshot(source, label, cache_value)
        if label.startswith("at start"):
            if isinstance(self.staged, BaseException):
                raise self.staged
            self.staged()


@pytest.mark.parametrize("staged", [RuntimeError, KeyboardInterrupt],
                         ids=["exception", "ctrl-c"])
@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_an_ending_in_the_opening_snapshot_still_closes_the_usage(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, runner,
        staged):
    """A usage whose opening snapshot failed is still the run's to close.

    The snapshot is a usage query, and the source may have started what it
    keeps running (Codex's quota server) before the query raised or Ctrl+C was
    pressed in it. The sequential runner stored the pair only once
    `open_usage` returned, so the ending closed nothing; the parallel runner
    opened it outside its boundary, so the ending had no epilogue at all. A
    press there must not let the sequential run launch its turn either.
    """
    endings = _count_close_runs(monkeypatch)
    source = _ClosingSource()
    policy = _FailingOpenPolicy(
        press_ctrl_c if staged is KeyboardInterrupt
        else staged("staged: the usage endpoint broke"))
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: source)
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    if runner == "sequential":
        driver = _SecondCommandDriver(policy=policy)
        run = _seq_run_raising(driver, str(tmp_path))
        name = "claude"
    else:
        driver = MemListDriver(["products/only.md"])
        driver.limit_policy = policy
        args = _par_args(str(tmp_path))
        run = lambda: parallel.run_parallel(driver, args,  # noqa: E731
                                            app_name="pytest-abnormal",
                                            wait_on_start=False)
        name = "parallel claude"

    if staged is KeyboardInterrupt:
        with pytest.raises(SystemExit) as exit_info:
            run()
        assert exit_info.value.code == 130
        ending = "interrupted"
    else:
        with pytest.raises(RuntimeError, match="staged"):
            run()
        ending = "unhandled RuntimeError"

    assert endings == [ending], f"not closed down exactly once: {endings}"
    assert source.closed, "the source the opening query started was left running"
    assert policy.snapshots[-1] == f"at end ({name}: {ending})"
    if runner == "sequential":
        # The fleet makes its mailboxes after the usage: none can hold a note.
        out = capsys.readouterr().out
        assert NOTE in out and "undelivered operator note" in out


def test_a_failing_step_of_an_exiting_door_keeps_its_exit_code(
        tmp_path, monkeypatch, capsys, exit_pushes):
    """The driver's exit 3 survives a housekeeping step that raises.

    The ending is already decided and recorded when `exit_run` closes the run
    down; an exception out of that (here the report of undelivered notes)
    used to replace `SystemExit(3)` with a traceback and exit 1.
    """
    def report_fails(mailbox):
        raise RuntimeError("staged: the console is gone")

    monkeypatch.setattr(operator, "report_undelivered_notes", report_fails)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **k: 0)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)

    with pytest.raises(SystemExit) as exit_info:
        cyclecore.run_loop(_StoppingDriver(commands=1), _seq_args(str(tmp_path)),
                           app_name="pytest-abnormal", wait_on_start=False)

    assert exit_info.value.code == 3, "the housekeeping's failure took the exit"
    err = capsys.readouterr().err
    assert "closing the run down failed" in err and "staged" in err, (
        "the failing step went unreported")


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_a_second_ctrl_c_while_the_reason_is_recorded_abandons_the_push(
        tmp_path, monkeypatch, capsys, loaded_mailbox, runner):
    """The interrupt's reason is written as the run leaves — where Ctrl+C comes.

    A second Ctrl+C inside `exitlog.set_reason` (a lock and a file write) left
    the boundary as a bare KeyboardInterrupt: no housekeeping, no 130. Then it
    was swallowed, and the ending went on to wait for its exit push without
    limit — over a stuck push the operator had to press Ctrl+C a third time.
    The second Ctrl+C is the operator abandoning that push: none is waited
    for (none is started), the snapshot and the notes still happen, 130.

    Staged with an exit push that never comes back on its own, so a run that
    waits for it is told apart by its time alone.
    """
    endings = _count_close_runs(monkeypatch)
    stuck = threading.Event()
    pushes = []

    def stuck_push(policy, project_dir, abort=None):
        pushes.append(abort)
        stuck.wait(timeout=STUCK_PUSH_S)

    monkeypatch.setattr(runlifecycle, "final_git_push", stuck_push)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    real_set_reason = exitlog.set_reason
    cut = []

    def ctrl_c_in_the_write(reason, **fields):
        real_set_reason(reason, **fields)
        if reason == runlifecycle.INTERRUPTED_REASON and not cut:
            cut.append(reason)
            press_ctrl_c()

    monkeypatch.setattr(exitlog, "set_reason", ctrl_c_in_the_write)
    if runner == "sequential":
        monkeypatch.setattr(cyclecore, "run_claude_streaming",
                            _noting_turn(loaded_mailbox))
        driver = _SecondCommandDriver(on_second=press_ctrl_c)
        run = _seq_run_raising(driver, str(tmp_path))
        snapshot = "at end (claude: interrupted)"
    else:
        def interrupt(threads):
            for t in threads:
                t.join()
            loaded_mailbox.submit(NOTE)     # no worker left to splice it
            press_ctrl_c()

        monkeypatch.setattr(parallel, "join_workers", interrupt)
        monkeypatch.setattr(parallel, "run_job",
                            lambda job_id, command, mailbox=None: (0, None, None))
        driver = MemListDriver(["products/only.md"])
        args = _par_args(str(tmp_path))
        run = lambda: parallel.run_parallel(driver, args,  # noqa: E731
                                            app_name="pytest-abnormal",
                                            wait_on_start=False)
        snapshot = "at end (parallel claude: interrupted)"

    started = time.monotonic()
    try:
        with pytest.raises(SystemExit) as exit_info:
            run()
    except KeyboardInterrupt:
        pytest.fail("the second Ctrl+C left the run as a bare KeyboardInterrupt")
    finally:
        elapsed = time.monotonic() - started
        stuck.set()

    assert cut, "the reason was never recorded — nothing staged"
    assert elapsed < STUCK_PUSH_S / 2, (
        f"the second Ctrl+C still waited for the exit push: {elapsed:.1f} s")
    assert pushes == [], "the push the operator abandoned was started anyway"
    assert exit_info.value.code == 130
    assert endings == ["interrupted"], f"not closed down exactly once: {endings}"
    assert driver.limit_policy.snapshots[-1] == snapshot
    out = capsys.readouterr().out
    assert "the exit push is abandoned" in out
    assert NOTE in out and "undelivered operator note" in out
    exitlog.finish()
    assert runlifecycle.INTERRUPTED_REASON in capsys.readouterr().out


def test_a_ctrl_c_while_a_normal_ending_s_reason_is_recorded_exits_130(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox):
    """The first Ctrl+C, landing in `end_run`'s record, still ends with 130.

    Swallowing a Ctrl+C there let the run RETURN its result as if nobody had
    pressed it. The record now comes before the housekeeping, so the press is
    one inside the ending: it gives up the exit push before it starts, and
    the snapshot and the notes still happen.
    """
    real_set_reason = exitlog.set_reason
    cut = []

    def ctrl_c_in_the_write(reason, **fields):
        real_set_reason(reason, **fields)
        if reason != runlifecycle.INTERRUPTED_REASON and not cut:
            cut.append(reason)
            press_ctrl_c()

    monkeypatch.setattr(exitlog, "set_reason", ctrl_c_in_the_write)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        _noting_turn(loaded_mailbox))
    driver = OneShotDriver()

    try:
        with pytest.raises(SystemExit) as exit_info:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl+C left end_run as a bare KeyboardInterrupt")

    assert cut, "the normal ending's reason was never recorded — nothing staged"
    assert exit_info.value.code == 130, "the Ctrl+C was lost"
    assert exit_pushes == [], "the push the operator abandoned was started"
    assert driver.limit_policy.snapshots[-1] == "at end (claude)"
    out = capsys.readouterr().out
    assert "the exit push is abandoned" in out
    assert NOTE in out and "undelivered operator note" in out
    exitlog.finish()
    assert runlifecycle.INTERRUPTED_REASON in capsys.readouterr().out


def test_a_ctrl_c_in_the_region_s_teardown_keeps_the_door_s_exit(
        tmp_path, monkeypatch, capsys, exit_pushes):
    """A door inside the sequential region exits; Ctrl+C in the teardown after.

    `StatusApp.stop` raises the first exception it meets, so a Ctrl+C there
    replaced the driver's `SystemExit(3)` with a bare KeyboardInterrupt, and
    the boundary — already closed by the door — let it go. A press after the
    door has done its housekeeping changes nothing: the door's exit stands.
    """
    real_stop = statusline.StatusApp.stop

    def ctrl_c_in_the_teardown(app):
        real_stop(app)
        press_ctrl_c()

    monkeypatch.setattr(statusline.StatusApp, "stop", ctrl_c_in_the_teardown)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **k: 0)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)

    try:
        with pytest.raises(SystemExit) as exit_info:
            cyclecore.run_loop(_StoppingDriver(commands=1),
                               _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl+C in the teardown replaced the door's exit")

    assert exit_info.value.code == 3
    assert len(exit_pushes) == 1, f"not closed down once: {exit_pushes}"


def test_a_ctrl_c_in_an_exception_s_exit_push_keeps_the_exception(
        tmp_path, monkeypatch, capsys, loaded_mailbox):
    """Ctrl+C while an exception's exit push runs gives up the push only.

    The exception stays the ending: the snapshot and the notes still happen,
    and what leaves `run_loop` is the exception, not the interrupt.
    """
    endings = _count_close_runs(monkeypatch)
    pushes = []
    stuck = threading.Event()

    def ctrl_c_in_the_push(policy, project_dir, abort=None):
        pushes.append((policy, project_dir))
        press_ctrl_c()
        stuck.wait(timeout=STUCK_PUSH_S)   # given up by the press, not done

    monkeypatch.setattr(runlifecycle, "final_git_push", ctrl_c_in_the_push)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        _noting_turn(loaded_mailbox))
    staged = RuntimeError("staged: the state file vanished")
    driver = _SecondCommandDriver(on_second=staged)

    try:
        with pytest.raises(RuntimeError) as raised:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    except (KeyboardInterrupt, SystemExit) as replaced:
        pytest.fail(f"the Ctrl+C replaced the exception: {replaced!r}")
    finally:
        stuck.set()

    assert raised.value is staged
    assert endings == ["unhandled RuntimeError"], f"not closed once: {endings}"
    assert pushes, "the exit push never ran — nothing staged"
    assert driver.limit_policy.snapshots[-1] == \
        "at end (claude: unhandled RuntimeError)"
    out = capsys.readouterr().out
    assert "the exit push is abandoned" in out
    assert NOTE in out and "undelivered operator note" in out
