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

import threading
import time

import pytest

from llm_loop import (cyclecore, exitlog, limits, operator, parallel,
                      runlifecycle, statusline, stopchannel, streamrender,
                      usage)
from llm_loop.agentwork import ClaudeCommand, Driver, LoopStop
from llm_loop.drivers import StateFileDriver
from llm_loop.limits import LimitPolicy, SessionLimit
from llm_loop.usage import RateLimitEvent

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

    `close` used to wait for the pool's lock, which a worker holds across its
    claim, and that wait was where a second Ctrl+C landed (it waits no more —
    see the stuck-claim pin below — but a Ctrl+C can still land in any step
    of it). The hearing used to mark the
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

    for door in (runlifecycle.end_run, runlifecycle.exit_run):
        tree = ast.parse(inspect.getsource(door))
        called = {node.func.id for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)}

        assert "close_run" in called, (
            f"{door.__name__} stopped delegating to close_run — the normal "
            f"ending and the sys.exit endings are doing different housekeeping "
            f"again")


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
    """A provider process whose stream is cut by Ctrl+C mid-turn.

    Handed to the REAL `streamrender.run_agent_streaming`, so the pin covers
    what that function does with the interrupt, not what a stub does. The note
    is typed while the turn runs (see the five-errors pin for why then).
    """

    stdin = None

    def __init__(self, mailbox):
        self._mailbox = mailbox

    @property
    def stdout(self):
        self._mailbox.submit(NOTE)
        raise KeyboardInterrupt
        yield  # pragma: no cover — makes this a generator, as a pipe iterates

    def wait(self):  # pragma: no cover — the stream never ends
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
    """One command, then `on_second` — raised, or returned when not an
    exception — from the second `next_command`; `on_summary` likewise from
    `final_summary`."""

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
        return None

    def final_summary(self):
        if isinstance(self.on_summary, BaseException):
            raise self.on_summary
        return None


@pytest.mark.parametrize("where", ["turn", "quota hold", "refusal wait",
                                   "driver"])
def test_ctrl_c_anywhere_in_a_sequential_run_still_closes_it_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, where):
    """Ctrl+C is the commonest way a sequential run ends, and it left nothing.

    The turn, the quota hold and the wait after a refusal each answered it
    with a `sys.exit(130)` of their own, straight past `close_run`: no exit
    push, no `at end` line, no report of the notes. Each is staged at the real
    site, so a site that exits on its own again is read here as an "exit 130"
    ending, not as the interrupt's.
    """
    endings = _count_close_runs(monkeypatch)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)

    def succeeds(*args, **kwargs):
        loaded_mailbox.submit(NOTE)
        return 0

    driver = _SecondCommandDriver()
    if where == "turn":
        monkeypatch.setattr(streamrender, "start_agent_process",
                            lambda *a: _InterruptedStream(loaded_mailbox))
        monkeypatch.setattr(streamrender, "reap_agent_process", lambda proc: None)
    elif where == "quota hold":
        # Before the first turn, so the note staged by the fixture is still
        # queued: the hold comes before the prompt takes the notes.
        driver = _SecondCommandDriver(policy=_HoldingPolicy())
        monkeypatch.setattr(runlifecycle, "usage_source_for",
                            lambda p: _OverTheCeiling())

        def ctrl_c(seconds, should_stop=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(limits, "sleep_unless", ctrl_c)
    elif where == "refusal wait":
        driver = _AlwaysWorkDriver()
        monkeypatch.setattr(cyclecore, "run_claude_streaming", succeeds)
        monkeypatch.setattr(
            cyclecore, "last_rate_limit_event",
            lambda: RateLimitEvent("rejected", "five_hour", time.time() + 3600))

        def ctrl_c(seconds, should_stop=None):
            raise KeyboardInterrupt

        monkeypatch.setattr(stopchannel, "sleep_unless", ctrl_c)
    else:
        driver = _SecondCommandDriver(on_second=KeyboardInterrupt())
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
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (claude: interrupted)",
        reason=runlifecycle.INTERRUPTED_REASON)


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
def test_ctrl_c_in_an_exiting_door_s_push_exits_130(
        tmp_path, monkeypatch, capsys, loaded_mailbox, door):
    """The two `sys.exit` doors end a Ctrl+C in their push the way Ctrl+C ends.

    `close_run` gives up the push and raises the interrupt on; the doors used
    to let it out bare — neither 130 nor the driver's code, and no record of
    the interrupt.
    """
    pushes = []

    def ctrl_c_in_the_push(policy, project_dir, abort=None):
        pushes.append((policy, project_dir))
        raise KeyboardInterrupt

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

    try:
        with pytest.raises(SystemExit) as exit_info:
            cyclecore.run_loop(driver, _seq_args(str(tmp_path)),
                               app_name="pytest-abnormal", wait_on_start=False)
    except KeyboardInterrupt:
        pytest.fail("the Ctrl+C left the door as a bare KeyboardInterrupt")

    assert exit_info.value.code == 130
    _assert_closed_down(
        pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot=f"at end (claude: {ending})",
        reason=runlifecycle.INTERRUPTED_REASON)


@pytest.mark.parametrize("staged", [KeyboardInterrupt, RuntimeError],
                         ids=["ctrl-c", "exception"])
def test_an_ending_in_the_parallel_preparation_still_closes_the_run_down(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, staged):
    """Between `open_usage` and the status region the fleet had no boundary.

    Staged where the worker pool is built: the usage is open, the mailboxes
    exist, the pusher has not started.
    """
    endings = _count_close_runs(monkeypatch)
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())

    def pool_fails(*args, **kwargs):
        raise staged("staged: the pool could not be built")

    monkeypatch.setattr(parallel, "WorkerPool", pool_fails)
    driver = MemListDriver(["products/only.md"])
    args = _par_args(str(tmp_path))

    if staged is KeyboardInterrupt:
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
    """The opening snapshot raises `staged`; every other one is recorded."""

    def __init__(self, staged):
        super().__init__()
        self.staged = staged

    def log_snapshot(self, source, label="", cache_value=True):
        super().log_snapshot(source, label, cache_value)
        if label.startswith("at start"):
            raise self.staged


@pytest.mark.parametrize("staged", [RuntimeError, KeyboardInterrupt],
                         ids=["exception", "ctrl-c"])
@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_an_ending_in_the_opening_snapshot_still_closes_the_usage(
        tmp_path, monkeypatch, capsys, exit_pushes, loaded_mailbox, runner,
        staged):
    """A usage whose opening snapshot failed is still the run's to close.

    The snapshot is a usage query, and the source may have started what it
    keeps running (Codex's quota server) before the query raised or was cut
    by Ctrl+C. The sequential runner stored the pair only once `open_usage`
    returned, so the ending closed nothing; the parallel runner opened it
    outside its boundary, so the ending had no epilogue at all.
    """
    endings = _count_close_runs(monkeypatch)
    source = _ClosingSource()
    policy = _FailingOpenPolicy(staged("staged: the usage endpoint broke"))
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
            raise KeyboardInterrupt

    monkeypatch.setattr(exitlog, "set_reason", ctrl_c_in_the_write)
    if runner == "sequential":
        monkeypatch.setattr(cyclecore, "run_claude_streaming",
                            _noting_turn(loaded_mailbox))
        driver = _SecondCommandDriver(on_second=KeyboardInterrupt())
        run = _seq_run_raising(driver, str(tmp_path))
        snapshot = "at end (claude: interrupted)"
    else:
        def interrupt(threads):
            for t in threads:
                t.join()
            loaded_mailbox.submit(NOTE)     # no worker left to splice it
            raise KeyboardInterrupt

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

    `end_run` records its reason after the housekeeping, and swallowing a
    Ctrl+C there let the run RETURN its result as if nobody had pressed it.
    """
    real_set_reason = exitlog.set_reason
    cut = []

    def ctrl_c_in_the_write(reason, **fields):
        real_set_reason(reason, **fields)
        if reason != runlifecycle.INTERRUPTED_REASON and not cut:
            cut.append(reason)
            raise KeyboardInterrupt

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
    _assert_closed_down(
        exit_pushes, driver.limit_policy, capsys, str(tmp_path),
        snapshot="at end (claude)",
        reason=runlifecycle.INTERRUPTED_REASON)


def test_a_ctrl_c_in_the_region_s_teardown_keeps_the_door_s_exit(
        tmp_path, monkeypatch, capsys, exit_pushes):
    """A door inside the sequential region exits; Ctrl+C in the teardown after.

    `StatusApp.stop` raises the first exception it meets, so a Ctrl+C there
    replaced the driver's `SystemExit(3)` with a bare KeyboardInterrupt, and
    the boundary — already closed by the door — let it go.
    """
    real_stop = statusline.StatusApp.stop

    def ctrl_c_in_the_teardown(app):
        real_stop(app)
        raise KeyboardInterrupt

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

    def ctrl_c_in_the_push(policy, project_dir, abort=None):
        pushes.append((policy, project_dir))
        raise KeyboardInterrupt

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

    assert raised.value is staged
    assert endings == ["unhandled RuntimeError"], f"not closed once: {endings}"
    assert pushes, "the exit push never ran — nothing staged"
    assert driver.limit_policy.snapshots[-1] == \
        "at end (claude: unhandled RuntimeError)"
    out = capsys.readouterr().out
    assert "the exit push is abandoned" in out
    assert NOTE in out and "undelivered operator note" in out
