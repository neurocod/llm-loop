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
                      runlifecycle, stopchannel, streamrender, usage)
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
