"""The git-push policy, and the repository each runner applies it to.

`gitpush` used to read the runner's own project-root global itself (spelled
`cyclecore.PROJECT_DIR` then; the root is `projectroot` now), so "which
repository" was never passed anywhere and could not be got wrong. Now it is an
argument, and a
handover is exactly the thing that regresses silently: the engine is vendored
under a host project whose root is NOT the process cwd (see
`projectroot.set_project_root`), so a caller that quietly substituted `os.getcwd()`
would still push a repository, still print "git push: done", and still pass any
test that only asked whether git ran.

Every pin here is therefore built on DIVERGENCE — the project root is a tmp_path
that is provably not the directory pytest is standing in — and asserts the
directory git was actually handed, never merely that a push happened.
"""

import _thread
import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from llm_loop import (ctrlc, cyclecore, exitlog, gitpush, operator, ownership,
                      parallel, projectroot, runlifecycle)
from llm_loop.stopchannel import RunStopReason

from _runfixtures import (FakeGitModule, MemListDriver, OneShotDriver,
                          StubPolicy, StubSource, capture_run_context,
                          isolated_run, par_args, record_exit_pushes,
                          root_not_cwd, seq_args)
from llm_loop.agentwork import ClaudeCommand


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    """Every pin here points a real, non-dry runner at a tmp_path: without this
    the run's exit record and tee outlive the test, and every later file
    inherits a tmp project root that no longer exists — green today, and the
    standard seed of an order-dependent flake."""
    with isolated_run(monkeypatch, tmp_path):
        yield


# The runs below push under this policy: every branch of it that runs git at all
# is taken (see `FakeGitModule`), where the fixtures' default `none` takes none.
PUSHING = "after_new_commits"


def test_git_runs_where_it_is_told_not_where_the_process_stands(tmp_path, monkeypatch):
    """The policy's own contract: the caller names the repository."""
    fake = FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    root = root_not_cwd(tmp_path)

    gitpush.maybe_git_push(gitpush.GitPushPolicy.AFTER_NEW_COMMITS, 0.0, root)

    assert [argv[:2] for argv, _ in fake.calls] == [
        ("git", "rev-list"), ("git", "push")]
    assert fake.dirs == {root}, \
        f"the policy ran git somewhere other than the root it was given: {fake.calls}"


def test_the_sequential_runner_pushes_the_project_it_was_pointed_at(
        tmp_path, monkeypatch):
    """`run_loop` must hand `gitpush` its --project-dir, not the process cwd.

    Both of the loop's push sites are covered by one run, and the COUNT is what
    covers them: `maybe_git_push` runs at the top of every PASS through the loop
    — the one that serves the command and the one that finds the queue empty —
    and `final_git_push` once on the way out, so three. Asserting merely "a push
    happened" would leave either site free to disappear; measured, when
    `final_git_push` was extracted: neutering its `git_push` left this file
    green.

    The second pass's check is a handshake too: it is only posted, and a check
    still queued when the run's ending begins stands down for the exit push
    (`push_turn` reads the boundary). So the driver reports the queue empty only
    once that check has pushed — without it, a slow CI worker saw two pushes.
    """
    fake = FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)

    def agent(*args, **kwargs):
        assert fake.pushed.wait(timeout=PUMP_WAIT_S)
        # Re-armed for the second pass, whose check is posted after we return.
        fake.pushed.clear()
        return 0

    class EmptyOnceCheckedDriver(OneShotDriver):
        def next_command(self):
            if self.served:
                assert fake.pushed.wait(timeout=PUMP_WAIT_S), \
                    "the second pass's push check never pushed"
            return super().next_command()

    monkeypatch.setattr(cyclecore, "run_claude_streaming", agent)
    root = root_not_cwd(tmp_path)

    cyclecore.run_loop(EmptyOnceCheckedDriver(), seq_args(root, git_push=PUSHING),
                       app_name="pytest-gitpush")

    assert len(fake.pushes) == 3, \
        f"expected two per-pass pushes and the exit push: {fake.calls}"
    assert fake.dirs == {root}, \
        f"the sequential runner pushed the wrong repository: {fake.calls}"


def test_a_sequential_agent_starts_while_git_push_is_running(tmp_path, monkeypatch):
    # The test body took 0.02 s measured 2026-10-02; the existing 10 s
    # HELD_PUSH_TIMEOUT_S leaves room for slower CI while bounding a regression.
    class HeldPushGit(FakeGitModule):
        def __init__(self):
            super().__init__()
            self.push_started = threading.Event()
            self.release = threading.Event()
            self.push_finished = threading.Event()
            self.threads = set()

        def run(self, argv, **kwargs):
            self.threads.add(threading.current_thread().name)
            if tuple(argv)[:2] == ("git", "push") and not self.push_started.is_set():
                self.push_started.set()
                self.release.wait(timeout=HELD_PUSH_TIMEOUT_S)
                self.push_finished.set()
            return super().run(argv, **kwargs)

    class TwoShotDriver(OneShotDriver):
        def next_command(self):
            if self.served == 2:
                return None
            self.served += 1
            return ClaudeCommand("do the thing", "", f"thing-{self.served}")

    fake = HeldPushGit()
    observed = []

    def agent(*args, **kwargs):
        if not observed:
            observed.append(fake.push_started.wait(timeout=HELD_PUSH_TIMEOUT_S))
        else:
            observed.append(not fake.push_finished.is_set())
            fake.release.set()
        return 0

    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", agent)
    try:
        cyclecore.run_loop(TwoShotDriver(),
                           seq_args(root_not_cwd(tmp_path), git_push=PUSHING),
                           app_name="pytest-gitpush")
    finally:
        fake.release.set()

    assert observed == [True, True], (
        "the next agent waited for the previous turn's git push to finish: "
        f"{observed}")
    assert fake.threads == {"pusher"}, \
        f"git ran on the agent's thread: {fake.threads}"


def test_a_sequential_run_an_exception_ends_runs_its_git_before_it_leaves(
        tmp_path, monkeypatch):
    """An exception out of the loop closes the run down; no git outlives it.

    The run's boundary (`runlifecycle.RunBoundary`) gives the exception the
    epilogue every ending gets: the push in flight finishes, the check queued
    behind it stands down for the exit push (`push_turn` reads the boundary),
    and the exit push runs — all before `run_loop` lets the exception go on.
    None may be left for the owner to start once the run has left, beside
    whatever this process runs next.
    """
    # The test body took 0.02 s measured 2026-10-02; 10 s bounds a broken
    # handshake without making a loaded CI worker fail on normal scheduling.
    class HeldPushGit(FakeGitModule):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if tuple(argv)[:2] == ("git", "push"):
                self.release.wait(timeout=HELD_PUSH_TIMEOUT_S)
            return result

    class TwoShotDriver(OneShotDriver):
        def next_command(self):
            if self.served == 2:
                return None
            self.served += 1
            return ClaudeCommand("do the thing", "", f"thing-{self.served}")

    fake = HeldPushGit()
    made = []
    real_owner = ownership.OwnerThread

    def owner(*args, **kwargs):
        pusher = real_owner(*args, **kwargs)
        made.append(pusher)
        return pusher

    calls = []

    def agent(*args, **kwargs):
        calls.append("agent")
        if len(calls) == 1:
            assert fake.pushed.wait(timeout=HELD_PUSH_TIMEOUT_S)
            return 0
        assert made[0].backlog == 2, "the second push check was not queued"
        raise RuntimeError("agent failed")

    real_close_run = runlifecycle.close_run

    def close_run_releasing_the_push(*args, **kwargs):
        # The held push is let go only once the ending has begun, so the
        # check queued behind it runs after that — let go from the agent, it
        # could run before the exception reached the boundary.
        fake.release.set()
        return real_close_run(*args, **kwargs)

    monkeypatch.setattr(runlifecycle, "close_run", close_run_releasing_the_push)
    monkeypatch.setattr(cyclecore, "OwnerThread", owner)
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", agent)
    try:
        with pytest.raises(RuntimeError, match="agent failed"):
            cyclecore.run_loop(TwoShotDriver(),
                               seq_args(root_not_cwd(tmp_path), git_push=PUSHING),
                               app_name="pytest-gitpush")
    finally:
        fake.release.set()
    calls_at_exit = list(fake.calls)

    assert made[0].close(timeout=HELD_PUSH_TIMEOUT_S)
    assert fake.calls == calls_at_exit, \
        f"git started after the run had left: {fake.calls[len(calls_at_exit):]}"
    # The held push and the exit push; the check queued between them is the
    # exit push's to make, and run it would be one more `git push` waited for.
    assert len(fake.pushes) == 2, \
        f"the exception's epilogue did not push exactly once more: {fake.calls}"


def test_the_parallel_runner_pushes_the_project_it_was_pointed_at(
        tmp_path, monkeypatch):
    """The same handover from the other runner, on its exit push.

    Its periodic pusher wakes on a 60 s timer, so the exit push is the site a
    test can reach; both read the same `projectroot.project_dir()`. One pending
    item rather than none, because a run with an empty list reports "nothing to
    do" and returns BEFORE the exit push — see `MemListDriver`.
    """
    fake = FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    root = root_not_cwd(tmp_path)

    try:
        parallel.run_parallel(MemListDriver(["products/only.md"]),
                              par_args(root, jobs=1, git_push=PUSHING),
                              app_name="pytest-gitpush")
    except SystemExit:
        pass

    # Exactly one, and it is the exit push: the periodic pusher's first turn is
    # a whole PUSH_PUMP_INTERVAL_S (60 s) away, so nothing else can have pushed.
    assert len(fake.pushes) == 1, \
        f"the exit push did not happen exactly once: {fake.calls}"
    assert fake.dirs == {root}, \
        f"the parallel runner pushed the wrong repository: {fake.calls}"


# How long the pusher pin gives the background thread to take one turn. Only the
# FAILING case ever approaches it: a healthy pump wakes every
# PUMP_INTERVAL_S (0.01 below) and the worker is released the moment it pushes.
PUMP_WAIT_S = 10.0


def test_the_parallel_pusher_pushes_the_project_it_was_pointed_at(
        tmp_path, monkeypatch):
    """The periodic pusher's handover — the site the exit push does not cover.

    This is where a long run's pushing actually happens; the exit push only
    mops up what the last interval left. It is also the only one of the four
    handovers that a normal run cannot reach in a test: the pump's first turn
    only arms its PUSH_PUMP_INTERVAL_S clock, so with the shipped minute a run
    that ends in under a second never reaches its push at all — and a pin over
    it was green with `os.getcwd()` substituted.

    The interval is shortened and the worker is held until the pusher has
    actually pushed, which is a handshake rather than a sleep: the run cannot
    finish before the thread under test has taken its turn, and cannot hang if
    it never does.

    `saw_push` is what makes it a pin instead of a formality. The exit push
    would set the same event a moment later, so "a push happened" proves
    nothing; what it records is that a push was seen WHILE a worker was still
    running, which only the pusher thread can produce.
    """
    fake = FakeGitModule()
    saw_push = []

    def held_job(job_id, command, mailbox=None):
        saw_push.append(fake.pushed.wait(timeout=PUMP_WAIT_S))
        return 0, None, None

    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    monkeypatch.setattr(parallel, "run_job", held_job)
    root = root_not_cwd(tmp_path)

    try:
        parallel.run_parallel(MemListDriver(["products/only.md"]),
                              par_args(root, jobs=1, git_push=PUSHING),
                              app_name="pytest-gitpush")
    except SystemExit:
        pass

    assert saw_push == [True], \
        f"the background pusher never took a turn — the pin proved nothing: {saw_push}"
    assert fake.dirs == {root}, \
        f"the parallel pusher pushed the wrong repository: {fake.calls}"


def test_the_git_push_knob_is_live_in_a_parallel_run(tmp_path, monkeypatch):
    """`--git-push` is an editable knob in BOTH runners, not only the sequential one.

    It was frozen here: the policy was read into a local at startup and captured
    by the pusher's closure, so a fleet run offered the flag on the command line
    and then ignored every edit of it — a run launched `--git-push none` could
    not be told to start pushing by any edit of the knob. The
    sequential runner had had the knob for as long as RunSettings existed, and
    nothing said the two disagreed.

    Launched with `none` on purpose: it is the value that used to make the run
    skip creating a pusher thread at all, so a run that pushes ANYTHING here is
    a run that read the policy after the edit rather than before it.
    """
    fake = FakeGitModule()
    made = capture_run_context(monkeypatch)
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    saw_push = []

    def edit_the_knob_then_wait(job_id, command, mailbox=None):
        # The run's own knob registry, whose setters write its RunSettings.
        made["ctx"].registry.get(gitpush.GIT_PUSH_SETTING).set("after_new_commits")
        saw_push.append(fake.pushed.wait(timeout=PUMP_WAIT_S))
        return 0, None, None

    monkeypatch.setattr(parallel, "run_job", edit_the_knob_then_wait)
    root = root_not_cwd(tmp_path)
    args = par_args(root, jobs=1, git_push="none")

    try:
        parallel.run_parallel(MemListDriver(["products/only.md"]), args,
                              app_name="pytest-gitpush")
    except SystemExit:
        pass

    assert saw_push == [True], (
        "the run never pushed after the policy was edited — the parallel "
        "runner is still reading a frozen copy of --git-push")
    assert fake.dirs == {root}, \
        f"the parallel runner pushed the wrong repository: {fake.calls}"


class _OverlapWatchingGit(FakeGitModule):
    """`FakeGitModule` that HOLDS each `git push` and records who ran what.

    `threads` is the name of the thread each git call ran on. The run's git has
    one owner (the `pusher`), so the whole of this pin's claim is that this set
    is `{"pusher"}`: one thread cannot run two pushes at once. An exit push
    made anywhere else — the main thread, as it was before the owner — puts a
    second name here whether or not the staging below got the timing right.

    `max_live` is the most pushes ever in flight at once, and `beside_a_push`
    every git command — `rev-list` included — that started while a push was in
    flight: the exit push is the WHOLE of `final_git_push`, so the count it reads
    to decide "is there anything to push?" must not run beside a push either;
    that is how "nothing to push" could be printed about a repository that was
    being pushed at that moment, and `max_live` cannot see it — a lone
    `rev-list` is not a second push. They are what an exclusion that is not by
    thread (a lock, once) would have to answer for.

    A push blocks on `release` instead of sleeping, so the overlap is staged
    rather than raced; `held` is whether each held push was released by
    `release` rather than by the timeout.

    Every push after the first starts LATE_EXIT_PUSH_S late. The pin's second
    push is the exit push, and the delay is what makes "close_run returned
    before it" observable: a close_run that does not wait would otherwise
    usually lose the race to a push that costs nothing, and pass.
    """

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.live = 0
        self.max_live = 0
        self.beside_a_push = []
        self.threads = set()
        self.held = []
        self._live_lock = threading.Lock()

    def run(self, argv, **kwargs):
        with self._live_lock:
            self.threads.add(threading.current_thread().name)
            if self.live:
                self.beside_a_push.append(tuple(argv)[:2])
        if tuple(argv)[:2] != ("git", "push"):
            return super().run(argv, **kwargs)
        if self.pushed.is_set():
            time.sleep(LATE_EXIT_PUSH_S)
        with self._live_lock:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        try:
            return super().run(argv, **kwargs)
        finally:
            # A bounded wait: a broken staging must fail the assertions, not
            # hang the suite.
            self.held.append(self.release.wait(timeout=HELD_PUSH_TIMEOUT_S))
            with self._live_lock:
                self.live -= 1


# Upper bound on how long a staged push is held if nothing releases it.
HELD_PUSH_TIMEOUT_S = 10.0

# How late `_OverlapWatchingGit` starts every push after the first. Only the
# failing case depends on it (a close_run returning before the exit push has
# run); a passing run merely waits it out once.
LATE_EXIT_PUSH_S = 0.2


def _signal_on_close_run(monkeypatch, event: threading.Event,
                         on_return=None) -> None:
    """Set `event` the moment the run enters its epilogue's housekeeping, and
    call `on_return` (if given) the moment that housekeeping returns.

    Wrapped on `runlifecycle`, where both doors resolve the name: `end_run`
    calls `close_run` through the module, and the interrupt calls it directly.
    """
    real_close_run = runlifecycle.close_run

    def close_run(*args, **kwargs):
        event.set()
        result = real_close_run(*args, **kwargs)
        if on_return is not None:
            on_return()
        return result

    monkeypatch.setattr(runlifecycle, "close_run", close_run)


def test_every_git_call_of_a_parallel_run_is_made_by_the_pusher(
        tmp_path, monkeypatch):
    """Two `git push`es at once is the thing the run's pusher exists to stop.

    The exit push is handed to the pusher (`runlifecycle.close_run`) rather
    than made beside it, because a `git push` gets a 300 s subprocess timeout
    and the run must not race a second `git` against one still in flight. That
    case is STAGED here, not hoped for: the pusher's periodic push is held until
    the run has entered `close_run`, so the exit push is asked for while the
    pusher is provably inside `git push`. A handshake, not a sleep — an earlier
    version of this pin slept 0.5 s, which on a loaded box degrades to a silent
    vacuous pass.

    Make the exit push on the calling thread instead and `threads` names
    `MainThread`, whatever the timing did. And `close_run` must WAIT for the
    exit push, not merely hand it over: the process may exit the moment it
    returns, and the pusher is a daemon.
    """
    fake = _OverlapWatchingGit()
    exit_asked = fake.release
    pushes_when_close_run_returned = []
    _signal_on_close_run(
        monkeypatch, exit_asked,
        on_return=lambda: pushes_when_close_run_returned.append(
            len(fake.pushes)))
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None:
                            (fake.pushed.wait(timeout=PUMP_WAIT_S),
                             (0, None, None))[1])
    root = root_not_cwd(tmp_path)

    try:
        parallel.run_parallel(MemListDriver(["products/only.md"]),
                              par_args(root, jobs=1, git_push=PUSHING),
                              app_name="pytest-gitpush")
    except SystemExit:
        pass
    finally:
        fake.release.set()

    assert fake.held[:1] == [True], (
        "the pusher's push was not still in flight when the run asked for its "
        f"exit push, so this run measured nothing about the two: {fake.held}")
    assert len(fake.pushes) >= 2, \
        f"the periodic push and the exit push did not both happen: {fake.calls}"
    # The whole count, not 2: a periodic turn may slip in between the held
    # push's release and the close, and the exit push then comes third.
    assert pushes_when_close_run_returned == [len(fake.pushes)], (
        "close_run returned before the exit push had run — it handed the push "
        f"over without waiting for it: {pushes_when_close_run_returned}")
    assert fake.threads == {"pusher"}, (
        f"git ran on a thread other than the run's pusher: {fake.threads}")
    assert fake.max_live == 1, \
        f"two git pushes were in flight at once: {fake.calls}"
    # The SCOPE, not just the push: `git_unpushed_count` is part of the exit
    # push too, so no git call of any kind may start while a push is in flight.
    assert fake.beside_a_push == [], (
        "a git command ran while a push was in flight, so the exit push was not "
        f"made whole behind the pusher's: {fake.beside_a_push}")


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
    def exit_push_fails(policy, project_dir, abort=None):
        raise OSError("staged: the repository is not reachable")

    monkeypatch.setattr(runlifecycle, "final_git_push", exit_push_fails)
    monkeypatch.setattr(runlifecycle, "usage_source_for",
                        lambda provider: StubSource())
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    box = _undelivered_note(monkeypatch)
    driver = MemListDriver(["products/only.md"])

    result = parallel.run_parallel(driver,
                                   par_args(root_not_cwd(tmp_path), jobs=1,
                                            git_push=PUSHING,
                                            ignore_usage=False,
                                            no_statusline=True),
                                   app_name="pytest-gitpush")

    assert result.reason == RunStopReason.NO_WORK
    assert driver.limit_policy.snapshots[-1] == "at end (parallel claude)", (
        f"the closing usage snapshot was lost: {driver.limit_policy.snapshots}")
    captured = capsys.readouterr()
    assert "undelivered operator note" in captured.out and NOTE in captured.out, \
        "the note nobody delivered was never reported"
    assert box.submitted, "the staged note was never typed"
    assert "the exit push failed" in captured.err
    assert "Traceback" in captured.err, "the failed exit push lost its traceback"
    assert "OSError: staged: the repository is not reachable" in captured.err
    exitlog.finish()
    assert "=== run ended: no more work in the queue" in capsys.readouterr().out


# What the operator typed and nobody delivered (see `_undelivered_note`).
NOTE = "please look at the third file"


class _NoteBox:
    """The run's one mailbox, with NOTE typed after the workers have ended.

    Typed then, not before the run: a note in the mailbox while a worker still
    claims is spliced into that worker's prompt, and is delivered after all.
    """

    def __init__(self):
        self.box = operator.Mailbox()
        self.submitted = False

    def type_note(self):
        self.box.submit(NOTE)
        self.submitted = True


def _undelivered_note(monkeypatch) -> _NoteBox:
    """Give the run one mailbox holding NOTE once `join_workers` has returned."""
    note = _NoteBox()
    monkeypatch.setattr(operator, "Mailbox", lambda: note.box)
    real_join = parallel.join_workers

    def join_then_type(threads):
        real_join(threads)
        note.type_note()

    monkeypatch.setattr(parallel, "join_workers", join_then_type)
    return note


# How long the Ctrl+C pin lets the main thread settle into its wait before it
# interrupts it, and afterwards lets a late interrupt land. Only the failing
# cases depend on it (an unbounded wait, a close_run that does not wait).
CTRL_C_SETTLE_S = 0.1


class _HeldCountGit(FakeGitModule):
    """`FakeGitModule` whose `rev-list` — the exit push's first git call —
    waits for `release`; `counting` says it has started."""

    def __init__(self):
        super().__init__()
        self.counting = threading.Event()
        self.release = threading.Event()

    def run(self, argv, **kwargs):
        if tuple(argv)[:2] == ("git", "rev-list"):
            self.counting.set()
            self.release.wait(timeout=HELD_PUSH_TIMEOUT_S)
        return super().run(argv, **kwargs)


def test_aborting_a_periodic_count_starts_no_push(tmp_path, monkeypatch):
    # Under 0.005 s measured 2026-10-02; the existing 10 s bound leaves room
    # for CI scheduling while still failing a lost release.
    fake = _HeldCountGit()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    abort = gitpush.PushAbort()
    pusher = threading.Thread(
        target=gitpush.maybe_git_push,
        args=(gitpush.GitPushPolicy.AFTER_NEW_COMMITS, 0.0,
              root_not_cwd(tmp_path)), kwargs={"abort": abort}, daemon=True)
    pusher.start()
    try:
        assert fake.counting.wait(timeout=HELD_PUSH_TIMEOUT_S)
        abort.set()
    finally:
        fake.release.set()
    pusher.join(timeout=HELD_PUSH_TIMEOUT_S)

    assert not pusher.is_alive()
    assert fake.pushes == [], \
        f"a push started after the periodic check was abandoned: {fake.calls}"


def test_aborted_git_wait_does_not_wait_for_another_runs_push():
    # 0.26 s measured 2026-10-02, including one 0.25 s lock poll; the
    # existing 10 s timeout leaves ample room for slower CI scheduling.
    abort = gitpush.PushAbort()
    waiting = threading.Event()
    result = []

    def call_git():
        waiting.set()
        result.append(gitpush._run_git(["git", "rev-list"], ".", 30, abort))

    lock = gitpush._GIT_CALL_LOCK
    lock.acquire()
    worker = threading.Thread(target=call_git, daemon=True)
    try:
        worker.start()
        assert waiting.wait(timeout=HELD_PUSH_TIMEOUT_S)
        abort.set()
        worker.join(timeout=HELD_PUSH_TIMEOUT_S)
        assert not worker.is_alive(), "aborted git kept waiting for another run"
    finally:
        lock.release()
        worker.join(timeout=HELD_PUSH_TIMEOUT_S)

    assert result == [None]


def test_exit_interrupt_cancels_queued_periodic_push(tmp_path, monkeypatch):
    # Under 0.005 s measured 2026-10-02; the 10 s handshake bound fails a
    # stuck pusher without requiring a particular scheduler interleaving.
    class HeldPushGit(FakeGitModule):
        def __init__(self):
            super().__init__()
            self.release = threading.Event()

        def run(self, argv, **kwargs):
            result = super().run(argv, **kwargs)
            if tuple(argv)[:2] == ("git", "push") and len(self.pushes) == 1:
                self.release.wait(timeout=HELD_PUSH_TIMEOUT_S)
            return result

    fake = HeldPushGit()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    projectroot.set_project_root(root_not_cwd(tmp_path))
    abort = gitpush.PushAbort()
    pusher = ownership.OwnerThread("pusher", maxsize=1).start()
    policy = gitpush.GitPushPolicy.AFTER_NEW_COMMITS
    pusher.post(gitpush.maybe_git_push, policy, 0.0,
                projectroot.project_dir(), timeout=0)
    assert fake.pushed.wait(timeout=HELD_PUSH_TIMEOUT_S)
    assert pusher.try_post(lambda: gitpush.maybe_git_push(
        policy, 0.0, projectroot.project_dir(), abort=abort))

    def interrupted_wait(owner, final, deadline_s=None, abandoned=None):
        # What the wait answers once the operator's Ctrl+C has given it up.
        owner.close(timeout=0, final=final)
        return runlifecycle.PushWait.ABANDONED

    monkeypatch.setattr(runlifecycle, "_wait_for_exit_push", interrupted_wait)
    closed = []

    class ClosingUsage:
        source = StubSource()
        name = "claude"

        def close(self, ending):
            fake.release.set()
            closed.append(pusher.close(timeout=HELD_PUSH_TIMEOUT_S))

    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=False, progress=None,
        settings=runlifecycle.RunSettings(git_push=policy),
        registry=None, status_enabled=False)
    try:
        runlifecycle.close_run(ctx, usages=[ClosingUsage()], pusher=pusher,
                               push_abort=abort)
    finally:
        fake.release.set()
        pusher.close(timeout=HELD_PUSH_TIMEOUT_S)

    assert closed == [True]
    assert len(fake.pushes) == 1, \
        f"a periodic push started after Ctrl+C abandoned it: {fake.calls}"


def test_ctrl_c_while_the_exit_push_is_waited_for_gives_up_the_push_only(
        tmp_path, monkeypatch, capsys):
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
    """
    fake = _HeldCountGit()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    projectroot.set_project_root(root_not_cwd(tmp_path))
    policy = StubPolicy()
    usage = runlifecycle.RunUsage(StubSource(), policy, "parallel claude")
    mailbox = operator.Mailbox()
    mailbox.submit(NOTE)
    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=False, progress=None,
        settings=runlifecycle.RunSettings(
            git_push=gitpush.GitPushPolicy(PUSHING)),
        registry=None, status_enabled=False)
    pusher = ownership.OwnerThread("pusher").start()

    def ctrl_c_once_the_exit_push_runs():
        if fake.counting.wait(timeout=HELD_PUSH_TIMEOUT_S):
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
                                       ending="interrupted", mailbox=mailbox,
                                       pusher=pusher)
                returned_after = interrupt.presses
                # A close_run that did not wait returns before the Ctrl+C is
                # sent; it is then heard here, still inside the capture.
                interrupter.join(timeout=HELD_PUSH_TIMEOUT_S)
                time.sleep(CTRL_C_SETTLE_S)
            except KeyboardInterrupt:
                raised.append("a bare KeyboardInterrupt")
    finally:
        fake.release.set()

    assert raised == [], "Ctrl+C inside a run was raised, not pressed"
    assert returned_after == 1, (
        f"close_run returned before the Ctrl+C was heard: {returned_after}")
    assert pusher.close(timeout=HELD_PUSH_TIMEOUT_S), \
        "the abandoned exit push never finished"
    assert fake.pushes == [], (
        f"the abandoned exit push still ran `git push`: {fake.calls}")
    assert policy.snapshots == ["at end (parallel claude: interrupted)"], (
        f"Ctrl+C cost the closing usage snapshot: {policy.snapshots}")
    out = capsys.readouterr().out
    assert "the exit push is abandoned" in out
    assert "final git push on exit" not in out, \
        "the abandoned exit push still announced a push it was not going to make"
    assert "undelivered operator note" in out and NOTE in out, \
        "Ctrl+C cost the report of the notes nobody delivered"


def test_ctrl_c_during_the_exit_push_announcement_starts_no_push(
        tmp_path, monkeypatch):
    """An abort heard while the push prints its announcement stops the push.

    The announcement sits between the count and the push and can block on a
    stalled console; the abort was asked before it, so a Ctrl+C heard during
    it still let `git push` start. Staged: the print is held, the abort set,
    the print let go.
    """
    fake = FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    announcing = threading.Event()
    let_print = threading.Event()

    def held_print(*args, **kwargs):
        if args and "final git push on exit" in str(args[0]):
            announcing.set()
            let_print.wait(timeout=HELD_PUSH_TIMEOUT_S)

    monkeypatch.setattr(gitpush, "print", held_print, raising=False)
    abort = gitpush.PushAbort()
    root = root_not_cwd(tmp_path)
    pusher = threading.Thread(
        target=gitpush.final_git_push,
        args=(gitpush.GitPushPolicy(PUSHING), root), kwargs={"abort": abort},
        name="pusher", daemon=True)
    pusher.start()
    try:
        assert announcing.wait(timeout=HELD_PUSH_TIMEOUT_S), \
            "the exit push never announced itself — nothing staged"
        abort.set()
    finally:
        let_print.set()
    pusher.join(timeout=HELD_PUSH_TIMEOUT_S)

    assert not pusher.is_alive(), "the exit push never returned"
    assert [argv[:2] for argv, _ in fake.calls] == [("git", "rev-list")], (
        f"git started after the push was abandoned: {fake.calls}")


def test_an_abort_waits_out_a_git_start_in_progress():
    """`PushAbort.set` returns only once no start can follow it.

    "Is it set? then start" is two steps; a `set` between them would return
    while a child was still about to start. The start is held in the middle
    here, and `set` must not return until it is over.
    """
    abort = gitpush.PushAbort()
    spawning = threading.Event()
    let_spawn = threading.Event()
    order = []

    def spawn():
        spawning.set()
        let_spawn.wait(timeout=HELD_PUSH_TIMEOUT_S)
        order.append("child started")
        return "the child"

    starter = threading.Thread(target=abort.start, args=(spawn,), daemon=True)
    setter = threading.Thread(
        target=lambda: (abort.set(), order.append("set returned")), daemon=True)
    starter.start()
    try:
        assert spawning.wait(timeout=HELD_PUSH_TIMEOUT_S), "the start never ran"
        setter.start()
        # A `set` that does not wait returns inside this; one that does is
        # still waiting at its end. Only the failing case depends on it.
        setter.join(timeout=CTRL_C_SETTLE_S)
    finally:
        let_spawn.set()
    starter.join(timeout=HELD_PUSH_TIMEOUT_S)
    setter.join(timeout=HELD_PUSH_TIMEOUT_S)

    assert order == ["child started", "set returned"], (
        f"set returned while a git start was still in progress: {order}")
    assert abort.start(lambda: pytest.fail("a start after set ran")) is None


class _TimingOutProcess:
    """A git child whose `communicate(timeout=...)` times out; records the rest."""

    def __init__(self, argv):
        self.argv = argv
        self.returncode = None
        self.steps = []

    def communicate(self, timeout=None):
        if timeout is not None:
            self.steps.append("communicate(timeout)")
            raise subprocess.TimeoutExpired(self.argv, timeout)
        self.steps.append("communicate")
        return "", None

    def kill(self):
        self.steps.append("kill")

    def wait(self, timeout=None):
        self.steps.append("wait")
        return -9

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("windows, reaped_by", [(False, "wait"),
                                                (True, "communicate")],
                         ids=["posix", "windows"])
def test_a_timed_out_git_is_reaped_the_way_subprocess_run_does(
        monkeypatch, windows, reaped_by):
    """After the kill: POSIX waits for the child, Windows drains the pipe.

    A second `communicate()` on POSIX waits for the pipe's EOF, and a hook or
    credential helper that inherited git's stdout holds that open past the
    kill — the 30/300 s timeout then bounds nothing. Both branches are pinned
    on every host through a stand-in process; the real descendant is staged
    below, where the platform allows it.
    """
    made = []

    def start_timing_out(argv, **kwargs):
        made.append(_TimingOutProcess(argv))
        return made[-1]

    monkeypatch.setattr(gitpush, "subprocess",
                        FakeGitModule(popen=start_timing_out))
    monkeypatch.setattr(gitpush, "_WINDOWS", windows)

    with pytest.raises(subprocess.TimeoutExpired):
        gitpush._run_git(["git", "push"], ".", 300, gitpush.PushAbort())

    [proc] = made
    assert proc.steps == ["communicate(timeout)", "kill", reaped_by], (
        f"the killed git was not reaped as subprocess.run reaps it: {proc.steps}")


# The descendant pin's timeout, and how long it lets `_run_git` take to come
# back from it. NOT measured: no POSIX host among the project's machines
# (2026-09-29), and on Windows the pin is skipped. Wide by construction
# instead: the child needs one interpreter start to spawn its descendant
# before GIT_TIMEOUT_S, and the healthy return after it is a kill and a
# wait; only the failing case (the unbounded read) waits out RETURN_BOUND_S,
# and DESCENDANT_SLEEP_S keeps that case hanging past it.
GIT_TIMEOUT_S = 5.0
RETURN_BOUND_S = 60.0
DESCENDANT_SLEEP_S = 300

_HOLDS_THE_PIPE = (
    "import subprocess, sys, time\n"
    "held = subprocess.Popen([sys.executable, '-c',\n"
    "                         'import time; time.sleep(%d)'])\n"
    "with open(sys.argv[1], 'w') as f:\n"
    "    f.write(str(held.pid))\n"
    "time.sleep(%d)\n" % (DESCENDANT_SLEEP_S, DESCENDANT_SLEEP_S))


@pytest.mark.skipif(sys.platform == "win32", reason=(
    "Windows drains the pipe after the kill, as subprocess.run does there: a "
    "descendant holding it holds the reap too, and there is no bound to pin"))
def test_a_timed_out_git_returns_although_a_descendant_holds_its_stdout(
        tmp_path):
    """The kill ends git, not the hook git started: its end must not be awaited."""
    pid_file = tmp_path / "descendant.pid"
    argv = [sys.executable, "-c", _HOLDS_THE_PIPE, str(pid_file)]
    outcome = []

    def run():
        try:
            gitpush._run_git(argv, str(tmp_path), GIT_TIMEOUT_S,
                             gitpush.PushAbort())
        except subprocess.TimeoutExpired:
            outcome.append("timed out")

    caller = threading.Thread(target=run, daemon=True)
    caller.start()
    try:
        caller.join(timeout=GIT_TIMEOUT_S + RETURN_BOUND_S)
        returned = not caller.is_alive()
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass

    assert pid_file.exists(), (
        "the descendant never started before the timeout — nothing staged")
    assert returned, ("the timed-out git was reaped by waiting for a pipe its "
                      "descendant still holds")
    assert outcome == ["timed out"]


# How long the Ctrl+C pin below stretches the run's wind-down (the console's
# close) so that a pump still running would push in it: at a 0.01 s interval,
# dozens of turns. Only the failing case depends on it.
WIND_DOWN_S = 0.5


def test_no_periodic_push_starts_once_the_operator_pressed_ctrl_c(
        tmp_path, monkeypatch):
    """After Ctrl+C the pump stops; only the exit push is still to come.

    The wind-down after an interrupt — the workers re-joined, the console
    closed — can take a while, and a periodic push begun in it is one more
    `git push` for the exit push to queue behind while the operator waits.
    One turn may already be under way when the interrupt lands; none may start
    after it.
    """
    fake = FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    record_exit_pushes(monkeypatch)
    at_interrupt = []

    def interrupt_once_the_pump_pushed(threads):
        for t in threads:
            t.join()
        at_interrupt.append(fake.pushed.wait(timeout=PUMP_WAIT_S))
        at_interrupt.append(len(fake.pushes))
        ctrlc.current().press()

    real_close_console = parallel._close_console

    def slow_wind_down():
        time.sleep(WIND_DOWN_S)
        real_close_console()

    monkeypatch.setattr(parallel, "join_workers", interrupt_once_the_pump_pushed)
    monkeypatch.setattr(parallel, "_close_console", slow_wind_down)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))

    with pytest.raises(SystemExit):
        parallel.run_parallel(MemListDriver(["products/only.md"]),
                              par_args(root_not_cwd(tmp_path), jobs=1,
                                       git_push=PUSHING, no_statusline=True),
                              app_name="pytest-gitpush")

    pumped, before = at_interrupt
    assert pumped, "the pump never pushed before the interrupt — nothing measured"
    assert len(fake.pushes) - before <= 1, (
        f"the pump went on pushing after Ctrl+C: {before} push(es) before it, "
        f"{len(fake.pushes)} in all")


class _HeldGitLock:
    """Stands in for `gitpush._GIT_CALL_LOCK`, held by "an earlier run".

    `hold` takes it the way a git call an abandoned run left running holds
    it; `contended` fires once the pusher asks for it while it is held.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.holding = False
        self.contended = threading.Event()

    def acquire(self, blocking=True, timeout=-1):
        if self.holding and threading.current_thread().name == "pusher":
            self.contended.set()
        return self._lock.acquire(blocking, timeout)

    def release(self):
        self._lock.release()

    def hold(self):
        self._lock.acquire()
        self.holding = True

    def let_go(self):
        if self.holding:
            self.holding = False
            self._lock.release()


def test_a_periodic_check_behind_another_run_s_git_starts_none_after_the_run(
        tmp_path, monkeypatch):
    """An abandoned exit push cancels the periodic check waiting ahead of it.

    The pump's checks ran without the run's `PushAbort`, so one waiting for
    the process-wide git lock — held by a git call an earlier, abandoned run
    left running — waited unbounded, and once that call let go it started
    `rev-list` and `git push` after this run had passed its exit push's
    deadline and left. Staged: the lock is held while the pump waits for it,
    the region raises, the run leaves past its (shortened) deadline, and only
    then is the lock let go; no git may start after that.
    """
    fake = FakeGitModule()
    lock = _HeldGitLock()
    owners = []
    real_owner = ownership.OwnerThread

    def recording_owner(name, **kwargs):
        owner = real_owner(name, **kwargs)
        if name == "pusher":
            owners.append(owner)
        return owner

    calls_at_hold = []

    def fail_while_the_pump_waits(threads):
        for t in threads:
            t.join()
        lock.hold()
        calls_at_hold.append(len(fake.calls))
        assert lock.contended.wait(timeout=PUMP_WAIT_S), \
            "the pump never asked for git — nothing staged"
        raise RuntimeError("staged: the join broke")

    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(gitpush, "_GIT_CALL_LOCK", lock)
    monkeypatch.setattr(ownership, "OwnerThread", recording_owner)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    monkeypatch.setattr(runlifecycle, "UNWIND_PUSH_DEADLINE_S", 0.5)
    monkeypatch.setattr(parallel, "join_workers", fail_while_the_pump_waits)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))

    try:
        with pytest.raises(RuntimeError, match="staged"):
            parallel.run_parallel(MemListDriver(["products/only.md"]),
                                  par_args(root_not_cwd(tmp_path), jobs=1,
                                           git_push=PUSHING, no_statusline=True),
                                  app_name="pytest-gitpush")
    finally:
        lock.let_go()

    [pusher] = owners
    assert pusher.close(timeout=HELD_PUSH_TIMEOUT_S), \
        "the pusher never finished once the git lock was let go"
    [before] = calls_at_hold
    assert fake.calls[before:] == [], (
        f"git started after the run had abandoned its push: "
        f"{fake.calls[before:]}")


def test_a_run_that_unwinds_past_its_epilogue_closes_its_pusher(
        tmp_path, monkeypatch):
    """An exception after the fleet must not leave the run's pusher pushing.

    `close_run` closes the pusher, and the run's boundary
    (`runlifecycle.RunBoundary`) brings this ending — the closing report's
    `pending_total` raising — there too. It used to leave the pusher open,
    pushing every minute for the rest of the process, beside the next run's
    pusher under a batching wrapper.
    """
    owners = []
    real_owner = ownership.OwnerThread

    def recording_owner(name, **kwargs):
        owner = real_owner(name, **kwargs)
        if name == "pusher":
            owners.append(owner)
        return owner

    class _ReportFails(MemListDriver):
        fleet_done = False

        def pending_total(self):
            if self.fleet_done:
                raise RuntimeError("staged: the list file vanished")
            return super().pending_total()

    driver = _ReportFails(["products/only.md"])
    real_join = parallel.join_workers

    def join_then_break_the_list(threads):
        real_join(threads)
        driver.fleet_done = True

    monkeypatch.setattr(ownership, "OwnerThread", recording_owner)
    monkeypatch.setattr(parallel, "join_workers", join_then_break_the_list)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    record_exit_pushes(monkeypatch)

    with pytest.raises(RuntimeError, match="staged: the list file vanished"):
        parallel.run_parallel(driver,
                              par_args(root_not_cwd(tmp_path), jobs=1,
                                       no_statusline=True),
                              app_name="pytest-gitpush")

    [pusher] = owners
    ran_on = []
    try:
        # A closed owner runs a post on its caller, at once; an open one
        # queues it for its own thread.
        pusher.post(lambda: ran_on.append(threading.current_thread().name))
        assert ran_on == [threading.current_thread().name], (
            "the run unwound and left its pusher open")
    finally:
        pusher.close(timeout=HELD_PUSH_TIMEOUT_S)
