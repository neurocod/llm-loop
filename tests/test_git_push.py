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

import os
import subprocess
import sys
import threading
import time

import pytest

from llm_loop import cyclecore, gitpush, parallel, projectroot
from llm_loop.agentwork import ClaudeCommand, Driver
from llm_loop.drivers import ListFileDriver


@pytest.fixture(autouse=True)
def _restore_streams():
    """Both runners tee sys.stdout/stderr into their log and never put them back;
    undo that so one test's tee does not follow the next one."""
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


@pytest.fixture(autouse=True)
def _restore_project_root():
    """Put the process-wide project root back after a run has moved it.

    Every pin here points a runner at a tmp_path, and the stop sentinel and the
    mirror log's file name are both derived from that root. This file collects
    second alphabetically, so without this every later file would inherit a tmp
    root that no longer exists — green today, and the standard seed of an
    order-dependent flake.

    ONE global to restore, not three: the sentinel and the log name used to be
    mirrors with setters of their own, so this fixture was repairing a root that
    had been copied into `stopchannel` and `console` as well. They read it now
    (see `projectroot`), so putting the root back puts everything back.
    """
    previous = projectroot.project_dir()
    yield
    projectroot.set_project_root(previous)


class _FakeGitModule:
    """Stands in for `gitpush.subprocess`, recording (argv, cwd) per call.

    A replacement MODULE rather than a patched `subprocess.run`: the real
    attribute is shared by every module in the process, so patching it would
    also silently rewire the provider launcher and anything else a runner
    reaches for during the same test.

    Every `git` invocation succeeds and `rev-list --count` answers 1, so the
    policy takes each branch that runs git at all rather than short-circuiting
    on "nothing to push".

    `pushed` fires on each `git push`. The parallel pin needs it: its pusher
    runs on a thread of its own, and the only way to know it has taken a turn
    without guessing at a sleep is to wait for the push itself.
    """

    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self):
        self.calls = []
        self.pushed = threading.Event()

    def run(self, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs.get("cwd")))
        if tuple(argv)[:2] == ("git", "push"):
            self.pushed.set()
        return subprocess.CompletedProcess(argv, 0, stdout="1")

    @property
    def dirs(self):
        return {cwd for _, cwd in self.calls}

    @property
    def pushes(self):
        """Only the `git push` calls.

        Asserting on `calls` cannot tell a push from the `rev-list` that decides
        whether to push, so a run that stopped pushing entirely still filled
        `calls` and still passed. Measured: neutering the exit push's `git_push`
        left every pin in this file green.
        """
        return [call for call in self.calls if call[0][:2] == ("git", "push")]


def _elsewhere(tmp_path) -> str:
    """A project root that is provably not where this process stands.

    The assertion is the point: on a machine where pytest happened to run from
    tmp_path, every pin in this file would pass while proving nothing.
    """
    root = os.path.abspath(str(tmp_path))
    assert os.path.normcase(root) != os.path.normcase(os.getcwd()), \
        "the project root and the process cwd are the same directory, so these " \
        "pins cannot tell a handed-over root from an ambient one"
    return root


class _OneShotDriver(Driver):
    """Hands out a single command, then reports the work exhausted."""

    def __init__(self):
        self.served = 0
        self.limit_policy = _StubPolicy()

    def next_command(self):
        if self.served:
            return None
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")


class _StubPolicy:
    """A LimitPolicy that never reads the usage report and never pauses."""

    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="",
                       cache_value=True, should_stop=None):
        return False, session_start


class _MemListDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real provider).

    One item rather than none: a run with nothing pending reports "nothing to
    do" and returns BEFORE its exit push, so an empty list would make this pin
    green without ever reaching the code it is about.
    """

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


def _seq_args(project_dir):
    ns = type("NS", (), {})()
    ns.max = None
    ns.dry_run = False
    ns.raw = False
    ns.start_in = None
    ns.git_push = "after_new_commits"
    ns.project_dir = project_dir
    ns.cost = False
    return ns


def _par_args(project_dir):
    ns = type("NS", (), {})()
    ns.jobs = 1
    ns.max = None
    ns.dry_run = False
    ns.git_push = "after_new_commits"
    ns.project_dir = project_dir
    ns.ignore_usage = True
    return ns


def test_git_runs_where_it_is_told_not_where_the_process_stands(tmp_path, monkeypatch):
    """The policy's own contract: the caller names the repository."""
    fake = _FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    root = _elsewhere(tmp_path)

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
    """
    fake = _FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)
    root = _elsewhere(tmp_path)

    cyclecore.run_loop(_OneShotDriver(), _seq_args(root),
                       app_name="pytest-gitpush")

    assert len(fake.pushes) == 3, \
        f"expected two per-pass pushes and the exit push: {fake.calls}"
    assert fake.dirs == {root}, \
        f"the sequential runner pushed the wrong repository: {fake.calls}"


def test_the_parallel_runner_pushes_the_project_it_was_pointed_at(
        tmp_path, monkeypatch):
    """The same handover from the other runner, on its exit push.

    Its periodic pusher wakes on a 60 s timer, so the exit push is the site a
    test can reach; both read the same `projectroot.project_dir()`. One pending
    item rather than none, because a run with an empty list reports "nothing to
    do" and returns BEFORE the exit push — see `_MemListDriver`.
    """
    fake = _FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    root = _elsewhere(tmp_path)

    try:
        parallel.run_parallel(_MemListDriver(["products/only.md"]),
                              _par_args(root), app_name="pytest-gitpush")
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
    handovers that a normal run cannot reach in a test: the pump asks
    `shared.stop.wait(PUSH_PUMP_INTERVAL_S)` BEFORE pushing, so with the shipped
    minute a run that ends in under a second never enters the body at all — and
    a pin over it was green with `os.getcwd()` substituted.

    The interval is shortened and the worker is held until the pusher has
    actually pushed, which is a handshake rather than a sleep: the run cannot
    finish before the thread under test has taken its turn, and cannot hang if
    it never does.

    `saw_push` is what makes it a pin instead of a formality. The exit push
    would set the same event a moment later, so "a push happened" proves
    nothing; what it records is that a push was seen WHILE a worker was still
    running, which only the pusher thread can produce.
    """
    fake = _FakeGitModule()
    saw_push = []

    def held_job(job_id, command, mailbox=None):
        saw_push.append(fake.pushed.wait(timeout=PUMP_WAIT_S))
        return 0, None, None

    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    monkeypatch.setattr(parallel, "run_job", held_job)
    root = _elsewhere(tmp_path)

    try:
        parallel.run_parallel(_MemListDriver(["products/only.md"]),
                              _par_args(root), app_name="pytest-gitpush")
    except SystemExit:
        pass

    assert saw_push == [True], \
        f"the background pusher never took a turn — the pin proved nothing: {saw_push}"
    assert fake.dirs == {root}, \
        f"the parallel pusher pushed the wrong repository: {fake.calls}"


class _OverlapWatchingGit(_FakeGitModule):
    """`_FakeGitModule` that HOLDS each `git push` and records concurrency.

    `max_live` is the most pushes ever in flight at once. git is not safe to call
    concurrently, so the whole point of the run's push lock is that this number
    is 1; without the lock around the exit push it is 2.

    The first push blocks on `release` instead of sleeping, so the overlap is
    staged rather than raced: the pusher is provably still inside `git push` when
    the main thread reaches the exit push.
    """

    def __init__(self):
        super().__init__()
        self.release = threading.Event()
        self.live = 0
        self.max_live = 0
        self._live_lock = threading.Lock()

    def run(self, argv, **kwargs):
        if tuple(argv)[:2] != ("git", "push"):
            return super().run(argv, **kwargs)
        with self._live_lock:
            self.live += 1
            self.max_live = max(self.max_live, self.live)
        try:
            return super().run(argv, **kwargs)
        finally:
            # A bounded wait: a broken staging must fail the assertions, not
            # hang the suite.
            self.release.wait(timeout=HELD_PUSH_TIMEOUT_S)
            with self._live_lock:
                self.live -= 1


# Upper bound on how long a staged push is held if nothing releases it.
HELD_PUSH_TIMEOUT_S = 10.0
# How long the stager waits, after the first push starts, before releasing it.
# It only has to outlast "a worker returns, two joins, the status area closes",
# all of which are immediate here — and erring long only makes the overlap more
# certain, never less.
HANDOVER_S = 0.5


def test_the_exit_push_never_runs_beside_the_background_pusher(
        tmp_path, monkeypatch):
    """Two `git push`es at once is the thing the run's push lock exists to stop.

    The exit push takes that lock because the join before it CAN TIME OUT: a
    `git push` gets a 300 s subprocess timeout and the join waits
    PUSHER_JOIN_TIMEOUT_S, so "the pusher has been joined" is not a fact the exit
    push may assume. Both constants are shortened here so the timed-out case is
    reachable at all — with the shipped five seconds the pusher always finishes
    first and this pin would be green whatever the exit push did.

    Staged, not raced: the pusher's first push blocks until the stager releases
    it HANDOVER_S later, by which time the main thread is already at the exit
    push. Remove the `with push_lock:` around it and `max_live` reads 2.
    """
    fake = _OverlapWatchingGit()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    monkeypatch.setattr(parallel, "PUSHER_JOIN_TIMEOUT_S", 0.01)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None:
                            (fake.pushed.wait(timeout=PUMP_WAIT_S),
                             (0, None, None))[1])
    root = _elsewhere(tmp_path)

    def stage_the_handover():
        if fake.pushed.wait(timeout=PUMP_WAIT_S):
            time.sleep(HANDOVER_S)
        fake.release.set()

    stager = threading.Thread(target=stage_the_handover, name="stager",
                              daemon=True)
    stager.start()
    try:
        parallel.run_parallel(_MemListDriver(["products/only.md"]),
                              _par_args(root), app_name="pytest-gitpush")
    except SystemExit:
        pass
    finally:
        fake.release.set()
        stager.join(timeout=PUMP_WAIT_S)

    assert fake.max_live >= 1, \
        "no push happened at all, so nothing about concurrency was measured"
    assert fake.max_live == 1, \
        f"two git pushes were in flight at once: {fake.calls}"
