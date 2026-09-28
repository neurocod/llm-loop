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

import subprocess
import threading

import pytest

from llm_loop import cyclecore, gitpush, parallel, runlifecycle, statusline
from llm_loop.stopchannel import RunStopReason

from _runfixtures import (MemListDriver, OneShotDriver, isolated_run, par_args,
                          root_not_cwd, seq_args)


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    """Every pin here points a real, non-dry runner at a tmp_path: without this
    the run's exit record and tee outlive the test, and every later file
    inherits a tmp project root that no longer exists — green today, and the
    standard seed of an order-dependent flake."""
    with isolated_run(monkeypatch, tmp_path):
        yield


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


# The runs below push under this policy: every branch of it that runs git at all
# is taken (see `_FakeGitModule`), where the fixtures' default `none` takes none.
PUSHING = "after_new_commits"


def test_git_runs_where_it_is_told_not_where_the_process_stands(tmp_path, monkeypatch):
    """The policy's own contract: the caller names the repository."""
    fake = _FakeGitModule()
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
    """
    fake = _FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)
    root = root_not_cwd(tmp_path)

    cyclecore.run_loop(OneShotDriver(), seq_args(root, git_push=PUSHING),
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
    do" and returns BEFORE the exit push — see `MemListDriver`.
    """
    fake = _FakeGitModule()
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
    fake = _FakeGitModule()
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


def _capture_status_app(monkeypatch) -> dict:
    """Hand the test the run's own StatusApp, and with it the knob registry.

    Through the seam a run already has — `app.registry` is the registry the
    status app is handed, i.e. `ctx.registry` — rather than by replacing
    `runlifecycle.knob_registry`, so the pin edits the very registry whose
    setters write the run's RunSettings. `test_parallel_statusline` reaches it
    the same way.
    """
    made = {}
    real_app_class = statusline.StatusApp        # captured before the patch

    def _app(**kwargs):
        made["app"] = real_app_class(**kwargs)
        return made["app"]

    monkeypatch.setattr(statusline, "StatusApp", _app)
    return made


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
    fake = _FakeGitModule()
    made = _capture_status_app(monkeypatch)
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    saw_push = []

    def edit_the_knob_then_wait(job_id, command, mailbox=None):
        # The run's own knob registry (`ctx.registry`, handed to the app).
        made["app"].registry.get(gitpush.GIT_PUSH_SETTING).set("after_new_commits")
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


class _OverlapWatchingGit(_FakeGitModule):
    """`_FakeGitModule` that HOLDS each `git push` and records who ran what.

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


def _signal_on_close_run(monkeypatch, event: threading.Event) -> None:
    """Set `event` the moment the run enters its epilogue's housekeeping.

    Wrapped on `runlifecycle`, where both doors resolve the name: `end_run`
    calls `close_run` through the module, and the interrupt calls it directly.
    """
    real_close_run = runlifecycle.close_run

    def close_run(*args, **kwargs):
        event.set()
        return real_close_run(*args, **kwargs)

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
    `MainThread`, whatever the timing did.
    """
    fake = _OverlapWatchingGit()
    exit_asked = fake.release
    _signal_on_close_run(monkeypatch, exit_asked)
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
    its record says. The owner reports what a call raised on stderr and goes on
    (`ownership.OwnerThread`), which is the behaviour pinned here.
    """
    def exit_push_fails(policy, project_dir):
        raise OSError("staged: the repository is not reachable")

    monkeypatch.setattr(runlifecycle, "final_git_push", exit_push_fails)
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))

    result = parallel.run_parallel(MemListDriver(["products/only.md"]),
                                   par_args(root_not_cwd(tmp_path), jobs=1,
                                            git_push=PUSHING),
                                   app_name="pytest-gitpush")

    assert result.reason == RunStopReason.NO_WORK
    assert "pusher: OSError: staged: the repository is not reachable" in \
        capsys.readouterr().err
