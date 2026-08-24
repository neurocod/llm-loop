"""The git-push policy, and the repository each runner applies it to.

`gitpush` used to read `cyclecore.PROJECT_DIR` itself, so "which repository" was
never passed anywhere and could not be got wrong. Now it is an argument, and a
handover is exactly the thing that regresses silently: the engine is vendored
under a host project whose root is NOT the process cwd (see
`cyclecore.set_project_root`), so a caller that quietly substituted `os.getcwd()`
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

import pytest

from llm_loop import cyclecore, gitpush, parallel
from llm_loop.cyclecore import ClaudeCommand, Driver
from llm_loop.drivers import ListFileDriver


@pytest.fixture(autouse=True)
def _restore_streams():
    """Both runners tee sys.stdout/stderr into their log and never put them back;
    undo that so one test's tee does not follow the next one."""
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


class _FakeGitModule:
    """Stands in for `gitpush.subprocess`, recording (argv, cwd) per call.

    A replacement MODULE rather than a patched `subprocess.run`: the real
    attribute is shared by every module in the process, so patching it would
    also silently rewire the provider launcher and anything else a runner
    reaches for during the same test.

    Every `git` invocation succeeds and `rev-list --count` answers 1, so the
    policy takes each branch that runs git at all rather than short-circuiting
    on "nothing to push".
    """

    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self):
        self.calls = []

    def run(self, argv, **kwargs):
        self.calls.append((tuple(argv), kwargs.get("cwd")))
        return subprocess.CompletedProcess(argv, 0, stdout="1")

    @property
    def dirs(self):
        return {cwd for _, cwd in self.calls}


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

    Both of the loop's push sites are covered by one run: the per-iteration
    `maybe_git_push` at the top of the iteration and the final push on the way
    out.
    """
    fake = _FakeGitModule()
    monkeypatch.setattr(gitpush, "subprocess", fake)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *args, **kwargs: 0)
    root = _elsewhere(tmp_path)

    cyclecore.run_loop(_OneShotDriver(), _seq_args(root),
                       app_name="pytest-gitpush")

    assert fake.calls, "the run pushed nothing at all — the pin proved nothing"
    assert fake.dirs == {root}, \
        f"the sequential runner pushed the wrong repository: {fake.calls}"


def test_the_parallel_runner_pushes_the_project_it_was_pointed_at(
        tmp_path, monkeypatch):
    """The same handover from the other runner, on its exit push.

    Its periodic pusher wakes on a 60 s timer, so the exit push is the site a
    test can reach; both read the same `cyclecore.project_dir()`, and a run over
    a drained list reaches the exit push immediately.
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

    assert fake.calls, "the run pushed nothing at all — the pin proved nothing"
    assert fake.dirs == {root}, \
        f"the parallel runner pushed the wrong repository: {fake.calls}"
