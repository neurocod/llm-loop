"""Pin that every wait of a budgeted ending takes what is LEFT of its budget.

One pin per limit wired into `runlifecycle.StopBudget` through `stopclock`:
each stages the wait the way it hangs in life, under a spent budget, and asks
that the step returns without waiting. Each limit is shortened or the hang
bounded so that a mutation reverting the wiring fails on the elapsed time in
seconds, not by hanging the suite.
"""

import subprocess
import sys
import threading
import time

import pytest

from llm_loop import (codex_usage, ownership, parallel, procend, runlifecycle,
                      stopclock, usage)

# What "did not wait" means below. A spent budget's step costs a few
# milliseconds (a `taskkill` started, a lock tried); every wait it replaces is
# a second or more, so half a second tells the two apart either way.
NO_WAIT_S = 0.5


def spent():
    """An ending whose budget ran out as it began."""
    return stopclock.ending(time.monotonic())


def test_an_inner_ending_never_extends_the_outer_one():
    with stopclock.ending(time.monotonic() + 1.0):
        with stopclock.ending(None):
            assert 0 < stopclock.left() <= 1.0
        with stopclock.ending(time.monotonic() + 100.0):
            assert stopclock.left() <= 1.0
    assert stopclock.left() is None
    assert stopclock.within(7.0) == 7.0


def test_a_thread_outside_the_ending_keeps_its_own_limit():
    seen = []
    with spent():
        worker = threading.Thread(target=lambda: seen.append(stopclock.within(2.0)))
        worker.start()
        worker.join()
    assert seen == [2.0]


class _UnendingChild:
    """A child no ask and no kill ends — a stuck handle — whose `wait` takes
    all the time it is given; `taskkill` against it hangs as well."""

    pid = 4242

    def __init__(self):
        self.waits = []

    def poll(self):
        return None

    def wait(self, timeout=None):
        self.waits.append(timeout)
        time.sleep(timeout)
        raise subprocess.TimeoutExpired("child", timeout)

    def terminate(self):
        pass

    def kill(self):
        pass


def test_ending_a_process_tree_on_a_spent_budget_does_not_wait(monkeypatch):
    """`procend.end_process_tree`: REAP_GRACE_S twice plus `taskkill`, unwaited."""
    def hanging_taskkill(argv, timeout=None, **kwargs):
        time.sleep(timeout)
        raise subprocess.TimeoutExpired(argv, timeout)

    monkeypatch.setattr(procend.subprocess, "run", hanging_taskkill)
    monkeypatch.setattr(procend.subprocess, "Popen",
                        lambda *args, **kwargs: None)
    child = _UnendingChild()
    started = time.monotonic()
    with spent():
        procend.end_process_tree(child)
    elapsed = time.monotonic() - started
    assert elapsed < NO_WAIT_S, (
        f"the escalation waited on a spent budget: {elapsed:.2f} s "
        f"(waits {child.waits})")


def test_a_spent_budget_still_ends_the_tree():
    """Only the waiting is skipped: the child is ended all the same."""
    child = subprocess.Popen([sys.executable, "-c",
                              "import time; time.sleep(60)"])
    try:
        started = time.monotonic()
        with spent():
            procend.end_process_tree(child)
        elapsed = time.monotonic() - started
        assert elapsed < NO_WAIT_S, (
            f"the escalation waited on a spent budget: {elapsed:.2f} s")
        # Still ENDED, only not waited for: the kill was issued.
        assert child.wait(timeout=10) is not None
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


class _HeldLock:
    """Holds `lock` on another thread until released — a query in flight."""

    def __init__(self, lock):
        self._lock = lock
        self._release = threading.Event()
        self._held = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self):
        with self._lock:
            self._held.set()
            self._release.wait(timeout=30)

    def __enter__(self):
        self._thread.start()
        assert self._held.wait(timeout=5)
        return self

    def __exit__(self, *exc):
        self._release.set()
        self._thread.join(timeout=5)


def _codex_source(spawned):
    def argv():
        spawned.append(True)
        return [sys.executable, "-c", "import time; time.sleep(60)"]
    return codex_usage.CodexUsageSource(argv=argv)


def test_the_codex_snapshot_on_a_spent_budget_starts_no_server():
    """`APP_SERVER_TIMEOUT`: no server started, no handshake waited for."""
    spawned = []
    source = _codex_source(spawned)
    try:
        started = time.monotonic()
        with spent():
            source.get_usage(cache_value=False)
        assert time.monotonic() - started < NO_WAIT_S
        assert spawned == [], "a quota server was started on a spent budget"
    finally:
        source.close()


def test_the_codex_snapshot_does_not_wait_for_a_query_in_flight():
    spawned = []
    source = _codex_source(spawned)
    try:
        with _HeldLock(source._lock):
            started = time.monotonic()
            with spent():
                source.get_usage(cache_value=False)
            elapsed = time.monotonic() - started
        assert elapsed < NO_WAIT_S, (
            f"the snapshot waited for the query in flight: {elapsed:.2f} s")
    finally:
        source.close()


def test_closing_the_codex_source_does_not_wait_for_a_query_in_flight():
    source = _codex_source([])
    with _HeldLock(source._lock):
        started = time.monotonic()
        with spent():
            source.close()
        elapsed = time.monotonic() - started
    assert elapsed < NO_WAIT_S, (
        f"close waited for the query in flight: {elapsed:.2f} s")
    assert source._closed, "a close that could not wait left the source open"
    source.close()                            # the exit hook's second close


def test_the_claude_snapshot_on_a_spent_budget_makes_no_request(monkeypatch):
    """`usage.HTTP_TIMEOUT` x HTTP_ATTEMPTS and the pause between them."""
    requests = []

    def refuse(request, timeout):
        requests.append(timeout)
        raise OSError("staged: refused")

    monkeypatch.setattr(usage, "oauth_token", lambda: "token")
    monkeypatch.setattr(usage.urllib.request, "urlopen", refuse)
    with spent():
        assert usage.UsageSource().query_usage_json() is None
    assert requests == [], "a usage request was made on a spent budget"


def test_the_claude_snapshot_is_given_only_what_is_left(monkeypatch):
    timeouts = []

    def refuse(request, timeout):
        timeouts.append(timeout)
        raise OSError("staged: refused")

    monkeypatch.setattr(usage, "oauth_token", lambda: "token")
    monkeypatch.setattr(usage.urllib.request, "urlopen", refuse)
    with stopclock.ending(time.monotonic() + 0.3):
        usage.UsageSource().query_usage_json()
    assert timeouts and all(t <= 0.3 for t in timeouts), timeouts


def test_the_console_drain_on_a_spent_budget_does_not_wait(monkeypatch, capsys):
    """`parallel.CONSOLE_CLOSE_TIMEOUT_S`, staged as a console that never
    comes back (shortened to 3 s so a reverted wiring fails, not hangs)."""
    monkeypatch.setattr(parallel, "CONSOLE_CLOSE_TIMEOUT_S", 3.0)
    owner = ownership.OwnerThread("console-lines").start()
    monkeypatch.setattr(parallel, "_console", owner)
    stuck = threading.Event()
    owner.post(stuck.wait, 30)
    try:
        started = time.monotonic()
        with spent():
            parallel._close_console()
        elapsed = time.monotonic() - started
    finally:
        stuck.set()
        owner.close(timeout=5)
    assert elapsed < NO_WAIT_S, (
        f"the console drain waited on a spent budget: {elapsed:.2f} s")
    assert "still unwritten" in capsys.readouterr().err


class _Pusher:
    """Records the timeouts the ending gives `close`; never finishes on its
    own, as a pusher stuck inside `git push`."""

    thread = object()

    def __init__(self):
        self.timeouts = []

    def close(self, timeout=None, final=None):
        self.timeouts.append(timeout)
        return False


def _ctx():
    return runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=False, progress=None,
        settings=runlifecycle.RunSettings(), registry=None,
        status_enabled=False)


def test_an_unwinding_ending_takes_its_budget_before_the_release(monkeypatch):
    """The release of what the boundary holds (the console's drain) is the
    first wait of an unwinding ending, and spends the ending's budget."""
    monkeypatch.setattr(runlifecycle, "final_git_push",
                        lambda *args, **kwargs: None)
    seen = []

    class _Held:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            seen.append(stopclock.left())

    boundary = runlifecycle.RunBoundary(_ctx(), counts=lambda: (0, 0))
    with pytest.raises(RuntimeError):
        with boundary:
            boundary.hold(_Held())
            raise RuntimeError("staged")
    assert seen and seen[0] is not None, (
        "the held resources were released outside the ending's budget")
    assert seen[0] <= runlifecycle.UNWIND_PUSH_DEADLINE_S


def test_the_pusher_closes_on_a_spent_budget_without_waiting(monkeypatch,
                                                             capsys):
    """`RunBoundary.__exit__`'s `pusher.close(0.5)` and the spent branch's
    `EXIT_PUSH_POLL_S` take what is left: nothing."""
    monkeypatch.setattr(runlifecycle, "UNWIND_PUSH_DEADLINE_S", 0.0)
    pusher = _Pusher()
    boundary = runlifecycle.RunBoundary(_ctx(), counts=lambda: (0, 0),
                                        pusher=pusher)
    with pytest.raises(RuntimeError):
        with boundary:
            raise RuntimeError("staged")
    assert pusher.timeouts and all(t == 0 for t in pusher.timeouts), (
        f"the pusher was waited for on a spent budget: {pusher.timeouts}")
    assert "did not finish within 0 s" in capsys.readouterr().err
