"""Pin that every wait of a budgeted ending takes what is LEFT of its budget.

One pin per limit wired into `runlifecycle.StopBudget` through `stopclock`:
each stages the wait the way it hangs in life, under a spent budget, and asks
that the step returns without waiting. Each limit is shortened or the hang
bounded so that a mutation reverting the wiring fails on the elapsed time in
seconds, not by hanging the suite.
"""

import io
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_loop import (codex_usage, ownership, parallel, procend, runlifecycle,
                      stopclock, termio, usage)

# What "did not wait" means below. A spent budget's step costs a few
# milliseconds (a lock tried, a thread started); every wait it replaces is
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


# A child that starts a grandchild sharing its stdout, marks the file in
# argv[1] once it has, and sleeps: the shape of an npm shim in front of the
# provider CLI.
_TREE = ("import subprocess, sys, time\n"
         "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
         " stdout=sys.stdout)\n"
         "open(sys.argv[1], 'w').close()\n"
         "time.sleep(60)\n")

# The run: starts the tree on its own stdout, hands its parent (the test) a
# second handle to the tree's job — so the job outlives the run, and only the
# explicit termination can end the tree — ends it on a spent budget, and
# leaves AT ONCE (`os._exit`: no shutdown, no thread finishes): the moment an
# unattended run's ending is spent and it exits. It prints the handle and how
# long the ending took.
_RUN = ("import ctypes, os, sys, time\n"
        "from ctypes import wintypes\n"
        "from llm_loop import procend, stopclock\n"
        "marker, tree = sys.argv[1], sys.argv[2]\n"
        "child = procend.start_tree([sys.executable, '-c', tree, marker],"
        " stdout=sys.stdout)\n"
        "k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
        "k.GetCurrentProcess.restype = wintypes.HANDLE\n"
        "k.OpenProcess.restype = wintypes.HANDLE\n"
        "k.DuplicateHandle.argtypes = [wintypes.HANDLE] * 3 + ["
        "ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL,"
        " wintypes.DWORD]\n"
        "parent = k.OpenProcess(0x0040, False, os.getppid())\n"  # PROCESS_DUP_HANDLE
        "held = wintypes.HANDLE()\n"
        "assert k.DuplicateHandle(k.GetCurrentProcess(), procend._trees[child]._handle,"
        " parent, ctypes.byref(held), 0, False, 2)\n"
        "print('held', held.value, flush=True)\n"
        "while not os.path.exists(marker):\n"
        "    time.sleep(0.05)\n"
        "started = time.monotonic()\n"
        "with stopclock.ending(time.monotonic()):\n"
        "    procend.end_process_tree(child)\n"
        "print('left', round(time.monotonic() - started, 3), flush=True)\n"
        "os._exit(0)\n")

# How long the tree may take to end once the run has left: a job's
# termination over two processes, well under a second on an idle machine;
# ten times that for a loaded one.
TREE_END_S = 10.0


@pytest.mark.skipif(os.name != "nt", reason="a tree is budget-bounded only in its "
                    "job; on POSIX the ending keeps its own limits")
def test_a_run_that_leaves_on_a_spent_budget_leaves_no_descendant(tmp_path):
    """A spent budget ends a `procend.start_tree` tree without waiting — its
    job is terminated by identity, so nothing is handed to a reaper that
    could outlive the run — and the run that leaves right after it leaves no
    descendant behind, although the job itself lives on in the test's handle.

    The run, its child and the grandchild all hold one pipe: its EOF is the
    proof that none of them is left."""
    import ctypes
    source = str(Path(procend.__file__).resolve().parents[1])
    env = dict(os.environ, PYTHONPATH=source)
    run = subprocess.Popen([sys.executable, "-c", _RUN,
                            str(tmp_path / "tree-up"), _TREE],
                           stdout=subprocess.PIPE, env=env)
    out = []
    reader = threading.Thread(target=lambda: out.append(run.stdout.read()),
                              daemon=True)
    reader.start()
    # The run's own start included: the whole test took 3.5 s, measured
    # 2026-11-19 under three other slots' load.
    reader.join(TREE_END_S + 30)
    held = None
    try:
        assert not reader.is_alive(), (
            "a descendant outlived the run and still holds the pipe")
        text = out[0] if out else b""
        assert b"held" in text and b"left" in text, out
        held = int(text.split(b"held", 1)[1].split()[0])
        took = float(text.split(b"left", 1)[1].split()[0])
        assert took < NO_WAIT_S, f"the ending waited on a spent budget: {took} s"
    finally:
        if held is None and reader.is_alive():
            run.kill()  # it holds the job: its exit lets the tree go
        if held:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(held))
        run.wait(timeout=10)


class _HeldLock:
    """Holds `lock` on another thread until released — a query in flight."""

    def __init__(self, lock, hold_s=30.0):
        self._lock = lock
        self._hold_s = hold_s
        self._release = threading.Event()
        self._held = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self):
        with self._lock:
            self._held.set()
            self._release.wait(timeout=self._hold_s)

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


def test_the_exit_hook_keeps_the_deadline_of_a_close_that_gave_up():
    """The source a spent close left open is closed again at interpreter exit
    (`_close_open_sources`); that close must not wait for the query in flight
    the ending would not wait for. The query is held 3 s, so a hook that
    waits fails on time rather than hanging."""
    source = _codex_source([])
    try:
        with _HeldLock(source._lock, hold_s=3.0):
            with spent():
                source.close()
            started = time.monotonic()
            codex_usage._close_open_sources()
            elapsed = time.monotonic() - started
        assert elapsed < NO_WAIT_S, (
            f"the exit hook waited for the query in flight: {elapsed:.2f} s")
    finally:
        codex_usage._open_sources.discard(source)


class _Server:
    """A quota server stand-in that records being stopped."""

    def __init__(self):
        self.stopped = threading.Event()

    def stop(self):
        self.stopped.set()


def test_a_close_that_cannot_wait_hands_the_stop_to_the_holder(monkeypatch):
    """The query in flight stops the server as it lets go of the lock, after
    a spent-budget close marked the source closed — on the release itself,
    whichever path the holder took, so no check-then-release window leaves
    the server running."""
    source = _codex_source([])
    server = source._server = _Server()
    holding = threading.Event()
    finish = threading.Event()

    def query():
        holding.set()
        finish.wait(timeout=30)
        return None

    monkeypatch.setattr(source, "_query_locked", query)
    holder = threading.Thread(target=lambda: source.get_usage(cache_value=False),
                              daemon=True)
    holder.start()
    try:
        assert holding.wait(timeout=5)
        with spent():
            source.close()
        assert not server.stopped.is_set(), "closed under the holder"
    finally:
        finish.set()
        holder.join(timeout=5)
    assert server.stopped.wait(timeout=5), (
        "the holder let go of the lock and left the server running")
    assert source not in codex_usage._open_sources


class _SlowServer:
    """A server whose stop is under way until `release` is set."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def stop(self):
        self.started.set()
        self.release.wait(timeout=30)


def test_a_close_during_a_holder_s_stop_waits_for_it(monkeypatch):
    """The window after the holder detached the server and let the lock go,
    before its stop is done: a second close then finds no server, and must
    still neither return early nor drop the source from the exit hook's
    registry — the stop is not done yet."""
    source = _codex_source([])
    server = source._server = _SlowServer()
    finish = threading.Event()
    holding = threading.Event()

    def query():
        holding.set()
        finish.wait(timeout=30)
        return None

    monkeypatch.setattr(source, "_query_locked", query)
    holder = threading.Thread(target=lambda: source.get_usage(cache_value=False),
                              daemon=True)
    holder.start()
    closer = None
    try:
        assert holding.wait(timeout=5)
        with spent():
            source.close()                  # cannot wait: hands the stop over
        finish.set()
        assert server.started.wait(timeout=5), "the holder began no stop"
        # Inside the window. A spent close returns, keeping the registry:
        with spent():
            source.close()
        assert source in codex_usage._open_sources, (
            "a spent close dropped a source whose stop is still under way")
        # An unbudgeted close waits for the stop.
        closer = threading.Thread(target=source.close, daemon=True)
        closer.start()
        closer.join(timeout=0.5)
        assert closer.is_alive(), "close returned while the stop was under way"
        assert source in codex_usage._open_sources
    finally:
        finish.set()
        server.release.set()
        holder.join(timeout=5)
        if closer is not None:
            closer.join(timeout=5)
    assert not closer.is_alive()
    assert source not in codex_usage._open_sources


def test_the_input_reader_s_join_on_a_spent_budget_does_not_wait():
    """`termio.TerminalInput.stop`'s 1 s join, staged as a reader that never
    comes back."""
    reader = termio.TerminalInput(stream=io.StringIO())
    stuck = threading.Event()
    reader._thread = threading.Thread(target=stuck.wait, args=(30,), daemon=True)
    reader._thread.start()
    try:
        started = time.monotonic()
        with spent():
            reader.stop()
        elapsed = time.monotonic() - started
    finally:
        stuck.set()
    assert elapsed < NO_WAIT_S, (
        f"the input reader was joined on a spent budget: {elapsed:.2f} s")


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


class _TricklingResponse:
    """A body that arrives a byte at a time, each well inside a per-socket
    timeout — no single operation ever times out — and then breaks off: an
    attempt takes 1 s, both with the pause between them 3 s."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __init__(self, failures):
        self._failures = failures

    def read(self):
        for _ in range(10):
            time.sleep(0.1)
        self._failures.append(time.monotonic())
        raise OSError("staged: the body broke off")


def test_the_claude_snapshot_is_bounded_whole_not_per_socket_operation(
        monkeypatch, capsys):
    failures = []
    monkeypatch.setattr(usage, "oauth_token", lambda: "token")
    monkeypatch.setattr(usage.urllib.request, "urlopen",
                        lambda request, timeout: _TricklingResponse(failures))
    started = time.monotonic()
    with stopclock.ending(time.monotonic() + 0.3):
        data = usage.UsageSource().query_usage_json()
    elapsed = time.monotonic() - started
    assert elapsed < 0.3 + NO_WAIT_S, (
        f"the snapshot outlasted its budget on a trickling body: {elapsed:.2f} s")
    assert data is None
    first = capsys.readouterr()
    assert "within the stop budget" in first.out
    assert first.err == "", first.err
    # The abandoned read goes on to fail on its own — every attempt, and the
    # worker then ends; 3 s of it (two 1 s attempts and the 1 s pause), so
    # 10 s is room on a loaded machine.
    workers = [t for t in threading.enumerate() if t.name == "usage-read"]
    assert workers, "no worker read the snapshot"
    for worker in workers:
        worker.join(timeout=10)
    assert not any(w.is_alive() for w in workers), "the late read never ended"
    assert len(failures) == usage.HTTP_ATTEMPTS, (
        f"the late read did not fail every attempt: {len(failures)}")
    late = capsys.readouterr()
    assert late.out == "" and late.err == "", (
        f"the abandoned read spoke up after the budget's line:\n"
        f"out: {late.out!r}\nerr: {late.err!r}")


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


def test_an_explicit_exit_keeps_its_budget_through_the_teardown(monkeypatch):
    """`boundary.exit` reaches `__exit__` as its own SystemExit, with the run
    already closed: the releases and the last pusher close there still spend
    the exit's budget rather than running unbounded after it."""
    monkeypatch.setattr(runlifecycle, "UNWIND_PUSH_DEADLINE_S", 0.0)
    seen = []

    class _Held:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            seen.append(stopclock.left())

    pusher = _Pusher()
    boundary = runlifecycle.RunBoundary(_ctx(), counts=lambda: (0, 0),
                                        pusher=pusher)
    with pytest.raises(SystemExit):
        with boundary:
            boundary.hold(_Held())
            boundary.exit(3, ending="staged stop", reason="staged stop")
    assert seen == [0.0], (
        f"the held resources were released outside the exit's budget: {seen}")
    assert pusher.timeouts and all(t == 0 for t in pusher.timeouts), (
        f"the pusher was waited for past the exit's budget: {pusher.timeouts}")


@pytest.fixture
def stuck_push(monkeypatch):
    """An exit push that never comes back on its own (held 5 s at most)."""
    release = threading.Event()
    monkeypatch.setattr(runlifecycle.projectroot, "project_dir", lambda: ".")
    monkeypatch.setattr(runlifecycle, "final_git_push",
                        lambda *args, **kwargs: release.wait(timeout=5))
    yield
    release.set()


@pytest.mark.parametrize("inner", [None, 30.0], ids=["no budget", "longer"])
def test_an_inner_close_run_never_outwaits_an_outer_ending(stuck_push, inner,
                                                           capsys):
    """The push waits for the EFFECTIVE deadline: an outer ending of 0.3 s
    bounds an inner `close_run` that has no budget, or a longer one."""
    budget = None if inner is None else runlifecycle.StopBudget(inner)
    started = time.monotonic()
    with stopclock.ending(time.monotonic() + 0.3):
        runlifecycle.close_run(_ctx(), usages=[], budget=budget)
    elapsed = time.monotonic() - started
    assert elapsed < 0.3 + NO_WAIT_S, (
        f"the inner push outwaited the outer ending: {elapsed:.2f} s")
    assert "did not finish within" in capsys.readouterr().err
