"""Pin `procend.start_tree` on Windows: a tree in a job of its own is ended by
that identity — `TerminateJobObject` — within the stop budget, the ending is
safe to re-enter from a Ctrl+C hook, and an outer job still contains it.

Where a pin is about the explicit termination, the test holds a second
handle to the job (`_held`), so the kill-on-close of the tree's own handle
cannot stand in for it.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_loop import codex_usage, procend, stopclock, win_job

windows = pytest.mark.skipif(os.name != "nt", reason="job objects are Windows'")

# What "did not wait": an ending by identity is a system call or two; every
# wait it replaces is a second or more.
NO_WAIT_S = 0.5
# How long a killed tree may take to let go of its pipe: well under a second
# idle, ten times that loaded.
TREE_END_S = 10.0
# A Python child's start under three other slots' load: 30 s.
START_S = 30.0

# A child that starts a grandchild sharing its stdout, writes both pids to
# argv[1] once it has, and sleeps: the shape of an npm shim in front of the
# provider CLI.
_TREE = ("import os, subprocess, sys, time\n"
         "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
         " stdout=sys.stdout)\n"
         "open(sys.argv[1] + '.tmp', 'w').write(f'{os.getpid()} {g.pid}')\n"
         "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
         "time.sleep(60)\n")


def _wait_for(path: Path) -> list:
    deadline = time.monotonic() + START_S
    while not path.exists():
        assert time.monotonic() < deadline, "the tree never started"
        time.sleep(0.02)
    return [int(pid) for pid in path.read_text().split()]


def _reader(stream):
    thread = threading.Thread(target=stream.read, daemon=True)
    thread.start()
    return thread


def _kernel():
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.DuplicateHandle.argtypes = [wintypes.HANDLE, wintypes.HANDLE, wintypes.HANDLE,
                                  ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
                                  wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.WaitForSingleObject.restype = wintypes.DWORD
    k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    return k


def _held(proc) -> int:
    """A second handle to `proc`'s job: the job outlives the tree's own."""
    import ctypes
    from ctypes import wintypes
    k = _kernel()
    out = wintypes.HANDLE()
    me = k.GetCurrentProcess()
    assert k.DuplicateHandle(me, procend._trees[proc]._handle, me, ctypes.byref(out),
                             0, False, 2), ctypes.WinError(ctypes.get_last_error())
    return out.value


def _dies(pid, timeout_s=TREE_END_S) -> bool:
    """Whether `pid` ends within `timeout_s`; killed regardless afterwards."""
    k = _kernel()
    handle = k.OpenProcess(0x100000 | 0x0001, False, pid)  # SYNCHRONIZE | TERMINATE
    if not handle:
        return True  # already gone
    try:
        if k.WaitForSingleObject(handle, int(timeout_s * 1000)) == 0:
            return True
        k.TerminateProcess(handle, 1)
        return False
    finally:
        k.CloseHandle(handle)


@windows
def test_a_tree_on_a_spent_budget_ends_at_once_with_no_descendant(tmp_path):
    marker = tmp_path / "tree-up"
    proc = procend.start_tree([sys.executable, "-c", _TREE, str(marker)],
                              stdout=subprocess.PIPE)
    reader = _reader(proc.stdout)
    held = _held(proc)
    try:
        _wait_for(marker)
        started = time.monotonic()
        with stopclock.ending(time.monotonic()):
            procend.end_process_tree(proc)
        took = time.monotonic() - started
        assert took < NO_WAIT_S, f"the ending waited on a spent budget: {took:.2f} s"
        # Child and grandchild both hold the pipe: EOF is the proof neither is
        # left — with the job still open through `held`.
        reader.join(TREE_END_S)
        assert not reader.is_alive(), "a descendant outlived the ending"
    finally:
        _kernel().CloseHandle(held)
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=TREE_END_S)


class _ReenteringKernel:
    """A kernel32 stand-in whose call `at` runs `hook` inside it once — a
    SIGINT handler running an `ask` between two bytecodes of the first."""

    def __init__(self, at, hook):
        self.calls = []
        self._at = at
        self._hook = hook

    def _enter(self, name, handle):
        self.calls.append((name, handle))
        if name == self._at and self._hook is not None:
            hook, self._hook = self._hook, None
            hook()
        return 1

    def TerminateJobObject(self, handle, code):
        return self._enter("terminate", handle)

    def CloseHandle(self, handle):
        return self._enter("close", handle)


@pytest.mark.parametrize("at", ["terminate", "close"])
def test_a_ctrl_c_hook_re_entering_the_ending_neither_deadlocks_nor_uses_a_closed_handle(
        monkeypatch, at):
    """`ctrlc.Interrupt.press` runs the turn's `ask` hook from the signal
    handler, on the main thread, inside whatever that thread was doing — the
    first press's `ask`, or the ending's `close`. Run in a thread joined with
    a timeout, so a deadlock fails instead of hanging the suite."""
    tree = procend._JobTree(4242)
    reentered = []

    def hook():
        tree.ask(None)
        reentered.append(True)

    kernel = _ReenteringKernel(at, hook)
    monkeypatch.setattr(win_job, "kernel32", lambda: kernel)
    first = (lambda: tree.ask(None)) if at == "terminate" else tree.close
    worker = threading.Thread(target=first, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive(), f"a re-entered ending deadlocked: {kernel.calls}"
    assert reentered == [True], f"the staged re-entry never ran: {kernel.calls}"
    if at == "terminate":
        assert kernel.calls == [("terminate", 4242), ("terminate", 4242)], (
            f"the first ask and the re-entered one did not both terminate: {kernel.calls}")
    else:
        assert kernel.calls == [("close", 4242)], (
            f"the re-entered ask used a handle being closed: {kernel.calls}")
    tree.close()


_EXITS_LEAVING_A_HOLDER = (
    "import subprocess, sys\n"
    "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
    " stdout=sys.stdout)\n"
    "print('{\"id\": 0, \"result\": {}}', flush=True)\n"
    "sys.stdin.read()\n")


def test_a_quota_server_that_exits_on_eof_leaves_no_holder_of_its_stdout():
    """The server exits on its stdin's close, inside the grace, and leaves a
    descendant holding stdout: the stop still ends the tree (its job, or its
    group through the keeper on POSIX), so the pump sees
    EOF now rather than when the descendant dies on its own."""
    server = codex_usage._QuotaServer([sys.executable, "-c", _EXITS_LEAVING_A_HOLDER], 0)
    try:
        assert procend.owns(server.proc)
        server.request({"method": "initialize", "params": {}},
                       time.monotonic() + START_S)
        server.stop()
        deadline = time.monotonic() + TREE_END_S
        while not server.connection.finished and time.monotonic() < deadline:
            try:
                server.connection.take(time.monotonic() + 0.2, "eof")
            except Exception:  # a timeout, or `Closed` at the EOF awaited
                pass
        assert server.connection.finished, "a descendant kept the server's stdout open"
    finally:
        procend.end_process_tree(server.proc)


# A run inside a job of its own, as `llm_agents.cores` hold starts it
# (KILL_ON_JOB_CLOSE | BREAKAWAY_OK): it starts a tree, hands its parent (the
# test) a second handle to the tree's own job — so that job outlives the
# run, and only the OUTER job's closing can end the tree — and sleeps.
_NESTED = ("import ctypes, os, sys, time\n"
           "from ctypes import wintypes\n"
           "from llm_loop import procend\n"
           "tree, marker = sys.argv[1], sys.argv[2]\n"
           "proc = procend.start_tree([sys.executable, '-c', tree, marker])\n"
           "k = ctypes.WinDLL('kernel32', use_last_error=True)\n"
           "k.GetCurrentProcess.restype = wintypes.HANDLE\n"
           "k.OpenProcess.restype = wintypes.HANDLE\n"
           "k.DuplicateHandle.argtypes = [wintypes.HANDLE] * 3 + ["
           "ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD, wintypes.BOOL,"
           " wintypes.DWORD]\n"
           "parent = k.OpenProcess(0x0040, False, os.getppid())\n"  # PROCESS_DUP_HANDLE
           "held = wintypes.HANDLE()\n"
           "assert k.DuplicateHandle(k.GetCurrentProcess(), procend._trees[proc]._handle,"
           " parent, ctypes.byref(held), 0, False, 2)\n"
           "print('held', held.value, flush=True)\n"
           "time.sleep(60)\n")


@windows
def test_closing_an_outer_job_ends_a_run_s_owned_tree(tmp_path):
    """No CREATE_BREAKAWAY_FROM_JOB: the child's job nests inside the run's,
    so the outer job's closing still ends the child and the grandchild —
    while the child's own job is kept open by the test."""
    import ctypes
    source = str(Path(procend.__file__).resolve().parents[1])
    marker = tmp_path / "tree-up"
    handle = win_job.create(None, win_job.KILL_ON_JOB_CLOSE | win_job.BREAKAWAY_OK)
    closed = False
    run = None
    held = None
    try:
        run = win_job.start(handle, [sys.executable, "-c", _NESTED, _TREE, str(marker)],
                            stdout=subprocess.PIPE,
                            env=dict(os.environ, PYTHONPATH=source))
        line = run.stdout.readline()
        assert line.startswith(b"held "), line
        held = int(line.split()[1])
        pids = _wait_for(marker)
        win_job.kernel32().CloseHandle(handle)
        closed = True
        survivors = [pid for pid in pids if not _dies(pid)]
        assert survivors == [], f"the outer job's close left {survivors} alive"
    finally:
        if not closed:
            win_job.kernel32().CloseHandle(handle)
        if held:
            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(held))
        if run is not None:
            run.kill()
            run.wait(timeout=TREE_END_S)
