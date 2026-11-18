"""`win_job`'s suspended start, through each job that uses it: a grandchild
the child spawns in its very first statement is still inside the job, so
closing the job ends it too.

The window is made wide on purpose: the stand-in `popen` sleeps after
creating the process, as a loaded machine may between `Popen` and the
assignment. A child running in that window (not started suspended) spawns
its grandchild outside the job, and the grandchild survives the close.
"""
import os
import subprocess
import sys
import time

import pytest

from llm_agents import cores, win_job
from llm_agents.child_job import ChildJob
from llm_agents.git import live_workers

pytestmark = pytest.mark.skipif(os.name != "nt", reason="job objects are Windows'")

_CHILD = """
import subprocess, sys, time
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "w").write(str(grandchild.pid))
time.sleep(60)
"""

# Python's start-up alone is ~30 ms; 1.5 s is far past it, so an unsuspended
# child has certainly spawned its grandchild before the assignment.
_WINDOW_S = 1.5
# A kill-on-close exit measured 0.0053 s (test_claude_agent_protocol.py,
# 2026-09-08); 10 s. A child start under load: 30 s.
_EXIT_TIMEOUT_MS = 10_000
_START_TIMEOUT_S = 30.0


def _slow_popen(argv, **options):
    child = subprocess.Popen(argv, **options)
    time.sleep(_WINDOW_S)
    return child


def _child_job(argv):
    job = ChildJob()
    return job, job.start(argv, _slow_popen), job.close


def _cores_job(argv):
    job = cores.Job.create()
    return job, job.start(argv, dict(os.environ), _slow_popen), job.close


def _worker_job(argv):
    job = live_workers.WorkerJob.create()
    child = _slow_popen(argv, creationflags=job.creationflags)
    job.adopt(child)
    return job, child, job.close


def _open(pid):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x0001, False, pid)  # SYNCHRONIZE | TERMINATE
    assert handle, ctypes.WinError(ctypes.get_last_error())
    return kernel, handle


def test_a_child_that_cannot_join_the_job_is_killed_never_run():
    started = []

    def popen(argv, **options):
        started.append(subprocess.Popen(argv, **options))
        return started[-1]

    with pytest.raises(OSError):
        win_job.start(0, [sys.executable, "-c", "import time; time.sleep(60)"], popen)
    assert started[0].poll() is not None, "a child outside the job was left running"


@pytest.mark.parametrize("make", [_child_job, _cores_job, _worker_job],
                         ids=["ChildJob", "cores.Job", "WorkerJob"])
def test_an_instantly_spawned_grandchild_dies_with_the_job(tmp_path, make):
    script = tmp_path / "child.py"
    script.write_text(_CHILD, encoding="utf-8")
    pid_file = tmp_path / "grandchild.pid"
    job, child, close = make([sys.executable, str(script), str(pid_file)])
    grandchild = None
    try:
        deadline = time.monotonic() + _START_TIMEOUT_S
        while not (pid_file.exists() and pid_file.read_text()) and time.monotonic() < deadline:
            time.sleep(0.02)
        grandchild = _open(int(pid_file.read_text()))
        close()
        kernel, handle = grandchild
        assert kernel.WaitForSingleObject(handle, _EXIT_TIMEOUT_MS) == 0, \
            "the grandchild outlived the job: it was spawned before the child was in it"
        assert child.wait(timeout=_EXIT_TIMEOUT_MS / 1000) is not None
    finally:
        close()
        if child.poll() is None:
            child.kill()
        child.wait()
        if grandchild is not None:
            kernel, handle = grandchild
            if kernel.WaitForSingleObject(handle, 0) == 258:  # WAIT_TIMEOUT: still alive
                kernel.TerminateProcess(handle, 1)
            kernel.CloseHandle(handle)
