"""Process-probe ownership, watchdog and descendant cleanup contracts."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from _process_probe import run_probe

# The bound on a child Python that stages one press and exits: interpreter
# start plus the package import, 0.27 / 0.25 / 0.28 s measured 2026-10-03; a press
# that deadlocks never ends, so the bound only has to be far above that.
CHILD_TIMEOUT_S = 30


@pytest.mark.parametrize('expires', [False, True])
def test_job_close_waits_for_active_members_before_releasing_handle(expires):
    from _process_job import Job
    from types import SimpleNamespace
    events = []
    active = iter([1, 0])
    def terminate(handle, code):
        events.append('terminate')
        return True
    def query(handle, kind, buffer, size, returned):
        buffer._obj.active = next(active)
        events.append('query')
        return True
    job = Job.__new__(Job)
    job._lock = threading.Lock()
    job.handle = 77
    job.api = SimpleNamespace(TerminateJobObject=terminate,
                              QueryInformationJobObject=query,
                              CloseHandle=lambda handle: events.append(f'close {handle}'))
    if expires:
        with pytest.raises(TimeoutError):
            job.close(time.monotonic())
        assert events == ['terminate', 'query', 'close 77']
    else:
        job.close(time.monotonic() + CHILD_TIMEOUT_S)
        assert events == ['terminate', 'query', 'query', 'close 77']


def test_concurrent_job_close_releases_handle_once():
    from _process_job import Job
    from types import SimpleNamespace
    entered, release = threading.Event(), threading.Event()
    closed = []
    def close_handle(handle):
        closed.append(handle)
        entered.set()
        assert release.wait(CHILD_TIMEOUT_S)
    job = Job.__new__(Job)
    job._lock = threading.Lock()
    job.handle = 77
    job.api = SimpleNamespace(CloseHandle=close_handle)
    job._end_members = lambda handle, deadline: None
    first = threading.Thread(target=job.close, args=(time.monotonic() + CHILD_TIMEOUT_S,))
    second = threading.Thread(target=job.close, args=(time.monotonic() + CHILD_TIMEOUT_S,))
    first.start()
    try:
        assert entered.wait(CHILD_TIMEOUT_S)
        second.start()
        second.join(CHILD_TIMEOUT_S)
        assert not second.is_alive()
        assert closed == [77]
        assert job.handle is None
    finally:
        release.set()
        first.join(CHILD_TIMEOUT_S)
        if second.ident is not None:
            second.join(CHILD_TIMEOUT_S)


@pytest.mark.skipif(os.name != 'nt', reason='Windows job watchdog')
def test_watchdog_can_end_probe_without_main_timeout():
    done = run_probe([sys.executable, '-c',
                      'import time; time.sleep(30); print("finished")'],
                     timeout=CHILD_TIMEOUT_S, watchdog=1)
    # Kill-on-close can report exit code zero on Windows. Finishing the body
    # (or reaching the main timeout) would mean the watchdog failed to act.
    assert 'finished' not in done.stdout


def test_probe_rescue_covers_app_server_startup():
    done = run_probe([sys.executable,
                      str(Path(__file__).with_name('_silent_interrupt_probe.py')),
                      'codex', '1', 'stall'], timeout=CHILD_TIMEOUT_S)
    assert done.returncode == 0, done.stdout + done.stderr
    measured = json.loads(done.stdout.splitlines()[-1])
    assert measured['rescued']
    assert measured['returncodes'] and all(code is not None for code in measured['returncodes'])


@pytest.mark.skipif(os.name != 'nt', reason='Windows process-tree timeout cleanup')
@pytest.mark.parametrize('parent_exits', [False, True])
def test_probe_timeout_reaps_provider_child(tmp_path, parent_exits):
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    # One overall deadline includes readiness, watchdog joining, job polling,
    # and the test-owned process wait. 60 s also doubles the existing 30 s
    # startup watchdog bound; completed cases measured 5.060 / 5.073 s.
    overall_deadline = time.monotonic() + 60
    def remaining_ms():
        return max(0, int((overall_deadline - time.monotonic()) * 1000))
    handles = []
    def release_handle():
        for handle in handles:
            kernel.TerminateProcess(handle, 1)
            kernel.WaitForSingleObject(handle, remaining_ms())
            kernel.CloseHandle(handle)
    # Retain identity before cleanup; reopening a PID afterwards can inspect
    # (and terminate) an unrelated process if Windows has reused that PID.
    pid_file = tmp_path / 'pid'
    script = ("import subprocess, sys, time; "
              "child = subprocess.Popen([sys.executable, '-c', "
              "'import os,sys,time; from pathlib import Path; "
              "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)', sys.argv[1]], "
              "stdout=None, stderr=None); "
              "time.sleep(30)")
    if parent_exits:
        script = script.rsplit('; time.sleep(30)', 1)[0]
    # Interpreter + child startup measured below 0.8 s, 2026-11-09.
    measured = []
    def ready(proc):
        deadline = min(overall_deadline, time.monotonic() + CHILD_TIMEOUT_S)
        while not pid_file.exists():
            assert time.monotonic() < deadline, 'child never became ready'
            time.sleep(0.01)
        while not pid_file.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        handle = kernel.OpenProcess(0x100001, False, int(pid_file.read_text()))
        assert handle, ctypes.get_last_error()
        handles.append(handle)
        if parent_exits:
            assert proc.wait(timeout=max(0, overall_deadline - time.monotonic())) == 0
            assert proc.poll() is not None
        measured.append(time.monotonic())
    # The watchdog fires before the overall deadline, leaving time to join it.
    # No cleanup operation starts a fresh timeout budget.
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_probe([sys.executable, '-c', script, str(pid_file)], timeout=5,
                      before_wait=ready, watchdog=30, deadline=overall_deadline)
        # Zero job membership was observed before a process handle signaled
        # in the previous regression. Job accounting is not a process join:
        # measured lag 0.046056 / 0.045247 s (live/exited parent), Windows
        # 2026-11-09. Cover it with the SAME overall deadline, not a new budget.
        wait_started = time.monotonic()
        assert kernel.WaitForSingleObject(handles[0], remaining_ms()) == 0, 'provider survived probe timeout'
        print(f'handle wait after zero membership: {time.monotonic() - wait_started:.6f} s')
    finally:
        release_handle()
    # Timeout plus cleanup after readiness: 5.060 s (live parent), 5.073 s
    # (exited parent), Windows 2026-11-09. 15 s exceeds twice both measurements.
    elapsed = time.monotonic() - measured[0]
    print(f'orphan cleanup after readiness: {elapsed:.3f} s')
    assert elapsed < 15, 'orphan kept captured pipes open'


