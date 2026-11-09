"""Bounded launch and cleanup of isolated test process trees."""
import os
import signal
import subprocess
import sys
import threading
import time


def run_probe(argv, timeout, *, before_wait=None, watchdog=None, deadline=None):
    """Timeout owns the entire probe tree, including a stalled provider."""
    # Completed timeout + cleanup measured 5.060 / 5.073 s on Windows,
    # 2026-11-09. The 15 s allowance exceeds twice either complete interval.
    # Callers may supply one absolute deadline shared with their own waits.
    if deadline is None:
        deadline = time.monotonic() + max(timeout, watchdog or 0) + 15
    def remaining():
        return max(0, deadline - time.monotonic())
    job = None
    launch = argv
    if os.name == 'nt':
        from _process_job import Job
        job = Job()
        # No descendants can start before assignment: bootstrap waits for a
        # byte on stdin. The job handle is not inherited by the child.
        launch = [sys.executable, '-c',
                  'import subprocess,sys; sys.stdin.buffer.read(1); '
                  'sys.exit(subprocess.call(sys.argv[1:]))', *argv]
    try:
        proc = subprocess.Popen(launch, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                stdin=subprocess.PIPE if job else None,
                                text=True, start_new_session=os.name != 'nt')
    except BaseException:
        if job:
            job.close(deadline)
        raise
    def stop_tree():
        if job:
            job.close(deadline)
        else:
            os.killpg(proc.pid, signal.SIGKILL)
        if proc.poll() is None:
            proc.kill()  # Also covers failure to assign the bootstrap.

    rescue_errors = []
    def rescue():
        try:
            stop_tree()
        except Exception as exc:
            rescue_errors.append(exc)
    timer = threading.Timer(watchdog, rescue) if watchdog is not None else None
    def finish_watchdog():
        if timer:
            timer.cancel()
            timer.join(remaining())
            if timer.is_alive():
                raise TimeoutError('probe watchdog did not stop before deadline')
            if rescue_errors:
                raise rescue_errors[0]
    if timer:
        timer.start()
    try:
        try:
            try:
                if job:
                    job.assign(proc)
                    proc.stdin.write('x')
                    proc.stdin.close()
                    proc.stdin = None
                if before_wait is not None:
                    before_wait(proc)
                stdout, stderr = proc.communicate(timeout=min(timeout, remaining()))
            finally:
                # Join the watchdog before the main thread takes cleanup ownership.
                finish_watchdog()
        except BaseException:
            stop_tree()
            proc.wait(timeout=remaining())
            raise
        else:
            # A successful bootstrap may still have descendants in its job.
            if job:
                job.close(deadline)
            return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    finally:
        if proc.stdin:
            proc.stdin.close()
        # Windows communicate readers may still own the stream locks after a
        # deadline failure. Do not turn bounded cleanup into an unbounded close.
        for name in ('stdout', 'stderr'):
            reader = getattr(proc, name + '_thread', None)
            if reader is None or not reader.is_alive():
                getattr(proc, name).close()


