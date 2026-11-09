"""Isolated real-transport SIGINT measurement, invoked by test_ctrlc."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_loop import ctrlc, providers, streamrender


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
    rescue_errors = []
    def rescue():
        try:
            if job:
                job.close(deadline)
            elif os.name != 'nt':
                os.killpg(proc.pid, signal.SIGKILL)
            if proc.poll() is None:
                proc.kill()
        except Exception as exc:
            rescue_errors.append(exc)
    timer = threading.Timer(watchdog, rescue) if watchdog is not None else None
    def stop_watchdog():
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
        if job:
            job.assign(proc)
            proc.stdin.write('x')
            proc.stdin.close()
            proc.stdin = None
        if before_wait is not None:
            before_wait(proc)
        stdout, stderr = proc.communicate(timeout=min(timeout, remaining()))
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    except BaseException:
        stop_watchdog()
        if os.name == 'nt':
            job.close(deadline)
            if proc.poll() is None:
                proc.kill()  # Also covers failure to assign the bootstrap.
            proc.wait(timeout=remaining())
        else:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=remaining())
        raise
    finally:
        try:
            stop_watchdog()
        finally:
            if job:
                job.close(deadline)
        if proc.stdin:
            proc.stdin.close()
        # Windows communicate readers may still own the stream locks after a
        # deadline failure. Do not turn bounded cleanup into an unbounded close.
        for name in ('stdout', 'stderr'):
            reader = getattr(proc, name + '_thread', None)
            if reader is None or not reader.is_alive():
                getattr(proc, name).close()


def main():
    provider = sys.argv[1]
    bound = float(sys.argv[2])
    stalled = len(sys.argv) > 3 and sys.argv[3] == 'stall'
    child = """
import json, sys, time
if sys.argv[2] == 'stall':
    print('startup-stalled', flush=True)
    time.sleep(30)
if sys.argv[1] == 'codex':
    for line in sys.stdin:
        request = json.loads(line)
        method = request['method']
        if 'id' not in request:
            continue
        result = {'thread': {'id': 'thread'}} if method == 'thread/start' else {'turn': {'id': 'turn'}} if method == 'turn/start' else {}
        print(json.dumps({'id': request['id'], 'result': result}), flush=True)
        if method == 'turn/start':
            break
print('silent-child-ready', flush=True)
time.sleep(30)
"""
    providers.runtime_argv = lambda argv, name: [sys.executable, '-u', '-c', child, name,
                                                'stall' if stalled else 'turn']
    stamps = {}
    processes = []
    threads = []
    spawn_lock = threading.Lock()
    expired = False
    start = subprocess.Popen

    def spawn(argv, **kwargs):
        proc = start(argv, **kwargs)
        if argv[0] != sys.executable:
            return proc
        with spawn_lock:
            processes.append(proc)
            if expired:
                proc.kill()
        def watch():
            proc.wait()
            stamps['ended'] = time.monotonic()
        thread = threading.Thread(target=watch)
        threads.append(thread)
        thread.start()
        return proc

    subprocess.Popen = spawn
    original_print = print

    def output(*args, **kwargs):
        original_print(*args, **kwargs)
        if args and str(args[0]).strip() == 'silent-child-ready':
            def press():
                stamps['press'] = time.monotonic()
                if os.name == 'nt':
                    signal.raise_signal(signal.SIGINT)
                else:
                    # POSIX raise_signal targets the timer thread; a console
                    # signal targets the process and interrupts its pipe read.
                    os.kill(os.getpid(), signal.SIGINT)
            timer = threading.Timer(0.2, press)
            threads.append(timer)
            timer.start()

    streamrender.print = output
    def rescue():
        nonlocal expired
        stamps['rescued'] = True
        with spawn_lock:
            expired = True
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
    backstop = threading.Timer(bound, rescue)
    backstop.start()
    try:
        with ctrlc.captured() as interrupt:
            code = streamrender.run_agent_streaming(['fake'], provider, raw=True, prompt='test')
            assert code == ctrlc.EXIT_CODE
            assert interrupt.presses == 1
    except SystemExit as exc:
        assert exc.code == ctrlc.EXIT_CODE
    except providers.appserver.Closed:
        if not stalled:
            raise
    finally:
        stamps['exit'] = time.monotonic()
        backstop.cancel()
        backstop.join()
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
        for thread in threads:
            thread.join()
    if stalled:
        original_print(json.dumps({'rescued': stamps.get('rescued', False),
                                  'returncodes': [proc.returncode for proc in processes]}))
        return
    original_print(json.dumps({'provider': provider,
        'child_delay': stamps['ended'] - stamps['press'],
        'exit_delay': stamps['exit'] - stamps['press'],
        'rescued': stamps.get('rescued', False)}))


if __name__ == '__main__':
    main()
