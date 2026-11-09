"""Isolated real-transport SIGINT measurement, invoked by test_ctrlc."""
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from llm_loop import ctrlc, providers, streamrender


def main():
    provider = sys.argv[1]
    bound = float(sys.argv[2])
    child = """
import json, sys, time
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
    providers.runtime_argv = lambda argv, name: [sys.executable, '-u', '-c', child, name]
    stamps = {}
    processes = []
    threads = []
    start = streamrender.start_agent_process

    def spawn(*args):
        proc = start(*args)
        processes.append(proc)
        def watch():
            proc.wait()
            stamps['ended'] = time.monotonic()
        thread = threading.Thread(target=watch)
        threads.append(thread)
        thread.start()
        return proc

    streamrender.start_agent_process = spawn
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
        stamps['rescued'] = True
        for proc in processes:
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
    finally:
        stamps['exit'] = time.monotonic()
        backstop.cancel()
        backstop.join()
        for thread in threads:
            thread.join()
    original_print(json.dumps({'provider': provider,
        'child_delay': stamps['ended'] - stamps['press'],
        'exit_delay': stamps['exit'] - stamps['press'],
        'rescued': stamps.get('rescued', False)}))


if __name__ == '__main__':
    main()
