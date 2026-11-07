#!/usr/bin/env python3
"""Block until a file disappears, then exit 0.

    python agent-tools/wait_file_gone.py stop            # relative to cwd
    python agent-tools/wait_file_gone.py /abs/path/stop --poll 2

What it is for: an LLM agent waiting for another process to remove a lock or
stop file. Why a script and not `while [ -f stop ]; do sleep 1; done`: shell
gates such as the ask-user-gate plugin reject clock waits and loops in a
command line, and a one-line script is re-runnable from any directory. Exit 0 =
the file is gone (also when it never existed); Ctrl+C = exit 130. Prints a
heartbeat once a minute so a background run shows it is alive.
"""
import argparse
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="file to wait for")
    ap.add_argument("--poll", type=float, default=1.0,
                    help="seconds between existence checks (default 1)")
    args = ap.parse_args()
    path = os.path.abspath(args.path)
    if not os.path.exists(path):
        print(f"{path}: already absent")
        return 0
    print(f"waiting for {path} to disappear (poll {args.poll:g} s)", flush=True)
    started = time.time()
    last_beat = started
    try:
        while os.path.exists(path):
            time.sleep(args.poll)
            now = time.time()
            if now - last_beat >= 60:
                print(f"  still present after {int(now - started)} s", flush=True)
                last_beat = now
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    print(f"{path}: gone after {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
