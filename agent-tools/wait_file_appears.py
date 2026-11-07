#!/usr/bin/env python3
r"""Block until a file exists, then exit 0; exit 1 if --timeout runs out first.

    python agent-tools/wait_file_appears.py out/run/x.log
    python agent-tools/wait_file_appears.py /abs/path/done --poll 2 --timeout 580
    python agent-tools/wait_file_appears.py run.log --contains "# \S+ after"

What it is for: an LLM agent waiting for a background job to produce a file.
The mirror of wait_file_gone.py, for the same reason: shell gates such as the
ask-user-gate plugin reject clock waits and loops in a command line, and a
reviewed script the human allowed once runs without a permission prompt. Keep
--timeout under the calling tool's foreground cap (600 s for Claude Code's
shell tool) so the wait ends by itself rather than being killed. Prints a
heartbeat once a minute. For a file that keeps growing, `follow_file.py` also
prints what arrives.
"""
import argparse
import os
import re
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="file to wait for")
    ap.add_argument("--poll", type=float, default=1.0,
                    help="seconds between existence checks (default 1)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="give up after this many seconds (exit 1)")
    ap.add_argument("--contains", default=None,
                    help="also wait until the file's text matches this regex")
    args = ap.parse_args()
    path = os.path.abspath(args.path)
    pattern = re.compile(args.contains) if args.contains else None

    def ready() -> bool:
        if not os.path.exists(path):
            return False
        if pattern is None:
            return True
        with open(path, encoding="utf-8", errors="replace") as handle:
            return pattern.search(handle.read()) is not None

    if ready():
        print(f"{path}: already present")
        return 0
    print(f"waiting for {path} to appear (poll {args.poll:g} s)", flush=True)
    started = time.time()
    last_beat = started
    try:
        while not ready():
            now = time.time()
            if args.timeout is not None and now - started >= args.timeout:
                print(f"{path}: still absent after {now - started:.1f} s (timeout)")
                return 1
            time.sleep(args.poll)
            if now - last_beat >= 60:
                print(f"  still absent after {int(now - started)} s", flush=True)
                last_beat = now
    except KeyboardInterrupt:
        print("interrupted")
        return 130
    print(f"{path}: present after {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
