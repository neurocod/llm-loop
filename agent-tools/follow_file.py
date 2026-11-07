#!/usr/bin/env python3
r"""Print a growing file (existing content first) until a pattern shows up or a timeout.

    python agent-tools/follow_file.py <task output file> --preset task --quiet --tail 80
    python agent-tools/follow_file.py server.log --until "listening on \d+" --timeout 120
    python agent-tools/follow_file.py run.txt --define-preset batch "=== summary ===" \
        --preset batch --timeout 580

What it is for: an LLM agent that has started a long job in the background
needs to wait for it, and the obvious ways to wait are the ones an agent harness
keeps stopping on. A `while ...; do sleep 1; done` line is refused by shell
gates such as this repository's ask-user-gate plugin, and a waiter script the
agent writes into a scratch directory asks the human for permission every time
it runs, because nobody ever approved that script. Agents wrote such waiters
again and again (a bash `for ... sleep 1` loop over `wc`/`sed`/`grep`, then two
Python variants), each one a prompt. One reviewed script that the human allows
once is the same wait without the prompt.

Siblings for the simpler cases: `wait_file_appears.py` (a file exists, or
matches once) and `wait_file_gone.py` (a stop file disappears).

Exit codes: 0 -- a pattern matched (and stayed matched for --settle polls);
1 -- --timeout ran out first; 130 -- interrupted. A file that does not exist yet
is waited for, not an error. Keep --timeout under the calling tool's foreground
cap (600 s for Claude Code's shell tool) so the wait ends by itself rather than
being killed.

Named end markers (`--preset`) save the caller from re-deriving a pattern per
run. A host project names its own with `--define-preset`, typically from a
wrapper script that passes them on every call, so the project's markers live in
the project rather than here.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

# The presets that are true wherever this script runs:
# - task: the Claude Code harness appends `[exited with code N]` to a background
#   task's output file (`Exit code:` in older trailers).
PRESETS: dict[str, list[str]] = {
    "task": [r"\[exited with code -?\d+\]", r"^Exit code:"],
}


def read_from(path: str, offset: int) -> tuple[str, int]:
    """Text appended since `offset` and the new offset; ("", offset) when absent or unchanged."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return "", offset
    if size < offset:
        offset = 0  # truncated or rewritten: start over rather than miss lines
    if size == offset:
        return "", offset
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
    # Hold back a trailing partial line: the writer may be mid-line.
    cut = data.rfind(b"\n") + 1
    if cut == 0:
        return "", offset
    return data[:cut].decode("utf-8", errors="replace"), offset + cut


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("path", help="file to follow (may not exist yet)")
    ap.add_argument("--until", action="append", default=[], metavar="REGEX",
                    help="stop once any line matches this regex (repeatable; multiline mode)")
    ap.add_argument("--preset", action="append", default=[], metavar="NAME",
                    help="add a named set of end patterns (repeatable); built in: "
                         + ", ".join(sorted(PRESETS)))
    ap.add_argument("--define-preset", nargs=2, action="append", default=[],
                    metavar=("NAME", "REGEX"),
                    help="add REGEX to the preset NAME, creating it if new (repeatable); "
                         "how a host project's wrapper supplies its own markers")
    ap.add_argument("--settle", type=int, default=1,
                    help="polls the match must persist before exiting, so trailing lines "
                         "after the marker are printed too (default 1)")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between reads (default 1)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="give up after this many seconds (exit 1)")
    ap.add_argument("--quiet", action="store_true",
                    help="do not stream lines; print only the last --tail lines at the end")
    ap.add_argument("--tail", type=int, default=0,
                    help="at exit, print the last N lines of the file (default 0)")
    args = ap.parse_args()

    presets = {name: list(patterns) for name, patterns in PRESETS.items()}
    for name, pattern in args.define_preset:
        presets.setdefault(name, []).append(pattern)
    unknown = [name for name in args.preset if name not in presets]
    if unknown:
        ap.error(f"unknown preset {', '.join(unknown)}; known: {', '.join(sorted(presets))}")

    patterns = [re.compile(p, re.MULTILINE) for p in args.until]
    for name in args.preset:
        patterns += [re.compile(p, re.MULTILINE) for p in presets[name]]
    if not patterns and args.timeout is None:
        ap.error("give --until/--preset, --timeout, or both: otherwise this never ends")

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    path = os.path.abspath(args.path)
    started = time.monotonic()
    last_beat = started
    offset = 0
    seen = ""          # everything read so far, for patterns spanning old lines
    matched_polls = 0
    code = 1
    try:
        while True:
            chunk, offset = read_from(path, offset)
            if chunk:
                seen += chunk
                if not args.quiet:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                last_beat = time.monotonic()
            if patterns and any(p.search(seen) for p in patterns):
                matched_polls += 1
                if matched_polls >= args.settle:
                    code = 0
                    break
            now = time.monotonic()
            if args.timeout is not None and now - started >= args.timeout:
                print(f"[follow_file] timeout after {now - started:.0f} s: no end marker in {path}",
                      flush=True)
                break
            if now - last_beat >= 60:
                print(f"[follow_file] no new lines for {int(now - last_beat)} s "
                      f"({int(now - started)} s total)", flush=True)
                last_beat = now
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print("[follow_file] interrupted")
        return 130
    if args.tail > 0:
        lines = seen.splitlines()
        print("\n".join(lines[-args.tail:]))
    return code


if __name__ == "__main__":
    sys.exit(main())
