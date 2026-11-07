#!/usr/bin/env python3
"""Read how full a subagent's context is, and block until it reaches a mark.

    python agent_context_watch.py <agent-id> --once
    python agent_context_watch.py <agent-id> --target-pct 50

A Claude Code orchestrator is not shown a subagent's context percentage — but
the agent's transcript
(`~/.claude/projects/<project>/<session>/subagents/agent-<id>.jsonl`) carries
it: every assistant message is logged with the API's `usage` record, and

    input_tokens + cache_creation_input_tokens + cache_read_input_tokens
        + output_tokens

of the LAST one is the context the agent is holding. Divided by the window
(`--window`, by default 1 000 000 tokens) it is the percentage the user's UI
shows. Checked against the UI on 2026-09-21:
the user read 39 % at 3533 s into a run that ended at 396 757 tokens at 3737 s,
and the harness's own `subagent_tokens` for that run was 399 455.

Read the transcript above and not the harness's own `tasks/<agent-id>.output`:
whatever that file holds at a given moment, it is the agent's whole JSONL and
reading it costs the ORCHESTRATOR the context it is trying to measure.

`--pct-now` is a CHECK, not an input the reading needs: pass the percentage the
user reads off the UI and the script prints the window that reading implies, so
a model with another window shows up as a disagreement instead of as a quietly
wrong percentage.

Why a script and not an `until ...; do sleep` loop: a shell gate (such as
llm-loop's `ask_user_gate.py`) rejects clock waits in a command line (see
`wait_file_gone.py`). Run it with the tool's `run_in_background`, so the
completion notification is the alarm. The last line names what ended the wait
(`target` / `timeout` / `gone` / `idle`).

What this replaced, so nobody rebuilds it: an estimate extrapolated from the
agent's AGE and its transcript SIZE, calibrated by one reading from the user.
Both lines over-read the second half of a run — the clock while the agent waits
on tests (82 s per percent early, 140 later), the size wildly once it reads
screenshots (base64 is bytes, not tokens: 66 % estimated against a true 39 %).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time

USAGE_AT = re.compile(rb'"usage":\{')
FIELDS = (b"input_tokens", b"cache_creation_input_tokens",
          b"cache_read_input_tokens", b"output_tokens")
# A usage record is ~300 bytes; the slice only has to reach past its own fields.
USAGE_SPAN = 600


def transcript_of(agent: str) -> str | None:
    """The newest transcript whose name carries this agent id (or the path itself)."""
    if os.path.exists(agent):
        return os.path.abspath(agent)
    pattern = os.path.join(
        os.path.expanduser("~"), ".claude", "projects", "*", "*", "subagents",
        f"agent-{agent}*.jsonl",
    )
    found = sorted(glob.glob(pattern), key=os.path.getmtime)
    return found[-1] if found else None


def tokens_of(record: bytes) -> int | None:
    """Context tokens of one usage record, or None if a field is missing."""
    total = 0
    for field in FIELDS:
        # Anchored on the quote so `output_tokens` cannot match inside
        # `output_tokens_details`, nor `input_tokens` inside the cache fields.
        hit = re.search(rb'"' + field + rb'":(\d+)', record)
        if hit is None:
            return None
        total += int(hit.group(1))
    return total


class Reader:
    """Last usage record of a growing transcript, read incrementally."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.offset = 0
        self.tokens: int | None = None
        self.carry = b""

    def poll(self) -> int | None:
        with open(self.path, "rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        self.offset += len(chunk)
        data = self.carry + chunk
        for hit in USAGE_AT.finditer(data):
            tokens = tokens_of(data[hit.start():hit.start() + USAGE_SPAN])
            if tokens is not None:
                self.tokens = tokens
        # A record cut in half by the read boundary is finished by the next one.
        self.carry = data[-USAGE_SPAN:]
        return self.tokens


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("agent", help="agent id (or a transcript path)")
    ap.add_argument("--window", type=int, default=1_000_000,
                    help="context window of the agent's model, tokens (default 1000000)")
    ap.add_argument("--pct-now", type=float,
                    help="the percentage the user's UI shows right now; printed "
                         "against the reading as the window it implies")
    ap.add_argument("--target-pct", type=float, default=50.0,
                    help="percentage that ends the wait (default 50)")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="seconds after which the wait ends regardless (default 3600)")
    ap.add_argument("--idle", type=float, default=900.0,
                    help="seconds without transcript growth that end the wait: the "
                         "agent finished or hangs (default 900)")
    ap.add_argument("--poll", type=float, default=10.0,
                    help="seconds between readings (default 10)")
    ap.add_argument("--once", action="store_true",
                    help="print the current reading and exit")
    args = ap.parse_args()

    path = transcript_of(args.agent)
    if path is None:
        print(f"gone: no transcript for {args.agent}")
        return 1
    reader = Reader(path)
    started = time.time()

    def line(tag: str) -> str:
        tokens = reader.tokens
        if tokens is None:
            return f"{tag}: no usage record in {path} yet"
        return (f"{tag}: {100 * tokens / args.window:.1f} % "
                f"({tokens} of {args.window} tokens)")

    reader.poll()
    if args.pct_now is not None and reader.tokens is not None and args.pct_now > 0:
        implied = reader.tokens * 100 / args.pct_now
        print(f"check: the UI's {args.pct_now:g} % against {reader.tokens} tokens "
              f"implies a window of {implied:.0f} (using {args.window})", flush=True)
    if args.once:
        print(line("now"))
        return 0
    try:
        while True:
            if not os.path.exists(path):
                print(f"gone: {path}")
                return 0
            tokens = reader.poll()
            if tokens is not None and 100 * tokens / args.window >= args.target_pct:
                print(line("target"))
                return 0
            if time.time() - os.path.getmtime(path) >= args.idle:
                print(line("idle"))
                return 0
            if time.time() - started >= args.timeout:
                print(line("timeout"))
                return 0
            time.sleep(args.poll)
    except KeyboardInterrupt:
        print(line("interrupted"))
        return 130


if __name__ == "__main__":
    sys.exit(main())
