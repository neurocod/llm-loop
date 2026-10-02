"""State timings reconstructed from the sequential runner's mirror log."""

from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

from . import console


def iteration_started(iteration: int, state: str) -> str:
    """Machine-only start record; JSON preserves arbitrary state labels."""
    return json.dumps({"event": "iteration_start", "iteration": iteration,
                       "state": _STATE_PREFIX.sub("", state).strip()
                       or "(no label)"}, ensure_ascii=False)


def iteration_finished(iteration: int, returncode: int,
                       elapsed_seconds: Optional[float] = None) -> str:
    """Close a timed iteration even when the provider failed or emitted no result.

    The runner prints this after the provider returns, before driver hooks and
    quota waits. Older logs only have provider result lines; those approximate
    the end and cannot measure the provider's shutdown after its last result.
    """
    elapsed = (f", elapsed {elapsed_seconds:.3f} s"
               if elapsed_seconds is not None else "")
    return f"=== Iteration {iteration} finished (exit {returncode}{elapsed}) ==="


_RECORD = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
_HEADER = re.compile(r"^=== Iteration (\d+) === \[(.*)\]$")
_FINISHED = re.compile(
    r"^=== Iteration (\d+) finished \(exit -?\d+"
    r"(?:, elapsed (\d+(?:\.\d+)?) s)?\) ===$")
_DONE = re.compile(r"^  · done(?: \(.*\))?$")
_FAILED = re.compile(
    r"^(?:Claude Code|Codex CLI) exited with code -?\d+ "
    r"\(error #\d+ in a row\)\.$")
_STATE_PREFIX = re.compile(r"^current state:\s*", re.IGNORECASE)
_TAGGED = re.compile(r"^pid=(\d+) kind=(output|timing) (.*)$")


@dataclass
class StateTiming:
    iterations: int = 0
    total_seconds: float = 0.0
    partial_iterations: int = 0


def _read_legacy_stats(lines: Iterable[str]) -> Dict[str, StateTiming]:
    """Sum observed iteration wall time, grouped by the labels in the log.

    A finished marker's monotonic duration is authoritative; timestamp-only
    markers and old logs measure wall time. The last provider `done`
    line ends the iteration. An unfinished iteration contributes only time
    observed through its last log line, and is marked partial, never extended
    to the next launch or the reader's current time. Quota/operator pauses and
    run epilogues close that fallback before their idle time can be charged.
    Rotation may leave orphan lines before the first header: ignore them.
    Percentages therefore describe recorded iteration time, not uptime.
    """
    timings: Dict[str, StateTiming] = {}
    state = None
    number = None
    started = last = ended = None

    def flush(explicit_end=None, elapsed_seconds=None):
        nonlocal state, number, started, last, ended
        if state is not None:
            end = explicit_end or ended or last
            row = timings.setdefault(state, StateTiming())
            row.iterations += 1
            # Local wall-clock stamps can move backwards after a clock reset.
            row.total_seconds += (elapsed_seconds if elapsed_seconds is not None
                                  else max(0.0, (end - started).total_seconds()))
            row.partial_iterations += int(explicit_end is None and ended is None)
        state = number = started = last = ended = None

    for line in lines:
        record = _RECORD.fullmatch(line.rstrip("\r\n"))
        if record is None:
            continue
        try:
            stamp = datetime.strptime(record[1], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        message = record[2]
        header = _HEADER.fullmatch(message)
        if header is not None:
            flush()
            number = int(header[1])
            label = header[2].rsplit(" · ", 1)[0]
            state = _STATE_PREFIX.sub("", label).strip() or "(no label)"
            started = last = stamp
            continue
        finished = _FINISHED.fullmatch(message)
        if finished is not None and int(finished[1]) == number:
            flush(stamp, float(finished[2]) if finished[2] is not None else None)
            continue
        if _FAILED.fullmatch(message):
            flush(stamp)
            continue
        # Only top-level runner messages count. A tool's numbered source dump,
        # quote or command containing these words must not split an iteration.
        if (message.startswith(("  · project root:", "Final state:",
                                "=== run ended:", "  ⏸ ",
                                "  ⚠ the previous run left no exit record",
                                "  · another "))
                or message.startswith("  ⏳ ")):
            flush()
            continue
        if state is not None:
            last = stamp
            if _DONE.fullmatch(message):
                ended = stamp
    flush()
    return timings


@dataclass
class _ActiveIteration:
    number: int
    state: str
    started: datetime
    last: datetime


def read_stats(lines: Iterable[str]) -> Dict[str, StateTiming]:
    """Read tagged timing events per pid, with best-effort legacy fallback.

    New mirror records distinguish machine timings from ordinary stdout, so
    agent prose cannot end an iteration. Each process has its own cursor; a
    killed process contributes observed time only, without charging downtime
    before another launch. Old untagged logs cannot disambiguate concurrent
    writers or exact copies of runner messages in agent output.
    """
    timings: Dict[str, StateTiming] = {}
    active: Dict[str, _ActiveIteration] = {}

    def flush(pid, seconds=None):
        item = active.pop(pid, None)
        if item is None:
            return
        row = timings.setdefault(item.state, StateTiming())
        row.iterations += 1
        row.total_seconds += (seconds if seconds is not None else
                              max(0.0, (item.last - item.started).total_seconds()))
        row.partial_iterations += int(seconds is None)

    def legacy_lines():
        for line in lines:
            record = _RECORD.fullmatch(line.rstrip("\r\n"))
            tagged = _TAGGED.fullmatch(record[2]) if record else None
            if tagged is None:
                yield line
                continue
            try:
                stamp = datetime.strptime(record[1], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            pid, kind, message = tagged.groups()
            if kind == "output":
                if pid in active:
                    active[pid].last = stamp
                continue
            try:
                event = json.loads(message)
            except ValueError:
                event = None
            if (isinstance(event, dict) and event.get("event") == "iteration_start"
                    and isinstance(event.get("iteration"), int)
                    and isinstance(event.get("state"), str)):
                flush(pid)
                active[pid] = _ActiveIteration(event["iteration"], event["state"],
                                                stamp, stamp)
                continue
            finished = _FINISHED.fullmatch(message)
            if (finished is not None and pid in active
                    and int(finished[1]) == active[pid].number):
                seconds = (float(finished[2]) if finished[2] is not None else
                           max(0.0, (stamp - active[pid].started).total_seconds()))
                flush(pid, seconds)

    legacy = _read_legacy_stats(legacy_lines())
    for pid in list(active):
        flush(pid)
    for state, row in legacy.items():
        merged = timings.setdefault(state, StateTiming())
        merged.iterations += row.iterations
        merged.total_seconds += row.total_seconds
        merged.partial_iterations += row.partial_iterations
    return timings


def _clock(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def report_stats(app_name: str = "runCycle",
                 path: Optional[Union[str, Path]] = None) -> None:
    """Read a log without modifying it and print totals and per-iteration means."""
    path = Path(path) if path is not None else console.log_file_path(app_name)
    print(f"Reading mirror log: {path}")
    try:
        # The tee splits only on LF; an agent's embedded CR is still data.
        # Universal-newline mode would turn that data into a new timing record.
        with open(path, encoding="utf-8", errors="replace", newline="\n") as source:
            timings = read_stats(source)
    except FileNotFoundError:
        print(f"No mirror log at {path} yet — nothing to report.")
        return
    if not timings:
        print("No timestamped iteration headers in this log — nothing to report.")
        return

    total = sum(row.total_seconds for row in timings.values())
    count = sum(row.iterations for row in timings.values())
    rows = [("State", "Iterations", "Total", "Average", "% total")]
    for state, row in timings.items():
        percent = row.total_seconds / total * 100 if total else 0.0
        rows.append((state, str(row.iterations), _clock(row.total_seconds),
                     _clock(row.total_seconds / row.iterations), f"{percent:.1f}%"))
    rows.append(("TOTAL", str(count), _clock(total), _clock(total / count),
                 "100.0%" if total else "0.0%"))
    widths = [max(len(row[i]) for row in rows) for i in range(5)]
    for index, row in enumerate(rows):
        if index in (1, len(rows) - 1):
            print("-+-".join("-" * width for width in widths))
        print(" | ".join(value.ljust(width) if i == 0 else value.rjust(width)
                         for i, (value, width) in enumerate(zip(row, widths))))
    partial = sum(row.partial_iterations for row in timings.values())
    print("Times: HH:MM:SS; recorded iteration time excludes waits between iterations.")
    if partial:
        print(f"Includes {partial} partial iteration(s), measured through their last log line.")
