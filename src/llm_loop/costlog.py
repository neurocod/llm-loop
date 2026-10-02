"""costlog.py - what a run cost, read back out of the mirror log.

Sequential runs delimit sessions with iteration_header and print successful
Claude costs through done_line. Pool runs write pool_started/pool_result machine
records, since concurrent workers cannot share a sequential session cursor.

Those two lines are therefore a CONTRACT between three modules — the two that
print them and the one that parses them — and this module is where it lives:
the printers build the lines with `iteration_header` and `done_line`, and the
patterns below are written against those two functions rather than against a
copy of their output in a comment. That is the reason for a module of its own
rather than a home in either printer: while the patterns sat in `cyclecore`
the "done" line had already moved to `streamrender`, and nothing but a
doc-comment tied the two together. tests/test_cost_log.py renders both lines
through the real printers and parses them back, which is the check a comment
could not be.

Not `console`: that module writes the log and knows nothing of what the lines
in it mean; this one is handed `console.log_file_path` / `LOG_MAX_BYTES` and
reads, the same shape as `exitlog` writing its own file beside the mirror.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional, Union

from . import console


# --- the two lines, as printed -----------------------------------------------

def iteration_header(iteration: int) -> str:
    """"=== Iteration N ===" — the head of an iteration's banner line.

    Iteration 1's is a RUN boundary for `report_costs`; the caller adds the
    separator and the state/model label around it.
    """
    return f"=== Iteration {iteration} ==="


def result_suffix(duration_ms: Optional[float], cost: Optional[float]) -> str:
    """" (12.0 c, $0.5000)" — a Claude turn's duration and dollar cost.

    Shared by the "done" line and the failed-result line, though only the first
    is summed: a failed turn is not a completed billed iteration. Either figure
    may be missing, and a line without BOTH is not counted (`_COST_RE`).
    """
    bits = []
    if duration_ms is not None:
        bits.append(f"{duration_ms / 1000:.1f} c")
    if cost is not None:
        bits.append(f"${cost:.4f}")
    return f" ({', '.join(bits)})" if bits else ""


def done_line(duration_ms: Optional[float], cost: Optional[float]) -> str:
    """The line a successful Claude turn prints — the one `report_costs` sums.

    Codex prints "· done (tokens: …)" instead, which `_COST_RE` deliberately
    does not match: codex reports tokens, not dollars, so a codex run has no
    per-session spend to total up.
    """
    return f"  · done{result_suffix(duration_ms, cost)}"


# --- reading them back ---------------------------------------------------------

# Built from the printer itself, so a re-worded header cannot leave it behind.
# The trailing " ===" is what keeps iteration 1 from matching 11, 12, …, so each
# match is a genuine run boundary.
_SESSION_RE = re.compile(re.escape(iteration_header(1)))
# Hand-written, because it has to capture the dollar figure out of a formatted
# number; the round trip through `done_line` is pinned by tests instead.
_COST_RE = re.compile(r"done \(\s*[\d.]+ c,\s*\$([\d.]+)\)")
_MACHINE_RE = re.compile(r"^\S+ \S+ pid=(\d+) kind=timing (.*)$")


def pool_started(session: str) -> str:
    """A distinct pool invocation, independent of sequential promotion headers."""
    return json.dumps({"event": "pool_start", "session": session})


def pool_result(worker: int, duration_ms: Optional[float],
                cost: Optional[float]) -> str:
    """A successful turn's delta, with the same two-figure rule as done_line.

    Worker identity is scoped to its pid and most recent pool iteration start.
    Human worker summaries are never counted, including failed/retried jobs.
    """
    return json.dumps({"event": "pool_cost", "worker": worker,
                       "elapsed_ms": duration_ms, "cost": cost})


def named_log(path: Union[str, "os.PathLike[str]"]) -> Path:
    """The --cost-log path as a Path, refusing one that names no log file.

    Raises ValueError for a blank string and for anything that normalises to
    the current directory: `Path("")` IS `Path(".")`, so once an empty string
    has been wrapped in a Path the only trace left of it is ".", and reading
    "." fails later as a PermissionError after the report has already printed
    its "Reading mirror log: ." header. A directory is never a log, so a
    literal "." is refused with it.
    """
    text = os.fspath(path)
    if not text.strip() or Path(text) == Path("."):
        raise ValueError(f"--cost-log needs a log file, got {path!r}")
    return Path(text)


def report_costs(app_name: str = "runCycle",
                 path: Optional[Union[str, Path]] = None) -> None:
    """Print per-session (per-run) cost totals parsed from the mirror log, then
    exit — the standalone counterpart reached via the --cost flag.

    A sequential session starts at its "=== Iteration 1 ===" header; a pool
    session starts at pool_started and scopes its workers by pid/run ID. Each
    successful turn with both duration and cost contributes its dollar cost. We
    print a line per session, a grand total, and how full the log is against the
    rotation limit (LOG_MAX_BYTES). With no `path`, the log is resolved via
    log_file_path(app_name), so --cost reports on the very log this entry point
    writes — under the project root already chosen by --project-dir.

    `path` (the --cost-log flag) names a log this entry point does NOT write:
    a rotated backup (`<app>-<project>.log.1`) or a copy taken elsewhere. It is
    the one case app_name cannot reach, since rotation renames files out from
    under log_file_path.

    A `path` that names no log (`named_log`) raises ValueError instead of
    meaning "no path": the caller named a log, and reporting on a different one
    would look like an answer.
    """
    path = (named_log(path) if path is not None
            else console.log_file_path(app_name))
    # Always name the log we are reading, so an empty report is unambiguous
    # (right file, no data) rather than looking like a silent failure.
    print(f"Reading mirror log: {path}")
    sessions = []  # mutable rows: [header, total_cost, count]
    sequential = None
    pools = {}
    workers = {}

    try:
        with open(path, encoding="utf-8", errors="replace", newline="\n") as f:
            for line in f:
                machine = _MACHINE_RE.fullmatch(line.rstrip("\r\n"))
                if machine is not None:
                    try:
                        event = json.loads(machine[2])
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    pid = machine[1]
                    session = event.get("session")
                    worker = event.get("worker")
                    kind = event.get("event")
                    if kind == "pool_start" and isinstance(session, str):
                        row = [line.strip(), 0.0, 0]
                        pools[pid, session] = row
                        sessions.append(row)
                    elif (kind == "pool_iteration_start"
                          and isinstance(session, str) and isinstance(worker, int)):
                        workers[pid, worker] = pools.get((pid, session))
                    elif kind == "pool_cost" and isinstance(worker, int):
                        row = workers.get((pid, worker))
                        cost = event.get("cost")
                        duration = event.get("elapsed_ms")
                        if (row is not None and isinstance(cost, (int, float))
                                and isinstance(duration, (int, float))):
                            # Match sequential done_line's four-decimal rounding.
                            row[1] += float(f"{cost:.4f}")
                            row[2] += 1
                    continue
                if _SESSION_RE.search(line):
                    sequential = [line.strip(), 0.0, 0]
                    sessions.append(sequential)
                else:
                    m = _COST_RE.search(line)
                    if m and sequential is not None:
                        sequential[1] += float(m.group(1))
                        sequential[2] += 1
    except FileNotFoundError:
        print(f"No mirror log at {path} yet — nothing to report.")
        return

    grand = 0.0
    grand_count = 0
    for i, (h, t, c) in enumerate(sessions, 1):
        print(f"Session {i}: {c} costs, ${t:.4f}  | {h}")
        grand += t
        grand_count += c

    print("-" * 60)
    print(f"TOTAL: {len(sessions)} sessions, {grand_count} costs, ${grand:.4f}")
    if not sessions:
        # The log exists but held no run boundaries / cost lines. Point at the
        # likely cause rather than leaving a bare zero.
        print(f"  (log has no '{iteration_header(1)}' / '· done (… c, $…)' "
              "lines — no completed billed iterations recorded here)")

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    limit = console.LOG_MAX_BYTES
    pct = size / limit * 100 if limit else 0.0
    print(f"LOG: {size / 1024 / 1024:.2f} / {limit / 1024 / 1024:.0f} MB "
          f"({pct:.1f}% full, rotates at 100%)")
    print(f"     {path}")
