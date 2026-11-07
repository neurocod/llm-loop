#!/usr/bin/env python3
r"""Print the machine's CPU load and the heaviest processes (Windows).

    python cpu_load.py            # load % + top 8 processes by CPU
    python cpu_load.py --top 15 --match "node|chrome|cl|MSBuild"
    python cpu_load.py --json      # the same selection, as load/procs JSON
    python cpu_load.py --json --own-tree 1234 --find myserver
    python cpu_load.py --json --stream --own-tree 1234   # a line per pair, until killed
    python cpu_load.py --json --own-tree 1234 --orphan-image node --orphan-image chrome

`--own-tree PID` keeps PID and every process it started (by `CreatingProcessID`,
transitively, checked against start times: `own_tree`) out of the ranking and
reports their share of the machine, the sampling's own cost included, as `own`,
and the share of processes positively known to be outside it as `others`.
Processes of an `--orphan-image` started during the run whose creator has
disappeared are `unattributed`, including their descendants: they may be this
run's orphans. Without any `--orphan-image`, `unattributed` is always 0.
`--find REGEX` lists every process whose WHOLE name matches, at any CPU,
outside that tree, as `found`. `--json` also names the span every figure
averages over, `fromMs`/`toMs` (`interval_ms`).

Why: a red test or a slow benchmark read while another agent builds or tests on
the same machine is not yet a defect (4-5 concurrent agents slowed one pytest
run from 57 s to 400-800 s), so agents check the load before judging -- and
kept writing their own `*.ps1` for it into a scratch directory, which asks the
human for permission on every run. This is that check as a reviewed tool.

How: a one-shot call is ONE PowerShell call (`QUERY`) that takes two RAW
perf-counter samples SAMPLE_GAP_MS apart; `--stream` is one PowerShell loop
that takes a raw sample every SAMPLE_GAP_MS and answers each consecutive pair
(`stream_query`). Either way this script does the arithmetic (`all_rates`), so
every figure is the average over its pair -- the machine's load as busy share
of all logical CPUs, each process's as % of ONE core (a busy multi-threaded process can read
above 100). Not the formatted counters: their `name#N` instances renumber as
processes of one name come and go, and one live call printed `node#4 4900%` on
16 logical CPUs (2026-11-03). Raw samples are keyed by process id instead.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import threading

# Long enough for a counter that ticks in 100 ns to average over several
# scheduler quanta; about half of a quiet call (QUERY_TIMEOUT_S has the timings).
# A plain integer constant on purpose: a caller in another language may mirror
# it and pin the mirror against this line.
SAMPLE_GAP_MS = 700

# `$take`: one raw sample of every process's counters, the projection both
# schedules read (`QUERY`'s two samples, `stream_query`'s loop) and `all_rates`,
# `own_tree` and `interval_ms` take apart.
TAKE = (
    "$take = { @(Get-CimInstance Win32_PerfRawData_PerfProc_Process | "
    "Select-Object Name, IDProcess, CreatingProcessID, ElapsedTime, PercentProcessorTime, "
    "Timestamp_Sys100NS) }; "
)

QUERY = (
    TAKE +
    f"$first = & $take; Start-Sleep -Milliseconds {SAMPLE_GAP_MS}; $second = & $take; "
    # $PID: the query's own PowerShell, which would otherwise rank itself.
    "@{ cpus = [Environment]::ProcessorCount; self = $PID; first = $first; second = $second } "
    "| ConvertTo-Json -Depth 3 -Compress"
)

# The WMI provider host serves the query itself and ranked in every live call.
NOT_RANKED = re.compile(r"^WmiPrvSE$", re.IGNORECASE)

# Full `--json` calls (two raw samples) measured 1.463 / 1.478 / 1.443 s on
# 2026-11-03 at 19-27 % CPU. No loaded figure for this query yet; the earlier
# one-sample query took up to 8.0 s at 70-77 % CPU, so 20 s stays -- over 13
# times the slowest quiet call, still twice a loaded one plus SAMPLE_GAP_MS.
QUERY_TIMEOUT_S = 20


def _base_name(name: str) -> str:
    """`node#4` -> `node`: the instance suffix is a slot, not an identity."""
    return re.sub(r"#\d+$", "", name or "")


def image_key(name: str) -> str:
    """How a process name is compared with an `--orphan-image`: base name,
    case-insensitive, without `.exe`."""
    key = _base_name(name).lower()
    return key[:-4] if key.endswith(".exe") else key


def _delta(old: dict, new: dict) -> tuple[float, float]:
    """(window, CPU time used in it), both in 100 ns, between two raw rows of one instance."""
    return (new["Timestamp_Sys100NS"] - old["Timestamp_Sys100NS"],
            new["PercentProcessorTime"] - old["PercentProcessorTime"])


def _rows(data: dict, key: str) -> list:
    """PowerShell serializes a single row as an object rather than an array."""
    rows = data.get(key) or []
    return [rows] if isinstance(rows, dict) else rows


def rates(procs: list, exclude_pids: set) -> list:
    """The RANKED processes' % of one core: `all_rates`'s processes
    without `exclude_pids` (the query's own PowerShell, the caller) and without
    NOT_RANKED (the WMI provider serving the query)."""
    return [p for p in procs
            if p["IDProcess"] not in exclude_pids and not NOT_RANKED.match(p["Name"])]


def all_rates(data: dict, cpus: int) -> tuple[int | None, list]:
    """Machine load % and per-process % of one core from two raw samples.

    A process is kept only when it is in BOTH samples under the same pid and the
    same base name (a pid reused inside the gap is a different process). A rate
    above 100 x logical CPUs is a counter anomaly, not a process, and is dropped.
    Every other process is in, the sampler's own included: the machine load
    counts them, so a share held against it must too (`own_share`).
    """
    first, second = _rows(data, "first"), _rows(data, "second")

    def idle(sample):
        return next((p for p in sample if p.get("Name") == "Idle"), None)

    load = None
    a, b = idle(first), idle(second)
    if a and b:
        span, idle_used = _delta(a, b)
        if span > 0:
            idle_share = idle_used / (span * cpus)
            load = round(min(100.0, max(0.0, 100.0 * (1.0 - idle_share))))

    # `_Total` and `Idle` both carry pid 0 (live query 2026-11-03), so leaving
    # them out of `before` leaves them out of the ranking too.
    before = {p["IDProcess"]: p for p in first
              if p.get("Name") not in ("_Total", "Idle") and p.get("IDProcess") is not None}
    procs = []
    for p in second:
        pid = p.get("IDProcess")
        if pid not in before:
            continue
        base = _base_name(p.get("Name"))
        old = before[pid]
        if _base_name(old.get("Name")) != base:
            continue
        span, used = _delta(old, p)
        if span <= 0 or used < 0:
            continue
        percent = 100.0 * used / span
        if percent > 100.0 * cpus:
            continue
        procs.append({"Name": base, "IDProcess": pid, "PercentProcessorTime": round(percent)})
    return load, procs


# FILETIME (100 ns since 1601-01-01 UTC) of the Unix epoch.
FILETIME_EPOCH = 116_444_736_000_000_000


def interval_ms(data: dict) -> tuple[int, int] | None:
    """The span the load is averaged over, as Unix-epoch ms: the `Idle` row's
    `Timestamp_Sys100NS` in each sample, or None without an Idle pair.

    That stamp is a UTC FILETIME (live query 2026-11-07, UTC+2 machine:
    134384962825759399 against `[DateTime]::UtcNow.ToFileTimeUtc()`
    134384962826653699 a call later), so it is on the clock of a caller's
    `Date.now()` or `time.time()`. Why it is reported: a caller that judges a
    stretch of time should count a sample only when THIS span lies inside the
    stretch -- the query starts ~0.7 s after it is asked and ends later still,
    so the asking time says nothing about which moments it saw.
    """
    first, second = _rows(data, "first"), _rows(data, "second")
    a = next((p for p in first if p.get("Name") == "Idle"), None)
    b = next((p for p in second if p.get("Name") == "Idle"), None)
    if not (a and b):
        return None
    return (int((a["Timestamp_Sys100NS"] - FILETIME_EPOCH) // 10_000),
            int((b["Timestamp_Sys100NS"] - FILETIME_EPOCH) // 10_000))


def own_tree(data: dict, root: int, not_parents: frozenset = frozenset(),
             *, unattributed: set | None = None,
             orphan_images: frozenset = frozenset()) -> set:
    """`root` and every process it started, transitively, in either sample.

    The parent link is `CreatingProcessID`, the pid of the creator AT CREATION:
    a process whose creator died keeps that pid, and Windows hands pids out
    again, so the link alone lets a recycled pid in this tree adopt a stranger
    -- another agent's test runner or server, whose load the caller then does
    not see. A link counts only when the parent that holds the pid NOW started
    no later than the child: the raw `ElapsedTime` of
    `Win32_PerfRawData_PerfProc_Process` is the process's start as a FILETIME
    (live query 2026-11-04: `System` 134380979226678505, `csrss`
    134380979327102747, 10 s later at boot), so a pid reused after the child's
    start fails the comparison. A row without it is no parent and no child:
    an unverifiable link cannot establish ownership.
    Pid 0 (`Idle`/`_Total`) is never a parent: it would adopt the whole machine;
    neither is anything in `not_parents` -- the sampler's own processes, whose
    one child (the query's PowerShell) is in `not_parents` itself, so any other
    link through their pids is a stale one.

    When requested, `unattributed` receives the other possible own processes:
    an orphan whose image is in `orphan_images` (`image_key` form) created since
    `root` started, and all its verified descendants. Other orphan images
    remain external.
    A short-lived cmd/npm creator can disappear before BOTH samples, leaving
    no link back to root. Counting its child as external would excuse this
    run's own regression. A recycled creator pid is equally unverifiable.
    An orphan predating root cannot belong to the run, so remains external;
    missing start times cannot prove that and are treated as uncertain.
    """
    rows = _rows(data, "first") + _rows(data, "second")
    started = {p["IDProcess"]: p["ElapsedTime"] for p in rows
               if p.get("IDProcess") and isinstance(p.get("ElapsedTime"), (int, float))}
    children: dict = {}
    for p in rows:
        pid, parent = p.get("IDProcess"), p.get("CreatingProcessID")
        if (pid and parent and pid != parent and parent not in not_parents
                and pid in started and parent in started and started[parent] <= started[pid]):
            children.setdefault(parent, set()).add(pid)
    tree, todo = {root}, [root]
    while todo:
        for child in children.get(todo.pop(), ()):
            if child not in tree:
                tree.add(child)
                todo.append(child)
    if unattributed is not None:
        root_start = started.get(root)
        for p in rows:
            pid, parent = p.get("IDProcess"), p.get("CreatingProcessID")
            if not pid or pid in tree or pid in not_parents:
                continue
            if image_key(p.get("Name")) not in orphan_images:
                continue
            start = started.get(pid)
            if root_start is None or start is None:
                unattributed.add(pid)
            elif start >= root_start and not (
                    parent and parent != pid and parent not in not_parents
                    and parent in started and started[parent] <= start):
                unattributed.add(pid)
        todo = list(unattributed)
        while todo:
            for child in children.get(todo.pop(), ()):
                if child not in tree and child not in unattributed:
                    unattributed.add(child)
                    todo.append(child)
    return tree


def own_share(procs: list, tree: set, cpus: int) -> int:
    """The share of the whole machine, %, of `tree` and of the sampling itself.

    `procs` is `all_rates`'s, unfiltered: the sampler's python and PowerShell
    and the WMI provider serving the query (NOT_RANKED) are left out of the
    RANKING, but the machine load counts them, so a share left out of `own`
    reads as a neighbour's -- about 2 points of the machine in live calls.
    The provider is not in any tree (a service host starts it), and counted
    here as the query's cost.
    """
    return round(sum(p["PercentProcessorTime"] for p in procs
                     if p["IDProcess"] in tree or NOT_RANKED.match(p["Name"])) / cpus)


def others_share(procs: list, tree: set, cpus: int) -> int:
    """The share of the whole machine, %, of processes POSITIVELY known to be
    outside `tree` (pass the sampler's and unattributed pids in it): in both samples (`all_rates`
    keeps no other), not in the tree, not the WMI provider (NOT_RANKED, the
    query's cost, in `own_share`).

    Not `load - own`: a process absent from the first sample -- a child the
    caller's run spawned inside the gap, whose tree link `own_tree` sees but
    whose rate `all_rates` cannot take -- is in the load and in neither share,
    and `load - own` would call it a neighbour's, which would excuse the run's
    own slowness as somebody else's load.
    """
    return round(sum(p["PercentProcessorTime"] for p in procs
                     if p["IDProcess"] not in tree and not NOT_RANKED.match(p["Name"])) / cpus)


def found(data: dict, pattern: re.Pattern, tree: set) -> list:
    """Every process of the second sample whose whole base name matches, outside `tree`."""
    return [{"Name": _base_name(p.get("Name")), "IDProcess": p.get("IDProcess")}
            for p in _rows(data, "second")
            if p.get("IDProcess") and p.get("IDProcess") not in tree
            and pattern.fullmatch(_base_name(p.get("Name")))]


class QueryError(Exception):
    """A raw query this script cannot turn into an answer; the message is the stderr line."""


def answer(data: dict, args: argparse.Namespace) -> tuple[int, list, tuple[int, int] | None, dict]:
    """(load, ranked procs, span, own/others/unattributed/found) of one raw
    first/second pair, or QueryError: what `main` prints, once for a call and
    once per pair of `--stream`."""
    try:
        sampler = frozenset({data.get("self"), os.getpid()})
        cpus = data.get("cpus") or os.cpu_count() or 1
        load, all_procs = all_rates(data, cpus)
        span = interval_ms(data)
        procs = rates(all_procs, sampler)
        unattributed = set()
        images = frozenset(image_key(name) for name in args.orphan_image)
        tree = (own_tree(data, args.own_tree, sampler, unattributed=unattributed,
                         orphan_images=images)
                if args.own_tree is not None else set())
        extra = {}
        if args.own_tree is not None:
            extra["own"] = own_share(all_procs, tree | sampler, cpus)
            extra["others"] = others_share(all_procs, tree | sampler | unattributed, cpus)
            extra["unattributed"] = round(sum(p["PercentProcessorTime"] for p in all_procs
                if p["IDProcess"] in unattributed and not NOT_RANKED.match(p["Name"])) / cpus)
            procs = [p for p in procs if p["IDProcess"] not in tree | unattributed]
        if args.find is not None:
            extra["found"] = found(data, re.compile(args.find, re.IGNORECASE), tree)
    except (AttributeError, KeyError, TypeError) as exc:
        raise QueryError(f"CPU query returned invalid JSON: {exc!r}") from exc
    if load is None:
        raise QueryError("CPU query returned no usable Idle counter pair")
    if args.match:
        pattern = re.compile(args.match, re.IGNORECASE)
        procs = [p for p in procs if pattern.search(p["Name"])]
    procs.sort(key=lambda p: p["PercentProcessorTime"], reverse=True)
    return load, procs, span, extra


def json_line(load: int, procs: list, span: tuple[int, int] | None, extra: dict, top: int) -> str:
    return json.dumps({"load": load, "procs": procs[:top],
                       "fromMs": span[0] if span else None, "toMs": span[1] if span else None, **extra})


# `--stream` starts a raw sample every SAMPLE_GAP_MS, or as soon as the last
# one is out when its enumeration took longer (~250 ms at 100 % load from
# vitest, 8 reads 213-464 ms, 2026-11-07). Each snapshot is both the second of
# one pair and the first of the next: the spans TILE the stream's life with no
# gap, which is what a caller judging a stretch of time needs -- a stretch
# holds a whole pair once it is twice the pair long, so a 5 s stretch does up
# to 2.5 s pairs. The one-shot call cannot give that: a 60 s window of such
# calls every 3-4 s left 15 % of its 5 s stretches without a whole sample
# inside (2026-11-07, others 80-98 %).
#
# Two measured facts for a reader's budgets, plain integer constants so a
# mirror in another language can be pinned against them: the longest pair, ms
# -- 1.6-2.0 s under a neighbour's cl.exe build (2026-11-07) -- and the slowest
# start, ms from the spawn to the first read: PowerShell's start alone varied
# 0.9-6.3 s under load (2026-11-07, others 80-98 %). This script uses neither.
STREAM_PAIR_MS = 2_000
STREAM_START_MS = 6_300


def stream_query(caller: int) -> str:
    """The PowerShell loop of `--stream`. It runs while `caller` (this script's
    pid) does, so a Python killed without its tree leaves no PowerShell
    enumerating every second for nobody. "Does" is the pid AND the start time
    it had when the loop began: Windows hands pids out again, and a stranger
    holding the pid would keep the orphan going."""
    alive = f"(Get-Process -Id {caller} -ErrorAction SilentlyContinue).StartTime"
    return (
        TAKE +
        f"$born = {alive}; "
        f"while ($born -and {alive} -eq $born) {{ "
        "$clock = [Diagnostics.Stopwatch]::StartNew(); "
        "@{ cpus = [Environment]::ProcessorCount; self = $PID; rows = & $take } "
        "| ConvertTo-Json -Depth 3 -Compress; "
        f"$left = {SAMPLE_GAP_MS} - $clock.ElapsedMilliseconds; "
        "if ($left -gt 0) { Start-Sleep -Milliseconds $left } }"
    )


# How many of PowerShell's last stderr lines `--stream` keeps for its failure
# message: the error that ended it and a few before, not a run's worth of
# non-terminating CIM errors.
STDERR_TAIL_LINES = 20
# Bound even a newline-free error before it reaches the tail. Keep a useful
# diagnostic suffix per line, with a separate total budget for long lines.
# These are character limits (Python Unicode uses at most four bytes each).
STDERR_READ_CHARS = 1024
STDERR_LINE_CHARS = 2048
STDERR_TAIL_CHARS = 8192
# Join only after killing and reaping the writer, which releases its pipe.
# A stuck reader must not hold shutdown forever. No timing sample yet: the
# former 5 s bound, raised to max(T + 10 s, T * 2).
STDERR_JOIN_S = 15


def _drain(pipe, tail: collections.deque) -> None:
    """Read `pipe` to its end, keeping the last lines in `tail`. Unread, a run
    of non-terminating `Get-CimInstance` errors fills the pipe (a few KB) and
    blocks PowerShell in its write -- the stream then goes silent, not dead."""
    continued = False
    while True:
        chunk = pipe.readline(STDERR_READ_CHARS)
        if not chunk:
            break
        if not continued:
            tail.append("")
        tail[-1] = (tail[-1] + chunk.rstrip("\r\n"))[-STDERR_LINE_CHARS:]
        while sum(map(len, tail)) > STDERR_TAIL_CHARS:
            tail.popleft()
        continued = not chunk.endswith("\n")


def _stdout_to_devnull() -> None:
    """Point stdout at the null device after its reader went away: the exit
    flush of what is still buffered would otherwise fail again and make the
    interpreter exit 120 instead of `stream`'s 0 (Python's own recipe for
    SIGPIPE, `signal` docs)."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):
        pass


def stream(args: argparse.Namespace) -> int:
    """`--stream`: one JSON line per pair of consecutive raw samples, until the
    reader goes away (a broken pipe) or PowerShell ends. A reader that is done
    kills this process's tree, or closes the pipe."""
    query = stream_query(os.getpid())
    try:
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", query],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
        )
    except OSError as exc:
        print(f"error: CPU query failed: {exc}", file=sys.stderr)
        return 1
    tail: collections.deque = collections.deque(maxlen=STDERR_TAIL_LINES)
    drain = threading.Thread(target=_drain, args=(proc.stderr, tail), daemon=True)
    drain.start()
    try:
        previous = None
        for line in proc.stdout:
            if not line.strip():
                continue
            try:
                current = json.loads(line)
                rows = current["rows"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"error: CPU query returned invalid JSON: {exc!r}", file=sys.stderr)
                return 1
            if previous is not None:
                data = {"cpus": current.get("cpus"), "self": current.get("self"),
                        "first": previous, "second": rows}
                try:
                    print(json_line(*answer(data, args), args.top), flush=True)
                except QueryError as exc:
                    print(f"error: {exc}", file=sys.stderr)
                    return 1
            previous = rows
    except (BrokenPipeError, OSError):
        _stdout_to_devnull()
        return 0
    finally:
        try:
            proc.kill()
        finally:
            try:
                proc.wait()
            finally:
                drain.join(STDERR_JOIN_S)
    said = "\n".join(list(tail)).strip()
    print(f"error: CPU query failed:\n{said}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--top", type=int, default=8, help="processes to list (default 8)")
    ap.add_argument("--json", action="store_true", help="emit load/procs JSON instead of the table")
    ap.add_argument("--stream", action="store_true",
                    help="with --json: one JSON line per consecutive pair of samples, until killed (`stream`)")
    ap.add_argument("--match", default=None,
                    help="only processes whose name matches this regex (case-insensitive)")
    ap.add_argument("--own-tree", type=int, default=None, metavar="PID",
                    help="report PID and its descendants as `own`, not in the ranking")
    ap.add_argument("--orphan-image", action="append", default=[], metavar="NAME",
                    help="with --own-tree, repeatable: an orphan of this image (case-insensitive, "
                         ".exe optional) started during the run counts as `unattributed`, "
                         "not as outside load; with none, `unattributed` is always 0")
    ap.add_argument("--find", default=None, metavar="REGEX",
                    help="list processes whose whole name matches (case-insensitive), "
                         "outside --own-tree, as `found`")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.stream and not args.json:
        print("error: --stream needs --json", file=sys.stderr)
        return 2
    if sys.platform != "win32":
        print("error: CPU query requires Windows", file=sys.stderr)
        return 1
    if args.stream:
        return stream(args)
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", QUERY],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=QUERY_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"error: CPU query failed: {exc}", file=sys.stderr)
        return 1
    if proc.returncode != 0 or not proc.stdout.strip():
        print(f"error: CPU query failed:\n{proc.stderr.strip()}", file=sys.stderr)
        return 1
    try:
        load, procs, span, extra = answer(json.loads(proc.stdout), args)
    except json.JSONDecodeError as exc:
        print(f"error: CPU query returned invalid JSON: {exc!r}", file=sys.stderr)
        return 1
    except QueryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json_line(load, procs, span, extra, args.top))
        return 0
    print(f"CPU load: {load} % (averaged over {SAMPLE_GAP_MS} ms; processes in % of one core)")
    if "own" in extra:
        print(f"  own tree of pid {args.own_tree}: {extra['own']} % of the machine, not ranked; "
              f"outside it: {extra['others']} %; unattributed: {extra['unattributed']} %")
    for p in procs[:args.top]:
        print(f"  {p['PercentProcessorTime']:>5}  {p['IDProcess']:>7}  {p['Name']}")
    for p in extra.get("found", []):
        print(f"  found: {p['Name']} (pid {p['IDProcess']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
