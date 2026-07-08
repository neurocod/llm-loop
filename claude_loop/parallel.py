"""
parallel.py - parallel sibling of the sequential ListFileDriver loop.

Processes the files listed in a ListFileDriver's list, but with N worker threads
running `claude` concurrently instead of one file at a time. The work
parallelises cleanly because each item is independent: a worker reads its own
source and writes its own target; the only shared mutable state is the list file
itself (each finished path is struck out of it), guarded by one lock, so the run
stays idempotent — stop any time and relaunch and whatever is still listed gets
picked up again.

What this shares with the sequential runner, and what it deliberately drops:

  * Reused — the ListFileDriver (list parsing, is_pending, command_for, strike),
    build_claude_argv, the /usage session-limit machinery, the git-push policy
    and the rotating mirror log.
  * Dropped — the live token-by-token Markdown rendering. cyclecore's stream
    renderer keeps module-global state that cannot serve several concurrent
    streams without garbling, so here each worker prints one compact, fully
    formed line per event, prefixed `[job k]`, under a single output lock. You
    trade the live view for throughput — the right call for mechanical bulk work.

CLI mirrors the family (see `--help`); the additions are `-j/--jobs N`
(default 10) and `-C/--project-dir`. `--max-runs N` caps the *total* number of
files processed this run (across all workers), not iterations.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import threading
import time
from typing import Optional

from . import cyclecore
from . import limits
from .cyclecore import (
    ClaudeCommand,
    GitPushPolicy,
    build_claude_argv,
    git_push,
    git_unpushed_count,
    maybe_git_push,
    print_markup,
    set_project_root,
    _describe_tool,
    _short,
)
from .usage import UsageSource
from .drivers import ListFileDriver

# Default worker count. The work is cheap and fully independent, so a handful of
# concurrent jobs is the sweet spot before the shared session budget, not CPU,
# becomes the bottleneck.
DEFAULT_JOBS = 10

# Per-file retry budget: a path that fails this many times in a row is parked in
# the `failed` set so it stops blocking the queue (and is reported at the end)
# instead of being retried forever.
MAX_ATTEMPTS = 3

# Serialises every line printed by any worker so the compact per-job lines never
# interleave mid-line (each print is atomic, the renderers are not thread-safe).
_emit_lock = threading.Lock()


def parse_args(argv=None, *, prog: str = "parallel",
               description: Optional[str] = None) -> argparse.Namespace:
    """CLI for the parallel runner: the family's options plus -j/--jobs.

    A trimmed copy of cyclecore.parse_args (it can't be reused directly: it has
    no --jobs, and its --max-runs means *iterations*, which here is redefined as
    a total-files cap). Every long option keeps its single-letter alias.
    """
    p = argparse.ArgumentParser(
        prog=prog,
        description=description or "Parallel autonomous loop running N "
                                  "concurrent `claude` workers over a list file.",
    )
    p.add_argument("-j", "--jobs", type=int, default=None, metavar="N",
                   help="number of concurrent workers (default: the driver's "
                        f"`jobs`, else {DEFAULT_JOBS})")
    p.add_argument("-m", "--max-runs", "--max", dest="max", type=int, default=None,
                   metavar="N",
                   help="stop after processing N files total, across all workers "
                        "(default: drain the whole list); --max is a deprecated alias")
    p.add_argument("-d", "--dry-run", action="store_true",
                   help="only print the commands that would run, don't run claude "
                        "and don't touch the list")
    p.add_argument("-g", "--git-push", dest="git_push",
                   choices=[pol.value for pol in GitPushPolicy],
                   default=cyclecore.GIT_PUSH_POLICY.value,
                   help="when to `git push`: none | after_new_commits | each_hour "
                        f"(default: {cyclecore.GIT_PUSH_POLICY.value})")
    p.add_argument("-C", "--project-dir", dest="project_dir", metavar="DIR",
                   default=None,
                   help="project root: cwd for git/claude, base for the stop "
                        "file and the list's relative paths "
                        "(default: the current working directory)")
    p.add_argument("--ignore-usage", action="store_true",
                   help="don't pause on the Current-session /usage limit "
                        "(by default the workers pause together when the session "
                        "budget is exhausted)")
    return p.parse_args(argv)


# --- output helpers: every emit goes through the shared lock --------------------

def _emit_markup(plain: str, markup: str) -> None:
    with _emit_lock:
        print_markup(plain, markup)


def _job_tag(job_id: int) -> tuple:
    """(plain, markup) prefix identifying a worker, e.g. '[job 2]'."""
    return f"[job {job_id}]", f"[cyan][job {job_id}][/]"


def emit_job(job_id: int, plain: str, style: Optional[str] = None) -> None:
    """One compact line attributed to a worker (styled on screen, plain in log)."""
    tag_plain, tag_markup = _job_tag(job_id)
    body_markup = f"[{style}]{plain}[/]" if style else plain
    _emit_markup(f"{tag_plain} {plain}", f"{tag_markup} {body_markup}")


def emit_tool(job_id: int, name: str, detail: str) -> None:
    """A tool-call line for a worker: '[job k] ⚙ Write: path'."""
    tag_plain, tag_markup = _job_tag(job_id)
    head_plain = f"{tag_plain}   ⚙ {name}"
    head_markup = f"{tag_markup}   [yellow]⚙[/] [bold yellow]{name}[/]"
    if detail:
        _emit_markup(f"{head_plain}: {detail}", f"{head_markup}: {detail}")
    else:
        _emit_markup(head_plain, head_markup)


# --- the actual claude round-trip for one file ---------------------------------

def run_job(job_id: int, command: ClaudeCommand) -> tuple:
    """Run one `claude` command, rendering a compact per-job trace.

    Unlike cyclecore's streaming renderer this prints only the key events — each
    tool call, any failed tool result, and the final cost line — one atomic line
    at a time, so several of these can run at once without their output
    colliding. Returns (returncode, cost_usd, duration_s).
    """
    argv = build_claude_argv(command)
    try:
        proc = subprocess.Popen(
            argv, cwd=cyclecore.project_dir(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except FileNotFoundError:
        emit_job(job_id, "executable 'claude' not found on PATH.", "bold red")
        return 2, None, None

    cost_usd = None
    duration_s = None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue  # non-JSON CLI diagnostics — skip in compact mode
        et = ev.get("type")
        if et == "assistant":
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_use":
                    name = block.get("name", "?")
                    detail = _describe_tool(name, block.get("input", {}) or {})
                    emit_tool(job_id, name, detail)
        elif et == "user":
            # Only surface *failed* tool results; successes would just be noise
            # at high concurrency.
            for block in ev.get("message", {}).get("content", []):
                if block.get("type") == "tool_result" and block.get("is_error"):
                    content = block.get("content", "")
                    if isinstance(content, list):
                        content = " ".join(
                            c.get("text", "") for c in content
                            if isinstance(c, dict)
                        )
                    emit_job(job_id, f"  ✗ {_short(content, 160)}", "red")
        elif et == "result":
            cost_usd = ev.get("total_cost_usd")
            dur = ev.get("duration_ms")
            duration_s = dur / 1000 if dur is not None else None
    return proc.wait(), cost_usd, duration_s


# --- shared queue state, guarded by one lock -----------------------------------

class Shared:
    """Cross-worker state behind a single lock: the list cursor and run stats.

    The list file (owned by `driver`) is the source of truth for what remains;
    `in_progress` keeps workers from claiming the same line, `failed` parks lines
    that exhausted their retry budget, and the counters bound/report the run. All
    access is under `lock`.
    """

    def __init__(self, driver: ListFileDriver, max_items: Optional[int]):
        self.driver = driver
        self.lock = threading.Lock()
        self.in_progress = set()      # raw lines a worker is currently handling
        self.failed = set()           # raw lines parked after MAX_ATTEMPTS
        self.attempts = {}            # raw line -> failed-attempt count
        self.claimed = 0              # files claimed this run (for --max-runs)
        self.done = 0                 # files processed successfully
        self.max_items = max_items
        self.stop = threading.Event()  # set on stop-file / no-work / fatal

    def claim(self) -> Optional[str]:
        """Reserve the next pending list line, or signal why there is none.

        Returns the claimed raw line, or None. When None, `stop` is set iff the
        queue is truly drained (nothing pending and nothing in flight); a None
        with `stop` clear means "busy, try again shortly".
        """
        with self.lock:
            if self.max_items is not None and self.claimed >= self.max_items:
                self.stop.set()
                return None
            pending = [ln for ln in self.driver.pending_lines()
                       if ln not in self.in_progress
                       and ln not in self.failed]
            if not pending:
                # Drained only if no one else is still working; otherwise back off.
                if not self.in_progress:
                    self.stop.set()
                return None
            # Random rather than list order so the run spreads evenly across a
            # category-grouped list (the head would drain one category first).
            line = random.choice(pending)
            self.in_progress.add(line)
            self.claimed += 1
            return line

    def release(self, line: str) -> None:
        """Return a claimed-but-unprocessed line to the queue.

        Used when a worker claims a line but then bails out before running it
        (the run was stopped while it sat in the session-limit gate): drop it from
        `in_progress` so the drained check stays accurate, and undo its --max
        reservation so the count reflects only files actually processed.
        """
        with self.lock:
            self.in_progress.discard(line)
            if self.claimed > 0:
                self.claimed -= 1

    def finish(self, line: str, ok: bool) -> tuple:
        """Record an item's outcome: strike it on success, or count/park a fail.

        Returns (done, remaining): files processed this run (across all workers)
        and how many are still pending — so the caller can report progress.
        """
        with self.lock:
            self.in_progress.discard(line)
            if ok:
                self.done += 1
                self.driver.strike(line)
                self.attempts.pop(line, None)
            else:
                self.attempts[line] = self.attempts.get(line, 0) + 1
                if self.attempts[line] >= MAX_ATTEMPTS:
                    self.failed.add(line)
            remaining = len(self.driver.pending_lines())
            return self.done, remaining


# --- worker loop ---------------------------------------------------------------

def worker(job_id: int, shared: Shared, source: Optional[UsageSource],
           policy, session_start_box: list, usage_lock: threading.Lock) -> None:
    """One worker thread: claim -> (usage gate) -> run -> record, repeat.

    Loops until the shared stop flag is set (queue drained, --max-runs hit, or stop
    file). A claim that returns None with the flag still clear means everything
    left is in flight elsewhere, so we briefly back off and retry. `source`/`policy`
    are None when --ignore-usage disables the session-limit gate.
    """
    while not shared.stop.is_set():
        if os.path.exists(cyclecore.STOP_FILE):
            # First worker to see it removes it and stops the whole run.
            with shared.lock:
                if os.path.exists(cyclecore.STOP_FILE):
                    os.remove(cyclecore.STOP_FILE)
                    emit_job(job_id, "stop file detected — stopping (removed it).",
                             "bold red")
                shared.stop.set()
            break

        # Claim work FIRST, before the session-limit gate. The gate can block for a
        # long time when the budget is spent, and its wait loop does not watch the
        # stop flag — so a worker that paused there before claiming would wedge and
        # never notice the queue draining to empty around it, hanging the run's
        # final join(). Claiming first means an empty/drained queue stops the worker
        # here (claim() sets the stop flag) and it never enters the gate with
        # nothing to do; only a worker actually holding a file ever pauses.
        line = shared.claim()
        if line is None:
            if shared.stop.is_set():
                break
            time.sleep(2)  # busy: others hold the rest — back off and retry
            continue

        # Session-limit gate, now that we hold real work: one worker checks at a
        # time (cheap, /usage is TTL-cached), and a pause blocks every worker that
        # reaches it — so the whole fleet idles together when the budget is spent.
        if source is not None:
            with usage_lock:
                if not shared.stop.is_set():
                    paused, new_start = policy.check_and_wait(
                        source, session_start_box[0])
                    if paused:
                        session_start_box[0] = new_start

        # The run may have been stopped (drained elsewhere, stop file, or --max
        # reached) while we waited for the lock or paused on the budget: return the
        # claimed file to the queue and exit without starting another claude run.
        if shared.stop.is_set():
            shared.release(line)
            break

        command = shared.driver.command_for(line)
        emit_job(job_id, f"▶ {command.label}", "bold cyan")
        rc, cost_usd, dur = run_job(job_id, command)
        ok = rc == 0
        done_total, remaining = shared.finish(line, ok)

        bits = []
        if dur is not None:
            bits.append(f"{dur:.1f}s")
        if cost_usd is not None:
            bits.append(f"${cost_usd:.4f}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        if ok:
            emit_job(job_id,
                     f"✓ {command.label}{suffix}  "
                     f"[{done_total} done this run, {remaining} left]", "green")
        else:
            parked = line in shared.failed
            tail = " — parked after repeated failures" if parked else " — will retry"
            emit_job(job_id, f"✗ {command.label} (exit {rc}){suffix}{tail}",
                     "bold red")


def run_parallel(driver: ListFileDriver, args: argparse.Namespace,
                 app_name: str = "parallel") -> None:
    """Drain `driver`'s list with N concurrent `claude` workers.

    The parallel counterpart of cyclecore.run_loop: same session-limit, git-push
    and mirror-log machinery, but a thread pool over independent list items
    instead of one sequential Driver loop.
    """
    # Worker count precedence: explicit -j/--jobs on the CLI, then the driver's
    # `jobs` attribute (a subclass may pin it), then the engine default.
    jobs = args.jobs
    if jobs is None:
        jobs = getattr(driver, "jobs", None)
    if jobs is None:
        jobs = DEFAULT_JOBS
    jobs = max(1, jobs)
    git_push_policy = GitPushPolicy(args.git_push)

    # Anchor every project-relative operation before anything reads the root.
    set_project_root(getattr(args, "project_dir", None))

    # Mirror all output to the rotating log, same as run_loop — under its own app
    # name so this runner's log doesn't fight the sequential one's.
    logger = cyclecore._setup_file_logging(app_name)
    sys.stdout = cyclecore._TeeToLog(sys.stdout, logger)
    sys.stderr = cyclecore._TeeToLog(sys.stderr, logger)
    print(f"  · project root: {cyclecore.project_dir()}")
    print(f"  · logging to {cyclecore.log_file_path(app_name)}")
    print(f"  · jobs: {jobs}  ·  git push policy: {git_push_policy.value}")

    list_file_rel = driver.list_file
    pending_now = driver.pending_lines()
    if not pending_now:
        print(f"Nothing pending in {list_file_rel} — nothing to do.")
        return

    # Dry-run: list the commands that would run (capped by --max-runs), touch nothing.
    if args.dry_run:
        limit = args.max if args.max is not None else len(pending_now)
        print(f"DRY-RUN: {min(limit, len(pending_now))} of {len(pending_now)} "
              f"pending file(s) would be processed across {jobs} worker(s):")
        for line in pending_now[:limit]:
            print("  " + " ".join(build_claude_argv(driver.command_for(line))))
        return

    # Usage gate: a shared UsageSource (query/cache) plus the Driver's LimitPolicy
    # (which quotas to gate on). --ignore-usage turns both off.
    source = None if args.ignore_usage else UsageSource()
    policy = None if args.ignore_usage else (
        driver.limit_policy or limits.default_policy())
    usage_lock = threading.Lock()
    session_start_box = [time.time()]  # shared, refreshed when a window resets

    if source is not None:
        policy.log_snapshot(source, "at start (parallel)")

    shared = Shared(driver, args.max)

    # Push pending commits up front on EACH_HOUR/AFTER_NEW_COMMITS, then let a
    # background pusher apply the policy on its cadence while workers run. git is
    # not thread-safe to call concurrently, so all pushes go through one thread.
    last_push_box = [0.0]
    push_lock = threading.Lock()

    def push_pump():
        while not shared.stop.wait(60):
            with push_lock:
                last_push_box[0] = maybe_git_push(git_push_policy, last_push_box[0])

    threads = [
        threading.Thread(target=worker, name=f"job{j}",
                         args=(j, shared, source, policy, session_start_box,
                               usage_lock),
                         daemon=True)
        for j in range(1, jobs + 1)
    ]
    pusher = None
    if git_push_policy != GitPushPolicy.NONE:
        pusher = threading.Thread(target=push_pump, name="pusher", daemon=True)

    for t in threads:
        t.start()
    if pusher is not None:
        pusher.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C) — signalling workers to stop…")
        shared.stop.set()
        for t in threads:
            t.join(timeout=5)
        sys.exit(130)

    shared.stop.set()  # release the pusher's wait()
    if pusher is not None:
        pusher.join(timeout=5)

    # Final push on the way out (unless policy is NONE), mirroring run_loop.
    if git_push_policy != GitPushPolicy.NONE:
        count = git_unpushed_count()
        if count is None or count > 0:
            print("  · final git push on exit…")
            with push_lock:
                git_push()
        else:
            print("  · final git push: nothing to push.")

    if source is not None:
        policy.log_snapshot(source, "at end (parallel)", cache_value=False)

    remaining = len(driver.pending_lines())
    print(f"\nProcessed {shared.done} file(s) this run; "
          f"{remaining} still pending in {list_file_rel}.")
    if shared.failed:
        print(f"  ⚠ {len(shared.failed)} file(s) parked after "
              f"{MAX_ATTEMPTS} failed attempts:")
        for line in sorted(shared.failed):
            print(f"      {os.path.basename(line.strip())}")
