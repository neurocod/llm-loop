"""
cyclecore.py - reusable engine behind autonomous Claude/Codex CLI loops.

This module holds everything that is *not* specific to one particular task:
command-line parsing, the rotating mirror log, the git-push policy, the whole
token-usage/session-window machinery, stream-json rendering, and the generic
`run_loop()` that ties it together. The only thing it does **not** decide is
*what work to do each iteration* — that is supplied by a `Driver` (see below),
so the same engine drives both:

  * runCycle.py    — a state-machine driver reading products/currentState.md;
  * runTranslate.py — a list driver translating files from products/list.md.

A Driver answers three questions for the loop, via three hooks:

  * next_command()  -> AgentCommand | None
        The command to run this iteration, or None when there is no more work
        and the loop should stop normally. It may raise LoopStop to abort the
        whole run (e.g. an error state that needs a human).
  * on_success(rc)  -> None
        Called after an iteration whose provider CLI exited 0 — the place to record
        progress (mark a file done, advance a cursor, …). Default: no-op.
  * final_summary() -> str | None
        A closing line printed on the way out (e.g. "Final state: …"). Optional.

Everything below is lifted verbatim from the original single-file runCycle.py,
with the few state-specific pieces (which file to read, which prompt to send,
which model to pick) factored out into the Driver protocol.

Claude token-limit handling is driven by the account's real usage figures rather than
guessed from error counts, in two layers:

  * proactive — before each iteration, and again immediately after any non-zero
    Claude exit, the loop asks the Driver's `limit_policy` for the current
    quota percentages and pauses if any watched one is at/over its ceiling. The
    reading lives in usage.py (UsageSource, an HTTP GET of the usage endpoint)
    and the pausing policy in limits.py (LimitPolicy and the ready-made
    SessionLimit / DayNightLimit / WeeklyLimit rules).
  * reactive — every Claude run streams its own rate-limit verdict, which this module
    picks out of the stream it is already parsing (see RateLimitEvent): a
    "rejected" means the wall was hit, and the loop waits that quota out even if
    the proactive reading was unavailable.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union

from . import exitlog, operator, providers
from .providers import (
    build_agent_argv as provider_argv,
    note_channel,
    prompt_on_stdin,
    provider_spec,
    set_live_messages,
    start_agent_process,
    usage_source_for,
)

# Claude sessions last ~5 hours; after a token-limit error we wait out that window.
CLAUDE_SESSION_DURATION = 5 * 60 * 60 + 3  # 5 hours as seconds and + 3s as a safety margin
LIMIT_RETRY_THRESHOLD = CLAUDE_SESSION_DURATION

# The usage-limit policy (which quota to gate on, what ceiling to allow,
# when to pause) lives in limits.py / usage.py, chosen per project via a Driver's
# `limit_policy` attribute — see Driver and run_loop.


class GitPushPolicy(Enum):
    """When the loop should run `git push` between iterations.

    Checked at the start of every iteration (see ``maybe_git_push``):

      * ``NONE``            — never push automatically.
      * ``AFTER_NEW_COMMITS`` — push whenever HEAD is ahead of its upstream
        (i.e. there are local commits that haven't been pushed yet).
      * ``EACH_HOUR``       — push at most once per hour, and only when there is
        something to push.
    """
    NONE = "none"
    AFTER_NEW_COMMITS = "after_new_commits"
    EACH_HOUR = "each_hour"


class RunStopReason(Enum):
    """Why a runner returned normally."""

    STOP_FILE = "stop_file"
    LIMIT_REACHED = "limit_reached"
    NO_WORK = "no_work"
    DRIVER_STOP = "driver_stop"
    DRY_RUN = "dry_run"


class RunResult(NamedTuple):
    """Structured normal-exit result used by higher-level coordinators."""

    reason: RunStopReason
    attempted: int = 0
    completed: int = 0
    remaining: Optional[int] = None


# What each stop reason is called in the `=== run ended: … ===` line, which is
# the one place a reader looks when asking "why did this stop?". Phrased for
# that reader rather than reusing the enum's wire value.
STOP_REASON_TEXT = {
    RunStopReason.STOP_FILE: "stop file requested",
    RunStopReason.LIMIT_REACHED: "iteration limit reached (--max-runs)",
    RunStopReason.NO_WORK: "no more work in the queue",
    RunStopReason.DRIVER_STOP: "the driver stopped the run",
    RunStopReason.DRY_RUN: "dry run finished",
}


# Default push policy. Override on the command line with --git-push.
GIT_PUSH_POLICY = GitPushPolicy.EACH_HOUR

# EACH_HOUR cadence: push no more often than this many seconds.
GIT_PUSH_INTERVAL = 3600  # seconds — one hour

# The Windows console is often cp1252 — switch output to UTF-8 so we can print Cyrillic.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# The project root: the working directory of the project being driven. All
# subprocesses (git, provider CLI) run with this as their cwd, the relative
# state/list paths a Driver is given are resolved against it, and the stop/log
# file names are derived from it.
#
# IMPORTANT: this is deliberately *not* the directory this module lives in. The
# module is meant to be vendored as a submodule under some host project, so the
# code location and the project root are different directories. It defaults to
# the current working directory (so running a thin wrapper from the project root
# "just works") and can be overridden with set_project_root() — which run_loop()
# calls from the --project-dir/-C option. Use project_dir() to read it so a
# later set_project_root() is always picked up.
PROJECT_DIR = os.getcwd()
# A manual brake: `touch stop` (create a file named "stop" in the project root)
# and the loop halts at the next iteration boundary - the running iteration
# finishes its one state transition first. The file stays present while the
# application winds down, then the outermost stop-file lifecycle removes it just
# before exit so other launchers can use it as a mutex. Recomputed by
# set_project_root().
STOP_FILE = os.path.join(PROJECT_DIR, "stop")

# How often wait_for_stop_file_clear() re-checks the sentinel while it holds a
# launch back. Short enough that removing the file feels immediate.
STOP_POLL_SECONDS = 2

_stop_file_lifecycle_lock = threading.Lock()
_stop_file_lifecycle_depth = 0
_detected_stop_file: Optional[str] = None


@contextmanager
def stop_file_lifecycle():
    """Keep a detected stop sentinel until the outermost application exits.

    Runners enter this lifecycle themselves so direct users retain one-shot stop
    semantics. A project wrapper can enter it once around its complete main()
    lifecycle; nested sequential, parallel, and periodic runner calls then leave
    the sentinel in place through all wrapper-level cleanup.
    """
    global _stop_file_lifecycle_depth, _detected_stop_file
    with _stop_file_lifecycle_lock:
        if _stop_file_lifecycle_depth == 0:
            _detected_stop_file = None
        _stop_file_lifecycle_depth += 1
    try:
        yield
    finally:
        with _stop_file_lifecycle_lock:
            _stop_file_lifecycle_depth -= 1
            outermost = _stop_file_lifecycle_depth == 0
            detected = _detected_stop_file if outermost else None
            if outermost:
                _detected_stop_file = None
        if detected is not None:
            try:
                os.remove(detected)
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"warning: could not remove stop file on application exit "
                      f"({detected}): {exc}", file=sys.stderr)
            else:
                print("Stop file removed on application exit.")


def mark_stop_file_detected() -> None:
    """Latch the current stop path for outermost-lifecycle exit cleanup."""
    global _detected_stop_file
    with _stop_file_lifecycle_lock:
        _detected_stop_file = STOP_FILE


# How long an interactive run counts down before it acts on a stop request. The
# status line's `s` key both sets and clears the sentinel, so a mis-press must be
# undoable — but a runner that latched the file and exited a millisecond later
# would make "press s again" a race the user always loses. Five seconds is long
# enough to notice the countdown row and press the key again, short enough that
# a deliberate stop still feels immediate.
STOP_GRACE_SECONDS = 5.0


def confirm_stop_request(app=None, grace: float = STOP_GRACE_SECONDS,
                         poll: float = 0.25) -> bool:
    """True if the pending stop sentinel should be acted on now.

    The grace exists ONLY for a sentinel this run's own `s` key created
    (`app.stop_requested_here`): a piped run, a CI run, or a script that wrote
    the file itself gets today's behaviour, unslowed — automation must not wait
    out a countdown for a keypress that is never coming. Intended as the one
    definition of "the user really meant it" for both runners; the parallel
    claim-loop adopts it when it grows a status line of its own.
    """
    if (app is None or not getattr(app, "enabled", False)
            or not getattr(app, "stop_requested_here", False) or grace <= 0):
        return True
    app.update(phase="stopping")
    deadline = time.time() + grace
    while True:
        if not os.path.exists(STOP_FILE):   # pressed `s` again — undo everything
            app.update(phase="idle", stop_pending=False)
            app.note("stop cancelled — continuing")
            return False
        remaining = deadline - time.time()
        if remaining <= 0:
            return True
        app.note(f"stopping in {int(remaining) + 1}s — press s to cancel")
        time.sleep(min(poll, remaining))


def set_project_root(path: Optional[str]) -> str:
    """Point the engine at the project root (cwd for git/provider CLI, base for the
    stop file and relative Driver paths). `path` None/empty means "keep the
    current value" (which defaults to the process cwd). Returns the resolved
    absolute path.

    The runners are single-process, so a module-level singleton set once at
    startup is enough — and it keeps cyclecore.PROJECT_DIR / cyclecore.STOP_FILE
    working for the parallel runner and the drivers, which read them directly.
    """
    global PROJECT_DIR, STOP_FILE
    if path:
        PROJECT_DIR = os.path.abspath(path)
        STOP_FILE = os.path.join(PROJECT_DIR, "stop")
    return PROJECT_DIR


def project_dir() -> str:
    """The current project root (see set_project_root)."""
    return PROJECT_DIR


# Markers that identify a project root when walking up the directory tree. `.git`
# is matched as either a directory (normal clone) or a file (git worktree /
# submodule), which os.path.exists covers for both.
ROOT_MARKERS = (".git", ".hg", ".svn")


def find_project_root(start: Optional[str] = None) -> Optional[str]:
    """Walk up from `start` (the current working directory by default) until a
    directory containing a VCS marker (`ROOT_MARKERS`) is found, and return it.

    A wrapper that anchors the search to its own file location (rather than the
    process cwd) gets a project root that is independent of where the loop was
    launched from: run it from the repo root, from a subdirectory, or from
    anywhere else and it lands on the same root — so git/provider CLI run there, the
    stop file lives there, and the model loads the root CLAUDE.md the same way
    every time. Returns None if no marker is found up to the filesystem root,
    leaving the engine's default (the current working directory) in place.
    """
    path = os.path.abspath(start if start else os.getcwd())
    while True:
        if any(os.path.exists(os.path.join(path, m)) for m in ROOT_MARKERS):
            return path
        parent = os.path.dirname(path)
        if parent == path:  # reached the filesystem root without a match
            return None
        path = parent

# A copy of everything printed to the screen is mirrored, line by line, to a
# rotating log file under the user's home dir (NOT the project tree) so cycle
# runs leave a durable record without cluttering the repo. The project folder
# name and the launching app name are baked into the file name so several
# projects/entry points write to separate logs instead of fighting over one file.
LOG_DIR = Path.home() / ".runCycle" / "logs"

# Rotation policy for the mirror log. Module-level constants rather than numbers
# inside the handler setup, so anything reporting how full the log is measures it
# against the very limit that rotates it instead of restating the figure.
LOG_MAX_BYTES = 25 * 1024 * 1024
# Deep enough that a burst of output cannot rotate an interesting segment off
# the end of the chain before anyone reads it: at 3 backups a single preview run
# displaced the failure a live run was recording, and it was gone for good.
LOG_BACKUP_COUNT = 5


def log_file_path(app_name: str = "runCycle") -> Path:
    """Path of the rotating mirror log for a given entry point.

    The project folder name and `app_name` are both baked in, so e.g.
    runCycle.py and runTranslate.py launched from the same project still write
    to separate logs (runCycle-<project>.log vs runTranslate-<project>.log).
    """
    return LOG_DIR / f"{app_name}-{os.path.basename(PROJECT_DIR)}.log"


# Per-run cost accounting parsed straight back out of the mirror log: every run's
# first iteration logs a "=== Iteration 1 ===" header (see run_loop) and every
# successful iteration logs a "done (… c, $…)" line (see _render_event's "result"
# branch). Summing the dollar figures between headers reconstructs per-run spend
# with no extra bookkeeping. The patterns live here, next to the code that emits
# both lines, so they can't drift apart. Note "=== Iteration 1 ===" matches only
# iteration 1 (the "1 ===" boundary rules out "11", "12", …), so each match is a
# genuine run boundary.
_SESSION_RE = re.compile(r"=== Iteration 1 ===")
_COST_RE = re.compile(r"done \(\s*[\d.]+ c,\s*\$([\d.]+)\)")


def report_costs(app_name: str = "runCycle",
                 path: Optional[Union[str, Path]] = None) -> None:
    """Print per-session (per-run) cost totals parsed from the mirror log, then
    exit — the standalone counterpart reached via the --cost flag.

    A "session" is one run of the loop, delimited by its "=== Iteration 1 ==="
    header; within it every "done (… c, $…)" line contributes its dollar cost. We
    print a line per session, a grand total, and how full the log is against the
    rotation limit (LOG_MAX_BYTES). With no `path`, the log is resolved via
    log_file_path(app_name), so --cost reports on the very log this entry point
    writes — under the project root already chosen by --project-dir.

    `path` (the --cost-log flag) names a log this entry point does NOT write:
    a rotated backup (`<app>-<project>.log.1`) or a copy taken elsewhere. It is
    the one case app_name cannot reach, since rotation renames files out from
    under log_file_path.
    """
    path = Path(path) if path else log_file_path(app_name)
    # Always name the log we are reading, so an empty report is unambiguous
    # (right file, no data) rather than looking like a silent failure.
    print(f"Reading mirror log: {path}")
    sessions = []  # list of (header, total_cost, count)
    header = None
    total = 0.0
    count = 0

    def flush():
        if header is not None:
            sessions.append((header, total, count))

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if _SESSION_RE.search(line):
                    flush()
                    header = line.strip()
                    total = 0.0
                    count = 0
                else:
                    m = _COST_RE.search(line)
                    if m and header is not None:
                        total += float(m.group(1))
                        count += 1
    except FileNotFoundError:
        print(f"No mirror log at {path} yet — nothing to report.")
        return
    flush()

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
        print("  (log has no '=== Iteration 1 ===' / '· done (… c, $…)' lines — "
              "no completed billed iterations recorded here)")

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    pct = size / LOG_MAX_BYTES * 100 if LOG_MAX_BYTES else 0.0
    print(f"LOG: {size / 1024 / 1024:.2f} / {LOG_MAX_BYTES / 1024 / 1024:.0f} MB "
          f"({pct:.1f}% full, rotates at 100%)")
    print(f"     {path}")


# The app-specific logger that owns the mirror-log file handler, set by
# _setup_file_logging. `_log_plain` (the Rich path) must target *this* logger:
# the handler lives on "runCycle.<app_name>" (which does not propagate), so
# logging to a bare "runCycle" would silently drop the message. Kept in a module
# global because the Rich print helpers have no reference to the configured logger.
_FILE_LOGGER: Optional[logging.Logger] = None


# How long a handler waits before retrying a rotation that failed. Long enough
# that a wedged rename is attempted once a minute rather than once per line.
ROLLOVER_RETRY_SECONDS = 60.0


class _MirrorLogHandler(RotatingFileHandler):
    """A rotating handler that survives another process holding the same log.

    Running two loops side by side is normal here (a sequential run, a parallel
    run, the grow-kit pass), and same-named runs share one mirror log. On Windows
    a rename fails while another process has the file open, and the stock handler
    reports that through `logging.raiseExceptions`, i.e. by printing to
    `sys.stderr` — which is the `_TeeToLog` mirror, which logs the line, which
    fails again: an unbounded recursion that ends the run with a RecursionError
    over a *log file*. Measured, not theorised: two runs colliding on a 25 MB
    rollover killed the second one outright.

    So a failed rotation is not an error here. We keep appending to the current
    file (briefly past the size cap, which the next successful rotation trims)
    and try again later.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._retry_rollover_at = 0.0

    def doRollover(self) -> None:
        if time.time() < self._retry_rollover_at:
            return  # a recent attempt failed; the other holder still has it
        try:
            super().doRollover()
        except OSError:
            self._retry_rollover_at = time.time() + ROLLOVER_RETRY_SECONDS

    def handleError(self, record) -> None:
        """Swallow. The default writes the traceback to `sys.stderr`, which is
        the tee — see the class docstring for why that cannot be allowed."""


def _setup_file_logging(app_name: str = "runCycle") -> logging.Logger:
    """Configure the rotating file logger at log_file_path(app_name)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"runCycle.{app_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:  # avoid duplicate handlers if called twice
        handler = _MirrorLogHandler(
            log_file_path(app_name), maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    global _FILE_LOGGER
    _FILE_LOGGER = logger
    return logger


class _TeeToLog:
    """Wrap a console stream so everything printed is also captured into the file
    logger, one record per line.

    Partial writes (streaming tokens emitted with ``end=""``) are buffered until a
    newline, so the file holds clean, complete lines while the screen keeps showing
    live token-by-token output.
    """

    # Set while this thread is inside a logging call, so anything the logging
    # machinery itself prints goes to the screen only. Without it a handler that
    # reports a failure through stderr feeds its own report back into the logger
    # that just failed, and the run dies of recursion (see _MirrorLogHandler).
    _in_logging = threading.local()

    def __init__(self, stream, logger: logging.Logger):
        self._stream = stream
        self._logger = logger
        self._buf = ""

    def write(self, text: str) -> int:
        self._stream.write(text)
        if getattr(self._in_logging, "active", False):
            return len(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._in_logging.active = True
            try:
                self._logger.info(line)
            finally:
                self._in_logging.active = False
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        # Delegate everything else (encoding, isatty, fileno, ...) to the stream.
        return getattr(self._stream, name)


class ConsumedByWrapperAction(argparse.Action):
    """An option this parser only DOCUMENTS — the wrapper reads it out of argv
    itself, before parsing (see Driver.add_cli_options).

    Reaching the parser therefore means the wrapper's own scan did not match
    what was typed. argparse would otherwise accept it (an abbreviation like
    `--grow` for `--grow-kit` is one this parser resolves and that scan does
    not), store it in a namespace nobody reads, and run the DEFAULT mode — a
    flag that appears to work and silently does nothing. Saying so is the whole
    job of this action.

    Declare a value-taking switch with `nargs=1, metavar="N"` so --help shows
    its argument; the default `nargs=0` documents a bare flag.
    """

    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs,
                         default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # `option_string` is the canonical spelling, not what was typed — which
        # is the point: it names the option the abbreviation resolved to, and
        # the one to type instead.
        parser.error(
            f"{option_string} never reached the argv scan that reads it, "
            f"which runs before this parser. Spell it out in full; "
            f"abbreviations are resolved here and are invisible there. (If it "
            f"WAS spelled in full, that scan and this help have drifted "
            f"apart.)")


def parse_args(argv=None, *, prog: str = "runCycle.py",
               description: Optional[str] = None,
               extra_options: Optional[Callable[[argparse.ArgumentParser],
                                                None]] = None
               ) -> argparse.Namespace:
    """Command-line interface shared by every entry point. Every long option has
    a single-letter alias.

    `prog`/`description` let each entry script label its own --help text while
    reusing the exact same option set (so there is no duplicated argument code).

    `extra_options` is handed the parser once the shared options are on it, and
    is how a wrapper documents the flags IT consumes before this parser ever
    runs (a mode switch that decides which parser to use cannot be one of this
    parser's options). Passing the parser rather than a block of help text is
    what keeps such a flag formatted, aligned and grouped like every other
    option — and lets the hook add a real, parsed option when the flag is not
    one the wrapper strips.
    """
    p = argparse.ArgumentParser(
        prog=prog,
        description=description or "Autonomous loop driving an LLM CLI.",
    )
    # Most option names are kept in sync with continuous_claude.py (kebab-case,
    # and --max-runs for the iteration cap). The former spellings (--max,
    # --startIn) stay on as accepted aliases so existing invocations keep
    # working.
    p.add_argument("-m", "--max-runs", "--max", dest="max", type=int, default=None,
                   metavar="N",
                   help="stop after N iterations (default: run forever); "
                        "--max is a deprecated alias")
    p.add_argument("--codex", action="store_const", const="codex",
                   dest="provider", default=None,
                   help="run Codex CLI instead of the Driver's default provider")
    p.add_argument("-d", "--dry-run", action="store_true",
                   help="only print the commands, don't run the LLM CLI")
    p.add_argument("-c", "--cost", action="store_true",
                   help="print per-session cost totals from the mirror log and "
                        "exit (no loop is run)")
    p.add_argument("--cost-log", dest="cost_log", metavar="LOG",
                   help="report on this log file instead of this entry point's "
                        "own — a rotated backup (<app>-<project>.log.1) or a "
                        "copy; implies --cost")
    # No -r short flag: -r is --review-prompt in continuous_claude.py, so it is
    # left free here rather than reused for --raw.
    p.add_argument("--raw", action="store_true",
                   help="print raw JSON events (for debugging)")
    p.add_argument("-s", "--start-in", "--startIn", dest="start_in", metavar="DURATION",
                   help="wait this long before starting the loop, e.g. 29m, 1h30m")
    p.add_argument("-g", "--git-push", dest="git_push",
                   choices=[pol.value for pol in GitPushPolicy],
                   default=GIT_PUSH_POLICY.value,
                   help="when to `git push` at the start of each iteration: "
                        "none | after_new_commits | each_hour "
                        f"(default: {GIT_PUSH_POLICY.value})")
    p.add_argument("-C", "--project-dir", dest="project_dir", metavar="DIR",
                   default=None,
                   help="project root: cwd for git/provider CLI, base for the stop "
                        "file and the Driver's relative paths "
                        "(default: the current working directory)")
    # No short alias: this is a rescue hatch for an odd terminal, not a knob to
    # reach for. LLM_LOOP_STATUSLINE=0 does the same without editing a
    # command line, and no TTY disables it by itself.
    p.add_argument("--no-statusline", dest="no_statusline", action="store_true",
                   help="do not pin the interactive status rows at the bottom of "
                        "the terminal (same as LLM_LOOP_STATUSLINE=0)")
    # Same shape of rescue hatch as --no-statusline, and for the same reason: it
    # turns off a transport, not a feature. A note typed with the `m` key still
    # reaches the agent — with the next iteration's prompt instead of the one
    # already running.
    p.add_argument("--no-live-messages", dest="no_live_messages",
                   action="store_true",
                   help="do not keep the agent's stdin open for notes typed "
                        "during an iteration; they wait for the next prompt "
                        f"instead (same as {providers.LIVE_MESSAGES_ENV}=0)")
    if extra_options is not None:
        extra_options(p)
    return p.parse_args(argv)


def parse_duration(text: str) -> float:
    """Parse a duration like '29m', '1h', '90s', '1h30m' into seconds.

    A bare number is treated as minutes ('29' == '29m'). Raises ValueError on
    anything it can't make sense of.
    """
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    if text.isdigit():  # bare number — minutes
        return int(text) * 60

    units = {"h": 3600, "m": 60, "s": 1}
    total = 0.0
    matched = False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
        total += float(value) * units[unit]
        matched = True
    if not matched:
        raise ValueError(f"cannot parse duration: {text!r}")
    return total


def git_unpushed_count() -> Optional[int]:
    """Number of local commits ahead of the upstream branch (HEAD not yet pushed).

    Returns the count, or None if it can't be determined (no upstream configured,
    git missing, not a repo, …) — in which case callers treat a push as worth
    attempting rather than silently skipping.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-list", "--count", "@{u}..HEAD"],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        return int((proc.stdout or "").strip())
    except ValueError:
        return None


def git_push() -> bool:
    """Run `git push`, printing the outcome. Returns True on success."""
    try:
        proc = subprocess.run(
            ["git", "push"],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            timeout=300,
        )
    except FileNotFoundError:
        print_error("  · git push skipped: 'git' not found on PATH.")
        return False
    except subprocess.TimeoutExpired:
        print_error("  · git push timed out.")
        return False
    if proc.returncode == 0:
        print_done("  · git push: done.")
        return True
    print_error(f"  · git push failed (exit {proc.returncode}): "
                f"{_short(proc.stdout or '')}")
    return False


def maybe_git_push(policy: GitPushPolicy, last_push: float) -> float:
    """Apply the GitPushPolicy at the start of an iteration.

    `last_push` is the epoch time of the previous push attempt (0.0 if never).
    Returns the updated `last_push` so the caller can carry it to the next
    iteration. A no-op for NONE; pushes when commits are pending for
    AFTER_NEW_COMMITS; for EACH_HOUR pushes pending commits at most once an hour.
    """
    if policy == GitPushPolicy.NONE:
        return last_push

    if policy == GitPushPolicy.AFTER_NEW_COMMITS:
        count = git_unpushed_count()
        if count is None or count > 0:
            if git_push():
                return time.time()
        return last_push

    if policy == GitPushPolicy.EACH_HOUR:
        now = time.time()
        if now - last_push < GIT_PUSH_INTERVAL:
            return last_push
        # An hour has passed — push if there is anything to push, and reset the
        # timer either way so we re-check at most once per hour.
        count = git_unpushed_count()
        if count is None or count > 0:
            git_push()
        return now

    return last_push


class AgentCommand(NamedTuple):
    """One unit of work for the loop: the prompt to send, the model to use, and a
    short label shown in the iteration header. Drivers build these in
    next_command(); build_agent_argv() turns one into the provider's full argv.

    An empty `model` means "no --model flag": the selected provider then uses
    its configured default, which is the common case.
    """
    prompt: str
    model: str = ""
    label: str = ""
    provider: str = ""


ClaudeCommand = AgentCommand


def build_agent_argv(command: AgentCommand, provider: Optional[str] = None) -> list:
    """Full provider command line for one unit of work."""
    provider = provider or command.provider or "claude"
    return provider_argv(command, provider, project_dir())


def build_claude_argv(command: ClaudeCommand) -> list:
    """Full `claude` command line for one ClaudeCommand.

    The flags are identical for every task; only the prompt and the model vary,
    so this is the single place those two are spliced into the otherwise fixed
    argv (stream-json + partial messages so the loop can render work live). An
    empty `command.model` omits --model entirely, letting the CLI pick its own
    configured default.
    """
    return build_agent_argv(command, "claude")


def _short(text: str, limit: int = 200) -> str:
    """Single-line truncated version of text for compact output."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_BACKSLASH_RUN_RE = re.compile(r"\\+")


def undouble_backslashes(text: str) -> str:
    r"""Halve uniformly doubled backslashes in a command line, for display only.

    Codex reports the command it ran with every backslash doubled — its own
    escaping, not JSON's (json.loads already removed one level, and the quotes
    in the same value arrive unescaped) — so a Windows path reaches the screen
    and the log as C:\\WINDOWS\\... and cannot be copied without hand-editing.

    Halving is the exact inverse only while EVERY run of consecutive backslashes
    has even length: a UNC \\server\share arrives as four and halves back to its
    correct two. A single odd run means the string was never uniformly doubled,
    so it is returned untouched — showing a raw string beats corrupting one.
    That is why this is not `text.replace("\\\\", "\\")`, which would eat the
    UNC pair and mangle half-escaped strings.

    Apply it to values that are definitionally command lines, before _short, so
    truncation counts the characters the reader actually sees.
    """
    runs = _BACKSLASH_RUN_RE.findall(text)
    if not runs or any(len(run) % 2 for run in runs):
        return text
    return _BACKSLASH_RUN_RE.sub(lambda m: "\\" * (len(m.group(0)) // 2), text)


def _describe_tool(name: str, ti: dict) -> str:
    """Short human-readable description of a tool call and its arguments."""
    if name == "Bash":
        return f"$ {_short(undouble_backslashes(str(ti.get('command', ''))))}"
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return ti.get("file_path", ti.get("notebook_path", ""))
    if name in ("Glob", "Grep"):
        loc = f" in {ti['path']}" if ti.get("path") else ""
        return f"{ti.get('pattern', '')}{loc}"
    if name == "Skill":
        return ti.get("skill", "")
    if name == "Task" or name == "Agent":
        return _short(ti.get("description", ti.get("prompt", "")))
    if name == "TodoWrite":
        todos = ti.get("todos", [])
        return f"{len(todos)} items"
    # fallback: the first meaningful field
    for key in ("url", "query", "description", "prompt"):
        if ti.get(key):
            return _short(ti[key])
    return _short(json.dumps(ti, ensure_ascii=False)) if ti else ""


# Optional pretty Markdown rendering of the assistant's streamed text via Rich.
# The model emits its answer as Markdown; with Rich installed we render it live
# (bold, headings, lists, code fences, tables) instead of dumping the raw
# `**...**` source to the screen. Without Rich the script falls back to plain
# token streaming, so it keeps working unchanged (just `pip install rich` to get
# the formatting).
try:
    from rich.console import Console as _RichConsole
    from rich.live import Live as _RichLive
    from rich.markdown import Markdown as _RichMarkdown
    from rich.markup import escape as _rich_escape
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def _esc(text: str) -> str:
    """Escape Rich markup metacharacters in dynamic text (e.g. a Bash command or
    a file path containing '['). No-op when Rich is unavailable."""
    return _rich_escape(str(text)) if _RICH_AVAILABLE else str(text)


def _real_stream():
    """The underlying console stream, unwrapping the line-logging tee.

    Rich's Live repaints the frame many times a second using cursor-movement
    escape codes that must not end up in the file log, so its output goes
    straight to the real terminal rather than through `_TeeToLog`.
    """
    out = sys.stdout
    return getattr(out, "_stream", out)


def _log_plain(text: str) -> None:
    """Mirror a finished Markdown block to the file log as clean plain text.

    Used on the Rich path, where the live frames bypass the tee — we still want
    the assistant's words in the log, just without the ANSI/redraw noise.

    Targets the app-specific logger configured by _setup_file_logging (which owns
    the file handler and does not propagate); falling back to a bare "runCycle"
    logger only if logging was never set up (e.g. in tests). Using the wrong
    logger name here silently drops every Rich-path line — including the
    "=== Iteration N ===" headers and "· done (… c, $…)" cost lines that
    report_costs parses — leaving --cost to report 0 sessions.
    """
    logger = _FILE_LOGGER or logging.getLogger("runCycle")
    for line in text.splitlines():
        logger.info(line)


class _MarkdownStream:
    """Render one assistant text block as live-updating Markdown.

    The model streams Markdown token by token; we accumulate it and let Rich
    re-render the whole block inside a `Live` region on each delta, so formatting
    appears in realtime. When Rich is unavailable we degrade to the original
    behaviour: print a `💬` header and stream the raw tokens inline.
    """

    def __init__(self):
        self._buf = ""
        self._live = None
        self._console = None

    def start(self) -> None:
        self._buf = ""
        if _RICH_AVAILABLE:
            self._console = _RichConsole(file=_real_stream())
            self._console.print("\n[dim]  💬[/dim]")
            self._live = _RichLive(
                _RichMarkdown(""),
                console=self._console,
                refresh_per_second=12,
                vertical_overflow="visible",
                # Nothing else prints during a text block, so we don't need Rich
                # to hijack stdout/stderr (which would fight with _TeeToLog).
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
        else:
            print("\n  💬 ", end="", flush=True)

    def feed(self, text: str) -> None:
        self._buf += text
        if self._live is not None:
            self._live.update(_RichMarkdown(self._buf))
        else:
            print(text, end="", flush=True)

    def stop(self) -> None:
        if self._live is not None:
            self._live.update(_RichMarkdown(self._buf))
            self._live.stop()
            self._live = None
            self._console = None
            # Guarantee the next output (tool calls, etc.) starts on a fresh line,
            # regardless of how Live left the cursor on this terminal.
            print(file=_real_stream())
            if self._buf.strip():
                _log_plain(self._buf)
        else:
            print(flush=True)  # finish the inline line in fallback mode
        self._buf = ""


def _render_markdown_block(text: str) -> None:
    """Print a complete Markdown string formatted (Rich) or plain (fallback).

    Used for non-streaming assistant text (when --include-partial-messages is off
    we never see deltas, only the final block).
    """
    text = text.strip()
    if not text:
        return
    if _RICH_AVAILABLE:
        console = _RichConsole(file=_real_stream())
        console.print("[dim]  💬[/dim]")
        console.print(_RichMarkdown(text))
        _log_plain(text)
    else:
        print(f"\n  💬 {text}")


def print_markup(plain: str, markup: str) -> None:
    """Print a status line from hand-written Rich markup: styled on screen, plain
    in the log. The low-level core of the print_* family — use `print_styled`
    (text + a style name) for uniform lines and call this directly only when a
    line needs different styles per segment (e.g. a coloured glyph + plain text).

    With Rich available the `markup` string (Rich console markup: colours, bold,
    italic, underline) is rendered straight to the real terminal, while a clean
    `plain` copy is mirrored to the file log — so colour/redraw escapes never end
    up in the log. Without Rich it degrades to a plain `print` (screen + log via
    the tee). Note: terminals can't switch *font family*; only colour and the
    bold/italic/underline attributes are available.
    """
    if _RICH_AVAILABLE:
        _RichConsole(file=_real_stream()).print(markup)
        _log_plain(plain)
    else:
        print(plain)


def print_styled(text: str, style: str) -> None:
    """Print a whole line in one Rich style, routed through `print_markup`.

    The single-style sibling of `print_markup`: callers pass plain `text` plus a
    Rich style (`"green"`, `"bold red"`, …); the plain copy goes to the log and
    the styled copy to the screen. Markup metacharacters in `text` are escaped,
    so a stray '[' is shown literally instead of being read as a tag. For lines
    that need *different* styles per segment (a coloured glyph next to plain
    text), call `print_markup` directly with hand-written markup.
    """
    print_markup(text, f"[{style}]{_esc(text)}[/]")


# Colour scale for the usage percentages (session/week quotas, and the ceilings
# they are judged against): comfortable below GREEN_BELOW, alarming above
# RED_ABOVE, watch-it in between. Both bounds are exclusive, so exactly 60% and
# exactly 90% read as the middle band.
PERCENT_GREEN_BELOW = 60.0
PERCENT_RED_ABOVE = 90.0
PERCENT_STYLES = ("green", "yellow", "bold red")  # low, middle, high

# "44%", "7.5 %" — the figure plus its sign, as it appears in a printed line.
_PERCENT_IN_TEXT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")


def percent_style(value: float) -> str:
    """The palette entry for one percentage — see PERCENT_GREEN_BELOW/RED_ABOVE."""
    if value < PERCENT_GREEN_BELOW:
        return PERCENT_STYLES[0]
    if value > PERCENT_RED_ABOVE:
        return PERCENT_STYLES[2]
    return PERCENT_STYLES[1]


def markup_percents(text: str) -> str:
    """Rich markup for `text` with every percentage coloured by percent_style.

    Colouring the *rendered line* rather than each figure at its format site is
    what keeps one scale across lines that are assembled in several places (a
    rule's own `describe()`, the usage/ceiling line, the usage-report summary
    lines) — and what lets a line quoted from elsewhere be coloured at all. The
    non-percentage parts are escaped, so a '[' in a label stays literal.
    """
    out = []
    last = 0
    for m in _PERCENT_IN_TEXT_RE.finditer(text):
        out.append(_esc(text[last:m.start()]))
        value = float(m.group(0).rstrip("% \t"))
        out.append(f"[{percent_style(value)}]{m.group(0)}[/]")
        last = m.end()
    out.append(_esc(text[last:]))
    return "".join(out)


def print_percents(text: str) -> None:
    """Print a line whose percentages are colour-coded on screen (plain in the
    log). A no-op difference from `print` when Rich is unavailable.

    Flushed, because these lines include the once-a-minute countdown printed
    while a run is paused on a limit — the one place output has to appear as it
    is written rather than when a buffer happens to fill.
    """
    print_markup(text, markup_percents(text))
    sys.stdout.flush()


# Named single-style specialisations, each delegating to print_styled. Centralise
# the loop's palette here so a colour is changed in one place, not at every call.
def print_done(text: str) -> None:
    print_styled(text, "green")


def print_error(text: str) -> None:
    print_styled(text, "bold red")


def report_undelivered_notes(mailbox) -> None:
    """Say so when the run ends holding notes nobody ever saw.

    A queued note promises "the next iteration" — and there is not always a next
    one (the list drained, --max-runs, a stop request, a quota pause that
    outlasts the run). Whoever typed it during the last iteration would
    otherwise have no way to learn it was never delivered, and the text itself
    is printed so it can be pasted into the next run rather than retyped.
    """
    if mailbox is None:
        return
    notes = mailbox.take_queued()
    if not notes:
        return
    print_error(f"\n  ⚠ the run ended holding {len(notes)} undelivered operator "
                f"note(s) — there was no next iteration to carry them:")
    for note in notes:
        print_error(f"      {note}")


def print_note(text: str) -> None:
    """An operator note, at the point in the stream where the agent received it.

    Printed rather than merely shown on the status row because the status row is
    transient and the mirror log is the run's record: an agent that changes
    course mid-iteration is unexplainable later unless the sentence that made it
    do so sits in the log next to the turn it landed in.
    """
    print_markup(f"  ✉ operator note: {text}",
                 f"  [magenta]✉[/] [bold magenta]operator note:[/] {_esc(text)}")


def print_tool(name: str, detail: str = "") -> None:
    """A tool-call line: a yellow gear glyph and the bold-yellow tool name,
    followed by the (plain, possibly empty) detail. Multi-segment, so it builds
    markup and calls `print_markup` rather than print_styled; the shared head is
    written once instead of being repeated across the with/without-detail forms.
    """
    head_plain = f"  ⚙ {name}"
    head_markup = f"  [yellow]⚙[/] [bold yellow]{_esc(name)}[/]"
    if detail:
        print_markup(f"{head_plain}: {detail}", f"{head_markup}: {_esc(detail)}")
    else:
        print_markup(head_plain, head_markup)


# Streaming print state: the single content-block index text is currently flowing
# into (assistant replies stream one text block at a time), plus its live renderer.
_active_text_index = None
_md_stream = _MarkdownStream()

# The session cost already reported by earlier `result` events of the process
# now streaming. Reset by run_agent_streaming; see the result branch for why the
# figure has to be differenced at all.
_turn_cost_base = 0.0


# --- the free half of the limit machinery: the run's own rate-limit events -----
#
# Every `claude` run streams a line of its own, built from the ratelimit headers
# the API already returned to it:
#
#   {"type":"rate_limit_event","rate_limit_info":{"status":"allowed",
#     "resetsAt":1786807200,"rateLimitType":"five_hour", …}}
#
# Reading it costs nothing — that stream is parsed anyway — and unlike a queried
# figure it cannot be stale or unavailable: it is the wire's own verdict on the
# request that just went out. It carries no percentage, so it cannot drive a
# ceiling; it is the backstop *under* the proactive check in limits.py. "rejected"
# means this run hit the wall, and `resetsAt` says when that quota comes back.

# Quota id -> the name the CLI itself uses for it in limit messages.
RATE_LIMIT_LABELS = {
    "five_hour": "session limit",
    "seven_day": "weekly limit",
    "seven_day_opus": "Opus limit",
    "seven_day_sonnet": "Sonnet limit",
    "seven_day_overage_included": "usage-credit limit",
}


class RateLimitEvent(NamedTuple):
    """One rate_limit_event: which quota it is about, how it stands, when it resets.

    `status` is the API's own verdict — "allowed", "allowed_warning" (close to
    the wall) or "rejected" (refused). `resets_at` is epoch seconds, or None when
    the event carried no reset time.
    """
    status: str
    limit_type: str
    resets_at: Optional[float]

    @property
    def label(self) -> str:
        return RATE_LIMIT_LABELS.get(self.limit_type, self.limit_type or "limit")

    def describe(self) -> str:
        when = f", resets {_fmt_moment(self.resets_at)}" if self.resets_at else ""
        return f"{self.label} {self.status}{when}"


def rate_limit_event_from(ev: dict) -> Optional[RateLimitEvent]:
    """The RateLimitEvent carried by a stream-json event, or None if it isn't one."""
    if ev.get("type") != "rate_limit_event":
        return None
    info = ev.get("rate_limit_info") or {}
    resets = info.get("resetsAt")
    return RateLimitEvent(
        status=str(info.get("status") or "unknown"),
        limit_type=str(info.get("rateLimitType") or ""),
        resets_at=float(resets) if isinstance(resets, (int, float)) else None,
    )


# The last verdict seen by run_claude_streaming, cleared when a run starts — so it
# always describes the run that just finished, never the one before it.
_last_rate_limit_event = None


def last_rate_limit_event() -> Optional[RateLimitEvent]:
    """The rate-limit verdict of the most recent run_claude_streaming call (None
    if that run streamed no rate_limit_event)."""
    return _last_rate_limit_event


def _render_claude_event(ev: dict, partial: bool, mailbox=None) -> None:
    """Print a single stream-json event in the style of interactive mode.

    partial=True — --include-partial-messages is enabled: we print text from the
    deltas (`stream_event`), and from the final `assistant` we take only the tool
    calls, so as not to duplicate already-printed text.

    `mailbox` turns the replay of an operator note (`--replay-user-messages`)
    into a console line. Rendered here rather than where the note was typed
    because this is the main thread, between two events — the one place a line
    can be printed without cutting into the live Markdown block — and because
    the replay is the CLI confirming delivery, not the console assuming it.
    """
    et = ev.get("type")

    if et == "system" and ev.get("subtype") == "init":
        model = ev.get("model", "?")
        print(f"  · session started (model {model})")
        return

    if et == "rate_limit_event":
        global _last_rate_limit_event
        _last_rate_limit_event = rate_limit_event_from(ev)
        # "allowed" is the normal case and would be one more line per run; the
        # two states that mean something get shown.
        if _last_rate_limit_event.status != "allowed":
            print_error(f"  ⚠ rate limit: {_last_rate_limit_event.describe()}")
        return

    # Streaming deltas (Anthropic streaming events, wrapped in stream_event).
    if et == "stream_event":
        global _active_text_index
        inner = ev.get("event", {})
        it = inner.get("type")
        if it == "content_block_start":
            if inner.get("content_block", {}).get("type") == "text":
                _active_text_index = inner.get("index")
                _md_stream.start()
        elif it == "content_block_delta":
            d = inner.get("delta", {})
            if d.get("type") == "text_delta" and inner.get("index") == _active_text_index:
                _md_stream.feed(d.get("text", ""))
        elif it == "content_block_stop":
            if inner.get("index") == _active_text_index:
                _md_stream.stop()  # finalize the Markdown render / line
                _active_text_index = None
        return

    if et == "assistant":
        for block in ev.get("message", {}).get("content", []):
            bt = block.get("type")
            if bt == "text":
                if partial:
                    continue  # already printed streaming from the deltas
                _render_markdown_block(block.get("text", ""))
            elif bt == "tool_use":
                name = block.get("name", "?")
                detail = _describe_tool(name, block.get("input", {}) or {})
                print_tool(name, detail)
        return

    if et == "user":
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text" and mailbox is not None:
                note = mailbox.claim_echo(block.get("text", ""))
                if note is not None:
                    print_note(note)
                continue
            if block.get("type") != "tool_result":
                continue
            content = block.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            is_err = block.get("is_error")
            mark = "✗" if is_err else "✓"
            line = _short(content, 160)
            if line:
                color = "red" if is_err else "green"
                print_markup(f"    {mark} {line}",
                             f"    [{color}]{mark}[/] {_esc(line)}")
        return

    if et == "result":
        global _turn_cost_base
        cost = ev.get("total_cost_usd")
        if cost is not None:
            # `total_cost_usd` is the SESSION's running total, not this turn's
            # (measured: two trivial turns in one process reported $0.2015 then
            # $0.2204, while their durations were 2243 ms and 1991 ms — the
            # second figure is the first plus $0.019, not a second $0.2). A
            # process emits more than one `result` whenever a note typed late is
            # answered as its own turn, and `report_costs` sums these lines, so
            # each line shows what its turn ADDED.
            cost, _turn_cost_base = cost - _turn_cost_base, cost
        dur = ev.get("duration_ms")
        bits = []
        if dur is not None:
            bits.append(f"{dur / 1000:.1f} c")
        if cost is not None:
            bits.append(f"${cost:.4f}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        if ev.get("subtype") != "success" or ev.get("is_error"):
            print_error(f"  ⚠ result: {ev.get('subtype', 'error')}{suffix}")
        else:
            print_done(f"  · done{suffix}")
        return


def _render_codex_event(ev: dict) -> None:
    """Render one event from ``codex exec --json``."""
    event_type = ev.get("type")
    item = ev.get("item") or {}
    item_type = item.get("type")

    if event_type == "thread.started":
        thread_id = ev.get("thread_id") or "?"
        print(f"  · session started (thread {thread_id})")
        return

    if event_type == "item.completed" and item_type == "agent_message":
        _render_markdown_block(str(item.get("text") or ""))
        return

    if event_type == "item.started" and item_type == "command_execution":
        command = _short(undouble_backslashes(str(item.get("command", ""))))
        print_tool("Bash", command)
        return

    if event_type == "item.completed" and item_type == "command_execution":
        exit_code = item.get("exit_code")
        command = _short(undouble_backslashes(str(item.get("command", ""))), 160)
        output = _short(item.get("aggregated_output", ""), 160)
        mark = "✓" if exit_code in (None, 0) else "✗"
        detail = f"exit {exit_code}: {command}" if exit_code is not None else command
        if output:
            detail += f" — {output}"
        style = "green" if exit_code in (None, 0) else "red"
        print_markup(f"    {mark} {detail}", f"    [{style}]{mark}[/] {_esc(detail)}")
        return

    if event_type == "item.completed" and item_type == "file_change":
        changes = item.get("changes") or []
        paths = [str(change.get("path")) for change in changes
                 if isinstance(change, dict) and change.get("path")]
        print_tool("Edit", ", ".join(paths) or "file changes applied")
        return

    if event_type == "turn.completed":
        usage = ev.get("usage") or {}
        bits = []
        if usage:
            bits.append(f"input {usage.get('input_tokens', 0)}")
            cached = usage.get("cached_input_tokens", 0)
            if cached:
                bits.append(f"cached {cached}")
            bits.append(f"output {usage.get('output_tokens', 0)}")
        suffix = f" (tokens: {', '.join(bits)})" if bits else ""
        print_done(f"  · done{suffix}")
        return

    if event_type in ("error", "turn.failed"):
        error = ev.get("message") or ev.get("error") or item.get("error") or ev
        print_error(f"  ⚠ result: {_short(error)}")


def run_agent_streaming(cmd: list, provider: str, raw: bool,
                        partial: bool = True, prompt: str = "",
                        mailbox=None) -> int:
    """Run one provider CLI, parse its JSONL and render progress live.

    Claude's rate-limit verdict, if streamed, is left in
    ``last_rate_limit_event()``. Codex limits are queried separately through
    its app-server before and after turns.

    `mailbox` is this run's `operator.Mailbox`: while the turn is in flight it
    holds the process's stdin, so a note typed at the console reaches the agent
    mid-iteration. The channel is closed on the turn's `result` event and again
    in the `finally` — closing stdin is what ends a streaming-input session, so
    an unclosed pipe is a run that never returns, and the redundancy is
    deliberate: a crash that swallows `result` must not be able to hang the loop.
    """
    global _last_rate_limit_event, _turn_cost_base
    _last_rate_limit_event = None
    _turn_cost_base = 0.0     # a new process starts a new cost total
    spec = provider_spec(provider)
    try:
        proc = start_agent_process(cmd, provider, prompt, PROJECT_DIR)
    except FileNotFoundError:
        print(f"Executable {spec.executable!r} not found. "
              f"Is {spec.display_name} installed and on PATH?")
        sys.exit(2)

    provider_failed = False
    try:
        with note_channel(proc, provider, mailbox) as channel:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                if raw:
                    print(line)
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    # non-JSON line (e.g. CLI diagnostics) — print it as is
                    print(line)
                    continue
                if not isinstance(ev, dict):
                    # A JSON scalar/array is diagnostic output, not a JSONL event.
                    print(line)
                    continue
                if provider == "claude":
                    # Before rendering, so the console cannot take a note for a
                    # turn that has already reported its result.
                    if ev.get("type") == "result":
                        channel.close()
                    _render_claude_event(ev, partial, mailbox)
                else:
                    _render_codex_event(ev)
                    provider_failed = provider_failed or ev.get("type") in (
                        "error", "turn.failed")
        # Outside the `with`, so the pipe is already closed: waiting on a
        # process whose stdin is still open is the hang this whole seam exists
        # to prevent.
        returncode = proc.wait()
        return 1 if returncode == 0 and provider_failed else returncode
    except KeyboardInterrupt:
        proc.terminate()
        print("\nInterrupted by user (Ctrl+C).")
        sys.exit(130)


def run_claude_streaming(cmd: list, raw: bool, partial: bool,
                         prompt: str = "", mailbox=None) -> int:
    """Backward-compatible Claude-only wrapper.

    `prompt` is required whenever the live-message transport is on — the argv
    then carries no task, only flags.
    """
    return run_agent_streaming(cmd, "claude", raw, partial, prompt=prompt,
                               mailbox=mailbox)


def _fmt_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _fmt_left(seconds: float) -> str:
    """"4d3h" / "3h24m" / "24m" — a duration in the two largest units that matter.

    The zero-valued smaller unit is dropped ("3h", not "3h0m"), and anything under
    a minute reads "<1m" rather than "0m", so a countdown never looks like it is
    already over. Two units is the point: a weekly window has days left, and
    "4320 min" is not a quantity anyone reads.
    """
    total = max(0, int(seconds))
    days, rest = divmod(total, 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60
    if days:
        return f"{days}d{hours}h" if hours else f"{days}d"
    if hours:
        return f"{hours}h{minutes}m" if minutes else f"{hours}h"
    return f"{minutes}m" if minutes else "<1m"


def _fmt_moment(ts: float) -> str:
    """Like _fmt_clock, but names the day too once the moment is far enough away
    that a bare clock reading would be ambiguous — a weekly quota resets days out,
    and "12:59:59" alone reads as "in a few hours"."""
    if ts - time.time() < 18 * 3600:
        return _fmt_clock(ts)
    return datetime.fromtimestamp(ts).strftime("%b %d, %H:%M")


def wait_until(target_ts: float, reason: str = None) -> None:
    """Sleep until wall-clock time reaches target_ts, printing a periodic countdown.

    Used after a probable token-limit error, or once the LimitPolicy decides the
    account's real usage figures leave no room: we idle until the 5-hour session
    window should have refreshed. `reason` overrides the default opening line.
    Ctrl+C interrupts the wait and stops the script.
    """
    if reason is None:
        reason = ("Looks like the token limit is exhausted. Waiting until "
                  f"{_fmt_clock(target_ts)} (until the 5-hour session window refreshes)…")
    print(f"  ⏳ {reason}")
    try:
        while True:
            now = time.time()
            remaining = target_ts - now
            if remaining <= 0:
                break
            print(f"    … {_fmt_left(remaining)} left (now {_fmt_clock(now)})",
                  flush=True)
            time.sleep(min(remaining, 60))
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)
    print("  ▶ The session window should have refreshed — continuing the loop.")


def wait_for_stop_file_clear() -> None:
    """Hold a launch back while a stop request is still pending, and return once
    the sentinel is gone.

    The stop file is a request aimed at a *running* loop, and the run that obeys
    it claims it and clears it after cleanup. A launch that finds one already
    there therefore has no good way to start: removing it would silently cancel
    someone else's brake, and
    obeying it would exit before doing any work at all. So it waits instead —
    for the loop that owns the request to clear it on its way out, or for the
    user to delete the file by hand — and then starts clean. Ctrl+C interrupts
    the wait and stops the script.

    Not called on a dry run: that mode never touches the sentinel and reports it
    instead.
    """
    if not os.path.exists(STOP_FILE):
        return
    print(f"  ⏸ Stop file present at startup ({STOP_FILE}) — waiting for it to "
          f"go away before starting (remove it to begin; Ctrl+C to abort)…")
    waited = 0
    try:
        while os.path.exists(STOP_FILE):
            time.sleep(STOP_POLL_SECONDS)
            waited += STOP_POLL_SECONDS
            if waited % 60 == 0:
                print(f"    … still waiting ({waited // 60} min, now "
                      f"{_fmt_clock(time.time())})", flush=True)
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)
    print("  ▶ Stop file removed — starting.")


def wait_before_start(spec: str) -> None:
    """Idle for the duration given by --start-in before the loop begins.

    Lets you launch the script and walk away; work kicks off after the delay.
    Ctrl+C interrupts the wait and stops the script.
    """
    try:
        seconds = parse_duration(spec)
    except ValueError as e:
        print(f"Invalid --start-in value {spec!r}: {e}")
        sys.exit(2)
    if seconds <= 0:
        return
    target_ts = time.time() + seconds
    print(f"  ⏳ --start-in {spec}: waiting until {_fmt_clock(target_ts)} before starting…")
    try:
        while True:
            now = time.time()
            remaining = target_ts - now
            if remaining <= 0:
                break
            print(f"    … {_fmt_left(remaining)} left (now {_fmt_clock(now)})",
                  flush=True)
            time.sleep(min(remaining, 60))
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)
    print("  ▶ Starting the loop.")


class RunSettings:
    """The script's own knobs, held in one MUTABLE object the loop re-reads.

    Plain locals froze these at startup, which made "edit the limits while the
    run goes" (the status line's `l` key) impossible without touching the loop
    body. The loop now reads this object at every iteration boundary, so moving
    `--max-runs` from 40 to 60 mid-run takes effect at the next boundary and
    nothing else has to change.
    """

    def __init__(self, *, max_runs: Optional[int] = None,
                 git_push: "GitPushPolicy" = None):
        self.max_runs = max_runs
        self.git_push = git_push or GitPushPolicy(GIT_PUSH_POLICY)


def _script_settings(run_settings: RunSettings, statusline) -> "SettingsRegistry":
    """The script's knobs as a SettingsRegistry — the display AND edit surface.

    One registry is the single source of truth for both the pinned row
    (`status_entries()`) and the reproducing command line (`overrides()`), so a
    figure on screen can never disagree with the flag that would reproduce it;
    the flags are checked against `cmdline.FLAG_ALIASES` at registration.
    `statusline` is passed in because cyclecore must not import it at module
    level (statusline imports cyclecore).
    """
    registry = statusline.SettingsRegistry()
    registry.add(statusline.NumberSetting(
        "max-runs", "--max-runs",
        lambda: run_settings.max_runs,
        lambda value: setattr(run_settings, "max_runs",
                              None if value is None else int(value)),
        minimum=1,
        # Editable and reproducible, but not a field of its own: the counter
        # already ends in this number (`iter 11/40`), or in the list's size when
        # that is the smaller of the two — see InvocationProgress.summary_fields.
        # Off the row it also stops printing `max-runs off` for every run that
        # never set one.
        show_in_status=False))
    registry.add(statusline.Setting(
        "git-push", "--git-push",
        lambda: run_settings.git_push.value,
        lambda value: setattr(run_settings, "git_push", GitPushPolicy(value))))
    return registry


class LoopStop(Exception):
    """Raised by a Driver to abort the whole run (not a normal completion).

    `exit_code` is the process exit status: non-zero for an error stop that needs
    a human (the loop sys.exit()s immediately, skipping the final push), 0 for a
    clean stop. `message` is printed before exiting.
    """

    def __init__(self, message: str, exit_code: int = 0):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class Driver:
    """What the generic loop needs from a task. Subclass and override.

    A Driver is customised two ways, both declarative:

      * class attributes for the labels the entry points use — ``app_name``
        (names the rotating mirror log) and ``prog`` (the --help program name)
        both default to None and are then derived from the invoked script's
        filename (runTranslate.py -> "runTranslate" / "runTranslate.py"), so a
        typical wrapper need not set them at all; ``description`` (the --help
        description) is free prose, so set it if you want one; ``limit_policy``
        (a limits.LimitPolicy) picks the usage-limit specialisation, defaulting
        to a day/night session rule when unset. Override any of them on your
        subclass to pin an explicit value.
      * methods for behaviour — ``next_command()``, ``model()``, ``on_success()``,
        ``final_summary()``. Override the ones you need; the rest keep their
        default.

    The loop owns all the scaffolding (stop file, git push, usage limits,
    --max-runs, streaming render); the Driver only decides *what work to
    do*. A project wrapper is then just::

        class MyDriver(StateFileDriver):
            state_file = "products/currentState.md"
            app_name   = "runCycle"

        if __name__ == "__main__":
            MyDriver.main()

    ``main()`` parses the shared CLI and hands a fresh instance to run_loop(); the
    subclass never touches parse_args / run_loop by hand.
    """

    # --- labels used by the entry points (override on the subclass) -----------
    # None => derive from the invoked script's filename (see resolved_app_name /
    # resolved_prog); set an explicit string to override.
    app_name: Optional[str] = None      # names the rotating mirror log file
    prog: Optional[str] = None          # --help program name
    description: Optional[str] = None   # --help description (None = generic)
    provider: str = "claude"            # may be overridden by --codex

    # --- usage-limit specialisation (declarative, like the labels above) ------
    # A limits.LimitPolicy picking which quota(s) to gate on and at what
    # ceiling; None => the engine's default (a day/night session rule, see
    # limits.default_policy). Set it as a class attribute to specialise, e.g.
    #   limit_policy = LimitPolicy([SessionLimit(80)])            # flat session
    #   limit_policy = LimitPolicy([WeeklyLimit(90)])             # weekly cap
    #   limit_policy = LimitPolicy([DayNightLimit(), WeeklyLimit(90)])  # composite
    # LimitPolicy/rules are stateless, so a shared default instance is safe here.
    limit_policy = None

    @classmethod
    def resolved_app_name(cls) -> str:
        """The mirror-log label: ``app_name`` if set, else the invoked script's
        filename stem (runTranslate.py -> "runTranslate"). Deriving it keeps each
        entry point on its own log without the wrapper having to spell it out; the
        fallback covers odd argv[0] values (e.g. ``-c``)."""
        if cls.app_name:
            return cls.app_name
        return Path(sys.argv[0]).stem or "runCycle"

    @classmethod
    def resolved_prog(cls) -> str:
        """The --help program name: ``prog`` if set, else the invoked script's
        basename (runTranslate.py -> "runTranslate.py")."""
        if cls.prog:
            return cls.prog
        return os.path.basename(sys.argv[0]) or "runCycle.py"

    @classmethod
    def add_cli_options(cls, parser: argparse.ArgumentParser) -> None:
        """Add this wrapper's own options to the shared --help. Default: none.

        Called by main() and main_parallel() with the parser that is about to
        run, so an entry point can document (or genuinely add) options the
        engine knows nothing about. The usual case is a MODE switch — one the
        wrapper must read out of argv itself, because it decides which of the
        two parsers runs at all, and which therefore can never be a plain option
        of either. Undocumented, such a flag exists only in prose, and `--help`
        answers "there is no such option" to a user who is looking straight at
        the one they want.

        Both entry points call it, so a flag spelled the same in both modes is
        documented in both --helps from one override. Build the option strings
        from the same constants the wrapper's argv scan uses; a second spelling
        typed out here is a spelling that can drift.
        """

    def next_command(self) -> Optional[AgentCommand]:
        """The command to run this iteration, or None when work is exhausted and
        the loop should stop normally. May raise LoopStop to abort the run."""
        raise NotImplementedError

    def model(self) -> str:
        """The selected provider's model for this iteration.

        Called by next_command() implementations to fill in AgentCommand.model.
        The default returns "" — no --model flag, so the provider CLI uses its
        own configured model. Override this (the single model knob) to pin a
        specific model, pick a cheaper/faster one for mechanical work (e.g. a
        list driver translating files needs less than the main state machine), or
        vary the model per iteration (read whatever state you like inside).
        """
        return ""

    def on_success(self, returncode: int) -> None:
        """Called after an iteration whose provider CLI exited 0 — record progress
        here (mark a file done, advance a cursor). Default: nothing to do."""

    def final_summary(self) -> Optional[str]:
        """An optional closing line printed on the way out (after the final
        git push). Return None for no summary."""
        return None

    @classmethod
    def main(cls, argv=None) -> RunResult:
        """Parse the shared CLI and run the sequential loop over a fresh instance.

        This is the whole body of a project wrapper: subclass, override the
        methods you need, then ``if __name__ == "__main__": MyDriver.main()``.
        ``prog`` labels the --help text and ``app_name`` names the log; both are
        taken from the (sub)class or derived from the script filename when unset,
        and ``description`` is the (optional) --help blurb.
        """
        args = parse_args(argv, prog=cls.resolved_prog(),
                          description=cls.description,
                          extra_options=cls.add_cli_options)
        return run_loop(cls(), args, app_name=cls.resolved_app_name())


@stop_file_lifecycle()
def run_loop(driver: Driver, args: argparse.Namespace,
             app_name: str = "runCycle", *, setup_logging: bool = True,
             wait_on_start: bool = True, progress=None) -> RunResult:
    """Drive the selected provider per `driver`, with all shared lifecycle
    machinery. This is the former runCycle.main(), generalised: the only thing
    that changed is that "read currentState.md and pick a prompt" became
    `driver.next_command()`, and the closing "Final state" line became
    `driver.final_summary()`.

    `progress` is the whole invocation's InvocationProgress, for a wrapper that
    makes several runner calls in one process (see run_parallel); left None, this
    call is the invocation and owns its own figures.
    """
    # The usage-limit query/parse (UsageSource) and pausing policy (LimitPolicy)
    # live in their own modules; imported here to avoid an import cycle
    # (limits/usage/statusline import cyclecore for its helpers). drivers imports
    # this module, so ListFileDriver comes in here rather than at module level.
    from . import limits, statusline
    from .drivers import ListFileDriver

    owns_progress = progress is None
    if owns_progress:
        progress = statusline.InvocationProgress(max_items=args.max)

    provider = getattr(args, "provider", None) or driver.provider
    spec = provider_spec(provider)
    driver.provider = provider
    # The live knobs (see RunSettings): read at each iteration boundary, never
    # snapshotted into locals, so the status line's editor can move them mid-run.
    run_settings = RunSettings(
        max_runs=args.max,                          # None = no limit
        git_push=GitPushPolicy(args.git_push))      # when to `git push` each iteration
    # When a finite iteration cap is given (-m/--max-runs) the run is short and
    # bounded on purpose, so the usage-limit machinery (the LimitPolicy
    # pause-on-limit logic) is skipped — we just run the requested iterations
    # without ever waiting out a window. Decided once, from the value the run was
    # LAUNCHED with: it also governs whether this run talks to the usage endpoint
    # at all, which is a property of the invocation, not of the current cap.
    ignore_usage_limits = (args.max is not None
                           or not spec.supports_usage_limits)
    dry_run = args.dry_run
    raw = args.raw
    start_in = args.start_in      # e.g. "29m" — delay before the loop starts
    # Decided per invocation, before the first argv is built: the transport is
    # what --no-live-messages turns off, and both the argv and the process's
    # stdin have to agree about it. Set in BOTH directions — a wrapper that
    # calls two runners in one process (see runGenerateModels' periodic mode)
    # would otherwise have the first `--no-live-messages` phase decide the
    # transport for every phase after it.
    set_live_messages(not getattr(args, "no_live_messages", False))

    # Anchor every project-relative operation (git/provider cwd, the stop file, the
    # log name, the Driver's paths) to the chosen root before anything reads it.
    set_project_root(getattr(args, "project_dir", None))

    # --cost: report per-run spend from the mirror log and exit, without touching
    # the loop, the tee, git, or the usage gate. Done here (after the root is
    # set, so the log path resolves) rather than in the loop body proper.
    # --cost-log implies --cost: naming a log to read and getting a loop run
    # instead would be a silent misfire, and there is nothing else it could mean.
    cost_log = getattr(args, "cost_log", None)
    if getattr(args, "cost", False) or cost_log:
        report_costs(app_name, cost_log)
        return RunResult(RunStopReason.NO_WORK)

    # Mirror all screen output into a rotating log file under the home dir —
    # except for a dry run, which is a preview and not a run: its output would
    # otherwise displace real runs' records out of the shared rotating log (a
    # preview once pushed ~26 MB through it, and the failure it was launched to
    # explain rotated off the end). Said on screen so the missing log is visible
    # rather than mysterious.
    if setup_logging and not dry_run:
        logger = _setup_file_logging(app_name)
        sys.stdout = _TeeToLog(sys.stdout, logger)
        sys.stderr = _TeeToLog(sys.stderr, logger)
    if not dry_run:
        # After the tee, so the report of a run that vanished lands in the very
        # log whose abrupt end it explains. Idempotent per process: the periodic
        # wrapper calls this runner repeatedly and keeps one record.
        exitlog.begin(app_name, LOG_DIR, os.path.basename(PROJECT_DIR))
    print(f"  · project root: {PROJECT_DIR}")
    if dry_run:
        print(f"  · dry run: nothing is mirrored to {log_file_path(app_name)}")
    else:
        print(f"  · logging to {log_file_path(app_name)}")
    print(f"  · provider: {spec.display_name}")
    if not _RICH_AVAILABLE:
        print("  · Markdown rendering is off (the 'rich' library is missing). "
              "Enable it with:")
        print(f"      {sys.executable} -m pip install rich")

    # A stop request pending from another run: wait it out rather than consume
    # it, so this launch starts on a clean sentinel instead of stopping on its
    # first iteration boundary. Before --start-in: the point is to begin as soon
    # as the brake is off, not to burn the delay while it is still on.
    if not dry_run and wait_on_start:
        wait_for_stop_file_clear()

    if start_in and not dry_run:
        wait_before_start(start_in)

    session_start = time.time()   # start of the current 5-hour session window
    consecutive_errors = 0        # reset to 0 after any successful iteration
    usage_source = usage_source_for(provider)
    limit_policy = None
    if usage_source is not None:
        limit_policy = driver.limit_policy or limits.default_policy(provider)
    last_git_push = 0.0           # epoch time of the last `git push` (0 = never)
    print(f"  · git push policy: {run_settings.git_push.value}")
    if ignore_usage_limits:
        print(f"  · usage limit policy: disabled (bounded run, "
              f"--max {run_settings.max_runs})")
    else:
        print_percents(f"  · usage limit policy: {limit_policy.describe()}")

    # Bookend the run with a usage snapshot (the policy's watched quotas) so each
    # run records where it started; the matching end-of-run snapshot is below.
    if not dry_run and usage_source is not None:
        limit_policy.log_snapshot(usage_source, "at start (iteration 1)")

    # The pinned status area. A Job is the unit of display in both runners, so
    # the sequential loop is a run with exactly one Job — no branch anywhere in
    # the status line separates it from `-j N`. Disabled it is a Null object, so
    # every call below stays a no-op and the loop behaves exactly as before.
    settings = _script_settings(run_settings, statusline)
    # A list driver's list is the source of truth for how much work this
    # invocation has: the summary row counts items STRUCK out of it, not
    # iterations, so a retried item is not progress and a preflight that strikes
    # finished ones is. Any other Driver has no such total and counts iterations.
    list_driven = isinstance(driver, ListFileDriver)
    if list_driven:
        progress.track_list(len(driver.pending_lines()))
    # One mailbox for the whole run: the console writes to it, the loop below
    # empties it into the next prompt, and run_agent_streaming lends it the
    # running turn's stdin. A dry run gets none — there is no agent to talk to.
    mailbox = None if dry_run else operator.Mailbox()
    app = statusline.StatusApp(
        # From the invocation's pool: a wrapper's next runner call resumes this
        # row instead of starting a fresh Job at iteration 1.
        status=statusline.LoopStatus(jobs=progress.jobs(1)),
        settings=settings,
        messages=mailbox,
        enabled=not dry_run and not getattr(args, "no_statusline", False))
    app.update(
        provider=provider,
        **progress.summary_fields(),
        # Only a list driver has a pick order; read it defensively so any other
        # Driver simply reports no `rand` marker.
        random_order=str(getattr(driver, "pick_order", "")) == "random",
        script_limits=settings.status_entries(),
    )

    iteration = 0
    completed = 0
    stop_reason = RunStopReason.NO_WORK
    stop_file_noted = False       # dry-run: report the sentinel once, not per iteration
    dry_run_prompt_shown = False  # dry-run: show job 1's prompt once, not per pass
    with app:
        if usage_source is not None:
            # Inside `with`, not before it: push_quotas is silent until start()
            # has marked the app enabled, so priming it earlier left requirement
            # #1 — the provider's live figures — blank until the first limit
            # check (i.e. for the whole run when the checks are skipped). The
            # reading itself is already paid for by the start-of-run snapshot
            # above, so this costs no round-trip.
            statusline.push_quotas(app, usage_source, limit_policy)
            if not ignore_usage_limits:
                # Only for a run that is allowed to talk to the usage endpoint at
                # all: the poll forces a FRESH reading (cache_value=False), so on
                # a bounded `-m N` run — whose whole point is not to touch the
                # usage machinery — it would be an unsolicited HTTP GET every
                # interval (a codex app-server call every 300 s).
                app.add_service(statusline.QuotaRefresher(
                    app, usage_source, limit_policy, provider=provider))
        while True:
            # The caps are read LIVE (see RunSettings) and republished here, so an
            # edit made while the run is going takes effect at this boundary and
            # is what the pinned row shows. Under a wrapper the invocation's cap
            # is the wrapper's to set — this call's cap only sizes one batch.
            if owns_progress:
                progress.max_items = run_settings.max_runs
            app.update(**progress.summary_fields(),
                       script_limits=settings.status_entries())
            if os.path.exists(STOP_FILE):
                # The sentinel is removed only after the outer application has
                # finished cleanup. A dry run must not claim it. `-d` is routinely
                # used to preview commands while a real loop is running — and that is
                # exactly when a
                # stop request is pending — so removing it here would silently cancel
                # someone else's stop, and the loop it was meant to halt would run on.
                # Report it and leave it for whoever it was written for.
                if dry_run:
                    if not stop_file_noted:
                        print("Stop file present — a real run would have waited for "
                              "it at startup, and stops here if it appears mid-run. "
                              "Left in place (a dry run never consumes it).")
                        stop_file_noted = True
                elif confirm_stop_request(app):
                    mark_stop_file_detected()
                    print("Stop file detected — stopping; it remains in place until "
                          "the application exits.")
                    app.update(phase="stopping")
                    stop_reason = RunStopReason.STOP_FILE
                    break
                # Cancelled inside the interactive grace — carry on with no trace.

            # Git push policy: evaluated at the start of every iteration.
            if not dry_run:
                last_git_push = maybe_git_push(run_settings.git_push, last_git_push)

            max_runs = run_settings.max_runs
            if max_runs is not None and iteration >= max_runs:
                print(f"Iteration limit reached (--max-runs {max_runs}). Stopping.")
                stop_reason = RunStopReason.LIMIT_REACHED
                break

            # Proactive limit check: read the real Current-session usage from the
            # account and pause cleanly between iterations if it is already at/over
            # the threshold, instead of running an iteration that would hit the wall.
            if not dry_run and not ignore_usage_limits:
                app.update(phase="waiting")
                paused, session_start = limit_policy.check_and_wait(usage_source, session_start)
                # The check just paid for a usage reading; publishing it here is
                # what puts the provider's live limits on the status row without
                # a second HTTP round-trip (the UsageSource cache serves it).
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if paused:
                    consecutive_errors = 0  # fresh window — start counting errors anew

            # Ask the driver what to do next. None => no more work (stop cleanly);
            # LoopStop => abort the run (e.g. an error state needing a human).
            try:
                command = driver.next_command()
            except LoopStop as stop:
                print(stop.message)
                if stop.exit_code:
                    exitlog.set_reason(
                        f"the driver stopped the run (exit {stop.exit_code}): "
                        f"{stop.message.splitlines()[0]}")
                    sys.exit(stop.exit_code)
                stop_reason = RunStopReason.DRIVER_STOP
                break
            if command is None:
                print("No more work — stopping.")
                stop_reason = RunStopReason.NO_WORK
                break

            # Notes typed while nothing was running (or while the transport was
            # off) ride this prompt — see Mailbox.splice for the ordering.
            if mailbox is not None:
                spliced, notes = mailbox.splice(command.prompt)
                if notes:
                    command = command._replace(prompt=spliced)
                    for note in notes:
                        print_note(note)

            iteration += 1
            state_label = command.label or "(no label)"
            # Show the model this iteration will use right in the header, so the
            # per-iteration model is visible up front (an empty command.model means
            # no --model flag — the CLI falls back to its own configured default).
            model_label = command.model or "cli default"
            # Same three calls the parallel workers make, on this run's one Job:
            # the Job clock times THIS iteration, the run clock (latched once)
            # times the whole run.
            started_at = time.time()
            app.mark_run_started(started_at)
            # The Job bumps its own counter (no `iteration=`): the local counter
            # restarts with every runner call, and pinning the row to it is what
            # kept a periodic run's job rows at 1.
            app.job(1).start(item=state_label, model=command.model,
                             prompt=command.prompt, now=started_at)
            if not list_driven:
                progress.note_iteration()
            app.update(**progress.summary_fields(), phase="running")
            # What a post-mortem needs from a run that never got to write an
            # ending: which item it was on when it stopped existing.
            exitlog.note(phase=f"iteration {iteration} — {state_label}",
                         iterations=iteration, completed=completed)
            print_markup(
                f"\n=== Iteration {iteration} === [{state_label} · {model_label}]",
                f"\n[bold cyan]=== Iteration {iteration} ===[/] "
                f"[dim]\\[{state_label} · {model_label}][/]",
            )

            cmd = build_agent_argv(command, provider)
            if dry_run:
                print("DRY-RUN:", " ".join(cmd))
                if prompt_on_stdin(provider):
                    # The argv above is complete but not self-contained: the
                    # prompt travels on stdin, so the preview has to show it
                    # separately or it shows a command with no task in it.
                    print("STDIN:", command.prompt)
                if not dry_run_prompt_shown:
                    # The argv line above is what will be executed, but for
                    # claude the whole prompt sits inside one joined `-p …`
                    # token and is unreadable — and reading the prompt is what a
                    # dry run is for. Printed once, for job 1.
                    dry_run_prompt_shown = True
                    print(statusline.format_prompt_block(
                        job_id=1, label=state_label, prompt=command.prompt,
                        width=statusline.screen_width()))
                # looping forever in dry-run is pointless — nothing is actually done,
                # so the driver would keep handing back the same first unit of work.
                if run_settings.max_runs is None:
                    print("(dry-run without --max-runs: running a single iteration and exiting)")
                    stop_reason = RunStopReason.DRY_RUN
                    break
                continue

            if provider == "claude":
                returncode = run_claude_streaming(
                    cmd, raw, partial=True, prompt=command.prompt,
                    mailbox=mailbox)
            else:
                returncode = run_agent_streaming(
                    cmd, provider, raw, partial=False, prompt=command.prompt)
            app.job(1).finish()
            app.update(phase="idle")

            if returncode == 0:
                consecutive_errors = 0
                completed += 1
                driver.on_success(returncode)
                if list_driven:
                    # on_success struck the item, so the list itself now says how
                    # far the invocation has got.
                    progress.note_remaining(len(driver.pending_lines()))
                    app.update(**progress.summary_fields())

            # The backstop under the proactive check (see RateLimitEvent): this run's
            # own verdict from the wire. A refusal needs no figure and no query to be
            # trusted, so it is honoured whatever the usage report said — including
            # when the report was unavailable, which is the case this exists for.
            # Checked for both outcomes: a run refused on its last turn may still have
            # exited 0 with its work recorded above.
            refusal = last_rate_limit_event() if provider == "claude" else None
            if (not ignore_usage_limits and refusal is not None
                    and refusal.status == "rejected"):
                # +5s so we come back after the reset, not exactly on it.
                target_ts = (refusal.resets_at
                             or time.time() + CLAUDE_SESSION_DURATION) + 5
                app.update(phase="paused")
                wait_until(target_ts,
                           reason=f"Hit the {refusal.label} — this run was refused. "
                                  f"Waiting until {_fmt_moment(target_ts)} for that "
                                  f"window to refresh…")
                usage_source.invalidate()  # the figures behind the refusal are stale
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if refusal.limit_type == "five_hour":
                    session_start = time.time()
                consecutive_errors = 0  # fresh window — start counting errors anew
                continue

            if returncode == 0:
                continue

            # Non-zero exit — the cause is ambiguous (a network blip / one-off CLI
            # hiccup, or the session's token limit). Rather than guessing from a
            # second consecutive error, ask the account directly: read the usage
            # figures right away and let the real Current-session percentage decide.
            consecutive_errors += 1
            elapsed = time.time() - session_start
            print_error(f"{spec.display_name} exited with code {returncode} "
                        f"(error #{consecutive_errors} in a row).")

            if not ignore_usage_limits:
                app.update(phase="waiting")
                paused, session_start = limit_policy.check_and_wait(
                    usage_source, session_start, note=" (checked after error)")
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if paused:
                    consecutive_errors = 0  # fresh window — start counting errors anew
                    continue

            # Session is under the limit — this was a transient failure, not token
            # exhaustion. Retry, but don't spin forever if something is truly broken.
            if consecutive_errors < 5:
                if usage_source is None:
                    print("  ↻ Provider quota status is unavailable — retrying immediately.")
                else:
                    print("  ↻ Session under the allowed limit — likely transient. "
                          "Retrying immediately.")
                continue
            else:
                quota_state = ("with provider quota status unavailable"
                               if usage_source is None
                               else "with the session under the allowed limit")
                print(f"  ⚠ {consecutive_errors} errors in a row {quota_state} after "
                      f"{int(elapsed // 60)} min. Stopping.")
                exitlog.set_reason(
                    f"{consecutive_errors} provider errors in a row "
                    f"(last exit code {returncode})",
                    iterations=iteration, completed=completed)
                sys.exit(returncode)

    # Final push: regardless of the EACH_HOUR cadence, push any pending commits
    # on the way out so work isn't left only on the local branch — unless the
    # policy is NONE (never auto-push).
    if not dry_run and run_settings.git_push != GitPushPolicy.NONE:
        count = git_unpushed_count()
        if count is None or count > 0:
            print("  · final git push on exit…")
            git_push()
        else:
            print("  · final git push: nothing to push.")

    # End-of-run usage snapshot (the policy's watched quotas), mirroring the one
    # logged before iteration 1 — so each run records where it finished. Forced
    # fresh (cache_value=False) so it reflects the true post-run state rather
    # than a possibly-recent cached reading from the last limit check.
    if not dry_run and usage_source is not None:
        limit_policy.log_snapshot(usage_source, "at end (after last cycle)",
                                  cache_value=False)

    report_undelivered_notes(mailbox)

    # Closing line, if the driver has one (e.g. "Final state: …").
    summary = driver.final_summary()
    if summary:
        print(f"\n{summary}")
    # The reason this runner returned. Recorded rather than printed: a wrapper
    # may call several runners, and the `=== run ended: … ===` line belongs to
    # the process, so the last reason set wins and exitlog prints it on exit.
    exitlog.set_reason(STOP_REASON_TEXT.get(stop_reason, stop_reason.value),
                       iterations=iteration, completed=completed)
    return RunResult(stop_reason, iteration, completed)
