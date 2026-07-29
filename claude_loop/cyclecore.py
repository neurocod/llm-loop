"""
cyclecore.py - reusable engine behind the autonomous Claude-CLI loops.

This module holds everything that is *not* specific to one particular task:
command-line parsing, the rotating mirror log, the git-push policy, the whole
token-usage/session-window machinery, stream-json rendering, and the generic
`run_loop()` that ties it together. The only thing it does **not** decide is
*what work to do each iteration* — that is supplied by a `Driver` (see below),
so the same engine drives both:

  * runCycle.py    — a state-machine driver reading products/currentState.md;
  * runTranslate.py — a list driver translating files from products/list.md.

A Driver answers three questions for the loop, via three hooks:

  * next_command()  -> ClaudeCommand | None
        The command to run this iteration, or None when there is no more work
        and the loop should stop normally. It may raise LoopStop to abort the
        whole run (e.g. an error state that needs a human).
  * on_success(rc)  -> None
        Called after an iteration whose `claude` exited 0 — the place to record
        progress (mark a file done, advance a cursor, …). Default: no-op.
  * final_summary() -> str | None
        A closing line printed on the way out (e.g. "Final state: …"). Optional.

Everything below is lifted verbatim from the original single-file runCycle.py,
with the few state-specific pieces (which file to read, which prompt to send,
which model to pick) factored out into the Driver protocol.

Token-limit handling is driven by the CLI's own usage report rather than guessed
from error counts. Before each iteration, and again immediately after any
non-zero `claude` exit, the loop asks the Driver's `limit_policy` to consult
`claude -p "/usage"` and pause if any watched quota is at/over its ceiling. The
querying/parsing lives in usage.py (UsageSource) and the pausing policy in
limits.py (LimitPolicy and the ready-made SessionLimit / DayNightLimit /
WeeklyLimit rules); this module only wires them into the loop.
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import NamedTuple, Optional

# Claude sessions last ~5 hours; after a token-limit error we wait out that window.
CLAUDE_SESSION_DURATION = 5 * 60 * 60 + 3  # 5 hours as seconds and + 3s as a safety margin
SESSION_DURATION = 3600 # or: if session is started at the end of a window, it may continue more from the start next time
LIMIT_RETRY_THRESHOLD = CLAUDE_SESSION_DURATION

# The usage-limit policy (which /usage quota to gate on, what ceiling to allow,
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
# subprocesses (git, claude, /usage) run with this as their cwd, the relative
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
# finishes its one state transition first. The file is consumed on stop so the
# next launch starts clean. Recomputed by set_project_root().
STOP_FILE = os.path.join(PROJECT_DIR, "stop")

# How often wait_for_stop_file_clear() re-checks the sentinel while it holds a
# launch back. Short enough that removing the file feels immediate.
STOP_POLL_SECONDS = 2


def set_project_root(path: Optional[str]) -> str:
    """Point the engine at the project root (cwd for git/claude, base for the
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
    anywhere else and it lands on the same root — so git/claude run there, the
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

# Rotation policy for the mirror log. Exposed as module-level constants so other
# tools (e.g. sum_session_costs.py) can report how full the log is against the
# same limit instead of hard-coding it.
LOG_MAX_BYTES = 25 * 1024 * 1024
LOG_BACKUP_COUNT = 3


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


def report_costs(app_name: str = "runCycle") -> None:
    """Print per-session (per-run) cost totals parsed from the mirror log, then
    exit — the standalone counterpart reached via the --cost flag.

    A "session" is one run of the loop, delimited by its "=== Iteration 1 ==="
    header; within it every "done (… c, $…)" line contributes its dollar cost. We
    print a line per session, a grand total, and how full the log is against the
    rotation limit (LOG_MAX_BYTES). Resolve the log via log_file_path(app_name),
    so --cost reports on the very log this entry point writes.
    """
    path = log_file_path(app_name)
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


def _setup_file_logging(app_name: str = "runCycle") -> logging.Logger:
    """Configure a rotating file logger at log_file_path(app_name) (25 MB x 3)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"runCycle.{app_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:  # avoid duplicate handlers if called twice
        handler = RotatingFileHandler(
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

    def __init__(self, stream, logger: logging.Logger):
        self._stream = stream
        self._logger = logger
        self._buf = ""

    def write(self, text: str) -> int:
        self._stream.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._logger.info(line)
        return len(text)

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name):
        # Delegate everything else (encoding, isatty, fileno, ...) to the stream.
        return getattr(self._stream, name)


def parse_args(argv=None, *, prog: str = "runCycle.py",
               description: Optional[str] = None) -> argparse.Namespace:
    """Command-line interface shared by every entry point. Every long option has
    a single-letter alias.

    `prog`/`description` let each entry script label its own --help text while
    reusing the exact same option set (so there is no duplicated argument code).
    """
    p = argparse.ArgumentParser(
        prog=prog,
        description=description or "Autonomous loop driving the Claude CLI.",
    )
    # Option names are kept in sync with continuous_claude.py (kebab-case, and
    # --max-runs for the iteration cap). The former spellings (--max, --startIn,
    # --maxStrike) stay on as accepted aliases so existing invocations keep working.
    p.add_argument("-m", "--max-runs", "--max", dest="max", type=int, default=None,
                   metavar="N",
                   help="stop after N iterations (default: run forever); "
                        "--max is a deprecated alias")
    p.add_argument("-d", "--dry-run", action="store_true",
                   help="only print the commands, don't run claude")
    p.add_argument("-c", "--cost", action="store_true",
                   help="print per-session cost totals from the mirror log and "
                        "exit (no loop is run)")
    # No -r short flag: -r is --review-prompt in continuous_claude.py, so it is
    # left free here rather than reused for --raw.
    p.add_argument("--raw", action="store_true",
                   help="print raw JSON events (for debugging)")
    p.add_argument("-s", "--start-in", "--startIn", dest="start_in", metavar="DURATION",
                   help="wait this long before starting the loop, e.g. 29m, 1h30m")
    p.add_argument("-S", "--max-strike", "--maxStrike", dest="max_strike",
                   metavar="DURATION",
                   help="per-session work budget before a pre-emptive pause, e.g. 3h")
    p.add_argument("-g", "--git-push", dest="git_push",
                   choices=[pol.value for pol in GitPushPolicy],
                   default=GIT_PUSH_POLICY.value,
                   help="when to `git push` at the start of each iteration: "
                        "none | after_new_commits | each_hour "
                        f"(default: {GIT_PUSH_POLICY.value})")
    p.add_argument("-C", "--project-dir", dest="project_dir", metavar="DIR",
                   default=None,
                   help="project root: cwd for git/claude, base for the stop "
                        "file and the Driver's relative paths "
                        "(default: the current working directory)")
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


class ClaudeCommand(NamedTuple):
    """One unit of work for the loop: the prompt to send, the model to use, and a
    short label shown in the iteration header. Drivers build these in
    next_command(); build_claude_argv() turns one into the full `claude` argv.

    An empty `model` means "no --model flag": the `claude` CLI then uses whatever
    model it is configured to (its own default), which is the common case.
    """
    prompt: str
    model: str = ""
    label: str = ""


def build_claude_argv(command: ClaudeCommand) -> list:
    """Full `claude` command line for one ClaudeCommand.

    The flags are identical for every task; only the prompt and the model vary,
    so this is the single place those two are spliced into the otherwise fixed
    argv (stream-json + partial messages so the loop can render work live). An
    empty `command.model` omits --model entirely, letting the CLI pick its own
    configured default.
    """
    argv = ["claude", "-p", command.prompt]
    if command.model:
        argv += ["--model", command.model]
    argv += [
        "--permission-mode", "acceptEdits",
        # tools allowed without interactive confirmation
        "--allowedTools", "Bash Edit Write Read Glob Grep WebFetch WebSearch",
        # a stream of events instead of a single final answer — to see the work in progress
        "--output-format", "stream-json",
        "--verbose",
        # text deltas as they are generated (we print token by token)
        "--include-partial-messages",
    ]
    return argv


def _short(text: str, limit: int = 200) -> str:
    """Single-line truncated version of text for compact output."""
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _describe_tool(name: str, ti: dict) -> str:
    """Short human-readable description of a tool call and its arguments."""
    if name == "Bash":
        return f"$ {_short(ti.get('command', ''))}"
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


# Named single-style specialisations, each delegating to print_styled. Centralise
# the loop's palette here so a colour is changed in one place, not at every call.
def print_done(text: str) -> None:
    print_styled(text, "green")


def print_error(text: str) -> None:
    print_styled(text, "bold red")


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


def _render_event(ev: dict, partial: bool) -> None:
    """Print a single stream-json event in the style of interactive mode.

    partial=True — --include-partial-messages is enabled: we print text from the
    deltas (`stream_event`), and from the final `assistant` we take only the tool
    calls, so as not to duplicate already-printed text.
    """
    et = ev.get("type")

    if et == "system" and ev.get("subtype") == "init":
        model = ev.get("model", "?")
        print(f"  · session started (model {model})")
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
        cost = ev.get("total_cost_usd")
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


def run_claude_streaming(cmd: list, raw: bool, partial: bool) -> int:
    """Runs claude, parses stream-json on the fly and prints the work in progress."""
    try:
        proc = subprocess.Popen(
            cmd, cwd=PROJECT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except FileNotFoundError:
        print("Executable 'claude' not found. Is the Claude CLI installed and on PATH?")
        sys.exit(2)

    try:
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
            _render_event(ev, partial)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nInterrupted by user (Ctrl+C).")
        sys.exit(130)


def _fmt_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def wait_until(target_ts: float, reason: str = None) -> None:
    """Sleep until wall-clock time reaches target_ts, printing a periodic countdown.

    Used after a probable token-limit error (or once the --max-strike budget is
    spent): we idle until the 5-hour session window should have refreshed.
    `reason` overrides the default opening line. Ctrl+C interrupts the wait and
    stops the script.
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
            mins = int(remaining // 60) + 1
            print(f"    … ~{mins} min left (now {_fmt_clock(now)})", flush=True)
            time.sleep(min(remaining, 60))
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)
    print("  ▶ The session window should have refreshed — continuing the loop.")


def wait_for_stop_file_clear() -> None:
    """Hold a launch back while a stop request is still pending, and return once
    the sentinel is gone.

    The stop file is a request aimed at a *running* loop, and the run that obeys
    it consumes it. A launch that finds one already there therefore has no good
    way to start: consuming it would silently cancel someone else's brake, and
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
            mins = int(remaining // 60) + 1
            print(f"    … ~{mins} min left (now {_fmt_clock(now)})", flush=True)
            time.sleep(min(remaining, 60))
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)
    print("  ▶ Starting the loop.")


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
    --max-runs/--max-strike, streaming render); the Driver only decides *what work to
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

    # --- usage-limit specialisation (declarative, like the labels above) ------
    # A limits.LimitPolicy picking which /usage quota(s) to gate on and at what
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

    def next_command(self) -> Optional[ClaudeCommand]:
        """The command to run this iteration, or None when work is exhausted and
        the loop should stop normally. May raise LoopStop to abort the run."""
        raise NotImplementedError

    def model(self) -> str:
        """The Claude model to drive this iteration's command.

        Called by next_command() implementations to fill in ClaudeCommand.model.
        The default returns "" — no --model flag, so the `claude` CLI uses its
        own configured model. Override this (the single model knob) to pin a
        specific model, pick a cheaper/faster one for mechanical work (e.g. a
        list driver translating files needs less than the main state machine), or
        vary the model per iteration (read whatever state you like inside).
        """
        return ""

    def on_success(self, returncode: int) -> None:
        """Called after an iteration whose `claude` exited 0 — record progress
        here (mark a file done, advance a cursor). Default: nothing to do."""

    def final_summary(self) -> Optional[str]:
        """An optional closing line printed on the way out (after the final
        git push). Return None for no summary."""
        return None

    @classmethod
    def main(cls, argv=None) -> None:
        """Parse the shared CLI and run the sequential loop over a fresh instance.

        This is the whole body of a project wrapper: subclass, override the
        methods you need, then ``if __name__ == "__main__": MyDriver.main()``.
        ``prog`` labels the --help text and ``app_name`` names the log; both are
        taken from the (sub)class or derived from the script filename when unset,
        and ``description`` is the (optional) --help blurb.
        """
        args = parse_args(argv, prog=cls.resolved_prog(),
                          description=cls.description)
        run_loop(cls(), args, app_name=cls.resolved_app_name())


def run_loop(driver: Driver, args: argparse.Namespace,
             app_name: str = "runCycle") -> None:
    """Drive the Claude CLI per `driver`, with all the shared session/limit/push
    machinery. This is the former runCycle.main(), generalised: the only thing
    that changed is that "read currentState.md and pick a prompt" became
    `driver.next_command()`, and the closing "Final state" line became
    `driver.final_summary()`.
    """
    # The usage-limit query/parse (UsageSource) and pausing policy (LimitPolicy)
    # live in their own modules; imported here to avoid an import cycle
    # (limits/usage import cyclecore for its helpers).
    from . import limits
    from .usage import UsageSource

    max_iters = args.max          # None = no limit
    # When a finite iteration cap is given (-m/--max-runs) the run is short and
    # bounded on purpose, so the usage-limit machinery (the LimitPolicy
    # pause-on-limit logic) is skipped — we just run the requested iterations
    # without ever waiting out a window.
    ignore_usage_limits = max_iters is not None
    dry_run = args.dry_run
    raw = args.raw
    start_in = args.start_in      # e.g. "29m" — delay before the loop starts
    git_push_policy = GitPushPolicy(args.git_push)  # when to `git push` each iteration
    max_strike = args.max_strike  # e.g. "3h" — per-session work budget before a pre-emptive pause
    max_strike_seconds = None
    if max_strike:
        try:
            max_strike_seconds = parse_duration(max_strike)
        except ValueError as e:
            print(f"Invalid --max-strike value {max_strike!r}: {e}")
            sys.exit(2)

    # Anchor every project-relative operation (git/claude cwd, the stop file, the
    # log name, the Driver's paths) to the chosen root before anything reads it.
    set_project_root(getattr(args, "project_dir", None))

    # --cost: report per-run spend from the mirror log and exit, without touching
    # the loop, the tee, git, or /usage. Done here (after the root is set, so the
    # log path resolves) rather than in the loop body proper.
    if getattr(args, "cost", False):
        report_costs(app_name)
        return

    # Mirror all screen output into a rotating log file under the home dir.
    logger = _setup_file_logging(app_name)
    sys.stdout = _TeeToLog(sys.stdout, logger)
    sys.stderr = _TeeToLog(sys.stderr, logger)
    print(f"  · project root: {PROJECT_DIR}")
    print(f"  · logging to {log_file_path(app_name)}")
    if not _RICH_AVAILABLE:
        print("  · Markdown rendering is off (the 'rich' library is missing). "
              "Enable it with:")
        print(f"      {sys.executable} -m pip install rich")

    # A stop request pending from another run: wait it out rather than consume
    # it, so this launch starts on a clean sentinel instead of stopping on its
    # first iteration boundary. Before --start-in: the point is to begin as soon
    # as the brake is off, not to burn the delay while it is still on.
    if not dry_run:
        wait_for_stop_file_clear()

    if start_in and not dry_run:
        wait_before_start(start_in)

    session_start = time.time()   # start of the current 5-hour session window
    consecutive_errors = 0        # reset to 0 after any successful iteration
    usage_source = UsageSource()  # queries/caches `claude -p "/usage"`
    limit_policy = driver.limit_policy or limits.default_policy()  # pauses on limit
    last_git_push = 0.0           # epoch time of the last `git push` (0 = never)
    print(f"  · git push policy: {git_push_policy.value}")
    if ignore_usage_limits:
        print(f"  · usage limit policy: disabled (bounded run, --max {max_iters})")
    else:
        print(f"  · usage limit policy: {limit_policy.describe()}")

    # Bookend the run with a usage snapshot (the policy's watched quotas) so each
    # run records where it started; the matching end-of-run snapshot is below.
    if not dry_run:
        limit_policy.log_snapshot(usage_source, "at start (iteration 1)")

    iteration = 0
    stop_file_noted = False       # dry-run: report the sentinel once, not per iteration
    while True:
        if os.path.exists(STOP_FILE):
            # The sentinel is *consumed* when it stops a run: one file, one stop.
            # A dry run must not consume it. `-d` is routinely used to preview
            # commands while a real loop is running — and that is exactly when a
            # stop request is pending — so removing it here would silently cancel
            # someone else's stop, and the loop it was meant to halt would run on.
            # Report it and leave it for whoever it was written for.
            if dry_run:
                if not stop_file_noted:
                    print("Stop file present — a real run would have waited for "
                          "it at startup, and stops here if it appears mid-run. "
                          "Left in place (a dry run never consumes it).")
                    stop_file_noted = True
            else:
                os.remove(STOP_FILE)
                print("Stop file detected — stopping (the file has been removed).")
                break

        # Git push policy: evaluated at the start of every iteration.
        if not dry_run:
            last_git_push = maybe_git_push(git_push_policy, last_git_push)

        if max_iters is not None and iteration >= max_iters:
            print(f"Iteration limit reached (--max-runs {max_iters}). Stopping.")
            break

        # --maxStrike: once a finished iteration pushes us past the per-session
        # work budget, pause pre-emptively (same as a token-limit hit) so the
        # current unit of work stays whole and we don't run into the limit
        # mid-iteration. Checked between iterations, never in the middle of one.
        if max_strike_seconds is not None and iteration > 0:
            elapsed = time.time() - session_start
            if elapsed > max_strike_seconds:
                print(f"  ⌛ max-strike budget ({max_strike}) reached after "
                      f"{int(elapsed // 60)} min of work — pausing for the next "
                      f"session so this run stays whole.")
                target_ts = session_start + SESSION_DURATION
                wait_until(target_ts,
                           reason=f"maxStrike: pausing pre-emptively until "
                                  f"{_fmt_clock(target_ts)} (until the 5-hour session "
                                  f"window refreshes)…")
                session_start = time.time()
                consecutive_errors = 0  # fresh window — start counting errors anew

        # Proactive limit check: ask the CLI for the real Current-session usage
        # and pause cleanly between iterations if it is already at/over the
        # threshold, instead of running an iteration that would hit the wall.
        if not dry_run and not ignore_usage_limits:
            paused, session_start = limit_policy.check_and_wait(usage_source, session_start)
            if paused:
                consecutive_errors = 0  # fresh window — start counting errors anew

        # Ask the driver what to do next. None => no more work (stop cleanly);
        # LoopStop => abort the run (e.g. an error state needing a human).
        try:
            command = driver.next_command()
        except LoopStop as stop:
            print(stop.message)
            if stop.exit_code:
                sys.exit(stop.exit_code)
            break
        if command is None:
            print("No more work — stopping.")
            break

        iteration += 1
        state_label = command.label or "(no label)"
        # Show the model this iteration will use right in the header, so the
        # per-iteration model is visible up front (an empty command.model means
        # no --model flag — the CLI falls back to its own configured default).
        model_label = command.model or "cli default"
        print_markup(
            f"\n=== Iteration {iteration} === [{state_label} · {model_label}]",
            f"\n[bold cyan]=== Iteration {iteration} ===[/] "
            f"[dim]\\[{state_label} · {model_label}][/]",
        )

        cmd = build_claude_argv(command)
        if dry_run:
            print("DRY-RUN:", " ".join(cmd))
            # looping forever in dry-run is pointless — nothing is actually done,
            # so the driver would keep handing back the same first unit of work.
            if max_iters is None:
                print("(dry-run without --max-runs: running a single iteration and exiting)")
                break
            continue

        returncode = run_claude_streaming(cmd, raw, partial=True)

        if returncode == 0:
            consecutive_errors = 0
            driver.on_success(returncode)
            continue

        # Non-zero exit — the cause is ambiguous (a network blip / one-off CLI
        # hiccup, or the session's token limit). Rather than guessing from a
        # second consecutive error, ask the CLI directly: query /usage right away
        # and let the real Current-session percentage decide.
        consecutive_errors += 1
        elapsed = time.time() - session_start
        print_error(f"claude exited with code {returncode} "
                    f"(error #{consecutive_errors} in a row).")

        if not ignore_usage_limits:
            paused, session_start = limit_policy.check_and_wait(
                usage_source, session_start, note=" (checked after error)")
            if paused:
                consecutive_errors = 0  # fresh window — start counting errors anew
                continue

        # Session is under the limit — this was a transient failure, not token
        # exhaustion. Retry, but don't spin forever if something is truly broken.
        if consecutive_errors < 5:
            print(f"  ↻ Session under the allowed limit — likely transient. "
                  f"Retrying immediately.")
            continue
        else:
            print(f"  ⚠ {consecutive_errors} errors in a row with the session under "
                  f"the allowed limit after {int(elapsed // 60)} min — not a "
                  f"limit. Stopping.")
            sys.exit(returncode)

    # Final push: regardless of the EACH_HOUR cadence, push any pending commits
    # on the way out so work isn't left only on the local branch — unless the
    # policy is NONE (never auto-push).
    if not dry_run and git_push_policy != GitPushPolicy.NONE:
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
    if not dry_run:
        limit_policy.log_snapshot(usage_source, "at end (after last cycle)",
                                  cache_value=False)

    # Closing line, if the driver has one (e.g. "Final state: …").
    summary = driver.final_summary()
    if summary:
        print(f"\n{summary}")
