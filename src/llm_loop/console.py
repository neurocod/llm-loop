"""console.py - how a run talks to the person watching it, and to the record.

Everything the loop prints goes through here, and everything printed is written
TWICE: styled to the terminal, and plain to a rotating mirror log. That is one
job, not two, which is why the log lives in this module rather than beside the
runner that produces the lines:

  * both writers of the log are printing paths — `TeeToLog`, which wraps
    `sys.stdout` for the whole run, and `_log_plain`, the second sink of
    `print_markup` for the Rich path whose live frames must never reach a file
    (see `real_stream`). Nothing else writes to it;
  * the split between them is a TERMINAL detail: escape codes and cursor
    repaints belong on a screen and nowhere else, and deciding that for each
    line is what `print_markup` is;
  * so a runner that owned the log would own a file it never writes, while the
    module that writes it would have to ask permission to.

What deliberately did NOT come along, and the test for it: reading this log
back for `--cost` means knowing what two of the runners' lines MEAN ("=== Iteration
1 ===", "· done (… c, $…)"), which is not a printing question — so the lines and
their patterns are `costlog`'s, and it imports `log_file_path`. `exitlog` is the
same shape from the other side: it is handed `LOG_DIR` and writes its own file
beside the mirror, so it stays a module of its own and imports the constant.

Names without a leading underscore here are the PACKAGE's, not the front door's:
`fmt_clock`, `TeeToLog`, `real_stream` and the others published in `c087700` are
read by the runners and by the status line, and `_` was dropped so it can go on
meaning "this file's business" — see tests/test_package_privacy.py, which is what
keeps that true. What an ADOPTER may name is `__init__.__all__`, pinned
separately.

The rule for anything added here: this module must not import a runner. The one
thing it needs that a runner used to own — the project root, whose folder name
this log is filed under — comes from `projectroot`, a leaf module below both, so
it is READ rather than handed over. It used to be handed over (`set_log_project`
pushed a copy in here), and the copy is what made the handover a thing that
could silently stop happening.
"""

import contextlib
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from . import compactline
from . import projectroot


# --- mirror-log path -------------------------------------------------------

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

    Derived on every call, never cached in a module global here: --project-dir
    moves the root after import, and a copy taken at import time would file
    every project's log under whatever directory the process started in. That
    copy existed (`_LOG_PROJECT`, pushed in by `set_log_project`) until the root
    became a leaf both modules can read — see `projectroot`.
    """
    return LOG_DIR / f"{app_name}-{os.path.basename(projectroot.project_dir())}.log"


# --- mirror-log writer -----------------------------------------------------

# The app-specific logger that owns the mirror-log file handler, set by
# setup_file_logging. `_log_plain` (the Rich path) must target *this* logger:
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
    `sys.stderr` — which is the `TeeToLog` mirror, which logs the line, which
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


class _MirrorLogFormatter(logging.Formatter):
    def format(self, record):
        # Timing records are written directly, never inferred from agent output.
        # The pid separates concurrent runs that share the app/project log.
        record.loop_record = getattr(record, "loop_record", "output")
        return super().format(record)


def record_timing(message: str) -> None:
    """Write a machine timing record without passing through the output tee."""
    if _FILE_LOGGER is not None:
        _FILE_LOGGER.info(message, extra={"loop_record": "timing"})


def setup_file_logging(app_name: str = "runCycle") -> logging.Logger:
    """Configure the rotating file logger at log_file_path(app_name).

    Idempotent per destination: a second call that finds this module's own
    mirror already writing `log_file_path(app_name)` adds none. Two tests, both
    needed:

      * OUR handler, not any handler: a host may hang its own on the logger
        (pytest 9.1 attaches capture handlers to every non-propagating logger
        at the start of each phase), and "has handlers" then silently left the
        run without a log file. Foreign handlers are left alone;
      * the file, not the class: the path is derived per call (the project
        root and `LOG_DIR` can move between two runs of one process), and a
        mirror kept for its class alone went on writing the FIRST project's
        log while the second run printed "logging to" its own. A mirror of
        ours aimed anywhere else is closed and replaced.

    Pinned by `tests/test_mirror_log.py`:
    `test_a_foreign_handler_does_not_switch_the_mirror_off` and
    `test_a_second_project_in_one_process_gets_its_own_log`.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"runCycle.{app_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    path = log_file_path(app_name)
    # `FileHandler` stores its path absolute; normcase for Windows' spelling.
    target = os.path.normcase(os.path.abspath(str(path)))
    kept = False
    for handler in list(logger.handlers):
        if not isinstance(handler, _MirrorLogHandler):
            continue
        if not kept and os.path.normcase(handler.baseFilename) == target:
            kept = True
            continue
        handler.close()
        logger.removeHandler(handler)
    if not kept:
        handler = _MirrorLogHandler(
            path, maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
        )
        handler.setFormatter(
            _MirrorLogFormatter(
                "%(asctime)s pid=%(process)d kind=%(loop_record)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    global _FILE_LOGGER
    _FILE_LOGGER = logger
    return logger


class TeeToLog:
    """Wrap a console stream so everything printed is also captured into the file
    logger, one record per line.

    Partial writes (streaming tokens emitted with ``end=""``) are buffered until a
    newline, so the file holds clean, complete lines while the screen keeps showing
    live token-by-token output.

    One write at a time per tee, screen and log together (`_lock`): more than
    one thread writes here during a parallel run (see `route_through`), and
    unguarded two of them lose each other's text in `_buf` or log their lines
    in the other order from the screen. A stream that is stuck holds the lock
    for as long as the write is — where the next writer's own write to that
    stream would have blocked anyway. Each tee has its own lock, so a stuck
    stdout never holds up the stderr tee that reports it. The logger is called
    inside the lock, which is safe because the mirror handler never writes a
    stream (`_MirrorLogHandler.handleError`); reentrant, because a handler
    that does, writing to this same tee, comes back here on the same thread.
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
        self._lock = threading.RLock()

    def write(self, text: str) -> int:
        with self._lock:
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


# --- printing --------------------------------------------------------------

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
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


_DEPENDENCY_WARNING_SHOWN = False


def warn_missing_dependencies() -> None:
    """Warn after the log tee is ready, once even across a wrapper's batches."""
    global _DEPENDENCY_WARNING_SHOWN
    if RICH_AVAILABLE or _DEPENDENCY_WARNING_SHOWN:
        return
    # cmdline imports clispec -> termio -> console; defer this edge until the
    # modules are initialized rather than closing an import-time cycle.
    from . import cmdline

    _DEPENDENCY_WARNING_SHOWN = True
    print("  WARNING: Not all dependencies are installed: 'rich' is missing. "
          "Some functionality is unavailable: Markdown formatting, colored "
          "output, and accurate Unicode display widths. Continuing in plain text.")
    # Written for the shell this console most likely is (`& ` included for
    # PowerShell). An embedded or frozen interpreter has sys.executable == "",
    # which no line can name; this runs at startup, so that must not raise.
    shell = cmdline.paste_shell()
    install = None
    if sys.executable:
        try:
            install = cmdline.quote(
                [sys.executable, "-m", "pip", "install", "rich"], shell)
        except cmdline.NotPasteable:
            pass
    if install is None:
        print("  Install with: pip install rich "
              "(into the Python that runs this program)")
        return
    shell_hint = " (PowerShell)" if shell == cmdline.POWERSHELL else ""
    print(f"  Install with{shell_hint}: {install}")


def real_stream():
    """The underlying console stream, unwrapping the line-logging tee.

    Rich's Live repaints the frame many times a second using cursor-movement
    escape codes that must not end up in the file log, so its output goes
    straight to the real terminal rather than through `TeeToLog`.
    """
    out = sys.stdout
    return getattr(out, "_stream", out)


def _log_plain(text: str) -> None:
    """Mirror a finished Markdown block to the file log as clean plain text.

    Used on the Rich path, where the live frames bypass the tee — we still want
    the assistant's words in the log, just without the ANSI/redraw noise.

    Targets the app-specific logger configured by setup_file_logging (which owns
    the file handler and does not propagate); falling back to a bare "runCycle"
    logger only if logging was never set up (e.g. in tests). Using the wrong
    logger name here silently drops every Rich-path line — including the
    "=== Iteration N ===" headers and "· done (… c, $…)" cost lines that
    report_costs parses — leaving --cost to report 0 sessions.
    """
    logger = _FILE_LOGGER or logging.getLogger("runCycle")
    for line in text.splitlines():
        logger.info(line)


# Repaints per second of a streaming Markdown block, and so also the ceiling on
# how often its text is PARSED: Rich's Markdown parses in its constructor, and
# `MarkdownStream` builds one only when `Live` asks for a frame. Building one per
# delta instead made the work quadratic in the block (synthetic 600 deltas,
# 9.2 KiB: 5.9 s of parsing in the reader thread vs 0.016 s for one final parse,
# measured 2026-10-01). With parsing tied to this rate the block parses once
# per frame that saw new text — see the measurement beside
# `MarkdownStream._frame`. Raising it buys smoother
# typing for proportionally more parses of a long block.
LIVE_REFRESH_PER_SECOND = 12

# Plain operator output must use the active Live's Console: writing beneath its
# frame through stdout makes the next refresh erase that output. The lock orders
# printing against start/stop; Live's refresh thread never takes it.
_LIVE_OUTPUT_LOCK = threading.Lock()
_live_console = None


class MarkdownStream:
    """Render one assistant text block as live-updating Markdown.

    The model streams Markdown token by token; `feed` only accumulates it, and
    Rich's `Live` asks `_frame` for the block on each repaint, so formatting
    appears in realtime at `LIVE_REFRESH_PER_SECOND` without a parse per delta.
    When Rich is unavailable we degrade to the original behaviour: print a `💬`
    header and stream the raw tokens inline.
    """

    def __init__(self):
        self._buf = ""
        self._live = None
        self._console = None
        # The buffer object the cached Markdown was parsed from, and that
        # Markdown. Compared by identity: every feed of new text makes a new
        # string, and an unchanged one must not be parsed again on the next
        # repaint.
        self._parsed_from = None
        self._parsed = None

    def _frame(self):
        """The block as Markdown, parsed only if text arrived since the last frame.

        Called by `Live` — from its refresh thread under the Live lock, and from
        `start`/`stop` — never by `feed`. Measured 2026-10-01 (two runs) with
        the same synthetic ~600 deltas / 9.2 KiB: back to back, 2 parses and
        <0.001 s across all feeds; paced over 3 s, 37 parses taking 0.31-0.32 s,
        all off the reader thread, and 0.005 s across all feeds.
        """
        buf = self._buf
        if buf is not self._parsed_from:
            self._parsed = _RichMarkdown(buf)
            self._parsed_from = buf
        return self._parsed

    def start(self) -> None:
        global _live_console
        self._buf = ""
        self._parsed_from = None
        self._parsed = None
        if RICH_AVAILABLE:
            self._console = _RichConsole(file=real_stream())
            self._console.print("\n[dim]💬[/dim]")
            self._live = _RichLive(
                get_renderable=self._frame,
                console=self._console,
                refresh_per_second=LIVE_REFRESH_PER_SECOND,
                vertical_overflow="visible",
                # Operator lines use this Console; redirecting stdout/stderr
                # would fight with TeeToLog.
                redirect_stdout=False,
                redirect_stderr=False,
            )
            with _LIVE_OUTPUT_LOCK:
                self._live.start()
                _live_console = self._console
        else:
            print("\n💬 ", end="", flush=True)

    def feed(self, text: str) -> None:
        self._buf += text
        if self._live is None:
            print(text, end="", flush=True)

    def stop(self) -> None:
        global _live_console
        if self._live is not None:
            # `Live.stop` repaints once more through `_frame`, so the final
            # frame is the whole block even if no refresh saw the last delta.
            with _LIVE_OUTPUT_LOCK:
                self._live.stop()
                _live_console = None
            self._live = None
            self._console = None
            # Guarantee the next output (tool calls, etc.) starts on a fresh line,
            # regardless of how Live left the cursor on this terminal.
            print(file=real_stream())
            if self._buf.strip():
                _log_plain(self._buf)
        else:
            print(flush=True)  # finish the inline line in fallback mode
        self._buf = ""
        self._parsed_from = None
        self._parsed = None


def render_markdown_block(text: str) -> None:
    """Print a complete Markdown string formatted (Rich) or plain (fallback).

    Used for non-streaming assistant text (when --include-partial-messages is off
    we never see deltas, only the final block).
    """
    text = text.strip()
    if not text:
        return
    if RICH_AVAILABLE:
        console = _RichConsole(file=real_stream())
        console.print("[dim]💬[/dim]")
        console.print(_RichMarkdown(text))
        _log_plain(text)
    else:
        print(f"\n💬 {text}")


# --- routing: who writes the console while an owner holds it ---------------


class _Route:
    """One `route_through` window: its owner, and the lines it could not take.

    `stalled` is set by a post that found no room and cleared by one that did;
    while set, a post takes only room that is free at once. `written_directly`
    counts the lines handed back to their caller, and `closed` is set when the
    window closes and reports that count; both are under `_ROUTE_LOCK`, which
    orders a count against the close (see `post`).
    """

    def __init__(self, owner, post_timeout: float):
        self.owner = owner
        self.post_timeout = post_timeout
        self.stalled = False
        self.written_directly = 0
        self.closed = False

    def post(self, call, *args) -> bool:
        """Hand one write to the owner; False when the caller must make it.

        A line handed back is counted: before the close, into the close's
        report; after it — a poster that took the route, waited, and timed
        out once the window had closed and reported — it says so itself,
        since nothing else would.
        """
        timeout = 0 if self.stalled else self.post_timeout
        if self.owner.post(call, *args, timeout=timeout):
            self.stalled = False
            return True
        with _ROUTE_LOCK:
            self.stalled = True
            self.written_directly += 1
            late = self.closed
        if late:
            print(f"  ⚠ {self.owner.name}: 1 line written directly, out of "
                  f"order, after the window closed — its queue stayed full.",
                  file=sys.stderr)
        return False


# Guards installing and removing the route, and its counters. Never held across
# a write or a post: a reader takes `_route` with one plain read (atomic), so a
# console that is stuck cannot hold up the install, the removal or a poster.
_ROUTE_LOCK = threading.Lock()
_route: Optional[_Route] = None

# Threads whose stdout writes are being diverted: ident -> list of chunks.
# Read without the lock by `_ThreadScopedCapture` and `_on_console` (one dict
# lookup, atomic); written only under it.
_capture_lock = threading.Lock()
_captured: dict = {}


class _ThreadScopedCapture:
    """A `sys.stdout` stand-in that diverts ONE thread's writes into a buffer.

    Needed because the status line's background quota poll reaches
    `usage.query_usage_json`, which prints its diagnostics ("no usage
    figures: … 401 …"). Printed from a daemon thread they land at an arbitrary
    point of the scrolling output — possibly mid-token inside a rich `Live`
    block — and in the mirror log that `--cost` parses. Replacing `sys.stdout`
    outright for the duration would steal the LOOP's own output too, so the
    diversion is keyed on the thread that asked for it; every other thread
    passes straight through.
    """

    def __init__(self, target):
        self._target = target

    def write(self, text):
        buffer = _captured.get(threading.get_ident())
        if buffer is None:
            return self._target.write(text)
        buffer.append(text)
        return len(text)

    def flush(self):
        if _captured.get(threading.get_ident()) is None:
            self._target.flush()

    def __getattr__(self, name):
        return getattr(self._target, name)


@contextlib.contextmanager
def capture_stdout_here():
    """Collect THIS thread's stdout writes; yields the list of chunks.

    A captured thread is never routed (`_on_console`): posted to a route's
    owner, its line would be written on the owner's thread — past the capture,
    onto the screen and into the mirror log — and the status line's refresher,
    having captured nothing, would read the source as recovered. That check
    lives beside the route rather than as a flag the capturer must remember to
    set, which is how backlog 0616's leak happened. Does not nest: an inner
    capture on the same thread ends the outer one.
    """
    ident = threading.get_ident()
    buffer: list = []
    with _capture_lock:
        if not isinstance(sys.stdout, _ThreadScopedCapture):
            sys.stdout = _ThreadScopedCapture(sys.stdout)
        _captured[ident] = buffer
    try:
        yield buffer
    finally:
        with _capture_lock:
            _captured.pop(ident, None)
            proxy = sys.stdout
            # Uninstall only once nobody is capturing any more, and only if the
            # proxy is still ours — the loop installs its own tee over stdout.
            if not _captured and isinstance(proxy, _ThreadScopedCapture):
                sys.stdout = proxy._target


@contextlib.contextmanager
def route_through(owner, *, post_timeout: float):
    """While inside, every console write of this module is run by `owner`.

    `owner` is an `ownership.OwnerThread`. `print_markup` and `print_line` —
    and so the whole print_* family and `LINES` — post their write to it
    instead of making it, so a line printed by a thread the runner does not own
    (the usage gate on a worker, the background git pusher) lands in the one
    stream the owner writes, after every line already queued there. A thread
    inside `capture_stdout_here` is not routed (the status line's refresher,
    whose usage diagnostics are captured, not printed). Why this lives here and not
    with the runner: those threads reach the console through this module's
    names, imported by value, which the runner cannot intercept without
    patching another module's globals.

    The window may be opened before `owner.start()` and closed after
    `owner.close()`: with no owner thread a post runs on its caller (see
    `OwnerThread.post`), which is exactly what a write did before the route.
    `parallel.run_parallel` opens it that way round, so while the owner is
    healthy no line is routed to an owner that has not yet got the lines
    before it, and none written directly ahead of what it still holds. On the
    owner's own thread a write is made at once — the owner cannot wait for a
    queue only it empties.

    What the route does NOT promise is order around a console that is stuck.
    A close that times out (`parallel._close_console`) leaves the owner
    writing its backlog as a daemon: a line posted just before the window
    closed is queued behind that backlog and lost with it if the process exits
    first, and a line written after the window closed is written at once,
    ahead of it. The lines written directly inside the window (below) are out
    of order too, by design.

    THE BLOCKING POLICY — a poster waits at most `post_timeout` for room in the
    owner's queue, and past it writes the line ITSELF, out of order, counting
    it. Chosen over the two alternatives:

      * over an unbounded `post`: a poster must never wait for the owner
        indefinitely. The gate holds `usage_lock` while it prints, and the
        exit push waits for the pusher to finish the push it is printing about;
      * over `try_post` with a counted drop: these lines are the run's record —
        the mirror log is written by the same call — and a quota pause or a
        failed push missing from it is worse than one that is out of order. A
        drop is the right answer for a diagnostic repeated per event (backlog
        0763), not for a line said once.

    A parallel run's WORKERS post to the owner directly, not through this
    route, and wait without a bound: why they differ is
    `parallel._emit_markup`'s to say.

    A queue full for the whole of `post_timeout` means the console itself is
    stuck, so the direct write may block there as well — which is where the
    write blocked before this route existed, so the worst case is the old
    behaviour after a bounded wait, not a new hang. And it is paid once, not
    per line: past one such timeout the route is STALLED, and a poster only
    takes room that is free at once (`try_post`), writing the line itself
    otherwise, until a post finds room again. Without that, every line of the
    gate — which holds `usage_lock` meanwhile — would wait its own
    `post_timeout` before reaching the point where the old write blocked.

    A line written directly is written whole (`_print_flushed`, and the tee's
    lock in `TeeToLog`), so it can land between two of the owner's lines but
    not inside one. The count is reported to stderr once the window closes
    (a later one reports itself: `_Route.post`).

    A write that raises on the owner costs that line only and is reported by
    the owner (`OwnerThread._report`); it does not unwind the thread that
    printed. Refused if a route is already open: two owners would each believe
    they order the console.
    """
    route = _Route(owner, post_timeout)
    global _route
    with _ROUTE_LOCK:
        if _route is not None:
            raise RuntimeError(f"the console is already routed through "
                               f"{_route.owner.name!r}")
        _route = route
    try:
        yield
    finally:
        with _ROUTE_LOCK:
            _route = None
            route.closed = True
            direct = route.written_directly
        if direct:
            print(f"  ⚠ {owner.name}: {direct} line(s) written directly, out of "
                  f"order — its queue had no room (the first after waiting "
                  f"{post_timeout:g} s).", file=sys.stderr)


def _on_console(call, *args) -> None:
    """Make one console write — on the route's owner while one is open.

    `call` is the write itself (never `print_markup`, which would route again
    on the owner's side and recurse there).
    """
    route = _route
    if (route is not None and threading.get_ident() not in _captured
            and not route.owner.owns_current_thread
            and route.post(call, *args)):
        return
    call(*args)


def _render_markup(plain: str, markup: str) -> None:
    """The write behind `print_markup`, on whichever thread makes it.

    Flushed on both paths, because a line may be the once-a-minute countdown of
    a run paused on a limit (`print_percents`) — the one place output has to
    appear as it is written rather than when a buffer happens to fill — and the
    flush has to be made by the thread that wrote the line, after it: a flush on
    the poster's side would run before a routed line was written at all. Rich
    flushes its own writes; the plain path is `_print_flushed`.
    """
    if RICH_AVAILABLE:
        _RichConsole(file=real_stream()).print(markup)
        _log_plain(plain)
    else:
        _print_flushed(plain)


def _print_flushed(text: str) -> None:
    """One whole line, in ONE write, then flushed (see `_render_markup`).

    Not `print`: it writes the text and the newline in two calls, and a line
    written directly past a stuck route (`route_through`) can land between
    them — "queued" + "direct\\n" + "\\n".
    """
    stream = sys.stdout
    stream.write(text + "\n")
    stream.flush()


def _print_above_live(text: str) -> None:
    with _LIVE_OUTPUT_LOCK:
        if _live_console is not None:
            _live_console.print(text, markup=False, highlight=False, emoji=False,
                                soft_wrap=True)
            _log_plain(text)
            return
        _print_flushed(text)


def print_markup(plain: str, markup: str) -> None:
    """Print a status line from hand-written Rich markup: styled on screen, plain
    in the log. The low-level core of the print_* family — use `print_styled`
    (text + a style name) for uniform lines and call this directly only when a
    line needs different styles per segment (e.g. a coloured glyph + plain text).

    With Rich available the `markup` string (Rich console markup: colours, bold,
    italic, underline) is rendered straight to the real terminal, while a clean
    `plain` copy is mirrored to the file log — so colour/redraw escapes never end
    up in the log. Without Rich it degrades to the plain line (screen + log via
    the tee). Note: terminals can't switch *font family*; only colour and the
    bold/italic/underline attributes are available.

    Inside `route_through` the write is the owner's, not the caller's.
    """
    _on_console(_render_markup, plain, markup)


def print_line(text: str) -> None:
    """`print(text, flush=True)`, but inside `route_through` written by the owner.

    For the plain lines of code that can run beside a parallel run's workers —
    the usage gate, the usage sources. A bare `print` there would be written
    past the owner, and ahead of the gate's own `print_percents` lines queued a
    moment earlier. Always flushed, like `print_markup` (`_render_markup`).
    """
    _on_console(_print_flushed, text)


def print_operator_line(text: str) -> None:
    """Print literal operator output above any live Markdown frame and log it.

    Rich must not interpret markup or insert line breaks into a pasteable command.
    """
    _on_console(_print_above_live, text)


# This runner's compact lines: no job tag, straight to the console. The sink is
# a lambda rather than `print_markup` itself so that the name is resolved per
# line — the width pins replace it to read the plain copy of what was printed,
# and a captured function would sail past them (see `compactline.LineWriter`).
LINES = compactline.LineWriter(lambda plain, markup: print_markup(plain, markup))


def print_styled(text: str, style: str) -> None:
    """Print a whole line in one Rich style, routed through `print_markup`.

    The single-style sibling of `print_markup`: callers pass plain `text` plus a
    Rich style (`"green"`, `"bold red"`, …); the plain copy goes to the log and
    the styled copy to the screen. Markup metacharacters in `text` are escaped,
    so a stray '[' is shown literally instead of being read as a tag. For lines
    that need *different* styles per segment (a coloured glyph next to plain
    text), call `print_markup` directly with hand-written markup.
    """
    print_markup(text, f"[{style}]{compactline.esc(text)}[/]")


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
        out.append(compactline.esc(text[last:m.start()]))
        value = float(m.group(0).rstrip("% \t"))
        out.append(f"[{percent_style(value)}]{m.group(0)}[/]")
        last = m.end()
    out.append(compactline.esc(text[last:]))
    return "".join(out)


def print_percents(text: str) -> None:
    """Print a line whose percentages are colour-coded on screen (plain in the
    log). A no-op difference from `print` when Rich is unavailable.
    """
    print_markup(text, markup_percents(text))


# Named single-style specialisations, each delegating to print_styled. Centralise
# the loop's palette here so a colour is changed in one place, not at every call.
def print_done(text: str) -> None:
    print_styled(text, "green")


def print_error(text: str) -> None:
    print_styled(text, "bold red")


def print_note(text: str) -> None:
    """An operator note, at the point in the stream where the agent received it.

    Printed rather than merely shown on the status row because the status row is
    transient and the mirror log is the run's record: an agent that changes
    course mid-iteration is unexplainable later unless the sentence that made it
    do so sits in the log next to the turn it landed in.
    """
    print_markup(f"  ✉ operator note: {text}",
                 f"  [magenta]✉[/] [bold magenta]operator note:[/] "
                 f"{compactline.esc(text)}")


# --- time formatting -------------------------------------------------------
#
# Here rather than in a runner because a duration is READ, not computed: the
# same "3h24m" appears in countdown lines and limit rules' own sentences.
# The pinned status area adapts compound hour/minute durations to clock notation.

def fmt_clock(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def fmt_left(seconds: float) -> str:
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


def fmt_moment(ts: float) -> str:
    """Like fmt_clock, but names the day too once the moment is far enough away
    that a bare clock reading would be ambiguous — a weekly quota resets days out,
    and "12:59:59" alone reads as "in a few hours"."""
    if ts - time.time() < 18 * 3600:
        return fmt_clock(ts)
    return datetime.fromtimestamp(ts).strftime("%b %d, %H:%M")
