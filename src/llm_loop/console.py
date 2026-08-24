"""console.py - how a run talks to the person watching it, and to the record.

Everything the loop prints goes through here, and everything printed is written
TWICE: styled to the terminal, and plain to a rotating mirror log. That is one
job, not two, which is why the log lives in this module rather than beside the
runner that produces the lines:

  * both writers of the log are printing paths — `_TeeToLog`, which wraps
    `sys.stdout` for the whole run, and `_log_plain`, the second sink of
    `print_markup` for the Rich path whose live frames must never reach a file
    (see `_real_stream`). Nothing else writes to it;
  * the split between them is a TERMINAL detail: escape codes and cursor
    repaints belong on a screen and nowhere else, and deciding that for each
    line is what `print_markup` is;
  * so a runner that owned the log would own a file it never writes, while the
    module that writes it would have to ask permission to.

What deliberately did NOT come along, and the test for it: `report_costs` READS
this log back and parses the runner's own vocabulary out of it ("=== Iteration 1
===", "· done (… c, $…)") — those lines are emitted by `run_loop` and the event
renderers, so its patterns belong next to THEM, in `cyclecore`. `exitlog` is the
same shape from the other side: it is handed `LOG_DIR` and writes its own file
beside the mirror, so it stays a module of its own and imports the constant.

The rule for anything added here: this module must not import a runner. The one
thing it needs that a runner used to own — the project root, whose folder name
this log is filed under — comes from `projectroot`, a leaf module below both, so
it is READ rather than handed over. It used to be handed over (`set_log_project`
pushed a copy in here), and the copy is what made the handover a thing that
could silently stop happening.
"""

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
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


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
# same "3h24m" has to appear on a pinned status row, in a countdown line, and in
# a limit rule's own sentence, and three modules formatting it themselves is how
# the three drift. Everything that speaks to the person watching the run lives
# in this module, and that includes how long it says something will take.

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


