"""statusline.py - the interactive status area pinned under a running loop.

While the loop streams the agent's output, a few rows stay pinned at the bottom
of the terminal: which iteration is running, on what item and model, how long it
has been going, the provider's live quota figures and the script's own limits,
plus a legend of the keys the run answers to.

Shape of the solution (and why it is hand-rolled): the output above the rows is
an ordinary scrolling subprocess stream, so the pinned area is a DECSTBM scroll
region and nothing more. A full-screen framework (textual/prompt_toolkit) would
have to own that stream, and `rich.live.Live` is already taken by the streaming
Markdown renderer - only one may be active at a time, and `redirect_stdout`
would bypass the mirror-log tee and silently empty `--cost`. Rich is still used
where it is good: styling and cell-accurate truncation (`rich.cells.cell_len`
knows about double-width glyphs and emoji, `len()` does not).

Two things are load-bearing and easy to get wrong:

  * every escape goes to ``cyclecore._real_stream()``. ``sys.stdout`` is the
    ``_TeeToLog`` mirror, and cursor-movement bytes in the log corrupt the run
    record that ``--cost`` parses.
  * anything that fails here disables the status line (a `NullTerminal` takes
    over) and the run continues. A cosmetic feature must never be able to stop
    an eight-hour loop.

The object model is deliberately wider than wave 1 needs: `Segment`, `Row`,
`Mode`, `Action`, `Setting` and `InputEvent` all exist so later features land by
ADDING a subclass and registering it - never by editing a dispatch chain, adding
a boolean to `LoopStatus`, or hard-coding a key into the legend.
"""

import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from . import cyclecore

__all__ = [
    "Action",
    "ElapsedSegment",
    "HelpAction",
    "InputEvent",
    "InputSource",
    "IterationSegment",
    "Job",
    "JobCountSegment",
    "JobRow",
    "Key",
    "KeyLegendRow",
    "LabelSegment",
    "Layout",
    "LoopStatus",
    "Mode",
    "Mouse",
    "NoteRow",
    "NormalMode",
    "NullInputSource",
    "NullTerminal",
    "NumberSetting",
    "PercentSetting",
    "ProviderSegment",
    "QuotaRefresher",
    "QuotaSegment",
    "Resize",
    "Row",
    "ScriptLimitSegment",
    "Segment",
    "SegmentRow",
    "Setting",
    "SettingsRegistry",
    "StatusApp",
    "StopAction",
    "StopSegment",
    "Terminal",
    "TerminalInput",
    "colorize",
    "format_elapsed",
    "format_prompt_block",
    "push_quotas",
    "quota_rows",
    "render_rows",
    "screen_width",
    "terminal_for",
]

# Environment kill switch, honoured next to --no-statusline so a hostile terminal
# can be worked around without editing any command line.
ENV_FLAG = "CLAUDE_LOOP_STATUSLINE"

# Below these the reserved region would eat the visible scrollback, so we simply
# do not pin anything.
MIN_COLUMNS = 40
MIN_LINES = 8

# Repaint cadence: fast enough that the elapsed clock ticks like a clock, slow
# enough to be invisible next to the agent's own output.
REFRESH_SECONDS = 0.5

# How long a key-feedback note stays on screen before the row goes quiet again.
NOTE_TTL = 8.0

# Quota refresh cadence. Claude's UsageSource is one cached HTTP GET (~0.3 s);
# the codex source shells out to its app-server, so it gets a longer interval.
CLAUDE_QUOTA_REFRESH = 120.0
CODEX_QUOTA_REFRESH = 300.0

PHASE_GLYPHS = {
    "idle": "·",
    "running": "⟳",
    "paused": "⏸",
    "waiting": "⏳",
    "stopping": "⏹",
}

# Shown when a command carries no --model: the provider CLI picks its own.
CLI_DEFAULT_MODEL = "cli default"


# --- text helpers --------------------------------------------------------------

try:  # optional, and the only reason rich is touched here
    from rich.cells import cell_len as _cell_len
except ImportError:  # pragma: no cover - exercised only without rich installed
    def _cell_len(text: str) -> int:
        return len(text)


def cell_width(text: str) -> int:
    """Terminal columns `text` occupies (double-width aware when rich is here)."""
    return _cell_len(text)


def fit(text: str, width: int) -> str:
    """`text` cut to at most `width` columns, ending in '…' when something was cut.

    A row that wraps would push the reserved region up by a line and desynchronise
    every subsequent repaint, so truncation is not cosmetic here.
    """
    if width <= 0:
        return ""
    if cell_width(text) <= width:
        return text
    out = []
    used = 0
    for ch in text:
        w = cell_width(ch)
        if used + w > width - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


def pad(text: str, width: int) -> str:
    """`text` fitted to exactly `width` columns (truncated, then space-padded)."""
    text = fit(text, width)
    return text + " " * max(0, width - cell_width(text))


def format_elapsed(seconds: Optional[float]) -> str:
    """"4m12s" / "1h02m" — two units, the larger one first; "" for None."""
    if seconds is None:
        return ""
    total = max(0, int(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def screen_width(default: int = 100, maximum: int = 120) -> int:
    """A sane width for full-width blocks printed into the scroll area."""
    try:
        columns = shutil.get_terminal_size(fallback=(default, 24)).columns
    except Exception:
        columns = default
    return max(40, min(columns, maximum))


def format_prompt_block(*, job_id: int, label: str, prompt: str,
                        width: int = 100) -> str:
    """A prompt, headed and delimited, ready to be printed anywhere.

    Pure on purpose: it is the ONE renderer for every place a prompt is shown
    (the dry run, wave 4's prompt viewer, any later "why did this iteration do
    that" diagnostic), so those cannot drift apart. The prompt body is kept
    verbatim between the rules — no indent, no prefix — so it stays selectable
    and pasteable; the header carries the size, which is the figure that
    explains a slow or truncated turn.
    """
    prompt = prompt or ""
    width = max(24, width)
    head = (f"─ prompt · job {job_id} · {label or '(no label)'} · "
            f"{len(prompt)} chars ")
    head = fit(head, width)
    head += "─" * max(0, width - cell_width(head))
    return "\n".join([head, prompt.rstrip("\n"), "─" * width])


def short_quota_label(label: str) -> str:
    """"Current session" -> "session": the rules' labels are prose for the log,
    and a pinned row has no room for prose."""
    text = label.strip().lower()
    if text.startswith("current "):
        text = text[len("current "):]
    if text.startswith("week"):
        return "week/sonnet" if "sonnet" in text else "week"
    return text or label


# ANSI styles used by the painted rows. The pure renderer produces plain text and
# `colorize` adds the codes afterwards — so truncation arithmetic never has to
# count escape bytes, and tests read plain strings.
_SGR = {
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "bold red": "\x1b[1;31m",
    "dim": "\x1b[2m",
}
_SGR_RESET = "\x1b[0m"
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")


def colorize(line: str) -> str:
    """Colour every percentage in a rendered row on the shared usage scale.

    Reuses `cyclecore.percent_style` so the pinned rows and the scrolling log
    lines agree about what "alarming" looks like.
    """
    out = []
    last = 0
    for match in _PERCENT_RE.finditer(line):
        style = _SGR.get(cyclecore.percent_style(
            float(match.group(0).rstrip("% \t"))), "")
        out.append(line[last:match.start()])
        out.append(f"{style}{match.group(0)}{_SGR_RESET}" if style else match.group(0))
        last = match.end()
    out.append(line[last:])
    return "".join(out)


# --- data model ----------------------------------------------------------------


@dataclass
class Job:
    """One unit of display: a worker (or the sequential loop) and what it is on.

    A sequential run is a run with exactly one Job, so no renderer anywhere has a
    "sequential vs parallel" branch — and the parallel runner only has to feed
    data. Every mutator takes the lock because wave-3 workers update their own
    Job from N threads while the repaint thread reads them all.

    Mutating a Job notifies nobody on purpose: the repaint thread picks the new
    values up within a tick, so a worker never blocks on the terminal.
    """

    job_id: int = 1
    running: bool = False
    iteration: int = 0          # iterations this Job has started
    item: str = ""              # label of the unit of work in flight
    model: str = ""             # exactly what driver.model() returned
    prompt: str = ""            # full prompt in flight (wave 4 shows it)
    started_at: float = 0.0     # start of the CURRENT iteration, 0.0 when idle

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def elapsed(self, now: Optional[float] = None) -> Optional[float]:
        """Duration of the iteration running RIGHT NOW (None when idle).

        Deliberately not the run's total: this is the number that says whether a
        job is wedged. The run total lives on the summary row.
        """
        started = self.started_at
        if not started:
            return None
        return max(0.0, (time.time() if now is None else now) - started)

    def start(self, *, item: str = "", model: str = "", prompt: str = "",
              iteration: Optional[int] = None, now: Optional[float] = None) -> None:
        """Claim an iteration: bump the counter and record what is in flight."""
        with self._lock:
            self.iteration = (self.iteration + 1) if iteration is None else iteration
            self.item = item
            self.model = model or ""
            self.prompt = prompt
            self.started_at = time.time() if now is None else now
            self.running = True

    def finish(self) -> None:
        """Release the iteration; the row goes idle and its clock stops."""
        with self._lock:
            self.running = False
            self.started_at = 0.0

    def update(self, **fields) -> None:
        with self._lock:
            for name, value in fields.items():
                setattr(self, name, value)

    def snapshot(self) -> "Job":
        """A detached copy, so one painted row cannot mix two iterations."""
        with self._lock:
            return Job(self.job_id, self.running, self.iteration, self.item,
                       self.model, self.prompt, self.started_at)


@dataclass
class LoopStatus:
    """Everything the rows can show. Later waves add Rows and Segments, not flags.

    `quotas` is [(label, percent|None, ceiling|None, reset_ts|None)] and
    `script_limits` is [(label, text)] — both are lists rather than named fields
    exactly so a new figure is a new tuple, not a new attribute plus a renderer
    edit.
    """

    jobs: List[Job] = field(default_factory=lambda: [Job(1)])
    iteration: int = 0                       # iterations started across all Jobs
    max_iterations: Optional[int] = None
    random_order: bool = False               # renders the `rand` marker
    provider: str = "claude"
    run_started_at: float = 0.0              # first iteration of the whole run
    phase: str = "idle"                      # idle|running|paused|waiting|stopping
    quotas: List[Tuple] = field(default_factory=list)
    script_limits: List[Tuple[str, str]] = field(default_factory=list)
    stop_pending: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if not self.jobs:      # a run ALWAYS has at least one Job to display
            self.jobs = [Job(1)]

    def update(self, **fields) -> None:
        with self._lock:
            for name, value in fields.items():
                setattr(self, name, value)

    def job(self, job_id: int = 1) -> Job:
        """The Job with this id, created on first use (ids are 1-based)."""
        with self._lock:
            for job in self.jobs:
                if job.job_id == job_id:
                    return job
            job = Job(job_id)
            self.jobs.append(job)
            self.jobs.sort(key=lambda j: j.job_id)
            return job

    def elapsed(self, now: Optional[float] = None) -> Optional[float]:
        """Total time since the run's FIRST iteration — the run's cost so far."""
        if not self.run_started_at:
            return None
        return max(0.0, (time.time() if now is None else now) - self.run_started_at)

    def mark_run_started(self, now: Optional[float] = None) -> None:
        """Latch the run clock on the first iteration; later calls are no-ops."""
        with self._lock:
            if not self.run_started_at:
                self.run_started_at = time.time() if now is None else now

    def snapshot(self) -> "LoopStatus":
        with self._lock:
            copy = LoopStatus(
                jobs=[j.snapshot() for j in self.jobs],
                iteration=self.iteration,
                max_iterations=self.max_iterations,
                random_order=self.random_order,
                provider=self.provider,
                run_started_at=self.run_started_at,
                phase=self.phase,
                quotas=list(self.quotas),
                script_limits=list(self.script_limits),
                stop_pending=self.stop_pending,
                note=self.note,
            )
        return copy


# --- segments ------------------------------------------------------------------


class Segment:
    """One field of a row. `text` returns None to be omitted entirely."""

    def text(self, status: LoopStatus, now: Optional[float] = None) -> Optional[str]:
        raise NotImplementedError


class IterationSegment(Segment):
    """"⟳ iter 12/40 rand" — phase glyph, counter, optional cap and order marker."""

    def text(self, status, now=None):
        glyph = PHASE_GLYPHS.get(status.phase, "·")
        text = f"{glyph} iter {status.iteration}"
        if status.max_iterations is not None:
            text += f"/{status.max_iterations}"
        if status.random_order:
            text += " rand"
        return text


class ProviderSegment(Segment):
    """"claude/opus" — the provider, plus the model when a single Job pins one.

    With several Jobs the model belongs on each JobRow (they may differ), so only
    the provider is shown here.
    """

    def text(self, status, now=None):
        if len(status.jobs) == 1:
            model = status.jobs[0].model or CLI_DEFAULT_MODEL
            return f"{status.provider}/{model}"
        return status.provider


class JobCountSegment(Segment):
    def text(self, status, now=None):
        return f"{len(status.jobs)} jobs" if len(status.jobs) > 1 else None


class LabelSegment(Segment):
    """The unit of work in flight.

    Not in the default layout — JobRow already shows the item in aligned columns,
    and repeating it on the summary row costs width that the quota figures need.
    Kept as a seam for a layout that has no job rows.
    """

    def text(self, status, now=None):
        for job in status.jobs:
            if job.running and job.item:
                return job.item
        return None


class ElapsedSegment(Segment):
    """The run's total elapsed time (the summary clock, not a job's)."""

    def text(self, status, now=None):
        return format_elapsed(status.elapsed(now)) or None


class QuotaSegment(Segment):
    """One provider quota: "session 43% (ceil 95%, 2h11m)"."""

    def __init__(self, index: int):
        self.index = index

    def text(self, status, now=None):
        try:
            label, percent, ceiling, reset_ts = status.quotas[self.index]
        except (IndexError, ValueError):
            return None
        if percent is None:
            return f"{label} n/a"
        detail = []
        if ceiling is not None:
            detail.append(f"ceil {ceiling:.0f}%")
        if reset_ts:
            detail.append(cyclecore._fmt_left(
                reset_ts - (time.time() if now is None else now)))
        suffix = f" ({', '.join(detail)})" if detail else ""
        return f"{label} {percent:.0f}%{suffix}"


class ScriptLimitSegment(Segment):
    """One of the script's own limits — "max-runs 40", "max-strike 3h", a ceiling.

    Indexed rather than named so wave 2 can grow the list from the
    SettingsRegistry (see `SettingsRegistry.status_entries`) without a renderer
    edit; pass a Setting directly to render a live knob.
    """

    def __init__(self, index: Optional[int] = None, setting: "Setting" = None):
        self.index = index
        self.setting = setting

    def text(self, status, now=None):
        if self.setting is not None:
            return f"{self.setting.name} {self.setting.format()}"
        try:
            label, value = status.script_limits[self.index]
        except (IndexError, TypeError, ValueError):
            return None
        return f"{label} {value}" if value else label


class StopSegment(Segment):
    """The pending-stop marker. Loud on purpose: the loop may take minutes to
    notice the sentinel, and the user needs to know cancelling is still possible."""

    def text(self, status, now=None):
        if not status.stop_pending:
            return None
        return "STOP pending — press s to cancel"


# --- rows ----------------------------------------------------------------------


class Row:
    """Exactly one terminal line, already truncated to `width`."""

    def render(self, status: LoopStatus, width: int,
               now: Optional[float] = None) -> str:
        raise NotImplementedError


class SegmentRow(Row):
    """Segments joined with ' · ', omitting the ones that return None."""

    separator = " · "

    def __init__(self, segments: Sequence[Segment], prefix: str = " "):
        self.segments = list(segments)
        self.prefix = prefix

    def render(self, status, width, now=None):
        parts = [s.text(status, now) for s in self.segments]
        body = self.separator.join(p for p in parts if p)
        return fit(self.prefix + body, width)


class JobRow(Row):
    """One Job: " job 1 ▶ garlic.md   opus   iter 3   3m01s".

    Keeps `job_id` rather than the Job object so a mouse click on this screen row
    maps back to the live Job (wave 4) even after the job list was rebuilt.
    """

    item_width = 26
    model_width = 12

    def __init__(self, job_id: int):
        self.job_id = job_id

    def _job(self, status: LoopStatus) -> Optional[Job]:
        for job in status.jobs:
            if job.job_id == self.job_id:
                return job
        return None

    def render(self, status, width, now=None):
        job = self._job(status)
        if job is None:
            return ""
        glyph = "▶" if job.running else "·"
        item = job.item if (job.running and job.item) else "idle"
        model = job.model or CLI_DEFAULT_MODEL
        elapsed = format_elapsed(job.elapsed(now))
        line = (f" job {job.job_id} {glyph} {pad(item, self.item_width)} "
                f"{pad(model, self.model_width)} "
                f"iter {job.iteration:<4}")
        if elapsed:
            line += f" {elapsed}"
        return fit(line, width)


class KeyLegendRow(Row):
    """"keys: s stop · h help" — BUILT from the registered Actions.

    Takes a callable rather than a list so a key added in a later wave (or one
    only available in the current Mode) appears here by registration alone; a
    literal string would go stale the first time that happened.
    """

    def __init__(self, entries: Callable[[], Sequence[Tuple[str, str]]] = None):
        self.entries = entries or (lambda: ())

    def render(self, status, width, now=None):
        parts = [f"{key} {label}" for key, label in self.entries() if key]
        if not parts:
            return ""
        return fit(" keys: " + " · ".join(parts), width)


class NoteRow(Row):
    """Transient feedback, or the active Mode's own prompt line.

    Always rendered (possibly blank): the reserved region is sized by row count,
    so a row that sometimes disappears would shift every other row.
    """

    def render(self, status, width, now=None):
        return fit(f" {status.note}", width) if status.note else ""


class Layout:
    """The row set: summary, one JobRow per Job, the legend, the note.

    One Layout serves both the sequential and the parallel runner — a sequential
    run is a run with one Job. Subclass only if a future mode needs a different
    set of rows.
    """

    def __init__(self, legend_entries: Callable[[], Sequence[Tuple[str, str]]] = None):
        self._legend_entries = legend_entries

    def summary_row(self, status: LoopStatus) -> Row:
        segments: List[Segment] = [IterationSegment(), ProviderSegment(),
                                   JobCountSegment(), ElapsedSegment()]
        # Built per paint from the data, so a new quota or limit shows up without
        # touching this file.
        segments += [QuotaSegment(i) for i in range(len(status.quotas))]
        segments += [ScriptLimitSegment(i) for i in range(len(status.script_limits))]
        segments.append(StopSegment())
        return SegmentRow(segments)

    def rows(self, status: LoopStatus) -> List[Row]:
        rows: List[Row] = [self.summary_row(status)]
        rows += [JobRow(job.job_id) for job in status.jobs]
        rows.append(KeyLegendRow(self._legend_entries))
        rows.append(NoteRow())
        return rows


def render_rows(status: LoopStatus, width: int, *, now: Optional[float] = None,
                layout: Optional[Layout] = None) -> List[str]:
    """The pinned area as plain text. Pure: no terminal, no clock of its own.

    All rendering lives here (and in the Rows) precisely so it can be tested
    without a terminal; the Terminal class only moves bytes.
    """
    layout = layout or Layout()
    snapshot = status.snapshot()
    return [row.render(snapshot, width, now) for row in layout.rows(snapshot)]


# --- terminal ------------------------------------------------------------------


def _enable_windows_vt(stream) -> bool:
    """Turn on ENABLE_VIRTUAL_TERMINAL_PROCESSING; True if escapes will work.

    Rich has usually done this already, but the status line must not depend on
    rich being installed. A console that refuses simply gets no status line.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class Terminal:
    """The reserved bottom region: bytes only, no idea what a row means.

    Mechanics: a DECSTBM scroll region shrinks the scrolling area by `rows`, so
    ordinary output keeps scrolling above untouched. Repaints save/restore the
    cursor around absolute moves, which is also what heals the rows after rich's
    Live has been moving the cursor around for a streamed Markdown block.
    """

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else cyclecore._real_stream()
        self._lock = threading.Lock()
        self._rows = 0
        self._size = (0, 0)
        self.failed = False

    @property
    def active(self) -> bool:
        return self._rows > 0 and not self.failed

    def size(self) -> Tuple[int, int]:
        """(columns, lines) of the terminal right now."""
        size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines

    def _write(self, text: str) -> bool:
        try:
            self._stream.write(text)
            self._stream.flush()
            return True
        except Exception:
            # A closed/odd stream must not take the loop down with it.
            self.failed = True
            return False

    def reserve(self, rows: int) -> bool:
        """Pin `rows` lines at the bottom. Safe to call again to resize."""
        if self.failed or rows <= 0:
            return False
        columns, lines = self.size()
        if columns < MIN_COLUMNS or lines < MIN_LINES or rows >= lines - 1:
            return False
        with self._lock:
            # Scroll only for the rows we are *adding*: re-establishing the
            # region on every resize must not walk the output up the screen.
            grown = max(0, rows - self._rows)
            self._rows = rows
            self._size = (columns, lines)
            first = lines - rows + 1
            return self._write(
                "\n" * grown
                + f"\x1b[{first};1H\x1b[0J"        # clear the whole reserved area
                + f"\x1b[1;{lines - rows}r"        # DECSTBM: shrink the scroll area
                + f"\x1b[{lines - rows};1H")       # park at the bottom of it

    def paint(self, lines: Sequence[str]) -> bool:
        if not self.active:
            return False
        with self._lock:
            columns, total = self._size
            first = total - self._rows + 1
            out = ["\x1b7"]  # save cursor (DECSC) — output above must not move
            for i in range(self._rows):
                text = lines[i] if i < len(lines) else ""
                out.append(f"\x1b[{first + i};1H\x1b[2K{text}")
            out.append("\x1b8")  # restore cursor (DECRC)
            return self._write("".join(out))

    def release(self) -> None:
        """Reset the scroll region, clear the rows, leave a clean cursor."""
        with self._lock:
            if self._rows <= 0:
                return
            _columns, total = self._size
            first = total - self._rows + 1
            out = ["\x1b7"]
            for i in range(self._rows):
                out.append(f"\x1b[{first + i};1H\x1b[2K")
            out.append("\x1b8")
            out.append("\x1b[r")             # full-height scroll region again
            out.append(f"\x1b[{first};1H")   # sit on the first freed line
            self._rows = 0
            self._write("".join(out))


class NullTerminal(Terminal):
    """Null Object: chosen when there is no TTY, when disabled, or after any
    error — so the controller has no `if enabled:` scattered through it."""

    def __init__(self):
        self._rows = 0
        self.failed = False

    @property
    def active(self) -> bool:
        return False

    def size(self):
        return (0, 0)

    def reserve(self, rows):
        return False

    def paint(self, lines):
        return False

    def release(self):
        return None


def terminal_for(stream=None, *, enabled: bool = True) -> Terminal:
    """A real Terminal when pinning rows is safe here, else a NullTerminal."""
    if not enabled or os.environ.get(ENV_FLAG) == "0":
        return NullTerminal()
    stream = stream if stream is not None else cyclecore._real_stream()
    try:
        if not stream.isatty():
            return NullTerminal()
    except Exception:
        return NullTerminal()
    if not _enable_windows_vt(stream):
        return NullTerminal()
    size = shutil.get_terminal_size(fallback=(0, 0))
    if size.columns < MIN_COLUMNS or size.lines < MIN_LINES:
        return NullTerminal()
    return Terminal(stream)


# --- input ---------------------------------------------------------------------


class InputEvent:
    """Something the user did. Subclasses carry the payload."""


@dataclass
class Key(InputEvent):
    char: str


@dataclass
class Mouse(InputEvent):
    """An SGR mouse report. Produced from wave 4 on; the type exists now so the
    Mode/Action dispatch already speaks it."""

    button: int
    x: int
    y: int
    pressed: bool = True


@dataclass
class Resize(InputEvent):
    columns: int
    lines: int


# Named keys. A `Key.char` longer than one character is one of these symbolic
# names, so a Mode can ask for "left" without knowing whether the terminal spelt
# it as a CSI sequence (POSIX) or a two-byte scan code (Windows).
CSI_KEYS = {
    "A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
    "1~": "home", "3~": "delete", "4~": "end", "5~": "pgup", "6~": "pgdn",
    "7~": "home", "8~": "end",
}
# msvcrt reports these as a '\x00'/'\xe0' lead byte plus a scan code.
WINDOWS_KEYS = {
    "H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end",
    "I": "pgup", "Q": "pgdn", "S": "delete",
}
_WINDOWS_LEAD = ("\x00", "\xe0")


def decode_escape(sequence: str) -> Optional[InputEvent]:
    """A CSI sequence -> an InputEvent, or None to drop it.

    The single seam for escape-driven input: wave 4 turns
    `ESC [ < b ; x ; y M|m` into a Mouse here and nothing else changes. Unknown
    sequences are dropped rather than delivered as a burst of Keys, which would
    look to the user like random keypresses.
    """
    body = sequence[2:]                       # strip the leading "ESC ["
    name = CSI_KEYS.get(body)
    return Key(name) if name else None


class InputSource:
    """Produces InputEvents on a daemon thread."""

    def start(self, handler: Callable[[InputEvent], None]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class NullInputSource(InputSource):
    """No TTY (or disabled): yields nothing, so no key is ever read."""

    def start(self, handler):
        return None

    def stop(self):
        return None


class _EscapeDecoder:
    """Assembles keystrokes into InputEvents: plain chars, CSI sequences (POSIX)
    and msvcrt scan codes (Windows), so both platforms deliver the same
    symbolic `Key("left")` / `Key("pgup")`.

    A bare ESC is only known to be a bare ESC once nothing follows it, hence
    `flush()` — the reader calls it when its poll times out. Wave 2's edit mode
    needs Esc to mean cancel, so the distinction is not academic.
    """

    def __init__(self, scan_codes: bool = os.name == "nt"):
        # `scan_codes` off on POSIX: there a NUL is a normal (if odd) character,
        # and treating it as a lead byte would swallow the key after it.
        self.scan_codes = scan_codes
        self._buf = ""

    def feed(self, char: str) -> List[InputEvent]:
        if self._buf in _WINDOWS_LEAD and self._buf:   # lead byte + scan code
            name = WINDOWS_KEYS.get(char)
            self._buf = ""
            return [Key(name)] if name else []
        if not self._buf:
            if self.scan_codes and char in _WINDOWS_LEAD:
                self._buf = char
                return []
            if char == "\x1b":
                self._buf = char
                return []
            return [Key(char)]
        self._buf += char
        if len(self._buf) == 2 and char not in "[O":
            buf, self._buf = self._buf, ""
            return [Key("\x1b"), Key(buf[1])]
        if len(self._buf) >= 3 and "@" <= char <= "~":
            buf, self._buf = self._buf, ""
            event = decode_escape(buf)
            return [event] if event is not None else []
        return []

    def flush(self) -> List[InputEvent]:
        if self._buf == "\x1b":
            self._buf = ""
            return [Key("\x1b")]
        return []


class TerminalInput(InputSource):
    """Raw-ish key reader: msvcrt on Windows, termios+select on POSIX.

    cbreak, never raw: ISIG stays on so Ctrl+C keeps behaving exactly as it does
    today. On Windows msvcrt hands '\\x03' over instead, so it is turned back
    into a KeyboardInterrupt on the main thread.
    """

    poll_seconds = 0.05

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._decoder = _EscapeDecoder()

    def usable(self) -> bool:
        try:
            return bool(self._stream) and self._stream.isatty()
        except Exception:
            return False

    def start(self, handler):
        if self._thread is not None or not self.usable():
            return
        self._stop.clear()
        target = self._run_windows if os.name == "nt" else self._run_posix
        self._thread = threading.Thread(target=self._guard, args=(target, handler),
                                        name="statusline-input", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def _guard(self, target, handler):
        try:
            target(handler)
        except Exception:
            return  # a broken reader costs the keys, never the run

    def _emit(self, handler, char: str) -> None:
        if char == "\x03":  # Ctrl+C on the Windows path: keep the usual meaning
            import _thread

            _thread.interrupt_main()
            return
        for event in self._decoder.feed(char):
            handler(event)

    def _run_windows(self, handler):
        import msvcrt

        while not self._stop.is_set():
            if msvcrt.kbhit():
                self._emit(handler, msvcrt.getwch())
                continue
            for event in self._decoder.flush():
                handler(event)
            self._stop.wait(self.poll_seconds)

    def _run_posix(self, handler):
        import select
        import termios
        import tty

        fd = self._stream.fileno()
        saved = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)  # cbreak, not raw: ISIG (Ctrl+C) stays enabled
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], self.poll_seconds)
                if not ready:
                    for event in self._decoder.flush():
                        handler(event)
                    continue
                data = os.read(fd, 64).decode("utf-8", "replace")
                for char in data:
                    self._emit(handler, char)
        finally:
            # Guaranteed restore: a terminal left in cbreak is unusable.
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)


# --- actions and modes ---------------------------------------------------------


class Action:
    """One key the run answers to. Register it and the legend grows by itself."""

    key: str = ""
    keys: Tuple[str, ...] = ()
    help: str = ""

    def all_keys(self) -> Tuple[str, ...]:
        return self.keys or ((self.key,) if self.key else ())

    def help_text(self, app: "StatusApp") -> str:
        """Legend label; overridden when it depends on state (stop vs cancel)."""
        return self.help

    def available(self, app: "StatusApp") -> bool:
        return True

    def run(self, app: "StatusApp") -> None:
        raise NotImplementedError


class StopAction(Action):
    """Graceful stop, implemented as the existing `stop` sentinel file.

    Writing the file rather than setting a flag reuses the whole tested
    lifecycle — sequential loop, parallel workers, periodic batches, removal at
    application exit — instead of adding a second, parallel stop path. The loop
    may not look at the file for minutes, so the same key cancels the request.
    """

    key = "s"
    help = "stop"

    def help_text(self, app):
        return "cancel stop" if app.status.stop_pending else "stop"

    def run(self, app):
        if app.status.stop_pending:
            app.cancel_stop()
        else:
            app.request_stop()


class HelpAction(Action):
    key = "h"
    keys = ("h", "?")
    help = "help"

    def run(self, app):
        parts = [f"{'/'.join(a.all_keys())} {a.help_text(app)}"
                 for a in app.actions if a.available(app)]
        app.note("keys — " + " · ".join(parts))


class Mode:
    """A screen state on a stack: it may add rows and may consume events."""

    name = "mode"

    def __init__(self, app: "StatusApp"):
        self.app = app

    def rows(self, status: LoopStatus) -> List[Row]:
        return []

    def handle(self, event: InputEvent) -> bool:
        """True when the event was consumed (and must not fall through)."""
        return False


class NormalMode(Mode):
    """The base of the stack: dispatches registered Actions, nothing else."""

    name = "normal"

    def handle(self, event):
        if not isinstance(event, Key):
            return False
        action = self.app.action_for(event.char)
        if action is not None:
            action.run(self.app)
            return True
        if event.char.isprintable() and event.char.strip():
            self.app.note(f"unknown key {event.char!r} — press h for help")
        return True


# --- settings ------------------------------------------------------------------


class Setting:
    """ONE editable knob: the single source of truth for the limit editor AND for
    the reproducing command line.

    `flag` is the canonical CLI spelling `cmdline.FLAG_ALIASES` speaks, which is
    what lets `SettingsRegistry.overrides()` be handed straight to
    `cmdline.render` — an edited ceiling and the command line that reproduces it
    can then never disagree.
    """

    def __init__(self, name: str, flag: str, getter: Callable[[], object],
                 setter: Callable[[object], None], *, step: float = 1,
                 minimum: Optional[float] = None, maximum: Optional[float] = None):
        self.name = name
        self.flag = flag
        self._get = getter
        self._set = setter
        self.step = step
        self.minimum = minimum
        self.maximum = maximum
        self.initial = getter()

    @property
    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (self.minimum, self.maximum)

    def get(self):
        return self._get()

    def set(self, value) -> None:
        self._set(self.clamp(value))

    def clamp(self, value):
        if isinstance(value, (int, float)):
            if self.minimum is not None:
                value = max(self.minimum, value)
            if self.maximum is not None:
                value = min(self.maximum, value)
        return value

    def nudge(self, steps: float = 1) -> None:
        value = self.get()
        if value is None:
            return
        self.set(type(value)(value + steps * self.step))

    def changed(self) -> bool:
        return self.get() != self.initial

    def format(self) -> str:
        value = self.get()
        return "off" if value is None else str(value)

    def cli_value(self):
        """The value for `cmdline`: a string, True for a bare flag, or None to
        drop the flag entirely."""
        value = self.get()
        return None if value is None else str(value)


class PercentSetting(Setting):
    """A quota ceiling (session / weekly / day / night), 0..100."""

    def __init__(self, name, flag, getter, setter, *, step: float = 1):
        super().__init__(name, flag, getter, setter, step=step,
                         minimum=0, maximum=100)

    def format(self):
        value = self.get()
        return "off" if value is None else f"{float(value):.0f}%"

    def cli_value(self):
        value = self.get()
        return None if value is None else f"{float(value):.0f}"


class NumberSetting(Setting):
    """A plain count or duration — --max-runs, --max-strike."""


class SettingsRegistry:
    """Ordered Settings, and the bridge to the command-line renderer."""

    def __init__(self, settings: Sequence[Setting] = ()):
        self._settings: List[Setting] = list(settings)

    def add(self, setting: Setting) -> Setting:
        self._settings.append(setting)
        return setting

    def __iter__(self):
        return iter(self._settings)

    def __len__(self):
        return len(self._settings)

    def get(self, name: str) -> Optional[Setting]:
        for setting in self._settings:
            if setting.name == name:
                return setting
        return None

    def status_entries(self) -> List[Tuple[str, str]]:
        """[(label, text)] for `LoopStatus.script_limits` — one entry per knob."""
        return [(s.name, s.format()) for s in self._settings]

    def overrides(self, *, changed_only: bool = True) -> dict:
        """Exactly the dict `cmdline.render` consumes: {canonical flag: value}.

        Only edited knobs by default — an unedited setting is already spelled
        (or deliberately absent) in the original argv, and restating it would
        make the reproducing line drift from the one that was actually run.
        """
        out = {}
        for setting in self._settings:
            if changed_only and not setting.changed():
                continue
            out[setting.flag] = setting.cli_value()
        return out


# --- quota feed ----------------------------------------------------------------


def quota_rows(usage_source, limit_policy=None, *, cache_value: bool = True,
               now: Optional[float] = None) -> List[Tuple]:
    """The provider's live quotas as `LoopStatus.quotas` tuples.

    Reads through the SAME cached UsageSource the limit machinery uses, so
    calling it right after a limit check costs no HTTP round-trip at all. Any
    failure yields no rows: a status line must never be able to raise into the
    loop (or into the background refresher's thread).
    """
    if usage_source is None:
        return []
    try:
        usage = usage_source.get_usage(cache_value)
        moment = time.time() if now is None else now
        rules = list(getattr(limit_policy, "rules", None) or ())
        if rules:
            rows = []
            for rule in rules:
                reading = rule.reading(usage)
                rows.append((short_quota_label(rule.label), reading.percent,
                             rule.ceiling(reading, moment), reading.reset_ts))
            return rows
        return [(label, reading.percent, None, reading.reset_ts)
                for label, reading in (("session", usage.session),
                                       ("week", usage.week_all))]
    except Exception:
        return []


def push_quotas(app: "StatusApp", usage_source, limit_policy=None, *,
                cache_value: bool = True) -> None:
    """Refresh the status line's quota figures from a reading already paid for.

    Silent when the status line is off: nobody would see the figures, and
    `get_usage` on a cold cache is a real HTTP round-trip — a disabled status
    line must cost the run exactly nothing.
    """
    if app is None or usage_source is None or not app.enabled:
        return
    app.update(quotas=quota_rows(usage_source, limit_policy,
                                 cache_value=cache_value))


class QuotaRefresher:
    """Low-frequency background poll so the pinned figures do not go stale while
    a long iteration runs. Uses the shared cached source, hence the low rate."""

    def __init__(self, app: "StatusApp", usage_source, limit_policy=None, *,
                 provider: str = "claude", interval: Optional[float] = None):
        self.app = app
        self.usage_source = usage_source
        self.limit_policy = limit_policy
        self.interval = interval if interval is not None else (
            CODEX_QUOTA_REFRESH if provider == "codex" else CLAUDE_QUOTA_REFRESH)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if (self._thread is not None or self.usage_source is None
                or not self.app.enabled):
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="statusline-quotas",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            # cache_value=False: this poll exists precisely to age out the cache.
            push_quotas(self.app, self.usage_source, self.limit_policy,
                        cache_value=False)


# --- the controller ------------------------------------------------------------


class StatusApp:
    """Owns the Terminal, the Layout, the input, the Mode stack, the Actions, the
    SettingsRegistry, the repaint thread and the LoopStatus.

    Public surface the loops use: start()/stop() (or `with`), update(**fields),
    note(text), register_action(a), job(job_id). Everything else is an internal
    detail that a later wave extends by registering something.
    """

    def __init__(self, *, status: Optional[LoopStatus] = None,
                 terminal: Optional[Terminal] = None,
                 input_source: Optional[InputSource] = None,
                 layout: Optional[Layout] = None,
                 settings: Optional[SettingsRegistry] = None,
                 enabled: bool = True, refresh: float = REFRESH_SECONDS,
                 stop_file: Optional[str] = None,
                 default_actions: bool = True):
        self.status = status or LoopStatus()
        self.settings = settings or SettingsRegistry()
        self.terminal = terminal if terminal is not None else terminal_for(
            enabled=enabled)
        self.layout = layout or Layout(self.legend_entries)
        self.actions: List[Action] = []
        self.modes: List[Mode] = [NormalMode(self)]
        self.refresh = refresh
        self._stop_file = stop_file
        self._input = input_source
        self._paint_stop = threading.Event()
        self._paint_thread: Optional[threading.Thread] = None
        self._note_at = 0.0
        self._reserved = 0
        self._started = False
        self._services: List[object] = []
        if default_actions:
            self.register_action(StopAction())
            self.register_action(HelpAction())

    # --- lifecycle ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self.terminal.active

    def add_service(self, service) -> object:
        """Attach a start()/stop() helper to the app's lifetime (e.g. the quota
        refresher). Tying it here is what guarantees its thread dies with the
        run — run_loop is called repeatedly by periodic wrappers."""
        self._services.append(service)
        if self._started:
            self._start_service(service)
        return service

    def _start_service(self, service) -> None:
        try:
            service.start()
        except Exception:
            pass

    def start(self) -> "StatusApp":
        if self._started:
            return self
        self._started = True
        try:
            self.status.update(stop_pending=os.path.exists(self.stop_file))
            if isinstance(self.terminal, NullTerminal):
                return self   # disabled: no region to pin, no keys to read
            if not self._reserve(len(self.rows())):
                self.disable()
                return self
            self._paint()
            if self._input is None:
                self._input = (TerminalInput() if self.terminal.active
                               else NullInputSource())
            self._input.start(self.handle_event)
            self._paint_thread = threading.Thread(
                target=self._repaint_loop, name="statusline-paint", daemon=True)
            self._paint_thread.start()
        except Exception:
            self.disable()
        for service in self._services:
            self._start_service(service)
        return self

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        for service in self._services:
            try:
                service.stop()
            except Exception:
                pass
        self._paint_stop.set()
        thread, self._paint_thread = self._paint_thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        if self._input is not None:
            try:
                self._input.stop()
            except Exception:
                pass
        try:
            self.terminal.release()
        except Exception:
            pass

    def __enter__(self) -> "StatusApp":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False   # never swallow the loop's exceptions (incl. SystemExit)

    def disable(self) -> None:
        """Permanently swap in a NullTerminal — the run continues unaffected."""
        try:
            self.terminal.release()
        except Exception:
            pass
        self.terminal = NullTerminal()

    # --- state -------------------------------------------------------------

    def update(self, **fields) -> None:
        self.status.update(**fields)
        self._paint()

    def note(self, text: str) -> None:
        self._note_at = time.time()
        self.update(note=text)

    def job(self, job_id: int = 1) -> Job:
        """The Job with this id (created on first use). Wave 3 hands one per
        worker; the sequential loop uses job 1."""
        return self.status.job(job_id)

    def mark_run_started(self, now: Optional[float] = None) -> None:
        self.status.mark_run_started(now)

    @property
    def stop_file(self) -> str:
        # Read late: --project-dir moves cyclecore.STOP_FILE after import.
        return self._stop_file or cyclecore.STOP_FILE

    def request_stop(self) -> None:
        """Create the sentinel. The loop claims and removes it (see
        `cyclecore.stop_file_lifecycle`); we deliberately do not latch it here,
        so a cancel before the loop notices leaves no trace."""
        try:
            with open(self.stop_file, "w", encoding="utf-8") as fh:
                fh.write("")
        except OSError as exc:
            self.note(f"could not create {self.stop_file}: {exc}")
            return
        self.status.update(stop_pending=True)
        self.note("stop requested — the loop halts at the next iteration boundary")

    def cancel_stop(self) -> None:
        try:
            os.remove(self.stop_file)
        except FileNotFoundError:
            pass
        except OSError as exc:
            self.note(f"could not remove {self.stop_file}: {exc}")
            return
        self.status.update(stop_pending=False)
        self.note("stop request cancelled")

    # --- actions, modes, events --------------------------------------------

    def register_action(self, action: Action) -> Action:
        self.actions.append(action)
        return action

    def action_for(self, char: str) -> Optional[Action]:
        for action in self.actions:
            if char in action.all_keys() and action.available(self):
                return action
        return None

    def legend_entries(self) -> List[Tuple[str, str]]:
        """(key, label) per available Action — the KeyLegendRow's only source."""
        return [(a.key or (a.all_keys() or ("",))[0], a.help_text(self))
                for a in self.actions if a.available(self)]

    def push_mode(self, mode: Mode) -> None:
        self.modes.append(mode)
        self._paint()

    def pop_mode(self) -> None:
        if len(self.modes) > 1:
            self.modes.pop()
            self._paint()

    @property
    def mode(self) -> Mode:
        return self.modes[-1]

    def handle_event(self, event: InputEvent) -> None:
        try:
            if isinstance(event, Resize):
                self._reserve(len(self.rows()))
                self._paint()
                return
            for mode in reversed(self.modes):
                if mode.handle(event):
                    break
            self._paint()
        except Exception:
            self.disable()

    # --- rendering ---------------------------------------------------------

    def rows(self) -> List[Row]:
        snapshot = self.status.snapshot()
        return self.layout.rows(snapshot) + self.mode.rows(snapshot)

    def render(self, width: Optional[int] = None,
               now: Optional[float] = None) -> List[str]:
        """The pinned rows as plain text (no escapes) — what the tests read."""
        if width is None:
            width = self.terminal.size()[0] or 100
        snapshot = self.status.snapshot()
        rows = self.layout.rows(snapshot) + self.mode.rows(snapshot)
        return [row.render(snapshot, width, now) for row in rows]

    def _reserve(self, rows: int) -> bool:
        """True when the region has the requested shape (or there is none to keep)."""
        if isinstance(self.terminal, NullTerminal):
            return True
        if self.terminal.reserve(rows):
            self._reserved = rows
            return True
        return False

    def _paint(self) -> None:
        if not self.terminal.active:
            return
        try:
            columns, _lines = self.terminal.size()
            rows = self.render(columns)
            if len(rows) != self._reserved:
                # A Mode added or dropped a row: resize the region rather than
                # painting into lines the terminal is still scrolling.
                if not self._reserve(len(rows)):
                    return
            self.terminal.paint([colorize(line) for line in rows])
        except Exception:
            self.disable()

    def _repaint_loop(self) -> None:
        last_size = self.terminal.size()
        ticks = 0
        while not self._paint_stop.wait(self.refresh):
            try:
                ticks += 1
                size = self.terminal.size()
                if size != last_size:
                    last_size = size
                    self.handle_event(Resize(size[0], size[1]))
                if self._note_at and time.time() - self._note_at > NOTE_TTL:
                    self._note_at = 0.0
                    self.status.update(note="")
                if ticks % 4 == 0:
                    # Also catches a sentinel created by hand (`touch stop`).
                    pending = os.path.exists(self.stop_file)
                    if pending != self.status.stop_pending:
                        self.status.update(stop_pending=pending)
                self._paint()
            except Exception:
                self.disable()
                return
