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

import atexit
import os
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

from . import cmdline, cyclecore

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
    "MessageAction",
    "MessageMode",
    "MessagePromptRow",
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
    "QuotaRow",
    "QuotaSegment",
    "Resize",
    "Row",
    "RuleRow",
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
    "fit_tail",
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
ENV_FLAG = "LLM_LOOP_STATUSLINE"

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

# Shown wherever a pending stop is announced — the phase glyph and the pending
# marker both. A stop request is the one state the user may want to undo, so it
# gets a glyph that carries across a screenful of scrolling output. Double-width:
# measure it with `cell_width`, never len().
STOP_GLYPH = "🚫"

PHASE_GLYPHS = {
    "idle": "·",
    "running": "⟳",
    "paused": "⏸",
    "waiting": "⏳",
    "stopping": STOP_GLYPH,
}

# Shown when a command carries no --model: the provider CLI picks its own.
CLI_DEFAULT_MODEL = "cli default"

# Chrome: the furniture that frames the toolbar. One style for both the opening
# rule and the value separators, so nothing but the data is at full intensity —
# and both are ordinary characters, so a terminal with no colour still shows the
# separation (the rule IS the separation; degrading it to an empty line would
# merge the toolbar into the scrolling output above).
SEPARATOR = " | "
# Sub-separator INSIDE one field, currently only a quota's: to its left the
# provider's own figures, to its right what our policy says about them
# ("week 63% (17h27m) / ceil 95%"). A quieter mark than the field separator on
# purpose — the two halves are one subject seen from two sides, not two fields.
POLICY_SEPARATOR = " / "
# U+2500, not an underscore: the box-drawing glyph is designed to touch both
# cell edges, so a row of them reads as one continuous line, while underscores
# leave a gap at every cell boundary (and sit on the baseline, below where a
# separator belongs).
RULE_CHAR = "─"
CHROME_STYLE = "dim"


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


def fit_tail(text: str, width: int) -> str:
    """`text` cut to `width` columns from the LEFT, marked with a leading '…'.

    The mirror image of `fit`, for the one place where the END of a string is
    the part worth showing: a line being typed, whose interesting character is
    the one just entered. Wrapping is not an option there either — the reserved
    region is sized in whole rows.
    """
    if width <= 0:
        return ""
    if cell_width(text) <= width:
        return text
    out = []
    used = 0
    for ch in reversed(text):
        w = cell_width(ch)
        if used + w > width - 1:
            break
        out.append(ch)
        used += w
    return "…" + "".join(reversed(out))


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
    and a pinned row has no room for prose.

    The fallback path only: a quota the provider reports carries its own `short`
    in usage.QUOTAS, and this is how a CUSTOM rule watching something else still
    gets a readable label."""
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
# One pass over the line: a percentage takes the usage scale's colour, a pipe or
# a run of rule glyphs takes the muted chrome style. Built from RULE_CHAR so the
# two cannot drift apart when the glyph changes.
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_STYLED_RE = re.compile(
    rf"\d+(?:\.\d+)?\s*%|{re.escape(RULE_CHAR)}{{2,}}|\|")


def colorize(line: str) -> str:
    """Colour a rendered row: percentages on the usage scale, chrome muted.

    Reuses `cyclecore.percent_style` so the pinned rows and the scrolling log
    lines agree about what "alarming" looks like. Adding the codes here rather
    than in the Rows keeps the truncation arithmetic counting characters instead
    of escape bytes — and keeps the rows readable as plain strings in tests.

    One exception to the scale: the figure right of POLICY_SEPARATOR
    ("week 50% (3d14h) / ceil 95%") is a THRESHOLD, not a reading. On the usage
    scale it would glow red permanently while nothing is wrong — a red that
    means "the ceiling is high", which is the opposite of what red means
    everywhere else on the row. So it borrows the colour of the reading it
    qualifies, and the field as a whole shows one state: how the account is
    doing against this ceiling.

    When the provider reported no figure at all ("session n/a / ceil 95%" — the
    codex source has no session window) there is no reading to borrow from, and
    the ceiling falls back to green rather than to the scale: nothing is being
    spent against it, so an alarming colour would announce a problem that
    cannot exist. The same rule states it: the field shows ONE state, and here
    that state is "not gating anything".
    """
    out = []
    last = 0
    reading_style = None    # style of the last provider figure, for its policy half
    reading_end = 0         # ...and where it ended, to see what separates the two
    field_start = 0         # start of the field being painted (after the last pipe)
    for match in _STYLED_RE.finditer(line):
        token = match.group(0)
        if _PERCENT_RE.fullmatch(token):
            # Only what stands between this figure and its own field's reading —
            # never text from the field before the pipe, which would make a
            # neighbour's separator look like this field's.
            gap = line[max(field_start, reading_end):match.start()]
            if POLICY_SEPARATOR in gap:
                # Policy half of the field: the reading's colour, or green when
                # the provider had no reading to give.
                style = (reading_style if reading_style is not None
                         else _SGR.get(cyclecore.PERCENT_STYLES[0], ""))
            else:
                style = _SGR.get(cyclecore.percent_style(
                    float(token.rstrip("% \t"))), "")
                reading_style, reading_end = style, match.end()
        else:
            style = _SGR.get(CHROME_STYLE, "")
            if token == SEPARATOR.strip():
                reading_style = None        # next field, next reading
                field_start = match.end()
        out.append(line[last:match.start()])
        out.append(f"{style}{token}{_SGR_RESET}" if style else token)
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


class QuotaRow(NamedTuple):
    """One quota as the pinned row shows it — two halves from two sources.

    The first three fields are the PROVIDER's: which window, how much of it is
    spent, when it resets. They are what the account says and are shown for every
    window the provider reports, whether or not the run is gated on it.

    `policy` is ours: whatever the LimitRule watching this window contributes
    (its ceiling, by default), or "" when no rule watches it. Keeping it a
    rendered string rather than a number is what lets a rule say something other
    than a ceiling without any change here.
    """

    label: str
    percent: Optional[float]
    reset_ts: Optional[float]
    policy: str = ""


@dataclass
class LoopStatus:
    """Everything the rows can show. Later waves add Rows and Segments, not flags.

    `quotas` is a list of `QuotaRow` and `script_limits` is [(label, text)] —
    both are lists rather than named fields exactly so a new figure is a new
    entry, not a new attribute plus a renderer edit.
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
    # "" when nothing is pending, else the cyclecore.StopSource value that is
    # asking for the stop. The SOURCE, not a bool, because the two read
    # differently on the row: only a `s` request can be taken back from here.
    stop_pending: str = ""
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


class InvocationProgress:
    """The figures of one INVOCATION, kept alive across several runner calls.

    A StatusApp and its Jobs are built inside run_parallel/run_loop, so every
    counter that lives in them restarts when the call does. One process is one
    invocation, but a wrapper may slice it into several calls — the periodic
    runner (--grow-kit-periodically N) calls run_parallel once per batch — and
    then per-call counters are the wrong quantity: the summary row froze at the
    batch cap and every job row restarted at 1. So the invocation-wide figures
    live here, and the wrapper hands the same instance to every call it makes.
    A runner given none makes its own, which is the single-call case unchanged.

    Two things are kept:

      * the Job pool, reused by job_id — the Job objects already ARE the
        per-worker state, so a worker resuming with the same Job keeps its count
        correct by construction, with no second bookkeeping to disagree with it;
      * list progress, where `done` is DERIVED as (pending when the invocation
        began − pending now) rather than counted. That is self-correcting: items
        a preflight strikes, or work a previous run finished, move the figure
        with the list instead of leaving a counter to drift away from it.

    The cap is one global number for the whole invocation (`max_items`), never
    the wrapper's per-batch cap — a batch cap is an internal detail of how the
    work is sliced and says nothing about how much the run has to do.
    """

    def __init__(self, max_items: Optional[int] = None):
        self._lock = threading.Lock()
        self.max_items = max_items   # the invocation's --max (None = uncapped)
        self._pool = {}              # job_id -> Job, reused by every runner call
        self._baseline = None        # list items pending when the invocation began
        self._remaining = None       # list items pending now
        self._iterations = 0         # non-list runs count iterations instead

    def jobs(self, count: int) -> List[Job]:
        """The `count` job rows of one runner call, reused across calls.

        The pool grows to the widest call ever made; a narrower call gets only
        the rows its workers will actually feed, so no row is resurrected for a
        worker that is not running — while its count is kept for a later, wider
        batch that does run it.
        """
        with self._lock:
            for job_id in range(1, count + 1):
                if job_id not in self._pool:
                    self._pool[job_id] = Job(job_id)
            return [self._pool[job_id] for job_id in range(1, count + 1)]

    def track_list(self, pending: int) -> None:
        """Latch the list's size at the start of the invocation (first call wins).

        Later calls must NOT re-baseline: their list is already short by whatever
        the earlier batches finished, and re-reading it there is what made the
        denominator describe a batch instead of the run.
        """
        with self._lock:
            if self._baseline is None:
                self._baseline = pending
                self._remaining = pending

    def note_remaining(self, remaining: int) -> None:
        """Record how many list items are still pending (the source of `done`)."""
        with self._lock:
            if self._baseline is not None:
                self._remaining = remaining

    def note_iteration(self) -> None:
        """Count one iteration of a non-list run (a list run derives it instead)."""
        with self._lock:
            self._iterations += 1

    def summary_fields(self) -> dict:
        """`iteration`/`max_iterations` for the summary row, as update() kwargs.

        A list-backed run reads as items COMPLETED out of the smaller of what the
        list held at the start and the cap — `done` clamped into the total, so a
        capped run cannot report more than it promised even when a preflight
        strikes extra items. Any other driver counts its own iterations and gets a
        denominator only when one was actually given.
        """
        with self._lock:
            if self._baseline is None:
                return {"iteration": self._iterations,
                        "max_iterations": self.max_items}
            total = (self._baseline if self.max_items is None
                     else min(self._baseline, self.max_items))
            done = min(max(self._baseline - self._remaining, 0), total)
            return {"iteration": done, "max_iterations": total}


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
    """One quota window: "session 43% (2h11m) / ceil 95%".

    Left of the slash is the provider's report — the figure and how long the
    window still has to run, which together are what a percentage actually means
    (43% with two hours left is a different situation from 43% with ten minutes
    left). Right of it, when a rule watches this window, is what our own policy
    makes of it. A window nobody gates on simply has no right-hand half; it is
    still shown, because it is still spending the account's budget.
    """

    def __init__(self, index: int):
        self.index = index

    def text(self, status, now=None):
        try:
            row = QuotaRow(*status.quotas[self.index])
        except (IndexError, TypeError, ValueError):
            return None
        if row.percent is None:
            # No figure for this window — the provider does not meter it, or the
            # report was silent. Say so rather than dropping the row, so a quota
            # that stops being reported is visible instead of invisible.
            provider = f"{row.label} n/a"
        else:
            left = (cyclecore._fmt_left(
                row.reset_ts - (time.time() if now is None else now))
                if row.reset_ts else "")
            provider = f"{row.label} {row.percent:.0f}%" + (f" ({left})" if left else "")
        return provider + (POLICY_SEPARATOR + row.policy if row.policy else "")


class ScriptLimitSegment(Segment):
    """One of the script's own limits — "max-runs 40", a ceiling.

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
    notice the request, and the user needs to know cancelling is still possible.

    A stop FILE is somebody else's request — this run only obeys it — so the row
    must not offer a cancel the `s` key will not perform."""

    def text(self, status, now=None):
        if not status.stop_pending:
            return None
        # Once the phase itself is "stopping" the row already opens with the
        # glyph, and printing it twice on one line reads as noise, not urgency.
        glyph = "" if PHASE_GLYPHS.get(status.phase) == STOP_GLYPH \
            else f"{STOP_GLYPH} "
        tail = ("press s to cancel"
                if status.stop_pending == cyclecore.StopSource.KEY.value
                else "stop file present")
        return f"{glyph}STOP pending — {tail}"


# --- rows ----------------------------------------------------------------------


class Row:
    """Exactly one terminal line, already truncated to `width`."""

    def render(self, status: LoopStatus, width: int,
               now: Optional[float] = None) -> str:
        raise NotImplementedError


class RuleRow(Row):
    """The full-width rule of underscores that opens the status area.

    A Row like every other one on purpose: the reserved region is sized by the
    row COUNT, so a rule special-cased in the Terminal would put the region one
    line out of step with what is painted. Rendered as plain characters, dimmed
    later by `colorize` — it must survive a terminal that shows no colour, since
    it is the only thing separating the toolbar from the scrolling output.
    """

    def render(self, status, width, now=None):
        return RULE_CHAR * max(0, width)


class SegmentRow(Row):
    """Segments joined with the chrome separator, omitting the None ones."""

    separator = SEPARATOR

    def __init__(self, segments: Sequence[Segment], prefix: str = " "):
        self.segments = list(segments)
        self.prefix = prefix

    def render(self, status, width, now=None):
        parts = [s.text(status, now) for s in self.segments]
        body = self.separator.join(p for p in parts if p)
        return fit(self.prefix + body, width)


class JobRow(Row):
    """One Job: " job 1 ▶ opus | iter 3 | 3m01s | garlic.md".

    Keeps `job_id` rather than the Job object so a mouse click on this screen row
    maps back to the live Job (wave 4) even after the job list was rebuilt.

    The item comes LAST and is the only unpadded cell, so it spends whatever is
    left of the line: it is the one field with no bound (a config path can be any
    length), while every field before it is short and fixed. The fixed cells are
    padded — including an empty elapsed for an idle job — so the item column
    starts at the same offset on every row and the names stay scannable.
    """

    elapsed_width = 6
    # The model column is sized from the data instead of pinned, because a model
    # name is an identifier the reader may want to recognise or copy, and
    # "gpt-5.6-ter…" says less than the two extra columns cost. Still bounded, so
    # one absurd name cannot squeeze the item column off the row; and it is the
    # widest name across ALL jobs, so the rows stay aligned under each other.
    model_width_max = 24

    def __init__(self, job_id: int):
        self.job_id = job_id

    def model_width(self, status: LoopStatus) -> int:
        widths = [cell_width(job.model or CLI_DEFAULT_MODEL)
                  for job in status.jobs]
        return min(self.model_width_max, max(widths, default=0))

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
        # Same separator as every other row (the leading columns stay padded, so
        # the job rows still line up under each other with -j N).
        # The glyph already separates the job from its model, so no pipe there.
        cells = [f" job {job.job_id} {glyph} {pad(model, self.model_width(status))}",
                 f"iter {job.iteration:<4}",
                 pad(elapsed, self.elapsed_width),
                 item]
        return fit(SEPARATOR.join(cells), width)


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
        return fit(" keys: " + SEPARATOR.join(parts), width)


class NoteRow(Row):
    """Transient feedback, or the active Mode's own prompt line.

    Always rendered (possibly blank): the reserved region is sized by row count,
    so a row that sometimes disappears would shift every other row.
    """

    def render(self, status, width, now=None):
        return fit(f" {status.note}", width) if status.note else ""


class Layout:
    """The row set: the rule, summary, one JobRow per Job, the legend, the note.

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
        rows: List[Row] = [RuleRow(), self.summary_row(status)]
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

    def paint(self, lines: Sequence[str], *, reassert: bool = False) -> bool:
        if not self.active:
            return False
        with self._lock:
            columns, total = self._size
            first = total - self._rows + 1
            out = ["\x1b7"]  # save cursor (DECSC) — output above must not move
            if reassert:
                # The region is not sticky: an `ESC[r`, an alt-screen switch or a
                # full reset from anything downstream (a provider CLI, a pager)
                # un-pins it for good, and every absolute row write below would
                # then scroll away with the output. Re-stating it costs one short
                # escape on the periodic repaint; DECSTBM homes the cursor, which
                # the DECRC at the end undoes.
                out.append(f"\x1b[1;{total - self._rows}r")
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

    def paint(self, lines, *, reassert=False):
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

    def restore_tty(self) -> None:
        """Undo any terminal mode this source set. Nothing to undo by default.

        Separate from `stop()` because it must be callable from a signal handler
        or from atexit, where the reader thread's own `finally` will never run.
        """
        return None


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
        # (fd, saved termios attrs) while the tty is in cbreak, else None.
        self._tty_state: Optional[Tuple[int, object]] = None

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
        self.restore_tty()   # the thread normally did it; make sure of it here

    def restore_tty(self):
        """Take the tty back out of cbreak, from any thread, at most once."""
        state, self._tty_state = self._tty_state, None
        if state is None:
            return
        fd, saved = state
        try:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except Exception:
            pass

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
            # Published so a signal handler / atexit can undo it: this thread is
            # a daemon, and a signal kills it without running the finally below.
            self._tty_state = (fd, saved)
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
            self.restore_tty()


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
    """Graceful stop of THIS run, as an in-process request (see StopSource.KEY).

    Deliberately not the `stop` sentinel file: several loops are routinely
    launched in one project root, and a file stops all of them at once, which
    makes the key unusable for "stop this one". The file stays the cross-process
    channel; the key addresses the run whose terminal it was typed into. The
    loop may not look at the request for minutes, so the same key withdraws it —
    but only its own: a stop file this run merely obeys is not ours to remove.
    """

    key = "s"
    help = "stop"

    def help_text(self, app):
        return "cancel stop" if app.stop_requested_here else "stop"

    def run(self, app):
        if app.stop_requested_here:
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
        app.note("keys — " + SEPARATOR.join(parts))


class Mode:
    """A screen state on a stack: it may add rows, consume events, and name its
    own keys."""

    name = "mode"

    def __init__(self, app: "StatusApp"):
        self.app = app

    def rows(self, status: LoopStatus) -> List[Row]:
        return []

    def legend(self) -> Optional[Sequence[Tuple[str, str]]]:
        """(key, label) pairs to show INSTEAD of the Actions', or None.

        A mode that consumes every key makes the registered Actions unreachable,
        and a legend still advertising them is not decoration — it is wrong
        about what the next keystroke does.
        """
        return None

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


# --- typing a note to the running agent ----------------------------------------


class MessagePromptRow(Row):
    """The line being typed, pinned under the status rows.

    Holds the Mode rather than a string so the caret follows the buffer without
    anything having to push updates into a row object.
    """

    prefix = " ✉ "
    caret = "▏"
    # Shown while the line is empty — which is also the state the editor returns
    # to after Enter, so it doubles as "you are still in here".
    hint = "  Enter sends · Esc leaves"

    def __init__(self, mode: "MessageMode"):
        self.mode = mode

    def render(self, status, width, now=None):
        body = fit_tail(self.mode.buffer + self.caret,
                        max(0, width - cell_width(self.prefix)))
        line = self.prefix + body
        if not self.mode.buffer:
            line += self.hint
        return fit(line, width)


class MessageMode(Mode):
    """Typing a note. Consumes EVERY key while it is on the stack.

    A Mode and not just an Action because of what must NOT happen while a
    sentence is being typed: `s` is the stop key in NormalMode, and a note
    containing the word "stop" would otherwise halt the run halfway through
    being written. Consuming everything is the feature.
    """

    name = "message"

    def __init__(self, app):
        super().__init__(app)
        self.buffer = ""

    def rows(self, status):
        return [MessagePromptRow(self)]

    def legend(self):
        """While typing, `s` is a letter — so the legend must not offer it."""
        return [("Enter", "send"), ("Esc", "clear / leave")]

    def handle(self, event):
        if not isinstance(event, Key):
            return False        # Resize and friends still belong to the app
        char = event.char
        if char in ("\r", "\n"):
            self.submit()
        elif char == "\x1b":
            self.escape()
        elif char in ("\x08", "\x7f"):   # msvcrt / POSIX spellings of backspace
            self.buffer = self.buffer[:-1]
        elif len(char) == 1 and char.isprintable():
            self.buffer += char
        # Anything else (arrows, page keys, control characters) is swallowed
        # rather than dispatched: see the class docstring.
        return True

    def submit(self):
        """Send, and STAY in the editor.

        Leaving on Enter looked tidier and had a sharp edge: keys arrive one at
        a time from one burst, so a pasted `…\\nand stop after this` sent the
        first line, popped the mode, and handed the `s` of the second line to
        StopAction — the run halting is not what the person typing a note asked
        for. Staying means the mode is left only by an explicit Esc, and sending
        several notes in a row costs nothing.
        """
        text = self.buffer.strip()
        self.buffer = ""
        delivery = self.app.messages.submit(text) if self.app.messages else None
        self.app.note(delivery.message if delivery is not None
                      else "empty note discarded")

    def escape(self):
        """Esc clears a half-typed line; Esc on an empty line leaves.

        Two steps for the same reason submit no longer pops: a terminal reports
        Alt+key as ESC followed by the key, so a single-step Esc would let that
        key through to the normal dispatch.
        """
        if self.buffer:
            self.buffer = ""
            self.app.note("note discarded — Esc again to leave")
            return
        self.app.pop_mode()


class MessageAction(Action):
    """`m` — send the agent a note.

    Registered only when the run has a mailbox, which is what makes the key
    honest: several concurrent workers share one terminal and one keyboard, so
    there is no unambiguous "the agent" to talk to, and the runner hands out no
    mailbox in that case (see parallel.run_parallel).
    """

    key = "m"
    help = "message"

    def available(self, app):
        return getattr(app, "messages", None) is not None

    def help_text(self, app):
        waiting = app.messages.queued_count if app.messages else 0
        return f"message ({waiting} queued)" if waiting else "message"

    def run(self, app):
        # No note announcing the mode: the editor's own row carries the hint
        # while the line is empty, and the note row is where each delivery
        # reports itself a moment later.
        app.push_mode(MessageMode(app))


# --- settings ------------------------------------------------------------------


class Setting:
    """ONE editable knob: the single source of truth for the limit editor AND for
    the reproducing command line.

    `flag` is the canonical CLI spelling `cmdline.FLAG_ALIASES` speaks, which is
    what lets `SettingsRegistry.overrides()` be handed straight to
    `cmdline.render` — an edited ceiling and the command line that reproduces it
    can then never disagree.

    `show_in_status=False` keeps a knob editable and reproducible while leaving
    it off the pinned row — for the one case where a field would repeat what
    another field already says (see the `max-runs` registrations, whose value is
    the counter's own denominator). Row width is the scarce resource up there:
    every field costs the quota figures characters, and a second spelling of a
    number already on screen is the cheapest one to drop.
    """

    def __init__(self, name: str, flag: str, getter: Callable[[], object],
                 setter: Callable[[object], None], *, step: float = 1,
                 minimum: Optional[float] = None, maximum: Optional[float] = None,
                 show_in_status: bool = True):
        self.name = name
        self.flag = flag
        self._get = getter
        self._set = setter
        self.step = step
        self.minimum = minimum
        self.maximum = maximum
        self.show_in_status = show_in_status
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

    def __init__(self, name, flag, getter, setter, *, step: float = 1,
                 show_in_status: bool = True):
        super().__init__(name, flag, getter, setter, step=step,
                         minimum=0, maximum=100,
                         show_in_status=show_in_status)

    def format(self):
        value = self.get()
        return "off" if value is None else f"{float(value):.0f}%"

    def cli_value(self):
        value = self.get()
        return None if value is None else f"{float(value):.0f}"


class NumberSetting(Setting):
    """A plain count or duration — e.g. --max-runs."""


class SettingsRegistry:
    """Ordered Settings, and the bridge to the command-line renderer."""

    def __init__(self, settings: Sequence[Setting] = ()):
        self._settings: List[Setting] = []
        for setting in settings:
            self.add(setting)

    def add(self, setting: Setting) -> Setting:
        # Validated here so a mistyped flag fails at registration — i.e. at
        # startup, on every run — instead of raising KeyError out of
        # `cmdline.render` the first time somebody presses the key that shows
        # the reproducing command line.
        if setting.flag not in cmdline.FLAG_ALIASES:
            raise KeyError(
                f"setting {setting.name!r}: unknown canonical flag "
                f"{setting.flag!r}; known: {sorted(cmdline.FLAG_ALIASES)}")
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
        """[(label, text)] for `LoopStatus.script_limits` — one entry per knob
        that asked to be shown (`Setting.show_in_status`)."""
        return [(s.name, s.format()) for s in self._settings if s.show_in_status]

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


def _policy_part(rule, reading, moment: float) -> str:
    """A rule's own half of its row — never allowed to cost the provider's half.

    A custom `status()` is third-party code running on the repaint path, so a
    raise here degrades one field to the bare figures instead of blanking the
    whole quota area (which is what the single outer try/except would do).
    """
    if rule is None:
        return ""
    try:
        return rule.status(reading, moment) or ""
    except Exception:
        return ""


def quota_rows(usage_source, limit_policy=None, *, cache_value: bool = True,
               now: Optional[float] = None) -> List[QuotaRow]:
    """The provider's live quotas as `LoopStatus.quotas` rows.

    The rows follow the PROVIDER, not the policy: every window the account is
    metered on gets one — the 5-hour session and the week always, the others when
    there is something to say about them — so the reader sees the whole budget
    and not just the slice the run happens to be gated on. That the loop watches
    one window and not another is a property of the policy, and it shows up as
    the policy's own half of the row (see `LimitRule.status`), not as a window
    silently missing from the display.

    Reads through the SAME cached UsageSource the limit machinery uses, so
    calling it right after a limit check costs no HTTP round-trip at all. Any
    failure yields no rows: a status line must never be able to raise into the
    loop (or into the background refresher's thread).
    """
    if usage_source is None:
        return []
    try:
        snapshot = usage_source.get_usage(cache_value)
        moment = time.time() if now is None else now
        rule_for = getattr(limit_policy, "rule_for", lambda quota: None)
        rows = []
        known = set()
        for quota, reading in snapshot.readings():
            known.add(quota.field)
            rule = rule_for(quota.field)
            # A window this plan does not have (an unused Sonnet-only week) is
            # noise on a row that has to fit in a terminal — unless a rule gates
            # on it, in which case its absence is exactly what the reader needs
            # to see.
            if not quota.always and reading.percent is None and rule is None:
                continue
            rows.append(QuotaRow(quota.short, reading.percent, reading.reset_ts,
                                 _policy_part(rule, reading, moment)))
        # A custom rule may watch something the snapshot has no field for; it
        # still gets a row, built the way it used to be for every rule.
        for rule in getattr(limit_policy, "rules", None) or ():
            if rule.quota in known:
                continue
            known.add(rule.quota)
            reading = rule.reading(snapshot)
            rows.append(QuotaRow(short_quota_label(rule.label), reading.percent,
                                 reading.reset_ts,
                                 _policy_part(rule, reading, moment)))
        return rows
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


# Threads whose stdout writes are being diverted: ident -> list of chunks.
_capture_lock = threading.Lock()
_captured: dict = {}


class _ThreadScopedCapture:
    """A `sys.stdout` stand-in that diverts ONE thread's writes into a buffer.

    Needed because the background quota poll reaches `usage.query_usage_json`,
    which prints its diagnostics ("no usage figures: … 401 …"). Printed from a
    daemon thread they land at an arbitrary point of the scrolling output —
    possibly mid-token inside a rich `Live` block — and in the mirror log that
    `--cost` parses. Replacing `sys.stdout` outright for the duration would
    steal the LOOP's own output too, so the diversion is keyed on the thread
    that asked for it; every other thread passes straight through.
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


@contextmanager
def capture_stdout_here():
    """Collect THIS thread's stdout writes; yields the list of chunks."""
    ident = threading.get_ident()
    buffer: List[str] = []
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
        self._last_report = ""

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
            # Its diagnostics are captured, never printed — see
            # `_ThreadScopedCapture` — and surfaced on the note row instead,
            # which is the one place a background message can appear without
            # corrupting the stream or the mirror log.
            with capture_stdout_here() as chunks:
                push_quotas(self.app, self.usage_source, self.limit_policy,
                            cache_value=False)
            self._report("".join(chunks))

    def _report(self, text: str) -> None:
        """Put a captured diagnostic on the note row — once per distinct message.

        Silently swallowing it forever would hide a stale OAuth token behind
        figures that simply stop moving; repeating the same line every interval
        would sit permanently on the note row instead of the run's own feedback.
        """
        text = " ".join(text.split())
        if text == self._last_report:
            return
        self._last_report = text
        if not text:
            return                  # recovered: next failure is reportable again
        try:
            self.app.note(f"quota refresh: {text}")
        except Exception:
            pass


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
                 messages=None,
                 enabled: bool = True, refresh: float = REFRESH_SECONDS,
                 stop_file: Optional[str] = None,
                 default_actions: bool = True):
        self.status = status or LoopStatus()
        self.settings = settings or SettingsRegistry()
        # This run's operator.Mailbox, or None when there is nobody to address
        # (a dry run, several concurrent workers). Registering the key on the
        # same condition keeps the legend from offering what it cannot do.
        self.messages = messages
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
        self._requested_stop = False      # has `s` been pressed here? (see below)
        # Guards the request flag against the sentinel poll: the key press, the
        # cancel and the repaint's re-sync all move the pair together.
        self._stop_lock = threading.Lock()
        self._atexit_registered = False
        self._signal_handlers: List[Tuple[int, object]] = []
        if default_actions:
            self.register_action(StopAction())
            self.register_action(MessageAction())
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
            with self._stop_lock:
                self.status.update(stop_pending=self._sentinel_pending())
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
            self._install_emergency_restore()
        except Exception:
            self.disable()
        for service in self._services:
            self._start_service(service)
        return self

    # Signals worth a last-resort restore. SIGKILL / `taskkill /F` are
    # unblockable by definition — nothing can save the terminal there.
    _RESTORE_SIGNALS = ("SIGTERM", "SIGHUP", "SIGBREAK")

    def _install_emergency_restore(self) -> None:
        """Put the terminal back even when no `finally` of ours runs.

        stop() covers a normal return, an exception, SystemExit and
        KeyboardInterrupt — but a signal unwinds nothing, and the key reader's
        termios restore sits in a daemon thread that simply dies with the
        process. Without this a `kill`/`taskkill` on a long run leaves the
        DECSTBM region pinned and, on POSIX, the tty in cbreak.
        """
        atexit.register(self._emergency_restore)
        self._atexit_registered = True
        if threading.current_thread() is not threading.main_thread():
            return          # only the main thread may install signal handlers
        import signal

        for name in self._RESTORE_SIGNALS:
            number = getattr(signal, name, None)
            if number is None:
                continue
            try:
                previous = signal.getsignal(number)
                signal.signal(number, self._signal_restore)
            except (ValueError, OSError, RuntimeError):
                continue
            self._signal_handlers.append((number, previous))

    def _signal_restore(self, signum, frame) -> None:
        """Restore, then let the signal do what it was going to do."""
        self._emergency_restore()
        import signal

        previous = dict(self._signal_handlers).get(signum, signal.SIG_DFL)
        if callable(previous):
            previous(signum, frame)
            return
        # Re-raise under the original disposition: swallowing a termination
        # signal would make the process survive a kill it was never meant to.
        signal.signal(signum, previous if previous is not None else signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _remove_emergency_restore(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._emergency_restore)
            self._atexit_registered = False
        handlers, self._signal_handlers = self._signal_handlers, []
        if not handlers:
            return
        import signal

        for number, previous in handlers:
            try:
                signal.signal(number, previous)
            except (ValueError, OSError, RuntimeError):
                pass

    def _emergency_restore(self) -> None:
        """Release the region and the tty. Safe to call twice, never raises."""
        try:
            self.terminal.release()
        except Exception:
            pass
        try:
            if self._input is not None:
                self._input.restore_tty()
        except Exception:
            pass

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        self._remove_emergency_restore()
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

    @property
    def stop_requested_here(self) -> bool:
        """True while THIS run's `s` key is asking it to stop.

        In-process by design (see `cyclecore.StopSource`): it addresses the one
        run whose terminal the key was typed into, so concurrent loops sharing a
        project root can be stopped one at a time.

        `cyclecore.confirm_stop_request`'s cancel grace hangs off this rather
        than off `enabled`: a stop file written by somebody else (a script's
        `touch stop`, another run) must halt this run as promptly as it always
        did — nobody is sitting at this terminal waiting to press `s` again.
        """
        return self._requested_stop

    def request_stop(self) -> None:
        """Ask this run to stop. Sets the flag only — no file is written, so a
        cancel before the loop notices leaves no trace, an abrupt kill leaves no
        sentinel to clean up, and the loop next door keeps running."""
        with self._stop_lock:
            self._requested_stop = True
            self.status.update(stop_pending=cyclecore.StopSource.KEY.value)
        self.note("stop requested — this run halts at the next iteration "
                  "boundary (other runs are unaffected)")

    def cancel_stop(self) -> None:
        """Withdraw this run's own request. A stop file is left alone: it is not
        ours to remove, and the run stays pending on it."""
        with self._stop_lock:
            self._requested_stop = False
            self.status.update(stop_pending=self._sentinel_pending())
        self.note("stop request cancelled")

    def _sentinel_pending(self) -> str:
        """The stop-file half of `stop_pending` (call under `_stop_lock`)."""
        try:
            return (cyclecore.StopSource.FILE.value
                    if os.path.exists(self.stop_file) else "")
        except OSError:
            return ""

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
        """(key, label) per available Action — the KeyLegendRow's only source.

        A Mode that answers `legend()` replaces that list wholesale: while it
        holds every key, the Actions are not reachable and must not be offered.
        """
        own = self.mode.legend()
        if own is not None:
            return list(own)
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
                # A refused reserve() leaves the OLD geometry in place, so
                # carrying on would paint absolute rows outside the new screen
                # with the region set for the old one. Disable instead —
                # start() answers the same refusal the same way.
                if not self._reserve(len(self.rows())):
                    self.disable()
                    return
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

    def _paint(self, *, reassert: bool = False) -> None:
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
            self.terminal.paint([colorize(line) for line in rows],
                                reassert=reassert)
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
                    # Catches a sentinel created by hand (`touch stop`) or by a
                    # neighbouring run. It never touches `_requested_stop`: the
                    # key's request is this process's own and outlives whatever
                    # anyone does to the file. Under the same lock as the key
                    # press so the flag and the row cannot disagree.
                    with self._stop_lock:
                        pending = (cyclecore.StopSource.KEY.value
                                   if self._requested_stop
                                   else self._sentinel_pending())
                        if pending != self.status.stop_pending:
                            self.status.update(stop_pending=pending)
                # Re-assert the region on the periodic repaint: see Terminal.paint.
                self._paint(reassert=True)
            except Exception:
                self.disable()
                return
