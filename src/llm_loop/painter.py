"""Terminal frame protocol and the ordered data owner feeding its frames."""

import collections
import functools
import os
import sys
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

from . import diaglog, ownership, termio, textwidth

# Repaint cadence: fast enough that the elapsed clock ticks like a clock, slow
# enough to be invisible next to the agent's own output.
REFRESH_SECONDS = 0.5

# With the key trace on, who shares the console is logged every this many
# periodic ticks (10 at REFRESH_SECONDS = every 5 s): a descendant of the agent
# attached to the console can take keys from its one input queue
# (`diaglog.console_snapshot`), and the snapshot dates when it came and went.
CONSOLE_SNAPSHOT_TICKS = 10

# Coalesce a typing/paste burst into at most 30 frames per second. This is a UI
# cadence, not a readiness timeout: terminal writes must not pace the key reader.
INPUT_REFRESH_SECONDS = 1 / 30

# How long `Painter.open` waits for its own "first frame not done" stderr line
# before going on without it. The line is written on a short-lived thread of its
# own: a first frame that timed out means a terminal write is stuck, and stderr
# is very likely the same frozen console (a Windows console frozen by a mouse
# selection, which thaws on the next key). Written on the opener, it held a
# parallel run's main thread BEFORE `parallel.join_workers`, the one place that
# hears Ctrl+C. A healthy write returns long before this bound, so the line keeps
# its place in the output; past it the opener goes on and the line lands if
# the console thaws before the process exits, and is lost (without a crash, see
# `_stderr_writer`) if it does not.
OPEN_WARNING_WAIT_SECONDS = 1.0

# The painter thread's name: the pins tell "written on the painter" from
# "written beside it" by it.
PAINTER_THREAD_NAME = "statusline-paint"

# How many posted calls may wait for the painter before a post waits (or, for
# `Painter.try_post` and `Painter.post_key`, is refused). The owner's own bound
# and for its reasons (`ownership.DEFAULT_MAXSIZE`); named here, and read when a
# Painter is built, so a pin can make "full" a few keys away.
PAINTER_QUEUE_MAXSIZE = ownership.DEFAULT_MAXSIZE

# The region's life, as `Painter._region` records it.
#   UNOPENED — never opened: nobody owns the terminal, and a caller driving
#     the app by hand (a pin, a run naming its window early) paints and
#     resizes it itself.
#   OPEN — between `_open_region` and `_close_region`: the idle hook ticks it.
#   CLOSED — released (or disabled). Neither a frame nor a Resize may touch
#     the terminal now: a call posted after a timed-out stop() still runs on
#     the painter, after the release, and `Terminal.reserve` would re-pin the
#     region nobody is left to release. Until the next `_open_region`.
_REGION_UNOPENED, _REGION_OPEN, _REGION_CLOSED = "unopened", "open", "closed"

# How long a thread that hands the painter a call AND needs its effect before
# going on (`handle_event` / `disable` called directly while the app is started)
# waits for the painter to run it — queueing it included. A round trip is one
# wake-up of the painter: 0.02 ms median, 0.08 ms worst, three runs of 200 calls
# on `ownership.OwnerThread` (measured 2026-09-28; 20-23 ms median, 48 ms worst
# on 2026-09-25, when a posted call also waited out the INPUT_REFRESH_SECONDS
# burst window, and the bound was set against those). Only a painter stuck in
# a terminal write gets near this. A call that was queued still runs once it
# is free — the caller merely stops waiting; one that found no room in the
# queue for this long is dropped. Either is named on stderr.
POSTED_CALL_WAIT_SECONDS = 5.0

def _call_name(call: Callable) -> str:
    """`_handle_event` -> "handle_event": a posted call as a warning names it."""
    return (getattr(call, "__name__", None) or type(call).__name__).lstrip("_")


class _LatestCall:
    """A queued `StateOwner.post_latest` reading: `call(value)` with the value it
    holds when the owner runs it, which a later reading may have replaced."""

    def __init__(self, owner: "StateOwner", group, kind, call: Callable, value):
        self._owner = owner
        self.group = group
        self.kind = kind
        self.call = call
        self.value = value
        self.taken = False          # (latest lock) the owner has read `value`
        self.__name__ = _call_name(getattr(call, "func", call))

    def __call__(self) -> None:
        self.call(self._owner._take_latest(self))


class StateOwner:
    """Ordered data changes, coalesced readings and publication between frames.

    The terminal protocol supplies its current thread and a frame request;
    data ownership also works synchronously while that thread is absent.
    """

    def __init__(self, *, thread: Callable[[], Optional[threading.Thread]],
                 request_frame: Callable[[], None],
                 published: Callable[[], None]):
        self._thread = thread
        self._request_frame = request_frame
        self.published = published
        # State calls posted and not yet run (see `post`). A deque of its
        # own rather than the owner's queue: that one is bounded, and a state
        # call refused for room would be a change lost — a row left saying
        # "running" for the rest of the run — where a refused frame request
        # costs nothing the next frame does not repaint.
        self._state_calls: collections.deque = collections.deque()
        # The `post_latest` reading still pending per group, and the lock that
        # orders replacing its value against the owner taking it to run it.
        self._latest: Dict[object, "_LatestCall"] = {}
        self._latest_lock = threading.Lock()
        # Orders whoever runs the state calls. While a painter runs that is the
        # painter alone and the lock is never contended; it is there for the
        # stretches with no painter — before `open()`, after `close()`, and a
        # whole run with the status line off, whose workers each run their own
        # posts on their own thread — and for the hand-over between the two.
        # Held only across the state calls and `published`: no wait, no
        # terminal write. Re-entrant for a state call that reads `snapshot`
        # (`_applying` then answers it with the copy published so far).
        self._state_lock = threading.RLock()
        self._applying = False      # (state lock) a batch is running

    @property
    def pending(self) -> bool:
        """Whether an idle pass needs to apply data and schedule a frame."""
        return bool(self._state_calls)

    def post(self, call: Callable, *args, frame: bool = True,
             group=None) -> None:
        """`call(*args)` changes the app's data: run it on the data's owner and
        ask for a frame of the result — never waiting for the painter.

        The data has the terminal's owner: on the painter itself, and with no
        painter running, the call runs here and now (`apply`); from any
        other thread it is queued and the frame request it rides (`request_frame`)
        runs it on the painter, ahead of the frame. Queued calls keep the order
        they were posted in, from every thread. A call is never dropped: one
        whose frame request found the queue full runs on the painter's next
        idle pass or frame, whichever comes first (`_on_idle`, `_draw`), or
        at the next `apply` once the painter has handed back.

        A failing call costs that call only (it is recorded, not raised): the
        caller is a run that asked for a cosmetic change and has already gone on.

        `frame=False` asks for no frame, for a caller whose next step draws one
        anyway: `StatusApp.start`, whose change would otherwise be painted on
        the caller just before the painter's first frame paints it again.

        `group` names the data `call` changes when a `post_latest` reading
        may change it too, so the reading keeps its place behind this call.
        """
        if group is not None:
            with self._latest_lock:
                self._end_latest(group)
        self._state_calls.append((call, args))
        self.apply()          # here only on the owner, or with none
        if frame:
            self._request_frame()

    def post_latest(self, group, kind, call: Callable, value) -> None:
        """`post(call, value)` for a reading whose latest value wins.

        `group` names the data the reading changes (a Job row's id), `kind`
        the reading. While the last call posted for `group` is a reading of
        the same `kind` that has not run yet, `value` replaces its value
        instead of queueing a second call: applying the two in a row would
        leave the data exactly as applying the later one does. Order is kept
        by `post(..., group=g)`: any other change to that data ends the
        pending reading, so a reading posted after it is a new call behind it.
        Calls for other groups do not touch this data and need no order
        against it; a poster racing this one from another thread has none.

        A replaced call has its frame request already asked for (or kept by
        `_on_idle` / `_draw`), so a replacement asks for none.
        """
        with self._latest_lock:
            pending = self._latest.get(group)
            if (pending is not None and pending.kind == kind
                    and not pending.taken):
                pending.value = value
                return
            pending = _LatestCall(self, group, kind, call, value)
            self._latest[group] = pending
        self.post(pending)

    def _end_latest(self, group, pending=None) -> None:
        """No later reading of `group` folds into its pending one (`pending`
        only: the one that ran). Under `_latest_lock`."""
        if pending is None or self._latest.get(group) is pending:
            self._latest.pop(group, None)

    def _take_latest(self, pending: _LatestCall):
        """(state call) `pending`'s value as it runs: no later reading folds
        into it once read."""
        with self._latest_lock:
            pending.taken = True
            self._end_latest(pending.group, pending)
            return pending.value

    def apply(self) -> bool:
        """Run the state calls queued so far, then `published`, if this thread
        is the data's owner — the painter, or any caller while there is none.
        False, with nothing run, on any other thread: the painter runs them.

        A caller with no painter takes the state lock for the batch, so callers
        racing here (a parallel run with the status line off) run the calls one
        at a time and in the order they were posted.
        """
        thread = self._thread()
        if thread is not None and thread is not threading.current_thread():
            return False
        with self._state_lock:
            # Asked again from inside a state call (one that reads `snapshot`):
            # the batch already running is this thread's, and it runs the
            # rest in order once that call returns.
            if self._applying or not self._state_calls:
                return True
            self._applying = True
            try:
                while self._state_calls:
                    call, args = self._state_calls.popleft()
                    try:
                        call(*args)
                    except Exception as exc:
                        diaglog.record("Painter state call raised",
                                       _call_name(call), exc=exc)
                try:
                    self.published()
                except Exception as exc:
                    diaglog.record("Painter publish raised", exc=exc)
            finally:
                self._applying = False
        return True


class Painter:
    """The one thread that writes the status line's terminal, and its protocol.

    While it is open every terminal write happens on it — the frames, the
    region's pinning and release, a resize, a disable — and every other thread
    posts (`request_frame`, `run_and_wait`, `try_post`, `post_key`) and goes
    on. Before `open()` and once `close()` has handed back there is no painter
    thread, and the caller writes itself: that is how a run names its window
    before the region is pinned, and how a pin drives an app by hand.

    The protocol, all of it here:

      * one frame request in flight is the whole queue (`request_frame`), and
        a burst of them is one frame, INPUT_REFRESH_SECONDS after its first
        (`_schedule_frame`); the periodic frame comes every `refresh`
        (`_on_idle`);
      * nothing writes the terminal once the region is released, and a region
        the last `close()` released is not re-pinned before the next `open()`
        (`_REGION_CLOSED`): a frame, a resize and the idle hook all ask;
      * a key the reader cannot post is dropped and counted, never waited for
        (`post_key`), and the count is named once the painter is back;
      * the app's DATA is the terminal owner's too: a change to it is a state
        call (`StateOwner.post`) run by the owner before anything it paints reads
        the data, and the readers on other threads get the copy `published`
        took after the last batch (`StatusApp.snapshot`).

    WHAT is painted is the app's, handed in as callables and only ever called
    by the terminal's owner: `render(width)` the rows as plain text, `title()`
    the window title, `row_count()` how many rows the region needs,
    `tick(count)` the app's checks that ride on the periodic frame,
    `colorize(line)` the app's styling,
    `note(text)` where a report of dropped keys goes, and `published()` the
    app's copy of its data for readers, taken after state calls ran.
    """

    def __init__(self, terminal: termio.Terminal, *,
                 render: Callable[[int], List[str]],
                 title: Callable[[], str],
                 row_count: Callable[[], int],
                 tick: Callable[[int], None],
                 note: Callable[[str], None],
                 colorize: Callable[[str], str],
                 published: Callable[[], None] = lambda: None,
                 refresh: float = REFRESH_SECONDS):
        # Swapped for a NullTerminal by `disable`, for good (read: `terminal`).
        self._terminal = terminal
        self.refresh = refresh
        self._render = render
        self._colorize = colorize
        self._title = title
        self._row_count = row_count
        self._app_tick = tick
        self._note = note
        self.state = StateOwner(
            thread=lambda: self.thread, request_frame=self.request_frame,
            published=published)
        self._owner = ownership.OwnerThread(
            PAINTER_THREAD_NAME, maxsize=PAINTER_QUEUE_MAXSIZE,
            idle=self._on_idle)
        # Set while a frame request is posted and not yet run: one in flight
        # is the whole queue, because every request asks for the same thing —
        # "draw the state as it is NOW" — so ten before the painter gets to the
        # first are one frame, not ten. Cleared by the request itself BEFORE
        # the frame it asks for is drawn, so a request that saw it set is
        # always covered by a frame drawn after it.
        self._frame_posted = False
        # Keys the reader could not post because the queue was full (see
        # `post_key`), and how many of them `_report_dropped_keys` has named.
        self._keys_dropped = 0
        self._keys_reported = 0
        # The last stderr line `_warn` wrote, so a stuck painter is named once
        # rather than once per caller that gave up on it.
        self._last_warning = ""
        # Painter-only from here down. The (rows, columns, lines) a refused
        # region last skipped a frame at, so the diagnostics name it once.
        self._skip_logged: Optional[Tuple[int, int, int]] = None
        # When the requested frame is due (a
        # monotonic time, None: none requested) — INPUT_REFRESH_SECONDS after
        # the first request of a burst, so the burst is one frame.
        self._frame_due: Optional[float] = None
        # Where the region is in its life (`_REGION_*`), moved by
        # `_open_region`, `_close_region` and `disable`.
        self._region = _REGION_UNOPENED
        self._next_tick = 0.0          # when the periodic frame is next due
        self._ticks = 0
        self._last_size = (0, 0)       # the terminal size the region was cut for
        # Rows the region is pinned at. Written only by `_reserve`, which only
        # the terminal's owner calls.
        self._reserved = 0

    # --- reads -------------------------------------------------------------

    @property
    def terminal(self) -> termio.Terminal:
        """The terminal the frames go to: a NullTerminal once `disable` ran.

        Read-only: swapping it is `disable`'s, on the terminal's owner. Any
        thread may read it (the app's `enabled`, the emergency restore), but
        a write through it from outside the owner lands beside a live frame.
        """
        return self._terminal

    @property
    def owns_current_thread(self) -> bool:
        return self._owner.owns_current_thread

    @property
    def thread(self) -> Optional[threading.Thread]:
        """The painter thread (a closing one included), None while there is none."""
        return self._owner.thread

    @property
    def backlog(self) -> int:
        """Posted calls not yet finished, the one running included."""
        return self._owner.backlog

    @property
    def frame_requested(self) -> bool:
        """True while a frame request is posted and not yet taken."""
        return self._frame_posted

    @property
    def keys_dropped(self) -> int:
        """Keys `post_key` found no room for, since this painter was built."""
        return self._keys_dropped

    # --- lifecycle ---------------------------------------------------------

    def open(self, timeout: float, *, on_refused: Callable[[], None]) -> None:
        """Pin the region and draw the first frame, on the painter; waits for
        it at most `timeout`, and names a wait that timed out on stderr.

        A painter an earlier `close()` gave up waiting for is still closing,
        and this reopens THAT one (the policy: `OwnerThread.start`): the region
        is pinned after its last frame and release, not beside it — and queued
        past the queue's bound, so a full one cannot hold this thread.
        `on_refused` runs on the painter when the terminal refuses the region,
        whenever that lands (see `_open_region`).
        """
        self._frame_posted = False
        self._owner.start(first=functools.partial(self._open_region, on_refused))
        if not self.drain(timeout):
            self._warn_stuck("the first frame", timeout,
                             wait=OPEN_WARNING_WAIT_SECONDS)

    def close(self, timeout: float) -> bool:
        """The last frame and the release, on the painter, then no painter;
        True once it has handed back within `timeout` (a timeout is named on
        stderr).

        The release is the owner's `final`, behind everything posted before it
        and past the queue's bound, so this wait keeps its bound however full
        the queue is. A painter stuck in a frame outlives the wait still owning
        the terminal (closing), and releases it itself once the write returns:
        nothing is released from here beside a live write.
        """
        if self._owner.close(timeout, final=self._close_region):
            return True
        self._warn_stuck("the last frame and the release", timeout)
        return False

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until every call posted before this one has run."""
        return self._owner.drain(timeout)

    # --- posting -----------------------------------------------------------

    def try_post(self, call: Callable, *args) -> bool:
        """`call(*args)` on the painter, never waiting; False when the queue
        is full (see `ownership.OwnerThread.try_post`)."""
        return self._owner.try_post(call, *args)

    def post_key(self, call: Callable, *args) -> None:
        """The key reader's post: never waits for room.

        The queue is full only while a terminal write is stuck, and a reader
        held there could not read the Ctrl+C that Windows delivers as a key
        (`TerminalInput._emit`). So a key that finds it full is dropped and
        counted, and `_report_dropped_keys` names the count once the painter
        is back.
        """
        if not self._owner.try_post(call, *args):
            self._keys_dropped += 1
            diaglog.record("key dropped, painter queue full",
                           f"{self._keys_dropped} so far, backlog "
                           f"{self._owner.backlog}")
        else:
            diaglog.trace("key posted", f"backlog {self._owner.backlog}")

    def run_and_wait(self, call: Callable, *args) -> None:
        """`call(*args)` on the thread that owns the terminal, waited for.

        On the painter itself, and with no painter running, that is here and
        now. From any other thread it is posted, and the caller — which reads
        its effect next — waits for it, for at most POSTED_CALL_WAIT_SECONDS
        in all: queueing it included, so a full queue behind a stuck write
        costs the call (named on stderr), never an unbounded wait.
        """
        if self._owner.owns_current_thread:
            call(*args)
            return
        deadline = time.monotonic() + POSTED_CALL_WAIT_SECONDS
        if not self._owner.post(call, *args,
                                timeout=POSTED_CALL_WAIT_SECONDS):
            self._warn_stuck(f"{_call_name(call)} (not queued, dropped)",
                             POSTED_CALL_WAIT_SECONDS)
            return
        if not self.drain(max(0.0, deadline - time.monotonic())):
            self._warn_stuck(_call_name(call), POSTED_CALL_WAIT_SECONDS)

    def request_frame(self) -> None:
        """Ask for a frame of the state as it is now — drawn on the painter
        while one runs.

        While the painter runs it is the one thread that renders and writes the
        terminal; a worker's `update`, a key, a quota poll only post a request
        and return, so the renderers never run on two threads at once and no
        caller waits for a slow terminal. On the painter itself the request is
        folded into the frame it draws once the calls queued behind it have run
        (see `_on_idle`). With no painter running the caller draws itself.
        """
        if self._owner.owns_current_thread:
            self._schedule_frame()
            return
        # A set flag with nothing queued is a request that never got there —
        # this thread's twin interrupted between flag and post, or a queue the
        # owner dropped dying — so it is posted again rather than trusted. A
        # second request racing the first is harmless: both are one frame.
        if self._frame_posted and self._owner.backlog:
            return
        self._frame_posted = True
        try:
            queued = self._owner.try_post(self._take_frame_request)
        except BaseException:
            self._frame_posted = False
            raise
        if not queued:
            # Queue full: the periodic frame draws this state instead.
            self._frame_posted = False

    # --- the terminal's owner only (the painter, or the caller with none) --

    # The public three check the caller (`_check_owner`); the painter's own
    # paths call the unchecked `_reserve` / `_resize` / `_disable`, so a frame
    # drawn inline while a painter is just starting fails no worse than before.

    def _check_owner(self, method: str) -> None:
        """Raise unless this thread owns the terminal: the painter while one
        runs (a closing one included), any caller while there is none.

        `StatusApp.painter` is public, so `app.painter.disable()` from a worker
        would otherwise skip `run_and_wait` and write next to a live frame.
        A RuntimeError rather than an `assert`, which `python -O` strips.
        """
        if self._owner.thread is None or self._owner.owns_current_thread:
            return
        raise RuntimeError(
            f"Painter.{method}() called on {threading.current_thread().name!r} "
            f"while {PAINTER_THREAD_NAME} owns the terminal; hand it over with "
            f"run_and_wait")

    def reserve(self, rows: int) -> bool:
        """True when the region has the requested shape (or there is none to keep)."""
        self._check_owner("reserve")
        return self._reserve(rows)

    def resize(self) -> None:
        """Re-cut the region for the terminal's size now, and repaint it."""
        self._check_owner("resize")
        self._resize()

    def disable(self) -> None:
        """Release the region and swap in a NullTerminal, for good."""
        self._check_owner("disable")
        self._disable()

    def _reserve(self, rows: int) -> bool:
        if isinstance(self.terminal, termio.NullTerminal):
            return True
        if self.terminal.reserve(rows):
            self._reserved = rows
            return True
        return False

    def _resize(self) -> None:
        if self._region == _REGION_CLOSED:
            return      # released: nothing to re-cut (see _REGION_*)
        # A refused reserve() leaves the OLD geometry in place, so carrying on
        # would paint absolute rows outside the new screen with the region set
        # for the old one. Disable instead — `_open_region` answers the same
        # refusal the same way.
        if not self._reserve(self._row_count()):
            self._disable()
            return
        self.request_frame()

    def _disable(self) -> None:
        if not isinstance(self._terminal, termio.NullTerminal):
            diaglog.record("Painter disabled the terminal for good, the status "
                           "line stops repainting")
        self._region = _REGION_CLOSED   # nothing left for the idle hook to tick
        try:
            self._terminal.release()
        except Exception:
            pass
        self._terminal = termio.NullTerminal()

    # --- frames --------------------------------------------------------------

    def _take_frame_request(self) -> None:
        """A posted frame request, or with no painter the frame itself. The
        state calls queued before it run first (see `StateOwner.post`)."""
        self._frame_posted = False
        if self._owner.owns_current_thread:
            self.state.apply()
            self._schedule_frame()
        else:
            self._draw()

    def _schedule_frame(self) -> None:
        """(painter) A burst's first request sets its frame's time; the rest of
        the burst lands in that frame."""
        if self._frame_due is None:
            self._frame_due = time.monotonic() + INPUT_REFRESH_SECONDS

    def _draw(self, *, reassert: bool = False) -> None:
        """One frame: the title, then the rows. On the painter, or on the
        caller when there is none (see `request_frame`). None once the region
        is closed (see `_REGION_CLOSED`). The state calls still queued run
        first, a closed region included: the data outlives the region."""
        self.state.apply()
        self._frame_due = None
        if self._region == _REGION_CLOSED:
            return
        # Before the `active` gate, and on the same path as the rows: the title
        # is what a person sees while the terminal is behind another window, so
        # it must follow every state change the rows follow — including the ones
        # that arrive before any region is pinned (a run names its window as
        # soon as `update` gives it a state, which is before `start()`).
        # Once the terminal has been released it refuses the write, so a late
        # painter cannot re-name a window nobody will clean up again.
        try:
            self.terminal.set_title(self._title(), reassert=reassert)
        except Exception:
            pass
        if not self.terminal.active:
            return
        try:
            started = time.monotonic()
            columns, _lines = self.terminal.size()
            # The bottom-right cell can immediately scroll a Windows console.
            # Budget the margin BEFORE windowing the editor, or its caret is cut.
            rows = self._render(max(0, columns - textwidth.LINE_RIGHT_MARGIN))
            if len(rows) != self._reserved:
                # A Mode added or dropped a row: resize the region rather than
                # painting into lines the terminal is still scrolling.
                if not self._reserve(len(rows)):
                    # Every frame skips while the refusal lasts: named once
                    # per shape, not once per tick.
                    shape = (len(rows), columns, _lines)
                    if shape != self._skip_logged:
                        self._skip_logged = shape
                        diaglog.record("Painter frames skipped, region refused",
                                       f"{len(rows)} rows, {columns}x{_lines}")
                    return
            self._skip_logged = None
            painted = self.terminal.paint([self._colorize(line) for line in rows],
                                          reassert=reassert)
            if not painted:
                diaglog.trace("frame NOT painted, the terminal refused it")
            elif diaglog.keytrace_enabled():
                diaglog.trace("frame painted",
                              f"{(time.monotonic() - started) * 1000:.1f} ms, "
                              f"{columns} columns, last row {rows[-1]!r}"
                              if rows else "no rows")
        except Exception as exc:
            diaglog.record("Painter._draw raised", exc=exc)
            self._disable()

    # --- the painter's own calls -------------------------------------------

    def _open_region(self, on_refused: Callable[[], None]) -> None:
        """(painter) Pin the region and draw the first frame — `open()`'s
        `first`, run after whatever an earlier close left queued.

        A refusal disables the terminal and calls `on_refused`: the app stops
        a key reader it may already have started while this waited behind a
        stuck write — a refused region means no keys, whenever it lands.
        """
        try:
            self._last_size = self.terminal.size()
            self._ticks = 0
            self._next_tick = time.monotonic() + self.refresh
            self._frame_due = None
            self._region = _REGION_OPEN
            if not self._reserve(self._row_count()):
                self._disable()
                on_refused()
                return
            self._draw()
        except Exception as exc:
            diaglog.record("Painter._open_region raised", exc=exc)
            self._disable()

    def _close_region(self) -> None:
        """(painter) The last frame, then the release — `close()`'s `final`.

        It runs behind everything posted before close(); a call posted after
        it (a `run_and_wait` from a thread close() did not wait for) still
        runs, after the release, and finds the region closed (`_REGION_CLOSED`).

        The frame is drawn whether or not one is due: the state a caller set
        right before close() — the run's final `phase="idle"` — is still inside
        its burst window, and would otherwise never reach the screen.
        """
        try:
            # Keys lost behind the stuck write close() gave up on: no tick is
            # left to name them.
            self._report_dropped_keys()
            self._draw()
        finally:
            self._region = _REGION_CLOSED
            try:
                self.terminal.release()
            except Exception:
                pass

    def _on_idle(self) -> Optional[float]:
        """(painter) The `idle` hook: a requested frame once its burst window
        has passed, the periodic one every `refresh`; answers when it is due
        next. Nothing outside an open region (see `_REGION_OPEN`): a
        reopened painter's first idle pass, before `_open_region` has run,
        must not tick — and re-pin — the region the last close released.

        State calls a full queue kept their frame request from (`StateOwner.post`)
        run here, once the queue has drained, rather than a `refresh` later.
        """
        if self.state.pending:
            self.state.apply()
            self._schedule_frame()
        if self._region != _REGION_OPEN:
            return None
        try:
            now = time.monotonic()
            if now >= self._next_tick:
                self._next_tick = now + self.refresh
                self._tick()
            elif self._frame_due is not None and now >= self._frame_due:
                self._draw()
        except Exception as exc:
            diaglog.record("Painter._on_idle raised", exc=exc)
            self._disable()
        due = self._next_tick
        if self._frame_due is not None:
            due = min(due, self._frame_due)
        return max(0.0, due - time.monotonic())

    def _tick(self) -> None:
        """(painter) The periodic frame and the checks that ride on it."""
        # The app's checks read the data: state calls a full queue left
        # behind (see `StateOwner.post`) run before them, not only before the frame.
        self.state.apply()
        self._ticks += 1
        size = self.terminal.size()
        if size != self._last_size:
            self._last_size = size
            try:
                self._resize()
            except Exception as exc:
                diaglog.record("Painter resize on tick raised", exc=exc)
                self._disable()
        self._report_dropped_keys()
        if self._ticks % CONSOLE_SNAPSHOT_TICKS == 0 and diaglog.keytrace_enabled():
            diaglog.trace("console", diaglog.console_snapshot())
        self._app_tick(self._ticks)
        # Re-assert the region on the periodic repaint: see
        # termio.Terminal.paint.
        self._draw(reassert=True)

    def _report_dropped_keys(self) -> None:
        """(painter) Say so once keys were lost to a full queue (see
        `post_key`): on the note row, and on stderr for the log."""
        dropped = self._keys_dropped
        if dropped == self._keys_reported:
            return
        self._keys_reported = dropped
        text = (f"{dropped} key(s) dropped so far — the terminal was not "
                f"keeping up; press them again")
        self._warn(text)
        self._note(text)

    def _warn_stuck(self, what: str, seconds: float, *,
                    wait: Optional[float] = None) -> None:
        """Name a wait on the painter that timed out: a stuck painter is
        otherwise invisible, since its own frames are what is stuck. `wait`
        as for `_warn`."""
        self._warn(f"{what} not done within {seconds:g} s — a terminal write "
                   f"is stuck; the painter carries on once it returns",
                   wait=wait)

    def _warn(self, text: str, *, wait: Optional[float] = None) -> None:
        """One stderr line, not repeated back to back.

        With `wait`, the line is written on a thread of its own and the caller
        waits for it at most that long (see OPEN_WARNING_WAIT_SECONDS).
        """
        line = f"{PAINTER_THREAD_NAME}: {text}"
        if line == self._last_warning:
            return
        self._last_warning = line
        diaglog.record("Painter warning", text)
        if wait is None:
            _print_stderr(line)
            return
        writer = threading.Thread(target=_stderr_writer(line),
                                  name=f"{PAINTER_THREAD_NAME}-warning",
                                  daemon=True)
        writer.start()
        writer.join(wait)


def _print_stderr(line: str) -> None:
    try:
        print(line, file=sys.stderr)
    except Exception:
        pass


def _stderr_writer(line: str) -> Callable[[], None]:
    """(caller) A write of `line` for a daemon thread that may outlive the
    process's main thread.

    A daemon stuck in `print` on a frozen stderr holds the `BufferedWriter`
    lock, and the interpreter's shutdown then waits for that lock about a
    second and aborts with "Fatal Python error: _enter_buffered_busy"
    (exit 0xC0000409), losing the line anyway. So the text layer is flushed
    here, on the caller, and the daemon writes the encoded bytes to the raw
    stream underneath, which takes no Python-level lock: a write still stuck
    at exit dies with the process, and the line is lost without a crash.
    A stderr without a raw stream (a test's stand-in) is printed to as is.
    """
    stream = sys.stderr
    raw = getattr(getattr(stream, "buffer", None), "raw", None)
    if raw is None:
        return lambda: _print_stderr(line)
    try:
        stream.flush()
        data = (line + os.linesep).encode(stream.encoding or "utf-8",
                                          stream.errors or "backslashreplace")
    except Exception:
        return lambda: None

    def write() -> None:
        view = memoryview(data)
        try:
            while view:
                written = raw.write(view)
                if not written:
                    return
                view = view[written:]
        except Exception:
            pass
    return write


