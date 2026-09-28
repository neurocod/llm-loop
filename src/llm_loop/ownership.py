"""ownership.py - one thread owns a shared resource; everybody else posts to it.

The rule this module exists for: a resource several threads use gets exactly one
thread that touches it, and the others hand it work through a queue instead of
taking a lock around it. A lock says "whoever gets here first, and hope every
path remembered to ask"; an owner says "only this thread, ever", which is a fact
a reader can check by looking at one loop. It is the Qt threads + signals/slots
model without Qt: the value is not the syntax but that there is no shared
mutable state left to guard.

`OwnerThread` is that owner and nothing more — FIFO order, a bounded queue,
`drain` to wait for what was posted so far, `close` to hand the resource back,
`start(first=)` / `close(final=)` for the calls that set the resource up and
put it back, and an `idle` hook for the work an owner does on its own clock.
What the resource IS lives with its user: `parallel` owns the console's worker
lines with one of these, `statusline.Painter` the pinned rows.
"""

import collections
import sys
import threading
import time
from typing import Callable, Optional

__all__ = ["OwnerThread", "DEFAULT_MAXSIZE", "IDLE_DELAY_MAX"]

# How many posted calls may wait before `post` blocks. Not a throughput knob: it
# is what keeps a stuck resource (a console the operator froze by selecting
# text, a pipe nobody reads) from turning into unbounded memory. Below the cap a
# poster never waits for the resource — which is the point of having an owner —
# and at the cap it waits exactly as it did when it took a lock around the
# write, so the worst case is the old behaviour, not a new one. A worker line is
# a few hundred bytes, so the cap costs a few MB at most.
DEFAULT_MAXSIZE = 10_000

# The longest delay an `idle` answer is taken at. A clamp, not a policy: a hook
# answering `inf` (or a year) means "not for a long while", and `Condition.wait`
# refuses a timeout past `threading.TIMEOUT_MAX` with an OverflowError that
# would end the owner. Being asked again after an hour costs a well-behaved hook
# nothing — it answers its own delay again.
IDLE_DELAY_MAX = 3600.0

# The owner's three states. CLOSING is the one that matters: `close` has been
# asked, but the thread is still running what was posted, so it still OWNS the
# resource — a post made now joins its queue rather than running beside it.
_OPEN, _CLOSING, _CLOSED = "open", "closing", "closed"


class OwnerThread:
    """The one thread that runs every call posted to it, in the order posted.

    A posted call that raises — anything, `SystemExit` included — costs that
    call only: the exception is reported (see `_report`) and the owner goes on
    with the next one, so a resource that fails can neither unwind the thread
    that merely asked for the write nor end the owner and leave the posters
    queueing into a thread nobody runs.

    Before `start()` and once the owner has exited there is no owner, and
    `post` runs the call on the caller's thread, exception included: the
    resource has been handed back to whoever is calling, which is what it was
    before an owner existed. Between `close()` and the owner's exit the thread
    still owns it (see `_CLOSING`).

    Every wait here is on a `threading.Condition` that is released while
    waiting, and nothing holds its lock across a call or a blocking put — which
    is what lets `close(timeout)` and `drain(timeout)` keep their bound however
    stuck the resource is.

    The bounded queue holds back posters, never the owner's own life cycle:
    `start(first=)` and `close(final=)` queue the call that sets the resource
    up and the one that puts it back past `maxsize`, so neither a restart nor
    a hand-back waits for room behind a stuck resource. The bound is on posts,
    then: every such call comes on top of it, repeated ones are not merged,
    and a caller that opens and closes over a stuck resource in a loop grows
    the queue by its own calls.

    `idle` is the owner's own work — a periodic repaint, a frame coalesced over
    a burst of posted calls. The owner calls it when its queue is empty and it
    is due, and it returns in how many seconds it is due again (clamped to
    IDLE_DELAY_MAX), or None for "not until something is posted". It is due
    after every call posted in the current window (the call may have made work
    for it — asked for a frame) and once the delay it last returned has passed;
    never before the window's first post — an owner reopened while closing
    included, whose last window's backlog does not count — since what it works
    on is usually set up by one. And it is never STARTED once the owner has seen
    `close()`: a closing owner finishes the backlog and hands back. An idle pass
    the owner had already begun when `close()` was asked finishes first, ahead
    of the backlog, so it still cannot land after anything the backlog (a
    `final` included) does. It fails like a posted call does — reported, then
    due again only after the next post — and so does an answer that is not a
    delay; it is not counted by `drain`, which waits for posted calls only.
    """

    def __init__(self, name: str, *, maxsize: int = DEFAULT_MAXSIZE,
                 on_error: Optional[Callable[[BaseException], None]] = None,
                 idle: Optional[Callable[[], Optional[float]]] = None):
        self.name = name
        self._maxsize = maxsize
        self._idle = idle
        self._lock = threading.Lock()
        # One condition for every change a waiter can be waiting for: room in
        # the queue, a call finished, the state moved.
        self._changed = threading.Condition(self._lock)
        self._items: collections.deque = collections.deque()
        self._state = _CLOSED
        self._thread: Optional[threading.Thread] = None
        # Bumped by every `start()` that opens (a reopen included). A queued
        # call carries the window it was posted in, and only the current
        # window's calls make `idle` due (see the class docstring).
        self._window = 0
        # Calls ever queued / ever finished (run or given up on). `drain` waits
        # for the second to reach what the first was when it was asked.
        self._queued = 0
        self._finished = 0
        self._on_error = on_error or self._report
        self._last_report = ""
        self.failures = 0

    @property
    def backlog(self) -> int:
        """Calls posted and not yet finished, the one running included."""
        with self._changed:
            return self._queued - self._finished

    @property
    def thread(self) -> Optional[threading.Thread]:
        """The owner thread (a closing one included), None while there is none."""
        return self._thread

    @property
    def owns_current_thread(self) -> bool:
        thread = self._thread
        return thread is not None and threading.current_thread() is thread

    def start(self, first: Optional[Callable[[], object]] = None
              ) -> "OwnerThread":
        """Open: a new owner thread, or the one still closing, kept on.

        `first` — the call that sets the resource up for this window — is
        queued behind whatever an earlier window left and past `maxsize` (see
        the class docstring), so opening never waits for room. Queued on an
        owner that is already open too, as the next call.

        The restart policy, and the only one in this package (the status line's
        painter is one of these too): a `close` that timed out leaves its
        thread running the backlog, and opening again keeps THAT thread as the
        owner instead of starting a second one beside it. A second thread would
        be two writers over one resource for as long as the first is stuck —
        its stuck write landing after the new owner's first ones, its hand-back
        undoing what the new owner set up. Kept on, the old window's last calls
        and the new window's first run in the order they were posted. What it
        costs is that the new window's first call waits for the stuck one; any
        second writer would wait there too, on whatever lock the resource
        itself holds across the write, and could only get ahead of it where the
        resource has no such lock — which is the corruption, not a remedy.
        """
        with self._changed:
            if self._state == _CLOSING:
                self._state = _OPEN
                self._window += 1
                self._changed.notify_all()
            elif self._state == _CLOSED:
                # A new window reports its own failures, even one worded like
                # the last window's (see `_report`).
                self._last_report = ""
                self._window += 1
                self._thread = threading.Thread(target=self._run,
                                                name=self.name, daemon=True)
                self._state = _OPEN
                self._thread.start()
            if first is not None:
                self._enqueue(first, ())
        return self

    def post(self, call: Callable, *args,
             timeout: Optional[float] = None) -> bool:
        """Run `call(*args)` on the owner; waits only while the queue is full.

        True once the call is queued (or has run: with no owner, and on the
        owner itself, where it runs at once, inline — the owner cannot wait for
        room in a queue only it empties). With a `timeout`, False when the
        queue had no room for that long, and the call is then NOT queued: for
        a caller that must go on whether or not the resource ever comes back.
        """
        if self.owns_current_thread:
            call(*args)
            return True
        deadline = _deadline(timeout)
        with self._changed:
            while self._state != _CLOSED and len(self._items) >= self._maxsize:
                left = _remaining_or_none(deadline)
                if left is not None and left <= 0:
                    return False
                self._changed.wait(left)
            if self._state != _CLOSED:
                self._enqueue(call, args)
                return True
        call(*args)     # no owner: the caller's call, and the caller's exception
        return True

    def try_post(self, call: Callable, *args) -> bool:
        """`post` that never waits: False, with nothing run, when the queue is full.

        For a caller that must not be held by the resource at all (an
        interrupt handler); what to do with a refused call is its decision.
        """
        return self.post(call, *args, timeout=0)

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until every call posted before this one has run.

        True when it has; False on a timeout, or when asked from the owner
        itself (its own queue cannot empty while it waits on it). The whole
        wait, lock included, keeps `timeout`.
        """
        if self.owns_current_thread:
            return False
        deadline = _deadline(timeout)
        if not self._lock.acquire(timeout=_remaining(deadline)):
            return False
        try:
            if self._state == _CLOSED:
                return not self._items
            target = self._queued
            self._changed.wait_for(
                lambda: self._finished >= target or self._state == _CLOSED,
                _remaining_or_none(deadline))
            return self._finished >= target
        finally:
            self._lock.release()

    def close(self, timeout: Optional[float] = None, *,
              final: Optional[Callable[[], object]] = None) -> bool:
        """Run what was posted, then stop owning. True once the thread has ended.

        Bounded by `timeout` as a whole. On a timeout the thread keeps owning
        the resource (CLOSING) while it finishes the backlog as a daemon; what
        it has not written by the time the process exits is lost with it. Asked
        again, `close` answers for the same thread — True only once it is gone.
        Asked from the owner itself it only marks the close and returns False:
        the thread cannot wait for its own exit.

        `final` — the call that puts the resource back — is queued behind the
        backlog and past `maxsize` (see the class docstring), so the bound
        holds however full the queue is and the hand-back still happens, on
        the owner, whenever the resource returns. A post made while closing
        still joins the queue and so runs after it: `final`'s own resource
        must refuse what comes after it. With no owner, `final` runs here, on
        the caller, exception included, as a `post` would.
        """
        deadline = _deadline(timeout)
        if final is not None:
            # Unbounded on purpose: losing `final` to a moment's contention
            # would leave the resource set up for good, and no holder of this
            # lock keeps it across a call or a wait, so the wait is a few
            # instructions long.
            self._lock.acquire()
        elif not self._lock.acquire(timeout=_remaining(deadline)):
            return False
        try:
            thread = self._thread
            if thread is not None and final is not None:
                self._enqueue(final, ())
            if self._state == _OPEN:
                self._state = _CLOSING
                self._changed.notify_all()
            if thread is not None:
                if thread is threading.current_thread():
                    return False
                self._changed.wait_for(lambda: self._state == _CLOSED,
                                       _remaining_or_none(deadline))
                if self._state != _CLOSED:
                    return False
        finally:
            self._lock.release()
        if thread is None:
            if final is not None:
                final()
            return True
        # CLOSED is set by the thread's last locked step; the join is for the
        # few instructions after it.
        thread.join(_remaining_or_none(deadline))
        return not thread.is_alive()

    def _enqueue(self, call: Callable, args: tuple) -> None:
        """Queue one call, tagged with its window (caller holds the lock)."""
        self._items.append((call, args, self._window))
        self._queued += 1
        self._changed.notify_all()

    def _run(self) -> None:
        # When `idle` is next due (monotonic), or None: after the next post.
        idle_due: Optional[float] = None
        window = None       # the window `idle_due` was worked out in
        try:
            while True:
                with self._changed:
                    while True:
                        # Asked again after every wake, not once per call: a
                        # close and a reopen can both land while this thread
                        # sleeps, and what the last window made due — a delay
                        # the hook answered — is not this one's.
                        if window != self._window:
                            window, idle_due = self._window, None
                        if self._items or self._state != _OPEN:
                            break
                        if idle_due is None:
                            self._changed.wait()
                            continue
                        left = idle_due - time.monotonic()
                        if left <= 0:
                            break
                        self._changed.wait(left)
                    if self._items:
                        call, args, posted_in = self._items.popleft()
                        self._changed.notify_all()  # room for a waiting poster
                    elif self._state == _OPEN:
                        call = None                 # nothing posted: idle is due
                    else:
                        # Closing and nothing left: hand the resource back in
                        # the same locked step that saw the queue empty, so no
                        # post can land between the look and the hand-back.
                        self._close_locked()
                        return
                if call is None:
                    idle_due = self._invoke(self._idle_pass, ())
                    continue
                try:
                    self._invoke(call, args)
                finally:
                    with self._changed:
                        self._finished += 1
                        self._changed.notify_all()
                        current = posted_in == self._window
                if self._idle is not None and current:
                    idle_due = time.monotonic()
        finally:
            # Only reachable still owning if something got past `_invoke` — the
            # owner must never die silently with posters left queueing into it.
            # Asked by identity, not by state: after a normal hand-back a
            # `start()` may already have opened a NEW owner, whose open state
            # this thread must not close.
            with self._changed:
                if self._thread is threading.current_thread():
                    self._close_locked()

    def _close_locked(self) -> None:
        """No owner from here: later posts run on their caller (lock held).

        Calls still queued can only be here if the owner is dying abnormally
        (something got past `_invoke`); they are dropped, and counted as
        finished so no `drain` — this window's or the next's — waits for them.
        """
        self._finished += len(self._items)
        self._items.clear()
        self._state = _CLOSED
        self._thread = None
        self._changed.notify_all()

    def _invoke(self, call: Callable, args: tuple):
        """`call(*args)`'s answer, or None once its failure is reported."""
        try:
            return call(*args)
        except BaseException as exc:    # noqa: BLE001 - see the class docstring
            self._fail(exc)
            return None

    def _fail(self, exc: BaseException) -> None:
        self.failures += 1
        try:
            self._on_error(exc)
        except BaseException:           # noqa: BLE001 - the reporter may not end us
            pass

    def _idle_pass(self) -> Optional[float]:
        """Run `idle`; when it is next due (monotonic), or None.

        Run through `_invoke` whole, answer included, so nothing the answer
        does can get past it and end the owner: an answer that is not a delay
        — a string, NaN — raises here and is reported as the hook's failure.
        A delay past IDLE_DELAY_MAX (`inf`, `10**1000`) is clamped to it
        BEFORE `float()`, which would overflow on an int that large.
        """
        delay = self._idle()
        if delay is None:
            return None
        if not isinstance(delay, (int, float)) or delay != delay:
            raise TypeError(f"idle answered {delay!r}, not a delay in "
                            f"seconds or None")
        return time.monotonic() + float(min(max(0, delay), IDLE_DELAY_MAX))

    def _report(self, exc: BaseException) -> None:
        """One stderr line per distinct failure, not one per failed call.

        A closed stdout fails every line after the first, and repeating the same
        complaint for each of them would bury whatever else stderr is saying —
        the same reasoning as `statusline.QuotaRefresher._report`. Written to
        stderr because the resource that failed is usually stdout.
        """
        text = f"{self.name}: {type(exc).__name__}: {exc}"
        if text == self._last_report:
            return
        self._last_report = text
        print(text, file=sys.stderr)


def _deadline(timeout: Optional[float]) -> Optional[float]:
    return None if timeout is None else time.monotonic() + max(0.0, timeout)


def _remaining_or_none(deadline: Optional[float]) -> Optional[float]:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _remaining(deadline: Optional[float]) -> float:
    """`Lock.acquire`'s spelling: -1 is "no bound"."""
    remaining = _remaining_or_none(deadline)
    return -1 if remaining is None else remaining
