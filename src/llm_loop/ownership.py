"""ownership.py - one thread owns a shared resource; everybody else posts to it.

The rule this module exists for: a resource several threads use gets exactly one
thread that touches it, and the others hand it work through a queue instead of
taking a lock around it. A lock says "whoever gets here first, and hope every
path remembered to ask"; an owner says "only this thread, ever", which is a fact
a reader can check by looking at one loop. It is the Qt threads + signals/slots
model without Qt: the value is not the syntax but that there is no shared
mutable state left to guard.

`OwnerThread` is that owner and nothing more — FIFO order, a bounded queue,
`drain` to wait for what was posted so far, `close` to hand the resource back.
What the resource IS lives with its user (`parallel` owns the console's worker
lines with one of these).
"""

import collections
import sys
import threading
import time
from typing import Callable, Optional

__all__ = ["OwnerThread", "DEFAULT_MAXSIZE"]

# How many posted calls may wait before `post` blocks. Not a throughput knob: it
# is what keeps a stuck resource (a console the operator froze by selecting
# text, a pipe nobody reads) from turning into unbounded memory. Below the cap a
# poster never waits for the resource — which is the point of having an owner —
# and at the cap it waits exactly as it did when it took a lock around the
# write, so the worst case is the old behaviour, not a new one. A worker line is
# a few hundred bytes, so the cap costs a few MB at most.
DEFAULT_MAXSIZE = 10_000

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
    """

    def __init__(self, name: str, *, maxsize: int = DEFAULT_MAXSIZE,
                 on_error: Optional[Callable[[BaseException], None]] = None):
        self.name = name
        self._maxsize = maxsize
        self._lock = threading.Lock()
        # One condition for every change a waiter can be waiting for: room in
        # the queue, a call finished, the state moved.
        self._changed = threading.Condition(self._lock)
        self._items: collections.deque = collections.deque()
        self._state = _CLOSED
        self._thread: Optional[threading.Thread] = None
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
    def owns_current_thread(self) -> bool:
        thread = self._thread
        return thread is not None and threading.current_thread() is thread

    def start(self) -> "OwnerThread":
        """Open: a new owner thread, or the one still closing, kept on.

        A `close` that timed out leaves its thread running the backlog; opening
        again keeps THAT thread as the owner instead of starting a second one
        beside it over the same resource.
        """
        with self._changed:
            if self._state == _OPEN:
                return self
            if self._state == _CLOSING:
                self._state = _OPEN
                self._changed.notify_all()
                return self
            # A new window reports its own failures, even one worded like the
            # last window's (see `_report`).
            self._last_report = ""
            self._thread = threading.Thread(target=self._run, name=self.name,
                                            daemon=True)
            self._state = _OPEN
            self._thread.start()
        return self

    def post(self, call: Callable, *args) -> None:
        """Run `call(*args)` on the owner; blocks only while the queue is full.

        On the owner itself the call runs at once, inline: the owner cannot
        wait for room in a queue only it empties.
        """
        if self.owns_current_thread:
            call(*args)
            return
        with self._changed:
            while self._state != _CLOSED and len(self._items) >= self._maxsize:
                self._changed.wait()
            if self._state != _CLOSED:
                self._enqueue(call, args)
                return
        call(*args)     # no owner: the caller's call, and the caller's exception

    def try_post(self, call: Callable, *args) -> bool:
        """`post` that never waits: False, with nothing run, when the queue is full.

        For a caller that must not be held by the resource at all (an
        interrupt handler); what to do with a refused call is its decision.
        """
        if self.owns_current_thread:
            call(*args)
            return True
        with self._changed:
            if self._state != _CLOSED:
                if len(self._items) >= self._maxsize:
                    return False
                self._enqueue(call, args)
                return True
        call(*args)
        return True

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

    def close(self, timeout: Optional[float] = None) -> bool:
        """Run what was posted, then stop owning. True once the thread has ended.

        Bounded by `timeout` as a whole. On a timeout the thread keeps owning
        the resource (CLOSING) while it finishes the backlog as a daemon; what
        it has not written by the time the process exits is lost with it. Asked
        again, `close` answers for the same thread — True only once it is gone.
        """
        deadline = _deadline(timeout)
        if not self._lock.acquire(timeout=_remaining(deadline)):
            return False
        try:
            thread = self._thread
            if self._state == _OPEN:
                self._state = _CLOSING
                self._changed.notify_all()
            if thread is None:
                return True
            if thread is threading.current_thread():
                return False
            self._changed.wait_for(lambda: self._state == _CLOSED,
                                   _remaining_or_none(deadline))
            if self._state != _CLOSED:
                return False
        finally:
            self._lock.release()
        # CLOSED is set by the thread's last locked step; the join is for the
        # few instructions after it.
        thread.join(_remaining_or_none(deadline))
        return not thread.is_alive()

    def _enqueue(self, call: Callable, args: tuple) -> None:
        """Queue one call (caller holds the lock)."""
        self._items.append((call, args))
        self._queued += 1
        self._changed.notify_all()

    def _run(self) -> None:
        try:
            while True:
                with self._changed:
                    while not self._items and self._state == _OPEN:
                        self._changed.wait()
                    if not self._items:
                        # Closing and nothing left: hand the resource back in
                        # the same locked step that saw the queue empty, so no
                        # post can land between the look and the hand-back.
                        self._close_locked()
                        return
                    call, args = self._items.popleft()
                    self._changed.notify_all()      # room for a waiting poster
                try:
                    self._invoke(call, args)
                finally:
                    with self._changed:
                        self._finished += 1
                        self._changed.notify_all()
        finally:
            # Only reachable with the state still open if something got past
            # `_invoke` — the owner must never die silently with posters left
            # queueing into it.
            with self._changed:
                if self._state != _CLOSED:
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

    def _invoke(self, call: Callable, args: tuple) -> None:
        try:
            call(*args)
        except BaseException as exc:    # noqa: BLE001 - see the class docstring
            self.failures += 1
            try:
                self._on_error(exc)
            except BaseException:       # noqa: BLE001 - the reporter may not end us
                pass

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
