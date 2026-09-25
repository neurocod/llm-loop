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

import queue
import sys
import threading
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


class OwnerThread:
    """The one thread that runs every call posted to it, in the order posted.

    A posted call that raises costs that call only: the exception is reported
    (see `_report`) and the owner goes on with the next one, so a resource that
    fails — a closed pipe under the console — can no longer unwind the thread
    that merely asked for the write.

    Before `start()` and after `close()` there is no owner, and `post` runs the
    call on the caller's thread, exception included: the resource has been
    handed back to whoever is calling, which is what it was before an owner
    existed.
    """

    def __init__(self, name: str, *, maxsize: int = DEFAULT_MAXSIZE,
                 on_error: Optional[Callable[[BaseException], None]] = None):
        self.name = name
        self._maxsize = maxsize
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        # Guards only the open/closed decision against `post`: a post that saw
        # the owner open must land in the queue before `close` enqueues its
        # stop, or it would sit behind the stop with nobody to run it.
        self._state_lock = threading.Lock()
        self._open = False
        self._thread: Optional[threading.Thread] = None
        self._on_error = on_error or self._report
        self._last_report = ""
        self.failures = 0

    @property
    def owns_current_thread(self) -> bool:
        thread = self._thread
        return thread is not None and threading.current_thread() is thread

    def start(self) -> "OwnerThread":
        """Open (again): a new thread over a new queue.

        New on every start, so a thread a timed-out `close` left finishing its
        backlog keeps its own queue and never becomes a second consumer of this
        one.
        """
        with self._state_lock:
            if self._open:
                return self
            self._queue = queue.Queue(maxsize=self._maxsize)
            # A new window reports its own failures, even one worded like the
            # last window's (see `_report`).
            self._last_report = ""
            self._thread = threading.Thread(target=self._run, args=(self._queue,),
                                            name=self.name, daemon=True)
            self._open = True
            self._thread.start()
        return self

    def post(self, call: Callable, *args) -> None:
        """Run `call(*args)` on the owner; blocks only while the queue is full."""
        with self._state_lock:
            if self._open:
                self._queue.put((call, args))
                return
        call(*args)     # no owner: the caller's call, and the caller's exception

    def drain(self, timeout: Optional[float] = None) -> bool:
        """Wait until every call posted before this one has run.

        True when it has; False on a timeout, or when asked from the owner
        itself (its own queue cannot empty while it waits on it).
        """
        if self.owns_current_thread:
            return False
        done = threading.Event()
        with self._state_lock:
            if not self._open:
                return True
            self._queue.put((done.set, ()))
        return done.wait(timeout)

    def close(self, timeout: Optional[float] = None) -> bool:
        """Run what was posted, then stop owning. True when the thread ended.

        Later posts run on their caller (see the class docstring). A timeout
        leaves the thread finishing its backlog as a daemon; what it has not
        written by the time the process exits is lost with it.
        """
        with self._state_lock:
            if not self._open:
                return True
            self._open = False
            self._queue.put(None)
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def _run(self, own_queue: "queue.Queue") -> None:
        while True:
            item = own_queue.get()
            if item is None:
                return
            call, args = item
            self._invoke(call, args)

    def _invoke(self, call: Callable, args: tuple) -> None:
        try:
            call(*args)
        except Exception as exc:
            self.failures += 1
            try:
                self._on_error(exc)
            except Exception:
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
