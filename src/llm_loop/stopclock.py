"""The deadline of the ending in progress, for the waits that cannot be handed it.

`runlifecycle.StopBudget` is one budget per ending, and every wait of that
ending must take what is LEFT of it. Most of those waits sit several calls
below the ending — the quota server's lock and EOF wait inside
`CodexUsageSource.close`, the HTTP read behind `LimitPolicy.log_snapshot` —
and their callers' signatures are a protocol a dozen test fakes implement. So the ending publishes its deadline here, for the thread
that runs it, and each wait caps its own limit with `within`. A wait outside
any ending — a turn's reap, a status-line read — sees no deadline and keeps
its limit.

Thread-local by construction (`contextvars`: a new thread starts with no
deadline): the ending's thread is the one whose waits make the stop long, and
a worker or a refresher that outlives the ending is not waited for by it.

A leaf on purpose — standard library only — so `usage` and `codex_usage`
reach it without importing the run's lifecycle.

An HTTP read, whose timeout restarts per socket operation and skips the DNS
lookup, cannot be cut short in place: it runs whole on a daemon worker the
ending joins for what is left (`usage.UsageSource.query_usage_json`).

What a deadline here does NOT bound:

  * ending a child's process tree (`procend.end_process_tree`): up to the
    grace twice plus `taskkill`, 6 s per tree past the budget at worst. A
    hand-off that does not wait kills by numeric pid, which can hit a reused
    pid; the safe fix is a process lifetime the OS owns, not built yet (see
    there);
  * a print into a console that has stopped draining (a selection freezing a
    Windows console, a pipe nobody reads). The write blocks the very thread
    that would ask `within`, and the stream has no timeout to give it; a
    thread to print on only moves the hang to the interpreter's exit flush.
    The run's console owner bounds what IT writes (`parallel._close_console`),
    and nothing bounds the ending's own lines.
"""

import contextlib
import contextvars
import time
from typing import Iterator, Optional

_DEADLINE: "contextvars.ContextVar[Optional[float]]" = contextvars.ContextVar(
    "llm_loop_ending_deadline", default=None)


@contextlib.contextmanager
def ending(deadline: Optional[float]) -> Iterator[None]:
    """Publish `deadline` (`time.monotonic()` seconds, None for none) as this
    thread's ending deadline for the block. An inner ending may only shorten
    an outer one: a budget once announced is never extended under it."""
    outer = _DEADLINE.get()
    if outer is not None and (deadline is None or deadline > outer):
        deadline = outer
    token = _DEADLINE.set(deadline)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def deadline() -> Optional[float]:
    """This thread's ending deadline (`time.monotonic()`), None without one —
    for a step deferred past the ending that must keep its bound."""
    return _DEADLINE.get()


def left() -> Optional[float]:
    """Seconds left to this thread's ending, 0.0 once spent, None without one."""
    deadline = _DEADLINE.get()
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def within(limit: Optional[float]) -> Optional[float]:
    """`limit` capped by what is left of the ending; None only when both are.

    A wait's own limit stays its limit outside an ending; inside one, the
    smaller of the two. 0.0 means "do not wait": the caller still does what
    needs no waiting (a kill, a flag), and skips what does.
    """
    remaining = left()
    if remaining is None:
        return limit
    if limit is None:
        return remaining
    return min(limit, remaining)


def spent() -> bool:
    """True once this thread's ending deadline has passed."""
    remaining = left()
    return remaining is not None and remaining <= 0
