"""ctrlc.py - the operator's Ctrl+C as an event a run polls, not an exception.

Python delivers Ctrl+C as `KeyboardInterrupt`, raised at whatever bytecode the
main thread happens to be running: inside a turn, a quota hold, a reason being
written to the exit record, the status region's teardown, the exit push. A run
that has to close itself down — push, snapshot the quotas, report the notes
nobody delivered — then needs an `except KeyboardInterrupt` around every step of
that closing too, and a second Ctrl+C inside each of THOSE. That matrix is what
four rounds of review on the run's ending kept finding new cells of
(llm-loop 282acc0, 2255e6f, ce11713).

So while a run is open (`captured`) SIGINT raises nothing. Its handler counts
the press on the run's `Interrupt`, and the run reads that object where it
already waits — the turn's stream, the quota hold, the pause, the wait for the
exit push — and decides there what a press means:

  * the FIRST press asks the run to stop: the turn in flight is ended (its CLI
    reaped), a wait returns, and the runner takes its interrupt ending (exit
    `EXIT_CODE`) with the same housekeeping as every other ending;
  * a press heard once an ending has begun (`Interrupt.since`) gives up what
    that ending waits for — the exit push, the workers' join — and nothing else:
    the snapshots and the notes are not waits, and still happen.

When a press is noticed: a wait polls in slices of a quarter second or less
(`stopchannel.STOP_RECHECK_SECONDS`, `runlifecycle.EXIT_PUSH_POLL_S`), and
`time.sleep` on the main thread is woken by the signal anyway. A turn notices
at the CLI's next output line when the press arrives as a SIGINT on Windows:
the main thread is then blocked in a pipe read that the console's Ctrl+C does
not interrupt, and Python runs a signal handler only between bytecodes — the
same latency `KeyboardInterrupt` had there. A press read as a key by the status
line (`deliver`) comes on the key reader's thread and runs the turn's
`on_press` hook at once, which ends the CLI and with it the read.
"""

import contextlib
import signal
import threading
from typing import Callable, Iterator, List, Optional

__all__ = ["EXIT_CODE", "Interrupt", "asked", "captured", "current", "deliver"]

# The exit code of a run the operator ended with Ctrl+C: 128 + SIGINT, what a
# shell reports for a process SIGINT killed. A constant so a wrapper telling an
# interrupt from a failure (a phase that "gave up") names it rather than 130.
EXIT_CODE = 130


class Interrupt:
    """The Ctrl+C presses one run has heard. Safe from any thread.

    `press` is called by the SIGINT handler (on the main thread, between two
    bytecodes) or by the status line's key reader (on its own thread), and only
    counts; it never raises into whatever the run was doing.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._presses = 0
        self._hooks: List[Callable[[], None]] = []

    def press(self) -> None:
        """One Ctrl+C: counted, then every `on_press` hook run once for it.

        A hook that raises costs its own effect only — a handler must not be
        the thing that unwinds the run it is telling to stop.
        """
        with self._lock:
            self._presses += 1
            hooks = list(self._hooks)
        for hook in hooks:
            try:
                hook()
            except Exception:       # noqa: BLE001 - see the docstring
                pass

    @property
    def presses(self) -> int:
        return self._presses

    @property
    def requested(self) -> bool:
        """Has the operator pressed Ctrl+C in this run at all?"""
        return self._presses > 0

    def since(self, mark: int) -> bool:
        """Was Ctrl+C pressed after `mark` (an earlier reading of `presses`)?

        How an ending tells "the press that caused me" from "a press telling me
        to stop waiting": it reads `presses` as it begins and asks this.
        """
        return self._presses > mark

    @contextlib.contextmanager
    def on_press(self, hook: Callable[[], None]) -> Iterator[None]:
        """Run `hook` on every press made while the block runs.

        Not run for a press made before the block: a caller that must act on one
        already heard asks `requested` first.
        """
        with self._lock:
            self._hooks.append(hook)
        try:
            yield
        finally:
            with self._lock:
                self._hooks.remove(hook)


class _Unheard(Interrupt):
    """What `current` answers outside a run: never pressed, whatever is asked."""

    def press(self) -> None:
        pass


_UNHEARD = _Unheard()
_active: Optional[Interrupt] = None


def current() -> Interrupt:
    """The open run's Interrupt, or one that is never pressed outside a run.

    Read through this function rather than handed down every call chain:
    `streamrender.run_agent_streaming`, `limits.LimitPolicy.check_and_wait` and
    `stopchannel.sleep_unless` are public, and an embedder calling them outside
    a run gets exactly what it got before — Python's own KeyboardInterrupt.
    """
    return _active if _active is not None else _UNHEARD


def asked(should_stop: Optional[Callable[[], bool]] = None) -> bool:
    """Ctrl+C, or the caller's own stop channels: is a wait to end now?

    The one question every wait of a run asks, so a hold cannot listen to `s`
    and the stop file and still sit out a Ctrl+C.
    """
    return current().requested or (should_stop is not None
                                    and bool(should_stop()))


@contextlib.contextmanager
def captured() -> Iterator[Interrupt]:
    """Hear SIGINT as presses on a new Interrupt while the block runs.

    The handler goes in only on the main thread — the only one Python lets
    install one, and the one SIGINT is delivered to — and the one it replaced
    is put back on the way out, so a wrapper's own code between two runs sees
    Python's KeyboardInterrupt again. A runner opened inside an open run (a
    wrapper nesting them) shares the outer one's Interrupt: one press, one run
    of the process to stop.
    """
    global _active
    if _active is not None:
        yield _active
        return
    interrupt = Interrupt()
    previous = None
    installed = False
    if threading.current_thread() is threading.main_thread():
        try:
            previous = signal.signal(signal.SIGINT,
                                     lambda signum, frame: interrupt.press())
            installed = True
        except (ValueError, OSError):
            pass
    _active = interrupt
    try:
        yield interrupt
    finally:
        _active = None
        if installed:
            signal.signal(signal.SIGINT,
                          previous if previous is not None
                          else signal.default_int_handler)


def deliver() -> None:
    """A Ctrl+C read as a key ('\\x03' from the Windows key reader).

    Pressed on the open run's Interrupt from the reader's own thread — at once,
    with no main thread to wait for. Outside a run it keeps its usual meaning:
    a KeyboardInterrupt on the main thread.
    """
    interrupt = _active
    if interrupt is not None:
        interrupt.press()
        return
    import _thread

    _thread.interrupt_main()
