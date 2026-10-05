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
from typing import Callable, Iterator, List, NoReturn, Optional, Tuple

__all__ = ["EXIT_CODE", "INTERRUPTED_REASON", "WAIT_INTERRUPTED_LINE",
           "Interrupt", "asked", "captured", "current", "deliver", "leave",
           "leave_wait"]

# The exit code of a run the operator ended with Ctrl+C: 128 + SIGINT, what a
# shell reports for a process SIGINT killed. A constant so a wrapper telling an
# interrupt from a failure (a phase that "gave up") names it rather than 130.
EXIT_CODE = 130

# What the exit record says about a run the operator ended with Ctrl+C —
# from every door of the epilogue (`runlifecycle`) and from `captured`.
INTERRUPTED_REASON = "interrupted by the operator (Ctrl+C)"

# What a wait Ctrl+C ended prints, from every wait — the quota hold, the
# countdown after a refusal and the two before the run's boundary alike.
WAIT_INTERRUPTED_LINE = "\nWait interrupted by user (Ctrl+C)."


class Interrupt:
    """The Ctrl+C presses one run has heard. Safe from any thread.

    `press` is called by the SIGINT handler (on the main thread, between two
    bytecodes) or by the status line's key reader (on its own thread), and only
    counts; it never raises into whatever the run was doing.

    `press` TAKES NO LOCK, and that is load-bearing: the SIGINT handler runs on
    the main thread between any two bytecodes, the ones inside a `with lock:`
    of the main thread's own included — `on_press` registers a hook every turn
    — and a handler waiting for a plain lock its own thread holds waits for
    ever (llm-loop review of 0072, F1). So a press is a `list.append` (atomic
    under the GIL, from either thread) and the hooks are an immutable tuple,
    REPLACED whole by `on_press`; `_lock` only orders two registrations.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: List[None] = []
        self._hooks: Tuple[Callable[[], None], ...] = ()

    def press(self) -> None:
        """One Ctrl+C: counted, then every `on_press` hook run once for it.

        A hook that raises costs its own effect only — a handler must not be
        the thing that unwinds the run it is telling to stop.
        """
        self._pressed.append(None)
        for hook in self._hooks:    # one read of an immutable tuple
            try:
                hook()
            except Exception:       # noqa: BLE001 - see the docstring
                pass

    @property
    def presses(self) -> int:
        return len(self._pressed)

    @property
    def requested(self) -> bool:
        """Has the operator pressed Ctrl+C in this run at all?"""
        return self.presses > 0

    def since(self, mark: int) -> bool:
        """Was Ctrl+C pressed after `mark` (an earlier reading of `presses`)?

        How an ending tells "the press that caused me" from "a press telling me
        to stop waiting": it reads `presses` as it begins and asks this.
        """
        return self.presses > mark

    @contextlib.contextmanager
    def on_press(self, hook: Callable[[], None]) -> Iterator[None]:
        """Run `hook` on every press made while the block runs.

        Not run for a press made before the block: a caller that must act on one
        already heard asks `requested` first — AFTER entering the block, so a
        press between its own check and the registration is not missed.
        """
        with self._lock:
            self._hooks = self._hooks + (hook,)
        try:
            yield
        finally:
            with self._lock:
                hooks = list(self._hooks)
                hooks.remove(hook)
                self._hooks = tuple(hooks)


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

    A press the run never got to act on — after its last look (`RunBoundary.end`),
    in what the runner does past its epilogue — does not let the block RETURN:
    a returned result tells a wrapper to start its next phase, the opposite of
    what the operator asked. It leaves as the interrupt instead,
    INTERRUPTED_REASON and `SystemExit(EXIT_CODE)`. A block already leaving by
    an exception or an exit keeps it (`runlifecycle.exit_run`: an ending is
    chosen once); a wrapper that must tell "the phase failed" from "the
    operator pressed Ctrl+C while the phase was failing" opens the capture
    around the runner itself, and asks the Interrupt once the runner has left.

    The handler is put back BEFORE the Interrupt stops being the open run's,
    and the late press is looked for after both: a SIGINT in between is the
    replaced handler's again (a KeyboardInterrupt), never a press nobody reads.
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
        if installed:
            signal.signal(signal.SIGINT,
                          previous if previous is not None
                          else signal.default_int_handler)
        _active = None
    # Reached only when the block returned: an exception or an exit has left
    # through the `finally` above with its own ending.
    if interrupt.requested:
        leave()


def leave(**counts) -> NoReturn:
    """The interrupt's ending: INTERRUPTED_REASON in the run's exit record,
    then exit EXIT_CODE. `counts` (iterations=, completed=) go into the
    record beside the reason."""
    # Local: exitlog is a run's record, and ctrlc stays importable below it.
    from . import exitlog

    exitlog.set_reason(INTERRUPTED_REASON, **counts)
    raise SystemExit(EXIT_CODE)


def leave_wait() -> NoReturn:
    """A wait before the run's boundary that Ctrl+C ended: said, exit EXIT_CODE.

    The press inside a run, and a KeyboardInterrupt outside one, alike. The
    reason is `leave`'s, without counts: the run's prologue has already opened
    its exit record (`runlifecycle.begin_run`), which otherwise closed as
    "reason not recorded". Nothing else of an ending: no usage is open yet,
    and no pusher is started (see `runlifecycle.RunBoundary`).
    """
    from . import exitlog

    print(WAIT_INTERRUPTED_LINE)
    exitlog.set_reason(INTERRUPTED_REASON)
    raise SystemExit(EXIT_CODE)


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
