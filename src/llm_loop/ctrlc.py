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
(`stopchannel.STOP_RECHECK_SECONDS`, `runlifecycle.EXIT_PUSH_POLL_S`). On
Windows a wakeup socket hears SIGINT even while the main thread is blocked
reading a silent CLI. Its watcher runs idempotent process-ending hooks to
release the pipe read; only the Python SIGINT callback counts the press.
"""

import contextlib
import logging
import os
import select
import signal
import socket
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

    `press` is called by the SIGINT handler or the status line's key reader.
    The wakeup drain only releases blocked I/O; press is the sole counting
    method. Neither path raises into whatever the run was doing.

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
        self._wake_hooks: Tuple[Callable[[], None], ...] = ()

    def wake(self) -> None:
        """Release blocking I/O without counting a press.

        Only idempotent process-ending hooks opt in. The Python SIGINT
        callback will later count the press and run its ordinary hooks.
        """
        for hook in self._wake_hooks:
            try:
                hook()
            except Exception:
                pass

    def press(self) -> None:
        """One Ctrl+C: counted, then every `on_press` hook run once for it.

        A hook that raises costs its own effect only — a handler must not be
        the thing that unwinds the run it is telling to stop.
        """
        self._pressed.append(None)
        self._notify_hooks()

    def _notify_hooks(self) -> None:
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
    def on_press(self, hook: Callable[[], None], *, wake: bool = False) -> Iterator[None]:
        """Run `hook` on every press made while the block runs.

        Not run for a press made before the block: a caller that must act on one
        already heard asks `requested` first — AFTER entering the block, so a
        press between its own check and the registration is not missed.

        `wake=True` opts an idempotent I/O-release hook into early invocation
        by signal bytes, before the Python callback can run. Such a hook may
        run again when the callback counts the press; ordinary hooks do not.
        """
        with self._lock:
            self._hooks = self._hooks + (hook,)
            if wake:
                self._wake_hooks = self._wake_hooks + (hook,)
        try:
            yield
        finally:
            with self._lock:
                hooks = list(self._hooks)
                hooks.remove(hook)
                self._hooks = tuple(hooks)
                if wake:
                    wake_hooks = list(self._wake_hooks)
                    wake_hooks.remove(hook)
                    self._wake_hooks = tuple(wake_hooks)


class _Unheard(Interrupt):
    """What `current` answers outside a run: never pressed, whatever is asked."""

    def press(self) -> None:
        pass


_UNHEARD = _Unheard()
_active: Optional[Interrupt] = None


class _SignalWakeup:
    """Wake blocked provider I/O; NEVER count signal bytes.

    The Python callback alone calls Interrupt.press. Whether a C signal
    arrives before installation, while attached, or after detachment, its
    callback counts once, even if delayed across either boundary. A byte
    merely invokes idempotent wake hooks so the blocked main thread can run
    that callback. Byte delivery and callback execution can occur in either
    order without changing the count. No descriptor inspection is needed
    in the handler, and no pending-byte credit can swallow a later press.

    Exact physical-keypress counting is impossible with this API: two C
    signals before Python runs can write two bytes but coalesce into one
    pending Python callback; a full wakeup socket can also drop bytes.
    We preserve Python's counting semantics: exactly one count per callback,
    never an extra count, and never zero for an isolated delivered SIGINT.
    Bursts may coalesce, as outside capture. Wake hooks can run more than once
    and must be idempotent. Ordinary on_press hooks run only from press().

    Two physical Ctrl+C events before Python services the first can coalesce
    into one callback: the second-press exit contract is per callback. This
    is acceptable for a human pressing again after the first Python callback
    has been serviced: the next is distinct. Observing provider termination
    alone does not prove this, since the watcher can terminate it first.
    A rapid double tap before callback service is not two guaranteed presses.

    Foreign notification is NOT lossless during ownership changes. Activation
    probes by installing -1, then restores a foreign fd if one was present.
    Even a declined takeover has that gap. Close likewise restores -1 and
    reinstalls any overlapping foreign replacement returned by the exchange.
    Between either pair of calls, foreign wakeup bytes are lost (Python
    handlers may still run). Neither gap has a wall-clock bound: scheduling
    or a signal handler can delay restoration arbitrarily. Cooperative,
    non-overlapping ownership is required where lossless foreign notification
    matters; there is no public getter/compare-and-swap to guarantee it here.
    """

    def __init__(self, interrupt: Interrupt):
        self.interrupt = interrupt
        self.reader, self.writer = socket.socketpair()
        try:
            self.reader.setblocking(False)
            self.writer.setblocking(False)
        except BaseException:
            self.reader.close()
            self.writer.close()
            raise
        self.lock = threading.RLock()
        self.stopping = False
        self.thread = threading.Thread(target=self.watch, name='ctrlc-wakeup', daemon=True)
        self.started = False
        self.owns_fd = False

    @staticmethod
    def current_fd():
        # Python has no get_wakeup_fd. Only the main thread can change it,
        # so an exchange and restore also detects an overlapping owner.
        fd = signal.set_wakeup_fd(-1)
        if fd != -1:
            signal.set_wakeup_fd(fd)
        return fd

    def activate(self):
        # Do not steal another event loop's notifications (including SIGBREAK).
        # Keep ordinary handler/line-polling behavior when it owns this slot.
        if self.current_fd() != -1:
            return False
        signal.set_wakeup_fd(self.writer.fileno())
        self.owns_fd = True
        self.thread.start()
        self.started = True
        return True

    def handle(self):
        self.interrupt.press()

    def drain(self):
        # Only the watcher consumes bytes; the handler never touches sockets.
        count = 0
        with self.lock:
            while True:
                try:
                    value = self.reader.recv(1)
                except BlockingIOError:
                    break
                if not value:
                    break
                if value[0] == signal.SIGINT:
                    count += 1
        for _ in range(count):
            self.interrupt.wake()

    def watch(self):
        while not self.stopping:
            select.select([self.reader], [], [])
            self.drain()

    def close(self):
        try:
            if self.owns_fd:
                # Restore the -1 we replaced. The exchange's return value
                # lets an overlapping replacement keep its slot; Python has
                # no compare-and-swap API. A declined takeover never touches it.
                fd = signal.set_wakeup_fd(-1)
                self.owns_fd = False
                if fd != self.writer.fileno() and fd != -1:
                    signal.set_wakeup_fd(fd)
        except (ValueError, OSError):
            # Installation may itself be unavailable (e.g. a sub-interpreter).
            # The sockets still need closing on that fallback path.
            pass
        finally:
            self.stopping = True
            try:
                self.writer.send(b'\x00')
                if self.started:
                    interrupted = None
                    while True:
                        try:
                            self.thread.join()
                            break
                        except KeyboardInterrupt as exc:
                            # Real SIGINT still uses our non-raising handler.
                            # An injected/asynchronous exception must not skip
                            # resource closure either; re-raise after joining.
                            interrupted = exc
                    if interrupted is not None:
                        raise interrupted
            finally:
                try:
                    self.drain()
                finally:
                    self.reader.close()
                    self.writer.close()


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

    Wakeup delivery is detached and its watcher joined while SIGINT still
    counts presses on this run. Then the previous handler is restored, and
    the open Interrupt is cleared even if restoration or cleanup raises.
    A signal after handler restoration belongs only to that previous handler.
    """
    global _active
    if _active is not None:
        yield _active
        return
    interrupt = Interrupt()
    previous = None
    installed = False
    wakeup = None
    if threading.current_thread() is threading.main_thread():
        try:
            previous = signal.signal(signal.SIGINT,
                                     lambda signum, frame: wakeup.handle()
                                     if wakeup is not None else interrupt.press())
            installed = True
        except (ValueError, OSError):
            pass
    _active = interrupt
    try:
        if installed and os.name == 'nt':
            try:
                wakeup = _SignalWakeup(interrupt)
                if not wakeup.activate():
                    logging.getLogger(__name__).warning(
                        'SIGINT wakeup disabled: foreign wakeup fd owns notifications; '
                        'using output polling')
            except (ValueError, OSError) as exc:
                if wakeup is not None:
                    wakeup.close()
                    wakeup = None
                logging.getLogger(__name__).warning(
                    'SIGINT wakeup unavailable; using output polling: %s', exc)
        yield interrupt
    finally:
        try:
            if wakeup is not None:
                wakeup.close()
                wakeup = None
        finally:
            try:
                if installed:
                    signal.signal(signal.SIGINT,
                                  previous if previous is not None
                                  else signal.default_int_handler)
            finally:
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
