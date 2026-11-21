"""Fleet claim state, admission and settlement behind one shared lock."""

import os
import threading
from typing import Callable, NamedTuple, Optional
from uuid import uuid4

from . import exitlog, stopchannel
from .drivers import ListFileDriver, Settled
from .stopchannel import RunStopReason


class Finished(NamedTuple):
    """What `Shared.finish` reports back to the worker that finished an item."""

    done: int                      # items done this run, across all workers
    remaining: int                 # still pending in the list
    verdict: Optional[Settled]     # the driver's verdict; None for a failed turn
    parked: bool                   # the ledger has parked the item's key


def settled_done(verdict: Optional[Settled]) -> bool:
    """Does `verdict` count as done? STRUCK and GONE do; REFUSED and a failed
    turn (None) do not. One reading for the counter and the report line."""
    return verdict is not None and verdict is not Settled.REFUSED


class Shared:
    """Cross-worker state behind a single lock: the list cursor and run stats.

    The list file (owned by `driver`) is the source of truth for what remains;
    `in_progress` keeps workers from claiming the same item, the ledger parks items
    that exhausted their retry budget, and the counters bound/report the run. All
    access to THAT state is under `lock`.

    `settings` is the exception, and it is not this object's state: the run's
    knobs are written by whoever holds the keyboard (the status line's editor,
    on its own thread) and only READ here, under the lock, at the moment a claim
    is decided — see `max_items`.
    """

    def __init__(self, driver: ListFileDriver, settings, *, progress=None):
        self.driver = driver
        self.progress = progress
        self.log_session = uuid4().hex
        # The run's live knobs (runlifecycle.RunSettings), not a copy of the cap
        # taken here: `--max-runs` is editable from the status line, and the
        # claim loop below is the one place that enforces it — so it has to read
        # the value the editor writes, at the moment it claims. See `max_items`.
        self.settings = settings
        self.lock = threading.Lock()
        # Everything below is per ITEM KEY (`driver.item_key`, the raw line by
        # default), never per raw line: a key in `in_progress` keeps every line
        # with that key unclaimable, so two spellings of one item never run at
        # once, and they share one retry budget.
        #
        # Keys a worker is handling — claimed, running, or still SETTLING: a
        # finished turn's key leaves only in `settle`, after `item_finished`
        # has answered (see `finish`).
        self.in_progress = set()
        # The subset of those whose provider turn is actually in flight. Not the
        # same question as `in_progress`, and the difference is what a pending
        # stop turns on: a worker parked on the usage gate holds a claim but is
        # not doing anything a stop would interrupt, so it must not be counted as
        # work in flight — see busy().
        self.running = set()
        # Line in flight -> the key it was CLAIMED under. Every later step
        # (`start_turn`, `finish`, `settle`, `release`) reads the key here
        # rather than asking `item_key` again: an answer that changed in
        # between would leave the claimed key in `in_progress` for ever, and
        # `_exhausted` with it. One line per key, since a held key's twins
        # cannot be claimed.
        self._keys = {}
        self.claimed = 0              # files claimed this run (for --max-runs)
        self.started = 0              # provider turns started, including interrupted ones
        self.done = 0                 # files processed successfully
        self.stop = threading.Event()  # cancel/wake the run on stop-file / no-work
        self.claims_closed = threading.Event()  # max reached: finish in-flight work
        # A stop request closes claims too, but REVERSIBLY (see reopen_claims).
        # Deliberately a second flag: cancelling a mis-pressed `s` must never be
        # able to reopen a run whose claims closed because it hit --max-runs or
        # drained the list, and one shared flag could not tell those apart.
        self.stop_requested = threading.Event()
        self.stop_owner = None        # job_id deciding the request's fate
        self.stop_reason = None
        # Why the driver asked the run to hand control back (see
        # request_driver_handback). Read by the workers as well as written: it
        # is what releases one parked on the usage gate, and what tells a claim
        # held there that it may go back to the queue. The latch's own rule is
        # stopchannel's; `self.lock` still guards its coupling to
        # `claims_closed` and `stop_reason`.
        self.handback = stopchannel.DriverHandback()
        # Whether the fleet has already said it is paused (see note_pause): the
        # `p` key is read by every worker, and a fleet of eight would otherwise
        # announce one keypress eight times.
        self.paused_noted = False

    @property
    def max_items(self) -> Optional[int]:
        """The live `--max-runs` cap: how many FILES this run may claim.

        Read through the run's settings on every ask rather than copied in, so
        an edit made from the status line lands in the one place that enforces
        the cap. It has to be a read and not a snapshot: this runner has no
        iteration boundary on the main thread where a copy could be refreshed —
        which is exactly why the knob used to be missing here.
        """
        return self.settings.max_runs

    def key(self, line: str):
        """The item key of `line` (`ListFileDriver.item_key`)."""
        return self.driver.item_key(line)

    def _pending(self) -> list:
        """Lines whose key nobody holds and nobody gave up on (call under
        `lock`)."""
        pending = []
        for ln in self.driver.pending_lines():
            key = self.key(ln)
            if key not in self.in_progress and not self.driver.ledger.is_parked(key):
                pending.append(ln)
        return pending

    def _exhausted(self, pending: list) -> bool:
        """Has this run run out of work to give? (call under `lock`)

        The two ways it ends of its own accord — the item cap, and a queue that
        drained with nobody still working — latched here rather than inside
        `claim`, because a worker that is NOT claiming has to be able to ask.
        See `exhausted`, and the pause hold in `parallel.worker` that uses it.

        A queue that is merely empty right now is not exhausted: the rest is in
        flight elsewhere, and the caller backs off and retries.
        """
        self._handback_locked(self.driver.needs_human())
        if self.claims_closed.is_set():
            return True
        if self.max_items is not None and self.claimed >= self.max_items:
            self.stop_reason = RunStopReason.LIMIT_REACHED
            self.claims_closed.set()
            return True
        if not pending and not self.in_progress:
            self.stop_reason = RunStopReason.NO_WORK
            self.claims_closed.set()
            self.stop.set()
            return True
        return False

    def exhausted(self) -> bool:
        """True once this run has no work left to give, latching why.

        Asked by a worker held on `p`: a pause is about not STARTING work, and a
        run with none left has no start to hold back — while a fleet that waits
        for one anyway never sets `stop`, so `parallel.run_parallel`'s join() never
        returns and the whole run hangs on a key that was only meant to slow it
        down. It is the same reading `claim` acts on, from the same lock, so the
        two cannot come to different verdicts about the run being over.
        """
        with self.lock:
            return self._exhausted(self._pending())

    def claim(self) -> Optional[str]:
        """Reserve the next pending list line, or signal why there is none.

        Returns the claimed raw line, or None. `claims_closed` distinguishes a
        clean max-items boundary (finish in-flight work, issue nothing new) from
        a temporarily busy queue; `stop_requested` closes claims for a pending
        stop sentinel; `stop` cancels claims held at a usage gate.
        """
        with self.lock:
            if self.stop_requested.is_set():
                return None
            pending = self._pending()
            if self._exhausted(pending):
                return None
            if not pending:
                return None   # busy: others hold the rest — the caller backs off
            # The driver decides the order (random by default, list order when
            # it sets pick_order = "list"); we are already under the lock, which
            # is the thread-safety pick() is documented to expect.
            line = self.driver.pick(pending)
            key = self.key(line)
            self.in_progress.add(key)
            self._keys[line] = key
            self.claimed += 1
            in_flight = sorted(os.path.basename(ln.strip())
                               for ln in self._keys)
            claimed, done = self.claimed, self.done
            counts = (self.progress.count_fields()
                      if self.progress is not None else
                      dict(iterations=claimed, completed=done))
        # Outside the lock — this writes a file, and every worker contends for
        # that lock. It leaves behind what the run had in flight, which is what
        # a post-mortem of a killed run has to start from (see exitlog).
        exitlog.note(phase=f"in flight: {', '.join(in_flight)}",
                     **counts)
        return line

    def release(self, line: str) -> None:
        """Return a claimed-but-unprocessed line to the queue.

        Used when a worker claims a line but then bails out before running it
        (the run was stopped while it sat in the session-limit gate), or when
        Ctrl+C cut its turn short: drop it from
        `in_progress` so the drained check stays accurate, and undo its --max
        reservation so the count reflects only files actually processed.
        """
        with self.lock:
            key = self._drop_locked(line)
            # A turn Ctrl+C cut short is released too (`worker`), after its
            # `start_turn`: a release that left the line in `running` would
            # make busy() true forever and hang every later stop request.
            self.running.discard(key)
            if self.claimed > 0:
                self.claimed -= 1

    def _held_key(self, line: str):
        """The key `line` was claimed under (call under `lock`); asked of
        `item_key` afresh only for a line this run never claimed."""
        return self._keys[line] if line in self._keys else self.key(line)

    def _drop_locked(self, line: str):
        """`line`'s key is no longer in flight; returns it (call under
        `lock`)."""
        key = self._held_key(line)
        self._keys.pop(line, None)
        self.in_progress.discard(key)
        return key

    def start_turn(self, line: str) -> bool:
        """Admit a turn atomically against handback; False leaves it unstarted."""
        with self.lock:
            if self.handback.pending:
                return False
            self.running.add(self._held_key(line))
            self.started += 1
            if self.progress is not None:
                self.progress.record_counts(self.started, self.done)
            return True

    def abandon(self, line: str, started: bool) -> None:
        """Give a claim back when its worker is dying, so the run can still end.

        Called from the guard in `worker` when anything between the claim and
        `finish` raises. Whatever else it does, it must drop the line from
        `in_progress`: a dying worker latches no stop, so a claim left there
        makes `_exhausted` false for ever and every surviving worker spins in
        the back-off until the run is killed by hand.

        `started` picks between the two ways of giving a claim back, and they
        are not interchangeable:

          * BEFORE `start_turn` nothing has been attempted — the provider was
            never launched, the target file was never touched, and the claim's
            --max-runs reservation was paid for a file nobody started. So the
            line goes back to the queue verbatim (`release`) and the next worker
            picks it up; the run must be able to complete it.
          * AFTER `start_turn` a turn really was attempted, so it is recorded as
            a failed one (`finish(line, False)`). Not cosmetic: `finish` is what
            records a failure in the ledger, which can park the key.
            A file whose turn reliably kills its
            worker (a provider CLI that dies mid-stream, a console that refuses
            a write) would otherwise be handed back untouched and kill the next
            worker, and the next — a fleet of ten dies ten times over one line.
            Counted, the same line stops the run after
            AttemptLedger.MAX_ATTEMPTS workers and is reported as failed,
            which is also what it is.

        The exception itself is not caught here: the guard re-raises, so the
        thread still ends the way it was always going to. What changes is that
        the REST of the fleet can now finish the run.
        """
        if started:
            self.finish(line, False)
        else:
            self.release(line)

    def stop_asked(self, app=None) -> bool:
        """Is anything asking this run to stop — its own latch, or a channel?

        The worker's one reading of the question. It used to be spelled three
        ways in one function (the gate's predicate, the hold loop's condition,
        the checks around them), which is three places that have to agree about
        a question with two sources.
        """
        return self.stop.is_set() or stopchannel.pending_stop(app) is not None

    def finish(self, line: str, ok: bool, *, settling: bool = False
               ) -> "Finished":
        """Record an item's outcome: settle it on success, or count/park a fail.

        Success is the driver's verdict, not the exit code's: `ok` asks
        `driver.settle(line)`, and a `Settled.REFUSED` is counted exactly like
        a failed turn — an attempt against the key, parked at
        AttemptLedger.MAX_ATTEMPTS,
        not "done". GONE (the line was already off the list) is done: nothing
        is left to retry.

        `settling` keeps the key in `in_progress` until `settle` — the worker's
        way of making `item_finished`'s answer part of the completion (see
        `settle`); without it the key is dropped here, as for a dying worker
        (`abandon`).

        Returns `Finished(done, remaining, verdict, parked)`: files done this
        run (across all workers), how many are still pending, the driver's
        verdict (None for a failed turn), and whether the item's key is now
        parked.
        """
        with self.lock:
            key = (self._held_key(line) if settling
                   else self._drop_locked(line))
            self.running.discard(key)
            try:
                verdict = self.driver.settle(line) if ok else None
                if settled_done(verdict):
                    self.done += 1
                    if self.progress is not None:
                        self.progress.record_counts(self.started, self.done)
                    self.driver.ledger.succeed(key)
                else:
                    self.driver.ledger.fail(key, line)
                remaining = self.driver.pending_total()
            except BaseException:
                # No `settle` follows a `finish` that raised (the worker never
                # reaches its `try`), so the key goes now or the run can never
                # read as drained.
                self._drop_locked(line)
                raise
            return Finished(self.done, remaining, verdict,
                            self.driver.ledger.is_parked(key))

    def settle(self, line: str, answer) -> bool:
        """The end of a `finish(..., settling=True)`: drop the key from
        `in_progress` and latch `item_finished`'s `answer`, in ONE section of
        `lock`. True for the worker that should announce the answer.

        One section because the two used to be apart, and the run's ending
        depended on who won the gap: with the last key already dropped, a
        worker in the back-off read the queue as drained and latched NO_WORK,
        which then refused the hand-back — so a driver asking for a human got
        NO_WORK (or DRIVER_PAUSE) depending on timing. Now nothing can read
        the run as drained while a hook is still answering.
        """
        with self.lock:
            self._drop_locked(line)
            return self._handback_locked(answer)

    # --- the stop request: closing claims is not the same as ending the run ----

    def request_stop(self, job_id: int) -> tuple:
        """Close new claims for a pending sentinel; returns (owner, first).

        Work in flight is untouched — and this close is undoable, which is what
        makes `s` a toggle for as long as a job row is still moving.

        `owner` marks the ONE worker that decides the request's fate: N workers
        each running the cancel countdown would fight over the note row and each
        latch on its own deadline. `first` is True only on the pass that opened
        the request, so the log line is written once however many workers see it.
        """
        with self.lock:
            first = not self.stop_requested.is_set()
            if first:
                self.stop_owner = job_id
            self.stop_requested.set()
            return self.stop_owner == job_id, first

    def reopen_claims(self) -> bool:
        """The sentinel went away again: resume claiming. True if it had closed.

        Only the stop close is undone. `claims_closed` (--max-runs, drained list)
        is final by design: a max-items stop must not become cancellable just
        because somebody removed a stop file.
        """
        with self.lock:
            if not self.stop_requested.is_set():
                return False
            self.stop_requested.clear()
            self.stop_owner = None
            return True

    def latch_stop(self, source, app, announce: Callable[[str], None]) -> bool:
        """End the run on a stop request, for good. True for the worker that did it.

        `announce` writes the line, and has no default on purpose: a worker's
        line goes to the console's owner (`job_lines(...).line`), and a
        fallback to a bare `print` would be a worker writing the console itself.

        Returning True exactly once is what keeps the announcement (and the
        lifecycle latch) single when every worker sees the request at the same
        time — which is why the tail runs here rather than being handed back to
        the winner: `stopchannel.commit_stop` latches the sentinel for cleanup and
        names the reason, and this is the one place that decides there is a
        winner at all. Both runners get that tail from there, so a new stop
        channel is one edit, not two.

        The transition finishes INSIDE the lock and the line is written OUTSIDE
        it, in that order and never the other way round — two separate reasons:

          * order, because a write that raises must not take the latch with it.
            Announcing first left `stop_reason` unset and `stop` clear when the
            console refused the line, and the run reported NO_WORK for a stop
            FILE it had obeyed (see `stopchannel.commit_stop`);
          * outside, because `shared.lock` is also the lock every other worker
            needs to claim, finish or release a file, and a console that blocks
            would hold all of them for as long as the write is stuck.

        What is still taken under `shared.lock` is `commit_stop`'s `os.path.exists`
        and the stop-file lifecycle lock behind `mark_stop_file_detected` — a
        stat and a global assignment, both of which have to be the winner's, and
        neither of which claims `shared.lock` back.
        """
        with self.lock:
            if self.stop.is_set():
                return False
            self.stop_reason, announcement = stopchannel.commit_stop(app, source)
            self.claims_closed.set()
            self.stop.set()
        announce(announcement)
        return True

    def request_driver_handback(self, reason: str) -> bool:
        """The driver asked to hand control back: close claims, keep the fleet.

        The same close `--max-runs` uses, and for the same reason it is not a
        `stop`: work in flight is not cancelled, only new claims are refused, so
        the run winds down without throwing away a turn somebody has paid for.

        True for the caller that latched it — one line per request however many
        workers finish an item at once. A close that is already in place wins:
        the cap and the drained queue are final endings (`claims_closed` is not
        reopenable), and a pause arriving a moment later must not relabel them.
        """
        with self.lock:
            return self._handback_locked(reason)

    def _handback_locked(self, answer) -> bool:
        """`request_driver_handback`'s body (call under `lock`).

        A `NeedsHuman` is the exception to "a close already in place wins": it
        latches whatever was latched before it (the cap, the drained queue, a
        pause), since it is about the work being stuck, which those endings
        say nothing about. A stop channel's reason is overwritten too — the
        exit code is what a script reads, and it must say "stuck".
        """
        if isinstance(answer, stopchannel.NeedsHuman):
            if not self.handback.latch(answer):
                return False
            self.stop_reason = RunStopReason.NEEDS_HUMAN
            self.claims_closed.set()
            return True
        if self.claims_closed.is_set() or not self.handback.latch(answer):
            return False
        self.stop_reason = RunStopReason.DRIVER_PAUSE
        self.claims_closed.set()
        return True

    def note_pause(self, paused: bool) -> bool:
        """True for the worker that should announce this pause (or its release).

        One line per transition, whoever gets here first — the alternative is a
        line per worker per poll, which would bury the run's own output under
        the very hold it is reporting.
        """
        with self.lock:
            if paused == self.paused_noted:
                return False
            self.paused_noted = paused
            return True

    def busy(self) -> bool:
        """Is any provider turn actually running right now?

        The question a pending stop asks, and it is deliberately narrower than
        "is any line claimed": the grace exists so the toggle stays usable for
        as long as the user can see a job row MOVING, and a worker parked on the
        usage gate moves nothing. Counting its claim here would hang the
        decision on the very hold the user is trying to escape — the worker
        waits for the verdict while the verdict waits for the worker.
        """
        with self.lock:
            return bool(self.running)
