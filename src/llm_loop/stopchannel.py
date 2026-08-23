"""
stopchannel.py - how a run is asked to stop or to hold, and what that costs.

One vocabulary, spoken by everything that can end a run: the two channels a
request arrives on (`StopSource` - the in-process `s` key and the `stop`
sentinel on disk), the reading of them (`pending_stop` / `latched_stop`), the
countdown that lets an interactive request be taken back
(`confirm_stop_request`), the tail that acts on one (`commit_stop`), the
sentinel's lifecycle across nested runner calls (`stop_file_lifecycle`), the
`p` key's hold (`pause_requested` / `wait_while_paused`), and the outcome a
runner reports afterwards (`RunResult`, `RunStopReason`, `STOP_REASON_TEXT`).

Apart from the runners because BOTH of them speak it, and so does a host
wrapper that slices one invocation into several runner calls. While this was
part of cyclecore, `parallel.py` had to import the module that defines the
sequential `run_loop` in order to ask "is anything asking us to stop?" - a
dependency on an address rather than on a subject, and the kind that makes
adding a stop channel look like a change to the sequential runner. Nothing
here imports the rest of the package, which is what keeps that true.

The sentinel is anchored on the project root, which this module does not own:
`cyclecore.set_project_root` hands it over through `set_stop_root`. See
STOP_FILE.
"""

import os
import sys
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional, Tuple


class RunStopReason(Enum):
    """Why a runner returned normally."""

    STOP_FILE = "stop_file"
    STOP_KEY = "stop_key"
    LIMIT_REACHED = "limit_reached"
    NO_WORK = "no_work"
    DRIVER_STOP = "driver_stop"
    DRY_RUN = "dry_run"


class StopSource(Enum):
    """Which channel is asking a run to stop. The two are NOT interchangeable.

    KEY is in-process and belongs to one run: the `s` key sets a flag inside
    that process, so several loops launched in the same project root can be
    stopped one at a time. It leaves nothing on disk — nothing to clean up, and
    nothing for a concurrent run to trip over.

    FILE is the shared `stop` sentinel in the project root: it halts EVERY run
    watching that root, and it survives the process. That is what makes it the
    cross-process handshake — one run ends, and a launch waiting on the file
    (wait_for_stop_file_clear) starts once the file is gone — so the key must
    not write it and a key-stop must not remove it.
    """

    KEY = "key"
    FILE = "file"


class RunResult(NamedTuple):
    """Structured normal-exit result used by higher-level coordinators."""

    reason: RunStopReason
    attempted: int = 0
    completed: int = 0
    remaining: Optional[int] = None


# What each stop reason is called in the `=== run ended: … ===` line, which is
# the one place a reader looks when asking "why did this stop?". Phrased for
# that reader rather than reusing the enum's wire value.
STOP_REASON_TEXT = {
    RunStopReason.STOP_FILE: "stop file requested",
    RunStopReason.STOP_KEY: "stop requested with the s key",
    RunStopReason.LIMIT_REACHED: "iteration limit reached (--max-runs)",
    RunStopReason.NO_WORK: "no more work in the queue",
    RunStopReason.DRIVER_STOP: "the driver stopped the run",
    RunStopReason.DRY_RUN: "dry run finished",
}

# The reasons that mean "a human asked this INVOCATION to stop", as opposed to
# "this runner call ran out of work". A wrapper that slices one invocation into
# several runner calls — the periodic batch loop in a host project — must end on
# any of them: each call builds its own StatusApp, so a key request that only
# ended one batch is gone by the next, and the keypress is silently absorbed
# batch after batch. Membership, never `== STOP_FILE`, so a new channel here
# reaches those wrappers by itself.
REQUESTED_STOP_REASONS = frozenset(
    {RunStopReason.STOP_FILE, RunStopReason.STOP_KEY})


# A manual brake shared by every run rooted here: `touch stop` (create a file
# named "stop" in the project root) and the loop halts at the next iteration
# boundary - the running iteration finishes its one state transition first. The
# file stays present while the application winds down, then the outermost
# stop-file lifecycle removes it just before exit so other launchers can use it
# as a mutex. Recomputed by set_stop_root().
#
# Deliberately NOT what the status line's `s` key writes: see StopSource. The
# file is the cross-process channel (and the handshake for chaining runs); the
# key is the per-run one.
#
# The cwd is only the pre-launch default: `cyclecore.set_project_root` calls
# `set_stop_root` at import and again from --project-dir/-C. Readers that
# outlive that call must read the ATTRIBUTE (`stopchannel.STOP_FILE`), never a
# `from … import` copy of it, or a `-C` run watches the launch directory's
# sentinel while it works somewhere else — which looks exactly like a stop file
# nobody obeys. `stop_file_for()` is the reader that cannot be got wrong.
STOP_FILE = os.path.join(os.getcwd(), "stop")


def set_stop_root(root: str) -> None:
    """Anchor the shared sentinel on `root` (see STOP_FILE).

    The project root belongs to the engine, not to this module — a stop channel
    has no opinion about where git runs or where a Driver's paths resolve — so
    it is handed over rather than read back out of cyclecore, which would be
    the import this split exists to remove.
    """
    global STOP_FILE
    STOP_FILE = os.path.join(root, "stop")

# How often wait_for_stop_file_clear() re-checks the sentinel while it holds a
# launch back. Short enough that removing the file feels immediate.
STOP_POLL_SECONDS = 2

# How often a wait that is holding the run re-reads the stop channels. It is the
# responsiveness of `s` while nothing is running — the cancel countdown polls at
# the same rate — and the only cost is one os.path.exists per interval.
STOP_RECHECK_SECONDS = 0.25

_stop_file_lifecycle_lock = threading.Lock()
_stop_file_lifecycle_depth = 0
_detected_stop_file: Optional[str] = None


@contextmanager
def stop_file_lifecycle():
    """Keep a detected stop sentinel until the outermost application exits.

    Runners enter this lifecycle themselves so direct users retain one-shot stop
    semantics. A project wrapper can enter it once around its complete main()
    lifecycle; nested sequential, parallel, and periodic runner calls then leave
    the sentinel in place through all wrapper-level cleanup.
    """
    global _stop_file_lifecycle_depth, _detected_stop_file
    with _stop_file_lifecycle_lock:
        if _stop_file_lifecycle_depth == 0:
            _detected_stop_file = None
        _stop_file_lifecycle_depth += 1
    try:
        yield
    finally:
        with _stop_file_lifecycle_lock:
            _stop_file_lifecycle_depth -= 1
            outermost = _stop_file_lifecycle_depth == 0
            detected = _detected_stop_file if outermost else None
            if outermost:
                _detected_stop_file = None
        if detected is not None:
            try:
                os.remove(detected)
            except FileNotFoundError:
                pass
            except OSError as exc:
                print(f"warning: could not remove stop file on application exit "
                      f"({detected}): {exc}", file=sys.stderr)
            else:
                print("Stop file removed on application exit.")


def mark_stop_file_detected(path: Optional[str] = None) -> None:
    """Latch a stop path for outermost-lifecycle exit cleanup (default: this
    project root's). Take the path from `stop_file_for(app)` when there is an
    app, so the file that is removed is the file that was obeyed."""
    global _detected_stop_file
    with _stop_file_lifecycle_lock:
        _detected_stop_file = path or STOP_FILE


def stop_file_for(app=None) -> str:
    """The sentinel path this run watches: the app's own, else this root's.

    One reader for both, because a StatusApp built with an explicit `stop_file=`
    would otherwise show a row about one file while the runner obeyed another.
    """
    return getattr(app, "stop_file", None) or STOP_FILE


def pending_stop(app=None) -> Optional[StopSource]:
    """What is asking this run to stop right now, and through which channel.

    The single place that answers "should we be stopping?", so the two channels
    cannot drift apart between the sequential and the parallel runner. The key
    is checked first: when a run holds both, the interactive request is the one
    with a human behind it, and it is the one that can still be cancelled —
    which is what this answer is for. For what to ACT on, see `latched_stop`.
    """
    if getattr(app, "stop_requested_here", False):
        return StopSource.KEY
    if os.path.exists(stop_file_for(app)):
        return StopSource.FILE
    return None


def latched_stop(app=None) -> Optional[StopSource]:
    """The channel a run that is stopping NOW must record and clean up after.

    Differs from `pending_stop` in the one case that matters: while both are up,
    `pending_stop` names KEY, because that is the request a human can still take
    back — but a run that stops with a sentinel on disk is the run that has to
    consume it. Report the key press instead and the file outlives the loop it
    was written for, and the next launch (`wait_for_stop_file_clear`, which has
    no timeout) waits on it forever.
    """
    if os.path.exists(stop_file_for(app)):
        return StopSource.FILE
    return pending_stop(app)


# The half of a stop-file announcement that does not depend on what found the
# sentinel: the run ends, the FILE stays. A constant rather than a literal in
# each sentence, because that promise is made in three places — both runners'
# stop tail (`commit_stop`) and the wrapper that looks between periodic phases
# (runGenerateModels._detect_periodic_stop) — and the copies had already drifted
# into different spellings of it.
STOP_FILE_KEPT_CLAUSE = ("stopping; it remains in place until "
                         "the application exits.")


def commit_stop(app, source: StopSource) -> Tuple[RunStopReason, str]:
    """Act on a stop a run has decided to obey: latch the sentinel, name the
    reason, and hand back the line to announce — saying it is the caller's job.

    The tail of the decision, shared by both runners. Its head already was —
    `pending_stop`, `latched_stop`, `confirm_stop_request` — while this half was
    written twice, and the copies had drifted into two wordings for one event.
    That drift is the cheap symptom: what it costs is that a new stop channel,
    or a change to what a stop consumes and cleans up, is two edits in two
    modules with nothing to catch the missed one (both halves are Python in one
    process, so there is no mirror gate to fail).

    The line is RETURNED, never written here, and that is correctness rather
    than taste: in the parallel runner this tail runs inside `shared.lock`, on
    the single worker that won the latch, and a console write fails for reasons
    that have nothing to do with the run — a closed pipe (`… | head`), a code
    page that cannot spell the em dash, rich itself. Written from in here, such
    an exception unwound the winner BEFORE the reason and the `stop` event were
    set: the other workers found the run unlatched, died on the same write in
    turn, and a run that obeyed a stop FILE reported NO_WORK on the way out.
    Everything that decides the outcome therefore finishes before this returns,
    and nothing on the way to it can fail (see `parallel.Shared.latch_stop`,
    which is where the ordering is spent, and the pin
    `test_parallel_stop_file_latches_even_when_the_console_write_fails`).

    NOT `pending_stop`, and not the caller's `source` either: what may still be
    cancelled and what must be cleaned up are different questions once the run
    is committed to stopping, so the channel is re-read through `latched_stop`
    here. `source` is the fallback for a sentinel that vanished between the
    caller's decision and this call, and it has no default on purpose: with both
    channels silent and nothing named, an unnamed stop would be reported as a
    key press — silently, and in the one window where the truth is hardest to
    reconstruct afterwards.

    That vanished-sentinel window resolves to `source`, i.e. to STOP_FILE, with
    a path that no longer exists marked for exit cleanup. Both halves are
    deliberate. The run is stopping because of a FILE — nobody pressed `s`, and
    STOP_KEY would name a human who was never there — and marking a path that is
    already gone costs nothing, since `stop_file_lifecycle` treats a missing
    file as done; the opposite mistake, not marking a sentinel that comes back,
    leaves it for the next launch's `wait_for_stop_file_clear` to wait on
    forever. The parallel runner has read it this way since it grew a stop file;
    the sequential one used to answer STOP_KEY here and mark nothing, and that
    difference ended when the two tails became this one.
    """
    latched = latched_stop(app) or source
    if latched is StopSource.FILE:
        # Only a stop FILE is latched for removal at application exit — a key
        # request never touched the disk. The outermost lifecycle removes the
        # sentinel only after every worker and all wrapper-level cleanup is done.
        mark_stop_file_detected(stop_file_for(app))
        return (RunStopReason.STOP_FILE,
                f"Stop file detected — {STOP_FILE_KEPT_CLAUSE}")
    return (RunStopReason.STOP_KEY,
            "Stop requested with the s key — stopping this run "
            "(no stop file written; other runs are unaffected).")


# How long an interactive run counts down before it acts on a stop request. The
# status line's `s` key both sets and clears the request, so a mis-press must be
# undoable — but a runner that ended a millisecond later would make "press s
# again" a race the user always loses. Five seconds is long enough to notice the
# countdown row and press the key again, short enough that a deliberate stop
# still feels immediate.
STOP_GRACE_SECONDS = 5.0


def sleep_unless(seconds: float, should_stop=None,
                 poll: float = STOP_RECHECK_SECONDS) -> bool:
    """Sleep up to `seconds`, cutting it short as soon as `should_stop()` says so.

    Returns True when it returned early. A hold that outlasts an iteration takes
    this rather than `time.sleep`, because an uninterruptible one is a run that
    ignores its own brakes: with the fleet parked on the usage gate there is
    nobody left to notice `s` or the stop file, and the keypress reads as "the
    program hung" — which from outside is exactly what it looks like.
    `should_stop` None keeps the plain sleep, for callers with no stop channel
    to watch.
    """
    if should_stop is None:
        if seconds > 0:
            time.sleep(seconds)
        return False
    deadline = time.time() + seconds
    while True:
        if should_stop():
            return True
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        time.sleep(min(poll, remaining))


def confirm_stop_request(app=None, grace: float = STOP_GRACE_SECONDS,
                         poll: float = STOP_RECHECK_SECONDS) -> bool:
    """True if the pending stop request should be acted on now.

    The grace exists ONLY for a request this run's own `s` key made
    (`app.stop_requested_here`): a piped run, a CI run, or a script that wrote
    the stop file gets today's behaviour, unslowed — automation must not wait
    out a countdown for a keypress that is never coming. Intended as the one
    definition of "the user really meant it" for both runners; the parallel
    claim-loop adopts it when it grows a status line of its own.

    Cancelling means BOTH channels went quiet: pressing `s` again while a stop
    file is also present does not resume the run, because the file was never
    this key's to withdraw.
    """
    if (app is None or not getattr(app, "enabled", False)
            or not getattr(app, "stop_requested_here", False) or grace <= 0):
        return True
    app.update(phase="stopping")
    deadline = time.time() + grace
    while True:
        if pending_stop(app) is None:       # pressed `s` again — undo everything
            app.update(phase="idle", stop_pending="")
            app.note("stop cancelled — continuing")
            return False
        remaining = deadline - time.time()
        if remaining <= 0:
            return True
        app.note(f"stopping in {int(remaining) + 1}s — press s to cancel")
        time.sleep(min(poll, remaining))


def pause_requested(app=None) -> bool:
    """Is the `p` key holding this run at its iteration boundaries?

    The one reader of the flag, so the two runners cannot drift apart about what
    a pause is — and so a run with no status line at all (piped output, CI,
    --no-statusline) answers "no" without either of them testing for it.

    In-process only, and deliberately: unlike a stop there is no file channel to
    pause a fleet with. A pause is what somebody watching this terminal asks
    for, and the run they are watching is the one that holds.
    """
    return bool(getattr(app, "paused", False))


def wait_while_paused(app=None, should_stop=None,
                      poll: float = STOP_RECHECK_SECONDS) -> float:
    """Hold while `p` is up; returns how long the hold lasted (0.0 if never).

    Called at an iteration BOUNDARY, which is the only point where holding is
    free: nothing is in flight, no API request is open, and the files the loop
    reads next are nobody's to race — which is the whole point, since the run is
    usually held in order to edit one of them.

    `should_stop` is the run's stop channels (see `sleep_unless`): a hold with no
    timer on it must end the moment somebody asks the run to stop, or `s` would
    be answered only by whatever is holding it. It does not DECIDE the stop —
    the caller goes back to its loop head, which is the single place that knows
    what a request means (cancel grace for a key, none for a file).
    """
    if not pause_requested(app):
        return 0.0
    started = time.time()
    while pause_requested(app):
        if should_stop is not None and should_stop():
            break
        time.sleep(poll)
    return time.time() - started


def wait_for_stop_file_clear() -> None:
    """Hold a launch back while a stop request is still pending, and return once
    the sentinel is gone.

    The stop file is a request aimed at a *running* loop, and the run that obeys
    it claims it and clears it after cleanup. A launch that finds one already
    there therefore has no good way to start: removing it would silently cancel
    someone else's brake, and
    obeying it would exit before doing any work at all. So it waits instead —
    for the loop that owns the request to clear it on its way out, or for the
    user to delete the file by hand — and then starts clean. Ctrl+C interrupts
    the wait and stops the script.

    Not called on a dry run: that mode never touches the sentinel and reports it
    instead.
    """
    if not os.path.exists(STOP_FILE):
        return
    print(f"  ⏸ Stop file present at startup ({STOP_FILE}) — waiting for it to "
          f"go away before starting (remove it to begin; Ctrl+C to abort)…")
    waited = 0
    try:
        while os.path.exists(STOP_FILE):
            time.sleep(STOP_POLL_SECONDS)
            waited += STOP_POLL_SECONDS
            if waited % 60 == 0:
                # The wall clock, spelled with the stdlib rather than borrowed
                # from cyclecore's `_fmt_clock`: the same %H:%M:%S, and one
                # import fewer between a stop channel and a runner.
                print(f"    … still waiting ({waited // 60} min, now "
                      f"{time.strftime('%H:%M:%S')})", flush=True)
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)
    print("  ▶ Stop file removed — starting.")
