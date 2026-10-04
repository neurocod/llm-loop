"""
cyclecore.py - reusable engine behind autonomous Claude/Codex CLI loops.

This module holds everything that is *not* specific to one particular task:
command-line parsing and the generic `run_loop()` that ties a run together.
Turning ONE provider stream into what a watcher sees — and the single-stream
state that goes with it, the run's own rate-limit verdict included — is
`streamrender`, which this loop calls and does not own; the words that stream is
made of are `wire`, shared with the parallel runner's renderer.
Reading a quota and pausing on it are `usage`/`limits`, and when a run
pushes what it has committed is `gitpush` — both runners apply those and neither
owns them. What the run PRINTS is `console` — with the rotating mirror log, which
is the second copy of every printed line and therefore belongs to the printer;
what `--cost` reads back OUT of that log is `costlog`, together with the two
lines it reads, which this loop and `streamrender` print through it. What is
asked of a run from OUTSIDE it —
the `s` key and the `stop` sentinel, the `p` key's hold, and the reason a
runner reports on the way out — is `stopchannel`, a module of its own, because
the parallel runner and a host wrapper speak that vocabulary too and neither
should have to import the sequential loop to do so.

The only thing this engine does **not** decide is *what work to do each
iteration* — that is supplied by a `Driver`, and the whole vocabulary of work
(`Driver`, `AgentCommand`, `build_agent_argv`, `LoopStop`) is `agentwork`, its
own module, because the parallel runner and every host wrapper speak it too and
neither runner owns it. This one drives both shipped drivers:

  * runCycle.py    — a state-machine driver reading products/currentState.md;
  * runTranslate.py — a list driver translating files from products/list.md.

Everything below is lifted verbatim from the original single-file runCycle.py,
with the few state-specific pieces (which file to read, which prompt to send,
which model to pick) factored out into the Driver protocol.

Claude token-limit handling is driven by the account's real usage figures rather than
guessed from error counts, in two layers:

  * proactive — before each iteration, and again immediately after any non-zero
    Claude exit, the loop asks the account's process-wide policy
    (`limits.process_policy`, settled by the first run's Driver) for the current
    quota percentages and pauses if any watched one is at/over its ceiling. The
    reading lives in usage.py (UsageSource, an HTTP GET of the usage endpoint)
    and the pausing policy in limits.py (LimitPolicy and the ready-made
    SessionLimit / DayNightLimit / WeeklyLimit rules).
  * reactive — every Claude run streams its own rate-limit verdict, which the
    renderer picks out of the stream it is already parsing and latches for this
    iteration to read (`usage.RateLimitEvent` says what one is; the latch is
    `streamrender`'s, and the block above it there says why): a "rejected"
    means the wall was hit, and the loop waits that quota out even if the
    proactive reading was unavailable.
"""

import argparse
import math
import queue
import signal
import sys
import textwrap
import time
from typing import Callable, Optional

# `statusline` was imported inside `run_loop` for years, on the grounds that
# hoisting it would change what a bare `import llm_loop.cyclecore` drags in.
# Measured 2026-08-24 and false: importing a submodule runs the package's
# `__init__`, which imports it unconditionally, so it is already in
# `sys.modules` before this line is reached. The cycle the local import really
# was for is gone too — it does not import this module any more.
from . import (clispec, console, costlog, ctrlc, exitlog, operator,
               projectroot, providers, runlifecycle, statlog, statusline, stopchannel,
               termio, textwidth)
# The vocabulary of WORK — what a unit of it is, how it becomes an argv, and the
# Driver protocol that produces them — is `agentwork`, for the same reason as the
# rest of this list: both runners execute that contract and neither owns it.
#
# Three names, not the six that module exports: these are the ones this loop
# CALLS. The rest of the vocabulary reaches an embedder through `__init__`, which
# takes all six straight from `agentwork` — so re-importing them here would only
# create second addresses for names this file never mentions again.
from .agentwork import Driver, LoopStop, build_agent_argv
# What the run PRINTS, and the mirror log that is the second copy of it, are
# `console` (see its header for why those are one module). The line helpers are
# imported by name because this module calls them on nearly every path; the
# log's own names are reached through the module instead — `LOG_DIR`, the
# handler and the tee are configured and replaced, and a second binding here
# would be a second address for a test or a wrapper to miss.
from .console import (
    fmt_clock,
    fmt_left,
    fmt_moment,
    print_error,
    print_note,
    print_percents,
)
# The vocabulary of stopping and pausing is `stopchannel`, its own module,
# because both runners and a host wrapper speak it and none of them should have
# to import the sequential runner to do so.
#
# Reached through the module, never `from .stopchannel import …`: a name
# imported here would be a SECOND address for it, so a test (or a wrapper) that
# replaced `stopchannel.pause_requested` would change what the parallel runner
# does and not what this one does. One address, one thing to patch.
#
# The stale half of this rule is deleted rather than reworded: it used to add
# that `STOP_FILE` moves when --project-dir does, so a `from … import` would
# freeze it at the launch directory. That constant is gone — the sentinel is
# `stop_file_path()`, derived on read — and a function imported by name would
# NOT freeze. Only the second-address argument was ever load-bearing.
from .providers import prompt_on_stdin
# Rendering ONE provider stream into a terminal — starting the CLI, printing its
# events, and the single-stream state that only makes sense with one run in
# flight — is `streamrender`, its own module.
#
# Imported BY NAME, deliberately, and that is the load-bearing half of this
# move: 19 pins across six test files replace `cyclecore.run_claude_streaming` /
# `cyclecore.run_agent_streaming`, and a `from … import` is exactly what keeps
# those bites landing — the loop below calls the name in THIS module's globals,
# which is the name a `monkeypatch.setattr(cyclecore, …)` rebinds. Reaching
# through `streamrender.` instead stops the patch reaching the call, and what
# runs then is MEASURED rather than feared (2026-08-24, `try_patch` over both
# spellings with `providers.start_agent_process` guarded so nothing could
# actually start): every such pin goes red on `SystemExit: 2`, and on a machine
# where `claude` IS on PATH the test suite launches a real agent instead.
# `last_rate_limit_event` comes the same way and for the same reason
# (`tests/test_usage_limits.py` reads it back off this module).
from .streamrender import (
    last_rate_limit_event,
    run_agent_streaming,
    run_claude_streaming,
)
# The git-push policy is `gitpush`, its own module, for the same reason as the
# two above: both runners apply it and neither owns it. Only the per-iteration
# call is named here now — the exit push, the policy enum and its status label
# are the shared prologue/epilogue's (`runlifecycle`), which is the one place
# both runners open and close a run through.
from .gitpush import PushAbort, maybe_git_push
from .ownership import OwnerThread
# What is known about a quota lives in `usage`, so the limit rules (and the
# parallel runner) can use it without importing this one. Only the length of the
# window a token-limited run waits out is named here — the verdict's vocabulary
# is read where the stream is parsed (`streamrender`), and the latch holding the
# last such verdict lives there too.
from .usage import CLAUDE_SESSION_DURATION

# The usage-limit policy (which quota to gate on, what ceiling to allow,
# when to pause) lives in limits.py / usage.py, chosen per project via a Driver's
# `limit_policy` attribute — see Driver and run_loop.


# The Windows console is often cp1252 — switch output to UTF-8 so we can print Cyrillic.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# The project root lives in `projectroot`, its own leaf module, and is reached
# through it: `projectroot.project_dir()`. Not re-exported here on purpose —
# this runner is a CONSUMER of the root, not its home, and a `from … import`
# would put a second address on the same fact, which is what the split removed
# (see that module's header for the two mirrors it deleted). The package's front
# door still spells them `llm_loop.project_dir` / `set_project_root` /
# `find_project_root`, so an embedder's address is unchanged.


def parse_args(argv=None, *, prog: str = "runCycle.py",
               description: Optional[str] = None,
               extra_options: Optional[Callable[[argparse.ArgumentParser],
                                                None]] = None
               ) -> argparse.Namespace:
    """Command-line interface shared by every entry point. Every long option has
    a single-letter alias.

    `prog`/`description` let each entry script label its own --help text while
    reusing the exact same option set (so there is no duplicated argument code).

    `extra_options` is handed the parser once the shared options are on it, and
    is how a wrapper adds options of its own — its mode switches above all
    (`modeswitch.register`), which its argv scan reads to pick THIS parser and
    this parser then parses with the rest of the line.

    The options themselves are declared once for the whole family in `clispec`,
    which is also where the parallel runner's parser and the alias table
    `cmdline` strips an argv with come from. This function is the sequential
    mode's name for that parser, and nothing else.
    """
    return clispec.build_parser(clispec.SEQUENTIAL, prog=prog,
                                description=description,
                                extra_options=extra_options).parse_args(argv)


def is_report(args: argparse.Namespace) -> bool:
    """Does this command line ask for a report instead of a run?

    `--log`, `--cost`, `--stat` and `--cost-log` answer from the mirror log and exit
    before the prologue (see `run_loop`, the one reader that acts on it). A host
    asks the same question before it does anything a run needs and a report must
    not do — take the script lock, wait out a pending stop file, thaw a frozen
    kit — because a report is how progress is read DURING a live run, and those
    steps would act on that run's shared state. Asked of this function rather
    than spelled per host, so a report flag added here is one no host can miss.

    `--cost-log` counts by its PRESENCE, never its truthiness: an empty path read
    as "absent" started the loop. Read with getattr because the parallel parser
    declares none of these flags (a namespace from it is never a report).
    """
    return bool(getattr(args, "log", False)
                or getattr(args, "cost", False)
                or getattr(args, "stat", False)
                or getattr(args, "cost_log", None) is not None)


def _count_down_to(target_ts: float, should_stop=None) -> bool:
    """Idle until `target_ts`, printing what is left about once a minute.

    The body every timed wait shares; they differ only in the lines they print
    around it, which is why this holds none of them. Returns True when it left
    early because `should_stop()` or the run's Ctrl+C (`ctrlc.asked`) asked it
    to; the caller tells the two apart.
    """
    while True:
        now = time.time()
        remaining = target_ts - now
        if remaining <= 0:
            return False
        print(f"    … {fmt_left(remaining)} left (now {fmt_clock(now)})",
              flush=True)
        if stopchannel.sleep_unless(min(remaining, 60), should_stop):
            return True


def wait_until(target_ts: float, reason: str = None, should_stop=None) -> bool:
    """Sleep until wall-clock time reaches target_ts, printing a periodic countdown.

    Used after a probable token-limit error, or once the LimitPolicy decides the
    account's real usage figures leave no room: we idle until the 5-hour session
    window should have refreshed. `reason` overrides the default opening line.
    Inside a run Ctrl+C ends the wait and True is returned, the caller asking
    the run's Interrupt (`ctrlc`) next; outside one it is Python's
    KeyboardInterrupt.

    `should_stop` is the run's stop channels (see `stopchannel.sleep_unless`):
    a hold that can last hours must end the moment a human asks it to, and
    returns True when that is why it returned. Without one the wait runs to
    `target_ts` as before.
    """
    if reason is None:
        reason = ("Looks like the token limit is exhausted. Waiting until "
                  f"{fmt_clock(target_ts)} (until the 5-hour session window refreshes)…")
    print(f"  ⏳ {reason}")
    if _count_down_to(target_ts, should_stop):
        if ctrlc.current().requested:
            print("\nWait interrupted by user (Ctrl+C).")
        else:
            print("  ⏹ Stop requested — leaving the wait.")
        return True
    print("  ▶ The session window should have refreshed — continuing the loop.")
    return False


def _wait_clock(seconds: float) -> str:
    """HH:MM:SS, omitting zero hours; callers choose elapsed/remaining rounding."""
    minutes, seconds = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return (f"{hours:02d}:" if hours else "") + f"{minutes:02d}:{seconds:02d}"


def _interactive_start_wait(seconds: float, *, enabled: bool) -> bool:
    """Own the terminal only until startup; False asks for the plain fallback.

    The input thread only queues events. Deadline changes and painting belong
    to this thread, so a key cannot race a timeout or leave a second reader
    competing with the loop's status line. A monotonic deadline keeps clock
    corrections from changing the requested delay. Repaints bypass the mirror
    log through Terminal; redirected output only records the wait's start.
    """
    terminal = termio.terminal_for(enabled=enabled)
    reader = termio.TerminalInput()
    events = queue.Queue()
    started = time.monotonic()
    deadline = started + seconds
    restore_signals = []

    def terminate(signum, frame):
        # Turn default termination into unwinding so the reader restores cbreak
        # and the screen releases its rows. Custom/ignored signals stay owned
        # by the embedder. SIGKILL cannot be handled by any terminal owner.
        raise SystemExit(128 + signum)

    try:
        if not reader.usable() or isinstance(terminal, termio.NullTerminal):
            return False
        for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
            number = getattr(signal, name, None)
            if number is None or signal.getsignal(number) != signal.SIG_DFL:
                continue
            try:
                signal.signal(number, terminate)
            except (ValueError, OSError, RuntimeError):
                continue  # Only the main thread can install signal handlers.
            restore_signals.append(number)
        if not terminal.reserve(4):
            return False
        reader.start(events.put)
        geometry = None
        while True:
            now = time.monotonic()
            remaining = max(0, deadline - now)
            size = terminal.size()
            width = max(1, size[0] - 1)
            rows = [statusline.RULE_CHAR * width]
            rows += textwrap.wrap(
                f"Elapsed {_wait_clock(now - started)}  |  "
                f"Remaining {_wait_clock(math.ceil(remaining))}", width)
            rows += textwrap.wrap("Press a key (no Enter needed):", width)
            rows += textwrap.wrap(
                "[q] quit  [+] +1 min  [-] -1 min  [space] start now", width)
            # Re-reserve on resize so long hints never wrap into the scroll area.
            layout = (size, len(rows))
            if layout != geometry and terminal.reserve(len(rows)):
                geometry = layout
            if layout == geometry:
                terminal.paint(rows)
            if remaining <= 0:
                return True
            if ctrlc.current().requested:
                raise KeyboardInterrupt     # the runner's Ctrl+C: see the caller
            try:
                # Keypresses must not shift ticks away from second boundaries.
                until_tick = 1.0 - ((time.monotonic() - started) % 1.0)
                event = events.get(timeout=min(until_tick, remaining))
            except queue.Empty:
                continue
            if not isinstance(event, termio.Key):
                continue
            if event.char.lower() == "q":
                print("Start cancelled by user (q).")
                sys.exit(0)
            if event.char == " ":
                deadline = time.monotonic()
            elif event.char == "+":
                deadline += 60
            elif event.char == "-":
                deadline -= 60
    finally:
        reader.stop()
        terminal.release()
        for number in restore_signals:
            signal.signal(number, signal.SIG_DFL)


def wait_before_start(spec: str, *, interactive: bool = True) -> None:
    """Idle for the duration given by --start-in before the loop begins.

    Lets you launch the script and walk away; work kicks off after the delay.
    Before the loop's status line exists, this wait owns its own keys: q exits,
    +/- adjusts the deadline by a minute, and Space starts immediately. Ctrl+C
    exits with `ctrlc.EXIT_CODE` — a press on the runner's Interrupt or, outside
    a runner, a KeyboardInterrupt. Without a terminal, wait silently after the
    opening line.

    The parser has already refused a malformed `spec` (`clispec.duration`); the
    refusal here covers a namespace a host builds past it.
    """
    try:
        seconds = clispec.parse_duration(spec)
    except ValueError as e:
        print(f"Invalid --start-in value {spec!r}: {e}")
        sys.exit(2)
    if seconds <= 0:
        return
    target_ts = time.time() + seconds
    print(f"  ⏳ --start-in {spec}: waiting until {fmt_clock(target_ts)} before starting…",
          flush=True)
    try:
        if (not _interactive_start_wait(seconds, enabled=interactive)
                and stopchannel.sleep_unless(max(0, target_ts - time.time()))):
            raise KeyboardInterrupt         # the runner's Ctrl+C
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(ctrlc.EXIT_CODE)
    print("  ▶ Starting the loop.")


@stopchannel.stop_file_lifecycle()
@ctrlc.captured()
def run_loop(driver: Driver, args: argparse.Namespace,
             app_name: str = "runCycle", *, setup_logging: bool = True,
             wait_on_start: bool = True, progress=None) -> stopchannel.RunResult:
    """Drive the selected provider per `driver`, with all shared lifecycle
    machinery. This is the former runCycle.main(), generalised: the only thing
    that changed is that "read currentState.md and pick a prompt" became
    `driver.next_command()`, and the closing "Final state" line became
    `driver.final_summary()`.

    `progress` is the whole invocation's InvocationProgress, for a wrapper that
    makes several runner calls in one process (see run_parallel); left None, this
    call is the invocation and owns its own figures.

    Ctrl+C is the run's `ctrlc.Interrupt` for the whole call, not an exception:
    every wait ends on it, a turn in flight is ended (`run_agent_streaming`),
    and the loop leaves through `RunBoundary.interrupt` at its next look —
    the loop head (every wait goes back there), or straight after a turn it
    cut short.
    """
    pusher = OwnerThread("pusher", maxsize=1)
    push_abort = PushAbort()
    try:
        return _run_loop(driver, args, app_name, setup_logging=setup_logging,
                         wait_on_start=wait_on_start, progress=progress,
                         pusher=pusher, push_abort=push_abort)
    finally:
        # An exit before `_run_loop`'s boundary bypasses close_run and leaves
        # the owner started with nothing posted to it — the first push check is
        # posted inside the boundary, which hands every later ending to
        # close_run — so this only ends the thread. Idle owner close: 0.1 ms
        # median, 0.98 ms worst of 600 closes measured 2026-09-29; 0.5 s lets
        # close acquire its short lock.
        pusher.close(timeout=0.5)


def _run_loop(driver: Driver, args: argparse.Namespace, app_name: str, *,
              setup_logging: bool, wait_on_start: bool, progress,
              pusher: OwnerThread, push_abort: PushAbort) -> stopchannel.RunResult:
    # --cost/--stat: read the mirror log and exit, without touching
    # the loop, the tee, git, or the usage gate. BEFORE the prologue, whose first
    # act is to raise the tee and open an exit record: a report is not a run, and
    # it must neither be mirrored into the shared log nor leave a record behind.
    # It does need the project root, so it anchors it — `begin_run` anchors the
    # same value again, which is what makes doing it twice free.
    # --cost-log implies --cost: naming a log to read and getting a loop run
    # instead would be a silent misfire, and there is nothing else it could mean.
    # So it is the flag's PRESENCE (`is not None`) that picks the report, never
    # its truthiness: an empty path read as "absent" started the loop. The parser
    # refuses an empty path (`clispec.log_file`); a namespace built by hand that
    # carries one anyway is refused by `costlog.named_log` - first of all, so
    # that not even `--log` answers a run that asked for an impossible report.
    # Which flags make a report is `is_report`'s rule, the one hosts ask too.
    projectroot.set_project_root(getattr(args, "project_dir", None))
    cost_log = getattr(args, "cost_log", None)
    if cost_log is not None:
        costlog.named_log(cost_log)
    if is_report(args):
        # A path query must not open the log or start any part of the loop.
        if getattr(args, "log", False):
            print(console.log_file_path(app_name))
        elif getattr(args, "stat", False):
            statlog.report_stats(app_name, cost_log)
        else:
            costlog.report_costs(app_name, cost_log)
        return stopchannel.RunResult(stopchannel.RunStopReason.NO_WORK)

    # `runlifecycle.begin_run` is the prologue both runners share.
    ctx = runlifecycle.begin_run(driver, args, app_name, progress,
                                 setup_logging=setup_logging)
    provider, spec = ctx.provider, ctx.spec
    progress = ctx.progress
    # The live knobs (see RunSettings): read where they are USED, never
    # snapshotted into locals, so an edit through `ctx.registry` moves them mid-run.
    run_settings = ctx.settings
    # A finite cap at launch disables automatic quota waits and background
    # polling even if the live cap changes later. Provider quota support is
    # re-evaluated on each switch; bounded runs still show usage snapshots.
    ignore_usage_limits = args.max is not None
    dry_run = ctx.dry_run
    raw = args.raw
    start_in = args.start_in      # e.g. "29m" — delay before the loop starts

    # A stop request pending from another run: wait it out rather than consume
    # it, so this launch starts on a clean sentinel instead of stopping on its
    # first iteration boundary. Before --start-in: the point is to begin as soon
    # as the brake is off, not to burn the delay while it is still on.
    if not dry_run and wait_on_start:
        stopchannel.wait_for_stop_file_clear()

    if start_in and not dry_run:
        wait_before_start(start_in, interactive=ctx.status_enabled)

    session_start = time.time()   # start of the current 5-hour session window
    consecutive_errors = 0        # reset to 0 after any successful iteration
    # The selected provider's RunUsage (None without a usage endpoint), and the
    # two halves of it the gate below reads, split from it wherever it is set.
    usage = None
    usage_source, limit_policy = runlifecycle.usage_halves(usage)
    # Usage pairs and fallback session clocks belong to accounts, not the whole
    # mixed-provider run. Populate lazily: the launch default may never run.
    # The pairs are opened through the run's boundary, which holds every one
    # of them for the closing snapshots — all of them, not the selected one,
    # so every ending answers every `at start (…)`.
    usage_states = {}

    provider_refusals = {}
    quota_refresher = None
    last_git_push = 0.0           # owned by the pusher thread

    def push_turn() -> None:
        nonlocal last_git_push
        # A check still queued once the run's ending has begun is the exit
        # push's job, made right behind it: run, it is one more `git push`
        # the ending waits for (the parallel pump stops at the press instead,
        # its workers winding down after it).
        if boundary.closed:
            return
        last_git_push = maybe_git_push(run_settings.git_push, last_git_push,
                                       projectroot.project_dir(), abort=push_abort)

    if not dry_run:
        pusher.start()
    if ignore_usage_limits:
        print(f"  · usage limit policy: disabled (bounded run, "
              f"--max {run_settings.max_runs})")

    # One mailbox for the whole run: the console writes to it, the loop below
    # empties it into the next prompt, and run_agent_streaming lends it the
    # running turn's stdin. A dry run gets none — there is no agent to talk to.
    mailbox = None if dry_run else operator.Mailbox()
    app = runlifecycle.open_status(ctx, driver, job_count=1,
                                   messages=mailbox)
    # Only state-driven loops can name a breakpoint. Keep the collection local
    # to this invocation so a later runner call starts without old console input.
    from .breakpoints import Breakpoints
    state_name = getattr(driver, "state_name", None)
    breakpoints = Breakpoints(state_name) if callable(state_name) else None
    if breakpoints is not None:
        app.register_action(statusline.BreakpointAction(breakpoints))
    app.register_action(statusline.WeeklyLimitAction(
        lambda: None if ignore_usage_limits else limit_policy))

    iteration = 0
    completed = 0
    paused_since = 0.0            # start of the pause being held (0.0 = none)
    # Why a driver hook asked to hand control back, latched until the loop head
    # acts on it (see Driver.item_started / item_finished).
    handback = stopchannel.DriverHandback()
    stop_reason = stopchannel.RunStopReason.NO_WORK
    stop_file_noted = False       # dry-run: report the sentinel once, not per iteration
    dry_run_prompt_shown = False  # dry-run: show job 1's prompt once, not per pass
    # next_command may claim work. A pause at the quota gate must retain that
    # claim until it is launched instead of asking the driver for another item.
    pending_command = None
    refresh_pending_command = False

    interrupt = ctrlc.current()

    def stop_pending() -> bool:
        """Is Ctrl+C, a stop channel or a state breakpoint asking for this run
        right now?

        Handed to every hold that can outlast an iteration (the usage gate, the
        post-refusal wait). It only reports — the loop head is the single place
        that decides what a request means, cancel grace included.
        """
        return (interrupt.requested
                or stopchannel.pending_stop(app) is not None
                or (breakpoints is not None and breakpoints.reached() is not None))

    # The run's one epilogue boundary: usages open lazily inside the loop,
    # through it, and every ending from here on — the doors below, Ctrl+C,
    # an exception out of the driver — closes all of them, once. Outer to the
    # region, so the region is released before the housekeeping prints; the
    # doors inside the loop close down within it, as they always did.
    boundary = runlifecycle.RunBoundary(
        ctx, counts=lambda: (iteration, completed),
        mailbox=mailbox, pusher=pusher, push_abort=push_abort)
    with boundary, app:
        while True:
            # The caps are read LIVE (see RunSettings) and republished here, so an
            # edit made while the run is going is what the pinned row shows at
            # this boundary. Only republished: the edit already moved the knob
            # and the denominator (see `runlifecycle.knob_registry`).
            app.update(**progress.summary_fields(),
                       script_limits=ctx.registry.status_entries())
            # Ctrl+C before every other request: it is the operator leaving,
            # with no grace to count down and no state to consult.
            if interrupt.requested:
                boundary.interrupt()
            pending = stopchannel.pending_stop(app)
            if pending is stopchannel.StopSource.FILE and dry_run:
                # The sentinel is removed only after the outer application has
                # finished cleanup. A dry run must not claim it. `-d` is routinely
                # used to preview commands while a real loop is running — and that is
                # exactly when a
                # stop request is pending — so removing it here would silently cancel
                # someone else's stop, and the loop it was meant to halt would run on.
                # Report it and leave it for whoever it was written for. (There is
                # no key branch to write here: a dry run's status line is disabled,
                # so `s` is never even read.)
                if not stop_file_noted:
                    print("Stop file present — a real run would have waited for "
                          "it at startup, and stops here if it appears mid-run. "
                          "Left in place (a dry run never consumes it).")
                    stop_file_noted = True
            elif pending is not None and stopchannel.confirm_stop_request(app):
                if interrupt.requested:     # pressed during the grace
                    boundary.interrupt()
                # Reason first, line second: `commit_stop` hands the line back
                # instead of writing it (the order the parallel runner spends a
                # lock on — see there). What this branch gets out of that order
                # is narrower than what the fleet gets, and the difference is
                # worth naming here rather than inheriting the fleet's promise.
                # A refused write cannot lose the SENTINEL: `commit_stop` marked
                # it for cleanup before returning, and `stop_file_lifecycle`'s
                # finally still removes it while the exception unwinds. It does
                # lose the REASON: `stop_reason` is a local, nothing around this
                # loop catches, and a raising print leaves run_loop with no
                # RunResult at all (exitlog then prints "reason not recorded").
                # Left that way on purpose — one thread, no other worker to
                # mislead, so the exception IS this run's ending instead of a
                # wrong answer about why it ended.
                stop_reason, announcement = stopchannel.commit_stop(app, pending)
                print(announcement)
                app.update(phase="stopping")
                break
            # Cancelled inside the interactive grace — carry on with no trace.

            reached = breakpoints.reached() if breakpoints is not None else None
            if reached is not None:
                stop_reason = stopchannel.RunStopReason.BREAKPOINT
                print(f"Breakpoint reached: '{reached}'. Stopping cleanly.")
                app.update(phase="stopping")
                break

            # Git push policy: requested at the start of every iteration. A
            # slow push cannot hold the next agent; one pending check suffices
            # while the owner is busy, and the exit push checks again. The
            # agent may change HEAD while a push runs; a later check sends its
            # commits, while a history rewrite can make that push fail as
            # non-fast-forward (reported by git_push).
            if not dry_run:
                pusher.try_post(push_turn)

            max_runs = run_settings.max_runs
            if max_runs is not None and iteration >= max_runs:
                print(f"Iteration limit reached (--max-runs {max_runs}). Stopping.")
                stop_reason = stopchannel.RunStopReason.LIMIT_REACHED
                break

            # A pause one of the driver's per-item hooks asked for, acted on at
            # the boundary rather than where it was latched: an iteration that
            # has started is never cancelled (see Driver.item_started), and the
            # paths between a finished item and this line — a retry after a
            # non-zero exit, the wait after a rate-limit refusal — all come back
            # through here, so one check covers every one of them.
            #
            # HERE, and not further down, for the reason the cap above is here:
            # everything below this line either holds the run (the `p` key) or
            # waits on the account (the usage gate), and a run with no iteration
            # left to hold back must end instead of standing held for a boundary
            # that will never come. Held there, `p` would keep the caller from
            # getting control back until somebody released it, and the gate
            # would keep it until the quota window reset.
            if handback.pending:
                print(f"  ⏸ {handback.reason} — "
                      f"{stopchannel.DRIVER_HANDBACK_CLAUSE}")
                stop_reason = stopchannel.RunStopReason.DRIVER_PAUSE
                break

            # The `p` key's hold. AFTER the cap check, so a run that has no
            # iteration left to hold back ends instead of standing paused for a
            # boundary that will never come; BEFORE `driver.next_command()`,
            # which is what the key is for — the state file, the queue and the
            # tree are all quiet while it holds, so an edit made now is what the
            # next iteration reads.
            if stopchannel.pause_requested(app):
                if not paused_since:
                    # Announced once per pause, not once per pass: a stop
                    # requested and then withdrawn inside the hold sends the
                    # loop back through here, and that is the same pause.
                    paused_since = time.time()
                    print(f"\n  {statusline.PAUSE_GLYPH} Paused — press p to "
                          f"resume, s to stop, m to queue a note for the next "
                          f"iteration.")
                    exitlog.note(phase="paused (p key)",
                                 **progress.record_counts(iteration, completed))
                stopchannel.wait_while_paused(app, should_stop=stop_pending)
                # Back to the head rather than on: a stop pressed during the hold
                # is the head's to act on (with its cancel grace), and the caps
                # and quotas are re-read there.
                continue
            if paused_since:
                # Only here, where the loop is actually going on to work: a hold
                # the stop channels ended never reaches this line, and "pause
                # released" is not what happened to a run that is stopping.
                print(f"  ▶ Pause released after "
                      f"{statusline.format_elapsed(time.time() - paused_since)}.")
                paused_since = 0.0
                refresh_pending_command = pending_command is not None

            # Ask the driver what to do next. None => no more work (stop cleanly);
            # LoopStop => abort the run (e.g. an error state needing a human).
            try:
                if pending_command is None:
                    pending_command = driver.next_command()
                elif refresh_pending_command:
                    pending_command = driver.refresh_command(pending_command)
                refresh_pending_command = False
                command = pending_command
            except LoopStop as stop:
                print(stop.message)
                if stop.exit_code:
                    # The same epilogue every other ending gets — an exit code
                    # is not a licence to strand what the run committed or to
                    # swallow the notes nobody delivered. The door records the
                    # reason before the housekeeping.
                    boundary.exit(
                        stop.exit_code, ending="driver stopped the run",
                        reason=(f"the driver stopped the run (exit "
                                f"{stop.exit_code}): "
                                f"{stop.message.splitlines()[0]}"))
                stop_reason = stopchannel.RunStopReason.DRIVER_STOP
                break
            if command is None:
                print("No more work — stopping.")
                stop_reason = stopchannel.RunStopReason.NO_WORK
                break

            app.job(1).select(command.model)
            selected_provider = command.provider or ctx.provider
            if selected_provider != provider or provider not in usage_states:
                if provider in usage_states:
                    usage_states[provider] = (usage, session_start)
                provider = selected_provider
                spec = providers.provider_spec(provider)
                if provider not in usage_states:
                    usage_states[provider] = (
                        boundary.open_usage(driver, provider), time.time())
                usage, session_start = usage_states[provider]
                usage_source, limit_policy = runlifecycle.usage_halves(usage)
                ignore_usage_limits = (args.max is not None or usage_source is None
                                       or not spec.supports_usage_limits)
                if quota_refresher is not None:
                    quota_refresher.set_source(usage_source, limit_policy,
                                               provider=provider)
                app.update(provider=provider, quotas=[])
                statusline.push_quotas(app, usage_source, limit_policy)
                if not ignore_usage_limits:
                    if iteration == 0:
                        print_percents(f"  · {provider} usage limit policy: "
                                       f"{limit_policy.describe()}")
                    if not dry_run and quota_refresher is None:
                        quota_refresher = app.add_service(statusline.QuotaRefresher(
                            app, usage_source, limit_policy, provider=provider))

            # A refusal blocks this account's next command, not another
            # provider's next step. Keep the deadline if an operator interrupts
            # the wait and subsequently cancels the stop request.
            if provider in provider_refusals and not ignore_usage_limits:
                refusal, target_ts = provider_refusals[provider]
                app.update(phase="paused")
                wait_until(target_ts,
                           reason=f"Hit the {refusal.label} — this run was refused. "
                                  f"Waiting until {fmt_moment(target_ts)} for that "
                                  f"window to refresh…",
                           should_stop=stop_pending)
                app.update(phase="idle")
                if stop_pending() or stopchannel.pause_requested(app):
                    continue
                del provider_refusals[provider]
                usage_source.invalidate()
                statusline.push_quotas(app, usage_source, limit_policy)
                if refusal.limit_type == "five_hour":
                    session_start = time.time()
                consecutive_errors = 0

            # Gate the account that will actually execute this command. Keep the
            # pending claim if the operator pauses or cancels a stop in this wait.
            if not dry_run and not ignore_usage_limits:
                app.update(phase="waiting")
                app.job(1).update(waiting_for_usage=True)
                try:
                    paused, session_start = limit_policy.check_and_wait(
                        usage_source, session_start, should_stop=stop_pending,
                        report_status=iteration == 0)
                finally:
                    app.job(1).update(waiting_for_usage=False)
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if paused:
                    consecutive_errors = 0
                if stop_pending() or stopchannel.pause_requested(app):
                    continue

            # A breakpoint may have been entered after selecting the command,
            # including during a quota wait — and Ctrl+C anywhere since the
            # head, the opening usage snapshot included. Return to the boundary
            # before launch.
            if interrupt.requested or (breakpoints is not None
                                       and breakpoints.reached() is not None):
                continue

            # Notes typed while nothing was running (or while the transport was
            # off) ride this prompt — see Mailbox.splice for the ordering.
            if mailbox is not None:
                spliced, notes = mailbox.splice(command.prompt)
                if notes:
                    command = command._replace(prompt=spliced)
                    for note in notes:
                        print_note(note)

            pending_command = None
            iteration += 1
            state_label = command.label or "(no label)"
            # Show the model this iteration will use right in the header, so the
            # per-iteration model is visible up front (an empty command.model means
            # no --model flag — the CLI falls back to its own configured default).
            model_label = f"{provider}/{command.model or 'cli default'}"
            # Same three calls the parallel workers make, on this run's one Job:
            # the Job clock times THIS iteration, the run clock (latched once)
            # times the whole run.
            started_at = time.time()
            iteration_clock = time.monotonic()
            app.mark_run_started(started_at)
            # The Job bumps its own counter (no `iteration=`): the local counter
            # restarts with every runner call, and pinning the row to it is what
            # kept a periodic run's job rows at 1.
            app.job(1).start(item=state_label, model=command.model,
                             prompt=command.prompt, now=started_at)
            # Read only by a row with no total (InvocationProgress.summary_fields).
            progress.note_iteration()
            counts = progress.record_counts(iteration, completed)
            app.update(**progress.summary_fields(), phase="running")
            # What a post-mortem needs from a run that never got to write an
            # ending: which item it was on when it stopped existing.
            exitlog.note(phase=f"iteration {iteration} — {state_label}",
                         **counts)
            # Through the module, unlike the line helpers above: this is the one
            # printed line the run must be able to READ BACK (`costlog` parses
            # iteration 1's header as a run boundary, hence its wording comes
            # from there), and the pins that capture printed lines replace
            # `console.print_markup`. A binding of our own here would be a third
            # address none of them reaches, so the header would sail past every
            # one of them uncaptured.
            separator = "_" * max(1, min(65, textwidth.terminal_columns()
                                         - textwidth.LINE_RIGHT_MARGIN))
            header = costlog.iteration_header(iteration)
            console.print_markup(
                f"{separator}\n{header} [{state_label} · {model_label}]",
                f"[dim]{separator}[/]\n[bold cyan]{header}[/] "
                f"[dim]\\[{state_label} · {model_label}][/]",
            )
            if not dry_run:
                console.record_timing(statlog.iteration_started(iteration, state_label))

            cmd = build_agent_argv(command, provider)
            if dry_run:
                print("DRY-RUN:", " ".join(cmd))
                if prompt_on_stdin(provider):
                    # The argv above is complete but not self-contained: the
                    # prompt travels on stdin, so the preview has to show it
                    # separately or it shows a command with no task in it.
                    print("STDIN:", command.prompt)
                if not dry_run_prompt_shown:
                    # The argv line above is what will be executed, but for
                    # claude the whole prompt sits inside one joined `-p …`
                    # token and is unreadable — and reading the prompt is what a
                    # dry run is for. Printed once, for job 1.
                    dry_run_prompt_shown = True
                    print(statusline.format_prompt_block(
                        job_id=1, label=state_label, prompt=command.prompt,
                        width=textwidth.screen_width()))
                # looping forever in dry-run is pointless — nothing is actually done,
                # so the driver would keep handing back the same first unit of work.
                if run_settings.max_runs is None:
                    print("(dry-run without --max-runs: running a single iteration and exiting)")
                    stop_reason = stopchannel.RunStopReason.DRY_RUN
                    break
                continue

            # The start-of-item hook, with the command as it will be sent (the
            # notes above are already spliced in). It cannot cancel this
            # iteration — the loop head is where a pause is acted on — so a
            # reason returned here is latched and this turn still runs.
            handback.latch(driver.item_started(command))

            with statusline.describing(app.job(1)):
                if provider == "claude":
                    returncode = run_claude_streaming(
                        cmd, raw, partial=True, prompt=command.prompt,
                        mailbox=mailbox)
                else:
                    returncode = run_agent_streaming(
                        cmd, provider, raw, partial=False, prompt=command.prompt,
                        mailbox=mailbox)
            app.job(1).finish()
            # A turn Ctrl+C cut short is not an error to count or retry. One
            # that got through before the press is credited; the head ends it.
            if interrupt.requested and returncode != 0:
                boundary.interrupt()
            timing = statlog.iteration_finished(
                iteration, returncode, time.monotonic() - iteration_clock)
            console.record_timing(timing)
            print(timing)
            app.update(phase="idle")

            if returncode == 0:
                consecutive_errors = 0
                completed += 1
                exitlog.note(**progress.record_counts(iteration, completed))
                driver.on_success(returncode)
                # on_success recorded the item, so the driver's own count now
                # says how far the invocation has got.
                remaining = driver.pending_total()
                if remaining is not None:
                    progress.note_remaining(remaining)
                app.update(**progress.summary_fields())

            # The end-of-item hook: after on_success, so the driver's own queue
            # is up to date, and after the outcome either way — a failed
            # iteration is still an iteration whose side effects are on disk.
            # Called even after `item_started` asked for a hand-back, as the
            # fleet calls it; the latch keeps the FIRST reason (see
            # DriverHandback).
            handback.latch(driver.item_finished(command, returncode))

            # Preserve the wire verdict even when the quota endpoint has no
            # figures. A refused final turn may still exit 0 and advance the
            # state, so defer its wait until this provider is selected again.
            refusal = last_rate_limit_event() if provider == "claude" else None
            if (not ignore_usage_limits and refusal is not None
                    and refusal.status == "rejected" and not handback.pending):
                # +5s so we come back after the reset, not exactly on it.
                provider_refusals[provider] = (
                    refusal, (refusal.resets_at
                              or time.time() + CLAUDE_SESSION_DURATION) + 5)
                continue

            if returncode == 0:
                continue

            # Non-zero exit — the cause is ambiguous (a network blip / one-off CLI
            # hiccup, or the session's token limit). Rather than guessing from a
            # second consecutive error, ask the account directly: read the usage
            # figures right away and let the real Current-session percentage decide.
            consecutive_errors += 1
            elapsed = time.time() - session_start
            print_error(f"{spec.display_name} exited with code {returncode} "
                        f"(error #{consecutive_errors} in a row).")

            if not ignore_usage_limits and not handback.pending:
                # The pause exclusion is the refusal branch's (see there): this
                # gate can hold for a whole window, and the head is one `continue`
                # away from ending the run. The error itself is still counted and
                # printed above — only the waiting is skipped.
                app.update(phase="waiting")
                app.job(1).update(waiting_for_usage=True)
                try:
                    paused, session_start = limit_policy.check_and_wait(
                        usage_source, session_start, note=" (checked after error)",
                        should_stop=stop_pending, report_status=False)
                finally:
                    app.job(1).update(waiting_for_usage=False)
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if paused:
                    consecutive_errors = 0  # fresh window — start counting errors anew
                    continue
                if stop_pending():
                    # Kept separate from the reset above: a request that is
                    # withdrawn inside the grace must not have zeroed the
                    # five-errors-in-a-row brake on its way past, or a genuinely
                    # broken provider gets to loop for free.
                    continue

            # Session is under the limit — this was a transient failure, not token
            # exhaustion. Retry, but don't spin forever if something is truly broken.
            if consecutive_errors < 5:
                if usage_source is None:
                    print("  ↻ Provider quota status is unavailable — retrying immediately.")
                else:
                    print("  ↻ Session under the allowed limit — likely transient. "
                          "Retrying immediately.")
                continue
            else:
                quota_state = ("with provider quota status unavailable"
                               if usage_source is None
                               else "with the session under the allowed limit")
                print(f"  ⚠ {consecutive_errors} errors in a row {quota_state} after "
                      f"{int(elapsed // 60)} min. Stopping.")
                # The epilogue matters most here: a run that gave up after five
                # failures may have committed four good iterations, and the
                # notes an operator typed at the console are the likeliest
                # explanation of what went wrong.
                boundary.exit(returncode, ending="provider errors in a row",
                              reason=(f"{consecutive_errors} provider errors "
                                      f"in a row (last exit code {returncode})"))

    with boundary:
        # This run's own closing line, if the driver has one (e.g. "Final
        # state: …"). Before the shared epilogue, which is housekeeping: the
        # run reports on its work first, then the run is closed down.
        summary = driver.final_summary()
        if summary:
            print(f"\n{summary}")
        # The exit push follows every check queued while agents were running.
        return boundary.end(
            stopchannel.RunResult(stop_reason, iteration, completed))
