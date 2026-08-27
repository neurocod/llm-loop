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
what `--cost` reads back OUT of that log is still here, because the lines it
parses are emitted here. What is asked of a run from OUTSIDE it —
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
    Claude exit, the loop asks the Driver's `limit_policy` for the current
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
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Union

# `limits` and `statusline` were imported inside `run_loop` for years, on the
# grounds that hoisting them would change what a bare `import llm_loop.cyclecore`
# drags in. Measured 2026-08-24 and false: importing a submodule runs the
# package's `__init__`, which imports both unconditionally, so they are already
# in `sys.modules` before this line is reached. The cycle the local import
# really was for is gone too — neither module imports this one any more.
from . import (clispec, console, exitlog, limits, operator,
               projectroot, providers, runlifecycle, statusline, stopchannel,
               textwidth)
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
from .providers import prompt_on_stdin, usage_source_for
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
from .gitpush import maybe_git_push
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


# Per-run cost accounting parsed straight back out of the mirror log: every run's
# first iteration logs a "=== Iteration 1 ===" header (see run_loop) and every
# successful iteration logs a "done (… c, $…)" line. Summing the dollar figures
# between headers reconstructs per-run spend with no extra bookkeeping.
#
# Only the FIRST of those two lines is emitted next to its pattern (run_loop,
# below). The second is `streamrender._render_claude_event`'s "result" branch.
# Two files, so they CAN drift: this pattern is what a re-worded "done" line has
# to be checked against. `_render_codex_event` prints "· done (tokens: …)",
# which this deliberately does not match — codex reports tokens, not dollars, so
# a codex run has no per-session spend to total up.
# Note "=== Iteration 1 ===" matches only
# iteration 1 (the "1 ===" boundary rules out "11", "12", …), so each match is a
# genuine run boundary.
_SESSION_RE = re.compile(r"=== Iteration 1 ===")
_COST_RE = re.compile(r"done \(\s*[\d.]+ c,\s*\$([\d.]+)\)")


def report_costs(app_name: str = "runCycle",
                 path: Optional[Union[str, Path]] = None) -> None:
    """Print per-session (per-run) cost totals parsed from the mirror log, then
    exit — the standalone counterpart reached via the --cost flag.

    A "session" is one run of the loop, delimited by its "=== Iteration 1 ==="
    header; within it every "done (… c, $…)" line contributes its dollar cost. We
    print a line per session, a grand total, and how full the log is against the
    rotation limit (LOG_MAX_BYTES). With no `path`, the log is resolved via
    log_file_path(app_name), so --cost reports on the very log this entry point
    writes — under the project root already chosen by --project-dir.

    `path` (the --cost-log flag) names a log this entry point does NOT write:
    a rotated backup (`<app>-<project>.log.1`) or a copy taken elsewhere. It is
    the one case app_name cannot reach, since rotation renames files out from
    under log_file_path.
    """
    path = Path(path) if path else console.log_file_path(app_name)
    # Always name the log we are reading, so an empty report is unambiguous
    # (right file, no data) rather than looking like a silent failure.
    print(f"Reading mirror log: {path}")
    sessions = []  # list of (header, total_cost, count)
    header = None
    total = 0.0
    count = 0

    def flush():
        if header is not None:
            sessions.append((header, total, count))

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if _SESSION_RE.search(line):
                    flush()
                    header = line.strip()
                    total = 0.0
                    count = 0
                else:
                    m = _COST_RE.search(line)
                    if m and header is not None:
                        total += float(m.group(1))
                        count += 1
    except FileNotFoundError:
        print(f"No mirror log at {path} yet — nothing to report.")
        return
    flush()

    grand = 0.0
    grand_count = 0
    for i, (h, t, c) in enumerate(sessions, 1):
        print(f"Session {i}: {c} costs, ${t:.4f}  | {h}")
        grand += t
        grand_count += c

    print("-" * 60)
    print(f"TOTAL: {len(sessions)} sessions, {grand_count} costs, ${grand:.4f}")
    if not sessions:
        # The log exists but held no run boundaries / cost lines. Point at the
        # likely cause rather than leaving a bare zero.
        print("  (log has no '=== Iteration 1 ===' / '· done (… c, $…)' lines — "
              "no completed billed iterations recorded here)")

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    limit = console.LOG_MAX_BYTES
    pct = size / limit * 100 if limit else 0.0
    print(f"LOG: {size / 1024 / 1024:.2f} / {limit / 1024 / 1024:.0f} MB "
          f"({pct:.1f}% full, rotates at 100%)")
    print(f"     {path}")


class ConsumedByWrapperAction(argparse.Action):
    """An option this parser only DOCUMENTS — the wrapper reads it out of argv
    itself, before parsing (see Driver.add_cli_options).

    Reaching the parser therefore means the wrapper's own scan did not match
    what was typed. argparse would otherwise accept it (an abbreviation like
    `--grow` for `--grow-kit` is one this parser resolves and that scan does
    not), store it in a namespace nobody reads, and run the DEFAULT mode — a
    flag that appears to work and silently does nothing. Saying so is the whole
    job of this action.

    Declare a value-taking switch with `nargs=1, metavar="N"` so --help shows
    its argument; the default `nargs=0` documents a bare flag.
    """

    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs,
                         default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # `option_string` is the canonical spelling, not what was typed — which
        # is the point: it names the option the abbreviation resolved to, and
        # the one to type instead.
        parser.error(
            f"{option_string} never reached the argv scan that reads it, "
            f"which runs before this parser. Spell it out in full; "
            f"abbreviations are resolved here and are invisible there. (If it "
            f"WAS spelled in full, that scan and this help have drifted "
            f"apart.)")


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
    is how a wrapper documents the flags IT consumes before this parser ever
    runs (a mode switch that decides which parser to use cannot be one of this
    parser's options). Passing the parser rather than a block of help text is
    what keeps such a flag formatted, aligned and grouped like every other
    option — and lets the hook add a real, parsed option when the flag is not
    one the wrapper strips.

    The options themselves are declared once for the whole family in `clispec`,
    which is also where the parallel runner's parser and the alias table
    `cmdline` strips an argv with come from. This function is the sequential
    mode's name for that parser, and nothing else.
    """
    return clispec.build_parser(clispec.SEQUENTIAL, prog=prog,
                                description=description,
                                extra_options=extra_options).parse_args(argv)


def parse_duration(text: str) -> float:
    """Parse a duration like '29m', '1h', '90s', '1h30m' into seconds.

    A bare number is treated as minutes ('29' == '29m'). Raises ValueError on
    anything it can't make sense of.
    """
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    if text.isdigit():  # bare number — minutes
        return int(text) * 60

    units = {"h": 3600, "m": 60, "s": 1}
    total = 0.0
    matched = False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
        total += float(value) * units[unit]
        matched = True
    if not matched:
        raise ValueError(f"cannot parse duration: {text!r}")
    return total


def _count_down_to(target_ts: float, should_stop=None) -> bool:
    """Idle until `target_ts`, printing what is left about once a minute.

    The body every timed wait shares; they differ only in the lines they print
    around it, which is why this holds none of them. Returns True when it left
    early because `should_stop()` asked it to. Ctrl+C ends the process from
    here, as it always did.
    """
    try:
        while True:
            now = time.time()
            remaining = target_ts - now
            if remaining <= 0:
                return False
            print(f"    … {fmt_left(remaining)} left (now {fmt_clock(now)})",
                  flush=True)
            if stopchannel.sleep_unless(min(remaining, 60), should_stop):
                return True
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)


def wait_until(target_ts: float, reason: str = None, should_stop=None) -> bool:
    """Sleep until wall-clock time reaches target_ts, printing a periodic countdown.

    Used after a probable token-limit error, or once the LimitPolicy decides the
    account's real usage figures leave no room: we idle until the 5-hour session
    window should have refreshed. `reason` overrides the default opening line.
    Ctrl+C interrupts the wait and stops the script.

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
        print("  ⏹ Stop requested — leaving the wait.")
        return True
    print("  ▶ The session window should have refreshed — continuing the loop.")
    return False


def wait_before_start(spec: str) -> None:
    """Idle for the duration given by --start-in before the loop begins.

    Lets you launch the script and walk away; work kicks off after the delay.
    Ctrl+C interrupts the wait and stops the script.

    The only wait that runs its clock out (no `should_stop`), because there is
    nothing here to stop yet: this is before the status line exists, so there is
    no `s` key to press, and a sentinel that appears meanwhile was waited out
    just above and is honoured at the first iteration boundary anyway.
    """
    try:
        seconds = parse_duration(spec)
    except ValueError as e:
        print(f"Invalid --start-in value {spec!r}: {e}")
        sys.exit(2)
    if seconds <= 0:
        return
    target_ts = time.time() + seconds
    print(f"  ⏳ --start-in {spec}: waiting until {fmt_clock(target_ts)} before starting…")
    _count_down_to(target_ts)
    print("  ▶ Starting the loop.")


@stopchannel.stop_file_lifecycle()
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
    """
    # --cost: report per-run spend from the mirror log and exit, without touching
    # the loop, the tee, git, or the usage gate. BEFORE the prologue, whose first
    # act is to raise the tee and open an exit record: a report is not a run, and
    # it must neither be mirrored into the shared log nor leave a record behind.
    # It does need the project root, so it anchors it — `begin_run` anchors the
    # same value again, which is what makes doing it twice free.
    # --cost-log implies --cost: naming a log to read and getting a loop run
    # instead would be a silent misfire, and there is nothing else it could mean.
    projectroot.set_project_root(getattr(args, "project_dir", None))
    cost_log = getattr(args, "cost_log", None)
    if getattr(args, "cost", False) or cost_log:
        report_costs(app_name, cost_log)
        return stopchannel.RunResult(stopchannel.RunStopReason.NO_WORK)

    # `runlifecycle.begin_run` is the prologue both runners share.
    ctx = runlifecycle.begin_run(driver, args, app_name, progress,
                                 setup_logging=setup_logging)
    provider, spec = ctx.provider, ctx.spec
    progress, owns_progress = ctx.progress, ctx.owns_progress
    # The live knobs (see RunSettings): read where they are USED, never
    # snapshotted into locals, so the status line's editor can move them mid-run.
    run_settings = ctx.settings
    # When a finite iteration cap is given (-m/--max-runs) the run is short and
    # bounded on purpose, so the usage-limit machinery (the LimitPolicy
    # pause-on-limit logic) is skipped — we just run the requested iterations
    # without ever waiting out a window. Decided once, from the value the run was
    # LAUNCHED with: it also governs whether this run talks to the usage endpoint
    # at all, which is a property of the invocation, not of the current cap.
    ignore_usage_limits = (args.max is not None
                           or not spec.supports_usage_limits)
    dry_run = ctx.dry_run
    raw = args.raw
    start_in = args.start_in      # e.g. "29m" — delay before the loop starts
    if not console.RICH_AVAILABLE:
        print("  · Markdown rendering is off (the 'rich' library is missing). "
              "Enable it with:")
        print(f"      {sys.executable} -m pip install rich")

    # A stop request pending from another run: wait it out rather than consume
    # it, so this launch starts on a clean sentinel instead of stopping on its
    # first iteration boundary. Before --start-in: the point is to begin as soon
    # as the brake is off, not to burn the delay while it is still on.
    if not dry_run and wait_on_start:
        stopchannel.wait_for_stop_file_clear()

    if start_in and not dry_run:
        wait_before_start(start_in)

    session_start = time.time()   # start of the current 5-hour session window
    consecutive_errors = 0        # reset to 0 after any successful iteration
    usage_source = usage_source_for(provider)
    limit_policy = None
    if usage_source is not None:
        limit_policy = driver.limit_policy or limits.default_policy(provider)
    last_git_push = 0.0           # epoch time of the last `git push` (0 = never)
    if ignore_usage_limits:
        print(f"  · usage limit policy: disabled (bounded run, "
              f"--max {run_settings.max_runs})")
    else:
        print_percents(f"  · usage limit policy: {limit_policy.describe()}")

    # Bookend the run with a usage snapshot (the policy's watched quotas) so each
    # run records where it started; the matching end-of-run snapshot is below.
    if not dry_run and usage_source is not None:
        limit_policy.log_snapshot(usage_source, "at start (iteration 1)")

    # The pinned status area. A Job is the unit of display in both runners, so
    # the sequential loop is a run with exactly one Job — no branch anywhere in
    # the status line separates it from `-j N`. Disabled it is a Null object, so
    # every call below stays a no-op and the loop behaves exactly as before.
    settings = runlifecycle.script_settings(
        run_settings, progress if owns_progress else None)
    # A driver that knows how much work it has is the source of truth for it:
    # the summary row then counts items FINISHED out of that total, not
    # iterations, so a retried item is not progress and a preflight that strikes
    # finished ones is. Asked of the driver (Driver.pending_total) rather than
    # read off a list file, so a queue that is not a list — the kit-promotion
    # pass draining its requests folder — gets a denominator too instead of
    # counting bare iterations into a row that reads "iter 1" forever.
    tracked_total = driver.pending_total()
    if tracked_total is not None:
        progress.track_total(tracked_total)
    # One mailbox for the whole run: the console writes to it, the loop below
    # empties it into the next prompt, and run_agent_streaming lends it the
    # running turn's stdin. A dry run gets none — there is no agent to talk to.
    mailbox = None if dry_run else operator.Mailbox()
    app = statusline.StatusApp(
        # From the invocation's pool: a wrapper's next runner call resumes this
        # row instead of starting a fresh Job at iteration 1.
        status=statusline.LoopStatus(jobs=progress.jobs(1)),
        settings=settings,
        messages=mailbox,
        enabled=not dry_run and not getattr(args, "no_statusline", False))
    app.update(
        provider=provider,
        **progress.summary_fields(),
        # Only a list driver has a pick order; read it defensively so any other
        # Driver simply reports no `rand` marker.
        random_order=str(getattr(driver, "pick_order", "")) == "random",
        script_limits=settings.status_entries(),
    )

    iteration = 0
    completed = 0
    paused_since = 0.0            # start of the pause being held (0.0 = none)
    # Why a driver hook asked to hand control back, latched until the loop head
    # acts on it (see Driver.item_started / item_finished).
    handback_reason = None
    stop_reason = stopchannel.RunStopReason.NO_WORK
    stop_file_noted = False       # dry-run: report the sentinel once, not per iteration
    dry_run_prompt_shown = False  # dry-run: show job 1's prompt once, not per pass

    def stop_pending() -> bool:
        """Is either stop channel asking for this run right now?

        Handed to every hold that can outlast an iteration (the usage gate, the
        post-refusal wait). It only reports — the loop head is the single place
        that decides what a request means, cancel grace included.
        """
        return stopchannel.pending_stop(app) is not None

    with app:
        if usage_source is not None:
            # Inside `with`, not before it: push_quotas is silent until start()
            # has marked the app enabled, so priming it earlier left requirement
            # #1 — the provider's live figures — blank until the first limit
            # check (i.e. for the whole run when the checks are skipped). The
            # reading itself is already paid for by the start-of-run snapshot
            # above, so this costs no round-trip.
            statusline.push_quotas(app, usage_source, limit_policy)
            if not ignore_usage_limits:
                # Only for a run that is allowed to talk to the usage endpoint at
                # all: the poll forces a FRESH reading (cache_value=False), so on
                # a bounded `-m N` run — whose whole point is not to touch the
                # usage machinery — it would be an unsolicited HTTP GET every
                # interval (a codex app-server call every 300 s).
                app.add_service(statusline.QuotaRefresher(
                    app, usage_source, limit_policy, provider=provider))
        while True:
            # The caps are read LIVE (see RunSettings) and republished here, so an
            # edit made while the run is going is what the pinned row shows at
            # this boundary. Only republished: the edit itself already moved both
            # the knob and the row's denominator, in the one setter both runners
            # register (`runlifecycle.script_settings`). Re-assigning the
            # denominator here as well would be a second writer for one number,
            # and the parallel runner — which has no boundary to re-read at —
            # could not have it.
            app.update(**progress.summary_fields(),
                       script_limits=settings.status_entries())
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

            # Git push policy: evaluated at the start of every iteration.
            if not dry_run:
                last_git_push = maybe_git_push(run_settings.git_push,
                                               last_git_push, projectroot.project_dir())

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
            if handback_reason is not None:
                print(f"  ⏸ {handback_reason} — "
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
                    exitlog.note(phase="paused (p key)", iterations=iteration,
                                 completed=completed)
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

            # Proactive limit check: read the real Current-session usage from the
            # account and pause cleanly between iterations if it is already at/over
            # the threshold, instead of running an iteration that would hit the wall.
            if not dry_run and not ignore_usage_limits:
                app.update(phase="waiting")
                paused, session_start = limit_policy.check_and_wait(
                    usage_source, session_start, should_stop=stop_pending)
                # The check just paid for a usage reading; publishing it here is
                # what puts the provider's live limits on the status row without
                # a second HTTP round-trip (the UsageSource cache serves it).
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if paused:
                    consecutive_errors = 0  # fresh window — start counting errors anew
                if stop_pending() or stopchannel.pause_requested(app):
                    # Asked to stop or to hold while parked on the limit. Back to
                    # the loop head, which owns both decisions (a key press still
                    # gets its cancel grace, a stop file still ends the run at
                    # once) — deciding here would be a second, divergent copy.
                    #
                    # The pause half is not symmetry: the gate is the longest
                    # hold in the engine, so it is exactly where somebody watching
                    # a run that is doing nothing reaches for `p`. Without this
                    # re-read the flag was set, the gate did not watch it, and the
                    # window opening started a full iteration under a row that
                    # already said PAUSED.
                    continue

            # Ask the driver what to do next. None => no more work (stop cleanly);
            # LoopStop => abort the run (e.g. an error state needing a human).
            try:
                command = driver.next_command()
            except LoopStop as stop:
                print(stop.message)
                if stop.exit_code:
                    # The reason FIRST, then the housekeeping: this ending is
                    # already an abnormal one, and the record of why must not
                    # depend on a push or a usage query surviving. Then the same
                    # epilogue every other ending gets — an exit code is not a
                    # licence to strand what the run committed or to swallow the
                    # notes nobody delivered.
                    exitlog.set_reason(
                        f"the driver stopped the run (exit {stop.exit_code}): "
                        f"{stop.message.splitlines()[0]}",
                        iterations=iteration, completed=completed)
                    runlifecycle.close_run(
                        ctx, usage_source=usage_source,
                        limit_policy=limit_policy,
                        snapshot_label="at end (driver stopped the run)",
                        mailbox=mailbox)
                    sys.exit(stop.exit_code)
                stop_reason = stopchannel.RunStopReason.DRIVER_STOP
                break
            if command is None:
                print("No more work — stopping.")
                stop_reason = stopchannel.RunStopReason.NO_WORK
                break

            # Notes typed while nothing was running (or while the transport was
            # off) ride this prompt — see Mailbox.splice for the ordering.
            if mailbox is not None:
                spliced, notes = mailbox.splice(command.prompt)
                if notes:
                    command = command._replace(prompt=spliced)
                    for note in notes:
                        print_note(note)

            iteration += 1
            state_label = command.label or "(no label)"
            # Show the model this iteration will use right in the header, so the
            # per-iteration model is visible up front (an empty command.model means
            # no --model flag — the CLI falls back to its own configured default).
            model_label = command.model or "cli default"
            # Same three calls the parallel workers make, on this run's one Job:
            # the Job clock times THIS iteration, the run clock (latched once)
            # times the whole run.
            started_at = time.time()
            app.mark_run_started(started_at)
            # The Job bumps its own counter (no `iteration=`): the local counter
            # restarts with every runner call, and pinning the row to it is what
            # kept a periodic run's job rows at 1.
            app.job(1).start(item=state_label, model=command.model,
                             prompt=command.prompt, now=started_at)
            if tracked_total is None:
                progress.note_iteration()
            app.update(**progress.summary_fields(), phase="running")
            # What a post-mortem needs from a run that never got to write an
            # ending: which item it was on when it stopped existing.
            exitlog.note(phase=f"iteration {iteration} — {state_label}",
                         iterations=iteration, completed=completed)
            # Through the module, unlike the line helpers above: this is the one
            # printed line the run must be able to READ BACK (`report_costs`
            # parses "=== Iteration 1 ===" as a run boundary), and the pins that
            # capture printed lines replace `console.print_markup`. A binding of
            # our own here would be a third address none of them reaches, so the
            # header would sail past every one of them uncaptured.
            console.print_markup(
                f"\n=== Iteration {iteration} === [{state_label} · {model_label}]",
                f"\n[bold cyan]=== Iteration {iteration} ===[/] "
                f"[dim]\\[{state_label} · {model_label}][/]",
            )

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
            handback_reason = handback_reason or driver.item_started(command)

            if provider == "claude":
                returncode = run_claude_streaming(
                    cmd, raw, partial=True, prompt=command.prompt,
                    mailbox=mailbox)
            else:
                returncode = run_agent_streaming(
                    cmd, provider, raw, partial=False, prompt=command.prompt,
                    mailbox=mailbox)
            app.job(1).finish()
            app.update(phase="idle")

            if returncode == 0:
                consecutive_errors = 0
                completed += 1
                driver.on_success(returncode)
                if tracked_total is not None:
                    # on_success recorded the item, so the driver's own count now
                    # says how far the invocation has got.
                    remaining = driver.pending_total()
                    if remaining is not None:
                        progress.note_remaining(remaining)
                        app.update(**progress.summary_fields())

            # The end-of-item hook: after on_success, so the driver's own queue
            # is up to date, and after the outcome either way — a failed
            # iteration is still an iteration whose side effects are on disk.
            # `or` keeps the FIRST reason: a pause already asked for is not
            # withdrawn by a later hook answering None.
            handback_reason = handback_reason or driver.item_finished(command, returncode)

            # The backstop under the proactive check (see RateLimitEvent): this run's
            # own verdict from the wire. A refusal needs no figure and no query to be
            # trusted, so it is honoured whatever the usage report said — including
            # when the report was unavailable, which is the case this exists for.
            # Checked for both outcomes: a run refused on its last turn may still have
            # exited 0 with its work recorded above.
            # `handback_reason` excluded: this wait is the longest hold below the
            # loop head (a whole quota window), and the head is about to end the
            # run anyway. Sitting it out first would hand control back hours
            # after it was asked for — the window matters to the NEXT run, and
            # that one is the caller's to start.
            refusal = last_rate_limit_event() if provider == "claude" else None
            if (not ignore_usage_limits and refusal is not None
                    and refusal.status == "rejected" and handback_reason is None):
                # +5s so we come back after the reset, not exactly on it.
                target_ts = (refusal.resets_at
                             or time.time() + CLAUDE_SESSION_DURATION) + 5
                app.update(phase="paused")
                wait_until(target_ts,
                           reason=f"Hit the {refusal.label} — this run was refused. "
                                  f"Waiting until {fmt_moment(target_ts)} for that "
                                  f"window to refresh…",
                           should_stop=stop_pending)
                usage_source.invalidate()  # the figures behind the refusal are stale
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if refusal.limit_type == "five_hour":
                    session_start = time.time()
                consecutive_errors = 0  # fresh window — start counting errors anew
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

            if not ignore_usage_limits and handback_reason is None:
                # The pause exclusion is the refusal branch's (see there): this
                # gate can hold for a whole window, and the head is one `continue`
                # away from ending the run. The error itself is still counted and
                # printed above — only the waiting is skipped.
                app.update(phase="waiting")
                paused, session_start = limit_policy.check_and_wait(
                    usage_source, session_start, note=" (checked after error)",
                    should_stop=stop_pending)
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
                # The reason FIRST, then the housekeeping — see the driver-stop
                # exit above for why that order. The epilogue matters most here:
                # a run that gave up after five failures may have committed four
                # good iterations, and the notes an operator typed at the console
                # are the likeliest explanation of what went wrong.
                exitlog.set_reason(
                    f"{consecutive_errors} provider errors in a row "
                    f"(last exit code {returncode})",
                    iterations=iteration, completed=completed)
                runlifecycle.close_run(
                    ctx, usage_source=usage_source, limit_policy=limit_policy,
                    snapshot_label="at end (provider errors in a row)",
                    mailbox=mailbox)
                sys.exit(returncode)

    # This run's own closing line, if the driver has one (e.g. "Final state: …").
    # Before the shared epilogue, which is housekeeping: the run reports on its
    # work first, then the run is closed down.
    summary = driver.final_summary()
    if summary:
        print(f"\n{summary}")
    # `runlifecycle.end_run` is the epilogue both runners share. No `push_lock`,
    # because this runner has one thread and nothing to exclude — see
    # `gitpush.final_git_push` for why the lock belongs to the caller that has
    # threads rather than to the call.
    return runlifecycle.end_run(
        ctx, stop_reason, iterations=iteration, completed=completed,
        usage_source=usage_source, limit_policy=limit_policy,
        snapshot_label="at end (after last cycle)", mailbox=mailbox)
