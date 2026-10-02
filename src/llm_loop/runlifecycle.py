"""The prologue and the epilogue every run has, and the live knobs both read.

Two runners open and close a run: `cyclecore.run_loop` and `parallel.run_parallel`.
What happens between those two moments is genuinely different — one Driver loop
against N workers over a queue — but the opening and the closing are the same
run, and they used to be written twice. Measured before this module existed
(2026-08-24, `ast.unparse` over both function bodies): 22 of the statements in
`run_parallel`'s prologue were verbatim-identical to statements in `run_loop`'s,
and the epilogues ran the same four steps in the same order with one line in
common. What kept the two halves in step was PROSE — 17 cross-references in
`parallel.py` to "the sequential runner", two of them literally "See the same
branch/call in cyclecore.run_loop". A rule that lives in a comment is a rule the
second half re-derives instead of inheriting, and the git-push knob is what that
cost: it was a live setting in one runner and a frozen local in the other,
though both spell the same flag.

The epilogue is one body, `close_run`, behind three doors. `end_run` is the
ending a runner RETURNS from; `exit_run` is the same housekeeping for the
endings that `sys.exit` instead — a driver stopping the run with an exit code,
five provider errors in a row, Ctrl+C; and `RunBoundary`, each runner's one
boundary, holds both and closes whatever unwinds the run past them. The
exiting endings used to write only `exitlog.set_reason`, so the endings with
the most to explain were the ones that pushed nothing and reported nothing.

THE ORDER OF THE STEPS IS LOAD-BEARING, and this module exists to state it once
rather than to tidy it:

  * the tee goes up BEFORE `exitlog.begin`, because `begin` is what prints the
    report of the PREVIOUS run that vanished, and that report has to land in the
    mirror log whose abrupt end it explains. Swap them and the report goes to a
    terminal nobody is reading any more. Pinned by
    `tests/test_exit_reason.py::test_the_report_of_a_vanished_run_lands_in_the_log`;
  * the exit push is the WHOLE of `final_git_push`, the `git_unpushed_count`
    inside it included, made by the caller's pusher because it may still be
    inside `git push`.
    Reading the count beside it is how "nothing to push" could be printed about
    a repository that was being pushed at that moment. Pinned by
    `tests/test_git_push.py`.
"""

import os
import sys
import traceback
from typing import (Any, Callable, Iterable, NamedTuple, NoReturn, Optional,
                    Tuple)

from . import (console, diaglog, exitlog, limits, operator, projectroot,
               statusline, stopchannel)
from .gitpush import (
    GIT_PUSH_POLICY,
    GIT_PUSH_SETTING,
    GitPushPolicy,
    PushAbort,
    final_git_push,
)
from .ownership import OwnerThread
from .providers import provider_spec, set_live_messages, usage_source_for
from .stopchannel import RunResult
from .scriptlock import ensure_script_lock
from .usage import UsageSource


class RunSettings:
    """The script's own knobs, held in one MUTABLE object the run re-reads.

    Plain locals froze these at startup, which made "edit the limits while the
    run goes" (through `knob_registry`'s setters) impossible without touching the
    runner body. Both runners read this object where the value is USED — the sequential
    one at every iteration boundary, the parallel one from its claim loop and
    from its pusher — so moving `--max-runs` from 40 to 60 mid-run takes effect
    at the next boundary and nothing else has to change.

    `max_runs` counts different things in the two runners and deliberately keeps
    one name: iterations for the sequential loop, FILES claimed for the parallel
    one (see `clispec.PARALLEL`, where that difference is written down). It is
    the same flag either way, so it is the same knob.
    """

    def __init__(self, *, max_runs: Optional[int] = None,
                 git_push: Optional[GitPushPolicy] = None):
        self.max_runs = max_runs
        self.git_push = git_push or GitPushPolicy(GIT_PUSH_POLICY)


def knob_registry(run_settings: RunSettings, progress=None) -> Any:
    """The script's knobs as a SettingsRegistry — the display AND edit surface.

    One registry is the single source of truth for both the pinned row
    (`status_entries()`) and the reproducing command line (`overrides()`), so a
    figure on screen can never disagree with the flag that would reproduce it;
    the flags are checked against `cmdline.FLAG_ALIASES` at registration.

    `progress` is the invocation's `InvocationProgress` when this run OWNS the
    figures, and None when a wrapper above it does. An edited `--max-runs` moves
    the summary row's denominator, and this is the one place that carries it
    across: under a wrapper the displayed cap is the wrapper's to set, and this
    call only sizes the current batch. Doing it in the setter rather than at a
    runner's own boundary is what lets the parallel runner have the knob at all
    — it has no boundary on the main thread to re-read anything at.
    """
    registry = statusline.SettingsRegistry()

    def set_max_runs(value):
        run_settings.max_runs = None if value is None else int(value)
        if progress is not None:
            progress.max_items = run_settings.max_runs

    registry.add(statusline.NumberSetting(
        "max-runs", "--max-runs",
        lambda: run_settings.max_runs,
        set_max_runs,
        minimum=1,
        # Editable and reproducible, but not a field of its own: the counter
        # already ends in this number (`iter 11/40`), or in the list's size when
        # that is the smaller of the two — see InvocationProgress.summary_fields.
        # Off the row it also stops printing `max-runs off` for every run that
        # never set one.
        show_in_status=False))
    registry.add(statusline.Setting(
        GIT_PUSH_SETTING, "--git-push",
        lambda: run_settings.git_push.value,
        lambda value: setattr(run_settings, "git_push", GitPushPolicy(value))))
    return registry


class RunContext(NamedTuple):
    """What the shared prologue settled; the runner and the epilogue read it back.

    `settings` is the live one (see RunSettings) — every later read of the cap or
    the push policy goes through it, never through a local taken here, which is
    the whole reason it is handed on rather than unpacked.
    """

    provider: str
    spec: Any
    dry_run: bool
    progress: Any
    settings: RunSettings
    # The knob registry over `settings` (see knob_registry).
    registry: Any
    # Whether the pinned status area is drawn: never for a dry run, and not
    # under `--no-statusline`. Off, open_status hands back the Null app.
    status_enabled: bool


def begin_run(driver, args, app_name: str, progress=None, *,
              setup_logging: bool = True) -> RunContext:
    """Everything both runners do before they have any work to show for it.

    In order, and the order is the point (see the module header): settle the
    provider, settle the run's live knobs, anchor the project root, decide the
    live-message transport, raise the tee, open the exit record, print the
    header every run starts with. A runner adds its own header lines after this
    returns.

    The header covers what BOTH runners have to say, the git-push policy
    included: it is a knob both of them register, so its line was drifting into
    two spellings of one fact — a bare one here and a half-line tacked onto the
    parallel runner's worker count.

    `setup_logging=False` is the wrapper's hook: a host script that already tees
    its own output must not have a second tee stacked on top of the first.

    `args` is read strictly: `provider`, `dry_run`, `project_dir` and
    `no_live_messages` are declared by `clispec` for BOTH modes, so a namespace
    from either parser always has them, and a default here could only ever cover
    for a caller that built its namespace by hand and left one out — the caller
    that gets an AttributeError instead.
    """
    launch = None
    if not args.dry_run:
        launch = ensure_script_lock(
            app_name=app_name,
            project_dir=os.path.abspath(
                args.project_dir or projectroot.project_dir()))

    provider = args.provider or driver.provider
    spec = provider_spec(provider)
    driver.provider = provider

    # No wrapper above us: this call is the whole invocation, so its own --max is
    # the invocation cap and it owns the figures.
    owns_progress = progress is None
    if owns_progress:
        progress = statusline.InvocationProgress(max_items=args.max)

    settings = RunSettings(max_runs=args.max,
                           git_push=GitPushPolicy(args.git_push))
    registry = knob_registry(settings, progress if owns_progress else None)
    dry_run = bool(args.dry_run)

    # Anchor every project-relative operation (git/provider cwd, the stop file,
    # the log name, the Driver's paths) before anything reads the root.
    projectroot.set_project_root(args.project_dir)

    # Decided per invocation, before the first argv is built: the transport is
    # what --no-live-messages turns off, and both the argv and the process's
    # stdin have to agree about it. Set in BOTH directions — a wrapper that calls
    # two runners in one process (see runGenerateModels' parallel mode, which
    # alternates product batches with kit-promotion passes) would
    # otherwise have the first `--no-live-messages` phase decide the transport
    # for every phase after it.
    set_live_messages(not args.no_live_messages)

    # Mirror all screen output into a rotating log file under the home dir —
    # except for a dry run, which is a preview and not a run: its output would
    # otherwise displace real runs' records out of the shared rotating log (a
    # preview once pushed ~26 MB through it, and the failure it was launched to
    # explain rotated off the end). Said on screen so the missing log is visible
    # rather than mysterious.
    if setup_logging and not dry_run:
        logger = console.setup_file_logging(app_name)
        sys.stdout = console.TeeToLog(sys.stdout, logger)
        sys.stderr = console.TeeToLog(sys.stderr, logger)
    if not dry_run:
        # AFTER the tee, and that is the load-bearing half of this order: `begin`
        # prints the report of a PREVIOUS run that vanished, and the report has
        # to land in the mirror log whose abrupt end it explains. Idempotent per
        # process: a batching wrapper calls a runner repeatedly and keeps one
        # record.
        record = exitlog.begin(app_name, console.LOG_DIR,
                               os.path.basename(projectroot.project_dir()))
        if record is not None:
            record.note(script_lock=launch.record())
    print(f"  · project root: {projectroot.project_dir()}")
    if dry_run:
        print(f"  · dry run: nothing is mirrored to "
              f"{console.log_file_path(app_name)}")
    else:
        print(f"  · logging to {console.log_file_path(app_name)}")
        print(f"  · {launch.summary()}")
        if diaglog.log_path() is not None:
            print(f"  · status line diagnostics: {diaglog.log_path()}"
                  + (" (key trace on)" if diaglog.keytrace_enabled() else ""))
    print(f"  · provider: {spec.display_name}")
    print(f"  · git push policy: {settings.git_push.value}")
    console.warn_missing_dependencies()
    return RunContext(provider=provider, spec=spec,
                      dry_run=dry_run, progress=progress,
                      settings=settings, registry=registry,
                      status_enabled=(not dry_run and
                                      not getattr(args, "no_statusline", False)))


def open_status(ctx: RunContext, driver, *, job_count: int,
                messages) -> "statusline.StatusApp":
    """The pinned status area, built the same way by both runners.

    A Job is the unit of display in both runners, so the sequential loop is a
    run with exactly one Job and the parallel one a run with N — no branch
    anywhere in the status line separates them. The Jobs come from
    `progress.jobs`. Disabled (a dry run, or `--no-statusline`), the app is a
    Null object and every call on it is a no-op. `messages` is the runner's own
    wiring: one Mailbox for the loop, a MailboxSet for the workers, None for a
    dry run.

    Records the queue as it stands (Driver.pending_total →
    InvocationProgress.track_total / note_remaining) before the first update.

    A runner registers its own actions on the returned app and enters it; the
    quota priming stays with the runner, because only the parallel one knows
    its account before the first command.
    """
    progress = ctx.progress
    total = driver.pending_total()
    if total is not None:
        progress.track_total(total)
        progress.note_remaining(total)
    app = statusline.StatusApp(
        status=statusline.LoopStatus(jobs=progress.jobs(job_count)),
        messages=messages,
        enabled=ctx.status_enabled)
    app.update(
        provider=ctx.provider,
        **progress.summary_fields(),
        # Only a list driver has a pick order; read it defensively so any other
        # driver simply reports no `rand` marker.
        random_order=str(getattr(driver, "pick_order", "")) == "random",
        script_limits=ctx.registry.status_entries(),
    )
    return app


class RunUsage:
    """One account's usage source and the policy that reads it — opened together.

    The pair is the unit: a source with no policy has nobody to decide what its
    figures mean, and a policy with no source has nothing to read. The runners
    used to assemble the two separately and in different words, and the closing
    snapshot relied on them having been set together without anything checking
    it; a None here now fails where the pair is BUILT, not at the end of a run.

    `name` is what the snapshots are labelled with, and it is what makes the
    opening snapshot and the closing one a pair in the log: `at start (name)` is
    answered by `at end (name)`, or by `at end (name: ending)` when the run
    ended somewhere worth naming (see `close`) — on every ending once it is
    open (see `close_run`).
    """

    __slots__ = ("source", "policy", "name")

    def __init__(self, source, policy, name: str):
        if source is None or policy is None:
            raise ValueError(f"usage '{name}' needs both a source and a policy "
                             f"(source={source!r}, policy={policy!r})")
        self.source = source
        self.policy = policy
        self.name = name

    def open(self) -> None:
        """The start-of-run snapshot of the policy's watched quotas."""
        self.policy.log_snapshot(self.source, f"at start ({self.name})")

    def close(self, ending: Optional[str] = None) -> None:
        """The end-of-run snapshot answering `open`'s.

        Forced fresh (cache_value=False) so it reflects the true post-run state
        rather than a possibly-recent cached reading from the last limit check.
        `ending` names an abnormal ending NEXT TO the usage's own name, not in
        place of it: a mixed-provider run closes one usage per account under the
        same ending, and two bare `at end (interrupted)` lines would not say
        which account each figure belongs to.
        """
        label = self.name if ending is None else f"{self.name}: {ending}"
        self.policy.log_snapshot(self.source, f"at end ({label})",
                                 cache_value=False)


def open_usage(driver, provider: str, *, name: Optional[str] = None,
               dry_run: bool) -> Optional[RunUsage]:
    """The provider's usage pair with its opening snapshot, or None without one.

    None when the provider has no usage endpoint (`usage_source_for`); whether
    to open one at all is the runner's call. `name` labels the snapshots and
    defaults to the provider. The policy is the Driver's specialisation when it
    has one, the provider's default otherwise. A dry run gets the pair (the gate
    and the status line read it) but no snapshot, because it is not a run.
    """
    source = usage_source_for(provider)
    if source is None:
        return None
    usage = RunUsage(source,
                     driver.limit_policy or limits.default_policy(provider),
                     name or provider)
    if not dry_run:
        usage.open()
    return usage


def usage_halves(usage: Optional[RunUsage]
                 ) -> Tuple[Optional[UsageSource], Optional[limits.LimitPolicy]]:
    """`(source, policy)` of a usage pair, or `(None, None)` without one.

    The gate, the status line and the parallel workers take the two halves
    separately; this is the one place they are split, so a runner never holds a
    source and a policy that came from different pairs.
    """
    return (usage.source, usage.policy) if usage is not None else (None, None)


def close_run(ctx: RunContext, *,
              usages: Iterable[Optional[RunUsage]],
              ending: Optional[str] = None,
              mailbox=None,
              pusher: Optional[OwnerThread] = None,
              push_abort: Optional[PushAbort] = None) -> None:
    """The housekeeping half of the epilogue, for every ending a run can have.

    Push what is still local, record where the quotas finished, report the notes
    nobody delivered. A runner prints its own closing report BEFORE calling this
    — "Final state: …", "Processed N file(s) …" — because that is the run talking
    about its work, and everything here is closing it down.

    Separate from `end_run` because most endings are NOT returns: the driver
    stopping the run with an exit code, five provider errors in a row, and
    Ctrl+C all `sys.exit`, so they have a reason to record but no `RunResult`
    to hand back. They used to write only `exitlog.set_reason` and leave — no
    exit push, no closing snapshot, no report of undelivered notes — which
    meant the endings that most need a post-mortem were the ones that left the
    least behind, and an operator's commits sat local until some later run
    happened to push them. Each of those now comes here through `exit_run`,
    and an ending no runner wrote through `RunBoundary.unwind`.

    `pusher` is the thread that owns the caller's git, and the exit push is
    handed to it as its `final` — queued behind
    a push it has in flight, and waited for, however long that push takes: the
    exit push is the WHOLE of `final_git_push`, `git_unpushed_count` included,
    so no git call of it can run beside that push. The pusher is closed here on
    every ending that reaches this function, a dry run's included (the runner
    closes it on the ones that do not). Without one the push is made here, on
    the caller. The
    policy is read AT the push, off the live settings, either way. On the pusher
    a push that raises is reported with its traceback on stderr and the
    housekeeping below still runs; made here it propagates. A KeyboardInterrupt
    from the pusher is returned to this thread for the normal interrupt ending.

    Ctrl+C while the exit push is waited for — on the pusher or here — gives
    up the push, not the housekeeping: see the body for the order, and
    `gitpush.final_git_push` for what happens to a git call already running.
    The sequential runner shares `push_abort` with its periodic checks so
    an interrupt also cancels checks queued behind the current push.

    `usages` is EVERY usage the run opened, not the one it ended on: a
    mixed-provider sequential run opens one per account it selects, and each is
    closed here. Required, so a runner states what it opened rather than
    defaulting to closing nothing. A None in it is an account without a usage
    endpoint (see `open_usage`) and has nothing to close. Each close is guarded
    on its own: a snapshot that raises costs its own `at end` line, not the
    other accounts' lines and not the report of undelivered notes after them.

    Every `at start (…)` is answered: each runner holds one boundary from its
    first `open_usage` to this call (`RunBoundary`), so an ending that is
    neither a return nor an exit the runner wrote — Ctrl+C anywhere, a provider
    executable that is not installed, an exception out of the driver,
    `driver.final_summary` included — still comes here, once. What exits before
    any usage is open (the waits for a stale stop file and for `--start-in`)
    has nothing to answer and leaves on its own `sys.exit`.
    """
    usages = list(usages)           # walked twice: snapshots, then sources
    abort = push_abort or PushAbort()
    exit_interrupt = []

    def exit_push():
        final_git_push(ctx.settings.git_push, projectroot.project_dir(),
                       abort=abort)

    def exit_push_on_pusher():
        # Reported here, whole, rather than by the owner: `OwnerThread._report`
        # is one line with no traceback, and it drops a failure worded like one
        # it already reported — a periodic push that failed the same way would
        # silence the exit push's. stderr is teed to the mirror log.
        try:
            exit_push()
        except KeyboardInterrupt as caught:
            exit_interrupt.append(caught)
        except Exception:
            print("  ⚠ the exit push failed; what is still local stays local:",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    # A Ctrl+C while the exit push runs abandons the PUSH, not the ending: the
    # push starts no further git (`abort`), the snapshots and the notes below
    # still happen, and the interrupt is raised again once they have — so it
    # still ends the run, only after its housekeeping. A Ctrl+C inside that
    # housekeeping skips what is left of the snapshots, never the notes.
    interrupt: Optional[KeyboardInterrupt] = None
    try:
        if ctx.dry_run:
            if pusher is not None:
                pusher.close()
        elif pusher is not None:
            _wait_for_exit_push(pusher, exit_push_on_pusher)
            if exit_interrupt:
                raise exit_interrupt[0]
        else:
            exit_push()
    except KeyboardInterrupt as caught:
        interrupt = caught
        abort.set()
        print("  ⚠ Ctrl+C: the exit push is abandoned — what is still local "
              "stays local until a later run pushes it.")

    # End-of-run usage snapshots, one answering each `open_usage` that logged —
    # so each run records where every account it used finished. `ending` names
    # the abnormal endings (see RunUsage.close).
    if not ctx.dry_run:
        try:
            for usage in usages:
                if usage is None:
                    continue
                try:
                    usage.close(ending)
                except Exception as error:
                    print(f"  ⚠ usage at end ({usage.name}) could not be read: "
                          f"{type(error).__name__}: {error}")
        except KeyboardInterrupt as caught:
            interrupt = interrupt or caught

    # Stop what a source keeps running between reads (Codex's quota server),
    # dry runs included: the gate and the status line read the source there
    # too. Endings that never get here leave it to the source's atexit hook.
    for usage in usages:
        if usage is None:
            continue
        close_source = getattr(usage.source, "close", None)
        if close_source is None:
            continue
        try:
            close_source()
        except Exception as error:
            print(f"  ⚠ usage source ({usage.name}) did not close: "
                  f"{type(error).__name__}: {error}")
        except KeyboardInterrupt as caught:
            interrupt = interrupt or caught

    operator.report_undelivered_notes(mailbox)
    if interrupt is not None:
        raise interrupt


# How often the wait for the exit push wakes up. Not a bound on the push — that
# is waited for however long it takes — but how late a Ctrl+C can be heard: up
# to Python 3.13 a lock wait cannot be interrupted, so an unbounded one kept the
# operator waiting for the whole `git push` (300 s subprocess timeout). A
# quarter of a second is below what a person notices, and a wake-up that finds
# nothing to do costs a few instructions.
EXIT_PUSH_POLL_S = 0.25


def _wait_for_exit_push(pusher: OwnerThread, push) -> None:
    """Hand `push` to `pusher` as its `final` and wait until the owner has ended.

    Queued behind a push the owner has in flight, and waited for in short
    slices (EXIT_PUSH_POLL_S) so a Ctrl+C lands here on every Python, where
    the caller turns it into an abandoned push.
    """
    if pusher.close(timeout=0, final=push):
        return
    while not pusher.close(timeout=EXIT_PUSH_POLL_S):
        pass


# What the exit record says about a run the operator ended with Ctrl+C, from
# any door of the epilogue (`end_run`, `exit_run`, `RunBoundary.unwind`).
INTERRUPTED_REASON = "interrupted by the operator (Ctrl+C)"


def end_run(ctx: RunContext, result: RunResult, *,
            usages: Iterable[Optional[RunUsage]],
            mailbox=None,
            pusher: Optional[OwnerThread] = None,
            push_abort: Optional[PushAbort] = None) -> RunResult:
    """Everything both runners do when the work is over and they RETURN.

    The housekeeping is `close_run`; this adds what only a normal ending has — a
    `RunResult` — by recording the reason of the caller's and returning it.

    The reason is RECORDED rather than printed: a wrapper may call several
    runners, the `=== run ended: … ===` line belongs to the process, so the last
    reason set wins and exitlog prints it on the way out.

    A Ctrl+C that abandoned the exit push (`close_run` raises it on once its
    housekeeping is done) ends the run the way Ctrl+C ends it everywhere else
    in both runners: INTERRUPTED_REASON recorded, `sys.exit(130)`. Let out
    bare, the KeyboardInterrupt would leave a traceback and an exit code of the
    interpreter's choosing (0xC000013A on Windows) from this one door only.
    """
    reason = result.reason
    interrupted = False
    try:
        close_run(ctx, usages=usages, mailbox=mailbox, pusher=pusher,
                  push_abort=push_abort)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        # Recorded whether or not the housekeeping got through; an exception
        # out of it is recorded over this by the excepthook.
        exitlog.set_reason(
            INTERRUPTED_REASON if interrupted
            else stopchannel.STOP_REASON_TEXT.get(reason, reason.value),
            iterations=result.attempted, completed=result.completed)
    if interrupted:
        sys.exit(130)
    return result


def exit_run(ctx: RunContext, code, *,
             usages: Iterable[Optional[RunUsage]],
             ending: str,
             iterations: int,
             completed: int,
             mailbox=None,
             pusher: Optional[OwnerThread] = None,
             push_abort: Optional[PushAbort] = None) -> NoReturn:
    """`close_run`, then `sys.exit(code)`: the door of every ending that exits.

    The caller records its reason BEFORE calling this, so the record does not
    depend on the push or a usage query surviving. A Ctrl+C that abandoned the
    exit push (`close_run` raises it on once its housekeeping is done) ends the
    run as an interrupt instead, exactly as it does through `end_run`:
    INTERRUPTED_REASON recorded over the caller's reason, `sys.exit(130)`.
    `iterations`/`completed` go with that record.
    """
    try:
        close_run(ctx, usages=usages, ending=ending, mailbox=mailbox,
                  pusher=pusher, push_abort=push_abort)
    except KeyboardInterrupt:
        exitlog.set_reason(INTERRUPTED_REASON, iterations=iterations,
                           completed=completed)
        sys.exit(130)
    sys.exit(code)


class RunBoundary:
    """One runner's epilogue boundary: from its first `open_usage` to `close_run`.

    Every ending inside it closes the run down exactly once. The endings the
    runner writes go through its doors — `end` (a return, `end_run`) and
    `exit` (`exit_run`) — and `closed` turns true as one is taken, so the
    boundary leaves an ending that is already being closed alone. Whatever else
    unwinds the block is `unwind`'s:

      * KeyboardInterrupt — Ctrl+C wherever it lands: inside a turn
        (`streamrender.run_agent_streaming`), a quota hold
        (`limits.LimitPolicy.check_and_wait`), the wait after a refusal
        (`cyclecore._count_down_to`), the driver, the status region's waits.
        Recorded as INTERRUPTED_REASON, closed as "interrupted", exit 130;
      * SystemExit — an exit nobody closed down, such as `streamrender`'s
        exit 2 for a provider executable that is not installed: recorded as
        that exit, closed as "exit N", the code kept;
      * any other exception — out of the driver or the runner body: closed as
        "unhandled <Type>" and let go on unchanged, for the excepthook to
        record. A Ctrl+C in this housekeeping abandons the push (see
        `close_run`) and the exception stays the ending; an exception the
        housekeeping raises itself is reported on stderr, never put in the
        original's place.

    What a run opens as it goes is read when the ending comes: `usages` and
    `counts` (`(iterations, completed)` for the exit record) are callables, and
    `mailbox`/`pusher` are attributes a runner sets once it has them.

    Used as a context manager (`with boundary, app:` — the region is released
    before the housekeeping prints) or by hand from an `except` (`unwind`).
    """

    def __init__(self, ctx: RunContext, *,
                 usages: Callable[[], Iterable[Optional[RunUsage]]],
                 counts: Callable[[], Tuple[int, int]],
                 mailbox=None,
                 pusher: Optional[OwnerThread] = None,
                 push_abort: Optional[PushAbort] = None):
        self.ctx = ctx
        self.usages = usages
        self.counts = counts
        self.mailbox = mailbox
        self.pusher = pusher
        self.push_abort = push_abort
        self.closed = False

    def __enter__(self) -> "RunBoundary":
        return self

    def __exit__(self, exc_type, error, traceback_) -> bool:
        if error is not None:
            self.unwind(error)
        return False

    def _close_kwargs(self) -> dict:
        return dict(usages=self.usages(), mailbox=self.mailbox,
                    pusher=self.pusher, push_abort=self.push_abort)

    def end(self, result: RunResult) -> RunResult:
        """The normal ending: `end_run`."""
        self.closed = True
        return end_run(self.ctx, result, **self._close_kwargs())

    def exit(self, code, *, ending: str) -> NoReturn:
        """An ending the runner exits from: `exit_run`. Record the reason first."""
        self.closed = True
        iterations, completed = self.counts()
        exit_run(self.ctx, code, ending=ending, iterations=iterations,
                 completed=completed, **self._close_kwargs())

    def unwind(self, error: BaseException) -> None:
        """Close the run `error` is unwinding; returns only for an exception,
        which the caller lets go on. A no-op once a door has been taken."""
        if self.closed:
            return
        self.closed = True
        iterations, completed = self.counts()
        if isinstance(error, KeyboardInterrupt):
            # Announced, if at all, where it landed (a turn, a wait); the exit
            # record names it either way.
            exitlog.set_reason(INTERRUPTED_REASON, iterations=iterations,
                               completed=completed)
            exit_run(self.ctx, 130, ending="interrupted", iterations=iterations,
                     completed=completed, **self._close_kwargs())
        if isinstance(error, SystemExit):
            code = 0 if error.code is None else error.code
            exitlog.set_reason(exitlog.describe_exception(SystemExit, error),
                               iterations=iterations, completed=completed)
            # The first line only: `sys.exit("message")` may carry a paragraph.
            first_line = (str(code).splitlines() or [""])[0]
            exit_run(self.ctx, code, ending=f"exit {first_line}",
                     iterations=iterations, completed=completed,
                     **self._close_kwargs())
        try:
            close_run(self.ctx, ending=f"unhandled {type(error).__name__}",
                      **self._close_kwargs())
        except KeyboardInterrupt:
            pass
        except Exception:
            print("  ⚠ closing the run down failed while an exception unwound "
                  "it:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
