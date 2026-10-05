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

import contextlib
import os
import sys
import time
import traceback
from enum import Enum
from typing import (Any, Callable, Iterable, List, NamedTuple, NoReturn,
                    Optional, Tuple)

from . import (console, ctrlc, diaglog, exitlog, limits, operator,
               projectroot, statusline, stopchannel)
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
    progress.start_run()

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
               dry_run: bool,
               register: Optional[Callable[[Optional[RunUsage]], None]] = None
               ) -> Optional[RunUsage]:
    """The provider's usage pair with its opening snapshot, or None without one.

    None when the provider has no usage endpoint (`usage_source_for`); whether
    to open one at all is the runner's call. `name` labels the snapshots and
    defaults to the provider. The policy is the process's one for that account
    (`limits.process_policy`), settled by the first run that opens it. A dry run gets the pair (the gate
    and the status line read it) but no snapshot, because it is not a run.

    `register` is handed what this returns BEFORE the opening snapshot: the
    snapshot is a usage query, which can raise after it has started what the
    source keeps running (Codex's quota server), and a runner's boundary can
    only close a pair it holds. The runners open through
    `RunBoundary.open_usage`, which is that registration.
    """
    source = usage_source_for(provider)
    if source is None:
        if register is not None:
            register(None)
        return None
    usage = RunUsage(source,
                     limits.process_policy(provider, driver.limit_policy),
                     name or provider)
    if register is not None:
        register(usage)
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
              push_abort: Optional[PushAbort] = None,
              budget: Optional["StopBudget"] = None,
              heard: Optional[int] = None) -> None:
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

    The exit push is the WHOLE of `final_git_push`, `git_unpushed_count`
    included, made on an owner thread and waited for here: on `pusher`, the
    thread that owns the caller's git, as its `final` — queued behind a push it
    has in flight, so no git call of the exit push runs beside that push — or,
    with no RUNNING `pusher` (none, or one the run made and had not started:
    the parallel preparation fails between the two), on a throwaway owner.
    Never on the caller: a push made inline is a wait nothing can give up, and
    an owner that never started runs its `final` inline. The pusher is closed
    here on every ending that reaches this function, a dry run's included
    (`RunBoundary` says what happens to it on the ones that do not). The
    policy is read AT the push, off the live settings. A push that raises is
    reported with its traceback on stderr, and the housekeeping below still
    runs.

    The wait for the push ends one of three ways (`_wait_for_exit_push`): the
    push is done; the ending's `budget` is spent (see StopBudget — None, an
    ending with the operator present, has none); or the operator pressed
    Ctrl+C past the ending's own presses — `heard`, the run's
    `ctrlc.Interrupt.presses` as the ending began (None reads it here). Either
    of the last two abandons the push and says so: `push_abort` is set, so no
    further git starts, and a git call already running finishes on its own as
    a daemon (see `gitpush.final_git_push`). Either of them already true when
    the push would begin abandons it unstarted. What a press does to the
    ENDING is the door's question (`end_run`, `exit_run`), not this
    function's. The sequential runner shares `push_abort` with its periodic
    checks so an abandoned push also cancels checks queued behind the current
    one.

    `usages` is EVERY usage the run opened, not the one it ended on: a
    mixed-provider sequential run opens one per account it selects, and each is
    closed here. Required, so a runner states what it opened rather than
    defaulting to closing nothing. A None in it is an account without a usage
    endpoint (see `open_usage`) and has nothing to close. Each close is guarded
    on its own: a snapshot that raises costs its own `at end` line, not the
    other accounts' lines and not the report of undelivered notes after them.
    The snapshots and the notes are not waits, and no Ctrl+C skips them.

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
    interrupt = ctrlc.current()
    mark = interrupt.presses if heard is None else heard

    def exit_push_on_pusher():
        # Reported here, whole, rather than by the owner: `OwnerThread._report`
        # is one line with no traceback, and it drops a failure worded like one
        # it already reported — a periodic push that failed the same way would
        # silence the exit push's. stderr is teed to the mirror log.
        try:
            final_git_push(ctx.settings.git_push, projectroot.project_dir(),
                           abort=abort)
        except Exception:
            print("  ⚠ the exit push failed; what is still local stays local:",
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def abandoned() -> bool:
        return interrupt.since(mark)

    if ctx.dry_run:
        if pusher is not None:
            pusher.close()
    elif abandoned() or (budget is not None and budget.expired):
        # Closed so its own clock starts no further check; a git call it has
        # in flight finishes on its own as a daemon, and `abort` keeps
        # anything queued behind that from starting git. Bounded: an owner's
        # lock is never held across a call.
        abort.set()
        if pusher is not None:
            pusher.close(timeout=EXIT_PUSH_POLL_S)
        _say_push_given_up(PushWait.ABANDONED if abandoned()
                           else PushWait.DEADLINE, budget)
    else:
        if pusher is None or pusher.thread is None:
            pusher = OwnerThread("exit pusher").start()
        outcome = _wait_for_exit_push(pusher, exit_push_on_pusher, budget,
                                      abandoned=abandoned)
        if outcome is not PushWait.DONE:
            abort.set()
            _say_push_given_up(outcome, budget)

    # End-of-run usage snapshots, one answering each `open_usage` that logged —
    # so each run records where every account it used finished. `ending` names
    # the abnormal endings (see RunUsage.close).
    if not ctx.dry_run:
        for usage in usages:
            if usage is None:
                continue
            try:
                usage.close(ending)
            except Exception as error:
                print(f"  ⚠ usage at end ({usage.name}) could not be read: "
                      f"{type(error).__name__}: {error}")

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

    operator.report_undelivered_notes(mailbox)


# What a run says when the operator's Ctrl+C gave up its exit push.
ABANDONED_PUSH = ("  ⚠ Ctrl+C: the exit push is abandoned — what is still local "
                  "stays local until a later run pushes it.")


# How often the wait for the exit push wakes up. Not a bound on the push — an
# ending with an operator present waits for it however long it takes — but how
# late that operator's Ctrl+C is heard: a press is an event the wait polls
# (`ctrlc`), not an exception that cuts it. A quarter of a second is below what
# a person notices, and a wake-up that finds nothing to do costs a few
# instructions.
EXIT_PUSH_POLL_S = 0.25


class StopBudget:
    """The time one ending may spend waiting, taken ONCE as the ending begins.

    Every wait of the ending is given what is LEFT, never a bound of its own:
    bounds set one per wait add up behind each other, and each of them was
    found by a separate review (llm-loop 2255e6f, ce11713) — one budget is
    one number to read and one place to change. Spent, no wait begins any
    more: no further git starts, and the rest of the ending (the snapshots,
    the notes) is done without waiting. `seconds` None is no budget: an
    ending with the operator present, who gives a wait up with Ctrl+C.
    """

    def __init__(self, seconds: Optional[float]):
        self.seconds = seconds
        self._deadline = (None if seconds is None
                          else time.monotonic() + seconds)

    def left(self) -> Optional[float]:
        """Seconds still to spend, 0.0 once spent, None without a budget."""
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - time.monotonic())

    @property
    def expired(self) -> bool:
        left = self.left()
        return left is not None and left <= 0


class PushWait(Enum):
    """How the wait for an exit push ended (`_wait_for_exit_push`)."""

    DONE = "done"
    DEADLINE = "deadline"       # the ending's StopBudget was spent first
    ABANDONED = "abandoned"     # the operator's Ctrl+C gave it up


def _say_push_given_up(outcome: PushWait,
                       budget: Optional[StopBudget]) -> None:
    if outcome is PushWait.ABANDONED:
        print(ABANDONED_PUSH)
        return
    print(f"  ⚠ the exit push did not finish within {budget.seconds:g} s "
          f"and is abandoned: no further git starts, the one running "
          f"finishes on its own — what is still local stays local.",
          file=sys.stderr)


def _wait_for_exit_push(pusher: OwnerThread, push,
                        budget: Optional[StopBudget] = None,
                        abandoned: Callable[[], bool] = lambda: False
                        ) -> PushWait:
    """Hand `push` to `pusher` as its `final` and wait until the owner has ended.

    Queued behind a push the owner has in flight, and waited for in short
    slices (EXIT_PUSH_POLL_S) so `abandoned` — the operator's Ctrl+C — is asked
    at least that often, and never past what is left of `budget`. A wait that
    outlasts its first slice under a budget says so, since nobody may be
    watching to know why the run has gone quiet.
    """
    if pusher.close(timeout=0, final=push):
        return PushWait.DONE
    announced = False
    while True:
        if abandoned():
            return PushWait.ABANDONED
        slice_s = EXIT_PUSH_POLL_S
        left = None if budget is None else budget.left()
        if left is not None:
            if left <= 0:
                return PushWait.DEADLINE
            slice_s = min(slice_s, left)
        if pusher.close(timeout=slice_s):
            return PushWait.DONE
        if left is not None and not announced:
            announced = True
            print(f"  · waiting at most {round(left, 1):g} s for the exit "
                  f"push before the run goes on leaving…")


# The StopBudget of an ending nobody may be watching: one no runner wrote — an
# exception or an exit unwinding the run past its doors (`RunBoundary.unwind`)
# — and the runner's own `RunBoundary.exit` (the driver stopping the run, five
# provider errors in a row: endings of a run left going overnight). In practice
# what it bounds is the exit push, the one wait such an ending has; the closing
# usage snapshots are network reads with their own timeouts, outside it. Bounded because
# nobody may be there to press Ctrl+C: a batch run unwinding an exception
# otherwise sat out a push in flight and the whole `final_git_push` behind it
# (a 300 s subprocess timeout per `git push`, and on Windows a git descendant
# holding the pipe outlives even that — see `gitpush._reap_killed`) before its
# traceback, in silence. A push that gets through needs seconds: a `git push
# --dry-run` round trip took 2.5 s (the host repository) and 1.1 s (llm-loop),
# measured 2026-10-03, and a real push carries objects on top. A minute keeps a
# normal exit push and a periodic one in flight ahead of it; past it the push
# is taken to be stuck. What is lost is that push only: no further git starts
# (`PushAbort`), the one running finishes on its own as a daemon, and the
# snapshots and the notes still follow. The interrupt's door has no budget:
# there an operator is present to give the push up with Ctrl+C again.
UNWIND_PUSH_DEADLINE_S = 60.0


# Re-exported: the doors here write it, and `ctrlc` owns the wording.
INTERRUPTED_REASON = ctrlc.INTERRUPTED_REASON


def _exit_interrupted(iterations: int, completed: int) -> NoReturn:
    """The ending of a run Ctrl+C ended, with the run's counts (`ctrlc.leave`)."""
    ctrlc.leave(iterations=iterations, completed=completed)


def end_run(ctx: RunContext, result: RunResult, *,
            usages: Iterable[Optional[RunUsage]],
            local_counts: Optional[Tuple[int, int]] = None,
            mailbox=None,
            pusher: Optional[OwnerThread] = None,
            push_abort: Optional[PushAbort] = None) -> RunResult:
    """Everything both runners do when the work is over and they RETURN.

    The housekeeping is `close_run`; this adds what only a normal ending has — a
    `RunResult` — by recording the reason of the caller's and returning it.

    The reason is RECORDED rather than printed: a wrapper may call several
    runners, the `=== run ended: … ===` line belongs to the process, so the last
    reason set wins and exitlog prints it on the way out. Recorded before the
    housekeeping, as every door's is: the record must not depend on a push or
    a usage query surviving.

    The ending is chosen before the housekeeping and the housekeeping does not
    rewrite it (`exit_run`), with ONE exception here: a Ctrl+C pressed while
    the run closes down gives up its waits (`close_run`) and does not let the
    run RETURN. A returned result tells the caller to go on — a wrapper to
    start its next phase — which is the opposite of what the operator asked;
    an exiting ending needs no such exception, the process is leaving anyway.
    It ends as the interrupt instead (`_exit_interrupted`).
    """
    reason = result.reason
    # A parallel RunResult counts cap reservations; a started turn interrupted
    # by Ctrl+C returns that reservation but still belongs to the run total.
    iterations, completed = (local_counts if local_counts is not None else
                             (result.attempted, result.completed))
    counts = ctx.progress.record_counts(iterations, completed)
    interrupt = ctrlc.current()
    mark = interrupt.presses
    exitlog.set_reason(stopchannel.STOP_REASON_TEXT.get(reason, reason.value),
                       **counts)
    close_run(ctx, usages=usages, mailbox=mailbox, pusher=pusher,
              push_abort=push_abort, heard=mark)
    if interrupt.since(mark):
        _exit_interrupted(**counts)
    return result


def exit_run(ctx: RunContext, code, *,
             usages: Iterable[Optional[RunUsage]],
             ending: str,
             mailbox=None,
             pusher: Optional[OwnerThread] = None,
             push_abort: Optional[PushAbort] = None,
             budget: Optional[StopBudget] = None,
             heard: Optional[int] = None) -> NoReturn:
    """`close_run`, then `sys.exit(code)`: the door of every ending that exits.

    The ending — the caller's reason, recorded BEFORE this is called, and
    `code` — is decided before the housekeeping, and nothing in the
    housekeeping rewrites it. A Ctrl+C pressed past the ending's own presses
    (`heard`, see `close_run`) gives up the exit push and nothing more: the
    driver's exit 3 stays exit 3 and keeps the reason that explains it. Any
    exception out of the housekeeping is reported on stderr and the run still
    leaves with `code`: a failing step of closing it down must not turn exit
    3 into a traceback and exit 1.
    """
    try:
        close_run(ctx, usages=usages, ending=ending, mailbox=mailbox,
                  pusher=pusher, push_abort=push_abort, budget=budget,
                  heard=heard)
    except Exception:
        print(f"  ⚠ closing the run down failed; it still exits with its own "
              f"code ({code}):", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    sys.exit(code)


class RunBoundary:
    """One runner's epilogue boundary: from its first `open_usage` to `close_run`.

    Every ending inside it closes the run down exactly once. The endings the
    runner writes go through its doors — `end` (a return, `end_run`), `exit`
    (`exit_run`) and `interrupt` (the operator's Ctrl+C) — and `closed` turns
    true as one is taken, so the boundary leaves an ending that is already
    being closed alone. An exiting door records its reason itself, from
    `counts`, BEFORE the housekeeping: no caller has to remember the order,
    and a door added later cannot forget it. Whatever else unwinds the block is
    `unwind`'s:

      * KeyboardInterrupt — only where SIGINT is not the run's event
        (`ctrlc.captured`: a runner called off the main thread) or a caller
        raised one itself: taken through `interrupt`;
      * SystemExit — an exit nobody closed down, such as `streamrender`'s
        exit 2 for a provider executable that is not installed: recorded as
        that exit, closed as "exit N", the code kept;
      * any other exception — out of the driver or the runner body: closed as
        "unhandled <Type>" and let go on unchanged, for the excepthook to
        record. An exception the housekeeping raises itself is reported on
        stderr, never put in the original's place.

    The last two, and the `exit` door, have a StopBudget of
    UNWIND_PUSH_DEADLINE_S, taken as the ending begins: nobody may be there.
    `interrupt` has none, its operator being there to press Ctrl+C again,
    and neither has `end` (a normal ending). Every ending's code and reason
    are chosen before its housekeeping and kept: a Ctrl+C pressed once it has begun gives up the exit push
    (`close_run`) and nothing more — except that a RETURN turns into the
    interrupt's exit (`end_run` says why).

    What a run acquires as it goes, the boundary holds, and every ending
    releases it exactly once — the way a destructor would:

      * the usage pairs, opened through `open_usage` here, which holds each
        one from BEFORE its opening snapshot (a query that can raise after
        the source has started what it keeps running); closed by `close_run`;
      * whatever must stop before the housekeeping begins — a fleet that
        must not go on starting agents through the exit push — registered
        with `on_ending` and released, last acquired first, as the ending
        begins (`_begin_ending`);
      * the pusher, the owner of the run's git (`pusher`, set by the runner
        once it has one): `close_run` hands it the exit push and closes it on
        every ending, and `__exit__` tells it to close once more, without
        waiting, for an ending whose housekeeping raised before it got there.
        Before the boundary opens there is no pusher to release: neither
        runner STARTS one outside its boundary, so an exit before it — the
        waits for a stale stop file and for `--start-in`, a report, the
        prologue failing — leaves no git owner behind and needs no path of
        its own. One the run made and had not started yet is `close_run`'s
        throwaway-owner case;
      * what is shown while the run works — the status region, and the
        parallel runner's console route with its owner — entered through
        `hold` and released, last held first, at `release_held`: where the
        runner's work ends on a normal ending, and at the latest as the
        `with` block is left. That makes the order of the two halves of the
        ending the boundary's, not the caller's:

          - an ending that UNWINDS the block (an exception, a SystemExit nobody
            wrote, a KeyboardInterrupt) releases them FIRST and is closed down
            after, so the housekeeping is not printed over a pinned region;
          - a door taken while they are still held closes the run down UNDER
            them, and they are released as its exit leaves the block. Only
            the sequential loop does that, and correctly: it prints inside
            its region on every iteration and pushes there on every pass, so
            a few more lines are what that region already carries. The
            parallel runner releases them before any of its doors.

    `counts` (`(iterations, completed)` for the exit record) is read when the
    ending comes, and `mailbox`/`pusher` are attributes a runner sets once it
    has them.

    Used as a context manager, or by hand from an `except` (`unwind`).
    """

    def __init__(self, ctx: RunContext, *,
                 counts: Callable[[], Tuple[int, int]],
                 mailbox=None,
                 pusher: Optional[OwnerThread] = None,
                 push_abort: Optional[PushAbort] = None):
        self.ctx = ctx
        self.counts = counts
        self.mailbox = mailbox
        self.pusher = pusher
        self.push_abort = push_abort
        # Every usage pair this run opened, each held from before its opening
        # snapshot; None for an account without a usage endpoint.
        self.usages: List[Optional[RunUsage]] = []
        self._releases = contextlib.ExitStack()
        self._held = contextlib.ExitStack()
        self.closed = False

    def __enter__(self) -> "RunBoundary":
        return self

    def __exit__(self, exc_type, error, traceback_) -> bool:
        try:
            try:
                self.release_held()
            except BaseException as released:
                # Raised past the original, as a nested `with` would have: the
                # region's own teardown failing (a Ctrl+C in its bounded waits)
                # is the ending the block now unwinds with.
                if not self.closed:
                    self.unwind(released)
                raise
            if error is not None and not self.closed:
                self.unwind(error)
        finally:
            if self.pusher is not None:
                # A short wait, not 0: close takes the owner's lock with this
                # timeout, and a zero try can miss it and leave the pusher OPEN.
                self.pusher.close(timeout=0.5)
        return False

    def hold(self, resource):
        """Enter `resource` (a context manager) and hold it until
        `release_held`; returns what its `__enter__` returned. One whose
        `__enter__` raised is not held, as with `with`."""
        return self._held.enter_context(resource)

    def release_held(self) -> None:
        """Release what `hold` holds, last held first; once — a second call,
        and the boundary's own at the end of its block, find nothing left.

        Released without the exception that may be unwinding the block: none
        of them asks. One that raises costs the ones under it nothing
        (ExitStack runs them all) and is raised once they have run."""
        self._held.close()

    def open_usage(self, driver, provider: str, *,
                   name: Optional[str] = None) -> Optional[RunUsage]:
        """`open_usage` for this run, the pair held here from before its
        opening snapshot, so every ending closes it."""
        return open_usage(driver, provider, name=name,
                          dry_run=self.ctx.dry_run,
                          register=self.usages.append)

    def on_ending(self, release: Callable[[], None]) -> None:
        """Run `release` once as the run's ending begins, whichever ending it
        is, before its housekeeping — after every release registered later."""
        self._releases.callback(release)

    def _begin_ending(self) -> None:
        """Take the ending (`closed`) and release what `on_ending` holds.

        A release that raises is reported on stderr and costs the releases
        under it nothing (ExitStack runs them all), and never the ending: the
        ending is already decided, and a failing release must not replace it.
        """
        self.closed = True
        try:
            self._releases.close()
        except Exception:
            print("  ⚠ releasing what the run held failed; it closes down "
                  "anyway:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    def _close_kwargs(self) -> dict:
        return dict(usages=list(self.usages), mailbox=self.mailbox,
                    pusher=self.pusher, push_abort=self.push_abort)

    def _door(self, code, *, ending: str, reason: str,
              budget: Optional[StopBudget] = None,
              heard: Optional[int] = None) -> NoReturn:
        # This ending's own presses, read BEFORE the releases and the record:
        # a press during either is past them, and gives up the exit push.
        if heard is None:
            heard = ctrlc.current().presses
        self._begin_ending()
        iterations, completed = self.counts()
        counts = (self.ctx.progress.record_counts(iterations, completed)
                  if self.ctx.progress is not None else
                  dict(iterations=iterations, completed=completed))
        exitlog.set_reason(reason, **counts)
        exit_run(self.ctx, code, ending=ending, budget=budget, heard=heard,
                 **self._close_kwargs())

    def end(self, result: RunResult) -> RunResult:
        """The normal ending: `end_run` — or `interrupt`, once Ctrl+C was
        pressed. A run asks its Interrupt at its boundaries, and this is the
        last of them: a press since the one before is the run's ending."""
        if ctrlc.current().requested:
            self.interrupt()
        self._begin_ending()
        return end_run(self.ctx, result, local_counts=self.counts(),
                       **self._close_kwargs())

    def exit(self, code, *, ending: str, reason: str) -> NoReturn:
        """An ending the runner exits from: `reason` recorded, then `exit_run`
        under a StopBudget of UNWIND_PUSH_DEADLINE_S — the driver stopping
        the run and five provider errors in a row come in unattended runs."""
        self._door(code, ending=ending, reason=reason,
                   budget=StopBudget(UNWIND_PUSH_DEADLINE_S))

    def interrupt(self) -> NoReturn:
        """The operator's Ctrl+C: INTERRUPTED_REASON, "interrupted", exit
        `ctrlc.EXIT_CODE`. Any press past the FIRST gives up the exit push.

        The first, because that is where this ending began, whenever the
        runner got to its door: the fleet's join (`parallel._Interrupt.hear`)
        has already given a second press its meaning — stop waiting — and a
        mark read here, after it, waited for a third press over a stuck push
        (llm-loop review of 0072, F2). Zero for a KeyboardInterrupt with no
        press behind it: then the first press is already past it."""
        self._door(ctrlc.EXIT_CODE, ending="interrupted",
                   reason=INTERRUPTED_REASON,
                   heard=min(ctrlc.current().presses, 1))

    def unwind(self, error: BaseException) -> None:
        """Close the run `error` is unwinding; returns only for an exception,
        which the caller lets go on. A no-op once a door has been taken."""
        if self.closed:
            return
        if isinstance(error, KeyboardInterrupt):
            self.interrupt()
        # Taken once, here: everything this ending waits for spends it.
        budget = StopBudget(UNWIND_PUSH_DEADLINE_S)
        if isinstance(error, SystemExit):
            code = 0 if error.code is None else error.code
            # The first line only: `sys.exit("message")` may carry a paragraph.
            first_line = (str(code).splitlines() or [""])[0]
            self._door(code, ending=f"exit {first_line}",
                       reason=exitlog.describe_exception(SystemExit, error),
                       budget=budget)
        self._begin_ending()
        iterations, completed = self.counts()
        if self.ctx.progress is not None:
            exitlog.note(**self.ctx.progress.record_counts(iterations, completed))
        try:
            close_run(self.ctx, ending=f"unhandled {type(error).__name__}",
                      budget=budget, **self._close_kwargs())
        except Exception:
            print("  ⚠ closing the run down failed while an exception unwound "
                  "it:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
