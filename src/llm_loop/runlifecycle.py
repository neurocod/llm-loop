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

The epilogue comes in two doors for one body. `end_run` is the ending a runner
RETURNS from; `close_run` is the same housekeeping for the three endings that
`sys.exit` instead — a driver stopping the run with an exit code, five provider
errors in a row, Ctrl+C in the parallel runner. Those three used to write only
`exitlog.set_reason`, so the endings with the most to explain were the ones that
pushed nothing and reported nothing.

THE ORDER OF THE STEPS IS LOAD-BEARING, and this module exists to state it once
rather than to tidy it:

  * the tee goes up BEFORE `exitlog.begin`, because `begin` is what prints the
    report of the PREVIOUS run that vanished, and that report has to land in the
    mirror log whose abrupt end it explains. Swap them and the report goes to a
    terminal nobody is reading any more. Pinned by
    `tests/test_exit_reason.py::test_the_report_of_a_vanished_run_lands_in_the_log`;
  * the exit push takes the caller's lock around the WHOLE of `final_git_push`,
    the `git_unpushed_count` inside it included, because the runner with threads
    may still have a pusher inside `git push` (see `parallel.PUSHER_JOIN_TIMEOUT_S`).
    Reading the count outside the lock is how "nothing to push" could be printed
    about a repository that was being pushed at that moment. Pinned by
    `tests/test_git_push.py`.
"""

import contextlib
import os
import sys
from typing import Any, NamedTuple, Optional

from . import (console, exitlog, limits, operator, projectroot, statusline,
               stopchannel)
from .gitpush import (
    GIT_PUSH_POLICY,
    GIT_PUSH_SETTING,
    GitPushPolicy,
    final_git_push,
)
from .providers import provider_spec, set_live_messages, usage_source_for
from .stopchannel import RunResult
from .scriptlock import ensure_script_lock


class RunSettings:
    """The script's own knobs, held in one MUTABLE object the run re-reads.

    Plain locals froze these at startup, which made "edit the limits while the
    run goes" (the status line's `l` key) impossible without touching the runner
    body. Both runners read this object where the value is USED — the sequential
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


def script_settings(run_settings: RunSettings, progress=None) -> Any:
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
    # The knob registry over `settings` (see script_settings).
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
    """
    if not getattr(args, "dry_run", False):
        ensure_script_lock()

    provider = getattr(args, "provider", None) or driver.provider
    spec = provider_spec(provider)
    driver.provider = provider

    # No wrapper above us: this call is the whole invocation, so its own --max is
    # the invocation cap and it owns the figures.
    owns_progress = progress is None
    if owns_progress:
        progress = statusline.InvocationProgress(max_items=args.max)

    settings = RunSettings(max_runs=args.max,
                           git_push=GitPushPolicy(args.git_push))
    registry = script_settings(settings, progress if owns_progress else None)
    dry_run = bool(getattr(args, "dry_run", False))

    # Anchor every project-relative operation (git/provider cwd, the stop file,
    # the log name, the Driver's paths) before anything reads the root.
    projectroot.set_project_root(getattr(args, "project_dir", None))

    # Decided per invocation, before the first argv is built: the transport is
    # what --no-live-messages turns off, and both the argv and the process's
    # stdin have to agree about it. Set in BOTH directions — a wrapper that calls
    # two runners in one process (see runGenerateModels' parallel mode, which
    # alternates product batches with kit-promotion passes) would
    # otherwise have the first `--no-live-messages` phase decide the transport
    # for every phase after it.
    set_live_messages(not getattr(args, "no_live_messages", False))

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
        exitlog.begin(app_name, console.LOG_DIR,
                      os.path.basename(projectroot.project_dir()))
    print(f"  · project root: {projectroot.project_dir()}")
    if dry_run:
        print(f"  · dry run: nothing is mirrored to "
              f"{console.log_file_path(app_name)}")
    else:
        print(f"  · logging to {console.log_file_path(app_name)}")
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
        settings=ctx.registry,
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
    answered by `at end (name)` unless the run ended somewhere worth naming
    instead (see `close`). Only the usage a run ENDS on is closed: a
    mixed-provider sequential run opens one per provider it selects, and the
    others' openings stay unanswered.
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
        `ending` names an abnormal ending in place of the usage's own name.
        """
        self.policy.log_snapshot(self.source, f"at end ({ending or self.name})",
                                 cache_value=False)


def open_usage(driver, provider: str, name: str, *,
               dry_run: bool) -> Optional[RunUsage]:
    """The provider's usage pair with its opening snapshot, or None without one.

    None when the provider has no usage endpoint (`usage_source_for`); whether
    to open one at all is the runner's call — `--ignore-usage` skips this in the
    parallel runner. The policy is the Driver's specialisation when it has one,
    the provider's default otherwise. A dry run gets the pair (the gate and the
    status line read it) but no snapshot, because it is not a run.
    """
    source = usage_source_for(provider)
    if source is None:
        return None
    usage = RunUsage(source,
                     driver.limit_policy or limits.default_policy(provider),
                     name)
    if not dry_run:
        usage.open()
    return usage


def usage_halves(usage: Optional[RunUsage]) -> tuple:
    """`(source, policy)` of a usage pair, or `(None, None)` without one.

    The gate, the status line and the parallel workers take the two halves
    separately; this is the one place they are split, so a runner never holds a
    source and a policy that came from different pairs.
    """
    return (usage.source, usage.policy) if usage is not None else (None, None)


def close_run(ctx: RunContext, *,
              usage: Optional[RunUsage] = None,
              ending: Optional[str] = None,
              mailbox=None,
              push_lock=None) -> None:
    """The housekeeping half of the epilogue, for every ending a run can have.

    Push what is still local, record where the quotas finished, report the notes
    nobody delivered. A runner prints its own closing report BEFORE calling this
    — "Final state: …", "Processed N file(s) …" — because that is the run talking
    about its work, and everything here is closing it down.

    Separate from `end_run` because three endings are NOT returns: the driver
    stopping the run with an exit code, five provider errors in a row, and Ctrl+C
    in the parallel runner all `sys.exit`, so they have a reason to record but no
    `RunResult` to hand back. They used to write only `exitlog.set_reason` and
    leave — no exit push, no closing snapshot, no report of undelivered notes —
    which meant the endings that most need a post-mortem were the ones that left
    the least behind, and an operator's commits sat local until some later run
    happened to push them. Each of those now calls this and then exits.

    `push_lock` is the caller's mutual exclusion, and it wraps the WHOLE of
    `final_git_push`, the `git_unpushed_count` inside it included. Only the
    runner with threads passes one: the sequential runner has nothing to exclude,
    and `gitpush.final_git_push` deliberately does not lock for itself (see its
    docstring). The policy is read INSIDE that lock, off the live settings — a
    knob edited while a pusher is mid-push must not be read half-applied.
    """
    if not ctx.dry_run:
        # No lock is the SINGLE-THREADED case, not a missing one: the runner with
        # threads is the only caller with anything to exclude.
        with push_lock if push_lock is not None else contextlib.nullcontext():
            final_git_push(ctx.settings.git_push, projectroot.project_dir())

    # End-of-run usage snapshot, answering the one `open_usage` logged — so each
    # run records where it finished. `ending` names the abnormal endings; a
    # normal one closes under the usage's own name (see RunUsage.close).
    if not ctx.dry_run and usage is not None:
        usage.close(ending)

    operator.report_undelivered_notes(mailbox)


def end_run(ctx: RunContext, result: RunResult, *,
            usage: Optional[RunUsage] = None,
            mailbox=None,
            push_lock=None) -> RunResult:
    """Everything both runners do when the work is over and they RETURN.

    The housekeeping is `close_run`; this adds the two things only a normal
    ending has — a `RunStopReason` to name it by, and the `RunResult` handed back
    to whoever called the runner.

    The reason is RECORDED rather than printed: a wrapper may call several
    runners, the `=== run ended: … ===` line belongs to the process, so the last
    reason set wins and exitlog prints it on the way out.
    """
    close_run(ctx, usage=usage, mailbox=mailbox, push_lock=push_lock)
    reason = result.reason
    exitlog.set_reason(stopchannel.STOP_REASON_TEXT.get(reason, reason.value),
                       iterations=result.attempted, completed=result.completed)
    return result
