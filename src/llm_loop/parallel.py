"""
parallel.py - parallel sibling of the sequential ListFileDriver loop.

Processes the files listed in a ListFileDriver's list, but with N worker threads
running the selected provider concurrently instead of one file at a time. The work
parallelises cleanly because each item is independent: a worker reads its own
source and writes its own target; the only shared mutable state is the list file
itself (each finished path is struck out of it), guarded by one lock, so the run
stays idempotent — stop any time and relaunch and whatever is still listed gets
picked up again.

What this shares with the sequential runner, and what it deliberately drops:

  * Reused — the ListFileDriver (list parsing, is_pending, pick, command_for,
    strike),
    build_agent_argv, the usage session-limit machinery, the git-push policy
    and the rotating mirror log. Every one of those is now a module of its own,
    so this runner no longer imports the sequential one at all: the two are
    siblings over shared parts, not a runner and its borrower. Opening and
    closing a run is `runlifecycle`, and it is shared as CODE rather than as a
    comment saying "the same as over there" — which is what the two runners used
    to have, and what let `--git-push` be a live knob in one of them and a frozen
    local in the other.
  * Dropped — the live token-by-token Markdown rendering. cyclecore's stream
    renderer keeps module-global state that cannot serve several concurrent
    streams without garbling, so here each worker posts one compact, fully
    formed line per event, prefixed `[job k]`, to the one thread that writes
    them (`_console`). You trade the live view for throughput — the right call
    for mechanical bulk work.

CLI mirrors the family (see `--help`) because both parsers are built from the
same table, `clispec.OPTIONS`; what this mode adds there is `-j/--jobs N`
(`clispec.DEFAULT_JOBS`), and `--max-runs N` caps the *total* number of files
processed this run (across all workers), not iterations.
"""

import argparse
import collections
import json
import os
import sys
import threading
import time
from typing import Callable, Optional
from uuid import uuid4

from . import clispec
from . import compactline
from . import console
from . import costlog
from . import ctrlc
from . import exitlog
from . import operator
from . import ownership
from . import projectroot
from . import providers
from . import runlifecycle
from . import statusline
from . import statlog
from . import stopchannel
from . import textwidth
# BY NAME, not through the module, and that is the repair of a real defect: this
# function used to reach `usage.rate_limit_event_from` while its codex branch
# bound a LOCAL called `usage` (`usage = ev.get("usage") or {}`), which made the
# name local to the whole function — so every rate-limit verdict in a claude
# worker raised UnboundLocalError instead of being reported, and the parallel
# runner's half of the limit backstop never worked. The local is gone (the token
# counts come through `wire` now) and a bare name cannot be shadowed by one
# appearing again. Pinned by
# `test_providers.test_a_worker_surfaces_the_runs_own_rate_limit_verdict`.
from .usage import rate_limit_event_from
# The words the provider stream is made of, shared with the other renderer:
# this runner and `streamrender` read the SAME events, and 32 of the literals
# they read them with used to be spelled in both files (see `wire`).
from . import wire
from .agentwork import (
    AgentCommand,
    build_agent_argv,
)
from .console import print_markup, route_through
# Only the per-turn call: the exit push (and the policy enum with it) belongs to
# the shared epilogue, `runlifecycle.end_run`, which is where both runners close
# a run down and therefore the one place that decides how the exit push is
# guarded.
from .gitpush import PushAbort, maybe_git_push
from .stopchannel import RunResult, RunStopReason
from .providers import (ask_agent_process_to_end, note_channel, provider_spec,
                        reap_agent_process, start_agent_process)
from .drivers import ListFileDriver

# How many of a failed job's discarded non-JSON lines are kept as its failure
# explanation. The reason a dying CLI gives is in its last lines, and the bound
# is what keeps a chatty one from growing a long-running worker.
FAILURE_TAIL_LINES = 5

# How often the pusher applies the git-push policy while the workers run. Not the
# cadence of pushing — EACH_HOUR's hour is its own — but how finely that cadence
# is checked, which is why a minute is plenty. A named constant rather than a
# literal in the pump so a test can shorten it: without that, the pump's push is
# unreachable in a run that lasts less than one interval, and the handover it
# makes (which repository to push) had no pin at all.
PUSH_PUMP_INTERVAL_S = 60

# How long an interrupted run waits for its workers to notice `shared.stop` and
# come back before it closes the run down anyway — the whole fleet together,
# one `runlifecycle.StopBudget`, not each thread in turn (a fleet of ten used
# to cost ten times this). Bounded because the operator has already asked to
# leave. Every turn's CLI was already ended by the press itself (`run_job`'s
# `on_press` hook), so what is waited for is each worker reaping it and giving
# its claim back; a worker that outlasts this is a daemon the exit kills, its
# CLI's tree ended at the press, not by the reap it never gets to.
INTERRUPT_JOIN_TIMEOUT_S = 5

# Per-file retry budget: a path that fails this many times in a row is parked in
# the `failed` set so it stops blocking the queue (and is reported at the end)
# instead of being retried forever.
MAX_ATTEMPTS = 3

# How many pending items an uncapped `--dry-run` lists before summarising the
# rest. One item is one argv line with a whole prompt in it, so this is roughly
# a screenful of preview; the count a real run would process is printed either
# way, so the cap costs no information about the size of the queue.
DRY_RUN_LIST_LIMIT = 10

# The one thread that writes the workers' lines while a run is open: workers
# post, it prints, so the per-job lines cannot interleave and no two of them are
# ever inside `print_markup` at once. `run_parallel` opens it for the worker
# phase and closes it — having written everything posted — before its own
# closing report. Outside that window a line is written by whoever emits it: a
# test driving `run_job` directly, or a worker a Ctrl+C gave up joining that
# outlives the owner too.
#
# The workers post to it by name (`_emit_markup`); every other console write of
# the run — the usage gate on a worker, the background pusher — reaches it
# through `console.route_through`, which `run_parallel` holds open around it.
_console = ownership.OwnerThread("console-lines")

# How long `run_parallel` waits for `_console` to write what the workers posted
# before it prints its own closing report. A healthy console never gets near it:
# a FULL queue (`ownership.DEFAULT_MAXSIZE` compact lines) drains in 2.9 s into a
# file (2.92 / 2.91 / 2.86 s, measured 2026-09-25), and a console is slower than
# a file, hence ten times that. Past it the console is taken to be stuck (a
# selection freezing a Windows console, a pipe nobody reads), and the policy for
# what is still queued is: the owner keeps it and goes on writing it as a
# daemon, the run reports and exits without waiting, and whatever is unwritten
# when the process ends is lost. One stderr line says how much was left behind.
CONSOLE_CLOSE_TIMEOUT_S = 30.0

# What Ctrl+C prints. A constant because it can be printed from two places:
# posted to the console by `_Interrupt.hear`, or by `run_parallel` after the
# console has closed when the queue had no room for it.
INTERRUPT_ANNOUNCEMENT = ("\nInterrupted by user (Ctrl+C) — "
                          "signalling workers to stop…")


def _close_console() -> None:
    """Close `_console` within CONSOLE_CLOSE_TIMEOUT_S, naming what it left."""
    if _console.close(CONSOLE_CLOSE_TIMEOUT_S):
        return
    print(f"  ⚠ {_console.name}: {_console.backlog} line(s) still unwritten "
          f"after {CONSOLE_CLOSE_TIMEOUT_S:g} s — reporting without them.",
          file=sys.stderr)


class _Interrupt:
    """What the fleet does about the run's Ctrl+C (`ctrlc.Interrupt`).

    `hear` is the first press inside the status region: the fleet stopped,
    announced, and waited for a bounded time. `stop_workers` is the part of
    that every ending past the region needs, an exception's included.
    """

    def __init__(self, shared: "Shared", threads: "WorkerPool",
                 interrupt: ctrlc.Interrupt):
        self._shared = shared
        self._threads = threads
        self._interrupt = interrupt
        # The Ctrl+C line when the console's queue had no room for it (see
        # `hear`); the run prints it after its console has closed.
        self.announce_later: Optional[str] = None
        self._heard = False

    def hear(self) -> None:
        """What a Ctrl+C does to the fleet — once, however often it is asked."""
        if self._heard:
            return
        self._heard = True
        heard = self._interrupt.presses
        # Signal first, talk second: nothing about the console — a stalled
        # one included — may stand between Ctrl+C and the workers hearing it.
        # Queued, not printed, so the line lands after what the workers had
        # already said — and never waited for: a queue that is full is a
        # console that is not being written, and this line is then deferred
        # to the closing report rather than blocking the interrupt behind it.
        self.stop_workers()
        if not _console.try_post(print, INTERRUPT_ANNOUNCEMENT):
            self.announce_later = INTERRUPT_ANNOUNCEMENT
        # One budget for the whole fleet, and a further Ctrl+C gives up
        # waiting at once; the run's ending is this one either way. A worker
        # the region never got to start has nothing to wait for. Joined in
        # slices: a thread join does not wake for a signal, and the press is
        # an event polled here.
        budget = runlifecycle.StopBudget(INTERRUPT_JOIN_TIMEOUT_S)
        for t in self._threads:
            if t.ident is None:
                continue
            while t.is_alive():
                if self._interrupt.since(heard) or budget.expired:
                    return
                t.join(timeout=min(budget.left(),
                                   stopchannel.STOP_RECHECK_SECONDS))

    def stop_workers(self) -> None:
        """No further claim and no further worker — and no wait for either.

        What every ending that unwinds the run past the region asks of the
        fleet, an exception's included: its workers must not go on starting
        agents while the run closes down, and the process exits under them
        once it has. Neither announced nor joined, unlike `hear`: an
        exception says its own line, and nobody asked to wait for a turn.
        The stop before the pool: a worker that a `+` holding the pool's lock
        starts anyway leaves on it (`WorkerPool.close`). Both are one store,
        safe to repeat — `hear` and the boundary's release both call this.
        """
        self._shared.stop.set()
        self._threads.close()


def parse_args(argv=None, *, prog: str = "parallel",
               description: Optional[str] = None,
               extra_options: Optional[Callable[[argparse.ArgumentParser],
                                                None]] = None
               ) -> argparse.Namespace:
    """CLI for the parallel runner: the family's options plus -j/--jobs.

    This is no longer a trimmed copy of cyclecore.parse_args. Both parsers are
    built from the one table in `clispec`, which is also where the two real
    differences are now written down instead of inferred by diffing two
    functions: this mode's option list has -j/--jobs and no --cost/--raw/
    --start-in, and the five options whose meaning differs here (--max-runs is a
    total-files cap, not an iteration cap, and so on) carry their own help text.

    `extra_options` is the same wrapper hook cyclecore.parse_args documents — a
    mode switch is usually spelled the same way in both modes, so the two
    parsers have to offer the same seam or its --help would depend on which one
    the wrapper happened to reach.
    """
    return clispec.build_parser(clispec.PARALLEL, prog=prog,
                                description=description,
                                extra_options=extra_options).parse_args(argv)


# --- output helpers: every emit is posted to the console's owner ----------------

def _write_markup(plain: str, markup: str) -> None:
    """One whole line to the console — on `_console`'s thread while a run is open.

    Reads `print_markup` off this module per call rather than closing over it —
    which is what lets the pins replace it (see `compactline.LineWriter`).
    """
    print_markup(plain, markup)


def _emit_markup(plain: str, markup: str) -> None:
    """The sink under every worker's line: post it, do not write it."""
    _console.post(_write_markup, plain, markup)


def _job_tag(job_id: int) -> tuple:
    """(plain, markup) prefix identifying a worker, e.g. '[job 2]'."""
    return f"[job {job_id}]", f"[cyan]\\[job {job_id}][/]"


def job_lines(job_id: int) -> compactline.LineWriter:
    """The compact line shapes, tagged for one worker and posted to `_console`.

    The whole difference between this runner's output and the sequential one:
    every line carries `[job k] ` and one thread writes them all. Both are given
    to `compactline.LineWriter` here, so the shapes themselves — a tool call, a
    head plus what the row leaves beside it, an outcome — exist once for both
    runners, and the tag counts against the width of each of them.
    """
    tag_plain, tag_markup = _job_tag(job_id)

    def emit(plain: str, markup: str) -> None:
        console.record_timing(statlog.pool_worker_activity(job_id))
        _emit_markup(plain, markup)

    return compactline.LineWriter(emit, f"{tag_plain} ",
                                  f"{tag_markup} ")


def emit_note(lines: compactline.LineWriter, note: str) -> None:
    """An operator note attributed to a worker — this runner's `print_note`.

    One place for the glyph, the colour and the label, because a note is
    announced twice (when it rides a prompt, and when the CLI replays one that
    went in live) and the two must not drift into looking like different things.
    Here rather than on the writer because the sequential runner's note is not
    the same line with a tag added: `console.print_note` colours the glyph and
    the label separately, so there is no one shape for the two to share yet.
    """
    lines.line(f"✉ operator note: {note}", "magenta")


def join_workers(threads) -> None:
    """Wait for every worker to finish, or for the operator's Ctrl+C.

    A function of its own for one reason, and it is a real one: this is where a
    run spends all of its time, so it is where Ctrl+C is heard — the run's
    `ctrlc.Interrupt`, asked every STOP_RECHECK_SECONDS, since a thread join
    does not wake for a signal — and the interrupt's own ending (record the
    reason, push, snapshot, report the undelivered notes, exit 130) cannot be
    pinned unless a test can stage the press HERE. Staging it by patching
    `threading.Thread.join` instead would also hit the bounded re-join in
    `_Interrupt.hear`, i.e. it would break the code under test on its way in.
    """
    interrupt = ctrlc.current()
    join_all = getattr(threads, "join_all", None)
    if join_all is not None:
        join_all(stop=lambda: interrupt.requested)
        return
    for t in threads:
        while t.is_alive() and not interrupt.requested:
            t.join(timeout=stopchannel.STOP_RECHECK_SECONDS)


class WorkerPool:
    """The worker threads of one run, resized safely from the status-line thread.

    `join_all` closes additions atomically with observing that the last thread is
    gone. Without that handshake, `+` could append a worker just after the main
    thread had decided its original list was finished, and the run would tear its
    status area down around the new thread.
    """

    def __init__(self, threads, make_thread, prepare_worker,
                 remove_worker=lambda job_id, count: None,
                 finish_removal=lambda count: None):
        self._threads = list(threads)
        # The terminal input starts before run_parallel reaches start_initial().
        # Keep that original set apart so an immediate `+` (whose worker starts
        # at once) is not included and started a second time.
        self._initial_threads = tuple(self._threads)
        self._make_thread = make_thread
        self._prepare_worker = prepare_worker
        self._remove_worker = remove_worker
        self._finish_removal = finish_removal
        self._lock = threading.Lock()
        self._accepting = True
        self._worker_count = len(self._threads)
        self._visible_count = self._worker_count
        self._current = {
            job_id: thread
            for job_id, thread in enumerate(self._threads, start=1)
        }
        self._departed = set()

    def __iter__(self):
        # A stable snapshot for bounded interrupt cleanup and test probes.
        with self._lock:
            return iter(tuple(self._threads))

    def start_initial(self) -> None:
        for thread in self._initial_threads:
            thread.start()

    def grow(self) -> Optional[int]:
        """Start one more worker; return its id, or None once shutdown began."""
        with self._lock:
            if not self._accepting:
                return None
            self._worker_count += 1
            job_id = self._worker_count
            self._prepare_worker(job_id)
            thread = self._current.get(job_id)
            if thread is None:
                thread = self._make_thread(job_id)
                self._current[job_id] = thread
                self._threads.append(thread)
                self._departed.discard(job_id)
                thread.start()
            self._visible_count = max(self._visible_count, self._worker_count)
            return job_id

    def shrink(self) -> Optional[tuple]:
        """Retire the last worker after its current item; never remove worker 1."""
        with self._lock:
            if not self._accepting:
                return None
            if self._worker_count <= 1:
                return (None, 1)
            job_id = self._worker_count
            self._worker_count -= 1
            self._remove_worker(job_id, self._worker_count)
            return (job_id, self._worker_count)

    def retirement_requested(self, job_id: int) -> bool:
        """True at the retired worker's next claim boundary.

        Committing the departure under the pool lock makes a quick `+` either
        cancel the pending removal or start a replacement, never lose the slot
        in the small race while the old thread is returning.
        """
        with self._lock:
            return self._retire_locked(job_id)

    def claim(self, job_id: int, shared) -> tuple:
        """Atomically keep this worker in the pool and claim its next item.

        `-` takes this same lock. Once it returns to the input thread, the
        removed worker therefore cannot slip through a check/claim gap and turn
        an idle slot into one more full provider call.
        """
        with self._lock:
            if self._retire_locked(job_id):
                return False, None
            return True, shared.claim()

    def _retire_locked(self, job_id: int) -> bool:
        """Commit one requested retirement (call under `_lock`)."""
        if job_id <= self._worker_count:
            return False
        self._current.pop(job_id, None)
        self._departed.add(job_id)
        old_visible = self._visible_count
        while (self._visible_count > self._worker_count
               and self._visible_count in self._departed):
            self._visible_count -= 1
        if self._visible_count != old_visible:
            self._finish_removal(self._visible_count)
        return True

    def close(self) -> None:
        """No further worker from `+`, and no `-` either — and no wait.

        Written without the pool's lock on purpose: a worker holds that lock
        across its whole claim (`claim` → `Shared.claim` → the driver reading
        the list file), so a claim stuck in the driver's I/O held every
        ending that closed the pool — an exception's included, before its
        exit push's deadline had even begun. A bool store is atomic under the
        GIL, and `grow`/`shrink` read it under the lock, so every one that
        takes the lock after this sees it. One that already holds it may
        still start its worker; every caller sets `shared.stop` first
        (`_Interrupt.stop_workers`), and that worker leaves at its first look.
        """
        self._accepting = False

    def join_all(self, stop: Optional[Callable[[], bool]] = None) -> None:
        """Wait for every worker, added ones included; with `stop`, also
        return — leaving the pool open — as soon as `stop()` says so, asked
        every STOP_RECHECK_SECONDS."""
        index = 0
        while True:
            if stop is not None and stop():
                return
            with self._lock:
                if index >= len(self._threads):
                    self._accepting = False
                    return
                thread = self._threads[index]
            thread.join(None if stop is None
                        else stopchannel.STOP_RECHECK_SECONDS)
            if not thread.is_alive():
                index += 1


class ResizeWorkerPoolAction(statusline.Action):
    """`+`/`-` — resize this parallel run one worker at a time."""

    key = "+/-"
    keys = ("+", "-")
    help = "add/remove workers"

    def __init__(self, pool: WorkerPool):
        self.pool = pool

    def run_key(self, app, key):
        if key == "+":
            job_id = self.pool.grow()
            if job_id is None:
                app.note("worker pool is already stopping")
                return
            app.note(f"worker {job_id} started — {job_id} workers now active")
            return

        result = self.pool.shrink()
        if result is None:
            app.note("worker pool is already stopping")
            return
        job_id, count = result
        if job_id is None:
            app.note("at least one worker must remain active")
            return
        app.note(f"worker {job_id} retiring after its current file — "
                 f"{count} worker{'s' if count != 1 else ''} remain active")


# --- one provider round-trip for one file --------------------------------------

def run_job(job_id: int, command: AgentCommand, mailbox=None) -> tuple:
    """Run one provider command, rendering a compact per-job trace.

    Unlike cyclecore's streaming renderer this prints only the key events — each
    tool call, any failed tool result, and the final cost line — one whole line
    per post to `_console`, so several of these can run at once without their
    output colliding. Returns (returncode, cost_usd, duration_s).

    `mailbox` belongs to this worker; it lends the console this turn's stdin,
    exactly as the sequential runner does.
    """
    out = job_lines(job_id)
    provider = command.provider or "claude"
    spec = provider_spec(provider)
    argv = build_agent_argv(command, provider)
    try:
        proc = start_agent_process(
            argv, provider, command.prompt, projectroot.project_dir())
    except FileNotFoundError:
        out.line(f"executable {spec.executable!r} not found on PATH.", "bold red")
        return 2, None, None

    cost_usd = None
    duration_s = None
    provider_failed = False
    codex_outcome = wire.CodexOutcome()
    # The child's stderr is merged into its stdout (start_agent_process), so a
    # provider that dies with a plain-text message says so on these skipped
    # lines. Compact mode drops them, which is how a job once failed with
    # `exit 1` and no cause anywhere. Kept as a bounded tail — a chatty CLI must
    # not be able to grow a worker's memory — and printed only if the job fails.
    diagnostics = collections.deque(maxlen=FAILURE_TAIL_LINES)
    # The run's Ctrl+C ends this turn's CLI from the press itself, the way
    # `streamrender.run_agent_streaming` ends the sequential one's. Not left
    # to the console: the child is started without one of its own
    # (`providers._console_isolation`) and never hears Ctrl+C, and a run that
    # gives up joining this worker exits under it — a daemon thread dies
    # without its `finally`, so the reap below never comes and an agent nobody
    # sees goes on editing the tree after the run has left (review of 0072, F3).
    interrupt = ctrlc.current()
    # Everything from here down to `proc.wait()` runs with a child process
    # alive, and every step of it can raise: formatting an `out.*` line (and,
    # with no run open, the console write behind it), `note_channel`'s close, a
    # decoder error on the child's own stream. `wait()` is the only exit that
    # reaps, so an exception used to walk away from a running provider — see
    # `providers.reap_agent_process`.
    try:
        with note_channel(proc, provider, mailbox) as channel, \
                interrupt.on_press(lambda: ask_agent_process_to_end(proc)):
            # A press between the worker's last look and the hook going in
            # had nobody to end this CLI: asked here, after it is in.
            if interrupt.requested:
                ask_agent_process_to_end(proc)
            for item in proc.stdout:
                if interrupt.requested:
                    break
                if isinstance(item, dict):
                    # An app-server event arrives decoded
                    # (`providers._CodexEventStream`).
                    ev = item
                else:
                    line = item.rstrip("\n")
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        # Printed indented under a head of its own if the job fails.
                        diagnostics.append(
                            compactline.short(line, out.budget("  ")))
                        continue  # non-JSON CLI diagnostics — skip in compact mode
                    if not isinstance(ev, dict):
                        continue  # valid JSON can still be a diagnostic, not an event
                et = wire.event_type(ev)
                if provider != "codex":
                    # The row's resolved model and window (see `statusline.describing`).
                    statusline.observe_claude_event(ev)
                if provider == "codex":
                    statusline.observe_codex_event(ev)
                    codex_outcome.observe(ev)
                    if et in (wire.TURN_COMPLETED, wire.TURN_FAILED):
                        channel.close()
                    item = wire.codex_item(ev)
                    item_type = wire.codex_item_type(item)
                    if et == wire.ITEM_COMPLETED and item_type == wire.AGENT_MESSAGE:
                        for text in wire.codex_message_text(item).splitlines():
                            out.line(f"💬 {text}")
                    elif (et == wire.ITEM_COMPLETED
                          and item_type == wire.USER_MESSAGE
                          and mailbox is not None):
                        note = mailbox.claim_echo(
                            wire.codex_user_message_text(item))
                        if note is not None:
                            emit_note(out, note)
                    elif et == wire.ITEM_STARTED and item_type == wire.COMMAND_EXECUTION:
                        out.fitted("💻 ", wire.codex_command(item))
                    elif et == wire.ITEM_COMPLETED and item_type == wire.COMMAND_EXECUTION:
                        # An empty slot rather than the live renderer's dropped
                        # prefix, for a command that carried no code — see
                        # `wire.codex_exit_code` for why the default is asked for
                        # here rather than decided there.
                        out.fitted(f"📤 exit {wire.codex_exit_code(item, '')}: ",
                                   wire.codex_command(item))
                    elif et == wire.ITEM_COMPLETED and item_type == wire.FILE_CHANGE:
                        paths = wire.codex_changed_paths(item)
                        out.line(f"🛠️ {', '.join(paths) or 'file changes applied'}")
                    elif et == wire.TURN_COMPLETED:
                        counts = wire.codex_token_counts(ev)
                        if counts is not None:
                            tokens_in, cached, tokens_out = counts
                            out.line(f"tokens: input {tokens_in}, "
                                     f"cached {cached}, output {tokens_out}")
                    elif et in (wire.ERROR, wire.TURN_FAILED):
                        out.fitted("⚠ ", wire.codex_error(ev), "bold red")
                elif et == wire.ASSISTANT:
                    for block in wire.message_blocks(ev):
                        if wire.block_type(block) == wire.BLOCK_TOOL_USE:
                            out.tool_use(wire.tool_use_name(block),
                                         wire.tool_use_input(block))
                elif et == wire.USER:
                    # Only surface *failed* tool results; successes would just be
                    # noise at high concurrency. An operator note replayed back to us
                    # is the exception: it is the receipt for something a human
                    # typed, and it belongs in the log next to the turn it landed in.
                    for block in wire.message_blocks(ev):
                        if (wire.block_type(block) == wire.BLOCK_TEXT
                                and mailbox is not None):
                            note = mailbox.claim_echo(wire.block_text(block))
                            if note is not None:
                                emit_note(out, note)
                        if (wire.block_type(block) == wire.BLOCK_TOOL_RESULT
                                and wire.tool_result_failed(block)):
                            out.fitted("  ✗ ", wire.tool_result_text(block), "red")
                elif et == wire.RATE_LIMIT_EVENT:
                    # The run's own rate-limit verdict (see usage.RateLimitEvent).
                    # Surfaced, not acted on: with N workers the pause belongs to the
                    # shared usage gate, which sees the same wall as a pegged
                    # percentage when the next worker checks in.
                    #
                    # A LOCAL, deliberately: the sequential renderer latches its
                    # last verdict in a module global because it has exactly one
                    # stream: here there are `jobs` of them at once, so "the last
                    # verdict" is not a question this runner can answer.
                    rl = rate_limit_event_from(ev)
                    if rl is not None and rl.status != "allowed":
                        out.line(f"⚠ rate limit: {rl.describe()}", "bold red")
                elif et == wire.RESULT:
                    # Before the figures: once the turn has reported, the console
                    # must not be able to write into a session that is closing.
                    channel.close()
                    provider_failed = provider_failed or wire.result_failed(ev)
                    # A process emits a second `result` when a late note is answered
                    # as its own turn. The two figures then have to be combined
                    # differently, which is measured rather than assumed:
                    # `total_cost_usd` is the session's running total (so the last
                    # one is the job's cost), `duration_ms` is that turn's alone (so
                    # they add up).
                    previous_cost = cost_usd or 0.0
                    result_cost = wire.result_cost(ev)
                    cost_usd = wire.result_cost(ev, cost_usd)
                    dur = wire.result_duration_ms(ev)
                    if dur is not None:
                        duration_s = (duration_s or 0.0) + dur / 1000
                    if not wire.result_failed(ev):
                        console.record_timing(costlog.pool_result(
                            job_id, dur, (result_cost - previous_cost
                                          if result_cost is not None else None)))
        if interrupt.requested:
            # Whatever the CLI had done: the operator asked for the run, not
            # this turn, to end. The worker gives the claim back (`worker`).
            out.line("⏹ interrupted (Ctrl+C)", "yellow")
            return ctrlc.EXIT_CODE, cost_usd, duration_s
        # Outside the `with`, so the pipe is closed before we wait on the process:
        # a worker waiting on a CLI whose stdin is still open never returns, and a
        # run whose final join() never finishes is the whole fleet.
        returncode = proc.wait()
    finally:
        reap_agent_process(proc)
    if provider == "codex":
        provider_failed = codex_outcome.failed
        outcome_code = codex_outcome.exit_code(returncode)
        if outcome_code:
            out.line(codex_outcome.describe(returncode), "bold red")
        returncode = outcome_code
    if returncode == 0 and provider_failed:
        returncode = 1
    # Last resort only: a codex error/turn.failed already printed its own ⚠ line
    # live, so repeating the tail there would be noise. A failure with no other
    # explanation is exactly what must never print bare again. Emitted here, so
    # the lines sit immediately above the worker's ✗ verdict for this job.
    if returncode != 0 and not provider_failed and diagnostics:
        out.line(f"provider output before exit {returncode}:", "red")
        for text in diagnostics:
            out.line(f"  {text}", "red")
    return returncode, cost_usd, duration_s


# --- shared queue state, guarded by one lock -----------------------------------

class Shared:
    """Cross-worker state behind a single lock: the list cursor and run stats.

    The list file (owned by `driver`) is the source of truth for what remains;
    `in_progress` keeps workers from claiming the same line, `failed` parks lines
    that exhausted their retry budget, and the counters bound/report the run. All
    access to THAT state is under `lock`.

    `settings` is the exception, and it is not this object's state: the run's
    knobs are written by whoever holds the keyboard (the status line's editor,
    on its own thread) and only READ here, under the lock, at the moment a claim
    is decided — see `max_items`.
    """

    def __init__(self, driver: ListFileDriver, settings):
        self.driver = driver
        self.log_session = uuid4().hex
        # The run's live knobs (runlifecycle.RunSettings), not a copy of the cap
        # taken here: `--max-runs` is editable from the status line, and the
        # claim loop below is the one place that enforces it — so it has to read
        # the value the editor writes, at the moment it claims. See `max_items`.
        self.settings = settings
        self.lock = threading.Lock()
        self.in_progress = set()      # raw lines a worker is currently handling
        # The subset of those whose provider turn is actually in flight. Not the
        # same question as `in_progress`, and the difference is what a pending
        # stop turns on: a worker parked on the usage gate holds a claim but is
        # not doing anything a stop would interrupt, so it must not be counted as
        # work in flight — see busy().
        self.running = set()
        self.failed = set()           # raw lines parked after MAX_ATTEMPTS
        self.attempts = {}            # raw line -> failed-attempt count
        self.claimed = 0              # files claimed this run (for --max-runs)
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

    def _pending(self) -> list:
        """Lines nobody holds and nobody gave up on (call under `lock`)."""
        return [ln for ln in self.driver.pending_lines()
                if ln not in self.in_progress and ln not in self.failed]

    def _exhausted(self, pending: list) -> bool:
        """Has this run run out of work to give? (call under `lock`)

        The two ways it ends of its own accord — the item cap, and a queue that
        drained with nobody still working — latched here rather than inside
        `claim`, because a worker that is NOT claiming has to be able to ask.
        See `exhausted`, and the pause hold in `worker` that uses it.

        A queue that is merely empty right now is not exhausted: the rest is in
        flight elsewhere, and the caller backs off and retries.
        """
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
        for one anyway never sets `stop`, so `run_parallel`'s join() never
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
            self.in_progress.add(line)
            self.claimed += 1
            in_flight = sorted(os.path.basename(ln.strip())
                               for ln in self.in_progress)
            claimed, done = self.claimed, self.done
        # Outside the lock — this writes a file, and every worker contends for
        # that lock. It leaves behind what the run had in flight, which is what
        # a post-mortem of a killed run has to start from (see exitlog).
        exitlog.note(phase=f"in flight: {', '.join(in_flight)}",
                     iterations=claimed, completed=done)
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
            self.in_progress.discard(line)
            # A turn Ctrl+C cut short is released too (`worker`), after its
            # `start_turn`: a release that left the line in `running` would
            # make busy() true forever and hang every later stop request.
            self.running.discard(line)
            if self.claimed > 0:
                self.claimed -= 1

    def start_turn(self, line: str) -> None:
        """This claim is about to become a running provider turn (see busy())."""
        with self.lock:
            self.running.add(line)

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
            increments `attempts`, and `attempts` is the only thing that can
            ever park a line in `failed`. A file whose turn reliably kills its
            worker (a provider CLI that dies mid-stream, a console that refuses
            a write) would otherwise be handed back untouched and kill the next
            worker, and the next — a fleet of ten dies ten times over one line.
            Counted, the same line stops the run after MAX_ATTEMPTS workers and
            is reported as failed, which is also what it is.

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

    def finish(self, line: str, ok: bool) -> tuple:
        """Record an item's outcome: strike it on success, or count/park a fail.

        Returns (done, remaining): files processed this run (across all workers)
        and how many are still pending — so the caller can report progress.
        """
        with self.lock:
            self.in_progress.discard(line)
            self.running.discard(line)
            if ok:
                self.done += 1
                self.driver.strike(line)
                self.attempts.pop(line, None)
            else:
                self.attempts[line] = self.attempts.get(line, 0) + 1
                if self.attempts[line] >= MAX_ATTEMPTS:
                    self.failed.add(line)
            remaining = self.driver.pending_total()
            return self.done, remaining

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
            if self.claims_closed.is_set() or not self.handback.latch(reason):
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


# --- worker loop ---------------------------------------------------------------

def apply_stop_request(job_id: int, shared: Shared, app) -> bool:
    """React to a pending stop from one worker. True => leave the run now.

    Three outcomes, and telling them apart is the whole point:

      * nothing pending — carry on claiming (and reopen claims if a request that
        closed them has since been withdrawn, so a mis-pressed `s` costs
        nothing but the files not claimed in between);
      * this run's own interactive request (`s`) — close new claims and HOLD
        while any job is still in flight, leaving the toggle usable for as long
        as the user can see a job row moving;
      * a stop file — latch it and end the run.

    The grace is interactive-only, exactly as in `stopchannel.confirm_stop_request`:
    a stop file (a script's `touch stop`, another run) has nobody sitting here to
    press `s` again, so it must stop the run as promptly as it always did.
    """
    out = job_lines(job_id)
    pending = stopchannel.pending_stop(app)
    if pending is None:
        if shared.reopen_claims():
            out.line("stop request withdrawn — claiming files again.", "cyan")
            app.update(phase="running")
        return False

    owner, first = shared.request_stop(job_id)
    if pending is stopchannel.StopSource.KEY:
        if first:
            app.update(phase="stopping")
            out.line("stop requested — no new files will be claimed; "
                     "press s again to cancel while a job is still running.",
                     "yellow")
        # Held, not ended, while the owner decides: the request may yet be
        # withdrawn. `stop.wait` rather than sleep so the latch releases this
        # worker at once instead of after the poll interval.
        if not owner or shared.busy():
            shared.stop.wait(stopchannel.STOP_RECHECK_SECONDS)
            return False
        # Nothing left in flight: the same countdown the sequential loop holds
        # at its iteration boundary, so both runners define "the user really
        # meant it" identically. False => the request went away, and the next
        # pass reopens the claims.
        if not stopchannel.confirm_stop_request(app):
            return False
    # The tail — re-reading the channel to act on, latching the sentinel for
    # cleanup, the reason, the line — is `stopchannel.commit_stop`, reached under
    # the lock that picks the one worker who runs it (which is also the lock the
    # line is written after, not under). `pending` goes along as the fallback for
    # a sentinel that vanished in between.
    if shared.latch_stop(pending, app,
                         lambda text: out.line(text, "bold red")):
        app.update(phase="stopping")
    return True


def note_driver_handback(job_id: int, shared: Shared, reason: Optional[str]) -> None:
    """Act on what a driver hook returned: nothing, or the end of this run.

    One place for both hooks (`Driver.item_started` / `item_finished`), because
    the two differ only in WHEN they are asked — what a returned reason means is
    the same either way, and the announcement has to read the same too.
    """
    if not reason:
        return
    if shared.request_driver_handback(reason):
        job_lines(job_id).line(
            f"⏸ {reason} — {stopchannel.DRIVER_HANDBACK_CLAUSE}", "yellow")


def worker(job_id: int, shared: Shared, source: Optional[object],
           policy, session_start_box: list, usage_lock: threading.Lock,
           app=None, progress=None, mailbox=None,
           retirement_requested=None, claim_work=None) -> None:
    """One worker thread: claim -> (usage gate) -> run -> record, repeat.

    Loops until the queue drains, the claim cap closes, the stop sentinel is
    latched (see apply_stop_request), the driver asks to hand control back from
    one of its per-item hooks (see note_driver_handback), or the pool retires this
    worker at a claim boundary. A claim that returns None while every signal
    remains clear means
    everything left is in flight elsewhere, so we briefly back off and retry.
    `source`/`policy` are None when --ignore-usage disables the session-limit
    gate. `app` is this run's status line; each worker owns the Job of its own
    number, and Job's mutators are lock-guarded for exactly that. `progress`
    carries the summary-row figures of the whole invocation.
    """
    # A disabled StatusApp is a Null object: no terminal, no threads, every call
    # below a no-op — so the worker has no `if app is not None` in it.
    app = app if app is not None else statusline.StatusApp(enabled=False)
    progress = (progress if progress is not None
                else statusline.InvocationProgress())
    job = app.job(job_id)
    out = job_lines(job_id)
    # Asked beside `shared.stop`, not left to it: the main thread sets the stop
    # only once its join has noticed the press (`_Interrupt.hear`, a quarter
    # second later), and in between a worker went on to start a fresh CLI.
    interrupt = ctrlc.current()
    while not shared.stop.is_set() and not interrupt.requested:
        if retirement_requested is not None and retirement_requested(job_id):
            break
        if apply_stop_request(job_id, shared, app):
            break
        if shared.stop_requested.is_set():
            continue  # holding the grace: do not fall into the claim back-off

        # The `p` key's hold, and the reason it sits BEFORE the claim: a paused
        # worker must hold no file. Claiming first and then pausing would park
        # a line in `in_progress` for the length of the hold, where a stop
        # latched meanwhile releases it — and the whole promise of the key is
        # that it costs the run nothing. Files already in flight are not held
        # back: as in the sequential loop, what pauses is the START of work.
        # `shared.stop.wait` rather than sleep, so a latched stop releases this
        # worker at once, and the loop head above is what acts on it.
        if stopchannel.pause_requested(app):
            if shared.exhausted():
                break       # nothing left to hold back — see Shared.exhausted
            if shared.note_pause(True):
                out.line(f"{statusline.PAUSE_GLYPH} paused — no new "
                         "files will be claimed; press p to resume.", "yellow")
            shared.stop.wait(stopchannel.STOP_RECHECK_SECONDS)
            continue
        if shared.note_pause(False):
            out.line("▶ pause released — claiming files again.", "cyan")

        # Claim work FIRST, before the session-limit gate. The gate can block for a
        # long time when the budget is spent, and its wait loop does not watch the
        # stop flag — so a worker that paused there before claiming would wedge and
        # never notice the queue draining to empty around it, hanging the run's
        # final join(). Claiming first means an empty/drained queue stops the worker
        # here (claim() sets the stop flag) and it never enters the gate with
        # nothing to do; only a worker actually holding a file ever pauses.
        if claim_work is None:
            active, line = True, shared.claim()
        else:
            active, line = claim_work(job_id, shared)
        if not active:
            break
        if line is None:
            if shared.stop.is_set() or shared.claims_closed.is_set():
                break
            time.sleep(2)  # busy: others hold the rest — back off and retry
            continue

        # EVERYTHING from here down to `shared.finish` runs while this worker
        # holds a claim, and every step of it can raise: the usage gate does
        # network I/O and prints, a status-line update can fail,
        # `command_for`/`splice` build the prompt, and `run_job` drives a child
        # process. An exception here ends one thread and latches no stop, so the
        # claim it walks away from keeps `_exhausted` (`not pending and not
        # in_progress`) false for ever — `claim()` answers None, and the workers
        # still alive spin in the two-second back-off until somebody kills the
        # run. Measured 2026-08-24: 2 workers, 4 files, one BrokenPipeError
        # raised inside run_job — the second worker finished its three files and
        # `run_parallel` had still not returned 25 s later. The guard therefore
        # spans the whole region; `turn_started` is what tells `Shared.abandon`
        # which of its two ways of giving the claim back applies.
        turn_started = False
        try:
            # Session-limit gate, now that we hold real work: one worker checks at a
            # time (cheap, the reading is TTL-cached), and a pause blocks every worker that
            # reaches it — so the whole fleet idles together when the budget is spent.
            if source is not None:
                job.update(waiting_for_usage=True)
                try:
                    with usage_lock:
                        if not shared.stop.is_set():
                            # Both stop channels and a driver handback must
                            # release a fleet held here for a whole window.
                            paused, new_start = policy.check_and_wait(
                                source, session_start_box[0],
                                should_stop=lambda: (shared.stop_asked(app)
                                                     or shared.handback.pending))
                            if paused:
                                session_start_box[0] = new_start
                            # Publish the reading the check already paid for.
                            statusline.push_quotas(app, source, policy)
                finally:
                    job.update(waiting_for_usage=False)

            # A stop may have been latched, or a channel may have opened, while we
            # waited for the lock or paused on the budget. HOLD the claimed file
            # here rather than handing it back and going round: an `s` request can
            # still be withdrawn, and a claim that --max-runs or a drained queue has
            # closed cannot be made a second time — releasing it there loses that
            # file for the whole run, which is exactly what "a mis-pressed `s` costs
            # nothing" promises it will not do. Parked like this the worker is not
            # busy (no turn is running), so the verdict is not waiting on it either.
            # A max-items boundary only closes new claims, so already-claimed work
            # deliberately continues past all of this. The worker parked here may
            # be the one that wins the latch, and whatever follows the latch can
            # still raise (see `Shared.latch_stop`) with this claim held.
            while (stopchannel.pending_stop(app) is not None
                   and not shared.stop.is_set()):
                # The one place that decides what a request means — the key keeps
                # its cancel grace, a stop file does not. Read its verdict off
                # `shared.stop` rather than its return value: the loop has to end
                # on a stop latched by ANY worker, not only on this call's answer.
                apply_stop_request(job_id, shared, app)
            if shared.stop.is_set() or interrupt.requested:
                shared.release(line)
                break

            # A driver pause latched while this worker waited — for the lock, on
            # the budget, or in the hold above. The claim goes BACK, unlike the
            # close just above it and unlike --max-runs: nothing was attempted
            # with it (no turn started, no target touched), and unlike those two
            # endings this claim CAN be made again — the caller's whole reason
            # for asking is that it means to start another runner call. Held
            # instead, it would be a file paid for out of --max-runs that nobody
            # ran, and the fleet would sit out the quota window before the caller
            # got control back. AFTER the stop hold, so a stop that is also
            # pending is still this worker's to latch: an ending a human asked
            # for outranks one a driver did.
            if shared.handback.pending:
                shared.release(line)
                break

            # From here the claim is a turn in flight, so a stop request waits for
            # it. Marked before the command is built rather than around run_job:
            # everything below is part of starting this file, and the gap would be a
            # window in which the fleet looks idle while it is not. Paired with
            # `job.start(...)` below, which says the same thing to the status line;
            # move one and the other has to move with it.
            shared.start_turn(line)
            turn_started = True
            command = shared.driver.command_for(line)
            # Notes typed while this worker had no turn in flight ride its next
            # prompt. Every worker has a separate mailbox, so another worker can
            # neither consume nor receive them.
            if mailbox is not None:
                spliced, notes = mailbox.splice(command.prompt)
                if notes:
                    command = command._replace(prompt=spliced)
                    for note in notes:
                        emit_note(out, note)
            # The same three calls the sequential loop makes on its single Job: the
            # Job clock times THIS file, the run clock (latched once) times the run.
            # This is the display's half of `shared.start_turn(line)` above — the row
            # a stop request's grace is about; `shared.finish` and `job.finish` close
            # the pair the same way.
            started_at = time.time()
            app.mark_run_started(started_at)
            job.start(item=command.label, model=command.model,
                      prompt=command.prompt, now=started_at)
            app.update(phase="running")
            out.line(f"▶ {command.label}", "bold cyan")
            # The start-of-item hook (Driver.item_started), placed here so it
            # sees the command as it will actually be sent — operator notes
            # spliced in, the row already announced.
            note_driver_handback(job_id, shared,
                                 shared.driver.item_started(command))
            console.record_timing(statlog.pool_iteration_started(
                shared.log_session, job_id, command.label))
            provider_started = time.monotonic()
            with statusline.describing(job):
                try:
                    rc, cost_usd, dur = run_job(job_id, command, mailbox)
                finally:
                    console.record_timing(statlog.pool_iteration_finished(
                        shared.log_session, job_id,
                        time.monotonic() - provider_started))
        except BaseException:
            # Hand the claim back (see `Shared.abandon` for which way and why),
            # then let the exception go on ending the thread it was always going
            # to end. Rescuing the worker is not this guard's job; letting the
            # other workers reach the end of the run is.
            shared.abandon(line, turn_started)
            if turn_started:
                # The display's half of the same pair: `shared.start_turn` and
                # `job.start` opened it together, so both have to be closed
                # together here too, or a dead worker leaves a row that says it
                # is still running for the rest of the run. Second, because it is
                # cosmetic and `abandon` is what keeps the run able to end — a
                # status line that threw here must not take the claim with it.
                job.finish()
            raise
        # Recording the outcome is deliberately OUTSIDE the guard, and it is the
        # one place that does it: were it inside, a `finish` that raised half-way
        # (its `strike`/`pending_total` are file I/O on a real driver) would be
        # followed by the guard's `abandon` calling `finish` a second time, and
        # the line would be both struck and counted as a failed attempt. It needs
        # no guard: `finish` discards `in_progress` first thing under the lock,
        # so whatever happens after that, the run can still read as drained.
        job.finish()
        if rc != 0 and interrupt.requested:
            # Cut short by Ctrl+C (`run_job`), not failed: no attempt counted
            # against the file, no "will retry", no `item_finished` verdict on
            # a turn nobody let finish. The line goes back as it was, for the
            # next run to take.
            shared.release(line)
            break
        ok = rc == 0
        done_total, remaining = shared.finish(line, ok)
        # The summary counter moves on COMPLETION, not on the claim: a claimed
        # file is in flight, and counting it as progress would report N jobs'
        # worth of work that nothing has finished yet.
        progress.note_remaining(remaining)
        app.update(**progress.summary_fields())

        bits = []
        if dur is not None:
            bits.append(f"{dur:.1f}s")
        if cost_usd is not None:
            bits.append(f"${cost_usd:.4f}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        if ok:
            out.line(f"✓ {command.label}{suffix}  "
                     f"[{done_total} done this run, {remaining} left]", "green")
        else:
            parked = line in shared.failed
            tail = " — parked after repeated failures" if parked else " — will retry"
            out.line(f"✗ {command.label} (exit {rc}){suffix}{tail}", "bold red")

        # The end-of-item hook (Driver.item_finished). Outside the guard above
        # for the same reason `shared.finish` is — the claim is already given
        # back, so a hook that raises ends its own thread without taking the
        # run with it.
        note_driver_handback(job_id, shared,
                             shared.driver.item_finished(command, rc))


@stopchannel.stop_file_lifecycle()
@ctrlc.captured()
def run_parallel(driver: ListFileDriver, args: argparse.Namespace,
                 app_name: str = "parallel", *, setup_logging: bool = True,
                 wait_on_start: bool = True, progress=None) -> RunResult:
    """Drain `driver`'s list with N concurrent provider workers.

    The parallel counterpart of cyclecore.run_loop: same session-limit, git-push
    and mirror-log machinery, but a thread pool over independent list items
    instead of one sequential Driver loop.

    `progress` is the whole invocation's InvocationProgress. It matters to a
    wrapper that calls this once per batch: `args` are then BATCH arguments, so
    args.max is a slice size and this call cannot know the run's real work — the
    wrapper passes what it knows. Left None, this call is the invocation.

    Ctrl+C is the run's `ctrlc.Interrupt` for the whole call, not an exception:
    the join hears it (`join_workers`), the fleet is stopped (`_Interrupt.hear`)
    and the run leaves through `RunBoundary.interrupt` once the region and the
    console have closed.
    """
    # `runlifecycle.begin_run` is the prologue both runners share, under this
    # runner's own app name so its log does not fight the sequential one's.
    ctx = runlifecycle.begin_run(driver, args, app_name, progress,
                                 setup_logging=setup_logging)
    provider = ctx.provider
    progress = ctx.progress
    run_settings = ctx.settings
    dry_run = ctx.dry_run
    interrupt = ctrlc.current()

    # Worker count precedence: explicit -j/--jobs on the CLI, then the driver's
    # `jobs` attribute (a subclass may pin it), then the engine default. The
    # invocation remembers later `+` presses so a batching wrapper's next batch
    # starts at the widened count instead of silently shrinking again.
    jobs = args.jobs
    if jobs is None:
        jobs = getattr(driver, "jobs", None)
    if jobs is None:
        jobs = clispec.DEFAULT_JOBS
    jobs = max(1, jobs)
    jobs = progress.worker_count(jobs)
    print(f"  · jobs: {jobs}")

    list_file_rel = driver.list_file
    pending_now = driver.pending_lines()
    if not pending_now:
        print(f"Nothing pending in {list_file_rel} — nothing to do.")
        return RunResult(RunStopReason.NO_WORK, remaining=0)

    # Dry-run: list the commands that would run (capped by --max-runs), touch
    # nothing — including the stop sentinel, which only a real run claims (the
    # workers below are what detect it, and they never start here). Reported so
    # the preview says why a real run would not start yet.
    if dry_run:
        if os.path.exists(stopchannel.stop_file_path()):
            print("  · stop file present — a real run would wait here until it "
                  "went away. Left in place (a dry run never consumes it).")
        would_run = (len(pending_now) if args.max is None
                     else min(args.max, len(pending_now)))
        # Every pending item carries its whole prompt inside the joined argv, so
        # listing a full list is unreadable (1961 products once made ~26 MB of
        # preview). An explicit --max-runs is the user naming how many items they
        # want to see, so it wins; an uncapped preview is trimmed to a screenful
        # and says so — a silent truncation would be a lie about what is pending.
        listed = would_run if args.max is not None else min(
            would_run, DRY_RUN_LIST_LIMIT)
        print(f"DRY-RUN: {would_run} of {len(pending_now)} "
              f"pending file(s) would be processed across {jobs} worker(s):")
        first_command = None
        for line in pending_now[:listed]:
            command = driver.command_for(line)
            first_command = first_command or command
            print("  " + " ".join(build_agent_argv(command, provider)))
            if providers.prompt_on_stdin(provider):
                # The argv is complete but not self-contained when the prompt
                # travels on stdin; without this the preview shows flags only.
                print("    STDIN: " + command.prompt)
        if listed < would_run:
            print(f"  … and {would_run - listed} more pending — listing capped "
                  f"at {DRY_RUN_LIST_LIMIT}; pass -m/--max-runs N to preview "
                  f"exactly N.")
        if first_command is not None:
            # The argv lines above are what would be executed, but for claude the
            # whole prompt sits inside one joined `-p …` token and is unreadable —
            # and reading the prompt is what a dry run is for. Once, for job 1,
            # through the one formatter every prompt view shares.
            print(statusline.format_prompt_block(
                job_id=1, label=first_command.label, prompt=first_command.prompt,
                width=textwidth.screen_width()))
        return RunResult(RunStopReason.DRY_RUN, remaining=len(pending_now))

    # Same as the sequential runner: a stop request pending from another run is
    # waited out here, on the main thread, before any worker starts — otherwise
    # the first worker would claim it and stop the run before it did anything.
    if wait_on_start:
        stopchannel.wait_for_stop_file_clear()

    if getattr(args, "start_in", None):
        # Local import keeps the shared startup UI out of the module import cycle.
        from .cyclecore import wait_before_start
        wait_before_start(args.start_in, interactive=ctx.status_enabled)

    # Usage gate: one account's RunUsage (see runlifecycle.open_usage), shared by
    # every worker. One is the whole set this run closes:
    # `ListFileDriver.command_for` stamps every command with `driver.provider`,
    # which `begin_run` settled, so a fleet never switches accounts the way the
    # sequential loop does. Named after the runner AND the account, so its
    # `at end (…)` line says whose figures it holds the way a sequential run's
    # does. --ignore-usage leaves it unopened, so there is no source to gate on
    # and no policy to gate with.
    #
    # The run's one epilogue boundary, from that usage to `close_run` (see
    # `runlifecycle.RunBoundary`): every ending from here on closes the run
    # down once — the two doors below, and whatever else unwinds the `with`:
    # the opening snapshot (the boundary holds the pair from before it is
    # taken) and the preparation before the region included.
    shared = None
    # The run's one hold on its git, shared by the pump's periodic checks and
    # the exit push (`close_run`): an exit push abandoned — by Ctrl+C or past
    # an exception's deadline — must cancel a periodic check still waiting
    # behind it too. Without it such a check waited for `gitpush`'s
    # process-wide git lock unbounded and, once a git call an earlier
    # abandoned run left running had let go of it, started `rev-list` and
    # `git push` after this run had left.
    push_abort = PushAbort()
    boundary = runlifecycle.RunBoundary(
        ctx,
        counts=lambda: ((shared.claimed, shared.done) if shared is not None
                        else (0, 0)),
        push_abort=push_abort)
    with boundary:
        usage = (None if args.ignore_usage
                 else boundary.open_usage(driver, provider,
                                          name=f"parallel {provider}"))
        source, policy = runlifecycle.usage_halves(usage)
        usage_lock = threading.Lock()
        session_start_box = [time.time()]  # shared, refreshed when a window resets

        # The live knobs reach the workers through the object, never a local:
        # --max-runs reaches the claim loop through `Shared.max_items`, and
        # --git-push the pusher and the exit push through `run_settings.git_push`.
        # Copying the push policy into a local is what once made that knob do
        # nothing in this mode.
        shared = Shared(driver, run_settings)
        console.record_timing(costlog.pool_started(shared.log_session))

        # Each worker owns a mailbox for both live delivery and notes queued between
        # its turns. This is a growable set even at one worker: MessageAction opens
        # that sole address directly, while a later `+` can add another address
        # without replacing (and losing the contents of) worker 1's mailbox.
        mailboxes = boundary.mailbox = operator.MailboxSet(range(1, jobs + 1))
        app = runlifecycle.open_status(ctx, driver, job_count=jobs,
                                       messages=mailboxes)
        app.register_action(statusline.WeeklyLimitAction(
            lambda: policy))

        # The run's git has ONE owner, and every git call of the run is made on it:
        # the periodic pushes as its `idle` work, and the exit push as its `final`,
        # which `runlifecycle.close_run` hands it. The workers never push. git is
        # not safe to call concurrently, and one thread cannot run two pushes at
        # once — so the exit push queues behind a push in flight (`git push` has a
        # 300 s subprocess timeout) by construction, not by a lock whose holder a
        # bounded join could have given up on. Pinned by
        # `test_git_push.test_every_git_call_of_a_parallel_run_is_made_by_the_pusher`.
        #
        # Its life is the run's, not the fleet's: it keeps its cadence while the
        # workers wind down after a latched stop, and there is no wait for it
        # between the workers' end and `close_run`.
        #
        # The first push is one interval in, not up front: the owner asks `idle`
        # right after the window's first post (see `pusher.start` below), and that
        # turn only sets the clock — so a run shorter than the interval pushes only
        # on the way out.
        #
        # And none once the operator has pressed Ctrl+C (the run's Interrupt,
        # read here from the pusher's thread): the run is leaving, the exit push
        # is still to come, and a periodic push begun in the wind-down is one
        # more `git push` for the exit push to queue behind while the operator
        # waits.
        last_push = 0.0
        pump_armed = False

        def push_turn() -> Optional[float]:
            nonlocal last_push, pump_armed
            if interrupt.requested:
                return None
            if pump_armed:
                # The policy is read HERE, at the push, off the live knobs — never
                # captured in this closure. A run launched `--git-push none` whose
                # operator later turns pushing on must start pushing.
                last_push = maybe_git_push(run_settings.git_push, last_push,
                                           projectroot.project_dir(),
                                           abort=push_abort)
            pump_armed = True
            return PUSH_PUMP_INTERVAL_S

        pusher = boundary.pusher = ownership.OwnerThread("pusher", idle=push_turn)

        def retirement_requested(j):
            return threads.retirement_requested(j)

        def claim_work(j, worker_shared):
            return threads.claim(j, worker_shared)

        def make_worker(j):
            return threading.Thread(
                target=worker, name=f"job{j}",
                args=(j, shared, source, policy, session_start_box, usage_lock, app,
                      progress, mailboxes.mailbox(j), retirement_requested,
                      claim_work),
                daemon=True)

        def prepare_worker(j):
            if j not in mailboxes.target_ids:
                try:
                    mailboxes.activate(j)
                except KeyError:
                    mailboxes.add(j)
            progress.set_worker_count(j)
            # Use the invocation's Job object, not LoopStatus.job's fallback, so a
            # later batch resumes the added row's iteration count.
            app.update(jobs=progress.jobs(j))

        def remove_worker(j, count):
            mailboxes.deactivate(j)
            progress.set_worker_count(count)

        def finish_removal(count):
            app.update(jobs=progress.jobs(count))

        threads = WorkerPool(
            [make_worker(j) for j in range(1, jobs + 1)],
            make_worker, prepare_worker, remove_worker, finish_removal)
        app.register_action(ResizeWorkerPoolAction(threads))
        # Heard after the join and read after the status region has been
        # released. The interrupt does NOT exit from inside the `with app:`: the
        # closing report and the exit push would then be written over a pinned status
        # area, and the run would leave without either.
        #
        # A fact about THIS runner, not a rule — the sequential loop's two `sys.exit`
        # endings do close down inside its region, and correctly: it prints inside
        # the area on every iteration and pushes there on every pass, so a few more
        # lines are what that area is already carrying. Here the workers' output is
        # the area, and this is the one moment it stops being written to.
        interrupted = _Interrupt(shared, threads, interrupt)
        # Every ending stops the fleet before its housekeeping begins, an
        # exception's included: workers left claiming went on starting agents
        # through the exit push. The pump needs nothing of its own: `close_run`
        # closes the pusher, and a closed owner starts no idle pass — so none
        # is left pushing every minute beside the next run's pusher, under a
        # batching wrapper.
        boundary.on_ending(interrupted.stop_workers)

        # The region lives exactly as long as the workers do (run_loop releases it
        # the same way, before its final push): a batching wrapper alternates runs of
        # this runner with sequential ones, and two status areas pinned at
        # once would fight over the same rows. The console's owner lives as long as
        # the region, and its `close` in the `finally` is what puts every line the
        # workers posted on screen BEFORE the closing report below (within
        # CONSOLE_CLOSE_TIMEOUT_S — see there for a console that never comes back).
        #
        # The route is opened before the owner starts and closed after it has been
        # closed, so no write is routed to an owner that has not got, or no longer
        # has, the lines before it (see `console.route_through`). A poster waits for
        # room as long as the close waits for the whole backlog: both are "this
        # console is stuck", from the same measurement.
        with route_through(_console, post_timeout=CONSOLE_CLOSE_TIMEOUT_S):
            _console.start()
            try:
                with app:
                    if source is not None:
                        # Inside `with`, not before it: push_quotas is
                        # silent until start() has marked the app
                        # enabled. The reading is already paid for by the
                        # start-of-run snapshot above, so this costs no
                        # round-trip. The refresher only runs for a run
                        # that talks to the usage endpoint at all — with
                        # --ignore-usage `source` is None and nothing
                        # polls.
                        statusline.push_quotas(app, source, policy)
                        app.add_service(statusline.QuotaRefresher(
                            app, source, policy, provider=provider))
                    # A press before the fleet starts (the preparation, the
                    # opening snapshot, the region's own start) starts none
                    # of it.
                    if not interrupt.requested:
                        threads.start_initial()
                    # Started for EVERY run, including one launched with
                    # `--git-push none`: the policy is a knob, so "there
                    # is nothing to push on" is a fact about this instant,
                    # not about the run. `maybe_git_push` is a no-op for
                    # NONE, so the cost of a pump nobody has switched on
                    # is one thread asleep between turns. The no-op
                    # `first` is what makes `idle` due at all (an owner
                    # never runs it before its window's first post).
                    pusher.start(first=lambda: None)
                    join_workers(threads)
                    # Still inside the region, so the announcement and the
                    # wait for the workers' turns show under the pinned rows.
                    if interrupt.requested:
                        interrupted.hear()
                    app.update(phase="idle")
            finally:
                _close_console()

        # A press in the region's teardown, after the join had returned, stops
        # the fleet too; `hear` does nothing a second time.
        if interrupt.requested:
            interrupted.hear()
        if interrupted.announce_later is not None:
            print(interrupted.announce_later)

        # This run's own closing report, before the shared epilogue: the run
        # talks about its work first, and the housekeeping that closes it down
        # follows. What the pusher prints is not ordered against it: a periodic
        # push already under way when the workers ended may print its
        # "git push: done." below this report. Accepted rather than waited for —
        # waiting means sitting out a `git push` before a report that does not
        # mention pushing, and the exit push, which the report does precede,
        # comes after that push either way.
        remaining = driver.pending_total()
        print(f"\nProcessed {shared.done} file(s) this run; "
              f"{remaining} still pending in {list_file_rel}.")
        if shared.failed:
            print(f"  ⚠ {len(shared.failed)} file(s) parked after "
                  f"{MAX_ATTEMPTS} failed attempts:")
            for line in sorted(shared.failed):
                print(f"      {os.path.basename(line.strip())}")

        if interrupt.requested:
            # An interrupt is not a `RunStopReason` — nobody returns from here,
            # so there is no `RunResult` to carry one — but it IS an ending, and
            # it gets the same epilogue as any other. It used to get none at
            # all: no reason recorded, no exit push, no closing snapshot, no
            # report of the notes nobody delivered, so an operator who pressed
            # Ctrl+C left the run's commits sitting local and its mailbox
            # unread.
            #
            # THE COST, NAMED because an operator feels it: this can take
            # minutes. `close_run` hands the exit push to the pusher, and an
            # interrupt that lands while the pusher is inside `git push` waits
            # for a subprocess with a 300 s timeout (`gitpush.git_push`), on top
            # of INTERRUPT_JOIN_TIMEOUT_S for the whole fleet. Waited out
            # rather than bounded, and that is the decision: the pusher is
            # PUSHING, so the alternative to waiting is not a faster exit with
            # the same result, it is racing a second `git` against the first
            # one. The operator who will not wait presses Ctrl+C again: that
            # abandons the exit push (`close_run` says how), keeps the rest of
            # the epilogue, and still leaves with 130.
            boundary.interrupt()

        # `stop_reason` unset means no worker ever reached a verdict about the run:
        # every one of the endings — the cap, the drained queue, a latched stop —
        # writes it before a worker can leave. So the threads did not run out of
        # work, they died holding it (see Shared.abandon), and NO_WORK would tell
        # the reader the opposite of what happened. A queue that ends with lines
        # parked in `failed` is NOT this case: `_exhausted` latched NO_WORK there.
        reason = shared.stop_reason
        if reason is None:
            reason = RunStopReason.WORKERS_DIED
            print(f"  ⚠ every worker thread ended before the queue drained; "
                  f"{remaining} file(s) left unclaimed.")

        # `runlifecycle.end_run` is the epilogue both runners share; the exit
        # push in it is made by this run's pusher, behind whatever push it has
        # in flight (see `pusher` above, and `close_run`).
        return boundary.end(
            RunResult(reason, shared.claimed, shared.done, remaining))
