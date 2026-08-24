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
    streams without garbling, so here each worker prints one compact, fully
    formed line per event, prefixed `[job k]`, under a single output lock. You
    trade the live view for throughput — the right call for mechanical bulk work.

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

from . import clispec
from . import compactline
from . import exitlog
from . import limits
from . import operator
from . import projectroot
from . import providers
from . import runlifecycle
from . import statusline
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
from .console import print_markup
# Only the per-turn call: the exit push (and the policy enum with it) belongs to
# the shared epilogue, `runlifecycle.end_run`, which is where both runners close
# a run down and therefore the one place that decides how the exit push is
# guarded.
from .gitpush import maybe_git_push
from .stopchannel import RunResult, RunStopReason
from .providers import (note_channel, provider_spec, reap_agent_process,
                        start_agent_process, usage_source_for)
from .drivers import ListFileDriver

# How many of a failed job's discarded non-JSON lines are kept as its failure
# explanation. The reason a dying CLI gives is in its last lines, and the bound
# is what keeps a chatty one from growing a long-running worker.
FAILURE_TAIL_LINES = 5

# How often the background pusher wakes to apply the git-push policy. Not the
# cadence of pushing — EACH_HOUR's hour is its own — but how finely that cadence
# is checked, which is why a minute is plenty. A named constant rather than a
# literal in `push_pump` so a test can shorten it: without that, the pump's body
# is unreachable in a run that lasts less than one interval, and the handover it
# makes (which repository to push) had no pin at all.
PUSH_PUMP_INTERVAL_S = 60

# How long the run waits for the background pusher to finish its current turn
# before going on to the exit push. Short on purpose — a run that has done its
# work should not sit here — which is exactly why it is not a guarantee that the
# pusher has stopped: `git push` gets a 300 s subprocess timeout, so a pusher
# caught mid-push is still running (and still holding `push_lock`) when this
# returns. That is what makes the lock around the exit push load-bearing rather
# than decorative, and the reason this is a named constant is the same as
# PUSH_PUMP_INTERVAL_S's: a test cannot otherwise reach the timed-out case.
PUSHER_JOIN_TIMEOUT_S = 5

# How long an interrupted run waits for each worker to notice `shared.stop` and
# come back before it closes the run down anyway. Bounded because the operator
# has already asked to leave: a worker sitting in a provider turn cannot be made
# to return, and its process is reaped by the worker's own `finally` either way.
# The wait is per THREAD, so a fleet of ten can cost ten times this in the worst
# case — which is the price of letting a worker that is nearly done finish.
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

# Serialises every line printed by any worker so the compact per-job lines never
# interleave mid-line (each print is atomic, the renderers are not thread-safe).
_emit_lock = threading.Lock()


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


# --- output helpers: every emit goes through the shared lock --------------------

def _emit_markup(plain: str, markup: str) -> None:
    """The sink under every worker's line: one whole line written at a time.

    Reads `print_markup` off this module per call rather than closing over it —
    which is what lets the pins replace it (see `compactline.LineWriter`).
    """
    with _emit_lock:
        print_markup(plain, markup)


def _job_tag(job_id: int) -> tuple:
    """(plain, markup) prefix identifying a worker, e.g. '[job 2]'."""
    return f"[job {job_id}]", f"[cyan]\\[job {job_id}][/]"


def job_lines(job_id: int) -> compactline.LineWriter:
    """The compact line shapes, tagged for one worker and emitted under the lock.

    The whole difference between this runner's output and the sequential one:
    every line carries `[job k] ` and every write is serialised. Both are given
    to `compactline.LineWriter` here, so the shapes themselves — a tool call, a
    head plus what the row leaves beside it, an outcome — exist once for both
    runners, and the tag counts against the width of each of them.
    """
    tag_plain, tag_markup = _job_tag(job_id)
    return compactline.LineWriter(_emit_markup, f"{tag_plain} ",
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
    """Wait for every worker to finish — the one place Ctrl+C lands.

    A function of its own for one reason, and it is a real one: this is where a
    run spends all of its time, so it is where `KeyboardInterrupt` is delivered,
    and the interrupt's own ending (record the reason, push, snapshot, report the
    undelivered notes, exit 130) cannot be pinned unless a test can stage the
    interrupt HERE. Staging it by patching `threading.Thread.join` instead would
    also hit the bounded re-join inside the handler and the pusher's, i.e. it
    would break the code under test on its way in.
    """
    for t in threads:
        t.join()


# --- one provider round-trip for one file --------------------------------------

def run_job(job_id: int, command: AgentCommand, mailbox=None) -> tuple:
    """Run one provider command, rendering a compact per-job trace.

    Unlike cyclecore's streaming renderer this prints only the key events — each
    tool call, any failed tool result, and the final cost line — one atomic line
    at a time, so several of these can run at once without their output
    colliding. Returns (returncode, cost_usd, duration_s).

    `mailbox` is passed only by a single-worker run (see run_parallel); it lends
    the console this turn's stdin, exactly as the sequential runner does.
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
    # The child's stderr is merged into its stdout (start_agent_process), so a
    # provider that dies with a plain-text message says so on these skipped
    # lines. Compact mode drops them, which is how a job once failed with
    # `exit 1` and no cause anywhere. Kept as a bounded tail — a chatty CLI must
    # not be able to grow a worker's memory — and printed only if the job fails.
    diagnostics = collections.deque(maxlen=FAILURE_TAIL_LINES)
    # Everything from here down to `proc.wait()` runs with a child process
    # alive, and every step of it can raise: the console write behind each
    # `out.*` line, `note_channel`'s close, a decoder error on the child's own
    # stream. `wait()` is the only exit that reaps, so an exception used to
    # walk away from a running provider — see `providers.reap_agent_process`.
    try:
        with note_channel(proc, provider, mailbox) as channel:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    # Printed indented under a head of its own if the job fails.
                    diagnostics.append(compactline.short(line, out.budget("  ")))
                    continue  # non-JSON CLI diagnostics — skip in compact mode
                if not isinstance(ev, dict):
                    continue  # valid JSON can still be a diagnostic, not an event
                et = wire.event_type(ev)
                if provider == "codex":
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
                        provider_failed = True
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
                    # A process emits a second `result` when a late note is answered
                    # as its own turn. The two figures then have to be combined
                    # differently, which is measured rather than assumed:
                    # `total_cost_usd` is the session's running total (so the last
                    # one is the job's cost), `duration_ms` is that turn's alone (so
                    # they add up).
                    cost_usd = wire.result_cost(ev, cost_usd)
                    dur = wire.result_duration_ms(ev)
                    if dur is not None:
                        duration_s = (duration_s or 0.0) + dur / 1000
        # Outside the `with`, so the pipe is closed before we wait on the process:
        # a worker waiting on a CLI whose stdin is still open never returns, and a
        # run whose final join() never finishes is the whole fleet.
        returncode = proc.wait()
    finally:
        reap_agent_process(proc)
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
        (the run was stopped while it sat in the session-limit gate): drop it from
        `in_progress` so the drained check stays accurate, and undo its --max
        reservation so the count reflects only files actually processed.
        """
        with self.lock:
            self.in_progress.discard(line)
            # Defence only: every release today happens before start_turn, so
            # there is no running turn to forget. Kept because the alternative —
            # a release that leaves a line in `running` — would make busy() true
            # forever and hang every later stop request.
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

    def latch_stop(self, source, app=None, announce=None) -> bool:
        """End the run on a stop request, for good. True for the worker that did it.

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
        (announce or print)(announcement)
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


def worker(job_id: int, shared: Shared, source: Optional[object],
           policy, session_start_box: list, usage_lock: threading.Lock,
           app=None, progress=None, mailbox=None) -> None:
    """One worker thread: claim -> (usage gate) -> run -> record, repeat.

    Loops until the queue drains, the claim cap closes, or the stop sentinel is
    latched (see apply_stop_request). A claim that returns None while every signal
    remains clear means everything left is in flight elsewhere, so we briefly back
    off and retry. `source`/`policy` are None when --ignore-usage disables the
    session-limit gate. `app` is this run's status line; each worker owns the Job
    of its own number, and Job's mutators are lock-guarded for exactly that.
    `progress` carries the summary-row figures of the whole invocation.
    """
    # A disabled StatusApp is a Null object: no terminal, no threads, every call
    # below a no-op — so the worker has no `if app is not None` in it.
    app = app if app is not None else statusline.StatusApp(enabled=False)
    progress = (progress if progress is not None
                else statusline.InvocationProgress())
    job = app.job(job_id)
    out = job_lines(job_id)
    while not shared.stop.is_set():
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
        line = shared.claim()
        if line is None:
            if shared.stop.is_set() or shared.claims_closed.is_set():
                break
            time.sleep(2)  # busy: others hold the rest — back off and retry
            continue

        # EVERYTHING from here down to `shared.finish` runs while this worker
        # holds a claim, and every step of it can raise: the usage gate does
        # network I/O and prints, the console can refuse any of these lines,
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
                with usage_lock:
                    if not shared.stop.is_set():
                        # The pause watches both stop channels. Without that, a
                        # fleet parked on the wall is a fleet with nobody left to
                        # notice `s`: every worker is inside this hold (or blocked
                        # on the lock in front of it), the loop head that reads the
                        # request is unreachable, and the keypress looks like a
                        # hung program until the window resets hours later.
                        paused, new_start = policy.check_and_wait(
                            source, session_start_box[0],
                            should_stop=lambda: shared.stop_asked(app))
                        if paused:
                            session_start_box[0] = new_start
                        # The check just paid for a usage reading; publishing it
                        # here is what keeps the provider's live figures on the
                        # pinned row without a second round-trip (cache serves it).
                        statusline.push_quotas(app, source, policy)

            # A stop may have been latched, or a channel may have opened, while we
            # waited for the lock or paused on the budget. HOLD the claimed file
            # here rather than handing it back and going round: an `s` request can
            # still be withdrawn, and a claim that --max-runs or a drained queue has
            # closed cannot be made a second time — releasing it there loses that
            # file for the whole run, which is exactly what "a mis-pressed `s` costs
            # nothing" promises it will not do. Parked like this the worker is not
            # busy (no turn is running), so the verdict is not waiting on it either.
            # A max-items boundary only closes new claims, so already-claimed work
            # deliberately continues past all of this. This is also the hold whose
            # exception is EXPECTED by design: the worker parked here may be the one
            # that wins the latch, and its console write can fail for reasons that
            # have nothing to do with the run (see `Shared.latch_stop`).
            while (stopchannel.pending_stop(app) is not None
                   and not shared.stop.is_set()):
                # The one place that decides what a request means — the key keeps
                # its cancel grace, a stop file does not. Read its verdict off
                # `shared.stop` rather than its return value: the loop has to end
                # on a stop latched by ANY worker, not only on this call's answer.
                apply_stop_request(job_id, shared, app)
            if shared.stop.is_set():
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
            # Notes typed while no turn was in flight ride this prompt (only a
            # single-worker run has a mailbox at all — see run_parallel).
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
            rc, cost_usd, dur = run_job(job_id, command, mailbox)
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


@stopchannel.stop_file_lifecycle()
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
    """
    # `runlifecycle.begin_run` is the prologue both runners share, under this
    # runner's own app name so its log does not fight the sequential one's.
    ctx = runlifecycle.begin_run(driver, args, app_name, progress,
                                 setup_logging=setup_logging)
    provider = ctx.provider
    progress, owns_progress = ctx.progress, ctx.owns_progress
    run_settings = ctx.settings
    dry_run = ctx.dry_run

    # Worker count precedence: explicit -j/--jobs on the CLI, then the driver's
    # `jobs` attribute (a subclass may pin it), then the engine default. Not a
    # knob: the pool is built once, so unlike --max-runs and --git-push this one
    # really is decided here, which is why it is printed in the header instead.
    jobs = args.jobs
    if jobs is None:
        jobs = getattr(driver, "jobs", None)
    if jobs is None:
        jobs = clispec.DEFAULT_JOBS
    jobs = max(1, jobs)
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

    # Usage gate: a shared UsageSource (query/cache) plus the Driver's LimitPolicy
    # (which quotas to gate on). --ignore-usage turns both off.
    source = None if args.ignore_usage else usage_source_for(provider)
    policy = None
    if source is not None:
        policy = driver.limit_policy or limits.default_policy(provider)
    usage_lock = threading.Lock()
    session_start_box = [time.time()]  # shared, refreshed when a window resets

    if source is not None:
        policy.log_snapshot(source, "at start (parallel)")

    shared = Shared(driver, run_settings)

    # What the queue holds right now: the baseline on the first call of the
    # invocation, and how far it has got on every later one. Through the
    # driver's own count, not `len(pending_now)`, so a driver that overrides
    # `pending_total` cannot end up counted one way here and another way by the
    # sequential runner — the same driver serves both.
    pending_total = driver.pending_total()
    progress.track_total(pending_total)
    progress.note_remaining(pending_total)

    # The pinned status area. A Job is the unit of display in BOTH runners, so
    # this is the sequential loop's wiring with N Jobs instead of one — no branch
    # anywhere in the status line separates them. Disabled it is a Null object,
    # so every call below stays a no-op and the run behaves exactly as before.
    #
    # The same registry the sequential runner builds, from the same function, so
    # both modes offer the same knobs under the same labels. --max-runs reaches
    # the claim loop through `Shared.max_items`, and --git-push reaches the
    # pusher and the exit push through `run_settings.git_push` — neither is
    # copied into a local here, which is what the parallel runner used to do to
    # the push policy and is why that knob did nothing in this mode.
    settings = runlifecycle.script_settings(
        run_settings, progress if owns_progress else None)
    # A note is addressed to "the agent", and only a one-worker run has exactly
    # one of those: with N workers the same keystroke would have to pick a
    # recipient, and the console shows their output interleaved. So a mailbox
    # exists here only at jobs == 1 — which is also what keeps the `m` key out of
    # the legend of a fleet run rather than offering something ambiguous.
    mailbox = operator.Mailbox() if jobs == 1 else None
    app = statusline.StatusApp(
        # Jobs come from the invocation's pool, so a second batch resumes the
        # rows of the first instead of starting a fresh set at iteration 1.
        status=statusline.LoopStatus(jobs=progress.jobs(jobs)),
        settings=settings,
        messages=mailbox,
        enabled=not getattr(args, "no_statusline", False))
    app.update(
        provider=provider,
        # Files COMPLETED out of the run's real work (see InvocationProgress) —
        # not files claimed, and not the size of this batch.
        **progress.summary_fields(),
        # Only a list driver has a pick order; read defensively so any other
        # driver simply reports no `rand` marker.
        random_order=str(getattr(driver, "pick_order", "")) == "random",
        script_limits=settings.status_entries(),
    )

    # A background pusher applies the policy on its own cadence while the
    # workers run; the workers never push. git is not thread-safe to call
    # concurrently, so every push in the run goes through one thread and one
    # lock — including the exit push below, which takes it because the join
    # before it CAN TIME OUT (see PUSHER_JOIN_TIMEOUT_S). The old wording here
    # said that push "runs after this thread is joined but takes the lock
    # anyway", i.e. that the lock was decorative; a `git push` may sit in a
    # subprocess for five minutes, the join waits five seconds, so it is the
    # only thing standing between two concurrent pushes.
    #
    # The first push is one interval in, not up front: the loop asks
    # `shared.stop.wait` BEFORE pushing, so a run shorter than the interval
    # pushes only on the way out.
    last_push_box = [0.0]
    push_lock = threading.Lock()

    def push_pump():
        while not shared.stop.wait(PUSH_PUMP_INTERVAL_S):
            with push_lock:
                # The policy is read HERE, inside the lock, off the live knobs —
                # never captured in this closure. A run launched `--git-push
                # none` whose operator later turns pushing on must start
                # pushing, and a run turned off mid-push must not have its
                # policy read half-applied beside a push already in flight.
                last_push_box[0] = maybe_git_push(run_settings.git_push,
                                                  last_push_box[0],
                                                  projectroot.project_dir())

    threads = [
        threading.Thread(target=worker, name=f"job{j}",
                         args=(j, shared, source, policy, session_start_box,
                               usage_lock, app, progress, mailbox),
                         daemon=True)
        for j in range(1, jobs + 1)
    ]
    # Started for EVERY run, including one launched with `--git-push none`: the
    # policy is a knob now, so "there is nothing to push on" is a fact about this
    # instant, not about the run. `maybe_git_push` is a no-op for NONE, so the
    # cost of a pump nobody has switched on is one thread asleep in `wait`.
    pusher = threading.Thread(target=push_pump, name="pusher", daemon=True)

    # Set by the Ctrl+C branch below and read after the status region has been
    # released. The interrupt does NOT exit from inside the `with app:`: the
    # closing report and the exit push would then be written over a pinned status
    # area, and the run would leave without either.
    #
    # A fact about THIS runner, not a rule — the sequential loop's two `sys.exit`
    # endings do close down inside its region, and correctly: it prints inside
    # the area on every iteration and pushes there on every pass, so a few more
    # lines are what that area is already carrying. Here the workers' output is
    # the area, and this is the one moment it stops being written to.
    interrupted = False

    # The region lives exactly as long as the workers do (run_loop releases it
    # the same way, before its final push): a periodic run alternates batches of
    # this runner with sequential grow-kit sweeps, and two status areas pinned at
    # once would fight over the same rows.
    with app:
        if source is not None:
            # Inside `with`, not before it: push_quotas is silent until start()
            # has marked the app enabled. The reading is already paid for by the
            # start-of-run snapshot above, so this costs no round-trip. The
            # refresher only runs for a run that talks to the usage endpoint at
            # all — with --ignore-usage `source` is None and nothing polls.
            statusline.push_quotas(app, source, policy)
            app.add_service(statusline.QuotaRefresher(
                app, source, policy, provider=provider))
        for t in threads:
            t.start()
        pusher.start()

        try:
            join_workers(threads)
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C) — signalling workers to stop…")
            interrupted = True
            shared.stop.set()
            for t in threads:
                t.join(timeout=INTERRUPT_JOIN_TIMEOUT_S)

        if not interrupted:
            shared.stop.set()  # release the pusher's wait()
            pusher.join(timeout=PUSHER_JOIN_TIMEOUT_S)
        app.update(phase="idle")

    # This run's own closing report, before the shared epilogue: the run talks
    # about its work first, and the housekeeping that closes it down follows.
    remaining = driver.pending_total()
    print(f"\nProcessed {shared.done} file(s) this run; "
          f"{remaining} still pending in {list_file_rel}.")
    if shared.failed:
        print(f"  ⚠ {len(shared.failed)} file(s) parked after "
              f"{MAX_ATTEMPTS} failed attempts:")
        for line in sorted(shared.failed):
            print(f"      {os.path.basename(line.strip())}")

    if interrupted:
        # An interrupt is not a `RunStopReason` — nobody returns from here, so
        # there is no `RunResult` to carry one — but it IS an ending, and it gets
        # the same epilogue as any other. It used to get none at all: no reason
        # recorded, no exit push, no closing snapshot, no report of the notes
        # nobody delivered, so an operator who pressed Ctrl+C left the run's
        # commits sitting local and its mailbox unread. The reason goes down
        # first, so the record does not depend on the push surviving a second
        # Ctrl+C.
        #
        # THE COST, NAMED because an operator feels it: this can take minutes.
        # `close_run` takes `push_lock`, and an interrupt that lands while
        # `push_pump` is inside `git push` waits for a subprocess with a 300 s
        # timeout (`gitpush.git_push`), on top of `jobs` × INTERRUPT_JOIN_TIMEOUT_S
        # for the workers. Waited out rather than bounded, and that is the
        # decision: the thread holding the lock is PUSHING, so the alternative to
        # waiting is not a faster exit with the same result, it is racing a
        # second `git` against the first one. `pusher.join` is skipped for the
        # same reason it would be pointless — `shared.stop` is already set, the
        # pusher is a daemon, and the lock is what actually excludes it.
        exitlog.set_reason("interrupted by the operator (Ctrl+C)",
                           iterations=shared.claimed, completed=shared.done)
        runlifecycle.close_run(
            ctx, usage_source=source, limit_policy=policy,
            snapshot_label="at end (interrupted)", mailbox=mailbox,
            push_lock=push_lock)
        sys.exit(130)

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

    # `runlifecycle.end_run` is the epilogue both runners share. `push_lock` is
    # this runner's, and it wraps the WHOLE exit push (`git_unpushed_count`
    # included, and the reading of the policy with it) because the join above
    # may have given up on a pusher that is still inside `git push` — see
    # PUSHER_JOIN_TIMEOUT_S, and `end_run` for the rest of why.
    return runlifecycle.end_run(
        ctx, reason, iterations=shared.claimed, completed=shared.done,
        remaining=remaining, usage_source=source, limit_policy=policy,
        snapshot_label="at end (parallel)", mailbox=mailbox,
        push_lock=push_lock)
