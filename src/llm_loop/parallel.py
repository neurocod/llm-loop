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
    and the rotating mirror log.
  * Dropped — the live token-by-token Markdown rendering. cyclecore's stream
    renderer keeps module-global state that cannot serve several concurrent
    streams without garbling, so here each worker prints one compact, fully
    formed line per event, prefixed `[job k]`, under a single output lock. You
    trade the live view for throughput — the right call for mechanical bulk work.

CLI mirrors the family (see `--help`); the additions are `-j/--jobs N`
(default 10) and `-C/--project-dir`. `--max-runs N` caps the *total* number of
files processed this run (across all workers), not iterations.
"""

import argparse
import collections
import json
import os
import sys
import threading
import time
from typing import Callable, Optional

from . import compactline
from . import console
from . import cyclecore
from . import exitlog
from . import limits
from . import operator
from . import projectroot
from . import providers
from . import statusline
from . import stopchannel
from . import textwidth
from .agentwork import (
    AgentCommand,
    build_agent_argv,
)
from .console import print_markup
from .gitpush import (
    GIT_PUSH_POLICY,
    GitPushPolicy,
    git_push,
    git_unpushed_count,
    maybe_git_push,
)
from .stopchannel import RunResult, RunStopReason
from .providers import (note_channel, provider_spec, reap_agent_process,
                        set_live_messages, start_agent_process,
                        usage_source_for)
from .drivers import ListFileDriver

# Default worker count. The work is cheap and fully independent, so a handful of
# concurrent jobs is the sweet spot before the shared session budget, not CPU,
# becomes the bottleneck.
DEFAULT_JOBS = 10

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

    A trimmed copy of cyclecore.parse_args (it can't be reused directly: it has
    no --jobs, and its --max-runs means *iterations*, which here is redefined as
    a total-files cap). Every long option keeps its single-letter alias.

    `extra_options` is the same wrapper hook cyclecore.parse_args documents — a
    mode switch is usually spelled the same way in both modes, so the two
    parsers have to offer the same seam or its --help would depend on which one
    the wrapper happened to reach.
    """
    p = argparse.ArgumentParser(
        prog=prog,
        description=description or "Parallel autonomous loop running N "
                                  "concurrent LLM workers over a list file.",
    )
    p.add_argument("-j", "--jobs", type=int, default=None, metavar="N",
                   help="number of concurrent workers (default: the driver's "
                        f"`jobs`, else {DEFAULT_JOBS})")
    p.add_argument("-m", "--max-runs", "--max", dest="max", type=int, default=None,
                   metavar="N",
                   help="stop after processing N files total, across all workers "
                        "(default: drain the whole list); --max is a deprecated alias")
    p.add_argument("--codex", action="store_const", const="codex",
                   dest="provider", default=None,
                   help="run Codex CLI instead of the Driver's default provider")
    p.add_argument("-d", "--dry-run", action="store_true",
                   help="only print the commands that would run, don't run the LLM CLI "
                        "and don't touch the list")
    p.add_argument("-g", "--git-push", dest="git_push",
                   choices=[pol.value for pol in GitPushPolicy],
                   default=GIT_PUSH_POLICY.value,
                   help="when to `git push`: none | after_new_commits | each_hour "
                        f"(default: {GIT_PUSH_POLICY.value})")
    p.add_argument("-C", "--project-dir", dest="project_dir", metavar="DIR",
                   default=None,
                   help="project root: cwd for git/provider CLI, base for the stop "
                        "file and the list's relative paths "
                        "(default: the current working directory)")
    p.add_argument("--ignore-usage", action="store_true",
                   help="don't pause on the Current-session usage limit "
                        "(by default the workers pause together when the session "
                        "budget is exhausted)")
    # Accepted here too: the flag is documented as a general one, and a periodic
    # run hands these args to the sequential loop (which honours it), so a parser
    # that rejected it would exit 2 on a documented spelling.
    p.add_argument("--no-statusline", dest="no_statusline", action="store_true",
                   help="do not pin the interactive status rows at the bottom of "
                        "the terminal (same as LLM_LOOP_STATUSLINE=0)")
    p.add_argument("--no-live-messages", dest="no_live_messages",
                   action="store_true",
                   help="do not keep the agent's stdin open for notes typed "
                        "during an iteration; they wait for the next prompt "
                        f"instead (same as {providers.LIVE_MESSAGES_ENV}=0). "
                        "Notes need a single worker either way")
    if extra_options is not None:
        extra_options(p)
    return p.parse_args(argv)


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
                et = ev.get("type")
                if provider == "codex":
                    item = ev.get("item") or {}
                    item_type = item.get("type")
                    if et == "item.completed" and item_type == "agent_message":
                        for text in str(item.get("text") or "").splitlines():
                            out.line(f"💬 {text}")
                    elif et == "item.started" and item_type == "command_execution":
                        out.fitted("💻 ", compactline.undouble_backslashes(
                            str(item.get("command", ""))))
                    elif et == "item.completed" and item_type == "command_execution":
                        out.fitted(f"📤 exit {item.get('exit_code', '')}: ",
                                   compactline.undouble_backslashes(
                                       str(item.get("command", ""))))
                    elif et == "item.completed" and item_type == "file_change":
                        changes = item.get("changes") or []
                        paths = [str(change.get("path")) for change in changes
                                 if isinstance(change, dict) and change.get("path")]
                        out.line(f"🛠️ {', '.join(paths) or 'file changes applied'}")
                    elif et == "turn.completed":
                        usage = ev.get("usage") or {}
                        if usage:
                            out.line("tokens: "
                                     f"input {usage.get('input_tokens', 0)}, "
                                     f"cached {usage.get('cached_input_tokens', 0)}, "
                                     f"output {usage.get('output_tokens', 0)}")
                    elif et in ("error", "turn.failed"):
                        provider_failed = True
                        error = ev.get("message") or ev.get("error") or ev
                        out.fitted("⚠ ", error, "bold red")
                elif et == "assistant":
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "tool_use":
                            out.tool_use(block.get("name", "?"),
                                         block.get("input", {}) or {})
                elif et == "user":
                    # Only surface *failed* tool results; successes would just be
                    # noise at high concurrency. An operator note replayed back to us
                    # is the exception: it is the receipt for something a human
                    # typed, and it belongs in the log next to the turn it landed in.
                    for block in ev.get("message", {}).get("content", []):
                        if block.get("type") == "text" and mailbox is not None:
                            note = mailbox.claim_echo(block.get("text", ""))
                            if note is not None:
                                emit_note(out, note)
                        if block.get("type") == "tool_result" and block.get("is_error"):
                            content = block.get("content", "")
                            if isinstance(content, list):
                                content = " ".join(
                                    c.get("text", "") for c in content
                                    if isinstance(c, dict)
                                )
                            out.fitted("  ✗ ", content, "red")
                elif et == "rate_limit_event":
                    # The run's own rate-limit verdict (see cyclecore.RateLimitEvent).
                    # Surfaced, not acted on: with N workers the pause belongs to the
                    # shared usage gate, which sees the same wall as a pegged
                    # percentage when the next worker checks in.
                    rl = cyclecore.rate_limit_event_from(ev)
                    if rl is not None and rl.status != "allowed":
                        out.line(f"⚠ rate limit: {rl.describe()}", "bold red")
                elif et == "result":
                    # Before the figures: once the turn has reported, the console
                    # must not be able to write into a session that is closing.
                    channel.close()
                    # A process emits a second `result` when a late note is answered
                    # as its own turn. The two figures then have to be combined
                    # differently, which is measured rather than assumed:
                    # `total_cost_usd` is the session's running total (so the last
                    # one is the job's cost), `duration_ms` is that turn's alone (so
                    # they add up).
                    cost_usd = ev.get("total_cost_usd", cost_usd)
                    dur = ev.get("duration_ms")
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
    access is under `lock`.
    """

    def __init__(self, driver: ListFileDriver, max_items: Optional[int]):
        self.driver = driver
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
        self.max_items = max_items
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
    # Worker count precedence: explicit -j/--jobs on the CLI, then the driver's
    # `jobs` attribute (a subclass may pin it), then the engine default.
    provider = getattr(args, "provider", None) or driver.provider
    spec = provider_spec(provider)
    driver.provider = provider
    jobs = args.jobs
    if jobs is None:
        jobs = getattr(driver, "jobs", None)
    if jobs is None:
        jobs = DEFAULT_JOBS
    jobs = max(1, jobs)
    git_push_policy = GitPushPolicy(args.git_push)
    # No wrapper above us: this call is the whole invocation, so its own --max is
    # the invocation cap and it owns the figures (see the setting below).
    owns_progress = progress is None
    if owns_progress:
        progress = statusline.InvocationProgress(max_items=args.max)

    # Anchor every project-relative operation before anything reads the root.
    projectroot.set_project_root(getattr(args, "project_dir", None))

    # Decided before the first argv is built: the transport is what this turns
    # off, and the argv and the process's stdin have to agree about it. Both
    # directions, so a previous phase in the same process cannot decide it.
    set_live_messages(not getattr(args, "no_live_messages", False))

    # Mirror all output to the rotating log, same as run_loop — under its own app
    # name so this runner's log doesn't fight the sequential one's. A dry run is
    # a preview, not a run, and stays out of the shared record entirely: see the
    # same branch in cyclecore.run_loop for the incident behind it.
    if setup_logging and not args.dry_run:
        logger = console._setup_file_logging(app_name)
        sys.stdout = console._TeeToLog(sys.stdout, logger)
        sys.stderr = console._TeeToLog(sys.stderr, logger)
    if not args.dry_run:
        # See the same call in cyclecore.run_loop: after the tee, so a vanished
        # run's report lands in the log whose abrupt end it explains.
        exitlog.begin(app_name, console.LOG_DIR,
                      os.path.basename(projectroot.project_dir()))
    print(f"  · project root: {projectroot.project_dir()}")
    if args.dry_run:
        print("  · dry run: nothing is mirrored to "
              f"{console.log_file_path(app_name)}")
    else:
        print(f"  · logging to {console.log_file_path(app_name)}")
    print(f"  · provider: {spec.display_name}")
    print(f"  · jobs: {jobs}  ·  git push policy: {git_push_policy.value}")

    list_file_rel = driver.list_file
    pending_now = driver.pending_lines()
    if not pending_now:
        print(f"Nothing pending in {list_file_rel} — nothing to do.")
        return RunResult(RunStopReason.NO_WORK, remaining=0)

    # Dry-run: list the commands that would run (capped by --max-runs), touch
    # nothing — including the stop sentinel, which only a real run claims (the
    # workers below are what detect it, and they never start here). Reported so
    # the preview says why a real run would not start yet.
    if args.dry_run:
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

    shared = Shared(driver, args.max)

    # What the queue holds right now: the baseline on the first call of the
    # invocation, and how far it has got on every later one. Through the
    # driver's own count, not `len(pending_now)`, so a driver that overrides
    # `pending_total` cannot end up counted one way here and another way by the
    # sequential runner — the same driver serves both.
    pending_total = driver.pending_total()
    progress.track_total(pending_total)
    progress.note_remaining(pending_total)

    def set_max_items(value):
        value = None if value is None else int(value)
        shared.max_items = value
        # A cap edited mid-run moves the summary row's denominator too — but only
        # when this call IS the invocation. Under a wrapper the displayed cap is
        # the wrapper's to set; this one only sizes the current batch.
        if owns_progress:
            progress.max_items = value

    # The pinned status area. A Job is the unit of display in BOTH runners, so
    # this is the sequential loop's wiring with N Jobs instead of one — no branch
    # anywhere in the status line separates them. Disabled it is a Null object,
    # so every call below stays a no-op and the run behaves exactly as before.
    settings = statusline.SettingsRegistry()
    settings.add(statusline.NumberSetting(
        "max-runs", "--max-runs",
        # Bound to Shared, not to a copy: here --max-runs caps FILES and the
        # claim loop is what enforces it, so an edited cap (wave 2) lands in the
        # one place that reads it. The run's other flags (-j, --git-push) are
        # consumed when the threads are created and cannot be re-read mid-run,
        # so they are printed in the header rather than offered as knobs.
        lambda: shared.max_items,
        set_max_items,
        minimum=1,
        # Not a field of its own — the summary counter's denominator is this cap
        # (or the list's size, whichever is smaller). See the sequential
        # registration in cyclecore._script_settings.
        show_in_status=False))
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
    # lock — including the exit push below, which runs after this thread is
    # joined but takes the lock anyway, so the rule holds by reading the code
    # rather than by remembering the join.
    #
    # The first push is one interval in, not up front: the loop asks
    # `shared.stop.wait` BEFORE pushing, so a run shorter than the interval
    # pushes only on the way out.
    last_push_box = [0.0]
    push_lock = threading.Lock()

    def push_pump():
        while not shared.stop.wait(PUSH_PUMP_INTERVAL_S):
            with push_lock:
                last_push_box[0] = maybe_git_push(git_push_policy,
                                                  last_push_box[0],
                                                  projectroot.project_dir())

    threads = [
        threading.Thread(target=worker, name=f"job{j}",
                         args=(j, shared, source, policy, session_start_box,
                               usage_lock, app, progress, mailbox),
                         daemon=True)
        for j in range(1, jobs + 1)
    ]
    pusher = None
    if git_push_policy != GitPushPolicy.NONE:
        pusher = threading.Thread(target=push_pump, name="pusher", daemon=True)

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
        if pusher is not None:
            pusher.start()

        try:
            for t in threads:
                t.join()
        except KeyboardInterrupt:
            print("\nInterrupted by user (Ctrl+C) — signalling workers to stop…")
            shared.stop.set()
            for t in threads:
                t.join(timeout=5)
            sys.exit(130)

        shared.stop.set()  # release the pusher's wait()
        if pusher is not None:
            pusher.join(timeout=5)
        app.update(phase="idle")

    # Final push on the way out (unless policy is NONE), mirroring run_loop.
    if git_push_policy != GitPushPolicy.NONE:
        count = git_unpushed_count(projectroot.project_dir())
        if count is None or count > 0:
            print("  · final git push on exit…")
            with push_lock:
                git_push(projectroot.project_dir())
        else:
            print("  · final git push: nothing to push.")

    if source is not None:
        policy.log_snapshot(source, "at end (parallel)", cache_value=False)

    remaining = driver.pending_total()
    print(f"\nProcessed {shared.done} file(s) this run; "
          f"{remaining} still pending in {list_file_rel}.")
    if shared.failed:
        print(f"  ⚠ {len(shared.failed)} file(s) parked after "
              f"{MAX_ATTEMPTS} failed attempts:")
        for line in sorted(shared.failed):
            print(f"      {os.path.basename(line.strip())}")
    cyclecore.report_undelivered_notes(mailbox)
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
    # See run_loop's matching call: the closing line belongs to the process, so
    # the reason is recorded here and printed by exitlog on the way out.
    exitlog.set_reason(stopchannel.STOP_REASON_TEXT.get(reason, reason.value),
                       iterations=shared.claimed, completed=shared.done)
    return RunResult(reason, shared.claimed, shared.done, remaining)
