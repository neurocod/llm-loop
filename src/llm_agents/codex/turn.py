"""One turn: run the app-server, fold events, retry and follow up, and log results.

This is where the wire (protocol.py) and the state (registry.py) meet, and the
only place that knows an event stream becomes a `TurnOutcome`. Permissions,
workspace preparation, briefings and escalation decisions belong to sandbox.py.

"One turn" is what the CALLER asked for, not what went over the wire: a turn the
service dropped is resent here (`_retry_service_drops`), so one `run_turn` may
be several attempts on one thread, folded into one outcome. What an attempt
does about sub-agents that outlive its main answer is `subagent_wait`'s.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from llm_loop.wire import codex_app_event_thread, codex_app_from_sub_agent
from .. import host_policy
from ..answer_contract import (
    EXPECT_FOLLOWUP_PROMPT, FINISH_TASK_CLAUSE, expect_follow_up, expect_verdict,
    follow_up_failed, join_answers, needs_expect_follow_up,
)

from . import gates as gates_module
from . import sandbox
from .subagent_wait import (
    SubAgentWait, await_main_and_sub_agents, settle_sub_agents,
)
from .subagents import SubAgentAccount, SubAgentTracker
from .beacon import announce
from .events import (
    END_META, FOLLOWUP_EXPECT, FOLLOWUP_GATES, FOLLOWUP_META, OPENER_DONE,
    OPENER_META, OPENER_THREAD_OPEN, REPORT_META, RETRY_META, TURN_META, NETWORK_WAIT_META, completed_item, item_work, text_of,
)
from .paths import host_root
from .protocol import (
    AppServer, M_INITIALIZE, M_INITIALIZED, M_THREAD_NAME_SET, M_THREAD_RESUME,
    M_THREAD_START, M_TURN_START, N_ERROR, N_RATE_LIMITS, N_THREAD_STARTED,
    N_TOKEN_USAGE, N_TURN_COMPLETED, N_TURN_STARTED, NOISY_NOTIFICATIONS,
    STARTUP_TIMEOUT_S, THREAD_OPEN_TIMEOUT_S,
)
from .registry import AgentRecord
from .store import AgentStore
from .service import (
    NetworkWait, NetworkWaitTimeout, api_reachable, error_kind,
    is_service_failure, retried_note,
)

# How long a single ATTEMPT may run before we give up on it. Generous because
# the whole point is delegating work that takes minutes; the caller can lower
# it. It is `run_turn`'s own default, not merely the CLI's: as an argparse
# default alone it governed `spawn`/`send` while a programmatic turn had no
# bound at all unless its caller happened to pass one.
#
# Per attempt, not per call: a turn the service dropped is resent with the same
# bound, because a resend that inherited only the remainder would be sent to
# redo the work with no time to do it. The worst case a caller budgeting
# wall-clock must allow for is therefore
# `(MAX_SERVICE_RETRIES + 1) * timeout + sum(SERVICE_RETRY_PAUSES_S)`, where
# an attempt whose sub-agents outlive the main answer lasts up to
# `timeout + SUB_AGENT_GRACE_S + timeout + 2 * SUB_AGENT_INTERRUPT_S`: the
# main turn, the wait, the one turn sent about the children, and an interrupt
# drain at the wait's end and at the attempt's
# (`subagent_wait.await_main_and_sub_agents`).
# Like the attempt's own openers (`TURN_STARTUP_GRACE_S`), the formula leaves
# out the extra turn's `turn/start` request (up to
# `protocol.STARTUP_TIMEOUT_S`) and the server's teardown
# (`protocol.REAP_GRACE_S`, per stop).
# Network waits add at most `service.MAX_NETWORK_WAIT_S` across all attempts.
# A main thread continuing by itself re-arms `timeout` too, each time — which
# Codex does not do by default (`subagents.SUB_AGENT_GRACE_S`).
DEFAULT_TURN_TIMEOUT_S = 1800.0

# How long a turn may be silent BEFORE its own timeout starts counting.
# `run_turn` writes its opening log line first of all, and the turn's deadline
# is armed only after `turn/start` is accepted; the four openers in between
# (initialize, thread/start-or-resume, thread/name/set, turn/start) are each
# bounded separately and their RESPONSES are returned to the caller rather
# than dispatched (`AppServer.request`), so none of them writes anything to the
# event log. Four `protocol.STARTUP_TIMEOUT_S` cover them all but the thread's,
# which may wait for required MCP servers up to
# `protocol.THREAD_OPEN_TIMEOUT_S`: that one is bracketed by
# `events.OPENER_META` lines, and only between them is the log entitled to the
# extra silence (`THREAD_OPEN_GRACE_S`) — granted to every phase, it delayed
# the verdict on a worker that died mid-turn by ten minutes. A
# live-but-slow-starting turn can therefore leave the log untouched for this
# long, which is why
# `cli._turn_in_progress` adds it to the turn's timeout before calling a silent
# log the work of a killed process: judged by the turn timeout alone, a
# `--timeout 60` turn read as "killed?" while its server was still starting
# (review 2026-09-13). Erring long only delays that verdict; erring short
# reports a working agent as dead.
TURN_STARTUP_GRACE_S = 4 * STARTUP_TIMEOUT_S

# What the thread-open line adds to `TURN_STARTUP_GRACE_S` while its request
# is pending — so a new thread is allowed 840 s of silence from that line, more
# than the 780 s the rest of its openers can take from the turn's start.
THREAD_OPEN_GRACE_S = THREAD_OPEN_TIMEOUT_S

# How long to wait before each resend of a turn the SERVICE dropped, one entry
# per attempt — so the list's length IS the retry limit (user decision
# 2026-09-13: "a limit of 2, with pauses before resending").
#
# What the numbers stand on. The outage of 2026-09-13 failed three workers'
# turns within one minute of each other, each about three minutes into its turn
# (179 s, 192 s, 258 s), and an operator revived all three by hand with a
# `send` sent 23 s, 42 s and 74 s after the failure — every one of them
# finished its task on that second attempt, and the second attempts still saw
# the server reconnecting mid-turn. So a minute is measured to be enough for
# this outage, and no longer wait is measured at all: how long a BAD outage
# lasts is unknown, and the second pause errs long for that reason.
#
# Erring long costs wall-clock on a turn that is already lost (four minutes at
# worst, against a turn budget of thirty); erring short costs a whole thread
# resend — 1.2-1.7M input tokens on a worker — spent on a service still down.
#
# One ceiling to respect when raising these: a pause is a stretch in which the
# log is silent BY DESIGN, and `cli._turn_in_progress` calls a log silent for
# longer than the turn's timeout plus `TURN_STARTUP_GRACE_S` (240 s) the work
# of a killed process. The longest pause here stays under that grace, so the
# verdict holds even for a turn started with a one-second timeout — and the
# assert below is that sentence made mechanical, because the two constants are
# tuned by different people for different reasons and a `(60, 300)` here would
# otherwise report a live worker as dead with nothing turning red.
SERVICE_RETRY_PAUSES_S = (60.0, 180.0)

assert max(SERVICE_RETRY_PAUSES_S) < TURN_STARTUP_GRACE_S

MAX_SERVICE_RETRIES = len(SERVICE_RETRY_PAUSES_S)

# What the resent turn says. Not the original prompt: that would have the agent
# start its task over with a tree it has already half-edited. This exact text
# is the one an operator typed three times on 2026-09-13, and all three workers
# picked their work back up from it — which is the whole reason it names
# `git diff --stat` and the answer format rather than merely apologising.
SERVICE_RETRY_PROMPT = (
    "Your previous turn was cut off by a transient service error (\"stream "
    "disconnected: unable to verify model access\"), not by anything you did. "
    "The edits you already made are still in the working tree. Re-read `git "
    "diff --stat` to see where you stopped, then continue the same task from "
    "there to completion, including the gates and the final answer in the "
    "format the task asked for.\n"
)

# A local outage is not a model-access failure. Keep the continuation request
# explicit so the agent preserves its existing work and checks interrupted runs.
NETWORK_RETRY_PROMPT = (
    "Your previous turn timed out during a network outage while the CLI was "
    "waiting to reconnect. The edits you already made are still in the working "
    "tree. Re-read `git diff --stat` to see where you stopped. Check the logs "
    "of any interrupted commands before trusting their results, then continue "
    "the same task to completion, including its gates and requested report.\n"
)


# Measured 2026-09-13: four workers' final messages were not their reports.
# indoorlight and sections sent "read-only audit" notes despite edits,
# entrycount sent three lines after a follow-up, shadowlookup a self-instruction.
# Ask for a complete replacement file; the shared clause also requires the
# unfinished work to be completed before writing the report.
REPORT_FOLLOWUP_PROMPT = (
    "Your report is missing or incomplete: the selected answer does not "
    "match /{pattern}/. "
    f"{FINISH_TASK_CLAUSE}write the COMPLETE report as UTF-8 to "
    "{report}. Include every required section, including /{pattern}/. The "
    "wrapper reads that file as your answer.\n"
)


@dataclass
class TurnOutcome:
    thread_id: str = ""
    text: str = ""
    # This attempt's own final message, never inherited or replaced by a file.
    message_text: str = ""
    report_source: str = ""
    report_note: str = ""
    # None when no usable file was read; ignore surrounding whitespace.
    report_differs: Optional[bool] = None
    report_read_error: str = ""
    status: str = ""
    error: str = ""
    # How many follow-up turns `--expect` sent, and what it was looking for.
    follow_ups: int = 0
    expect: str = ""
    # Whether a completed answer satisfies `expect` once its follow-ups are
    # spent; None when nothing was judged (no expect, a failed turn or a
    # failed follow-up). False keeps the status — the answer is still owed to
    # the caller — but `cli.print_outcome` names the gap.
    expect_met: Optional[bool] = None
    # The server's own name for what went wrong (`codexErrorInfo`, folded to
    # one token by `service.error_kind`) — the field the retry decision reads,
    # because the human-facing `error` message is prose that changes.
    error_kind: str = ""
    # How many times this turn was resent after the service dropped it.
    service_retries: int = 0
    network_wait_s: float = 0.0
    network_wait_timeout: bool = False
    duration_s: float = 0.0
    commands: int = 0
    file_changes: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    context_tokens: int = 0   # what the thread occupies now (`last.totalTokens`)
    context_window: int = 0   # the model's window, when the server reports it
    log_path: str = ""
    items: list[str] = field(default_factory=list)
    # What the wrapper saw running the recorded gates after the turn
    # (`_check_gates`), the last run only; empty when the agent has none.
    gates: list[gates_module.GateRun] = field(default_factory=list)
    gate_follow_ups: int = 0
    # Why the recorded gates were NOT run, when they were not.
    gate_note: str = ""
    # Sub-agents still running when a main turn ended: this attempt's, and
    # once `run_turn` returns, every attempt's (`SubAgentAccount.fold`).
    sub_agents: SubAgentAccount = field(default_factory=SubAgentAccount)
    # Edits the wrapper refused to retry outside the sandbox, as logged
    # (`protocol.AppServer.file_decisions`), every attempt's: the agent heard
    # them as "rejected", so the caller is told whose refusal it was.
    declined_edits: list[dict] = field(default_factory=list)


class TurnRunner:
    """Collects one turn's events into a TurnOutcome.

    Only the MAIN thread's events decide the outcome: its `turn/completed` ends
    the turn, its `agentMessage` is the answer, its errors are the turn's. A
    sub-agent Codex spawns inside the turn (`wire.codex_app_from_sub_agent`) is
    activity — its commands and edits count, its tokens are billed — but its
    completion, answer and errors are the main thread's to act on, not ours
    (read as ours, a child's `turn/completed` ended the caller's turn with the
    CHILD's answer: `.codex-agents/review-a`, line 255).
    Its lifetime is the exception (`subs`): a child still running when the
    main thread completes keeps the server open
    (`subagent_wait.await_main_and_sub_agents`).
    """

    def __init__(self, earlier: Optional["TurnRunner"] = None, *,
                 week_used: Optional[Callable[..., None]] = None) -> None:
        """`earlier`: the attempt this one follows within the same caller
        turn (a service-drop resend or a follow-up), whose token readings
        this one inherits — see `_totals`.

        `week_used(percent, why)`: where the account's weekly utilisation
        goes (`agent_counter.Beacon.report_week_used`). Inherited from
        `earlier` too, so a resent attempt keeps feeding the badge."""
        self.outcome = TurnOutcome()
        self.network = NetworkWait(earlier.outcome.network_wait_s if earlier else 0.0)
        self.network.child_s = earlier.network.child_s if earlier else 0.0
        self.network_log: Callable[[dict], None] = lambda _message: None
        self.done = False
        self._week_used = week_used or (earlier._week_used if earlier else None)
        # Each thread's own cumulative `tokenUsage.total` (in, cached, out):
        # the main thread and every sub-agent report theirs separately
        # (measured on `.codex-agents/review-a`: the child's first reading
        # was 36k against the main thread's 115k), so the bill is their sum.
        # That holds only while a forked child's total starts at zero, i.e.
        # while threads stay paginated — the condition and its Codex source
        # are at `llm_loop.wire.codex_app_usage_sum`; `sandbox.thread_params`
        # must not add `historyMode` or `ephemeral`.
        #
        # Carried from the attempt before: the main thread's total runs on
        # across attempts by itself (`_carry_over`) and its next reading
        # replaces the entry, but a child's thread is over with the attempt
        # that spawned it and never reports again — started empty, the
        # follow-up of review-a dropped the child's 433k from the bill.
        self._totals: dict[str, tuple[int, int, int]] = (
            dict(earlier._totals) if earlier is not None else {})
        self._sum_totals()
        # Sub-agents of THIS attempt: its server's threads die with it, so
        # nothing carries over.
        self.subs = SubAgentTracker()

    # --- `subagent_wait.Attempt`

    @property
    def sub_agents(self) -> SubAgentAccount:
        return self.outcome.sub_agents

    def main_verdict(self) -> str:
        if _completed(self.outcome):
            return ""
        return self.outcome.error or self.outcome.status or "not completed"

    def answer(self) -> tuple:
        """What a completed main turn answered, to put back if a later main
        turn of the same attempt (a continuation, the turn about the
        children) fails."""
        return (self.outcome.text, self.outcome.message_text,
                self.outcome.status, self.outcome.error,
                self.outcome.error_kind)

    def restore_answer(self, kept: tuple) -> None:
        (self.outcome.text, self.outcome.message_text, self.outcome.status,
         self.outcome.error, self.outcome.error_kind) = kept

    def message(self) -> str:
        return self.outcome.message_text

    def on_sub_agent_event(self, message: dict) -> None:
        """`on_event` for sub-agents' events only: the interrupt drain after
        the attempt's verdict is final, where a late main `turn/completed`
        (a failure, even a service drop) must not overwrite a timeout or
        send the attempt to `_retry_service_drops`."""
        if codex_app_from_sub_agent(message, self.outcome.thread_id):
            self.on_event(message)

    def on_event(self, message: dict) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        if self.network.observe(message, time.monotonic(),
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                self.outcome.thread_id):
            self.log_network_wait()
        if method == N_RATE_LIMITS:
            # Account-wide, so whichever thread it came from is irrelevant.
            self._on_rate_limits(params)
            return
        item = completed_item(message)
        if codex_app_from_sub_agent(message, self.outcome.thread_id):
            self._on_sub_agent(message, item)
        elif method == N_THREAD_STARTED:
            # The first one is ours: a sub-agent's thread can only start
            # inside our turn, after `thread/start` has named the main one.
            self.outcome.thread_id = (self.outcome.thread_id
                                      or codex_app_event_thread(message))
        elif method == N_TURN_STARTED:
            # Ours, or the main thread continuing on its own after its answer
            # (`subagent_wait.await_main_and_sub_agents`): either way the
            # turn is not over.
            self.subs.main_started()
            self.done = False
        elif item is not None:
            self._on_item(item)
        elif method == N_TOKEN_USAGE:
            self._on_usage(params, codex_app_event_thread(message))
        elif method == N_ERROR:
            # `willRetry` means the server is already handling it (a dropped
            # response stream, a 429). Recording it as the turn's error would
            # report a failure for a turn that then completes normally.
            if params.get("willRetry"):
                return
            error = params.get("error") or params
            detail = error.get("message") if isinstance(error, dict) else error
            self.outcome.error = self.outcome.error or str(detail)
            self.outcome.error_kind = self.outcome.error_kind or error_kind(error)
        elif method == N_TURN_COMPLETED:
            turn = params.get("turn") or {}
            self.outcome.status = str(turn.get("status") or "completed")
            if self.outcome.status != "completed":
                error = turn.get("error") or {}
                detail = error.get("message") if isinstance(error, dict) else error
                # `or self.outcome.error` because a failed verdict may carry no
                # error object at all, and the loose `error` notification that
                # preceded it is then the only thing that named the reason —
                # overwriting it with the word "failed" threw away both the
                # message and (with it) the retry.
                self.outcome.error = str(
                    detail or self.outcome.error or self.outcome.status)
                # The verdict's own kind wins over the one a loose `error`
                # notification left: they agree on a service drop (measured
                # 2026-09-13), and when they do not, the turn's is the reason
                # the turn ended.
                self.outcome.error_kind = error_kind(error) or self.outcome.error_kind
            self.done = True
            self.subs.main_completed()

    def log_network_wait(self) -> None:
        self.outcome.network_wait_s = self.network.seconds(time.monotonic())
        self.network_log({"_meta": NETWORK_WAIT_META,
                          "active": self.network.started is not None,
                          "since": self.network.since, "tries": self.network.tries,
                          "waited_s": self.outcome.network_wait_s})

    def _on_rate_limits(self, params: dict) -> None:
        """The weekly figure for the on-screen counter, from the turn's OWN
        stream: no second `codex app-server` per agent (what
        `CodexUsageSource` would start) and nothing on the start path.

        `parse_rate_limits` decides which window is the week — by duration,
        not by the name `primary`/`secondary` — and is imported here, on the
        first report, because importing it with this module cost every Codex
        start 129 ms (median 309 vs 180 ms over 7 runs, 2026-11-06: it pulls
        `llm_loop.usage` and `urllib.request` with it).

        The figure is optional telemetry (`agent_counter`: nothing there may
        break a turn), so a failed import or a report the parser cannot read
        (`resetsAt: 1e20` throws in `datetime.fromtimestamp`) becomes "no
        figure" with the reason, and the turn goes on."""
        if self._week_used is None:
            return
        try:
            from llm_loop.codex_usage import parse_rate_limits
            week = parse_rate_limits(params).week_all.percent
        except Exception as error:  # noqa: BLE001 — see the docstring
            self._week_used(None, f"{N_RATE_LIMITS} could not be read: "
                                  f"{type(error).__name__}: {error}")
            return
        self._week_used(week, "" if week is not None else
                        f"{N_RATE_LIMITS} carried no window of a day or longer")

    def _on_item(self, item: dict) -> None:
        kind = str(item.get("type") or "")
        self.outcome.items.append(kind)
        self.subs.on_record(item)
        if kind == "agentMessage":
            text = text_of(item)
            if text:
                # Later messages replace earlier ones: the final assistant
                # message is the turn's answer, and a turn may emit several.
                self.outcome.text = text
                self.outcome.message_text = text
        self._count_work(item)

    def _count_work(self, item: dict) -> None:
        # What counts as a command or an edit is `events.item_work`'s to say,
        # not this loop's: `status` re-derives the same numbers from the log
        # and the two are printed to the same caller about the same turn.
        commands, file_changes = item_work(item)
        self.outcome.commands += commands
        self.outcome.file_changes += file_changes

    def _on_sub_agent(self, message: dict, item: Optional[dict]) -> None:
        """A sub-agent's event: its work and its tokens count, nothing else.

        Its errors are dropped too: a child that fails reports to the main
        thread, which decides whether the turn fails — recording the child's
        error here would fail a turn whose main thread then completes it.

        What it says about its own lifetime is kept, though — see `subs`: the
        main thread may answer before it is done. A grandchild's spawn is
        recorded on its parent's thread, so records count from any thread."""
        method = message.get("method")
        thread = codex_app_event_thread(message)
        if item is not None:
            self._count_work(item)
            self.subs.on_record(item)
            params = message.get("params") or {}
            self.subs.on_item(thread, str(params.get("turnId") or ""), item)
        elif method == N_TOKEN_USAGE:
            self._on_usage(message.get("params") or {}, thread,
                           occupancy=False)
        elif method == N_TURN_STARTED and thread:
            turn = (message.get("params") or {}).get("turn") or {}
            self.subs.on_turn_started(thread, str(turn.get("id") or ""))
        elif method == N_TURN_COMPLETED and thread:
            self.subs.on_turn_completed(
                thread, (message.get("params") or {}).get("turn") or {})

    def _on_usage(self, params: dict, thread: str = "", *,
                  occupancy: bool = True) -> None:
        """Two different numbers arrive here, and confusing them misleads badly.

        `total` accumulates across every request the turn made, so it counts the
        same thread text once per request — it answers "what did this cost".
        `last` is the most recent request alone, which is what the thread
        OCCUPIES in the model's window right now. Measured on one review turn:
        total 1.10M against last 88.8k, in a 258.4k window. Reporting the
        million as if it were occupancy would suggest the agent had long overrun
        a window it was in fact a third into.

        A sub-agent's reading (`occupancy=False`) adds to the bill and never
        to the occupancy: its window is not the one `send` will resume, and
        taking its `last` reported the child's context as the agent's.
        """
        usage = params.get("tokenUsage") or {}
        total = usage.get("total") or {}
        if isinstance(total, dict) and total:
            self._totals[thread] = (int(total.get("inputTokens") or 0),
                                    int(total.get("cachedInputTokens") or 0),
                                    int(total.get("outputTokens") or 0))
            self._sum_totals()
        if not occupancy:
            return
        last = usage.get("last") or {}
        if isinstance(last, dict) and last:
            self.outcome.context_tokens = int(last.get("totalTokens") or 0)
        window = usage.get("modelContextWindow")
        if isinstance(window, int) and window > 0:
            self.outcome.context_window = window

    def _sum_totals(self) -> None:
        readings = self._totals.values()
        self.outcome.input_tokens = sum(r[0] for r in readings)
        self.outcome.cached_input_tokens = sum(r[1] for r in readings)
        self.outcome.output_tokens = sum(r[2] for r in readings)


def run_turn(*, record: AgentRecord, prompt: str, resume: bool,
             timeout: float = DEFAULT_TURN_TIMEOUT_S, effort: str = "",
             expect: str = "",
             on_thread: Optional[Callable[[str], None]] = None,
             sleep: Callable[[float], None] = time.sleep,
             reachable: Callable[[], bool] = api_reachable,
             run_gate: Optional[gates_module.GateRunner] = None
             ) -> TurnOutcome:
    """Start (or resume) a thread, run one turn, return what it produced.

    `on_thread` fires the moment the thread id is known, minutes before the turn
    ends, so the caller can persist it. Waiting for the return value meant a
    Ctrl+C or a reboot mid-turn left a live Codex thread that no name pointed
    at — recoverable only through Codex's own history.

    A turn the SERVICE drops is resent here rather than reported — see
    `_retry_service_drops`. `sleep` and `reachable` are parameters so the gate
    can exercise that without waiting minutes or touching the network; nothing
    in production passes them.

    `expect` is a regex the ANSWER must match (multiline); an answer that does
    not is followed up once on the same thread — `_follow_up_on_missing`.

    `record.gates` are then run by the wrapper itself — `_check_gates`;
    `run_gate` is a parameter for the same reason as `sleep`.
    """
    store = AgentStore(record.name)
    started = time.monotonic()

    # The on-screen counter is told here rather than in `cli`, so a turn started
    # programmatically is as visible as one started from the command line — the
    # invisibility this fixes belongs to the TURN, not to the way it was asked
    # for. The label names the checkout too: one badge serves every clone.
    with announce(f"{record.name}@{host_root().name}") as counter, \
            store.event_log() as log:
        log({"_meta": TURN_META, "name": record.name, "resume": resume,
             "prompt": prompt, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
             "timeout": timeout})
        # The attempts of THIS turn, oldest first: a service drop is resent
        # (`_retry_service_drops` appends), and the last entry is always the
        # one that was running — including when an exception escapes.
        attempts = [TurnRunner(week_used=counter.report_week_used)]
        report = sandbox.report_path(record)
        try:
            # Clear only at the caller-turn boundary, before any attempt: a
            # send must not inherit yesterday's report, but a service retry
            # must retain the file its interrupted attempt already wrote.
            if report is not None:
                _clear_report(report, log)
            _run_turn_logged(record, prompt,
                             record.thread_id if resume else "",
                             timeout, effort, on_thread, log, attempts[0])
            _retry_service_drops(
                record=record, attempts=attempts, timeout=timeout,
                effort=effort, on_thread=on_thread, log=log, sleep=sleep,
                reachable=reachable)
            if report is not None:
                _read_report(attempts[-1].outcome, report)
            _follow_up_on_missing(
                record=record, attempts=attempts, expect=expect,
                timeout=timeout, effort=effort, on_thread=on_thread, log=log,
                report=report)
            _check_gates(
                record=record, attempts=attempts, timeout=timeout,
                effort=effort, on_thread=on_thread, log=log, report=report,
                # Looked up per call, not bound as the default: a default is
                # evaluated once at import, and `cli` passes none.
                run_gate=run_gate or gates_module.run_gate)
        finally:
            if attempts[-1].outcome.report_source:
                log({"_meta": REPORT_META, "name": record.name,
                     "path": str(report) if report is not None else None,
                     "source": attempts[-1].outcome.report_source,
                     "status": attempts[-1].outcome.status,
                     "differs": attempts[-1].outcome.report_differs,
                     **({"read_error": attempts[-1].outcome.report_read_error}
                        if attempts[-1].outcome.report_read_error else {})})
            # The closing line is what tells a turn in progress from one that
            # is over (`events.last_turn_trace`): without it `status` showed a
            # running worker as "0 turns · last -" for its whole first turn,
            # and a killed one as nothing at all (measured 2026-09-13). Written
            # on EVERY exit, including the operator-typo SystemExit that
            # `sandbox.prepare_workspace` raises for a missing workspace, so
            # an aborted turn does not read as one still running.
            log({"_meta": END_META, "name": record.name,
                 "status": attempts[-1].outcome.status or "aborted",
                 "at": time.strftime("%Y-%m-%d %H:%M:%S")})

    outcome = attempts[-1].outcome
    outcome.duration_s = time.monotonic() - started
    outcome.log_path = str(store.events_path)
    outcome.sub_agents = SubAgentAccount.fold(
        [runner.outcome.sub_agents for runner in attempts])
    outcome.text = outcome.sub_agents.under_answer(outcome.text, record.name)
    if outcome.text:
        # Only a turn that SAID something replaces the stored answer. Writing
        # `text or ""` erased a good report whenever the next message hit a
        # usage limit or a timeout, and `result` then printed nothing at all —
        # indistinguishable from an agent that never answered.
        store.write_answer(outcome.text)
    return outcome


def _service_drop(outcome: TurnOutcome) -> str:
    """The service's own name for why it dropped this turn, or "" when the turn
    was not the service's to lose.

    Two things must hold, and the second is what carries the weight. The status
    must be `failed`: an ordinary `timeout` is our own bound expiring
    (`DEFAULT_TURN_TIMEOUT_S`), and resending a turn we cut off ourselves would
    buy a second full-length one. And the SERVER must have named the reason —
    which a purely local failure (the app-server process dying, an OSError)
    never does, so it is refused for want of a kind rather than by a rule of its
    own. A local death that FOLLOWS a service error keeps that error's kind and
    is retried, deliberately: the stream had already broken for the reason this
    exists for. The exception is a timeout while the CLI explicitly waits for
    network: `network_wait_timeout` permits the same reachability check and
    resend. Exhausting the caller-wide network allowance is terminal.
    """
    if outcome.status == "timeout" and outcome.network_wait_timeout:
        return "responseStreamDisconnected"
    if outcome.status != "failed":
        return ""
    if not is_service_failure(outcome.error_kind, outcome.error):
        return ""
    return outcome.error_kind


def _carry_over(dropped: TurnOutcome, resent: TurnOutcome) -> None:
    """Fold a dropped attempt into the attempt that replaced it.

    The caller asked for ONE turn and is owed one turn's numbers. The commands
    and the edits are OURS to count — one per item seen — so they are summed:
    the agent really ran them before the stream died.

    The token totals are not ours and are NOT summed here. `tokenUsage.total`
    accumulates over the THREAD, not over the turn, and a resumed thread
    carries on from where the dropped attempt left off: measured on
    worker-e's `events.jsonl`, the last usage line before the
    drop and the first after the resume both read 400 578 input, and the turn
    that followed grew from there. Adding them would have reported nearly twice
    the bill the account was charged. The resend's `TurnRunner` was handed the
    dropped one's per-thread readings instead (`TurnRunner._totals`), so each
    thread's newest reading counts once. The context numbers are a level
    rather than a total: the newest wins, the dropped attempt's kept only when
    the resend reported none at all.

    Text carries over only if the resend produced none, for the reason
    `run_turn` guards `last.md`: an empty answer must never erase a real one.
    """
    resent.commands += dropped.commands
    resent.file_changes += dropped.file_changes
    resent.declined_edits[:0] = dropped.declined_edits
    if not resent.text:
        _keep_answer(dropped, resent)
    resent.thread_id = resent.thread_id or dropped.thread_id
    resent.context_tokens = resent.context_tokens or dropped.context_tokens
    resent.context_window = resent.context_window or dropped.context_window


def _keep_answer(kept: TurnOutcome, into: TurnOutcome) -> None:
    """Make `into` carry `kept`'s answer and everything that says where that
    answer came from. One list for both callers: a provenance field copied by
    one and forgotten by the other dropped the `--expect` note ("follow-up
    answered in the message") from the summary of a failed gate follow-up."""
    into.text = kept.text
    into.report_source = kept.report_source
    into.report_differs = kept.report_differs
    into.report_note = kept.report_note
    into.report_read_error = kept.report_read_error


def _retry_service_drops(*, record: AgentRecord, attempts: list[TurnRunner],
                         timeout: float, effort: str,
                         on_thread: Optional[Callable[[str], None]],
                         log: Callable[[dict], None],
                         sleep: Callable[[float], None],
                         reachable: Callable[[], bool]) -> None:
    """Resend a turn the service dropped, up to `MAX_SERVICE_RETRIES` times.

    The failure this exists for: on 2026-09-13 OpenAI dropped three workers'
    turns inside a minute ("stream disconnected before completion: Unable to
    verify model access", after the server's own five reconnects), and the
    thread and the edits in the tree were all intact — an operator typed one
    `send` per worker and all three finished. This is that `send`, automatic.

    Each slot is a pause and then a reachability check, and the CHECK decides
    whether the slot is spent or used: with no route to OpenAI the resend is
    not made at all (user decision 2026-09-13), which costs nothing and leaves
    the same refusal the operator sees today, with the reason in its text.

    The attempt in progress is APPENDED to `attempts` rather than returned, so
    that `run_turn`'s closing line names the attempt that was really running
    even when an exception (a Ctrl+C during the pause, a server that will not
    start) escapes from the middle of this.
    """
    retries = 0
    unreachable = 0
    for slot, pause in enumerate(SERVICE_RETRY_PAUSES_S, start=1):
        dropped = attempts[-1].outcome
        reason = _service_drop(dropped)
        if not reason:
            break
        thread_id = dropped.thread_id
        if not thread_id:
            # The drop landed before the thread existed, so there is nothing to
            # resume — and a fresh `thread/start` would be a different agent
            # answering the same prompt with none of its history.
            break
        slot_note = {"_meta": RETRY_META, "name": record.name, "attempt": slot,
                     "of": MAX_SERVICE_RETRIES, "reason": reason,
                     "paused_s": pause}
        # Written BEFORE the wait, and that is the point of there being two
        # lines: the dropped attempt has already logged its own failed
        # `turn/completed`, so for the whole of the pause `status` would
        # otherwise call this agent finished — and an operator who believes it
        # sends the next task into a thread that is about to wake up
        # (`events.last_turn_trace` reopens the turn on this line).
        log({**slot_note, "waiting": True,
             "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        sleep(pause)
        alive = bool(reachable())
        retries += alive
        unreachable += not alive
        log({**slot_note, "reachable": alive,
             "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        if not alive:
            continue
        attempts.append(TurnRunner(attempts[-1]))
        prompt = NETWORK_RETRY_PROMPT if dropped.network_wait_timeout else SERVICE_RETRY_PROMPT
        _run_turn_logged(record, prompt, thread_id, timeout,
                         effort, on_thread, log, attempts[-1])
        _carry_over(dropped, attempts[-1].outcome)
    attempts[-1].outcome.service_retries = retries
    _note_retries_in_error(attempts[-1].outcome, retries, unreachable)


def _clear_report(report: Path, log: Callable[[dict], None]) -> None:
    try:
        report.unlink(missing_ok=True)
    except OSError as exc:
        log({"_meta": "report_unlink_error", "path": str(report),
             "error": type(exc).__name__})


def _read_report(outcome: TurnOutcome, report: Optional[Path]) -> None:
    """Select the answer once after the service-attempt chain, before expect.

    A follow-up is another completed wire turn and gets its own read; neither
    persistence nor logging reopens the file. Failed attempts keep their
    message, because a partial write is not a completed report.
    """
    if outcome.message_text or not outcome.report_source:
        outcome.report_source = "message"
    if report is None or outcome.status != "completed" or outcome.error:
        return
    try:
        data = report.read_bytes()
    except OSError as exc:
        outcome.report_read_error = type(exc).__name__
        return
    # PowerShell 5.1 redirection writes UTF-16; its UTF-8 encoding adds a BOM.
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    text = data.decode(encoding, errors="replace")
    if text.strip():
        outcome.report_differs = text.strip() != outcome.message_text.strip()
        outcome.text = text
        outcome.report_source = "file"
        if outcome.sub_agents.reply:
            # The turn about the children wrote no report and the first one
            # was put back (`subagent_wait._sub_agent_turn`): its message —
            # the children's results, as the main thread used them — goes
            # under it HERE, where the answer is selected, so a later
            # follow-up that fails and keeps this answer (`_keep_answer`)
            # keeps the message too.
            outcome.text = (text.rstrip() + "\n\n"
                            + outcome.sub_agents.reply.strip() + "\n")


def _send_follow_up(*, record: AgentRecord, attempts: list[TurnRunner],
                    done: TurnOutcome, kind: str, meta: dict, prompt: str,
                    what: str, timeout: float, effort: str,
                    on_thread: Optional[Callable[[str], None]],
                    log: Callable[[dict], None],
                    report: Optional[Path]) -> bool:
    """One follow-up turn on `done`'s thread, appended to `attempts`.

    Writes the `FOLLOWUP_META` line of `kind` first (`meta` is what that kind
    carries), so both writers stamp the same envelope. With a report path the
    follow-up is asked for a replacement file, so the old one is cleared
    first and an unchanged report cannot pass as the new answer. The new
    outcome inherits `done`'s `--expect` count and pattern; the caller counts
    its own kind. False when the follow-up did not complete: the first answer
    is kept and the error names `what` was lost."""
    log({"_meta": FOLLOWUP_META, "kind": kind, "name": record.name, **meta,
         "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    if report is not None:
        _clear_report(report, log)
    attempts.append(TurnRunner(attempts[-1]))
    _run_turn_logged(record, prompt, done.thread_id, timeout, effort,
                     on_thread, log, attempts[-1])
    followed = attempts[-1].outcome
    followed.follow_ups = done.follow_ups
    followed.expect = done.expect
    _carry_over(done, followed)
    if followed.status != "completed" or followed.error:
        _keep_answer(done, followed)
        followed.expect_met = done.expect_met
        followed.error = follow_up_failed(what, followed.error or followed.status)
        return False
    if report is not None:
        _read_report(followed, report)
    return True


def _follow_up_on_missing(*, record: AgentRecord, attempts: list[TurnRunner],
                          expect: str, timeout: float, effort: str,
                          on_thread: Optional[Callable[[str], None]],
                          log: Callable[[dict], None],
                          report: Optional[Path] = None) -> None:
    """Ask once more, on the same thread, when the answer lacks what the
    caller's `expect` regex names — the automatic form of the `send` "cover
    the remaining N" that a Codex reviewer given five questions or fifteen
    commits needed every time (three resends in the wave of 2026-09-13, and
    the day after a 15-commit breakdown answered one question of five).

    Only a COMPLETED answer is judged: a failed or timed-out turn has its own
    report, and prodding it would spend a thread resend on a turn that was
    not short of a section but of a result. The two answers are joined
    (`join_answers`); the follow-up's own failure keeps the first answer and
    says in the error what was lost.
    A worker with a report path instead writes a complete replacement file;
    its contents have already replaced the message before the regex is tested.
    A message-only follow-up extends the saved first report, with a note
    explaining that the joined answer includes a message.

    The last completed answer records `expect_met`, re-judged after the
    follow-up: two answers both lacking the section once read as completed
    with "the first answer lacked", i.e. as a gap closed.
    """
    done = attempts[-1].outcome
    done.expect = expect
    # This call's own count, not `follow_ups`: `_check_gates` runs this again
    # over the gate follow-up's replacement answer.
    sent = 0
    while needs_expect_follow_up(expect, done.text, sent,
                                 completed=_completed(done)):
        if not done.thread_id:
            break
        sent += 1
        first_text = done.text
        prompt = (EXPECT_FOLLOWUP_PROMPT.format(pattern=expect) if report is None
                  else REPORT_FOLLOWUP_PROMPT.format(pattern=expect, report=report))
        ok = _send_follow_up(
            record=record, attempts=attempts, done=done, kind=FOLLOWUP_EXPECT,
            meta={"expect": expect, "report_source": done.report_source,
                  "report_differs": done.report_differs},
            prompt=prompt, what=expect_follow_up(expect), timeout=timeout,
            effort=effort, on_thread=on_thread, log=log, report=report)
        followed = attempts[-1].outcome
        followed.follow_ups += 1
        if not ok:
            return
        if report is None or followed.report_source == "message":
            followed.text = join_answers(first_text, followed.text)
            if done.report_source == "file" and followed.message_text:
                followed.report_source = "file"
                followed.report_differs = done.report_differs
                followed.report_note = "follow-up answered in the message"
        done = followed
    verdict = expect_verdict(expect, done.text, completed=_completed(done))
    if verdict is not None:
        done.expect_met = verdict


def _completed(outcome: TurnOutcome) -> bool:
    """A turn with a result to judge: completed and without an error."""
    return outcome.status == "completed" and not outcome.error


def _attach_gates(outcome: TurnOutcome, results: list[gates_module.GateRun],
                  follow_ups: int) -> None:
    outcome.gates = results
    outcome.gate_follow_ups = follow_ups
    block = gates_module.render(results, follow_ups)
    outcome.text = (outcome.text.rstrip() + "\n\n" if outcome.text else "") + block + "\n"


def _check_gates(*, record: AgentRecord, attempts: list[TurnRunner],
                 timeout: float, effort: str,
                 on_thread: Optional[Callable[[str], None]],
                 log: Callable[[dict], None], report: Optional[Path],
                 run_gate: gates_module.GateRunner) -> None:
    """Run the recorded gates outside the sandbox and put what they said
    under the answer; on a red one, send ONE follow-up on the same thread with
    the output tail, then run them all again (`gates` says why the wrapper,
    not the worker).

    Only a COMPLETED turn is gated, for `_follow_up_on_missing`'s reason: a
    failed turn's problem is not its gates. Each run of the gates shares the
    turn's timeout (`gates.run_gates`). The follow-up asks for a complete
    replacement report, since the first one's `### Gates` describes a tree
    that has since changed, and that report is held to `expect` like the
    first; if the follow-up itself fails the first answer is kept, with the
    first run's results under it and the loss named in the error.

    Without a host (`host_policy.EMPTY`) only gates this invocation's caller
    passed run (`spawn --gates-from-prompt`): a `send` replaying recorded
    gates outside the sandbox is the wrapper's own decision, not the caller's.
    """
    commands = record.gates
    if not commands:
        return
    done = attempts[-1].outcome
    if host_policy.active() is host_policy.EMPTY and "gates" not in host_policy.explicit():
        done.gate_note = ("gates not run: no host policy, and this call passed "
                          "none (spawn --gates-from-prompt)")
        return
    if done.status != "completed" or done.error:
        done.gate_note = "gates not run: the turn did not complete"
        return
    project = Path(record.cwd or host_root())
    results = gates_module.run_gates(commands, project, timeout, run_gate, log)
    if all(result.passed for result in results) or not done.thread_id:
        _attach_gates(done, results, 0)
        return
    ok = _send_follow_up(
        record=record, attempts=attempts, done=done, kind=FOLLOWUP_GATES,
        meta={"gates": [result.command for result in results
                        if not result.passed]},
        prompt=gates_module.follow_up_prompt(results, report),
        what="gate follow-up", timeout=timeout, effort=effort,
        on_thread=on_thread, log=log, report=report)
    if not ok:
        _attach_gates(attempts[-1].outcome, results, 1)
        return
    # The replacement answer owes the same contract as the first one: without
    # this a complete report replaced by "fixed" passed once the gate went
    # green (review 2026-09-30). The expect follow-up may edit the tree too,
    # so the gates run after it, on the last turn.
    _follow_up_on_missing(
        record=record, attempts=attempts, expect=done.expect, timeout=timeout,
        effort=effort, on_thread=on_thread, log=log, report=report)
    _attach_gates(attempts[-1].outcome,
                  gates_module.run_gates(commands, project, timeout, run_gate, log),
                  1)


def _note_retries_in_error(outcome: TurnOutcome, retries: int,
                           unreachable: int) -> None:
    """Say in the error itself what was spent before it.

    The error message is the half of the outcome that gets pasted into a
    report, so a turn that failed after minutes of waiting and a whole thread
    resent must say so there and not only in the summary line — including when
    the resend failed for some OTHER reason, which is the case the first draft
    stayed silent about.

    Guarded on `outcome.error` rather than on the count: appending to the empty
    error of a SUCCESSFUL retry would make `print_outcome` — which reads "no
    error" as the definition of ok — report the rescued turn as failed.
    """
    if not outcome.error:
        return
    notes = []
    if retries:
        notes.append(retried_note(retries))
    if unreachable:
        notes.append(f"service unreachable after {unreachable} check(s)")
    if notes:
        outcome.error = f"{outcome.error} ({'; '.join(notes)})"


def _run_turn_logged(record: AgentRecord, prompt: str, resume_thread: str,
                     timeout: float, effort: str,
                     on_thread: Optional[Callable[[str], None]],
                     log: Callable[[dict], None], runner: TurnRunner) -> None:
    """One attempt at the turn: `run_turn` owns the log and the closing line,
    this owns the server and fills `runner.outcome`.

    The thread to resume arrives as an ID rather than as a `resume` flag over
    `record.thread_id`, because a service-drop retry resumes the thread the
    attempt before it OPENED — which the record does not know about until its
    caller stores it (`on_thread`), and a programmatic caller need not."""
    workspace, sandbox_policy = sandbox.prepare_workspace(record)
    server = AppServer(workspace, log=log, approve=sandbox.escalation_judge(record),
                       approve_files=sandbox.file_change_judge(record))
    wait = SubAgentWait(log=log, name=record.name)
    runner.network_log = log
    try:
        server.request(M_INITIALIZE, {
            "clientInfo": {"name": "codex_agent", "title": "codex_agent",
                           "version": "1.0.0"},
            "capabilities": {"optOutNotificationMethods": NOISY_NOTIFICATIONS},
        })
        server.notify(M_INITIALIZED, {})

        # Bracketed for `status` (`events.OPENER_META`): the done line is in
        # the `finally`, so a failed open does not leave the grace standing.
        log({"_meta": OPENER_META, "phase": OPENER_THREAD_OPEN,
             "grace_s": THREAD_OPEN_GRACE_S,
             "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        try:
            if resume_thread:
                thread = server.request(
                    M_THREAD_RESUME,
                    {"threadId": resume_thread, **sandbox.thread_params(record)},
                    timeout=THREAD_OPEN_TIMEOUT_S, on_event=runner.on_event)
            else:
                thread = server.request(M_THREAD_START,
                                        sandbox.thread_params(record),
                                        timeout=THREAD_OPEN_TIMEOUT_S,
                                        on_event=runner.on_event)
        finally:
            log({"_meta": OPENER_META, "phase": OPENER_DONE,
                 "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        thread_id = str((thread.get("thread") or {}).get("id") or "")
        if not thread_id:
            raise RuntimeError(f"{M_THREAD_START} returned no thread id")
        runner.outcome.thread_id = thread_id
        if on_thread is not None:
            on_thread(thread_id)

        if not resume_thread:
            # Best-effort: the name is for humans reading `codex agents`,
            # so a server that rejects it must not fail the turn.
            #
            # Only a rejection, though. A dead server must still be reported
            # as one: swallowing it here consumed the end-of-stream sentinel
            # (it arrives once), and the next request then waited out its
            # whole timeout and blamed a slow model for a crash whose stderr
            # had already been discarded.
            try:
                server.request(M_THREAD_NAME_SET,
                               {"threadId": thread_id, "name": record.name})
            except TimeoutError:
                pass
            except RuntimeError:
                if server.finished:
                    raise

        turn_params: dict[str, Any] = {
            "threadId": thread_id,
            "input": [{"type": "text", "text": prompt}],
        }
        # `effort`, not `reasoningEffort` (schema bundle of codex-cli 0.149.0):
        # a misspelt field is ignored in silence. Why `sandboxPolicy` is an
        # object here: `sandbox.turn_sandbox_policy`.
        if effort:
            turn_params["effort"] = effort
        if sandbox_policy is not None:
            turn_params["sandboxPolicy"] = sandbox_policy
        server.request(M_TURN_START, turn_params, on_event=runner.on_event)
        await_main_and_sub_agents(server, runner, turn_params, timeout,
                                  sandbox.report_path(record), wait)
    except TimeoutError as exc:
        runner.outcome.status = "timeout"
        runner.outcome.error = str(exc)
        runner.outcome.network_wait_timeout = runner.outcome.thread_id in runner.network.pending
        if isinstance(exc, NetworkWaitTimeout):
            runner.outcome.network_wait_timeout &= not exc.exhausted
            if exc.exhausted:
                runner.outcome.status = "failed"
                runner.outcome.error_kind = ""
    except (RuntimeError, OSError) as exc:
        runner.outcome.status = runner.outcome.status or "failed"
        runner.outcome.error = runner.outcome.error or str(exc)
    finally:
        runner.network.stop(time.monotonic())
        if runner.network.total_s:
            runner.log_network_wait()
        # Before the close, on every exit: the close kills every thread on
        # the server, and a child killed unnamed is a worker's half-made
        # edits under a clean-looking report. Nested, because the settling
        # pumps for up to `SUB_AGENT_INTERRUPT_S`: a second Ctrl+C in it must
        # still not leave the app-server running.
        try:
            settle_sub_agents(server, runner, wait)
        finally:
            server.close()
            wait.restore_report()
            runner.outcome.declined_edits.extend(
                entry for entry in server.file_decisions
                if entry.get("decision") != "accept")
