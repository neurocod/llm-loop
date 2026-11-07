"""One attempt's sub-agent phase: wait for the children that outlived the
main answer, send the main thread the one turn about them, interrupt whatever
is left, and record what became of them in the attempt's `SubAgentAccount`.

The phase reaches the attempt only through `Attempt` and the app-server only
through `Server` — `turn.TurnRunner` and `protocol.AppServer` are one of each
— so that it can be driven by any loop that folds app-server events, not only
by `turn`'s. `turn._run_turn_logged` calls `await_main_and_sub_agents` once
the turn is started and `settle_sub_agents` from its `finally`.

The bounds are read at call time from `subagents` (`SUB_AGENT_GRACE_S`,
`SUB_AGENT_INTERRUPT_S`), so tests shorten them by patching that module.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

from ..answer_contract import follow_up_failed

from . import subagents
from .events import (
    SUB_AGENTS_CUT, SUB_AGENTS_END, SUB_AGENTS_FOLLOW_UP, SUB_AGENTS_META,
    SUB_AGENTS_WAITING,
)
from .protocol import M_TURN_INTERRUPT, M_TURN_START
from .service import NetworkWait
from .subagents import SubAgentAccount, SubAgentTracker


class Server(Protocol):
    """What the phase uses of an app-server connection: three calls and `finished`."""
    finished: bool

    def pump(self, deadline: float, on_event: Callable[[dict], None]) -> None:
        ...

    def request(self, method: str, params: dict, *,
                on_event: Optional[Callable[[dict], None]] = None) -> dict:
        ...

    def send_request(self, method: str, params: Optional[dict] = None) -> int:
        ...


class Attempt(Protocol):
    """What the phase needs of one attempt: its children, its main thread's
    state, its account, and a way to put a complete answer back when a
    later main turn of the same attempt fails."""
    # Fed by `on_event`.
    subs: SubAgentTracker
    # The main thread's current turn has completed. The phase clears it
    # before a turn it sends and sets it when it stops waiting for one.
    done: bool
    network: NetworkWait

    @property
    def sub_agents(self) -> SubAgentAccount:
        ...

    def on_event(self, message: dict) -> None:
        ...

    def on_sub_agent_event(self, message: dict) -> None:
        """`on_event` for sub-agents' events only: the drain after the
        attempt's verdict is final."""

    def main_verdict(self) -> str:
        """"" when the main thread's last turn completed without an error;
        otherwise why not (non-empty)."""

    def answer(self) -> Any:
        """The current answer and verdict, opaque, for `restore_answer`."""

    def restore_answer(self, kept: Any) -> None:
        ...

    def message(self) -> str:
        """The main thread's own last message (never a report file's)."""


@dataclass
class SubAgentWait:
    """One attempt's wait for its sub-agents (`await_main_and_sub_agents`)
    and its `SUB_AGENTS_META` lines."""
    log: Callable[[dict], None]
    name: str
    # The children's deadline once a main answer left some running; one per
    # attempt, whatever the main thread does after it.
    grace_end: Optional[float] = None
    started: float = 0.0
    # When the waiting stopped being for the children: the turn sent about
    # them is the main thread's work, not a wait (0 while still waiting).
    stopped: float = 0.0
    # The worker report to put back once the server is stopped
    # (`_sub_agent_turn`), as (path, bytes or None for "absent").
    report_restore: Optional[tuple[Path, Optional[bytes]]] = None

    def log_line(self, phase: str, **fields: Any) -> None:
        self.log({"_meta": SUB_AGENTS_META, "phase": phase, "name": self.name,
                  **fields, "at": time.strftime("%Y-%m-%d %H:%M:%S")})

    def restore_report(self) -> None:
        """Put back the report a failed turn about the children was asked to
        replace. Only once the server is stopped: a turn that timed out may
        still have been writing it until then."""
        if self.report_restore is not None:
            _restore_report(*self.report_restore)


def _over(deadline: float, what: str = "turn") -> None:
    """Raise the pump's own TimeoutError once `deadline` has passed.

    By the clock, not only by an empty queue: `Queue.get(timeout=0)` on a
    queue that always holds a message never expires, so a main thread or a
    child that keeps streaming (token usage, items) kept every loop of this
    kind past its deadline for as long as it streamed (found in review)."""
    if time.monotonic() >= deadline:
        raise TimeoutError(f"codex app-server did not answer {what} in time")


def _pump_main(server: Server, attempt: Attempt, deadline: float,
               wait: SubAgentWait) -> None:
    """Pump until the main thread's turn completes; TimeoutError past
    `deadline`. A child that outlived an earlier main answer is interrupted
    on ITS deadline meanwhile — a main thread busy with a turn of its own
    does not extend the children's wait."""
    baseline_s = attempt.network.seconds(time.monotonic())
    while not attempt.done:
        extended = attempt.network.deadline(deadline, baseline_s, time.monotonic())
        _over(extended)
        overdue = (attempt.subs.open_threads(late_only=True)
                   if wait.grace_end is not None else {})
        if overdue and time.monotonic() >= wait.grace_end:
            _interrupt_sub_agents(server, attempt, wait, overdue, keep_main=True)
            continue
        limit = min(extended, wait.grace_end) if overdue else extended
        try:
            server.pump(limit, attempt.on_event)
        except TimeoutError:
            _over(attempt.network.deadline(deadline, baseline_s, time.monotonic()))
        # A completed answer already delivered before teardown wins the race
        # with the ceiling, even if another thread is still reconnecting.
        if not attempt.done:
            attempt.network.deadline(deadline, baseline_s, time.monotonic())


def await_main_and_sub_agents(server: Server, attempt: Attempt,
                              turn_params: dict, timeout: float,
                              report: Optional[Path],
                              wait: SubAgentWait) -> None:
    """Pump until the main thread's turn completes — and, when sub-agents
    are still running then, until they are done or their deadline
    (`subagents.SUB_AGENT_GRACE_S`) passes.

    Then at most ONE more main turn per attempt (user decisions 2026-10-24):
    `DONE_TURN` with the results of the children that finished after the
    answer, or `RESTART_TURN` naming the ones that had to be cut, so the main
    thread can restart them or finish their work; its answer is the outcome
    (`_sub_agent_turn`). Children still running when that turn ends get no
    further wait after a restart turn, and only what is left of the deadline
    after a done turn; whatever is left open is cut by `settle_sub_agents`.

    A main thread that starts a turn by itself is a continuation, waited out
    under a re-armed `timeout` (rust-v0.164.0 does not do this by default:
    agent/control/api.rs, session/turn_input.rs; kept as a defensive path),
    and like the turn about the children it cannot cost the complete answer
    before it (`_continuation`). A main turn that did not complete is not
    waited past: a failed one is about to be resent on a fresh server
    (`turn._retry_service_drops`), where these children do not exist.
    """
    subs = attempt.subs
    account = attempt.sub_agents
    deadline = time.monotonic() + timeout
    try:
        while True:
            _pump_main(server, attempt, deadline, wait)
            if attempt.main_verdict():
                return
            kept = attempt.answer()
            if subs.open_threads(late_only=True):
                if account.turn == subagents.RESTART_TURN:
                    return  # no second wait: the settling cuts them
                if wait.grace_end is None:
                    wait.started = time.monotonic()
                    wait.grace_end = wait.started + subagents.SUB_AGENT_GRACE_S
                    wait.log_line(SUB_AGENTS_WAITING,
                                  open=[subs.path(thread) for thread in
                                        subs.open_threads(late_only=True)],
                                  grace_s=subagents.SUB_AGENT_GRACE_S)
                if not _pump_idle_main(server, attempt, wait):
                    return  # the server died; the children died with it
                # The main thread started a turn of its own meanwhile — it may
                # even have ended it, badly, inside an interrupt drain.
                if not attempt.done or attempt.main_verdict():
                    if not _continuation(server, attempt, timeout, wait, kept):
                        return
                continue
            if account.turn:
                return  # one turn per attempt; the rest is the block's
            clause = (f", and write that complete report to {report} again"
                      if report is not None else "")
            planned = subs.turn_prompt(clause, wait.name)
            if planned is None:
                return
            kind, text, finished, cut = planned
            wait.log_line(SUB_AGENTS_FOLLOW_UP, kind=kind,
                          finished=[subs.path(thread) for thread in finished],
                          cut=[subs.path(thread) for thread in cut])
            account.turn = kind
            if wait.grace_end is not None:
                wait.stopped = time.monotonic()
            if not _sub_agent_turn(server, attempt, turn_params, text, kind,
                                   timeout, report, wait):
                return  # untold: their results are quoted under the answer
            subs.tell(finished + cut)
    finally:
        if wait.grace_end is not None:
            account.wait_s = (wait.stopped or time.monotonic()) - wait.started


def _pump_idle_main(server: Server, attempt: Attempt,
                    wait: SubAgentWait) -> bool:
    """With the main thread idle, pump until its late children are all done,
    it starts a turn of its own, or the children's deadline passes — then
    interrupt them. False when the server died meanwhile."""
    subs = attempt.subs
    while attempt.done and subs.open_threads(late_only=True):
        if time.monotonic() >= wait.grace_end:
            _interrupt_sub_agents(server, attempt, wait,
                                  subs.open_threads(late_only=True),
                                  keep_main=True)
            break
        try:
            server.pump(wait.grace_end, attempt.on_event)
        except TimeoutError:
            continue
        except RuntimeError:
            # The main answer is in; the server's death ends the children,
            # not the answer (the settling names them).
            if server.finished:
                return False
            raise
    return True


def _set_report_aside(report: Optional[Path]) -> Optional[bytes]:
    """Read and remove the worker's report before a turn that is asked for a
    new one, as `turn._send_follow_up` clears it: an unchanged file must not
    pass for the new answer. The bytes come back for `_restore_report`."""
    if report is None:
        return None
    try:
        data: Optional[bytes] = report.read_bytes()
    except OSError:
        data = None
    try:
        report.unlink(missing_ok=True)
    except OSError:
        pass
    return data


def _restore_report(report: Optional[Path], data: Optional[bytes]) -> None:
    if report is None:
        return
    try:
        if data is None:
            report.unlink(missing_ok=True)
        else:
            report.write_bytes(data)
    except OSError:
        pass


def _sub_agent_turn(server: Server, attempt: Attempt, turn_params: dict,
                    text: str, kind: str, timeout: float,
                    report: Optional[Path], wait: SubAgentWait) -> bool:
    """Send the main thread the one turn about its children, on the same
    server (a resume elsewhere would lose them and their results), and wait
    for it under its own `timeout`.

    A turn that is rejected, fails or times out does not fail the attempt:
    the complete first answer and its status are put back, the worker's
    report too — once the server is stopped (`SubAgentWait.restore_report`),
    since a timed-out turn may still be writing it — and `sub_agents.note`
    says what was lost, so the gates still run and the exit code is the
    first answer's, as `turn._send_follow_up` keeps it. False then.

    A turn that completes without writing a new report keeps the first
    (complete) one, and its own message is joined under it
    (`sub_agents.reply`, `turn._read_report`)."""
    kept = attempt.answer()
    message_before = attempt.message()
    saved = _set_report_aside(report)
    # Before the request: its `turn/completed` may arrive while `request` is
    # still waiting for the reply.
    attempt.done = False
    attempt.subs.main_started()
    reason = _extra_main_turn(server, attempt, timeout, wait, {
        **turn_params, "input": [{"type": "text", "text": text}]})
    if not reason:
        if report is not None and saved is not None and not report.exists():
            _restore_report(report, saved)
            if attempt.message() != message_before:
                attempt.sub_agents.reply = attempt.message()
        return True
    if report is not None:
        wait.report_restore = (report, saved)
    _keep_first_answer(attempt, kept, f"sub-agent {kind} turn", reason)
    return False


def _extra_main_turn(server: Server, attempt: Attempt, timeout: float,
                     wait: SubAgentWait,
                     start: Optional[dict] = None) -> str:
    """Pump a main turn after the attempt's first answer — the one sent
    with `start` as its `turn/start` params, or a continuation — under its
    own `timeout`: why it did not complete, "" when it did."""
    try:
        if start is not None:
            server.request(M_TURN_START, start, on_event=attempt.on_event)
        _pump_main(server, attempt, time.monotonic() + timeout, wait)
    except (TimeoutError, RuntimeError, OSError) as exc:
        return str(exc)
    return attempt.main_verdict()


def _keep_first_answer(attempt: Attempt, kept: Any, what: str,
                       reason: str) -> None:
    """A main turn after the first answer failed: the answer and its status
    are put back, `sub_agents.note` names `what` was lost, and the main
    thread is no longer waited for."""
    attempt.restore_answer(kept)
    attempt.sub_agents.note = follow_up_failed(what, reason)
    attempt.done = True


def _continuation(server: Server, attempt: Attempt, timeout: float,
                  wait: SubAgentWait, kept: Any) -> bool:
    """Wait out a turn the main thread started by itself after its answer.
    One that times out or fails costs the continuation, not the answer
    before it (`_keep_first_answer`, as for `_sub_agent_turn`) — a failed
    status would otherwise send a complete answer to
    `turn._retry_service_drops`. False then; the main thread may still be
    running, which the settling's drain ignores."""
    reason = _extra_main_turn(server, attempt, timeout, wait)
    if reason:
        _keep_first_answer(attempt, kept, "the main thread's own continuation",
                           reason)
    return not reason


def _interrupt_sub_agents(server: Server, attempt: Attempt,
                          wait: SubAgentWait, threads: dict[str, str], *,
                          keep_main: bool) -> None:
    """`turn/interrupt` each of `threads` (thread -> open turn id) on its own
    thread — interrupting the main one leaves them running — and pump for up
    to `SUB_AGENT_INTERRUPT_S` so what they do on the way out is counted;
    then they are cut, answered or not.

    The server answers an interrupt when the turn aborts, just BEFORE that
    turn's `turn/completed` (app-server bespoke_event_handling.rs), so the
    reply is not waited for: the completion is. `keep_main` False is the
    drain after the attempt's verdict, where main-thread events are dropped
    (`Attempt.on_sub_agent_event`)."""
    if not threads:
        return
    wait.log_line(SUB_AGENTS_CUT,
                  cut=[attempt.subs.path(thread) for thread in threads])
    if not server.finished:
        on_event = attempt.on_event if keep_main else attempt.on_sub_agent_event
        end = time.monotonic() + subagents.SUB_AGENT_INTERRUPT_S
        try:
            for thread, turn_id in threads.items():
                server.send_request(M_TURN_INTERRUPT,
                                    {"threadId": thread, "turnId": turn_id})
            while any(attempt.subs.is_open(thread) for thread in threads):
                _over(end, "the interrupt")
                server.pump(end, on_event)
        except (TimeoutError, RuntimeError, OSError, ValueError):
            pass
    attempt.subs.mark_cut(threads)


def settle_sub_agents(server: Server, attempt: Attempt,
                      wait: SubAgentWait) -> None:
    """Interrupt whatever sub-agent is still open as the attempt ends (its
    verdict is final: main-thread events are dropped), then record the
    attempt's account of them and the `SUB_AGENTS_END` line. Runs from
    `finally` and never raises an OSError."""
    try:
        _interrupt_sub_agents(server, attempt, wait,
                              attempt.subs.open_threads(), keep_main=False)
        fates = attempt.subs.fates()
        account = attempt.sub_agents
        account.late = fates
        if not fates:
            return

        def listed(state: str) -> list[dict]:
            return [{"path": fate.path, "thread": fate.thread_id,
                     "files": fate.files, "told": fate.told,
                     "status": fate.status, "restarted": fate.restarted}
                    for fate in fates if fate.state == state]

        wait.log_line(SUB_AGENTS_END, cut=listed("cut"),
                      finished=listed("finished"),
                      waited_s=round(account.wait_s, 1),
                      turn=account.turn, note=account.note)
    except OSError:
        pass
