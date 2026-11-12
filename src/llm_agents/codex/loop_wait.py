"""Adapt the loop provider's connection to the shared sub-agent wait phase."""

# The Protocol's `X | None` annotations would be evaluated at import on 3.9.
from __future__ import annotations

import time
from typing import Callable, Protocol

from llm_loop import wire, stopchannel

from .service import NetworkWait
from .subagents import SubAgentAccount, SubAgentTracker
from .subagent_wait import SubAgentWait, await_main_and_sub_agents, settle_sub_agents
from . import subagents


class WaitStream(Protocol):
    """The provider-owned transport lent to the wait adapter on its reader thread.

    IDs are read-only; follow_turn also updates the operator-note target.
    next_event consumes one record before an absolute monotonic deadline,
    resolves pending replies (returning None), and returns notifications,
    diagnostics, or unmatched replies. It raises TimeoutError on expiry and
    RuntimeError/OSError on transport failure. request has its own bounded
    response deadline, dispatches intervening records to on_event, calls
    on_poll even while idle, and propagates its exception after unregistering
    the pending reply; a later reply can therefore appear in next_event.
    send_request does not await a reply. aggregate_usage returns a fresh sum,
    never mutable provider storage. The provider owns process/pipe lifetime;
    the adapter neither closes them nor accesses the connection directly.
    progress is cosmetic and may raise; stop_requested is the run's policy.
    """

    @property
    def thread_id(self) -> str: ...

    @property
    def turn_id(self) -> str: ...

    @property
    def finished(self) -> bool: ...

    def follow_turn(self, turn_id: str) -> None: ...
    def next_event(self, deadline: float) -> dict | str | None: ...
    def request(self, method: str, params: dict, *,
                on_event: Callable, on_poll: Callable) -> dict: ...
    def send_request(self, method: str, params: dict | None = None) -> int: ...
    def observe_usage(self, message: dict) -> None: ...
    def aggregate_usage(self) -> dict: ...
    def progress(self, event: dict) -> None: ...
    def stop_requested(self) -> bool: ...


class WaitStopped(RuntimeError):
    """The run requested an end after its initial answer."""


class LoopWait:
    """Keep terminal events private until children and the follow-up settle.

    The initial turn still streams normally. Only the phase after its answer
    is buffered, so neither renderer can close stdin on an intermediate
    completion or mistake a failed follow-up for a failed initial answer.
    """

    def __init__(self, stream: WaitStream):
        self.stream = stream
        self.subs = SubAgentTracker()
        self.subs.main_started()
        self.sub_agents = SubAgentAccount()
        self.network = NetworkWait()
        self.done = False
        self.terminal = None
        self.text = ""
        self.events = []
        self.settling = False
        self.stopped = False
        self.extra_dispatched = False
        self.extra_turn_id = ""
        self.extra_done = False
        self.progress_failed = False

    def progress(self, event):
        if self.progress_failed:
            return
        try:
            self.stream.progress(event)
        except Exception as exc:
            # A broken display must not prevent CUT from sending interrupts.
            self.progress_failed = True
            try:
                from llm_loop import diaglog
                diaglog.record("codex.sub_agents.progress_failed", str(exc))
            except Exception:
                pass  # Even the diagnostic sink may be the failed display.

    def check_stop(self):
        if not self.settling and self.stream.stop_requested():
            self.stopped = True
            raise WaitStopped("stop requested during sub-agent wait")

    @property
    def finished(self):
        return self.stream.finished

    def pump(self, deadline, on_event):
        while True:
            self.check_stop()
            limit = min(deadline, time.monotonic() + stopchannel.STOP_RECHECK_SECONDS)
            try:
                message = self.stream.next_event(limit)
                break
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise
        if isinstance(message, dict):
            on_event(message)
        elif message is not None:
            self.events.append(message)

    def request(self, method, params, *, on_event=None):
        self.check_stop()
        self.extra_dispatched = True
        # An automatic continuation may have completed before this request.
        self.extra_turn_id = ""
        self.extra_done = False
        result = self.stream.request(
            method, params, on_event=on_event,
            on_poll=self.check_stop)
        turn_id = wire.codex_app_turn_id(result)
        if turn_id:
            self.extra_turn_id = turn_id
            self.stream.follow_turn(turn_id)
        return result

    def send_request(self, method, params=None):
        return self.stream.send_request(method, params)

    def on_sub_agent_event(self, message):
        if wire.codex_app_from_sub_agent(message, self.stream.thread_id):
            self.observe(message)

    def on_event(self, message):
        if not isinstance(message, dict):
            self.events.append(message)
            return
        event = self.observe(message, continuing=True)
        if event is not None and wire.event_type(event) not in (
                wire.TURN_COMPLETED, wire.TURN_FAILED, wire.ERROR):
            self.events.append(event)

    def observe(self, message, continuing=False):
        params = message.get("params") or {}
        method = message.get("method")
        child = wire.codex_app_from_sub_agent(message, self.stream.thread_id)
        turn = params.get("turn") or {}
        item = params.get("item") if method == "item/completed" else None
        if child:
            thread = wire.codex_app_event_thread(message)
            if item:
                self.subs.on_record(item)
                self.subs.on_item(thread, str(params.get("turnId") or ""), item)
            elif method == wire.APP_TURN_STARTED:
                self.subs.on_turn_started(thread, str(turn.get("id") or ""))
            elif method == wire.APP_TURN_COMPLETED:
                self.subs.on_turn_completed(thread, turn)
            self.stream.observe_usage(message)
            return None
        if continuing and method == wire.APP_TURN_STARTED and turn.get("id"):
            self.extra_turn_id = str(turn["id"])
            self.stream.follow_turn(self.extra_turn_id)
            self.done = False
            self.subs.main_started()
        if not wire.codex_app_in_turn(message, self.stream.turn_id):
            return None
        self.network.observe(message, time.monotonic(), "", self.stream.thread_id)
        self.stream.observe_usage(message)
        if item:
            self.subs.on_record(item)
            if item.get("type") == "agentMessage":
                self.text = str(item.get("text") or "")
        event = wire.codex_app_event(message, self.stream.aggregate_usage())
        if method == wire.APP_TURN_COMPLETED:
            if str(turn.get("id") or "") == self.extra_turn_id:
                self.extra_done = True
            self.done = True
            self.subs.main_completed()
            self.terminal = event
        return event

    def main_verdict(self):
        if self.terminal and wire.event_type(self.terminal) == wire.TURN_COMPLETED:
            return ""
        return wire.codex_error(self.terminal or {}) or "main turn failed"

    def answer(self):
        return self.terminal, self.text, len(self.events)

    def restore_answer(self, kept):
        self.terminal, self.text, count = kept
        self.events[count:] = [event for event in self.events[count:]
                               if not (isinstance(event, dict) and
                                       wire.codex_item_type(wire.codex_item(event))
                                       == wire.AGENT_MESSAGE)]

    def message(self):
        return self.text

    def interrupt_follow_up(self):
        """Cancel even a start whose reply arrived after request cancellation.

        A turn/started notification or the abandoned request's late reply
        supplies the ID. Discovery and the interrupt drain share the existing
        interrupt budget; an unresponsive start gets no extra turn timeout.
        Drain through on_event so edits/commands and newly spawned children
        are counted, then finish restores the answer without discarding them.
        """
        if not self.extra_dispatched or self.extra_done:
            return
        deadline = time.monotonic() + subagents.SUB_AGENT_INTERRUPT_S
        interrupted = False
        try:
            while not self.extra_done and not self.finished:
                if time.monotonic() >= deadline:
                    break
                if self.extra_turn_id and not interrupted:
                    self.send_request("turn/interrupt", {
                        "threadId": self.stream.thread_id,
                        "turnId": self.extra_turn_id})
                    interrupted = True

                def drain(message):
                    turn_id = wire.codex_app_turn_id(message.get("result") or {})
                    if turn_id:
                        self.extra_turn_id = turn_id
                        self.stream.follow_turn(turn_id)
                    self.on_event(message)

                self.pump(deadline, drain)
        except (TimeoutError, RuntimeError, OSError, ValueError):
            pass  # Child settling and answer restoration still have to run.

    def finish(self):
        # Reuse the named worker's existing turn budget for the one follow-up.
        from .turn import DEFAULT_TURN_TIMEOUT_S

        wait = SubAgentWait(self.progress, None)
        initial = self.answer()
        try:
            self.check_stop()
            await_main_and_sub_agents(
                self, self, {"threadId": self.stream.thread_id},
                DEFAULT_TURN_TIMEOUT_S, None, wait)
        except WaitStopped as exc:
            self.sub_agents.note = str(exc)
        finally:
            self.settling = True
            try:
                if self.stopped:
                    try:
                        self.interrupt_follow_up()
                    finally:
                        self.restore_answer(initial)
            finally:
                settle_sub_agents(self, self, wait)
        account = self.sub_agents.under_answer(self.text, None) or self.sub_agents.block
        if account and (self.sub_agents.late or self.sub_agents.note):
            self.events.append({"type": wire.ITEM_COMPLETED, "item": {
                "type": wire.AGENT_MESSAGE, "text": account}})
        if self.terminal is not None:
            if wire.event_type(self.terminal) == wire.TURN_COMPLETED:
                usage = self.stream.aggregate_usage()
                if usage:
                    self.terminal["usage"] = usage
            self.events.append(self.terminal)
        return self.events
