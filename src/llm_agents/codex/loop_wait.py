"""Adapt the loop provider's connection to the shared sub-agent wait phase."""

import time

from llm_loop import appserver, wire

from .service import NetworkWait
from .subagents import SubAgentAccount, SubAgentTracker
from .subagent_wait import SubAgentWait, await_main_and_sub_agents, settle_sub_agents


class LoopWait:
    """Keep terminal events private until children and the follow-up settle.

    The initial turn still streams normally. Only the phase after its answer
    is buffered, so neither renderer can close stdin on an intermediate
    completion or mistake a failed follow-up for a failed initial answer.
    """

    def __init__(self, stream):
        self.stream = stream
        self.owner = stream._owner
        self.subs = SubAgentTracker()
        self.subs.main_started()
        self.sub_agents = SubAgentAccount()
        self.network = NetworkWait()
        self.done = False
        self.terminal = None
        self.text = ""
        self.events = []

    @property
    def finished(self):
        return self.owner.connection.finished

    def pump(self, deadline, on_event):
        message = self.stream._next_item(deadline)
        if isinstance(message, dict):
            if not self.owner.connection.resolve(message):
                on_event(message)
        else:
            self.events.append(message)

    def request(self, method, params, *, on_event=None):
        result = self.owner.connection.request(
            appserver.message(method, params), on_other=on_event)
        turn_id = wire.codex_app_turn_id(result)
        if turn_id:
            self.owner.turn_id = turn_id
        return result

    def send_request(self, method, params=None):
        return self.owner.connection.send_request(appserver.message(method, params))

    def on_sub_agent_event(self, message):
        if wire.codex_app_from_sub_agent(message, self.owner.thread_id):
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
        child = wire.codex_app_from_sub_agent(message, self.owner.thread_id)
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
            self.stream._observe_usage(message)
            return None
        if continuing and method == wire.APP_TURN_STARTED and turn.get("id"):
            self.owner.turn_id = str(turn["id"])
            self.done = False
            self.subs.main_started()
        if not wire.codex_app_in_turn(message, self.owner.turn_id):
            return None
        self.network.observe(message, time.monotonic(), "", self.owner.thread_id)
        self.stream._observe_usage(message)
        if item:
            self.subs.on_record(item)
            if item.get("type") == "agentMessage":
                self.text = str(item.get("text") or "")
        event = wire.codex_app_event(message, wire.codex_app_usage_sum(
            self.stream._usage_by_thread.values()))
        if method == wire.APP_TURN_COMPLETED:
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
        del self.events[count:]

    def message(self):
        return self.text

    def finish(self):
        # Reuse the named worker's existing turn budget for the one follow-up.
        from .turn import DEFAULT_TURN_TIMEOUT_S

        wait = SubAgentWait(lambda event: None, "loop")
        try:
            await_main_and_sub_agents(
                self, self, {"threadId": self.owner.thread_id},
                DEFAULT_TURN_TIMEOUT_S, None, wait)
        finally:
            settle_sub_agents(self, self, wait)
        account = self.sub_agents.under_answer("", "loop") or self.sub_agents.block
        if account:
            self.events.append({"type": wire.ITEM_COMPLETED, "item": {
                "type": wire.AGENT_MESSAGE, "text": account}})
        if self.terminal is not None:
            if wire.event_type(self.terminal) == wire.TURN_COMPLETED:
                usage = wire.codex_app_usage_sum(
                    self.stream._usage_by_thread.values())
                if usage:
                    self.terminal["usage"] = usage
            self.events.append(self.terminal)
        return self.events
