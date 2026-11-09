"""No real CLI: the server keeps working only while its input stays open."""

from collections import deque

import pytest

from llm_agents.codex import subagents
from llm_loop import appserver, parallel, providers, streamrender
from llm_loop.agentwork import AgentCommand
from _runfixtures import isolated_run


def turn(method, thread, turn_id, **fields):
    return {"method": method, "params": {
        "threadId": thread, "turn": {"id": turn_id, **fields}}}


def message(thread, turn_id, text):
    return {"method": "item/completed", "params": {
        "threadId": thread, "turnId": turn_id,
        "item": {"type": "agentMessage", "text": text}}}


class Server:
    def __init__(self, outcome):
        self.proc = self
        self.finished = False
        self.input_closed = False
        self.returncode = 0
        self.outcome = outcome
        self.calls = []
        self.events = deque([
            turn("turn/started", "child", "child-turn"),
            message("root", "first", "initial answer"),
            turn("turn/completed", "root", "first", status="completed"),
        ])
        if outcome != "timeout":
            self.events.extend([
                message("child", "child-turn", "late finding"),
                turn("turn/completed", "child", "child-turn", status="completed"),
            ])

    def take(self, deadline, what):
        assert not self.input_closed, "runner cut off a live child"
        if not self.events:
            self.finished = True
            raise appserver.Closed("fake server exited")
        return self.events.popleft()

    def resolve(self, event):
        return False

    def request(self, body, on_other=None):
        assert not self.input_closed
        self.calls.append(body)
        assert body["method"] == "turn/start"
        prompt = body["params"]["input"][0]["text"]
        assert "late finding" in prompt or "interrupted" in prompt
        if self.outcome == "rejected":
            raise appserver.RpcError("turn/start", "rejected follow-up")
        # Events can precede the response to turn/start.
        on_other(turn("turn/started", "root", "follow-up"))
        on_other(message("root", "follow-up", "combined answer"))
        if self.outcome == "failed":
            on_other(turn("turn/completed", "root", "follow-up", status="failed",
                          error={"message": "follow-up failed"}))
        else:
            on_other(turn("turn/completed", "root", "follow-up", status="completed"))
        return {"turn": {"id": "follow-up"}}

    def send_request(self, body):
        assert not self.input_closed
        self.calls.append(body)
        assert body["method"] == "turn/interrupt"
        self.events.append(turn("turn/completed", "child", "child-turn",
                                status="interrupted"))
        return len(self.calls)

    def close_input(self):
        self.input_closed = True

    def fail_pending(self, reason):
        pass

    def poll(self):
        return 0

    def wait(self, timeout=None):
        assert self.input_closed
        return 0


@pytest.mark.parametrize("runner", ["sequential", "raw", "parallel"])
@pytest.mark.parametrize("notes", [True, False])
@pytest.mark.parametrize("outcome", ["completed", "rejected", "failed", "timeout"])
def test_runners_wait_for_children(monkeypatch, tmp_path, capsys,
                                   runner, notes, outcome):
    with isolated_run(monkeypatch, tmp_path):
        providers.set_live_messages(notes)
        if outcome == "timeout":
            monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0)
        server = Server(outcome)
        proc = providers._CodexAppProcess(server, [], "root", "first")
        monkeypatch.setattr(streamrender, "start_agent_process", lambda *a: proc)
        monkeypatch.setattr(parallel, "start_agent_process", lambda *a: proc)
        if runner == "parallel":
            rc = parallel.run_job(1, AgentCommand("p", "", "job", "codex"))[0]
        else:
            rc = streamrender.run_agent_streaming(
                ["codex"], "codex", raw=runner == "raw")
        assert rc == 0
        assert server.input_closed
        assert sum(c["method"] == "turn/start" for c in server.calls) == 1
        assert sum(c["method"] == "turn/interrupt" for c in server.calls) == (
            1 if outcome == "timeout" else 0)
        output = capsys.readouterr().out
        if runner != "parallel":
            expected = {"rejected": "rejected follow-up", "failed": "follow-up failed"}
            assert expected.get(outcome, "combined answer") in output
        if outcome in ("rejected", "failed"):
            assert proc.stdout._wait.text == "initial answer"
            assert "combined answer" not in output


def test_notes_disabled_still_uses_controllable_transport(monkeypatch, tmp_path):
    with isolated_run(monkeypatch, tmp_path):
        providers.set_live_messages(False)
        argv = providers.build_agent_argv(
            AgentCommand("p", "model", sandbox_mode="read-only"), "codex", "/repo")
        assert argv == ["codex", "app-server", "--stdio"]
        assert argv.model == "model"
        assert argv.sandbox_mode == "read-only"


def test_nested_child_finishing_after_its_parent_is_in_follow_up():
    server = Server("completed")
    server.events.insert(3, turn("turn/started", "grandchild", "grandchild-turn"))
    server.events.extend([
        message("grandchild", "grandchild-turn", "nested finding"),
        turn("turn/completed", "grandchild", "grandchild-turn", status="completed"),
    ])
    proc = providers._CodexAppProcess(server, [], "root", "first")
    events = list(proc.stdout)
    prompt = server.calls[0]["params"]["input"][0]["text"]
    assert "late finding" in prompt and "nested finding" in prompt
    assert sum(e.get("type") == "turn.completed" for e in events) == 1
    assert server.input_closed


def test_server_death_during_child_wait_keeps_answer_and_names_cut_child():
    server = Server("completed")
    server.events = deque(list(server.events)[:3])
    proc = providers._CodexAppProcess(server, [], "root", "first")
    events = list(proc.stdout)
    assert events[-1]["type"] == "turn.completed"
    assert proc.stdout._wait.text == "initial answer"
    assert len(proc.stdout._wait.sub_agents.cut) == 1
    assert server.calls == []
