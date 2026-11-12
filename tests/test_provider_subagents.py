"""No real CLI: the server keeps working only while its input stays open."""

from collections import deque
import time

import pytest

from llm_agents.codex import subagents
from llm_loop import appserver, parallel, providers, streamrender, statusline
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
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(what)
        if not self.events:
            self.finished = True
            raise appserver.Closed("fake server exited")
        return self.events.popleft()

    def resolve(self, event):
        return False

    def request(self, body, on_other=None, on_poll=None):
        if on_poll:
            on_poll()
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


def test_initial_answer_precedes_wait_and_last_message_is_answer():
    server = Server("failed")
    proc = providers._CodexAppProcess(server, [], "root", "first")
    initial = next(proc.stdout)
    assert initial["item"]["text"] == "initial answer"
    assert not server.calls
    assert server.events[0]["method"] == "turn/completed"
    events = list(proc.stdout)
    answers = [e["item"]["text"] for e in events
               if e.get("item", {}).get("type") == "agent_message"]
    assert answers[-1].startswith("initial answer\n")
    assert "late finding" in answers[-1]


@pytest.mark.parametrize("runner", ["sequential", "raw", "parallel"])
def test_progress_is_immediate_in_row_output_and_log(monkeypatch, tmp_path, capsys, runner):
    from llm_loop import diaglog
    records = []
    monkeypatch.setattr(diaglog, "record", lambda *args: records.append(args))
    with isolated_run(monkeypatch, tmp_path):
        server = Server("completed")
        proc = providers._CodexAppProcess(server, [], "root", "first")
        # The handle the runners bind, on a disabled app: no painter, so its
        # posts run at once and `snapshot()` reads them.
        job = statusline.StatusApp(enabled=False).job(1)
        job.start(item="work")
        original = server.take
        checked = False

        def take(deadline, what):
            nonlocal checked
            if not checked and deadline:
                checked = True
                assert "waiting" in job.snapshot().progress
                assert "waiting" in capsys.readouterr().out
                assert any("waiting" in str(record) for record in records)
            return original(deadline, what)

        server.take = take
        monkeypatch.setattr(streamrender, "start_agent_process", lambda *a: proc)
        monkeypatch.setattr(parallel, "start_agent_process", lambda *a: proc)
        with statusline.describing(job):
            if runner == "parallel":
                parallel.run_job(1, AgentCommand("p", "", "job", "codex"))
            else:
                streamrender.run_agent_streaming(["codex"], "codex", raw=runner == "raw")
        assert any("follow_up" in str(record) for record in records)
        assert any('"phase": "end"' in str(record) for record in records)


def test_stop_during_wait_cuts_children_without_follow_up(monkeypatch):
    server = Server("timeout")
    proc = providers._CodexAppProcess(server, [], "root", "first")
    stopped = False
    def progress(event):
        nonlocal stopped
        stopped = True
    proc.stdout.progress = progress
    proc.stdout.stop_requested = lambda: stopped
    events = list(proc.stdout)
    assert [c["method"] for c in server.calls] == ["turn/interrupt"]
    assert proc.stdout._wait.sub_agents.cut
    assert events[-1]["type"] == "turn.completed"
    assert "initial answer" in events[-2]["item"]["text"]


@pytest.mark.parametrize("outcome", ["failed", "rejected", "request_timeout"])
def test_failed_follow_up_keeps_command_and_file_events(outcome):
    server = Server(outcome)
    original = server.request
    def request(body, on_other=None, **kwargs):
        on_other(turn("turn/started", "root", "follow-up"))
        for item in ({"type": "commandExecution", "command": "test", "exitCode": 1},
                     {"type": "fileChange", "changes": [{"path": "edited.py", "kind": "update"}]}):
            on_other({"method": "item/completed", "params": {
                "threadId": "root", "turnId": "follow-up", "item": item}})
        if outcome == "request_timeout":
            raise TimeoutError("turn/start timed out")
        return original(body, on_other=on_other, **kwargs)
    server.request = request
    proc = providers._CodexAppProcess(server, [], "root", "first")
    events = list(proc.stdout)
    kinds = [e.get("item", {}).get("type") for e in events]
    assert "command_execution" in kinds and "file_change" in kinds
    assert all("combined answer" not in str(e) for e in events)


def test_loop_child_text_is_not_truncated():
    server = Server("rejected")
    whole = "late finding" + "x" * (subagents.PROMPT_CHARS + 100)
    server.events[3] = message("child", "child-turn", whole)
    proc = providers._CodexAppProcess(server, [], "root", "first")
    events = list(proc.stdout)
    assert whole in server.calls[0]["params"]["input"][0]["text"]
    assert whole in events[-2]["item"]["text"]
    assert "logs loop" not in str(events)


def test_continuous_notifications_cannot_extend_request_deadline(monkeypatch):
    from types import SimpleNamespace
    clock = [0.0]
    monkeypatch.setattr(appserver.time, "monotonic", lambda: clock[0])
    pending = appserver.Pending(1, "turn/start")
    forgotten = []
    def take(deadline, what):
        clock[0] += 1
        assert clock[0] < 20, "request ignored its monotonic deadline"
        return {"method": "notification"}
    connection = SimpleNamespace(start_request=lambda body: pending,
                                 take=take, resolve=lambda item: False,
                                 forget=forgotten.append)
    with pytest.raises(TimeoutError):
        appserver.Connection.request(connection, {}, timeout=3, on_other=lambda e: None)
    assert clock[0] == 3
    assert forgotten == [pending]


def test_idle_deadline_cuts_child_with_fake_clock(monkeypatch):
    clock = [1.0]
    monkeypatch.setattr(appserver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 2)
    server = Server("timeout")
    original = server.take
    deadlines = []
    def take(deadline, what):
        if not server.events:
            deadlines.append(deadline)
            clock[0] = deadline
            raise TimeoutError(what)
        return original(deadline, what)
    server.take = take
    proc = providers._CodexAppProcess(server, [], "root", "first")
    list(proc.stdout)
    assert deadlines and max(deadlines) == 3.0
    assert proc.stdout._wait.sub_agents.cut
    assert proc.stdout._wait.sub_agents.wait_s == 2.0


@pytest.mark.parametrize("accepted", [True, False])
def test_note_typed_during_wait_is_acknowledged(monkeypatch, accepted):
    from llm_loop import operator
    server = Server("completed")
    proc = providers._CodexAppProcess(server, [], "root", "first")
    pending = appserver.Pending(90, "turn/steer")
    sent = []
    def start_request(body):
        sent.append(body)
        response = {"result": {}} if accepted else {"error": {"message": "no active turn"}}
        server.events.appendleft({"id": 90, **response})
        return pending
    def resolve(event):
        if event.get("id") == 90:
            pending.settle(event)
            return True
        return False
    server.start_request = start_request
    server.resolve = resolve
    def wait(self, timeout):
        # Model the reader servicing the acknowledgement while the note sender
        # waits, without starting a thread or sleeping.
        proc.stdout._wait.pump(time.monotonic() + timeout, proc.stdout._wait.on_event)
        return self.done
    monkeypatch.setattr(appserver.Pending, "wait", wait)
    mailbox = operator.Mailbox()
    channel = operator.AgentChannel(proc.stdin, mailbox)
    def progress(event):
        if event["phase"] == "waiting":
            delivery = mailbox.submit("include the late finding")
            assert delivery.live == accepted
            assert pending.done
    proc.stdout.progress = progress
    list(proc.stdout)
    assert sent[0]["method"] == "turn/steer"
    assert sent[0]["params"]["expectedTurnId"] == "first"
    assert mailbox.take_queued() == ([] if accepted else ["include the late finding"])
    channel.close()


@pytest.mark.parametrize("source", ["key", "file"])
def test_bound_run_stop_is_seen_during_idle_poll(monkeypatch, tmp_path, source):
    from types import SimpleNamespace
    clock = [1.0]
    monkeypatch.setattr(appserver.time, "monotonic", lambda: clock[0])
    app = SimpleNamespace(stop_requested_here=False, stop_file=str(tmp_path / "stop"))
    server = Server("timeout")
    original = server.take
    def take(deadline, what):
        if not server.events:
            if source == "key":
                app.stop_requested_here = True
            else:
                (tmp_path / "stop").touch()
            clock[0] = deadline
            raise TimeoutError(what)
        return original(deadline, what)
    server.take = take
    row = statusline.StatusApp(enabled=False).job(1)
    with statusline.describing(row, app):
        proc = providers._CodexAppProcess(server, [], "root", "first")
        list(proc.stdout)
    assert [c["method"] for c in server.calls] == ["turn/interrupt"]
    assert proc.stdout._wait.sub_agents.cut
    assert clock[0] < 2


@pytest.mark.parametrize("phase", ["waiting", "cut", "end"])
def test_raising_progress_does_not_skip_interrupts_or_account(monkeypatch, phase):
    from llm_loop import diaglog
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0)
    logged = []
    monkeypatch.setattr(diaglog, "record", lambda *args: logged.append(args))
    server = Server("timeout")
    proc = providers._CodexAppProcess(server, [], "root", "first")
    calls = []
    def progress(event):
        calls.append(event["phase"])
        if event["phase"] == phase:
            raise OSError("display unavailable")
    proc.stdout.progress = progress
    events = list(proc.stdout)
    assert any(call["method"] == "turn/interrupt" for call in server.calls)
    assert proc.stdout._wait.sub_agents.cut
    assert events[-1]["type"] == "turn.completed"
    assert "CUT" in events[-2]["item"]["text"]
    assert calls[-1] == phase
    assert logged == [("codex.sub_agents.progress_failed", "display unavailable")]


def test_sequential_progress_runs_on_console_owner(monkeypatch, tmp_path):
    import sys
    import threading
    from llm_loop import console, ownership
    with isolated_run(monkeypatch, tmp_path):
        server = Server("completed")
        proc = providers._CodexAppProcess(server, [], "root", "first")
        monkeypatch.setattr(streamrender, "start_agent_process", lambda *a: proc)
        writes = []
        original = sys.stdout
        class RecordingOutput:
            def write(self, text):
                if '"_meta": "subagents"' in text:
                    writes.append(threading.current_thread().name)
                return original.write(text)

            def __getattr__(self, name):
                return getattr(original, name)
        monkeypatch.setattr(sys, "stdout", RecordingOutput())
        owner = ownership.OwnerThread("wait-progress-owner").start()
        try:
            with console.route_through(owner, post_timeout=10.0):
                streamrender.run_agent_streaming(["codex"], "codex", raw=True)
                # Same handshake bound as test_output_owner: 0.49 s measured
                # for that file on 2026-09-25, with ample scheduling headroom.
                assert owner.drain(10.0)
        finally:
            owner.close(10.0)
        assert writes and set(writes) == {"wait-progress-owner"}


@pytest.mark.parametrize("when", ["before_response", "unknown_id", "after_response"])
@pytest.mark.parametrize("responds", [True, False])
@pytest.mark.parametrize("continuation", [False, True])
def test_stop_interrupts_dispatched_follow_up(monkeypatch, when, responds, continuation):
    clock = [1.0]
    monkeypatch.setattr(appserver.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(subagents, "SUB_AGENT_INTERRUPT_S", 2.0)
    server = Server("completed")
    if continuation:
        server.events.insert(3, turn("turn/started", "root", "continuation"))
        server.events.insert(4, message("root", "continuation", "automatic continuation"))
        server.events.insert(5, turn("turn/completed", "root", "continuation", status="completed"))
    stopped = [False]
    pending = {}
    requested = [False]
    real_take = server.take

    def item(kind, **fields):
        return {"method": "item/completed", "params": {
            "threadId": "root", "turnId": "follow-up", "item": {"type": kind, **fields}}}

    def start_request(body):
        server.calls.append(body)
        requested[0] = True
        reply = appserver.Pending(99, body["method"])
        pending[99] = reply
        start = [turn("turn/started", "root", "follow-up"),
                 turn("turn/started", "new-child", "new-child-turn"),
                 item("commandExecution", command="before stop", exitCode=0)]
        response = {"id": 99, "result": {"turn": {"id": "follow-up"}}}
        stop = {"method": "test/stop"}
        if when == "unknown_id":
            server.events.extend([stop, response, *start])
        else:
            server.events.extend([*start, *([stop, response] if when == "before_response"
                                          else [response, stop])])
        return reply

    def resolve(event):
        reply = pending.pop(appserver.response_id(event), None)
        if reply:
            reply.settle(event)
        return reply is not None

    def take(deadline, what):
        if server.events:
            event = real_take(deadline, what)
            if event.get("method") == "test/stop":
                stopped[0] = True
            return event
        assert requested[0]
        clock[0] = deadline
        raise TimeoutError(what)

    def send_request(body):
        server.calls.append(body)
        thread = body["params"]["threadId"]
        if thread == "root":
            server.events.extend([
                item("fileChange", changes=[{"path": "during-stop.py", "kind": "update"}]),
                message("root", "follow-up", "partial follow-up")])
        if responds or thread != "root":
            server.events.append(turn("turn/completed", thread,
                                      body["params"]["turnId"], status="interrupted"))
        return len(server.calls)

    server.start_request = start_request
    server.forget = lambda reply: pending.pop(reply.id, None)
    server.resolve = resolve
    server.take = take
    server.send_request = send_request
    server.request = lambda body, **kwargs: appserver.Connection.request(server, body, **kwargs)
    proc = providers._CodexAppProcess(server, [], "root", "first")
    proc.stdout.stop_requested = lambda: stopped[0]
    events = list(proc.stdout)
    interrupts = [call["params"] for call in server.calls if call["method"] == "turn/interrupt"]
    assert {"threadId": "root", "turnId": "follow-up"} in interrupts
    assert {"threadId": "new-child", "turnId": "new-child-turn"} in interrupts
    assert any(fate.thread_id == "new-child" for fate in proc.stdout._wait.sub_agents.cut)
    kinds = [event.get("item", {}).get("type") for event in events]
    assert "command_execution" in kinds and "file_change" in kinds
    assert "partial follow-up" not in str(events)
    assert events[-2]["item"]["text"].startswith("initial answer\n")
    assert events[-1]["type"] == "turn.completed"
    assert clock[0] <= 3.0
    assert not pending


def test_wait_stream_does_not_expose_provider_storage():
    proc = providers._CodexAppProcess(Server("completed"), [], "root", "first")
    assert not hasattr(proc.stdout, "owner")
    assert not hasattr(proc.stdout, "usage_by_thread")
    proc.stdout.follow_turn("next")
    assert proc.turn_id == proc.stdout.turn_id == "next"
    proc.stdout.observe_usage({"method": "thread/tokenUsage/updated", "params": {
        "threadId": "root", "tokenUsage": {"total": {"inputTokens": 9}}}})
    usage = proc.stdout.aggregate_usage()
    usage["input_tokens"] = 1000
    assert proc.stdout.aggregate_usage()["input_tokens"] == 9


def test_wait_stream_deadline_bounds_buffered_events(monkeypatch):
    monkeypatch.setattr(appserver.time, "monotonic", lambda: 3.0)
    proc = providers._CodexAppProcess(Server("completed"), [{"method": "buffered"}], "root", "first")
    with pytest.raises(TimeoutError):
        proc.stdout.next_event(3.0)
