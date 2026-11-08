"""`llm_loop.appserver`: the JSON-RPC transport both Codex app-server clients
stand on — the loop's provider process (`providers`) and the named worker
(`llm_agents.codex.protocol`).

The wire tests drive BOTH clients against the same stand-in server
(`llm_agents.codex.testing`), started as a real child, so a fix to the
transport is pinned once for the two of them.
"""

import io
import json
import subprocess
import sys
import time

import pytest

from llm_loop import appserver, procend, providers
from llm_loop.agentwork import AgentCommand
from llm_agents.codex import protocol
from llm_agents.codex.testing import wire as fake
from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401


class _Lines:
    """A stand-in process: stdout replays `items`, stdin is recorded."""

    class _Input(io.StringIO):
        def close(self):
            self.was_closed = True

    def __init__(self, *items):
        self.stdin = self._Input()
        self.stdout = io.StringIO("".join(
            item if isinstance(item, str) else json.dumps(item) + "\n"
            for item in items))
        self.returncode = 0

    def poll(self):
        return self.returncode

    def sent(self):
        return [json.loads(line) for line in self.stdin.getvalue().splitlines()]


# --- the transport on its own ------------------------------------------------


def test_a_server_request_carrying_the_awaited_id_is_not_the_reply():
    """The server numbers its own requests: one with the client's id is a
    question routed onwards, and the reply is the message WITHOUT a method."""
    question = {"jsonrpc": "2.0", "id": 0, "method": "item/tool/requestUserInput",
                "params": {}}
    connection = appserver.Connection(_Lines(
        question, {"id": 0, "result": {"answer": 42}}))
    seen = []

    result = connection.request(appserver.message("initialize"), 5.0,
                                on_other=seen.append)

    assert result == {"answer": 42}
    assert seen == [question]


def test_a_reply_settles_only_its_own_request():
    """Ids count from `first_id`; a reply to an id nobody awaits is stale."""
    connection = appserver.Connection(_Lines(
        {"id": 6, "result": {"stale": True}}, {"id": 7, "result": {"mine": True}}),
        first_id=7)
    seen = []

    assert connection.request(appserver.message("thread/start"), 5.0,
                              on_other=seen.append) == {"mine": True}
    assert seen == [{"id": 6, "result": {"stale": True}}]
    assert connection.proc.sent()[0]["id"] == 7


def test_an_error_reply_names_the_request_and_the_server_s_words():
    connection = appserver.Connection(_Lines(
        {"id": 0, "error": {"code": -32600, "message": "input too large"}}))

    with pytest.raises(appserver.RpcError, match="^turn/start failed: input too large$"):
        connection.request(appserver.message("turn/start"), 5.0)


def test_lines_that_are_not_json_objects_arrive_as_written():
    connection = appserver.Connection(_Lines(
        "npm warn deprecated\n", "[1]\n", "\n", {"method": "x"}))

    items = [connection.take(time.monotonic() + 5, "test") for _ in range(4)]

    assert items == ["npm warn deprecated\n", "[1]\n", "\n", {"method": "x"}]


def test_the_end_of_the_stream_is_seen_by_every_later_wait():
    connection = appserver.Connection(_Lines())

    for _ in range(2):
        with pytest.raises(appserver.Closed, match="during turn"):
            connection.take(time.monotonic() + 5, "turn")
    assert connection.finished


def test_a_write_after_the_input_closed_is_refused():
    connection = appserver.Connection(_Lines())
    connection.close_input()
    connection.close_input()  # idempotent

    with pytest.raises(appserver.InputClosed):
        connection.start_request(appserver.message("turn/steer"))
    assert connection.proc.stdin.was_closed


def test_a_silent_server_times_out_and_its_death_carries_its_stderr(tmp_path):
    """A real child: the wait is bounded, and once the child dies its last
    stderr lines name the cause."""
    script = ("import sys, time\n"
              "sys.stdin.readline()\n"
              "print('auth expired', file=sys.stderr, flush=True)\n"
              "time.sleep(1)\n")
    connection = appserver.Connection(appserver.spawn([sys.executable, "-c", script]))
    try:
        with pytest.raises(TimeoutError, match="did not answer initialize in time"):
            connection.request(appserver.message("initialize"), 0.3)
        with pytest.raises(appserver.Closed, match=r"\(stderr: auth expired\)"):
            # 1 s sleep + interpreter exit; 15 s bounds a loaded machine.
            connection.take(time.monotonic() + 15, "turn")
    finally:
        connection.close()


# --- ending the child (`procend.end_unless_exited`, `Connection.close`) -------


class _Dead:
    """A handle whose every OS call fails: the process is long gone and the
    OS has reclaimed it (Windows raises OSError from `wait` then). `pid` -1
    names no process, so `taskkill` fails and the escalation reaches the
    handle."""

    pid = -1

    def __init__(self, exited=None, poll_fails=False):
        self.exited = exited
        self.poll_fails = poll_fails
        self.calls = []

    def poll(self):
        self.calls.append("poll")
        if self.poll_fails:
            raise OSError("gone")
        return self.exited

    def terminate(self):
        self.calls.append("terminate")
        raise OSError("gone")

    def kill(self):
        self.calls.append("kill")
        raise OSError("gone")

    def wait(self, timeout=None):
        self.calls.append("wait")
        raise OSError("gone")


def test_close_ends_a_live_child_through_procend(monkeypatch):
    ended = []
    real = procend.end_process_tree
    monkeypatch.setattr(procend, "end_process_tree",
                        lambda proc: (ended.append(proc), real(proc)))
    child = appserver.spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        appserver.Connection(child).close()
        # Bounded by procend: ask + wait + kill + wait, REAP_GRACE_S each.
        assert child.wait(timeout=4 * procend.REAP_GRACE_S + 5) is not None
        assert ended == [child]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_an_exited_child_is_left_alone(monkeypatch):
    ended = []
    monkeypatch.setattr(procend, "end_process_tree", ended.append)
    done = _Dead(exited=0)
    procend.end_unless_exited(done)
    assert ended == [] and done.calls == ["poll"]


@pytest.mark.parametrize("poll_fails", [False, True])
def test_ending_never_raises_on_a_handle_the_os_took_back(poll_fails):
    dead = _Dead(poll_fails=poll_fails)
    procend.end_unless_exited(dead)  # every call raises OSError; none escapes
    if poll_fails:
        assert dead.calls == ["poll"]
    else:
        assert dead.calls[:2] == ["poll", "terminate"]
        assert "wait" in dead.calls


# --- both clients against the stand-in server --------------------------------


@pytest.fixture
def loop_server(monkeypatch, tmp_path):
    """Start the loop's live-message Codex process against the stand-in."""
    monkeypatch.setattr(providers, "_LIVE_MESSAGES", True)

    def start(scenario):
        argv = fake.fake_server_argv(tmp_path, scenario)
        monkeypatch.setattr(providers, "runtime_argv", lambda *_: argv)
        command = AgentCommand("do the thing", "gpt-test")
        return providers.start_agent_process(
            providers.build_agent_argv(command, "codex", str(tmp_path)),
            "codex", "do the thing", str(tmp_path))
    return start


def _drain(proc):
    """The turn's events, its input closed at the completion as the runners
    do (`providers.note_channel`): the server then exits and the stream ends."""
    events = []
    try:
        for event in proc.stdout:
            events.append(event)
            if event.get("type") == "turn.completed":
                proc.stdin.close()
        return events
    finally:
        proc.stdin.close()
        providers.reap_agent_process(proc)


@pytest.mark.parametrize("scenario", ["loop-turn", "id-clash"])
def test_the_loop_opens_its_turn_and_streams_it(loop_server, tmp_path, scenario):
    """"id-clash": the server asks a question carrying the id of the pending
    `thread/start`; taken for the reply, it opened no thread."""
    events = _drain(loop_server(scenario))

    assert [event["type"] for event in events] == [
        "thread.started", "item.completed", "turn.completed"]
    assert events[1]["item"]["text"] == "fake answer"
    assert [message.get("method") for message in fake.received(tmp_path)] == [
        "initialize", "initialized", "thread/start", "turn/start"]


def test_the_loop_s_startup_is_bounded(loop_server, monkeypatch):
    """A CLI that never answers `initialize` used to hang the run for good."""
    monkeypatch.setattr(appserver, "STARTUP_TIMEOUT_S", 0.5)
    started = time.monotonic()

    with pytest.raises(TimeoutError, match="did not answer initialize"):
        loop_server("mute-initialize")
    # 0.5 s wait + procend's bounded ending (3 * REAP_GRACE_S at worst); 40 s
    # is the repo's fallback max(T + 10 s, 2T) over the former unmeasured 20 s.
    assert time.monotonic() - started < 40


@pytest.mark.parametrize("scenario", ["plain", "id-clash"])
def test_the_worker_runs_a_turn_on_the_same_transport(tmp_path, monkeypatch,
                                                      scenario):
    """"id-clash" as for the loop: taken for the reply, the server's question
    opened no thread."""
    outcome = fake.run_fake_turn(tmp_path, monkeypatch, scenario)

    assert outcome.status == "completed"
    assert outcome.text == "fake answer"
    requests = [message for message in fake.received(tmp_path)
                if "method" in message]
    assert [message["method"] for message in requests][:4] == [
        "initialize", "initialized", "thread/start", "thread/name/set"]
    assert [message.get("id") for message in requests][:4] == [1, None, 2, 3]


def test_the_worker_starts_the_codex_cli_s_app_server(monkeypatch):
    monkeypatch.setattr(protocol, "_codex_executable", lambda: "codex.cmd")
    assert protocol._server_argv() == ["codex.cmd", "app-server", "--stdio"]
