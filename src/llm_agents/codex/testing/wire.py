"""The offline wire: a stand-in `codex app-server` and the helpers that read it.

The real `codex` CLI is never started. What talks to the client instead is
`fake_app_server.py` beside this module, launched as a REAL child process, so
a wire test exercises the actual pipes, encoding and line buffering rather
than a mock's idea of them. Every message the stand-in reads is appended to
`received.jsonl` in the test's directory, which is how a test sees what the
CLIENT put on the wire (the client discards most responses, so asserting on
its return values alone would miss the request side entirely).

This repository's tests and a host's tests import the same helpers from
here, so a host pins its own policy against the very wire the core is pinned
against. Standard library only: `monkeypatch` is pytest's fixture, passed in;
the fixtures themselves live in `fixtures` beside this module.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

from .. import gates, paths, protocol, turn
from ..registry import AgentRecord

# The stand-in's source: a script, not a module of this package. It is run
# by the interpreter under test from a copy in the test's directory
# (`install_fake_server`); never imported, since it reads its argv and stdin
# at the top level.
FAKE_APP_SERVER_PATH = Path(__file__).with_name("fake_app_server.py")
FAKE_APP_SERVER = FAKE_APP_SERVER_PATH.read_text(encoding="utf-8")

# The outage the retry rule was written for, copied verbatim out of
# worker-e's `events.jsonl` (2026-09-13) rather than written
# from memory — a shape invented here would pin nothing but itself. Three lines
# of one failure: the server reconnecting (`willRetry`, and the kind in its
# object spelling), the error that ends the turn (the kind degraded to the
# catch-all `"other"`, the sentence kept), and, from an earlier day, the other
# service failure these logs have seen. The stand-in is handed the drop's
# message on its command line, so both halves of the pin are this one string.
LOGGED_DROP = json.loads(
    '{"method": "error", "params": {"error": {"message": "stream disconnected '
    'before completion: Unable to verify model access right now. Please '
    'retry.", "codexErrorInfo": "other", "additionalDetails": null, '
    '"misalignment": null}, "willRetry": false, "threadId": '
    '"01a09779-2ded-7041-aaef-b795471eb850", "turnId": '
    '"01a09779-30f7-7850-850c-5c4f54018dd8"}, "emittedAtMs": 1789247876163}')
LOGGED_RECONNECT = json.loads(
    '{"method": "error", "params": {"error": {"message": "Reconnecting... 5/5",'
    ' "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": '
    'null}}, "additionalDetails": "stream disconnected before completion: '
    'Unable to verify model access right now. Please retry.", "misalignment": '
    'null}, "willRetry": true, "threadId": '
    '"01a09779-2ded-7041-aaef-b795471eb850", "turnId": '
    '"01a09779-30f7-7850-850c-5c4f54018dd8"}, "emittedAtMs": 1789247871949}')
LOGGED_OVERLOAD = json.loads(
    '{"method": "error", "params": {"error": {"message": "Selected model is at '
    'capacity. Please try a different model.", "codexErrorInfo": '
    '"serverOverloaded", "additionalDetails": null, "misalignment": null}, '
    '"willRetry": false, "threadId": "01a05fa3-c011-7a50-8a1f-e4ff435224da", '
    '"turnId": "01a05fa3-c273-7d51-9731-b4433f5629bd"}, "emittedAtMs": '
    "1788311174392}")
DROPPED_TURN_MESSAGE = LOGGED_DROP["params"]["error"]["message"]

# worker-c/events.jsonl, 2026-11-07: the steady part of the network outage.
# Keep the wire shape and emittedAtMs cadence; the test clock replays minutes
# instantly, without turning CPU scheduling delays into test failures.
LOGGED_NETWORK_WAIT = json.loads(
    '{"method": "error", "params": {"error": {"message": "Reconnecting... '
    'waiting for network", "codexErrorInfo": {"responseStreamDisconnected": '
    '{"httpStatusCode": null}}, "additionalDetails": "Connection failed: error '
    'sending request", "misalignment": null}, "willRetry": true, "threadId": '
    '"01a1b4cc-9df0-7261-9030-864b6d84b697", "turnId": '
    '"01a1b4cc-a550-79f2-a90f-acb4d804d0c5"}, "emittedAtMs": 1794038271329}')
NETWORK_WAIT_TIMES = [1794038271329, 1794038334670, 1794038398202,
                      1794038461795, 1794038525197]


def fake_server_argv(tmp_path: Path, scenario: str) -> list:
    """The command line that runs the stand-in playing `scenario`, from a copy
    in `tmp_path` (where it also writes `received.jsonl`)."""
    script = tmp_path / "fake_app_server.py"
    script.write_text(FAKE_APP_SERVER, encoding="utf-8")
    return [sys.executable, str(script), scenario, str(tmp_path),
            DROPPED_TURN_MESSAGE,
            json.dumps([LOGGED_NETWORK_WAIT, NETWORK_WAIT_TIMES])]


def install_fake_server(monkeypatch: Any, tmp_path: Path, scenario: str) -> None:
    """Point AppServer at the stand-in instead of the real CLI.

    Only the command line is swapped (`protocol._server_argv`): the child is
    still started, read and ended by the real transport, `llm_loop.appserver`,
    which the loop's provider process shares.
    """
    argv = fake_server_argv(tmp_path, scenario)
    monkeypatch.setattr(protocol, "_server_argv", lambda: argv)


def received(tmp_path: Path) -> list:
    """Everything the client sent, in order."""
    path = tmp_path / "received.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def first_request(tmp_path: Path, method: str) -> dict:
    """The first message the client sent with `method`; an AssertionError
    naming the method when it never sent one."""
    for message in received(tmp_path):
        if message.get("method") == method:
            return message
    raise AssertionError(f"the client never sent {method!r}")


def answers(tmp_path: Path) -> dict:
    """Client replies to server-initiated requests, keyed by request id."""
    return {message["id"]: message for message in received(tmp_path)
            if "method" not in message and message.get("id") is not None}


def turn_starts(tmp_path: Path) -> list:
    """Every `turn/start` the client sent: one per turn, follow-ups included."""
    return [message for message in received(tmp_path)
            if message.get("method") == "turn/start"]


def run_fake_turn(tmp_path: Path, monkeypatch: Any, scenario: str, *, name: str = "w",
                  escalate: str = "", sandbox: str = "read-only", role: str = "",
                  cwd: Optional[Path] = None, **kwargs):
    """One `turn.run_turn` against the stand-in playing `scenario`, the
    agent's state under `tmp_path`; `kwargs` go to `run_turn` (a 30 s
    `timeout` unless given)."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    install_fake_server(monkeypatch, tmp_path, scenario)
    record = AgentRecord(name=name, cwd=str(cwd or tmp_path), sandbox=sandbox,
                         escalate=escalate, role=role)
    kwargs.setdefault("timeout", 30.0)
    return turn.run_turn(record=record, prompt="do the thing",
                         resume=False, **kwargs)


# --- gates the wrapper runs itself ------------------------------------------

# A prompt whose `## Gates` section holds GATES, among fenced blocks that are
# not gates: one before the section, one in a section after it.
GATES_PROMPT = """Task: convert X to Y.

## Context
```
npm run not-a-gate
```

## Gates

Run these from the usual place:

```
npm run typecheck
# a comment, not a command

python tools/check/rename_gate.py oldName
```

### Notes under the gates
```powershell
npm test -- test/hash.test.ts
```

## Scope
```
npm run also-not-a-gate
```
"""

GATES = ["npm run typecheck", "python tools/check/rename_gate.py oldName",
         "npm test -- test/hash.test.ts"]


class GateRunner:
    """Stands in for `gates.run_gate`: every call is recorded, and each
    command answers from its own queue (the last answer repeats)."""

    def __init__(self, answers):
        self.answers = {command: list(codes) for command, codes in answers.items()}
        self.calls = []

    def __call__(self, command, cwd, timeout):
        self.calls.append((command, cwd))
        codes = self.answers[command]
        code = codes.pop(0) if len(codes) > 1 else codes[0]
        output = "all good" if code == 0 else "line\n" * 60 + "FAILED boom"
        return gates.GateRun(command, str(cwd), code, output, 1.0)


def gated_turn(tmp_path: Path, monkeypatch: Any, scenario: str, runner, commands=GATES,
               record: Optional[AgentRecord] = None, expect: str = ""):
    """A turn against the stand-in whose record carries `commands` as its
    gates, run by `runner` (a `GateRunner`); a read-only agent "w" unless
    `record` is given."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    install_fake_server(monkeypatch, tmp_path, scenario)
    if record is None:
        record = AgentRecord(name="w", cwd=str(tmp_path), sandbox="read-only")
    record.gates = list(commands)
    return turn.run_turn(record=record, prompt="do the thing", resume=False,
                         timeout=30.0, run_gate=runner, expect=expect)
