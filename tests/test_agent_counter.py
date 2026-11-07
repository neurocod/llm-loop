"""The badge client (`llm_agents.agent_counter`) with no counter built.

Every test here replaces the pipe, so none needs the optional
`extras/agent-counter` executable or Qt; the round trips against a built exe
belong to whoever builds it.
"""

import json
import os

import pytest

from llm_agents import agent_counter, host_policy
from llm_agents.host_policy import HostPolicy

# `announce` is a no-op off Windows (the counter speaks a Windows named pipe),
# so a test that reads what it wrote has nothing to read there.
windows_only = pytest.mark.skipif(os.name != "nt", reason="announce is a no-op off Windows")


class _Recorder:
    """A counter handle that keeps every line written to it."""

    def __init__(self, messages):
        self.messages = messages

    def write(self, data):
        self.messages.append(json.loads(data))

    def close(self):
        pass


@pytest.fixture
def recorded(monkeypatch):
    messages = []
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "1")
    monkeypatch.setenv("CLAUDE_AGENTS_COUNTER", "1")
    monkeypatch.setattr(agent_counter, "_open_or_start", lambda: _Recorder(messages))
    monkeypatch.setattr(agent_counter, "_week_complained", set())
    return messages


@windows_only
def test_the_weekly_remainder_rides_on_a_repeated_hello(recorded):
    """Used in, left out; floored after float noise; clamped; and only a
    CHANGED remainder is written (Codex repeats its report every 10-20 s)."""
    hello = {"op": "hello", "label": "claude:w", "agent_type": "claude"}
    with agent_counter.announce("claude:w", agent_type="claude") as beacon:
        for used in (18.0, 18.0, 18.5, 0.55 * 100, 150.0, -5.0):
            beacon.report_week_used(used)
    assert recorded == [hello] + [dict(hello, week_left=left) for left in (82, 81, 45, 0, 100)]


@windows_only
def test_no_weekly_figure_sends_nothing_and_says_so_once(recorded, capsys):
    for _ in range(2):
        with agent_counter.announce("claude:w", agent_type="claude") as beacon:
            beacon.report_week_used(None, "the event had no window")
    assert all("week_left" not in message for message in recorded)
    lines = [line for line in capsys.readouterr().err.splitlines() if "weekly-limit" in line]
    assert lines == ["agent_counter: no weekly-limit figure for claude "
                     "(the event had no window); the badge shows the count alone"]


@windows_only
def test_a_shown_figure_survives_a_later_report_without_one(recorded, capsys):
    with agent_counter.announce("codex:w", agent_type="codex") as beacon:
        beacon.report_week_used(12.0)
        beacon.report_week_used(None, "no long window")
    assert [message.get("week_left") for message in recorded] == [None, 88]
    assert "weekly-limit" not in capsys.readouterr().err


def test_a_disabled_counter_takes_reports_without_touching_the_wire(monkeypatch, capsys):
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "0")

    def explode():
        raise AssertionError("a disabled counter opened the pipe")

    monkeypatch.setattr(agent_counter, "_open_or_start", explode)
    with agent_counter.announce("claude:w", agent_type="claude") as beacon:
        beacon.report_week_used(10.0)
    assert "weekly-limit" not in capsys.readouterr().err


def test_an_unbuilt_counter_is_said_once_and_never_started(tmp_path, monkeypatch, capsys):
    """The badge is optional: a host that names an exe it never built gets no
    badge, no exception and no spawn — and ONE line saying why, however many
    turns the process runs. Called at `_open_or_start`, the step `announce`
    takes on Windows, so the pin holds on every OS."""
    monkeypatch.delenv(agent_counter.EXE_ENV, raising=False)
    monkeypatch.setattr(agent_counter, "_complained", False)
    monkeypatch.setattr(agent_counter, "_connect", lambda: None)
    spawned = []
    monkeypatch.setattr(agent_counter, "_spawn", spawned.append)
    with host_policy.installed(HostPolicy(counter_exes=("bin/counter.exe",)), tmp_path):
        assert [agent_counter._open_or_start() for _ in range(3)] == [None] * 3
    assert spawned == []
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("agent_counter:")]
    assert lines == ["agent_counter: agent counter unavailable (counter.exe is not built)"]
