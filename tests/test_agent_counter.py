"""The badge client (`llm_agents.agent_counter`) with no counter built.

Every test here replaces the pipe, so none needs the optional
`extras/agent-counter` executable or Qt; the round trips against a built exe
belong to whoever builds it. The text cross-checks against the C++ half read
its sources only, so this suite guards both halves of the wire.
"""

import json
import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from llm_agents import agent_counter, host_policy
from llm_agents.host_policy import HostPolicy

REGISTRY_CPP = Path(__file__).resolve().parents[1] / "extras" / "agent-counter" / "Registry.cpp"


def test_the_pipe_prefix_is_spelled_the_same_in_both_halves():
    """A disagreement is invisible at runtime: the client finds no server,
    starts one, and still cannot reach it, so nothing is ever counted."""
    assert f'"{agent_counter.PIPE_PREFIX}"' in REGISTRY_CPP.read_text(encoding="utf-8")


def test_the_week_left_field_is_spelled_the_same_in_both_halves():
    source = REGISTRY_CPP.read_text(encoding="utf-8")
    assert f'QStringLiteral("{agent_counter.WEEK_LEFT_FIELD}")' in source


def test_provider_vocabulary_matches_native_normalization():
    source = REGISTRY_CPP.read_text(encoding="utf-8")
    normalizer = re.search(r"QString normalizedAgentType\([^)]*\) \{(.*?)\n\}", source, re.S)
    assert normalizer is not None
    native_types = set(re.findall(r'QLatin1String\("([^"]+)"\)', normalizer.group(1)))
    assert native_types == set(agent_counter.KNOWN_AGENT_TYPES)

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


def _unavailable_lines(capsys):
    return [line for line in capsys.readouterr().err.splitlines()
            if line.startswith("agent_counter: agent counter unavailable")]


def test_the_once_holds_across_threads(monkeypatch, capsys):
    """`_complain` runs on a provider stream's thread as well as the turn's."""
    monkeypatch.setattr(agent_counter, "_complained", False)
    start = threading.Barrier(8)

    def complain():
        start.wait()
        agent_counter._complain("broken pipe")

    threads = [threading.Thread(target=complain) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert _unavailable_lines(capsys) == ["agent_counter: agent counter unavailable (broken pipe)"]


def test_a_missing_override_is_named_not_the_policys_exe(tmp_path, monkeypatch, capsys):
    missing = tmp_path / "nowhere" / "Counter.exe"
    monkeypatch.setenv(agent_counter.EXE_ENV, str(missing))
    monkeypatch.setattr(agent_counter, "_complained", False)
    monkeypatch.setattr(agent_counter, "_connect", lambda: None)
    with host_policy.installed(HostPolicy(counter_exes=("bin/counter.exe",)), tmp_path):
        assert agent_counter._open_or_start() is None
    assert _unavailable_lines(capsys) == [
        f"agent_counter: agent counter unavailable ({agent_counter.EXE_ENV}={missing} is not a file)"]


@pytest.mark.parametrize("prefix, spawns", [(agent_counter.PIPE_PREFIX, 1), ("old-name.v1.", 0)])
def test_an_exe_listening_elsewhere_is_never_started(tmp_path, monkeypatch, capsys, prefix, spawns):
    """A stale build would cost every turn `SPAWN_WAIT_S` of waiting in vain;
    it is asked its pipe name once per process and skipped with one line."""
    exe = tmp_path / "Counter.exe"
    exe.write_bytes(b"")
    monkeypatch.setenv(agent_counter.EXE_ENV, str(exe))
    monkeypatch.setattr(agent_counter, "_complained", False)
    monkeypatch.setattr(agent_counter, "_exe_pipe_names", {})
    monkeypatch.setattr(agent_counter, "_connect", lambda: None)
    monkeypatch.setattr(agent_counter, "SPAWN_WAIT_S", 0.0)
    asked, spawned = [], []
    user = agent_counter.pipe_path().rsplit(agent_counter.PIPE_PREFIX, 1)[1]

    def run(argv, **kwargs):
        asked.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout=f"{prefix}{user}\n", stderr="")

    monkeypatch.setattr(agent_counter.subprocess, "run", run)
    monkeypatch.setattr(agent_counter, "_spawn", spawned.append)
    for _ in range(3):
        assert agent_counter._open_or_start() is None
    assert asked == [[str(exe), "--pipe-name"]]
    assert len(spawned) == 3 * spawns
    lines = _unavailable_lines(capsys)
    assert len(lines) == 1 and ("old-name.v1." in lines[0]) == (spawns == 0), lines
