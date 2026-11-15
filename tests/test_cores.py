"""The core lease client (`llm_agents.cores`) against a fake counter.

The wire is replaced (`connect`/`transport_of`), so nothing here needs the
optional `extras/agent-counter` executable; the round trips against a built
one are the host's (MultiSkill: tools/agent_host/tests/test_cores_counter.py)
and `AgentCounter.exe --selftest`'s.
"""

import os
import sys
from pathlib import Path

import pytest

from llm_agents import agent_counter, cores

REGISTRY_CPP = Path(__file__).resolve().parents[1] / "extras" / "agent-counter" / "Registry.cpp"

windows_only = pytest.mark.skipif(os.name != "nt", reason="the broker speaks a Windows named pipe")


def test_the_default_wait_is_the_same_figure_in_both_halves():
    assert f"kDefaultWaitS = {cores.DEFAULT_WAIT_S}" in REGISTRY_CPP.read_text(encoding="utf-8")


class FakeCounter:
    """A transport playing the counter: `answers` are handed out one poll at a
    time (None = nothing arrived yet), `cores.Closed` ends the stream."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def poll(self):
        if not self.answers:
            return None
        answer = self.answers.pop(0)
        if answer is cores.Closed:
            raise cores.Closed("end of stream")
        return answer

    def close(self):
        self.closed = True


@pytest.fixture
def wire(monkeypatch):
    """`lease`/`run` options routing to one FakeCounter; lines said are kept."""
    monkeypatch.delenv(cores.LEASE_ENV, raising=False)
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "1")
    state = {"lines": []}

    def options(counter, handle=object()):
        state["counter"] = counter
        return {"connect": lambda complain: handle, "transport_of": lambda _: counter,
                "say": state["lines"].append}

    state["options"] = options
    return state


def _grant(cores_, total=16, **extra):
    return dict({"op": "grant", "cores": cores_, "total": total, "free": cores_,
                 "foreign": 0.0, "forced": False, "waited_s": 0.0}, **extra)


@windows_only
def test_a_grant_reaches_the_child_and_its_exit_code_comes_back(wire, tmp_path):
    counter = FakeCounter(None, _grant(6))
    seen = tmp_path / "seen"
    code = cores.run([sys.executable, "-c",
                      f"import os, sys; open({str(seen)!r}, 'w').write(os.environ['{cores.LEASE_ENV}']); "
                      f"sys.exit(7)"],
                     want=8, minimum=2, label="suite", wait_s=30, **wire["options"](counter))
    assert code == 7
    assert seen.read_text() == "6"
    assert counter.sent == [{"op": "lease", "want": 8, "min": 2, "label": "suite", "wait_s": 30}]
    assert counter.closed, "the lease is returned once the child is done"
    assert wire["lines"] == ["6 of 16 for suite (asked 8, min 2; free 6, foreign 0.0)"]


@windows_only
def test_the_lease_is_held_for_the_childs_whole_life(wire):
    counter = FakeCounter(_grant(4))
    with cores.lease(4, 1, "held", **wire["options"](counter)) as granted:
        assert granted == 4 and not counter.closed
    assert counter.closed


@windows_only
def test_a_queued_lease_says_it_waits_then_takes_the_grant(wire):
    counter = FakeCounter({"op": "queued", "ahead": 1, "free": 1, "total": 16}, None, None,
                          _grant(3))
    with cores.lease(4, 3, "waiter", wait_s=60, **wire["options"](counter)) as granted:
        assert granted == 3
    assert wire["lines"][0] == "waiting for 3 core(s) for waiter (free 1 of 16, 1 ahead), up to 60 s"
    assert len(wire["lines"]) == 2


@windows_only
def test_a_forced_grant_says_it_runs_over_the_machine(wire):
    counter = FakeCounter(_grant(2, total=8, forced=True, waited_s=300.2))
    with cores.lease(8, 2, "late", **wire["options"](counter)) as granted:
        assert granted == 2
    assert wire["lines"] == ["waited 300 s; late runs with its min 2 anyway, over the 8-core machine"]


@windows_only
@pytest.mark.parametrize("answers, reason", [
    (({"op": "error", "error": "unknown-op", "got": "lease"},),
     "the counter refused the lease (unknown-op, lease)"),
    ((cores.Closed,), "the counter closed the connection without a grant (a build without leases?)"),
    (({"op": "grant", "cores": 0},), "unexpected answer from the counter: {'op': 'grant', 'cores': 0}"),
])
def test_an_old_or_refusing_counter_is_no_broker_and_one_line(wire, tmp_path, answers, reason):
    """A counter without `lease` (refusing by name, or a pre-v2 build cutting
    the pipe) leaves the child its own defaults: LLM_CORES stays unset."""
    counter = FakeCounter(*answers)
    seen = tmp_path / "seen"
    code = cores.run([sys.executable, "-c",
                      f"import os; open({str(seen)!r}, 'w').write(os.environ.get('{cores.LEASE_ENV}', 'unset'))"],
                     want=4, label="old", **wire["options"](counter))
    assert code == 0 and seen.read_text() == "unset"
    assert wire["lines"] == [f"no core broker ({reason}); old runs with its own defaults "
                             f"({os.cpu_count()} cores on this machine)"]


@windows_only
def test_a_hung_counter_costs_its_deadline_then_falls_back(wire, monkeypatch):
    clock = iter(range(0, 1000, 10))
    counter = FakeCounter()
    options = wire["options"](counter)
    monkeypatch.setattr(cores, "negotiate", lambda *a, **k: _negotiate(*a, **k, clock=lambda: next(clock),
                                                                         sleep=lambda s: None))
    with cores.lease(4, 1, "hung", wait_s=20, **options) as granted:
        assert granted is None
    assert wire["lines"][0].startswith(f"no core broker (no grant within {20 + cores.ANSWER_SLACK_S:.0f} s)")


_negotiate = cores.negotiate


def test_without_a_counter_the_run_goes_on_with_one_line(wire, monkeypatch, tmp_path):
    """Not built: `_open_or_start` complains once through the lease's sink,
    and that reason is the lease's one line."""
    monkeypatch.setattr(agent_counter, "_connect", lambda: None)
    monkeypatch.setattr(agent_counter, "counter_exe", lambda: None)
    monkeypatch.delenv(agent_counter.EXE_ENV, raising=False)
    lines = []
    with cores.lease(4, 1, "unbuilt", say=lines.append) as granted:
        assert granted is None
    if os.name == "nt":
        assert lines == [f"no core broker (no counter executable (the host policy names none)); "
                         f"unbuilt runs with its own defaults ({os.cpu_count()} cores on this machine)"]
    else:
        assert len(lines) == 1 and "Windows named pipe" in lines[0]


def test_a_disabled_counter_never_touches_the_wire(monkeypatch):
    monkeypatch.delenv(cores.LEASE_ENV, raising=False)
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "0")

    def explode(complain):
        raise AssertionError("a disabled broker opened the pipe")

    lines = []
    with cores.lease(4, 1, "off", say=lines.append, connect=explode) as granted:
        assert granted is None
    assert len(lines) == 1 and lines[0].startswith("no core broker (")


def test_a_nested_run_takes_the_outer_lease_without_a_new_one(monkeypatch):
    monkeypatch.setenv(cores.LEASE_ENV, "3")

    def explode(complain):
        raise AssertionError("a nested lease asked the broker again")

    lines = []
    with cores.lease(8, 1, "inner", say=lines.append, connect=explode) as granted:
        assert granted == 3
    assert lines == ["inside a lease of 3: inner takes 3 without a new one"]


@windows_only
def test_the_child_exit_code_is_forwarded_without_a_broker(monkeypatch):
    monkeypatch.delenv(cores.LEASE_ENV, raising=False)
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "0")
    assert cores.run([sys.executable, "-c", "raise SystemExit(5)"], want=2, say=lambda line: None) == 5


def test_a_body_exception_propagates_and_still_returns_the_lease(wire):
    if os.name != "nt":
        pytest.skip("the broker speaks a Windows named pipe")
    counter = FakeCounter(_grant(2))
    with pytest.raises(RuntimeError):
        with cores.lease(2, 1, "boom", **wire["options"](counter)):
            raise RuntimeError("the body failed")
    assert counter.closed


@pytest.mark.parametrize("argv", [
    ["run", "--want", "0", "--", "x"],
    ["run", "--want", "2", "--min", "3", "--", "x"],
    ["run", "--want", "2"],
])
def test_the_command_line_refuses_an_impossible_lease(argv, capsys):
    with pytest.raises(SystemExit) as exit_:
        cores.main(argv)
    assert exit_.value.code == 2
