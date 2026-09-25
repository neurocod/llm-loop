"""One sequential run may cross provider and quota-account boundaries."""

from types import SimpleNamespace
import time

import pytest

from llm_loop import (cyclecore, projectroot, providers, runlifecycle,
                      statusline, stopchannel)
from llm_loop.agentwork import AgentCommand, Driver
from llm_loop.drivers import StateFileDriver
from llm_loop.usage import RateLimitEvent


class Source:
    def __init__(self, provider):
        self.provider = provider

    def invalidate(self):
        pass


class Policy:
    def __init__(self, provider, events):
        self.provider = provider
        self.events = events

    def describe(self):
        return self.provider

    def log_snapshot(self, source, *args, **kwargs):
        assert source.provider == self.provider

    def check_and_wait(self, source, session_start, **kwargs):
        assert source.provider == self.provider
        self.events.append((self.provider, session_start))
        return False, session_start + 10


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    events, sources, apps, refreshers = [], [], [], []
    previous_root = projectroot.project_dir()
    real_app = statusline.StatusApp

    def source_for(provider):
        sources.append(provider)
        return Source(provider)

    def make_app(**kwargs):
        app = real_app(**kwargs)
        apps.append(app)
        return app

    class Refresher:
        def __init__(self, app, source, policy, *, provider):
            self.selections = [provider]
            refreshers.append(self)

        def start(self):
            pass

        def stop(self):
            pass

        def set_source(self, source, policy, *, provider):
            assert source.provider == policy.provider == provider
            self.selections.append(provider)

    monkeypatch.setattr(runlifecycle, "usage_source_for", source_for)
    monkeypatch.setattr(runlifecycle.limits, "default_policy", lambda p: Policy(p, events))
    monkeypatch.setattr(statusline, "StatusApp", make_app)
    monkeypatch.setattr(statusline, "QuotaRefresher", Refresher)
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    args = SimpleNamespace(max=None, dry_run=False, raw=False, start_in=None,
                           git_push="none", project_dir=str(tmp_path), cost=False,
                           no_statusline=True, provider="claude")
    yield SimpleNamespace(args=args, events=events, sources=sources, apps=apps,
                          refreshers=refreshers)
    projectroot.set_project_root(previous_root)


def run(driver, runtime):
    return cyclecore.run_loop(driver, runtime.args, app_name="pytest-mixed",
                              setup_logging=False, wait_on_start=False)


def test_state_cycle_dispatches_each_provider_and_preserves_its_session(
        monkeypatch, tmp_path, runtime, capsys):
    state_file = tmp_path / "currentState.md"
    state_file.write_text("Current state: planning", encoding="utf-8")
    calls = []

    class Cycle(StateFileDriver):
        state_file = str(tmp_path / "currentState.md")

        def prompt(self):
            return f"Follow {self.state_file}"

        def model(self):
            return "claude/opus" if self.state_name() == "implementation" else "codex"

        def on_success(self, rc):
            states = {"planning": "implementation", "implementation": "cleanup",
                      "cleanup": "done"}
            state_file.write_text(f"Current state: {states[self.state_name()]}",
                                  encoding="utf-8")

    def codex(argv, provider, raw, partial, **kwargs):
        calls.append(provider)
        assert provider == "codex"
        assert argv[0] == "codex"
        assert argv.model == ""
        assert runtime.apps[0].status.provider == provider
        return 0

    def claude(argv, raw, partial, **kwargs):
        calls.append("claude")
        assert argv[0] == "claude"
        assert argv[argv.index("--model") + 1] == "opus"
        assert runtime.apps[0].status.provider == "claude"
        return 0

    monkeypatch.setattr(cyclecore, "run_agent_streaming", codex)
    monkeypatch.setattr(cyclecore, "run_claude_streaming", claude)
    monkeypatch.setattr(providers, "_LIVE_MESSAGES", True)
    check = Policy.check_and_wait

    def check_model(policy, source, session_start, **kwargs):
        expected = "opus" if source.provider == "claude" else ""
        assert runtime.apps[0].job(1).model == expected
        return check(policy, source, session_start, **kwargs)

    monkeypatch.setattr(Policy, "check_and_wait", check_model)
    driver = Cycle()
    result = run(driver, runtime)

    assert calls == ["codex", "claude", "codex"]
    assert runtime.sources == ["codex", "claude"]
    assert [p for p, _ in runtime.events] == calls
    assert runtime.events[2][1] == runtime.events[0][1] + 10
    assert runtime.refreshers[0].selections == calls
    assert driver.provider == "claude"
    assert result.reason == stopchannel.RunStopReason.DRIVER_STOP
    output = capsys.readouterr().out
    assert "codex/cli default" in output and "claude/opus" in output


def test_bare_command_provider_uses_launch_default_after_explicit_step(monkeypatch, runtime):
    commands = iter([AgentCommand("first", provider="codex"),
                     AgentCommand("second", "opus"), None])

    class Queue(Driver):
        def next_command(self):
            return next(commands)

    calls = []
    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda argv, provider, *a, **k: calls.append(provider) or 0)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *a, **k: calls.append("claude") or 0)
    run(Queue(), runtime)
    assert calls == ["codex", "claude"]


def test_dry_run_uses_step_provider_without_querying_quota(monkeypatch, runtime, capsys):
    class Queue(Driver):
        def next_command(self):
            return AgentCommand("work", "opus", provider="claude")

    runtime.args.provider = "codex"
    runtime.args.dry_run = True
    monkeypatch.setattr(Policy, "log_snapshot", lambda *a, **k: pytest.fail("quota read"))
    result = run(Queue(), runtime)
    assert "DRY-RUN: claude" in capsys.readouterr().out
    assert result.reason == stopchannel.RunStopReason.DRY_RUN
    assert runtime.events == []
    assert runtime.refreshers == []


def test_no_work_never_opens_the_default_provider_account(runtime):
    class Empty(Driver):
        def next_command(self):
            return None

    run(Empty(), runtime)
    assert runtime.sources == []
    assert runtime.events == []
    assert runtime.refreshers == []


@pytest.mark.parametrize("edited_state", ["planning", "done", "error"])
def test_state_edit_during_quota_pause_revalidates_pending_step(
        monkeypatch, tmp_path, runtime, edited_state):
    state = tmp_path / "currentState.md"
    state.write_text("Current state: implementation", encoding="utf-8")
    paused = False
    gated = False
    calls = []

    class Cycle(StateFileDriver):
        state_file = str(state)

        def prompt(self):
            return "Follow currentState.md"

        def model(self):
            return "claude/opus" if self.state_name() == "implementation" else "codex"

        def on_success(self, rc):
            state.write_text("Current state: done", encoding="utf-8")

    def gate(policy, source, session_start, **kwargs):
        nonlocal paused, gated
        if not gated:
            paused = gated = True
        return False, session_start

    def hold(*args, **kwargs):
        nonlocal paused
        state.write_text(f"Current state: {edited_state}", encoding="utf-8")
        paused = False

    monkeypatch.setattr(Policy, "check_and_wait", gate)
    monkeypatch.setattr(stopchannel, "pause_requested", lambda app: paused)
    monkeypatch.setattr(stopchannel, "wait_while_paused", hold)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *a, **k: pytest.fail("launched stale Claude step"))
    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda argv, provider, *a, **k: calls.append(provider) or 0)

    if edited_state == "error":
        with pytest.raises(SystemExit) as stopped:
            run(Cycle(), runtime)
        assert stopped.value.code == 1
    else:
        run(Cycle(), runtime)
    assert calls == (["codex"] if edited_state == "planning" else [])


@pytest.mark.parametrize("return_to_claude", [False, True])
def test_claude_refusal_only_blocks_its_next_command(monkeypatch, runtime, return_to_claude):
    items = [AgentCommand("implementation", "opus", provider="claude"),
             AgentCommand("cleanup", provider="codex")]
    if return_to_claude:
        items.append(AgentCommand("next implementation", "opus", provider="claude"))
    commands = iter([*items, None])
    calls, waits = [], []
    reset = time.time() + 3600

    class Queue(Driver):
        def next_command(self):
            return next(commands)

    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda *a, **k: calls.append("claude") or 0)
    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda argv, provider, *a, **k: calls.append(provider) or 0)
    monkeypatch.setattr(cyclecore, "last_rate_limit_event",
                        lambda: RateLimitEvent("rejected", "five_hour", reset)
                        if len(calls) == 1 else None)
    monkeypatch.setattr(cyclecore, "wait_until",
                        lambda target, **k: waits.append((target, list(calls))))

    run(Queue(), runtime)

    assert calls[:2] == ["claude", "codex"]
    assert waits == ([(reset + 5, ["claude", "codex"])] if return_to_claude else [])
