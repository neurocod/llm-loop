"""Console breakpoint input and stopping before the selected state runs."""

import pytest

from llm_loop import cyclecore, statusline as sl, termio
from llm_loop.breakpoints import Breakpoints
from llm_loop.drivers import StateFileDriver
from llm_loop.stopchannel import RunStopReason


def press(app, *keys):
    for key in keys:
        app.handle_event(termio.Key(key))


def test_names_are_trimmed_appended_and_match_whole_states():
    state = ["plan mode"]
    points = Breakpoints(lambda: state[0])
    assert points.add("  cleanup | codex-review ||  ") == ("cleanup", "codex-review")
    assert points.add(" plan mode ") == ("cleanup", "codex-review", "plan mode")
    assert points.reached() == "plan mode"
    state[0] = "Cleanup"
    assert points.reached() == "cleanup"
    state[0] = "cleanup-extra"
    assert points.reached() is None
    assert Breakpoints(lambda: "cleanup").names == ()


@pytest.mark.parametrize("enter", ["\r", "\n"])
def test_editor_appends_on_enter_and_esc_discards_only_the_draft(enter):
    points = Breakpoints(lambda: "implementation")
    app = sl.StatusApp(enabled=False)
    app.register_action(sl.BreakpointAction(points))
    press(app, "b", *"  plan mode | cleanup  ")
    assert points.names == ()
    assert not app.paused and not app.stop_requested_here
    assert "plan mode | cleanup" in app.render(120)[-1]
    press(app, enter)
    assert points.names == ("plan mode", "cleanup")
    assert isinstance(app.mode, sl.NormalMode)
    assert "cleanup" in app.status.note
    press(app, "b", *"codex-review", enter)
    assert points.names == ("plan mode", "cleanup", "codex-review")
    press(app, "b", *"discard this", "\x1b", "s")
    assert isinstance(app.mode, sl.BreakpointMode)
    assert app.mode.editor.buffer == "s"
    assert not app.stop_requested_here
    press(app, "\x1b", "\x1b")
    assert isinstance(app.mode, sl.NormalMode)
    assert points.names == ("plan mode", "cleanup", "codex-review")


def test_editor_empty_submit_and_cursor_editing():
    points = Breakpoints(lambda: "cleanup")
    app = sl.StatusApp(enabled=False)
    app.register_action(sl.BreakpointAction(points))
    press(app, "b", *" |  ||", "\r")
    assert points.names == ()
    press(app, "b", *"cleanu", "home", "x", "delete", "\x08", "c", "end", "p", "\r")
    assert points.names == ("cleanup",)


@pytest.mark.parametrize("enter", ["\r", "\n", "\r\n"])
def test_paste_tail_cannot_trigger_shortcuts_after_submit(enter):
    points = Breakpoints(lambda: "cleanup")
    app = sl.StatusApp(enabled=False)
    app.register_action(sl.BreakpointAction(points))
    source = termio.TerminalInput()
    app._input = source
    press(app, "b")
    for char in "cleanup" + enter + "spmb":
        source._emit(app.handle_event, char)
    assert points.names == ("cleanup",)
    assert isinstance(app.mode, sl.NormalMode)
    assert not app.stop_requested_here and not app.paused
    source._idle(app.handle_event)
    source._emit(app.handle_event, "b")
    assert isinstance(app.mode, sl.BreakpointMode)


class AdvancingDriver(StateFileDriver):
    def __init__(self):
        self.state = "implementation"
        self.issued = []

    def first_line(self):
        return "Current state: " + self.state

    def prompt(self):
        self.issued.append(self.state)
        return "go"

    def on_success(self, returncode):
        self.state = "cleanup"


@pytest.fixture
def runner(tmp_path, monkeypatch):
    apps = []
    real_start = sl.StatusApp.start

    def start(app):
        apps.append(app)
        return real_start(app)

    monkeypatch.setattr(sl.StatusApp, "start", start)
    # No account or subprocess is needed for a state-boundary test.
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: None)
    args = cyclecore.parse_args([
        "--project-dir", str(tmp_path), "--max-runs", "3",
        "--git-push", "none", "--no-statusline"])

    def run(driver, agent):
        monkeypatch.setattr(cyclecore, "run_claude_streaming", agent)
        return cyclecore.run_loop(driver, args, wait_on_start=False,
                                  app_name="pytest-breakpoint")

    return apps, args, run


@pytest.mark.parametrize("returncode", [0, 1])
def test_entered_during_agent_stops_before_matching_state(runner, capsys, returncode):
    apps, args, run = runner
    driver = AdvancingDriver()

    def agent(*unused, **kwargs):
        press(apps[0], "b", *" plan mode | cleanup ", "\r")
        driver.state = "cleanup"
        return returncode

    result = run(driver, agent)
    assert driver.issued == ["implementation"]
    assert result.attempted == 1
    assert result.reason is RunStopReason.BREAKPOINT
    assert "Breakpoint reached: 'cleanup'" in capsys.readouterr().out


def test_breakpoint_on_current_state_stops_before_first_command(runner, monkeypatch):
    apps, args, run = runner
    real_start = sl.StatusApp.start

    def start(app):
        result = real_start(app)
        press(app, "b", *"implementation", "\r")
        return result

    monkeypatch.setattr(sl.StatusApp, "start", start)
    driver = AdvancingDriver()
    result = run(driver, lambda *a, **kw: pytest.fail("agent must not start"))
    assert result.attempted == 0
    assert driver.issued == []
    assert result.reason is RunStopReason.BREAKPOINT


def test_breakpoint_added_after_selection_still_prevents_launch(runner, monkeypatch):
    apps, args, run = runner
    driver = AdvancingDriver()

    def source(provider):
        press(apps[0], "b", *"implementation", "\r")
        return None

    monkeypatch.setattr(cyclecore, "usage_source_for", source)
    result = run(driver, lambda *a, **kw: pytest.fail("agent must not start"))
    assert result.attempted == 0
    assert driver.issued == ["implementation"]
    assert result.reason is RunStopReason.BREAKPOINT


def test_breakpoint_releases_a_quota_wait_without_launching(runner, monkeypatch):
    apps, args, run = runner
    args.max = None
    driver = AdvancingDriver()
    waits = []

    class Policy:
        def describe(self):
            return "test quota"

        def log_snapshot(self, *a, **kw):
            pass

        def check_and_wait(self, source, session_start, *, should_stop):
            assert not should_stop()
            press(apps[0], "b", *"implementation", "\r")
            waits.append(should_stop())
            return True, session_start

    driver.limit_policy = Policy()
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: object())
    monkeypatch.setattr(sl, "push_quotas", lambda *a, **kw: None)
    monkeypatch.setattr(sl.StatusApp, "add_service", lambda self, service: service)
    result = run(driver, lambda *a, **kw: pytest.fail("agent must not start"))
    assert waits == [True]
    assert result.reason is RunStopReason.BREAKPOINT
    assert result.attempted == 0


def test_breakpoint_releases_a_manual_pause(runner, monkeypatch):
    from llm_loop import stopchannel

    apps, args, run = runner
    driver = AdvancingDriver()
    real_start = sl.StatusApp.start

    def start(app):
        result = real_start(app)
        press(app, "p")
        return result

    def wait(app, *, should_stop):
        assert app.paused
        press(app, "b", *"implementation", "\r")
        assert should_stop()

    monkeypatch.setattr(sl.StatusApp, "start", start)
    monkeypatch.setattr(stopchannel, "wait_while_paused", wait)
    result = run(driver, lambda *a, **kw: pytest.fail("agent must not start"))
    assert result.reason is RunStopReason.BREAKPOINT
    assert driver.issued == []
