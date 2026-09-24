"""Startup controls run on a fake clock: no agent launch or real delay."""

from types import SimpleNamespace

import pytest

from llm_loop import cyclecore, termio


@pytest.fixture
def waiting(monkeypatch):
    state = SimpleNamespace(now=0.0, actions=[], frames=[], stopped=False,
                            released=False, enabled=None, width=80)

    class Terminal:
        def reserve(self, rows):
            return state.enabled

        def size(self):
            return state.width, 24

        def paint(self, rows):
            state.frames.append(list(rows))

        def release(self):
            state.released = True

    class Reader:
        def usable(self):
            return True

        def start(self, handler):
            pass

        def stop(self):
            state.stopped = True

    class Events:
        def put(self, event):
            pass

        def get(self, timeout):
            if state.actions:
                action = state.actions.pop(0)
                if isinstance(action, BaseException):
                    raise action
                if callable(action):
                    return action()
                if isinstance(action, str):
                    return termio.Key(action)
            state.now += timeout
            raise cyclecore.queue.Empty

    def terminal_for(*, enabled):
        state.enabled = enabled
        return Terminal()

    def sleep(seconds):
        state.now += seconds

    monkeypatch.setattr(cyclecore.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(cyclecore.time, "time", lambda: 10000 + state.now)
    monkeypatch.setattr(cyclecore.time, "sleep", sleep)
    monkeypatch.setattr(cyclecore.queue, "Queue", Events)
    monkeypatch.setattr(termio, "terminal_for", terminal_for)
    monkeypatch.setattr(termio, "TerminalInput", Reader)
    return state


@pytest.mark.parametrize("seconds, expected", [
    (0, "00:00"), (59, "00:59"), (60, "01:00"), (3599, "59:59"),
    (3600, "01:00:00"), (3661, "01:01:01"), (360000, "100:00:00"),
])
def test_clock(seconds, expected):
    assert cyclecore._wait_clock(seconds) == expected


def test_countdown_repaints_without_logging_ticks(waiting, capsys):
    cyclecore.wait_before_start("2s")
    assert waiting.now == 2
    assert waiting.frames[0][0] == "Elapsed 00:00  |  Remaining 00:02"
    assert waiting.frames[1][0] == "Elapsed 00:01  |  Remaining 00:01"
    assert waiting.frames[-1][0] == "Elapsed 00:02  |  Remaining 00:00"
    assert "[q] quit" in " ".join(waiting.frames[0])
    output = capsys.readouterr().out
    assert "waiting until" in output
    assert "Elapsed" not in output
    assert "Remaining" not in output
    assert waiting.stopped and waiting.released


@pytest.mark.parametrize("actions, duration", [
    (["+"], 62), (["-"], 0), ([" ", "+"], 0),
    ([None, "+", "-"], 2), (["x"], 2),
])
def test_keys_adjust_remaining_or_start_immediately(waiting, actions, duration):
    waiting.actions = actions
    cyclecore.wait_before_start("2s")
    assert waiting.now == duration
    assert waiting.stopped and waiting.released


@pytest.mark.parametrize("action, code", [("q", 0), (KeyboardInterrupt(), 130)])
def test_quit_never_starts_and_restores_terminal(waiting, capsys, action, code):
    waiting.actions = [action]
    with pytest.raises(SystemExit) as caught:
        cyclecore.wait_before_start("2h")
    assert caught.value.code == code
    assert waiting.now == 0
    assert "Starting the loop" not in capsys.readouterr().out
    assert waiting.stopped and waiting.released


def test_disabled_ui_waits_silently(waiting, capsys):
    cyclecore.wait_before_start("2h", interactive=False)
    assert waiting.now == 7200
    assert waiting.frames == []
    assert len(capsys.readouterr().out.splitlines()) == 2


def test_narrow_terminal_keeps_all_controls(waiting):
    waiting.width = 40
    waiting.actions = [" "]
    cyclecore.wait_before_start("2h")
    assert all(len(row) < 40 for frame in waiting.frames for row in frame)
    assert "[space] start now" in " ".join(waiting.frames[0])


def test_fractional_wait_does_not_show_zero_early(waiting):
    cyclecore.wait_before_start("0.5s")
    assert waiting.frames[0][0].endswith("Remaining 00:01")
    assert waiting.now == 0.5


def test_default_termination_unwinds_and_restores_signal(waiting, monkeypatch):
    signal = cyclecore.signal
    handlers = {signal.SIGTERM: signal.SIG_DFL}
    monkeypatch.setattr(signal, "getsignal",
                        lambda number: handlers.get(number, signal.SIG_IGN))
    monkeypatch.setattr(signal, "signal", handlers.__setitem__)
    waiting.actions = [lambda: handlers[signal.SIGTERM](signal.SIGTERM, None)]
    with pytest.raises(SystemExit) as caught:
        cyclecore.wait_before_start("2h")
    assert caught.value.code == 128 + signal.SIGTERM
    assert waiting.stopped and waiting.released
    assert handlers[signal.SIGTERM] == signal.SIG_DFL


def test_keypress_does_not_shift_second_boundary(waiting):
    def press_plus_between_ticks():
        waiting.now = 0.9
        return termio.Key("+")

    waiting.actions = [press_plus_between_ticks, None, " "]
    cyclecore.wait_before_start("2s")
    assert waiting.now == 1.0
    assert waiting.frames[2][0] == "Elapsed 00:01  |  Remaining 01:01"
