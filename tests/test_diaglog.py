"""The status line's quiet branches name themselves in the diagnostics file."""

from pathlib import Path

import pytest

from llm_loop import diaglog, exitlog, operator, statusline as sl, termio

from _termfixtures import RecordingTerminal


class _Record:
    """Stands in for this process's exit record: only its path is read."""

    def __init__(self, path: Path):
        self.path = path


@pytest.fixture
def diag(tmp_path, monkeypatch):
    monkeypatch.setattr(exitlog, "_record",
                        _Record(tmp_path / "app-proj.123.run.json"))
    monkeypatch.delenv(diaglog.KEYTRACE_ENV, raising=False)
    return tmp_path / "app-proj.diag.log"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_nothing_is_written_without_an_exit_record(tmp_path, monkeypatch):
    monkeypatch.setattr(exitlog, "_record", None)
    diaglog.record("anything")
    assert diaglog.log_path() is None
    assert list(tmp_path.iterdir()) == []


def test_a_frame_that_raises_is_named_with_its_traceback(diag):
    terminal = RecordingTerminal()
    app = sl.StatusApp(terminal=terminal, messages=operator.Mailbox())
    assert app.painter.reserve(len(app.rows()))

    def broken(width):
        raise ValueError("render exploded")

    app.painter._render = broken
    app.painter.request_frame()
    text = _text(diag)
    assert "Painter._draw raised" in text
    assert "ValueError: render exploded" in text
    assert "disabled the terminal for good" in text
    assert isinstance(app.terminal, termio.NullTerminal)


def test_a_key_reader_that_dies_is_named(diag):
    reader = termio.TerminalInput()

    def boom(handler):
        raise RuntimeError("reader exploded")

    reader._guard(boom, lambda event: None)
    assert "reader died" in _text(diag)
    assert "RuntimeError: reader exploded" in _text(diag)


def test_key_trace_is_off_by_default_and_follows_every_stage_when_on(
        diag, monkeypatch):
    terminal = RecordingTerminal()
    app = sl.StatusApp(terminal=terminal, messages=operator.Mailbox())
    assert app.painter.reserve(len(app.rows()))
    app._apply_key(app._key_epoch, termio.Key("m"))
    assert _text(diag) == ""

    monkeypatch.setenv(diaglog.KEYTRACE_ENV, "1")
    app._apply_key(app._key_epoch, termio.Key("x"))
    app._apply_key(app._key_epoch - 1, termio.Key("y"))
    app.painter._draw()
    text = _text(diag)
    assert "key applied: Key(" in text and "'x'" in text
    assert "key discarded, stale epoch" in text
    assert "frame painted" in text
