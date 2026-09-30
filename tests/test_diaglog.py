"""The status line's quiet branches name themselves in the diagnostics file."""

import io
from pathlib import Path

import pytest

from llm_loop import diaglog, exitlog, operator, statusline as sl, termio

from _termfixtures import RecordingTerminal


class _Record:
    """Stands in for this process's exit record: its path and whether it ended."""

    def __init__(self, path: Path):
        self.path = path
        self.finished = False


@pytest.fixture
def record(tmp_path, monkeypatch):
    fake = _Record(tmp_path / "app-proj.123.run.json")
    monkeypatch.setattr(exitlog, "_record", fake)
    monkeypatch.delenv(diaglog.KEYTRACE_ENV, raising=False)
    return fake


@pytest.fixture
def diag(record, tmp_path):
    return tmp_path / "app-proj.diag.log"


def _text(path: Path) -> str:
    assert diaglog.flush(5.0), "the diagnostics writer did not drain"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_no_path_without_a_live_exit_record(record, monkeypatch):
    record.finished = True
    assert diaglog.log_path() is None
    monkeypatch.setattr(exitlog, "_record", None)
    assert diaglog.log_path() is None


def test_a_dotted_project_name_keeps_its_dots(tmp_path, monkeypatch):
    monkeypatch.setattr(exitlog, "_record",
                        _Record(tmp_path / "runCycle-foo.github.io.77.run.json"))
    assert diaglog.log_path() == tmp_path / "runCycle-foo.github.io.diag.log"


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


class _BrokenStream(io.StringIO):
    def write(self, text):
        raise OSError("console write failed")


def test_a_failed_console_write_is_named_once(diag):
    terminal = termio.Terminal(stream=_BrokenStream())
    assert not terminal._write("a")
    assert not terminal._write("b")
    text = _text(diag)
    assert text.count("Terminal write failed") == 1
    assert "OSError: console write failed" in text


def test_a_key_reader_that_dies_is_named(diag):
    reader = termio.TerminalInput()

    def boom(handler):
        raise RuntimeError("reader exploded")

    reader._guard(boom, lambda event: None)
    text = _text(diag)
    assert "reader died" in text
    assert "RuntimeError: reader exploded" in text


def test_a_refused_region_is_named_once_per_shape(diag, monkeypatch):
    terminal = RecordingTerminal()
    app = sl.StatusApp(terminal=terminal, messages=operator.Mailbox())
    assert app.painter.reserve(len(app.rows()))
    monkeypatch.setattr(app.painter, "_reserve", lambda rows: False)
    app.handle_event(termio.Key("m"))       # the editor adds a row
    for _ in range(3):
        app.painter._draw()
    assert _text(diag).count("frames skipped, region refused") == 1


def test_key_trace_is_off_by_default_and_names_each_stage_when_on(
        diag, monkeypatch):
    terminal = RecordingTerminal()
    app = sl.StatusApp(terminal=terminal, messages=operator.Mailbox())
    assert app.painter.reserve(len(app.rows()))
    reader = termio.TerminalInput()
    reader._emit(lambda event: None, "m")
    app._apply_key(app._key_epoch, termio.Key("m"))
    # Opening the editor is always named, with who shares the console; the
    # per-stage key lines are not, with the trace off.
    text = _text(diag)
    assert "note editor opened" in text
    assert "key " not in text

    monkeypatch.setenv(diaglog.KEYTRACE_ENV, "1")
    reader._emit(lambda event: None, "q")
    app.painter.post_key(lambda: None)      # no painter running: runs inline
    app._apply_key(app._key_epoch, termio.Key("x"))
    app._apply_key(app._key_epoch - 1, termio.Key("y"))
    app.painter._draw()
    text = _text(diag)
    assert "key read: 'q'" in text
    assert "key posted" in text
    assert "key applied: Key(char='x')" in text
    assert "key discarded, stale epoch: Key(char='y')" in text
    assert "frame painted" in text
