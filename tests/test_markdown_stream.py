"""Tests for console.MarkdownStream - one streamed assistant text block.

The pin that matters is the timing contract: a delta only appends, and the
block is parsed when Rich's `Live` asks for a frame. Parsing on every delta
read as correct on screen and made the stream reader's work quadratic in the
block, so the parse count is asserted, not just the rendered text.
"""

import io
import sys
import time

import pytest

from llm_loop import console

pytest.importorskip("rich")


def _chunks():
    """~600 deltas of a Markdown block whose last word is `omega`."""
    text = ("Some **bold** text and `code`.\n\n- item one\n- item two\n\n"
            * 120) + "omega"
    return [text[i:i + 10] for i in range(0, len(text), 10)], text


@pytest.fixture
def parses(monkeypatch):
    """Every Markdown source `console` parses, in order."""
    seen = []
    real = console._RichMarkdown

    def counting(text, *args, **kwargs):
        seen.append(text)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(console, "_RichMarkdown", counting)
    return seen


@pytest.fixture
def logged(monkeypatch):
    lines = []
    monkeypatch.setattr(console, "_log_plain", lines.append)
    return lines


def test_burst_of_deltas_is_not_parsed_per_delta(monkeypatch, parses, logged):
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(console, "RICH_AVAILABLE", True)
    chunks, text = _chunks()

    stream = console.MarkdownStream()
    stream.start()
    t0 = time.perf_counter()
    for chunk in chunks:
        stream.feed(chunk)
    stream.stop()
    elapsed = time.perf_counter() - t0

    # One parse at start, one at stop, at most one per refresh in between.
    ceiling = 3 + int(elapsed * console.LIVE_REFRESH_PER_SECOND)
    assert len(parses) <= ceiling < len(chunks)
    assert parses[-1] == text
    assert logged == [text]
    assert "omega" in out.getvalue()


def test_unchanged_text_is_not_parsed_again(monkeypatch, parses, logged):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(console, "RICH_AVAILABLE", True)
    stream = console.MarkdownStream()
    stream.start()
    stream.feed("hello")
    # Hold Live's refresh lock, as Rich does around every frame: a refresh
    # tick racing this thread's first `_frame` would parse twice and swap
    # `_parsed` without any defect in the code under test.
    with stream._live._lock:
        first = stream._frame()
        assert stream._frame() is first
        before = len(parses)
        stream._frame()
        assert len(parses) == before
    stream.stop()
    assert logged == ["hello"]


def test_plain_fallback_streams_complete_text(monkeypatch, parses):
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(console, "RICH_AVAILABLE", False)
    chunks, text = _chunks()

    stream = console.MarkdownStream()
    stream.start()
    for chunk in chunks:
        stream.feed(chunk)
    stream.stop()

    assert out.getvalue() == "\n💬 " + text + "\n"
    assert parses == []


def test_help_command_is_printed_above_the_live_frame(monkeypatch, logged):
    from rich.console import Console
    from rich.live import Live
    from llm_loop import statusline, termio

    out = io.StringIO()
    monkeypatch.setattr(sys, "stdout", out)
    literal_arg = "[red]:rocket:" + "x" * 200
    monkeypatch.setattr(sys, "argv", ["runCycle.py", "--max-runs", "7",
                                    "--wrapper-option", literal_arg])
    monkeypatch.setattr(console, "_RichConsole", lambda **kwargs: Console(
        **kwargs, force_terminal=True, force_interactive=True,
        legacy_windows=False, _environ={"TERM": "xterm"}, width=100, height=40))
    monkeypatch.setattr(console, "_RichLive", lambda **kwargs: Live(
        **kwargs, auto_refresh=False))
    monkeypatch.setattr(console, "RICH_AVAILABLE", True)
    stream = console.MarkdownStream()
    stream.start()
    try:
        stream.feed("\n\n".join(f"assistant row {i}" for i in range(7)))
        stream._live.refresh()
        out.seek(0)
        out.truncate()

        statusline.StatusApp(enabled=False).handle_event(termio.Key("h"))

        printed = out.getvalue()
        # Live clears its old frame BEFORE the command, then redraws below it.
        # Its next refresh can therefore erase only the assistant block.
        assert "\x1b[2K" in printed.split("Restart command:")[0]
        assert "assistant row 6" in printed.split("--max-runs 7")[1]
        assert literal_arg in printed
        assert logged[0].startswith("Restart command:\n")
        assert "--max-runs 7" in logged[0]
        stream._live.refresh()
    finally:
        stream.stop()

    console.print_line("after stream")
    assert out.getvalue().endswith("after stream\n")
