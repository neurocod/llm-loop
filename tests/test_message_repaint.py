"""Exercise the painted editor, including a terminal too slow for each key."""

import io
import queue
import re
import threading
import time

import pytest

from llm_loop import operator, statusline as sl, termio, textwidth


class RecordingTerminal(termio.Terminal):
    def __init__(self, columns=80):
        super().__init__(io.StringIO())
        self.columns = columns
        self.frames = queue.Queue()

    def size(self):
        return self.columns, 30

    def paint(self, lines, *, reassert=False):
        result = super().paint(lines, reassert=reassert)
        self.frames.put([re.sub(r"\x1b\[[0-9;]*m", "", line) for line in lines])
        return result


@pytest.mark.parametrize("text", ["a" * 200, "полка " * 100, "码" * 200],
                         ids=["ascii", "cyrillic", "wide"])
def test_painted_long_note_scrolls_without_touching_the_last_column(text):
    terminal = RecordingTerminal()
    app = sl.StatusApp(terminal=terminal, messages=operator.Mailbox())
    assert app._reserve(len(app.rows()))
    try:
        app.handle_event(termio.Key("m"))
        app.mode.editor.set(text)
        for columns in (80, 40, 120):
            terminal.columns = columns
            app.handle_event(termio.Resize(columns, 30))
            app._paint()
            frames = []
            while not terminal.frames.empty():
                frames = terminal.frames.get_nowait()
            assert all(textwidth.cell_width(row) < columns for row in frames)
            assert frames[-1].startswith(" ✉ …")
            assert frames[-1].endswith(text[-1] + sl.MessagePromptRow.caret)
        app.handle_event(termio.Key("home"))
        assert sl.MessagePromptRow.caret in terminal.frames.get_nowait()[-1]
        app.handle_event(termio.Key("\r"))
        assert app.messages.take_queued() == [text.strip()]
    finally:
        terminal.release()


def test_slow_paint_does_not_block_input_and_eventually_shows_the_latest_note():
    class Input(termio.NullInputSource):
        def start(self, handler):
            self.handler = handler

    terminal, source = RecordingTerminal(), Input()
    mailbox = operator.Mailbox()
    app = sl.StatusApp(terminal=terminal, input_source=source, messages=mailbox)
    text = "measure the shelf " * 100 + "FINAL"
    read = threading.Event()

    def type_note():
        # Open another mode first, then leave and re-enter after a send.
        for char in "pmfirst\r\x1bm" + text:
            source.handler(termio.Key(char))
        read.set()

    # The whole focused suite took 3.47 s on 2026-09-15; allow over twice that
    # even for this single blocked-writer regression on a loaded machine.
    budget = 15.0
    with app:
        terminal._lock.acquire()
        reader = threading.Thread(target=type_note, daemon=True)
        try:
            reader.start()
            assert read.wait(budget), "terminal painting stalled the key reader"
            assert app.mode.buffer == text
            assert mailbox.take_queued() == ["first"]
        finally:
            terminal._lock.release()
            reader.join(timeout=budget)

        deadline = time.monotonic() + budget
        while True:
            frame = terminal.frames.get(timeout=max(0, deadline - time.monotonic()))
            if frame[-1].endswith("FINAL" + sl.MessagePromptRow.caret):
                break
        assert all(textwidth.cell_width(row) < terminal.columns for row in frame)
        source.handler(termio.Key("\r"))
        assert mailbox.take_queued() == [text]
        assert not app.stop_requested_here


def test_arrows_show_a_literal_caret_at_the_insertion_position():
    app = sl.StatusApp(terminal=termio.NullTerminal(), messages=operator.Mailbox())
    for char in "mabcd":
        app.handle_event(termio.Key(char))
    app.handle_event(termio.Key("left"))
    app.handle_event(termio.Key("left"))
    assert sl.colorize(app.render(width=40)[-1]) == " ✉ ab|cd"
    app.handle_event(termio.Key("X"))
    app.handle_event(termio.Key("right"))
    assert sl.colorize(app.render(width=40)[-1]) == " ✉ abXc|d"
    app.handle_event(termio.Key("\r"))
    assert app.messages.take_queued() == ["abXcd"]
