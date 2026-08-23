"""How wide terminal text is, and how much of it one line may hold.

Everything that paints — the pinned rows, the sequential stream renderer, the
per-worker lines — cuts with these, so the pins live next to the primitives
rather than next to any one of their callers.
"""
import os

import pytest

from llm_loop import textwidth as tw


def test_fit_truncates_with_an_ellipsis():
    assert tw.fit("abcdef", 10) == "abcdef"
    assert tw.fit("abcdef", 4) == "abc…"
    assert tw.fit("abcdef", 0) == ""


def test_fit_tail_keeps_the_end_and_marks_the_cut():
    assert tw.fit_tail("abcdef", 10) == "abcdef"
    assert tw.fit_tail("abcdef", 4) == "…def"
    assert tw.fit_tail("abcdef", 0) == ""


# --- how wide one scrolling line may be ---------------------------------------
#
# The compact renderers used to cut their variable field at 140, 160 or 200
# characters depending on the call site and the provider. `line_budget` replaces
# all three with the terminal's own width; what the renderers do with it is
# pinned in test_providers.py.


# Every expectation below is a LITERAL. Restating the constant under test
# (`== tw.LINE_RIGHT_MARGIN`, `== tw.LEGACY_LINE_COLUMNS`) made three of these pins
# survive a mutation of the very number they were named after — measured, with
# try_patch --expect-fail.


def _columns(monkeypatch, columns):
    """Pretend the terminal is `columns` wide; 0 = there is no terminal.

    The fake honours the `fallback` it is handed, as shutil does — a fake that
    ignores it hides which fallback the caller asked for, and that argument is
    the whole of "no terminal means the legacy width, not shutil's 80".
    """
    def fake_size(fallback=(80, 24)):
        return os.terminal_size((columns, 30) if columns else fallback)

    monkeypatch.setattr(tw.shutil, "get_terminal_size", fake_size)


def test_the_budget_is_the_terminal_minus_the_prefix(monkeypatch):
    _columns(monkeypatch, 240)

    assert tw.terminal_columns() == 240
    assert tw.line_budget() == 239               # one column left unwritten
    assert tw.line_budget("[job 7] ") == 231     # ...and the head's eight


def test_a_prefix_is_measured_in_cells_not_characters(monkeypatch):
    """The heads are full of double-width glyphs; `len` would under-count them
    by one column each and the line would wrap by exactly that much."""
    pytest.importorskip("rich.cells")
    _columns(monkeypatch, 400)

    assert tw.cell_width("💻 ") == 3            # two cells plus the space
    assert tw.line_budget("💻 ") == 396


def test_no_terminal_hands_back_the_fallback_rather_than_eighty(monkeypatch):
    """Redirected to a file or a pipe there is no screen to fit, and the only
    reader left is the mirror log — where a cut cannot be undone."""
    _columns(monkeypatch, 0)

    assert tw.terminal_columns(fallback=7) == 7   # shutil's own 80 never shows
    assert tw.terminal_columns() == 200           # the legacy line width
    assert tw.line_budget("[job 7] ") == 200      # floored, not 191


def test_a_narrow_terminal_wraps_rather_than_record_less(monkeypatch):
    """Screen and mirror log get the same text, so cutting to a small window
    would shrink the run's record; 200 is what the fixed limits used to keep."""
    _columns(monkeypatch, 30)

    assert tw.line_budget("[job 12]   ⚙ NotebookEdit: ") == 200
    assert tw.LEGACY_LINE_COLUMNS == 200          # the figure it replaced
