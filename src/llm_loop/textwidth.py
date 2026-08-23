"""textwidth.py - how wide terminal text is, and how much of it a line may hold.

The bottom of the stack: it imports nothing from the package, because everything
that PAINTS depends on it. The pinned rows (`statusline`), the sequential stream
renderer (`cyclecore`) and the per-worker lines (`parallel`) all measure and cut
with these, and they used to reach for them through `statusline`, which imports
`cyclecore` — so the direction the renderers needed was the one that closed a
cycle, worked around with function-local imports.

Two kinds of thing live here, and they are not the same question:

  * `cell_width`/`fit`/`fit_tail`/`pad` — what a string COSTS on screen and how
    to cut it to a given width. Columns, not characters: `rich.cells.cell_len`
    knows about double-width glyphs and emoji, `len()` does not.
  * `terminal_columns`/`line_budget`/`screen_width` — how much width there IS.
"""

import shutil

__all__ = [
    "LEGACY_LINE_COLUMNS",
    "LINE_RIGHT_MARGIN",
    "cell_width",
    "fit",
    "fit_tail",
    "line_budget",
    "pad",
    "screen_width",
    "terminal_columns",
]


try:  # optional, and the only reason rich is touched here
    from rich.cells import cell_len as _cell_len
except ImportError:  # pragma: no cover - exercised only without rich installed
    def _cell_len(text: str) -> int:
        return len(text)


def cell_width(text: str) -> int:
    """Terminal columns `text` occupies (double-width aware when rich is here)."""
    return _cell_len(text)


def fit(text: str, width: int) -> str:
    """`text` cut to at most `width` columns, ending in '…' when something was cut.

    A row that wraps would push the reserved region up by a line and desynchronise
    every subsequent repaint, so truncation is not cosmetic here.
    """
    if width <= 0:
        return ""
    if cell_width(text) <= width:
        return text
    out = []
    used = 0
    for ch in text:
        w = cell_width(ch)
        if used + w > width - 1:
            break
        out.append(ch)
        used += w
    return "".join(out) + "…"


def fit_tail(text: str, width: int) -> str:
    """`text` cut to `width` columns from the LEFT, marked with a leading '…'.

    The mirror image of `fit`, for the one place where the END of a string is
    the part worth showing: a line being typed, whose interesting character is
    the one just entered. Wrapping is not an option there either — the reserved
    region is sized in whole rows.
    """
    if width <= 0:
        return ""
    if cell_width(text) <= width:
        return text
    out = []
    used = 0
    for ch in reversed(text):
        w = cell_width(ch)
        if used + w > width - 1:
            break
        out.append(ch)
        used += w
    return "…" + "".join(reversed(out))


def pad(text: str, width: int) -> str:
    """`text` fitted to exactly `width` columns (truncated, then space-padded)."""
    text = fit(text, width)
    return text + " " * max(0, width - cell_width(text))


# The widest of the fixed limits this measurement replaced (a Claude tool call's
# 200 characters). It serves twice, because both uses answer the same question —
# what a line may hold when the screen is not the constraint:
#
#  * no terminal at all (redirected to a file or a pipe, where the mirror log is
#    the only reader and a cut cannot be undone), rather than the 80-column guess
#    `get_terminal_size` hands out there;
#  * the FLOOR under any real width. Screen and log get the same text, so cutting
#    to a narrow window would shrink what the run recorded — and wrapping is what
#    140-200 characters always did on such a window anyway. So a terminal below
#    this wraps exactly as it did before, and only a wider one gains: nothing
#    records less than the fixed figures kept.
#
# The one line that can still record less is codex's command-plus-output pair,
# which has two variable fields to fit in one budget (see `cyclecore._fit_two`).
LEGACY_LINE_COLUMNS = 200
# The last cell of a row is not ours to fill: a terminal that auto-wraps on it
# turns an exactly-full line into two rows, and rich's own console is one column
# narrower than the terminal on the legacy Windows console (`legacy_windows` in
# `Console.size`), which would wrap every full line by one character.
LINE_RIGHT_MARGIN = 1


def terminal_columns(fallback: int = LEGACY_LINE_COLUMNS) -> int:
    """The terminal's real width in columns, or `fallback` when there is none.

    `shutil.get_terminal_size` reads $COLUMNS first and the real console second,
    so an explicit width set in the environment wins here as it does everywhere
    else. Asked for a fallback of zero rather than its default 80 so that "no
    terminal" is distinguishable from a narrow one — an 80 arrived at by guessing
    and an 80 measured off a screen deserve different answers.
    """
    try:
        columns = shutil.get_terminal_size(fallback=(0, 0)).columns
    except Exception:
        columns = 0
    return columns if columns > 0 else fallback


def line_budget(prefix: str = "") -> int:
    """Columns left on one scrolling line for its variable part, after `prefix`.

    The single source of every compact renderer's truncation width: the
    sequential stream renderer (`cyclecore`) and the per-worker lines
    (`parallel`) both ask this one question.

    For the ORDINARY output above the pinned region, where a line that does not
    fit merely wraps — not for the pinned rows themselves, which are sized by
    `statusline.Terminal` and must never wrap (see `fit`). The compact renderers
    used to cut their one variable field at 140, 160 or 200 characters depending
    on the call site and on which provider produced the event; on a 240-column
    terminal that showed a command cut in half with a third of the screen left
    blank, and on an 80-column one it wrapped the same line across two rows. Both
    are the same defect — a number that was never the screen's — so the fixed
    figures were replaced by this measurement.

    Not below `LEGACY_LINE_COLUMNS`, which is why this is a budget rather than a
    promise to fit: on a narrow terminal the caller is told it may write more
    than one row holds, deliberately (see the constant).

    `prefix` is the fixed head the caller is about to print in front of the
    variable part (indent, glyph, job tag, tool name), measured in cells so the
    double-width glyphs in it count for the two columns they occupy — as is the
    text the answer sizes (`cyclecore._short` cuts by cells too, or a command
    echoing CJK would overflow by one column per character). So pass the prefix
    itself, never its `len`.
    """
    return max(LEGACY_LINE_COLUMNS,
               terminal_columns(0) - cell_width(prefix) - LINE_RIGHT_MARGIN)


def screen_width(default: int = 100, maximum: int = 120) -> int:
    """A sane width for full-width blocks printed into the scroll area."""
    return max(40, min(terminal_columns(default), maximum))
