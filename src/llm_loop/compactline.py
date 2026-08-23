"""compactline.py - one event, one line: the shapes both runners print.

Sits between `textwidth` (how much width there is) and the two runners that
paint with it — the sequential stream renderer (`cyclecore`) and the N-worker
one (`parallel`). It imports neither: everything a line's bytes depend on is
here or below.

These shapes existed twice, once with a `[job k] ` tag in front and the write
taken under an output lock, so `parallel` reached across for six of
`cyclecore`'s rendering symbols (half of them private) and every new line shape
had to be written on both sides. Both differences are parameters of
`LineWriter` — the tag is a pair of prefixes, the lock lives inside the emit
callable — so a shape is written once and both runners get it.

Two kinds of thing live here, and they are not the same question:

  * the text a line is built FROM: `collapse`/`short`/`fit_two` cut it to a
    measured width, `esc` makes it safe to put in markup, `describe_tool` turns
    a tool call's JSON into one field, and `tool_line_head`/`mark_line_head`
    name the fixed part printed in front of that field — which is also the part
    the width is measured against;
  * `LineWriter`, which assembles them and hands the finished (plain, markup)
    pair to the sink it was built with.
"""

import json
import re
from typing import Optional

from . import textwidth

__all__ = [
    "BASH_DETAIL_HEAD",
    "LineWriter",
    "TOOL_DETAIL_SEP",
    "collapse",
    "describe_tool",
    "esc",
    "fit_two",
    "mark_line_head",
    "short",
    "tool_line_head",
    "undouble_backslashes",
]


try:  # optional, and the only reason rich is touched here
    from rich.markup import escape as _rich_escape
except ImportError:  # pragma: no cover - exercised only without rich installed
    _rich_escape = None


def esc(text) -> str:
    """Escape Rich markup metacharacters in dynamic text (e.g. a Bash command or
    a file path containing '['). No-op when Rich is unavailable."""
    return _rich_escape(str(text)) if _rich_escape is not None else str(text)


def collapse(text) -> str:
    """One line's worth of `text`: every run of whitespace becomes one space.

    Split from `short` because the two-field split needs to MEASURE a value
    before deciding how much of it to keep, and measuring the raw text would
    count newlines and indentation that are never printed.
    """
    return " ".join(str(text).split())


def short(text, limit: Optional[int] = None) -> str:
    """Single-line version of `text`, cut to `limit` terminal COLUMNS.

    `limit` defaults to a bare terminal line (`textwidth.line_budget()`); a
    caller that prints a head in front of the result passes what that head
    leaves (`LineWriter.budget(head)`) so the two together fill the width
    instead of overflowing it. The cut is delegated to `textwidth.fit`, which
    counts cells: cutting by characters here while the budget was measured in
    columns is how a command echoing CJK produced a 299-character line 580
    columns wide.
    """
    return textwidth.fit(collapse(text),
                         textwidth.line_budget() if limit is None else limit)


def fit_two(budget: int, first, second) -> tuple:
    """Cut two variable fields of ONE line to share its `budget`.

    Both fit — both are kept whole. Only one is oversized — it takes what the
    other leaves, so a short output never strands half a row of blank next to a
    command that was cut to make room for it. Both oversized — half each,
    because the two fields answer different questions (what ran, and why it
    failed) and a line showing only one of them answers half of it.

    That last case is the one place this records less than the fixed 160+160 it
    replaced (100 each at the floor). The alternative is the line those figures
    actually produced: 320 characters wrapped across three rows whatever the
    terminal.
    """
    first, second = collapse(first), collapse(second)
    wide_first = textwidth.cell_width(first)
    wide_second = textwidth.cell_width(second)
    if wide_first + wide_second <= budget:
        return first, second
    half = budget // 2
    if wide_first <= half:
        return first, textwidth.fit(second, budget - wide_first)
    if wide_second <= half:
        return textwidth.fit(first, budget - wide_second), second
    return textwidth.fit(first, half), textwidth.fit(second, budget - half)


_BACKSLASH_RUN_RE = re.compile(r"\\+")


def undouble_backslashes(text: str) -> str:
    r"""Halve uniformly doubled backslashes in a command line, for display only.

    Codex reports the command it ran with every backslash doubled — its own
    escaping, not JSON's (json.loads already removed one level, and the quotes
    in the same value arrive unescaped) — so a Windows path reaches the screen
    and the log as C:\\WINDOWS\\... and cannot be copied without hand-editing.

    Halving is the exact inverse only while EVERY run of consecutive backslashes
    has even length: a UNC \\server\share arrives as four and halves back to its
    correct two. A single odd run means the string was never uniformly doubled,
    so it is returned untouched — showing a raw string beats corrupting one.
    That is why this is not `text.replace("\\\\", "\\")`, which would eat the
    UNC pair and mangle half-escaped strings.

    Apply it to values that are definitionally command lines, before `short`, so
    truncation counts the characters the reader actually sees.
    """
    runs = _BACKSLASH_RUN_RE.findall(text)
    if not runs or any(len(run) % 2 for run in runs):
        return text
    return _BACKSLASH_RUN_RE.sub(lambda m: "\\" * (len(m.group(0)) // 2), text)


# What a Bash tool call's detail puts in front of the command, so the line reads
# as the shell prompt it is.
BASH_DETAIL_HEAD = "$ "

# What stands between a tool-call line's head and its detail. Named because two
# places need it as a value rather than as a literal: the width arithmetic
# (`tool_line_head`) counts it, and the printer drops it when there is no detail.
TOOL_DETAIL_SEP = ": "


def tool_line_head(name: str) -> str:
    """Everything a tool-call line prints in FRONT of its detail.

    Exists for the width arithmetic: `LineWriter.budget(tool_line_head(name))`
    is how much of the detail fits beside it. Kept next to the printer, so a head
    measured somewhere else cannot drift from the head printed; a worker's writer
    prepends its `[job k] ` tag to both (see `LineWriter`).
    """
    return f"  ⚙ {name}{TOOL_DETAIL_SEP}"


def mark_line_head(mark: str) -> str:
    """Everything an outcome line prints in FRONT of its body.

    The `mark` is a ✓/✗ standing for what a step returned, indented under the
    tool call it belongs to. Same reason as `tool_line_head`: the head is both
    measured (`LineWriter.budget(mark_line_head(mark))`) and printed, and the
    two spellings drift silently — the printed line then overflows the row by the
    difference, which is what the width measurement exists to prevent.
    """
    return f"    {mark} "


def describe_tool(name: str, ti: dict, limit: Optional[int] = None) -> str:
    """Short human-readable description of a tool call and its arguments.

    `limit` is what the caller's own line leaves for this text — the tool-call
    line is printed as `<indent>⚙ <name>: <this>`, so only the caller knows how
    wide the head in front of it is (parallel mode adds a `[job k]` tag). Left
    out, the description is cut to a bare terminal line.

    Every branch that echoes a JSON value goes through `short`, and not only for
    the width: these values come out of a provider's JSON, so a `file_path` that
    arrives as a number or a list used to be returned as-is and reach
    `LineWriter.tool`, whose f-string coerced it. A path is exactly as unbounded
    as a command — a Grep pattern once printed a 517-column line — and one call
    fixes both. (`TodoWrite` echoes a count rather than a value, so it needs
    neither.)
    """
    if limit is None:
        limit = textwidth.line_budget()
    if name == "Bash":
        # This head is part of the line too, so the command gets what is left
        # after it rather than the whole budget. Subtracted from the same value
        # that is printed: a literal 2 beside a literal "$ " is a number nothing
        # ties to the string it measures. `len` because it is ASCII — a glyph
        # here would have to be counted in cells (`textwidth.cell_width`).
        command = undouble_backslashes(str(ti.get("command", "")))
        return BASH_DETAIL_HEAD + short(command, limit - len(BASH_DETAIL_HEAD))
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return short(ti.get("file_path", ti.get("notebook_path", "")) or "",
                     limit)
    if name in ("Glob", "Grep"):
        loc = f" in {ti['path']}" if ti.get("path") else ""
        return short(f"{ti.get('pattern', '')}{loc}", limit)
    if name == "Skill":
        return short(ti.get("skill", "") or "", limit)
    if name == "Task" or name == "Agent":
        return short(ti.get("description", ti.get("prompt", "")) or "", limit)
    if name == "TodoWrite":
        todos = ti.get("todos", [])
        return f"{len(todos)} items"
    # fallback: the first meaningful field
    for key in ("url", "query", "description", "prompt"):
        if ti.get(key):
            return short(ti[key], limit)
    return short(json.dumps(ti, ensure_ascii=False), limit) if ti else ""


class LineWriter:
    """The compact line shapes, aimed at one sink and carrying one tag.

    `emit(plain, markup)` takes the two copies of a finished line: the plain one
    for the mirror log, the markup one for the screen. `cyclecore` owns one of
    these with no tag over `print_markup`; `parallel` builds one per worker,
    tagged `[job k] ` and emitting under the output lock, so the lines of N
    workers cannot interleave mid-line. Everything else about a line — where the
    head ends, what the body is cut to, which half is escaped — is the same on
    both sides and is decided here.

    Pass an emit that LOOKS ITS SINK UP rather than one bound to it. The width
    and markup pins replace `cyclecore.print_markup` / `parallel.print_markup`
    to read the plain copy of every line; a writer holding the original function
    would print past them and leave the pins green while they watched a function
    nothing calls.

    The tag is two spellings of one thing, because the markup copy has to escape
    its own brackets (`\\[job 3]`) to survive rich. Both include whatever
    separates the tag from the line — nothing is inserted between them here.
    """

    def __init__(self, emit, tag_plain: str = "", tag_markup: str = ""):
        self.emit = emit
        self.tag_plain = tag_plain
        self.tag_markup = tag_markup

    def budget(self, head: str = "") -> int:
        """Columns left for the variable part of this writer's line, after `head`.

        The tag is as much of the line's furniture as the glyph after it — a
        worker's text cut to a bare terminal line would overflow the row by
        exactly the tag — so it is measured with the head rather than beside it.
        """
        return textwidth.line_budget(self.tag_plain + head)

    def line(self, plain: str, style: Optional[str] = None) -> None:
        """One whole line, optionally in one Rich style.

        The body is escaped on its way into the markup: these lines carry
        commands, paths and error text, and rich reads a '[' in them as the
        start of a style tag. It does not complain — `[job 3]` inside a Grep
        pattern simply vanishes from the screen while the log copy still has it.
        """
        body_markup = esc(plain)
        if style:
            body_markup = f"[{style}]{body_markup}[/]"
        self.emit(f"{self.tag_plain}{plain}", f"{self.tag_markup}{body_markup}")

    def fitted(self, head: str, text, style: Optional[str] = None) -> None:
        """`head`, then as much of `text` as the row leaves beside it.

        The head is named once and used twice from here — measured, then
        printed. Spelled out at the call site instead, the two copies drift the
        moment one of them gains a glyph, and the line overflows by exactly the
        difference without anything failing.
        """
        self.line(head + short(text, self.budget(head)), style)

    def tool(self, name: str, detail="") -> None:
        """A tool-call line: a yellow gear glyph and the bold-yellow tool name,
        followed by the (plain, possibly empty) detail. Multi-segment, so it
        builds markup itself rather than taking a single style; the shared head
        is written once instead of being repeated across the with/without-detail
        forms.
        """
        head_plain = self.tag_plain + tool_line_head(name)
        head_markup = (f"{self.tag_markup}  [yellow]⚙[/] "
                       f"[bold yellow]{esc(name)}[/]{TOOL_DETAIL_SEP}")
        if detail:
            # Joined by f-string, not `+`: a detail is whatever a provider's JSON
            # put in the field, and a `+` against a number raises where the
            # printed line used to say `123`. It escapes the stream renderer and
            # ends the run — in a worker, while the thread still holds a claimed
            # item. See `describe_tool`, which is why one can no longer arrive.
            self.emit(f"{head_plain}{detail}", f"{head_markup}{esc(detail)}")
        else:
            cut = len(TOOL_DETAIL_SEP)
            self.emit(head_plain[:-cut], head_markup[:-cut])

    def tool_use(self, name: str, tool_input: dict) -> None:
        """A tool call straight from a provider event: describe it, then print.

        Both runners had these two steps spelled out at the dispatch site, which
        put the same head in front of the same measurement twice; here the head
        the line prints is the head its detail was sized against, by
        construction.
        """
        self.tool(name, describe_tool(name, tool_input,
                                      self.budget(tool_line_head(name))))

    def mark(self, mark: str, style: str, body: str) -> None:
        """An outcome line: the ✓/✗ in `style`, then the (plain) body beside it.

        Both providers report an outcome this way — a Claude tool_result and a
        codex command_execution completion — and they differ only in what they
        put in the body, so the glyph, the indent and the escaping are decided
        once here.
        """
        self.emit(f"{self.tag_plain}{mark_line_head(mark)}{body}",
                  f"{self.tag_markup}    [{style}]{mark}[/] {esc(body)}")
