#!/usr/bin/env python3
"""In-place text replacement in one file -- the checked stand-in for `sed -i`.

Why not `sed -i`: it is silent about the one thing that matters. A pattern that
matches nothing edits nothing and still exits 0, so a script that mutates a
constant and runs a test reports a confident PASS against unmutated code. This
refuses to write unless the number of occurrences is exactly what the caller
said it would be, so "my pattern was wrong" and "the edit landed" are different
outcomes rather than the same one.

Two more differences that matter on this repo:
 - literal by default (--regex opts in), because most edits are a constant or
   an identifier, and a literal needs no escaping of . ( ) [ ] * / $
 - line endings are preserved: a uniformly-CRLF file is written back as CRLF.
   Under --regex the text is normalised to LF for matching first, so ^...$
   anchors work on such a file -- in Python's MULTILINE mode `$` sits before
   the \\n but AFTER the \\r, so anchored patterns silently fail to match a
   CRLF file otherwise. That silent no-match is exactly the failure above.

Runs from anywhere; FILE is resolved against the current directory.

Usage:
  python tools/replace_in_file.py FILE --old TEXT --new TEXT
  python tools/replace_in_file.py FILE --old X --new Y --count 3   # expect 3
  python tools/replace_in_file.py FILE --old X --new Y --count any # any number
  python tools/replace_in_file.py FILE --regex --old '^const A = 1;$' --new 'const A = 0;'
  python tools/replace_in_file.py FILE --old X --new Y --dry-run

To mutate a file, run something, and put it back, use tools/try_patch.py --
it restores even when the command crashes or is interrupted.
"""

import argparse
import re
import sys
from pathlib import Path

CRLF = "\r\n"

# The report prints the changed lines, and on this repo those lines are often
# Cyrillic (docs) or carry box-drawing characters. A Windows console hands
# Python a cp1252 stdout, whose encoder RAISES on them -- after the file has
# already been written. The caller then reads exit 1 and a UnicodeEncodeError
# traceback for an edit that landed, and the obvious response (run it again)
# applies it twice. Printing is not worth failing over: degrade the characters,
# never the exit code.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # not a reconfigurable text stream
        pass


class EditError(Exception):
    """A refusal to write: the file, the pattern or the count is wrong."""


def read_text(path: Path) -> tuple[str, bool]:
    """Return (text, was_uniformly_crlf). Reads bytes so nothing is normalised."""
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise EditError(f"cannot read {path}: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EditError(f"{path} is not UTF-8: {exc}") from exc
    crlf = CRLF in text and text.count("\n") == text.count(CRLF)
    return text, crlf


def apply_replacement(text: str, old: str, new: str, regex: bool,
                      count: "int | None", crlf: bool) -> tuple[str, int]:
    """Return (new_text, hits). `count` None means "any number, but > 0"."""
    body = text.replace(CRLF, "\n") if crlf else text
    if regex:
        try:
            pattern = re.compile(old, re.MULTILINE)
        except re.error as exc:
            raise EditError(f"bad --regex pattern: {exc}") from exc
        hits = len(pattern.findall(body))
        replaced = pattern.sub(new, body)
    else:
        hits = body.count(old)
        replaced = body.replace(old, new)

    if hits == 0:
        raise EditError(
            f"pattern not found, nothing written: {old!r}\n"
            "  (a silent no-op here is what makes `sed -i` dangerous in a "
            "mutate-and-test script)"
        )
    if count is not None and hits != count:
        raise EditError(
            f"expected {count} occurrence(s), found {hits}; nothing written. "
            "Narrow the pattern, or pass --count to say how many you mean."
        )
    if replaced == body:
        raise EditError("the replacement is identical to the original text")
    return (replaced.replace("\n", CRLF) if crlf else replaced), hits


def changed_lines(before: str, after: str, limit: int = 10) -> list[str]:
    """Line-numbered view of what moved, for the caller's log."""
    old_lines = before.replace(CRLF, "\n").split("\n")
    new_lines = after.replace(CRLF, "\n").split("\n")
    out = []
    for i, (a, b) in enumerate(zip(old_lines, new_lines), start=1):
        if a != b:
            out.append(f"  {i}: - {a}")
            out.append(f"  {i}: + {b}")
            if len(out) >= limit * 2:
                out.append("  ...")
                break
    if len(old_lines) != len(new_lines):
        out.append(f"  (line count {len(old_lines)} -> {len(new_lines)})")
    return out


def write_text(path: Path, text: str) -> None:
    try:
        path.write_bytes(text.encode("utf-8"))
    except OSError as exc:
        raise EditError(f"cannot write {path}: {exc}") from exc


def parse_count(value: str) -> "int | None":
    if value == "any":
        return None
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("--count takes a number or 'any'")
    if n < 1:
        raise argparse.ArgumentTypeError("--count must be at least 1")
    return n


def add_edit_arguments(parser: argparse.ArgumentParser) -> None:
    """The edit half of the CLI, shared with try_patch.py."""
    parser.add_argument("--old", required=True, metavar="TEXT",
                        help="text to find (literal unless --regex)")
    parser.add_argument("--new", required=True, metavar="TEXT",
                        help="text to put in its place")
    parser.add_argument("--regex", action="store_true",
                        help="treat --old as a regex (MULTILINE; \\1 in --new)")
    parser.add_argument("--count", type=parse_count, default=1, metavar="N",
                        help="occurrences required, or 'any' (default: 1)")


# No read-edit-write-in-one helper lives here on purpose. The obvious one
# returns the text it read as "the original", which is only true for the FIRST
# edit of a file; a caller stacking two edits on one file would keep the second
# call's `before` -- a text that already carries the first mutation -- and use
# it to "restore". try_patch.py shipped exactly that and left mutations behind
# (see Touched there). Callers that need to put a file back must snapshot it
# once, before any edit, and keep that snapshot per FILE.


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replace text in one file, refusing to write on a "
                    "surprising number of matches.")
    parser.add_argument("file", type=Path, metavar="FILE")
    add_edit_arguments(parser)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, write nothing")
    options = parser.parse_args()

    try:
        before, crlf = read_text(options.file)
        after, hits = apply_replacement(before, options.old, options.new,
                                        options.regex, options.count, crlf)
        if not options.dry_run:
            write_text(options.file, after)
    except EditError as exc:
        print(f"replace_in_file: {exc}", file=sys.stderr)
        return 1

    verb = "would change" if options.dry_run else "changed"
    print(f"{options.file}: {verb} {hits} occurrence(s)")
    for line in changed_lines(before, after):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
