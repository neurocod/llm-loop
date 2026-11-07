#!/usr/bin/env python3
"""Cut whole-line ranges out of one file and append them, in order, to another.

What it is for: the mechanical half of splitting a large source file, done by
an LLM agent or a person. A case block of a thousand lines moved by hand
through an editor (or retyped by a model, which may drift a character) is a
thousand lines nobody can review as a MOVE, while a named range moved by a
script is one line of command plus the few edits around it that are the real
change.

    python agent-tools/move_lines.py <src> <dst> 4165-5345 1570-1631 [--dry-run]

Ranges are 1-based and inclusive, numbered against the source AS IT IS NOW
(every range is resolved before any is cut), and appended to `dst` in the order
given -- which need not be file order. `dst` is created when it does not exist.

WHAT IT REFUSES TO DO:

  * overlapping ranges, or one past the end of the source: an error, nothing
    written. "Moved the same line twice" and "silently stopped at EOF" are the
    two ways a line-number tool loses code.
  * `dst` naming the source itself, by any spelling (`a/../src.cpp`, a hard
    link): the append would be written and then overwritten by the cut, so the
    lines would simply be gone.

A moved range ending on the source's last line, when that line has no newline,
gets one in `dst` if anything is appended after it -- else two lines join.

Both files are handled as BYTES, so line endings survive (default text I/O on
Windows rewrites every one of them, which `.gitattributes` then hides from
`git diff`).

Each range is echoed with its first and last line, so the caller sees what the
numbers resolved to before trusting them. Build afterwards: this moves text, and
the compiler is what says the two halves still stand.
"""
import argparse
import os
import pathlib
import re
import sys

RANGE = re.compile(r"^(\d+)-(\d+)$")


class MoveError(Exception):
    """A refusal: nothing was written."""


def parse_ranges(specs: list[str], line_count: int) -> list[tuple[int, int]]:
    """The ranges as 0-based half-open (start, end); raises {@link MoveError}."""
    out: list[tuple[int, int]] = []
    for spec in specs:
        match = RANGE.match(spec)
        if not match:
            raise MoveError(f"range {spec!r} is not FIRST-LAST")
        first, last = int(match.group(1)), int(match.group(2))
        if first < 1 or last < first:
            raise MoveError(f"range {spec!r} is empty or starts before line 1")
        if last > line_count:
            raise MoveError(f"range {spec!r} ends past the source's {line_count} lines")
        out.append((first - 1, last))
    ordered = sorted(out)
    for (_, end), (start, _) in zip(ordered, ordered[1:]):
        if start < end:
            raise MoveError("ranges overlap; nothing written")
    return out


def same_file(src: pathlib.Path, dst: pathlib.Path) -> bool:
    """Whether `dst` is `src` under another name (path aliases, hard links)."""
    if src.resolve() == dst.resolve():
        return True
    try:
        return dst.exists() and os.path.samefile(src, dst)
    except OSError:
        return False


def eol_of(data: bytes) -> bytes:
    return b"\r\n" if b"\r\n" in data else b"\n"


def preview(line: bytes) -> str:
    text = line.decode("utf-8", errors="replace").rstrip("\r\n")
    return text if len(text) <= 70 else text[:67] + "..."


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cut line ranges out of SRC and append them, in order, to DST.")
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("ranges", nargs="+", help="FIRST-LAST, 1-based, inclusive")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would move; write nothing")
    options = parser.parse_args()
    src = pathlib.Path(options.src)
    dst = pathlib.Path(options.dst)
    try:
        if same_file(src, dst):
            raise MoveError(f"destination {dst} is the source itself; nothing written")
        data = src.read_bytes()
        lines = data.splitlines(keepends=True)
        ranges = parse_ranges(options.ranges, len(lines))
    except (OSError, MoveError) as exc:
        print(f"{src}: {exc}")
        return 1
    moved: list[bytes] = []
    for start, end in ranges:
        print(f"  {start + 1}-{end} ({end - start} lines)")
        print(f"    first: {preview(lines[start])}")
        print(f"    last:  {preview(lines[end - 1])}")
        if moved and not moved[-1].endswith(b"\n"):
            moved[-1] += eol_of(data)
        moved.extend(lines[start:end])
    if options.dry_run:
        print(f"would move {len(moved)} line(s) from {src} to {dst}")
        return 0
    cut = set()
    for start, end in ranges:
        cut.update(range(start, end))
    kept = [line for i, line in enumerate(lines) if i not in cut]
    existing = dst.read_bytes() if dst.exists() else b""
    if existing and not existing.endswith(b"\n"):
        existing += b"\r\n" if b"\r\n" in existing else b"\n"
    dst.write_bytes(existing + b"".join(moved))
    src.write_bytes(b"".join(kept))
    print(f"moved {len(moved)} line(s) from {src} to {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
