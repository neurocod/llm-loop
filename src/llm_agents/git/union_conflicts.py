#!/usr/bin/env python3
"""Resolve merge-conflict hunks in a file by UNION (both sides kept, in a chosen order).

Usage (from any directory):
    python -m llm_agents.git.union_conflicts FILE [--hunk N=SPEC ...] [--default SPEC] [--dry-run]

SPEC is one of: ours, theirs, ours,theirs (default), theirs,ours. Hunks are
numbered from 1 in file order. The tool refuses to write if the file has no
conflict markers, or if a hunk is a diff3-style one (a `|||||||` base section),
which needs a human rather than a union.

Why a tool and not an editor: two branches that both APPEND to a journal or a
backlog produce one conflict per shared insertion point, and the right answer
is almost always "keep both". Deleting the markers by hand over a thousand-line
hunk is where a paragraph goes missing silently. Exit code 1 on any refusal; 0
after writing (or on --dry-run).
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from pathlib import Path

OURS = "<<<<<<< "
SEP = "=======\n"
THEIRS = ">>>>>>> "
BASE = "||||||| "

SPECS = {"ours", "theirs", "ours,theirs", "theirs,ours"}


def resolve(text: str, specs: dict[int, str], default: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    hunk = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(OURS):
            out.append(line)
            i += 1
            continue
        hunk += 1
        ours: list[str] = []
        theirs: list[str] = []
        i += 1
        while i < len(lines) and lines[i] != SEP:
            if lines[i].startswith(BASE):
                raise SystemExit(f"hunk {hunk}: diff3 base section found; resolve by hand")
            ours.append(lines[i])
            i += 1
        if i >= len(lines):
            raise SystemExit(f"hunk {hunk}: no ======= separator")
        i += 1
        while i < len(lines) and not lines[i].startswith(THEIRS):
            theirs.append(lines[i])
            i += 1
        if i >= len(lines):
            raise SystemExit(f"hunk {hunk}: no >>>>>>> terminator")
        i += 1
        spec = specs.get(hunk, default)
        for part in spec.split(","):
            out.extend(ours if part == "ours" else theirs)
    return "".join(out), hunk


def help_text(command: str, precedent: str = "") -> str:
    """Keep the shared rules in the module docstring; hosts supply local context."""
    text = __doc__.replace("python -m llm_agents.git.union_conflicts", command)
    if precedent:
        # Reflow only the tail that grows; preserve the existing help layout.
        prefix, tail = text.split("hunk is where", 1)
        tail = "hunk is where" + " ".join(tail.splitlines())
        tail = tail.replace("silently.", f"silently{precedent}.")
        text = prefix + textwrap.fill(tail, width=80, break_on_hyphens=False) + "\n"
    return text


def main(command: str = "python -m llm_agents.git.union_conflicts",
         precedent: str = "") -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, description=help_text(command, precedent),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--hunk", action="append", default=[], metavar="N=SPEC")
    ap.add_argument("--default", default="ours,theirs", choices=sorted(SPECS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs: dict[int, str] = {}
    for item in args.hunk:
        m = re.fullmatch(r"(\d+)=(ours|theirs|ours,theirs|theirs,ours)", item)
        if not m:
            print(f"bad --hunk {item!r}; want N=SPEC with SPEC in {sorted(SPECS)}", file=sys.stderr)
            return 1
        specs[int(m.group(1))] = m.group(2)

    path = Path(args.file)
    text = path.read_text(encoding="utf-8")
    if OURS not in text:
        print(f"{path}: no conflict markers", file=sys.stderr)
        return 1
    resolved, count = resolve(text, specs, args.default)
    unknown = sorted(n for n in specs if n > count)
    if unknown:
        print(f"{path}: --hunk names hunk(s) {unknown} but the file has {count}", file=sys.stderr)
        return 1
    for n in range(1, count + 1):
        print(f"hunk {n}: {specs.get(n, args.default)}")
    if args.dry_run:
        return 0
    # open(), not write_text(newline=...), which needs Python 3.10.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(resolved)
    print(f"{path}: {count} hunk(s) resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
