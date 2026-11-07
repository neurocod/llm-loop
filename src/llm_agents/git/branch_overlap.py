#!/usr/bin/env python3
"""Show how two diverged branches overlap in the files they touched.

Usage (from any directory):
    python -m llm_agents.git.branch_overlap OURS THEIRS [--json]

For the merge base B = merge-base(OURS, THEIRS) it prints:
  * files changed only by OURS, only by THEIRS, and by BOTH (the conflict candidates);
  * renames and deletions THEIRS performed on files OURS touched — the places where
    a naive rebase/merge loses our hunks silently, because git sees a deleted or
    moved path rather than a conflicting one.

Rename detection uses `git diff -M50%` so a heavily refactored file still resolves
to its new home. Paths are repo-relative; exit code is always 0 (reporting, not a gate).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from ..gitlib import git

# How the usage line above spells this command; a host's stand-in passes its
# own spelling to `main`, which substitutes it in the help text.
COMMAND = "python -m llm_agents.git.branch_overlap"


def repo_root() -> str:
    return git(["rev-parse", "--show-toplevel"]).stdout.strip()


def changes(base: str, tip: str) -> dict[str, tuple[str, str | None]]:
    """Map old path -> (status, new path or None) for base..tip with rename detection."""
    out = git(["diff", "--name-status", "-M50%", "-z", base, tip]).stdout
    items = out.split("\0")
    result: dict[str, tuple[str, str | None]] = {}
    i = 0
    while i < len(items):
        status = items[i]
        if not status:
            i += 1
            continue
        if status[0] in "RC":
            old, new = items[i + 1], items[i + 2]
            result[old] = (status[0], new)
            i += 3
        else:
            result[items[i + 1]] = (status[0], None)
            i += 2
    return result


def main(command: str = COMMAND) -> int:
    ap = argparse.ArgumentParser(description=__doc__.replace(COMMAND, command),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ours")
    ap.add_argument("theirs")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = ap.parse_args()

    os.chdir(repo_root())
    base = git(["merge-base", args.ours, args.theirs]).stdout.strip()
    ours = changes(base, args.ours)
    theirs = changes(base, args.theirs)

    ours_paths = set(ours) | {n for _, n in ours.values() if n}
    theirs_paths = set(theirs) | {n for _, n in theirs.values() if n}
    both = sorted(ours_paths & theirs_paths)
    only_ours = sorted(ours_paths - theirs_paths)
    only_theirs = sorted(theirs_paths - ours_paths)

    # Files we touched that the other side moved or deleted (git sees no textual conflict).
    moved: list[tuple[str, str]] = []
    deleted: list[str] = []
    for path in sorted(ours_paths):
        st = theirs.get(path)
        if not st:
            continue
        if st[0] in "RC" and st[1]:
            moved.append((path, st[1]))
        elif st[0] == "D":
            deleted.append(path)

    report = {
        "base": base,
        "counts": {
            "ours": len(ours_paths),
            "theirs": len(theirs_paths),
            "both": len(both),
            "only_ours": len(only_ours),
            "only_theirs": len(only_theirs),
        },
        "both": both,
        "only_ours": only_ours,
        "moved_by_theirs": moved,
        "deleted_by_theirs": deleted,
    }
    if args.json:
        json.dump(report, sys.stdout, indent=1, ensure_ascii=False)
        return 0

    print(f"merge-base {base[:10]}")
    for k, v in report["counts"].items():
        print(f"  {k:12} {v}")
    print(f"\n== touched by BOTH ({len(both)}) ==")
    for p in both:
        o = ours.get(p, ("?", None))[0]
        t = theirs.get(p, ("?", None))[0]
        print(f"  {o}{t}  {p}")
    print(f"\n== ours, MOVED by theirs ({len(moved)}) ==")
    for old, new in moved:
        print(f"  {old} -> {new}")
    print(f"\n== ours, DELETED by theirs ({len(deleted)}) ==")
    for p in deleted:
        print(f"  {p}")
    print(f"\n== only ours ({len(only_ours)}) ==")
    for p in only_ours:
        print(f"  {ours.get(p, ('?', None))[0]}  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
