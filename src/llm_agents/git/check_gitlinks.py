#!/usr/bin/env python3
"""Refuse a submodule entry that only half exists.

A directory containing its own `.git` is recorded by `git add` as a GITLINK
(index mode 160000) whether or not anybody meant it to be a submodule -- and
`git add` does NOT write the matching `.gitmodules` stanza. The result is an
index entry no `git submodule` subcommand can resolve:

    fatal: no submodule mapping found in .gitmodules for path 'includes/polypartition'

That is not a warning attached to the offending path: `git submodule` walks
every gitlink and dies on the first unmapped one, so ONE stale entry breaks
`status`, `update`, `foreach` and every script built on them, repo-wide, for
everybody. The repository this grew up in carried exactly that for two years
(a commit about shader warnings re-added a vendored library's directory long
after both halves had been removed) until an agent read the fatal in its own
tool output.

The reverse mismatch -- a `.gitmodules` stanza whose path is not a gitlink --
is reported too. It is the shape left behind by deleting a submodule's line
from the index without editing `.gitmodules`, and it makes `git submodule
update` try to populate a path that no commit points at.

Usage (runs from any directory inside the work tree):
    python -m llm_agents.git.check_gitlinks                # whole index
    python -m llm_agents.git.check_gitlinks -- libs tools  # only these pathspecs
Exit status is 1 when something is out of step, so it works as a pre-commit
hook or a CI step; the kit's `commit.py` calls it on the paths it is about to
commit, which is the point that mistake would have been caught at.
"""

import argparse
import os
import sys
from functools import partial

from ..gitlib import git

git = partial(git, check=False, errors="replace")

# Index mode of a gitlink -- a commit pointer standing in for a subdirectory.
GITLINK_MODE = "160000"


def repo_root(cwd: str = None) -> str:
    proc = git(["rev-parse", "--show-toplevel"], cwd=cwd or os.getcwd())
    if proc.returncode != 0:
        raise SystemExit("error: not inside a git work tree")
    return proc.stdout.strip()


def index_gitlinks(base: str, pathspecs: list[str]) -> list[str]:
    """Paths recorded as gitlinks, repo-relative (`--full-name`), restricted to
    `pathspecs` when given. The pathspecs are read from `base`, the directory
    they were typed in: commit.py run from a subdirectory once had them read
    from the root, where `nested` matched nothing and an accidental nested
    repository went in as a gitlink unchecked (review 2026-11-03). Without
    pathspecs, `base` must be the root, or only its subtree is listed."""
    proc = git(["ls-files", "--stage", "-z", "--full-name", "--", *pathspecs],
               cwd=base)
    if proc.returncode != 0:
        raise SystemExit(f"error: git ls-files failed: {proc.stderr.strip()}")
    out = []
    for entry in proc.stdout.split("\0"):
        if not entry:
            continue
        # "<mode> <sha> <stage>\t<path>"
        meta, _, path = entry.partition("\t")
        if meta.split(" ", 1)[0] == GITLINK_MODE:
            out.append(path)
    return out


def mapped_paths(root: str) -> set:
    """Paths declared in `.gitmodules` (empty when the file is absent)."""
    proc = git(["config", "--file", ".gitmodules", "--get-regexp",
                r"^submodule\..*\.path$"], cwd=root)
    if proc.returncode != 0:
        return set()   # no .gitmodules, or no path keys in it
    return {line.split(" ", 1)[1].strip()
            for line in proc.stdout.splitlines() if " " in line}


def check(pathspecs: list[str], cwd: str = None) -> list[str]:
    """Human-readable problems; empty list means the two halves agree.
    `pathspecs` are relative to `cwd` (default: the current directory)."""
    root = repo_root(cwd)
    declared = mapped_paths(root)
    problems = []
    for path in index_gitlinks((cwd or root) if pathspecs else root, pathspecs):
        if path not in declared:
            problems.append(
                f"{path}: recorded as a submodule in the index (mode "
                f"{GITLINK_MODE}) but absent from .gitmodules. Either add the "
                f"[submodule] stanza, or drop the entry with "
                f"`git rm --cached {path}` if the directory is not meant to be "
                f"a submodule at all.")
    if not pathspecs:
        # Only meaningful over the whole index: a pathspec-scoped run cannot
        # tell "this stanza has no gitlink" from "its gitlink is out of scope".
        present = set(index_gitlinks(root, []))
        for path in sorted(declared - present):
            problems.append(
                f"{path}: declared in .gitmodules but nothing in the index "
                f"points at it. Remove the stanza, or re-add the submodule.")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False,
        description="Check that index gitlinks and .gitmodules agree.")
    parser.add_argument("pathspecs", nargs="*", metavar="PATHSPEC",
                        help="limit the gitlink scan to these paths")
    options = parser.parse_args()
    problems = check(options.pathspecs, cwd=os.getcwd())
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("gitlinks and .gitmodules agree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
