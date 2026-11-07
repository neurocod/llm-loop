#!/usr/bin/env python3
"""Whole-repo text search for a bare identifier.

Wraps `git grep` so the flags are not re-derived each time:
 -w   whole word, so `foo` does not match `foobar` (drop it with --substring)
 -I   skip binaries, which also means Git LFS pointer files
 --untracked   a brand-new script is not staged yet but still calls you
 and .gitignore is honoured for free, keeping node_modules/dist/logs out.

TWO passes, because git refuses to do it in one: `--untracked` and
`--recurse-submodules` are mutually exclusive ("fatal: --untracked not
supported with --recurse-submodules"), and dropping either one buys a silent
hole. Without `--recurse-submodules` a submodule is a gitlink, so `git grep`
walks straight past its contents -- measured 2026-08-24: an earlier one-pass
version printed "no live references" for a name a submodule held 24 times.
For a script whose whole job is to catch what a typed search cannot see,
answering "clean" about a directory it never opened is the worst failure it
has. So: pass 1 tracked+untracked in the superproject, pass 2 tracked
everywhere including submodules, both started at once, union. What still
escapes is an UNTRACKED file inside a submodule -- git offers no way to reach
it, and this is the note saying so rather than a gap someone has to
rediscover.

The search is always rooted at the top level of `--repo` (default: the
repository holding the current directory), not at the current directory.

A host repository passes its own layout as arguments, usually from a small
wrapper script of its own:
  --aside PREFIX=WHAT   hits under PREFIX are listed and counted as WHAT, but
                        never block: notes that describe the past, vendored
                        code whose names are not yours
  --exclude GLOB        a git glob pathspec never searched: stale copies of
                        files, never a reference

This is the search half of a two-script pair sharing one implementation:
  find_refs.py     search -- hits are information, the exit code stays 0
                   (>1 only when git grep itself failed or no repo was found)
  rename_gate.py   post-rename commit gate -- see its docstring

Usage:
  python find_refs.py someName
  python find_refs.py --substring PREFIX_    # partial/concatenated names
  python find_refs.py nameA nameB            # union of several names
  python find_refs.py -g src/render drawCar  # narrow to a subtree
  python find_refs.py -g '*.ts' handleWheel  # ... or to a file type
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

# Exit code of a search that could not run at all (no git, no repository):
# not a finding's 1, and not git grep's own >1 either.
NO_REPO_EXIT = 2


def _aside(text: str) -> tuple[str, str]:
    prefix, sep, what = text.partition("=")
    if not sep or not prefix or not what:
        raise argparse.ArgumentTypeError(f"expected PREFIX=WHAT, got {text!r}")
    return prefix, what


def build_parser(description: str, with_glob: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "names", nargs="+", metavar="NAME",
        help="identifier(s) to look for; several names are OR-ed",
    )
    parser.add_argument(
        "--substring", action="store_true",
        help="match inside longer words too (default is whole-word)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="list the non-blocking (--aside) hits in full instead of just counting them",
    )
    parser.add_argument(
        "--repo", metavar="DIR", default=None,
        help="search the repository holding DIR (default: the current directory's); "
             "a later --repo overrides an earlier one",
    )
    parser.add_argument(
        "--aside", action="append", type=_aside, default=[], metavar="PREFIX=WHAT",
        help="repeatable: hits whose path starts with PREFIX are listed as WHAT and never block",
    )
    parser.add_argument(
        "--exclude", action="append", default=[], metavar="GLOB",
        help="repeatable: a git glob pathspec left out of the search",
    )
    if with_glob:
        parser.add_argument(
            "-g", "--glob", action="append", metavar="PATHSPEC", default=[],
            help="limit the search to a git pathspec (repeatable): a subtree "
                 "'src/render' or a wildcard '*.ts'",
        )
    return parser


def repo_top(start: str | None) -> str | None:
    """The top level of the repository holding <start> (the cwd when None), or
    None after printing why there is none. A missing `git` is answered like a
    missing work tree, not with a traceback: whoever reads the output wants the
    tool and the fix named, which `FileNotFoundError` from a subprocess is not."""
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=start,
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", check=False)
    except OSError as exc:
        print(f"find_refs: cannot run git to locate the repository: {exc}", file=sys.stderr)
        return None
    if out.returncode != 0:
        print("find_refs: not inside a git work tree: " + out.stderr.strip(), file=sys.stderr)
        return None
    return out.stdout.strip()


def run(names, substring: bool, list_all: bool, globs, gate: bool, *,
        repo: str | None = None, asides: Sequence[tuple[str, str]] = (),
        excludes: Sequence[str] = ()) -> int:
    # A hit in a document may be in any script, and a Windows console defaults
    # stdout to a code page that cannot encode most of them: UnicodeEncodeError
    # mid-listing, measured on Cyrillic hits under cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Search from the top level whatever the caller's cwd is: git grep would
    # otherwise cover only the subtree below it, and print paths relative to
    # it, which would also break the --aside prefix match below.
    root = repo_top(repo)
    if root is None:
        return NO_REPO_EXIT

    pattern_args = []
    for name in names:
        pattern_args += ["-e", name]
    word = [] if substring else ["-w"]
    # Pathspecs go after the `--` separator, so a name and a path can never be
    # confused for one another.
    paths = ["--", *globs, *(f":(exclude,glob){glob}" for glob in excludes)]

    def grep(scope_flag: str) -> subprocess.Popen:
        return subprocess.Popen(
            ["git", "grep", "-n", "-I", scope_flag, *word, *pattern_args, *paths],
            cwd=root,
            text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    # Both passes are started before either is read. Each is a tree walk git
    # already spreads over threads, so side by side they mostly contend for
    # the same cores and disk: measured 2026-09-07, 394 ms + 242 ms alone,
    # 704 ms in sequence, 589 ms overlapped -- a sixth off, not a half.
    # The union below does not care which pass answered first.
    passes = [grep(flag) for flag in ("--untracked", "--recurse-submodules")]
    matched = []
    seen = set()
    # git grep: 0 = matched, 1 = no match, >1 = real failure. A failed pass is
    # returned, never swallowed: a partial answer that reads like a whole one is
    # the exact defect the two-pass shape exists to remove. Every pass is
    # waited for even after a failure, so no grep is left running unread.
    outputs = [proc.communicate() for proc in passes]
    for proc, (stdout, stderr) in zip(passes, outputs):
        if proc.returncode > 1:
            sys.stderr.write(stderr)
            return proc.returncode
    for stdout, _ in outputs:
        for line in stdout.splitlines():
            if line not in seen:
                seen.add(line)
                matched.append(line)

    live = []
    aside_hits: dict = {aside: [] for aside in asides}
    for line in matched:
        for aside in asides:
            if line.startswith(aside[0]):
                aside_hits[aside].append(line)
                break
        else:
            live.append(line)

    for line in live:
        print(line)

    for (prefix, what), lines in aside_hits.items():
        if not lines:
            continue
        print(f"\n{len(lines)} hit(s) in {prefix} ({what}) -- not blocking:")
        for line in lines if list_all else lines[:3]:
            print(f"  {line}")
        if not list_all and len(lines) > 3:
            print(f"  ... and {len(lines) - 3} more (--all to list)")

    scope = f" within {' '.join(globs)}" if globs else ""
    if not live:
        aside = [what for (_, what), lines in aside_hits.items() if lines]
        if aside:
            print(f"\nclean{scope}: nothing outside {' and '.join(aside)}")
        else:
            print(f"clean{scope}: no live references")
        return 0
    if not gate:
        # Search mode: report, do not judge. Exit 0 so a successful search
        # does not read as a failed command in a shell chain or a tool runner.
        print(f"\n{len(live)} hit(s){scope}")
        return 0
    print(f"\n{len(live)} live hit(s) -- check each is intentional before committing")
    return 1


def main() -> int:
    parser = build_parser(
        "Find every textual mention of a name across the repo.", with_glob=True,
    )
    options = parser.parse_args()
    return run(options.names, options.substring, options.all, options.glob,
               gate=False, repo=options.repo, asides=options.aside,
               excludes=options.exclude)


if __name__ == "__main__":
    sys.exit(main())
