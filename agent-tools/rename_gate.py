#!/usr/bin/env python3
"""Post-rename commit gate: classify every old-name hit before committing.

Run after renaming or deleting a symbol by any means, on the OLD name. Typed
searches -- the compiler, an LSP, an IDE -- are blind to references made
through a string: `obj[k]()`, `page.evaluate("...")` in browser test scripts,
`getMeshByName`, Qt's `SIGNAL()`/`invokeMethod("...")`. Run this afterwards,
not before: every static reference has already been rewritten, so whatever
still shows up is exactly what a typed search could not reach.

Deliberately has no pathspec filter: string-borne references can live outside
the edited files, so the repository-wide run is always required. For plain
searching and for classifying its hits (with optional -g narrowing), use
find_refs.py, which shares the implementation and takes the same --repo,
--aside and --exclude.

Covers submodules -- a rename must answer for them too. That took two
`git grep` passes to arrange; find_refs.py explains why, and what one hole is
left (an untracked file inside a submodule).

Exit code: 0 means no blocking hits; 1 means hits require classification
before committing, as described below; >1 means the search failed and must be
repaired and rerun. Hits under an --aside prefix are listed but do not block.
This is the inverse of grep: hits mean 1.

Exit 1 is "read every one of these", not "the rename is unfinished" -- the
listing includes the hits you meant to keep, and two shapes of intentional hit
recur. One is an old name that is also an ordinary word or a repo-wide per-file
spelling (measured 2026-09-14 in one repository: `stand` 3699 hits,
`makeDriver` 29, and in neither run was one of them the symbol that had just
moved). The other is a name surviving only in prose that describes the rename.
Both are answered the same way: say in the commit or the handoff which hits are
which. Committing after exit 1 is allowed only when every remaining hit has
been established as intentional or unrelated to the renamed symbol; fix
genuine surviving uses and rerun the gate first. Where the old name was a
FILE-LOCAL binding, no other file could reach it by any spelling, so the run
that carries the verdict is `find_refs.py -g <the files you changed>` and the
wide gate is there for the record. Never land on a listing nobody read: a
genuine string-borne hit hides in noise exactly as well as noise does.

Usage:
  python rename_gate.py oldFunctionName
  python rename_gate.py --substring OLD_PREFIX_
  python rename_gate.py nameA nameB
"""
import importlib.util
import sys
from pathlib import Path


def _find_refs():
    """find_refs.py beside this file, loaded BY PATH: the directory need not
    be on sys.path, and a module of the same name elsewhere cannot win."""
    path = Path(__file__).resolve().with_name("find_refs.py")
    spec = importlib.util.spec_from_file_location("_rename_gate_find_refs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    find_refs = _find_refs()
    parser = find_refs.build_parser(
        "Gate a rename: classify every old-name hit before committing; "
        "see this file's docstring for exit codes and intentional hits.",
        with_glob=False,
    )
    options = parser.parse_args()
    return find_refs.run(options.names, options.substring, options.all, [],
                         gate=True, repo=options.repo, asides=options.aside,
                         excludes=options.exclude)


if __name__ == "__main__":
    sys.exit(main())
