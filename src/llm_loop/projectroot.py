"""
projectroot.py - where the project being driven lives, and the one place that
answers it.

The root is the working directory of the host project: every subprocess (git,
the provider CLI) runs with it as cwd, a Driver's relative state/list paths
resolve against it, and the stop sentinel and the mirror log's file name are
both derived from it.

IMPORTANT: it is deliberately *not* the directory these modules live in. The
package is meant to be vendored as a submodule under some host project, so the
code location and the project root are different directories. It defaults to the
process cwd (so a thin wrapper run from the project root "just works") and is
overridden by `set_project_root()`, which both runners call from --project-dir/-C.

A LEAF on purpose: it imports `os` and nothing from this package, so every
module that needs the root can read it directly. That is what this module is
FOR. While the root lived in `cyclecore` (the sequential runner), the two
modules below it could not ask for it without importing a runner, so each kept a
MIRROR of the value with a setter of its own — `console._LOG_PROJECT` /
`set_log_project` and `stopchannel.STOP_FILE` / `set_stop_root` — which
`set_project_root` had to remember to push to. Three globals moved as one, so a
test that pointed a runner at a tmp_path had to repair all three, and a handover
that silently stopped happening left a project logging under, or watching a
sentinel in, whatever directory the process started in. There is one global now,
and the two former mirrors DERIVE from it on read (`console.log_file_path`,
`stopchannel.stop_file_path`), so there is nothing left to push and nothing left
to forget.

`gitpush` reached the same place from the other side and stayed there: it takes
the repository as a PARAMETER (`cwd`), because a git push is about a repository
and not about "the project root, whatever that currently is". Anything with a
real choice of directory should keep doing that; this module is for the callers
that genuinely mean "wherever this run is rooted".
"""

import os
from typing import Optional


# The current project root. Read it through `project_dir()`, never `from …
# import PROJECT_DIR`: a copy taken at import time freezes at the launch
# directory, and a `-C` run would then act on the directory it was launched
# from while it works somewhere else — which looks exactly like a stop file
# nobody obeys and a log filed under the wrong project.
PROJECT_DIR = os.getcwd()


def set_project_root(path: Optional[str]) -> str:
    """Point the engine at the project root (cwd for git/provider CLI, base for
    the stop file, the mirror log's name and relative Driver paths). `path`
    None/empty means "keep the current value" (which defaults to the process
    cwd). Returns the resolved absolute path.

    The runners are single-process, so a module-level singleton set once at
    startup is enough. Setting it is all there is to it: everything derived from
    the root is computed on read, so this call has no second half that could be
    forgotten (see the module header for the mirrors that used to be it).
    """
    global PROJECT_DIR
    if path:
        PROJECT_DIR = os.path.abspath(path)
    return PROJECT_DIR


def project_dir() -> str:
    """The current project root (see set_project_root)."""
    return PROJECT_DIR


# Markers that identify a project root when walking up the directory tree. `.git`
# is matched as either a directory (normal clone) or a file (git worktree /
# submodule), which os.path.exists covers for both.
ROOT_MARKERS = (".git", ".hg", ".svn")


def find_project_root(start: Optional[str] = None) -> Optional[str]:
    """Walk up from `start` (the current working directory by default) until a
    directory containing a VCS marker (`ROOT_MARKERS`) is found, and return it.

    A wrapper that anchors the search to its own file location (rather than the
    process cwd) gets a project root that is independent of where the loop was
    launched from: run it from the repo root, from a subdirectory, or from
    anywhere else and it lands on the same root — so git/provider CLI run there, the
    stop file lives there, and the model loads the root CLAUDE.md the same way
    every time. Returns None if no marker is found up to the filesystem root,
    leaving the engine's default (the current working directory) in place.
    """
    path = os.path.abspath(start if start else os.getcwd())
    while True:
        if any(os.path.exists(os.path.join(path, m)) for m in ROOT_MARKERS):
            return path
        parent = os.path.dirname(path)
        if parent == path:  # reached the filesystem root without a match
            return None
        path = parent
