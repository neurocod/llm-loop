"""Where the worker keeps its state, and what may name it.

The leaf of the package: nothing here imports a sibling, so every other module
can depend on it without a cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from functools import partial
from pathlib import Path
from subprocess import SubprocessError

from .. import host_policy
from ..gitlib import git

# Every failure of this one call — no Git, no work tree, a hung Git — is
# answered the same way by `invocation_root` (keep the cwd), so the runner
# raises and the caller catches rather than each branch being read here.
git = partial(git, timeout=10)

STATE_DIR_NAME = ".codex-agents"
STATE_DIR_ENV = "CODEX_AGENTS_DIR"


# Names Win32 refuses to give a directory, with or without an extension:
# `NUL.txt` is the device, not a file. A record under one of these is claimed in
# the registry and then has nowhere to keep its log.
WINDOWS_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{digit}" for digit in range(1, 10)]
    + [f"LPT{digit}" for digit in range(1, 10)])


def agent_name(value: str) -> str:
    """Validate a name that becomes a directory under `.codex-agents/`.

    argparse type, so a bad name is rejected before any process starts rather
    than after the model work is paid for. Names travel into paths, so `..` and
    separators are refused outright.

    The last two checks are the same collision `Registry.reserve` refuses for
    names differing only in case, arriving by other doors: Win32 STRIPS a
    trailing dot from a path component, so `rev.` and `rev` are two registry
    keys sharing one directory and one `last.md`, and a device name is a
    directory that cannot exist at all. Refused here rather than in `reserve`
    because a name that cannot become a path is not a naming clash — there is
    nothing for the operator to rename it away from.
    """
    if not value or not all(ch.isalnum() or ch in "-_." for ch in value) \
            or value.strip(".") == "":
        raise argparse.ArgumentTypeError(
            f"bad agent name {value!r}: use letters, digits, '-', '_', '.'")
    if value.endswith("."):
        raise argparse.ArgumentTypeError(
            f"bad agent name {value!r}: Windows drops the trailing dot, so this "
            f"would share {value.rstrip('.')!r}'s files")
    if value.split(".")[0].upper() in WINDOWS_DEVICE_NAMES:
        raise argparse.ArgumentTypeError(
            f"bad agent name {value!r}: {value.split('.')[0]!r} is a reserved "
            f"Windows device name and cannot be a directory")
    return value


def host_root() -> Path:
    """The checkout the host's entry point names (`host_policy.installed`, per call), so
    the package runs from any cwd and keeps its state beside that entry
    point; without a host, the caller's (`invocation_root`), which
    `host_policy.announce_empty` says.

    Named by the host, not counted from this file's depth: a count
    (`parents[2]`, as it once was) relocates the state directory silently
    when the file moves, and the package is meant to move. The agents' default
    cwd is `invocation_root()` either way.
    """
    return host_policy.root() or invocation_root()


def invocation_root() -> Path:
    """The Git worktree containing the caller's cwd, or the cwd itself.

    The package can be invoked by an absolute path from another worktree.  In
    that case ``host_root()`` names the checkout containing the entry point,
    not the checkout the caller intends to work in.  Git supplies the right root
    for both ordinary checkouts and linked worktrees; outside Git, preserving
    the current directory keeps the command useful for non-repository tasks.
    """
    cwd = Path.cwd().resolve()
    try:
        found = git(["-C", str(cwd), "rev-parse", "--show-toplevel"]).stdout.strip()
    except (OSError, SubprocessError):
        return cwd
    return Path(found).resolve() if found else cwd


def state_dir() -> Path:
    """Where the registry and the per-agent logs live.

    `CODEX_AGENTS_DIR` overrides it — that is what the tests use to keep their
    writers off the real registry, and it also lets a run keep its state outside
    the checkout.

    Without the override the registry belongs to the host root (`host_root`):
    the checkout of the entry point the caller ran, not `--cwd` — an agent
    spawned through a slot's entry point lives in `<slot>/.codex-agents`,
    so `status`/`logs`/`result` must be called through the same path
    (measured 2026-09-13).
    """
    override = os.environ.get(STATE_DIR_ENV)
    return Path(override) if override else host_root() / STATE_DIR_NAME


def scratch_dir(name: str) -> Path:
    """The workspace handed to a `scratch-write` agent — see `sandbox.thread_params`
    — and the second writable root of a `workspace-write` one (`sandbox.scratch_root`).

    Keyed by the REGISTRY it belongs to as well as the name: the name is unique
    only within one registry, so two checkouts each spawning `rev` would
    otherwise be handed the same writable directory — one agent's files
    appearing inside another's workspace, in the one mode whose whole promise is
    isolation.

    Under the system temp directory, deliberately OUTSIDE the checkout. Two
    reasons, and the second is the expensive one:

      * the point of the mode is that the reviewed tree is not writable, and
        `workspace-write` makes exactly the workspace root writable — so the
        root must not be inside it, `.codex-agents/<name>/` included;
      * sandbox provisioning writes an `.agents` directory into the workspace
        root and hands it to a `CodexSandbox*` user. A directory of that kind
        inside the repo is what broke `workspace-write` for the whole checkout
        once already (see the `--sandbox` flag), and one we could no longer
        delete would break it again.
    """
    registry_key = hashlib.blake2s(
        str(state_dir().resolve()).casefold().encode("utf-8"),
        digest_size=4).hexdigest()
    return Path(tempfile.gettempdir()) / "codex_agent" / f"{name}-{registry_key}"
