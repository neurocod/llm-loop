"""Writable and protected paths, including Windows path spellings."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

# --- paths -----------------------------------------------------------------

# Path runs no agent writes, sandbox or not: the host policy's
# `protected_parts` (git's own store at least, `host_policy.GIT_STORE`).
# Matched as whole components anywhere in the path, case-folded, so
# `.gitignore` is not `.git` and a linked worktree's `.git` FILE is.
Protected = Iterable[tuple[str, ...]]


def protected_part(path: Path, protected_parts: Protected) -> str:
    """The `protected_parts` entry `path` lies in, as `a/b`, or "".

    Windows drops trailing dots and spaces from a path component, so
    `.git.` and `.git ` open `.git` (review 2026-11-03: both read as plain
    names while `.git` itself did not exist yet) — compared without them."""
    parts = [part.rstrip(" .").casefold() for part in path.parts]
    for protected in protected_parts:
        width = len(protected)
        for start in range(len(parts) - width + 1):
            if tuple(parts[start:start + width]) == protected:
                return "/".join(protected)
    return ""


def within(path: Path, roots: Iterable[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


class PathRefused(ValueError):
    """A path the judges will not compare with a root; the message says why."""


_LONG_PREFIX = re.compile(r"^[\\/]{2}\?[\\/]", re.IGNORECASE)
_LONG_UNC = re.compile(r"^[\\/]{2}\?[\\/]UNC[\\/]", re.IGNORECASE)
_DEVICE_PREFIX = re.compile(r"^[\\/]{2}\.[\\/]")
_DRIVE = re.compile(r"^[A-Za-z]:")
_URL = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]+://")


def _without_long_prefix(raw: str) -> str:
    """`\\\\?\\C:\\x` -> `C:\\x`, `\\\\?\\UNC\\s\\x` -> `\\\\s\\x`."""
    if _LONG_UNC.match(raw):
        return "\\\\" + raw[8:]
    if _LONG_PREFIX.match(raw):
        return raw[4:]
    return raw


def resolve_path(raw: str, base: Path) -> tuple[Path, Path]:
    """`raw` as (lexical absolute path, resolved path), against `base` when
    relative; raises `PathRefused` for what the judges do not compare.

    `\\\\?\\` prefixes are taken off (`\\\\?\\UNC\\s\\x` is `\\\\s\\x`), so
    both spellings of one path meet the same root; the device namespace
    (`\\\\.\\`) is refused, and so is a `\\\\?\\` path that names no drive
    letter or share after the prefix: `\\\\?\\GLOBALROOT\\Device\\...` and
    `\\\\?\\Volume{GUID}\\...` reach a volume by its device, and with the
    prefix gone they read as RELATIVE names inside the tree (review
    2026-11-03 granted both). A `:` after the drive designator names an NTFS
    alternate data stream — `.git:review-stream` writes into `.git` while
    reading as a file beside it (review 2026-11-03) — and a drive with no
    separator after it (`C:x`) is relative to that drive's own cwd, which the
    judge cannot see: both refused.

    RESOLVED, because a junction or symlink is a second way in: a link inside
    the tree that points elsewhere is elsewhere, and the lexical string looks
    like the tree (`_refuse_project_inside_workspace` has the same reason).
    The lexical path is returned too, for `protected_part`: a path that only
    LOOKS protected is refused as well.
    """
    if _URL.match(raw):
        raise PathRefused(f"{raw} is a URL, not a path")
    if _DEVICE_PREFIX.match(raw):
        raise PathRefused(f"{raw} is a device path")
    prefixed = bool(_LONG_PREFIX.match(raw))
    raw = _without_long_prefix(raw)
    drive = _DRIVE.match(raw)
    if prefixed and not (drive or raw.startswith("\\\\")):
        raise PathRefused(f"{raw} after `\\\\?\\` names a volume by its "
                          f"device, not by a drive letter or share")
    rest = raw[2:] if drive else raw
    if ":" in rest:
        raise PathRefused(f"{raw} names an alternate data stream (a `:` after "
                          f"the drive)")
    if drive and not rest.startswith(("\\", "/")):
        raise PathRefused(f"{raw} is relative to a drive's own working "
                          f"directory")
    try:
        joined = base / raw
        return Path(os.path.normpath(joined)), joined.resolve()
    except (OSError, ValueError) as exc:
        raise PathRefused(f"{raw} cannot be resolved ({exc})") from None


def path_refusal(raw: str, base: Path, roots: Iterable[Path],
                 protected_parts: Protected) -> str:
    """Why `raw` may not be written outside the sandbox, or "": it must
    resolve (`resolve_path`) inside `roots` and lie, lexically and resolved,
    outside `protected_parts`. Shared by the edit judge and every path a
    worker's command names."""
    try:
        lexical, resolved = resolve_path(raw, base)
    except PathRefused as refused:
        return str(refused)
    protected = (protected_part(lexical, protected_parts)
                 or protected_part(resolved, protected_parts))
    if protected:
        return f"{raw} is under {protected}/, which agents never write"
    if not within(resolved, roots):
        return f"{raw} is outside your writable roots"
    return ""
