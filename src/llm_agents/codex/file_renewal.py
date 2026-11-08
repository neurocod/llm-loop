"""Renew stale Windows files while preserving bytes and concurrent writes.

Windows only, and safe to import anywhere: every Win32 call is made inside a
function, and off Windows `_owner_sid` is None, so `renew_file` never acts.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Optional


FILE_ATTRIBUTE_READONLY = 0x1
_KEPT_ATTRIBUTES = 0x2 | 0x4  # hidden, system
_RENEW_CHUNK = 1 << 20


def _sid_string(sid: Any) -> Optional[str]:
    import ctypes
    advapi = ctypes.WinDLL("advapi32")
    kernel = ctypes.WinDLL("kernel32")
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p,
                                              ctypes.POINTER(ctypes.c_wchar_p)]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    text = ctypes.c_wchar_p()
    if not advapi.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        return None
    try:
        return text.value
    finally:
        kernel.LocalFree(ctypes.cast(text, ctypes.c_void_p))


def _owner_sid(path: Path) -> Optional[str]:
    """The owner of `path` as a SID string; None off Windows or on failure."""
    if os.name != "nt":
        return None
    import ctypes
    advapi = ctypes.WinDLL("advapi32")
    kernel = ctypes.WinDLL("kernel32")
    advapi.GetNamedSecurityInfoW.restype = ctypes.c_uint32
    advapi.GetNamedSecurityInfoW.argtypes = [
        ctypes.c_wchar_p, ctypes.c_int, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    owner = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    # SE_FILE_OBJECT = 1, OWNER_SECURITY_INFORMATION = 1
    if advapi.GetNamedSecurityInfoW(str(path), 1, 1, ctypes.byref(owner), None,
                                    None, None, ctypes.byref(descriptor)):
        return None
    try:
        return _sid_string(owner)
    finally:
        kernel.LocalFree(descriptor)


def _own_sids() -> frozenset[str]:
    """The owners a file of this process's own can have: its user
    (TokenUser = 1) and the default owner a NEW file gets (TokenOwner = 4).
    They differ in an elevated administrator's process, whose new files are
    owned by BUILTIN\\Administrators, S-1-5-32-544 (windows-latest CI, run as
    runneradmin, 2026-11-08); counting only the user made every file of an
    elevated run look stale. Empty off Windows or on failure."""
    return frozenset(sid for sid in (_token_sid(1), _token_sid(4)) if sid)


def _token_sid(info_class: int) -> Optional[str]:
    """The SID of a TOKEN_USER / TOKEN_OWNER of this process (both start
    with the SID pointer); None off Windows or on failure."""
    if os.name != "nt":
        return None
    import ctypes
    advapi = ctypes.WinDLL("advapi32")
    kernel = ctypes.WinDLL("kernel32")
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    advapi.OpenProcessToken.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                        ctypes.POINTER(ctypes.c_void_p)]
    advapi.GetTokenInformation.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32)]
    token = ctypes.c_void_p()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x8,  # QUERY
                                   ctypes.byref(token)):
        return None
    try:
        size = ctypes.c_uint32()
        advapi.GetTokenInformation(token, info_class, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token, info_class, buffer, size,
                                          ctypes.byref(size)):
            return None
        return _sid_string(ctypes.c_void_p.from_buffer(buffer).value)
    finally:
        kernel.CloseHandle(token)


def _set_attributes(path: Path, attributes: int) -> None:
    import ctypes
    kernel = ctypes.WinDLL("kernel32")
    kernel.SetFileAttributesW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32]
    kernel.SetFileAttributesW(str(path), attributes)


def _digest(path: Path) -> str:
    """sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(_RENEW_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def renew_file(path: Path) -> tuple[bool, str]:
    """Replace a STALE file by a byte-identical copy written as the real
    user: (renewed, why it was kept) — `(False, "")` for a file that needs
    nothing.

    Why (measured on a worker's tree 2026-11-03): under the elevated
    Windows sandbox a file the agent creates is OWNED by the sandbox account,
    and the sandbox may write it only through an ACE for the capability SID
    current at creation. Codex can regenerate those SIDs mid-turn
    (`~/.codex/cap_sid` is rewritten unlocked by every concurrent session —
    rust-v0.164.0 `windows-sandbox-rs/src/cap.rs:48-114`); the new SIDs are
    granted on the workspace root and pushed down by Windows running as the
    real user, who has no WRITE_DAC on sandbox-owned files, so exactly the
    agent's own new files keep the stale SIDs and turn unwritable
    (UnauthorizedAccessException, apply_patch failing on them alone). An
    in-place write outside the sandbox keeps that DACL; a NEW file in the
    same directory moved over it is owned by the real user and inherits the
    directory's current ACEs.

    Repairing the DACL in place is not open to the wrapper (checked
    2026-11-03 on a renewed file of that worker): the real user reaches a
    sandbox-owned file through `Authenticated Users:(M)` only — no
    WRITE_DAC, so no `icacls /reset` — and taking ownership needs an
    elevated token. Replacing needs only DELETE, which Modify includes.

    Only files that are actually stale are renewed: owned by another account
    than this process (`_owner_sid` outside `_own_sids`) -- in the measured
    case a `CodexSandbox*` account, but also e.g. Administrators for a file
    an elevated shell once made under a non-elevated wrapper: the real user
    has no WRITE_DAC on it either, so the new SIDs miss it the same way. A
    file the user owns gets the new capability SIDs pushed down like the
    rest of the tree. Not renewed either: a hard link (the copy would split
    it), a read-only file (the replace fails on it), a file in a directory
    the sandbox account owns (the copy would inherit that stale list; the
    reason says so instead of a "renewed" that changes nothing).

    The copy streams, and a concurrent write is detected rather than lost:
    size and mtime before the copy must match after the file is hashed
    again, and the hash the copy's, or the file is kept and the reason
    named. What remains is the
    window between that check and the rename — microseconds, while the
    agent itself waits on this approval. The copy keeps the file's mtime
    and its hidden/system attributes; the temporary file sits beside it
    (the new ACEs come from the DIRECTORY it is created in — a rename
    keeps a file's own) under a `*~` name git ignores, and is removed on
    every failure. Best effort: the grant stands either way."""
    if not path.is_file() or path.is_symlink():
        return False, ""
    owner, own = _owner_sid(path), _own_sids()
    if owner is None or not own or owner in own:
        return False, ""
    if _owner_sid(path.parent) == owner:
        # A directory the sandbox account created is stale the same way, and
        # the copy made in it inherits ITS access list (review 2026-11-03):
        # replacing would report "renewed" and leave the file as unwritable.
        # Stale means owned by the FILE's owner, the sandbox account — not
        # "anyone but the user": a tree owned by Administrators or
        # TrustedInstaller is not the sandbox's (third review 2026-11-03).
        return False, "its directory is stale too (owned by the file's owner, not this process)"
    try:
        before = os.stat(path)
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if before.st_nlink > 1:
        return False, "a hard link"
    attributes = getattr(before, "st_file_attributes", 0)
    if attributes & FILE_ATTRIBUTE_READONLY:
        return False, "read-only"
    temp = ""
    replaced = False
    try:
        handle, temp = tempfile.mkstemp(dir=path.parent,
                                        prefix=f".{path.name}.renew-",
                                        suffix="~")
        digest = hashlib.sha256()
        with os.fdopen(handle, "wb") as out, open(path, "rb") as source:
            for chunk in iter(lambda: source.read(_RENEW_CHUNK), b""):
                digest.update(chunk)
                out.write(chunk)
        os.utime(temp, ns=(before.st_atime_ns, before.st_mtime_ns))
        stamp = (before.st_size, before.st_mtime_ns)
        hashed = _digest(path)
        # Stat AFTER the hash as well: a write into a chunk the hash had
        # already read leaves the digest equal to the copy's while the file
        # is new (review 2026-11-03 lost `WXYZ` that way with the stat
        # before the hash only).
        after = os.stat(path)
        if (after.st_size, after.st_mtime_ns) != stamp or hashed != digest.hexdigest():
            return False, "changed while being renewed"
        os.replace(temp, path)
        replaced = True
        if attributes & _KEPT_ATTRIBUTES:
            _set_attributes(path, attributes & _KEPT_ATTRIBUTES)
        return True, ""
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        if temp and not replaced:
            try:
                os.unlink(temp)
            except OSError:
                pass
