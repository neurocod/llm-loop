"""The git subprocess runner the bridges (and a host's own git tools) share.

One runner rather than a `subprocess.run(["git", ...])` per caller, so the
hardening below — UTF-8 path decoding, no credential prompt on an automatic
push, a bounded push — is written once and every caller keeps its own
defaults through `functools.partial`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal, Optional, Union


class Fail(SystemExit):
    """Exit 1 with `error: <message>` on stderr (printed by the interpreter, once)."""

    def __init__(self, message: str):
        super().__init__(f"error: {message}")


# Bound on an automatic `git push`, s: a push of ~100 commits took ~5 s on
# 2026-09-06, so this is well over twenty times the measured cost — enough for a
# slow link, short enough that an auth stall does not eat a hook's whole budget
# (`terminal_prompt=False` is the other half of that guarantee).
PUSH_TIMEOUT_S = 120


def git(args: list[str], *, cwd: Union[str, Path, None] = None,
        check: bool = True, timeout: Optional[float] = None,
        capture_output: bool = True, text: bool = True,
        encoding: Optional[str] = "utf-8", errors: Optional[str] = None,
        terminal_prompt: Optional[bool] = None,
        failure: Literal["raise", "exit", "command"] = "raise",
        ) -> subprocess.CompletedProcess:
    """Run Git arguments (without the executable) and return CompletedProcess.

    `terminal_prompt=False` overlays GIT_TERMINAL_PROMPT=0 on the inherited
    environment: automatic pushes must not ask for credentials. Combined with
    `timeout=PUSH_TIMEOUT_S` (120 s; pushing about 100 commits took about 5 s,
    measured 2026-09-06), this prevents an authentication stall from eating a
    hook's budget and aborting an agent's start. None preserves the inherited
    environment and the existing unbounded timeout of other callers.

    UTF-8 decodes Git's path bytes independently of the Windows locale: cp1252
    can raise UnicodeDecodeError on Cyrillic filenames in a subprocess reader
    thread, leaving stdout as None. `encoding=None` retains the locale codec;
    `errors="replace"` gives tolerant decoding to callers that only scan the
    output. `text=False, encoding=None` keeps paths and blobs as bytes, without
    newline translation or decoding.

    `check=False` returns failed commands for the caller to inspect. Otherwise
    `failure` picks the CLI contract: "raise" uses subprocess's
    CalledProcessError, "exit" raises Fail with command, cwd and captured output,
    and "command" replays output to stderr before SystemExit. "exit" also turns
    TimeoutExpired into Fail even when check=False, so a timed-out automatic
    push can report a warning while keeping its local result. Capture can be
    disabled for commands whose output belongs directly on the console.
    """
    env = None if terminal_prompt is None else {
        **os.environ, "GIT_TERMINAL_PROMPT": "1" if terminal_prompt else "0",
    }
    try:
        proc = subprocess.run(
            ["git", *args], cwd=str(cwd) if cwd else None, env=env,
            capture_output=capture_output, text=text, encoding=encoding,
            errors=errors, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if failure == "exit":
            raise Fail(f"git {' '.join(args)} exceeded {timeout} s")
        raise
    if check and proc.returncode != 0:
        # `or ""`: an uncaptured run (`capture_output=False`) has both streams
        # as None, and a diagnosis must not turn into a TypeError on the way
        # out (review 2026-09-13).
        if failure == "exit":
            raise Fail(f"git {' '.join(args)} failed in {cwd or os.getcwd()}:\n"
                       f"{proc.stdout or ''}{proc.stderr or ''}".rstrip())
        if failure == "command":
            sys.stderr.write((proc.stdout or "") + (proc.stderr or ""))
            raise SystemExit(f"command failed ({proc.returncode}): git {' '.join(args)}")
        proc.check_returncode()
    return proc
