"""PowerShell as the operator's shell, for pins that type a command.

Windows PowerShell 5.1 (`powershell.exe`) by default; pwsh 7 (`PWSH`) is the
second one a line is pasted into, where it is installed — not on the author's
machine, on the GitHub Windows runner.

Imported by name (`from _pwsh import ...`), like `_runfixtures`. Several pins
hand a command to the shell an operator pastes it into, because that shell --
not argparse, not CreateProcess -- is where `''` gets dropped and a quoted
interpreter path stops being a command. One copy of the mechanics, because a
fix made to one of four copies (a closed stdin, an execution policy, the output
codec) reads as a flaky test on whichever machine the other three meet it.

The main repository's `tools/tests/test_replace_in_file_forward.py` imports
this file too, by its path: keep it free of imports from `llm_loop`.
"""

import base64
import os
import shutil
import subprocess
from typing import Optional, Sequence

import pytest

POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None
# 0.18-0.20 s per round trip into Python on an idle box (three runs), 2.4-6.6 s
# while a parallel suite loads the machine (two runs of three cases), both
# measured 2026-09-27; the budget only has to tell a hang from a slow box.
TIMEOUT_S = 60

needs_powershell = pytest.mark.skipif(
    POWERSHELL is None,
    reason="Windows PowerShell 5.1 is the shell under test (drops '' and "
           "re-parses the pasted line)")

# pwsh 7.3+ passes native arguments properly ($PSNativeCommandArgumentPassing),
# which is what `cmdline._powershell_word`'s refusals rest on; 7.0-7.2 still
# pass them the 5.1 way, so a pwsh older than this is no pwsh for the pins.
PWSH_MIN_VERSION = (7, 3)


def _pwsh_version(path: str) -> Optional[tuple]:
    """(major, minor) of the pwsh at `path`, or None if it does not say."""
    try:
        done = subprocess.run(
            [path, "-NoProfile", "-NonInteractive", "-Command",
             "$PSVersionTable.PSVersion.ToString()"],
            stdin=subprocess.DEVNULL, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=TIMEOUT_S)
        major, minor = done.stdout.strip().split(".")[:2]
        return int(major), int(minor)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


# Windows only, like POWERSHELL: the pins paste Windows-shaped lines.
_PWSH_FOUND = shutil.which("pwsh.exe") if os.name == "nt" else None
_PWSH_VERSION = _pwsh_version(_PWSH_FOUND) if _PWSH_FOUND else None
PWSH = (_PWSH_FOUND if _PWSH_VERSION is not None
        and _PWSH_VERSION >= PWSH_MIN_VERSION else None)

needs_pwsh = pytest.mark.skipif(
    PWSH is None,
    reason=("no pwsh here (the GitHub Windows runner has it)" if not _PWSH_FOUND
            else f"pwsh at {_PWSH_FOUND} is {_PWSH_VERSION}, not 7.3+")
    + ": the claim that 7.3+ reads the PowerShell line as 5.1 does goes "
      "unmeasured")


# PowerShell reads these four as the single quote too, so inside '...' each
# must be doubled like `'` itself or it ends the string. A copy of
# `llm_loop.cmdline._PS_SINGLE_QUOTES`, not an import (see the module
# docstring); test_pwsh_helper.py pins the two equal.
_PS_SINGLE_QUOTES = "'\u2018\u2019\u201a\u201b"


def ps_quote(text: str) -> str:
    """`text` as a PowerShell single-quoted literal: nothing inside expands."""
    return "'" + "".join(ch * 2 if ch in _PS_SINGLE_QUOTES else ch
                         for ch in text) + "'"


def invocation(program: Sequence[str], *words: str) -> str:
    """`& 'program' 'args' <words>`: `program` literal, each word PS source.

    `program` is quoted here (an interpreter under "Program Files" included);
    the words are passed through as typed, so `''` in them is PowerShell's own
    empty string -- the one it drops on the way to a native program.
    """
    return " ".join(["&", *map(ps_quote, program), *words])


def run_powershell(line: str, *, timeout: float = TIMEOUT_S,
                   shell: str = "5.1") -> subprocess.CompletedProcess:
    """Run one line of PowerShell source that calls a native program, in
    `shell`: "5.1" for POWERSHELL, "7" for PWSH (pwsh 7.3+). A name, not a
    path, so a missing pwsh is an assertion naming the mark to add rather
    than a quiet run in 5.1.

    The exit code is the native program's. Anything else is a failure, never
    0: a native program that never ran (the name not found) or a failing
    cmdlet stops the line with exit 1 ($ErrorActionPreference = 'Stop'), and
    a line that ran no native program at all exits 255 rather than
    `exit $null`, which is 0 (all three measured 2026-09-28 on 5.1).

    Native-command-oriented in its output too: a native program's stdout and
    stderr reach the caller as the bytes it wrote, but text written by
    PowerShell itself - a cmdlet's output, or its own error, which arrives on
    stderr as a `#< CLIXML` document - is encoded in the console code page
    (`Write-Output 'e-acute'` came back as U+FFFD, measured 2026-09-28).
    Setting [Console]::OutputEncoding here would repair that by switching the
    code page of the console powershell.exe shares with the test runner.

    -EncodedCommand (base64 of UTF-16LE) keeps Python's own command-line
    quoting out of what PowerShell parses. stdin is closed: a program that
    reads it when no command reaches it (the gate's hook mode) would otherwise
    wait for input that never comes, until the timeout.
    """
    if shell == "5.1":
        assert POWERSHELL is not None, "mark the test with needs_powershell"
        program = POWERSHELL
    elif shell == "7":
        assert PWSH is not None, "mark the test with needs_pwsh"
        program = PWSH
    else:
        raise ValueError(f"shell is '5.1' or '7', not {shell!r}")
    source = ("$ErrorActionPreference = 'Stop'; " + line
              + "; exit $(if ($null -eq $LASTEXITCODE) { 255 } "
                "else { $LASTEXITCODE })")
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    return subprocess.run([program, "-NoProfile", "-NonInteractive",
                           "-EncodedCommand", encoded],
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)
