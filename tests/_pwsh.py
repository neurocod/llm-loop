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
# which is what `cmdline._powershell_word`'s refusals rest on. Windows only, like
# POWERSHELL: the pins paste Windows-shaped lines.
PWSH = shutil.which("pwsh.exe") if os.name == "nt" else None

needs_pwsh = pytest.mark.skipif(
    PWSH is None,
    reason="no pwsh 7 here (the GitHub Windows runner has it): the claim that "
           "7.3+ reads the PowerShell line as 5.1 does goes unmeasured")


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
                   shell: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run one line of PowerShell source that calls a native program, in
    `shell` (POWERSHELL when None; PWSH for pwsh 7).

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
    if shell is None:
        assert POWERSHELL is not None, "mark the test with needs_powershell"
        shell = POWERSHELL
    source = ("$ErrorActionPreference = 'Stop'; " + line
              + "; exit $(if ($null -eq $LASTEXITCODE) { 255 } "
                "else { $LASTEXITCODE })")
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    return subprocess.run([shell, "-NoProfile", "-NonInteractive",
                           "-EncodedCommand", encoded],
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)
