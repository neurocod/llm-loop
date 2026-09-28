"""Windows PowerShell 5.1 as the operator's shell, for pins that type a command.

Imported by name (`from _pwsh import ...`), like `_runfixtures`. Several pins
hand a command to the shell an operator pastes it into, because that shell --
not argparse, not CreateProcess -- is where `''` gets dropped and a quoted
interpreter path stops being a command. One copy of the mechanics, because a
fix made to one of four copies (a closed stdin, an execution policy, the output
codec) reads as a flaky test on whichever machine the other three meet it.

The main repository's `tools/tests/test_replace_in_file_forward.py` imports
this file too, through `sys.path`: keep it free of imports from `llm_loop`.
"""

import base64
import os
import shutil
import subprocess
from typing import Sequence

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


def ps_quote(text: str) -> str:
    """`text` as a PowerShell single-quoted literal: nothing inside expands."""
    return "'" + text.replace("'", "''") + "'"


def invocation(program: Sequence[str], *words: str) -> str:
    """`& 'program' 'args' <words>`: `program` literal, each word PS source.

    `program` is quoted here (an interpreter under "Program Files" included);
    the words are passed through as typed, so `''` in them is PowerShell's own
    empty string -- the one it drops on the way to a native program.
    """
    return " ".join(["&", *map(ps_quote, program), *words])


def run_powershell(line: str, *,
                   timeout: float = TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run one line of PowerShell source; the exit code is the native one's.

    -EncodedCommand (base64 of UTF-16LE) keeps Python's own command-line
    quoting out of what PowerShell parses. stdin is closed: a program that
    reads it when no command reaches it (the gate's hook mode) would otherwise
    wait for input that never comes, until the timeout.
    """
    assert POWERSHELL is not None, "mark the test with needs_powershell"
    source = line + "; exit $LASTEXITCODE"
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive",
                           "-EncodedCommand", encoded],
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout)
