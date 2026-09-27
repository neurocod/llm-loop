"""The gate's diagnostic CLI, typed the way an operator types it.

`cpp/parity_check.py` hands both halves ready-made argv lists (ARGV_CASES), so
it pins the parsers but takes the shell's behaviour on trust. Here the command
goes through Windows PowerShell 5.1 itself, whose habit of dropping an empty
argument to a native program is the whole reason the contract exists:
`--check=` is the empty command, and `--check ""` -- which arrives as a bare
--check -- is a usage error that names `--check=`, in both implementations,
never a verdict. (The contract and why it is not "bare --check means empty":
ask_user_gate.py, at the --check add_argument.)
"""

import base64
import os
import shutil
import subprocess
import sys

import pytest

PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "claude-plugin", "ask-user-gate")
SCRIPT = os.path.join(PLUGIN, "hooks", "ask_user_gate.py")
BINARY = os.path.join(PLUGIN, "hooks",
                      "ask_user_gate.exe" if os.name == "nt" else "ask_user_gate")
POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None
# 0.18-0.20 s per PowerShell round trip into Python (three runs, measured
# 2026-09-27); the budget only has to tell a hang from a slow box.
TIMEOUT_S = 60

pytestmark = pytest.mark.skipif(
    POWERSHELL is None, reason="Windows PowerShell 5.1 is the shell that drops ''")

IMPLEMENTATIONS = [
    pytest.param([sys.executable, SCRIPT], id="python"),
    pytest.param([BINARY], id="c++", marks=pytest.mark.skipif(
        not os.path.isfile(BINARY),
        reason="the C++ gate is an opt-in build; see cpp/build.py")),
]


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def powershell(gate, *words: str) -> subprocess.CompletedProcess:
    """`& <gate> <words>` in PowerShell 5.1; each word is PowerShell source.

    -EncodedCommand keeps Python's own command-line quoting out of what
    PowerShell parses, so `''` is PowerShell's empty string and nothing else.
    """
    line = " ".join(["&", *map(_ps_quote, gate), *words]) + "; exit $LASTEXITCODE"
    encoded = base64.b64encode(line.encode("utf-16-le")).decode("ascii")
    return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive",
                           "-EncodedCommand", encoded],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=TIMEOUT_S)


@pytest.mark.parametrize("gate", IMPLEMENTATIONS)
@pytest.mark.parametrize("words", [["--check="],
                                   ["--check=", "--shell", "powershell"]],
                         ids=["alone", "before-a-flag"])
def test_check_equals_is_the_empty_command(gate, words):
    result = powershell(gate, *words)
    assert (result.returncode, result.stdout.strip()) == (0, "allowed"), \
        result.stdout + result.stderr


@pytest.mark.parametrize("gate", IMPLEMENTATIONS)
@pytest.mark.parametrize("words", [["--check", "''"],
                                   ["--check", "''", "--shell", "powershell"],
                                   ["--check", "''", "--platform=windows"]],
                         ids=["at-the-end", "before-a-flag", "before-flag=value"])
def test_check_of_a_dropped_empty_is_a_usage_error(gate, words):
    result = powershell(gate, *words)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "allowed" not in result.stdout
    assert "--check=" in result.stderr


@pytest.mark.parametrize("gate", IMPLEMENTATIONS)
def test_help_names_the_empty_spelling(gate):
    result = powershell(gate, "--help")
    assert result.returncode == 0, result.stderr
    assert "--check=" in result.stdout
