"""The ask-user-gate plugin's two implementations must not drift apart.

Until this file existed, `cpp/parity_check.py` was named in a README and run by
whoever remembered. That is a convention, not a mechanism: the live gate on a
machine that opted into the binary is a gitignored `.exe`, so editing
`hooks/ask_user_gate.py` leaves a stale binary guarding the session and nothing
says so -- not `git status`, not a test run. Here, a `pytest` does.

The parity case is skipped where the binary was never built, which is every CI
runner and every fresh clone. The script's own `--self-test` is not skipped:
it needs nothing but the checkout, and it is the half that every plugin install
actually runs.
"""

import os
import subprocess
import sys

import pytest

PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "claude-plugin", "ask-user-gate")
SCRIPT = os.path.join(PLUGIN, "hooks", "ask_user_gate.py")
PARITY = os.path.join(PLUGIN, "cpp", "parity_check.py")
BINARY = os.path.join(PLUGIN, "hooks",
                      "ask_user_gate.exe" if os.name == "nt" else "ask_user_gate")


def _run(argv):
    return subprocess.run([sys.executable] + argv, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


def test_reference_self_test():
    """The scanner, the wiring and both branches of the path resolver."""
    result = _run([SCRIPT, "--self-test"])
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not os.path.isfile(BINARY),
                    reason="the C++ gate is an opt-in build; see cpp/build.py")
def test_port_agrees_with_the_reference():
    """Verdict, exit code and refusal text, over the CLI and over hook mode.

    Failure here is not always "the port is wrong": a rule edited in the script
    alone fails it too, and that is the point -- whichever half moved, the pair
    stopped being one gate.
    """
    result = _run([PARITY])
    assert result.returncode == 0, result.stdout + result.stderr
