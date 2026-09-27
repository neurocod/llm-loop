"""The ask-user-gate plugin's two implementations must not drift apart.

Until this file existed, `cpp/parity_check.py` was named in a README and run by
whoever remembered. That is a convention, not a mechanism: the live gate on a
machine that opted into the binary is a gitignored `.exe`, so editing
`hooks/ask_user_gate.py` leaves a stale binary guarding the session and nothing
says so -- not `git status`, not a test run. Here, a `pytest` does.

The parity case is skipped where the binary was never built, which is every CI
runner and every fresh clone. A binary older than its source FAILS it instead:
parity_check refuses to compare yesterday's port (newer_sources there). The
script's own `--self-test` is not skipped: it needs nothing but the checkout,
and it is the half that every plugin install actually runs.
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


def test_reference_check_file_it_cannot_read_is_exit_2(tmp_path):
    """Not 1: exit 1 is "denied", and an unread file was never judged.

    Pinned here as well as in parity_check's ARGV_CASES because that half only
    runs where the binary was built.
    """
    result = _run([SCRIPT, "--check-file", str(tmp_path / "missing.txt")])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "cannot read" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("token, verdict", [
    ("-1", True), ("-.5", True), ("-2.5", True),
    ("-1e5", False), ("-1abc", False), ("-1.2.3", False), ("-.5x", False),
    ("-1_000", False), ("-5\n", False), ("-٥", False), ("-= x", False),
])
def test_reference_negative_number_rule_is_its_own(token, verdict):
    """`--check TOKEN`: a verdict (exit 0/1) or a usage error (exit 2).

    The rule is _Parser.NEGATIVE_NUMBER, not the interpreter's argparse, which
    changed it in 3.14; the port mirrors it, and this pins the reference half
    where there is no binary to compare against.
    """
    result = _run([SCRIPT, "--check", token])
    assert (result.returncode != 2) == verdict, result.stdout + result.stderr


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
