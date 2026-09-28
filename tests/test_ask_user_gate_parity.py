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

import argparse
import importlib.util
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


@pytest.mark.parametrize("argv, code, stdout", [
    (["--check", "-==x y"], 2, None),
    (["--tool", "-= x", "--check", "ls"], 2, None),
    (["--check=-= x"], 0, "allowed\n"),  # glued, so a value
    (["--help", "-= x"], 0, None),
    (["-= x", "--help"], 2, None),
    (["--help", "--c"], 0, None),
])
def test_reference_ambiguous_flag_is_refused_where_parsing_reaches_it(
        argv, code, stdout):
    """A `-=` token and an ambiguous prefix are flags, refused on arrival.

    3.9-3.11 argparse read `-= x` as a value and refused an ambiguous prefix
    before running anything, --help included; _Parser._get_option_tuples pins
    the newer reading, which the port shares (ARGV_CASES compare the two).
    3.12.14+ argparse already gives every one of these results natively, so
    there the rows guard nothing of the override: they bite on 3.9-3.11 only.
    The glued row is the control the other way -- a `-=` VALUE that an
    over-broad refusal would turn into a usage error -- and its verdict is
    read, not just its exit code: 0 is also what hook mode on an empty stdin
    returns.
    """
    result = _run([SCRIPT, "--platform", "windows"] + argv)
    assert result.returncode == code, result.stdout + result.stderr
    if stdout is not None:
        assert result.stdout == stdout, result.stdout + result.stderr


def _gate_module():
    spec = importlib.util.spec_from_file_location("ask_user_gate_ref", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_parser_without_help_refuses_a_dash_equals_token():
    """The stand-in's tuple width comes from a probe of its own.

    It was copied from a `--hel` lookup on the parser in use, which is empty
    without --help: `-= x` then raised IndexError instead of a usage error.
    """
    gate = _gate_module()
    parser = gate._Parser(add_help=False, exit_on_error=False)
    parser.add_argument("--check")
    parser.add_argument("--tool")
    with pytest.raises(argparse.ArgumentError, match="ambiguous option"):
        parser.parse_args(["-= x"])
    # A flag, so not --check's value.
    with pytest.raises(argparse.ArgumentError, match="expected one argument"):
        parser.parse_args(["--check", "-= x"])


def test_reference_parser_refuses_allow_abbrev_false():
    """Its rules are abbreviation rules; with abbreviations off, newer argparse
    reads `-= x` as a value, which the stand-in would contradict."""
    with pytest.raises(ValueError, match="allow_abbrev=False"):
        _gate_module()._Parser(allow_abbrev=False)


def test_reference_parser_refuses_an_option_tuple_layout_it_never_measured(
        monkeypatch):
    """A 5-field tuple must stop the parse loudly, not be padded with None."""
    gate = _gate_module()
    real = argparse.ArgumentParser._get_option_tuples

    def five_fields(self, option_string):
        return [match + (None,) for match in real(self, option_string)]

    monkeypatch.setattr(argparse.ArgumentParser, "_get_option_tuples",
                        five_fields)
    with pytest.raises(RuntimeError, match="teach _option_tuple_fields"):
        gate._option_tuple_fields()
    parser = gate._Parser(exit_on_error=False)
    parser.add_argument("--check")
    parser.add_argument("--tool")
    with pytest.raises(RuntimeError, match="teach _option_tuple_fields"):
        parser.parse_args(["--check", "-= x"])


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
