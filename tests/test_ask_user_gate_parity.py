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
import re
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


def test_reference_check_file_it_cannot_read_says_so(tmp_path):
    """The message, and no traceback; the exit code is an ARGV_CASES row."""
    result = _run([SCRIPT, "--check-file", str(tmp_path / "missing.txt")])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "cannot read" in result.stderr
    assert "Traceback" not in result.stderr


def _parity_module():
    spec = importlib.util.spec_from_file_location("parity_check_ref", PARITY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PARITY = _parity_module()


@pytest.mark.parametrize("argv, code, text", _PARITY.ARGV_CASES,
                         ids=[ascii(case[0]) for case in _PARITY.ARGV_CASES])
def test_reference_command_line(argv, code, text):
    """The reference half of parity_check's ARGV_CASES, binary or not.

    That script compares the two halves only where the port was built; this
    holds the reference to the table's own answers everywhere else, through
    the same argv_verdict, so both read one output the same way.
    """
    got_code, got_text = _PARITY.argv_verdict([sys.executable, SCRIPT], argv)
    assert got_code == code, got_text
    if text is not None:
        assert got_text == text


def test_port_options_are_the_reference_options():
    """The port's kOptions is a hand copy of the reference's option strings.

    Read from the source, so it bites on every checkout: the behavioural pin
    (parity_check's per-option ARGV_CASES rows) needs the binary, and an option
    added to the script alone would otherwise stay unknown to the port -- a
    usage error there, a verdict here -- until someone built it.
    """
    with open(os.path.join(PLUGIN, "cpp", "ask_user_gate.cpp"),
              encoding="utf-8") as handle:
        source = handle.read()
    block = re.search(r"constexpr CliOption kOptions\[\] = \{(.*?)\n\};", source,
                      re.DOTALL)
    assert block, "kOptions initializer not found in ask_user_gate.cpp"
    port = {name: arity == "Value" for name, arity in
            re.findall(r'\{"([^"]+)", Arity::(\w+)\}', block.group(1))}
    assert port == _PARITY.reference_options()


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
