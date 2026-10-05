"""Tests for tests/_pwsh.py - the harness other pins type their commands through.

A helper that reports 0 for a command that never ran, or quotes a path into
broken PowerShell source, turns every pin built on it into one that cannot
fail, so its own two promises are pinned here.
"""

import json
import sys

import pytest

import _pwsh
from _pwsh import (invocation, needs_powershell, needs_pwsh, ps_quote,
                   run_powershell)
from llm_loop import cmdline

_ECHO = "import json, sys; print(json.dumps(sys.argv[1:]))"


@pytest.mark.parametrize("text, expected", [
    ("it's", "'it''s'"),
    ("it\u2019s", "'it\u2019\u2019s'"),
    ("\u2018a\u201ab\u201b", "'\u2018\u2018a\u201a\u201ab\u201b\u201b'"),
    ("$x `y", "'$x `y'"),
])
def test_ps_quote_doubles_every_quote_powershell_ends_a_string_at(text,
                                                                   expected):
    assert ps_quote(text) == expected


def test_the_quote_set_is_cmdline_s():
    # `_pwsh` keeps a copy rather than importing `llm_loop` (the main
    # repository loads it by path); a quote found later goes into both.
    assert _pwsh._PS_SINGLE_QUOTES == cmdline._PS_SINGLE_QUOTES


@needs_powershell
def test_ps_quote_round_trips_through_powershell():
    text = "it's \u2018a\u2019 \u201ab\u201b $x `y"
    result = run_powershell(invocation([sys.executable, "-c", _ECHO],
                                       ps_quote(text)))
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == [text]


@pytest.mark.parametrize("shell", [
    pytest.param("5.1", marks=needs_powershell),
    pytest.param("7", marks=needs_pwsh)])
@pytest.mark.parametrize("line, expected", [
    ("& {py} -c 'import sys; sys.exit(3)'", 3),
    ("& {py} -c 'pass'", 0),
    # A bare `; exit $LASTEXITCODE` tail reports 0 for each of these.
    ("& 'C:\\no\\such\\program.exe' a", 1),
    ("& {py} -c 'pass'; Write-Error boom", 1),
    ("Write-Output ran-no-native-program", 255),
], ids=["native-exit-code", "native-success", "native-never-ran",
        "failing-cmdlet-last", "no-native-program"])
def test_run_powershell_never_reports_success_for_what_did_not_run(line,
                                                                   expected,
                                                                   shell):
    result = run_powershell(line.replace("{py}", ps_quote(sys.executable)),
                            shell=_pwsh.PWSH if shell == "7" else None)
    assert result.returncode == expected, result.stdout + result.stderr
