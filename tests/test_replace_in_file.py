"""replace_in_file's one way to delete, through the script's real command line.

Deleting is --remove TEXT, and an empty --new in any form is refused: the
spellings `--new ""`, a bare `--new` (what Windows PowerShell 5.1 makes of
`--new ""`, dropping the empty argument) and `--new=` are also what a
replacement lost on the way looks like, and this script writes for good (the
reasoning sits at add_edit_arguments). These cases hand the script exactly
those argvs, and the PowerShell cases at the end type the command the way an
operator does, so the dropped token is the shell's doing rather than this
file's assumption.
"""

import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[1] / "claude-plugin" / "ask-user-gate"
          / "bin" / "replace_in_file.py")
POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None
# One run is 0.06 s direct and 0.18-0.20 s through powershell.exe (three runs,
# measured 2026-09-27); the budget only has to tell a hang from a slow box.
TIMEOUT_S = 60

TEXT = "a = 1; guard();\nb = 2; guard();\n"
DELETED = "a = 1;\nb = 2;\n"


def _victim(tmp_path: Path) -> Path:
    path = tmp_path / "victim.txt"
    path.write_bytes(TEXT.encode("utf-8"))
    return path


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=TIMEOUT_S)


def _flat(text: str) -> str:
    """`text` with argparse's line wrapping undone: it wraps at $COLUMNS."""
    return " ".join(text.split())


def _assert_wrote(result: subprocess.CompletedProcess, victim: Path,
                  expected: str) -> None:
    assert result.returncode == 0, result.stdout + result.stderr
    assert victim.read_bytes().decode("utf-8") == expected


def _assert_refused(result: subprocess.CompletedProcess, victim: Path,
                    names: str = "") -> None:
    assert result.returncode == 2, result.stdout + result.stderr
    assert victim.read_bytes().decode("utf-8") == TEXT
    assert names in _flat(result.stderr), result.stderr


# --- an empty --new, however it arrives, is not a deletion -----------------------

@pytest.mark.parametrize("new", [["--new", "--count", "2"],   # PS 5.1, mid-line
                                 ["--count", "2", "--new"],   # ... at the end
                                 ["--new", "", "--count", "2"],
                                 ["--new=", "--count", "2"]],
                         ids=["bare-before-a-flag", "bare-at-the-end",
                              "empty-token", "equals"])
def test_an_empty_new_is_refused_naming_remove(tmp_path, new):
    victim = _victim(tmp_path)
    _assert_refused(_run(str(victim), "--old", " guard();", *new), victim,
                    "use --remove TEXT")


# --- --remove says the operation by name -----------------------------------------

def test_remove_deletes(tmp_path):
    victim = _victim(tmp_path)
    _assert_wrote(_run(str(victim), "--remove", " guard();", "--count", "any"),
                  victim, DELETED)


def test_remove_is_a_regex_under_regex(tmp_path):
    victim = _victim(tmp_path)
    _assert_wrote(_run(str(victim), "--regex", "--remove", r" guard\(\);$",
                       "--count", "2"), victim, DELETED)


def test_remove_keeps_the_count_check(tmp_path):
    # Two matches against the default --count 1: the refusal that makes this
    # script safer than `sed -i` must not be skipped by the new spelling.
    victim = _victim(tmp_path)
    result = _run(str(victim), "--remove", " guard();")
    assert result.returncode == 1, result.stdout + result.stderr
    assert victim.read_bytes().decode("utf-8") == TEXT


@pytest.mark.parametrize("extra", [["--old", "a = 1;"], ["--new", "x"],
                                   ["--new"]])
def test_remove_mixed_with_old_or_new_is_refused(tmp_path, extra):
    victim = _victim(tmp_path)
    _assert_refused(_run(str(victim), "--remove", " guard();", *extra,
                         "--count", "any"), victim)


@pytest.mark.parametrize("edit", [["--old", " guard();"], ["--new", "x"], []])
def test_an_edit_still_needs_both_halves(tmp_path, edit):
    victim = _victim(tmp_path)
    _assert_refused(_run(str(victim), *edit, "--count", "any"), victim)


@pytest.mark.parametrize("edit", [["--remove", ""], ["--old", "", "--new", "x"]],
                         ids=["remove", "old"])
def test_an_empty_pattern_is_refused(tmp_path, edit):
    victim = _victim(tmp_path)
    _assert_refused(_run(str(victim), *edit, "--count", "any"), victim,
                    "got no text")


# --- a repeated flag is refused, not silently narrowed to its last value ----------

@pytest.mark.parametrize("flag, edit", [
    ("--remove", ["--remove", "a = 1;", "--remove", " guard();"]),
    ("--old", ["--old", "a = 1;", "--old", " guard();", "--new", "x"]),
    ("--new", ["--old", " guard();", "--new", "x", "--new", "y"]),
], ids=["remove", "old", "new"])
def test_a_repeated_flag_is_refused(tmp_path, flag, edit):
    victim = _victim(tmp_path)
    _assert_refused(_run(str(victim), *edit, "--count", "2"), victim,
                    f"{flag} given 2 times")


def test_help_names_the_delete_spelling():
    result = _run("--help")
    assert result.returncode == 0, result.stderr
    assert "--remove TEXT" in result.stdout
    assert "to delete, use --remove" in _flat(result.stdout)


# --- the operator's shell ------------------------------------------------------

def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def powershell(*words: str) -> subprocess.CompletedProcess:
    """Run `& python SCRIPT <words>` in Windows PowerShell 5.1, words verbatim.

    Each word is PowerShell source, so `''` is PowerShell's own empty string --
    the one it drops on the way to a native program. -EncodedCommand keeps
    Python's own command-line quoting out of what PowerShell parses.
    """
    line = " ".join(["&", _ps_quote(sys.executable), _ps_quote(str(SCRIPT)),
                     *words]) + "; exit $LASTEXITCODE"
    encoded = base64.b64encode(line.encode("utf-16-le")).decode("ascii")
    return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive",
                           "-EncodedCommand", encoded],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=TIMEOUT_S)


needs_powershell = pytest.mark.skipif(
    POWERSHELL is None, reason="Windows PowerShell 5.1 is the shell that drops ''")


@needs_powershell
@pytest.mark.parametrize("tail", [["--new", "''", "--count", "2"],
                                  ["--count", "2", "--new", "''"],
                                  # the lost value itself: never assigned
                                  ["--new", "$replacementNeverSet",
                                   "--count", "2"]],
                         ids=["empty-mid-line", "empty-at-the-end",
                              "unset-variable"])
def test_powershell_empty_new_is_refused(tmp_path, tail):
    victim = _victim(tmp_path)
    _assert_refused(powershell(_ps_quote(str(victim)), "--old", "' guard();'",
                               *tail), victim, "use --remove TEXT")


@needs_powershell
def test_powershell_remove_deletes(tmp_path):
    victim = _victim(tmp_path)
    _assert_wrote(powershell(_ps_quote(str(victim)), "--remove", "' guard();'",
                             "--count", "2"), victim, DELETED)
