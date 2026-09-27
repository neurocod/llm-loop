"""Tests for llm_loop.cmdline - the reproducing command line.

The module's whole job is "remove every spelling of a flag, then append the new
value", so most of these tests are one spelling each: a missed spelling leaves
the old value on the line next to the new one, which reads as correct and is not.
"""

import base64
import json
import os
import shutil
import subprocess
import sys

import pytest

from llm_loop import clispec, cmdline
from llm_loop.cmdline import quote, rebuild_argv, render


# --- removal: every spelling of one flag ---------------------------------------

@pytest.mark.parametrize("argv", [
    ["--max-runs", "5"],
    ["--max-runs=5"],
    ["--max", "5"],
    ["--max=5"],
    ["-m", "5"],
    ["-m5"],
])
def test_every_max_runs_spelling_is_replaced(argv):
    assert rebuild_argv(argv, {"--max-runs": 9}) == ["--max-runs", "9"]


@pytest.mark.parametrize("argv", [
    ["--start-in", "29m"],
    ["--startIn", "29m"],
    ["--start-in=29m"],
    ["-s", "29m"],
    ["-s29m"],
])
def test_every_start_in_spelling_is_replaced(argv):
    assert rebuild_argv(argv, {"--start-in": "1h"}) == ["--start-in", "1h"]


@pytest.mark.parametrize("argv", [
    ["--project-dir", "D:/proj"],
    ["--project-dir=D:/proj"],
    ["-C", "D:/proj"],
    ["-CD:/proj"],
])
def test_every_project_dir_spelling_is_replaced(argv):
    assert rebuild_argv(argv, {"--project-dir": "D:/other"}) == [
        "--project-dir", "D:/other"]


def test_repeated_spellings_are_all_removed():
    # argparse would keep the last one; leaving any behind hides the override.
    argv = ["-m", "1", "--max", "2", "--max-runs=3", "-m4"]
    assert rebuild_argv(argv, {"--max-runs": 7}) == ["--max-runs", "7"]


def test_removal_only_override_drops_the_flag():
    assert rebuild_argv(["--raw", "-m", "5"], {"--max-runs": None}) == ["--raw"]


def test_dangling_value_flag_at_end_is_removed():
    assert rebuild_argv(["--raw", "-m"], {"--max-runs": None}) == ["--raw"]


# --- untouched argv -------------------------------------------------------------

def test_empty_overrides_is_a_no_op():
    argv = ["-p", "-j", "3", "--random", "--max-runs", "5", "--", "-m", "9"]
    assert rebuild_argv(argv, {}) == argv
    assert rebuild_argv(argv, {}) is not argv     # a copy, never the caller's list


def test_wrapper_only_flags_survive_an_override():
    # `--finish` is the value-taking one: an unlisted flag of that shape has its
    # VALUE read as a free-standing token, so the folder would be stripped too.
    argv = ["-p", "-j", "3", "--random", "--finish", "products/configs/x",
            "--max-runs", "5"]
    assert rebuild_argv(argv, {"--max-runs": 2}) == [
        "-p", "-j", "3", "--random", "--finish", "products/configs/x",
        "--max-runs", "2"]


def test_unknown_flags_are_copied_verbatim():
    argv = ["--some-future-flag", "--another=1", "positional"]
    assert rebuild_argv(argv, {"--raw": True}) == argv + ["--raw"]


def test_value_that_looks_like_a_flag_is_not_scanned():
    # A project directory literally named "--max-runs" is still a value.
    argv = ["-C", "--max-runs", "-m", "5"]
    assert rebuild_argv(argv, {"--max-runs": None}) == ["-C", "--max-runs"]


# --- the `--` tail --------------------------------------------------------------

def test_passthrough_tail_is_preserved_verbatim():
    argv = ["-m", "5", "--", "-m", "5", "--raw", "--max=9"]
    assert rebuild_argv(argv, {"--max-runs": 1}) == [
        "--max-runs", "1", "--", "-m", "5", "--raw", "--max=9"]


def test_only_the_first_bare_dashdash_splits():
    argv = ["--", "--", "-m", "1"]
    assert rebuild_argv(argv, {"--max-runs": 2}) == [
        "--max-runs", "2", "--", "--", "-m", "1"]


# --- boolean flags --------------------------------------------------------------

def test_boolean_flag_added_and_removed():
    assert rebuild_argv([], {"--no-statusline": True}) == ["--no-statusline"]
    assert rebuild_argv(["--no-statusline"], {"--no-statusline": None}) == []
    assert rebuild_argv(["--no-statusline"], {"--no-statusline": False}) == []


def test_boolean_flag_is_not_duplicated():
    assert rebuild_argv(["-d"], {"--dry-run": True}) == ["--dry-run"]


def test_boolean_flag_rejects_a_value():
    with pytest.raises(ValueError):
        rebuild_argv([], {"--raw": "yes"})


def test_value_flag_rejects_true():
    with pytest.raises(ValueError):
        rebuild_argv([], {"--max-runs": True})


def test_unknown_canonical_flag_is_rejected():
    with pytest.raises(KeyError):
        rebuild_argv([], {"--maxRuns": 5})


# --- append order ---------------------------------------------------------------

def test_overrides_are_appended_in_table_order():
    overrides = {"--no-statusline": True, "--weekly-limit": 90,
                 "--session-limit": 80, "--max-runs": 3}
    reversed_overrides = dict(reversed(list(overrides.items())))
    expected = ["--max-runs", "3", "--session-limit", "80",
                "--weekly-limit", "90", "--no-statusline"]
    assert rebuild_argv([], overrides) == expected
    assert rebuild_argv([], reversed_overrides) == expected


def test_zero_is_a_value_not_a_removal():
    assert rebuild_argv([], {"--max-runs": 0}) == ["--max-runs", "0"]


# --- empty values ---------------------------------------------------------------
# `--flag=`, one token: the pair `--flag ""` is lost to PowerShell 5.1 on paste
# (see cmdline._empty_value). The round trip through the shell itself is pinned
# below, under render.

def test_an_empty_override_is_one_equals_token():
    assert rebuild_argv(["-m", "5"], {"--project-dir": ""}) == [
        "-m", "5", "--project-dir="]


@pytest.mark.parametrize("argv, expected", [
    (["--project-dir", ""], ["--project-dir="]),
    (["-C", "", "--raw"], ["--project-dir=", "--raw"]),
    (["--max", ""], ["--max="]),            # a long alias keeps its spelling
    (["--project-dir=", "--raw"], ["--project-dir=", "--raw"]),
])
def test_a_copied_empty_value_is_respelled_as_one_token(argv, expected):
    assert rebuild_argv(argv, {}) == expected


def test_an_empty_token_after_dashdash_is_left_alone():
    argv = ["-C", "D:/p", "--", "--project-dir", ""]
    assert rebuild_argv(argv, {}) == argv


# --- the table's address --------------------------------------------------------
# Its SHAPE is pinned where it is now declared, in test_clispec.py: this module
# consumes the table, it no longer owns it. What stays this module's business is
# that both names still answer here, since `statusline.SettingsRegistry` and the
# host wrappers reach for `cmdline.FLAG_ALIASES`, not for clispec's.

def test_the_table_still_answers_at_this_module_s_address():
    assert cmdline.FLAG_ALIASES is clispec.FLAG_ALIASES
    assert cmdline.Flag is clispec.Flag
    assert set(cmdline.__all__) >= {"FLAG_ALIASES", "Flag"}



# --- render ---------------------------------------------------------------------

def test_render_prefixes_interpreter_and_script():
    line = render(["-m", "5"], {"--max-runs": 2},
                  executable="python", script="runGenerateModels.py")
    assert line == "python runGenerateModels.py --max-runs 2"


def test_render_quotes_paths_with_spaces():
    line = render(["-p"], {"--project-dir": r"C:\my project"},
                  executable="python", script="run models.py")
    if os.name == "nt":
        assert line == 'python "run models.py" -p --project-dir "C:\\my project"'
    else:
        assert line == "python 'run models.py' -p --project-dir 'C:\\my project'"


def test_render_defaults_the_script_to_argv0(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runCycle.py", "-m", "5"])
    assert render(["-m", "5"], {}, executable="python") == "python runCycle.py -m 5"


POWERSHELL = shutil.which("powershell.exe") if os.name == "nt" else None
# 2.4-4.2 s per test here (three cases, measured 2026-09-27; the ask-user-gate
# CLI's round trips measured 0.18-0.20 s the same day on a warm shell); the
# budget only has to tell a hang from a slow box.
PS_TIMEOUT_S = 60
_ECHO_ARGV = "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"


@pytest.mark.skipif(POWERSHELL is None,
                    reason="Windows PowerShell 5.1 is the shell that drops \"\"")
@pytest.mark.parametrize("argv, overrides", [
    (["-m", "5"], {"--project-dir": ""}),
    (["-C", "", "-m", "5"], {"--max-runs": 2}),
    (["-p", "--finish", "products/configs/x"],
     {"--start-in": "", "--project-dir": r"C:\my project"}),
], ids=["empty-override", "copied-empty", "empty-among-values"])
def test_the_rendered_line_round_trips_through_powershell(
        tmp_path, argv, overrides):
    """Paste the line into PowerShell 5.1 and read back the argv it delivers.

    Comparing renderer text alone is what let `--project-dir ""` pass: it is
    the right CreateProcess spelling and still arrives as a bare flag.
    """
    echo = tmp_path / "echo_argv.py"
    echo.write_text(_ECHO_ARGV, encoding="utf-8")
    line = render(argv, overrides, executable=sys.executable, script=str(echo))
    # `& ` because PowerShell reads a QUOTED first token (an interpreter under
    # "Program Files") as a string expression, not a command; the rendered line
    # does not carry it, since cmd.exe would reject it.
    source = "& " + line + "; exit $LASTEXITCODE"
    encoded = base64.b64encode(source.encode("utf-16-le")).decode("ascii")
    result = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive",
                             "-EncodedCommand", encoded],
                            capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=PS_TIMEOUT_S)
    assert result.returncode == 0, result.stdout + result.stderr
    delivered = json.loads(result.stdout)
    assert delivered == rebuild_argv(argv, overrides), line

    # And it MEANS what the run meant: the same namespace as the pair spelling.
    parser = clispec.build_parser(clispec.SEQUENTIAL, prog="pytest")
    pairs = list(argv)
    for flag, value in overrides.items():
        pairs += [flag, str(value)]
    assert parser.parse_known_args(delivered) == parser.parse_known_args(pairs)


def test_quote_round_trips_through_the_local_shell_rules():
    parts = ["python", "a b.py", "--max-runs", "5"]
    if os.name == "nt":
        assert quote(parts) == 'python "a b.py" --max-runs 5'
    else:
        assert quote(parts) == "python 'a b.py' --max-runs 5"
