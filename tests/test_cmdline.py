"""Tests for llm_loop.cmdline - the reproducing command line.

The module's whole job is "remove every spelling of a flag, then append the new
value", so most of these tests are one spelling each: a missed spelling leaves
the old value on the line next to the new one, which reads as correct and is not.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_loop.cmdline import FLAG_ALIASES, Flag, quote, rebuild_argv, render


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
    ["--max-strike", "3h"],
    ["--maxStrike", "3h"],
    ["--max-strike=3h"],
    ["-S", "3h"],
    ["-S3h"],
])
def test_every_max_strike_spelling_is_replaced(argv):
    assert rebuild_argv(argv, {"--max-strike": "4h"}) == ["--max-strike", "4h"]


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
    argv = ["-p", "-j", "3", "--random", "--grow-kit-periodically", "4",
            "--max-runs", "5"]
    assert rebuild_argv(argv, {"--max-runs": 2}) == [
        "-p", "-j", "3", "--random", "--grow-kit-periodically", "4",
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


# --- the flag table -------------------------------------------------------------

def test_canonical_key_is_among_its_own_aliases():
    for canonical, spec in FLAG_ALIASES.items():
        assert isinstance(spec, Flag)
        assert canonical in spec.aliases


def test_no_spelling_is_claimed_by_two_flags():
    seen = {}
    for canonical, spec in FLAG_ALIASES.items():
        for alias in spec.aliases:
            assert alias not in seen, f"{alias} in {seen.get(alias)} and {canonical}"
            seen[alias] = canonical


def test_deprecated_aliases_are_known():
    # The three spellings cyclecore.parse_args still accepts for compatibility.
    assert "--max" in FLAG_ALIASES["--max-runs"].aliases
    assert "--startIn" in FLAG_ALIASES["--start-in"].aliases
    assert "--maxStrike" in FLAG_ALIASES["--max-strike"].aliases


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


def test_quote_round_trips_through_the_local_shell_rules():
    parts = ["python", "a b.py", "--max-runs", "5"]
    if os.name == "nt":
        assert quote(parts) == 'python "a b.py" --max-runs 5'
    else:
        assert quote(parts) == "python 'a b.py' --max-runs 5"
