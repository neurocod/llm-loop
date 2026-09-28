"""Tests for llm_loop.cmdline - the reproducing command line.

The module's whole job is "remove every spelling of a flag, then append the new
value", so most of these tests are one spelling each: a missed spelling leaves
the old value on the line next to the new one, which reads as correct and is not.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from _pwsh import needs_powershell, run_powershell
from llm_loop import clispec, cmdline
from llm_loop.cmdline import (POSIX, POWERSHELL, NotPasteable, paste_shell,
                              quote, rebuild_argv, render)


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



# --- render: one line, for the shell it is pasted into -------------------------

@pytest.mark.parametrize("shell, expected", [
    (POWERSHELL, "& python runGenerateModels.py --max-runs 2"),
    (POSIX, "python runGenerateModels.py --max-runs 2"),
])
def test_render_prefixes_interpreter_and_script(shell, expected):
    line = render(["-m", "5"], {"--max-runs": 2}, executable="python",
                  script="runGenerateModels.py", shell=shell)
    assert line == expected


@pytest.mark.parametrize("shell, expected", [
    (POWERSHELL, r"& python 'run models.py' -p --project-dir 'C:\my project'"),
    (POSIX, r"python 'run models.py' -p --project-dir 'C:\my project'"),
])
def test_render_quotes_paths_with_spaces(shell, expected):
    line = render(["-p"], {"--project-dir": r"C:\my project"},
                  executable="python", script="run models.py", shell=shell)
    assert line == expected


def test_render_defaults_the_script_to_argv0(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["runCycle.py", "-m", "5"])
    assert render(["-m", "5"], {}, executable="python",
                  shell=POSIX) == "python runCycle.py -m 5"


def test_the_default_shell_is_this_os_s_console():
    assert paste_shell() == (POWERSHELL if os.name == "nt" else POSIX)
    parts = ["python", "a b.py", "--max-runs", "5"]
    assert quote(parts) == quote(parts, paste_shell())
    assert render(["-m", "5"], {}, executable="python", script="x.py") == \
        render(["-m", "5"], {}, executable="python", script="x.py",
               shell=paste_shell())


def test_an_unknown_shell_is_refused():
    with pytest.raises(ValueError, match="cmd"):
        quote(["python"], "cmd")


@pytest.mark.parametrize("word, expected", [
    # Bare: PowerShell 5.1 delivers each of these verbatim (measured).
    ("5", "5"), ("1e5", "1e5"), ("--max-runs", "--max-runs"), ("-m5", "-m5"),
    ("--project-dir=", "--project-dir="), (r"C:\a\b", r"C:\a\b"),
    ("D:/proj", "D:/proj"), ("--", "--"),
    # Quoted: each would be expanded, split or dropped if left bare.
    ("$HOME", "'$HOME'"), ("a`b", "'a`b'"), ("a;b", "'a;b'"), ("a&b", "'a&b'"),
    ("a|b", "'a|b'"), ("@a", "'@a'"), ("(a)", "'(a)'"), ("{a}", "'{a}'"),
    ("a,b", "'a,b'"), ("#a", "'#a'"), ("--%x", "'--%x'"), ("a b", "'a b'"),
    ("-foo.bar", "'-foo.bar'"), ("\u043f\u0440\u043e", "'\u043f\u0440\u043e'"),
    # Every quote PowerShell ends a '...' string at is doubled.
    ("it's", "'it''s'"), ("it\u2019s", "'it\u2019\u2019s'"),
])
def test_a_powershell_word_is_bare_only_when_delivered_verbatim(word, expected):
    assert quote([word], POWERSHELL) == "& " + expected


@pytest.mark.parametrize("word", ["", 'a"b', "C:\\my dir\\", "--%"],
                         ids=["empty", "double-quote",
                              "space-and-trailing-backslash", "stop-parsing"])
def test_an_argument_powershell_cannot_deliver_is_refused(word):
    with pytest.raises(NotPasteable):
        quote(["python", word], POWERSHELL)
    # The POSIX line has a spelling for each of them.
    assert shlex.split(quote(["python", word], POSIX)) == ["python", word]


def test_an_empty_value_after_dashdash_is_refused_for_powershell():
    # The one empty value `rebuild_argv` does not respell (see its docstring).
    with pytest.raises(NotPasteable):
        render(["--", ""], {}, executable="python", script="x.py",
               shell=POWERSHELL)


# --- the rendered line, pasted into the shell itself ---------------------------
# Comparing renderer text alone is what let `--project-dir ""` pass: it was the
# right CreateProcess spelling and still arrived as a bare flag. So each case is
# handed, exactly as printed, to the shell it was written for, and the argv the
# program receives is read back.

_ECHO_ARGV = "import json, sys\nprint(json.dumps(sys.argv[1:]))\n"


def _msys_bash():
    """A bash that runs this interpreter, or None.

    On Windows only an MSYS one (Git Bash) qualifies: `bash` there may be WSL's
    launcher, which runs a different machine that has no such path.
    """
    bash = shutil.which("bash")
    if bash and os.name == "nt" and not os.path.isfile(
            os.path.join(os.path.dirname(bash), "msys-2.0.dll")):
        return None
    return bash


BASH = _msys_bash()
# 0.95-2.9 s per case through Git Bash (two runs of the six cases, measured
# 2026-09-28); the budget only has to tell a hang from a slow box.
BASH_TIMEOUT_S = 60


def _deliver_by_powershell(line, tmp_path):
    return run_powershell(line)


def _deliver_by_bash(line, tmp_path):
    # From a file, not `bash -c`: the line then reaches bash untouched by any
    # command-line quoting of our own. MSYS rewrites POSIX-looking arguments to
    # a native program as Windows paths unless told not to.
    script = tmp_path / "line.sh"
    script.write_text(line + "\n", encoding="utf-8")
    env = dict(os.environ, MSYS_NO_PATHCONV="1", MSYS2_ARG_CONV_EXCL="*")
    return subprocess.run([BASH, str(script)], env=env,
                          stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=BASH_TIMEOUT_S)


SHELLS = [
    pytest.param(POWERSHELL, _deliver_by_powershell, id="powershell",
                 marks=needs_powershell),
    pytest.param(POSIX, _deliver_by_bash, id="bash", marks=pytest.mark.skipif(
        BASH is None, reason="no bash to paste into (on Windows: no Git Bash)")),
]

_METACHARACTERS = "C:\\a$b `c;d&e|f@g(h){i}[j],k#l%m'n\u2019o \u043f\u0440\u043e"


def _interpreter_under_a_space(tmp_path):
    """This interpreter, reached through a directory whose name has a space.

    A link to its real directory rather than a copy, which would lose the
    runtime beside it. The base interpreter, not a venv's: a Windows venv
    launcher looks for its pyvenv.cfg next to the path it was started by.
    """
    real = os.path.dirname(getattr(sys, "_base_executable", sys.executable))
    link = tmp_path / "Program Files x"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(real, str(link))
    else:
        os.symlink(real, str(link))
    return str(link / os.path.basename(sys.executable))


@pytest.mark.parametrize("shell, deliver", SHELLS)
@pytest.mark.parametrize("argv, overrides, spaced_interpreter", [
    (["-m", "5"], {"--project-dir": ""}, False),
    (["-C", "", "-m", "5"], {"--max-runs": 2}, False),
    # No `-p`: it selects the parallel parser, which has no --start-in, and the
    # meaning check below parses with the sequential one.
    (["--finish", "products/configs/x"],
     {"--start-in": "", "--project-dir": r"C:\my project"}, False),
    (["-m", "5"], {"--project-dir": _METACHARACTERS}, False),
    (["-m", "5", "--", "-foo.bar", "--%x", "@a", "a,b", "$x", "it's"], {},
     False),
    (["-m", "5"], {"--project-dir": r"C:\my project\sub"}, True),
], ids=["empty-override", "copied-empty", "empty-among-values",
        "metacharacters", "tail-words", "interpreter-under-a-space"])
def test_the_rendered_line_round_trips_through_its_shell(
        tmp_path, shell, deliver, argv, overrides, spaced_interpreter):
    echo = tmp_path / "echo_argv.py"
    echo.write_text(_ECHO_ARGV, encoding="utf-8")
    executable = (_interpreter_under_a_space(tmp_path) if spaced_interpreter
                  else sys.executable)
    line = render(argv, overrides, executable=executable, script=str(echo),
                  shell=shell)
    result = deliver(line, tmp_path)
    assert result.returncode == 0, line + "\n" + result.stdout + result.stderr
    delivered = json.loads(result.stdout)
    assert delivered == rebuild_argv(argv, overrides), line

    # And it MEANS what the run meant: the same namespace as the pair spelling.
    parser = clispec.build_parser(clispec.SEQUENTIAL, prog="pytest")
    pairs = list(argv)
    for flag, value in overrides.items():
        pairs += [flag, str(value)]
    assert parser.parse_known_args(delivered) == parser.parse_known_args(pairs)
