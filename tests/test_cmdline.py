"""Tests for llm_loop.cmdline - the reproducing command line.

The module's whole job is "remove every spelling of a flag, then append the new
value", so most of these tests are one spelling each: a missed spelling leaves
the old value on the line next to the new one, which reads as correct and is not.
"""

import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys

import pytest

from _pwsh import (PWSH, invocation, needs_powershell, needs_pwsh, ps_quote,
                   run_powershell)
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


@pytest.mark.parametrize("os_name, environ, expected", [
    ("nt", {}, POWERSHELL),
    ("nt", {"MSYSTEM": "MINGW64"}, POSIX),          # Git Bash
    ("nt", {"MSYSTEM": "UCRT64"}, POSIX),           # an MSYS2 shell
    ("nt", {"MSYSTEM": ""}, POWERSHELL),
    ("nt", {"SHELL": "/usr/bin/bash"}, POWERSHELL),  # not a console signal
    ("posix", {}, POSIX),
    ("posix", {"MSYSTEM": "MINGW64"}, POSIX),
])
def test_the_paste_shell_decision_table(os_name, environ, expected):
    assert cmdline._shell_for(os_name, environ) == expected


def test_the_default_shell_reads_this_process_s_environment(monkeypatch):
    # Native Windows Python started from Git Bash sees MSYSTEM (measured); bash
    # rejects the PowerShell line's leading `& `.
    monkeypatch.setenv("MSYSTEM", "MINGW64")
    assert paste_shell() == POSIX
    monkeypatch.delenv("MSYSTEM")
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
    # Split at the colon after a bare `--` (measured), so quoted everywhere.
    ("-C:foo", "'-C:foo'"), ("-a:b", "'-a:b'"), ("--x:y", "'--x:y'"),
    ("-CD:/proj", "'-CD:/proj'"), ("C:foo", "C:foo"),
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


@pytest.mark.parametrize("shell", [POWERSHELL, POSIX])
@pytest.mark.parametrize("word", [
    "a\0b", "a\nb", "a\rb", "a\tb", "a\x1b[2Jb", "a\x7fb", "a\x85b",
    "a\u2028b", "a\u200bb", "a\u202eb", "a\udcffb"],
    ids=["nul", "lf", "cr", "tab", "esc", "del", "nel", "line-separator",
         "zero-width-space", "bidi-override", "lone-surrogate"])
def test_an_argument_that_does_not_print_as_one_line_is_refused(shell, word):
    with pytest.raises(NotPasteable):
        quote(["python", word, "SENTINEL"], shell)
    with pytest.raises(NotPasteable):
        quote([word, "SENTINEL"], shell)     # the program's own path too


def test_a_nul_is_refused_by_its_name():
    # 5.1 delivered ["a\0b", "SENTINEL"] as ["a"] with exit 0: the silent cut
    # is the reason the message has to say what happened.
    with pytest.raises(NotPasteable, match="NUL"):
        quote(["python", "a\0b", "SENTINEL"], POWERSHELL)


@pytest.mark.parametrize("shell", [POWERSHELL, POSIX])
@pytest.mark.parametrize("word", ["a\u00a0b", "a\u3000b"],
                         ids=["no-break-space", "ideographic-space"])
def test_a_space_separator_is_quoted_not_refused(shell, word):
    line = quote(["python", word], shell)
    assert word in line and line.endswith("b'")


def test_a_posix_word_starting_with_equals_is_quoted_for_zsh():
    # zsh's EQUALS expands a bare `=foo` into the path of the command `foo`.
    assert quote(["python", "=foo", "a=b"], POSIX) == "python '=foo' a=b"
    assert shlex.split(quote(["python", "=it's"], POSIX)) == ["python", "=it's"]


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
    launcher, which runs a different machine that has no such path. Git's
    `Git/bin/bash.exe` is a launcher for `Git/usr/bin/bash.exe`, the one with
    `msys-2.0.dll` beside it, so it is resolved to that before the check.
    Search every PATH entry: a Windows app alias or WSL launcher may appear
    before the usable Git Bash executable.
    """
    if os.name != "nt":
        return shutil.which("bash")
    for directory in os.get_exec_path():
        bash = shutil.which("bash", path=directory)
        if not bash:
            continue
        beside_usr = os.path.join(os.path.dirname(os.path.dirname(bash)),
                                  "usr", "bin", os.path.basename(bash))
        for candidate in (bash, beside_usr):
            if os.path.isfile(os.path.join(os.path.dirname(candidate),
                                           "msys-2.0.dll")) \
                    and os.path.isfile(candidate):
                return candidate
    return None


@pytest.mark.skipif(os.name != "nt", reason="the MSYS check is Windows-only")
def test_git_s_bash_launcher_counts_as_git_bash_and_wsl_s_does_not(
        tmp_path, monkeypatch):
    for name in ("Git/bin/bash.exe", "Git/usr/bin/bash.exe",
                 "Git/usr/bin/msys-2.0.dll", "System32/bash.exe"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    found = {}
    monkeypatch.setattr(shutil, "which", lambda name, **kwargs: found["bash"])
    found["bash"] = str(tmp_path / "Git" / "bin" / "bash.exe")
    assert _msys_bash() == str(tmp_path / "Git" / "usr" / "bin" / "bash.exe")
    found["bash"] = str(tmp_path / "Git" / "usr" / "bin" / "bash.exe")
    assert _msys_bash() == found["bash"]
    found["bash"] = str(tmp_path / "System32" / "bash.exe")
    assert _msys_bash() is None


@pytest.mark.skipif(os.name != "nt", reason="the MSYS check is Windows-only")
@pytest.mark.parametrize("git_directory", ["Git/bin", "Git/usr/bin", None])
def test_git_bash_is_found_after_an_unusable_path_match(
        tmp_path, monkeypatch, git_directory):
    for name in ("WindowsApps/bash.exe", "Git/bin/bash.exe",
                 "Git/usr/bin/bash.exe", "Git/usr/bin/msys-2.0.dll"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    directories = [str(tmp_path / "WindowsApps")]
    if git_directory is not None:
        directories.append(str(tmp_path / git_directory))
    monkeypatch.setenv("PATH", os.pathsep.join(directories))

    expected = (str(tmp_path / "Git/usr/bin/bash.exe")
                if git_directory is not None else None)
    actual = _msys_bash()
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert os.path.normcase(actual) == os.path.normcase(expected)


BASH = _msys_bash()
# 0.95-2.9 s per case through Git Bash (two runs of the six cases, measured
# 2026-09-28); the budget only has to tell a hang from a slow box.
BASH_TIMEOUT_S = 60


def _deliver_by_powershell(line, tmp_path):
    return run_powershell(line)


def _deliver_by_pwsh(line, tmp_path):
    return run_powershell(line, shell=PWSH)


def _deliver_by_bash(line, tmp_path):
    # From a file, not `bash -c`: the line then reaches bash untouched by any
    # command-line quoting of our own. MSYS rewrites POSIX-looking arguments to
    # a native program as Windows paths unless told not to - so this pins the
    # QUOTING only; a real Git Bash paste of `/foo` delivers
    # `C:/Program Files/Git/foo` (see `cmdline.paste_shell`).
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
    # The same PowerShell line, pasted into pwsh 7.3+: `render` says it reads
    # the line as 5.1 does.
    pytest.param(POWERSHELL, _deliver_by_pwsh, id="pwsh", marks=needs_pwsh),
    pytest.param(POSIX, _deliver_by_bash, id="bash", marks=pytest.mark.skipif(
        BASH is None, reason="no bash to paste into (on Windows: no Git Bash)")),
]

_METACHARACTERS = ("C:\\a$b `c;d&e|f@g(h){i}[j],k#l%m'n\u2019o \u043f\u0440\u043e"
                   "\u00a0p=q")


def _interpreter_under_a_space(tmp_path):
    """This interpreter, reached through a directory whose name has a space.

    A link to its real directory rather than a copy, which would lose the
    runtime beside it. The base interpreter, not a venv's: a Windows venv
    launcher looks for its pyvenv.cfg next to the path it was started by.
    """
    base = getattr(sys, "_base_executable", sys.executable)
    link = tmp_path / "Program Files x"
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(os.path.dirname(base), str(link))
    else:
        os.symlink(os.path.dirname(base), str(link))
    # The base's own name: a Linux venv's `python` may have no namesake in
    # the base directory (/usr/bin holds `python3` only).
    return str(link / os.path.basename(base))


@pytest.mark.parametrize("shell, deliver", SHELLS)
@pytest.mark.parametrize("argv, overrides, spaced_interpreter", [
    (["-m", "5"], {"--project-dir": ""}, False),
    (["-C", "", "-m", "5"], {"--max-runs": 2}, False),
    (["--finish", "products/configs/x"],
     {"--start-in": "", "--project-dir": r"C:\my project"}, False),
    (["-m", "5"], {"--project-dir": _METACHARACTERS}, False),
    (["-m", "5", "--", "-foo.bar", "--%x", "@a", "a,b", "$x", "it's"], {},
     False),
    # 5.1 splits a dash word at its colon only after a bare `--`; both sides.
    (["-a:b", "-CD:/proj", "--", "-C:foo", "-a:b", "-CD:/proj", "--x:y"], {},
     False),
    (["-m", "5", "--", "=foo", "--x=a:b"], {}, False),
    (["-m", "5"], {"--project-dir": r"C:\my project\sub"}, True),
], ids=["empty-override", "copied-empty", "empty-among-values",
        "metacharacters", "tail-words", "colon-words", "equals-words",
        "interpreter-under-a-space"])
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

    # And it MEANS what the run meant: the same outcome as the pair spelling.
    # Outcome, not namespace: the parser refuses an empty -C or --start-in
    # (`clispec.directory`, `clispec.duration`), and a refusal is a meaning too
    # — the one both spellings must share.
    pairs = list(argv)
    for flag, value in overrides.items():
        pairs += [flag, str(value)]
    assert _parse_outcome(delivered) == _parse_outcome(pairs)


@pytest.mark.parametrize("shell", [
    pytest.param("5.1", marks=needs_powershell),
    pytest.param("7", marks=needs_pwsh)])
@pytest.mark.parametrize("word", ["", 'a"b', "C:\\my dir\\"],
                         ids=["empty", "double-quote",
                              "space-and-trailing-backslash"])
def test_the_refused_words_are_where_5_1_and_pwsh_part(tmp_path, shell, word):
    """The premise of three of `_powershell_word`'s refusals: 5.1 mangles
    each word as quoted, and pwsh 7.3+ delivers it as typed — so a
    pre-escape that repaired 5.1 would break pwsh."""
    echo = tmp_path / "echo_argv.py"
    echo.write_text(_ECHO_ARGV, encoding="utf-8")
    line = invocation([sys.executable, str(echo)], ps_quote(word), "SENTINEL")
    result = run_powershell(line, shell=PWSH if shell == "7" else None)
    assert result.returncode == 0, line + "\n" + result.stdout + result.stderr
    delivered = json.loads(result.stdout)
    if shell == "7":
        assert delivered == [word, "SENTINEL"], line
    else:
        assert delivered != [word, "SENTINEL"], line


def _parse_outcome(argv):
    """The sequential parser's namespace for `argv`, or its exit code and
    message when it refuses the line."""
    parser = clispec.build_parser(clispec.SEQUENTIAL, prog="pytest")
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            return parser.parse_known_args(argv)
    except SystemExit as exc:
        return exc.code, err.getvalue()


# --- the one production caller: the rich-install hint at startup ---------------

def _missing_rich_hint(monkeypatch, capsys):
    from llm_loop import console
    monkeypatch.setattr(console, "RICH_AVAILABLE", False)
    monkeypatch.setattr(console, "_DEPENDENCY_WARNING_SHOWN", False)
    console.warn_missing_dependencies()
    return capsys.readouterr().out


@pytest.mark.parametrize("msystem, shell, label", [
    (None, POWERSHELL if os.name == "nt" else POSIX,
     " (PowerShell)" if os.name == "nt" else ""),
    ("MINGW64", POSIX, ""),
])
def test_the_rich_hint_is_a_line_for_the_paste_shell(monkeypatch, capsys,
                                                     msystem, shell, label):
    if msystem is None:
        monkeypatch.delenv("MSYSTEM", raising=False)
    else:
        monkeypatch.setenv("MSYSTEM", msystem)
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\Py\python.exe")
    out = _missing_rich_hint(monkeypatch, capsys)
    install = quote([sys.executable, "-m", "pip", "install", "rich"], shell)
    assert f"  Install with{label}: {install}\n" in out


def test_the_rich_hint_survives_an_interpreter_with_no_path(monkeypatch,
                                                            capsys):
    # Embedded / frozen: sys.executable is "", which no line can name. This
    # runs at startup; raising here would stop the run over a missing extra.
    monkeypatch.setattr(sys, "executable", "")
    out = _missing_rich_hint(monkeypatch, capsys)
    assert "Install with: pip install rich" in out
