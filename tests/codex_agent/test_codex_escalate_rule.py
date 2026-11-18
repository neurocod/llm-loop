"""The `escalate` rule matched against each command of a line, whole
(`command_escalate.escalate_refusal`), and the literal reader under it
(`command_shell._words`): round 4 of 2432's review (item 2445)."""

import re

import pytest

from llm_agents.codex.command_escalate import escalate_refusal
from llm_agents.codex.command_shell import Unparsed, Word, _words
from llm_agents.codex.testing import PS, worker_judge

GRANTED = "matches the escalate rule"


def _refused(ask, command):
    verdict = ask(command)
    assert not verdict, (command, verdict.reason)
    return verdict.reason


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_a_rule_is_not_searched_for_in_the_line(tmp_path, monkeypatch, role):
    """A search for `npm test` granted all of these: a git commit whose
    message is `npm test` (git past the broker, the worker's hooks), a
    second command chained after the match, an alias swapped in before it."""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"npm test", role=role)
    for command in ("git -c core.hooksPath=.githooks commit --allow-empty -m 'npm test'",
                    "npm test; python -c 'print(1)'",
                    "npm test && python -c 'print(1)'",
                    "Set-Alias npm C:\\x\\evil.exe; npm test",
                    "npm test | python -c 'print(1)'"):
        _refused(ask, command)
    for command in ("npm test", "npm  test", "npm test; npm test",
                    "npm test 2>&1 | Select-Object -Last 40"):
        verdict = ask(command)
        assert verdict and verdict.reason == GRANTED, (command, verdict.reason)


def test_a_rule_grants_no_redirect_relative_program_or_method_call(tmp_path, monkeypatch):
    """Even a rule that matches the text: `>` writes wherever it names,
    `.\\npm` is the file the agent left in its tree, `% Delete` calls a
    method on every object piped in."""
    _, ask = worker_judge(tmp_path, monkeypatch, role="reviewer",
                          escalate=r"(\.\\)?npm test.*|% Delete")
    assert "writes no file" in _refused(ask, "npm test > C:\\anywhere\\x.txt")
    assert "writes no file" in _refused(ask, "npm test 2> x.txt")
    assert "given by path" in _refused(ask, ".\\npm test")
    assert "runs a command it is handed" in _refused(ask, "npm test | % Delete")
    assert ask("npm test -- a.test.ts").reason == GRANTED


def _why(command: str, rule: str) -> str:
    """The escalate rule's own reason (a worker's verdict names its shape
    refusal instead), for `command` in Codex's wrapper."""
    return escalate_refusal(PS.format(command), re.compile(rule))


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_a_rule_grants_no_program_given_by_a_path(tmp_path, monkeypatch, role):
    """Review of 2445: `C:\\<slot>\\git.exe` passed as an absolute path and
    was then judged as `git`; a path names a file the agent may have
    written, and the rule has no roots to tell."""
    rule = r"(\S*[\\/:])?(npm|git)(\.exe)? .*"
    _, ask = worker_judge(tmp_path, monkeypatch, role=role, escalate=rule)
    for command in ("C:\\w\\npm.exe test", "C:/w/npm test", "C:npm.exe test",
                    "\\\\host\\share\\npm test", "C:\\slot\\git.exe status"):
        _refused(ask, command)
        assert "given by path" in _why(command, rule), command
    assert _why("npm.exe test", rule) == ""


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_git_and_its_family_are_refused_whatever_the_rule_says(tmp_path, monkeypatch, role):
    """Both reviews of 2445: a "read-only" git runs the fsmonitor, external
    diff and textconv programs of a config the agent plants (`status`,
    `diff`, `log -p`, `show`, `blame`, `ls-files -m`) and `diff --output`
    writes anywhere — a reviewer's as much as a worker's, and `gitk`,
    `scalar` and `git-*` read the same config. Git reads inside the
    sandbox; no rule lets it out."""
    rule = r"(?i)(git|gitk|scalar|git-\S+).*"
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=rule, role=role)
    for command in ("git status", "git log --format=%H -1", "git diff",
                    "git diff --no-ext-diff", "git log -p", "git show HEAD",
                    "git blame a.txt", "git ls-files -m", "git cat-file --textconv HEAD:a",
                    "git diff --output=C:/anywhere/x.txt", "GIT.EXE status",
                    "git.cmd status", "git commit -m x", "git rebase main",
                    "git -c core.hooksPath=.githooks status", "gitk --all",
                    "scalar list", "git-receive-pack x", "Git-Upload-Pack.exe x"):
        _refused(ask, command)
        assert "does not leave the sandbox" in _why(command, rule), command
    assert _why("gitlab-ci-lint x", r"gitlab-ci-lint.*") == ""


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_a_cmd_shim_s_metacharacters_are_refused(tmp_path, monkeypatch, role):
    """Codex review of 2445: `npm.cmd test 'a&whoami'` matched `npm(\\.cmd)?
    test( .*)?`, PowerShell handed the unspaced quoted word to the shim bare
    and cmd ran `whoami`. Claude's: a `%…%` pair expands in cmd
    (`%CMDCMDLINE%` brings quotes). The rule cannot see whether a name
    resolves to a `.cmd`, so every escalated command is judged as one."""
    rule = r"(npm(\.cmd)?|shimx) test( .*)?"
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=rule, role=role)
    for command in ("npm.cmd test 'a&whoami'", "npm test 'a|whoami'",
                    "npm test 'a<b'", "npm test 'a>b'", "npm test 'a^b'",
                    "shimx test 'a&whoami'", "npm test 'a b!'",
                    "shimx test x%CMDCMDLINE%", "shimx test %PATH%",
                    "shimx test 'x&echo INJECT' %CMDCMDLINE%",
                    "shimx test --a=%x --b=y%"):
        _refused(ask, command)
        assert "cmd" in _why(command, rule), command
    for command in ("npm test -- -t 'a & b'", "shimx test --format=%H",
                    "shimx test 100%"):
        assert _why(command, rule) == "", command


def test_honest_gates_pass_as_recorded(tmp_path, monkeypatch):
    """The anchored gate rule (`gates.escalate_pattern`) under the whole-
    command match: quotes kept as typed, either slash, `2>&1` read through."""
    gates = ["npm run typecheck",
             "node node_modules/vitest/vitest.mjs run --root C:/w test/x.test.ts",
             "node C:\\w\\node_modules\\vitest\\vitest.mjs run --root C:\\w test/x.test.ts",
             "python tools/try_patch.py --file a.ts --old 'a' --new 'b//c' -- npm test",
             "python C:/w/tools/selftest_run.py physics",
             "python tools/webgame/list_endpoints.py --check"]
    _, ask = worker_judge(tmp_path, monkeypatch, gates=gates, role="reviewer")
    for command in (*gates, "npm run typecheck 2>&1",
                    "node node_modules\\vitest\\vitest.mjs run --root C:/w test/x.test.ts"):
        verdict = ask(command)
        assert verdict and verdict.reason == GRANTED, (command, verdict.reason)
    for command in ("npm run typecheck -- --x", "npm run typecheck > out.txt"):
        _refused(ask, command)


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_a_typographic_quote_is_refused_inside_a_string(tmp_path, monkeypatch, role):
    """PowerShell closes `'…` on `’`: an ASCII-only reader saw one literal
    and PowerShell ran the tail — past the worker's shapes too."""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"python -m pytest.*", role=role)
    for command in ("python -m pytest -k 'x\u2019 ; Set-Content a.txt b'",
                    'python -m pytest -k "x\u201d ; Set-Content a.txt b"'):
        assert "typographic" in _refused(ask, command)
    assert ask("python -m pytest -k 'x ; y'")


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_a_lone_carriage_return_ends_a_command(tmp_path, monkeypatch, role):
    """Both reviews of 2445: PowerShell 5.1 ends a statement at a lone CR;
    read as a space, `npm test\\rSet-Content …` matched `npm test( .*)?`."""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"npm test( .*)?", role=role)
    for command in ("npm test\rSet-Content C:\\x\\a.txt b",
                    "npm test\rpython -c 'print(1)'",
                    "npm test -k 'a\rb'", "npm test -k 'a\nb'"):
        _refused(ask, command)
    for command in ("npm test\r\nnpm test", "npm test\rnpm test -- x"):
        assert ask(command).reason == GRANTED, command


# Each splits words in PowerShell 5.1 (or is its parameter dash) but was
# plain text to an ASCII reader (reviews of 2445); `\x1b` and U+00E9 stand
# for the rest of the non-printable and non-ASCII range.
_UNREADABLE = ["\u00a0", "\u2003", "\u3000", "\f", "\v", "\u0085", "\u2028",
               "\u2013", "\u2014", "\x00", "\x1b", "\x7f", "\u00e9"]


@pytest.mark.parametrize("char", _UNREADABLE, ids=lambda c: f"U+{ord(c):04X}")
@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_a_character_outside_printable_ascii_is_refused_anywhere(
        tmp_path, monkeypatch, role, char):
    """Deny by default (`command_shell.unreadable_character`): outside a
    string and inside one, under the escalate rule and the worker shapes."""
    _, ask = worker_judge(tmp_path, monkeypatch, role=role,
                          escalate=r"(git|npm test|python -m pytest)\s.*")
    for command in (f"git{char}-c{char}core.fsmonitor=x{char}status",
                    f"npm test{char}x",
                    f"python -m pytest -k 'a{char}b'",
                    f"npm test 2>&1 | Select-Object {char}First:1",
                    f"npm test | Select-String x {char}Path:C:/outside/a.txt"):
        assert "U+" in _refused(ask, command), ascii(command)


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_a_quoted_string_powershell_5_splits_is_refused(tmp_path, monkeypatch, role):
    """Measured on Windows PowerShell 5.1 (`_native_quoting_refusal`): its
    legacy native quoting hands `-k 'a\\" -p b'` on as `-k`, `a"`, `-p`, `b`,
    a spaced value ending in `\\` swallows the next arguments, an empty one
    vanishes — while the reader saw one word each."""
    _, ask = worker_judge(tmp_path, monkeypatch, role=role,
                          escalate=r"python -m pytest -k \S+")
    for command in ("python -m pytest -k 'a\\\" -p b'",
                    "python -m pytest -k 'x\" \"-p\" \"evil'",
                    "python -m pytest -k \"a\"\"b\"",
                    "python -m pytest -k 'a b\\'", "python -m pytest -k \"a b\\\"",
                    "python -m pytest -k ''", "python -m pytest -k \"\""):
        assert "PowerShell 5.1" in _refused(ask, command), command
    assert ask("python -m pytest -k 'C:\\d\\'")
    assert _why("python -m pytest -k 'C:\\d\\'", r"python -m pytest -k \S+") == ""


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_only_codex_s_own_preamble_is_taken_off(tmp_path, monkeypatch, role):
    """Codex review of 2445: any `try { … OutputEncoding … } catch {}` was
    stripped as the wrapper's, so the agent's own code ran under a rule
    for `npm test`. Only `UTF8_OUTPUT_PREFIX` of codex-rs is the wrapper's."""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"npm test", role=role)
    shell = '"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command '
    preamble = "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n"
    assert ask(f"{shell}'{preamble}npm test'", wrap=False).reason == GRANTED
    for body in ("try { Write-Output OutputEncoding } catch {}; npm test",
                 "try { [Console]::OutputEncoding=1; Set-Content a b } catch {}\nnpm test",
                 f"{preamble}{preamble}npm test"):
        verdict = ask(f"{shell}'{body}'", wrap=False)
        assert not verdict, (body, verdict.reason)


def test_powershell_literals_the_reader_follows():
    """Measured 2026-11-18 on Windows PowerShell 5.1 (`_bracket_literal`):
    `%` and an inner `[ ]` reach a native program as typed; the stop-parsing
    word, a leading `[`, a `[` in a parameter word, and both outside a
    PowerShell wrapper do not."""
    assert _words("x --format=%H 100% tests/x.py::t[1] a]b", "powershell") == [
        Word("x"), Word("--format=%H"), Word("100%"), Word("tests/x.py::t[1]"),
        Word("a]b")]
    assert _words("x --cov=src.pkg -x:a.b -1.5 a-b.c", "powershell") == [
        Word("x"), Word("--cov=src.pkg"), Word("-x:a.b"), Word("-1.5"),
        Word("a-b.c")]
    for text in ("x --% y", "x [int]", "x -k=a[x]", "x -Dfoo.bar=1",
                 "x -k=a.b", "x -m.x"):
        with pytest.raises(Unparsed):
            _words(text, "powershell")
    for text in ("x --format=%H", "x a[0]"):
        with pytest.raises(Unparsed):
            _words(text, "")
