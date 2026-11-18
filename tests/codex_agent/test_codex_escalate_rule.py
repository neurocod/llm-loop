"""The `escalate` rule matched against each command of a line, whole
(`command_escalate.escalate_refusal`), and the literal reader under it
(`command_shell._words`): round 4 of 2432's review (item 2445)."""

import pytest

from llm_agents.codex.command_shell import Unparsed, Word, _words
from llm_agents.codex.testing import worker_judge

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
    assert "relative path" in _refused(ask, ".\\npm test")
    assert "runs a command it is handed" in _refused(ask, "npm test | % Delete")
    assert ask("npm test -- a.test.ts").reason == GRANTED


def test_a_workers_git_only_reads_whatever_the_rule_says(tmp_path, monkeypatch):
    """A worker's commit and rebase go through the brokers; a rule matching
    git does not reopen them, and no global option runs its hooks."""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"git .*")
    for command in ("git commit -m x", "git rebase main",
                    "git -c core.hooksPath=.githooks status",
                    "git -C C:\\other log"):
        _refused(ask, command)
    for command in ("git status", "git log --format=%H -1"):
        verdict = ask(command)
        assert verdict and verdict.reason == GRANTED, (command, verdict.reason)
    _, reviewer = worker_judge(tmp_path, monkeypatch, escalate=r"git .*", role="reviewer")
    assert reviewer("git commit -m x").reason == GRANTED


def test_honest_gates_pass_as_recorded(tmp_path, monkeypatch):
    """The anchored gate rule (`gates.escalate_pattern`) under the whole-
    command match: quotes kept as typed, either slash, `2>&1` read through."""
    gates = ["npm run typecheck",
             "node node_modules/vitest/vitest.mjs run --root C:/w test/x.test.ts",
             "python tools/try_patch.py --file a.ts --old 'a' --new 'b//c' -- npm test"]
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


def test_powershell_literals_the_reader_follows():
    """Measured 2026-11-18 on Windows PowerShell 5.1 (`_bracket_literal`):
    `%` and an inner `[ ]` reach a native program as typed; the stop-parsing
    word, a leading `[`, a `[` in a parameter word, and both outside a
    PowerShell wrapper do not."""
    assert _words("x --format=%H 100% tests/x.py::t[1] a]b", "powershell") == [
        Word("x"), Word("--format=%H"), Word("100%"), Word("tests/x.py::t[1]"),
        Word("a]b")]
    for text in ("x --% y", "x [int]", "x -k=a[x]"):
        with pytest.raises(Unparsed):
            _words(text, "powershell")
    for text in ("x --format=%H", "x a[0]"):
        with pytest.raises(Unparsed):
            _words(text, "")
