"""The `escalate` rule: an operator's regex (`spawn --escalate`) or the
recorded gates (`gates.escalate_pattern`), judged against a command an agent
asks to run outside the sandbox (`sandbox.escalation_judge`)."""

from __future__ import annotations

import re
from typing import Pattern

from .command_dispatch import _pipe_refusal
from .command_shell import (
    Token,
    Unparsed,
    Word,
    _MERGE,
    _REDIRECTS,
    _SEPARATORS,
    _lexed,
    split_shell,
)

# Programs that run a command they are handed rather than being one: a rule
# matching `npm test` in their argument would grant whatever they make of it
# (round 3 of 2432, Codex: `& $env:ComSpec /c 'npm test'`). `%` and
# `foreach-object` call a method named by their argument on every input
# (`… | % Delete`). Lower-case, without `.exe`.
_INDIRECT_PROGRAMS = frozenset({
    ".", "invoke-expression", "iex", "invoke-command", "icm", "start-process",
    "saps", "start", "start-job", "sajb", "start-threadjob", "invoke-item", "ii",
    "powershell", "pwsh", "cmd", "bash", "sh", "zsh", "wsl", "env", "runas",
    "wscript", "cscript", "mshta", "rundll32", "conhost", "forfiles",
    "%", "foreach", "foreach-object"})

_PROGRAM_SUFFIXES = (".exe", ".com", ".cmd", ".bat", ".ps1")


def _program(word: str) -> str:
    """`word` as a program name: lower-case, no `.exe`-like suffix."""
    program = word.lower()
    for suffix in _PROGRAM_SUFFIXES:
        if program.endswith(suffix):
            return program[:-len(suffix)]
    return program


def _stages(lexed: list[tuple[Token, str]]) -> list[tuple[list[tuple[Token, str]], bool]]:
    """Every command of the line: (its tokens, whether a pipe feeds it)."""
    stages: list[tuple[list[tuple[Token, str]], bool]] = [([], False)]
    for token, source in lexed:
        if token in _SEPARATORS:
            stages.append(([], False))
        elif token == "|":
            stages.append(([], True))
        else:
            stages[-1][0].append((token, source))
    return [(tokens, piped) for tokens, piped in stages if tokens or piped]


def escalate_refusal(command: str, pattern: Pattern[str], worker: bool) -> str:
    """Why `pattern` does not grant `command`, or "" when it does.

    The rule is matched against each COMMAND of the line, whole
    (`re.fullmatch`: `^`/`\\Z` in a rule are redundant, never needed), not
    searched for in the line: a search for `npm test` granted `npm test;
    python -c …`, `Set-Alias npm C:\\x\\evil.exe; npm test`, `npm test >
    <any path>` and a git commit whose message was `npm test` (round 4 of
    2432's review). So:

    - the line is read through Codex's wrapper into literal words
      (`command_shell._lexed`: no `$`, no `&` call, no `( )`, nothing
      outside printable ASCII), anything else refused;
    - it is cut at `;`, `&&`, `||` and `|`, and EVERY command must match on
      its own — one written as its words' source text joined by single
      spaces (quotes kept as typed, `2>&1` matched with and without it); a
      command fed by a pipe may be one of the text filters a worker's pipe
      may end in (`command_dispatch._pipe_refusal`) instead;
    - no command redirects to a file, none is a program that runs a command
      it is handed (`_INDIRECT_PROGRAMS`), none is a program given by a
      path: `.\\npm` or `C:\\<slot>\\git.exe` is a file the agent may have
      written, and the rule has no roots to tell it from a trusted one —
      a program is named and looked up on the PATH;
    - a worker's git is refused whatever the rule says (nothing after this
      function grants git, so the refusal is final). "Read-only"
      subcommands are no limit: `status`, `diff`, `log -p`, `show`, `blame`
      and `ls-files -m` run the fsmonitor, external diff or textconv
      programs of a repository config the worker can plant under any cwd
      of its workspace, and `diff --output=<path>` writes anywhere (both
      reviews of 2445). Its reads run inside the sandbox; its commits and
      rebases go through the brokers.

    An operator who wants arguments says so in the rule: `npm test( .*)?`.
    """
    shell, inner = split_shell(command)
    try:
        lexed = _lexed(inner, shell)
    except Unparsed as unparsed:
        return f"the escalate rule reads literal commands only: {unparsed}"
    stages = _stages(lexed)
    if not stages:
        return "an empty command"
    for tokens, piped in stages:
        why = _stage_refusal(tokens, piped, pattern, worker)
        if why:
            return why
    return ""


def _stage_refusal(tokens: list[tuple[Token, str]], piped: bool,
                   pattern: Pattern[str], worker: bool) -> str:
    if not tokens:
        return "a pipe into nothing"
    for token, _ in tokens:
        if token in _REDIRECTS:
            return f"`{token}`: a command the escalate rule grants writes no file"
    words = [token for token, _ in tokens if isinstance(token, Word)]
    if not words or words[0].quoted:
        return "a command with no plain program name"
    program = words[0].value
    if re.search(r"[\\/:]", program):
        return f"`{program}`: programs are named, not given by path"
    name = _program(program)
    if name in _INDIRECT_PROGRAMS:
        return f"`{program}` runs a command it is handed"
    if worker and name == "git":
        return ("a worker's git does not leave the sandbox: its reads run "
                "inside, its commits and rebases through the brokers")
    full = " ".join(source for _, source in tokens)
    bare = " ".join(source for token, source in tokens if token != _MERGE)
    if pattern.fullmatch(full) or pattern.fullmatch(bare):
        return ""
    if piped and not _pipe_refusal(words):
        return ""
    return f"`{bare}` does not match the escalate rule /{pattern.pattern}/"
