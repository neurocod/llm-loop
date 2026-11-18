"""The `escalate` rule: an operator's regex (`spawn --escalate`) or the
recorded gates (`gates.escalate_pattern`), judged against a command an agent
asks to run outside the sandbox (`sandbox.escalation_judge`)."""

from __future__ import annotations

import re
from typing import Pattern

from .command_dispatch import _pipe_refusal
from .command_node import _SHIM_BARE
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

# Git and the programs that ship with it (`C:\Program Files\Git\cmd` holds
# `gitk`, `scalar`, `git-receive-pack`, …): each reads the repository's
# config and hooks from its cwd, so fsmonitor, textconv and external diff
# run what the agent planted, and `diff --output=<path>` writes anywhere —
# outside the sandbox under an escalate grant, for a worker and a reviewer
# alike (both reviews of 2445). Their reads run inside the sandbox; a
# worker's commits and rebases go through the brokers.
_GIT_FAMILY = frozenset({"git", "gitk", "scalar"})


def _git_family(name: str) -> bool:
    return name in _GIT_FAMILY or name.startswith("git-")


# The rule names a program but cannot see what it resolves to, and `npm`,
# `npx`, `yarn` and any tool the PATH holds as a `.cmd`/`.bat` are cmd
# shims: PowerShell writes their command line for cmd.exe, which acts on
# `& | < > ^` in a word it sees unquoted (`npm.cmd test 'a&whoami'` runs
# `whoami`; Codex review of 2445), on `!`, a line break, and on a `%…%`
# pair anywhere (`%CMDCMDLINE%` brings its own quotes: `shim 'x&echo
# INJECT' %CMDCMDLINE%` ran `echo INJECT`, Claude review of 2445). So
# every escalated command is judged as if it were one. A single `%` stays
# (`--format=%H`): cmd leaves it be. PowerShell double-quotes a word holding
# whitespace, which is why `-k 'a or b'` passes (`command_node._shim_refusal`
# is the same rule for npm/npx, without the single `%`).
_SHIM_ALWAYS = re.compile(r"[!\n\r]")


def _shim_refusal(args: list[str]) -> str:
    for arg in args:
        quoted = bool(re.search(r"\s", arg)) and not arg.endswith("\\")
        if _SHIM_ALWAYS.search(arg) or (_SHIM_BARE.search(arg) and not quoted):
            return (f"{arg!r}: the program may be a cmd shim, and cmd acts on "
                    f"! anywhere and on & | < > ^ in a word with no space")
    if sum(arg.count("%") for arg in args) > 1:
        return ("two `%` in one command: cmd expands a `%…%` pair if the "
                "program is a cmd shim")
    return ""


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


def escalate_refusal(command: str, pattern: Pattern[str]) -> str:
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
    - git and its family (`_GIT_FAMILY`) are refused whatever the rule says
      and whoever asks. "Read-only" subcommands are no limit: `status`,
      `diff`, `log -p`, `show`, `blame` and `ls-files -m` run the
      fsmonitor, external diff or textconv programs of a repository config
      the agent can plant under any cwd of its workspace;
    - every command's arguments pass the cmd-shim rule (`_shim_refusal`)
      before the rule may grant it.

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
        why = _stage_refusal(tokens, piped, pattern)
        if why:
            return why
    return ""


def _stage_refusal(tokens: list[tuple[Token, str]], piped: bool,
                   pattern: Pattern[str]) -> str:
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
    if _git_family(name):
        return (f"`{program}` does not leave the sandbox: git reads run "
                f"inside, a worker's commits and rebases through the brokers")
    why = _shim_refusal([word.value for word in words[1:]])
    if why:
        return why
    full = " ".join(source for _, source in tokens)
    bare = " ".join(source for token, source in tokens if token != _MERGE)
    if pattern.fullmatch(full) or pattern.fullmatch(bare):
        return ""
    if piped and not _pipe_refusal(words):
        return ""
    return f"`{bare}` does not match the escalate rule /{pattern.pattern}/"
