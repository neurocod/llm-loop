"""Worker command composition and dispatch. Trust model: commands.py.

The shapes themselves — which programs, judged how — are the host's
(`HostPolicy.commands.shapes`, read off `CommandContext.policy`); the judges a
host builds them from are command_node.py, command_python.py and
command_commit.py, each handed `argv_refusal` for the commands it runs in turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from ..host_policy import HostPolicy, active as active_policy
from .command_arguments import (
    CommandContext,
    _arg_path_refusal,
)
from .command_shell import (
    Token,
    Unparsed,
    Word,
    _MERGE,
    _REDIRECTS,
    _segments,
    _words,
    split_shell,
)

@dataclass(frozen=True)
class Shape:
    """One program a worker may run outside the sandbox: its names, how the
    briefing and a decline name it, and the judge of its arguments."""
    names: tuple[str, ...]
    text: str
    judge: Callable[[list[str], Path, CommandContext], str]


def argv_refusal(argv: list[str], cwd: Path, ctx: CommandContext) -> str:
    """Why one command's words may not run outside, or "": the program,
    named and not given by path, must be a shape of the host's, whose judge
    then reads the rest. The judge a recursive shape (a runner of another
    command) is bound to."""
    if not argv:
        return "an empty command"
    program = argv[0].lower()
    if re.search(r"[\\/]", program):
        return f"{argv[0]}: programs are named, not given by path"
    for suffix in _PROGRAM_SUFFIXES:
        if program.endswith(suffix):
            program = program[:-len(suffix)]
            break
    shape = next((shape for shape in ctx.policy.commands.shapes
                  if program in shape.names), None)
    if shape is None:
        return f"`{argv[0]}` is not a worker command shape"
    return shape.judge([program, *argv[1:]], cwd, ctx)


_PROGRAM_SUFFIXES = (".exe", ".cmd", ".bat", ".ps1")


# A pipe may feed one of these and nothing else: they read text and print
# it. Not `Select-Object -First`: it closes the pipe early, and the program
# writing into it dies mid-run (try_patch.py's help: before its restore).
_PIPE_FILTERS = {
    "select-object": {"-last": True, "-skip": True, "-unique": False},
    "select": {"-last": True, "-skip": True, "-unique": False},
    "select-string": {"-pattern": True, "-simplematch": False,
                      "-notmatch": False, "-casesensitive": False,
                      "-context": True, "-allmatches": False},
    "sls": {"-pattern": True, "-simplematch": False, "-notmatch": False,
            "-casesensitive": False, "-context": True, "-allmatches": False},
    "out-string": {"-stream": False, "-width": True},
    "measure-object": {"-line": False, "-word": False, "-character": False},
}


def _pipe_refusal(words: list[Word]) -> str:
    if not words or words[0].quoted:
        return "a pipe into nothing"
    options = _PIPE_FILTERS.get(words[0].value.lower())
    if options is None:
        return f"a pipe into {words[0].value}"
    positional = 0
    at = 1
    while at < len(words):
        word = words[at]
        lowered = word.value.lower()
        if not word.quoted and lowered.startswith("-"):
            if lowered not in options:
                return f"{words[0].value} {word.value}"
            at += 2 if options[lowered] else 1
            continue
        positional += 1
        if positional > 1:
            return f"{words[0].value} with more than one positional argument"
        at += 1
    return ""


def _segment_refusal(segment: list[Token], cwd: Path,
                     ctx: CommandContext) -> str:
    pipes: list[list[Token]] = [[]]
    for token in segment:
        if token == "|":
            pipes.append([])
        else:
            pipes[-1].append(token)
    words: list[str] = []
    first: list[Token] = pipes[0]
    at = 0
    while at < len(first):
        token = first[at]
        if token == _MERGE:
            at += 1
            continue
        if token in _REDIRECTS:
            target = first[at + 1] if at + 1 < len(first) else None
            if not isinstance(target, Word):
                return f"`{token}` with no file"
            why = _arg_path_refusal(target.value, cwd, ctx)
            if why:
                return f"redirect {why}"
            at += 2
            continue
        if not isinstance(token, Word):
            return f"`{token}` where a word was expected"
        if at == 0 and token.quoted:
            return "a quoted program name"
        if '"' in token.value:
            # PowerShell 5.1 passes such a word to a program unescaped, and
            # the program splits it there: `'x" C:\\y'` arrives as two
            # arguments, the second a path the judge never saw.
            return (f"{token.value!r}: a `\"` inside an argument (PowerShell "
                    f"5.1 hands it on unescaped, and the program splits it)")
        words.append(token.value)
        at += 1
    for pipe in pipes[1:]:
        if any(not isinstance(token, Word) for token in pipe):
            return "an operator inside a pipe"
        why = _pipe_refusal([token for token in pipe if isinstance(token, Word)])
        if why:
            return why
    return argv_refusal(words, cwd, ctx)


def worker_command_refusal(command: str, cwd: Path,
                           ctx: CommandContext) -> str:
    """Why a worker's command may NOT leave the sandbox, or "" when it may.

    Granted only when the whole command — read through Codex's wrapper
    (`split_shell`) into literal words (`_words`) and cut at `;`, `&&`,
    `||` — is commands of the host's shapes (`ctx.policy.commands.shapes`),
    each judged by its shape: known options only, every path it names inside
    the writable roots and outside the host's `protected_parts`
    (`path_refusal`, resolved against `cwd` or the command's own
    `--prefix`/`--cwd`), and whatever else the host's judges hold it to.
    Anything else is a "no": a cmdlet, git, a nested shell, an operator the
    reader does not follow — and, under the empty policy, every program.
    What it guards against, and what not, is the commands.py docstring's trust
    model.
    """
    shell, inner = split_shell(command)
    try:
        tokens = _words(inner, shell)
    except Unparsed as unparsed:
        return f"{unparsed} (the judge reads plain words only)"
    segments = _segments(tokens)
    if not segments:
        return "an empty command"
    for segment in segments:
        why = _segment_refusal(segment, cwd, ctx)
        if why:
            return why
    return ""


def shapes_text(policy: Optional[HostPolicy] = None) -> str:
    """The host's shapes as the briefing lists them (the installed host's
    when `policy` is not given)."""
    if policy is None:
        policy = active_policy()
    if not policy.commands.shapes:
        return "none (the host policy names no command shapes)"
    return "; ".join(shape.text for shape in policy.commands.shapes)
