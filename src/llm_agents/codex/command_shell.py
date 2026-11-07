"""Codex shell wrappers and literal shell words."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Optional
from typing import Union

# --- Codex's wrapper -----------------------------------------------------

_SHELL_WRAPPER = re.compile(
    r"""^\s*"?[^"\s]*(?P<shell>powershell|pwsh|cmd)(?:\.exe)?"?\s+"""
    r"""(?:(?:-NoProfile|-NonInteractive|-NoLogo|/d|/s|/q)\s+)*"""
    r"""(?:-Command|-c|/c)\s+(?P<inner>.*)$""", re.DOTALL | re.IGNORECASE)
_ENCODING_PREAMBLE = re.compile(
    r"^try\s*\{[^}]*OutputEncoding[^}]*\}\s*catch\s*\{\s*\}\s*;?\s*", re.DOTALL)
_SHELL_NAMES = {"powershell": "powershell", "pwsh": "powershell", "cmd": "cmd"}
_SHELL_OPTIONS = frozenset({"-noprofile", "-noninteractive", "-nologo", "/d",
                            "/s", "/q"})
_SHELL_COMMAND = frozenset({"-command", "-c", "/c"})


def _shell_argv(argv: list[str]) -> Optional[tuple[str, str]]:
    """(shell, script) when `argv` is a shell started on a command string —
    its own options, then `-Command`/`-c`/`/c` and the script — else None."""
    if len(argv) < 3:
        return None
    program = re.split(r"[\\/]", argv[0])[-1].lower()
    program = program[:-4] if program.endswith(".exe") else program
    shell = _SHELL_NAMES.get(program)
    if shell is None:
        return None
    at = 1
    while at < len(argv) and argv[at].lower() in _SHELL_OPTIONS:
        at += 1
    if at + 1 >= len(argv) or argv[at].lower() not in _SHELL_COMMAND:
        return None
    return shell, " ".join(argv[at + 1:])


def split_shell(command: str) -> tuple[str, str]:
    """The shell Codex wrapped the command in, and the command itself.

    The shell is `"cmd"`, `"powershell"` (which `pwsh` folds into) or `""` when
    nothing recognisable wrapped it — and it is not decoration: the same word
    means different things under each. `rmdir /s /q x` deletes a tree under
    cmd, while under PowerShell `rmdir` is an alias of `Remove-Item` and `/s`
    is the PATH `C:\\s`, so only the wrapper can say whether a `/token` is a
    switch or a target (`sandbox.cleanup_targets`).

    On Windows a request's `command` is not `npm test` but
    `"C:\\...\\powershell.exe" -Command 'npm test'` (measured 2026-09-13: all
    three of a verification agent's requests came wrapped, and an anchored
    `^npm` matched none). The wrapper is the runner's, not the agent's, so the
    judge reads through it: the shell's own flags, the `-Command`/`/c` switch,
    its quoting, and the `[Console]::OutputEncoding` preamble the sandbox
    prepends.

    The quoting is POSIX shell quoting of the argv Codex starts (measured
    2026-11-03: 168 of 168 logged requests in 39 agents' events.jsonl split
    by `shlex.split` into exactly `[powershell.exe, -Command, <script>]`). A
    script holding a `'` arrives double-quoted with its backslashes doubled
    and every `!` cut out into single quotes (`--old 'if ("'!x) y;'"'`), so a
    reader that only stripped the outer quotes judged `C:\\\\g\\\\...` paths
    and quote debris rather than what PowerShell runs. A command that does
    not split that way (no Codex wrapper, unbalanced quotes, an unquoted
    backslashed path to the shell) is read by `_SHELL_WRAPPER` instead: one
    layer of matching quotes, PowerShell's `''` escape undone.
    """
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        argv = []
    wrapped = _shell_argv(argv)
    if wrapped is not None:
        shell, inner = wrapped
        return shell, _ENCODING_PREAMBLE.sub("", inner).strip()
    match = _SHELL_WRAPPER.match(command)
    if match is None:
        return "", command
    shell = match.group("shell").lower()
    inner = match.group("inner").strip()
    if len(inner) >= 2 and inner[0] == inner[-1] and inner[0] in "'\"":
        quote = inner[0]
        inner = inner[1:-1]
        if quote == "'":
            inner = inner.replace("''", "'")
    return ("cmd" if shell == "cmd" else "powershell",
            _ENCODING_PREAMBLE.sub("", inner).strip())


def unwrap_shell(command: str) -> str:
    """The command as the agent typed it, with Codex's shell wrapper taken off
    (`split_shell`). A command with no recognisable wrapper is judged as is."""
    return split_shell(command)[1]


# --- the words of a command ------------------------------------------------


class Unparsed(ValueError):
    """Syntax the reader does not follow; the message names it."""


@dataclass(frozen=True)
class Word:
    """One argument as the program receives it, and whether it was quoted."""
    value: str
    quoted: bool = False


# Operators the reader keeps; everything else that is not a word is Unparsed.
_SEPARATORS = (";", "&&", "||")
_REDIRECTS = (">", ">>", "2>", "2>>", "*>", "*>>")
_MERGE = "2>&1"
Token = Union[Word, str]

# Outside quotes, every one of these makes PowerShell do something other than
# pass a literal word: `$` and a backtick expand, `( ) { } @` start
# expressions, arrays and splats, `,` builds an array (`a,C:\x` is two
# arguments), `#` comments the rest out, `~` and wildcards expand for
# cmdlets, `[ ]` are type literals, `< %` and `!` are operators or reserved.
_PS_SPECIAL = frozenset("$`(){}@,#~!%^[]*?<")
# ...and the ones cmd.exe acts on outside quotes (`%` expands even inside).
_CMD_SPECIAL = frozenset("%^!()<'*?,;=@`")
# PowerShell takes the typographic quotes as quotes too.
_CURLY_QUOTES = frozenset("\u2018\u2019\u201a\u201b\u201c\u201d\u201e")


def _words(text: str, shell: str) -> list[Token]:
    """The words and operators of `text` under `shell` ("" reads as
    PowerShell, the stricter of the two); raises `Unparsed`.

    Only what passes a LITERAL string to a program is read: bare words of
    plain characters, single-quoted strings (PowerShell: `''` is a quote),
    double-quoted ones with nothing to expand, `;`/`&&`/`||` between
    commands, `|` (judged by `_pipe_refusal`), output redirection and
    `2>&1`. A quote glued to other text (`--x='a b'`, `'a'b`) is refused —
    PowerShell's rules for it differ by context, and guessing is how a
    reader grants what the shell does not run.
    """
    cmd = shell == "cmd"
    special = _CMD_SPECIAL if cmd else _PS_SPECIAL
    out: list[Token] = []
    current: list[str] = []
    at = 0
    end = len(text)

    def flush() -> None:
        if current:
            out.append(Word("".join(current)))
            current.clear()

    while at < end:
        char = text[at]
        if char in " \t\r":
            flush()
            at += 1
            continue
        if char == "\n":
            flush()
            out.append(";")
            at += 1
            continue
        if char in _CURLY_QUOTES:
            raise Unparsed(f"a typographic quote {char!r}")
        if char == '"' or (char == "'" and not cmd):
            if current:
                raise Unparsed("a quote glued to other text")
            value: list[str] = []
            at += 1
            while True:
                if at >= end:
                    raise Unparsed("unbalanced quotes")
                if text[at] == char:
                    if not cmd and at + 1 < end and text[at + 1] == char:
                        value.append(char)
                        at += 2
                        continue
                    break
                value.append(text[at])
                at += 1
            literal = "".join(value)
            if char == '"' and re.search(r"[$`]" if not cmd else r"[%!]",
                                         literal):
                raise Unparsed("a double-quoted string the shell expands")
            at += 1
            if at < end and text[at] not in " \t\r\n;|&>":
                raise Unparsed("a quote glued to other text")
            out.append(Word(literal, quoted=True))
            continue
        if not current and text.startswith(_MERGE, at):
            out.append(_MERGE)
            at += len(_MERGE)
            continue
        if not current and char in "2*" and text.startswith(">", at + 1):
            flush()
            op = char + (">>" if text.startswith(">>", at + 1) else ">")
            out.append(op)
            at += len(op)
            continue
        if char == ">":
            flush()
            op = ">>" if text.startswith(">>", at) else ">"
            out.append(op)
            at += len(op)
            continue
        if char == "&":
            flush()
            if text.startswith("&&", at):
                out.append("&&")
                at += 2
            elif cmd:
                out.append(";")
                at += 1
            else:
                raise Unparsed("`&` (PowerShell's call operator)")
            continue
        if char == "|":
            flush()
            if text.startswith("||", at):
                out.append("||")
                at += 2
            else:
                out.append("|")
                at += 1
            continue
        if char == ";" and not cmd:
            flush()
            out.append(";")
            at += 1
            continue
        if char in special:
            raise Unparsed(f"`{char}` outside quotes")
        current.append(char)
        at += 1
    flush()
    return out


def _segments(tokens: list[Token]) -> list[list[Token]]:
    """`tokens` cut at `;`, `&&` and `||`; empty commands dropped."""
    found: list[list[Token]] = [[]]
    for token in tokens:
        if token in _SEPARATORS:
            found.append([])
        else:
            found[-1].append(token)
    return [segment for segment in found if segment]
