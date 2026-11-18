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
# The ONE preamble Codex prepends to a PowerShell script, character for
# character: `UTF8_OUTPUT_PREFIX` in codex-rs/shell-command/src/powershell.rs
# (`prefix_powershell_script_with_utf8`, which also leaves a script already
# starting with it alone). Only this text is taken off: a pattern for "any
# `try { … OutputEncoding … } catch {}`" took `try { Write-Output
# OutputEncoding } catch {}; npm test` for the wrapper's and granted the
# agent's own code under a rule for `npm test` (Codex review of 2445).
_ENCODING_PREAMBLE = (
    "try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } catch {}\n")
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


def _without_preamble(script: str) -> str:
    """`script` with Codex's own preamble (`_ENCODING_PREAMBLE`) taken off
    the front, once, and stripped; anything else is the agent's code."""
    script = script.lstrip()
    if script.startswith(_ENCODING_PREAMBLE):
        script = script[len(_ENCODING_PREAMBLE):]
    return script.strip()


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
    its quoting, and the exact preamble Codex prepends
    (`_ENCODING_PREAMBLE`).

    The quoting is POSIX shell quoting of the argv Codex starts (measured
    2026-11-03: 168 of 168 logged requests in 39 agents' events.jsonl split
    by `shlex.split` into exactly `[powershell.exe, -Command, <script>]`). A
    script holding a `'` arrives double-quoted with its backslashes doubled
    and every `!` cut out into single quotes (`--old 'if ("'!x) y;'"'`), so a
    reader that only stripped the outer quotes judged `C:\\\\work\\\\...` paths
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
        return shell, _without_preamble(inner)
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
            _without_preamble(inner))


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
# expressions, arrays and splats, `,` builds an array (`a, C:\x` is two
# arguments, a trailing `x,` takes the NEXT LINE as more of them), `#`
# comments the rest out, `~` and wildcards expand for cmdlets (and for the
# scripts that glob their own arguments), `< !` are operators or reserved.
# `[ ]` are read by `_bracket_literal` instead, `%` by `_STOP_PARSING`.
_PS_SPECIAL = frozenset("$`(){}@,#~!^*?<")
# ...and the ones cmd.exe acts on outside quotes (`%` expands even inside).
_CMD_SPECIAL = frozenset("%^!()<'*?,;=@`")
# PowerShell takes the typographic quotes as quotes too — and closes a `'…`
# string on a `’` (any of the single ones, any of the double ones for `"…`).
# Refused ANYWHERE, inside a string as well: a reader that kept only the
# ASCII rule saw `'x’ ; Set-Content …'` as one literal while PowerShell ran
# the tail (review of 2432, round 4). Re-implementing the rule buys an agent
# nothing it cannot type in ASCII.
_CURLY_QUOTES = frozenset("\u2018\u2019\u201a\u201b\u201c\u201d\u201e")
# PowerShell's stop-parsing token: the rest of the line goes to the program
# raw, `%VAR%` expanded and `;` no separator. (PowerShell's other dashes \u2014
# en, em, horizontal bar \u2014 never reach here: `unreadable_character`.)
_STOP_PARSING = "--%"
_UNKNOWN_SHELL_SPECIAL = frozenset("%[]")


def unreadable_character(text: str) -> str:
    """Why `text` holds a character no judge reads, or "".

    Deny by default: only printable ASCII, a space, a tab and the line
    breaks (`\\r` and `\\n`, separators to the reader) \u2014 anywhere, inside a
    string too. PowerShell 5.1 gives meaning to characters a reader written
    for ASCII takes as plain text: it splits words at NBSP, U+2003, U+3000,
    `\\f`, `\\v`, U+0085 and U+2028 (`git<NBSP>-c<NBSP>\u2026` hid a git option
    from the escalate rule's guard), takes the en dash as a parameter prefix
    (`Select-Object \u2013First:1` passed the pipe filters' allowlist,
    `\u2013Path:C:/outside/\u2026` made Select-String read a file) and closes a string
    on a typographic quote (reviews of 2432 round 4 and of 2445). Listing
    those is never finished; an agent loses nothing it cannot type in ASCII.
    """
    for char in text:
        if char in _CURLY_QUOTES:
            return f"a typographic quote {char!r}"
        if not (" " <= char <= "~" or char in "\t\r\n"):
            return (f"the character U+{ord(char):04X} (only printable ASCII, "
                    f"spaces, tabs and line breaks are read)")
    return ""


def _bracket_literal(current: list[str]) -> bool:
    """Whether PowerShell keeps a `[` or `]` literal in the bare word
    `current` (measured 2026-11-18 on Windows PowerShell 5.1, argv of a
    native program): after the first character of a word that is no
    parameter (`tests/x.py::t[1]`, `a]b`). At its start `[int]` is a type
    literal in command position, and after a leading `-` PowerShell splits
    the word there (`-k=a[x]` arrives as `-k=a`, `[x]`).

    `%` needs no rule: the same measurement passed `--format=%H`, `100%`,
    `%a%` and `--%x` literally — only the whole word `--%` is not
    (`_STOP_PARSING`)."""
    return bool(current) and current[0] != "-"


def _native_quoting_refusal(literal: str) -> str:
    """Why PowerShell 5.1 would not hand the quoted string `literal` to a
    native program as ONE argument, or "".

    Its legacy quoting writes the command line without escaping: it wraps a
    value holding whitespace in `"…"` as is and drops an empty one.
    Measured 2026-11-18 (Windows PowerShell 5.1, a Python program printing
    its argv): `-k 'a\\" -p b'` arrives as `-k`, `a"`, `-p`, `b`;
    `-k 'a b\\' -p evil` and `-k "a b\\" -p evil` as `-k`, `a b" -p evil`;
    `-k '' -p evil` as `-k`, `-p`, `evil`. The reader's one word would be
    the rule's or the judge's reading, not the program's."""
    if '"' in literal:
        return "a `\"` inside a quoted string (PowerShell 5.1 hands it on unescaped)"
    if not literal:
        return "an empty quoted string (PowerShell 5.1 drops it)"
    if literal.endswith("\\") and re.search(r"[ \t]", literal):
        return ("a quoted string with a space ending in `\\` (PowerShell 5.1 "
                "quotes it into an escaped `\"`)")
    return ""


def _words(text: str, shell: str) -> list[Token]:
    """The words and operators of `text` under `shell` ("" reads as
    PowerShell without its measured relaxations, the strictest); raises
    `Unparsed`.

    Only what passes a LITERAL string to a program is read: bare words of
    plain characters, single-quoted strings (PowerShell: `''` is a quote),
    double-quoted ones with nothing to expand, `;`/`&&`/`||` between
    commands, `|` (judged by `_pipe_refusal`), output redirection and
    `2>&1`. A quote glued to other text (`--x='a b'`, `'a'b`) is refused —
    PowerShell's rules for it differ by context, and guessing is how a
    reader grants what the shell does not run — and so is any character
    outside printable ASCII (`unreadable_character`), a line break inside a
    string, and a string PowerShell would not pass as one argument
    (`_native_quoting_refusal`).
    """
    return [token for token, _ in _lexed(text, shell)]


def _lexed(text: str, shell: str) -> list[tuple[Token, str]]:
    """`_words`, each token with its source text (a quoted word's quotes
    included): what the escalate rule matches (`sandbox.escalate_refusal`)."""
    cmd = shell == "cmd"
    # The measured PowerShell relaxations (`%`, `[ ]`) hold under a
    # PowerShell wrapper only; with none the shell is unknown.
    special = (_CMD_SPECIAL if cmd else _PS_SPECIAL if shell == "powershell"
               else _PS_SPECIAL | _UNKNOWN_SHELL_SPECIAL)
    unreadable = unreadable_character(text)
    if unreadable:
        raise Unparsed(unreadable)
    out: list[tuple[Token, str]] = []
    current: list[str] = []
    started = 0
    at = 0
    end = len(text)

    def flush() -> None:
        if current:
            value = "".join(current)
            if not cmd and value == _STOP_PARSING:
                raise Unparsed(f"`{value}` (PowerShell's stop-parsing token)")
            out.append((Word(value), text[started:at]))
            current.clear()

    def operator(op: str) -> None:
        out.append((op, op))

    while at < end:
        char = text[at]
        if char in " \t":
            flush()
            at += 1
            continue
        if char in "\r\n":
            # A lone CR ends a statement in PowerShell 5.1 as LF does: read
            # as a space, `npm test\rSet-Content …` was granted by
            # `npm test( .*)?` (both reviews of 2445).
            flush()
            operator(";")
            at += 1
            continue
        if char == '"' or (char == "'" and not cmd):
            if current:
                raise Unparsed("a quote glued to other text")
            started = at
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
                if text[at] in "\r\n":
                    raise Unparsed("a line break inside a quoted string")
                value.append(text[at])
                at += 1
            literal = "".join(value)
            if char == '"' and re.search(r"[$`]" if not cmd else r"[%!]",
                                         literal):
                raise Unparsed("a double-quoted string the shell expands")
            if not cmd:
                why = _native_quoting_refusal(literal)
                if why:
                    raise Unparsed(why)
            at += 1
            if at < end and text[at] not in " \t\r\n;|&>":
                raise Unparsed("a quote glued to other text")
            out.append((Word(literal, quoted=True), text[started:at]))
            continue
        if not current and text.startswith(_MERGE, at):
            operator(_MERGE)
            at += len(_MERGE)
            continue
        if not current and char in "2*" and text.startswith(">", at + 1):
            op = char + (">>" if text.startswith(">>", at + 1) else ">")
            operator(op)
            at += len(op)
            continue
        if char == ">":
            flush()
            op = ">>" if text.startswith(">>", at) else ">"
            operator(op)
            at += len(op)
            continue
        if char == "&":
            flush()
            if text.startswith("&&", at):
                operator("&&")
                at += 2
            elif cmd:
                operator(";")
                at += 1
            else:
                raise Unparsed("`&` (PowerShell's call operator)")
            continue
        if char == "|":
            flush()
            op = "||" if text.startswith("||", at) else "|"
            operator(op)
            at += len(op)
            continue
        if char == ";" and not cmd:
            flush()
            operator(";")
            at += 1
            continue
        if char in special or (not cmd and char in "[]"
                               and not _bracket_literal(current)):
            raise Unparsed(f"`{char}` outside quotes")
        if not current:
            started = at
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
