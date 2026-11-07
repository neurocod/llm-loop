"""Command context, path arguments and literal option parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Optional
from ..host_policy import EMPTY, HostPolicy
from .command_paths import (
    _DRIVE,
    _URL,
    _without_long_prefix,
    path_refusal,
    resolve_path,
    within,
)


@dataclass(frozen=True)
class CommandContext:
    """What a worker's command is judged against: its workspace (resolved),
    every writable root (resolved), the scratch root pytest may clear, the
    spawn's online switch, and the host's rules (`policy`) with the flags
    they computed for these roots (`WorkerCommands.flags`)."""
    workspace: Path
    roots: tuple[Path, ...]
    scratch: Optional[Path] = None
    # The record bit `spawn --allow-online-probes` keeps (named so for the
    # registry's and the CLI's sake); only the host's judges read it.
    allow_online_probes: bool = False
    # How many `npm run` bodies deep the command being judged sits.
    npm_depth: int = 0
    commit_approvals: Optional[Path] = None
    policy: HostPolicy = EMPTY
    host_flags: frozenset[str] = frozenset()


# Flags whose VALUE is never a path, wherever they stand: try_patch's text,
# a test-name filter, a pytest expression.
_TEXT_FLAGS = frozenset({"--old", "--new", "--remove", "-t",
                         "--testnamepattern", "-k", "-m"})
# Flags whose value IS a path, even a bare name (`--prefix web`). pytest's
# `-o`/`--override-ini` take `key=value`, whose VALUE is the path
# (`-o cache_dir=../x`, `_value_refusal`).
_PATH_FLAGS = frozenset({
    "--prefix", "--root", "--cwd", "--file", "-c", "--dir", "--basetemp",
    "--rootdir", "--confcutdir", "--out", "--output", "--outdir",
    "--outputfile", "--config", "--from", "--project", "--log", "-o",
    "--override-ini", "--junitxml", "--junit-xml", "--log-file", "--path"})


def _names_output(flag: str) -> bool:
    """Whether a flag's name says its value is a place to write, judged as
    a path with spaces or without (`_path_like` takes a space for text):
    `--out*` (`--outputFile.junit`, `--out-dir`) and `*dir`/`*directory`
    (`--coverage.reportsDirectory`). Third review 2026-11-03 granted
    `--out-dir 'web/../.git/new dir'` and `--outputFile.junit
    'reports/../../../../outside file.xml'`."""
    return flag.startswith("--out") or flag.endswith(("dir", "directory"))
_WILDCARD = re.compile(r"[*?\[]")
# The key of a `key=value` word: a setting's name, never a path of its own.
_ASSIGN_KEY = re.compile(r"[A-Za-z_][\w.-]*")
# ...and the flags whose value is always one (pytest's ini overrides).
_ASSIGN_FLAGS = frozenset({"-o", "--override-ini"})


def _path_like(value: str) -> bool:
    """Whether a word that is not a known flag's value names a path: rooted,
    drive-prefixed, dot-relative or holding a separator (and no space, which
    marks text such as try_patch's `'a / b'`)."""
    if not value:
        return False
    if (_DRIVE.match(value) or value[0] in "\\/" or value[0] == "."
            or _URL.match(value)):
        return True
    return ("/" in value or "\\" in value) and not re.search(r"\s", value)


def _arg_path_refusal(value: str, base: Path, ctx: CommandContext) -> str:
    if _WILDCARD.search(_without_long_prefix(value)):
        return f"{value}: a wildcard in a path (the judge does not expand it)"
    return path_refusal(value, base, ctx.roots, ctx.policy.protected_parts)


def _value_refusal(value: str, base: Path, ctx: CommandContext,
                   is_path: bool = False, assign: bool = False) -> str:
    """Why one argument value may not be used, or "": judged as a path when
    `is_path` or when it looks like one (`_path_like`).

    A `key=value` word is judged by its VALUE too — `cache_dir=../../x`
    read whole is the name `cache_dir=..` and its parent, which normalises
    back INTO the tree (review 2026-11-03 granted `-o cache_dir=../../x`
    that way). `assign`: the flag may define its value as `key=value`
    (pytest's `_ASSIGN_FLAGS`), and then only the value is a path, bare
    names included. Otherwise the whole word is judged as well, unless a `:`
    in it shows it cannot be a relative path at all (`k=C:/x`): a program
    that took the word as a path would resolve it one level deeper than its
    value."""
    key, eq, right = value.partition("=")
    pair = bool(eq and right and _ASSIGN_KEY.fullmatch(key))
    if assign and pair:
        return _value_refusal(right, base, ctx, is_path=True)
    if pair:
        why = _value_refusal(right, base, ctx)
        if why or ":" in value:
            return why
    if value and (is_path or _path_like(value)):
        return _arg_path_refusal(value, base, ctx)
    return ""


def _args_refusal(args: list[str], base: Path, ctx: CommandContext,
                  bare_paths: bool = False) -> str:
    """Why the arguments of an allowed program may not run outside, or "".

    Every path they name — a known path flag's value (`_PATH_FLAGS`), a
    `--flag=value` / `-flag:value` value or a positional that looks like a
    path (`_path_like`), every positional when `bare_paths` (a program that
    WRITES its positionals: `outside` alone may be a junction out of the
    tree) — must pass `path_refusal` against `base` (`_value_refusal`);
    text flags (`_TEXT_FLAGS`) are skipped with their value."""
    skip = False
    path_next = False
    assign = False
    for arg in args:
        if skip:
            skip = False
            continue
        if path_next:
            path_next = False
            why = _value_refusal(arg, base, ctx, is_path=True, assign=assign)
            if why:
                return why
            continue
        if arg.startswith("-") and len(arg) > 1:
            flag, sep, value = _split_flag(arg)
            lowered = flag.lower()
            assign = lowered in _ASSIGN_FLAGS
            if lowered in _TEXT_FLAGS:
                skip = not sep
                continue
            path_flag = lowered in _PATH_FLAGS or _names_output(lowered)
            if sep:
                why = _value_refusal(value, base, ctx, assign=assign,
                                     is_path=path_flag)
                if why:
                    return why
                continue
            path_next = path_flag
            continue
        why = _value_refusal(arg, base, ctx, is_path=bare_paths)
        if why:
            return why
    if path_next:
        return f"{args[-1]} with no value"
    return ""


def _script_args_refusal(script: Path, args: list[str], cwd: Path,
                         ctx: CommandContext) -> str:
    """`_args_refusal` for the arguments of a node/tsx script, from every
    base the script may resolve them against: the cwd, its own folder and
    its package's (the scripts the third review read resolve `--out`
    against their package root or against their own folder). The judge
    does not know which, so a path must stay inside the roots from all
    three (third review 2026-11-03: from a package's scripts/, `--out
    ../../outside` read as the tree and wrote beside it)."""
    bases = [cwd, script.parent]
    for folder in script.parents:
        if not within(folder, ctx.roots):
            break
        if (folder / "package.json").is_file():
            bases.append(folder)
            break
    for base in dict.fromkeys(bases):
        why = _args_refusal(args, base, ctx)
        if why:
            return why
    return ""


def _split_flag(arg: str) -> tuple[str, str, str]:
    """`--a=b` / `-a:b` as (flag, separator, value); a bare flag has no
    separator. The first of `=` and `:` wins, as PowerShell splits `-a:b`."""
    cuts = [at for at in (arg.find("="), arg.find(":")) if at > 0]
    if not cuts:
        return arg, "", ""
    at = min(cuts)
    return arg[:at], arg[at], arg[at + 1:]


def _existing_file(raw: str, base: Path, ctx: CommandContext,
                   what: str) -> tuple[Optional[Path], str]:
    why = _arg_path_refusal(raw, base, ctx)
    if why:
        return None, why
    resolved = resolve_path(raw, base)[1]
    if not resolved.is_file():
        return None, f"{what} {raw} is not a file"
    return resolved, ""


@dataclass(frozen=True)
class _Option:
    takes_value: bool = False
    optional_value: bool = False
    check: Optional[Callable[[str, Path, CommandContext], str]] = None
    # Free text (commit.py's -m): a value word that argparse reads as a value
    # though it starts with `-` (`_argparse_text_value`) is one here too.
    text: bool = False


def _argparse_text_value(word: str, options: dict[str, _Option]) -> bool:
    """Whether argparse takes a `-`-led word as the VALUE of the option before
    it rather than as an option: it holds a space (`- bullet`, `-x fix`,
    `--fi x`) and its first two characters are no short option of the parser
    (`-h` included) — argparse matches `-F x` as `-F` glued to ` x` first.
    Measured on Python 3.13 with commit.py's parser (review 2026-11-03: `-m
    '- bullet'` was declined as a glued option). A tab is not a space there."""
    shorts = {flag for flag in options if len(flag) == 2 and flag[1] != "-"}
    return " " in word and word != "--" and word[:2] not in shorts | {"-h"}


def _parse_options(args: list[str], options: dict[str, _Option],
                   label: str, unknown_hint: str, cwd: Path, ctx: CommandContext,
                   *, remainder: bool = False, reject_option_value: bool = False
                   ) -> tuple[dict[str, str], list[str], str]:
    """Read full option names, checking values in order before later syntax.
    Return the last value of each option, the remainder, and any refusal.
    try_patch owns a REMAINDER and accepts option words as required values;
    build has neither rule. Optional values stop before an option word. A
    short option glued to its value (`-Fx`, `-F=x`, which argparse reads as
    `=x`) is refused where an option is read, not where a value is."""
    parsed: dict[str, str] = {}
    at = 0
    while at < len(args):
        arg = args[at]
        if remainder and arg == "--":
            at += 1
            break
        if remainder and not arg.startswith("-"):
            break
        if len(arg) > 2 and arg[1] != "-" and arg[:2] in options:
            return {}, [], (f"{label} {arg}: a short option glued to its value "
                            f"(spell it `{arg[:2]} <value>`)")
        flag, sep, value = arg.partition("=")
        option = options.get(flag)
        if option is None:
            return {}, [], (f"{label} option {arg} ({unknown_hint}: "
                            f"{', '.join(sorted(options))})")
        if option.takes_value and not sep:
            absent = at + 1 >= len(args)
            option_word = not absent and args[at + 1].startswith("-") and not (
                option.text and _argparse_text_value(args[at + 1], options))
            if option.optional_value and (absent or option_word):
                at += 1
                continue
            if absent or (reject_option_value and option_word):
                return {}, [], f"{label} {flag} with no value"
            at += 1
            value = args[at]
        elif sep and not option.takes_value:
            return {}, [], f"{label} {flag} takes no value"
        if option.check:
            why = option.check(value, cwd, ctx)
            if why:
                return {}, [], why
        parsed[flag] = value
        at += 1
    return parsed, args[at:], ""


# Recursive policies receive this dependency from the dispatcher.
_ArgvJudge = Callable[[list[str], Path, CommandContext], str]
