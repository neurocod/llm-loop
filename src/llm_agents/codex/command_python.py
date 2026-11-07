"""Python launcher, pytest and the judging of listed Python entrypoints.

The entrypoint TABLE is the host's (`WorkerCommands.shapes` binds it into
the python shape with `partial(_python, entries=..., declined=..., table=...)`);
this module reads `python <script>` against whatever table it is handed.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Mapping, Optional
from .command_arguments import (
    CommandContext,
    _ArgvJudge,
    _Option,
    _arg_path_refusal,
    _args_refusal,
    _existing_file,
    _parse_options,
    _value_refusal,
)
from .command_paths import (
    resolve_path,
)

_PYTHON_SWITCHES = frozenset({"-B", "-u", "-I", "-E", "-s"})
_PYTHON_X = frozenset({"utf8", "utf8=1"})


def _python(argv: list[str], cwd: Path, ctx: CommandContext, *,
            entries: Mapping[str, Entrypoint],
            declined: Mapping[str, str] = {},
            table: str = "the host policy's python entrypoints") -> str:
    words = argv[1:]
    if argv[0] == "py" and words and re.fullmatch(r"-3(\.\d+)?(-(32|64))?",
                                                  words[0]):
        # The Windows launcher's version pick: `py -3 x.py` is `python x.py`.
        words = words[1:]
    at = 0
    while at < len(words) and words[at].startswith("-") and words[at] != "-m":
        word = words[at]
        if word == "-X" and at + 1 < len(words) and words[at + 1] in _PYTHON_X:
            at += 2
            continue
        if word.startswith("-X") and word[2:] in _PYTHON_X:
            at += 1
            continue
        if word not in _PYTHON_SWITCHES:
            return f"python option {word} (-c runs code the judge cannot read)"
        at += 1
    if at >= len(words):
        return "python with no script"
    if words[at] == "-m":
        module = words[at + 1] if at + 1 < len(words) else ""
        if module != "pytest":
            return f"python -m {module} (only pytest)"
        return _pytest_refusal(words[at + 2:], cwd, ctx)
    script, why = _existing_file(words[at], cwd, ctx, "script")
    if why:
        return why
    try:
        relative = "/".join(script.relative_to(ctx.workspace).parts)
    except ValueError:
        return f"script {words[at]} is not in your working tree"
    entry, why = _python_entrypoint(relative, entries, declined, table)
    if entry is None:
        return why
    return entry.judge(words[at + 1:], cwd, ctx)


# The ini keys pytest's `-o`/`--override-ini` may set, and whether the value
# is a path (judged as one). None of them takes arguments or names a module.
# Any other key is declined, `addopts` above all: pytest splices its value
# into the command line, so `-o addopts=--basetemp=../<sibling slot>`
# deleted a sibling slot past every rule here (third review, 2026-11-03).
_PYTEST_INI = {"cache_dir": True, "console_output_style": False,
               "log_cli": False, "log_cli_level": False, "log_level": False,
               "faulthandler_timeout": False, "xfail_strict": False}
# pytest's short options that take no value: they may lead a glued cluster,
# argparse's way (`-qo addopts=x` is `-q -o addopts=x`). pytest itself
# refuses abbreviations of its long options (`allow_abbrev=False`).
_PYTEST_SHORT_BARE = frozenset("qvxslV")


def _pytest_option(args: list[str], at: int) -> tuple[str, str, int]:
    """(option, value, words used) of `args[at]` when it is `-p`, `-o`,
    `--override-ini` or `--basetemp` in any spelling — `-pX`, `-p X`,
    `-qpX`, `--override-ini=X`, `--override-ini X` — else ("", "", 1)."""
    arg = args[at]
    following = args[at + 1] if at + 1 < len(args) else ""
    if arg.startswith("--"):
        name, sep, value = arg.partition("=")
        if name not in ("--override-ini", "--basetemp"):
            return "", "", 1
        return (name, value, 1) if sep else (name, following, 2)
    if not arg.startswith("-"):
        return "", "", 1
    for index, char in enumerate(arg[1:], start=1):
        if char in "po":
            glued = arg[index + 1:]
            return (f"-{char}", glued, 1) if glued else (f"-{char}", following, 2)
        if char not in _PYTEST_SHORT_BARE:
            break
    return "", "", 1


def _pytest_refusal(args: list[str], cwd: Path, ctx: CommandContext) -> str:
    """pytest DELETES an existing `--basetemp`: allowed only on a directory
    that does not exist yet, one inside the scratch root, or one named for
    pytest (`.pytest-tmp`), never on a root itself. `-p <module>` imports
    any module as a plugin (`-p tools.git.commit`); only `-p no:<name>`,
    which turns one off, is read. `-o` sets only a `_PYTEST_INI` key, and
    `@file` (pytest reads arguments from it) is refused."""
    at = 0
    while at < len(args):
        if args[at].startswith("@"):
            return (f"pytest {args[at]}: reads arguments from a file the judge "
                    f"does not read")
        flag, value, used = _pytest_option(args, at)
        at += used
        if flag == "-p":
            if not value.startswith("no:"):
                return f"pytest -p {value} loads a module the judge cannot read"
            continue
        if flag in ("-o", "--override-ini"):
            key, eq, setting = value.partition("=")
            if not eq or key not in _PYTEST_INI:
                return (f"pytest {flag} {value}: only the ini keys "
                        f"{', '.join(sorted(_PYTEST_INI))} (addopts and the "
                        f"rest carry arguments the judge does not read)")
            if _PYTEST_INI[key]:
                why = _arg_path_refusal(setting, cwd, ctx) if setting else \
                    f"pytest {flag} {key}= with no value"
                if why:
                    return why
            continue
        if flag != "--basetemp":
            continue
        why = _arg_path_refusal(value, cwd, ctx) if value else \
            "--basetemp with no value"
        if why:
            return why
        target = resolve_path(value, cwd)[1]
        if target in ctx.roots or any(target in root.parents
                                      for root in ctx.roots):
            return f"--basetemp {value} is a writable root, which pytest deletes"
        named = "pytest" in target.name.casefold()
        in_scratch = ctx.scratch is not None and ctx.scratch in target.parents
        if target.exists() and not (named or in_scratch):
            return (f"--basetemp {value} exists and pytest deletes it; use a "
                    f"directory named for pytest or under your scratch root")
    return _args_refusal(args, cwd, ctx)


# try_patch.py's own options (`main` of claude-plugin/ask-user-gate/bin/try_patch.py
# in this repository):
# whether each takes a value, and whether that value is a path. Spelled in
# full: argparse also takes `--cw` for `--cwd`, which a reader matching the
# full name would run in the wrong directory. Not here: `--selftest` and
# `--recover`, which no worker step needs.
_TRY_PATCH_OPTIONS = {
    "--file": _Option(takes_value=True, check=partial(_value_refusal, is_path=True)),
    "--cwd": _Option(takes_value=True, check=partial(_value_refusal, is_path=True)),
    "--old": _Option(takes_value=True),
    "--new": _Option(takes_value=True, optional_value=True),
    "--remove": _Option(takes_value=True), "--count": _Option(takes_value=True),
    "--reject-exit": _Option(takes_value=True),
    "--regex": _Option(), "--expect-fail": _Option(), "--keep": _Option()}


def _try_patch_refusal(args: list[str], cwd: Path, ctx: CommandContext,
                       *, judge_argv: _ArgvJudge) -> str:
    """try_patch.py edits its `--file`s and runs a command outside the
    sandbox: each file must pass `path_refusal` (a bare name too), and the
    command must itself be a worker command, judged in try_patch's `--cwd`
    (`--cwd X` and `--cwd=X` alike — review 2026-11-03: the `=` form was
    checked but the command judged at the old cwd). The command is what
    follows `--`, or — argparse's REMAINDER — the first word that is no
    option's value."""
    parsed, tail, why = _parse_options(
        args, _TRY_PATCH_OPTIONS, "try_patch", "its options spelled in full",
        cwd, ctx, remainder=True)
    if why:
        return why
    run_in = resolve_path(parsed["--cwd"], cwd)[1] if "--cwd" in parsed else cwd
    return judge_argv(tail, run_in, ctx) if tail else ""


# --- the python entrypoints ------------------------------------------------


@dataclass(frozen=True)
class Entrypoint:
    """One script `python <path>` may start outside the sandbox: what it
    does (the reason it is granted) and the judge of its arguments."""
    may: str
    judge: _ArgvJudge


def _edits(args: list[str], cwd: Path, ctx: CommandContext) -> str:
    """A script that writes the files it names: every positional is one,
    a bare name included (`outside` alone may be a junction)."""
    return _args_refusal(args, cwd, ctx, bare_paths=True)


def _checks(*read_only: str) -> _ArgvJudge:
    """The judge of a generator that rewrites its checked-in output unless
    asked only to compare: exactly `--check`, plus the `read_only` flags of
    its own argparse, each spelled in full. Any other word declines —
    argparse takes `--w`/`--wri` for `--write`, and `build_registry.py
    --measure` writes too (third review 2026-11-03 granted both)."""
    allowed = frozenset({"--check", *read_only})

    def judge(args: list[str], cwd: Path, ctx: CommandContext) -> str:
        if "--check" not in args:
            return "this generator writes its output; run it with --check"
        other = next((arg for arg in args if arg not in allowed), None)
        if other is not None:
            return (f"{other}: a generator is granted only "
                    f"{' '.join(sorted(allowed))}")
        return ""

    return judge


def _declined_reason(relative: str, declined: Mapping[str, str]) -> str:
    """The reason `declined` gives for a script, by path prefix (a
    directory ending in `/`, or a `*` name pattern), or ""."""
    folded = relative.casefold()
    for pattern, reason in declined.items():
        pattern = pattern.casefold()
        if (fnmatch.fnmatchcase(folded, pattern) if "*" in pattern
                else folded == pattern or (pattern.endswith("/")
                                           and folded.startswith(pattern))):
            return reason
    return ""


def _python_entrypoint(relative: str, entries: Mapping[str, Entrypoint],
                       declined: Mapping[str, str], table: str
                       ) -> tuple[Optional[Entrypoint], str]:
    """The `entries` entry of a script (its path relative to the worker's
    tree, `/`-separated; case-folded, as Windows opens it), or None and why
    not: `declined`'s reason, else that it is not on `table`. A listed entry
    wins over its declined family."""
    folded = relative.casefold()
    for path, entry in entries.items():
        if path.casefold() == folded:
            return entry, ""
    reason = _declined_reason(relative, declined)
    if reason:
        return None, f"{relative}: {reason}"
    return None, (f"{relative} is not on {table} "
                  f"(run it inside the sandbox, where python works)")
