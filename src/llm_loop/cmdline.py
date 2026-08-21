"""cmdline - answer "what command line would reproduce this run?".

The interactive status line lets a run's settings be edited while it is going
(iteration cap, git-push policy, quota ceilings). Key `c` then has to show a line
the user can paste to relaunch with exactly those settings. That is this module:
it takes the run's ORIGINAL argv and a dict of overrides and gives back an argv
with every existing spelling of each overridden flag removed and the new value
appended.

Starting from the original argv rather than from parsed values is what keeps the
answer honest: flags this engine never parses - the host wrapper's -p/--parallel,
--grow-kit, --random, --grow-kit-periodically N - survive untouched, without this
module knowing what they mean.

The module is deliberately PURE: no terminal, no I/O, and no import of cyclecore,
limits or parallel. `statusline` imports this one (SettingsRegistry validates
every Setting.flag against FLAG_ALIASES) while cyclecore/limits/parallel import
statusline - so a single import from here in the other direction would close the
cycle.
"""

import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

__all__ = ["FLAG_ALIASES", "Flag", "rebuild_argv", "render"]


class Flag(NamedTuple):
    """One canonical flag: every spelling argparse accepts, and its arity."""

    aliases: Tuple[str, ...]
    takes_value: bool


# The flag table. Keys are the canonical long spellings the `overrides` dict
# speaks; `aliases` lists every spelling that must be REMOVED when that flag is
# overridden, canonical form included.
#
# Sourced from cyclecore.parse_args, parallel.parse_args and
# runGenerateModels.split_mode - including the deprecated aliases those parsers
# still accept (--max, --startIn). A spelling missing here is not a
# cosmetic bug: the old value would survive next to the new one and argparse
# would take the LAST occurrence, which happens to be right for a value flag and
# wrong for anything else - so keep this table complete instead of relying on it.
#
# Order matters and is part of the contract: overrides are appended in the order
# below, so the rendered line is deterministic regardless of dict ordering.
#
# --session-limit / --weekly-limit / --no-statusline have no parser yet (a later
# wave adds them); they are listed now so the status line can already express an
# edited ceiling as a command line.
FLAG_ALIASES: Dict[str, Flag] = {
    # value-taking options
    "--max-runs": Flag(("-m", "--max-runs", "--max"), True),
    "--start-in": Flag(("-s", "--start-in", "--startIn"), True),
    "--git-push": Flag(("-g", "--git-push"), True),
    "--project-dir": Flag(("-C", "--project-dir"), True),
    "--jobs": Flag(("-j", "--jobs"), True),
    "--session-limit": Flag(("--session-limit",), True),
    "--weekly-limit": Flag(("--weekly-limit",), True),
    "--grow-kit-periodically": Flag(("--grow-kit-periodically",), True),
    # store_true / store_const options
    "--codex": Flag(("--codex",), False),
    "--dry-run": Flag(("-d", "--dry-run"), False),
    "--raw": Flag(("--raw",), False),
    "--cost": Flag(("-c", "--cost"), False),
    "--ignore-usage": Flag(("--ignore-usage",), False),
    "--no-statusline": Flag(("--no-statusline",), False),
    "--parallel": Flag(("-p", "--parallel"), False),
    "--grow-kit": Flag(("--grow-kit",), False),
    "--random": Flag(("--random",), False),
}


def _split_passthrough(argv: List[str]) -> Tuple[List[str], List[str]]:
    """Split at the first bare `--`. The tail is never scanned or rewritten.

    Everything after `--` belongs to whatever the wrapper forwards; a token
    there that happens to spell `-m` is data, not a flag.
    """
    for i, arg in enumerate(argv):
        if arg == "--":
            return list(argv[:i]), list(argv[i:])
    return list(argv), []


def _lookup(arg: str, aliases: Dict[str, Flag]):
    """(canonical, spec, consumes_next) for the flag `arg` spells, else Nones.

    Three passes so that a longer spelling can never be shadowed by a shorter
    one that happens to be declared first (`--max-runs=5` vs the `--max` alias).
    """
    for canonical, spec in aliases.items():
        if arg in spec.aliases:
            return canonical, spec, spec.takes_value
    for canonical, spec in aliases.items():
        if not spec.takes_value:
            continue
        for alias in spec.aliases:
            if alias.startswith("--") and arg.startswith(alias + "="):
                return canonical, spec, False
    for canonical, spec in aliases.items():
        if not spec.takes_value:
            continue
        for alias in spec.aliases:
            # Glued short form (`-m5`, `-Cd:\proj`): argparse accepts it, and so
            # does the wrapper's own -C scan, so it must be removable here too.
            if (len(alias) == 2 and not alias.startswith("--")
                    and len(arg) > 2 and arg.startswith(alias)):
                return canonical, spec, False
    return None, None, False


def _validate(overrides: Dict[str, Any], aliases: Dict[str, Flag]) -> None:
    unknown = [key for key in overrides if key not in aliases]
    if unknown:
        raise KeyError(
            f"unknown canonical flag(s) {sorted(unknown)}; known: "
            f"{sorted(aliases)}")


def rebuild_argv(argv: List[str], overrides: Dict[str, Any], *,
                 aliases: Dict[str, Flag] = FLAG_ALIASES) -> List[str]:
    """The run's argv with `overrides` applied, ready to be quoted.

    `argv` is the argument list WITHOUT the program name (what `parse_args`
    gets, i.e. `sys.argv[1:]`). `overrides` maps a canonical long flag to its
    new value; `None` (or `False` for a boolean flag) means "just remove it",
    `True` sets a boolean flag. Non-string values are stringified, so an int
    iteration cap can be passed as an int.

    Removal walks the whole argv rather than the overridden flags alone: a
    value token that looks like a flag (`--project-dir --max-runs`, a directory
    literally so named) must not be mistaken for one. Flags this table does not
    know are copied through verbatim - which is exactly how the wrapper-only
    switches survive - but a value-taking flag missing from the table can still
    have its value misread, so add new flags here when the parsers grow them.
    """
    _validate(overrides, aliases)
    head, tail = _split_passthrough(argv)
    dropped = set(overrides)

    out: List[str] = []
    i = 0
    while i < len(head):
        canonical, _spec, consumes_next = _lookup(head[i], aliases)
        span = 1
        if canonical is not None and consumes_next and i + 1 < len(head):
            span = 2
        if canonical in dropped:
            i += span
            continue
        out.extend(head[i:i + span])
        i += span

    # Appended in table order, never in dict order: the line must be identical
    # for identical settings (it is shown to a human and compared by eye).
    for canonical, spec in aliases.items():
        if canonical not in overrides:
            continue
        value = overrides[canonical]
        if value is None or value is False:
            continue        # removal only
        if spec.takes_value:
            if value is True:
                raise ValueError(f"{canonical} needs a value, got True")
            out.append(canonical)
            out.append(str(value))
        else:
            if value is not True:
                raise ValueError(
                    f"{canonical} is a boolean flag; use True/None, "
                    f"got {value!r}")
            out.append(canonical)
    return out + tail


def quote(parts: List[str]) -> str:
    """Join argv into one pasteable line, quoted for the local shell."""
    if os.name == "nt":
        # cmd.exe/PowerShell parse the way CreateProcess does; list2cmdline is
        # the inverse of that parse, shlex is the inverse of the POSIX one.
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


def render(argv: List[str], overrides: Dict[str, Any], *,
           executable: str = sys.executable, script: Optional[str] = None,
           aliases: Dict[str, Flag] = FLAG_ALIASES) -> str:
    """The full copy-pasteable command line reproducing this run.

    `script` defaults to `sys.argv[0]` - the wrapper actually launched
    (runGenerateModels.py), not this module - so the line can be pasted as-is.
    """
    if script is None:
        script = sys.argv[0] if sys.argv else ""
    parts = [executable]
    if script:
        parts.append(script)
    parts.extend(rebuild_argv(argv, overrides, aliases=aliases))
    return quote(parts)
