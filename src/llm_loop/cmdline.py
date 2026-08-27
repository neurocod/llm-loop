"""cmdline - answer "what command line would reproduce this run?".

The interactive status line lets a run's settings be edited while it is going
(iteration cap, git-push policy, quota ceilings). Key `c` then has to show a line
the user can paste to relaunch with exactly those settings. That is this module:
it takes the run's ORIGINAL argv and a dict of overrides and gives back an argv
with every existing spelling of each overridden flag removed and the new value
appended.

Starting from the original argv rather than from parsed values is what keeps the
answer honest: flags this engine never parses - the host wrapper's -p/--parallel,
--grow-kit, --random, --finish FOLDER - survive untouched, without this
module knowing what they mean.

The module is deliberately PURE: no terminal, no I/O, and no import of cyclecore,
limits or parallel. `statusline` imports this one (SettingsRegistry validates
every Setting.flag against FLAG_ALIASES) while cyclecore/limits/parallel import
statusline - so a single import from here in the other direction would close the
cycle. `clispec` is below all of them and is safe to import; see there.
"""

import os
import shlex
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

# The flag table this module strips an argv with, and the record type it is made
# of, are DERIVED from the family's one option table rather than kept here: the
# hand-written copy that used to live at this spot carried a comment naming the
# two parsers it had been transcribed from, which is the whole reason the three
# could drift. Re-exported under these names because that is where every caller
# (statusline's SettingsRegistry, the tests, `rebuild_argv`'s own default) still
# reaches for them.
from .clispec import FLAG_ALIASES, Flag

__all__ = ["FLAG_ALIASES", "Flag", "rebuild_argv", "render"]


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
    have its value misread. A flag a parser offers cannot go missing any more
    (the table is derived from the same declaration the parsers are built from);
    a wrapper-only one still has to be declared in `clispec.OPTIONS` by hand.
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
