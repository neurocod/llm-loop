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
import re
import shlex
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

# The flag table this module strips an argv with, and the record type it is made
# of, are DERIVED from the family's one option table rather than kept here: the
# hand-written copy that used to live at this spot carried a comment naming the
# two parsers it had been transcribed from, which is the whole reason the three
# could drift. Re-exported under these names because that is where every caller
# (statusline's SettingsRegistry, the tests, `rebuild_argv`'s own default) still
# reaches for them.
from .clispec import FLAG_ALIASES, Flag

__all__ = ["FLAG_ALIASES", "Flag", "NotPasteable", "POSIX", "POWERSHELL",
           "paste_shell", "quote", "rebuild_argv", "render"]

# The shells a line can be rendered for; see `render` for which one is chosen.
POWERSHELL = "powershell"
POSIX = "posix"


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
            # Glued short form (`-m5`, `-Cd:\proj`): argparse accepts it, so it
            # must be removable here too.
            if (len(alias) == 2 and not alias.startswith("--")
                    and len(arg) > 2 and arg.startswith(alias)):
                return canonical, spec, False
    return None, None, False


def _empty_value(spelling: str, canonical: str) -> str:
    """The one token that gives the value-taking flag `spelling` an EMPTY value.

    `--flag=` rather than the pair `--flag ""`: argparse (and the host wrapper's
    own `--finish=` reader) takes both as the empty value, but only the single
    token survives the shell the line is pasted into. Windows PowerShell 5.1
    drops an empty argument to a native program, so a pasted `--project-dir ""`
    arrives as a bare `--project-dir`: argparse refuses it ("expected one
    argument") whether a flag follows or the line ends there, so the relaunch
    dies as a usage error. `list2cmdline` cannot help: `""` is the correct
    CreateProcess spelling, and the shell in between is what loses it.

    A short spelling is replaced by the canonical long one. argparse would take
    `-C=` as the empty value too (it splits a known short option at `=`), but
    the long `--flag=` is the one spelling the line needs, and the one a reader
    recognises as "empty value" rather than as a typo.
    """
    return (spelling if spelling.startswith("--") else canonical) + "="


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

    An EMPTY value - an override of `""` or a copied `-C ""` - comes out as the
    single token `--flag=` (see `_empty_value`), so this is the one place a
    copied flag is respelled. The `--` tail is never respelled: it is not this
    table's to read, so an empty token there is copied as is, and `quote`
    refuses it for PowerShell (NotPasteable) rather than print a line that
    loses it.
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
        if span == 2 and head[i + 1] == "":
            out.append(_empty_value(head[i], canonical))
        else:
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
            text = str(value)
            if text == "":
                out.append(_empty_value(canonical, canonical))
            else:
                out.append(canonical)
                out.append(text)
        else:
            if value is not True:
                raise ValueError(
                    f"{canonical} is a boolean flag; use True/None, "
                    f"got {value!r}")
            out.append(canonical)
    return out + tail


class NotPasteable(ValueError):
    """An argument no line for the target shell delivers to the program intact.

    Raised rather than printing a line that looks reproducible and is not; the
    message names the argument and the reason.
    """


def _shell_for(os_name: str, environ) -> str:
    """`paste_shell`'s decision, on the values it reads (a table to pin)."""
    if os_name != "nt":
        return POSIX
    # Git Bash and the MSYS2 shells set MSYSTEM (MINGW64, UCRT64, MSYS, ...)
    # and hand it to the native Windows Python they start (measured
    # 2026-09-28: `python` from Git Bash sees MSYSTEM=MINGW64, from
    # PowerShell sees none). SHELL is not used: Git Bash exports it too, but
    # it is also set machine-wide by tools and editor setups that never make
    # bash the console, so it would flip lines typed in PowerShell to POSIX.
    # The miss that remains: a PowerShell started FROM Git Bash inherits
    # MSYSTEM and gets the POSIX line.
    if environ.get("MSYSTEM"):
        return POSIX
    return POWERSHELL


def paste_shell() -> str:
    """The shell a line printed by this process is most likely pasted into.

    PowerShell on Windows (the console the author and the project's docs use;
    cmd.exe is not supported - see `render`), unless the process was started
    from Git Bash / MSYS2 (MSYSTEM set), and a POSIX sh everywhere else. A
    POSIX line pasted into Git Bash still passes through MSYS's path
    conversion on its way to a native program (`/foo` becomes
    `C:/Program Files/Git/foo`); the run's own argv, when it came from Git
    Bash, already holds the converted Windows forms, which MSYS leaves alone.
    """
    return _shell_for(os.name, os.environ)


# A word PowerShell delivers to a native program exactly as written, measured
# against Windows PowerShell 5.1 on 2026-09-28 (bare `1e5`, `0x10`, `1kb`,
# `--x=a.b`, `--`, `--flag=` all arrive verbatim; argument mode does not turn
# number-like words into numbers). Everything else is single-quoted: `$`,
# backtick, `;`, `&`, `|`, `@` (splatting: a bare `@a` delivers nothing), `(`,
# `{`, `,`, `#`, `%`, quotes and whitespace.
_PS_BARE = re.compile(r"[A-Za-z0-9_\-./\\:=]+")
# PowerShell reads these four as the single quote too, so inside '...' each
# must be doubled like `'` itself or it ends the string.
_PS_SINGLE_QUOTES = "'\u2018\u2019\u201a\u201b"


def _powershell_word(arg: str) -> str:
    """`arg` as one PowerShell word that reaches the program unchanged.

    Four arguments have no such word, and are refused rather than guessed.
    Three because PowerShell 5.1 builds the program's command line by pasting
    each value between `"` when it holds whitespace and escaping nothing, while
    pwsh 7.3+ escapes properly - so a pre-escape that repaired one would break
    the other:
      - `""`: 5.1 drops an empty argument (a flag's empty value never gets
        here: `rebuild_argv` spells it `--flag=`).
      - a `"`: 5.1 hands it over raw and the C runtime reads it as quoting
        (`a"b` arrives as `ab`); Windows paths cannot contain one.
      - whitespace with a trailing backslash: 5.1 prints `"C:\\my dir\\"`,
        whose `\\"` escapes the closing quote (`C:\\my dir"` arrives).
    And `--%`, which 5.1 drops even quoted (`'--%'`, `"--%"`, `('--%')` all
    deliver nothing, measured 2026-09-28): it compares the VALUE with its
    stop-parsing token. Whitespace alone needs nothing more: PowerShell adds
    the `"` itself.
    """
    if arg == "":
        raise NotPasteable(
            "PowerShell 5.1 drops an empty argument to a native program")
    if arg == "--%":
        raise NotPasteable(
            "PowerShell 5.1 drops a '--%' argument to a native program, "
            "quoted or not")
    if '"' in arg:
        raise NotPasteable(
            f"PowerShell 5.1 passes the double quote in {arg!r} unescaped")
    if arg.endswith("\\") and any(ch.isspace() for ch in arg):
        raise NotPasteable(
            f"PowerShell 5.1 turns the trailing backslash of {arg!r} into an "
            f"escaped quote")
    # `-foo.bar` and `-a=b.c` arrive split at the dot (5.1, measured), and so
    # do `-C:foo`, `-a=b:c` and `-CD:/proj` at the colon - but only once a bare
    # `--` precedes them on the line (`-C:` `foo`; before it, and first after
    # the program, they arrive whole; measured 2026-09-28). So a dash word
    # with a dot or a colon is quoted even though every character is plain,
    # wherever it stands - `--x=a.b` and `--x:y` too, measured intact: one
    # rule is easier to trust than a rule per position.
    if _PS_BARE.fullmatch(arg) and not (
            arg.startswith("-") and ("." in arg or ":" in arg)):
        return arg
    return "'" + "".join(ch * 2 if ch in _PS_SINGLE_QUOTES else ch
                         for ch in arg) + "'"


def _refuse_unprintable(arg: str) -> None:
    """NotPasteable unless every character of `arg` prints as itself.

    For either shell, because the line is PRINTED before it is pasted. A NUL
    ends the program's command line there (5.1 delivered `["a\\0b",
    "SENTINEL"]` as `["a"]`, exit 0; POSIX argv cannot hold one at all). A
    line feed, CR or U+2028 breaks the one printed line into several, a tab
    is a completion key when pasted into PSReadLine, and ESC turns the status
    line that prints the command into a terminal escape sequence (`\\x1b[2J`
    clears the screen). Refused by class rather than by list:
    `str.isprintable` is false for control (Cc), format (Cf: zero-width and
    bidi overrides, which make the line read differently from what it runs),
    line/paragraph separator, private-use, surrogate and unassigned
    characters. The one class let through is Zs, the space separators
    (U+00A0, U+3000, ...): they print as a space, both shells quote them
    (`shlex` quotes any non-ASCII-word character; `_PS_BARE` is ASCII), and
    5.1 delivered them intact (measured 2026-09-28).
    """
    for ch in arg:
        if ch == "\0":
            raise NotPasteable(
                f"{arg!r} holds a NUL, which ends a program's command line")
        if not ch.isprintable() and unicodedata.category(ch) != "Zs":
            raise NotPasteable(
                f"{arg!r} holds {ch!r}, which does not print as itself on a "
                f"one-line command")


def _posix_word(arg: str) -> str:
    """`shlex.quote`, plus quoting for a leading `=`.

    zsh (EQUALS, on by default) expands a word starting with `=` into the path
    of the command it names (`=python` -> `/usr/bin/python`), and `=` is in
    `shlex`'s safe set, so such a word gets single quotes of its own.
    """
    if arg.startswith("="):
        return "'" + arg.replace("'", "'\"'\"'") + "'"
    return shlex.quote(arg)


def quote(parts: List[str], shell: Optional[str] = None) -> str:
    """Join a program and its arguments into one line for `shell`.

    `shell` is POWERSHELL or POSIX, `paste_shell()` when omitted. The
    PowerShell line starts with the call operator `& `: without it a quoted
    first word (an interpreter under "Program Files") is a string expression
    and the paste fails with "Unexpected token". It raises NotPasteable for an
    argument no PowerShell line delivers intact (see `_powershell_word`), and
    for either shell for one that does not print as one line
    (`_refuse_unprintable`). The POSIX line is `shlex.join`'s quoting, which
    quotes every argument, however written, for sh, bash and zsh alike, plus
    `_posix_word`'s rule for zsh.
    """
    shell = paste_shell() if shell is None else shell
    if shell not in (POWERSHELL, POSIX):
        raise ValueError(
            f"unknown shell {shell!r}; known: {POWERSHELL}, {POSIX}")
    for part in parts:
        _refuse_unprintable(part)
    if shell == POWERSHELL:
        return " ".join(["&", *map(_powershell_word, parts)])
    return " ".join(map(_posix_word, parts))


def render(argv: List[str], overrides: Dict[str, Any], *,
           executable: str = sys.executable, script: Optional[str] = None,
           aliases: Dict[str, Flag] = FLAG_ALIASES,
           shell: Optional[str] = None) -> str:
    """The full copy-pasteable command line reproducing this run.

    `script` defaults to `sys.argv[0]` - the wrapper actually launched
    (runGenerateModels.py), not this module - so the line can be pasted as-is.

    The line is written for ONE shell, the one it will be pasted into: a line
    is only reproducible in the shell that parses it, and the two candidates
    disagree on nearly every metacharacter. `shell` defaults to `paste_shell()`
    - PowerShell on Windows, a POSIX sh elsewhere - because a status line is
    read in the console it runs in. The PowerShell line is written and measured
    for Windows PowerShell 5.1 (`powershell.exe`, present on every Windows); it
    carries no pre-escape for 5.1's native-argument passing, so pwsh 7.3+,
    which passes arguments properly, should read it the same - unmeasured, as
    the author's machine has no pwsh. cmd.exe is not a target: it would reject
    the leading `& ` and read `%` and `^` in values; the older CreateProcess-only
    line (`list2cmdline`) worked there and in no shell the author uses. Raises
    NotPasteable when an argument cannot be written for the shell (see
    `quote`); the caller shows that message instead of a line.
    """
    if script is None:
        script = sys.argv[0] if sys.argv else ""
    parts = [executable]
    if script:
        parts.append(script)
    parts.extend(rebuild_argv(argv, overrides, aliases=aliases))
    return quote(parts, shell)
