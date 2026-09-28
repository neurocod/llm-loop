"""modeswitch - a wrapper's mode switches, declared once.

A MODE switch decides which parser runs (`-p` picks the parallel runner, whose
option set differs from the sequential one), so a wrapper has to read it before
either parser exists. It used to be spelled in up to six places per switch: the
flag constant, a field of the wrapper's scan result, a branch of the scan, an
`add_argument` plus a row of a "which --help lists it" table, a row of the
check that compares the scan with the parser, and an unpacking at the dispatch
- with names that drifted between them. And a wrapper that STRIPPED the switch
out of argv before parsing handed the parser a different line than the user
typed: `-m -p 3` became `-m 3`, the word silently taking the flag's place.

Here a wrapper declares a table of `ModeSwitch` rows and three functions work
off it:

  * `scan(argv, switches)` reads the switches off argv, leaving it intact, to
    CHOOSE the parser. It only chooses;
  * `register(parser, switches, mode)` puts every switch into the chosen parser
    (`Driver.add_cli_options` is the seam), so that parser reads the WHOLE line,
    switches included, with one grammar: a value flag followed by a switch is
    argparse's "expected one argument", never a quiet substitution;
  * `refuse_disagreement(args, scanned, prog)` exits 2 when the parser read a
    switch differently from the scan — an abbreviation argparse resolves and
    the scan does not (`--rand`), which would otherwise run the default mode
    behind a flag that looks as if it worked.

A switch's `dest` names it everywhere: the scan result's attribute and the
parsed namespace's are the same name.
"""

import argparse
import re
import sys
from typing import (Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional,
                    Sequence, Tuple)

__all__ = [
    "ModeSwitch",
    "Scan",
    "refuse_disagreement",
    "register",
    "scan",
]


class ModeSwitch(NamedTuple):
    """One mode switch of a wrapper.

    `aliases` are every spelling, a long one last (it names the switch in
    messages). `takes_value` switches store their value (default None, `type`
    and `metavar` as argparse's); the others are store_true. `listed_in` names
    the help modes whose --help lists the switch — every mode PARSES every
    switch, listed or not, because argv still carries it. `implies` names the
    dest of a boolean switch this one turns on (a worker count asks for
    workers); such a switch is compared by presence only, its value being the
    parser's to read.

    A switch whose every spelling the chosen parser already offers is that
    parser's own option (the parallel runner's `-j/--jobs`): `register` leaves
    it alone, and its `dest` must be that option's.
    """

    aliases: Tuple[str, ...]
    dest: str
    takes_value: bool
    help: str
    listed_in: FrozenSet[str] = frozenset()
    implies: Optional[str] = None
    metavar: Optional[str] = None
    type: Optional[Callable[[str], Any]] = None

    @property
    def name(self) -> str:
        """The spelling messages use: the last alias, by convention the long one."""
        return self.aliases[-1]


class Scan:
    """What `scan` read: one attribute per switch `dest`.

    A boolean switch reads True when given or implied; a value switch reads its
    first value, or None. `seen` holds the dests given on the line (a value
    switch given without a value is seen and reads None), `repeated` the dests
    of value switches given more than once.
    """

    def __init__(self, switches: Sequence[ModeSwitch], values: Dict[str, Any],
                 seen: FrozenSet[str], repeated: Tuple[str, ...]):
        self.switches = tuple(switches)
        self.values = dict(values)
        self.seen = seen
        self.repeated = repeated

    def __getattr__(self, dest: str) -> Any:
        values = self.__dict__.get("values", {})
        if dest in values:
            return values[dest]
        raise AttributeError(f"no mode switch has dest {dest!r}")

    def __repr__(self) -> str:
        return f"Scan({self.values!r})"


def _check_table(switches: Sequence[ModeSwitch]) -> None:
    dests = [s.dest for s in switches]
    if len(set(dests)) != len(dests):
        raise ValueError(f"mode switch dests repeat: {dests}")
    reserved = {"switches", "values", "seen", "repeated"} & set(dests)
    if reserved:
        raise ValueError(f"mode switch dest shadows a Scan attribute: {reserved}")
    by_dest = {s.dest: s for s in switches}
    for s in switches:
        if s.implies is not None:
            target = by_dest.get(s.implies)
            if target is None or target.takes_value:
                raise ValueError(f"{s.name} implies {s.implies!r}, which is not "
                                 f"a boolean switch of the same table")


# argparse's own test for "this dash-led token is a number, not an option"
# (`ArgumentParser._negative_number_matcher`), valid while no option of the
# parser itself looks like a negative number — true of every parser here.
_NEGATIVE_NUMBER = re.compile(r"^-\d+$|^-\d*\.\d+$")


def _value_after(argv: List[str], i: int) -> Tuple[Optional[str], int]:
    """The value token after argv[i], as argparse would take it, and the index
    after it. A token that looks like an option is not a value (argparse then
    refuses the line), and is left to be read as a token of its own."""
    if i + 1 < len(argv):
        token = argv[i + 1]
        if not token.startswith("-") or _NEGATIVE_NUMBER.match(token):
            return token, i + 2
    return None, i + 1


def scan(argv: Sequence[str], switches: Sequence[ModeSwitch]) -> Scan:
    """Read `switches` off `argv`, which is left intact. Stops at `--`.

    The spellings are argparse's, in full: `--flag`, `--flag VALUE` and
    `--flag=VALUE`; a short `-f VALUE`, `-fVALUE` and `-f=VALUE`. An
    abbreviation is NOT matched — that is `refuse_disagreement`'s case.
    """
    _check_table(switches)
    argv = list(argv)
    values: Dict[str, Any] = {s.dest: (None if s.takes_value else False)
                              for s in switches}
    seen = set()
    repeated = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            break
        hit = None
        value = None
        nxt = i + 1
        for s in switches:
            for alias in s.aliases:
                if arg == alias:
                    hit = s
                    if s.takes_value:
                        value, nxt = _value_after(argv, i)
                elif s.takes_value and arg.startswith(alias + "="):
                    hit, value = s, arg[len(alias) + 1:]
                elif (s.takes_value and len(alias) == 2
                        and arg.startswith(alias) and len(arg) > 2):
                    hit, value = s, arg[2:]
                if hit is not None:
                    break
            if hit is not None:
                break
        if hit is not None:
            if hit.takes_value:
                if hit.dest in seen and hit.dest not in repeated:
                    repeated.append(hit.dest)
                if hit.dest not in seen:
                    values[hit.dest] = value
            else:
                values[hit.dest] = True
            seen.add(hit.dest)
            if hit.implies is not None:
                values[hit.implies] = True
        i = nxt
    return Scan(switches, values, frozenset(seen), tuple(repeated))


def register(parser: argparse.ArgumentParser, switches: Sequence[ModeSwitch],
             mode: str, *,
             group_description: str = "read by this wrapper to choose the "
                                       "parser, then parsed with the options "
                                       "above") -> None:
    """Put every switch into `parser`, listing in --help those whose
    `listed_in` names `mode`.

    Every switch is parsed in every mode: the scan only chose this parser, and
    the line it reads still carries the switch. Unlisted ones are
    `argparse.SUPPRESS`ed. A switch the parser already offers under every
    spelling is its own option and is skipped (see `ModeSwitch`); one it offers
    under only some spellings, or under another dest, is a conflict.

    Reads `parser._option_string_actions`: argparse offers no public lookup.
    """
    _check_table(switches)
    listed = [s for s in switches if mode in s.listed_in]
    target = (parser.add_argument_group("modes", group_description)
              if listed else parser)
    owned = parser._option_string_actions
    for s in switches:
        taken = [alias for alias in s.aliases if alias in owned]
        if taken:
            dests = {owned[alias].dest for alias in taken}
            if len(taken) != len(s.aliases) or dests != {s.dest}:
                raise ValueError(
                    f"{s.name}: the parser already offers {taken} (dest "
                    f"{sorted(dests)}); a mode switch must match all of an "
                    f"option's spellings and its dest, or none")
            continue
        help_text = s.help if mode in s.listed_in else argparse.SUPPRESS
        if s.takes_value:
            kwargs = dict(default=None, metavar=s.metavar)
            if s.type is not None:
                kwargs["type"] = s.type
        else:
            kwargs = dict(action="store_true")
        target.add_argument(*s.aliases, dest=s.dest, help=help_text, **kwargs)


class _Unreadable:
    """Stands for a scanned value the switch's `type` refuses: equal to nothing
    the parser can have produced."""


def _given(args: argparse.Namespace, switch: ModeSwitch) -> bool:
    """Whether the parser read `switch` as given on the line."""
    value = getattr(args, switch.dest)
    return value is not None if switch.takes_value else bool(value)


def _refuse(prog: str, message: str) -> None:
    print(f"{prog}: error: {message}", file=sys.stderr)
    sys.exit(2)


def refuse_disagreement(args: argparse.Namespace, scanned: Scan,
                        prog: str) -> None:
    """Exit 2 unless the parser read every switch as `scanned` did.

    They differ when argparse resolves a spelling the scan does not match — an
    abbreviation such as `--rand`, or a combined short flag — and running on
    the scan's reading would be the wrong mode behind a flag that appears to
    have worked. A value switch given twice is refused too: the scan keeps the
    first value and argparse the last, and a mode named twice is ambiguous
    anyway.
    """
    switches = scanned.switches
    by_dest = {s.dest: s for s in switches}
    for dest in scanned.repeated:
        _refuse(prog, f"{by_dest[dest].name} may be given only once")
    for s in switches:
        if s.implies is not None:
            continue                # compared through the switch it implies
        parsed = getattr(args, s.dest)
        impliers = [t for t in switches if t.implies == s.dest]
        if not s.takes_value:
            parsed = bool(parsed) or any(_given(args, t) for t in impliers)
        expected = getattr(scanned, s.dest)
        if s.takes_value and s.type is not None and expected is not None:
            # The parser already applied it to ITS reading, and exited had it
            # failed; the scan's reading is compared in the same terms. One the
            # type refuses is a reading the parser did not share.
            try:
                expected = s.type(expected)
            except (ValueError, TypeError, argparse.ArgumentTypeError):
                expected = _Unreadable
        if parsed != expected:
            also = "".join(f" (or {t.name})" for t in impliers)
            _refuse(prog, f"{s.name}{also} was read differently by the argv "
                          f"scan that picks the mode and by the parser. Spell "
                          f"it out in full; abbreviations are resolved by the "
                          f"parser and are invisible to the scan.")
