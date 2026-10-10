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

Here a wrapper declares a table of `ModeSwitch` rows and makes two calls with
it:

  * `register(parser, switches, mode)`, from each driver's `add_cli_options`,
    puts every switch into that driver's parser, so the parser reads the WHOLE
    line, switches included, with one grammar: a value flag followed by a
    switch is argparse's "expected one argument", never a quiet substitution;
  * `parse(argv, switches, choose)` is the dispatch: it scans the switches off
    argv (leaving it intact) only to CHOOSE a driver and its parser, parses the
    line with that parser, and exits 2 when the parser read a switch
    differently from the scan — an abbreviation argparse resolves and the scan
    does not (`--rand`), which would otherwise run the default mode behind a
    flag that looks as if it worked.

`scan` stays public as a read-only question about argv (is `--prompt` on the
line?); the steps of the dispatch are not offered separately, because a host
that copied only some of them read a switch and then ignored it. For the same
reason `Driver.main()` and `main_parallel()`, which never act on a switch,
refuse a line that gives one (`refuse_undispatched`). The README's "Wrapper
options in `--help`" shows a whole wrapper.

A switch's `dest` names it everywhere: the scan result's attribute and the
parsed namespace's are the same name.
"""

import argparse
import sys
from typing import (Any, Callable, Dict, FrozenSet, NamedTuple, Optional,
                    Sequence, Tuple)

from . import clispec

__all__ = [
    "Choice",
    "ModeSwitch",
    "Scan",
    "parse",
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
    it alone, its `dest` must be that option's, and the parser's reading of it
    is final — abbreviated or repeated, as for any other option of that parser
    (`-j 2 -j 3` runs 3 workers, as in every `main_parallel` host). That is
    sound only while the parser owning a switch is the one the switch itself
    chooses, as the parallel parser is for `-j`.
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
    last value, as argparse keeps it, or None. `seen` holds the dests given on
    the line (a value switch given without a value is seen and reads None),
    `counts` how many times each dest was given.
    """

    def __init__(self, switches: Sequence[ModeSwitch], values: Dict[str, Any],
                 counts: Dict[str, int]):
        self.switches = tuple(switches)
        self.values = dict(values)
        self.counts = dict(counts)
        self.seen = frozenset(dest for dest, n in counts.items() if n)

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
    reserved = {"switches", "values", "seen", "counts"} & set(dests)
    if reserved:
        raise ValueError(f"mode switch dest shadows a Scan attribute: {reserved}")
    by_dest = {s.dest: s for s in switches}
    for s in switches:
        if s.implies is not None:
            target = by_dest.get(s.implies)
            if target is None or target.takes_value:
                raise ValueError(f"{s.name} implies {s.implies!r}, which is not "
                                 f"a boolean switch of the same table")


class _Unscannable(Exception):
    """The scan parser met a line argparse refuses (`--random=x`)."""


class _ScanParser(argparse.ArgumentParser):
    """An argparse parser that raises instead of printing usage and exiting:
    the scan must not end the process, the chosen parser reports the error."""

    def error(self, message):
        raise _Unscannable(message)


class _Occurrences(argparse.Action):
    """Scan side: records each occurrence of a switch, in order — True for a
    boolean, the value (None when missing) for a value switch."""

    def __call__(self, parser, namespace, values, option_string=None):
        got = list(getattr(namespace, self.dest) or ())
        got.append(True if self.nargs == 0 else values)
        setattr(namespace, self.dest, got)


def _scan_parser(switches: Sequence[ModeSwitch]) -> argparse.ArgumentParser:
    """The grammar the scan reads argv with: argparse's own, on this Python.

    Hand-written, the scan was a second grammar that drifted from argparse's:
    which dash-led token is a value (`-`, `"-x y"`, and since 3.14 `-1x` are
    values to argparse) and short flags combined into one token (`-dp`). Here
    the table's switches and every option of `clispec.OPTIONS` are declared
    with their arity (a switch wins a spelling both declare), so argparse
    tokenises the line exactly as the chosen parser will; `-h/--help` too,
    since `-ph` combines with it.

    Two deliberate differences, neither of which can reach a runner:
      * `allow_abbrev=False`: an abbreviation is left unmatched, which is what
        `_refuse_disagreement` detects against the chosen parser;
      * a value takes `nargs='?'`, so a missing value (`--finish -p`) reads
        None instead of failing the scan; the chosen parser then says
        "expected one argument".
    A spelling neither the table nor `clispec.OPTIONS` declares (a host's own
    option) is an unknown one: its value, if it has one, is read as a token of
    its own — harmless, since a value that spells an option makes the chosen
    parser refuse the line anyway.
    """
    parser = _ScanParser(prog="modeswitch.scan", add_help=False,
                         allow_abbrev=False)
    for s in switches:
        parser.add_argument(*s.aliases, dest=s.dest, action=_Occurrences,
                            nargs="?" if s.takes_value else 0, default=None)
    taken = set(parser._option_string_actions)
    rows = [(("-h", "--help"), False)]
    rows += [(option.aliases, option.takes_value)
             for option in clispec.OPTIONS.values()]
    for n, (aliases, takes_value) in enumerate(rows):
        free = [alias for alias in aliases if alias not in taken]
        if not free:
            continue
        kwargs = (dict(nargs="?") if takes_value
                  else dict(action="store_true"))
        parser.add_argument(*free, dest=f"_option_{n}", **kwargs)
        taken.update(free)
    return parser


def scan(argv: Sequence[str], switches: Sequence[ModeSwitch]) -> Scan:
    """Read `switches` off `argv`, which is left intact.

    With argparse's grammar (see `_scan_parser`): every full spelling
    (`--flag VALUE`, `--flag=VALUE`, `-fVALUE`, `-f=VALUE`, `-dp`), nothing
    after `--`, and no abbreviation — that is `_refuse_disagreement`'s case. A
    line argparse refuses outright reads as no switch at all: the parser that
    choice picks refuses it too, with its own usage.

    A question about argv, not a dispatch: an abbreviated switch reads as
    absent here. Choose a mode with `parse`, which holds the scan to the
    parser's reading.
    """
    _check_table(switches)
    try:
        parsed, _rest = _scan_parser(switches).parse_known_args(list(argv))
    except _Unscannable:
        parsed = argparse.Namespace(**{s.dest: None for s in switches})
    values: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    for s in switches:
        got = getattr(parsed, s.dest) or []
        counts[s.dest] = len(got)
        if s.takes_value:
            values[s.dest] = got[-1] if got else None
        else:
            values[s.dest] = bool(got)
    for s in switches:
        if s.implies is not None and counts[s.dest]:
            values[s.implies] = True
    return Scan(switches, values, counts)


# The namespace attribute `register`'s actions count occurrences into: a dict
# of every registered dest to its count, so a dest missing from it is one the
# parser owns (see `ModeSwitch`).
_COUNTS = "_mode_switch_counts"
# Beside it, each registered dest's `ModeSwitch.name`, for a refusal to spell.
_NAMES = "_mode_switch_names"


class _Counted(argparse.Action):
    """`register`'s action: `store` (or `store_true` with nargs=0) that also
    counts the occurrence in `_COUNTS`. The dict is copied, never mutated: it
    starts as the parser's default object, shared by every parse."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True if self.nargs == 0 else values)
        counts = dict(getattr(namespace, _COUNTS))
        counts[self.dest] += 1
        setattr(namespace, _COUNTS, counts)


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

    Each registered switch also COUNTS its occurrences into the namespace
    (`_COUNTS`), whatever spelling argparse resolved, so `_refuse_disagreement`
    compares occurrences rather than spellings: `--finish=x --fin=x` is a
    second --finish exactly as `--finish=x --finish=x` is.

    Reads `parser._option_string_actions`: argparse offers no public lookup.
    """
    _check_table(switches)
    listed = [s for s in switches if mode in s.listed_in]
    target = (parser.add_argument_group("modes", group_description)
              if listed else parser)
    owned = parser._option_string_actions
    counts = dict(parser.get_default(_COUNTS) or {})
    names = dict(parser.get_default(_NAMES) or {})
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
            kwargs = dict(nargs=0, default=False)
        target.add_argument(*s.aliases, dest=s.dest, help=help_text,
                            action=_Counted, **kwargs)
        counts[s.dest] = 0
        names[s.dest] = s.name
    parser.set_defaults(**{_COUNTS: counts, _NAMES: names})


class _Unreadable:
    """Stands for a scanned value the switch's `type` refuses: equal to nothing
    the parser can have produced."""


def _refuse(prog: str, message: str) -> None:
    print(f"{prog}: error: {message}", file=sys.stderr)
    sys.exit(2)


def _read_alike(switch: ModeSwitch, parsed: Any, scanned: Any) -> bool:
    """Whether a value switch given once holds the same value on both sides.

    The parser already applied `type` to ITS reading, and exited had it
    failed; the scan's reading is compared in the same terms. One the type
    refuses is a reading the parser did not share.
    """
    if switch.type is not None and scanned is not None:
        try:
            scanned = switch.type(scanned)
        except (ValueError, TypeError, argparse.ArgumentTypeError):
            scanned = _Unreadable
    return parsed == scanned


def _refuse_disagreement(args: argparse.Namespace, scanned: Scan,
                         prog: str) -> None:
    """Exit 2 unless the parser read every registered switch as `scanned` did.

    They differ when argparse resolves a spelling the scan leaves unmatched —
    an abbreviation such as `--rand`, or a short flag combined with one only
    the host's parser knows — and running on the scan's reading would be the
    wrong mode behind a flag that appears to have worked. What is compared is
    OCCURRENCES, which `register` counts whatever the spelling, so the outcome
    never depends on how a switch was spelled:

      * a boolean switch: given on both sides, or on neither (`-p --para` is
        `-p -p`, which is harmless);
      * a value switch: given as often on both sides, and at most once — the
        mode it names would be ambiguous — with the same value.

    A switch the parser owns (`register` skipped it) is not compared: its
    reading is that parser's, as for any other option it offers (see
    `ModeSwitch`).
    """
    counts = getattr(args, _COUNTS, {})
    for s in scanned.switches:
        if s.dest not in counts:
            continue
        parsed_n, scanned_n = counts[s.dest], scanned.counts[s.dest]
        if s.takes_value and parsed_n > 1:
            _refuse(prog, f"{s.name} may be given only once")
        agree = (parsed_n == scanned_n and (
                     parsed_n == 0 or _read_alike(s, getattr(args, s.dest),
                                                  getattr(scanned, s.dest)))
                 if s.takes_value else bool(parsed_n) == bool(scanned_n))
        if not agree:
            _refuse(prog, f"{s.name} was read differently by the argv scan "
                          f"that picks the mode and by the parser. Spell it "
                          f"out in full and on its own: an abbreviation, or a "
                          f"short flag combined with one the scan does not "
                          f"know, is resolved by the parser alone.")


class Choice(NamedTuple):
    """What a `parse` chooser returns in the plain case: the driver class whose
    `resolved_prog`, `description` and `add_cli_options` label and extend the
    parser, and the engine function that builds and runs that parser
    (`parse_args` or `parse_parallel_args`). A host may return any object with
    these two attributes — a richer row of its own that also names the mode."""

    driver: type
    parse: Callable[..., argparse.Namespace]


def parse(argv: Optional[Sequence[str]], switches: Sequence[ModeSwitch],
          choose: Callable[[Scan], Any]) -> Tuple[Any, Scan, argparse.Namespace]:
    """A wrapper's whole dispatch: `(choice, scanned, args)` for `argv`.

    `scan` reads the switches, `choose(scanned)` picks the driver and its
    parser (see `Choice`), that parser reads the whole line — the switches
    included, through the driver's `add_cli_options`, which is expected to
    `register` this same table — and the two readings are held equal
    (`_refuse_disagreement`). A usage error, `--help` and a disagreement all
    exit here, before the host does anything. `argv` None is `sys.argv[1:]`,
    as for the engine's own parsers.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    scanned = scan(argv, switches)
    choice = choose(scanned)
    driver = choice.driver
    prog = driver.resolved_prog()
    args = choice.parse(argv, prog=prog, description=driver.description,
                        extra_options=driver.add_cli_options)
    _refuse_disagreement(args, scanned, prog)
    return choice, scanned, args


def refuse_undispatched(args: argparse.Namespace, prog: str) -> None:
    """Exit 2 when `args` carries a registered switch the line gave.

    For the engine's own entry points, `Driver.main()` and `main_parallel()`:
    they parse a driver's `add_cli_options` but never act on a mode switch,
    so a host that registered its table and then dispatched through them would
    run the default mode behind `-p`. A switch the line does not give changes
    nothing there, so only a given one is refused.
    """
    counts = getattr(args, _COUNTS, {})
    names = getattr(args, _NAMES, {})
    given = [dest for dest, n in counts.items() if n]
    if given:
        _refuse(prog, f"{names.get(given[0], given[0])} is a mode switch this "
                      f"entry point never reads; the wrapper has to dispatch "
                      f"through llm_loop.modeswitch.parse")
