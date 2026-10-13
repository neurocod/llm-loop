"""clispec - the one declaration of every option this family's command lines carry.

Three things need the same knowledge and used to hold three copies of it: the
sequential parser, the parallel parser (whose header called itself "a trimmed
copy of cyclecore.parse_args"), and the hand-written alias table `cmdline` needs
to strip a flag out of an argv. They drifted exactly as far as nobody was
reading: seven options were offered by both parsers, two of those declarations
byte-identical, and the alias table's own comment said where its rows had been
copied from.

So `OPTIONS` below is the single table, and everything else is DERIVED from it:

  * `build_parser(mode, ...)` builds either parser out of it, in the order the
    mode's list names — which is the order `--help` prints, so that order is
    part of the declaration and not an accident of where an `add_argument` line
    happened to sit;
  * `FLAG_ALIASES` (re-exported by `cmdline`, which is where callers still name
    it) is the same table projected down to what an argv rewriter needs: every
    spelling of a flag, and whether it eats the next token. `flag_aliases(args)`
    adds the flags a host recorded (`record_flags`; `modeswitch.register` for
    its mode switches), so no host's flags are rows here.

A row that `build_parser` never adds is legitimate and carries `kwargs=None`:
`--session-limit`/`--weekly-limit` are ceilings the status line can edit into a
command line although neither engine parser offers them (a host's own hook
may). The alias table must know those spellings anyway — it is what stops their
values from being misread as free-standing tokens.

The mode's option list and `OPTIONS` are checked against each other, and against
what argparse actually built, by `tests/test_clispec.py`. That gate is the
reason this table can be trusted as the only copy. What it cannot see is a host's
own `extra_options` hook, so the parser half of it is `unstrippable_flags`,
public for each host to run over the parsers it builds.
"""

import argparse
import re
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from . import costlog
from . import providers
from . import termio
from .gitpush import GIT_PUSH_POLICY, GitPushPolicy

__all__ = [
    "DEFAULT_JOBS",
    "FLAG_ALIASES",
    "Flag",
    "OPTIONS",
    "OPTION_ORDER",
    "Option",
    "PARALLEL",
    "RESERVED_DEST_PREFIX",
    "SEQUENTIAL",
    "build_parser",
    "directory",
    "duration",
    "flag_aliases",
    "log_file",
    "parse_duration",
    "record_flags",
    "recorded_flags",
    "unstrippable_flags",
]

# The two parsers this table serves. A mode picks BOTH the ordered option list
# and, where the two runners mean different things by the same flag, which help
# text is printed - so there is one name for "which command line is this".
SEQUENTIAL = "sequential"
PARALLEL = "parallel"

# Default worker count. The work is cheap and fully independent, so a handful of
# concurrent jobs is the sweet spot before the shared session budget, not CPU,
# becomes the bottleneck. `--jobs` itself defaults to None and `run_parallel`
# applies this number; what forces the constant to live HERE is the help text
# below, which has to print it. (The other direction is closed anyway: `parallel`
# imports this module, so this module cannot import `parallel`.)
DEFAULT_JOBS = 10


def log_file(text: str) -> str:
    """argparse `type=` of --cost-log: a log path, never an empty one.

    The flag's PRESENCE is what selects the report (see `run_loop`), so an empty
    value cannot mean "absent" there without starting the agent loop the user
    asked not to run. And it arrives in two spellings for one intent: PowerShell
    5.1 drops the `""` of `--cost-log ""`, which argparse then refuses as a bare
    flag ("expected one argument"), while `--cost-log=` delivers "". Refusing the
    empty value makes both a usage error, exit 2, before anything else runs.

    Which paths name no log is `costlog.named_log`'s rule, the one `run_loop`
    applies to a namespace built past this parser; here it only becomes a
    usage error, so the two cannot disagree on a spelling.
    """
    try:
        costlog.named_log(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "needs a log file; use --cost without --cost-log to report on this "
            "entry point's own log") from None
    return text


def directory(text: str) -> str:
    """argparse `type=` of -C/--project-dir: a directory, never an empty one.

    An empty value has no meaning of its own, only accidents: `--project-dir=`
    delivers "", which reads as "no root given" (the launch cwd) to the engine
    and as `Path("")` - also the launch cwd, but by a different road - to a
    wrapper that anchors its default elsewhere; while PowerShell 5.1 drops the
    `""` of `--project-dir ""`, which argparse then refuses as a bare flag. Two
    spellings of one input, two outcomes; refusing the empty value makes both a
    usage error, exit 2, before anything runs. Omitting the flag is how one asks
    for the default root. Returns the text unchanged: resolving it is
    `projectroot.set_project_root`'s job, which refuses a blank path as well.
    """
    if not text.strip():
        raise argparse.ArgumentTypeError(
            "needs a directory; omit the flag to use the default project root")
    return text


# One or more `<number><unit>` parts, spaces allowed after each: 1h30m, 90s,
# `1h 30m`, 1.5h.
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)\s*([hms])\s*")
_DURATION = re.compile(r"(?:\d+(?:\.\d+)?\s*[hms]\s*)+")


def parse_duration(text: str) -> float:
    """Parse a duration like '29m', '1h', '90s', '1h30m' into seconds.

    A bare number is treated as minutes ('29' == '29m'). Raises ValueError on
    anything it can't make sense of.

    Lives here, beside its `type=` validator `duration`, rather than in the
    runner that waits on it: this module cannot import `cyclecore` (which
    imports this one), and the check has to run inside the parser.
    """
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    if text.isdigit():  # bare number — minutes
        return int(text) * 60
    # The WHOLE text, not the parts that happen to match: searching for
    # `<number><unit>` read `1h30` as 1 h (the 30 dropped) and `abc1m` as 1 min.
    if not _DURATION.fullmatch(text):
        raise ValueError(f"cannot parse duration: {text!r}")
    units = {"h": 3600, "m": 60, "s": 1}
    return sum(float(value) * units[unit]
               for value, unit in _DURATION_PART.findall(text))


def duration(text: str) -> str:
    """argparse `type=` of --start-in: a duration `parse_duration` reads.

    Checked at parse time because the runner reads the value only when the wait
    begins — after the prologue, and in a host wrapper after its script lock,
    stop-file wait and whatever else it does on the way in — so a typo used to
    fail late, and differently per host. An empty value (`--start-in=`) is
    refused too: it used to mean "no delay" while `--start-in ""`, which
    PowerShell 5.1 delivers as a bare flag, was a usage error. A line that
    waits for nothing still fails: a dry run or a report (`--cost`, `--log`)
    never read the value and used to run on past a typo in it; it is exit 2
    now, like any other malformed option on that line.

    Returns the text unchanged, so `args.start_in` stays the spelling the wait
    announces.
    """
    try:
        parse_duration(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{exc}; expected e.g. 29m, 1h30m, 90s, or a bare number of "
            f"minutes") from None
    return text


class Flag(NamedTuple):
    """One canonical flag: every spelling argparse accepts, and its arity."""

    aliases: Tuple[str, ...]
    takes_value: bool


class Option(NamedTuple):
    """One option of the family: what argparse needs, plus what an argv rewriter
    needs.

    `aliases` lists every spelling, canonical first-or-anywhere but always
    included; `takes_value` is the arity the rewriter uses to decide whether the
    NEXT token belongs to this flag - and is checked against the arity argparse
    derives from `kwargs`, so the two cannot disagree quietly.

    `kwargs` is handed to `add_argument` verbatim (minus the flag strings and
    `help`); `None` means `build_parser` never adds this flag and the row
    serves the argv rewriter alone; see the module header. `help` is
    what both modes print, and `parallel_help` replaces it in the parallel
    parser for the options the two runners genuinely mean differently (an
    iteration cap vs a total-files cap, and so on).
    """

    aliases: Tuple[str, ...]
    takes_value: bool
    kwargs: Optional[Dict[str, Any]] = None
    help: str = ""
    parallel_help: Optional[str] = None


# The table. Keys are the canonical long spellings that `cmdline`'s `overrides`
# dict speaks, and the ORDER is part of the contract: overrides are appended in
# it, so a rendered command line is deterministic regardless of dict ordering.
# (The order each parser prints is a different order, and each mode's list below
# states its own.)
OPTIONS: Dict[str, Option] = {
    # --- value-taking options ---------------------------------------------------
    # Most option names are kept in sync with continuous_claude.py (kebab-case,
    # and --max-runs for the iteration cap). The former spellings (--max,
    # --startIn) stay on as accepted aliases so existing invocations keep working.
    "--max-runs": Option(
        aliases=("-m", "--max-runs", "--max"),
        takes_value=True,
        kwargs=dict(dest="max", type=int, default=None, metavar="N"),
        help="stop after N iterations (default: run forever); "
             "--max is a deprecated alias",
        parallel_help="stop after processing N files total, across all workers "
                      "(default: drain the whole list); --max is a deprecated alias",
    ),
    "--start-in": Option(
        aliases=("-s", "--start-in", "--startIn"),
        takes_value=True,
        kwargs=dict(dest="start_in", metavar="DURATION", type=duration),
        help="wait this long before starting the loop, e.g. 29m, 1h30m",
    ),
    "--git-push": Option(
        aliases=("-g", "--git-push"),
        takes_value=True,
        kwargs=dict(dest="git_push",
                    choices=[pol.value for pol in GitPushPolicy],
                    default=GIT_PUSH_POLICY.value),
        help="when to `git push` at the start of each iteration: "
             "none | after_new_commits | each_hour "
             f"(default: {GIT_PUSH_POLICY.value})",
        parallel_help="when to `git push`: none | after_new_commits | each_hour "
                      f"(default: {GIT_PUSH_POLICY.value})",
    ),
    "--project-dir": Option(
        aliases=("-C", "--project-dir"),
        takes_value=True,
        kwargs=dict(dest="project_dir", metavar="DIR", default=None,
                    type=directory),
        help="project root: cwd for git/provider CLI, base for the stop "
             "file and the Driver's relative paths "
             "(default: the current working directory)",
        parallel_help="project root: cwd for git/provider CLI, base for the stop "
                      "file and the list's relative paths "
                      "(default: the current working directory)",
    ),
    "--jobs": Option(
        aliases=("-j", "--jobs"),
        takes_value=True,
        kwargs=dict(type=int, default=None, metavar="N"),
        help="number of concurrent workers (default: the driver's "
             f"`jobs`, else {DEFAULT_JOBS})",
    ),
    # No parser yet (a later wave adds them); listed so the status line can
    # already express an edited ceiling as a command line.
    "--session-limit": Option(aliases=("--session-limit",), takes_value=True),
    "--weekly-limit": Option(aliases=("--weekly-limit",), takes_value=True),
    "--cost-log": Option(
        aliases=("--cost-log",),
        takes_value=True,
        kwargs=dict(dest="cost_log", metavar="LOG", type=log_file),
        help="report on this log file instead of this entry point's "
             "own — a rotated backup (<app>-<project>.log.1) or a "
             "copy; implies --cost unless --stat is selected",
    ),
    # --- store_true / store_const options ---------------------------------------
    "--codex": Option(
        aliases=("--codex",),
        takes_value=False,
        kwargs=dict(action="store_const", const="codex", dest="provider",
                    default=None),
        help="run Codex CLI instead of the Driver's default provider",
    ),
    "--dry-run": Option(
        aliases=("-d", "--dry-run"),
        takes_value=False,
        kwargs=dict(action="store_true"),
        help="only print the commands, don't run the LLM CLI",
        parallel_help="only print the commands that would run, don't run the LLM CLI "
                      "and don't touch the list",
    ),
    # No -r short flag: -r is --review-prompt in continuous_claude.py, so it is
    # left free here rather than reused for --raw.
    "--raw": Option(
        aliases=("--raw",),
        takes_value=False,
        kwargs=dict(action="store_true"),
        help="print raw JSON events (for debugging)",
    ),
    "--log": Option(
        aliases=("--log",),
        takes_value=False,
        kwargs=dict(action="store_true"),
        help="print the mirror log file path and exit (no loop is run)",
    ),
    "--cost": Option(
        aliases=("-c", "--cost"),
        takes_value=False,
        kwargs=dict(action="store_true"),
        help="print per-session cost totals from the mirror log and "
             "exit (no loop is run)",
    ),
    "--stat": Option(
        aliases=("--stat",),
        takes_value=False,
        kwargs=dict(action="store_true"),
        help="print a table of total time, average per iteration and percentage "
             "by state from the mirror log and exit (no loop is run)",
    ),
    "--ignore-usage": Option(
        aliases=("--ignore-usage",),
        takes_value=False,
        kwargs=dict(action="store_true"),
        help="don't pause on the Current-session usage limit "
             "(by default the workers pause together when the session "
             "budget is exhausted)",
    ),
    # No short alias: this is a rescue hatch for an odd terminal, not a knob to
    # reach for. LLM_LOOP_STATUSLINE=0 does the same without editing a command
    # line, and no TTY disables it by itself.
    "--no-statusline": Option(
        aliases=("--no-statusline",),
        takes_value=False,
        kwargs=dict(dest="no_statusline", action="store_true"),
        help="do not pin the interactive status rows at the bottom of "
             f"the terminal (same as {termio.ENV_FLAG}=0)",
    ),
    # Same shape of rescue hatch as --no-statusline, and for the same reason: it
    # turns off a transport, not a feature. A note typed with the `m` key still
    # reaches the agent - with the next iteration's prompt instead of the one
    # already running.
    "--no-live-messages": Option(
        aliases=("--no-live-messages",),
        takes_value=False,
        kwargs=dict(dest="no_live_messages", action="store_true"),
        help="do not keep the agent's stdin open for notes typed "
             "during an iteration; they wait for the next prompt "
             f"instead (same as {providers.LIVE_MESSAGES_ENV}=0)",
        parallel_help="do not keep the agent's stdin open for notes typed "
                      "during an iteration; they wait for the next prompt "
                      f"instead (same as {providers.LIVE_MESSAGES_ENV}=0). "
                      "Notes need a single worker either way",
    ),
}


# Each mode's option set AND the order `--help` prints it in. Two lists rather
# than a flag on each row because the orders differ (the parallel runner leads
# with the option that makes it parallel), and because reading one list answers
# "what does this command line accept?" without walking the whole table.
OPTION_ORDER: Dict[str, Tuple[str, ...]] = {
    SEQUENTIAL: (
        "--max-runs",
        "--codex",
        "--dry-run",
        "--log",
        "--cost",
        "--stat",
        "--cost-log",
        "--raw",
        "--start-in",
        "--git-push",
        "--project-dir",
        "--no-statusline",
        "--no-live-messages",
    ),
    PARALLEL: (
        "--jobs",
        "--max-runs",
        "--codex",
        "--dry-run",
        "--start-in",
        "--git-push",
        "--project-dir",
        "--ignore-usage",
        # Accepted here too: the flag is documented as a general one, and a
        # batching wrapper hands these args to the sequential loop (which honours
        # it), so a parser that rejected it would exit 2 on a documented
        # spelling.
        "--no-statusline",
        "--no-live-messages",
    ),
}

# Underscored because nothing outside this file reads it — which is what `_`
# means here now (see tests/test_package_privacy.py). Each mode's default
# `--help` blurb; a caller that passes `description=` overrides it.
_DESCRIPTIONS: Dict[str, str] = {
    SEQUENTIAL: "Autonomous loop driving an LLM CLI.",
    PARALLEL: "Parallel autonomous loop running N concurrent LLM workers "
              "over a list file.",
}


def build_parser(mode: str, *, prog: str, description: Optional[str] = None,
                 extra_options: Optional[Callable[[argparse.ArgumentParser],
                                                  None]] = None
                 ) -> argparse.ArgumentParser:
    """The parser for one mode, built from `OPTIONS` in that mode's order.

    Returned rather than run, which is the whole point of the split: a gate can
    walk a parser's actions and check them against the table, and that check is
    impossible while the only way to reach a parser is to hand it an argv and
    have it call `sys.exit`.

    `extra_options` is handed the finished parser and is how a wrapper adds
    options of its own — its mode switches above all (`modeswitch.register`):
    the wrapper's argv scan reads them to choose THIS parser, which then parses
    them with the rest of the line. It runs last so a wrapper's flags print
    after the family's.
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or _DESCRIPTIONS[mode],
    )
    for name in OPTION_ORDER[mode]:
        option = OPTIONS[name]
        if option.kwargs is None:                       # pragma: no cover - gated
            raise ValueError(f"{name} is offered by {mode} but declares no parser "
                             f"arguments")
        help_text = option.help
        if mode == PARALLEL and option.parallel_help is not None:
            help_text = option.parallel_help
        parser.add_argument(*option.aliases, help=help_text, **option.kwargs)
    if extra_options is not None:
        extra_options(parser)
    return parser


# The alias table, projected out of the one above. `cmdline` re-exports it under
# the name its callers already use; deriving it is what makes "every flag the
# parsers offer is strippable from an argv" true by construction instead of by
# somebody remembering to copy a row.
FLAG_ALIASES: Dict[str, Flag] = {
    name: Flag(option.aliases, option.takes_value)
    for name, option in OPTIONS.items()
}

# Every dest starting with this is the engine's: the parser defaults it keeps
# records under (`record_flags`'s, and `modeswitch.register`'s occurrence
# counts) end up as attributes of every namespace the parser returns, so an
# option under such a dest would overwrite a record with its value, or have
# `set_defaults` overwrite its default with a record. One prefix rather than a
# list of names, so this module refuses `modeswitch`'s record without naming it.
RESERVED_DEST_PREFIX = "_llm_loop_"
# The parser default `record_flags` writes: {dest: Flag}.
_RECORD = RESERVED_DEST_PREFIX + "flags"


def _reserved(dests) -> List[str]:
    return sorted({d for d in dests if d.startswith(RESERVED_DEST_PREFIX)})


def record_flags(parser: argparse.ArgumentParser,
                 flags: Dict[str, Flag]) -> None:
    """Tell the argv rewriter about `parser`'s own flags, {dest: Flag}: each
    becomes a row of `flag_aliases(parser)` and of every namespace it parses.

    A host's value-taking flag needs a row as much as an engine option does
    (see `unstrippable_flags`); `modeswitch.register` records a host's mode
    switches through here, and a hook records an ordinary value flag the same
    way. Repeated calls add to the record, a dest given again replacing its
    row. The rows are bare `Flag`s, never a host's richer row (a `ModeSwitch`
    carries frozensets and a `type` callable): the record ends up in every
    namespace, which a host may `json.dumps(vars(args))` or pickle. A dest under `RESERVED_DEST_PREFIX`, among `flags` or the parser's
    options, is a ValueError; an option added under one after this call is
    `unstrippable_flags`'s to report.

    Reads `parser._actions`: argparse offers no public walk of its options.
    """
    collide = _reserved(set(flags) | {a.dest for a in parser._actions})
    if collide:
        raise ValueError(f"dest {collide} starts with {RESERVED_DEST_PREFIX!r}, "
                         f"where llm_loop keeps its records; give that option "
                         f"another dest")
    record = dict(recorded_flags(parser))
    record.update((dest, Flag(tuple(flag.aliases), flag.takes_value))
                  for dest, flag in flags.items())
    parser.set_defaults(**{_RECORD: record})


def recorded_flags(source: Any) -> Dict[str, Flag]:
    """What `record_flags` put into `source` — a parser, or a namespace it
    parsed — as {dest: Flag}; empty when nothing (None included).

    A record that is not a dict is a ValueError: an option's value, under a
    reserved dest (`record_flags` refuses such an option added before it,
    `unstrippable_flags` one added after it, whatever its value).
    """
    if isinstance(source, argparse.ArgumentParser):
        record = source.get_default(_RECORD)
    else:
        record = getattr(source, _RECORD, None)
    if record is not None and not isinstance(record, dict):
        raise ValueError(f"{_RECORD} holds {record!r}, not record_flags's "
                         f"record: an option uses it as its dest")
    return record or {}


def flag_aliases(source: Any = None) -> Dict[str, Flag]:
    """The rewriter's vocabulary for one host: `FLAG_ALIASES`, then a row per
    flag recorded into `source` (`recorded_flags`; None: the engine's rows
    alone).

    A recorded flag that IS an engine row — the parallel runner's -j/--jobs,
    which the sequential parser gets as a host's mode switch — keeps that row.
    One sharing only some spellings with a row, or differing in arity, is a
    ValueError: the rewriter would strip one flag as the other.
    """
    table = dict(FLAG_ALIASES)
    owner = {alias: name for name, flag in table.items() for alias in flag.aliases}
    for aliases, takes_value in recorded_flags(source).values():
        name = aliases[-1]
        clash = sorted({owner[a] for a in aliases if a in owner})
        if not clash:
            table[name] = Flag(tuple(aliases), takes_value)
            continue
        same = (len(clash) == 1
                and set(table[clash[0]].aliases) == set(aliases)
                and table[clash[0]].takes_value == takes_value)
        if not same:
            raise ValueError(f"recorded flag {name} shares a spelling with "
                             f"the engine's {clash}")
    return table


def unstrippable_flags(parser: argparse.ArgumentParser) -> List[str]:
    """What `cmdline.rebuild_argv` would get wrong about `parser`'s options, one
    line per spelling; empty when nothing.

    The contract of the `extra_options` / `Driver.add_cli_options` seam, published
    so every host can hold its own hook to it: `build_parser` is a loop over
    `OPTIONS`, so the hook is the only way a spelling the table has never heard
    of reaches a parser. The table checked is the one the run's rewriter uses,
    `flag_aliases(parser)`: a host's flag is declared by being recorded
    (`record_flags`, which `modeswitch.register` calls for a host's mode
    switches). Three things break a rebuilt command line:

      * a VALUE-taking spelling the table does not declare. The rewriter copies
        an unknown flag through verbatim and then reads its value as a token of
        its own, so a folder or a count that happens to spell `-m` is stripped
        together with whatever follows it;
      * a declared spelling whose arity argparse disagrees with. `takes_value`
        decides whether the NEXT token belongs to the flag: wrong, and removing
        it either eats a neighbour or leaves an orphan value on the line;
      * a spelling of ANY arity but none or one token (`nargs` `'?'`, `'*'`,
        `'+'`, `N > 1`), declared or not. The rewriter strips a flag plus at
        most one token, so a bare `nargs='?'` flag eats its neighbour and an
        `nargs=2` one leaves an orphan value behind.

    An undeclared switch (`nargs == 0`) is NOT reported: it has no value to
    misread and is copied through as it stands. An option of any arity under a
    `RESERVED_DEST_PREFIX` dest is: it shadows a record the engine reads, this
    table's own or the mode switches' occurrence counts.

    Reads `parser._actions`: argparse offers no public walk of its options.
    """
    aliases = flag_aliases(parser)
    owner = {alias: canonical
             for canonical, flag in aliases.items()
             for alias in flag.aliases}
    problems = []
    for action in parser._actions:
        if "--help" in action.option_strings:
            continue                # argparse's own; no table declares it
        if _reserved([action.dest]):
            problems.append(
                f"{'/'.join(action.option_strings) or action.dest}: dest "
                f"{action.dest!r} starts with {RESERVED_DEST_PREFIX!r}, where "
                f"llm_loop keeps its records")
            continue
        if action.option_strings and action.nargs not in (None, 0, 1):
            problems.append(
                f"{action.option_strings[0]}: nargs={action.nargs!r}, but a "
                f"rebuilt argv strips a flag with at most one value token")
            continue
        takes_value = action.nargs != 0
        for spelling in action.option_strings:
            canonical = owner.get(spelling)
            if canonical is None:
                if takes_value:
                    problems.append(
                        f"{spelling} takes a value but is neither in "
                        f"clispec.OPTIONS nor recorded for the rewriter "
                        f"(clispec.record_flags, which modeswitch.register "
                        f"calls), so a rebuilt argv misreads its value")
            elif aliases[canonical].takes_value != takes_value:
                problems.append(
                    f"{spelling}: the rewriter's row {canonical!r} says "
                    f"takes_value={aliases[canonical].takes_value}, "
                    f"argparse built nargs={action.nargs!r}")
    return problems
