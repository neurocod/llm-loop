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
    spelling of a flag, and whether it eats the next token.

A row that no parser offers is legitimate and carries `kwargs=None`: the wrapper
reads `--parallel`/`--grow-kit`/`--random`/`--grow-kit-periodically` out of argv
before either parser exists, and `--session-limit`/`--weekly-limit` are ceilings
the status line can edit into a command line that has no parser yet. The alias
table must know those spellings anyway — it is what stops their values from
being misread as free-standing tokens.

The mode's option list and `OPTIONS` are checked against each other, and against
what argparse actually built, by `tests/test_clispec.py`. That gate is the
reason this table can be trusted as the only copy.
"""

import argparse
from typing import Any, Callable, Dict, NamedTuple, Optional, Tuple

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
    "SEQUENTIAL",
    "build_parser",
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
    `help`); `None` means no parser offers this flag at all - see the module
    header. `help` is what both modes print, and `parallel_help` replaces it in
    the parallel parser for the options the two runners genuinely mean
    differently (an iteration cap vs a total-files cap, and so on).
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
        kwargs=dict(dest="start_in", metavar="DURATION"),
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
        kwargs=dict(dest="project_dir", metavar="DIR", default=None),
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
    # Wrapper-only: runGenerateModels reads these two out of argv before either
    # parser. Both take a value, which is the reason they must be listed even
    # though nothing here parses them: an unlisted value-taking flag has its
    # VALUE read as a free-standing token, and a folder or a count that happens
    # to spell `-m` is then stripped along with the token after it.
    "--grow-kit-periodically": Option(aliases=("--grow-kit-periodically",),
                                      takes_value=True),
    "--finish": Option(aliases=("--finish",), takes_value=True),
    "--cost-log": Option(
        aliases=("--cost-log",),
        takes_value=True,
        kwargs=dict(dest="cost_log", metavar="LOG"),
        help="report on this log file instead of this entry point's "
             "own — a rotated backup (<app>-<project>.log.1) or a "
             "copy; implies --cost",
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
    "--cost": Option(
        aliases=("-c", "--cost"),
        takes_value=False,
        kwargs=dict(action="store_true"),
        help="print per-session cost totals from the mirror log and "
             "exit (no loop is run)",
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
    # Wrapper-only, all three: see --grow-kit-periodically above.
    "--parallel": Option(aliases=("-p", "--parallel"), takes_value=False),
    "--grow-kit": Option(aliases=("--grow-kit",), takes_value=False),
    "--random": Option(aliases=("--random",), takes_value=False),
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
        "--cost",
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
        "--git-push",
        "--project-dir",
        "--ignore-usage",
        # Accepted here too: the flag is documented as a general one, and a
        # periodic run hands these args to the sequential loop (which honours
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

    `extra_options` is handed the finished parser and is how a wrapper documents
    the flags IT consumes before this parser ever runs (a mode switch that
    decides WHICH parser to use cannot be one of that parser's options). It runs
    last so a wrapper's flags print after the family's.
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
