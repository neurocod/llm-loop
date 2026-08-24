"""Pin that the family's options really are declared in ONE place.

`clispec.OPTIONS` is now the only table: both parsers are built from it and
`cmdline.FLAG_ALIASES` is projected out of it. That makes the two old drifts
impossible by construction — but only for options that go THROUGH the table. An
`add_argument` written straight into a parser would still be a flag no argv
rewriter knows, and that is a silent bug rather than a loud one: `rebuild_argv`
copies an unknown flag through verbatim, so an unknown VALUE-taking flag has its
value read as a separate token, and an override lands next to the stale setting
it was meant to replace. The rendered command line then looks right and is not.

So the gate walks the parsers argparse actually built and demands every option
string be a spelling the alias table knows, with the same arity. It is the check
the previous shape could not express: while `parse_args` only ever built a
parser, fed it argv and let it `sys.exit`, there was no parser object to walk.
"""

import argparse

import pytest

from llm_loop import clispec, cyclecore, parallel
from llm_loop.cmdline import FLAG_ALIASES


MODES = [clispec.SEQUENTIAL, clispec.PARALLEL]


def _known_wrapper_options(parser):
    """A wrapper's `Driver.add_cli_options`, standing in for every host that has
    one.

    `build_parser` is a loop over the table, so inside this package the only way
    an undeclared flag can reach a parser is through this hook — which means a
    gate that never passes one is checking the loop, not the seam where a host
    project actually adds options. The flags used here are real ones a wrapper
    registers (runGenerateModels does, for both), and they belong to the table:
    a hook that stays inside it must leave the parser clean.
    """
    parser.add_argument("-p", "--parallel", action="store_true")
    parser.add_argument("--random", action="store_true")


def _built(mode, *, hooked=False):
    return clispec.build_parser(
        mode, prog="runGate.py",
        extra_options=_known_wrapper_options if hooked else None)


def _real_actions(parser):
    """Every action except argparse's own -h/--help, which no table declares."""
    return [a for a in parser._actions if "--help" not in a.option_strings]


def _spelling_owner():
    """Which canonical flag each spelling belongs to."""
    return {alias: canonical
            for canonical, spec in FLAG_ALIASES.items()
            for alias in spec.aliases}


# --- the gate ------------------------------------------------------------------

@pytest.mark.parametrize("hooked", [False, True], ids=["bare", "wrapper-hook"])
@pytest.mark.parametrize("mode", MODES)
def test_every_spelling_a_parser_offers_is_strippable(mode, hooked):
    owner = _spelling_owner()
    unknown = [opt
               for action in _real_actions(_built(mode, hooked=hooked))
               for opt in action.option_strings
               if opt not in owner]

    assert unknown == [], (
        f"{mode} parser offers flags no argv rewriter knows: {unknown}. Declare "
        f"them in clispec.OPTIONS instead of calling add_argument directly.")


@pytest.mark.parametrize("hooked", [False, True], ids=["bare", "wrapper-hook"])
@pytest.mark.parametrize("mode", MODES)
def test_the_table_and_argparse_agree_on_arity(mode, hooked):
    # `takes_value` decides whether the NEXT token is this flag's value. Wrong,
    # and removing the flag either eats a neighbouring token or leaves an orphan
    # value on the line.
    owner = _spelling_owner()
    for action in _real_actions(_built(mode, hooked=hooked)):
        canonical = owner.get(action.option_strings[0])
        if canonical is None:
            continue            # the test above owns "the table has never heard of it"
        assert FLAG_ALIASES[canonical].takes_value == (action.nargs != 0), (
            f"{canonical}: table says takes_value="
            f"{FLAG_ALIASES[canonical].takes_value}, argparse built nargs="
            f"{action.nargs!r}")


@pytest.mark.parametrize("mode", MODES)
def test_a_parser_offers_every_spelling_its_row_declares(mode):
    # The other direction: a row may not promise a spelling the parser does not
    # accept, or `rebuild_argv` would strip a flag that a relaunch then rejects.
    # What it pins is one expression — that `build_parser` splats the WHOLE
    # alias tuple. Narrow, and worth saying so: handing argparse `aliases[0]`
    # instead is a one-character edit that keeps every --help line looking right
    # while quietly retiring `--max` and `--startIn`.
    offered = {opt
               for action in _real_actions(_built(mode))
               for opt in action.option_strings}
    for name in clispec.OPTION_ORDER[mode]:
        missing = [a for a in clispec.OPTIONS[name].aliases if a not in offered]
        assert missing == [], f"{mode} parser is missing {missing} of {name}"


# --- the table's own shape ------------------------------------------------------

@pytest.mark.parametrize("mode", MODES)
def test_each_mode_names_only_rows_that_can_build_an_option(mode):
    for name in clispec.OPTION_ORDER[mode]:
        assert name in clispec.OPTIONS, f"{mode} names {name}, which is not a row"
        assert clispec.OPTIONS[name].kwargs is not None, (
            f"{name} is offered by {mode} but declares no add_argument keywords")


def test_a_row_no_parser_offers_declares_no_parser_keywords():
    """The encoding of "the wrapper reads this one itself".

    Parser keywords on a row nobody builds are the residue of a half-finished
    wiring — the flag reads as supported and is not.
    """
    offered = set(clispec.OPTION_ORDER[clispec.SEQUENTIAL])
    offered |= set(clispec.OPTION_ORDER[clispec.PARALLEL])
    for name, option in clispec.OPTIONS.items():
        if name in offered:
            continue
        assert option.kwargs is None, (
            f"{name} carries add_argument keywords but no parser offers it")
        assert option.help == "" and option.parallel_help is None, (
            f"{name} carries help text no --help can print")


def test_a_parallel_help_override_belongs_to_a_shared_option():
    """A second help text is only meaningful where both modes offer the flag."""
    both = (set(clispec.OPTION_ORDER[clispec.SEQUENTIAL])
            & set(clispec.OPTION_ORDER[clispec.PARALLEL]))
    stray = [name for name, option in clispec.OPTIONS.items()
             if option.parallel_help is not None and name not in both]

    assert stray == [], f"parallel_help on options the parallel parser is alone in: {stray}"


def test_the_alias_table_is_the_option_table():
    """No row may be dropped on the way to the rewriter, and none reordered.

    The comparison of KEYS is the load-bearing half. A projection that skipped
    the parser-less rows would still satisfy every other test in this file — the
    parsers would look complete — while `--parallel`, `--grow-kit`, `--random`
    and `--finish` silently left the rewriter's vocabulary. The order matters
    for a different reason: overrides are appended in it, so a rendered command
    line must not depend on dict insertion luck.
    """
    assert list(FLAG_ALIASES) == list(clispec.OPTIONS)
    for name, option in clispec.OPTIONS.items():
        assert FLAG_ALIASES[name] == clispec.Flag(option.aliases,
                                                  option.takes_value)


# --- the two entry points still reach the same table ---------------------------

@pytest.mark.parametrize("parse,mode", [(cyclecore.parse_args, clispec.SEQUENTIAL),
                                        (parallel.parse_args, clispec.PARALLEL)])
def test_each_entry_point_parses_what_its_mode_declares(parse, mode):
    # The public signatures are unchanged; what changed is where the options
    # come from. Parsing every spelling of every row is the cheapest proof that
    # this entry point is wired to that mode's list and not the other's.
    for name in clispec.OPTION_ORDER[mode]:
        option = clispec.OPTIONS[name]
        for alias in option.aliases:
            argv = [alias, "1"] if option.takes_value else [alias]
            if name == "--git-push":
                argv = [alias, "none"]
            parse(argv)             # argparse exits 2 on an unknown option


@pytest.mark.parametrize("mode", MODES)
def test_a_wrapper_hook_still_lands_on_the_built_parser(mode):
    seen = []

    def hook(parser):
        assert isinstance(parser, argparse.ArgumentParser)
        seen.append(parser.prog)

    clispec.build_parser(mode, prog="runGate.py", extra_options=hook)

    assert seen == ["runGate.py"]
