"""Pin that the family's options really are declared in ONE place.

`clispec.OPTIONS` is now the only table: both parsers are built from it and
`cmdline.FLAG_ALIASES` is projected out of it. That makes the two old drifts
impossible by construction — but only for options that go THROUGH the table. An
`add_argument` written straight into a parser would still be a flag no argv
rewriter knows, and that is a silent bug rather than a loud one: `rebuild_argv`
copies an unknown flag through verbatim, so an unknown VALUE-taking flag has its
value read as a separate token, and an override lands next to the stale setting
it was meant to replace. The rendered command line then looks right and is not.

So the gate walks the parsers argparse actually built — `clispec.unstrippable_flags`,
public because every host's own hook needs the same walk — and demands that
every value-taking spelling be one the alias table knows, and every known
spelling have the table's arity. It is the check the previous shape could not
express: while `parse_args` only ever built a parser, fed it argv and let it
`sys.exit`, there was no parser object to walk.
"""

import argparse

import pytest

from llm_loop import clispec, cyclecore, modeswitch, parallel
from llm_loop.cmdline import FLAG_ALIASES, rebuild_argv
from llm_loop.modeswitch import ModeSwitch


MODES = [clispec.SEQUENTIAL, clispec.PARALLEL]

# A host's mode switches, a value-taking one among them: none is a row of
# `clispec.OPTIONS`, and registering them is what declares them to the rewriter.
HOST_SWITCHES = (
    ModeSwitch(("-p", "--parallel"), "parallel", False, "run workers"),
    ModeSwitch(("--random",), "random", False, "random order"),
    ModeSwitch(("--finish",), "finish", True, "one folder", metavar="FOLDER"),
)


def _known_wrapper_options(parser):
    """A wrapper's `Driver.add_cli_options`, standing in for every host that has
    one.

    `build_parser` is a loop over the table, so inside this package the only way
    an undeclared flag can reach a parser is through this hook — which means a
    gate that never passes one is checking the loop, not the seam where a host
    project actually adds options. The hook registers a mode-switch table, as a
    host does: a hook that goes through `modeswitch.register` must leave the
    parser clean.
    """
    modeswitch.register(parser, HOST_SWITCHES, "any")


def _built(mode, *, hooked=False):
    return clispec.build_parser(
        mode, prog="runGate.py",
        extra_options=_known_wrapper_options if hooked else None)


def _real_actions(parser):
    """Every action except argparse's own -h/--help, which no table declares."""
    return [a for a in parser._actions if "--help" not in a.option_strings]


def _hooked(mode, *add_argument_calls):
    """`mode`'s parser with a hook that makes exactly these `add_argument` calls."""
    def hook(parser):
        for args, kwargs in add_argument_calls:
            parser.add_argument(*args, **kwargs)
    return clispec.build_parser(mode, prog="runGate.py", extra_options=hook)


# --- the gate ------------------------------------------------------------------

@pytest.mark.parametrize("hooked", [False, True], ids=["bare", "wrapper-hook"])
@pytest.mark.parametrize("mode", MODES)
def test_every_spelling_a_parser_offers_is_strippable(mode, hooked):
    problems = clispec.unstrippable_flags(_built(mode, hooked=hooked))

    assert problems == [], (
        f"{mode} parser: {problems}. Declare the flag in clispec.OPTIONS, or "
        f"register it as a mode switch, instead of calling add_argument "
        f"directly.")


# --- ...and it bites: each failure the gate exists for, at the hook ------------
# Hosts run `unstrippable_flags` over their own hooks, where nothing in this
# package can see them, so what it must refuse is pinned here once.

@pytest.mark.parametrize("mode", MODES)
def test_an_undeclared_value_taking_flag_is_reported(mode):
    parser = _hooked(mode, (("--rogue",), dict(metavar="X")))

    problems = clispec.unstrippable_flags(parser)

    assert len(problems) == 1 and problems[0].startswith("--rogue "), problems


# A boolean row the mode's parser does not offer, so a hook can add it.
_UNOFFERED_SWITCH = {clispec.SEQUENTIAL: "--ignore-usage",
                     clispec.PARALLEL: "--raw"}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("which,kwargs", [
    # boolean in the table, handed a value by the hook
    ("switch", dict(nargs=1)),
    # value-taking in the table, registered as a switch
    ("value", dict(action="store_true")),
], ids=["switch-given-a-value", "value-made-a-switch"])
def test_a_declared_flag_with_the_wrong_arity_is_reported(mode, which, kwargs):
    spelling = _UNOFFERED_SWITCH[mode] if which == "switch" else "--session-limit"
    problems = clispec.unstrippable_flags(_hooked(mode, ((spelling,), kwargs)))

    assert len(problems) == 1 and problems[0].startswith(f"{spelling}:"), problems


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("spelling,kwargs", [
    # declared value flag whose value became optional: bare, it eats a neighbour
    ("--session-limit", dict(nargs="?")),
    # undeclared, two values: the second one is left behind as an orphan
    ("--rogue", dict(nargs=2)),
], ids=["declared-optional-value", "undeclared-two-values"])
def test_a_flag_with_more_than_one_possible_value_is_reported(mode, spelling, kwargs):
    problems = clispec.unstrippable_flags(_hooked(mode, ((spelling,), kwargs)))

    assert len(problems) == 1 and problems[0].startswith(f"{spelling}: nargs="), problems


@pytest.mark.parametrize("mode", MODES)
def test_an_undeclared_switch_is_left_to_pass_through(mode):
    # The line the gate draws, and the reason it is allowed: the rewriter copies
    # a flag it does not know through verbatim, and a switch has no value that
    # could be misread as the next token.
    parser = _hooked(mode, (("--wrapper-only",), dict(action="store_true")))

    assert clispec.unstrippable_flags(parser) == []
    assert rebuild_argv(["--wrapper-only", "-m", "1"], {"--max-runs": 2}) == [
        "--wrapper-only", "--max-runs", "2"]


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
    """The encoding of "only the status line writes this one".

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


def test_canonical_key_is_among_its_own_aliases():
    for canonical, spec in FLAG_ALIASES.items():
        assert isinstance(spec, clispec.Flag)
        assert canonical in spec.aliases


def test_no_spelling_is_claimed_by_two_flags():
    seen = {}
    for canonical, spec in FLAG_ALIASES.items():
        for alias in spec.aliases:
            assert alias not in seen, f"{alias} in {seen.get(alias)} and {canonical}"
            seen[alias] = canonical


def test_deprecated_aliases_are_known():
    # The spellings the sequential parser still accepts for compatibility. They
    # exist only to be removable from an argv, so nothing else would miss them.
    assert "--max" in FLAG_ALIASES["--max-runs"].aliases
    assert "--startIn" in FLAG_ALIASES["--start-in"].aliases


def test_the_alias_table_is_the_option_table():
    """No row may be dropped on the way to the rewriter, and none reordered.

    The comparison of KEYS is the load-bearing half. A projection that skipped
    the parser-less rows would still satisfy every other test in this file — the
    parsers would look complete — while `--session-limit` and `--weekly-limit`
    silently left the rewriter's vocabulary. The order matters for a different
    reason: overrides are appended in it, so a rendered command line must not
    depend on dict insertion luck.
    """
    assert list(FLAG_ALIASES) == list(clispec.OPTIONS)
    for name, option in clispec.OPTIONS.items():
        assert FLAG_ALIASES[name] == clispec.Flag(option.aliases,
                                                  option.takes_value)


# --- a host's mode switches, projected from its own table -----------------------

@pytest.mark.parametrize("mode", MODES)
def test_a_registered_switch_is_a_row_of_the_hosts_vocabulary(mode):
    parser = _built(mode, hooked=True)
    args = parser.parse_args([])

    for source in (parser, args):
        aliases = clispec.flag_aliases(source)
        assert list(aliases)[:len(FLAG_ALIASES)] == list(FLAG_ALIASES)
        assert aliases["--finish"] == clispec.Flag(("--finish",), True)
        assert aliases["--parallel"] == clispec.Flag(("-p", "--parallel"), False)
    assert clispec.flag_aliases(None) == FLAG_ALIASES


def test_a_switch_that_is_an_engine_row_keeps_that_row():
    # The sequential parser has no -j, so a host's -j/--jobs switch is
    # registered there — and is the very row the parallel parser builds from.
    jobs = (ModeSwitch(("-j", "--jobs"), "jobs", True, "workers", metavar="N",
                       type=int),)
    parser = clispec.build_parser(
        clispec.SEQUENTIAL, prog="runGate.py",
        extra_options=lambda p: modeswitch.register(p, jobs, "any"))

    assert clispec.flag_aliases(parser) == FLAG_ALIASES
    assert clispec.unstrippable_flags(parser) == []


def test_a_switch_the_parser_already_offers_is_a_row_too():
    # `register` adds no option for a switch the parser owns, but the parser
    # reads it all the same, so the rewriter must know its arity: without the
    # row `--finish ""` is copied as two tokens, which PowerShell 5.1 cannot
    # deliver (`quote` refuses it), and the gate reports `--finish`.
    def hook(parser):
        parser.add_argument("--finish")
        modeswitch.register(parser, HOST_SWITCHES, "any")
    parser = clispec.build_parser(clispec.SEQUENTIAL, prog="runGate.py",
                                  extra_options=hook)
    args = parser.parse_args(["--finish", ""])

    assert clispec.flag_aliases(args)["--finish"] == clispec.Flag(
        ("--finish",), True)
    assert clispec.unstrippable_flags(parser) == []
    assert rebuild_argv(["--finish", ""], {},
                        aliases=clispec.flag_aliases(args)) == ["--finish="]


def test_an_option_over_the_switch_record_is_reported():
    # Added after `register` (which refuses one added before it): the gate
    # names it, and a value it parsed in place of the record is refused.
    def hook(parser):
        modeswitch.register(parser, HOST_SWITCHES, "any")
        parser.add_argument("--record", dest=clispec.REGISTERED_SWITCHES)
    parser = clispec.build_parser(clispec.SEQUENTIAL, prog="runGate.py",
                                  extra_options=hook)

    [problem] = clispec.unstrippable_flags(parser)
    assert problem.startswith("--record: dest '_mode_switches'")
    with pytest.raises(ValueError, match="an option uses it as its dest"):
        clispec.flag_aliases(parser.parse_args(["--record", "x"]))


@pytest.mark.parametrize("aliases, takes_value", [
    (("-s", "--soon"), True),           # one spelling of --start-in
    (("--session-limit",), False),      # the row's spelling, the wrong arity
])
def test_a_switch_sharing_an_engine_spelling_is_refused(aliases, takes_value):
    # A bare parser, so registering cannot collide with an engine option; the
    # rewriter strips the ENGINE's spellings whatever the host's parser offers.
    parser = argparse.ArgumentParser(prog="runGate.py")
    modeswitch.register(parser, (ModeSwitch(aliases, "clash", takes_value, "x"),),
                        "any")

    with pytest.raises(ValueError, match="shares a spelling"):
        clispec.flag_aliases(parser)


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
