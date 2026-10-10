"""`modeswitch`: one table of a wrapper's mode switches, the registration that
puts the switches into a driver's parser, and `parse`, the dispatch — the scan
that picks a parser, the parse, and the check that the two readings agree.

The pins dispatch a real argv through `modeswitch.parse` with two real drivers
whose `add_cli_options` registers the table, as a wrapper does; the scan and
the registration are also pinned on their own.
"""

import argparse

import pytest

from llm_loop import ListFileDriver, clispec, cyclecore, modeswitch, parallel
from llm_loop.modeswitch import ModeSwitch

SEQ, PAR = clispec.SEQUENTIAL, clispec.PARALLEL


def _folder(text):
    if not text.strip():
        raise argparse.ArgumentTypeError("needs a folder")
    return text


SWITCHES = (
    ModeSwitch(("-p", "--parallel"), "parallel", False, "run workers",
               listed_in=frozenset({SEQ})),
    ModeSwitch(("-j", "--jobs"), "jobs", True, "worker count",
               listed_in=frozenset({SEQ}), implies="parallel", metavar="N",
               type=int),
    ModeSwitch(("--random",), "random", False, "random order",
               listed_in=frozenset({SEQ, PAR})),
    ModeSwitch(("--finish",), "finish", True, "one folder",
               listed_in=frozenset({SEQ}), metavar="FOLDER", type=_folder),
)


class _PinDriver(ListFileDriver):
    prog = "runPin.py"
    list_file = "queue.md"

    @classmethod
    def add_cli_options(cls, parser):
        modeswitch.register(parser, SWITCHES, SEQ)


class _PinParallelDriver(_PinDriver):
    prog = "runPinParallel.py"

    @classmethod
    def add_cli_options(cls, parser):
        modeswitch.register(parser, SWITCHES, PAR)


def _choose(scanned):
    if scanned.parallel:
        return modeswitch.Choice(_PinParallelDriver, parallel.parse_args)
    return modeswitch.Choice(_PinDriver, cyclecore.parse_args)


def _dispatch(argv):
    """A wrapper's dispatch: `modeswitch.parse` with the two pin drivers."""
    _choice, scanned, args = modeswitch.parse(argv, SWITCHES, _choose)
    return scanned, args


def _refused(argv, capsys) -> str:
    with pytest.raises(SystemExit) as exc:
        _dispatch(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


# --- the scan -------------------------------------------------------------------

@pytest.mark.parametrize("argv, parallel, jobs_seen", [
    ([], False, False),
    (["-p"], True, False),
    (["--parallel"], True, False),
    (["-j", "3"], True, True),
    (["-j3"], True, True),
    (["-j=3"], True, True),
    (["--jobs=3"], True, True),
    (["--jobs", "3"], True, True),
    (["--", "-p"], False, False),
    # Short flags combined into one token, with an engine flag (clispec).
    (["-dp"], True, False),
    (["-pd"], True, False),
    (["-dj3"], True, True),
    # A line argparse refuses reads as no switch; the parser it picks refuses it.
    (["-p", "--random=x"], False, False),
])
def test_the_scan_reads_every_argparse_spelling_and_stops_at_dashdash(
        argv, parallel, jobs_seen):
    scanned = modeswitch.scan(argv, SWITCHES)

    assert scanned.parallel is parallel
    assert ("jobs" in scanned.seen) is jobs_seen


def test_the_scan_leaves_argv_intact():
    argv = ["-m", "1", "-p", "--finish", "x"]
    modeswitch.scan(argv, SWITCHES)

    assert argv == ["-m", "1", "-p", "--finish", "x"]


def test_a_dash_led_token_is_not_a_value():
    # As argparse: `--finish -p` gives --finish no value and -p its own reading.
    scanned = modeswitch.scan(["--finish", "-p"], SWITCHES)

    assert scanned.finish is None and "finish" in scanned.seen
    assert scanned.parallel is True


def _argparse_reads(value):
    """What a plain argparse `--finish` option makes of `value` on THIS Python:
    the value, or None when argparse takes it for an option instead."""
    parser = argparse.ArgumentParser(prog="plain")
    parser.add_argument("--finish")
    try:
        return parser.parse_args(["--finish", value]).finish
    except SystemExit:
        return None


# Dash-led tokens that are values to argparse: `-` and a token with a space on
# every version, `-1x` from 3.14 (argparse's negative-number test widened). A
# hand-written copy of argparse's test missed all three.
@pytest.mark.parametrize("value", ["-", "-x y", "-1x", "-5", "-.5"])
def test_the_scan_takes_a_value_exactly_when_argparse_does(value, capsys):
    expected = _argparse_reads(value)
    capsys.readouterr()

    assert modeswitch.scan(["--finish", value], SWITCHES).finish == expected
    if expected is None:
        assert "argument --finish: expected one argument" in _refused(
            ["--finish", value], capsys)
    else:
        assert _dispatch(["--finish", value])[1].finish == expected


# --- scan -> parse -> refuse ----------------------------------------------------

@pytest.mark.parametrize("argv, driver, runner", [
    ([], _PinDriver, cyclecore.parse_args),
    (["-j", "2"], _PinParallelDriver, parallel.parse_args),
])
def test_parse_returns_the_choice_that_parsed_the_line(argv, driver, runner):
    choice, scanned, args = modeswitch.parse(argv, SWITCHES, _choose)

    assert choice == (driver, runner)
    assert scanned.parallel is (driver is _PinParallelDriver)
    assert hasattr(args, "raw") is (driver is _PinDriver)  # sequential-only


def test_the_chosen_driver_labels_the_usage(capsys):
    # Its prog, not the other driver's: the parser is built from the choice.
    err = _refused(["-p", "--no-such-option"], capsys)

    assert err.startswith("usage: runPinParallel.py")


def test_no_argv_is_the_process_command_line(monkeypatch):
    monkeypatch.setattr("sys.argv", ["runPin.py", "--finish", "f"])

    _choice, scanned, args = modeswitch.parse(None, SWITCHES, _choose)

    assert scanned.finish == "f" and args.finish == "f"


@pytest.mark.parametrize("argv", [
    ["-p", "-m", "2"], ["-j", "2"], ["--random", "--dry-run"],
    ["--finish", "a/b", "-m", "1"], ["--finish=a/b"],
])
def test_a_line_the_scan_and_the_parser_agree_on_runs(argv):
    scanned, args = _dispatch(argv)             # no SystemExit

    assert args.finish == scanned.finish
    assert args.random is scanned.random


@pytest.mark.parametrize("flag", ["-m", "-C", "--git-push"])
@pytest.mark.parametrize("switch", [["-p"], ["--random"], ["--finish", "f"]])
def test_a_value_flag_never_takes_a_switch_as_its_value(flag, switch, capsys):
    # `-m -p 3` used to reach the parser as `-m 3` once -p was stripped.
    err = _refused([flag, *switch, "3"], capsys)

    assert "expected one argument" in err


@pytest.mark.parametrize("argv, named", [
    (["--rand"], "--random"),
    (["--para"], "--parallel"),
    (["--job", "2"], "--jobs"),
    (["--fin", "f"], "--finish"),
])
def test_an_abbreviated_switch_is_refused_by_name(argv, named, capsys):
    err = _refused(argv, capsys)

    assert f"error: {named} was read differently" in err


# Occurrences are compared, not spellings: an abbreviation of a switch given in
# full elsewhere on the line ends as the full spelling would.
@pytest.mark.parametrize("argv", [
    ["--finish", "a", "--finish", "b"], ["--finish=a", "--finish=a"],
    ["--finish=a", "--fin=a"], ["--fin=a", "--finish", "a"],
])
def test_a_value_switch_given_twice_is_refused_however_spelled(argv, capsys):
    assert "error: --finish may be given only once" in _refused(argv, capsys)


@pytest.mark.parametrize("argv", [["-p", "-p"], ["-p", "--parallel"],
                                  ["-p", "--para"], ["--parallel", "--par"]])
def test_a_boolean_switch_given_twice_is_given(argv):
    scanned, args = _dispatch(argv)

    assert scanned.parallel is True and args.parallel is True


# -j in the parallel parser is that parser's own option: last value wins, as in
# every `main_parallel` host, abbreviated or not.
@pytest.mark.parametrize("argv", [["-j", "2", "-j", "3"], ["-j2", "--job", "3"],
                                  ["-p", "-j", "2", "--jobs=3"],
                                  ["-p", "--jo", "3"]])
def test_the_parallel_parsers_own_option_reads_as_it_does_there(argv):
    scanned, args = _dispatch(argv)

    assert scanned.parallel is True and args.jobs == 3


@pytest.mark.parametrize("argv, jobs", [(["-dp"], None), (["-pd"], None),
                                        (["-dj3"], 3)])
def test_combined_short_flags_run_as_argparse_reads_them(argv, jobs):
    scanned, args = _dispatch(argv)

    assert scanned.parallel is True
    assert args.dry_run is True and args.jobs == jobs


@pytest.mark.parametrize("argv", [["--finish="], ["--finish", " "]])
def test_a_switch_type_refuses_at_parse_time(argv, capsys):
    assert "argument --finish: needs a folder" in _refused(argv, capsys)


# --- registration ---------------------------------------------------------------

def _help(mode) -> str:
    parser = clispec.build_parser(
        mode, prog="runPin.py",
        extra_options=lambda p: modeswitch.register(p, SWITCHES, mode))
    return parser.format_help()


def test_help_lists_exactly_the_switches_of_its_mode():
    seq, par = _help(SEQ), _help(PAR)

    for spelling in ("--parallel", "--random", "--finish FOLDER", "--jobs N"):
        assert spelling in seq
    assert "--random" in par
    assert "[--finish" not in par and "[-p]" not in par


def test_an_unlisted_switch_is_still_parsed():
    parser = clispec.build_parser(
        PAR, prog="runPin.py",
        extra_options=lambda p: modeswitch.register(p, SWITCHES, PAR))

    assert parser.parse_args(["-p", "--finish", "f"]).finish == "f"


def test_a_switch_the_parser_already_offers_is_left_to_it():
    # The parallel parser's own -j/--jobs: registering it again would be an
    # argparse conflict, and its dest is the switch's.
    parser = clispec.build_parser(
        PAR, prog="runPin.py",
        extra_options=lambda p: modeswitch.register(p, SWITCHES, PAR))

    assert parser.parse_args(["-j", "4"]).jobs == 4


def test_a_half_owned_switch_is_a_conflict():
    clash = (ModeSwitch(("-j", "--workers"), "jobs", True, "x"),)

    with pytest.raises(ValueError, match="--workers"):
        clispec.build_parser(
            PAR, prog="runPin.py",
            extra_options=lambda p: modeswitch.register(p, clash, PAR))


def test_the_switches_survive_a_rebuilt_command_line():
    for mode in (SEQ, PAR):
        parser = clispec.build_parser(
            mode, prog="runPin.py",
            extra_options=lambda p, m=mode: modeswitch.register(p, SWITCHES, m))
        assert clispec.unstrippable_flags(parser) == []


@pytest.mark.parametrize("bad, match", [
    ((ModeSwitch(("-a",), "x", False, ""), ModeSwitch(("-b",), "x", False, "")),
     "repeat"),
    ((ModeSwitch(("-a",), "seen", False, ""),), "shadows"),
    ((ModeSwitch(("-a",), "a", True, "", implies="b"),), "implies"),
])
def test_a_malformed_table_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        modeswitch.scan([], bad)
