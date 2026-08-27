"""Tests for the wrapper's seam into the shared --help.

A wrapper mode switch is read out of argv before either parser exists, so
nothing lists it unless `Driver.add_cli_options` puts it there — and a flag that
`--help` does not mention is a flag its user concludes does not exist. These
pin the two halves: the hook reaches both entry points, and the documenting
action refuses to be a silent no-op.
"""

import pytest

from llm_loop import ConsumedByWrapperAction, ListFileDriver, cyclecore, parallel


MODE_FLAG = "--grow-kit"


def add_mode(parser):
    parser.add_argument(MODE_FLAG, action=ConsumedByWrapperAction,
                        help="a mode this wrapper reads itself")


class HookedDriver(ListFileDriver):
    prog = "runHooked.py"
    list_file = "queue.md"

    @classmethod
    def add_cli_options(cls, parser):
        add_mode(parser)


def _help_of(main, capsys) -> str:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    return capsys.readouterr().out


# --- the hook reaches both entry points ----------------------------------------

@pytest.mark.parametrize("parse", [cyclecore.parse_args, parallel.parse_args])
def test_both_parsers_offer_the_same_seam(parse, capsys):
    # Same wrapper, same flag, either mode: a switch documented in only one of
    # the two --helps is documented by accident.
    with pytest.raises(SystemExit):
        parse(["--help"], extra_options=add_mode)

    assert MODE_FLAG in capsys.readouterr().out


@pytest.mark.parametrize("entry_point", ["main", "main_parallel"])
def test_a_driver_carries_its_options_into_help(entry_point, capsys):
    assert MODE_FLAG in _help_of(getattr(HookedDriver, entry_point), capsys)


def test_a_driver_without_the_hook_gets_the_help_it_always_had(capsys):
    class PlainDriver(ListFileDriver):
        prog = "runPlain.py"
        list_file = "queue.md"

    assert MODE_FLAG not in _help_of(PlainDriver.main, capsys)


# --- the documenting action ----------------------------------------------------

def test_a_documented_switch_stays_out_of_the_parsed_namespace():
    # The wrapper acted on it long before this parser ran; leaving a stale copy
    # in the namespace invites a second, disagreeing reader of the same flag.
    args = cyclecore.parse_args([], extra_options=add_mode)

    assert not hasattr(args, "grow_kit")


def test_reaching_the_parser_is_an_error_naming_the_option(capsys):
    # argparse resolves `--grow-k`; the wrapper's argv scan does not. Accepted
    # here, it would run the DEFAULT mode while looking like it worked.
    with pytest.raises(SystemExit) as exit_info:
        cyclecore.parse_args(["--grow-k"], extra_options=add_mode)

    assert exit_info.value.code == 2
    assert MODE_FLAG in capsys.readouterr().err


def test_a_value_taking_switch_shows_its_argument(capsys):
    def add_batched(parser):
        parser.add_argument("--batches-of", action=ConsumedByWrapperAction,
                            nargs=1, metavar="N", help="batches of N")

    with pytest.raises(SystemExit):
        cyclecore.parse_args(["--help"], extra_options=add_batched)

    assert "--batches-of N" in capsys.readouterr().out
