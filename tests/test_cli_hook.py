"""Tests for the wrapper's seam into the shared --help.

A wrapper mode switch is scanned off argv to choose a parser, so nothing lists
it unless `Driver.add_cli_options` puts it there — and a flag that `--help` does
not mention is a flag its user concludes does not exist. These pin that the
hook reaches both entry points; what a registered switch then does (parsed with
the rest of the line, a disagreeing scan refused) is test_modeswitch.py's.
"""

import pytest

from llm_loop import ListFileDriver, ModeSwitch, cyclecore, modeswitch, parallel


MODE_FLAG = "--grow-kit"
SWITCHES = (ModeSwitch((MODE_FLAG,), "grow_kit", False,
                       "a mode this wrapper scans for",
                       listed_in=frozenset({"any"})),)


def add_mode(parser):
    modeswitch.register(parser, SWITCHES, "any")


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


@pytest.mark.parametrize("entry_point", ["main", "main_parallel"])
def test_an_entry_point_that_never_reads_a_switch_refuses_one(
        entry_point, monkeypatch, capsys):
    # A host that registers its table but dispatches through the engine's
    # own entry point would run the default mode behind the switch.
    def run(*_args, **_kwargs):
        pytest.fail("a runner started behind an unread mode switch")
    monkeypatch.setattr(cyclecore, "run_loop", run)
    monkeypatch.setattr(parallel, "run_parallel", run)

    with pytest.raises(SystemExit) as exit_info:
        getattr(HookedDriver, entry_point)([MODE_FLAG])

    assert exit_info.value.code == 2
    assert (f"error: {MODE_FLAG} is a mode switch this entry point never "
            f"reads" in capsys.readouterr().err)


@pytest.mark.parametrize("parse", [cyclecore.parse_args, parallel.parse_args])
def test_a_registered_switch_is_parsed_with_the_line(parse):
    # The switch is no longer taken out of argv first: the parser the scan
    # picked reads it, so it lands in the namespace under its own dest.
    assert parse([MODE_FLAG], extra_options=add_mode).grow_kit is True
    assert parse([], extra_options=add_mode).grow_kit is False
