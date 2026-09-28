"""The `type=` validators of `clispec`, pinned from a real argv in both modes.

A value-taking flag reaches the engine in two spellings for one input under
PowerShell 5.1: `--flag=` delivers "", while `--flag ""` loses its `""` and
arrives as a bare flag that argparse refuses. The validators make the first a
usage error too, so both spellings end the same way — exit 2 from the parser,
before anything runs — in every host, not only in a wrapper that checks for
itself.
"""

import pytest

from llm_loop import cyclecore, parallel, projectroot

PARSERS = [cyclecore.parse_args, parallel.parse_args]
PARSER_IDS = ["sequential", "parallel"]

_ROOT_REFUSAL = "argument -C/--project-dir: needs a directory"


def _refusal(parse, argv, capsys) -> str:
    """Parse `argv`, which must be refused with exit 2; return stderr."""
    with pytest.raises(SystemExit) as exc:
        parse(argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


# --- -C / --project-dir ---------------------------------------------------------

@pytest.mark.parametrize("parse", PARSERS, ids=PARSER_IDS)
@pytest.mark.parametrize("argv", [
    ["-C", ""], ["--project-dir", ""], ["--project-dir="], ["-C="],
    ["--project-dir", "   "], ["--project-dir=  "],
], ids=["short-pair", "long-pair", "long-equals", "short-equals",
        "long-blank", "equals-blank"])
def test_an_empty_project_dir_is_a_usage_error(parse, argv, capsys):
    # Matched whole: the usage line every argparse error prints already names
    # `-C DIR`, so the bare flag would hold for a refusal of anything else.
    assert _ROOT_REFUSAL in _refusal(parse, argv, capsys)


@pytest.mark.parametrize("parse", PARSERS, ids=PARSER_IDS)
def test_the_bare_flag_powershell_leaves_is_refused_too(parse, capsys):
    # The other spelling of the same input: `--project-dir ""` under PS 5.1.
    err = _refusal(parse, ["--project-dir"], capsys)

    assert "argument -C/--project-dir: expected one argument" in err


@pytest.mark.parametrize("parse", PARSERS, ids=PARSER_IDS)
def test_a_directory_is_passed_through_unchanged(parse, tmp_path):
    # Returned as typed: resolving it is `set_project_root`'s job.
    assert parse(["-C", str(tmp_path)]).project_dir == str(tmp_path)
    assert parse([]).project_dir is None


@pytest.mark.parametrize("path", ["", "  "])
def test_set_project_root_refuses_a_blank_path(path):
    # A namespace built past the parser must not turn "" into "keep the root".
    before = projectroot.project_dir()

    with pytest.raises(ValueError, match="--project-dir needs a directory"):
        projectroot.set_project_root(path)

    assert projectroot.project_dir() == before


def test_set_project_root_keeps_the_root_on_none():
    before = projectroot.project_dir()

    assert projectroot.set_project_root(None) == before


# --- -s / --start-in (sequential only: the parallel parser has no delay) --------

_START_IN_REFUSAL = "argument -s/--start-in/--startIn: "


@pytest.mark.parametrize("argv, why", [
    (["-s", "bogus"], "cannot parse duration: 'bogus'"),
    (["--start-in=soon"], "cannot parse duration: 'soon'"),
    (["--startIn", "h"], "cannot parse duration: 'h'"),
    (["--start-in="], "empty duration"),
    (["--start-in", "  "], "empty duration"),
], ids=["short", "equals", "deprecated-alias", "equals-empty", "blank"])
def test_a_malformed_start_in_is_a_usage_error(argv, why, capsys):
    # At parse time, not when the wait begins: a runner reads the value only
    # after its prologue, and a host wrapper after its own startup steps.
    err = _refusal(cyclecore.parse_args, argv, capsys)

    assert _START_IN_REFUSAL + why in err


@pytest.mark.parametrize("spelling", ["1h30m", "29", "90s", "0"])
def test_a_well_formed_start_in_is_kept_as_typed(spelling):
    # As typed, not in seconds: the wait announces the spelling it was given.
    assert cyclecore.parse_args(["-s", spelling]).start_in == spelling
    assert cyclecore.parse_args([]).start_in is None
