"""`runlifecycle.begin_run` reads the command-line contract strictly, and the
pins' shared fixtures (`_runfixtures`) are what let it.

The two halves belong together. `begin_run` used to read four attributes through
`getattr` with a default, although `clispec` declares all four for both modes:
no parser could ever leave one out, so each default existed only to cover a
namespace written out by hand and incomplete. The pins built those namespaces
eleven times over; now they take them from the parser, and an incomplete caller
is an error instead of a run with a silently chosen default.
"""

import pytest

from llm_loop import clispec, console, exitlog, projectroot, runlifecycle

from _runfixtures import (NoWorkDriver, par_args, root_named_unlike_cwd,
                          root_not_cwd, seq_args)

# What `begin_run` reads without a fallback.
STRICT = ("provider", "dry_run", "project_dir", "no_live_messages")

BUILDERS = {
    "sequential": seq_args,
    "parallel": lambda project_dir, **fields: par_args(project_dir, jobs=1,
                                                       **fields),
}


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    """The project root put back, and a run that defaulted `dry_run` to False —
    the regression the refusal pins exist for — kept off the real exit record."""
    monkeypatch.setattr(console, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(exitlog, "_record", None)
    previous = projectroot.project_dir()
    yield
    exitlog.finish()
    projectroot.set_project_root(previous)


def _begin(args):
    # A dry run: no script lock, no tee, no exit record — nothing to undo.
    return runlifecycle.begin_run(NoWorkDriver(), args, "pytest-run-args",
                                  setup_logging=False)


@pytest.mark.parametrize("mode", [clispec.SEQUENTIAL, clispec.PARALLEL])
def test_both_parsers_declare_everything_begin_run_reads_strictly(mode):
    parsed = vars(clispec.build_parser(mode, prog="pytest").parse_args([]))
    assert set(STRICT) <= set(parsed), (
        f"the {mode} parser no longer declares {set(STRICT) - set(parsed)}, so "
        f"begin_run would fail on a namespace that parser built")


@pytest.mark.parametrize("runner", sorted(BUILDERS))
def test_a_namespace_from_either_parser_opens_the_run(tmp_path, runner):
    """The control for the refusal below: a complete namespace is accepted, so
    the AttributeError there comes from the attribute that was taken away."""
    ctx = _begin(BUILDERS[runner](tmp_path, dry_run=True))

    assert ctx.dry_run is True
    assert projectroot.project_dir() == str(tmp_path)


@pytest.mark.parametrize("runner", sorted(BUILDERS))
@pytest.mark.parametrize("name", STRICT)
def test_a_namespace_missing_a_contract_attribute_is_refused(tmp_path, runner,
                                                             name):
    args = BUILDERS[runner](tmp_path, dry_run=True)
    delattr(args, name)

    with pytest.raises(AttributeError, match=name):
        _begin(args)


def test_the_builders_refuse_a_field_their_parser_does_not_declare(tmp_path):
    """A misspelt override would set an attribute no code reads and leave the
    pin running on the default it meant to change. The two modes' option sets
    differ, so each builder answers for its own parser."""
    with pytest.raises(TypeError, match="jobs"):
        seq_args(tmp_path, jobs=2)
    with pytest.raises(TypeError, match="cost"):
        par_args(tmp_path, jobs=1, cost=True)


def test_root_not_cwd_refuses_the_directory_the_process_stands_in(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(AssertionError, match="same directory"):
        root_not_cwd(tmp_path)


def test_a_root_can_differ_from_the_cwd_and_still_share_its_name(
        tmp_path, monkeypatch):
    """Why there are two "elsewhere" helpers and not one: the mirror log's file
    name carries only the folder's basename, and a directory that `root_not_cwd`
    accepts can still carry the same basename as the cwd."""
    twin = tmp_path / "elsewhere" / "some-project"
    twin.mkdir(parents=True)
    monkeypatch.chdir(twin)

    root_not_cwd(tmp_path)          # a different directory: accepted
    with pytest.raises(AssertionError, match="share a name"):
        root_named_unlike_cwd(tmp_path)
