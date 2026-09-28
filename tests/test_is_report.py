"""`cyclecore.is_report` and `run_loop` agree on every flag of the sequential
command line.

A host asks `is_report` before its own startup steps — a script lock, a wait on
the stop file, a kit thaw — because a report is how a live run's progress is
read, and those steps would act on that run's shared state. So a flag that
`run_loop` answers as a report while the predicate says "run" is a report that
takes a live run's lock; the opposite is a run that skips its guards. Checked
from a real argv, one option at a time, over every option the parser offers,
so a report flag added to either side alone fails here.
"""

import pytest

from llm_loop import (clispec, costlog, cyclecore, parallel, runlifecycle,
                      stopchannel)

from _runfixtures import NoWorkDriver, isolated_run

# A value each value-taking option accepts; the parser refuses anything else.
_SAMPLE_VALUES = {
    "--max-runs": "1",
    "--start-in": "1",
    "--git-push": "none",
    "--cost-log": "named.log",
}

# What the rule says today. Pinned as a control: agreement alone would also hold
# for a predicate and a runner that both forgot the same flag.
_REPORT_OPTIONS = {"--log", "--cost", "--cost-log"}


class _RunStarted(Exception):
    """`begin_run` was reached: the command line was treated as a run."""


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    with isolated_run(monkeypatch, tmp_path):
        yield


def _argv_for(name, root):
    option = clispec.OPTIONS[name]
    words = [name]
    if option.takes_value:
        words.append(str(root) if name == "--project-dir"
                     else _SAMPLE_VALUES[name])
    # -C first so every case reads its logs and stop file under tmp_path.
    return ["-C", str(root), "--git-push", "none", *words]


def _run_loop_reports(args, monkeypatch) -> bool:
    """Whether `run_loop` answered `args` as a report (and never began a run)."""
    def begin_run(*_args, **_kwargs):
        raise _RunStarted

    monkeypatch.setattr(runlifecycle, "begin_run", begin_run)
    monkeypatch.setattr(costlog, "report_costs", lambda *_a, **_kw: None)
    try:
        result = cyclecore.run_loop(NoWorkDriver(), args,
                                    app_name="pytest-is-report",
                                    wait_on_start=False)
    except _RunStarted:
        return False
    assert result.reason is stopchannel.RunStopReason.NO_WORK
    return True


@pytest.mark.parametrize("name", clispec.OPTION_ORDER[clispec.SEQUENTIAL])
def test_the_predicate_and_run_loop_agree_on_each_option(
        name, tmp_path, monkeypatch):
    args = cyclecore.parse_args(_argv_for(name, tmp_path))

    predicted = cyclecore.is_report(args)

    assert _run_loop_reports(args, monkeypatch) is predicted
    assert predicted is (name in _REPORT_OPTIONS)


def test_a_line_with_no_report_flag_is_a_run(tmp_path, monkeypatch):
    args = cyclecore.parse_args(["-C", str(tmp_path), "--git-push", "none"])

    assert cyclecore.is_report(args) is False
    assert _run_loop_reports(args, monkeypatch) is False


def test_a_parallel_namespace_is_never_a_report(tmp_path):
    # The parallel parser declares none of the report flags; the predicate
    # reads them with defaults so a host can ask it of either namespace.
    assert cyclecore.is_report(parallel.parse_args(["-C", str(tmp_path)])) is False
