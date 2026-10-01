"""The usage pair both runners open: one source, one policy, two snapshots.

`runlifecycle.RunUsage` is the unit the closing snapshot relies on — the source
and the policy set together — and `open_usage` is the one place that sets them.
Before it, each runner assembled the pair in its own words and the invariant had
no guard; these pins are that guard.
"""

import ast
import inspect

import pytest

from llm_loop import cyclecore, limits, operator, parallel, runlifecycle
from llm_loop.stopchannel import RunStopReason

from _runfixtures import (MemListDriver, OneShotDriver, StubPolicy, StubSource,
                          isolated_run, par_args, record_exit_pushes, seq_args)


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    with isolated_run(monkeypatch, tmp_path):
        yield


@pytest.fixture
def exit_pushes(monkeypatch):
    return record_exit_pushes(monkeypatch)


class _Driver:
    def __init__(self, policy=None):
        self.limit_policy = policy


@pytest.mark.parametrize("source, policy", [(None, object()), (object(), None)])
def test_a_usage_pair_refuses_a_missing_half(source, policy):
    with pytest.raises(ValueError, match="both a source and a policy"):
        runlifecycle.RunUsage(source, policy, "claude")


def test_no_usage_endpoint_opens_nothing(monkeypatch):
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: None)
    policy = StubPolicy()

    assert runlifecycle.open_usage(_Driver(policy), "claude",
                                   dry_run=False) is None
    assert policy.snapshots == []


def test_opening_pairs_the_drivers_policy_and_logs_the_start(monkeypatch):
    source = object()
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: source)
    policy = StubPolicy()

    usage = runlifecycle.open_usage(_Driver(policy), "claude", name="parallel",
                                    dry_run=False)

    assert (usage.source, usage.policy) == (source, policy)
    assert policy.logged == [(source, "at start (parallel)", True)]


def test_opening_falls_back_to_the_providers_default_policy(monkeypatch):
    source = object()
    default = StubPolicy()
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: source)
    monkeypatch.setattr(limits, "default_policy",
                        lambda provider: default if provider == "codex" else None)

    usage = runlifecycle.open_usage(_Driver(), "codex", dry_run=True)

    assert (usage.policy, usage.name) == (default, "codex")
    assert default.snapshots == [], "a dry run is not a run: no opening snapshot"


def test_the_closing_snapshot_answers_the_opening_one():
    source, policy = object(), StubPolicy()
    usage = runlifecycle.RunUsage(source, policy, "claude")

    usage.open()
    usage.close()
    usage.close("interrupted")

    assert policy.logged == [
        (source, "at start (claude)", True),
        # Fresh, not cached: the closing figures are the post-run state.
        (source, "at end (claude)", False),
        # The ending beside the name, not over it: closed per account, a bare
        # `at end (interrupted)` twice would not say whose figures are whose.
        (source, "at end (claude: interrupted)", False),
    ]


def test_closing_a_run_answers_every_usage_it_opened(exit_pushes):
    """`close_run` closes each usage handed to it, not only the last one.

    A mixed-provider sequential run opens one usage per account it selects;
    closing only the one it ended on left the others' `at start (…)` lines
    unanswered in the log. A None is an account without a usage endpoint and
    closes nothing.
    """
    policy = StubPolicy()
    claude = runlifecycle.RunUsage("claude-source", policy, "claude")
    codex = runlifecycle.RunUsage("codex-source", policy, "codex")
    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=False, progress=None,
        settings=runlifecycle.RunSettings(), registry=None,
        status_enabled=False)

    runlifecycle.close_run(ctx, usages=[claude, None, codex],
                           ending="interrupted")

    assert [(source, label) for source, label, _fresh in policy.logged] == [
        ("claude-source", "at end (claude: interrupted)"),
        ("codex-source", "at end (codex: interrupted)"),
    ]
    assert len(exit_pushes) == 1, "the housekeeping around the snapshots ran"


def test_one_failing_close_costs_only_its_own_line(exit_pushes, capsys):
    """A snapshot that raises must not skip the accounts after it, nor the notes.

    The closes run one after another, so an unguarded one that raised took every
    later account's `at end` line and `report_undelivered_notes` with it.
    """
    class _BrokenSource:
        def get_usage(self, cache_value=True):
            raise OSError("usage endpoint unreachable")

    policy = StubPolicy()
    broken = runlifecycle.RunUsage(_BrokenSource(), limits.default_policy("claude"),
                                   "claude")
    codex = runlifecycle.RunUsage("codex-source", policy, "codex")
    mailbox = operator.Mailbox()
    mailbox.submit("look at the third file")
    ctx = runlifecycle.RunContext(
        provider="claude", spec=None, dry_run=False, progress=None,
        settings=runlifecycle.RunSettings(), registry=None,
        status_enabled=False)

    runlifecycle.close_run(ctx, usages=[broken, codex], mailbox=mailbox)

    assert policy.snapshots == ["at end (codex)"]
    out = capsys.readouterr().out
    assert "usage at end (claude) could not be read" in out
    assert "usage endpoint unreachable" in out
    assert "look at the third file" in out, "the undelivered note was skipped"


def test_a_sequential_run_that_returns_closes_the_usage_it_opened(
        tmp_path, monkeypatch, exit_pushes):
    """The normal ending is the common one; the abnormal three are pinned in
    `test_abnormal_exit_epilogue`, this is the door that returns a RunResult."""
    monkeypatch.setattr(cyclecore, "run_claude_streaming", lambda *a, **k: 0)
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    # One command, then no more work: the ending a sequential run RETURNS from.
    driver = OneShotDriver()

    result = cyclecore.run_loop(driver, seq_args(tmp_path, no_statusline=True),
                                app_name="pytest-run-usage", wait_on_start=False)

    assert result.reason is RunStopReason.NO_WORK
    assert driver.limit_policy.snapshots == ["at start (claude)",
                                             "at end (claude)"]


def test_a_parallel_run_that_returns_closes_the_usage_it_opened(
        tmp_path, monkeypatch, exit_pushes):
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda p: StubSource())
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, None, None))
    driver = MemListDriver(["products/only.md"])

    # `ignore_usage` off, or the run opens no usage and there is no pair to close.
    args = par_args(tmp_path, jobs=1, ignore_usage=False, no_statusline=True)
    result = parallel.run_parallel(driver, args, app_name="pytest-run-usage",
                                   wait_on_start=False)

    assert result.reason is RunStopReason.NO_WORK
    # The account in the name, as a sequential run's lines have it: `parallel`
    # alone would not say whose figures the pair holds.
    assert driver.limit_policy.snapshots == ["at start (parallel claude)",
                                             "at end (parallel claude)"]


@pytest.mark.parametrize("runner", [cyclecore.run_loop, parallel.run_parallel])
def test_both_runners_open_usage_through_the_shared_path(runner):
    """Checked on the SOURCE: a runner that built its own pair again would still
    pass every behavioural pin whose stub source it happened to receive."""
    # The sequential public entry owns cleanup; its body opens usage.
    body = cyclecore._run_loop if runner is cyclecore.run_loop else runner
    tree = ast.parse(inspect.getsource(body))
    names = {node.attr if isinstance(node, ast.Attribute) else node.id
             for node in ast.walk(tree)
             if isinstance(node, (ast.Attribute, ast.Name))}

    assert "open_usage" in names
    assert not names & {"usage_source_for", "default_policy", "log_snapshot"}, (
        "the runner assembles the usage pair itself again")
