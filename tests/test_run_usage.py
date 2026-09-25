"""The usage pair both runners open: one source, one policy, two snapshots.

`runlifecycle.RunUsage` is the unit the closing snapshot relies on — the source
and the policy set together — and `open_usage` is the one place that sets them.
Before it, each runner assembled the pair in its own words and the invariant had
no guard; these pins are that guard.
"""

import ast
import inspect

import pytest

from llm_loop import cyclecore, limits, parallel, runlifecycle


class _RecordingPolicy:
    def __init__(self):
        self.snapshots = []

    def log_snapshot(self, source, label, cache_value=True):
        self.snapshots.append((source, label, cache_value))


class _Driver:
    def __init__(self, policy=None):
        self.limit_policy = policy


@pytest.mark.parametrize("source, policy", [(None, object()), (object(), None)])
def test_a_usage_pair_refuses_a_missing_half(source, policy):
    with pytest.raises(ValueError, match="both a source and a policy"):
        runlifecycle.RunUsage(source, policy, "claude")


def test_no_usage_endpoint_opens_nothing(monkeypatch):
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: None)
    policy = _RecordingPolicy()

    assert runlifecycle.open_usage(_Driver(policy), "claude", "claude",
                                   dry_run=False) is None
    assert policy.snapshots == []


def test_opening_pairs_the_drivers_policy_and_logs_the_start(monkeypatch):
    source = object()
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: source)
    policy = _RecordingPolicy()

    usage = runlifecycle.open_usage(_Driver(policy), "claude", "parallel",
                                    dry_run=False)

    assert (usage.source, usage.policy) == (source, policy)
    assert policy.snapshots == [(source, "at start (parallel)", True)]


def test_opening_falls_back_to_the_providers_default_policy(monkeypatch):
    source = object()
    default = _RecordingPolicy()
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: source)
    monkeypatch.setattr(limits, "default_policy",
                        lambda provider: default if provider == "codex" else None)

    usage = runlifecycle.open_usage(_Driver(), "codex", "codex", dry_run=True)

    assert usage.policy is default
    assert default.snapshots == [], "a dry run is not a run: no opening snapshot"


def test_the_closing_snapshot_answers_the_opening_one():
    source, policy = object(), _RecordingPolicy()
    usage = runlifecycle.RunUsage(source, policy, "claude")

    usage.open()
    usage.close()
    usage.close("interrupted")

    assert policy.snapshots == [
        (source, "at start (claude)", True),
        # Fresh, not cached: the closing figures are the post-run state.
        (source, "at end (claude)", False),
        (source, "at end (interrupted)", False),
    ]


@pytest.mark.parametrize("runner", [cyclecore.run_loop, parallel.run_parallel])
def test_both_runners_open_usage_through_the_shared_path(runner):
    """Checked on the SOURCE: a runner that built its own pair again would still
    pass every behavioural pin whose stub source it happened to receive."""
    tree = ast.parse(inspect.getsource(runner))
    names = {node.attr if isinstance(node, ast.Attribute) else node.id
             for node in ast.walk(tree)
             if isinstance(node, (ast.Attribute, ast.Name))}

    assert "open_usage" in names
    assert not names & {"usage_source_for", "default_policy", "log_snapshot"}, (
        "the runner assembles the usage pair itself again")
