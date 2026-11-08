"""Sandbox wire scenarios."""

from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401 — see no_badge's docstring
import json
import shutil
import pytest
from llm_agents.codex.testing import wire
from llm_agents.codex import cli
from llm_agents.codex import paths
from llm_agents.codex import sandbox
from llm_agents.codex.registry import AgentRecord
from llm_agents.codex.registry import Registry


def test_escalation_lets_out_exactly_the_named_gate_and_nothing_else(
        tmp_path, monkeypatch):
    """The sandbox cannot run the npm gates, so an `escalate` regex names the
    commands allowed OUT of it. The judge reads the request: the command must
    match, must run inside the agent's workspace, and must be a command (not
    stdin to a running one). Without the rule every request is declined, and
    the thread never even offers the model a way to ask."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "escalate", timeout=15.0,
                             escalate=r"^npm (test|run typecheck)\b")
    assert outcome.status == "completed"
    assert outcome.text == "escalation handled"
    answers = wire.answers(tmp_path)
    assert answers[9101]["result"] == {"decision": "accept"}
    assert answers[9102]["result"] == {"decision": "decline"}  # not a gate
    assert answers[9103]["result"] == {"decision": "decline"}  # other tree
    assert answers[9104]["result"] == {"decision": "decline"}  # stdin, not a command
    # Codex wraps the agent's command in its shell; the judge reads through it.
    assert answers[9105]["result"] == {"decision": "accept"}
    assert answers[9106]["result"] == {"decision": "decline"}
    with_rule = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(tmp_path), sandbox="workspace-write", escalate="^npm"))
    without = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(tmp_path), sandbox="workspace-write"))
    assert with_rule["approvalPolicy"] == "on-request"
    assert without["approvalPolicy"] == "never"
    assert "EXCEPTION" in with_rule["developerInstructions"]
    assert "EXCEPTION" not in without["developerInstructions"]
    # And the rule survives the registry, so a `send` keeps it.
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="kept", thread_id="t", escalate="^npm"))
    assert Registry().get("kept").escalate == "^npm"


def test_a_judge_that_raises_is_a_decline_not_a_dead_turn():
    """The server blocks on every approval: an exception out of a judge would
    end the turn with nothing answered."""
    from llm_agents.codex import protocol

    def broken(_request):
        raise OSError("bad path")

    verdict = protocol._judged(broken, {})
    assert not verdict
    assert protocol._reason_of(verdict) == {
        "reason": "the wrapper's judge failed: OSError: bad path"}
    # An AppServer given no edit judge declines edits the same way.
    refused = protocol._no_edits(["a.py"])
    assert not refused and "never leave the sandbox" in refused.reason


def test_a_workspace_write_agent_gets_a_second_root_outside_the_tree(
        tmp_path, monkeypatch):
    """A worker whose only writable place is the checkout puts its baselines
    and measuring scripts INTO the checkout (two workers, 2026-09-13). The
    protocol can say more than the thread's mode string: `turn/start` carries
    a `sandboxPolicy` with extra `writableRoots`. Three things must hold
    together — the root is on the wire, it exists before the turn (a missing
    root is one the agent cannot write to), and the briefing names it — and a
    `read-only` agent gets none of the three, or the mode would be a lie."""
    # The scratch path is keyed by the registry, so the registry is pointed at
    # the test's directory BEFORE the path is read (`wire.run_fake_turn` does the
    # same, later).
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    scratch = paths.scratch_dir("w")
    if scratch.exists():
        shutil.rmtree(scratch)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "plain",
                             sandbox="workspace-write")
    assert outcome.status == "completed"
    started = wire.first_request(tmp_path, "turn/start")["params"]
    assert started["sandboxPolicy"] == {
        "type": "workspaceWrite", "writableRoots": [str(scratch)],
        "networkAccess": True}
    # Existed when the server took `turn/start`, not merely by the end: the
    # fake server records that moment (review 2026-09-13 — checked after the
    # turn, a root created too late stayed green).
    roots_then = json.loads((tmp_path / "roots_at_turn_start.json")
                            .read_text(encoding="utf-8"))
    assert roots_then == {str(scratch): True}
    # Outside the tree, or the second root adds nothing the first lacks.
    assert tmp_path.resolve() not in scratch.resolve().parents
    # The briefing as SENT, not as a separate call would build it.
    opened = wire.first_request(tmp_path, "thread/start")["params"]
    assert str(scratch) in opened["developerInstructions"]

    read_only = wire.run_fake_turn(tmp_path, monkeypatch, "plain", name="r",
                               sandbox="read-only")
    assert read_only.status == "completed"
    assert "sandboxPolicy" not in wire.turn_starts(tmp_path)[-1]["params"]
    opened_ro = [message["params"] for message in wire.received(tmp_path)
                 if message.get("method") == "thread/start"][-1]
    assert str(paths.scratch_dir("r")) not in opened_ro["developerInstructions"]
    assert not paths.scratch_dir("r").exists()


