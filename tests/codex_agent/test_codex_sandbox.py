"""Sandbox configuration and role briefings."""

from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401 — see no_badge's docstring
from pathlib import Path
import pytest
from llm_agents.codex import cli
from llm_agents.codex import paths
from llm_agents.codex import sandbox
from llm_agents.codex import turn
from llm_agents.codex.registry import AgentRecord
from llm_agents.codex.registry import Registry


def test_a_turn_never_creates_the_project_directory_it_was_given(tmp_path,
                                                                 monkeypatch):
    """The same guard one level down, where a resumed agent arrives: a record
    whose project has since been moved must be told so, not have it recreated
    (an empty tree reads to the agent as a project with nothing in it)."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    gone = tmp_path / "gone"

    with pytest.raises(SystemExit) as refused:
        turn.run_turn(record=AgentRecord(name="rev", cwd=str(gone),
                                         sandbox="read-only"),
                      prompt="p", resume=True, timeout=5.0)

    assert "does not exist" in str(refused.value)
    assert not gone.exists()


def test_unattended_threads_never_wait_for_an_approver(tmp_path):
    record = AgentRecord(name="a", cwd=str(tmp_path), sandbox="read-only")
    params = sandbox.thread_params(record)
    assert params["approvalPolicy"] == "never"
    assert params["sandbox"] == "read-only"
    # A record from before the field existed must not fall back to the setting
    # that cannot run commands on this machine (see the --sandbox flag).
    assert sandbox.thread_params(AgentRecord(name="a"))["sandbox"] == "read-only"


def test_scratch_write_is_workspace_write_rooted_outside_the_project(tmp_path):
    """The whole mode is that one substitution: the protocol has no read-only
    sandbox WITH a writable root (`SandboxPolicy.readOnly` carries only
    `networkAccess`), so the only way to give a reviewer somewhere to write is
    `workspace-write` — and the only thing keeping the project unwritable under
    it is that the workspace root is somewhere else."""
    record = AgentRecord(name="rev", cwd=str(tmp_path), sandbox=sandbox.SCRATCH_WRITE)
    params = sandbox.thread_params(record)

    assert params["sandbox"] == "workspace-write"
    workspace = Path(params["cwd"])
    assert workspace == paths.scratch_dir("rev")
    # BOTH directions. Only the second one carries the promise: a project
    # sitting inside the writable workspace is a workspace-write agent wearing
    # this mode's label, and asserting only that the scratch is not in the
    # project would not notice.
    assert tmp_path not in workspace.parents and workspace != tmp_path
    assert workspace not in tmp_path.resolve().parents
    # The agent starts outside the project, so nothing tells it where the
    # project is, that it is read-only there, or how to get a test suite to run.
    assert str(tmp_path) in params["developerInstructions"]
    assert "--basetemp" in params["developerInstructions"]
    # ...and what the sandbox itself refuses, same as every sandboxed agent.
    assert "spawn EPERM" in params["developerInstructions"]


def test_the_other_sandboxes_still_work_in_the_project_itself(tmp_path):
    """`scratch-write` moves the workspace; the three protocol modes must not
    move with it, or a `workspace-write` agent would silently edit a scratch
    copy instead of the tree it was pointed at."""
    for policy in ("read-only", "workspace-write", "danger-full-access"):
        params = sandbox.thread_params(
            AgentRecord(name="rev", cwd=str(tmp_path), sandbox=policy))
        assert params["sandbox"] == policy
        assert params["cwd"] == str(tmp_path)
        # The scratch briefing (where the project is, how pytest must be run)
        # is the scratch mode's alone; the sandbox's own limits — piped Node
        # children refused — reach every SANDBOXED agent, and nobody else.
        briefing = params.get("developerInstructions", "")
        assert str(tmp_path) not in briefing
        assert "--basetemp" not in briefing
        if policy == "danger-full-access":
            # Full access has no sandbox to explain; what reaches every
            # agent is the sub-agent rule alone.
            assert briefing == sandbox.SUB_AGENT_FACT
        else:
            assert "spawn EPERM" in briefing
            assert briefing.endswith("\n\n" + sandbox.SUB_AGENT_FACT)


def test_scratch_write_refuses_a_project_inside_its_own_workspace(tmp_path,
                                                                  monkeypatch):
    """Point `--cwd` into the scratch directory and the mode still calls itself
    read-only-for-the-project while the project IS the writable workspace."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    workspace = paths.scratch_dir("rev")

    for project in (workspace, workspace / "checkout"):
        with pytest.raises(SystemExit) as refused:
            sandbox.thread_params(AgentRecord(name="rev", cwd=str(project),
                                              sandbox=sandbox.SCRATCH_WRITE))
        assert "writable" in str(refused.value)


def test_an_unknown_sandbox_is_named_rather_than_sent(tmp_path):
    """Only a hand-edited registry gets here (argparse refuses the rest), and
    the server's answer to an unknown sandbox is a failed turn that says nothing
    about the cause."""
    with pytest.raises(SystemExit) as refused:
        sandbox.thread_params(AgentRecord(name="rev", cwd=str(tmp_path),
                                          sandbox="read-write-ish"))
    assert "read-write-ish" in str(refused.value)


def test_writable_widens_the_policy_and_is_refused_where_it_cannot(
        tmp_path, monkeypatch):
    """`spawn --writable DIR` (user question 2026-09-13: "can an agent be
    given full access to chosen folders?"): the root goes on the wire beside
    the scratch root, the briefing names it, the registry keeps it for `send`,
    and the parser carries it. Refused, before anything is claimed or
    created: a directory that does not exist, a mode with no root list to
    widen, and — under `scratch-write` — a root that contains the project,
    which would make the read-only promise a lie."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    extra = tmp_path / "assets"
    extra.mkdir()
    work = tmp_path / "ws"
    work.mkdir()
    record = AgentRecord(name="w", cwd=str(work), sandbox="workspace-write",
                         writable=[str(extra)])
    policy = sandbox.turn_sandbox_policy(record)
    assert policy["writableRoots"] == [str(paths.scratch_dir("w")), str(extra)]
    assert str(extra) in sandbox.thread_params(record)["developerInstructions"]
    assert sandbox.prepare_workspace(record) == (work, policy)
    # A scratch-write reviewer gets it beside its scratch workspace.
    rev = AgentRecord(name="rev", cwd=str(work), sandbox=sandbox.SCRATCH_WRITE,
                      writable=[str(extra)])
    assert sandbox.turn_sandbox_policy(rev)["writableRoots"] == [str(extra)]
    assert str(extra) in sandbox.thread_params(rev)["developerInstructions"]
    # Refusals.
    missing = tmp_path / "typo"
    with pytest.raises(SystemExit) as refused:
        sandbox.prepare_workspace(AgentRecord(
            name="w", cwd=str(work), sandbox="workspace-write",
            writable=[str(missing)]))
    assert "--writable" in str(refused.value)
    assert not missing.exists()
    for mode in ("read-only", "danger-full-access"):
        with pytest.raises(SystemExit) as refused:
            sandbox.check_writable_roots(AgentRecord(
                name="w", cwd=str(work), sandbox=mode, writable=[str(extra)]))
        assert mode in str(refused.value)
    with pytest.raises(SystemExit) as hole:
        sandbox.check_writable_roots(AgentRecord(
            name="rev", cwd=str(work), sandbox=sandbox.SCRATCH_WRITE,
            writable=[str(tmp_path)]))
    assert "writable" in str(hole.value)
    # Survives the registry, so a `send` resumes with the same roots...
    Registry().put(record)
    assert Registry().get("w").writable == [str(extra)]
    # ...and the flag reaches the record through the parser.
    args = cli.build_parser().parse_args(
        ["spawn", "w", "--writable", str(extra), "p"])
    assert args.writable == [str(extra)]


def test_a_linked_worktree_workspace_is_told_git_cannot_write(tmp_path):
    """A worker in a pool slot spent its turn asking whether it might move files
    without `git mv` (2026-09-13); the fact that git writes fail there is the
    harness's to tell, up front. A linked worktree's `.git` is a FILE."""
    slot = tmp_path / "slot"
    slot.mkdir()
    (slot / ".git").write_text("gitdir: ../main/.git/worktrees/slot\n",
                               encoding="utf-8")
    told = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(slot), sandbox="workspace-write"))["developerInstructions"]
    assert "linked git worktree" in told
    assert "Move-Item" in told
    main = tmp_path / "main"
    (main / ".git").mkdir(parents=True)
    plain = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(main), sandbox="workspace-write"))["developerInstructions"]
    assert "linked git worktree" not in plain


