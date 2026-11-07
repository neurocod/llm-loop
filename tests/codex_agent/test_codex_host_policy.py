"""The core with no host (`EMPTY`), and with test hosts of its own.

What a host supplies is pinned by policies of this suite's own
(`core_policy.CORE_POLICY` and `TEST_POLICY` below), so the core's tests name
no project; a real host pins its own policy in its own repository.
"""

from test_codex_agent import _install_fake_server, _no_badge  # noqa: F401
import json
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest

from llm_agents.codex import cli, gates, paths, roles, sandbox, turn
from llm_agents.codex.command_arguments import _args_refusal
from llm_agents.codex.command_dispatch import Shape, argv_refusal
from llm_agents.codex.command_node import NpmScript, _npm
from llm_agents.codex.registry import AgentRecord, Registry
from llm_agents.codex.testing import PS as _PS
from llm_agents import agent_counter, host_policy
from llm_agents.host_policy import (
    EMPTY, GIT_STORE, HostPolicy, Reserved, WorkerCommands)

# One program this test host lets a worker run outside: `tool <paths>`.
TEST_POLICY = HostPolicy(
    commands=WorkerCommands(
        shapes=(
            Shape(("tool",), "`tool <paths>`", _args_refusal),
            Shape(("npm",), "`npm test`", partial(_npm, judge_argv=argv_refusal)),
        ),
        shapes_name="the test shapes",
        npm_scripts={"test": NpmScript("runs `tool`")},
        npm_name="the test npm table",
        flags=lambda roots: frozenset({"seen"}),
        reserved=Reserved(text=lambda record: "anything the test host forgot")),
    protected_parts=GIT_STORE + (("vault",),),
    gate_cwd=lambda command, project: project / "sub",
    worker_lines=("Test host line.",),
    commit_briefing=lambda where, model: f"commit with the test tool into {where}.",
    counter_exes=("bin/counter.exe",),
)


@pytest.fixture
def host():
    with host_policy.installed(TEST_POLICY, Path.cwd()):
        yield TEST_POLICY


@pytest.fixture
def empty():
    with host_policy.installed(EMPTY):
        yield EMPTY


def _judge(tmp_path, monkeypatch, **fields):
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    work = tmp_path / "ws"
    (work / "src").mkdir(parents=True, exist_ok=True)
    fields.setdefault("role", "worker")
    judge = sandbox.escalation_judge(AgentRecord(
        name="w", cwd=str(work), sandbox="workspace-write", **fields))
    if judge is None:  # Nothing may leave the sandbox: no judge is installed.
        return work, lambda command, cwd=None: None
    return work, lambda command, cwd=None: judge(
        {"command": _PS.format(command), "cwd": str(cwd or work)})


def test_without_a_host_no_command_leaves_the_sandbox_by_shape(
        tmp_path, monkeypatch, empty):
    """The empty policy is a policy, not a guess at one: a host's command
    shape is "not a worker command shape", and the briefing says none exist."""
    work, ask = _judge(tmp_path, monkeypatch)
    (work / "package.json").write_text(json.dumps({"scripts": {"test": "x"}}))
    for command in ("npm test", "python -m pytest", "npx vitest run",
                    "python tools/try_patch.py --file a -- npm test"):
        verdict = ask(command)
        assert not verdict and "is not a worker command shape" in verdict.reason, command
        assert sandbox.worker_decline_tail() in verdict.reason
    told = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(work), sandbox="workspace-write",
        role="worker"))["developerInstructions"]
    assert "none (the host policy names no command shapes)" in told
    assert "every other program" in told
    assert gates.gate_cwd("npm run typecheck", tmp_path) == tmp_path
    assert "Test host line." not in roles.role_briefing("worker")
    monkeypatch.delenv(agent_counter.EXE_ENV, raising=False)
    assert host_policy.counter_paths() == [] and agent_counter.counter_exe() is None


def test_without_a_host_only_this_calls_explicit_asks_leave_the_sandbox(
        tmp_path, monkeypatch, empty):
    """Under EMPTY the wrapper grants nothing on its own account: no cleanup
    delete, no edit retry, no recorded `--escalate` or gate replayed by a
    `send`. What the caller passed in THIS invocation still holds."""
    work, ask = _judge(tmp_path, monkeypatch, role="", escalate=r"^tool\b",
                       gates=["tool gate"])
    (work / ".pytest-tmp").mkdir()
    assert not ask("Remove-Item -Recurse -Force .pytest-tmp")
    assert not ask("tool src/a.txt") and not ask("tool gate")
    edits = sandbox.file_change_judge(AgentRecord(
        name="w", cwd=str(work), sandbox="workspace-write", escalate=r"^tool\b"))
    refused = edits([str(work / "src" / "a.txt")])
    assert not refused and "no host policy" in refused.reason
    host_policy.grant({"escalate"})
    _, ask = _judge(tmp_path, monkeypatch, role="", escalate=r"^tool\b",
                    gates=["tool gate"])
    assert ask("tool src/a.txt").reason == "matches the escalate rule"
    assert not ask("Remove-Item -Recurse -Force .pytest-tmp")
    host_policy.grant({"gates"})
    _, ask = _judge(tmp_path, monkeypatch, role="", gates=["tool gate"])
    assert ask("tool gate")


def test_without_a_host_recorded_gates_run_only_when_this_call_passed_them(
        tmp_path, monkeypatch, empty):
    ran = []

    def runner(command, cwd, timeout):
        ran.append(command)
        return gates.GateRun(command, str(cwd), 0, "ok", 0.1)

    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    _install_fake_server(monkeypatch, tmp_path, "plain")
    record = AgentRecord(name="w", cwd=str(tmp_path), sandbox="read-only",
                         gates=["tool gate"])
    outcome = turn.run_turn(record=record, prompt="p", resume=False,
                            timeout=30.0, run_gate=runner)
    assert ran == [] and "no host policy" in outcome.gate_note
    host_policy.grant({"gates"})
    record = AgentRecord(name="w2", cwd=str(tmp_path), sandbox="read-only",
                         gates=["tool gate"])
    turn.run_turn(record=record, prompt="p", resume=False, timeout=30.0,
                  run_gate=runner)
    assert ran == ["tool gate"]


def test_a_host_holds_for_one_call_and_a_call_without_one_says_so(
        tmp_path, monkeypatch, capsys, empty):
    """`main(policy=P)` installs P for that call only: the next call without
    a policy runs under EMPTY and announces it, naming what it still
    honours."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    seen = []
    monkeypatch.setattr(cli, "cmd_status", lambda args: seen.append(
        (host_policy.active(), host_policy.root())) or 0)
    assert cli.main(["status"], policy=TEST_POLICY, root=tmp_path) == 0
    assert seen[-1] == (TEST_POLICY, tmp_path.resolve())
    assert "no host policy" not in capsys.readouterr().err
    assert host_policy.active() is EMPTY and host_policy.root() is None
    assert cli.main(["status"]) == 0
    assert seen[-1] == (EMPTY, None)
    err = capsys.readouterr().err
    assert err.count("codex_agent: no host policy given") == 1
    assert "spawn --escalate REGEX, spawn --gates-from-prompt" in err
    with pytest.raises(ValueError):
        cli.main(["status"], policy=TEST_POLICY)


def test_only_spawn_s_own_flags_are_explicit_asks(tmp_path, monkeypatch, empty):
    """`spawn --escalate`/`--gates-from-prompt` grant for that call; a `send`
    replaying the record asks for nothing itself."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    prompt = tmp_path / "prompt.md"
    prompt.write_text("Task.\n\n## Gates\n\n```\ntool gate\n```\n", encoding="utf-8")
    seen = []
    for name in ("cmd_spawn", "cmd_send"):
        monkeypatch.setattr(cli, name, lambda args: seen.append(host_policy.explicit()) or 0)
    assert cli.main(["spawn", "w", "--prompt-file", str(prompt), "--cwd", str(tmp_path),
                     "--escalate", "^tool", "--gates-from-prompt"]) == 0
    assert cli.main(["spawn", "w", "task", "--cwd", str(tmp_path)]) == 0
    assert cli.main(["send", "w", "more"]) == 0
    assert seen == [{"escalate", "gates"}, frozenset(), frozenset()]


def test_a_test_host_s_shapes_paths_and_flags_are_the_ones_judged(
        tmp_path, monkeypatch, host):
    work, ask = _judge(tmp_path, monkeypatch)
    assert ask("tool src/a.txt").reason == "a worker command shape"
    assert ask("tool src/a.txt 2>&1 | Select-Object -Last 5")
    outside = ask(f"tool {tmp_path / 'elsewhere.txt'}")
    assert not outside and "outside your writable roots" in outside.reason
    vault = ask("tool vault/key.txt")
    assert not vault and "vault/" in vault.reason
    assert "the test shapes" in vault.reason
    (work / "package.json").write_text(json.dumps({"scripts": {
        "test": "tool src/a.txt", "build": "tool src/b.txt"}}))
    assert ask("npm test")
    refused = ask("npm run build")
    assert not refused and "is not on the test npm table" in refused.reason
    assert not ask("python -m pytest")
    # The edit judge holds the same protected runs.
    edits = sandbox.file_change_judge(AgentRecord(
        name="w", cwd=str(work), sandbox="workspace-write"))
    assert edits([str(work / "src" / "a.txt")])
    assert not edits([str(work / "vault" / "x")])


def test_a_test_host_s_briefing_gates_and_counter(tmp_path, monkeypatch, host):
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    told = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(tmp_path), sandbox="workspace-write",
        role="worker"))["developerInstructions"]
    assert "`tool <paths>`; `npm test`" in told
    assert "`.git/`, `vault/`" in told
    assert "anything the test host forgot" in told
    assert "Test host line." in told
    assert gates.gate_cwd("npm test", tmp_path) == tmp_path / "sub"
    runs = gates.run_gates(["x"], tmp_path, 5.0,
                           lambda c, cwd, t: gates.GateRun(c, str(cwd), 0, ""),
                           lambda line: None)
    assert runs[0].cwd == str(tmp_path / "sub")
    monkeypatch.delenv(agent_counter.EXE_ENV, raising=False)
    assert host_policy.counter_paths() == [Path.cwd().resolve() / "bin/counter.exe"]


def test_a_test_host_briefs_a_slot_worker_to_commit_in_its_words(
        tmp_path, monkeypatch, host):
    main = tmp_path / "main"
    main.mkdir()
    for args in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@x.invalid",
                                  "commit", "-q", "--allow-empty", "-m", "base"],
                 ["worktree", "add", "-q", "--detach", str(tmp_path / "main-wt1")]):
        subprocess.run(["git", *args], cwd=main, check=True, capture_output=True)
    slot = tmp_path / "main-wt1"
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    record = AgentRecord(name="w", cwd=str(slot), sandbox="workspace-write",
                         role="worker")
    assert sandbox.commit_briefed(record)
    told = sandbox.thread_params(record)["developerInstructions"]
    assert "commit with the test tool into your scratch directory" in told
    with host_policy.installed(replace(TEST_POLICY, commit_briefing=None)):
        assert not sandbox.commit_briefed(record)
        assert "commit with the test tool" not in sandbox.thread_params(
            record)["developerInstructions"]


# Written by the registry code as it was before the host policy existed
# (2026-11-07, before this package was published): its `__init__`, `registry`,
# `paths` and `gitlib` modules taken into a temp tree, imported from there,
# and `Registry().put(AgentRecord(name="old", thread_id="t-old",
# cwd="C:\\work\\slot", model="gpt-x", sandbox="workspace-write",
# created/last_turn/touched=2026-11-07 12:0x, turns=1, escalate="^tool\\b",
# writable=["C:\\work\\extra"], role="worker", gates=["tool gate"],
# allow_online_probes=True))` run with CODEX_AGENTS_DIR at a temp directory;
# its registry.json is the fixture as written. (A detached worktree of that
# revision does not check out here: MAX_PATH.)
OLD_REGISTRY = Path(__file__).with_name("fixtures") / "registry_pre_host_policy.json"


def test_a_record_written_before_the_host_policy_still_loads(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    raw = OLD_REGISTRY.read_bytes()
    (tmp_path / "registry.json").write_bytes(raw)
    old = json.loads(raw)["old"]
    record = Registry().get("old")
    assert record.thread_id == "t-old"
    assert record.allow_online_probes and record.role == "worker"
    assert record.gates == ["tool gate"] and record.writable == ["C:\\work\\extra"]
    assert record.to_json() == old
    Registry().put(record)
    # Content, not bytes: the checkout may carry the fixture with other
    # line endings than the writer used.
    assert json.loads((tmp_path / "registry.json").read_bytes()) == json.loads(raw)
