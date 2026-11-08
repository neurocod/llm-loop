"""Offline transport regressions using a real Python child, never live Claude."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import pytest

from llm_agents import host_policy
from llm_agents.claude import protocol


FAKE_CLAUDE = r'''
import json
from pathlib import Path
import subprocess
import sys
import time

scenario = sys.argv[1]
root = Path(sys.argv[2])
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")
if scenario == "blocked_input":
    (root / "ready").write_text("ready", encoding="utf-8")
    import threading
    threading.Event().wait()
payload = sys.stdin.read()
(root / "received.json").write_text(payload, encoding="utf-8")
message = json.loads(payload)
if scenario == "context":
    print(json.dumps({"type":"system","subtype":"init","model":"primary[1m]"}), flush=True)
    print(json.dumps({"type":"assistant","message":{"model":"primary","usage":{
        "input_tokens":10000,"cache_creation_input_tokens":40000,"cache_read_input_tokens":400000
    }}}), flush=True)
if scenario in ("rate_limit", "rate_limit_bare"):
    # The shape of .claude-agents/review-s/turn-0002, line 4.
    info = {"status": "allowed", "resetsAt": 1794402000, "rateLimitType": "seven_day"}
    if scenario == "rate_limit":
        info["unifiedWindows"] = {"seven_day": {
            "utilization": 0.18, "resetsAt": 1794402000, "status": "allowed"}}
    print(json.dumps({"type": "rate_limit_event", "rate_limit_info": info,
                      "session_id": message["session_id"]}), flush=True)
print("diagnostic before JSON", flush=True)
print("[]", flush=True)
print(json.dumps({"type": "assistant", "message": {"content": [
    {"type": "thinking", "thinking": "DO_NOT_EXPOSE_THINKING"},
    {"type": "text", "text": "DO_NOT_EXPOSE_PROGRESS"}]}}), flush=True)
print("stderr diagnostic", file=sys.stderr, flush=True)
if scenario == "interrupt":
    (root / "ready").write_text("ready", encoding="utf-8")
    while True:
        time.sleep(0.01)
if scenario == "no_result":
    sys.exit(0)
if scenario == "inherited_stdout":
    subprocess.Popen([sys.executable, "-c", "import threading; threading.Event().wait()"],
                     stdin=subprocess.DEVNULL)
result = {"type": "result", "subtype": "success", "is_error": False,
          "session_id": message["session_id"], "result": message["message"]["content"],
          "usage": {"input_tokens": 17, "output_tokens": 23},
          "total_cost_usd": 0.125, "permission_denials": []}
if scenario == "is_error":
    result.update(is_error=True, result="Provider refused the request")
elif scenario == "error_subtype":
    result.update(subtype="error_max_turns", errors=["Maximum turns exceeded"])
elif scenario == "denied":
    result["permission_denials"] = [{"tool_name": "Bash", "tool_input": {"command": "write"}}]
elif scenario == "context":
    result["modelUsage"] = {"auxiliary":{"contextWindow":200000},
                            "primary[1m]":{"canonicalModel":"primary","contextWindow":1000000}}
print(json.dumps(result, ensure_ascii=False), flush=True)
sys.exit(7 if scenario == "nonzero" else 0)
'''


@pytest.fixture
def fake_cli(tmp_path, monkeypatch):
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLAUDE, encoding="utf-8")
    agent = {"cwd": str(tmp_path), "session_id": str(uuid.uuid4()),
             "options": {"profile": "configured"}}

    def run(scenario="success", prompt="FINAL ANSWER", cancel=None, on_session=None, on_context=None,
            deadline=None, on_week_used=None):
        monkeypatch.setattr(protocol, "command", lambda agent, resume: [
            sys.executable, str(script), scenario, str(tmp_path)])
        outcome = protocol.run_turn(agent, {"resume": False, "prompt": prompt},
                                    tmp_path / "logs", cancel or (lambda: False),
                                    on_session=on_session, on_context=on_context,
                                    on_week_used=on_week_used, deadline=deadline)
        return outcome

    return run, tmp_path, agent


# The core tests' review host: a settings file naming the gate hook, a
# built gate newer than its sources, and a submodule for the templates.
SUBMODULE = "sub/module"


@pytest.fixture
def review_host(tmp_path):
    root = tmp_path / "host"
    gate = root / "gate"
    (gate / "cpp" / "third_party").mkdir(parents=True)
    (gate / "hooks").mkdir()
    (gate / "hooks" / "ask_user_gate.py").write_text("", encoding="utf-8")
    (gate / "hooks" / "ask_user_gate.exe").write_text("", encoding="utf-8")
    (root / "settings.json").write_text(json.dumps({"hooks": {"PreToolUse": [{
        "matcher": "Bash|PowerShell|Monitor", "hooks": [{"type": "command",
            "command": '"$CLAUDE_PROJECT_DIR/gate/hooks/ask_user_gate.exe"'}]}]}}),
        encoding="utf-8")
    profile = host_policy.ReviewProfile(settings="settings.json", gate="gate",
                                        submodule=SUBMODULE, note="Core test note.")
    with host_policy.installed(host_policy.HostPolicy(review=profile), root):
        yield root


def test_without_a_host_a_review_refuses_with_its_reason(fake_cli, monkeypatch):
    """No host names a gate hook, so the review profile does not start: a
    named refusal, not a FileNotFoundError about some path of a host."""
    run, root, agent = fake_cli
    agent["options"]["profile"] = "review"
    with host_policy.installed(host_policy.EMPTY):
        with pytest.raises(protocol.ReviewUnavailable, match="HostPolicy.review"):
            protocol.verify_review_gate()
        monkeypatch.setattr(protocol.shutil, "which", lambda name: "claude")
        with pytest.raises(protocol.ReviewUnavailable):
            protocol.command(agent, resume=False)
        outcome = protocol.run_turn(agent, {"resume": False, "prompt": "p"},
                                    root / "logs", lambda: False)
    assert outcome["status"] == "failed"
    assert outcome["error"].startswith("ReviewUnavailable: the review profile needs")
    assert not (root / "received.json").exists()


def test_success_waits_for_eof_and_returns_only_final_answer(fake_cli, capsys):
    run, root, agent = fake_cli
    outcome = run()
    assert outcome["status"] == "completed"
    assert outcome["answer"] == "FINAL ANSWER"
    assert outcome["error"] == ""
    assert outcome["session_id"] == agent["session_id"]
    assert outcome["usage"] == {"input_tokens": 17, "output_tokens": 23}
    assert outcome["total_cost_usd"] == 0.125
    assert capsys.readouterr().out == ""
    received = json.loads((root / "received.json").read_text(encoding="utf-8"))
    assert received["message"] == {"role": "user", "content": "FINAL ANSWER"}
    assert received["type"] == "user"
    assert received["parent_tool_use_id"] is None
    assert str(uuid.UUID(received["uuid"])) == received["uuid"]


@pytest.mark.parametrize("scenario,reason", [
    ("is_error", "Provider refused"),
    ("error_subtype", "Maximum turns exceeded"),
    ("no_result", "without a result"),
    ("nonzero", "exited 7"),
])
def test_failed_turns_are_not_reported_as_success(fake_cli, scenario, reason):
    run, _, _ = fake_cli
    outcome = run(scenario)
    assert outcome["status"] == "failed"
    assert reason in outcome["error"]
    if scenario == "nonzero":
        assert outcome["answer"] == "FINAL ANSWER"


def test_unicode_and_raw_diagnostics_survive_without_polluting_answer(fake_cli):
    run, root, _ = fake_cli
    answer = "Review: Привет, café, 日本語, 😀\nsecond line"
    outcome = run(prompt=answer)
    assert outcome["status"] == "completed"
    assert outcome["answer"] == answer
    logs = root / "logs"
    assert (logs / "answer.md").read_text(encoding="utf-8") == answer
    assert json.loads((logs / "result.json").read_text(encoding="utf-8")) == outcome
    raw = (logs / "events.jsonl").read_text(encoding="utf-8")
    assert "diagnostic before JSON" in raw
    assert "DO_NOT_EXPOSE_THINKING" in raw
    assert "DO_NOT_EXPOSE_PROGRESS" in raw
    assert "Привет" in raw
    assert (logs / "stderr.log").read_text(encoding="utf-8").strip() == "stderr diagnostic"


def test_interrupt_terminates_real_child_and_persists_outcome(fake_cli):
    run, root, _ = fake_cli
    outcome = run("interrupt", cancel=lambda: (root / "ready").exists())
    assert outcome["status"] == "interrupted"
    assert outcome["error"] == "Interrupted by caller"
    assert outcome["answer"] == ""
    assert (root / "logs" / "result.json").exists()


def test_permission_denials_are_preserved(fake_cli):
    run, _, _ = fake_cli
    outcome = run("denied")
    assert outcome["permission_denials"] == [
        {"tool_name": "Bash", "tool_input": {"command": "write"}}]


def test_session_is_recorded_before_outcome_is_returned(fake_cli):
    run, _, agent = fake_cli
    sessions = []
    outcome = run(on_session=sessions.append)
    assert outcome["status"] == "completed"
    assert sessions == [agent["session_id"]]


def test_completed_parent_does_not_wait_for_descendant_stdout(fake_cli):
    run, _, agent = fake_cli
    # Plain child 0.185 s measured 2026-09-08; the broken inherited-pipe case
    # consumed the entire 10 s caller deadline despite an already valid result.
    agent["options"]["turn_timeout"] = 10
    outcome = run("inherited_stdout")
    assert outcome["status"] == "completed", outcome
    assert outcome["answer"] == "FINAL ANSWER"


def test_interrupt_can_cancel_a_large_prompt_before_stdin_drains(fake_cli):
    run, root, agent = fake_cli
    # Same fake process startup measurement as the inherited-stdout test.
    agent["options"]["turn_timeout"] = 10
    outcome = run("blocked_input", prompt="x" * 1_000_000,
                  cancel=lambda: (root / "ready").exists())
    assert outcome["status"] == "interrupted", outcome


def test_a_passed_deadline_replaces_a_fresh_turn_timeout(fake_cli):
    run, root, agent = fake_cli
    # The worker's follow-up passes what is left of the caller's 600 s, here
    # 2 s — ten times the 0.185 s fake startup measured 2026-09-08. A fresh
    # 600 s clock would hang the never-answering child for ten minutes.
    agent["options"]["turn_timeout"] = 600
    started = time.monotonic()
    outcome = run("interrupt", deadline=started + 2)
    assert outcome["status"] == "failed", outcome
    assert outcome["error"] == "Turn exceeded caller deadline (600s)"
    assert time.monotonic() - started < 60


def test_context_callback_reports_live_input_then_measured_capacity(fake_cli):
    run, _, _ = fake_cli
    snapshots = []
    outcome = run("context", on_context=snapshots.append)
    assert outcome["status"] == "completed"
    assert any(s["input_tokens"] == 450000 and s["context_window"] is None for s in snapshots)
    assert snapshots[-1]["context_window"] == 1000000
    assert snapshots[-1]["used_percent"] == 45.0
    assert snapshots[-1]["warning"] is True
    assert outcome["context"] == snapshots[-1]


@pytest.mark.parametrize("scenario, expected", [
    ("rate_limit", [(18.0, "")]),
    ("rate_limit_bare", [(None, "rate_limit_event without unifiedWindows.seven_day.utilization")]),
    ("success", []),
])
def test_the_streams_weekly_figure_reaches_the_counter_callback(fake_cli, scenario, expected):
    """The on-screen counter's Claude figure comes off the turn's own stream
    (`protocol.week_used_percent`): a fraction there, a percent here, and a
    stream without it is a reason, not silence."""
    run, _, _ = fake_cli
    reported = []
    outcome = run(scenario, on_week_used=lambda used, why: reported.append((used, why)))
    assert outcome["status"] == "completed"
    assert reported == expected


def test_child_environment_keeps_auth_without_nested_session_markers(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "parent")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-placeholder-not-a-secret")
    env = protocol.child_environment()
    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "test-placeholder-not-a-secret"
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "CLAUDE_AGENT_REVIEW_GATE" not in env
    assert protocol.child_environment(review=True)["CLAUDE_AGENT_REVIEW_GATE"] == "1"


def test_review_command_ignores_ambient_grants_without_disabling_oauth(
        monkeypatch, tmp_path, review_host):
    monkeypatch.setattr(protocol.shutil, "which", lambda name: "fake-claude-executable")
    agent = {"session_id": str(uuid.uuid4()), "cwd": str(tmp_path),
             "options": {"profile": "review"}}
    argv = protocol.command(agent, resume=False)
    assert argv[argv.index("--session-id") + 1] == agent["session_id"]
    assert argv[argv.index("--permission-mode") + 1] == "dontAsk"
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    assert argv[argv.index("--setting-sources") + 1] == ""
    settings = json.loads(argv[argv.index("--settings") + 1])
    hooks = settings["hooks"]["PreToolUse"]
    assert set(settings) == {"hooks"} and set(settings["hooks"]) == {"PreToolUse"}
    assert len(hooks) == 1 and hooks[0]["matcher"] == "Bash|PowerShell|Monitor"
    # Bound to the host's gate, resolved under its root, not to $CLAUDE_PROJECT_DIR.
    assert hooks[0]["hooks"] == [{"type": "command", "command":
        f'"{(review_host / "gate/hooks/ask_user_gate.exe").resolve().as_posix()}"'}]
    grants = argv[argv.index("--allowedTools") + 1:argv.index("--setting-sources")]
    assert grants == ["Read", "Glob", "Grep",
                      *(f"Bash(git -C {SUBMODULE} {rule} *)" for rule in protocol.REVIEW_RULES)]
    for rule in grants[3:]:  # The wildcard follows the option terminator.
        terminator = " -- *)" if " grep " in rule else " --end-of-options *)"
        assert rule.count("*") == 1 and rule.endswith(terminator), rule
    instructions = argv[argv.index("--append-system-prompt") + 1]
    for rule in grants[3:]:  # The briefing spells out every template it may type.
        template = rule[len("Bash("):-len(" *)")].removeprefix("git ")
        assert template.removeprefix(f"-C {SUBMODULE} ") in instructions, rule
    assert "Core test note." in instructions
    assert "--strict-mcp-config" in argv and "--bare" not in argv
    assert "--dangerously-skip-permissions" not in argv
    resumed = protocol.command(agent, resume=True)
    assert resumed[resumed.index("--resume") + 1] == agent["session_id"]
    # A settings file whose hook entry changed is refused, not merged.
    (review_host / "settings.json").write_text(json.dumps({"hooks": {"PreToolUse": []}}))
    with pytest.raises(ValueError, match="no longer contain"):
        protocol.command(agent, resume=False)


@pytest.fixture(scope="module")
def review_git_repo(tmp_path_factory):
    """A real repository: one commit plus a modified worktree file to grep."""
    root = tmp_path_factory.mktemp("review git")
    env = os.environ.copy()
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)

    def git(*args):
        return subprocess.run(["git", *args], cwd=root, env=env, capture_output=True,
                              stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                              check=False)

    assert git("-c", "init.templateDir=", "init", "-q").returncode == 0
    (root / "sample.txt").write_text("review needle\n", encoding="utf-8")
    assert git("add", ".").returncode == 0
    assert git("-c", "user.name=Review Test", "-c", "user.email=review@example.invalid",
               "-c", "commit.gpgSign=false", "commit", "-qm", "review fixture").returncode == 0
    (root / "sample.txt").write_text("worktree needle\n", encoding="utf-8")
    marker = root / "helper-ran"
    (root / "helper.sh").write_text(f'#!/bin/sh\necho fired >> "{marker.as_posix()}"\n',
                                    encoding="utf-8")
    return root, marker, git


def review_template(rule):
    """Git arguments a grant fixes before its '*', minus `git` and its -C path."""
    words = rule[len("Bash(git "):-len(" *)")].split()
    return words[2:] if words[0] == "-C" else words


REVIEW_BASH = [rule for rule in protocol.review_allow(SUBMODULE) if rule.startswith("Bash(")]


@pytest.mark.parametrize("rule", REVIEW_BASH)
def test_review_git_templates_read_committed_objects(review_git_repo, rule):
    _, marker, git = review_git_repo
    args = review_template(rule)
    # The briefed operands: a hash, for grep a pattern first, a path after --.
    operands = (["needle", "HEAD", "--", "sample.txt"] if args[0] == "grep"
                else ["HEAD", "--", "sample.txt"])
    result = git(*args, *operands)
    # Red on git 2.43.0 if grep's terminator were `--end-of-options`
    # (`protocol.REVIEW_RULES`): "unable to resolve revision: needle".
    assert result.returncode == 0, (git("--version").stdout, result.stderr)
    expected = {"show": "review needle", "log": "review fixture", "grep": "HEAD:sample.txt:1"}
    assert ("sample.txt |" if "--stat" in args else expected[args[0]]) in result.stdout
    assert not marker.exists()


# Each row runs a program or writes a file once its option is parsed: the
# worktree grep matches `needle`, so -O starts helper.sh on sample.txt.
@pytest.mark.parametrize(("rule", "operands"), [
    *((rule, ["--output=written", "HEAD"]) for rule in REVIEW_BASH
      if review_template(rule)[0] in ("show", "log")),
    *((rule, operands) for rule in REVIEW_BASH if review_template(rule)[0] == "grep"
      # As the pattern, and after it: where `-e PATTERN` would let it through.
      for operands in (["-O./helper.sh", "needle"], ["needle", "-O./helper.sh"])),
])
def test_review_git_wildcard_cannot_reenable_options(review_git_repo, rule, operands):
    root, marker, git = review_git_repo
    for leftover in (root / "written", marker):  # A red row must not redden the next.
        leftover.unlink(missing_ok=True)
    result = git(*review_template(rule), *operands)
    assert not (root / "written").exists()
    assert not marker.exists()
    # The option became a bad revision, or an option after a non-option.
    assert result.returncode != 0, result.stdout


def test_review_refuses_to_start_without_compiled_gate(monkeypatch, review_host):
    original = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda path: False if path.name == "ask_user_gate.exe"
                        else original(path))
    with pytest.raises(FileNotFoundError, match="Review gate is missing"):
        protocol.verify_review_gate()


def test_review_turn_fails_before_launch_when_gate_is_missing(fake_cli, monkeypatch):
    run, root, agent = fake_cli
    agent["options"]["profile"] = "review"

    def missing_gate():
        raise FileNotFoundError("Review gate is missing")

    monkeypatch.setattr(protocol, "verify_review_gate", missing_gate)
    outcome = run()
    assert outcome["status"] == "failed"
    assert "Review gate is missing" in outcome["error"]
    assert not (root / "received.json").exists()


def test_review_profile_briefs_the_codex_reviewer_answer_contract(monkeypatch, review_host):
    """Both wrappers' reviews of one commit are merged before one fixer, so the
    review profile carries codex_agent's reviewer skeleton verbatim — the one
    that asks for missed siblings, a verdict and per-question coverage."""
    from llm_agents.answer_contract import REVIEWER_BRIEFING
    monkeypatch.setattr(protocol.shutil, "which", lambda name: "fake-claude-executable")
    agent = {"session_id": str(uuid.uuid4()), "options": {"profile": "review"}}
    argv = protocol.command(agent, resume=False)
    told = argv[argv.index("--append-system-prompt") + 1]
    assert told.endswith(REVIEWER_BRIEFING)
    headings = [line for line in told.splitlines() if line.startswith("## ")]
    assert headings == ["## Findings", "## Missed siblings", "## Verdict", "## Coverage"]
    assert "blocker / should / nit" in told
    # The shared briefing allows a requested gate; this profile's tools cannot run one.
    assert "report a gate the task asks you to run as not run" in told
    configured = protocol.command(agent | {"options": {"profile": "configured"}}, resume=False)
    assert "--append-system-prompt" not in configured


JOB_OWNER = r'''
import json
import os
from pathlib import Path
import subprocess
import sys

if sys.argv[1] == "grandchild":
    print(json.dumps({"grandchild": os.getpid()}), flush=True)
    sys.stdin.readline()
elif sys.argv[1] == "child":
    assert sys.stdin.readline().strip() == "assigned"
    grandchild = subprocess.Popen(
        [sys.executable, __file__, "grandchild"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    ready = json.loads(grandchild.stdout.readline())
    ready["child"] = os.getpid()
    print(json.dumps(ready), flush=True)
    sys.stdin.readline()
else:
    sys.path.insert(0, sys.argv[2])
    from llm_agents.child_job import ChildJob
    job = ChildJob()
    child = subprocess.Popen(
        [sys.executable, __file__, "child"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    job.assign(child)
    child.stdin.write("assigned\n")
    child.stdin.flush()
    print(child.stdout.readline().strip(), flush=True)
    sys.stdin.readline()
'''


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object lifetime contract")
def test_owner_crash_kills_assigned_child_and_grandchild(tmp_path):
    """Prove OS handle cleanup works when no Python finally block can run."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    helper = tmp_path / "job_owner.py"
    helper.write_text(JOB_OWNER, encoding="utf-8")
    owner = subprocess.Popen(
        [sys.executable, str(helper), "owner", str(Path(__file__).resolve().parents[2] / "src")],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", creationflags=subprocess.CREATE_NO_WINDOW)
    handles = []
    try:
        # Both descendants send readiness through pipes. The child cannot spawn
        # its grandchild until assignment, so this covers inherited membership.
        line = owner.stdout.readline()
        assert line, owner.stderr.read()
        ready = json.loads(line)
        for key in ("child", "grandchild"):
            handle = kernel.OpenProcess(0x100000 | 0x0001, False, ready[key])
            assert handle, ctypes.WinError(ctypes.get_last_error())
            handles.append(handle)
            assert kernel.WaitForSingleObject(handle, 0) == 258
        started = time.monotonic()
        owner.kill()  # TerminateProcess skips ChildJob.close and all finally blocks.
        owner.wait()
        for handle in handles:
            # 0.0053 s measured 2026-09-08 for both exits; allow 10 s per handle.
            assert kernel.WaitForSingleObject(handle, 10_000) == 0
        print(f"Owner crash killed child and grandchild in {time.monotonic() - started:.4f}s")
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.wait()
        for handle in handles:
            if kernel.WaitForSingleObject(handle, 0) == 258:
                kernel.TerminateProcess(handle, 1)
            kernel.CloseHandle(handle)
        for stream in (owner.stdin, owner.stdout, owner.stderr):
            stream.close()
