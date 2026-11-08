"""Gates the wrapper runs itself (`spawn --gates-from-prompt`, `gates.py`).

The wire is the offline stand-in (`llm_agents.codex.testing.wire`); the command
runner is a fake (`wire.GateRunner`), except in the tests about
`gates.run_gate` itself.
"""

import sys
from pathlib import Path

import pytest

from llm_agents.codex.testing import wire
# Fixture registration: see no_badge's docstring.
from llm_agents.codex.testing.fixtures import no_badge, worker_report  # noqa: F401
from llm_agents.codex.testing.wire import GATES, GATES_PROMPT as PROMPT
from llm_agents.codex import cli, events, gates, paths, sandbox, turn
from llm_agents.codex.registry import AgentRecord, Registry
from llm_agents.codex.store import AgentStore
from core_policy import core_main


def test_prompt_gates_reads_only_fenced_lines_under_the_gates_heading():
    assert gates.prompt_gates(PROMPT) == GATES
    # Bullets and inline code are prose: the flag must not guess commands.
    assert gates.prompt_gates("## Gates\n- `npm test`\n") == []
    with pytest.raises(gates.GatesError):
        gates.required_gates("## Gates\n- `npm test`\n")


def test_required_gates_refuses_a_second_section_and_powershell_syntax():
    template = PROMPT + "\n## Report\n### Gates\n```\nnpm run example\n```\n"
    with pytest.raises(gates.GatesError, match="2 `Gates` sections"):
        gates.required_gates(template)
    # A second Gates heading with no commands under it is prose, not a gate.
    assert gates.required_gates(PROMPT + "\n### Gates\nlist them here\n") == GATES
    for line in (r"python -m pytest x --basetemp $env:TEMP\t", "npm test `"):
        with pytest.raises(gates.GatesError, match="PowerShell"):
            gates.required_gates(f"## Gates\n```\n{line}\n```\n")


def test_derived_escalation_lets_out_the_gates_and_nothing_else(tmp_path):
    record = AgentRecord(name="w", cwd=str(tmp_path), sandbox="workspace-write",
                         gates=GATES)
    judge = sandbox.escalation_judge(record)
    assert judge is not None

    def asks(command, cwd=tmp_path):
        return judge({"command": command, "cwd": str(cwd)})

    assert asks("npm run typecheck")
    assert asks("npm  run typecheck", tmp_path / "app")
    # Either slash; the line exactly, nothing appended.
    assert asks(r"python tools\check\rename_gate.py oldName")
    assert asks("npm test -- test/hash.test.ts ")
    powershell = "\"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\""
    cmd = "C:\\WINDOWS\\system32\\cmd.exe /c"
    assert asks(f"{powershell} -Command 'npm run typecheck'")
    assert not asks("npm run typecheck; Remove-Item -Recurse src")
    assert not asks("npm run typecheck | Out-File x")
    assert not asks("npm run typecheckx")
    assert not asks("npm test")
    assert not asks("node -e 1")
    # A second command after a line break or inside a PowerShell expression,
    # raw and as the shell wrapper delivers it (review 2026-09-30).
    for smuggled in ("npm run typecheck\nWrite-Output x",
                     "npm run typecheck\r\nWrite-Output x",
                     "npm run typecheck (Write-Output x)",
                     "npm run\ntypecheck"):
        assert not asks(smuggled), smuggled
        assert not asks(f"{powershell} -Command '{smuggled}'"), smuggled
        assert not asks(f"{cmd} \"{smuggled}\""), smuggled
    # Literal arguments escape too: another shell, another tree, agent code.
    for extra in ("npm test -- test/hash.test.ts --script-shell=evil.exe",
                  "npm test -- test/hash.test.ts --prefix C:\\other",
                  "npm test -- test/hash.test.ts --reporter=dot",
                  "npm run typecheck -p evil"):
        assert not asks(extra), extra
    params = sandbox.thread_params(record)
    assert params["approvalPolicy"] == "on-request"
    assert "EXCEPTION" in params["developerInstructions"]
    # The operator's own rule still applies beside the derived one.
    both = sandbox.escalation_judge(AgentRecord(
        name="w", cwd=str(tmp_path), sandbox="workspace-write",
        gates=GATES, escalate=r"^npx tsx scripts[/\\]census\.ts$"))
    assert both({"command": "npx tsx scripts/census.ts", "cwd": str(tmp_path)})
    assert both({"command": "npm run typecheck", "cwd": str(tmp_path)})
    assert sandbox.escalation_judge(AgentRecord(name="w", cwd=str(tmp_path))) is None


def test_run_gate_reports_exit_output_and_timeout(tmp_path):
    python = f'"{sys.executable}"'
    ok = gates.run_gate(f"{python} -c \"print('hello'); raise SystemExit(3)\"",
                        tmp_path, 60.0)
    assert ok.exit_code == 3 and "hello" in ok.output and not ok.passed
    # 1 s bound on a 30 s sleep: the timeout path must return, not hang on
    # the child's output.
    slow = gates.run_gate(f"{python} -c \"import time; time.sleep(30)\"",
                          tmp_path, 1.0)
    assert slow.exit_code is None and "timed out after 1s" in slow.output
    assert slow.duration_s < 25
    missing = gates.run_gate("npm test", tmp_path / "absent", 60.0)
    assert missing.exit_code is None and "does not exist" in missing.output


def test_gate_runner_preserves_explicit_prefix_and_checks_its_exit(tmp_path):
    command = "npm --prefix app run typecheck"
    runner = wire.GateRunner({command: [1]})
    results = gates.run_gates([command], tmp_path, 60.0, runner, lambda _: None)
    assert runner.calls == [(command, tmp_path)]
    assert results[0].exit_code == 1 and not results[0].passed


def test_run_gate_reaps_the_process_when_the_wait_is_interrupted(
        tmp_path, monkeypatch):
    class _Proc:
        def wait(self, timeout):
            raise KeyboardInterrupt

    proc = _Proc()
    reaped = []
    monkeypatch.setattr(gates.subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(gates, "_reap", reaped.append)
    with pytest.raises(KeyboardInterrupt):
        gates.run_gate("npm test", tmp_path, 60.0)
    assert reaped == [proc]


def test_gates_share_one_budget(tmp_path):
    import time
    seen = []

    def slow(command, cwd, timeout):
        seen.append(timeout)
        time.sleep(0.2)
        return gates.GateRun(command, str(cwd), 0, "", 0.2)

    logged = []
    results = gates.run_gates(["npm test", "npm run typecheck"], tmp_path, 0.1,
                              slow, logged.append)
    # `deadline - now` with no tick between them is the budget plus float
    # rounding: `(t + 0.1) - t` read 0.10000000000582 on Windows 3.9/3.10,
    # where `t` is a large monotonic value and its last bits are coarse. The
    # 1e-6 slack covers that rounding only, not a clock step between reads.
    assert len(seen) == 1 and seen[0] <= 0.1 + 1e-6
    assert results[0].passed
    assert results[1].exit_code is None and "budget" in results[1].output
    assert logged[-1]["skipped"] is True


def test_a_red_gate_is_followed_up_once_with_its_tail_and_rerun(
        tmp_path, monkeypatch, capsys):
    runner = wire.GateRunner({GATES[0]: [1, 0], GATES[1]: [0], GATES[2]: [0]})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "plain", runner)
    assert outcome.status == "completed" and not outcome.error
    assert outcome.gate_follow_ups == 1 and len(runner.calls) == 6
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2
    assert starts[1]["params"]["threadId"] == "th-fake"
    follow_up = starts[1]["params"]["input"][0]["text"]
    assert "1 of 3 failed" in follow_up and "`npm run typecheck`" in follow_up
    assert "FAILED boom" in follow_up
    # The tail, not the whole output.
    assert follow_up.count("line\n") == gates.GATE_TAIL_LINES - 1
    assert "(3/3 green, after 1 gate follow-up)" in outcome.text
    assert cli.print_outcome("w", outcome, quiet=False) == 0
    trace = AgentStore("w").last_turn()
    # Counted apart from an --expect follow-up, as the outcome counts it.
    assert trace.ended and trace.gate_follow_ups == 1 and trace.follow_ups == 0
    assert [line.get("kind") for line in AgentStore("w").events()
            if line.get("_meta") == events.FOLLOWUP_META] == [events.FOLLOWUP_GATES]
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "w", "--tail", "0"]))
    logged = capsys.readouterr().out
    assert "--- follow-up: red gates npm run typecheck" in logged
    assert "--- gate: npm run typecheck -> exit 1 (1.0s)" in logged


def test_a_follow_up_line_without_a_kind_reads_as_before(
        tmp_path, monkeypatch, capsys):
    # Logs written before `kind` existed told the two apart by the `gates`
    # list alone; they must still read that way in `logs` and `status`.
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="old", thread_id="t1"))
    legacy_gates = {"_meta": events.FOLLOWUP_META, "gates": ["npm test"]}
    legacy_expect = {"_meta": events.FOLLOWUP_META, "expect": "^## X"}
    AgentStore("old").append_events([
        {"_meta": events.TURN_META, "name": "old", "prompt": "p",
         "at": "2026-10-02 10:00:00", "timeout": 600.0},
        legacy_expect, legacy_gates, legacy_gates])
    cli.cmd_status(cli.build_parser().parse_args(["status", "old"]))
    printed = capsys.readouterr().out
    assert ("RUNNING since 2026-10-02 10:00:00 "
            "(0 cmd, 0 edits, 1 follow-up, 2 gate follow-up)") in printed
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "old", "--tail", "0"]))
    logged = capsys.readouterr().out
    assert logged.count("--- follow-up: the answer lacked /^## X/") == 1
    assert logged.count("--- follow-up: red gates npm test") == 2


def test_an_unknown_follow_up_kind_is_refused():
    # A third kind left out of `_FOLLOWUP_KINDS` must not pass for one of the
    # two, whichever keys its line happens to carry.
    for keys in ({}, {"gates": ["npm test"]}, {"expect": "^## X"}):
        with pytest.raises(ValueError, match="unknown follow-up kind 'retry'"):
            events.follow_up_kind({"_meta": events.FOLLOWUP_META,
                                   "kind": "retry", **keys})


def test_a_gate_still_red_after_the_follow_up_fails_the_call(
        tmp_path, monkeypatch, capsys):
    runner = wire.GateRunner({GATES[0]: [1], GATES[1]: [0], GATES[2]: [2]})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "plain", runner)
    assert len(wire.turn_starts(tmp_path)) == 2  # one follow-up, not a loop
    assert "(1/3 green, after 1 gate follow-up)" in outcome.text
    assert "- FAIL `npm test -- test/hash.test.ts`" in outcome.text
    assert "FAILED boom" in outcome.text
    assert cli.print_outcome("w", outcome, quiet=False) == 1
    assert "gates 1/3 green, after 1 gate follow-up" in capsys.readouterr().out


def test_the_gate_follow_up_answer_is_held_to_expect(tmp_path, monkeypatch):
    # "partial": the first answer has `## Findings`, every later one only
    # `## Coverage` — so the gate follow-up's replacement drops the section.
    runner = wire.GateRunner({GATES[0]: [1, 0], GATES[1]: [0], GATES[2]: [0]})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "partial", runner,
                          expect=r"^## Findings")
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 3  # the turn, the gate follow-up, the expect one
    assert "## Findings" in starts[2]["params"]["input"][0]["text"]
    assert outcome.follow_ups == 1 and outcome.gate_follow_ups == 1
    # Judged again after the expect follow-up inside the gate path: neither
    # the gate follow-up's answer nor its Coverage-only addition has Findings.
    assert outcome.expect_met is False
    # The gates ran after the last turn that could edit the tree.
    assert len(runner.calls) == 2 * len(GATES)
    assert outcome.text.index("## Wrapper gates (3/3 green") > \
        outcome.text.rindex("## Coverage")
    # Each writer stamps its own kind, and the trace counts them apart.
    assert _follow_up_kinds() == [events.FOLLOWUP_GATES, events.FOLLOWUP_EXPECT]
    trace = AgentStore("w").last_turn()
    assert trace.follow_ups == 1 and trace.gate_follow_ups == 1


def test_expect_then_gate_then_expect_follow_ups_are_counted_apart(
        tmp_path, monkeypatch):
    # "partial-twice": no answer ever has `## Coverage`, so the first answer
    # gets an expect follow-up, the red gate one more, and its replacement an
    # expect follow-up again inside `_check_gates`.
    runner = wire.GateRunner({GATES[0]: [1, 0], GATES[1]: [0], GATES[2]: [0]})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "partial-twice", runner,
                          expect=r"^## Coverage")
    assert len(wire.turn_starts(tmp_path)) == 4
    assert outcome.follow_ups == 2 and outcome.gate_follow_ups == 1
    assert _follow_up_kinds() == [events.FOLLOWUP_EXPECT, events.FOLLOWUP_GATES,
                                  events.FOLLOWUP_EXPECT]
    trace = AgentStore("w").last_turn()
    assert trace.ended and trace.follow_ups == 2 and trace.gate_follow_ups == 1


def _follow_up_kinds(name="w"):
    return [line.get("kind") for line in AgentStore(name).events()
            if line.get("_meta") == events.FOLLOWUP_META]


def test_a_failed_gate_follow_up_keeps_the_incomplete_verdict(
        tmp_path, monkeypatch, capsys):
    # Both answers lack `## Coverage`, then the gate follow-up fails: the kept
    # answer is still the incomplete one, so the Incomplete line must survive.
    runner = wire.GateRunner({GATES[0]: [1], GATES[1]: [0], GATES[2]: [0]})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "partial-twice-then-refuse",
                          runner, expect=r"^## Coverage")
    assert len(wire.turn_starts(tmp_path)) == 3
    assert "gate follow-up failed" in outcome.error
    assert outcome.follow_ups == 1 and outcome.expect_met is False
    cli.print_outcome("w", outcome, quiet=True)
    assert "Incomplete" in capsys.readouterr().out


def test_a_failed_gate_follow_up_keeps_the_expect_note(
        tmp_path, monkeypatch, capsys, worker_report):
    # The expect follow-up answers in the message (the note), the gate is red,
    # and the gate follow-up says something before it fails: the kept answer
    # keeps its whole provenance, the note included.
    record, report = worker_report
    runner = wire.GateRunner({GATES[0]: [1], GATES[1]: [0], GATES[2]: [0]})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "report-fallback-then-fail",
                          runner, record=record, expect=r"^### Not done")
    assert len(wire.turn_starts(tmp_path)) == 3
    assert "gate follow-up failed" in outcome.error
    assert outcome.report_source == "file" and outcome.follow_ups == 1
    assert outcome.report_note == "follow-up answered in the message"
    cli.print_outcome("w", outcome, quiet=False)
    assert outcome.report_note in capsys.readouterr().out.splitlines()[0]


def test_the_gate_follow_up_report_file_is_held_to_expect(
        tmp_path, monkeypatch, worker_report):
    record, report = worker_report
    runner = wire.GateRunner({GATES[0]: [1, 0], GATES[1]: [0], GATES[2]: [0]})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "report-gate-regress", runner,
                          record=record, expect=r"^### Not done")
    assert len(wire.turn_starts(tmp_path)) == 3
    assert outcome.status == "completed" and not outcome.error
    assert outcome.report_source == "file" and outcome.follow_ups == 1
    assert outcome.text.startswith(report.read_text(encoding="utf-8").rstrip())
    assert "### Not done / open questions" in outcome.text
    assert "(3/3 green, after 1 gate follow-up)" in outcome.text


def test_a_turn_that_did_not_complete_runs_no_gates(tmp_path, monkeypatch):
    runner = wire.GateRunner({command: [0] for command in GATES})
    outcome = wire.gated_turn(tmp_path, monkeypatch, "refuse", runner)
    assert outcome.status == "failed" and runner.calls == []
    assert outcome.gate_note == "gates not run: the turn did not complete"
    assert "Wrapper gates" not in outcome.text


def test_follow_up_prompt_names_the_report_file():
    red = [gates.GateRun("npm test", "w", 1, "boom"),
           gates.GateRun("npm run typecheck", "w", 0, "")]
    prompt = gates.follow_up_prompt(red, Path("C:/scratch/report.md"))
    assert "C:" in prompt and "report.md" in prompt and "1 of 2 failed" in prompt
    assert "npm run typecheck" not in prompt
    assert "final message" in gates.follow_up_prompt(red, None)


def test_spawn_records_the_gates_and_send_reruns_them(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    wire.install_fake_server(monkeypatch, tmp_path, "plain")
    runner = wire.GateRunner({command: [0] for command in GATES})
    monkeypatch.setattr(gates, "run_gate", runner)
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text(PROMPT, encoding="utf-8")
    assert core_main(["spawn", "w", "--prompt-file", str(prompt_file),
                     "--cwd", str(tmp_path), "--sandbox", "read-only",
                     "--gates-from-prompt"]) == 0
    assert Registry().get("w").gates == GATES
    assert core_main(["send", "w", "fix one more thing"]) == 0
    assert len(runner.calls) == 2 * len(GATES)
    cli.cmd_status(cli.build_parser().parse_args(["status", "w"]))
    assert "3 gates" in capsys.readouterr().out


def test_spawn_refuses_the_flag_without_gates_before_claiming_the_name(
        tmp_path, monkeypatch):
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    with pytest.raises(SystemExit, match="no fenced command block"):
        core_main(["spawn", "w", "task without gates", "--cwd", str(tmp_path),
                  "--gates-from-prompt"])
    assert Registry().get("w") is None
    assert not AgentStore("w").has_events()
