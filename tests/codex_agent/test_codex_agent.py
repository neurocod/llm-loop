"""Regression tests for the delegated Codex worker (`llm_agents.codex`).

Everything here runs offline: the real `codex` CLI is never started. What talks
to the client instead is a stand-in app-server (`llm_agents.codex.testing.wire`)
launched as a REAL child process, so the wire tests exercise the actual pipes,
encoding and line buffering rather than a mock's idea of them.

What is pinned is the part that fails SILENTLY in production — the registry
shared by parallel agents, and the request/approval shapes, where a wrong field
name reads as "the agent had nothing to say" rather than as an error.

Every monkeypatch below names the module that LOOKS THE TARGET UP, which is the
CONSUMER, not the one that defines it: the package is written in `from x import
y` style, so each consumer holds its own binding and rebinding `turn.run_turn`
leaves `cli.run_turn` pointing at the original. A patch on the wrong side does
not fail — it passes while the real code runs, which for these tests means
spending a real Codex turn per run.

Where a name's definer and its only consumer are the same module
(`protocol.subprocess`, `registry.LOCK_WAIT_S`), both readings coincide; that is
a coincidence, not the rule.
"""

import dataclasses
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

from llm_agents.codex import (
    cli, events, gates as gates_module, paths, protocol,
    registry as registry_module, sandbox, service, subagent_wait, subagents,
    turn,
)
from llm_agents import agent_counter as beacon
from llm_agents import host_policy
from llm_agents.answer_contract import followed_up_note
from llm_agents.codex.paths import agent_name
from llm_agents.codex.registry import AgentRecord, Registry, RegistryError
from llm_agents.codex.store import AgentStore
from llm_agents.codex.turn import TurnOutcome, TurnRunner
from llm_agents.codex.testing import wire
from llm_agents.codex.testing.fixtures import no_badge, worker_report  # noqa: F401
from llm_agents.codex.testing.wire import (
    LOGGED_DROP, LOGGED_NETWORK_WAIT, LOGGED_OVERLOAD, LOGGED_RECONNECT,
    NETWORK_WAIT_TIMES,
)
from core_policy import core_main

WRITERS = 6


@pytest.fixture
def network_clock(monkeypatch):
    """Real fake-server pipes, virtual elapsed time from the recorded events.

    All three deadline consumers see the same monotonic clock. Startup keeps
    protocol.STARTUP_TIMEOUT_S; virtual work budgets below are scenario
    boundaries, not guesses about how fast this machine runs.
    """
    now = [time.monotonic()]
    clock = SimpleNamespace(monotonic=lambda: now[0], strftime=time.strftime)
    for module in (turn, protocol, subagent_wait):
        monkeypatch.setattr(module, "time", clock)
    original_take = protocol.AppServer._take

    def timed_take(server, deadline, what):
        message = getattr(server, "_test_pending_message", None)
        if message is None:
            message = original_take(server, deadline, what)
        if "emittedAtMs" in message:
            if not hasattr(server, "_test_network_epoch"):
                server._test_network_epoch = now[0]
            arrives = (server._test_network_epoch
                       + (message["emittedAtMs"] - NETWORK_WAIT_TIMES[0]) / 1000)
            if arrives > deadline:
                # Model the empty queue waking BEFORE the next event. Merely
                # jumping the clock in on_event skips this deadline entirely.
                server._test_pending_message = message
                now[0] = deadline
                raise TimeoutError(f"codex app-server did not answer {what} in time")
            now[0] = arrives
        server._test_pending_message = None
        return message

    monkeypatch.setattr(protocol.AppServer, "_take", timed_take)
    original = TurnRunner.on_event
    progress = []

    def timed_event(runner, message):
        original(runner, message)
        if runner.network.started is not None:
            store = AgentStore("w")
            progress.append(cli._turn_in_progress(store, store.last_turn()))

    monkeypatch.setattr(TurnRunner, "on_event", timed_event)
    return progress


def test_network_wait_completes_beyond_attempt_budget(
        tmp_path, monkeypatch, network_clock, capsys):
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, "network-complete", timeout=100,
        reachable=lambda: False, sleep=lambda _: None)
    assert outcome.status == "completed" and not outcome.error
    assert outcome.service_retries == 0
    assert outcome.network_wait_s == pytest.approx(253.868)
    assert outcome.network_wait_s > 100
    assert any("RECONNECTING since " in line and "(5 tries)" in line
               for line in network_clock)
    assert cli.print_outcome("w", outcome, quiet=False) == 0
    assert "waited 4.2 min for network" in capsys.readouterr().out
    assert core_main(["logs", "w", "--tail", "0"]) == 0
    printed = capsys.readouterr().out
    assert "RECONNECTING since " in printed and "(5 tries)" in printed
    assert "waited 4.2 min for network" in printed
    assert AgentStore("w").last_turn().reconnecting_since == ""


def test_network_wait_ceiling_is_terminal(tmp_path, monkeypatch, network_clock):
    monkeypatch.setattr(service, "MAX_NETWORK_WAIT_S", 100.0)
    checks = []
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, "network-ceiling", timeout=100,
        reachable=lambda: checks.append(True), sleep=lambda _: None)
    assert outcome.status == "failed"
    assert "no network for 1.7 min (wait limit reached)" in outcome.error
    assert outcome.network_wait_s == 100
    assert not checks and outcome.service_retries == 0


def test_network_wait_timeout_resends_same_thread(
        tmp_path, monkeypatch, network_clock):
    checks, pauses = [], []
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, "network-stale", timeout=100,
        reachable=lambda: checks.append(True) or True, sleep=pauses.append)
    assert outcome.status == "completed" and not outcome.error
    assert outcome.service_retries == 1
    assert outcome.network_wait_s == 366
    assert checks == [True] and pauses == [turn.SERVICE_RETRY_PAUSES_S[0]]
    resumed = [m for m in wire.received(tmp_path) if m.get("method") == "thread/resume"]
    assert len(resumed) == 1 and resumed[0]["params"]["threadId"] == "th-fake"
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2 and starts[1]["params"]["threadId"] == "th-fake"
    prompt = starts[1]["params"]["input"][0]["text"]
    assert prompt == turn.NETWORK_RETRY_PROMPT
    assert "network outage" in prompt and "git diff --stat" in prompt
    assert "unable to verify model access" not in prompt


@pytest.mark.parametrize("scenario", ["network-429", "network-stream-drop"])
def test_non_network_retries_keep_attempt_budget(
        tmp_path, monkeypatch, network_clock, scenario):
    checks = []
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, scenario, timeout=100,
        reachable=lambda: checks.append(True), sleep=lambda _: None)
    assert outcome.status == "timeout"
    assert "did not answer turn in time" in outcome.error
    assert outcome.network_wait_s == 0 and not network_clock
    assert outcome.service_retries == 0 and not checks


def test_network_wait_ceiling_is_shared_by_resends(
        tmp_path, monkeypatch, network_clock):
    monkeypatch.setattr(service, "MAX_NETWORK_WAIT_S", 600.0)
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, "network-ceiling-across-retries", timeout=100,
        reachable=lambda: True, sleep=lambda _: None)
    assert outcome.status == "failed" and outcome.service_retries == 1
    assert "no network for 10.0 min (wait limit reached)" in outcome.error
    assert outcome.network_wait_s == 600


def test_network_wait_recovery_keeps_only_elapsed_credit():
    wait = service.NetworkWait(main_thread=LOGGED_NETWORK_WAIT["params"]["threadId"])
    wait.observe(LOGGED_NETWORK_WAIT, 10, "2026-11-07 12:30:00")
    # A quiet queue must wait for the next heartbeat even with only 20 s of
    # work left; on recovery it gets back only the 60 s actually waited.
    assert wait.deadline(30, 0, 10) == 376
    wait.observe({"method": "item/started", "params": {
        "threadId": LOGGED_NETWORK_WAIT["params"]["threadId"]}},
        70, "2026-11-07 12:31:00")
    assert wait.deadline(30, 0, 70) == 90
    assert wait.seconds(1000) == 60 and wait.started is None
    wait.observe(LOGGED_NETWORK_WAIT, 80, "2026-11-07 12:32:00")
    assert wait.tries == 1 and wait.since == "2026-11-07 12:32:00"
    wait.observe({"method": "error", "params": {**LOGGED_RECONNECT["params"],
                 "threadId": LOGGED_NETWORK_WAIT["params"]["threadId"]}},
                 90, "2026-11-07 12:32:10")
    assert wait.seconds(1000) == 70 and wait.started is None


def test_network_wait_completion_at_ceiling_keeps_answer(
        tmp_path, monkeypatch, network_clock):
    monkeypatch.setattr(service, "MAX_NETWORK_WAIT_S", 100)
    checks = []
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, "network-complete-at-ceiling", timeout=100,
        reachable=lambda: checks.append(True), sleep=lambda _: None)
    assert outcome.status == "completed" and not outcome.error
    assert outcome.text == "complete answer"
    assert outcome.network_wait_s == 100 and not checks


@pytest.mark.parametrize("gap", [366.0, 128.0])
def test_network_wait_rev1277_recovers_without_resend(
        tmp_path, monkeypatch, network_clock, gap):
    # The measured gap is covered by the production allowance; even the old,
    # under-measured allowance must only stop crediting, not kill the turn.
    monkeypatch.setattr(service, "NETWORK_WAIT_GAP_S", gap)
    checks = []
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, "network-rev-d", timeout=1800,
        reachable=lambda: checks.append(True), sleep=lambda _: None)
    assert outcome.status == "completed" and not outcome.error
    assert outcome.text == "recovered without a resend"
    assert outcome.service_retries == 0 and not checks
    assert outcome.network_wait_s == pytest.approx(min(gap, 182.785) + 77.717)


def test_network_wait_staleness_keeps_remaining_work_budget():
    wait = service.NetworkWait(main_thread=LOGGED_NETWORK_WAIT["params"]["threadId"])
    wait.observe(LOGGED_NETWORK_WAIT, 187.371, "2026-10-15 12:30:00")
    # Wake on the freshness deadline, as an empty protocol queue does.
    end = wait.deadline(1800, 0, 187.371)
    assert end == pytest.approx(553.371)
    assert wait.deadline(1800, 0, end) == pytest.approx(2166)
    assert wait.started is None
    with pytest.raises(service.NetworkWaitTimeout, match="attempt budget expired"):
        wait.deadline(1800, 0, 2166)
    # Recovery of the newest waiting thread must not retract credit already
    # earned while it was fresh, even if the remaining thread is stale.
    shared = service.NetworkWait()
    shared.observe(LOGGED_NETWORK_WAIT, 0, "2026-10-15 12:30:00")
    newer = {**LOGGED_NETWORK_WAIT, "params": {
        **LOGGED_NETWORK_WAIT["params"], "threadId": "newer"}}
    shared.observe(newer, 100, "2026-10-15 12:31:40")
    shared.observe({"method": "item/started", "params": {"threadId": "newer"}},
                   400, "2026-10-15 12:36:40")
    assert shared.seconds(1000) == 400 and shared.started is None


def test_sub_agent_only_network_wait_extends_main_budget(
        tmp_path, monkeypatch, network_clock):
    monkeypatch.setattr(AgentStore, "events_silent_s", lambda _: service.NETWORK_WAIT_GAP_S)
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, "network-child-only", timeout=100,
        reachable=lambda: False, sleep=lambda _: None)
    assert outcome.status == "completed" and not outcome.error
    assert outcome.network_wait_s == pytest.approx(253.868)
    assert outcome.service_retries == 0
    assert any("(5 tries)" in line for line in network_clock)
    assert all("process was killed?" not in line for line in network_clock)


@pytest.mark.parametrize("scenario, credit", [
    ("network-child-stale", 366.0),
    ("network-child-heartbeats", 732.0),
])
def test_child_wait_with_busy_main_times_out_without_resend(
        tmp_path, monkeypatch, network_clock, scenario, credit):
    checks = []
    outcome = wire.run_fake_turn(
        tmp_path, monkeypatch, scenario, timeout=1800,
        reachable=lambda: checks.append(True) or True, sleep=lambda _: None)
    assert outcome.status == "timeout"
    assert "did not answer turn in time" in outcome.error
    assert not outcome.network_wait_timeout
    assert outcome.network_wait_s == credit
    assert outcome.service_retries == 0 and not checks
    assert len(wire.turn_starts(tmp_path)) == 1


@pytest.mark.parametrize("waiting_thread", ["main", "child"])
def test_only_main_pending_makes_deadline_retryable(waiting_thread):
    wait = service.NetworkWait(main_thread="main")
    event = {**LOGGED_NETWORK_WAIT, "params": {
        **LOGGED_NETWORK_WAIT["params"], "threadId": waiting_thread}}
    wait.observe(event, 10, "2026-11-07 12:30:00")
    assert wait.deadline(1800, 0, 376) == 2166
    if waiting_thread == "main":
        with pytest.raises(service.NetworkWaitTimeout) as caught:
            wait.deadline(1800, 0, 2166)
        assert not caught.value.exhausted
    else:
        assert wait.deadline(1800, 0, 2166) == 2166


def test_child_credit_cap_survives_recovery_and_preserves_main_credit():
    wait = service.NetworkWait(main_thread="main")
    event = {**LOGGED_NETWORK_WAIT, "params": {
        **LOGGED_NETWORK_WAIT["params"], "threadId": "child"}}
    for start in (0, 300, 600):
        wait.observe(event, start, "2026-11-07 12:30:00")
        wait.observe({"method": "item/started", "params": {"threadId": "child"}},
                     start + 300, "2026-11-07 12:35:00")
    assert wait.seconds(900) == 732
    event["params"]["threadId"] = "main"
    wait.observe(event, 900, "2026-11-07 12:45:00")
    wait.observe({"method": "item/started", "params": {"threadId": "main"}},
                 1000, "2026-11-07 12:46:40")
    assert wait.deadline(1800, 0, 1000) == 1800 + 732 + 100
    runner = TurnRunner()
    runner.network = wait
    runner.outcome.network_wait_s = wait.seconds(1000)
    resumed = TurnRunner(earlier=runner)
    resumed.network.observe(event, 1100, "2026-11-07 12:48:20", "other-main")
    assert resumed.network.seconds(1200) == 832


# --- the wire -------------------------------------------------------------


@pytest.mark.parametrize("scenario, source, differs", [
    ("report-file", "file", True),
    ("report-same", "file", False),
    ("report-newline", "file", False),
    ("report-absent", "message", None),
    ("report-empty", "message", None),
])
def test_worker_answer_source_and_stored_result(
        tmp_path, monkeypatch, capsys, worker_report, scenario, source, differs):
    record, report = worker_report
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, scenario,
                             sandbox=record.sandbox, role=record.role)
    message = "## Report\n### Not done / open questions\nnone"
    expected = report.read_text(encoding="utf-8") if source == "file" else message
    assert outcome.status == "completed" and not outcome.error
    assert outcome.text == expected and outcome.report_source == source
    notes = [m for m in AgentStore("w").events()
             if m.get("_meta") == events.REPORT_META]
    assert notes == [{"_meta": events.REPORT_META, "name": "w",
                      "status": "completed",
                      "path": str(report), "source": source, "differs": differs,
                      **({"read_error": "FileNotFoundError"}
                         if scenario == "report-absent" else {})}]
    assert AgentStore("w").read_answer() == expected
    cli.cmd_result(cli.build_parser().parse_args(["result", "w"]))
    assert capsys.readouterr().out == expected
    cli.print_outcome("w", outcome, quiet=False)
    assert f" · {source}" in capsys.readouterr().out.splitlines()[0]
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "w"]))
    report_line = f"--- report: {source}"
    if differs:
        report_line += " (differs from the message)"
    assert report_line in capsys.readouterr().out.splitlines()
    assert AgentStore("w").last_turn().ended
    briefing = wire.first_request(tmp_path, "thread/start")["params"]["developerInstructions"]
    assert str(report) in briefing


@pytest.mark.parametrize("scenario", ["report-utf8-bom", "report-utf16", "report-cp1252"])
def test_worker_report_decodes_shell_encodings(
        tmp_path, monkeypatch, worker_report, scenario):
    record, report = worker_report
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, scenario,
                             sandbox=record.sandbox, role=record.role)
    expected = "## Report\n### Done\nEdited café.\n### Not done / open questions\nnone\n"
    if scenario == "report-cp1252":
        expected = expected.replace("é", "\ufffd")
    assert outcome.text.replace("\r\n", "\n") == expected
    assert outcome.report_source == "file"
    assert AgentStore("w").read_answer() == expected


def test_worker_report_directory_falls_back_to_message(
        tmp_path, monkeypatch, worker_report):
    record, report = worker_report
    report.parent.mkdir(parents=True, exist_ok=True)
    report.mkdir()
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "report-directory",
                             sandbox=record.sandbox, role=record.role)
    assert outcome.status == "completed" and not outcome.error
    assert outcome.report_source == "message"
    assert outcome.text == "## Report\n### Not done / open questions\nnone"
    assert AgentStore("w").read_answer() == outcome.text
    # Windows reports PermissionError for directories; POSIX uses IsADirectoryError.
    try:
        report.read_bytes()
    except OSError as exc:
        expected_error = type(exc).__name__
    notes = list(AgentStore("w").events())
    selection = next(m for m in notes if m.get("_meta") == events.REPORT_META)
    assert selection["read_error"] == expected_error
    assert any(m.get("_meta") == "report_unlink_error" for m in notes)


@pytest.mark.parametrize("scenario", ["report-partial", "report-missing"])
def test_worker_expect_reads_the_file_and_requests_a_complete_file(
        tmp_path, monkeypatch, worker_report, scenario):
    record, report = worker_report
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, scenario,
                             sandbox=record.sandbox, role=record.role,
                             expect="^### Not done")
    assert outcome.status == "completed" and outcome.follow_ups == 1
    assert outcome.text == report.read_text(encoding="utf-8")
    assert outcome.text.count("## Report") == 1
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2
    prompt = starts[1]["params"]["input"][0]["text"]
    assert str(report) in prompt and "COMPLETE report" in prompt
    assert "finish it first" in prompt
    assert "without repeating" not in prompt


def test_worker_send_does_not_reuse_the_previous_report(
        tmp_path, monkeypatch, worker_report):
    record, report = worker_report
    first = wire.run_fake_turn(tmp_path, monkeypatch, "report-stale",
                           sandbox=record.sandbox, role=record.role)
    assert first.report_source == "file" and report.exists()
    record.thread_id = first.thread_id
    second = turn.run_turn(record=record, prompt="next task", resume=True)
    assert second.status == "completed" and second.report_source == "message"
    assert second.text != first.text and not report.exists()
    notes = [m for m in AgentStore("w").events()
             if m.get("_meta") == events.REPORT_META]
    assert [m["source"] for m in notes] == ["file", "message"]


@pytest.mark.parametrize("failed", [False, True])
def test_worker_followup_without_a_file_keeps_one_answer(
        tmp_path, monkeypatch, capsys, worker_report, failed):
    record, report = worker_report
    outcome = wire.run_fake_turn(tmp_path, monkeypatch,
                             "report-failed" if failed else "report-fallback",
                             sandbox=record.sandbox, role=record.role,
                             expect="^### Not done")
    assert outcome.follow_ups == 1 and not report.exists()
    if failed:
        assert outcome.status == "failed" and "first answer is kept" in outcome.error
        assert outcome.report_source == "file"
        assert outcome.text == "## Report\n### Done\nEdited café."
    else:
        assert outcome.status == "completed" and not outcome.error
        assert outcome.report_source == "file"
        assert outcome.text == ("## Report\n### Done\nEdited café.\n\n"
                                "## Report\n### Not done / open questions\nnone")
        assert outcome.report_note == "follow-up answered in the message"
        cli.print_outcome("w", outcome, quiet=False)
        assert outcome.report_note in capsys.readouterr().out.splitlines()[0]
    assert AgentStore("w").read_answer() == outcome.text


def test_worker_repeated_message_keeps_file_provenance(
        tmp_path, monkeypatch, worker_report):
    record, report = worker_report
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "report-repeat",
                             sandbox=record.sandbox, role=record.role,
                             expect="^### Not done")
    assert outcome.status == "completed" and outcome.follow_ups == 1
    assert outcome.text == "## Report\n### Done\nEdited café."
    assert outcome.report_source == "file"
    assert outcome.report_note == "follow-up answered in the message"


def test_worker_locked_report_cleanup_does_not_abort_either_attempt(
        tmp_path, monkeypatch, worker_report):
    record, report = worker_report
    unlink = Path.unlink

    def locked(path, *args, **kwargs):
        if path == report:
            raise PermissionError("report is locked")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", locked)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "report-partial",
                             sandbox=record.sandbox, role=record.role,
                             expect="^### Not done")
    assert outcome.status == "completed" and not outcome.error
    assert outcome.follow_ups == 1 and outcome.report_source == "file"
    assert "### Not done" in outcome.text
    errors = [m for m in AgentStore("w").events()
              if m.get("_meta") == "report_unlink_error"]
    assert errors == [{"_meta": "report_unlink_error", "path": str(report),
                       "error": "PermissionError"}] * 2


def test_worker_report_is_read_once_after_service_retries(
        tmp_path, monkeypatch, worker_report):
    record, report = worker_report
    reads = []
    read_bytes = Path.read_bytes

    def capture(path, *args, **kwargs):
        if path == report:
            reads.append(int((tmp_path / "attempts.txt").read_text(encoding="utf-8")))
        return read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", capture)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "report-retry",
                             sandbox=record.sandbox, role=record.role,
                             sleep=lambda _: None, reachable=lambda: True)
    assert outcome.status == "completed" and outcome.service_retries == 1
    assert outcome.report_source == "file" and "Edited café." in outcome.text
    assert reads == [2]


@pytest.mark.parametrize("scenario", ["report-silent", "report-file-only"])
def test_worker_silent_followup_keeps_file_provenance(
        tmp_path, monkeypatch, worker_report, scenario):
    record, report = worker_report
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, scenario,
                             sandbox=record.sandbox, role=record.role,
                             expect="^### Not done")
    assert outcome.status == "completed" and not outcome.error
    assert outcome.follow_ups == 1 and outcome.message_text == ""
    assert outcome.text == "## Report\n### Done\nEdited café."
    assert outcome.report_source == "file" and outcome.report_differs is True
    assert outcome.report_note == ""
    assert AgentStore("w").read_answer() == outcome.text


@pytest.mark.parametrize("interrupted", [False, True])
def test_worker_aborted_turn_has_no_report_selection(
        tmp_path, monkeypatch, worker_report, interrupted):
    record, report = worker_report
    if interrupted:
        def interrupt(record):
            raise KeyboardInterrupt
        monkeypatch.setattr(sandbox, "prepare_workspace", interrupt)
    else:
        record.cwd = str(tmp_path / "missing")
    with pytest.raises(KeyboardInterrupt if interrupted else SystemExit):
        turn.run_turn(record=record, prompt="task", resume=False)
    notes = list(AgentStore("w").events())
    assert not any(m.get("_meta") == events.REPORT_META for m in notes)
    assert notes[-1]["status"] == "aborted"


def test_worker_failed_attempt_does_not_select_its_file(
        tmp_path, monkeypatch, worker_report):
    record, report = worker_report
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "report-terminal-failure",
                             sandbox=record.sandbox, role=record.role)
    assert report.is_file()
    assert outcome.status == "failed" and outcome.error_kind == "usageLimitExceeded"
    assert outcome.report_source == "message" and outcome.report_differs is None
    assert outcome.text == "## Report\n### Not done / open questions\nnone"
    assert outcome.follow_ups == 0 and outcome.service_retries == 0
    notes = list(AgentStore("w").events())
    assert next(m for m in notes if m.get("_meta") == events.REPORT_META)["status"] == "failed"
    assert AgentStore("w").read_answer() == outcome.text


def test_spawn_prints_the_same_report_path_as_the_briefing(
        tmp_path, monkeypatch, capsys, worker_report):
    record, report = worker_report
    wire.install_fake_server(monkeypatch, tmp_path, "report-file")
    assert core_main(["spawn", record.name, "task", "--cwd", str(tmp_path),
                     "--role", "worker"]) == 0
    assert f"[codex:w] report {report}" in capsys.readouterr().out
    briefing = wire.first_request(tmp_path, "thread/start")["params"]["developerInstructions"]
    assert str(report) in briefing
    cli.cmd_status(cli.build_parser().parse_args(["status", "w"]))
    assert f"report={report}" in capsys.readouterr().out
    assert core_main(["send", "w", "next task"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == f"[codex:w] report {report}"
    assert core_main(["send", "w", "next task", "--quiet"]) == 0
    assert f"[codex:w] report {report}" not in capsys.readouterr().out


def test_initialize_opts_out_of_delta_notifications_on_the_wire(tmp_path, monkeypatch):
    """The point of the wrapper: the caller pays for the answer, not the
    stream. Deltas are the bulk of the traffic and are never rendered — but a
    list that never reaches `initialize` opts out of nothing."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "plain")
    assert outcome.status == "completed"
    capabilities = wire.first_request(tmp_path, "initialize")["params"]["capabilities"]
    assert capabilities["optOutNotificationMethods"] == protocol.NOISY_NOTIFICATIONS
    assert "item/agentMessage/delta" in capabilities["optOutNotificationMethods"]
    assert protocol.N_ITEM_COMPLETED not in capabilities["optOutNotificationMethods"]


def test_a_full_turn_returns_the_final_message_and_its_counters(tmp_path, monkeypatch):
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "plain", timeout=10.0)
    assert outcome.status == "completed"
    assert outcome.error == ""
    assert outcome.thread_id == "th-fake"
    assert outcome.text == "fake answer"
    assert (outcome.commands, outcome.file_changes) == (1, 2)
    assert (outcome.input_tokens, outcome.output_tokens) == (120, 8)
    assert AgentStore("w").read_answer() == "fake answer"
    started = wire.first_request(tmp_path, "turn/start")["params"]
    assert started["threadId"] == "th-fake"
    assert started["input"] == [{"type": "text", "text": "do the thing"}]


def test_an_answer_lacking_the_expected_section_is_followed_up_once(
        tmp_path, monkeypatch, capsys):
    """Codex answers one question of five and stops; the operator's fix was
    one `send` naming the gap (three times on 2026-09-13, once the day
    after). `--expect` sends it: the second turn resumes the SAME thread, the
    prompt names the missing pattern, the two answers are joined oldest
    first, the work of both is counted, and the log reopens the turn for
    `status` while it runs."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "partial",
                             expect=r"^## Coverage", timeout=15.0)
    assert outcome.status == "completed" and outcome.error == ""
    assert outcome.text == "## Findings\n1. one thing\n\n## Coverage\nall of them"
    assert outcome.follow_ups == 1 and outcome.expect == r"^## Coverage"
    assert outcome.commands == 2
    assert AgentStore("w").read_answer() == outcome.text
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2
    follow_up = starts[1]["params"]["input"][0]["text"]
    assert "^## Coverage" in follow_up and "without repeating" in follow_up
    # A worker's missing section is usually unfinished WORK (`topology`,
    # `lotrules` 2026-09-13: gates unrun, report says "NOT RUN"), so the
    # follow-up orders the work finished before the answer is completed.
    assert "finish it first" in follow_up
    # Resumed, not restarted: a fresh thread would not have the first answer.
    resumed = wire.first_request(tmp_path, "thread/resume")["params"]
    assert resumed["threadId"] == "th-fake"
    trace = AgentStore("w").last_turn()
    assert trace.ended and trace.follow_ups == 1 and trace.commands == 2
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "w"]))
    assert "--- follow-up: the answer lacked /^## Coverage/" in capsys.readouterr().out
    assert outcome.expect_met is True
    assert cli.print_outcome("w", outcome, quiet=False) == 0
    printed = capsys.readouterr().out
    assert followed_up_note("^## Coverage", 1) in printed
    assert "Incomplete" not in printed


def test_a_sub_agent_finishing_first_neither_ends_the_turn_nor_answers_it(
        tmp_path, monkeypatch):
    """Four reviews of 2026-10-24 came back as "answered a subset": Codex had
    handed half the review to a child, the child's `turn/completed` was read
    as ours, and its `final_answer` ("none in the client portion", Coverage
    "not reviewed by this subtask") was returned while the main thread was
    still working on the server half (`llm_loop.wire.codex_app_from_sub_agent`
    has the census). The turn ends on the MAIN thread's completion, the answer
    is the main thread's — also against a child that speaks after it — the child's
    error is not the turn's, and the `--expect` judge reads the main answer,
    which has its Coverage, so nothing is resent. Both children ended before
    the main answer, so nothing is waited for either: no grace, no line."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent",
                             expect=r"^## Coverage", timeout=15.0)
    assert outcome.sub_agents.late == [] and outcome.sub_agents.wait_s == 0
    assert _sub_agent_lines() == [] and outcome.duration_s < 10
    main_answer = "## Findings\n1. server half\n## Coverage\nall of them"
    # First, because it is the defect: the pre-fix wrapper returned
    # "## Findings\nnone in the client portion" here, and a wrapper that
    # takes the children's items as the main thread's returns the late
    # child's "## Findings\nthe late child's subtask".
    assert outcome.text == main_answer and outcome.message_text == main_answer
    assert (outcome.status, outcome.error) == ("completed", "")
    assert outcome.thread_id == "th-fake"
    assert outcome.follow_ups == 0 and outcome.expect_met is True
    assert len(wire.turn_starts(tmp_path)) == 1
    assert AgentStore("w").read_answer() == main_answer
    # The child's command was really run; its tokens were really billed; its
    # window is not the one a `send` resumes.
    assert outcome.commands == 2
    assert outcome.input_tokens == 433471 + 501566
    assert outcome.output_tokens == 1837 + 2163
    assert outcome.context_tokens == 79835
    logged = list(AgentStore("w").events())
    completions = [i for i, message in enumerate(logged)
                   if message.get("method") == protocol.N_TURN_COMPLETED]
    end = next(i for i, message in enumerate(logged)
               if message.get("_meta") == events.END_META)
    assert len(completions) == 3 and completions[-1] < end
    assert logged[completions[-1]]["params"]["threadId"] == "th-fake"


@pytest.mark.parametrize("scenario", ["sub-agent-follow-up", "sub-agent-retry"])
def test_a_sub_agents_tokens_outlive_the_attempt_that_spawned_it(
        tmp_path, monkeypatch, scenario):
    """Every attempt of one caller turn starts a fresh `TurnRunner`, and the
    main thread's cumulative total carries the earlier attempts' spend by
    itself — a child's does not: its thread is over, and the next attempt
    hears nothing of it. review-a spent 433k on its child in attempt 1;
    after the `--expect` follow-up the summary showed the main thread's
    total alone. The bill is the main thread's newest total plus every
    child's last, whichever attempt it ran in."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, scenario,
                             expect=r"^## Coverage", timeout=15.0,
                             sleep=lambda _: None, reachable=lambda: True)
    assert outcome.status == "completed" and "## Coverage" in outcome.text
    assert len(wire.turn_starts(tmp_path)) == 2
    assert outcome.input_tokens == 560000 + 433471
    assert outcome.cached_input_tokens == 480000 + 382336
    assert outcome.output_tokens == 2500 + 1837
    assert outcome.context_tokens == 81000


def _sub_agent_lines(name="w"):
    return [message for message in AgentStore(name).events()
            if message.get("_meta") == events.SUB_AGENTS_META]


def _interrupts(tmp_path):
    return [message["params"] for message in wire.received(tmp_path)
            if message.get("method") == protocol.M_TURN_INTERRUPT]


def _asked(start):
    return start["params"]["input"][0]["text"]


def _fates(outcome):
    return [(fate.path, fate.state, fate.files, fate.told)
            for fate in outcome.sub_agents.late]


def _cli(capsys, *argv):
    capsys.readouterr()
    args = cli.build_parser().parse_args(list(argv))
    args.func(args)
    return capsys.readouterr().out


@pytest.mark.parametrize("explicit", [None, "low", "medium", "high", "xhigh"])
def test_cli_effort_on_wire_is_per_turn(tmp_path, monkeypatch, capsys, explicit):
    """A resumed thread must receive today's default, not its last override."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    wire.install_fake_server(monkeypatch, tmp_path, "ok")
    options = [] if explicit is None else ["--effort", explicit]
    _cli(capsys, "spawn", "effort", "task", "--cwd", str(tmp_path),
         "--sandbox", "read-only", "--quiet", *options)
    stored = Registry().get("effort")
    assert stored.thread_id == "th-fake"
    assert "effort" not in stored.to_json()
    _cli(capsys, "send", "effort", "continue", "--quiet", *options)
    _cli(capsys, "send", "effort", "continue again", "--quiet")
    starts = wire.turn_starts(tmp_path)
    assert [start["params"].get("effort") for start in starts] == [
        explicit or "medium", explicit or "medium", "medium"]
    assert len([m for m in wire.received(tmp_path)
                if m.get("method") == "thread/resume"]) == 2


def test_a_child_outliving_the_main_answer_is_waited_for_and_its_result_sent(
        tmp_path, monkeypatch, capsys):
    """rev-a (line 708): the main thread answered after a 10 s `wait_agent`
    with two children running, and closing the server killed them. Now the
    server stays open (user decision 2026-10-24) and, once the child is
    done, the main thread gets one more turn on the same server — with the
    child's result IN ITS TEXT: the stand-in has no inbox, so "saw: the
    kid's result" is only possible if the prompt carried it."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-live",
                             timeout=15.0)
    assert (outcome.status, outcome.error) == ("completed", "")
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2 and starts[1]["params"]["threadId"] == "th-fake"
    asked = _asked(starts[1])
    assert "finished only after your final answer" in asked
    assert "the kid's result" in asked and "kid.py" in asked
    assert outcome.message_text == "## Findings\nsaw: the kid's result, kid.py"
    assert not any(message.get("method") == "thread/resume"
                   for message in wire.received(tmp_path))
    assert _interrupts(tmp_path) == []
    assert _fates(outcome) == [("/root/kid", "finished", ["kid.py"], True)]
    assert outcome.sub_agents.cut == [] and outcome.sub_agents.note == ""
    assert outcome.sub_agents.turn == subagents.DONE_TURN
    assert 0.1 < outcome.sub_agents.wait_s < 20
    assert ("- /root/kid: finished after the main answer; its result was "
            "passed to the main thread; touched kid.py.") in outcome.text
    assert AgentStore("w").read_answer() == outcome.text
    lines = _sub_agent_lines()
    assert [line["phase"] for line in lines] == [
        events.SUB_AGENTS_WAITING, events.SUB_AGENTS_FOLLOW_UP,
        events.SUB_AGENTS_END]
    assert (lines[1]["kind"], lines[1]["finished"]) == ("done", ["/root/kid"])
    trace = AgentStore("w").last_turn()
    assert trace.ended and trace.status == "completed" and trace.cut_sub_agents == ()
    printed = _cli(capsys, "logs", "w", "--tail", "0")
    assert ("--- sub-agents still running after the main answer: /root/kid "
            "(waiting up to 20.0s)") in printed
    assert ("--- sub-agents finished: /root/kid; their results sent to the "
            "main thread") in printed
    assert ("--- sub-agent /root/kid finished after the main answer; result "
            "passed to the main thread; touched kid.py") in printed
    assert cli.print_outcome("w", outcome, quiet=False) == 0
    assert "then gave their results to the main thread" in capsys.readouterr().out


def test_a_child_known_only_by_its_spawn_record_is_waited_for(
        tmp_path, monkeypatch):
    """The parent's `subAgentActivity started` record precedes the child's
    `turn/started` (rev-a: lines 52 and 57, 36 ms); a main turn
    completing in that gap left a child the wrapper did not know was
    running, and it closed the server on it."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-gap",
                             timeout=15.0)
    assert _sub_agent_lines()[0]["open"] == ["/root/kid"]
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2 and "the kid's result" in _asked(starts[1])
    assert _fates(outcome) == [("/root/kid", "finished", ["kid.py"], True)]


def test_an_older_turns_late_completion_does_not_close_the_newer_one(
        tmp_path, monkeypatch):
    """A child thread runs several turns (rev-a: `/root/nav` started
    `01a12549-e200` and later `01a1254c-3b6f`); the completion of the older
    one, arriving late, said nothing about the newer — read as the child's
    end, it let the wrapper close the server on a running child."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.5)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-stale",
                             timeout=15.0)
    assert _interrupts(tmp_path) == [{"threadId": "th-kid", "turnId": "t-kid2"}]
    assert [fate.path for fate in outcome.sub_agents.cut] == ["/root/kid"]
    # Nor do its words speak for the newer turn.
    asked = _asked(wire.turn_starts(tmp_path)[1])
    assert "kid2 progress" in asked and "stale text" not in asked


def test_a_cut_child_and_grandchild_are_named_and_the_main_thread_asked_to_restart(
        tmp_path, monkeypatch, capsys):
    """The deadline passes: each still-open thread — the grandchild too,
    whose spawn only its parent's thread recorded — gets its own
    `turn/interrupt`, and the main thread gets ONE turn naming them, their
    files and that their work may need restarting (user decision
    2026-10-24: "the worker can track it and restart its sub-agent"). That
    turn's answer is the outcome; the cut is still said everywhere a caller
    looks."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.5)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-stuck",
                             timeout=15.0)
    assert (outcome.status, outcome.error) == ("completed", "")
    assert _interrupts(tmp_path) == [
        {"threadId": "th-kid", "turnId": "t-kid"},
        {"threadId": "th-grand", "turnId": "t-grand"}]
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2
    asked = _asked(starts[1])
    for told in ("restart it", "/root/kid/grand", "kid.py", "grand.py"):
        assert told in asked and told in outcome.message_text
    assert outcome.sub_agents.turn == subagents.RESTART_TURN
    assert _fates(outcome) == [("/root/kid", "cut", ["kid.py"], True),
                               ("/root/kid/grand", "cut", ["grand.py"], True)]
    assert ("- /root/kid: CUT — interrupted while still running; the main "
            "thread was asked to restart it or finish its work; touched "
            "kid.py.") in outcome.text
    assert AgentStore("w").read_answer() == outcome.text
    lines = _sub_agent_lines()
    assert [line["phase"] for line in lines] == [
        events.SUB_AGENTS_WAITING, events.SUB_AGENTS_CUT,
        events.SUB_AGENTS_FOLLOW_UP, events.SUB_AGENTS_END]
    assert lines[1]["cut"] == ["/root/kid", "/root/kid/grand"]
    assert lines[3]["cut"] == [
        {"path": "/root/kid", "thread": "th-kid", "files": ["kid.py"],
         "told": True, "status": "interrupted", "restarted": False},
        {"path": "/root/kid/grand", "thread": "th-grand",
         "files": ["grand.py"], "told": True, "status": "interrupted",
         "restarted": False}]

    for quiet in (False, True):
        assert cli.print_outcome("w", outcome, quiet=quiet) == 0
        assert ("[codex:w] warning: 2 sub-agent(s) cut while still running: "
                "/root/kid (1 file(s)), /root/kid/grand (1 file(s)) — the main "
                "thread was asked to restart them" in capsys.readouterr().out)
    printed = _cli(capsys, "logs", "w", "--tail", "0")
    assert ("--- asking the main thread to restart or finish the cut "
            "sub-agents: /root/kid, /root/kid/grand (finished: none)") in printed
    assert ("--- sub-agent /root/kid/grand CUT: interrupted; the main thread "
            "was asked to restart it; touched grand.py") in printed
    Registry().put(AgentRecord(name="w", thread_id="th-fake", sandbox="read-only"))
    printed = _cli(capsys, "status", "w")
    assert "last turn cut 2 sub-agent(s): /root/kid, /root/kid/grand" in printed
    assert "RUNNING" not in printed
    # An ended turn is not a running one, whatever it cut.
    assert "(1 idle agent(s) not listed" in _cli(capsys, "status", "--running")


def test_a_cut_and_a_finished_child_go_into_one_restart_turn(
        tmp_path, monkeypatch):
    """One child finishes after the answer, the other has to be cut: one
    turn carries both — the finished one's result and the cut one's news."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.5)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-mixed",
                             timeout=15.0)
    assert _interrupts(tmp_path) == [{"threadId": "th-b", "turnId": "t-b"}]
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2
    for told in ("A's result", "/root/b", "a.py", "b.py"):
        assert told in _asked(starts[1]) and told in outcome.message_text
    assert _fates(outcome) == [("/root/a", "finished", ["a.py"], True),
                               ("/root/b", "cut", ["b.py"], True)]


def _fate_of(outcome, path):
    return next(fate for fate in outcome.sub_agents.late if fate.path == path)


def test_a_cut_child_the_restart_turn_resumes_on_its_thread_is_not_cut_again(
        tmp_path, monkeypatch):
    """The user's decision at work: the restart turn resumes the cut child
    on its own thread (Codex v2 `followup_task` starts a new turn there) and
    waits for it. That new turn belongs to the main turn that started it —
    read as part of the expired wait, it was interrupted at once."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.5)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-revive",
                             timeout=15.0)
    assert (outcome.status, outcome.error) == ("completed", "")
    assert _interrupts(tmp_path) == [{"threadId": "th-kid", "turnId": "t-kid"}]
    assert outcome.message_text == "## Findings\nrestarted the kid"
    kid = _fate_of(outcome, "/root/kid")
    assert (kid.state, kid.restarted, kid.files) == (
        "finished", True, ["kid.py", "kid2.py"])
    assert outcome.sub_agents.cut == []
    assert "restarted by the main thread after its cut" in outcome.text
    assert "> the kid's second result" in outcome.text


def test_a_grandchild_spawned_during_the_wait_is_waited_for_too(
        tmp_path, monkeypatch):
    """A child spawning its own child while the main thread is idle: the
    grandchild runs on the same deadline. Otherwise the turn about the
    children went out with it still running, and its result vanished."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-idle-grand",
                             timeout=15.0)
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2
    asked = _asked(starts[1])
    assert "the kid's result" in asked and "the grand's result" in asked
    grand = _fate_of(outcome, "/root/kid/grand")
    assert (grand.state, grand.told) == ("finished", True)


@pytest.mark.parametrize("scenario", ["sub-agent-done-fail",
                                      "sub-agent-done-reject"])
def test_a_turn_about_the_children_that_fails_leaves_their_results_quoted(
        tmp_path, monkeypatch, scenario):
    """Told is what a COMPLETED turn carried: a rejected (an input over
    Codex's size limit) or failed one delivered nothing, so the results are
    quoted under the kept answer and the failure is noted there — in
    `result` too, not only in the summary."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, scenario, timeout=15.0)
    assert (outcome.status, outcome.error) == ("completed", "")
    assert outcome.message_text == "## Findings\nwithout the kid"
    assert _fates(outcome) == [("/root/kid", "finished", ["kid.py"], False)]
    stored = AgentStore("w").read_answer()
    assert "> the kid's result" in stored
    assert "Note: sub-agent done turn failed" in stored


def test_a_restart_turn_that_times_out_keeps_the_answer_and_the_cut(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.3)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-restart-timeout",
                             timeout=2.0)
    assert (outcome.status, outcome.error) == ("completed", "")
    assert outcome.message_text == "## Findings\nwithout the kid"
    assert outcome.sub_agents.note.startswith(
        "sub-agent restart turn failed: codex app-server did not answer")
    assert _fates(outcome) == [("/root/kid", "cut", ["kid.py"], False)]
    cli.print_outcome("w", outcome, quiet=True)
    assert "check the tree for edits the answer does not describe" in (
        capsys.readouterr().out)


def test_a_child_the_restart_turn_starts_and_leaves_running_is_cut_and_named(
        tmp_path, monkeypatch):
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.3)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-restart-spawn",
                             timeout=15.0)
    assert outcome.message_text == "## Findings\nstarted a new one"
    assert _interrupts(tmp_path) == [{"threadId": "th-kid", "turnId": "t-kid"},
                                     {"threadId": "th-new", "turnId": "t-new"}]
    assert _fates(outcome) == [("/root/kid", "cut", ["kid.py"], True),
                               ("/root/new", "cut", ["new.py"], False)]
    assert "- /root/new: CUT — interrupted while still running; touched new.py." in outcome.text


# What the `subrep-*` stand-in writes as the report of its first answer.
FIRST_REPORT = "## Report\n### Done\nfirst\n### Not done / open questions\nnone\n"


def test_a_late_report_write_cannot_replace_the_restored_report(
        tmp_path, monkeypatch, worker_report):
    """The turn about the children times out while the main thread is still
    writing the report: put back before the server is stopped, the first
    report was overwritten by the partial one and read as the answer."""
    record, report = worker_report
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "subrep-timeout",
                             sandbox=record.sandbox, role=record.role,
                             timeout=2.0)
    first = FIRST_REPORT
    assert report.read_text(encoding="utf-8") == first
    assert outcome.text.startswith(first) and outcome.status == "completed"


def test_a_turn_about_the_children_without_a_new_report_keeps_the_first(
        tmp_path, monkeypatch, worker_report):
    """The report was set aside for the turn; one that completes answering
    in its message only must not cost the complete first report — it is
    put back, and the message goes under it."""
    record, report = worker_report
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "subrep-silent",
                             sandbox=record.sandbox, role=record.role,
                             timeout=15.0)
    first = FIRST_REPORT
    assert report.read_text(encoding="utf-8") == first
    assert outcome.text.startswith(
        first + "\nmerged the kid's result in the message only\n")
    assert outcome.report_source == "file" and outcome.sub_agents.note == ""


def test_the_reply_joined_under_a_kept_report_survives_a_failed_gate_follow_up(
        tmp_path, monkeypatch, worker_report):
    """The turn about the children answered in its message only; joined
    under the put-back report only at the very end, it was dropped when a
    gate follow-up failed and kept the answer before it — with the children
    marked delivered, so not quoted either."""
    record, report = worker_report
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    wire.install_fake_server(monkeypatch, tmp_path, "subrep-silent-gatefail")
    gated = AgentRecord(name="w", cwd=str(tmp_path), sandbox=record.sandbox,
                        role=record.role, gates=["npm test"])

    def red(command, cwd, timeout):
        return gates_module.GateRun(command, str(cwd), 1, "FAILED boom", 1.0)

    outcome = turn.run_turn(record=gated, prompt="do the thing", resume=False,
                            timeout=15.0, run_gate=red)
    assert outcome.gate_follow_ups == 1 and "gate follow-up failed" in outcome.error
    assert "merged the kid's result in the message only" in outcome.text
    assert "merged the kid's result in the message only" in (
        AgentStore("w").read_answer())


def test_a_capped_child_answer_points_at_a_command_that_prints_it_whole(
        tmp_path, monkeypatch, capsys):
    """The quote of a long child message is capped; the command it names
    must print the rest — the line-per-action `logs` keeps one line."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-long",
                             timeout=15.0)
    command = subagents.full_text_command("w", "/root/kid")
    assert command in outcome.text and "TAIL-MARK" not in outcome.text
    printed = _cli(capsys, "logs", "w", "--child", "/root/kid")
    # The stand-in's `LONG_ANSWER`, whole and once.
    whole = "L" * 30000 + "TAIL-MARK"
    assert whole in printed and printed.count("TAIL-MARK") == 1


def test_a_child_whose_turn_failed_is_not_passed_off_as_finished(
        tmp_path, monkeypatch):
    """A failed child turn's last message is commentary, not a result: the
    turn about it names the verdict and labels the message partial."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-child-fail", timeout=15.0)
    asked = _asked(wire.turn_starts(tmp_path)[1])
    assert "### Finished (failed: child broke): /root/kid" in asked
    assert "PARTIAL" in asked and "half-done notes" in asked


def test_a_child_that_completed_during_its_interrupt_is_finished_not_cut(
        tmp_path, monkeypatch):
    """Its verdict came back `completed`: it finished on its own, so it is
    passed on as finished, not reported as interrupted."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.3)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-drain-done",
                             timeout=15.0)
    assert _fates(outcome) == [("/root/kid", "finished", ["kid.py"], True)]
    asked = _asked(wire.turn_starts(tmp_path)[1])
    assert "### Finished: /root/kid" in asked and "Interrupted" not in asked
    assert outcome.sub_agents.turn == subagents.DONE_TURN


def test_a_failed_continuation_costs_the_continuation_not_the_answer(
        tmp_path, monkeypatch):
    """A complete answer, then a turn the main thread started by itself and
    the service dropped: read as the attempt's verdict, it sent the complete
    answer to `_retry_service_drops`."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-continue-fail",
                             timeout=15.0, sleep=lambda _: None,
                             reachable=lambda: True)
    assert (outcome.status, outcome.error) == ("completed", "")
    assert outcome.message_text == "## Findings\nwithout the kid"
    assert outcome.service_retries == 0 and len(wire.turn_starts(tmp_path)) == 1
    assert outcome.sub_agents.note.startswith(
        "the main thread's own continuation failed")
    assert [fate.path for fate in outcome.sub_agents.cut] == ["/root/kid"]


def test_a_long_child_answer_is_capped_in_the_turn_about_it():
    """Codex rejects a turn over 1 048 576 characters: each quoted message is
    capped, the rest pointed at in `logs`, and the total is bounded."""
    tracker = subagents.SubAgentTracker()
    for n in range(15):
        thread = f"th-{n}"
        tracker.on_record({"type": "subAgentActivity", "kind": "started",
                           "agentThreadId": thread, "agentPath": f"/root/c{n}"})
        tracker.on_turn_started(thread, f"t-{n}")
    tracker.main_completed()
    for n in range(15):
        tracker.on_turn_completed(f"th-{n}", {"id": f"t-{n}", "status": "completed",
                                              "items": [{"type": "agentMessage",
                                                         "text": "x" * 50_000}]})
    kind, text, finished, cut = tracker.turn_prompt("", "w")
    assert kind == subagents.DONE_TURN and len(finished) == 15
    assert len(text) < subagents.PROMPT_CHARS + 20_000
    assert ("more characters: `python -m llm_agents.codex logs w --child "
            "/root/c0` prints them") in text
    assert "(omitted: the quoted messages reached" in text


def test_a_main_thread_continuing_by_itself_is_waited_out_not_prodded(
        tmp_path, monkeypatch, capsys):
    """rust-v0.164.0 does not wake the main thread when a child finishes
    (agent/control/api.rs:129-135, session/turn_input.rs:734-746), so this
    is a defensive path: a turn the main thread starts by itself is waited
    out like the first and its answer replaces the first. A child finishing
    INSIDE it is not sent back — nothing says the turn read it — so its
    answer is quoted to the caller, labelled unread."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-continue",
                             timeout=15.0)
    assert (outcome.status, outcome.error) == ("completed", "")
    assert outcome.message_text == "## Findings\ncontinued"
    assert len(wire.turn_starts(tmp_path)) == 1 and _interrupts(tmp_path) == []
    assert outcome.sub_agents.turn == ""
    assert _fates(outcome) == [("/root/kid", "finished", ["kid.py"], False)]
    assert "UNREAD by the main thread; touched kid.py. Its answer:\n  > the kid's result" in outcome.text
    assert [line["phase"] for line in _sub_agent_lines()] == [
        events.SUB_AGENTS_WAITING, events.SUB_AGENTS_END]
    assert "UNREAD by the main thread (quoted under the answer)" in _cli(
        capsys, "logs", "w", "--tail", "0")


def test_a_main_thread_busy_with_its_own_turn_does_not_extend_the_wait(
        tmp_path, monkeypatch):
    """The children's deadline holds while the main thread streams a turn
    of its own: a review reproduced 1000 s of waiting under a 600 s grace
    when only an idle main thread was checked. The interrupt is logged
    before that turn completes, and the cut child is then the restart
    turn's."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.3)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-busy-continue",
                             timeout=15.0)
    logged = list(AgentStore("w").events())
    cut_at = next(i for i, message in enumerate(logged)
                  if message.get("phase") == events.SUB_AGENTS_CUT)
    continued = next(i for i, message in enumerate(logged)
                     if message.get("method") == protocol.N_TURN_COMPLETED
                     and (message["params"].get("turn") or {}).get("id") == "t-main2")
    assert cut_at < continued
    assert _interrupts(tmp_path) == [{"threadId": "th-kid", "turnId": "t-kid"}]
    assert outcome.sub_agents.turn == subagents.RESTART_TURN
    assert len(wire.turn_starts(tmp_path)) == 2


class _BusyServer:
    """A server whose stream never pauses: `Queue.get(timeout=0)` on a queue
    that always holds a message never expires, so only a clock check ends a
    loop over it."""
    finished = False

    def __init__(self):
        self.pumps = 0
        self.sent = []

    def pump(self, deadline, on_event):
        self.pumps += 1
        if self.pumps > 200000:
            raise AssertionError("pumped long past the deadline")
        on_event({"method": protocol.N_TOKEN_USAGE, "params": {
            "threadId": "th-kid", "tokenUsage": {}}})

    def send_request(self, method, params=None):
        self.sent.append((method, params))
        return len(self.sent)


def test_every_wait_ends_by_the_clock_even_on_a_stream_that_never_pauses(
        monkeypatch):
    """The main-turn loop, the children's wait and the interrupt drain each
    check the clock themselves."""
    monkeypatch.setattr(subagents, "SUB_AGENT_INTERRUPT_S", 0.05)
    runner = TurnRunner()
    runner.outcome.thread_id = "th-main"
    wait = subagent_wait.SubAgentWait(log=lambda message: None, name="w")
    with pytest.raises(TimeoutError):
        subagent_wait._pump_main(_BusyServer(), runner,
                                 time.monotonic() + 0.05, wait)

    runner.subs.on_turn_started("th-kid", "t-kid")
    runner.subs.main_completed()
    runner.done = True
    wait.grace_end = time.monotonic() + 0.05
    busy = _BusyServer()
    assert subagent_wait._pump_idle_main(busy, runner, wait) is True
    assert busy.sent == [(protocol.M_TURN_INTERRUPT,
                          {"threadId": "th-kid", "turnId": "t-kid"})]
    assert not runner.subs.is_open("th-kid")


def test_a_late_main_failure_during_the_drain_keeps_the_timeout_verdict(
        tmp_path, monkeypatch):
    """The attempt timed out with a child open; interrupting the child, the
    server reports the never-finished main turn as failed by a service drop.
    Read, it turned the timeout into a "failed" the retry logic resends."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-late-root",
                             timeout=1.0, sleep=lambda _: None,
                             reachable=lambda: True)
    assert outcome.status == "timeout" and "did not answer" in outcome.error
    assert outcome.service_retries == 0 and len(wire.turn_starts(tmp_path)) == 1
    assert _interrupts(tmp_path) == [{"threadId": "th-kid", "turnId": "t-kid"}]
    assert [fate.path for fate in outcome.sub_agents.cut] == ["/root/kid"]


def test_a_failed_main_turn_cuts_its_children_at_once(tmp_path, monkeypatch):
    """A main turn that failed is not waited past (`_retry_service_drops`
    would resend it on a fresh server anyway): its child is interrupted and
    named without a waiting line or a turn about it."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-fail",
                             timeout=15.0)
    assert outcome.status == "failed" and "usage limit" in outcome.error
    assert outcome.duration_s < 10 and len(wire.turn_starts(tmp_path)) == 1
    assert _interrupts(tmp_path) == [{"threadId": "th-kid", "turnId": "t-kid"}]
    assert [line["phase"] for line in _sub_agent_lines()] == [
        events.SUB_AGENTS_CUT, events.SUB_AGENTS_END]
    assert _fates(outcome) == [("/root/kid", "cut", ["kid.py"], False)]


def test_a_child_that_ignores_its_interrupt_costs_the_drain_bound_once(
        tmp_path, monkeypatch):
    """No `turn/completed` ever comes for the interrupted child: it is
    written off after `SUB_AGENT_INTERRUPT_S`, and not interrupted again
    when the attempt ends."""
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.3)
    monkeypatch.setattr(subagents, "SUB_AGENT_INTERRUPT_S", 0.5)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-deaf",
                             timeout=15.0)
    assert outcome.duration_s < 8
    assert _interrupts(tmp_path) == [{"threadId": "th-kid", "turnId": "t-kid"}]
    assert [fate.path for fate in outcome.sub_agents.cut] == ["/root/kid"]
    assert outcome.sub_agents.turn == subagents.RESTART_TURN


@pytest.mark.parametrize("scenario", ["subrep-live", "subrep-fail"])
def test_a_worker_report_is_set_aside_for_the_sub_agent_turn(
        tmp_path, monkeypatch, capsys, worker_report, scenario):
    """As `_send_follow_up` clears it: the report is gone when the turn about
    the children starts, so an unchanged file cannot pass for its answer —
    and when that turn fails, the file, the complete first answer and its
    status are put back (the gates still run on a completed turn), with the
    loss named."""
    record, report = worker_report
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 20.0)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, scenario,
                             sandbox=record.sandbox, role=record.role,
                             timeout=15.0)
    assert (tmp_path / "report_at_wrapper.txt").read_text(encoding="utf-8") == "False"
    first = FIRST_REPORT
    assert (outcome.status, outcome.error) == ("completed", "")
    if scenario == "subrep-live":
        assert outcome.text.startswith(
            "## Report\n### Done\nmerged the kid's result\n")
        assert outcome.report_source == "file" and outcome.sub_agents.note == ""
        return
    assert report.read_text(encoding="utf-8") == first
    assert outcome.text.startswith(first) and outcome.report_source == "file"
    assert outcome.sub_agents.note.startswith(
        "sub-agent done turn failed: You've hit your usage limit.")
    assert cli.print_outcome("w", outcome, quiet=True) == 0
    assert "Note: sub-agent done turn failed" in capsys.readouterr().out
    # `result` too, and the child the failed turn never delivered is quoted.
    stored = AgentStore("w").read_answer()
    assert "Note: sub-agent done turn failed" in stored
    assert "UNREAD by the main thread" in stored and "> the kid's result" in stored


def test_with_no_answer_the_sub_agent_account_does_not_become_the_stored_answer(
        tmp_path, monkeypatch, capsys):
    """A main thread that never said anything: the account of its children
    is printed, but stored as the answer it would replace an earlier turn's
    real one in `last.md` with a list of children."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    AgentStore("w").write_answer("an earlier turn's answer")
    monkeypatch.setattr(subagents, "SUB_AGENT_GRACE_S", 0.3)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "sub-agent-mute",
                             timeout=15.0)
    assert outcome.text == ""
    assert outcome.sub_agents.block.startswith(subagents.BLOCK_HEADING)
    assert AgentStore("w").read_answer() == "an earlier turn's answer"
    cli.print_outcome("w", outcome, quiet=True)
    assert subagents.BLOCK_HEADING in capsys.readouterr().out


def test_a_caller_turn_s_sub_agent_account_folds_each_field_by_its_rule():
    """The attempts of one caller turn (a service resend, follow-ups) fold
    into one account: children concatenated, waits summed, the turn and the
    note of the last attempt that had one, the reply of the last attempt,
    and no block — it is rendered after the fold."""
    first = subagents.SubAgentFate(path="/root/a", thread_id="th-a")
    second = subagents.SubAgentFate(path="/root/b", thread_id="th-b",
                                    state="cut")
    folded = subagents.SubAgentAccount.fold([
        subagents.SubAgentAccount(late=[first], wait_s=2.0,
                                  turn=subagents.DONE_TURN, note="lost one",
                                  block="stale", reply="first reply"),
        subagents.SubAgentAccount(late=[second], wait_s=3.5,
                                  turn=subagents.RESTART_TURN, reply="second"),
        subagents.SubAgentAccount(),
    ])
    assert folded.late == [first, second] and folded.cut == [second]
    assert folded.wait_s == 5.5
    assert folded.turn == subagents.RESTART_TURN
    assert folded.note == "lost one"
    assert folded.reply == "" and folded.block == ""
    assert subagents.SubAgentAccount.fold([]) == subagents.SubAgentAccount()


def test_a_sub_agent_account_field_without_a_fold_rule_fails_the_fold():
    """A field the fold has no rule for raises instead of reading as empty
    on the caller turn's outcome."""
    @dataclasses.dataclass
    class Wider(subagents.SubAgentAccount):
        extra: int = 0

    with pytest.raises(TypeError, match="extra"):
        Wider.fold([Wider(extra=1)])


def _waiting_log(store, *more):
    store.append_events([
        {"_meta": events.TURN_META, "name": "wait", "prompt": "p",
         "at": "2026-10-24 12:00:00", "timeout": 60.0},
        _event(protocol.N_TURN_STARTED, {"threadId": "th-main", "turn": {
            "id": "t-main"}}),
        _event(protocol.N_TURN_COMPLETED, {"threadId": "th-main", "turn": {
            "id": "t-main", "status": "completed"}}),
        {"_meta": events.SUB_AGENTS_META, "phase": events.SUB_AGENTS_WAITING,
         "name": "wait", "open": ["/root/a", "/root/b"], "grace_s": 600.0},
        *more])


def test_a_turn_waiting_for_sub_agents_is_running_for_as_long_as_it_said(
        tmp_path, monkeypatch, capsys):
    """While the wrapper waits, the log may stay silent past the turn's
    timeout plus the startup grace — `status` called that a killed process
    — so the waiting line names its grace and `status` adds it; past that
    too, it is a killed process again."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="wait", thread_id="th-main",
                               sandbox="read-only"))
    store = AgentStore("wait")
    _waiting_log(store)
    for silent_s, running in ((60 + turn.TURN_STARTUP_GRACE_S + 300, True),
                              (60 + turn.TURN_STARTUP_GRACE_S + 600 + 30, False)):
        stamp = time.time() - silent_s
        os.utime(store.events_path, (stamp, stamp))
        printed = _cli(capsys, "status", "wait")
        assert ("RUNNING since 2026-10-24 12:00:00" in printed) is running, silent_s
        assert ("killed" in printed) is not running, silent_s
        assert "waiting for up to 2 sub-agent(s)" in printed


def test_a_sub_agent_wait_stays_running_across_main_turns(
        tmp_path, monkeypatch, capsys):
    """A main turn completing inside the wait (a continuation, or the turn
    sent about the children) is not the end: `status --running` lists the
    agent until the closing line."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="wait", thread_id="th-main",
                               sandbox="read-only"))
    store = AgentStore("wait")
    _waiting_log(store,
                 _event(protocol.N_TURN_STARTED, {"threadId": "th-main", "turn": {
                     "id": "t-main2"}}),
                 _event(protocol.N_TURN_COMPLETED, {"threadId": "th-main", "turn": {
                     "id": "t-main2", "status": "completed"}}))
    assert "RUNNING since" in _cli(capsys, "status", "--running")
    store.append_events([
        {"_meta": events.SUB_AGENTS_META, "phase": events.SUB_AGENTS_END,
         "name": "wait", "cut": [], "finished": []},
        {"_meta": events.END_META, "name": "wait", "status": "completed"}])
    assert "(1 idle agent(s) not listed" in _cli(capsys, "status", "--running")


def test_status_and_logs_tell_a_sub_agent_from_the_main_thread(
        tmp_path, monkeypatch, capsys):
    """The log readers make the same call as the live turn. `status` on a log
    without a closing line (the wrapper still running, or one written before
    that line existed) must not call the turn over at the child's completion,
    and `logs` marks the child's lines with its path — unmarked, its final
    answer read as the main thread's in the middle of the turn. A resumed turn
    has no `thread/started` in the log, so the main thread is learnt from the
    first `turn/started`."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    main = {"threadId": "th-main", "turnId": "t-main"}
    child = {"threadId": "th-child", "turnId": "t-child"}
    store = AgentStore("split")
    store.append_events([
        {"_meta": events.TURN_META, "name": "split", "prompt": "p",
         "at": "2026-10-24 03:40:00", "timeout": 600.0},
        _event(protocol.N_TURN_STARTED, {"threadId": "th-main", "turn": {
            "id": "t-main", "rootTurnId": "t-main"}}),
        _event(protocol.N_ITEM_COMPLETED, {**main, "item": {
            "type": "subAgentActivity", "kind": "started",
            "agentThreadId": "th-child", "agentPath": "/root/client_review"}}),
        _event(protocol.N_TURN_STARTED, {"threadId": "th-child", "turn": {
            "id": "t-child", "rootTurnId": "t-main"}}),
        _event(protocol.N_ITEM_COMPLETED, {**child, "item": {
            "type": "commandExecution", "command": "git show a"}}),
        _event(protocol.N_ITEM_COMPLETED, {**child, "item": {
            "type": "agentMessage", "text": "none in the client portion"}}),
        _event(protocol.N_TURN_COMPLETED, {"threadId": "th-child", "turn": {
            "id": "t-child", "rootTurnId": "t-main", "status": "completed"}}),
    ])
    trace = store.last_turn()
    assert trace is not None and not trace.ended and trace.commands == 1

    cli.cmd_logs(cli.build_parser().parse_args(["logs", "split"]))
    printed = capsys.readouterr().out
    assert "--- sub-agent /root/client_review started" in printed
    assert "[/root/client_review] $ git show a" in printed
    assert "[/root/client_review] = none in the client portion" in printed

    store.append_events([
        _event(protocol.N_ITEM_COMPLETED, {**main, "item": {
            "type": "agentMessage", "text": "server half reviewed"}}),
        _event(protocol.N_TURN_COMPLETED, {"threadId": "th-main", "turn": {
            "id": "t-main", "rootTurnId": "t-main", "status": "completed"}}),
    ])
    trace = store.last_turn()
    assert trace.ended and trace.status == "completed"
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "split"]))
    assert "\n= server half reviewed" in capsys.readouterr().out


def test_logs_label_a_grandchild_and_show_an_interrupted_child(
        tmp_path, monkeypatch, capsys):
    """A child that spawns its own child records the spawn on ITS thread,
    so labels learnt from the main thread's records alone printed the
    grandchild's lines as a bare "[sub-agent]". And the main thread
    interrupting a child (`kind: interrupted`, seven times in the logs up to
    2026-10-24, e.g. `.codex-agents/rev-f` line 204) is as much the child's
    fate as its completion, and printed nothing."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    main = {"threadId": "th-main", "turnId": "t-main"}
    child = {"threadId": "th-child", "turnId": "t-child"}
    grand = {"threadId": "th-grand", "turnId": "t-grand"}
    store = AgentStore("nest")
    store.append_events([
        {"_meta": events.TURN_META, "name": "nest", "prompt": "p",
         "at": "2026-10-24 05:00:00", "timeout": 600.0},
        _event(protocol.N_TURN_STARTED, {"threadId": "th-main", "turn": {
            "id": "t-main", "rootTurnId": "t-main"}}),
        _event(protocol.N_ITEM_COMPLETED, {**main, "item": {
            "type": "subAgentActivity", "kind": "started",
            "agentThreadId": "th-child", "agentPath": "/root/server"}}),
        _event(protocol.N_ITEM_COMPLETED, {**child, "item": {
            "type": "subAgentActivity", "kind": "started",
            "agentThreadId": "th-grand", "agentPath": "/root/server/net"}}),
        _event(protocol.N_ITEM_COMPLETED, {**grand, "item": {
            "type": "commandExecution", "command": "git show c"}}),
        # A child's own record of its parent names the parent, not itself.
        _event(protocol.N_ITEM_COMPLETED, {**child, "item": {
            "type": "subAgentActivity", "kind": "interacted",
            "agentThreadId": "th-main", "agentPath": "/root"}}),
        _event(protocol.N_ITEM_COMPLETED, {**main, "item": {
            "type": "subAgentActivity", "kind": "interrupted",
            "agentThreadId": "th-child", "agentPath": "/root/server"}}),
    ])
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "nest"]))
    printed = capsys.readouterr().out.splitlines()
    assert "[/root/server/net] $ git show c" in printed
    assert "--- sub-agent /root/server/net started" in printed
    assert "--- sub-agent /root/server interrupted" in printed
    assert not any("/root interacted" in line for line in printed)


def test_an_answer_still_incomplete_after_the_follow_up_says_so(
        tmp_path, monkeypatch, capsys):
    """Two answers without `## Coverage` once came out completed, exit 0, with
    only "the first answer lacked" — read by the orchestrator as a gap the
    follow-up closed. The joined answer is re-judged: the status and exit code
    stay (the answer is kept), and the `Incomplete` line is the one
    `claude_agent` prints, so one search finds it in both reviews."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "partial-twice",
                             expect=r"^## Coverage", timeout=15.0)
    assert outcome.status == "completed" and outcome.error == ""
    assert outcome.text == "## Findings\n1. one thing\n\n2. another thing"
    assert outcome.follow_ups == 1 and outcome.expect_met is False
    assert len(wire.turn_starts(tmp_path)) == 2  # one follow-up, not a loop
    for quiet in (False, True):
        assert cli.print_outcome("w", outcome, quiet=quiet) == 0
        assert ("[codex:w] Incomplete: the answer still lacks /^## Coverage/ "
                "after 1 follow-up(s)") in capsys.readouterr().out


def test_an_empty_completed_follow_up_keeps_the_first_answer(
        tmp_path, monkeypatch, capsys):
    """A follow-up that completes without a word must not erase the first
    answer (`_carry_over` restores it before `join_answers` sees it), and the
    answer is still judged incomplete rather than passed as a gap closed."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "partial-then-silent",
                             expect=r"^## Coverage", timeout=15.0)
    assert outcome.status == "completed" and outcome.error == ""
    assert outcome.text == "## Findings\n1. one thing"
    assert outcome.follow_ups == 1 and outcome.expect_met is False
    assert AgentStore("w").read_answer() == outcome.text
    assert cli.print_outcome("w", outcome, quiet=False) == 0
    printed = capsys.readouterr().out
    # Literal, not `followed_up_note(...)`: the oracle must not share the
    # wording it checks.
    assert "1 follow-up(s): the first answer lacked /^## Coverage/" in printed
    assert ("[codex:w] Incomplete: the answer still lacks /^## Coverage/ "
            "after 1 follow-up(s)") in printed


def test_an_answer_that_matches_is_not_followed_up(tmp_path, monkeypatch):
    """The other half of the contract: a match costs nothing — no second
    turn, no resume, the answer as it came."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "plain",
                             expect=r"fake answer", timeout=10.0)
    assert outcome.status == "completed"
    assert outcome.text == "fake answer"
    assert outcome.follow_ups == 0 and outcome.expect_met is True
    assert len(wire.turn_starts(tmp_path)) == 1
    assert not any(message.get("method") == "thread/resume"
                   for message in wire.received(tmp_path))


def test_each_approval_request_is_answered_in_its_own_shape(tmp_path, monkeypatch):
    """Three different reply shapes share one table, and each wrong one fails
    quietly in its own way: the server BLOCKS on an approval until it is
    answered, so a refusal it cannot parse stalls the turn until the timeout.
    The turn reaching `turn/completed` here is the proof that all four answers
    were accepted."""
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "approvals", timeout=15.0)
    assert outcome.status == "completed"
    assert outcome.text == "approvals handled"
    answers = wire.answers(tmp_path)
    # No `decision` field exists for permissions: an empty granted profile IS
    # the refusal.
    assert answers[9001]["result"] == {"permissions": {}}
    assert answers[9002]["result"] == {"decision": "decline"}
    # The legacy pair uses ReviewDecision, which has no "decline" member.
    assert "denied" in answers[9003]["result"]["decision"]
    assert answers[9005]["result"] == {"decision": "decline"}


def test_an_unknown_server_request_gets_an_error_not_an_invented_answer(
        tmp_path, monkeypatch):
    """A made-up payload can be accepted as a real answer and change what the
    agent does; a JSON-RPC error cannot."""
    wire.run_fake_turn(tmp_path, monkeypatch, "approvals", timeout=15.0)
    unknown = wire.answers(tmp_path)[9004]
    assert "result" not in unknown
    assert unknown["error"]["code"] == -32601


def test_a_server_that_dies_mid_turn_is_reported_instead_of_waited_out(
        tmp_path, monkeypatch):
    """Without the end-of-stream sentinel the client would sit out the whole
    turn timeout and then blame a slow model for a crash."""
    began = time.monotonic()
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "die", timeout=5.0)
    elapsed = time.monotonic() - began
    assert outcome.status == "failed"
    assert "closed the connection" in outcome.error
    assert elapsed < 4.0


def test_the_thread_id_is_handed_over_before_the_turn_ends(tmp_path, monkeypatch):
    """`on_thread` exists so a Ctrl+C mid-turn cannot orphan a live Codex
    thread. The stand-in withholds `turn/completed` until the callback's file
    appears, so a callback that fires only at the end reads as "no"."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    wire.install_fake_server(monkeypatch, tmp_path, "callback")
    seen = tmp_path / "thread_seen"
    outcome = turn.run_turn(
        record=AgentRecord(name="w", cwd=str(tmp_path), sandbox="read-only"),
        prompt="p", resume=False, timeout=30.0,
        on_thread=lambda thread_id: seen.write_text(thread_id, encoding="utf-8"))
    assert outcome.text == "callback:yes"
    assert seen.read_text(encoding="utf-8") == "th-fake"


def test_a_textless_turn_does_not_erase_the_previous_answer(tmp_path, monkeypatch):
    """A turn that hit a usage limit says nothing; writing its empty text over
    `last.md` loses a good report and makes `result` print nothing at all."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    AgentStore("w").write_answer("previous answer")
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "silent")
    assert outcome.text == ""
    assert AgentStore("w").read_answer() == "previous answer"


# --- a turn the service dropped -------------------------------------------

# Every other member of `CodexErrorInfo`. None of these may cost a resend: the
# thread goes over the wire whole, 1.2-1.7M input tokens on a worker, and an
# account limit or a policy refusal answers identically a minute later.
NOT_THE_SERVICES_FAULT = [
    "contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded",
    "rateLimitExceeded", "cyberPolicy", "misalignmentPolicyViolation",
    "unauthorized", "badRequest", "threadRollbackFailed", "sandboxError",
    "activeTurnNotSteerable", "other", "",
]

# ...and the whole enum, as `codex app-server generate-json-schema --out <dir>`
# printed it on 2026-09-13 (`CodexErrorInfo`, thirteen bare strings and five
# single-key objects). The sentinel: a member added upstream belongs to neither
# list, so this is the one place that turns red instead of the new failure
# quietly joining "not ours to retry" — which is how a real outage would stop
# being retried with nobody told (docs/agents/enum-sentinel.md).
CODEX_ERROR_INFO_MEMBERS = frozenset({
    "contextWindowExceeded", "sessionBudgetExceeded", "usageLimitExceeded",
    "rateLimitExceeded", "serverOverloaded", "cyberPolicy",
    "misalignmentPolicyViolation", "internalServerError", "unauthorized",
    "badRequest", "threadRollbackFailed", "sandboxError", "other",
    "httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected", "responseTooManyFailedAttempts",
    "activeTurnNotSteerable",
})


def test_the_drop_predicate_is_pinned_to_the_outage_it_was_written_for():
    """`codexErrorInfo` arrives in two spellings and the terminal one is the
    catch-all: while reconnecting, the server names
    `responseStreamDisconnected`; the error that actually fails the turn
    degrades to `"other"` and only the message still says what happened. Read
    the kind alone and the outage is invisible; read the message alone and
    every unrelated `other` becomes retryable."""
    dropped = LOGGED_DROP["params"]["error"]
    assert service.error_kind(dropped) == "other"
    assert service.is_service_failure("other", dropped["message"])
    reconnecting = LOGGED_RECONNECT["params"]["error"]
    assert service.error_kind(reconnecting) == "responseStreamDisconnected"
    assert service.is_service_failure(
        service.error_kind(reconnecting), reconnecting["message"])
    overloaded = LOGGED_OVERLOAD["params"]["error"]
    assert service.error_kind(overloaded) == "serverOverloaded"
    assert service.is_service_failure(
        service.error_kind(overloaded), overloaded["message"])

    for kind in NOT_THE_SERVICES_FAULT:
        assert not service.is_service_failure(kind, "You've hit your usage limit.")
    assert service.error_kind({"message": "x"}) == ""
    assert service.error_kind("not a payload at all") == ""
    # Every member of the enum is classified by one list or the other, so an
    # upstream addition cannot slip into "not ours to retry" unnoticed.
    assert (service.SERVICE_FAILURE_KINDS | set(NOT_THE_SERVICES_FAULT)
            ) - {""} == CODEX_ERROR_INFO_MEMBERS
    assert not service.SERVICE_FAILURE_KINDS & set(NOT_THE_SERVICES_FAULT)

    # ...and the same line, read the way a turn reads it: the retryable
    # `willRetry` reconnect must not become the verdict, and the verdict's own
    # kind is what the retry decision sees.
    runner = TurnRunner()
    runner.on_event(LOGGED_RECONNECT)
    runner.on_event(LOGGED_DROP)
    runner.on_event(_event("turn/completed", {"turn": {
        "status": "failed", "error": LOGGED_DROP["params"]["error"]}}))
    assert runner.outcome.error_kind == "other"
    assert turn._service_drop(runner.outcome) == "other"
    # A local failure the server never explained carries no kind, so there is
    # nothing to forgive it by — note that the message alone does not qualify
    # it either, which is what makes the kind load-bearing here.
    assert turn._service_drop(TurnOutcome(
        status="failed", error="codex app-server closed the connection")) == ""
    assert turn._service_drop(TurnOutcome(
        status="failed", error=dropped["message"])) == ""
    # ...but a local death that FOLLOWS a service error keeps that error's kind
    # and IS resent: the stream had already broken for the reason this exists
    # for, and `_run_turn_logged` preserves both fields when it catches.
    assert turn._service_drop(TurnOutcome(
        status="failed", error_kind="other",
        error=dropped["message"])) == "other"
    # A timeout is our own bound expiring, whatever the server said before it.
    assert turn._service_drop(TurnOutcome(
        status="timeout", error_kind="serverOverloaded")) == ""


def test_a_dropped_turn_is_resent_on_the_same_thread_after_a_pause(
        tmp_path, monkeypatch, capsys):
    """The 2026-09-13 recovery, automatic: pause, check, resume the SAME
    thread with the continuation message — the thread and the edits in the tree
    survive the drop, so starting over would throw away both."""
    slept, checked, during = [], [], []

    def wait(pause):
        # Read the log from INSIDE the pause: this is the minutes-long window
        # in which the dropped attempt's failed `turn/completed` is the last
        # word, and an operator asking `status` must still be told the turn is
        # alive rather than sent looking for a finished one.
        slept.append(pause)
        trace = AgentStore("w").last_turn()
        during.append((trace.ended, trace.status, trace.retries))

    def alive():
        checked.append(True)
        return True

    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "drop-once",
                             sleep=wait, reachable=alive)

    assert during == [(False, "", 0)]

    assert outcome.status == "completed"
    assert outcome.text == "picked up after the drop"
    assert outcome.service_retries == 1
    assert slept == [turn.SERVICE_RETRY_PAUSES_S[0]]
    assert len(checked) == 1
    # The same thread, resumed — not a second `thread/start`, which would be a
    # fresh agent answering with none of the history it is asked to continue.
    sent = wire.received(tmp_path)
    assert len([m for m in sent if m.get("method") == "thread/start"]) == 1
    resumed = [m for m in sent if m.get("method") == "thread/resume"]
    assert len(resumed) == 1 and resumed[0]["params"]["threadId"] == "th-fake"
    starts = wire.turn_starts(tmp_path)
    assert len(starts) == 2
    assert starts[0]["params"]["input"][0]["text"] == "do the thing"
    assert starts[1]["params"]["input"][0]["text"] == turn.SERVICE_RETRY_PROMPT
    assert "git diff --stat" in turn.SERVICE_RETRY_PROMPT
    # The dropped attempt's work was really done, so it is kept...
    assert (outcome.commands, outcome.file_changes) == (1, 1)
    # ...but the token totals are the SERVER's running count over the thread,
    # not ours to add up: these two readings are the real ones from either side
    # of the outage's resume, and summing them would report 868k of input on an
    # account that was charged 468k.
    assert outcome.input_tokens == 467645
    assert outcome.cached_input_tokens == 415232
    assert outcome.output_tokens == 3075
    assert outcome.context_tokens == 67273
    # A rescued turn is a SUCCESSFUL turn: nothing may be left in `error`,
    # which `print_outcome` reads as the definition of failure.
    assert outcome.error == ""
    assert cli.print_outcome("w", outcome, quiet=True) == 0

    # And it is visible afterwards: one line in the log, which `logs` renders
    # and `status` counts, with the turn still ending once.
    trace = AgentStore("w").last_turn()
    assert trace is not None
    assert trace.ended and trace.status == "completed" and trace.retries == 1
    capsys.readouterr()
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "w"]))
    printed = capsys.readouterr().out
    assert "--- retry 1/2 after a service drop (other): waiting 60.0s" in printed
    assert ("--- retry 1/2 after a service drop (other): paused 60.0s, api "
            "reachable" in printed)


def test_an_unreachable_api_spends_the_slot_without_resending(tmp_path,
                                                              monkeypatch):
    """With no route to OpenAI the attempt is not made at all (user decision
    2026-09-13): the pauses still pass and the checks still happen, but the
    turn is never resent and the refusal says why."""
    slept, checked = [], []

    def down():
        checked.append(False)
        return False

    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "drop-always",
                             sleep=slept.append, reachable=down)

    assert outcome.status == "failed"
    assert outcome.service_retries == 0
    assert len(wire.turn_starts(tmp_path)) == 1
    assert slept == list(turn.SERVICE_RETRY_PAUSES_S)
    assert len(checked) == turn.MAX_SERVICE_RETRIES
    assert "stream disconnected" in outcome.error
    assert f"service unreachable after {turn.MAX_SERVICE_RETRIES} check(s)" \
        in outcome.error
    trace = AgentStore("w").last_turn()
    assert trace is not None and trace.ended and trace.retries == 0


def test_a_service_that_keeps_dropping_stops_at_the_limit(tmp_path, monkeypatch):
    """Two retries and no more: each one resends the whole thread (1.2-1.7M
    input tokens on a worker), so an outage outliving them has to reach the
    operator as the failure it is."""
    slept = []
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "drop-always",
                             sleep=slept.append, reachable=lambda: True)

    assert outcome.status == "failed"
    assert outcome.service_retries == turn.MAX_SERVICE_RETRIES == 2
    assert len(wire.turn_starts(tmp_path)) == turn.MAX_SERVICE_RETRIES + 1
    assert slept == list(turn.SERVICE_RETRY_PAUSES_S)
    assert service.retried_note(turn.MAX_SERVICE_RETRIES) in outcome.error
    # Each attempt ran the same command and made the same edit, and the caller
    # is owed the whole of what its agent did...
    assert outcome.commands == turn.MAX_SERVICE_RETRIES + 1
    assert outcome.file_changes == turn.MAX_SERVICE_RETRIES + 1
    # ...but not three times the bill: every attempt reported the same running
    # thread total, because that is what the number means.
    assert outcome.input_tokens == 400578
    assert outcome.output_tokens == 2869


def test_a_resend_that_fails_for_another_reason_still_names_the_retry(
        tmp_path, monkeypatch):
    """The error message is the half of the outcome that gets pasted into a
    report. A turn dropped, resent, and then refused for an unrelated reason
    has spent minutes and a whole thread on the wire, and the first draft said
    so only when the SECOND failure happened to be a drop too."""
    slept = []
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "drop-then-refuse",
                             sleep=slept.append, reachable=lambda: True)

    assert outcome.status == "failed"
    assert outcome.error_kind == "usageLimitExceeded"
    assert "usage limit" in outcome.error
    assert service.retried_note(1) in outcome.error
    assert outcome.service_retries == 1
    # And the second failure is not a drop, so the second slot goes unused.
    assert len(wire.turn_starts(tmp_path)) == 2
    assert slept == [turn.SERVICE_RETRY_PAUSES_S[0]]


def test_a_failure_that_is_not_the_services_is_never_resent(tmp_path, monkeypatch):
    """A usage limit is the loudest of these: it fails a turn exactly like an
    outage does, answers identically a minute later, and a resend of the thread
    is the most expensive thing this script can do."""
    slept, checked = [], []
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "refuse",
                             sleep=slept.append,
                             reachable=lambda: checked.append(True) or True)

    assert outcome.status == "failed"
    assert outcome.error_kind == "usageLimitExceeded"
    assert outcome.service_retries == 0
    assert (slept, checked) == ([], [])
    assert len(wire.turn_starts(tmp_path)) == 1
    # Nothing was appended to the message either: a turn nobody retried must
    # not be explained as one that was.
    assert "retried" not in outcome.error and "unreachable" not in outcome.error


def test_the_summary_line_owns_up_to_a_retried_turn(capsys):
    """The caller sees one summary line and the answer. Without this the retry
    is invisible from there: minutes of wall-clock and a second attempt's
    numbers, reported as an ordinary turn that simply took longer."""
    cli.print_outcome("w", TurnOutcome(status="completed", text="done",
                                       service_retries=2), quiet=False)
    assert service.retried_note(2) in capsys.readouterr().out
    cli.print_outcome("w", TurnOutcome(status="completed", text="done"),
                      quiet=False)
    assert "retried" not in capsys.readouterr().out


def test_a_turn_waiting_between_attempts_is_still_running(tmp_path, monkeypatch,
                                                          capsys):
    """A dropped attempt writes its own failed `turn/completed`, and the log is
    then silent for the length of the pause. Left standing, that verdict would
    have `status` report the turn as finished-and-failed while its next attempt
    was running — and the closing line, the one thing that really ends a turn,
    is written once, after the last attempt."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="busy", thread_id="th-1",
                               sandbox="workspace-write"))
    lines = [
        {"_meta": events.TURN_META, "name": "busy", "prompt": "p",
         "at": "2026-09-13 10:00:00", "timeout": 600.0},
        {"method": protocol.N_ITEM_COMPLETED, "params": {"item": {
            "type": "commandExecution", "command": "git diff --stat"}}},
        LOGGED_DROP,
        _event(protocol.N_TURN_COMPLETED, {"turn": {
            "status": "failed", "error": LOGGED_DROP["params"]["error"]}}),
        {"_meta": events.RETRY_META, "name": "busy", "attempt": 1, "of": 2,
         "reason": "other", "paused_s": 60.0, "waiting": True,
         "at": "2026-09-13 10:00:30"},
        {"_meta": events.RETRY_META, "name": "busy", "attempt": 1, "of": 2,
         "reason": "other", "paused_s": 60.0, "reachable": True,
         "at": "2026-09-13 10:01:30"},
    ]
    busy = AgentStore("busy")
    busy.append_events(lines)

    trace = busy.last_turn()
    assert trace is not None
    assert not trace.ended and trace.retries == 1
    cli.cmd_status(cli.build_parser().parse_args(["status", "busy"]))
    printed = capsys.readouterr().out
    assert "RUNNING since 2026-09-13 10:00:00" in printed
    assert service.retried_note(1) in printed

    # The variants below each get a log of their own: appended to the one
    # above, they would be judged as part of the same turn.
    #
    # The waiting line alone — the state that lasts for minutes — already
    # reopens the turn, and counts no resend because none has been made.
    waiting_only = AgentStore("waiting-only")
    waiting_only.append_events(lines[:-1])
    trace = waiting_only.last_turn()
    assert not trace.ended and trace.retries == 0

    # A slot the reachability check refused is not a resend either: nothing was
    # sent and the agent redid no work.
    refused = AgentStore("refused")
    refused.append_events(lines[:-1] + [{**lines[-1], "reachable": False}])
    assert refused.last_turn().retries == 0


def test_the_reachability_probe_believes_any_http_answer(monkeypatch):
    """Unauthenticated on purpose: a 401 or a 405 proves the machine can reach
    OpenAI, which is the whole question — whether this account may run the
    model is what the resent turn itself is about to find out. A 5xx is the one
    status that means "cannot serve", and everything below HTTP is a plain no."""

    class _Reply:
        def __init__(self, status):
            self.status = status

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def answering(status):
        return lambda request, timeout=None: _Reply(status)

    def refusing(error):
        def open_it(request, timeout=None):
            raise error
        return open_it

    for status in (200, 401, 405):
        monkeypatch.setattr(service.urllib.request, "urlopen", answering(status))
        assert service.api_reachable() is True
    monkeypatch.setattr(service.urllib.request, "urlopen", answering(503))
    assert service.api_reachable() is False
    for code in (401, 405):
        monkeypatch.setattr(service.urllib.request, "urlopen", refusing(
            urllib.error.HTTPError(service.API_PROBE_URL, code, "no", {}, None)))
        assert service.api_reachable() is True
    monkeypatch.setattr(service.urllib.request, "urlopen", refusing(
        urllib.error.HTTPError(service.API_PROBE_URL, 502, "no", {}, None)))
    assert service.api_reachable() is False
    for broken in (urllib.error.URLError("name resolution failed"),
                   TimeoutError("timed out"), OSError("connection refused")):
        monkeypatch.setattr(service.urllib.request, "urlopen", refusing(broken))
        assert service.api_reachable() is False


# --- registry -------------------------------------------------------------


def _write_one(index: int) -> None:
    """One registry writer, at module level so `spawn` can pickle it.

    `cwd` carries the CHILD's view of `state_dir()`, which is the only way the
    parent can tell an honoured environment override from a lucky one.
    """
    Registry().put(AgentRecord(name=f"racer{index}", thread_id=f"t{index}",
                               cwd=str(paths.state_dir())))


def test_parallel_writers_all_survive_in_the_registry(tmp_path, monkeypatch):
    """Running agents in parallel is the point, and they all finish by writing
    here; an unlocked read-modify-write silently drops all but one of them."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    context = mp.get_context("spawn")
    workers = [context.Process(target=_write_one, args=(i,)) for i in range(WRITERS)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
    finally:
        # Non-daemon children: a writer left alive would hang the interpreter
        # at exit long after this test reported its failure.
        for worker in workers:
            if worker.is_alive():
                worker.terminate()

    stored = Registry().load()
    assert sorted(stored) == sorted(f"racer{i}" for i in range(WRITERS))
    assert all(stored[f"racer{i}"].thread_id == f"t{i}" for i in range(WRITERS))


def test_the_environment_override_reaches_a_spawned_child(tmp_path, monkeypatch):
    """A fresh interpreter must resolve the state directory the same way, or
    the parallel test above would be writing to the real registry while
    claiming to be isolated."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    worker = mp.get_context("spawn").Process(target=_write_one, args=(0,))
    try:
        worker.start()
        worker.join(timeout=60)
    finally:
        if worker.is_alive():
            worker.terminate()
    assert worker.exitcode == 0
    stored = Registry().load()
    assert stored["racer0"].cwd == str(tmp_path)


def test_registry_round_trips_a_record(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.put(AgentRecord(name="rev", thread_id="abc", sandbox="read-only",
                             turns=3))
    stored = registry.get("rev")
    assert stored is not None
    assert (stored.thread_id, stored.sandbox, stored.turns) == ("abc", "read-only", 3)
    assert registry.drop("rev") is True
    assert registry.get("rev") is None


def test_an_unreadable_registry_is_an_error_not_an_empty_one(tmp_path, monkeypatch):
    """"Missing" and "unreadable" as one answer meant silent mass deletion: the
    failed read returned {}, the caller added its own record and wrote back a
    registry containing only itself."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.path.parent.mkdir(parents=True, exist_ok=True)
    registry.path.write_text('{"keeper": {"threadId": "keep", }', encoding="utf-8")

    with pytest.raises(RegistryError):
        registry.load()
    with pytest.raises(RegistryError):
        registry.put(AgentRecord(name="newcomer", thread_id="n1"))
    assert "keep" in registry.path.read_text(encoding="utf-8")

    registry.path.write_text('["not", "an", "object"]', encoding="utf-8")
    with pytest.raises(RegistryError):
        registry.load()


def test_a_stale_lock_is_broken_rather_than_waited_on_forever(tmp_path, monkeypatch):
    """`taskkill /F` (which this script itself uses) leaves the lock file
    behind. Waiting it out and then writing unlocked would disable mutual
    exclusion permanently, so the lock must be REMOVED, not ignored."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(registry_module, "LOCK_WAIT_S", 0.2)
    registry = Registry()
    registry.lock_path.parent.mkdir(parents=True, exist_ok=True)
    registry.lock_path.write_text("dead pid", encoding="utf-8")

    registry.put(AgentRecord(name="rev", thread_id="abc"))

    assert registry.get("rev").thread_id == "abc"
    assert not registry.lock_path.exists()


def test_reserve_refuses_a_name_that_is_already_taken(tmp_path, monkeypatch):
    """Claiming happens before the turn, which is minutes long: two spawns that
    both passed a separate check both ran, and the loser's thread ended up with
    no registry entry at all."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.reserve(AgentRecord(name="rev", thread_id="first"), replace=False)

    with pytest.raises(SystemExit):
        registry.reserve(AgentRecord(name="rev", thread_id="second"), replace=False)
    assert registry.get("rev").thread_id == "first"

    registry.reserve(AgentRecord(name="rev", thread_id="third"), replace=True)
    assert registry.get("rev").thread_id == "third"


def test_reserve_refuses_a_name_that_differs_only_in_case(tmp_path, monkeypatch):
    """Windows paths are case-insensitive while dict keys are not, so `review`
    and `Review` would be two records sharing one state directory and one
    `last.md`."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.reserve(AgentRecord(name="review", thread_id="first"), replace=False)

    for replace in (False, True):
        with pytest.raises(SystemExit):
            registry.reserve(AgentRecord(name="Review", thread_id="clash"),
                             replace=replace)
    assert sorted(registry.load()) == ["review"]


def test_a_busy_registry_file_is_retried_not_lost(tmp_path, monkeypatch):
    """On Windows `os.replace` FAILS while a reader holds the target, and an
    unhandled failure costs the caller the answer a finished turn already paid
    for."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    real_replace = Path.replace
    refusals = []

    def flaky_replace(self, target):
        if len(refusals) < 2:
            refusals.append(str(target))
            raise PermissionError(5, "WinError 5: another process holds it")
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    Registry().put(AgentRecord(name="rev", thread_id="abc"))

    monkeypatch.setattr(Path, "replace", real_replace)
    assert len(refusals) == 2
    assert Registry().get("rev").thread_id == "abc"


# --- bookkeeping ----------------------------------------------------------


def test_a_completed_turn_is_counted_and_stamped(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    record = AgentRecord(name="rev", thread_id="", turns=2,
                         last_turn="2000-01-01 00:00:00")
    registry.put(record)

    cli._remember_turn(registry, record, TurnOutcome(
        thread_id="th-9", status="completed", text="done"))

    stored = registry.get("rev")
    assert stored.turns == 3
    assert stored.last_turn != "2000-01-01 00:00:00"
    assert stored.thread_id == "th-9"


def test_a_failed_turn_is_not_counted_but_its_thread_is_kept(tmp_path, monkeypatch):
    """A turn that never started is not a turn: counting it made `status` claim
    work that produced no answer. The thread id still has to be stored, or the
    agent becomes unresumable."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    record = AgentRecord(name="rev", thread_id="", turns=2,
                         last_turn="2000-01-01 00:00:00")
    registry.put(record)

    cli._remember_turn(registry, record, TurnOutcome(
        thread_id="th-9", status="failed", error="usage limit reached"))

    stored = registry.get("rev")
    assert stored.turns == 2
    assert stored.last_turn == "2000-01-01 00:00:00"
    assert stored.thread_id == "th-9"


def test_send_refuses_a_record_whose_thread_was_never_stored(tmp_path, monkeypatch):
    """Without the guard `send` fell through to `thread/start`: a brand new
    thread with none of the agent's history, answering as if it had context."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="ghost", thread_id=""))
    turns = []

    def spy(**kwargs):
        turns.append(kwargs)
        return TurnOutcome(thread_id="th-new", status="completed", text="hi")

    monkeypatch.setattr(cli, "run_turn", spy)
    args = cli.build_parser().parse_args(["send", "ghost", "hello"])
    with pytest.raises(SystemExit):
        cli.cmd_send(args)
    assert turns == []


def test_send_resumes_the_recorded_thread(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="rev", thread_id="th-1"))
    turns = []

    def spy(**kwargs):
        turns.append(kwargs)
        return TurnOutcome(thread_id="th-1", status="completed", text="hi")

    monkeypatch.setattr(cli, "run_turn", spy)
    args = cli.build_parser().parse_args(["send", "rev", "hello", "there"])
    assert cli.cmd_send(args) == 0
    assert len(turns) == 1
    assert turns[0]["resume"] is True
    assert turns[0]["prompt"] == "hello there"
    assert turns[0]["record"].thread_id == "th-1"


def test_result_and_logs_read_exactly_what_the_turn_wrote(tmp_path, monkeypatch,
                                                          capsys):
    """`turn` writes the two per-agent files and four `cli` commands read, move
    and print them, so their names are a contract between two modules (kept in
    `paths`). Spell either side differently and nothing raises: `result` and
    `logs` just report an agent that has never run."""
    assert wire.run_fake_turn(tmp_path, monkeypatch, "plain").text == "fake answer"
    capsys.readouterr()

    cli.cmd_result(cli.build_parser().parse_args(["result", "w"]))
    assert "fake answer" in capsys.readouterr().out

    cli.cmd_logs(cli.build_parser().parse_args(["logs", "w"]))
    printed = capsys.readouterr().out
    assert "$ git status" in printed
    assert "~ update a.py" in printed


def test_status_shows_the_sandbox_the_turn_will_actually_use(tmp_path, monkeypatch,
                                                             capsys):
    """`status` printing its own default told the operator an agent could edit
    files while its turns ran read-only and refused every write. Both sides now
    read `AgentRecord.effective_sandbox`, and this is the pin that they do."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.put(AgentRecord(name="old", thread_id="t1"))  # written before the field
    registry.put(AgentRecord(name="rw", thread_id="t2", sandbox="workspace-write"))

    cli.cmd_status(cli.build_parser().parse_args(["status"]))
    printed = capsys.readouterr().out

    assert "read-only" in printed.splitlines()[0]
    assert "workspace-write" in printed.splitlines()[1]
    assert sandbox.thread_params(registry.get("old"))["sandbox"] == "read-only"
    assert sandbox.thread_params(registry.get("rw"))["sandbox"] == "workspace-write"


def test_a_reviewer_prompt_with_numbered_questions_expects_every_number():
    """`rev_lotrules` (2026-09-13) answered questions 4 and 5, listed exactly
    those under `## Coverage`, and the bare heading let it through. The
    default now names each question in order; a prompt without numbered
    lines (a `send` follow-up) keeps the bare heading, and an explicit
    `--expect` still wins."""
    from llm_agents.codex.roles import expected_answer
    from llm_agents.answer_contract import coverage_expect
    prompt = "Review X.\n\n1. **A.** why\n2. **B.** how\n\n3. C\nnot 4. a question"
    pattern = coverage_expect(prompt)
    assert pattern == "(?ms)^## Coverage.*^1:.*^2:.*^3:"
    import re
    full = "## Findings\nnone\n## Coverage\n1: answered\n2: nothing found\n3: answered\n"
    assert re.search(pattern, full, re.MULTILINE)
    partial = "## Coverage\n4: answered\n5: answered\n"
    assert not re.search(pattern, partial, re.MULTILINE)
    assert not re.search(pattern, "## Coverage\n2: answered\n1: answered\n3: x",
                         re.MULTILINE)
    assert coverage_expect("continue where you stopped") == "^## Coverage"
    assert expected_answer("reviewer", None, prompt) == pattern
    assert expected_answer("reviewer", "^## Verdict", prompt) == "^## Verdict"
    assert expected_answer("worker", None, prompt) == "^### Not done"


def test_the_reviewer_briefing_is_the_one_claude_agent_reviews_with():
    """The orchestrator merges a Codex and a Claude review of one commit; the
    two briefings are one text (`llm_agents.answer_contract`), so neither
    wrapper can drift the skeleton alone."""
    from llm_agents.codex.roles import role_briefing
    from llm_agents.answer_contract import REVIEWER_BRIEFING
    from llm_agents.claude.protocol import review_instructions
    assert role_briefing("reviewer") == REVIEWER_BRIEFING
    assert review_instructions().endswith(REVIEWER_BRIEFING)


@pytest.mark.parametrize("role,default", [
    ("", ""), ("worker", "^### Not done"), ("reviewer", "^## Coverage"),
])
@pytest.mark.parametrize("explicit", [None, "^## Coverage", ""])
def test_spawn_and_send_resolve_expect_from_the_persisted_role(
        tmp_path, monkeypatch, role, default, explicit):
    """Exercise both CLI consumers: a send has no --role flag of its own."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    turns = []

    def capture(**kwargs):
        turns.append(kwargs)
        return TurnOutcome(thread_id="th-role", status="completed", text="ok")

    monkeypatch.setattr(cli, "run_turn", capture)
    options = [] if explicit is None else [f"--expect={explicit}"]
    args = cli.build_parser().parse_args([
        "spawn", "role", "task", "--cwd", str(tmp_path), "--quiet",
        *(["--role", role] if role else []), *options,
    ])
    assert cli.cmd_spawn(args) == 0
    stored = Registry().get("role")
    assert stored.role == role
    assert stored.thread_id == "th-role"
    assert cli.cmd_send(cli.build_parser().parse_args([
        "send", "role", "continue", "--quiet", *options])) == 0
    assert [call["expect"] for call in turns] == [
        default if explicit is None else explicit] * 2
    assert [call["resume"] for call in turns] == [False, True]
    assert [call["record"].role for call in turns] == [role, role]
    # An explicit override affects only that invocation, not the role's next turn.
    assert cli.cmd_send(cli.build_parser().parse_args([
        "send", "role", "continue again", "--quiet"])) == 0
    assert turns[-1]["expect"] == default


@pytest.mark.parametrize("policy", ["read-only", "scratch-write"])
@pytest.mark.parametrize("existing", [False, True])
def test_worker_refusal_preserves_the_registry_and_previous_answer(
        tmp_path, monkeypatch, policy, existing):
    """Reject before reserve, including reuse of a name with an answer to lose."""
    state = tmp_path / "state"
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(state))
    registry = Registry()
    registry.put(AgentRecord(name="other", thread_id="th-other"))
    if existing:
        registry.put(AgentRecord(name="worker", thread_id="th-old", turns=3))
        AgentStore("worker").write_answer("old answer")
    before = registry.path.read_bytes()

    def never(*args, **kwargs):
        raise AssertionError("the name must not be reserved and the turn must not start")

    monkeypatch.setattr(Registry, "reserve", never)
    monkeypatch.setattr(cli, "run_turn", never)
    args = cli.build_parser().parse_args([
        "spawn", "worker", "task", "--cwd", str(tmp_path),
        "--role", "worker", "--sandbox", policy,
    ])
    with pytest.raises(SystemExit, match="--role worker needs workspace-write or danger-full-access"):
        cli.cmd_spawn(args)
    assert registry.path.read_bytes() == before
    if existing:
        assert AgentStore("worker").read_answer() == "old answer"
        assert AgentStore("worker").archived() == []
    else:
        assert registry.get("worker") is None


@pytest.mark.parametrize("command", ["spawn", "send"])
def test_an_invalid_expect_regex_is_refused_before_the_turn(
        tmp_path, monkeypatch, command):
    """`--expect` is compiled at the CLI boundary, as `claude_agent` does: an
    invalid regex used to surface as `re.error` from `answers` only after the
    first turn had been paid for. Nothing is reserved,
    touched or sent."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    registry = Registry()
    registry.put(AgentRecord(name="rev", thread_id="th-old", cwd=str(tmp_path),
                             role="reviewer"))
    before = registry.path.read_bytes()

    def never(*args, **kwargs):
        raise AssertionError("the name must not be claimed and the turn must not start")

    monkeypatch.setattr(Registry, "reserve", never)
    monkeypatch.setattr(cli, "run_turn", never)
    argv = ([command, "rev", "task", "--expect", "("]
            + (["--cwd", str(tmp_path)] if command == "spawn" else []))
    with pytest.raises(SystemExit, match=r"Invalid --expect regex '\('"):
        core_main(argv)
    assert registry.path.read_bytes() == before


@pytest.mark.parametrize("policy", ["read-only", "scratch-write"])
def test_a_persisted_worker_under_an_unwritable_sandbox_is_refused_on_send(
        tmp_path, monkeypatch, policy):
    """The refusal is a rule of the TURN, not of `spawn`: a record written by
    an older version (or by hand) reaches `send` without passing `cmd_spawn`,
    and resumed it would be briefed to edit a tree it cannot write (found in
    review). `prepare_workspace` runs on every turn, so it is the gate."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    record = AgentRecord(name="worker", thread_id="th-old", cwd=str(tmp_path),
                         sandbox=policy, role="worker")
    with pytest.raises(SystemExit, match="--role worker needs workspace-write"):
        sandbox.prepare_workspace(record)
    record.role = "reviewer"
    sandbox.prepare_workspace(record)  # a reviewer is fine under every mode


@pytest.mark.parametrize("role,policy", [
    ("reviewer", "read-only"), ("reviewer", "scratch-write"),
    ("reviewer", "workspace-write"), ("reviewer", "danger-full-access"),
    ("worker", "workspace-write"), ("worker", "danger-full-access"),
])
def test_spawn_accepts_each_allowed_role_sandbox_pair(
        tmp_path, monkeypatch, role, policy):
    """Reviewers need no project writes; workers need them even before a turn."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "state"))
    turns = []

    def capture(**kwargs):
        turns.append(sandbox.thread_params(kwargs["record"]))
        return TurnOutcome(thread_id="th-accepted", status="completed", text="ok")

    monkeypatch.setattr(cli, "run_turn", capture)
    args = cli.build_parser().parse_args([
        "spawn", "accepted", "task", "--cwd", str(tmp_path), "--quiet",
        "--role", role, "--sandbox", policy,
    ])
    assert cli.cmd_spawn(args) == 0
    assert len(turns) == 1
    stored = Registry().get("accepted")
    assert (stored.role, stored.sandbox, stored.thread_id) == (role, policy, "th-accepted")


def test_registry_round_trips_roles_and_status_keeps_old_records_unchanged(
        tmp_path, monkeypatch, capsys):
    """Load an actual old JSON entry lacking role, not a new empty-role record."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.path.write_text(json.dumps({"old": {"threadId": "t-old"}}),
                             encoding="utf-8")
    assert registry.get("old").role == ""
    for role in ("worker", "reviewer"):
        registry.put(AgentRecord(name=role, thread_id=f"t-{role}",
                                 sandbox="workspace-write", role=role))
        assert Registry().get(role).role == role
    assert Registry().get("old").role == ""
    cli.cmd_status(cli.build_parser().parse_args(["status"]))
    assert capsys.readouterr().out.splitlines() == [
        "old: thread t-old · 0 turns · read-only · last -",
        "reviewer: thread t-reviewer · 0 turns · workspace-write · role reviewer · last -",
        "worker: thread t-worker · 0 turns · workspace-write · role worker · last -"
        f" · report={sandbox.report_path(registry.get('worker'))}",
    ]


# (command line, prompt it parses to — or the error text it must print).
# Every row holds on 3.9-3.13 alike; the rules are `cli._PromptParser`'s.
_PROMPT_COMMAND_LINES = [
    ("spawn w a b", ["a", "b"]),
    ("send w a b", ["a", "b"]),
    ("send w", []),
    ("spawn w --quiet a b", ["a", "b"]),
    ("spawn --quiet w a", ["a"]),
    ("spawn w a --quiet b", ["a", "b"]),
    ("spawn w a --quiet b --effort high c", ["a", "b", "c"]),
    ("spawn w a b --tim 5 c", ["a", "b", "c"]),
    ("spawn w --sandbox read-only a --keep b", ["a", "b"]),
    ("spawn w --cwd X hello", ["hello"]),
    ("send w --timeout 5 -", ["-"]),
    ("spawn w a --quiet -", ["a", "-"]),
    ("spawn w --quiet -- -x", ["-x"]),
    ("spawn w a -- --quiet -- b", ["a", "--quiet", "--", "b"]),
    ("send w -- -", ["-"]),
    ("spawn w --quiet a -5", "unrecognized arguments: -5"),
    ("spawn w --quiet -x", "unrecognized arguments: -x"),
    ("spawn w a --bogus b", "unrecognized arguments: --bogus b"),
    ("send w a --bogus b -- c", "unrecognized arguments: --bogus b"),
    ("spawn w a --qiet b", "unrecognized arguments: --qiet b"),
    ("spawn w --e high a", "ambiguous option: --e"),
]


@pytest.mark.parametrize("line,expected", _PROMPT_COMMAND_LINES,
                         ids=[line for line, _ in _PROMPT_COMMAND_LINES])
def test_prompt_command_lines_parse_alike(line, expected, capsys):
    """`spawn` and `send` read their prompt words by `cli._PromptParser`'s
    rules, not argparse's, which changed between versions: options anywhere
    before `--`, everything after it prompt text, and an unknown option an
    error naming it and the words after it rather than prompt text."""
    argv = line.split()
    parse = cli.build_parser().parse_args
    if isinstance(expected, list):
        args = parse(argv)
        assert args.prompt == expected
        assert args.name == "w"
        return
    with pytest.raises(SystemExit):
        parse(argv)
    err = capsys.readouterr().err
    assert expected in err
    # The error names the words from the first unknown one on, and nothing else.
    if expected.startswith("unrecognized"):
        assert err.rstrip().endswith(expected)


def test_prompt_option_values_still_parse():
    """The options themselves are argparse's: values, abbreviations."""
    args = cli.build_parser().parse_args(
        "spawn w a --tim 5 --effort high --sandbox read-only b".split())
    assert (args.timeout, args.effort, args.sandbox) == (5.0, "high", "read-only")


def test_a_mistyped_cwd_is_refused_instead_of_created(tmp_path, monkeypatch):
    """`run_turn` used to create whatever workspace it was handed, so
    `spawn rev --sandbox read-only --cwd <repo>/revieww` put a stray directory
    INTO the project — under the one mode that exists to touch nothing."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    project = tmp_path / "project"
    project.mkdir()
    typo = project / "revieww"

    def never(**kwargs):
        raise AssertionError("the turn must not start")

    monkeypatch.setattr(cli, "run_turn", never)
    args = cli.build_parser().parse_args(
        ["spawn", "rev", "--sandbox", "read-only", "--cwd", str(typo), "hello"])
    with pytest.raises(SystemExit):
        cli.cmd_spawn(args)

    assert not typo.exists()
    assert Registry().get("rev") is None  # nor was the name claimed


def test_a_spawn_that_cannot_read_its_prompt_destroys_nothing(tmp_path, monkeypatch):
    """Everything `spawn` does before the turn is destructive — `reserve`
    overwrites the record, the archive moves `last.md` aside — so a typo in
    `--prompt-file` used to cost a working agent for a turn that never ran:
    thread id gone, answer archived, `send` refusing it as "no thread
    recorded"."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.put(AgentRecord(name="rev", thread_id="th-old", turns=3))
    AgentStore("rev").write_answer("old answer")

    def never(**kwargs):
        raise AssertionError("the turn must not start without a prompt")

    monkeypatch.setattr(cli, "run_turn", never)
    args = cli.build_parser().parse_args(
        ["spawn", "rev", "--prompt-file", str(tmp_path / "typo.md")])
    with pytest.raises(OSError):
        cli.cmd_spawn(args)

    stored = registry.get("rev")
    assert (stored.thread_id, stored.turns) == ("th-old", 3)
    assert AgentStore("rev").read_answer() == "old answer"
    assert AgentStore("rev").archived() == []


# --- event parsing --------------------------------------------------------


def _event(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "method": method, "params": params}


def _completed_item(item: dict) -> dict:
    return _event("item/completed", {"item": item})


def test_completed_item_decoding_and_text_extraction_have_one_home():
    item = {"type": "agentMessage", "content": [
        {"type": "text", "text": "first"},
        {"type": "image", "text": "ignored"},
        {"type": "text", "text": "second"},
    ]}

    assert events.completed_item(_completed_item(item)) is item
    assert events.completed_item(_event("turn/completed", {})) is None
    assert events.text_of(item) == "first\nsecond"


def test_turn_runner_reads_the_final_message_and_counts_the_work():
    """`agentMessage` carries a FLAT `text` field; reading it as a content list
    (the shape used by user input) would report an empty answer for a turn that
    actually produced one."""
    runner = TurnRunner()
    runner.on_event(_event("thread/started", {"thread": {"id": "th-1"}}))
    runner.on_event(_completed_item({"type": "commandExecution", "command": "git log"}))
    runner.on_event(_completed_item({"type": "fileChange", "changes": [
        {"path": "a.py", "kind": {"type": "update"}},
        {"path": "b.py", "kind": {"type": "add"}}]}))
    runner.on_event(_completed_item({"type": "agentMessage", "text": "first"}))
    runner.on_event(_completed_item({"type": "agentMessage", "text": "final answer"}))
    runner.on_event(_event("thread/tokenUsage/updated", {"tokenUsage": {
        "last": {"inputTokens": 1, "cachedInputTokens": 0, "outputTokens": 2},
        "total": {"inputTokens": 100, "cachedInputTokens": 30, "outputTokens": 7}}}))
    runner.on_event(_event("turn/completed", {"turn": {"status": "completed"}}))

    assert runner.done is True
    assert runner.outcome.thread_id == "th-1"
    assert runner.outcome.text == "final answer"
    assert runner.outcome.commands == 1
    assert runner.outcome.file_changes == 2
    assert runner.outcome.input_tokens == 100
    assert runner.outcome.output_tokens == 7
    assert runner.outcome.error == ""


def test_a_live_turn_and_a_re_read_log_count_the_same_work(tmp_path, monkeypatch,
                                                           capsys):
    """The summary line reports what `TurnRunner` counted while the turn ran;
    `status` re-derives the same two numbers from the log afterwards, and
    `logs` prints a line per edit. A second spelling of "a fileChange is as
    many edits as it lists" would let one turn be `1 cmd, 3 edits` in one place
    and something else in the other, so all three read `events.item_work` —
    which the numbers alone cannot show (a copied branch counts the same), so
    the calls are counted too."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    stream = [
        _completed_item({"type": "commandExecution", "command": "git log"}),
        _completed_item({"type": "fileChange", "changes": [
            {"path": "a.py"}, {"path": "b.py"}]}),
        _completed_item({"type": "fileChange"}),   # the shape with no list
        _completed_item({"type": "agentMessage", "text": "done"}),
        _completed_item({"type": "mcpToolCall", "server": "s", "tool": "t"}),
    ]
    kinds = [message["params"]["item"]["type"] for message in stream]
    counted: list[str] = []
    real_item_work = events.item_work

    def spy(item):
        counted.append(str(item.get("type")))
        return real_item_work(item)

    # Patched on BOTH consumers because each holds its own binding; a consumer
    # that stopped calling it would keep the import and show up as silence.
    monkeypatch.setattr(events, "item_work", spy)
    monkeypatch.setattr(turn, "item_work", spy)

    runner = TurnRunner()
    for message in stream:
        runner.on_event(message)
    live, counted[:] = list(counted), []

    store = AgentStore("w")
    opened = {"_meta": events.TURN_META, "at": "2026-09-13 10:00:00",
              "timeout": 60.0}
    store.append_events([opened, *stream])
    trace = store.last_turn()
    replayed = counted[:]

    assert (runner.outcome.commands, runner.outcome.file_changes) == (1, 3)
    assert trace is not None
    assert (trace.commands, trace.file_changes) == (runner.outcome.commands,
                                                    runner.outcome.file_changes)
    assert live == kinds and replayed == kinds

    # The third reader of the same turn. A fileChange carrying no list is one
    # edit for the counters and used to be no line at all here, so `status`
    # said "3 edits" over a `logs` output that showed two.
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "w"]))
    printed = capsys.readouterr().out
    edits = [line for line in printed.splitlines() if line.startswith("~")]
    assert len(edits) == trace.file_changes


def test_a_failed_turn_reports_the_servers_reason():
    runner = TurnRunner()
    runner.on_event(_event("turn/completed", {"turn": {
        "status": "failed", "error": {"message": "usage limit reached"}}}))
    assert runner.done is True
    assert runner.outcome.status == "failed"
    assert "usage limit" in runner.outcome.error


def test_a_verdict_with_no_error_object_keeps_the_reason_already_given():
    """`turn/completed` may carry `status: failed` and no error at all, and
    the loose `error` notification before it is then the only thing that named
    the reason. Overwriting it with the word "failed" lost the message the
    caller is shown AND the kind's company — leaving a service drop that reads
    as an unexplained failure and is therefore never resent."""
    runner = TurnRunner()
    runner.on_event(LOGGED_DROP)
    runner.on_event(_event("turn/completed", {"turn": {"status": "failed"}}))
    assert runner.outcome.error == LOGGED_DROP["params"]["error"]["message"]
    assert turn._service_drop(runner.outcome) == "other"


def test_a_retryable_error_is_not_the_turns_verdict():
    """`willRetry` means the server is handling it; recording it would report a
    failure for a turn that then completes normally."""
    runner = TurnRunner()
    runner.on_event(_event("error", {"error": {"message": "stream dropped"},
                                     "willRetry": True}))
    runner.on_event(_event("turn/completed", {"turn": {"status": "completed"}}))
    assert runner.outcome.error == ""
    assert runner.outcome.status == "completed"


# --- options --------------------------------------------------------------


@pytest.mark.parametrize("bad", ["../escape", "sub/dir", "", "..", "a b"])
def test_agent_names_that_would_escape_the_state_directory_are_refused(bad):
    with pytest.raises(Exception):
        agent_name(bad)


@pytest.mark.parametrize("bad", ["rev.", "rev..", "CON", "nul.txt", "com1", "LPT9"])
def test_agent_names_that_collide_with_windows_path_rules_are_refused(bad):
    """The case clash `reserve` refuses, arriving by two other doors: Win32
    strips a trailing dot, so `rev.` is a second registry key over `rev`'s one
    directory and one `last.md`; and a device name is a directory that cannot be
    created at all, leaving a claimed record with nowhere to log."""
    with pytest.raises(Exception):
        agent_name(bad)


def test_ordinary_dotted_names_still_pass():
    """The guard above is about Windows path rules, not about dots: a name like
    `rev.2` is how a caller numbers a second run, and refusing it would push
    them into names that collide in worse ways."""
    assert agent_name("rev.2") == "rev.2"
    assert agent_name("console") == "console"


def test_status_hides_agents_nobody_will_resume(tmp_path, monkeypatch, capsys):
    """A registry of a hundred old review threads printed all of them on every
    `status`. The recent ones and the undated ones show; the rest are a count
    and one flag away."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 30 * 86400))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    registry.put(AgentRecord(name="ancient", thread_id="t1", last_turn=old))
    registry.put(AgentRecord(name="fresh", thread_id="t2", last_turn=now))
    registry.put(AgentRecord(name="undated", thread_id="t3"))

    cli.cmd_status(cli.build_parser().parse_args(["status"]))
    printed = capsys.readouterr().out
    assert "fresh:" in printed and "undated:" in printed
    assert "ancient:" not in printed
    assert "1 agent(s) idle" in printed and "--all" in printed

    cli.cmd_status(cli.build_parser().parse_args(["status", "--all"]))
    printed = capsys.readouterr().out
    assert "ancient:" in printed and "idle" not in printed

    # Asking for one agent by name is never filtered.
    cli.cmd_status(cli.build_parser().parse_args(["status", "ancient"]))
    assert "ancient:" in capsys.readouterr().out

    # A wider window is the flag's, not the constant's.
    cli.cmd_status(cli.build_parser().parse_args(["status", "--days", "60"]))
    assert "ancient:" in capsys.readouterr().out

    # Zero is "no window", not "nothing is recent enough" (review 2026-09-12).
    cli.cmd_status(cli.build_parser().parse_args(["status", "--days", "0"]))
    printed = capsys.readouterr().out
    assert "ancient:" in printed and "fresh:" in printed and "idle" not in printed


def test_status_counts_a_resumed_turn_as_activity_even_when_it_fails(
        tmp_path, monkeypatch, capsys):
    """`last_turn` is stamped only by a COMPLETED turn, so an agent whose
    `send` timed out a minute ago stayed filtered as weeks idle (review
    2026-09-12). The start of a turn is activity too, and it is recorded
    before the turn runs."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 30 * 86400))
    registry.put(AgentRecord(name="rev", thread_id="t1", last_turn=old))

    def failed(**kwargs):
        return TurnOutcome(thread_id="t1", status="failed", error="timed out")

    monkeypatch.setattr(cli, "run_turn", failed)
    cli.cmd_send(cli.build_parser().parse_args(["send", "rev", "--quiet", "again"]))
    capsys.readouterr()

    cli.cmd_status(cli.build_parser().parse_args(["status"]))
    printed = capsys.readouterr().out
    assert "rev:" in printed
    # The completed-turn stamp is untouched: the turn produced no answer.
    assert Registry().get("rev").last_turn == old


def test_a_turn_closes_its_own_log_entry_whatever_the_outcome(tmp_path, monkeypatch):
    """`status` tells a running turn from a finished one by the closing line
    `run_turn` writes, so that line must follow a completed turn, a failed one
    AND an operator-typo abort — the last one used to raise before the server
    ever started, leaving the opening line alone and the agent 'running'."""
    wire.run_fake_turn(tmp_path, monkeypatch, "plain")
    trace = AgentStore("w").last_turn()
    assert trace is not None and trace.ended and trace.status == "completed"
    assert (trace.commands, trace.file_changes) == (1, 2)
    assert trace.timeout_s == 30.0

    wire.run_fake_turn(tmp_path, monkeypatch, "die")
    trace = AgentStore("w").last_turn()
    assert trace is not None and trace.ended and trace.status == "failed"

    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    record = AgentRecord(name="typo", cwd=str(tmp_path / "missing"),
                         sandbox="read-only")
    with pytest.raises(SystemExit):
        turn.run_turn(record=record, prompt="x", resume=False, timeout=5.0)
    trace = AgentStore("typo").last_turn()
    assert trace is not None and trace.ended and trace.status == "aborted"


def test_status_running_lists_only_turns_in_progress(tmp_path, monkeypatch, capsys):
    """Polling three workers printed 25 lines, 22 of them finished reviewers
    still inside the recent window (2026-09-13). `--running` keeps the
    in-progress ones — an open log with no closing line — and counts the rest
    in one line, so the poll costs what it is worth."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    registry = Registry()
    registry.put(AgentRecord(name="busy", thread_id="t1", last_turn=now))
    registry.put(AgentRecord(name="done", thread_id="t2", last_turn=now))
    registry.put(AgentRecord(name="quiet", thread_id="t3"))
    for name, ended in (("busy", False), ("done", True)):
        lines = [{"_meta": events.TURN_META, "name": name, "prompt": "p",
                  "at": "2026-09-13 10:00:00", "timeout": 600.0}]
        if ended:
            lines.append({"_meta": events.END_META, "name": name,
                          "status": "completed", "at": "2026-09-13 10:01:00"})
        AgentStore(name).append_events(lines)

    cli.cmd_status(cli.build_parser().parse_args(["status", "--running"]))
    printed = capsys.readouterr().out
    assert "busy: " in printed and "RUNNING since" in printed
    assert "done: " not in printed and "quiet: " not in printed
    assert "(2 idle agent(s) not listed; `status` shows them)" in printed

    # Without the flag the same three are all there, unchanged.
    cli.cmd_status(cli.build_parser().parse_args(["status"]))
    printed = capsys.readouterr().out
    assert "busy: " in printed and "done: " in printed and "quiet: " in printed
    assert "idle agent(s) not listed" not in printed


def test_status_shows_a_turn_that_is_still_running(tmp_path, monkeypatch, capsys):
    """Three first-turn workers printed `0 turns · last -`, the line of an
    agent that never started (2026-09-13). The log's opening line without its
    closing line is a turn in progress; the same with a log silent for longer
    than that turn's timeout AND the silence the four openers are entitled to
    before the timeout is armed is a killed process, and says so."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="busy", thread_id="t1", sandbox="workspace-write"))
    store = AgentStore("busy")
    log = store.events_path
    opened = {"_meta": events.TURN_META, "name": "busy", "prompt": "p",
              "at": "2026-09-13 10:00:00", "timeout": 600.0}
    ran = {"method": protocol.N_ITEM_COMPLETED, "params": {"item": {
        "type": "commandExecution", "command": "git status", "exitCode": 0}}}
    edited = {"method": protocol.N_ITEM_COMPLETED, "params": {"item": {
        "type": "fileChange", "changes": [{"path": "a.py"}, {"path": "b.py"}]}}}
    store.append_events([opened, ran, edited])

    cli.cmd_status(cli.build_parser().parse_args(["status", "busy"]))
    printed = capsys.readouterr().out
    assert "RUNNING since 2026-09-13 10:00:00 (1 cmd, 2 edits)" in printed

    # Past its own timeout, but still inside the silence a turn is entitled to
    # before that timeout is armed (`turn.TURN_STARTUP_GRACE_S`): the openers
    # log nothing, so this may be a live turn whose server is still starting.
    #
    # Measured against the four openers themselves rather than against the
    # constant: `4 * protocol.STARTUP_TIMEOUT_S` is the whole point of the
    # grace, and reading the grace back out of `turn` pinned a turn that had
    # waited out one opener just as well as a turn that had waited out four.
    four_openers = 4 * protocol.STARTUP_TIMEOUT_S
    for silent_s in (601, 601 + four_openers - 5):
        starting = time.time() - silent_s
        os.utime(log, (starting, starting))
        cli.cmd_status(cli.build_parser().parse_args(["status", "busy"]))
        printed = capsys.readouterr().out
        assert "RUNNING since 2026-09-13 10:00:00" in printed, silent_s
        assert "killed" not in printed, silent_s

    # Silent for longer than that too: nobody is running it any more.
    stale = time.time() - (601 + four_openers + 30)
    os.utime(log, (stale, stale))
    cli.cmd_status(cli.build_parser().parse_args(["status", "busy"]))
    printed = capsys.readouterr().out
    assert "never ended" in printed and "killed" in printed
    assert "RUNNING" not in printed

    # The closing line ends it, and `logs` renders both bookkeeping lines.
    store.append_events([{"_meta": events.END_META, "status": "timeout",
                          "at": "2026-09-13 10:10:00"}])
    cli.cmd_status(cli.build_parser().parse_args(["status", "busy"]))
    printed = capsys.readouterr().out
    assert "RUNNING" not in printed and "never ended" not in printed
    cli.cmd_logs(cli.build_parser().parse_args(["logs", "busy"]))
    printed = capsys.readouterr().out
    assert "--- turn: p" in printed and "--- end: timeout" in printed

    # A log from before the closing line existed ends at the server's own
    # `turn/completed`, so an old finished turn is not reported as running. It
    # gets a log of its own: the closing line above already ends `busy`'s turn.
    old = {"method": protocol.N_TURN_COMPLETED,
           "params": {"turn": {"status": "completed"}}}
    Registry().put(AgentRecord(name="legacy", thread_id="t2",
                               sandbox="workspace-write"))
    AgentStore("legacy").append_events([opened, ran, old])
    cli.cmd_status(cli.build_parser().parse_args(["status", "legacy"]))
    printed = capsys.readouterr().out
    assert "legacy: " in printed and "RUNNING" not in printed


def test_the_last_turn_is_found_without_reading_the_earlier_ones(tmp_path,
                                                                 monkeypatch):
    """`status` asks `last_turn_trace` once per listed agent, and a log holds
    every turn its agent ever ran, so the trace is read from the last opening
    line on. That offset has to be exact with the line SEVERAL chunks from the
    end (the reader widens its window until it finds one) and with the window
    cutting a line in half, which is the usual case."""
    log = tmp_path / "events.jsonl"
    filler = json.dumps({"method": protocol.N_ITEM_COMPLETED, "params": {"item": {
        "type": "commandExecution", "command": "x" * 197, "exitCode": 0}}})
    older = json.dumps({"_meta": events.TURN_META, "at": "2026-09-01 09:00:00",
                        "timeout": 60.0, "prompt": "первый ход"},
                       ensure_ascii=False)
    latest = json.dumps({"_meta": events.TURN_META, "at": "2026-09-13 10:00:00",
                         "timeout": 600.0, "prompt": "второй ход"},
                        ensure_ascii=False)
    head = "\n".join([older, filler, filler]) + "\n"
    # Enough traffic AFTER the opening line to push it past several windows,
    # and a filler line whose length is coprime with the window so the cut
    # lands inside a line rather than on a boundary.
    after = 3 * events._TAIL_CHUNK_BYTES // len(filler) + 1
    body = "\n".join([latest] + [filler] * after) + "\n"
    log.write_bytes((head + body).encode("utf-8"))

    offset = len(head.encode("utf-8"))
    assert events._last_turn_offset(log) == offset
    first = next(iter(events.iter_log(log, start=events._last_turn_offset(log))))
    assert first.get("at") == "2026-09-13 10:00:00"

    # That the offset is USED, and not merely computed: the trace itself cannot
    # show it (a full parse of this log ends on the same turn, because each
    # opening line replaces the trace before it), so the one observable
    # difference is the argument — dropping it costs 0.178 s per agent instead
    # of 0.0007 s and nothing turns red.
    started_at: list[int] = []
    real_iter_log = events.iter_log

    def spy(path, *, start=0):
        started_at.append(start)
        return real_iter_log(path, start=start)

    monkeypatch.setattr(events, "iter_log", spy)
    trace = events.last_turn_trace(log)
    monkeypatch.undo()
    assert started_at == [offset]
    assert trace is not None
    assert trace.started == "2026-09-13 10:00:00" and trace.commands == after

    # A log that has no opening line at all is read from the top, and says so
    # by having no trace rather than by inventing one. An empty one is the same
    # answer without a single window to widen.
    plain = tmp_path / "plain.jsonl"
    plain.write_text(filler + "\n", encoding="utf-8")
    assert events._last_turn_offset(plain) == 0
    assert events.last_turn_trace(plain) is None
    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    assert events._last_turn_offset(empty) == 0
    assert events.last_turn_trace(empty) is None
    assert events.last_turn_trace(tmp_path / "absent.jsonl") is None

    # An opening line LONGER than the window: the first windows hold no line
    # end at all, and each one must widen rather than conclude there is no turn
    # (a worker's prompt is routinely tens of kilobytes, and a resend carries
    # the whole task again).
    huge = tmp_path / "huge.jsonl"
    huge.write_bytes((json.dumps(
        {"_meta": events.TURN_META, "at": "2026-09-13 11:00:00", "timeout": 60.0,
         "prompt": "ход " * (events._TAIL_CHUNK_BYTES // 4)}, ensure_ascii=False)
        + "\n" + filler + "\n").encode("utf-8"))
    assert events._last_turn_offset(huge) == 0
    huge_trace = events.last_turn_trace(huge)
    assert huge_trace is not None
    assert huge_trace.started == "2026-09-13 11:00:00" and huge_trace.commands == 1

    # A reader that starts mid-character loses that one line and no more. The
    # decode is `errors="replace"` for exactly this, and the `except` below it
    # is narrow enough that a strict one would escape instead of being taken
    # for an unparseable line.
    mid_char = offset + body.encode("utf-8").index("ход".encode("utf-8")) + 1
    from_mid = list(events.iter_log(log, start=mid_char))
    assert len(from_mid) == after
    assert all(message.get("method") == protocol.N_ITEM_COMPLETED
               for message in from_mid)


def test_status_treats_an_unreadable_latest_stamp_as_unknown_not_old(
        tmp_path, monkeypatch, capsys):
    """The latest stamp decides alone: one in a format this cannot read must
    not fall through to an old creation date and hide the agent."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    old = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - 30 * 86400))
    Registry().put(AgentRecord(name="odd", thread_id="t1", created=old,
                               last_turn="2026-09-12T23:00:00"))

    cli.cmd_status(cli.build_parser().parse_args(["status"]))
    assert "odd:" in capsys.readouterr().out


def test_two_registries_do_not_share_one_scratch_workspace(tmp_path, monkeypatch):
    """A name is unique inside ONE registry, so two checkouts both spawning
    `rev` would otherwise be handed the same writable directory — one agent's
    files inside another's workspace, in the mode whose point is isolation."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "here"))
    mine = paths.scratch_dir("rev")
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "elsewhere"))
    theirs = paths.scratch_dir("rev")

    assert mine != theirs
    assert mine.name.startswith("rev-") and theirs.name.startswith("rev-")


def test_a_fresh_scratch_agent_does_not_inherit_the_last_ones_workspace(
        tmp_path, monkeypatch):
    """The name is reusable by design, so without archiving, a new `rev` opens
    in the old `rev`'s working directory — and unlike a stale log, an agent
    reads and RUNS what it finds in its workspace."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    project = tmp_path / "project"
    project.mkdir()
    workspace = paths.scratch_dir("rev")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "leftover.py").write_text("print('stale')", encoding="utf-8")
    monkeypatch.setattr(cli, "run_turn", lambda **kwargs: TurnOutcome(
        thread_id="th-1", status="completed", text="hi"))

    # Through `spawn`, not through the helper: a test that calls
    # `_archive_scratch` itself passes just as happily when nothing calls it.
    assert cli.cmd_spawn(cli.build_parser().parse_args(
        ["spawn", "rev", "--sandbox", sandbox.SCRATCH_WRITE, "--cwd", str(project),
         "--quiet", "hello"])) == 0

    assert not (workspace / "leftover.py").exists()
    archived = [path for path in workspace.parent.iterdir()
                if path.name.startswith(f"{workspace.name}.")]
    assert len(archived) == 1
    assert (archived[0] / "leftover.py").read_text(encoding="utf-8") == "print('stale')"
    shutil.rmtree(archived[0], ignore_errors=True)
    shutil.rmtree(workspace, ignore_errors=True)


def test_status_reports_the_policy_not_the_protocol_mode(tmp_path, monkeypatch,
                                                         capsys):
    """`scratch-write` reaches the server as `workspace-write`, and printing
    what goes on the wire would tell the operator their reviewer can edit the
    project — the exact misreport `effective_sandbox` exists to prevent."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    Registry().put(AgentRecord(name="rev", thread_id="t", sandbox=sandbox.SCRATCH_WRITE))

    cli.cmd_status(cli.build_parser().parse_args(["status"]))

    printed = capsys.readouterr().out
    assert "scratch-write" in printed
    # And NOT the wire value beside it: a line carrying both would satisfy a
    # bare `in` check while still telling the operator the project is writable.
    assert "workspace-write" not in printed.replace("scratch-write", "")


def test_state_directory_follows_its_environment_override(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    assert paths.state_dir() == tmp_path
    monkeypatch.delenv(paths.STATE_DIR_ENV)
    assert paths.state_dir().name == paths.STATE_DIR_NAME


def test_the_host_root_is_named_by_the_host_not_counted_from_this_file(
        tmp_path, monkeypatch):
    """The registry lives under the root the host's entry point installs, and
    nowhere this package's own depth would put it: moving the package must
    not move a `.codex-agents`. Without a host the caller's checkout stands
    in (`invocation_root`), which `announce_empty` says."""
    monkeypatch.delenv(paths.STATE_DIR_ENV, raising=False)
    host = tmp_path / "host"
    host.mkdir()
    with host_policy.installed(host_policy.EMPTY, host):
        assert paths.host_root() == host.resolve()
        assert paths.state_dir() == host.resolve() / paths.STATE_DIR_NAME
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    monkeypatch.chdir(plain)
    with host_policy.installed(host_policy.EMPTY):
        assert host_policy.root() is None
        assert paths.host_root() == plain.resolve()
        assert paths.state_dir() == plain.resolve() / paths.STATE_DIR_NAME


def test_spawn_defaults_to_the_callers_linked_worktree(tmp_path, monkeypatch):
    """An absolute command path must not pull spawn back to its checkout."""
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"],
                   check=True)
    subprocess.run([
        "git", "-C", str(repository), "-c", "user.name=Test",
        "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "base",
    ], check=True)
    worktree = tmp_path / "другой-worktree"
    subprocess.run([
        "git", "-C", str(repository), "worktree", "add", "-q", "-b", "other",
        str(worktree),
    ], check=True)
    nested = worktree / "app" / "src"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path / "agent-state"))
    turns = []

    def capture(**kwargs):
        turns.append(kwargs)
        return TurnOutcome(thread_id="th-worktree", status="completed", text="ok")

    monkeypatch.setattr(cli, "run_turn", capture)
    args = cli.build_parser().parse_args([
        "spawn", "worktree-review", "--sandbox", "read-only", "--quiet", "review",
    ])

    assert cli.cmd_spawn(args) == 0
    assert Path(turns[0]["record"].cwd) == worktree.resolve()


def test_invocation_root_is_cwd_outside_git(tmp_path, monkeypatch):
    """The worker also supports plain directories, so Git is not mandatory."""
    plain = tmp_path / "plain"
    plain.mkdir()
    # --basetemp may live inside a checkout; keep this fixture outside Git.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.resolve()))
    monkeypatch.chdir(plain)

    assert paths.invocation_root() == plain.resolve()


@pytest.mark.parametrize("how", ["directory", "module"])
def test_the_package_runs_as_the_command(tmp_path, how):
    """`python <src>/llm_agents/codex` reaches `__main__.py` with `src/` NOT
    on the path, so the package's imports resolve only because that file
    puts it there; `python -m` needs none of it. A subprocess is the only way
    to test either: in-process the path is already fixed. Without a host the
    one line on stderr is the empty policy's notice; nothing else may be there,
    since a model reads this output (the PEP 366 spelling of the path fix
    prints a DeprecationWarning per run)."""
    command = ([str(SRC / "llm_agents" / "codex")] if how == "directory"
               else ["-m", "llm_agents.codex"])
    env = {**os.environ, paths.STATE_DIR_ENV: str(tmp_path)}
    if how == "module":
        env["PYTHONPATH"] = str(SRC)
    done = subprocess.run([sys.executable, *command, "status"], cwd=tmp_path,
                          capture_output=True, text=True, timeout=60, env=env)
    assert done.returncode == 0, done.stderr
    assert "no codex agents" in done.stdout
    assert done.stderr.startswith("codex_agent: no host policy given: ")
    assert done.stderr.count("\n") == 1, done.stderr


def test_the_directory_run_prefers_its_own_src(tmp_path):
    """Another llm-loop reachable through PYTHONPATH (an older revision) must
    not shadow the one beside the package being run: an older one fails on
    new names, a different one silently swaps the stream rule."""
    fake = tmp_path / "elsewhere" / "llm_loop"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("", encoding="utf-8")
    (fake / "wire.py").write_text("raise ImportError('the other copy')\n",
                                  encoding="utf-8")
    done = subprocess.run(
        [sys.executable, str(SRC / "llm_agents" / "codex"), "--help"],
        capture_output=True, text=True, timeout=60, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(fake.parent)})
    assert done.returncode == 0, done.stderr


# --- context window -------------------------------------------------------


def _usage(total: dict, last: dict, window: int) -> dict:
    return _event("thread/tokenUsage/updated", {"tokenUsage": {
        "total": total, "last": last, "modelContextWindow": window}})


def test_thread_occupancy_is_read_from_last_not_from_the_running_total():
    """The two numbers differ by an order of magnitude on a real turn (total
    1.10M against last 88.8k in a 258.4k window): `total` counts the thread once
    per request and answers "what did this cost", while `last` is what the
    thread OCCUPIES now. Reading occupancy off `total` would report a window
    overrun on a turn that was a third of the way in."""
    runner = TurnRunner()
    runner.on_event(_usage({"inputTokens": 1_082_097, "outputTokens": 20_340},
                           {"totalTokens": 88_764}, 258_400))
    assert runner.outcome.input_tokens == 1_082_097
    assert runner.outcome.context_tokens == 88_764
    assert runner.outcome.context_window == 258_400


def test_a_crowded_thread_is_told_to_start_a_new_agent():
    """Codex's window is about a quarter of this script's caller's, and a resume
    re-sends the whole thread — so the caller must hear about the ceiling before
    it pays for a `send` that starts from there."""
    crowded = TurnOutcome(context_tokens=200_000, context_window=258_400)
    assert "new `spawn`" in cli._context_pressure(crowded)
    roomy = TurnOutcome(context_tokens=20_000, context_window=258_400)
    assert cli._context_pressure(roomy) == ""
    # A server that reports no window must not produce a share of nothing.
    assert cli._context_pressure(TurnOutcome(context_tokens=200_000)) == ""
    assert "258.4k" in cli._context_share(crowded)


def test_a_malformed_registry_entry_is_an_error_not_a_silent_drop(tmp_path,
                                                                  monkeypatch):
    """An entry that is not an object used to be filtered out on read, so the
    next write deleted it — the same silent loss the file-level check exists to
    prevent, one level down."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.path.write_text('{"good": {"threadId": "t1"}, "broken": "oops"}',
                             encoding="utf-8")
    with pytest.raises(RegistryError):
        registry.load()


def test_reusing_a_name_reports_the_thread_it_displaced(tmp_path, monkeypatch):
    """Reuse is the default, so the id of the thread being left behind is the
    only way the caller can still find it (`codex agents` lists it); losing it
    silently would make reuse feel like deletion."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    registry = Registry()
    registry.reserve(AgentRecord(name="rev", thread_id="old-thread"), replace=True)
    displaced = registry.reserve(AgentRecord(name="rev"), replace=True)
    assert displaced == "old-thread"
    assert registry.reserve(AgentRecord(name="fresh"), replace=True) == ""


def test_a_reused_name_archives_the_previous_log_and_answer(tmp_path, monkeypatch):
    """Appending would make `logs`/`result` mix two unrelated threads under one
    name; deleting would throw away the findings that usually prompted the
    rerun."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    store = AgentStore("rev")
    store.append_events([{"_meta": events.TURN_META, "prompt": "old turn"}])
    store.write_answer("old answer")

    store.archive()

    assert not store.has_events()
    assert store.read_answer() is None
    archived_log, archived_answer = store.archived()
    assert [m.get("prompt") for m in events.iter_log(archived_log)] == ["old turn"]
    assert archived_answer.read_text(encoding="utf-8") == "old answer"
    # Nothing else in the directory: the archive moved, it did not copy.
    assert sorted(store.dir.iterdir()) == sorted(store.archived())


def test_two_archives_in_one_second_keep_both_runs(tmp_path, monkeypatch):
    """`Path.replace` overwrites a file target, so a second archive under the
    same second-resolution stamp (a `spawn` failing at once, then a scripted
    retry) used to replace the earlier run's log with the failed one's."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(time, "strftime", lambda fmt, *args: "20261002-120000")
    store = AgentStore("rev")
    for run in ("first", "second", "third"):
        store.append_events([{"_meta": events.TURN_META, "prompt": run}])
        store.write_answer(f"{run} answer")
        store.archive()

    archived = store.archived()
    logs, answers = archived[:3], archived[3:]
    # Oldest first, though `<stamp>-2` sorts before `<stamp>` as a plain string.
    assert [[m.get("prompt") for m in events.iter_log(p)] for p in logs] == [
        ["first"], ["second"], ["third"]]
    assert [p.read_text(encoding="utf-8") for p in answers] == [
        "first answer", "second answer", "third answer"]
    # A log and its answer share the suffix, so they still pair by name.
    assert [p.name.split(".")[1] for p in logs] == [
        p.name.split(".")[1] for p in answers]


def test_two_scratch_archives_in_one_second_do_not_fail(tmp_path, monkeypatch):
    """`replace` onto an existing workspace archive raises, so a second
    scratch `spawn` of one name in one second used to fail."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(time, "strftime", lambda fmt, *args: "20261002-120000")
    workspace = paths.scratch_dir("rev")
    try:
        for run in ("first", "second"):
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "left.txt").write_text(run, encoding="utf-8")
            cli._archive_scratch("rev")
        archived = sorted(path for path in workspace.parent.iterdir()
                          if path.name.startswith(f"{workspace.name}."))
        assert sorted((p / "left.txt").read_text(encoding="utf-8")
                      for p in archived) == ["first", "second"]
    finally:
        for path in workspace.parent.glob(f"{workspace.name}*"):
            shutil.rmtree(path, ignore_errors=True)


def test_a_torn_last_line_does_not_swallow_the_next_turn(tmp_path, monkeypatch):
    """A writer killed mid-line leaves no final newline; the next opening line
    glued onto that fragment would be one unparseable line, and `status` would
    keep reporting the turn before it."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    store = AgentStore("w")
    store.append_events([{"_meta": events.TURN_META, "prompt": "old",
                          "at": "2026-09-13 10:00:00"}])
    with store.events_path.open("a", encoding="utf-8") as handle:
        handle.write('{"method": "item/comp')
    store.append_events([{"_meta": events.TURN_META, "prompt": "new",
                          "at": "2026-09-13 11:00:00"}])
    assert store.last_turn().started == "2026-09-13 11:00:00"


def test_a_registry_failure_is_not_reported_as_a_failed_turn():
    """`run_turn` catches RuntimeError to turn protocol trouble into an outcome.
    A RegistryError caught there would be announced as a failed turn — and the
    thread id, the one thing that must survive, would be dropped with it."""
    assert not issubclass(RegistryError, RuntimeError)


# --- the on-screen counter ------------------------------------------------

# The other half of these lives in `extras/agent-counter/`, and is pinned by
# `AgentCounter.exe --selftest`: the count, an abrupt client death, the start
# race. Only a built exe can serve a named pipe (the stdlib can connect to one
# but cannot create one), so the round trip below is the single place both
# halves are exercised together — and it is skipped rather than faked when the
# exe is absent, because a green test that proved nothing is worse than a skip.
# The counter is a Windows named pipe: off Windows `agent_counter.announce`
# yields an empty Beacon before it opens anything, so there is nothing to pin.
windows_counter = pytest.mark.skipif(os.name != "nt",
                                     reason="announce is a no-op off Windows")


@windows_counter
def test_a_turn_tells_the_counter_that_it_started(tmp_path, monkeypatch):
    """The one thing no other test here pins: the PLACE of the announcement.

    Every other beacon test calls `announce` itself, and the autouse fixture
    switches it off for the turn tests — so deleting the context manager from
    `run_turn` left this whole suite green while no production turn registered
    ever again."""
    monkeypatch.delenv(beacon.DISABLE_ENV, raising=False)
    sent = []
    closed = []

    class _Handle:
        def write(self, data):
            sent.append(data)

        def close(self):
            closed.append(True)

    monkeypatch.setattr(beacon, "_open_or_start", _Handle)
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "plain", name="w")

    assert outcome.status == "completed"
    assert json.loads(sent[0])["label"].startswith("w@")
    assert json.loads(sent[0])["agent_type"] == "codex"
    # Closing IS the decrement, so a turn that ends must let the handle go.
    assert closed


class _RecordingHandle:
    def __init__(self, sent):
        self.sent = sent

    def write(self, data):
        self.sent.append(json.loads(data))

    def close(self):
        pass


@windows_counter
def test_a_turn_tells_the_counter_its_weekly_remainder(tmp_path, monkeypatch, capsys):
    """The figure comes off the turn's own `account/rateLimits/updated`, as
    the remainder, once per change; a report with no weekly window keeps
    what is shown. Repeated reports must not become repeated writes."""
    monkeypatch.delenv(beacon.DISABLE_ENV, raising=False)
    monkeypatch.setattr(beacon, "_week_complained", set())
    sent = []
    monkeypatch.setattr(beacon, "_open_or_start", lambda: _RecordingHandle(sent))
    outcome = wire.run_fake_turn(tmp_path, monkeypatch, "rate-limits", name="w")

    assert outcome.status == "completed"
    assert [message.get("week_left") for message in sent] == [None, 17, 10]
    assert all(message["agent_type"] == "codex" for message in sent)
    assert "weekly-limit" not in capsys.readouterr().err


@windows_counter
def test_a_turn_without_a_weekly_figure_tells_the_counter_log_once(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(beacon.DISABLE_ENV, raising=False)
    monkeypatch.setattr(beacon, "_week_complained", set())
    sent = []
    monkeypatch.setattr(beacon, "_open_or_start", lambda: _RecordingHandle(sent))
    wire.run_fake_turn(tmp_path, monkeypatch, "plain", name="w")

    assert all("week_left" not in message for message in sent)
    assert "agent_counter: no weekly-limit figure for codex" in capsys.readouterr().err


def test_a_resent_attempt_keeps_feeding_the_counter_weekly_figure():
    """`TurnRunner(earlier)` is how a service drop and a follow-up start their
    attempt, so the sink must travel with it or the badge goes stale."""
    reported = []
    first = turn.TurnRunner(week_used=lambda used, why: reported.append((used, why)))
    turn.TurnRunner(first).on_event({"method": "account/rateLimits/updated", "params": {
        "rateLimits": {"primary": {"usedPercent": 30, "windowDurationMins": 10080}}}})
    assert reported == [(30.0, "")]


def test_an_unreadable_rate_limit_report_does_not_break_the_turn():
    """The weekly figure is optional telemetry: a report the parser cannot
    read (`resetsAt` past datetime's range) is "no figure" with the reason,
    not an exception out of the turn's event pump."""
    reported = []
    runner = turn.TurnRunner(week_used=lambda used, why: reported.append((used, why)))
    runner.on_event({"method": "account/rateLimits/updated", "params": {
        "rateLimits": {"primary": {"usedPercent": 30, "windowDurationMins": 10080,
                                   "resetsAt": 1e20}}}})
    assert len(reported) == 1
    assert reported[0][0] is None
    assert "could not be read" in reported[0][1]


def test_the_kill_switch_keeps_a_gate_run_off_the_screen(monkeypatch):
    """With the switch off nothing may touch the wire — otherwise every test
    above this one puts a window on the operator's screen."""
    monkeypatch.setenv(beacon.DISABLE_ENV, "0")

    def explode():
        raise AssertionError("the beacon opened the pipe with the switch off")

    monkeypatch.setattr(beacon, "_open_or_start", explode)
    with beacon.announce("x"):
        pass


def test_codex_provider_counter_switch_survives_shared_client_extraction(monkeypatch):
    from llm_agents.codex import beacon as codex_beacon
    monkeypatch.setenv(beacon.DISABLE_ENV, "1")
    monkeypatch.setenv(codex_beacon.DISABLE_ENV, "0")

    def explode():
        raise AssertionError("Codex-specific switch did not suppress the shared counter")

    monkeypatch.setattr(beacon, "_open_or_start", explode)
    with codex_beacon.announce("codex-test"):
        pass


def test_an_absent_counter_costs_the_turn_nothing(monkeypatch, tmp_path):
    """The badge is a decoration, and a decoration that can fail a turn is worse
    than no decoration. An unbuilt exe must also not be started."""
    monkeypatch.delenv(beacon.DISABLE_ENV, raising=False)
    monkeypatch.setattr(beacon, "_complained", False)
    monkeypatch.setattr(beacon, "_connect", lambda: None)
    monkeypatch.setenv(beacon.EXE_ENV, str(tmp_path / "not-built.exe"))
    spawned = []
    monkeypatch.setattr(beacon, "_spawn", spawned.append)
    with beacon.announce("x"):
        pass
    assert spawned == []


def test_a_wire_that_breaks_mid_announcement_is_swallowed(monkeypatch):
    """Every OSError on this path is one the turn must survive: the counter can
    be killed between the connect and the hello."""
    monkeypatch.delenv(beacon.DISABLE_ENV, raising=False)
    monkeypatch.setattr(beacon, "_complained", False)

    def die():
        raise OSError("the pipe went away")

    monkeypatch.setattr(beacon, "_open_or_start", die)
    with beacon.announce("x"):
        pass


    # Other Codex/Claude workers can start or finish during this probe, so their
    # contributions must not turn our successful close into a failing count.
