"""A run that ends must say why — and a run that is killed must be reported.

The incident these pin: two long runs ended mid-command with 11 MB of mirror log
and nothing in it about the ending. Both had been killed from outside, which is
precisely the case no handler inside the dying process can log. So the closing
line covers everything the process can see, and the leftover record covers the
one thing it cannot.
"""

import json
import logging
import os
import sys

import pytest

from llm_loop import cyclecore, exitlog
from llm_loop.cyclecore import ClaudeCommand, Driver


class _StubPolicy:
    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="", cache_value=True):
        return False, session_start


class _NoWorkDriver(Driver):
    def __init__(self):
        self.limit_policy = _StubPolicy()

    def next_command(self):
        return None


def _seq_args(project_dir):
    ns = type("NS", (), {})()
    ns.max = None
    ns.dry_run = False
    ns.raw = False
    ns.start_in = None
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.cost = False
    ns.no_statusline = True
    return ns


@pytest.fixture(autouse=True)
def _isolated_record(tmp_path, monkeypatch):
    """Every test gets its own log dir and its own fresh record."""
    monkeypatch.setattr(cyclecore, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(exitlog, "_record", None)
    root = cyclecore.project_dir()
    streams = sys.stdout, sys.stderr
    yield
    exitlog.finish()
    sys.stdout, sys.stderr = streams
    cyclecore.set_project_root(root)
    for handler in list(logging.getLogger("runCycle.pytest-exit").handlers):
        handler.close()
    logging.getLogger("runCycle.pytest-exit").handlers = []


def _run_once(tmp_path):
    cyclecore.run_loop(_NoWorkDriver(), _seq_args(str(tmp_path)),
                       app_name="pytest-exit", wait_on_start=False)


def test_a_clean_run_names_its_ending_in_the_log(tmp_path):
    _run_once(tmp_path)
    exitlog.finish()

    written = cyclecore.log_file_path("pytest-exit").read_text(
        encoding="utf-8", errors="replace")
    assert "=== run ended: no more work in the queue" in written


def test_a_finished_run_leaves_no_record_behind(tmp_path):
    """The record's only meaning is "this run never got to end", so a run that
    did end must not leave one — else every launch reports a phantom kill."""
    _run_once(tmp_path)
    record = exitlog.current().path
    assert record.exists()          # while the run is live it is the liveness

    exitlog.finish()
    assert not record.exists()


def test_a_record_whose_owner_is_gone_is_reported_and_cleared(
        tmp_path, monkeypatch, capsys):
    """The whole point: the kill that logs nothing is named by the NEXT run."""
    logs = cyclecore.LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    dead = exitlog.record_path(logs, "pytest-exit", "proj", 424242)
    dead.write_text(json.dumps({
        "pid": 424242, "app": "pytest-exit", "project": "proj",
        "argv": "runGenerateModels.py --codex --random",
        "started": 1000.0, "alive_at": 2000.0,
        "phase": "iteration 10 — electric-guitar-hollow-body.md",
    }), encoding="utf-8")
    monkeypatch.setattr(exitlog, "pid_alive", lambda pid, started=None: False)

    orphans = exitlog.report_orphans("pytest-exit", logs, "proj")

    out = capsys.readouterr().out
    assert len(orphans) == 1
    assert "killed from outside" in out
    assert "pid 424242" in out
    assert "electric-guitar-hollow-body.md" in out       # what it was working on
    assert "--codex --random" in out                     # and how it was launched
    assert not dead.exists()        # cleared, so the next run does not re-report it


def test_a_record_whose_owner_still_runs_is_left_alone(
        tmp_path, monkeypatch, capsys):
    """Two runs of one wrapper is routine here. Reporting a live sibling as a
    corpse — and deleting its record — would break the report for real kills."""
    logs = cyclecore.LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    live = exitlog.record_path(logs, "pytest-exit", "proj", 424242)
    live.write_text(json.dumps({"pid": 424242, "started": 1000.0}),
                    encoding="utf-8")
    monkeypatch.setattr(exitlog, "pid_alive", lambda pid, started=None: True)

    orphans = exitlog.report_orphans("pytest-exit", logs, "proj")

    assert orphans == []
    assert live.exists()
    assert "is live (pid 424242" in capsys.readouterr().out


def test_the_liveness_probe_does_not_kill_what_it_asks_about(monkeypatch):
    """On Windows `os.kill(pid, 0)` is TerminateProcess with exit code 0 — the
    probe would kill the run it was asked about. This pins that it is not used."""
    def forbidden(*args, **kwargs):
        raise AssertionError("os.kill must never be part of a liveness probe "
                             "on Windows")

    if os.name == "nt":
        monkeypatch.setattr(os, "kill", forbidden)
    assert exitlog.pid_alive(os.getpid()) is True
    assert exitlog.pid_alive(424242, started=1000.0) is False


def test_a_recycled_pid_does_not_mask_a_kill():
    """A pid is reused within seconds on Windows. Our own pid with someone
    else's start time must read as gone, or the report is silently swallowed."""
    if os.name != "nt":
        pytest.skip("only Windows reports a process creation time here")
    assert exitlog.pid_alive(os.getpid(), started=1000.0) is False


def test_sys_exit_is_named_although_no_excepthook_sees_it(tmp_path, capsys):
    _run_once(tmp_path)
    with pytest.raises(SystemExit):
        with exitlog.guard():
            sys.exit("error: --grow-kit models nothing")
    exitlog.finish()

    assert "=== run ended: sys.exit: error: --grow-kit models nothing" \
        in capsys.readouterr().out


def test_an_unhandled_exception_is_named():
    assert exitlog.describe_exception(KeyboardInterrupt, KeyboardInterrupt()) \
        == "interrupted from the keyboard (Ctrl+C)"
    assert exitlog.describe_exception(ValueError, ValueError("bad seed\nmore")) \
        == "unhandled ValueError: bad seed"
