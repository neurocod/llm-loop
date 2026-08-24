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

from llm_loop import console, cyclecore, exitlog, parallel, projectroot
from llm_loop.agentwork import ClaudeCommand, Driver
from llm_loop.drivers import ListFileDriver


class _StubPolicy:
    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="",
                       cache_value=True, should_stop=None):
        return False, session_start


class _NoWorkDriver(Driver):
    def __init__(self):
        self.limit_policy = _StubPolicy()

    def next_command(self):
        return None


class _OneItemListDriver(ListFileDriver):
    """A one-item in-memory queue: enough for the parallel runner to open a run.

    One item and not none, because a parallel run with an empty list reports
    "nothing to do" and returns before it has done anything a pin can look at.
    """

    target_suffix = ".out.md"

    def __init__(self):
        super().__init__()
        self._items = ["products/only.md"]
        self.limit_policy = _StubPolicy()

    def prompt(self, source, target):
        return "do it"

    def model(self):
        return ""

    def pending_lines(self):
        return list(self._items)

    def strike(self, line):
        if line in self._items:
            self._items.remove(line)
            return True
        return False


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
    monkeypatch.setattr(console, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(exitlog, "_record", None)
    root = projectroot.project_dir()
    streams = sys.stdout, sys.stderr
    yield
    exitlog.finish()
    sys.stdout, sys.stderr = streams
    projectroot.set_project_root(root)
    _drop_logger("pytest-exit")


def _drop_logger(app_name):
    """Close and forget whatever mirror handler this logger currently holds."""
    logger = logging.getLogger(f"runCycle.{app_name}")
    for handler in list(logger.handlers):
        handler.close()
    logger.handlers = []


def _run_once(tmp_path):
    cyclecore.run_loop(_NoWorkDriver(), _seq_args(str(tmp_path)),
                       app_name="pytest-exit", wait_on_start=False)


def test_a_clean_run_names_its_ending_in_the_log(tmp_path):
    _run_once(tmp_path)
    exitlog.finish()

    written = console.log_file_path("pytest-exit").read_text(
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
    logs = console.LOG_DIR
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
    logs = console.LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    live = exitlog.record_path(logs, "pytest-exit", "proj", 424242)
    live.write_text(json.dumps({"pid": 424242, "started": 1000.0}),
                    encoding="utf-8")
    monkeypatch.setattr(exitlog, "pid_alive", lambda pid, started=None: True)

    orphans = exitlog.report_orphans("pytest-exit", logs, "proj")

    assert orphans == []
    assert live.exists()
    assert "is live (pid 424242" in capsys.readouterr().out


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_the_report_of_a_vanished_run_lands_in_the_log(
        tmp_path, monkeypatch, runner):
    """The ORDER inside the shared prologue: the tee goes up BEFORE exitlog.begin.

    `begin` is what prints the previous run's missing-ending report, and the
    whole point of that report is to explain a mirror log that stops mid-line.
    Printed before the tee exists it goes to a terminal nobody is reading any
    more, and the log it explains never gets it — which is exactly the log the
    next reader opens. Nothing pinned that order before this: the closing line
    is written at exit, long after the tee is up either way, so swapping the two
    left the whole suite green.

    Asserted against the FILE, never against capsys: the report reaches the
    screen whichever order the two are in, so a screen-based assertion cannot
    tell them apart. Both runners, because both open a run through the one
    prologue and the pin has to fail if either stops doing so.
    """
    logs = console.LOG_DIR
    logs.mkdir(parents=True, exist_ok=True)
    project = os.path.basename(str(tmp_path))
    app_name = "pytest-exit"
    # `console.setup_file_logging` attaches a mirror handler only to a logger
    # that has none, so a handler another test left on this name would keep this
    # run's output going to THAT test's file while this one's path is merely
    # printed — which reads exactly like the defect below and is not it.
    _drop_logger(app_name)
    dead = exitlog.record_path(logs, app_name, project, 424242)
    dead.write_text(json.dumps({
        "pid": 424242, "app": app_name, "project": project,
        "argv": "runGenerateModels.py --codex --random",
        "started": 1000.0, "alive_at": 2000.0,
        "phase": "iteration 10 — electric-guitar-hollow-body.md",
    }), encoding="utf-8")
    monkeypatch.setattr(exitlog, "pid_alive", lambda pid, started=None: False)

    if runner == "sequential":
        _run_once(tmp_path)
    else:
        monkeypatch.setattr(parallel, "run_job",
                            lambda job_id, command, mailbox=None: (0, None, None))
        args = _seq_args(str(tmp_path))
        args.jobs = 1
        args.ignore_usage = True
        parallel.run_parallel(_OneItemListDriver(), args, app_name=app_name,
                              wait_on_start=False)
    exitlog.finish()

    written = console.log_file_path(app_name).read_text(
        encoding="utf-8", errors="replace")
    assert "killed from outside" in written, (
        "the report of the previous run that vanished never reached the mirror "
        "log — exitlog.begin ran before the tee was up")
    assert "electric-guitar-hollow-body.md" in written


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
