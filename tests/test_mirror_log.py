"""The mirror log must never be able to kill — or lose — the run it is recording.

Two loops sharing one log is normal (a sequential run, a parallel run, the
grow-kit pass), and on Windows that makes rotation fail while the other process
holds the file. These pin the two halves that turned such a failure fatal.

The second group pins who may write there at all. A dry run is a preview, not a
run: previewing every pending item once pushed ~26 MB through the shared log and
rotated a live run's records — including the failure they were opened to
explain — off the end of the backup chain.
"""

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_loop import cyclecore, parallel
from llm_loop.cyclecore import ClaudeCommand, Driver
from llm_loop.drivers import ListFileDriver


@pytest.fixture(autouse=True)
def _restore_streams():
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


def _handler(tmp_path, **kwargs):
    handler = cyclecore._MirrorLogHandler(
        tmp_path / "mirror.log", maxBytes=64, backupCount=2,
        encoding="utf-8", **kwargs)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _logger(name, handler):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = [handler]
    return logger


def test_a_rotation_another_process_blocks_keeps_the_log_running(tmp_path, monkeypatch):
    """A failed rename must cost lines in the file, not the run."""
    handler = _handler(tmp_path)
    logger = _logger("runCycle.test-rotate", handler)

    def refuse(*args, **kwargs):
        raise PermissionError(32, "used by another process")

    monkeypatch.setattr(os, "rename", refuse)
    monkeypatch.setattr(os, "replace", refuse)
    for i in range(40):  # well past maxBytes, so rollover is attempted
        logger.info(f"line {i} " + "x" * 20)
    handler.close()

    written = (tmp_path / "mirror.log").read_text(encoding="utf-8")
    assert "line 39" in written  # kept appending instead of giving up


def test_a_failed_rotation_is_retried_on_a_timer_not_per_line(tmp_path, monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(1)
        raise PermissionError(32, "used by another process")

    handler = _handler(tmp_path)
    logger = _logger("runCycle.test-retry", handler)
    monkeypatch.setattr(os, "rename", refuse)
    monkeypatch.setattr(os, "replace", refuse)
    for i in range(30):
        logger.info(f"line {i} " + "x" * 20)
    handler.close()

    # One burst of renames (the backup chain), then the backoff holds them off;
    # without it every record past the cap would try again.
    assert len(attempts) <= handler.backupCount + 1


def test_a_logging_failure_never_recurses_through_the_tee(tmp_path):
    """The exact shape that killed a run: the handler reports to stderr, stderr
    is the tee, the tee logs it, the handler fails again."""
    depth = {"max": 0, "now": 0}

    class Exploding(logging.Handler):
        def emit(self, record):
            depth["now"] += 1
            depth["max"] = max(depth["max"], depth["now"])
            try:
                # What logging.Handler.handleError does by default.
                print("--- Logging error ---", file=sys.stderr)
            finally:
                depth["now"] -= 1

    handler = Exploding()
    logger = _logger("runCycle.test-recursion", handler)
    sys.stderr = cyclecore._TeeToLog(sys.stderr, logger)

    print("something the loop printed", file=sys.stderr)

    assert depth["max"] == 1  # the report did not re-enter the logger


def test_the_tee_still_logs_normally_after_a_guarded_call(tmp_path):
    handler = _handler(tmp_path)
    logger = _logger("runCycle.test-normal", handler)
    sys.stdout = cyclecore._TeeToLog(sys.stdout, logger)

    print("first")
    print("second")
    handler.close()

    written = (tmp_path / "mirror.log").read_text(encoding="utf-8")
    assert "first" in written and "second" in written


# --- a dry run is a preview, and stays out of the shared record ----------------


class _StubPolicy:
    """A LimitPolicy that never reads the usage report and never pauses."""

    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="", cache_value=True):
        return False, session_start


class _MemListDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real provider)."""

    target_suffix = ".out.md"
    pick_order = "list"          # deterministic: the tests name the first item

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self.limit_policy = _StubPolicy()

    def prompt(self, source, target):
        return f"do {os.path.basename(source)}"

    def pending_lines(self):
        return list(self._items)

    def strike(self, line):
        if line in self._items:
            self._items.remove(line)
            return True
        return False


class _NoWorkDriver(Driver):
    def __init__(self):
        self.limit_policy = _StubPolicy()

    def next_command(self):
        return None


class _OneShotDriver(_NoWorkDriver):
    def __init__(self):
        super().__init__()
        self.served = 0

    def next_command(self):
        if self.served:
            return None
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")


def _seq_args(project_dir, *, dry_run, max_runs=None):
    ns = type("NS", (), {})()
    ns.max = max_runs
    ns.dry_run = dry_run
    ns.raw = False
    ns.start_in = None
    ns.max_strike = None
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.cost = False
    ns.no_statusline = True
    return ns


def _par_args(project_dir, *, dry_run, max_runs=None):
    ns = type("NS", (), {})()
    ns.jobs = 2
    ns.max = max_runs
    ns.dry_run = dry_run
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.ignore_usage = True
    ns.no_statusline = True
    return ns


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Point the mirror log at the test's own directory, never the user's."""
    logs = tmp_path / "logs"
    monkeypatch.setattr(cyclecore, "LOG_DIR", logs)
    yield logs


@pytest.fixture(autouse=True)
def _restore_project_root():
    previous = cyclecore.project_dir()
    yield
    cyclecore.set_project_root(previous)


def _drop_logger(app_name):
    """Release the handler's file so the tmp dir can be cleaned up."""
    logger = logging.getLogger(f"runCycle.{app_name}")
    for handler in list(logger.handlers):
        handler.close()
    logger.handlers = []


def test_a_parallel_dry_run_writes_nothing_to_the_shared_log(
        tmp_path, log_dir, capsys):
    app_name = "pytest-dry-parallel"
    parallel.run_parallel(_MemListDriver(["products/a.md"]),
                          _par_args(str(tmp_path), dry_run=True),
                          app_name=app_name)

    assert not cyclecore.log_file_path(app_name).exists()
    out = capsys.readouterr().out
    # Visible, not mysterious: the preview names the log it is staying out of.
    assert "dry run: nothing is mirrored" in out
    assert app_name in out


def test_a_sequential_dry_run_writes_nothing_to_the_shared_log(
        tmp_path, log_dir, capsys):
    app_name = "pytest-dry-sequential"
    cyclecore.run_loop(_OneShotDriver(), _seq_args(str(tmp_path), dry_run=True),
                       app_name=app_name)

    assert not cyclecore.log_file_path(app_name).exists()
    out = capsys.readouterr().out
    assert "dry run: nothing is mirrored" in out
    # The header --cost parses is printed by a dry run too; keeping it out of
    # the file is what stops previews from inventing zero-cost sessions.
    assert "=== Iteration 1 ===" in out


def test_a_real_run_still_mirrors_to_the_shared_log(tmp_path, log_dir):
    """The other half of the fix: only the dry run lost its mirror."""
    app_name = "pytest-real-mirror"
    try:
        cyclecore.run_loop(_NoWorkDriver(),
                           _seq_args(str(tmp_path), dry_run=False),
                           app_name=app_name, wait_on_start=False)
        written = cyclecore.log_file_path(app_name).read_text(
            encoding="utf-8", errors="replace")
    finally:
        _drop_logger(app_name)
    assert "logging to" in written


def test_an_uncapped_parallel_dry_run_caps_its_listing(tmp_path, log_dir, capsys):
    """1961 pending items is not a preview. The cap is reported, never silent."""
    pending = [f"products/p{i}.md" for i in range(25)]
    parallel.run_parallel(_MemListDriver(pending),
                          _par_args(str(tmp_path), dry_run=True),
                          app_name="pytest-dry-cap")

    out = capsys.readouterr().out
    listed = out.count("--output-format")   # one per previewed argv line
    assert listed == parallel.DRY_RUN_LIST_LIMIT
    assert "25 of 25 pending file(s)" in out           # the queue size is honest
    assert f"… and {25 - listed} more pending" in out
    assert "--max-runs" in out                          # how to see fewer/more


def test_an_explicit_max_runs_lists_exactly_that_many(tmp_path, log_dir, capsys):
    pending = [f"products/p{i}.md" for i in range(25)]
    parallel.run_parallel(_MemListDriver(pending),
                          _par_args(str(tmp_path), dry_run=True, max_runs=3),
                          app_name="pytest-dry-cap")

    out = capsys.readouterr().out
    assert out.count("--output-format") == 3
    assert "more pending" not in out
