"""The `stop` sentinel and who may consume it.

A stop file is a one-shot signal: the run that honours it also removes it, so
the next launch starts clean. That makes *who* removes it a correctness
question. A `--dry-run` previews the commands a run would build, and it is
routinely used while a real loop is running — which is exactly when a stop is
pending. A dry run that consumed the sentinel would silently cancel someone
else's stop request, and the loop it was meant to halt would keep going.

So: a real run consumes it, a dry run reports it and leaves it in place. Both
runners are covered, since either entry point could regress independently.
"""

import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_loop import cyclecore, parallel
from claude_loop.cyclecore import ClaudeCommand, Driver
from claude_loop.drivers import ListFileDriver


@pytest.fixture(autouse=True)
def _restore_streams():
    """Both runners tee sys.stdout/stderr into their log and never put them back;
    undo that so one test's tee does not follow the next one."""
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


class _OneShotDriver(Driver):
    """Hands out a single command, then reports the work exhausted."""

    def __init__(self):
        self.served = 0
        self.limit_policy = _StubPolicy()

    def next_command(self):
        if self.served:
            return None
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")


class _StubPolicy:
    """A LimitPolicy that never queries /usage and never pauses."""

    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="", cache_value=True):
        return False, session_start


class _MemListDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real claude)."""

    target_suffix = ".ru.md"

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self.limit_policy = _StubPolicy()

    def prompt(self, source, target):
        return "do the thing"

    def pending_lines(self):
        return list(self._items)


def _seq_args(project_dir, *, dry_run):
    ns = type("NS", (), {})()
    ns.max = None
    ns.dry_run = dry_run
    ns.raw = False
    ns.start_in = None
    ns.max_strike = None
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.cost = False
    return ns


def _par_args(project_dir, *, dry_run):
    ns = type("NS", (), {})()
    ns.jobs = 2
    ns.max = None
    ns.dry_run = dry_run
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.ignore_usage = True
    return ns


def _stop_file(project_dir: Path) -> Path:
    path = project_dir / "stop"
    path.write_text("", encoding="utf-8")
    return path


# -- sequential runner ---------------------------------------------------------

def test_dry_run_leaves_the_stop_file(tmp_path, capsys):
    """A previewing run reports the sentinel and leaves it for the real run."""
    stop = _stop_file(tmp_path)
    driver = _OneShotDriver()
    cyclecore.run_loop(driver, _seq_args(str(tmp_path), dry_run=True),
                       app_name="pytest-stop")
    assert stop.exists(), "dry run consumed the stop file"
    out = capsys.readouterr().out
    assert "Stop file present" in out
    # It reports the sentinel *and* still previews — a pending stop must not
    # make -d useless, which is the whole reason it does not simply break here.
    assert "DRY-RUN:" in out
    assert driver.served == 1


def test_real_run_consumes_the_stop_file(tmp_path, capsys):
    """The run that honours the sentinel is the one that removes it."""
    stop = _stop_file(tmp_path)
    driver = _OneShotDriver()
    cyclecore.run_loop(driver, _seq_args(str(tmp_path), dry_run=False),
                       app_name="pytest-stop")
    assert not stop.exists(), "a real run left the stop file behind"
    assert "Stop file detected" in capsys.readouterr().out
    # Stopped before doing any work: the driver was never asked for a command.
    assert driver.served == 0


def test_dry_run_reports_the_stop_file_once(tmp_path, capsys):
    """--max-runs keeps the dry run looping; the note must not repeat per pass."""
    _stop_file(tmp_path)
    args = _seq_args(str(tmp_path), dry_run=True)
    args.max = 3
    cyclecore.run_loop(_OneShotDriver(), args, app_name="pytest-stop")
    assert capsys.readouterr().out.count("Stop file present") == 1


# -- parallel runner -----------------------------------------------------------

def test_parallel_dry_run_leaves_the_stop_file(tmp_path, capsys):
    """The workers are what remove the sentinel, and a dry run starts none."""
    stop = _stop_file(tmp_path)
    parallel.run_parallel(_MemListDriver(["products/a.md"]),
                          _par_args(str(tmp_path), dry_run=True),
                          app_name="pytest-stop-parallel")
    assert stop.exists(), "parallel dry run consumed the stop file"
    out = capsys.readouterr().out
    assert "stop file present" in out.lower()
    assert "DRY-RUN" in out


def test_parallel_worker_consumes_the_stop_file(tmp_path, monkeypatch):
    """A real parallel run still stops on the sentinel and removes it."""
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd: (0, 0.0, 0.01))
    stop = _stop_file(tmp_path)
    done = threading.Event()

    def go():
        try:
            parallel.run_parallel(_MemListDriver(["products/a.md"]),
                                  _par_args(str(tmp_path), dry_run=False),
                                  app_name="pytest-stop-parallel")
        except SystemExit:
            pass
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    assert done.wait(10), "run_parallel did not stop on the stop file"
    assert not stop.exists(), "a real parallel run left the stop file behind"


def test_stop_file_path_follows_the_project_root(tmp_path):
    """The sentinel is resolved against the chosen root, not the cwd — otherwise
    a -C run would watch the wrong file (and the tests would pass by accident)."""
    previous = cyclecore.project_dir()
    try:
        cyclecore.set_project_root(str(tmp_path))
        assert cyclecore.STOP_FILE == os.path.join(str(tmp_path), "stop")
    finally:
        cyclecore.set_project_root(previous)
