"""Termination guarantees for the parallel runner's worker loop.

Regression coverage for the fix where a worker could keep running (or block in
the session-limit gate) after the list had drained to empty. The worker must:

  * claim work *before* touching the usage gate, so an empty/drained queue stops
    it via claim() -> stop and it never enters the gate with nothing to do; and
  * exit promptly once the queue is drained, even with far more workers than
    files (the spare workers must not spin forever).
"""

import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_loop import parallel
from claude_loop.drivers import ListFileDriver


class _MemDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real claude)."""

    target_suffix = ".ru.md"

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self._lock = threading.Lock()

    def prompt(self, source, target):
        return "translate"

    def model(self):
        return ""

    def pending_lines(self):
        with self._lock:
            return list(self._items)

    def strike(self, line):
        with self._lock:
            if line in self._items:
                self._items.remove(line)
                return True
            return False


def _args(project_dir, jobs, *, ignore_usage=True, max_runs=None):
    ns = type("NS", (), {})()
    ns.jobs = jobs
    ns.max = max_runs
    ns.dry_run = False
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.ignore_usage = ignore_usage
    return ns


def _run_and_wait(driver, args, timeout=10.0):
    """Run run_parallel on a thread; return True iff it returned within timeout."""
    done = threading.Event()

    def go():
        try:
            parallel.run_parallel(driver, args, app_name="pytest-parallel")
        except SystemExit:
            pass
        finally:
            done.set()

    threading.Thread(target=go, daemon=True).start()
    return done.wait(timeout)


def test_drain_terminates_with_more_workers_than_files(tmp_path, monkeypatch):
    """20 workers, a few files: everyone must exit once the list drains."""
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd: (0, 0.0, 0.01))
    driver = _MemDriver([f"products/f{i}.md" for i in range(3)])
    assert _run_and_wait(driver, _args(str(tmp_path), jobs=20)), \
        "run_parallel did not terminate after the list drained"
    assert driver.pending_lines() == []


def test_worker_claims_before_touching_the_usage_gate(tmp_path, monkeypatch):
    """On an empty/drained queue no worker may enter the session-limit gate.

    The gate can block (over budget) with no regard for the stop flag, so a
    worker that reached it with nothing to do would wedge and hang the run. With
    an empty list every worker must stop at claim(), so the policy's
    check_and_wait is never called.
    """
    monkeypatch.setattr(parallel, "run_job", lambda job_id, cmd: (0, 0.0, 0.01))
    # A source object is truthy so the gate branch is taken *if reached*.
    monkeypatch.setattr(parallel, "UsageSource", lambda *a, **k: object())

    gate_calls = []

    class SpyPolicy:
        def log_snapshot(self, *a, **k):
            pass

        def check_and_wait(self, source, session_start, note="", cache_value=True):
            gate_calls.append(1)
            return False, session_start

    # Non-empty list that drains immediately: run_parallel starts workers, they
    # drain the single item, then must exit — the spare workers never gate.
    driver = _MemDriver(["products/only.md"])
    driver.limit_policy = SpyPolicy()
    assert _run_and_wait(driver, _args(str(tmp_path), jobs=20, ignore_usage=False))
    assert driver.pending_lines() == []
    # At most one gate call (the single worker that claimed the one item); the 19
    # spare workers claimed nothing and so never touched the gate.
    assert len(gate_calls) <= 1, f"idle workers entered the usage gate: {gate_calls}"


def test_release_returns_claim_to_queue():
    """release() drops the line from in_progress and undoes its --max reservation."""
    driver = _MemDriver(["a", "b"])
    shared = parallel.Shared(driver, max_items=5)
    line = shared.claim()
    assert line in ("a", "b")
    assert shared.claimed == 1 and line in shared.in_progress
    shared.release(line)
    assert shared.claimed == 0 and line not in shared.in_progress
    # The line is still pending (never struck), so a re-run picks it up.
    assert set(driver.pending_lines()) == {"a", "b"}
