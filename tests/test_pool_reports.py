"""Pool producers round-trip through the same reports as sequential promotions."""

import io
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from llm_loop import (console, costlog, cyclecore, parallel, runlifecycle,
                      statlog, streamrender)

from _runfixtures import MemListDriver, OneShotDriver, isolated_run, par_args, seq_args


class _Process:
    stdin = None

    def __init__(self, events, returncode=0):
        self.stdout = io.StringIO("".join(json.dumps(ev) + "\n" for ev in events))
        self.returncode = returncode

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode


def _result(cost, duration=1000, subtype="success"):
    event = {"type": "result", "subtype": subtype, "total_cost_usd": cost}
    if duration is not None:
        event["duration_ms"] = duration
    return event


def _wall_clock(monkeypatch):
    clock = threading.local()

    def monotonic():
        clock.calls = getattr(clock, "calls", 0) + 1
        return 100.0 if clock.calls % 2 else 112.345

    # Only the pool worker binding changes; background services keep real clocks.
    monkeypatch.setattr(parallel, "time", SimpleNamespace(
        time=parallel.time.time, sleep=parallel.time.sleep, monotonic=monotonic))


def _run_pool(tmp_path, items, app="pytest-pool-report", jobs=2):
    streams = sys.stdout, sys.stderr
    try:
        parallel.run_parallel(MemListDriver(items),
                              par_args(tmp_path, jobs=jobs, no_statusline=True),
                              app_name=app, wait_on_start=False)
    finally:
        sys.stdout, sys.stderr = streams
    return console.log_file_path(app)


@pytest.mark.parametrize("events,exit_code,count,cost,attempts", [
    ([_result(.2015), _result(.2204)], 0, 4, .4408, 1),
    ([_result(.75, subtype="error_max_turns")], 0, 0, 0, 3),
    ([_result(.5, duration=None)], 0, 0, 0, 1),
    # Match sequential policy: a successful result counts even if CLI exit fails.
    ([_result(.1)], 1, 6, .6, 3),
    # Failed turn advances the cumulative baseline but contributes no cost.
    ([_result(.2, subtype="error_max_turns"), _result(.25)], 0, 6, .3, 3),
])
def test_real_pool_mirror_reports_successful_turn_deltas_and_worker_wall_time(
        tmp_path, monkeypatch, capsys, events, exit_code, count, cost, attempts):
    _wall_clock(monkeypatch)
    monkeypatch.setattr(parallel, "start_agent_process",
                        lambda *args: _Process(events, exit_code))
    with isolated_run(monkeypatch, tmp_path):
        log = _run_pool(tmp_path, ["products/a.md", "products/b.md"])
        capsys.readouterr()
        before = log.read_bytes()
        costlog.report_costs(path=log)
        output = capsys.readouterr().out
        assert f"TOTAL: 1 sessions, {count} costs, ${cost:.4f}" in output
        with log.open(encoding="utf-8", newline="\n") as source:
            stats = statlog.read_stats(source)
        assert len(stats) == 2
        for row in stats.values():
            assert row.iterations == attempts
            assert row.total_seconds == pytest.approx(12.345 * attempts)
            assert row.partial_iterations == 0
        statlog.report_stats(path=log)
        assert "TOTAL" in capsys.readouterr().out
        assert log.read_bytes() == before


def test_real_pool_promotion_pool_log_keeps_sessions_and_timings_distinct(
        tmp_path, monkeypatch, capsys):
    _wall_clock(monkeypatch)
    monkeypatch.setattr(parallel, "start_agent_process",
                        lambda *args: _Process([_result(.2), _result(.25)]))

    def promotion(*args, **kwargs):
        streamrender._render_claude_event(_result(.1), True)
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", promotion)
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: None)
    app = "pytest-pool-promotion"
    with isolated_run(monkeypatch, tmp_path):
        _run_pool(tmp_path, ["products/a.md"], app, jobs=1)
        streams = sys.stdout, sys.stderr
        try:
            cyclecore.run_loop(OneShotDriver(),
                               seq_args(tmp_path, no_statusline=True),
                               app_name=app, wait_on_start=False)
        finally:
            sys.stdout, sys.stderr = streams
        log = _run_pool(tmp_path, ["products/a.md"], app, jobs=1)
        capsys.readouterr()
        costlog.report_costs(path=log)
        assert "TOTAL: 3 sessions, 5 costs, $0.6000" in capsys.readouterr().out
        stats = statlog.read_stats(log.read_text(encoding="utf-8").splitlines())
        assert len(stats) == 2
        assert stats.pop("the-thing").iterations == 1
        product = next(iter(stats.values()))
        assert product.iterations == 2
        assert product.total_seconds == pytest.approx(24.69)
        assert product.partial_iterations == 0


def test_interleaved_pool_workers_do_not_replace_same_pid_promotion_cursor():
    def timing(second, message):
        return f"2026-10-02 00:00:{second:02d} pid=7 kind=timing {message}\n"

    stats = statlog.read_stats([
        timing(0, statlog.iteration_started(1, "promotion")),
        timing(1, statlog.pool_iteration_started("run", 1, "a")),
        timing(2, statlog.pool_iteration_started("run", 2, "b")),
        timing(3, statlog.pool_iteration_finished("run", 2, 20)),
        timing(4, statlog.iteration_finished(1, 0, 5)),
        timing(5, statlog.pool_iteration_finished("run", 1, 30)),
    ])
    assert {state: row.total_seconds for state, row in stats.items()} == {
        "promotion": 5, "a": 30, "b": 20}
    assert all(row.iterations == 1 and row.partial_iterations == 0
               for row in stats.values())
