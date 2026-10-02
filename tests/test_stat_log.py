"""Read state timings from legacy logs and the runner's own printed records."""

import sys
from types import SimpleNamespace

import pytest

from llm_loop import (console, costlog, cyclecore, projectroot, runlifecycle,
                      statlog, streamrender)

from _runfixtures import OneShotDriver, isolated_run, seq_args


def _line(clock, message):
    return f"2026-10-02 {clock} {message}\n"


def _tagged_line(clock, pid, kind, message):
    return _line(clock, f"pid={pid} kind={kind} {message}")


def _header(iteration, state, model="claude/opus"):
    label = f"{state} · {model}" if model else state
    return f"{costlog.iteration_header(iteration)} [{label}]"


def _assert_timing(stats, state, count, seconds, partial=0):
    row = stats[state]
    assert row.iterations == count
    assert row.total_seconds == pytest.approx(seconds)
    assert row.partial_iterations == partial


def _write_log(path, records):
    path.write_text("".join(_line(clock, message)
                            for clock, message in records), encoding="utf-8")
    return path


def _table_rows(output):
    return {cells[0]: cells[1:] for line in output.splitlines()
            if " | " in line
            for cells in [[cell.strip() for cell in line.split(" | ")]]}


def test_repeated_states_sum_all_iterations_and_keep_custom_steps():
    records = [
        ("00:00:00", _header(1, "Current state: plan mode")),
        ("00:01:00", "  · done (60.0 c, $0.1000)"),
        ("02:00:00", _header(2, "CURRENT STATE: custom validation", "codex/gpt-6")),
        ("02:03:00", "  · done (tokens: 500)"),
        ("04:00:00", _header(3, "plan mode", model=None)),
        ("04:03:00", "  · done"),
    ]

    stats = statlog.read_stats(_line(clock, message) for clock, message in records)

    assert set(stats) == {"plan mode", "custom validation"}
    _assert_timing(stats, "plan mode", 2, 240)
    _assert_timing(stats, "custom validation", 1, 180)


def test_legacy_done_uses_last_turn_and_excludes_post_result_idle_time():
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "implementation")),
        _line("00:00:10", "  · done (10.0 c, $0.1000)"),
        _line("00:00:20", "  · tool continuation"),
        _line("00:01:00", "  · done (50.0 c, $0.2000)"),
        _line("03:00:00", "  · transient provider error; retrying"),
        _line("04:00:00", _header(2, "cleanup")),
        _line("04:00:10", "  · done (10.0 c, $0.0100)"),
    ])

    _assert_timing(stats, "implementation", 1, 60)
    _assert_timing(stats, "cleanup", 1, 10)


@pytest.mark.parametrize("failure", [
    "Claude Code exited with code 1 (error #1 in a row).",
    "Codex CLI exited with code 2 (error #2 in a row).",
])
def test_legacy_provider_failure_closes_before_a_quota_wait(failure):
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "implementation")),
        _line("00:00:20", "  · last observed activity"),
        _line("00:00:35", failure),
        _line("03:00:00", "  ⏳ Provider quota exhausted; waiting"),
        _line("06:00:00", _header(2, "implementation")),
        _line("06:00:25", "  · done (25.0 c, $0.1000)"),
    ])

    _assert_timing(stats, "implementation", 2, 60)


@pytest.mark.parametrize("boundary", [
    "  · project root: C:\\project",
    "=== run ended: stopped ===",
    "Final state: done",
    "  ⏳ Provider quota exhausted; waiting",
    "  ⏸ Paused by operator",
])
def test_partial_iteration_closes_before_a_pause_or_new_launch(boundary):
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "fix-review")),
        _line("00:00:25", "  · last observed activity"),
        _line("00:10:00", boundary),
        _line("05:00:00", "orphan after the closed iteration"),
        _line("06:00:00", _header(1, "fix-review")),
        _line("06:00:35", "  · done (35.0 c, $0.1000)"),
    ])

    _assert_timing(stats, "fix-review", 2, 60, partial=1)


def test_rotation_orphans_and_partial_eof_are_limited_to_observed_activity():
    stats = statlog.read_stats([
        _line("00:00:00", "  · done (60.0 c, $0.5000)"),
        _line("00:01:00", statlog.iteration_finished(7, 0)),
        _line("01:00:00", _header(8, "claude-review")),
        _line("01:00:40", "  · inspecting changes"),
        "unprefixed output with no timestamp\n",
    ])

    assert set(stats) == {"claude-review"}
    _assert_timing(stats, "claude-review", 1, 40, partial=1)


def test_a_new_header_closes_a_partial_iteration_without_charging_the_gap():
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "implementation")),
        _line("00:00:12", "  · editing a file"),
        _line("07:00:00", _header(2, "cleanup")),
    ])

    _assert_timing(stats, "implementation", 1, 12, partial=1)
    _assert_timing(stats, "cleanup", 1, 0, partial=1)


@pytest.mark.parametrize("warning", [
    "  ⚠ the previous run left no exit record; its process may have stopped",
    "  · another run is active; waiting for the script lock",
])
def test_legacy_orphan_warning_closes_before_the_relaunch_gap(warning):
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "implementation")),
        _line("00:00:15", "  · last observed activity"),
        _line("02:00:00", warning),
        _line("03:00:00", "  · startup work before project root is printed"),
        _line("04:00:00", "  · project root: C:\\project"),
        _line("05:00:00", _header(1, "cleanup")),
        _line("05:00:20", "  · done (20.0 c, $0.1000)"),
    ])

    _assert_timing(stats, "implementation", 1, 15, partial=1)
    _assert_timing(stats, "cleanup", 1, 20)


def test_tagged_concurrent_pids_keep_identical_iteration_numbers_independent():
    stats = statlog.read_stats([
        _tagged_line("00:00:00", 101, "timing",
                     statlog.iteration_started(1, "implementation")),
        _tagged_line("00:00:05", 202, "timing",
                     statlog.iteration_started(1, "codex-review")),
        _tagged_line("00:00:10", 101, "output", "editing"),
        _tagged_line("00:00:25", 202, "timing",
                     statlog.iteration_finished(1, 0)),
        _tagged_line("00:00:40", 101, "timing",
                     statlog.iteration_finished(1, 0)),
    ])

    assert set(stats) == {"implementation", "codex-review"}
    _assert_timing(stats, "implementation", 1, 40)
    _assert_timing(stats, "codex-review", 1, 20)


@pytest.mark.parametrize("fake", [
    _header(999, "spoof"),
    statlog.iteration_started(999, "spoof"),
    statlog.iteration_finished(1, 0, 999),
    "Final state: done",
    "  · done (999.0 c, $0.1000)",
    "Claude Code exited with code 1 (error #1 in a row).",
])
def test_tagged_output_cannot_start_or_finish_an_iteration(fake):
    stats = statlog.read_stats([
        _tagged_line("00:00:00", 303, "output", fake),
        _tagged_line("00:00:05", 101, "timing",
                     statlog.iteration_started(1, "codex-review")),
        _tagged_line("00:00:15", 101, "output", fake),
        _tagged_line("00:00:35", 101, "timing",
                     statlog.iteration_finished(1, 0)),
    ])

    assert set(stats) == {"codex-review"}
    _assert_timing(stats, "codex-review", 1, 30)


def test_tagged_killed_pid_is_bounded_by_its_own_last_activity():
    stats = statlog.read_stats([
        _tagged_line("00:00:00", 101, "timing",
                     statlog.iteration_started(1, "implementation")),
        _tagged_line("00:00:12", 101, "output", "editing a file"),
        _tagged_line("03:00:00", 202, "output",
                     "  ⚠ the previous run left no exit record"),
        _tagged_line("04:00:00", 202, "output", "startup still running"),
        _tagged_line("05:00:00", 202, "timing",
                     statlog.iteration_started(1, "cleanup")),
        _tagged_line("05:00:25", 202, "timing",
                     statlog.iteration_finished(1, 0)),
    ])

    _assert_timing(stats, "implementation", 1, 12, partial=1)
    _assert_timing(stats, "cleanup", 1, 25)


def test_tagged_and_legacy_logs_merge_totals_for_the_same_state():
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "implementation")),
        _tagged_line("00:00:05", 101, "timing",
                     statlog.iteration_started(1, "implementation")),
        _line("00:00:20", statlog.iteration_finished(1, 0)),
        _tagged_line("00:00:35", 101, "timing",
                     statlog.iteration_finished(1, 0, 30)),
        _line("00:01:00", _header(2, "cleanup")),
        _line("00:01:10", "  · done"),
    ])

    assert set(stats) == {"implementation", "cleanup"}
    _assert_timing(stats, "implementation", 2, 50)
    _assert_timing(stats, "cleanup", 1, 10)


@pytest.mark.parametrize("state", [
    "custom ] · delimiter | [claude/opus]",
    'custom "quoted" \\ path\nsecond line',
])
def test_tagged_start_json_preserves_arbitrary_state_labels(state):
    stats = statlog.read_stats([
        _tagged_line("00:00:00", 101, "timing",
                     statlog.iteration_started(1, state)),
        _tagged_line("00:00:20", 101, "timing",
                     statlog.iteration_finished(1, 0)),
    ])

    assert set(stats) == {state}
    _assert_timing(stats, state, 1, 20)


@pytest.mark.parametrize("fake", [
    '  > === Iteration 999 === [spoof · codex/gpt-6]',
    '42: === Iteration 999 === [spoof · codex/gpt-6]',
    '  · tool: "=== Iteration 999 === [spoof · codex/gpt-6]"',
    '42:   · done (1.0 c, $0.1000)',
    '  > === Iteration 1 finished (exit 0) ===',
    '42: Final state: done',
])
def test_quoted_and_numbered_tool_output_is_not_a_timing_marker(fake):
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "codex-review")),
        _line("00:00:10", fake),
        _line("00:00:30", statlog.iteration_finished(1, 0)),
    ])

    assert set(stats) == {"codex-review"}
    _assert_timing(stats, "codex-review", 1, 30)


def test_finished_marker_overrides_done_and_stops_before_driver_hooks():
    stats = statlog.read_stats([
        _line("00:00:00", _header(1, "implementation")),
        _line("00:00:10", "  · done (10.0 c, $0.1000)"),
        _line("00:00:20", statlog.iteration_finished(99, 0)),
        _line("00:00:30", statlog.iteration_finished(1, 1)),
        _line("03:00:00", "  · driver hook completed"),
    ])

    _assert_timing(stats, "implementation", 1, 30)


def test_explicit_monotonic_elapsed_overrides_backwards_wall_clock():
    stats = statlog.read_stats([
        _line("00:01:00", _header(1, "cleanup")),
        _line("00:00:10", statlog.iteration_finished(1, 0, 12.345)),
    ])

    _assert_timing(stats, "cleanup", 1, 12.345)


def test_backwards_legacy_wall_clock_is_clamped_to_zero():
    stats = statlog.read_stats([
        _line("00:01:00", _header(1, "cleanup")),
        _line("00:00:50", "  · done (10.0 c, $0.1000)"),
    ])

    _assert_timing(stats, "cleanup", 1, 0)


def test_report_shows_sum_mean_and_percentage_per_state(tmp_path, capsys):
    log = _write_log(tmp_path / "copy.log", [
        ("00:00:00", _header(1, "plan mode")),
        ("00:01:00", statlog.iteration_finished(1, 0)),
        ("00:02:00", _header(2, "plan mode")),
        ("00:05:00", statlog.iteration_finished(2, 0)),
        ("00:06:00", _header(3, "custom validation")),
        ("00:08:00", statlog.iteration_finished(3, 0)),
    ])
    before = log.read_bytes()

    statlog.report_stats("pytest-stat", log)

    output = capsys.readouterr().out
    rows = _table_rows(output)
    assert rows["State"] == ["Iterations", "Total", "Average", "% total"]
    assert rows["plan mode"] == ["2", "00:04:00", "00:02:00", "66.7%"]
    assert rows["custom validation"] == ["1", "00:02:00", "00:02:00", "33.3%"]
    assert rows["TOTAL"] == ["3", "00:06:00", "00:02:00", "100.0%"]
    assert str(log) in output
    assert log.read_bytes() == before


def test_report_zero_time_and_partial_counts_are_defined(tmp_path, capsys):
    log = _write_log(tmp_path / "copy.log", [
        ("00:00:00", _header(1, "cleanup")),
    ])

    statlog.report_stats(path=log)

    output = capsys.readouterr().out
    assert _table_rows(output)["cleanup"] == ["1", "00:00:00", "00:00:00", "0.0%"]
    assert "1 partial iteration" in output


@pytest.mark.parametrize("exists", [False, True])
def test_missing_or_empty_log_reports_nothing_without_creating_data(
        tmp_path, capsys, exists):
    path = tmp_path / "missing.log"
    if exists:
        path.write_text("", encoding="utf-8")

    statlog.report_stats(path=path)

    assert "nothing to report" in capsys.readouterr().out
    assert path.exists() is exists


def test_real_runner_mirror_log_roundtrips_elapsed_and_state(
        tmp_path, monkeypatch, capsys):
    elapsed = iter([100.0, 112.345])
    # Replace only the runner's time binding; background owners retain real clocks.
    monkeypatch.setattr(cyclecore, "time", SimpleNamespace(
        time=cyclecore.time.time, monotonic=lambda: next(elapsed),
        sleep=cyclecore.time.sleep))

    def agent(*args, **kwargs):
        streamrender._render_claude_event({
            "type": "assistant", "message": {"content": [{
                "type": "text", "text": "\n".join([
                    "These diagnostics are quoted from a previous run:",
                    "Final state: done",
                    statlog.iteration_finished(1, 0, 999),
                    _header(999, "spoof"),
                    statlog.iteration_started(999, "spoof"),
                ]),
            }]},
        }, False)
        streamrender._render_claude_event(
            {"type": "result", "subtype": "success", "duration_ms": 1000}, True)
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", agent)
    monkeypatch.setattr(cyclecore, "last_rate_limit_event", lambda: None)
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: None)
    with isolated_run(monkeypatch, tmp_path):
        out, err = sys.stdout, sys.stderr
        try:
            cyclecore.run_loop(OneShotDriver(),
                               seq_args(tmp_path, no_statusline=True),
                               app_name="pytest-stat-real", wait_on_start=False)
        finally:
            sys.stdout, sys.stderr = out, err
        log = console.log_file_path("pytest-stat-real")
        contents = log.read_text(encoding="utf-8")
        stats = statlog.read_stats(contents.splitlines())

    assert ("kind=timing " + statlog.iteration_started(1, "the-thing")) in contents
    assert ("kind=timing " + statlog.iteration_finished(1, 0, 12.345)) in contents
    assert "kind=output Final state: done" in contents
    assert ("kind=output " + statlog.iteration_finished(1, 0, 999)) in contents
    assert ("kind=output " + _header(999, "spoof")) in contents
    assert set(stats) == {"the-thing"}
    assert stats["the-thing"].iterations == 1
    assert stats["the-thing"].partial_iterations == 0
    assert stats["the-thing"].total_seconds == pytest.approx(12.345)


@pytest.mark.parametrize("named_log", [False, True])
def test_stat_cli_reports_without_beginning_a_run_or_modifying_log(
        tmp_path, monkeypatch, capsys, named_log):
    def refuse_run(*args, **kwargs):
        pytest.fail("--stat must not begin a run")

    with isolated_run(monkeypatch, tmp_path):
        projectroot.set_project_root(str(tmp_path))
        log = (tmp_path / "named.log" if named_log
               else console.log_file_path("pytest-stat-cli"))
        log.parent.mkdir(parents=True, exist_ok=True)
        _write_log(log, [
            ("00:00:00", _header(1, "cleanup")),
            ("00:00:45", statlog.iteration_finished(1, 0)),
        ])
        before = log.read_bytes()
        monkeypatch.setattr(runlifecycle, "begin_run", refuse_run)
        monkeypatch.setattr(cyclecore, "wait_before_start", refuse_run)
        argv = ["--stat", "--project-dir", str(tmp_path), "--start-in", "1"]
        if named_log:
            argv += ["--cost-log", str(log)]
        args = cyclecore.parse_args(argv)
        driver = OneShotDriver()

        assert cyclecore.is_report(args) is True
        cyclecore.run_loop(driver, args, app_name="pytest-stat-cli")

        assert driver.served == 0
        assert log.read_bytes() == before
        assert _table_rows(capsys.readouterr().out)["cleanup"] == [
            "1", "00:00:45", "00:00:45", "100.0%"]
