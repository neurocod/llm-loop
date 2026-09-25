"""The two lines `--cost` reads back must be the two lines the run prints.

`costlog.report_costs` parses a run's spend out of the mirror log from a header
`run_loop` prints and a "done" line `streamrender` prints. Those are three
modules, and until the patterns moved next to the wording (`costlog`) the only
thing tying them together was a comment — the "done" line had already moved to
another file once. So these pins do not hand-write the lines: they take them
from the REAL printers, write them into a log and read that log back.
"""

import sys

import pytest

from llm_loop import console, costlog, cyclecore, projectroot, streamrender

from _runfixtures import OneShotDriver, seq_args


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    out, err = sys.stdout, sys.stderr
    previous = projectroot.project_dir()
    monkeypatch.setattr(console, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(streamrender, "_turn_cost_base", 0.0)
    yield
    projectroot.set_project_root(previous)
    sys.stdout, sys.stderr = out, err


def _printed_header(tmp_path, capsys) -> str:
    """Iteration 1's banner, as a dry run of the real loop prints it."""
    cyclecore.run_loop(OneShotDriver(),
                       seq_args(tmp_path, dry_run=True, no_statusline=True),
                       app_name="pytest-cost-header")
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "Iteration" in ln]
    assert len(lines) == 1, lines
    return lines[0]


def _printed_result(capsys, **event) -> str:
    streamrender._render_claude_event({"type": "result", **event}, True)
    return capsys.readouterr().out.strip("\n")


def _report(tmp_path, capsys, lines) -> str:
    log = tmp_path / "copy.log"
    # The mirror log prefixes every record with a timestamp; parse it as written.
    log.write_text("".join(f"2026-09-25 00:00:00 {ln}\n" for ln in lines),
                   encoding="utf-8")
    costlog.report_costs("pytest-cost", log)
    return capsys.readouterr().out


def test_the_printed_header_and_done_line_are_what_the_report_sums(
        tmp_path, capsys):
    header = _printed_header(tmp_path, capsys)
    first = _printed_result(capsys, subtype="success", duration_ms=2243,
                            total_cost_usd=0.2015)
    # A second turn of the same process: the line shows what that turn ADDED.
    second = _printed_result(capsys, subtype="success", duration_ms=1991,
                             total_cost_usd=0.2204)

    out = _report(tmp_path, capsys, [header, first, second])

    assert "Session 1: 2 costs, $0.2204" in out
    assert "TOTAL: 1 sessions, 2 costs, $0.2204" in out


def test_a_later_iterations_header_is_not_a_run_boundary(tmp_path, capsys):
    done = costlog.done_line(1000, 0.5)
    out = _report(tmp_path, capsys, [
        costlog.iteration_header(1), done,
        costlog.iteration_header(11), done,
        costlog.iteration_header(12), done,
    ])

    assert "TOTAL: 1 sessions, 3 costs, $1.5000" in out


def test_a_failed_turn_is_shown_but_not_summed(tmp_path, capsys):
    """A failed result carries the same figures, and is not a completed billed
    iteration — the pattern must keep telling it from "done"."""
    failed = _printed_result(capsys, subtype="error_max_turns",
                             duration_ms=1000, total_cost_usd=0.75)
    assert "$0.7500" in failed

    out = _report(tmp_path, capsys, [costlog.iteration_header(1), failed])

    assert "TOTAL: 1 sessions, 0 costs, $0.0000" in out


def test_a_codex_done_line_is_not_a_cost(tmp_path, capsys):
    """Codex reports tokens, not dollars: its "done" must not parse as spend."""
    streamrender._render_codex_event({
        "type": "turn.completed",
        "usage": {"input_tokens": 12, "cached_input_tokens": 3,
                  "output_tokens": 4},
    })
    codex_done = capsys.readouterr().out.strip("\n")
    assert "done" in codex_done

    out = _report(tmp_path, capsys, [costlog.iteration_header(1), codex_done])

    assert "TOTAL: 1 sessions, 0 costs, $0.0000" in out
