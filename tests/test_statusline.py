"""Tests for the pinned status line.

Everything here exercises the PURE renderer (rows -> strings) or the controller
driven by synthetic events. Nothing asserts on raw escape sequences: the byte
level is the terminal's business and differs per emulator, while the rows are
what a human actually reads.
"""

import io
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_loop import statusline as sl


NOW = 1_700_000_000.0


def sequential_status(**fields):
    """A one-Job run, 12 iterations in, the job 4m12s into its current one."""
    status = sl.LoopStatus(
        jobs=[sl.Job(1, running=True, iteration=12, item="bmx-bike.md",
                     model="opus", started_at=NOW - 252)],
        iteration=12, max_iterations=40, provider="claude",
        run_started_at=NOW - 3600, phase="running",
        quotas=[("session", 43.0, 95.0, NOW + 7860),
                ("week", 61.0, 95.0, None)],
        script_limits=[("max-runs", "40")],
    )
    status.update(**fields)
    return status


def parallel_status(count=3):
    jobs = [sl.Job(i, running=i < count, iteration=i, item=f"item-{i}.md",
                   model="opus", started_at=(NOW - 60 * i) if i < count else 0.0)
            for i in range(1, count + 1)]
    return sl.LoopStatus(jobs=jobs, iteration=7, max_iterations=40,
                         provider="claude", run_started_at=NOW - 3720,
                         phase="running")


# --- the pure renderer ---------------------------------------------------------


def test_one_job_renders_summary_job_legend_and_note_rows():
    rows = sl.render_rows(sequential_status(), 200, now=NOW)

    assert len(rows) == 4          # summary + 1 job + legend + note
    summary, job_row = rows[0], rows[1]
    assert "iter 12/40" in summary
    assert "claude/opus" in summary
    assert "session 43% (ceil 95%, 2h11m)" in summary
    assert "week 61% (ceil 95%)" in summary
    assert "max-runs 40" in summary
    assert "job 1" in job_row and "bmx-bike.md" in job_row


def test_each_job_gets_its_own_row_and_an_idle_one_has_no_clock():
    rows = sl.render_rows(parallel_status(3), 200, now=NOW)

    assert len(rows) == 6          # summary + 3 jobs + legend + note
    assert "3 jobs" in rows[0]
    assert [f"job {i}" in rows[i] for i in (1, 2, 3)] == [True] * 3
    assert "item-3.md" not in rows[3] and "idle" in rows[3]
    assert "0m00s" not in rows[3]


def test_rows_never_exceed_the_terminal_width():
    for width in (40, 60, 100):
        for line in sl.render_rows(sequential_status(), width, now=NOW):
            assert sl.cell_width(line) <= width
        assert "…" in sl.render_rows(sequential_status(), 40, now=NOW)[0]


def test_rand_marker_follows_the_random_order_flag():
    assert "rand" in sl.render_rows(
        sequential_status(random_order=True), 200, now=NOW)[0]
    assert "rand" not in sl.render_rows(sequential_status(), 200, now=NOW)[0]


def test_max_marker_is_dropped_for_an_unbounded_run():
    summary = sl.render_rows(
        sequential_status(max_iterations=None), 200, now=NOW)[0]

    assert "iter 12" in summary and "/40" not in summary


def test_an_empty_model_reads_as_cli_default():
    status = sequential_status()
    status.jobs[0].update(model="")

    rows = sl.render_rows(status, 200, now=NOW)

    assert "claude/cli default" in rows[0]
    assert "cli default" in rows[1]


def test_stop_pending_marker_appears_only_while_pending():
    assert "STOP pending" not in sl.render_rows(
        sequential_status(), 200, now=NOW)[0]
    assert "STOP pending — press s to cancel" in sl.render_rows(
        sequential_status(stop_pending=True), 200, now=NOW)[0]


def test_the_two_clocks_are_different_clocks():
    """The job row times the CURRENT iteration; the summary times the whole run."""
    rows = sl.render_rows(sequential_status(), 200, now=NOW)

    assert "1h00m" in rows[0] and "1h00m" not in rows[1]
    assert "4m12s" in rows[1] and "4m12s" not in rows[0]


def test_quota_without_a_figure_reads_as_not_available():
    status = sequential_status(quotas=[("session", None, 95.0, None)])

    assert "session n/a" in sl.render_rows(status, 200, now=NOW)[0]


def test_note_row_carries_the_transient_message():
    rows = sl.render_rows(sequential_status(note="stop requested"), 200, now=NOW)

    assert rows[-1].strip() == "stop requested"


def test_colorize_only_wraps_percentages():
    line = sl.colorize(" session 43% (ceil 95%)")

    assert "\x1b[" in line
    assert "session" in line and line.count("\x1b[0m") == 2
    assert sl.colorize(" no figures here") == " no figures here"


# --- key legend derived from the registered Actions ----------------------------


class _RecordingAction(sl.Action):
    key = "x"
    help = "do the thing"

    def __init__(self):
        self.runs = 0

    def run(self, app):
        self.runs += 1


def test_key_legend_row_is_built_from_registered_actions():
    app = sl.StatusApp(enabled=False, default_actions=False)
    action = app.register_action(_RecordingAction())

    legend = app.render(width=200, now=NOW)[-2]

    assert legend.strip() == "keys: x do the thing"

    app.handle_event(sl.Key("x"))
    assert action.runs == 1


def test_default_actions_put_stop_and_help_in_the_legend():
    app = sl.StatusApp(enabled=False)

    legend = app.render(width=200, now=NOW)[-2]

    assert "s stop" in legend and "h help" in legend


def test_an_unavailable_action_leaves_the_legend():
    class Hidden(sl.Action):
        key = "z"
        help = "hidden"

        def available(self, app):
            return False

        def run(self, app):
            raise AssertionError("must not run")

    app = sl.StatusApp(enabled=False, default_actions=False)
    app.register_action(Hidden())

    assert app.render(width=200, now=NOW)[-2].strip() == ""
    app.handle_event(sl.Key("z"))   # dispatch must not reach it either


def test_help_key_writes_the_full_key_list_into_the_note_row():
    app = sl.StatusApp(enabled=False)

    app.handle_event(sl.Key("?"))

    assert "s stop" in app.status.note and "h/? help" in app.status.note


def test_an_unknown_key_points_at_the_help_key():
    app = sl.StatusApp(enabled=False)

    app.handle_event(sl.Key("q"))

    assert "press h for help" in app.status.note


# --- the stop key --------------------------------------------------------------


def test_stop_key_creates_the_sentinel_and_pressing_it_again_cancels(tmp_path):
    sentinel = tmp_path / "stop"
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))

    app.handle_event(sl.Key("s"))
    assert sentinel.exists() and app.status.stop_pending is True
    assert "cancel stop" in app.render(width=200, now=NOW)[-2]

    app.handle_event(sl.Key("s"))
    assert not sentinel.exists() and app.status.stop_pending is False


def test_stop_file_defaults_to_the_engine_sentinel(monkeypatch):
    from claude_loop import cyclecore

    monkeypatch.setattr(cyclecore, "STOP_FILE", "/somewhere/stop")

    assert sl.StatusApp(enabled=False).stop_file == "/somewhere/stop"


# --- the cancel grace (cyclecore.confirm_stop_request) -------------------------


class _FakeApp:
    """An interactive app whose note() can act — the seam a test needs to make
    "the user pressed s again" happen at a defined moment."""

    enabled = True

    def __init__(self, on_note=None):
        self.notes = []
        self.fields = {}
        self._on_note = on_note

    def update(self, **fields):
        self.fields.update(fields)

    def note(self, text):
        self.notes.append(text)
        if self._on_note is not None:
            self._on_note()


def test_a_stop_request_cancelled_inside_the_grace_leaves_no_trace(monkeypatch, tmp_path):
    from claude_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(cyclecore, "STOP_FILE", str(sentinel))
    app = _FakeApp(on_note=lambda: sentinel.unlink(missing_ok=True))

    assert cyclecore.confirm_stop_request(app, grace=5.0, poll=0.01) is False
    assert app.fields["phase"] == "idle" and app.fields["stop_pending"] is False
    assert "stop cancelled" in app.notes[-1]
    assert "press s to cancel" in app.notes[0]


def test_a_stop_request_still_pending_after_the_grace_stops_the_run(monkeypatch, tmp_path):
    from claude_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(cyclecore, "STOP_FILE", str(sentinel))
    app = _FakeApp()

    assert cyclecore.confirm_stop_request(app, grace=0.05, poll=0.01) is True
    assert app.fields["phase"] == "stopping"


def test_a_non_interactive_run_stops_without_any_grace(monkeypatch, tmp_path):
    """Automation (`touch stop` from a script, a piped run) must not be slowed."""
    from claude_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(cyclecore, "STOP_FILE", str(sentinel))
    started = time.monotonic()

    assert cyclecore.confirm_stop_request(None, grace=60) is True
    assert cyclecore.confirm_stop_request(
        sl.StatusApp(enabled=False), grace=60) is True
    assert time.monotonic() - started < 1.0


# --- disabled / no-TTY paths ---------------------------------------------------


class _FakeStream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_a_non_tty_stream_gets_a_null_terminal():
    assert isinstance(sl.terminal_for(_FakeStream(False)), sl.NullTerminal)


def test_the_env_flag_disables_the_status_line(monkeypatch):
    monkeypatch.setenv(sl.ENV_FLAG, "0")

    assert isinstance(sl.terminal_for(_FakeStream(True)), sl.NullTerminal)


def test_a_disabled_app_writes_nothing_and_still_serves_its_api():
    stream = _FakeStream(True)   # a real TTY: only `enabled=False` disables us
    app = sl.StatusApp(terminal=sl.terminal_for(stream, enabled=False))

    with app:
        app.update(iteration=3, phase="running")
        app.job(1).start(item="a.md", model="opus", now=NOW)
        app.note("hello")

    assert app.enabled is False
    assert stream.getvalue() == ""
    assert app.status.iteration == 3
    assert app.job(1).iteration == 1


def test_a_pinned_terminal_writes_its_rows_to_the_real_stream(monkeypatch):
    """The bytes are the terminal's business; that the ROWS land is ours."""
    stream = _FakeStream(True)
    monkeypatch.setattr(sl, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(sl.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))
    app = sl.StatusApp(terminal=sl.terminal_for(stream),
                       input_source=sl.NullInputSource(), refresh=60)

    with app:
        assert app.enabled is True
        app.update(iteration=4, phase="running")
        assert "iter 4" in stream.getvalue()

    assert app.enabled is False   # region released on the way out


def test_start_is_a_no_op_when_the_terminal_cannot_reserve(monkeypatch):
    """Any failure swaps in the Null terminal — the run must not care."""
    class Refusing(sl.Terminal):
        def __init__(self):
            super().__init__(stream=_FakeStream(True))

        def reserve(self, rows):
            raise RuntimeError("no region here")

    app = sl.StatusApp(terminal=Refusing())
    with app:
        app.update(phase="running")

    assert isinstance(app.terminal, sl.NullTerminal)


# --- Job / LoopStatus bookkeeping ---------------------------------------------


def test_job_start_and_finish_move_only_the_job_clock():
    job = sl.Job(1)

    job.start(item="a.md", model="opus", prompt="p", now=NOW)
    assert (job.running, job.iteration, job.item) == (True, 1, "a.md")
    assert job.elapsed(NOW + 30) == 30

    job.finish()
    assert job.running is False and job.elapsed(NOW + 30) is None

    job.start(item="b.md", now=NOW)
    assert job.iteration == 2


def test_run_clock_is_latched_once():
    status = sl.LoopStatus()

    status.mark_run_started(NOW)
    status.mark_run_started(NOW + 999)

    assert status.elapsed(NOW + 60) == 60


def test_a_status_always_has_at_least_one_job():
    assert len(sl.LoopStatus(jobs=[]).jobs) == 1
    assert sl.LoopStatus().job(3).job_id == 3


def test_snapshot_detaches_from_later_mutation():
    status = sequential_status()

    frozen = status.snapshot()
    status.update(iteration=99)
    status.jobs[0].update(item="other.md")

    assert frozen.iteration == 12 and frozen.jobs[0].item == "bmx-bike.md"


# --- settings registry ---------------------------------------------------------


def test_settings_registry_yields_the_overrides_dict_cmdline_consumes():
    box = {"max_runs": 5, "session": 95.0}
    registry = sl.SettingsRegistry()
    registry.add(sl.NumberSetting(
        "max-runs", "--max-runs", lambda: box["max_runs"],
        lambda v: box.__setitem__("max_runs", int(v))))
    ceiling = registry.add(sl.PercentSetting(
        "session ceiling", "--session-limit", lambda: box["session"],
        lambda v: box.__setitem__("session", float(v))))

    assert registry.overrides() == {}       # nothing edited yet

    ceiling.nudge(-5)

    assert box["session"] == 90.0
    assert registry.overrides() == {"--session-limit": "90"}
    assert registry.status_entries() == [("max-runs", "5"),
                                         ("session ceiling", "90%")]


def test_percent_setting_is_clamped_to_its_bounds():
    box = {"v": 99.0}
    setting = sl.PercentSetting("session", "--session-limit",
                                lambda: box["v"], lambda v: box.__setitem__("v", v))

    setting.nudge(5)
    assert box["v"] == 100
    setting.set(-10)
    assert box["v"] == 0


# --- quota feed ----------------------------------------------------------------


class _FakeUsageSource:
    def __init__(self, usage):
        self.usage = usage
        self.reads = 0

    def get_usage(self, cache_value=True):
        self.reads += 1
        return self.usage


def test_quota_rows_follow_the_policy_rules():
    from claude_loop.limits import LimitPolicy, SessionLimit
    from claude_loop.usage import Usage, UsageReading

    usage = Usage(UsageReading(43.0, NOW + 600), UsageReading(61.0, None),
                  UsageReading(None, None), [])
    source = _FakeUsageSource(usage)

    rows = sl.quota_rows(source, LimitPolicy([SessionLimit(80)]), now=NOW)

    assert rows == [("session", 43.0, 80.0, NOW + 600)]
    assert source.reads == 1


def test_a_disabled_status_line_never_reads_the_usage_endpoint():
    """`get_usage` on a cold cache is a real HTTP round-trip; a status line
    nobody can see must not buy one."""
    source = _FakeUsageSource(None)

    sl.push_quotas(sl.StatusApp(enabled=False), source)

    assert source.reads == 0


def test_quota_rows_never_raise():
    class Broken:
        def get_usage(self, cache_value=True):
            raise RuntimeError("endpoint down")

    assert sl.quota_rows(Broken()) == []
    assert sl.quota_rows(None) == []


# --- small helpers -------------------------------------------------------------


@pytest.mark.parametrize("seconds,expected", [
    (0, "0m00s"), (72, "1m12s"), (252, "4m12s"), (3600, "1h00m"),
    (3720, "1h02m"), (None, ""),
])
def test_format_elapsed(seconds, expected):
    assert sl.format_elapsed(seconds) == expected


def test_fit_truncates_with_an_ellipsis():
    assert sl.fit("abcdef", 10) == "abcdef"
    assert sl.fit("abcdef", 4) == "abc…"
    assert sl.fit("abcdef", 0) == ""


def test_format_prompt_block_heads_the_prompt_and_keeps_it_verbatim():
    block = sl.format_prompt_block(job_id=2, label="garlic.md",
                                   prompt="line one\nline two\n", width=60)
    head, body, foot = block.split("\n")[0], block.split("\n")[1:-1], \
        block.split("\n")[-1]

    assert "job 2" in head and "garlic.md" in head and "18 chars" in head
    assert sl.cell_width(head) == 60 and foot == "─" * 60
    assert body == ["line one", "line two"]   # verbatim: still pasteable


def test_format_prompt_block_survives_an_empty_prompt():
    block = sl.format_prompt_block(job_id=1, label="", prompt="", width=60)

    assert "(no label)" in block and "0 chars" in block


# --- the dry run prints job 1's prompt ----------------------------------------


def test_a_dry_run_prints_the_prompt_block_for_job_one(tmp_path, capsys):
    """The joined `-p …` argv line hides the prompt; the block is what shows it."""
    from claude_loop import cyclecore
    from claude_loop.cyclecore import AgentCommand, Driver

    class _OneShot(Driver):
        def next_command(self):
            return AgentCommand("do the thing, carefully", "", "the-thing")

    args = type("NS", (), {})()
    args.max = None
    args.dry_run = True
    args.raw = False
    args.start_in = None
    args.max_strike = None
    args.git_push = "none"
    args.project_dir = str(tmp_path)
    args.cost = False
    args.no_statusline = False

    previous = cyclecore.project_dir()
    streams = (sys.stdout, sys.stderr)
    try:
        cyclecore.run_loop(_OneShot(), args, app_name="pytest-statusline",
                           setup_logging=False, wait_on_start=False)
    finally:
        cyclecore.set_project_root(previous)
        sys.stdout, sys.stderr = streams

    out = capsys.readouterr().out
    assert "DRY-RUN:" in out                      # unchanged, tests match on it
    assert "prompt · job 1 · the-thing · 23 chars" in out
    assert "do the thing, carefully" in out
    assert out.count("prompt · job 1") == 1       # once, not per pass


def test_escape_decoder_names_the_special_keys_on_both_platforms():
    posix = sl._EscapeDecoder(scan_codes=False)

    assert [e.char for e in posix.feed("s")] == ["s"]
    assert posix.feed("\x1b") == [] and posix.feed("[") == []
    assert [e.char for e in posix.feed("A")] == ["up"]
    assert posix.feed("\x1b") == []
    assert [e.char for e in posix.flush()] == ["\x1b"]     # a bare Esc

    windows = sl._EscapeDecoder(scan_codes=True)
    assert windows.feed("\xe0") == []
    assert [e.char for e in windows.feed("K")] == ["left"]


def test_escape_decoder_drops_sequences_nothing_understands():
    decoder = sl._EscapeDecoder(scan_codes=False)

    for char in "\x1b[200~":                 # bracketed paste: not a keypress
        events = decoder.feed(char)

    assert events == []
