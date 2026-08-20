"""Tests for the pinned status line.

Everything here exercises the PURE renderer (rows -> strings) or the controller
driven by synthetic events. Nothing asserts on raw escape sequences: the byte
level is the terminal's business and differs per emulator, while the rows are
what a human actually reads.
"""

import io
import logging
import os
import sys
import threading
import time

import pytest

from llm_loop import statusline as sl


NOW = 1_700_000_000.0


def sequential_status(**fields):
    """A one-Job run, 12 iterations in, the job 4m12s into its current one."""
    status = sl.LoopStatus(
        jobs=[sl.Job(1, running=True, iteration=12, item="bmx-bike.md",
                     model="opus", started_at=NOW - 252)],
        iteration=12, max_iterations=40, provider="claude",
        run_started_at=NOW - 3600, phase="running",
        quotas=[sl.QuotaRow("session", 43.0, NOW + 7860, "ceil 95%"),
                sl.QuotaRow("week", 61.0, None, "ceil 95%")],
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


def test_one_job_renders_rule_summary_job_legend_and_note_rows():
    rows = sl.render_rows(sequential_status(), 200, now=NOW)

    assert len(rows) == 5          # rule + summary + 1 job + legend + note
    summary, job_row = rows[1], rows[2]
    assert "iter 12/40" in summary
    assert "claude/opus" in summary
    assert "session 43% (2h11m) / ceil 95%" in summary
    assert "week 61% / ceil 95%" in summary
    assert "max-runs 40" in summary
    assert "job 1" in job_row and "bmx-bike.md" in job_row


def test_the_item_comes_last_and_keeps_the_rest_of_the_line():
    """The one unbounded field goes where the space is, and is not truncated
    while the line has room — a cut config path is the least useful thing to
    show, and the fields before it are all short and fixed."""
    long_item = "products/configs/kitchenware-and-accessories-non-electric/loaf-pan.md"
    status = sequential_status()
    status.job(1).start(item=long_item, model="gpt-5.6-terra",
                        iteration=12, now=NOW - 252)

    job_row = sl.render_rows(status, 200, now=NOW)[2]

    assert job_row.endswith(long_item)          # last cell, nothing after it
    assert "…" not in job_row

    # Within one run the item column starts at the same offset on every row,
    # idle ones included — that is what the padded empty clock is for. (Across
    # runs it may differ: the model column is sized from that run's own names.)
    rows = sl.render_rows(parallel_status(3), 200, now=NOW)
    running, idle = rows[2], rows[4]
    assert running.index("item-1.md") == idle.index("idle")


def test_each_job_gets_its_own_row_and_an_idle_one_has_no_clock():
    rows = sl.render_rows(parallel_status(3), 200, now=NOW)

    assert len(rows) == 7          # rule + summary + 3 jobs + legend + note
    assert "3 jobs" in rows[1]
    assert [f"job {i}" in rows[i + 1] for i in (1, 2, 3)] == [True] * 3
    assert "item-3.md" not in rows[4] and "idle" in rows[4]
    assert "0m00s" not in rows[4]


def test_rows_never_exceed_the_terminal_width():
    for width in (40, 60, 100):
        rows = sl.render_rows(sequential_status(), width, now=NOW)
        for line in rows:
            assert sl.cell_width(line) <= width
            assert "\n" not in line        # exactly one terminal line per row
    assert "…" in sl.render_rows(sequential_status(), 40, now=NOW)[1]


def test_rand_marker_follows_the_random_order_flag():
    assert "rand" in sl.render_rows(
        sequential_status(random_order=True), 200, now=NOW)[1]
    assert "rand" not in sl.render_rows(sequential_status(), 200, now=NOW)[1]


def test_max_marker_is_dropped_for_an_unbounded_run():
    summary = sl.render_rows(
        sequential_status(max_iterations=None), 200, now=NOW)[1]

    assert "iter 12" in summary and "/40" not in summary


def test_an_empty_model_reads_as_cli_default():
    status = sequential_status()
    status.jobs[0].update(model="")

    rows = sl.render_rows(status, 200, now=NOW)

    assert "claude/cli default" in rows[1]
    assert "cli default" in rows[2]


def test_stop_pending_marker_appears_only_while_pending():
    assert "STOP pending" not in sl.render_rows(
        sequential_status(), 200, now=NOW)[1]
    pending = sl.render_rows(sequential_status(stop_pending=True), 200,
                             now=NOW)[1]
    assert f"{sl.STOP_GLYPH} STOP pending — press s to cancel" in pending

    # …but not twice on one line: once the phase is "stopping" the row already
    # opens with the same glyph.
    status = sequential_status(stop_pending=True)
    status.update(phase="stopping")
    stopping = sl.render_rows(status, 200, now=NOW)[1]
    assert stopping.count(sl.STOP_GLYPH) == 1
    assert "STOP pending — press s to cancel" in stopping


def test_the_two_clocks_are_different_clocks():
    """The job row times the CURRENT iteration; the summary times the whole run."""
    rows = sl.render_rows(sequential_status(), 200, now=NOW)

    assert "1h00m" in rows[1] and "1h00m" not in rows[2]
    assert "4m12s" in rows[2] and "4m12s" not in rows[1]


def test_quota_without_a_figure_reads_as_not_available():
    status = sequential_status(quotas=[sl.QuotaRow("session", None, None,
                                                   "ceil 95%")])

    assert "session n/a / ceil 95%" in sl.render_rows(status, 200, now=NOW)[1]


def test_a_quota_nobody_gates_on_shows_the_providers_half_alone():
    """The policy owns only the right-hand half; a window with no rule keeps its
    figures and simply says nothing about a ceiling."""
    status = sequential_status(quotas=[sl.QuotaRow("week", 61.0, NOW + 7860)])

    summary = sl.render_rows(status, 200, now=NOW)[1]
    assert "week 61% (2h11m)" in summary
    assert sl.POLICY_SEPARATOR not in summary


def test_note_row_carries_the_transient_message():
    rows = sl.render_rows(sequential_status(note="stop requested"), 200, now=NOW)

    assert rows[-1].strip() == "stop requested"


def test_colorize_styles_percentages_and_chrome_and_nothing_else():
    line = sl.colorize(" session 43% (ceil 95%)")

    assert "\x1b[" in line
    assert "session" in line and line.count("\x1b[0m") == 2
    assert sl.colorize(" no figures here") == " no figures here"


def test_the_policy_ceiling_borrows_the_colour_of_its_own_reading():
    green, red = sl._SGR["green"], sl._SGR["bold red"]
    line = sl.colorize(f"week 50% (3d14h){sl.POLICY_SEPARATOR}ceil 95%")

    assert line.count(green) == 2 and red not in line   # the ceiling is not alarming
    # ...but only for the reading in its OWN field: past the chrome pipe the
    # scale takes over again, and a comfortable ceiling cannot green a hot week.
    two = sl.colorize(f"week 95%{sl.SEPARATOR}session 50%{sl.POLICY_SEPARATOR}ceil 80%")
    assert two.count(red) == 1 and two.count(green) == 2


def test_a_ceiling_with_no_reading_behind_it_is_green_not_alarming():
    green, red = sl._SGR["green"], sl._SGR["bold red"]
    # --codex reports no session window at all; the ceiling gates nothing.
    line = sl.colorize(f"session n/a{sl.POLICY_SEPARATOR}ceil 95%")

    assert line.count(green) == 1 and red not in line
    # ...and the hot field before the pipe must not lend its colour across it.
    two = sl.colorize(f"week 95%{sl.SEPARATOR}session n/a{sl.POLICY_SEPARATOR}ceil 95%")
    assert two.count(red) == 1 and two.count(green) == 1


# --- chrome: the rule and the value separators ---------------------------------


def test_the_rule_row_opens_the_area_and_spans_the_width():
    for width in (40, 100, 200):
        rule = sl.render_rows(sequential_status(), width, now=NOW)[0]

        assert rule == sl.RULE_CHAR * width       # plain chars: no colour needed
        assert sl.cell_width(rule) == width


def test_values_inside_a_row_are_separated_by_the_chrome_pipe():
    rows = sl.render_rows(sequential_status(), 200, now=NOW)

    assert " | " in rows[1] and "iter 12/40 | claude/opus" in rows[1]
    assert " | " in rows[2]                        # job row columns
    assert " | " in sl.StatusApp(enabled=False).render(width=200, now=NOW)[-2]


def test_the_rule_and_the_separators_share_one_muted_style():
    dim = sl._SGR[sl.CHROME_STYLE]
    painted = sl.colorize(f"a{sl.SEPARATOR}b")

    assert painted == f"a {dim}|{sl._SGR_RESET} b"
    assert (sl.colorize(sl.RULE_CHAR * 8)
            == f"{dim}{sl.RULE_CHAR * 8}{sl._SGR_RESET}")


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
    from llm_loop import cyclecore

    monkeypatch.setattr(cyclecore, "STOP_FILE", "/somewhere/stop")

    assert sl.StatusApp(enabled=False).stop_file == "/somewhere/stop"


# --- the cancel grace (cyclecore.confirm_stop_request) -------------------------


class _FakeApp:
    """An interactive app whose note() can act — the seam a test needs to make
    "the user pressed s again" happen at a defined moment."""

    enabled = True
    stop_requested_here = True     # as if the sentinel came from the `s` key

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
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(cyclecore, "STOP_FILE", str(sentinel))
    app = _FakeApp(on_note=lambda: sentinel.unlink(missing_ok=True))

    assert cyclecore.confirm_stop_request(app, grace=5.0, poll=0.01) is False
    assert app.fields["phase"] == "idle" and app.fields["stop_pending"] is False
    assert "stop cancelled" in app.notes[-1]
    assert "press s to cancel" in app.notes[0]


def test_a_stop_request_still_pending_after_the_grace_stops_the_run(monkeypatch, tmp_path):
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(cyclecore, "STOP_FILE", str(sentinel))
    app = _FakeApp()

    assert cyclecore.confirm_stop_request(app, grace=0.05, poll=0.01) is True
    assert app.fields["phase"] == "stopping"


def test_a_non_interactive_run_stops_without_any_grace(monkeypatch, tmp_path):
    """Automation (`touch stop` from a script, a piped run) must not be slowed."""
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(cyclecore, "STOP_FILE", str(sentinel))
    started = time.monotonic()

    assert cyclecore.confirm_stop_request(None, grace=60) is True
    assert cyclecore.confirm_stop_request(
        sl.StatusApp(enabled=False), grace=60) is True
    assert time.monotonic() - started < 1.0


def test_a_sentinel_this_run_did_not_write_stops_it_at_once(monkeypatch, tmp_path):
    """`touch stop` from a script: the grace waits for a key nobody will press."""
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(cyclecore, "STOP_FILE", str(sentinel))
    external = _FakeApp()
    external.stop_requested_here = False    # written by somebody else
    started = time.monotonic()

    assert cyclecore.confirm_stop_request(external, grace=1.0, poll=0.01) is True
    assert time.monotonic() - started < 0.5
    assert external.notes == []             # no countdown was ever announced


def test_the_stop_key_is_what_marks_the_sentinel_as_ours(tmp_path):
    """The grace is keyed on `s`, so the flag must follow the key, not the file."""
    sentinel = tmp_path / "stop"
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))

    assert app.stop_requested_here is False
    sentinel.write_text("", encoding="utf-8")       # an external `touch stop`
    app.status.update(stop_pending=True)
    assert app.stop_requested_here is False

    app.handle_event(sl.Key("s"))                   # toggle: removes it
    app.handle_event(sl.Key("s"))                   # and writes it as ours
    assert sentinel.exists() and app.stop_requested_here is True

    app.handle_event(sl.Key("s"))
    assert app.stop_requested_here is False


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


def test_a_knob_can_stay_editable_without_taking_a_row_field():
    box = {"max_runs": 5}
    registry = sl.SettingsRegistry()
    hidden = registry.add(sl.NumberSetting(
        "max-runs", "--max-runs", lambda: box["max_runs"],
        lambda v: box.__setitem__("max_runs", int(v)), show_in_status=False))

    assert registry.status_entries() == []          # no field of its own
    assert registry.get("max-runs") is hidden       # but still an editable knob
    hidden.set(9)
    assert registry.overrides() == {"--max-runs": "9"}   # …and reproducible


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


def _usage(session=(43.0, NOW + 600), week=(61.0, None), sonnet=(None, None)):
    from llm_loop.usage import Usage, UsageReading

    return Usage(UsageReading(*session), UsageReading(*week),
                 UsageReading(*sonnet), [])


def test_quota_rows_follow_the_provider_and_the_policy_only_adds_its_half():
    """The row set is the account's windows; a rule contributes its ceiling to
    the one it watches and says nothing about the others."""
    from llm_loop.limits import LimitPolicy, SessionLimit

    source = _FakeUsageSource(_usage())

    rows = sl.quota_rows(source, LimitPolicy([SessionLimit(80)]), now=NOW)

    assert rows == [sl.QuotaRow("session", 43.0, NOW + 600, "ceil 80%"),
                    sl.QuotaRow("week", 61.0, None, "")]
    assert source.reads == 1


def test_both_windows_are_shown_without_any_policy_at_all():
    rows = sl.quota_rows(_FakeUsageSource(_usage()), None, now=NOW)

    assert [row.label for row in rows] == ["session", "week"]
    assert [row.policy for row in rows] == ["", ""]


def test_an_unreported_window_is_dropped_unless_a_rule_watches_it():
    """A plan without a Sonnet-only week has nothing to say about it — but if the
    run is gated on it, its absence is exactly what the reader needs to see."""
    from llm_loop.limits import LimitPolicy, WeeklyLimit

    source = _FakeUsageSource(_usage())

    assert [r.label for r in sl.quota_rows(source, now=NOW)] == ["session", "week"]

    policy = LimitPolicy([WeeklyLimit(90, sonnet_only=True)])
    rows = sl.quota_rows(source, policy, now=NOW)

    assert [r.label for r in rows] == ["session", "week", "week/sonnet"]
    assert rows[-1] == sl.QuotaRow("week/sonnet", None, None, "ceil 90%")


def test_a_rule_on_a_window_the_provider_never_reports_still_gets_a_row():
    """A custom rule may watch something outside the snapshot's fields; it keeps
    the old rule-driven row rather than disappearing."""
    from llm_loop.limits import LimitPolicy, LimitRule
    from llm_loop.usage import UsageReading

    class _Custom(LimitRule):
        quota = "burst"
        label = "Current burst"

        def reading(self, usage):
            return UsageReading(12.0, NOW + 60)

        def ceiling(self, reading, now):
            return 50.0

    rows = sl.quota_rows(_FakeUsageSource(_usage()), LimitPolicy([_Custom()]),
                         now=NOW)

    assert rows[-1] == sl.QuotaRow("burst", 12.0, NOW + 60, "ceil 50%")


def test_a_rule_may_say_more_than_a_ceiling():
    from llm_loop.limits import LimitPolicy, SessionLimit

    class _Chatty(SessionLimit):
        def status(self, reading, now):
            return f"ceil {self.ceiling(reading, now):.0f}% (night)"

    rows = sl.quota_rows(_FakeUsageSource(_usage()), LimitPolicy([_Chatty(80)]),
                         now=NOW)

    assert rows[0].policy == "ceil 80% (night)"


def test_a_raising_rule_costs_its_own_half_and_nothing_else():
    """Third-party `status()` runs on the repaint path; it must not be able to
    blank the provider's figures."""
    from llm_loop.limits import LimitPolicy, SessionLimit

    class _Broken(SessionLimit):
        def status(self, reading, now):
            raise RuntimeError("bad rule")

    rows = sl.quota_rows(_FakeUsageSource(_usage()), LimitPolicy([_Broken(80)]),
                         now=NOW)

    assert rows == [sl.QuotaRow("session", 43.0, NOW + 600, ""),
                    sl.QuotaRow("week", 61.0, None, "")]


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
    from llm_loop import cyclecore
    from llm_loop.cyclecore import AgentCommand, Driver

    class _OneShot(Driver):
        def next_command(self):
            return AgentCommand("do the thing, carefully", "", "the-thing")

    args = type("NS", (), {})()
    args.max = None
    args.dry_run = True
    args.raw = False
    args.start_in = None
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


# --- the mirror log must never see an escape ----------------------------------


class _CollectingHandler(logging.Handler):
    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        self.sink.append(record.getMessage())


def test_a_default_terminal_targets_the_real_stream_not_the_tee(monkeypatch):
    from llm_loop import cyclecore

    console = _FakeStream(True)
    monkeypatch.setattr(cyclecore, "_real_stream", lambda: console)

    assert sl.Terminal()._stream is console


def test_the_pinned_rows_never_reach_the_mirror_log(monkeypatch):
    """The feature's #1 non-negotiable: cursor bytes in the log corrupt the run
    record `--cost` parses, and every other test injects its own stream."""
    from llm_loop import cyclecore

    console = _FakeStream(True)
    logged = []
    logger = logging.getLogger("pytest-statusline-tee")
    logger.handlers = [_CollectingHandler(logged)]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    monkeypatch.setattr(sys, "stdout", cyclecore._TeeToLog(console, logger))
    monkeypatch.setattr(sl, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(sl.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))

    app = sl.StatusApp(input_source=sl.NullInputSource(), refresh=60)
    assert app.terminal._stream is console      # the tee was unwrapped
    with app:
        app.update(iteration=7, phase="running")
        print("ordinary output")                # still goes through the tee

    assert "\x1b" in console.getvalue() and "iter 7" in console.getvalue()
    assert logged == ["ordinary output"]
    assert all("\x1b" not in line for line in logged)


# --- teardown -----------------------------------------------------------------


@pytest.mark.parametrize("blow_up", [RuntimeError("boom"), SystemExit(3),
                                     KeyboardInterrupt()])
def test_every_way_out_releases_the_region(monkeypatch, blow_up):
    """Disabling is not enough — the region has to be reset on the way out, or
    the shell keeps scrolling inside it after the process is gone."""
    stream = _FakeStream(True)
    monkeypatch.setattr(sl, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(sl.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))
    app = sl.StatusApp(terminal=sl.terminal_for(stream),
                       input_source=sl.NullInputSource(), refresh=60)

    with pytest.raises(type(blow_up)):
        with app:
            app.update(iteration=1, phase="running")
            raise blow_up

    assert "\x1b[r" in stream.getvalue()        # DECSTBM reset, not just disabled
    assert app.terminal._rows == 0
    assert app.enabled is False


def test_a_signal_restores_the_terminal_and_still_lets_the_kill_through():
    """A signal unwinds no `finally` of ours, and the key reader is a daemon —
    without a handler the region stays pinned after the process is gone."""
    import signal

    stream = _FakeStream(True)
    app = sl.StatusApp(terminal=sl.Terminal(stream),
                       input_source=sl.NullInputSource(), refresh=60)
    chained = []
    previous = signal.signal(signal.SIGTERM, lambda *args: chained.append(args))
    try:
        with app:
            # == not `is`: a bound method is a fresh object on every attribute read
            assert signal.getsignal(signal.SIGTERM) == app._signal_restore
            app._signal_restore(signal.SIGTERM, None)
            assert app.terminal._rows == 0          # the region was given back
            assert chained                          # and the kill still happens
        assert signal.getsignal(signal.SIGTERM) != app._signal_restore
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_a_resize_the_region_cannot_satisfy_stops_painting(monkeypatch):
    """reserve() refuses without touching its geometry, so ignoring its answer
    left the rows being painted at the OLD coordinates."""
    stream = _FakeStream(True)
    size = [(100, 30)]
    monkeypatch.setattr(sl, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(sl.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size(size[0]))
    app = sl.StatusApp(terminal=sl.terminal_for(stream),
                       input_source=sl.NullInputSource(), refresh=60)

    with app:
        assert app.enabled is True
        assert "\x1b[27;1H" in stream.getvalue()     # 5 rows at the 30-line size
        size[0] = (100, 6)                           # no room for the region now
        mark = len(stream.getvalue())
        app.handle_event(sl.Resize(100, 6))
        app.update(iteration=99)

        assert app.enabled is False
        tail = stream.getvalue()[mark:]
        assert "iter" not in tail        # no row painted into the smaller screen
        assert "\x1b[r" in tail          # the region was given back instead


# --- concurrent writers (the net under wave 3's N workers) ---------------------


def test_concurrent_job_updates_never_tear_a_snapshot():
    status = sl.LoopStatus(jobs=[sl.Job(i) for i in range(1, 5)])
    stop = threading.Event()
    errors = []

    def churn(job):
        try:
            while not stop.is_set():
                job.start(item=f"item-{job.job_id}.md", model="opus")
                job.finish()
        except Exception as exc:              # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=churn, args=(job,), daemon=True)
               for job in status.jobs]
    for thread in threads:
        thread.start()
    try:
        for _ in range(300):
            frozen = status.snapshot()
            assert [j.job_id for j in frozen.jobs] == [1, 2, 3, 4]
            for job in frozen.jobs:
                # A painted row is ONE iteration, never half of two.
                assert (job.started_at > 0) is job.running
                assert job.item in ("", f"item-{job.job_id}.md")
            sl.render_rows(status, 120)
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2)

    assert not errors


# --- the loop's wiring: live caps, primed quotas, no forced polls ---------------


class _LiveTerminal(sl.Terminal):
    """Active from reserve() on, with no screen behind it.

    What these tests are about happens on either side of start() (push_quotas is
    silent until the app is enabled), which a NullTerminal — off always, and
    short-circuited by start() — cannot express and a real Terminal needs a tty
    for.
    """

    def __init__(self):
        super().__init__(stream=io.StringIO())
        self._on = False

    @property
    def active(self):
        return self._on

    def size(self):
        return (120, 30)

    def reserve(self, rows):
        self._on = True
        return True

    def paint(self, lines, *, reassert=False):
        return True

    def release(self):
        self._on = False


class _CountingSource:
    """A UsageSource that never leaves the process."""

    def __init__(self, percent=5.0):
        from llm_loop.usage import Usage, UsageReading

        reading = UsageReading(percent, NOW + 3600)
        self.usage = Usage(reading, reading, UsageReading(None, None),
                           [f"Current session: {percent:.0f}% used"])
        self.reads = []

    def get_usage(self, cache_value=True):
        self.reads.append(cache_value)
        return self.usage

    def invalidate(self):
        pass


def _run_with_status(monkeypatch, tmp_path, driver, *, on_app=None,
                     progress=None, **arg_fields):
    """run_loop with a live (screenless) status line; returns (app, source).

    `progress` stands in for a wrapper making several runner calls in one
    process; left None, the call is the whole invocation.
    """
    from llm_loop import cyclecore

    source = _CountingSource()
    monkeypatch.setattr(cyclecore, "usage_source_for", lambda provider: source)
    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial: 0)
    made = {}
    real_app_class = sl.StatusApp        # captured before the patch below

    def _app(**kwargs):
        kwargs.pop("enabled", None)
        app = real_app_class(terminal=_LiveTerminal(),
                             input_source=sl.NullInputSource(), refresh=60,
                             **kwargs)
        made["app"] = app
        if on_app is not None:
            on_app(app)
        return app

    monkeypatch.setattr(sl, "StatusApp", _app)

    args = type("NS", (), {})()
    args.max = 1
    args.dry_run = False
    args.raw = False
    args.start_in = None
    args.git_push = "none"
    args.project_dir = str(tmp_path)
    args.cost = False
    args.no_statusline = False
    for name, value in arg_fields.items():
        setattr(args, name, value)

    previous = cyclecore.project_dir()
    try:
        cyclecore.run_loop(driver, args, app_name="pytest-statusline",
                           setup_logging=False, wait_on_start=False,
                           progress=progress)
    finally:
        cyclecore.set_project_root(previous)
    return made["app"], source


def test_a_bounded_run_shows_the_quotas_and_never_polls_for_them(monkeypatch,
                                                                 tmp_path):
    """Priming before `with app:` published nothing (push_quotas needs an enabled
    app), and a `-m N` run has no later limit check to publish them either."""
    from llm_loop.cyclecore import AgentCommand, Driver

    class _OneShot(Driver):
        def next_command(self):
            return AgentCommand("do the thing", "", "the-thing")

    app, _source = _run_with_status(monkeypatch, tmp_path, _OneShot())

    assert app.status.quotas, "the provider's figures never reached the row"
    # -m N deliberately skips the usage machinery: no forced-fresh poll either.
    assert app._services == []


def test_an_unbounded_run_keeps_the_quota_figures_refreshed(monkeypatch, tmp_path):
    from llm_loop.cyclecore import Driver

    class _NoWork(Driver):
        def next_command(self):
            return None

    app, _source = _run_with_status(monkeypatch, tmp_path, _NoWork(), max=None)

    assert [type(s).__name__ for s in app._services] == ["QuotaRefresher"]
    assert app.status.quotas


def test_the_iteration_cap_is_read_live_from_the_settings_registry(monkeypatch,
                                                                   tmp_path):
    """Wave 2 edits --max-runs while the run goes, so the loop must not be
    comparing against a local it snapshotted at startup."""
    from llm_loop.cyclecore import AgentCommand, Driver

    class _RaisesItsOwnCap(Driver):
        app = None
        calls = 0

        def next_command(self):
            self.calls += 1
            if self.calls == 1:
                self.app.settings.get("max-runs").set(2)
            return AgentCommand("do the thing", "", f"item-{self.calls}")

    driver = _RaisesItsOwnCap()
    app, _source = _run_with_status(monkeypatch, tmp_path, driver,
                                    on_app=lambda a: setattr(driver, "app", a))

    assert driver.calls == 2                          # the raised cap took effect
    # An edited cap shows up as the counter's denominator — the one place it is
    # shown at all, since a field of its own would say the same number twice.
    assert app.status.max_iterations == 2
    assert "max-runs" not in dict(app.status.script_limits)
    assert app.settings.get("max-runs").get() == 2     # still an editable knob


def _counts_down(total):
    """A non-list Driver with `total` units of work and no queue behind it."""
    from llm_loop.cyclecore import AgentCommand, Driver

    class _CountsDown(Driver):
        provider = "codex"

        def __init__(self):
            self.calls = 0

        def model(self):
            return "gpt-5.6-terra"

        def next_command(self):
            if self.calls >= total:
                return None
            self.calls += 1
            return AgentCommand("do the thing", self.model(),
                                f"item-{self.calls}", self.provider)

    return _CountsDown()


def test_a_driver_with_no_list_gets_no_invented_denominator(monkeypatch, tmp_path):
    """Only a list says how much work a run has. Without one an uncapped run
    counts, and says nothing it cannot know."""
    from llm_loop import cyclecore

    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda cmd, provider, raw, partial, prompt: 0)
    app, _source = _run_with_status(monkeypatch, tmp_path, _counts_down(3),
                                    max=None, provider="codex")

    assert app.status.iteration == 3
    assert app.status.max_iterations is None
    summary = app.render(width=200)[1]
    assert "iter 3" in summary and "iter 3/" not in summary


def test_a_capped_run_with_no_list_shows_the_cap_it_was_given(monkeypatch,
                                                              tmp_path):
    """--max is the whole denominator here — there is no queue to be smaller."""
    from llm_loop import cyclecore

    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda cmd, provider, raw, partial, prompt: 0)
    app, _source = _run_with_status(monkeypatch, tmp_path, _counts_down(10),
                                    max=2, provider="codex")

    assert (app.status.iteration, app.status.max_iterations) == (2, 2)
    assert "iter 2/2" in app.render(width=200)[1]


def _list_driver(items):
    """A ListFileDriver whose list lives in memory (no files, no provider)."""
    from llm_loop.drivers import ListFileDriver

    class _MemList(ListFileDriver):
        provider = "codex"
        target_suffix = ".out.md"
        pick_order = "list"          # deterministic: the tests name the order

        def __init__(self):
            super().__init__()
            self.items = list(items)

        def prompt(self, source, target):
            return "do it"

        def model(self):
            return "gpt-5.6-terra"

        def pending_lines(self):
            return list(self.items)

        def strike(self, line):
            if line in self.items:
                self.items.remove(line)
                return True
            return False

    return _MemList()


def test_the_sequential_loop_counts_items_struck_not_iterations(monkeypatch,
                                                                tmp_path):
    """The list is the total in BOTH runners, so a retried item is not progress —
    while the job row keeps counting the iterations it actually ran."""
    from llm_loop import cyclecore

    codes = [1, 0, 0, 0]        # the first item fails once and comes back
    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda cmd, provider, raw, partial, prompt: codes.pop(0))
    driver = _list_driver([f"products/f{i}.md" for i in range(3)])
    app, _source = _run_with_status(monkeypatch, tmp_path, driver, max=None,
                                    provider="codex")

    assert (app.status.iteration, app.status.max_iterations) == (3, 3)
    assert app.status.jobs[0].iteration == 4      # attempts, not items
    assert "iter 3/3" in app.render(width=200)[1]


def test_a_second_runner_call_resumes_the_job_row(monkeypatch, tmp_path):
    """Both runners share the rule: a Job belongs to the invocation, not to the
    call that displayed it — and a per-call cap is not the invocation's."""
    from llm_loop import cyclecore

    monkeypatch.setattr(cyclecore, "usage_source_for",
                        lambda provider: _CountingSource())
    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda cmd, provider, raw, partial, prompt: 0)
    made = []
    real_app_class = sl.StatusApp        # captured before the patch below

    def _app(**kwargs):
        kwargs.pop("enabled", None)
        app = real_app_class(terminal=_LiveTerminal(),
                             input_source=sl.NullInputSource(), refresh=60,
                             **kwargs)
        made.append(app)
        return app

    monkeypatch.setattr(sl, "StatusApp", _app)

    args = type("NS", (), {})()
    for name, value in dict(max=2, dry_run=False, raw=False, start_in=None,
                            git_push="none", cost=False,
                            no_statusline=False, provider="codex",
                            project_dir=str(tmp_path)).items():
        setattr(args, name, value)

    progress = sl.InvocationProgress()
    previous = cyclecore.project_dir()
    try:
        for _call in (1, 2):        # what a periodic wrapper does, twice
            cyclecore.run_loop(_counts_down(2), args,
                               app_name="pytest-statusline",
                               setup_logging=False, wait_on_start=False,
                               progress=progress)
    finally:
        cyclecore.set_project_root(previous)

    first, second = made
    assert first.status.jobs[0] is second.status.jobs[0]
    assert (first.status.iteration, second.status.iteration) == (2, 4)
    assert second.status.jobs[0].iteration == 4
    # The 2 each call was allowed to run sized a batch, not the run: with no
    # invocation-level --max there is nothing honest to put after the slash.
    assert second.status.max_iterations is None


def test_a_setting_flag_is_checked_against_the_command_line_table():
    """A typo must fail at registration, not when `c` renders the command line."""
    with pytest.raises(KeyError):
        sl.SettingsRegistry().add(sl.NumberSetting(
            "max-runs", "--max-run", lambda: 1, lambda v: None))


# --- the background refresher stays out of the stream --------------------------


def test_a_background_quota_poll_notes_its_failure_instead_of_printing(capsys):
    """A daemon thread printing lands mid-stream and in the mirror log — possibly
    inside a rich Live block."""
    class _NoisySource:
        def get_usage(self, cache_value=True):
            print("  · no usage figures: the stored OAuth token was rejected (401)")
            raise RuntimeError("no figures")     # quota_rows swallows this

    app = sl.StatusApp(terminal=_LiveTerminal(),
                       input_source=sl.NullInputSource(), refresh=60)
    with app:
        refresher = sl.QuotaRefresher(app, _NoisySource(), interval=0.01)
        refresher.start()
        deadline = time.time() + 5
        while not app.status.note and time.time() < deadline:
            time.sleep(0.01)
        refresher.stop()

    assert "quota refresh" in app.status.note and "401" in app.status.note
    assert capsys.readouterr().out == ""          # nothing reached the stream


def test_capture_only_diverts_the_thread_that_asked_for_it(capsys):
    other = threading.Event()

    def elsewhere():
        print("from another thread")
        other.set()

    with sl.capture_stdout_here() as chunks:
        print("mine")
        threading.Thread(target=elsewhere, daemon=True).start()
        other.wait(5)

    assert "".join(chunks) == "mine\n"
    assert capsys.readouterr().out == "from another thread\n"
    assert not isinstance(sys.stdout, sl._ThreadScopedCapture)   # uninstalled


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
