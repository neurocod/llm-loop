"""Tests for the pinned status line.

Everything here exercises the PURE renderer (rows -> strings) or the controller
driven by synthetic events.

Row CONTENT is asserted as text: that is what a human reads, and how a row gets
painted differs per emulator. Escapes are asserted as BYTES only where the bytes
are themselves the observable — the title escape and the DECSTBM reset (a
terminal either accepts them or does not, and there is nothing above them to
look at), the top row the region reserves (the geometry arithmetic), and whether
any ESC at all reaches the mirror log (it must not). The title escape is spelled
out literally rather than built from `tio.TITLE_SET`, because an expectation
built from the constant mutates along with it and pins nothing.
"""

import io
import logging
import os
import sys
import threading
import time

import pytest

from llm_loop import console, cyclecore, projectroot, stopchannel, textwidth
from llm_loop import statusline as sl
from llm_loop import termio as tio


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
            assert textwidth.cell_width(line) <= width
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
    key = stopchannel.StopSource.KEY.value
    assert "STOP pending" not in sl.render_rows(
        sequential_status(), 200, now=NOW)[1]
    pending = sl.render_rows(sequential_status(stop_pending=key), 200,
                             now=NOW)[1]
    assert f"{sl.STOP_GLYPH} STOP pending — press s to cancel" in pending

    # …but not twice on one line: once the phase is "stopping" the row already
    # opens with the same glyph.
    status = sequential_status(stop_pending=key)
    status.update(phase="stopping")
    stopping = sl.render_rows(status, 200, now=NOW)[1]
    assert stopping.count(sl.STOP_GLYPH) == 1
    assert "STOP pending — press s to cancel" in stopping


def test_a_stop_file_marker_does_not_offer_a_cancel_the_key_cannot_do():
    """`s` withdraws only its own request, so the row must not promise otherwise."""
    row = sl.render_rows(
        sequential_status(stop_pending=stopchannel.StopSource.FILE.value),
        200, now=NOW)[1]

    assert "STOP pending — stop file present" in row
    assert "press s to cancel" not in row


def test_the_pause_marker_appears_only_while_paused():
    assert "PAUS" not in sl.render_rows(sequential_status(), 200, now=NOW)[1]

    row = sl.render_rows(sequential_status(paused=True), 200, now=NOW)[1]

    assert "PAUSING — the iteration in flight finishes first" in row


def test_a_request_made_mid_iteration_does_not_yet_read_as_held():
    """The safety line of the whole feature: the files are touchable only once
    the run is standing still, and the row must not say so a moment early — the
    note explaining it expires, the marker does not."""
    running = sequential_status(paused=True)            # the fixture job is busy
    held = sequential_status(paused=True)
    held.jobs[0].finish()

    running_row = sl.render_rows(running, 200, now=NOW)[1]
    held_row = sl.render_rows(held, 200, now=NOW)[1]

    assert "PAUSING" in running_row and "PAUSED" not in running_row
    assert running_row.lstrip().startswith("⟳")         # still running, and says so
    assert "PAUSED — press p to resume" in held_row
    # Held, the glyph outranks the phase — one ⏸, at the head of the row.
    assert held_row.lstrip().startswith(sl.PAUSE_GLYPH)
    assert held_row.count(sl.PAUSE_GLYPH) == 1


def test_a_rate_limit_hold_wears_the_glyph_without_claiming_to_be_the_p_key():
    """Both stand still and both show ⏸ — only the marker says which is which,
    and a run waiting out a window must not read as one a human held."""
    row = sl.render_rows(sequential_status(phase="paused"), 200, now=NOW)[1]

    assert row.lstrip().startswith(sl.PAUSE_GLYPH)
    assert "PAUSED" not in row


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
        assert textwidth.cell_width(rule) == width


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

    app.handle_event(tio.Key("x"))
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
    app.handle_event(tio.Key("z"))   # dispatch must not reach it either


def test_help_key_writes_the_full_key_list_into_the_note_row():
    app = sl.StatusApp(enabled=False)

    app.handle_event(tio.Key("?"))

    assert "s stop" in app.status.note and "h/? help" in app.status.note


def test_an_unknown_key_points_at_the_help_key():
    app = sl.StatusApp(enabled=False)

    app.handle_event(tio.Key("q"))

    assert "press h for help" in app.status.note


# --- the stop key --------------------------------------------------------------


def test_stop_key_requests_a_stop_and_pressing_it_again_cancels(tmp_path):
    sentinel = tmp_path / "stop"
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))

    app.handle_event(tio.Key("s"))
    assert app.stop_requested_here is True
    assert app.status.stop_pending == stopchannel.StopSource.KEY.value
    assert "cancel stop" in app.render(width=200, now=NOW)[-2]

    app.handle_event(tio.Key("s"))
    assert app.stop_requested_here is False and app.status.stop_pending == ""


def test_the_stop_key_writes_no_sentinel_so_a_neighbouring_run_keeps_going(
    tmp_path, monkeypatch
):
    """Why the key is a flag and not the file: several loops share one project
    root, and a file stops every one of them."""
    sentinel = tmp_path / "stop"
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    one = sl.StatusApp(enabled=False, stop_file=str(sentinel))
    two = sl.StatusApp(enabled=False, stop_file=str(sentinel))

    one.handle_event(tio.Key("s"))

    assert not sentinel.exists(), "the key press left a cross-process sentinel"
    assert stopchannel.pending_stop(one) is stopchannel.StopSource.KEY
    assert stopchannel.pending_stop(two) is None


def test_the_sentinel_a_run_watches_is_the_one_its_row_reports(tmp_path):
    """An app given an explicit stop_file must not have the runner obeying the
    project root's file while the row talks about another one."""
    sentinel = tmp_path / "elsewhere-stop"
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))
    assert stopchannel.stop_file_for(app) == str(sentinel)
    assert stopchannel.pending_stop(app) is None

    sentinel.write_text("", encoding="utf-8")

    assert stopchannel.pending_stop(app) is stopchannel.StopSource.FILE
    assert stopchannel.latched_stop(app) is stopchannel.StopSource.FILE


def test_what_may_be_cancelled_and_what_must_be_consumed_are_different(
    tmp_path, monkeypatch
):
    """Both channels up at once. The key is the pending request (a human can
    still take it back), but the FILE is what a run stopping now must consume —
    both runners latch on `latched_stop` for exactly this case."""
    sentinel = tmp_path / "stop"
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))
    app.handle_event(tio.Key("s"))
    sentinel.write_text("", encoding="utf-8")

    assert stopchannel.pending_stop(app) is stopchannel.StopSource.KEY
    assert stopchannel.latched_stop(app) is stopchannel.StopSource.FILE


def test_the_stop_file_remains_the_cross_process_channel(tmp_path, monkeypatch):
    """The fallback the key deliberately does not use: a file stops every run
    watching the root, and `s` neither writes nor withdraws it."""
    sentinel = tmp_path / "stop"
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))
    sentinel.write_text("", encoding="utf-8")           # another run's `touch stop`

    assert stopchannel.pending_stop(app) is stopchannel.StopSource.FILE
    # `s` is offered as "stop", not "cancel stop": this request is not ours.
    assert app.action_for("s").help_text(app) == "stop"

    app.handle_event(tio.Key("s"))                       # …and then pressed twice
    app.handle_event(tio.Key("s"))
    assert sentinel.exists(), "the s key removed somebody else's stop file"
    assert stopchannel.pending_stop(app) is stopchannel.StopSource.FILE


def test_stop_file_defaults_to_the_engine_sentinel(monkeypatch, tmp_path):
    """A StatusApp built without an explicit `stop_file=` watches the sentinel of
    whatever root the engine is pointed at — derived on read, so moving the root
    moves the row without anybody pushing it a new value."""
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))

    assert sl.StatusApp(enabled=False).stop_file == str(tmp_path / "stop")


# --- the pause key -------------------------------------------------------------


def test_pause_key_holds_the_run_and_pressing_it_again_resumes():
    app = sl.StatusApp(enabled=False)

    app.handle_event(tio.Key("p"))
    assert app.paused is True and app.status.paused is True
    assert stopchannel.pause_requested(app) is True
    assert "p resume" in app.render(width=200, now=NOW)[-2]

    app.handle_event(tio.Key("p"))
    assert app.paused is False and app.status.paused is False
    assert stopchannel.pause_requested(app) is False
    assert "p pause" in app.render(width=200, now=NOW)[-2]


def test_pausing_writes_nothing_to_disk_and_leaves_the_stop_channels_alone(
    tmp_path, monkeypatch
):
    """A pause is one run's own, like `s` — and it is not a stop: a run holding
    on `p` must not report a stop pending to anything that asks."""
    sentinel = tmp_path / "stop"
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))

    app.handle_event(tio.Key("p"))

    assert not sentinel.exists()
    assert stopchannel.pending_stop(app) is None
    assert app.stop_requested_here is False


def test_a_run_with_no_status_line_is_never_paused():
    """Piped output, CI, --no-statusline: nobody can press the key, and both
    runners must not have to ask whether there is a status line at all."""
    assert stopchannel.pause_requested(None) is False
    assert stopchannel.pause_requested(sl.StatusApp(enabled=False)) is False
    assert stopchannel.wait_while_paused(None) == 0.0


class _PausedApp:
    """An app whose `p` key is released after `polls` reads of the flag."""

    enabled = True

    def __init__(self, polls=3, stop_after=None):
        self.polls = 0
        self.limit = polls
        self.stop_after = stop_after    # reads after which `s` is pressed too
        self.fields = {}
        self.notes = []

    @property
    def paused(self):
        self.polls += 1
        return self.polls <= self.limit

    @property
    def stop_requested_here(self):
        return self.stop_after is not None and self.polls > self.stop_after

    def update(self, **fields):
        self.fields.update(fields)

    def note(self, text):
        self.notes.append(text)


def test_the_hold_lasts_exactly_as_long_as_the_key_is_up():
    app = _PausedApp(polls=3)

    held = stopchannel.wait_while_paused(app, poll=0.01)

    assert held > 0.0
    assert app.polls == 4          # three "still paused", then the release


def test_a_stop_pressed_during_the_hold_releases_it_at_once(monkeypatch, tmp_path):
    """The hold has no timer, so `s` must be answered by it and not only by
    whatever the run does next — and the decision stays with the loop head."""
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = _PausedApp(polls=1000, stop_after=2)

    held = stopchannel.wait_while_paused(
        app, should_stop=lambda: stopchannel.pending_stop(app) is not None,
        poll=0.01)

    assert held > 0.0
    assert app.polls < 10          # left on the stop, not on the pause
    assert app.paused is True      # …and the pause itself is untouched


# --- the cancel grace (stopchannel.confirm_stop_request) -------------------------


class _FakeApp:
    """An interactive app whose note() can act — the seam a test needs to make
    "the user pressed s again" happen at a defined moment."""

    enabled = True

    def __init__(self, on_note=None, requested_here=True):
        self.notes = []
        self.fields = {}
        self.stop_requested_here = requested_here   # as if `s` was pressed here
        self._on_note = on_note

    def update(self, **fields):
        self.fields.update(fields)

    def note(self, text):
        self.notes.append(text)
        if self._on_note is not None:
            self._on_note()


def test_a_stop_request_cancelled_inside_the_grace_leaves_no_trace(monkeypatch, tmp_path):
    from llm_loop import cyclecore

    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = _FakeApp()
    app._on_note = lambda: setattr(app, "stop_requested_here", False)  # `s` again

    assert stopchannel.confirm_stop_request(app, grace=5.0, poll=0.01) is False
    assert app.fields["phase"] == "idle" and app.fields["stop_pending"] == ""
    assert "stop cancelled" in app.notes[-1]
    assert "press s to cancel" in app.notes[0]


def test_a_stop_file_arriving_mid_grace_outlives_the_cancel(monkeypatch, tmp_path):
    """Taking back your own `s` cannot withdraw a request that was never yours."""
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = _FakeApp()

    def cancel_but_a_file_appears():
        app.stop_requested_here = False
        sentinel.write_text("", encoding="utf-8")

    app._on_note = cancel_but_a_file_appears

    assert stopchannel.confirm_stop_request(app, grace=0.2, poll=0.01) is True


def test_a_stop_request_still_pending_after_the_grace_stops_the_run(monkeypatch, tmp_path):
    from llm_loop import cyclecore

    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = _FakeApp()

    assert stopchannel.confirm_stop_request(app, grace=0.05, poll=0.01) is True
    assert app.fields["phase"] == "stopping"


def test_a_non_interactive_run_stops_without_any_grace(monkeypatch, tmp_path):
    """Automation (`touch stop` from a script, a piped run) must not be slowed."""
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    started = time.monotonic()

    assert stopchannel.confirm_stop_request(None, grace=60) is True
    assert stopchannel.confirm_stop_request(
        sl.StatusApp(enabled=False), grace=60) is True
    assert time.monotonic() - started < 1.0


def test_a_stop_this_run_did_not_request_stops_it_at_once(monkeypatch, tmp_path):
    """`touch stop` from a script: the grace waits for a key nobody will press."""
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    sentinel.write_text("", encoding="utf-8")
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    external = _FakeApp(requested_here=False)   # requested by somebody else
    started = time.monotonic()

    assert stopchannel.confirm_stop_request(external, grace=1.0, poll=0.01) is True
    assert time.monotonic() - started < 0.5
    assert external.notes == []             # no countdown was ever announced


def test_the_stop_key_is_what_marks_the_request_as_ours(tmp_path, monkeypatch):
    """The grace is keyed on `s`, so the flag must follow the key, not the file."""
    from llm_loop import cyclecore

    sentinel = tmp_path / "stop"
    monkeypatch.setattr(projectroot, "PROJECT_DIR", str(tmp_path))
    app = sl.StatusApp(enabled=False, stop_file=str(sentinel))

    assert app.stop_requested_here is False
    sentinel.write_text("", encoding="utf-8")       # an external `touch stop`
    assert app.stop_requested_here is False
    assert stopchannel.pending_stop(app) is stopchannel.StopSource.FILE

    app.handle_event(tio.Key("s"))                   # ours, on top of theirs
    assert app.stop_requested_here is True
    assert stopchannel.pending_stop(app) is stopchannel.StopSource.KEY

    app.handle_event(tio.Key("s"))                   # withdrawn — theirs remains
    assert app.stop_requested_here is False
    assert stopchannel.pending_stop(app) is stopchannel.StopSource.FILE


# --- disabled / no-TTY paths ---------------------------------------------------


class _FakeStream(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def test_a_non_tty_stream_gets_a_null_terminal():
    assert isinstance(tio.terminal_for(_FakeStream(False)), tio.NullTerminal)


def test_the_env_flag_disables_the_status_line(monkeypatch):
    monkeypatch.setenv(tio.ENV_FLAG, "0")

    assert isinstance(tio.terminal_for(_FakeStream(True)), tio.NullTerminal)


def test_a_disabled_app_writes_nothing_and_still_serves_its_api():
    stream = _FakeStream(True)   # a real TTY: only `enabled=False` disables us
    app = sl.StatusApp(terminal=tio.terminal_for(stream, enabled=False))

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
    monkeypatch.setattr(tio, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(tio.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))
    app = sl.StatusApp(terminal=tio.terminal_for(stream),
                       input_source=tio.NullInputSource(), refresh=60)

    with app:
        assert app.enabled is True
        app.update(iteration=4, phase="running")
        assert "iter 4" in stream.getvalue()

    assert app.enabled is False   # region released on the way out


def test_start_is_a_no_op_when_the_terminal_cannot_reserve(monkeypatch):
    """Any failure swaps in the Null terminal — the run must not care."""
    class Refusing(tio.Terminal):
        def __init__(self):
            super().__init__(stream=_FakeStream(True))

        def reserve(self, rows):
            raise RuntimeError("no region here")

    app = sl.StatusApp(terminal=Refusing())
    with app:
        app.update(phase="running")

    assert isinstance(app.terminal, tio.NullTerminal)


# --- window/tab title ----------------------------------------------------------


def test_the_title_is_the_summary_rows_first_field_then_the_running_items():
    """Literally the row's own words — that is the point of building it from the
    same Segment."""
    status = sequential_status()
    first_field = sl.render_rows(status, 200, now=NOW)[1].split(sl.SEPARATOR)[0]

    title = sl.title_text(status, now=NOW)

    assert title == "⟳ iter 12/40 · bmx-bike.md"
    assert title.startswith(first_field.strip())


def test_the_title_names_every_busy_job_and_no_idle_one():
    title = sl.title_text(parallel_status(3), now=NOW)

    assert title == "⟳ iter 7/40 · item-1.md · item-2.md"   # job 3 is idle


def test_an_idle_run_still_says_where_it_is():
    assert sl.title_text(sl.LoopStatus(iteration=4, max_iterations=9)) \
        == "· iter 4/9"


def test_control_characters_never_reach_the_title():
    """An item label is arbitrary text; a stray BEL would end the escape early
    and spill the rest of the title into the scrollback."""
    status = sequential_status()
    status.jobs[0].update(item="a\x07b\x1b[2Jc")

    assert "\x07" not in sl.title_text(status) and "\x1b" not in sl.title_text(status)


def test_the_title_is_written_on_change_only_and_given_back_on_exit(monkeypatch):
    stream = _FakeStream(True)
    monkeypatch.setattr(tio, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(tio.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))
    app = sl.StatusApp(terminal=tio.terminal_for(stream),
                       input_source=tio.NullInputSource(), refresh=60)

    # Spelled out rather than built from `tio.TITLE_SET`, which would mutate
    # along with the constant and leave its VALUE pinned by nothing: OSC 0 (title
    # AND icon/tab name in one escape) terminated by BEL (which every emulator
    # this runs under accepts, where some older ones refuse ST). Those bytes are
    # the whole feature; the reasoning for them lives above the constant.
    escape = "\x1b]0;⟳ iter 4/9 · garlic.md\x07"
    with app:
        app.update(iteration=4, max_iterations=9, phase="running")
        app.job(1).start(item="garlic.md", model="opus", now=NOW)
        app.update(phase="running")             # the item reaches the title here
        mark = len(stream.getvalue())
        app.update(note="anything")             # state the title does not carry
        assert "\x1b]0;" not in stream.getvalue()[mark:]   # so: no second write
        assert stream.getvalue().count(escape) == 1

    # The empty title, which is the documented way back to the profile's own.
    assert "\x1b]0;\x07" in stream.getvalue()   # the window gets its name back

    # A worker still finishing past a parallel Ctrl+C, or the quota refresher
    # whose join timed out: both paint after the teardown, and a name written
    # then is a name nothing is left to clear.
    mark = len(stream.getvalue())
    app.update(iteration=5)
    assert stream.getvalue()[mark:] == ""


def test_the_window_keeps_its_name_when_something_else_takes_it(monkeypatch):
    """The title is not ours alone, exactly as the scroll region is not (see
    `Terminal.paint`): a provider CLI or a pager can reset it with an escape,
    and on Windows a child process renames the window through SetConsoleTitle,
    which no escape of ours is answering. Without a re-assert on the periodic
    repaint the dedupe holds the window on that other name for the rest of the
    run — we already wrote ours, as far as we know.
    """
    stream = _FakeStream(True)
    monkeypatch.setattr(tio, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(tio.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))
    app = sl.StatusApp(terminal=tio.terminal_for(stream),
                       input_source=tio.NullInputSource(), refresh=0.01)

    escape = tio.TITLE_SET.format("⟳ iter 4/9 · garlic.md")
    with app:
        app.update(iteration=4, max_iterations=9, phase="running")
        app.job(1).start(item="garlic.md", model="opus", now=NOW)
        app.update(phase="running")
        # Nothing about the run changes from here on: only the repaint ticks.
        deadline = time.time() + 5
        while stream.getvalue().count(escape) < 2 and time.time() < deadline:
            time.sleep(0.02)

    assert stream.getvalue().count(escape) >= 2, \
        "the repaint never re-asserted the window title"


def test_disabling_mid_paint_cannot_leave_a_name_on_the_window(monkeypatch):
    """`_paint` reads `self.terminal`, then writes to it — and another thread
    may release that very terminal in between (a resize the screen cannot fit,
    any painting error). The released terminal has to refuse the write."""
    stream = _FakeStream(True)
    monkeypatch.setattr(tio, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(tio.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))
    terminal = tio.terminal_for(stream)
    app = sl.StatusApp(terminal=terminal, input_source=tio.NullInputSource(),
                       refresh=60)

    with app:
        app.update(iteration=8, phase="running")
        app.disable()                      # the app now holds a NullTerminal…
        mark = len(stream.getvalue())
        terminal.set_title("⟳ iter 9")     # …and the old one is asked anyway

    assert stream.getvalue()[mark:] == ""


def test_the_git_push_value_is_lit_like_a_healthy_figure():
    """Through `_script_settings`, not a hand-written field: the colouring is
    anchored on the knob's NAME, and this is what makes renaming it move the
    colour instead of quietly losing it."""
    settings = cyclecore._script_settings(cyclecore.RunSettings(), sl)
    line = sl.render_rows(sequential_status(script_limits=settings.status_entries()),
                          200, now=NOW)[1]

    assert "git-push each_hour" in line       # the plain row is unchanged
    assert f"git-push {sl._SGR['green']}each_hour{sl._SGR_RESET}" in sl.colorize(line)


def test_a_policy_word_is_coloured_only_where_it_is_the_git_push_value():
    """`none` is ordinary English — an item called none.md must stay plain."""
    status = sequential_status(script_limits=[("git-push", "none")])
    status.jobs[0].update(item="none")
    rows = sl.render_rows(status, 200, now=NOW)

    assert sl._SGR["green"] not in sl.colorize(rows[2])          # the job row
    assert f"git-push {sl._SGR['green']}none" in sl.colorize(rows[1])


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


def test_format_prompt_block_heads_the_prompt_and_keeps_it_verbatim():
    block = sl.format_prompt_block(job_id=2, label="garlic.md",
                                   prompt="line one\nline two\n", width=60)
    head, body, foot = block.split("\n")[0], block.split("\n")[1:-1], \
        block.split("\n")[-1]

    assert "job 2" in head and "garlic.md" in head and "18 chars" in head
    assert textwidth.cell_width(head) == 60 and foot == "─" * 60
    assert body == ["line one", "line two"]   # verbatim: still pasteable


def test_format_prompt_block_survives_an_empty_prompt():
    block = sl.format_prompt_block(job_id=1, label="", prompt="", width=60)

    assert "(no label)" in block and "0 chars" in block


# --- the dry run prints job 1's prompt ----------------------------------------


def test_a_dry_run_prints_the_prompt_block_for_job_one(tmp_path, capsys):
    """The joined `-p …` argv line hides the prompt; the block is what shows it."""
    from llm_loop import cyclecore
    from llm_loop.agentwork import AgentCommand, Driver

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

    previous = projectroot.project_dir()
    streams = (sys.stdout, sys.stderr)
    try:
        cyclecore.run_loop(_OneShot(), args, app_name="pytest-statusline",
                           setup_logging=False, wait_on_start=False)
    finally:
        projectroot.set_project_root(previous)
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
    screen = _FakeStream(True)
    monkeypatch.setattr(console, "real_stream", lambda: screen)

    assert tio.Terminal()._stream is screen


def test_the_pinned_rows_never_reach_the_mirror_log(monkeypatch):
    """The feature's #1 non-negotiable: cursor bytes in the log corrupt the run
    record `--cost` parses, and every other test injects its own stream."""
    screen = _FakeStream(True)
    logged = []
    logger = logging.getLogger("pytest-statusline-tee")
    logger.handlers = [_CollectingHandler(logged)]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    monkeypatch.setattr(sys, "stdout", console.TeeToLog(screen, logger))
    monkeypatch.setattr(tio, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(tio.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))

    app = sl.StatusApp(input_source=tio.NullInputSource(), refresh=60)
    assert app.terminal._stream is screen       # the tee was unwrapped
    with app:
        app.update(iteration=7, phase="running")
        print("ordinary output")                # still goes through the tee

    assert "\x1b" in screen.getvalue() and "iter 7" in screen.getvalue()
    assert logged == ["ordinary output"]
    assert all("\x1b" not in line for line in logged)


# --- teardown -----------------------------------------------------------------


@pytest.mark.parametrize("blow_up", [RuntimeError("boom"), SystemExit(3),
                                     KeyboardInterrupt()])
def test_every_way_out_releases_the_region(monkeypatch, blow_up):
    """Disabling is not enough — the region has to be reset on the way out, or
    the shell keeps scrolling inside it after the process is gone."""
    stream = _FakeStream(True)
    monkeypatch.setattr(tio, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(tio.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size((100, 30)))
    app = sl.StatusApp(terminal=tio.terminal_for(stream),
                       input_source=tio.NullInputSource(), refresh=60)

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
    app = sl.StatusApp(terminal=tio.Terminal(stream),
                       input_source=tio.NullInputSource(), refresh=60)
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
    monkeypatch.setattr(tio, "_enable_windows_vt", lambda s: True)
    monkeypatch.setattr(tio.shutil, "get_terminal_size",
                        lambda fallback=(0, 0): os.terminal_size(size[0]))
    app = sl.StatusApp(terminal=tio.terminal_for(stream),
                       input_source=tio.NullInputSource(), refresh=60)

    with app:
        assert app.enabled is True
        assert "\x1b[27;1H" in stream.getvalue()     # 5 rows at the 30-line size
        size[0] = (100, 6)                           # no room for the region now
        mark = len(stream.getvalue())
        app.handle_event(tio.Resize(100, 6))
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


class _LiveTerminal(tio.Terminal):
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
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)
    made = {}
    real_app_class = sl.StatusApp        # captured before the patch below

    def _app(**kwargs):
        kwargs.pop("enabled", None)
        app = real_app_class(terminal=_LiveTerminal(),
                             input_source=tio.NullInputSource(), refresh=60,
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

    previous = projectroot.project_dir()
    try:
        cyclecore.run_loop(driver, args, app_name="pytest-statusline",
                           setup_logging=False, wait_on_start=False,
                           progress=progress)
    finally:
        projectroot.set_project_root(previous)
    return made["app"], source


def test_a_bounded_run_shows_the_quotas_and_never_polls_for_them(monkeypatch,
                                                                 tmp_path):
    """Priming before `with app:` published nothing (push_quotas needs an enabled
    app), and a `-m N` run has no later limit check to publish them either."""
    from llm_loop.agentwork import AgentCommand, Driver

    class _OneShot(Driver):
        def next_command(self):
            return AgentCommand("do the thing", "", "the-thing")

    app, _source = _run_with_status(monkeypatch, tmp_path, _OneShot())

    assert app.status.quotas, "the provider's figures never reached the row"
    # -m N deliberately skips the usage machinery: no forced-fresh poll either.
    assert app._services == []


def test_an_unbounded_run_keeps_the_quota_figures_refreshed(monkeypatch, tmp_path):
    from llm_loop.agentwork import Driver

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
    from llm_loop.agentwork import AgentCommand, Driver

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


def test_a_paused_loop_holds_before_it_asks_the_driver_for_work(monkeypatch,
                                                                tmp_path):
    """The hold is only worth anything where it is: the state file the driver
    reads must be quiet while a human edits it, so a pause taken after
    `next_command()` would hold the run with the edit already missed."""
    from llm_loop import cyclecore
    from llm_loop.agentwork import AgentCommand, Driver

    class _TwoItems(Driver):
        calls = 0

        def next_command(self):
            self.calls += 1
            return AgentCommand("do the thing", "", f"item-{self.calls}")

    driver = _TwoItems()
    seen = []          # driver.calls at each read of the key, in order

    def pressed_after_the_first_iteration(app=None):
        seen.append(driver.calls)
        return 2 <= len(seen) <= 4      # up for three reads, then released

    monkeypatch.setattr(stopchannel, "pause_requested",
                        pressed_after_the_first_iteration)

    app, _source = _run_with_status(monkeypatch, tmp_path, driver, max=2)

    # Read once before iteration 1 (nothing to hold yet), then held at the
    # boundary after it — and for the whole hold the driver stayed on one call.
    assert seen[0] == 0
    assert seen[1:4] == [1, 1, 1]
    assert driver.calls == 2
    assert app.status.iteration == 2


def test_a_pause_pressed_at_the_usage_gate_holds_the_next_iteration(monkeypatch,
                                                                    tmp_path):
    """The gate is the longest hold in the engine — hours with nothing moving —
    so it is exactly where somebody reaches for `p`. The gate does not watch the
    pause, so the loop head has to re-read it: without that the window opened
    and a full iteration started under a row that already said PAUSED."""
    from llm_loop import cyclecore
    from llm_loop.agentwork import AgentCommand, Driver

    key = {"up": False, "reads": 0}

    def fake_pause_requested(app=None):
        if not key["up"]:
            return False
        key["reads"] += 1
        if key["reads"] > 3:
            key["up"] = False         # as if `p` were pressed a second time
        return key["up"]

    monkeypatch.setattr(stopchannel, "pause_requested", fake_pause_requested)

    class _PressesPAtTheGate:
        """A policy that never holds, but is where the key gets pressed."""

        rules = ()
        reached = False

        def describe(self):
            return "never holds"

        def log_snapshot(self, *a, **k):
            pass

        def rule_for(self, quota):
            return None

        def check_and_wait(self, source, session_start, note="",
                           cache_value=True, should_stop=None):
            self.reached = True
            key["up"] = True
            return False, session_start

    asked_while_paused = []

    class _OneItem(Driver):
        calls = 0

        def next_command(self):
            asked_while_paused.append(key["up"])
            self.calls += 1
            return (AgentCommand("do the thing", "", "item-1")
                    if self.calls == 1 else None)

    driver = _OneItem()
    policy = _PressesPAtTheGate()
    driver.limit_policy = policy

    _app, _source = _run_with_status(monkeypatch, tmp_path, driver, max=None)

    assert policy.reached, "the gate was never entered — the test proved nothing"
    assert asked_while_paused == [False, False]
    assert key["up"] is False and driver.calls == 2


def _counts_down(total):
    """A non-list Driver with `total` units of work and no queue behind it."""
    from llm_loop.agentwork import AgentCommand, Driver

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


def test_a_driver_with_no_total_gets_no_invented_denominator(monkeypatch, tmp_path):
    """Only the driver says how much work a run has (Driver.pending_total).
    Reporting none, an uncapped run counts, and says nothing it cannot know."""
    from llm_loop import cyclecore

    monkeypatch.setattr(cyclecore, "run_agent_streaming",
                        lambda cmd, provider, raw, partial, prompt: 0)
    app, _source = _run_with_status(monkeypatch, tmp_path, _counts_down(3),
                                    max=None, provider="codex")

    assert app.status.iteration == 3
    assert app.status.max_iterations is None
    summary = app.render(width=200)[1]
    assert "iter 3" in summary and "iter 3/" not in summary


def test_a_driver_with_a_queue_of_its_own_gets_a_denominator(monkeypatch,
                                                              tmp_path):
    """A queue does not have to be a list file to be countable: the kit-promotion
    pass drains a folder of requests, and its row used to read a bare `iter 1`
    however much was left in it — the one number a watcher wants.
    """
    from llm_loop import cyclecore
    from llm_loop.agentwork import AgentCommand, Driver

    class _DrainsAFolder(Driver):
        """Two requests; one iteration clears one of them."""

        def __init__(self):
            self.left = 2

        def pending_total(self):
            return self.left

        def next_command(self):
            if not self.left:
                return None
            return AgentCommand("promote", "", f"{self.left} request(s)")

        def on_success(self, returncode):
            self.left -= 1

    monkeypatch.setattr(cyclecore, "run_claude_streaming",
                        lambda cmd, raw, partial, prompt="", mailbox=None: 0)
    app, _source = _run_with_status(monkeypatch, tmp_path, _DrainsAFolder(),
                                    max=None)

    assert (app.status.iteration, app.status.max_iterations) == (2, 2)
    assert "iter 2/2" in app.render(width=200)[1]
    # The window name is the summary row's first field verbatim, so it carries
    # the same total — that is the whole point of building it from the Segment.
    assert sl.title_text(app.status).startswith("· iter 2/2")


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
                             input_source=tio.NullInputSource(), refresh=60,
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
    previous = projectroot.project_dir()
    try:
        for _call in (1, 2):        # what a periodic wrapper does, twice
            cyclecore.run_loop(_counts_down(2), args,
                               app_name="pytest-statusline",
                               setup_logging=False, wait_on_start=False,
                               progress=progress)
    finally:
        projectroot.set_project_root(previous)

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
                       input_source=tio.NullInputSource(), refresh=60)
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
