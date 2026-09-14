"""Numeric input and live weekly policy edits through the console keys."""

import pytest

from llm_loop import statusline as sl, termio
from llm_loop.limits import LimitPolicy, SessionLimit, WeeklyLimit
from llm_loop.usage import Usage, UsageReading


def press(app, *keys):
    for key in keys:
        app.handle_event(termio.Key(key))


@pytest.mark.parametrize("text", ["abc", "-+.", " \t", "²٣９", "mspw"])
def test_spinbox_refuses_non_ascii_digits(text):
    editor = sl.SpinBox(98)
    for key in text:
        editor.handle(key)
    assert editor.buffer == "98"


def test_spinbox_supports_cursor_edits_and_bounded_steps():
    editor = sl.SpinBox(98)
    for key in ("home", "delete", "1", "end", "\x08", "9"):
        editor.handle(key)
    assert editor.buffer == "19"
    editor.handle("up")
    assert editor.value == 20
    editor.handle("down")
    assert editor.value == 19
    editor.set("100")
    editor.handle("up")
    assert editor.value == 100
    editor.clear()
    assert editor.value is None
    editor.handle("down")
    assert editor.value == 0
    editor.handle("up")
    assert editor.value == 1


def make_app(policy):
    app = sl.StatusApp(enabled=False)
    policies = [policy]
    app.register_action(sl.WeeklyLimitAction(lambda: policies[0]))
    return app, policies


def test_open_cancel_apply_and_reopen_edit_the_real_gate():
    rule = WeeklyLimit(98)
    policy = LimitPolicy([SessionLimit(80), rule])
    app, _ = make_app(policy)
    app.update(quotas=[sl.QuotaRow("week", 98, None, "ceiling 98%")])
    press(app, "w")
    assert app.mode.editor.buffer == "98"
    assert "98|" in app.render(120)[-1]
    assert sl.colorize(app.render(120)[-1]) == app.render(120)[-1]
    press(app, "down", "\x1b", "\x1b")
    assert rule.limit == 98
    press(app, "w", "up", "\r")
    assert rule.limit == 99
    assert "99%" in app.status.quotas[0].policy
    assert app.status.quotas[0].percent == 98
    usage = Usage(UsageReading(0, None), UsageReading(98, None),
                  UsageReading(None, None), [])
    assert not policy._violations(policy._status(usage, 0))
    press(app, "down", "\r")
    assert policy._violations(policy._status(usage, 0))[0][0] is rule
    press(app, "\x1b", "\x1b", "w")
    assert app.mode.editor.buffer == "98"


@pytest.mark.parametrize("text", ["", "101", "9" * 5000])
def test_invalid_submission_stays_in_editor_without_changing_limit(text):
    rule = WeeklyLimit(98)
    app, _ = make_app(LimitPolicy([rule]))
    press(app, "w", "\x15", *text, "\r")
    assert isinstance(app.mode, sl.WeeklyLimitMode)
    assert rule.limit == 98
    assert "0 to 100" in app.status.note


def test_pasted_newlines_and_shortcuts_cannot_stop_or_pause_the_loop():
    rule = WeeklyLimit(98)
    app, _ = make_app(LimitPolicy([rule]))
    press(app, "w", "\x15", *"90\nstop\npause\nmessage")
    assert rule.limit == 90
    assert not app.stop_requested_here and not app.paused
    assert isinstance(app.mode, sl.WeeklyLimitMode)


def test_policy_switch_rejects_stale_draft_and_reopens_current_limit():
    old, new = WeeklyLimit(98), WeeklyLimit(90)
    app, policies = make_app(LimitPolicy([old]))
    press(app, "w", "down")
    policies[0] = LimitPolicy([new])
    press(app, "\r")
    assert (old.limit, new.limit) == (98, 90)
    press(app, "\x1b", "\x1b", "w")
    assert app.mode.editor.buffer == "90"
    press(app, "\x1b", "\x1b")
    policies[0] = None
    assert app.action_for("w") is None


def test_alt_shortcut_clears_draft_without_stopping():
    app, _ = make_app(LimitPolicy([WeeklyLimit(98)]))
    press(app, "w", "\x1b", "s")
    assert isinstance(app.mode, sl.WeeklyLimitMode)
    assert app.mode.editor.buffer == ""
    assert not app.stop_requested_here
