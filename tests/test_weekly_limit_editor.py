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
    refreshed = []
    app.register_action(sl.WeeklyLimitAction(
        lambda: policies[0], lambda: refreshed.append(True)))
    return app, policies, refreshed


def test_open_cancel_apply_and_reopen_edit_the_real_gate():
    rule = WeeklyLimit(98)
    policy = LimitPolicy([SessionLimit(80), rule])
    app, _, refreshed = make_app(policy)
    press(app, "w")
    assert app.mode.editor.buffer == "98"
    assert "98|" in app.render(120)[-1]
    press(app, "down", "\x1b")
    assert rule.limit == 98
    press(app, "w", "up", "\r")
    assert rule.limit == 99
    assert refreshed == [True]
    usage = Usage(UsageReading(0, None), UsageReading(98, None),
                  UsageReading(None, None), [])
    assert not policy._violations(policy._status(usage, 0))
    press(app, "down", "\r")
    assert policy._violations(policy._status(usage, 0))[0][0] is rule
    press(app, "\x1b", "w")
    assert app.mode.editor.buffer == "98"


@pytest.mark.parametrize("text", ["", "101", "9" * 5000])
def test_invalid_submission_stays_in_editor_without_changing_limit(text):
    rule = WeeklyLimit(98)
    app, _, refreshed = make_app(LimitPolicy([rule]))
    press(app, "w", "\x15", *text, "\r")
    assert isinstance(app.mode, sl.WeeklyLimitMode)
    assert rule.limit == 98
    assert not refreshed
    assert "0 to 100" in app.status.note


def test_pasted_newlines_and_shortcuts_cannot_stop_or_pause_the_loop():
    rule = WeeklyLimit(98)
    app, _, _ = make_app(LimitPolicy([rule]))
    press(app, "w", "\x15", *"90\nstop\npause\nmessage")
    assert rule.limit == 90
    assert not app.stop_requested_here and not app.paused
    assert isinstance(app.mode, sl.WeeklyLimitMode)


def test_policy_switch_rejects_stale_draft_and_reopens_current_limit():
    old, new = WeeklyLimit(98), WeeklyLimit(90)
    app, policies, refreshed = make_app(LimitPolicy([old]))
    press(app, "w", "down")
    policies[0] = LimitPolicy([new])
    press(app, "\r")
    assert (old.limit, new.limit) == (98, 90)
    assert not refreshed
    press(app, "\x1b", "w")
    assert app.mode.editor.buffer == "90"
    press(app, "\x1b")
    policies[0] = None
    assert app.action_for("w") is None

