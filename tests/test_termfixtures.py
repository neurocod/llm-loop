"""Pins for the doubles in `_termfixtures` that decide a pin's verdict.

`RecordingTerminal.wait_for_frame` is the only assertion several status-line
pins make, so a wait that returned the first frame whatever it showed would
leave them green while checking nothing.
"""

import time

import pytest

from _termfixtures import RecordingTerminal

# The budget of a wait that is MEANT to run out: nothing is painting, so how
# long it is decides only how long these pins take.
_EXPIRING_WAIT_S = 0.2


def _is_yes(frame):
    return frame == ["yes"]


def test_the_wait_skips_frames_that_do_not_match_and_returns_the_one_that_does():
    terminal = RecordingTerminal()
    terminal.frames.put(["no"])
    terminal.frames.put(["yes"])
    terminal.frames.put(["later"])

    assert terminal.wait_for_frame(_is_yes, what="yes") == ["yes"]
    # The frames before it were taken; the one after is left for the next wait.
    assert terminal.frames.get_nowait() == ["later"]


def test_a_wait_nothing_matches_fails_naming_the_wait_and_the_last_frame():
    terminal = RecordingTerminal()
    terminal.frames.put(["no"])
    terminal.frames.put(["still no"])

    with pytest.raises(AssertionError) as failure:
        terminal.wait_for_frame(_is_yes, what="the yes row",
                                timeout=_EXPIRING_WAIT_S)

    message = str(failure.value)
    assert "the yes row" in message
    assert "still no" in message


def test_a_predicates_own_what_names_the_wait():
    def named(frame):
        return _is_yes(frame)

    named.what = "the named wait"

    with pytest.raises(AssertionError, match="the named wait"):
        RecordingTerminal().wait_for_frame(named, timeout=_EXPIRING_WAIT_S)


class _RepaintsForever:
    """A `frames` queue that is never empty: a painter repainting a wrong frame.

    Past `honest_after` seconds it hands out the matching frame, so a wait
    that ignores its deadline while frames keep coming ends by RETURNING —
    which the pin reads as the failure — instead of hanging the suite.
    """

    def __init__(self, honest_after):
        self._until = time.monotonic() + honest_after

    def get(self, timeout=None):
        time.sleep(0.005)
        return ["yes"] if time.monotonic() > self._until else ["busy"]


def test_the_deadline_holds_while_wrong_frames_keep_arriving():
    terminal = RecordingTerminal()
    # 25 times the wait's budget: only a wait that never looks at its deadline
    # while the queue is full gets this far.
    terminal.frames = _RepaintsForever(honest_after=_EXPIRING_WAIT_S * 25)

    with pytest.raises(AssertionError, match="busy"):
        terminal.wait_for_frame(_is_yes, what="yes",
                                timeout=_EXPIRING_WAIT_S)
