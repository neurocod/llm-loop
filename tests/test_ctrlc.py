"""Ctrl+C inside a run is an event the run polls, never a KeyboardInterrupt.

The runner-level scenarios (a turn, a quota hold, the exit push, a second
press) are pinned in `test_abnormal_exit_epilogue.py` and `test_git_push.py`;
these pin the mechanism they all stand on, with the real signal.
"""

import signal
import threading
import time

import pytest

from llm_loop import ctrlc, stopchannel, termio


def test_sigint_inside_a_run_is_a_press_and_raises_nothing():
    with ctrlc.captured() as interrupt:
        signal.raise_signal(signal.SIGINT)
        # The handler runs between bytecodes; this line is past it.
        assert interrupt.presses == 1
        assert ctrlc.current() is interrupt
    assert ctrlc.current().requested is False, "the run's presses outlived it"


def test_the_handler_it_replaced_is_back_once_the_run_has_left():
    before = signal.getsignal(signal.SIGINT)
    with ctrlc.captured():
        assert signal.getsignal(signal.SIGINT) is not before
    assert signal.getsignal(signal.SIGINT) is before
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)


def test_a_runner_inside_a_run_shares_its_interrupt():
    with ctrlc.captured() as outer:
        with ctrlc.captured() as inner:
            assert inner is outer
        # The inner exit must not have taken the outer run's handler away.
        signal.raise_signal(signal.SIGINT)
        assert outer.presses == 1


def test_a_press_runs_the_hooks_of_its_moment_only_and_survives_a_bad_one():
    interrupt = ctrlc.Interrupt()
    called = []

    def broken():
        raise RuntimeError("staged: the hook broke")

    interrupt.press()           # before the block: no hook of it runs
    with interrupt.on_press(broken), interrupt.on_press(
            lambda: called.append(interrupt.presses)):
        interrupt.press()
    interrupt.press()           # after the block
    assert called == [2]
    assert interrupt.since(2) and not interrupt.since(3)


def test_a_wait_ends_on_a_press_from_another_thread():
    """What the key reader does with '\\x03' (`ctrlc.deliver`), mid-sleep."""
    with ctrlc.captured():
        threading.Timer(0.2, ctrlc.deliver).start()
        started = time.monotonic()
        assert stopchannel.sleep_unless(30) is True
        elapsed = time.monotonic() - started
    # 0.2 s timer + one STOP_RECHECK_SECONDS poll (0.25 s); 30 s if unheard.
    assert elapsed < 5, f"the wait sat out the press: {elapsed:.1f} s"


def test_the_key_reader_s_ctrl_c_is_a_press_inside_a_run():
    reader = termio.TerminalInput()
    with ctrlc.captured() as interrupt:
        reader._emit(lambda event: None, "\x03")
        assert interrupt.presses == 1


def test_the_key_reader_s_ctrl_c_is_a_keyboard_interrupt_outside_a_run():
    reader = termio.TerminalInput()
    with pytest.raises(KeyboardInterrupt):
        reader._emit(lambda event: None, "\x03")
        time.sleep(1)   # interrupt_main is heard between bytecodes
