"""Ctrl+C inside a run is an event the run polls, never a KeyboardInterrupt.

The runner-level scenarios (a turn, a quota hold, the exit push, a second
press) are pinned in `test_abnormal_exit_epilogue.py` and `test_git_push.py`;
these pin the mechanism they all stand on, with the real signal.
"""

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from llm_loop import ctrlc, stopchannel, termio, usage
from llm_loop.limits import LimitPolicy, SessionLimit

# The bound on a child Python that stages one press and exits: interpreter
# start plus the package import, 0.27 / 0.25 / 0.28 s measured 2026-10-03; a press
# that deadlocks never ends, so the bound only has to be far above that.
CHILD_TIMEOUT_S = 30


@contextlib.contextmanager
def pressed_run():
    """`ctrlc.captured()` for a block that presses: it leaves as the
    interrupt (see the late-press pin below), and nothing inside it raised."""
    with pytest.raises(SystemExit) as left:
        with ctrlc.captured() as interrupt:
            yield interrupt
    assert left.value.code == ctrlc.EXIT_CODE


def test_sigint_inside_a_run_is_a_press_and_raises_nothing():
    with pressed_run() as interrupt:
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


def test_a_sigint_while_the_main_thread_holds_the_hooks_lock_is_counted():
    """The handler runs between ANY two bytecodes of the main thread.

    `on_press` registers a hook every turn under the Interrupt's lock, on the
    main thread — the thread the SIGINT handler then runs on. A `press` that
    took that plain lock waited for its own thread for ever (review of 0072,
    F1). Staged in a child process: a deadlock there is a timeout here, not a
    hung test run.
    """
    script = textwrap.dedent("""
        import signal
        from llm_loop import ctrlc
        with ctrlc.captured() as interrupt:
            with interrupt._lock:
                signal.raise_signal(signal.SIGINT)
            print("presses", interrupt.presses)
    """)
    package_root = os.path.dirname(os.path.dirname(ctrlc.__file__))
    env = dict(os.environ, PYTHONPATH=package_root)
    try:
        done = subprocess.run([sys.executable, "-c", script], env=env,
                              capture_output=True, text=True,
                              timeout=CHILD_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        pytest.fail("a SIGINT under the hooks' lock deadlocked the main thread")
    assert "presses 1" in done.stdout, done.stderr
    # The press the block never acted on: it leaves as the interrupt.
    assert done.returncode == ctrlc.EXIT_CODE, done.stderr


def test_a_press_the_run_never_acted_on_does_not_let_it_return():
    """A press after the run's last look leaves as the interrupt, 130.

    A returned result tells a wrapper to start its next phase. A press in
    what a runner does past its epilogue (the sequential runner's last close
    of its pusher, say) was counted on an Interrupt nobody asked any more,
    and the run returned as if nobody had pressed.
    """
    with pytest.raises(SystemExit) as left:
        with ctrlc.captured():
            signal.raise_signal(signal.SIGINT)
    assert left.value.code == ctrlc.EXIT_CODE
    # Gone with the run: the next press is Python's own again.
    with pytest.raises(KeyboardInterrupt):
        signal.raise_signal(signal.SIGINT)


def test_a_block_leaving_by_its_own_exit_keeps_it_after_a_press():
    """An ending is chosen once: a press does not rewrite an exit 3."""
    with pytest.raises(SystemExit) as left:
        with ctrlc.captured() as interrupt:
            interrupt.press()
            raise SystemExit(3)
    assert left.value.code == 3


class _OverTheCeiling:
    def get_usage(self, cache_value=True):
        return usage.parse_usage({"five_hour": {"utilization": 90.0}})

    def invalidate(self):
        pass


def test_a_quota_hold_ends_on_ctrl_c_and_says_so(capsys):
    """`check_and_wait` asks the run's Interrupt itself, first.

    Its caller's stop channels are no stand-in: an embedder may pass none,
    and a parallel worker's are set only once the main thread's join has
    heard the press — until then the hold's sleep returned at once and the
    hold went round again. Here the caller's channel would also end it, and
    it must not be the one that did.
    """
    policy = LimitPolicy([SessionLimit(5)])
    asked = []

    def stop_channel():
        asked.append(1)
        return True

    with pressed_run() as interrupt:
        interrupt.press()
        paused, _start = policy.check_and_wait(_OverTheCeiling(), 0.0,
                                               should_stop=stop_channel)
    assert paused is True
    out = capsys.readouterr().out
    assert "Wait interrupted by user (Ctrl+C)." in out
    assert "Stop requested" not in out and asked == []


def test_the_stop_file_wait_leaves_on_the_run_s_ctrl_c(
        tmp_path, monkeypatch, capsys):
    """Held back by a stop file, a launch leaves on the press, not on the
    file going away — and does not go on to start."""
    sentinel = tmp_path / "stop"
    sentinel.write_text("")
    monkeypatch.setattr(stopchannel, "stop_file_path", lambda: str(sentinel))
    # The backstop: a wait deaf to the press ends when the file goes, and
    # fails below instead of hanging the suite.
    backstop = threading.Timer(5, sentinel.unlink)
    backstop.start()
    try:
        with pressed_run() as interrupt:
            interrupt.press()
            stopchannel.wait_for_stop_file_clear()
    finally:
        backstop.cancel()
    out = capsys.readouterr().out
    assert ctrlc.WAIT_INTERRUPTED_LINE in out
    assert "Stop file removed" not in out


def test_a_runner_inside_a_run_shares_its_interrupt():
    with pressed_run() as outer:
        with ctrlc.captured() as inner:
            assert inner is outer
            inner.press()
        # The inner exit must neither have left as the interrupt — that is
        # the outer run's to do — nor taken the outer run's handler away.
        signal.raise_signal(signal.SIGINT)
        assert outer.presses == 2


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
    with pressed_run():
        threading.Timer(0.2, ctrlc.deliver).start()
        started = time.monotonic()
        assert stopchannel.sleep_unless(30) is True
        elapsed = time.monotonic() - started
    # 0.2 s timer + one STOP_RECHECK_SECONDS poll (0.25 s); 30 s if unheard.
    assert elapsed < 5, f"the wait sat out the press: {elapsed:.1f} s"


def test_the_key_reader_s_ctrl_c_is_a_press_inside_a_run():
    reader = termio.TerminalInput()
    with pressed_run() as interrupt:
        reader._emit(lambda event: None, "\x03")
        assert interrupt.presses == 1


def test_the_key_reader_s_ctrl_c_is_a_keyboard_interrupt_outside_a_run():
    reader = termio.TerminalInput()
    with pytest.raises(KeyboardInterrupt):
        reader._emit(lambda event: None, "\x03")
        time.sleep(1)   # interrupt_main is heard between bytecodes
