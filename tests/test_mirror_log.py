"""The mirror log must never be able to kill the run it is recording.

Two loops sharing one log is normal (a sequential run, a parallel run, the
grow-kit pass), and on Windows that makes rotation fail while the other process
holds the file. These pin the two halves that turned such a failure fatal.
"""

import logging
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_loop import cyclecore


@pytest.fixture(autouse=True)
def _restore_streams():
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


def _handler(tmp_path, **kwargs):
    handler = cyclecore._MirrorLogHandler(
        tmp_path / "mirror.log", maxBytes=64, backupCount=2,
        encoding="utf-8", **kwargs)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _logger(name, handler):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = [handler]
    return logger


def test_a_rotation_another_process_blocks_keeps_the_log_running(tmp_path, monkeypatch):
    """A failed rename must cost lines in the file, not the run."""
    handler = _handler(tmp_path)
    logger = _logger("runCycle.test-rotate", handler)

    def refuse(*args, **kwargs):
        raise PermissionError(32, "used by another process")

    monkeypatch.setattr(os, "rename", refuse)
    monkeypatch.setattr(os, "replace", refuse)
    for i in range(40):  # well past maxBytes, so rollover is attempted
        logger.info(f"line {i} " + "x" * 20)
    handler.close()

    written = (tmp_path / "mirror.log").read_text(encoding="utf-8")
    assert "line 39" in written  # kept appending instead of giving up


def test_a_failed_rotation_is_retried_on_a_timer_not_per_line(tmp_path, monkeypatch):
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(1)
        raise PermissionError(32, "used by another process")

    handler = _handler(tmp_path)
    logger = _logger("runCycle.test-retry", handler)
    monkeypatch.setattr(os, "rename", refuse)
    monkeypatch.setattr(os, "replace", refuse)
    for i in range(30):
        logger.info(f"line {i} " + "x" * 20)
    handler.close()

    # One burst of renames (the backup chain), then the backoff holds them off;
    # without it every record past the cap would try again.
    assert len(attempts) <= handler.backupCount + 1


def test_a_logging_failure_never_recurses_through_the_tee(tmp_path):
    """The exact shape that killed a run: the handler reports to stderr, stderr
    is the tee, the tee logs it, the handler fails again."""
    depth = {"max": 0, "now": 0}

    class Exploding(logging.Handler):
        def emit(self, record):
            depth["now"] += 1
            depth["max"] = max(depth["max"], depth["now"])
            try:
                # What logging.Handler.handleError does by default.
                print("--- Logging error ---", file=sys.stderr)
            finally:
                depth["now"] -= 1

    handler = Exploding()
    logger = _logger("runCycle.test-recursion", handler)
    sys.stderr = cyclecore._TeeToLog(sys.stderr, logger)

    print("something the loop printed", file=sys.stderr)

    assert depth["max"] == 1  # the report did not re-enter the logger


def test_the_tee_still_logs_normally_after_a_guarded_call(tmp_path):
    handler = _handler(tmp_path)
    logger = _logger("runCycle.test-normal", handler)
    sys.stdout = cyclecore._TeeToLog(sys.stdout, logger)

    print("first")
    print("second")
    handler.close()

    written = (tmp_path / "mirror.log").read_text(encoding="utf-8")
    assert "first" in written and "second" in written
