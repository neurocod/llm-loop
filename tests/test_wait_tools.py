"""agent-tools' waiters: they end by themselves, on the marker or the timeout."""

import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "agent-tools"

# Every case below ends in well under a second; the cap only stops a hung
# waiter from stalling the suite.
CAP_S = 60


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOLS / script), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          check=False, timeout=CAP_S)


def test_follow_file_stops_on_the_built_in_task_preset(tmp_path):
    log = tmp_path / "task.out"
    log.write_text("working\n[exited with code 0]\n", encoding="utf-8")
    done = _run("follow_file.py", str(log), "--preset", "task", "--poll", "0.05")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "working" in done.stdout


def test_follow_file_takes_a_preset_the_caller_defines(tmp_path):
    log = tmp_path / "run.txt"
    log.write_text("a\n=== summary ===\n", encoding="utf-8")
    done = _run("follow_file.py", str(log), "--define-preset", "batch",
                "=== summary ===", "--preset", "batch", "--poll", "0.05",
                "--timeout", "30")
    assert done.returncode == 0, done.stdout + done.stderr


def test_follow_file_define_preset_extends_a_built_in_one(tmp_path):
    log = tmp_path / "task.out"
    log.write_text("[exited with code 3]\n", encoding="utf-8")
    done = _run("follow_file.py", str(log), "--define-preset", "task", "never",
                "--preset", "task", "--poll", "0.05", "--timeout", "30")
    assert done.returncode == 0, done.stdout + done.stderr


def test_follow_file_refuses_an_invalid_regex_when_it_is_defined(tmp_path):
    done = _run("follow_file.py", str(tmp_path / "x"), "--define-preset", "bad",
                "(", "--timeout", "30")
    assert done.returncode == 2
    assert "--define-preset bad: invalid regex" in done.stderr


def test_follow_file_refuses_an_unknown_preset_by_name(tmp_path):
    done = _run("follow_file.py", str(tmp_path / "x"), "--preset", "nope")
    assert done.returncode == 2
    assert "unknown preset nope" in done.stderr
    assert "task" in done.stderr


def test_follow_file_times_out_with_exit_1(tmp_path):
    done = _run("follow_file.py", str(tmp_path / "never"), "--until", "x",
                "--poll", "0.05", "--timeout", "0.2")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "timeout" in done.stdout


def test_follow_file_refuses_a_wait_that_never_ends(tmp_path):
    done = _run("follow_file.py", str(tmp_path / "x"))
    assert done.returncode == 2
    assert "never ends" in done.stderr


@pytest.mark.parametrize(("text", "contains", "code"), [
    ("ready\n", None, 0),
    ("ready\n", "re+ady", 0),
    ("ready\n", "absent", 1),
])
def test_wait_file_appears(tmp_path, text, contains, code):
    target = tmp_path / "done"
    target.write_text(text, encoding="utf-8")
    args = [str(target), "--poll", "0.05", "--timeout", "0.2"]
    if contains is not None:
        args += ["--contains", contains]
    done = _run("wait_file_appears.py", *args)
    assert done.returncode == code, done.stdout + done.stderr


def test_wait_file_appears_times_out_on_a_missing_file(tmp_path):
    done = _run("wait_file_appears.py", str(tmp_path / "never"),
                "--poll", "0.05", "--timeout", "0.2")
    assert done.returncode == 1, done.stdout + done.stderr
    assert "timeout" in done.stdout


def test_wait_file_gone_returns_at_once_for_an_absent_file(tmp_path):
    done = _run("wait_file_gone.py", str(tmp_path / "stop"))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "already absent" in done.stdout


def test_wait_file_gone_waits_for_the_file_to_go(tmp_path):
    stop = tmp_path / "stop"
    stop.write_text("", encoding="utf-8")
    waiter = subprocess.Popen(
        [sys.executable, str(TOOLS / "wait_file_gone.py"), str(stop), "--poll", "0.05"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    lines: "queue.Queue[str]" = queue.Queue()
    # A reader thread, so a waiter that stalls before its banner fails the test
    # after CAP_S instead of hanging a bare readline() forever.
    reader = threading.Thread(
        target=lambda: [lines.put(line) for line in waiter.stdout], daemon=True)
    reader.start()
    try:
        # The banner says the waiter saw the file, so the removal below is
        # what ends it rather than a race with its first check.
        banner = lines.get(timeout=CAP_S)
        assert "waiting for" in banner
        stop.unlink()
        waiter.wait(timeout=CAP_S)
        reader.join(timeout=CAP_S)
    finally:
        if waiter.poll() is None:
            waiter.kill()
        waiter.wait()
        waiter.stdout.close()
    out = "".join(lines.queue)
    assert waiter.returncode == 0, out
    assert "gone after" in out
