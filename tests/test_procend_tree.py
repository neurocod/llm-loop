"""Pin `procend.start_tree`: a tree whose lifetime the OS owns is ended by that
identity — its job on Windows, its process group on POSIX — within the stop
budget, and never by a numeric pid that may name someone else by then.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_loop import procend, stopclock, win_job

# What "did not wait" means: an ending by identity is a system call or two;
# every wait it replaces is a second or more.
NO_WAIT_S = 0.5
# How long a killed tree may take to let go of its pipe: well under a second
# idle, ten times that loaded.
TREE_END_S = 10.0
# A Python child's start under three other slots' load: 30 s.
START_S = 30.0

# A child that starts a grandchild sharing its stdout, marks argv[1] once it
# has, and sleeps: the shape of an npm shim in front of the provider CLI.
_TREE = ("import subprocess, sys, time\n"
         "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
         " stdout=sys.stdout)\n"
         "open(sys.argv[1], 'w').close()\n"
         "time.sleep(60)\n")


def _wait_for(path: Path) -> None:
    deadline = time.monotonic() + START_S
    while not path.exists():
        assert time.monotonic() < deadline, "the tree never started"
        time.sleep(0.02)


def _reader(stream):
    out = []
    thread = threading.Thread(target=lambda: out.append(stream.read()), daemon=True)
    thread.start()
    return thread


def test_a_tree_on_a_spent_budget_ends_at_once_with_no_descendant(tmp_path):
    marker = tmp_path / "tree-up"
    proc = procend.start_tree([sys.executable, "-c", _TREE, str(marker)],
                              stdout=subprocess.PIPE)
    reader = _reader(proc.stdout)
    try:
        assert procend.owns(proc), "the child was started without an owned lifetime"
        _wait_for(marker)
        started = time.monotonic()
        with stopclock.ending(time.monotonic()):
            procend.end_process_tree(proc)
        took = time.monotonic() - started
        assert took < NO_WAIT_S, f"the ending waited on a spent budget: {took:.2f} s"
        # The child and the grandchild both hold the pipe: EOF is the proof
        # that neither is left.
        reader.join(TREE_END_S)
        assert not reader.is_alive(), "a descendant outlived the ending"
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=TREE_END_S)


def _foreign():
    """A process nobody's tree holds: on POSIX the leader of a group of its
    own, so a `killpg` by its number would reach it."""
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            **({} if os.name == "nt" else {"start_new_session": True}))


def test_an_exited_tree_s_number_reused_by_another_process_is_not_ended():
    """The pid of an exited, reaped child can name another process later. The
    staged reuse: the tracked child exits, and its Popen is handed the pid of
    an unrelated live process. Every ending entry then acts on the tree's
    identity — the job, the reaped leader's group — and the other process
    lives on; a pid-based kill (`taskkill /T /PID`, a `killpg` after the
    reap) would end it."""
    proc = procend.start_tree([sys.executable, "-c", "pass"])
    foreign = _foreign()
    try:
        assert procend.owns(proc)
        proc.wait(timeout=START_S)
        proc.pid = foreign.pid
        procend.ask_agent_process_to_end(proc)
        procend.end_process_tree(proc)
        procend.end_unless_exited(proc)
        time.sleep(0.5)
        assert foreign.poll() is None, "an ending reached a process outside its tree"
    finally:
        foreign.kill()
        foreign.wait(timeout=TREE_END_S)


# A run inside a job of its own, as `llm_agents.cores` hold starts it
# (KILL_ON_JOB_CLOSE | BREAKAWAY_OK): its child joins a nested job of its own.
_NESTED = ("import sys\n"
           "from llm_loop import procend\n"
           "proc = procend.start_tree([sys.executable, '-c', 'pass'])\n"
           "proc.wait()\n"
           "print('owned' if procend.owns(proc) else 'unowned', flush=True)\n")


@pytest.mark.skipif(os.name != "nt", reason="job objects are Windows'")
def test_a_run_inside_a_job_still_owns_its_children(tmp_path):
    """No CREATE_BREAKAWAY_FROM_JOB is needed: the outer job's limits do not
    refuse a nested job, so a run under `cores hold` keeps its children's
    tree in a job of its own and the outer job still contains them."""
    source = str(Path(procend.__file__).resolve().parents[1])
    handle = win_job.create(None, win_job.KILL_ON_JOB_CLOSE | win_job.BREAKAWAY_OK)
    try:
        run = win_job.start(handle, [sys.executable, "-c", _NESTED],
                            stdout=subprocess.PIPE,
                            env=dict(os.environ, PYTHONPATH=source))
        out, _ = run.communicate(timeout=START_S)
    finally:
        win_job.kernel32().CloseHandle(handle)
    assert out.strip() == b"owned", out
