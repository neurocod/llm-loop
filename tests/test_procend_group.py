"""Pin `procend.start_tree` on POSIX: a tree in a process group of its own
is ended through the group's keeper — never by a group number — within the
stop budget, whoever reaped the leader first, whatever SIGCHLD is set to,
and also when this process dies without ending it.

Every staged member ignores SIGTERM, so only the keeper's SIGKILL can end
it: a pin that passes on a polite tree would pass on the old pid-based
sequence too. The proof that no member is left is the EOF of a pipe every
member holds."""

import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from llm_loop import procend, stopclock

pytestmark = pytest.mark.skipif(os.name == "nt", reason="process groups are POSIX'")

# The same bounds as test_procend_tree.py's Windows pins: an ending that does
# not wait is a write and a close; a killed tree lets go of its pipe well
# under a second idle, ten times that loaded; a Python child starts in 30 s
# under three other slots' load.
NO_WAIT_S = 0.5
TREE_END_S = 10.0
START_S = 30.0

# A child that ignores SIGTERM, starts a grandchild that ignores it too and
# shares its stdout, writes both pids to argv[1], and then sleeps — or, with
# argv[2] == "exit", exits at once, leaving the grandchild behind.
_STUBBORN_TREE = (
    "import os, signal, subprocess, sys, time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "g = subprocess.Popen([sys.executable, '-c', 'import signal, time;"
    " signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'],"
    " stdout=sys.stdout)\n"
    "open(sys.argv[1] + '.tmp', 'w').write(f'{os.getpid()} {g.pid}')\n"
    "os.replace(sys.argv[1] + '.tmp', sys.argv[1])\n"
    "if sys.argv[2] != 'exit':\n"
    "    time.sleep(60)\n")

# The same shape with SIGTERM's default action: what a Ctrl+C's ask ends.
_POLITE_TREE = _STUBBORN_TREE.replace(
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)", "pass")


def _start(tmp_path, source=_STUBBORN_TREE, then="sleep"):
    marker = tmp_path / "tree-up"
    proc = procend.start_tree([sys.executable, "-c", source, str(marker), then],
                              stdout=subprocess.PIPE)
    reader = threading.Thread(target=proc.stdout.read, daemon=True)
    reader.start()
    deadline = time.monotonic() + START_S
    while not marker.exists():
        assert time.monotonic() < deadline, "the tree never started"
        time.sleep(0.02)
    return proc, reader, [int(pid) for pid in marker.read_text().split()]


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _assert_ended(reader, pids):
    reader.join(TREE_END_S)
    survivors = [pid for pid in pids if _alive(pid)]
    assert not reader.is_alive(), f"a member still holds the pipe; alive: {survivors}"


def _cleanup(proc, pids):
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    try:
        proc.wait(timeout=TREE_END_S)
    except subprocess.TimeoutExpired:
        pass


@pytest.fixture
def no_group_numbers(monkeypatch):
    """Fail on any `killpg`: the ending names the group by the keeper's
    membership, so a reused group id can never be what it signals."""
    def refuse(pgid, signum):
        raise AssertionError(f"killpg({pgid}, {signum}) by number")
    monkeypatch.setattr(os, "killpg", refuse)


def test_a_stubborn_tree_on_a_spent_budget_ends_at_once(tmp_path, no_group_numbers):
    proc, reader, pids = _start(tmp_path)
    try:
        assert procend.owns(proc)
        started = time.monotonic()
        with stopclock.ending(time.monotonic()):
            procend.end_process_tree(proc)
        took = time.monotonic() - started
        assert took < NO_WAIT_S, f"the ending waited on a spent budget: {took:.2f} s"
        _assert_ended(reader, pids)
    finally:
        _cleanup(proc, pids)


def test_a_stubborn_tree_without_a_budget_ends_after_one_grace(tmp_path, no_group_numbers):
    """No budget: the ask, one grace for the leader (it ignores SIGTERM),
    then the keeper's SIGKILL — not the old two graces."""
    proc, reader, pids = _start(tmp_path)
    try:
        started = time.monotonic()
        procend.end_process_tree(proc)
        took = time.monotonic() - started
        assert took < procend.REAP_GRACE_S + TREE_END_S / 2, f"took {took:.2f} s"
        _assert_ended(reader, pids)
    finally:
        _cleanup(proc, pids)


def test_a_leader_reaped_by_another_waiter_still_takes_its_group(tmp_path,
                                                                no_group_numbers):
    """The leader exits first and a stray `wait()` reaps it — the move that
    unpins a group id. The grandchild it left is still ended."""
    proc, reader, pids = _start(tmp_path, then="exit")
    try:
        proc.wait(timeout=TREE_END_S)
        assert _alive(pids[1]), "the grandchild should outlive its parent here"
        procend.end_unless_exited(proc)
        _assert_ended(reader, pids)
    finally:
        _cleanup(proc, pids)


def test_the_ask_reaches_every_member(tmp_path, no_group_numbers):
    """The terminal's Ctrl+C reaches no member (the group is not the
    foreground one); the press's hook, `ask_agent_process_to_end`, does."""
    proc, reader, pids = _start(tmp_path, source=_POLITE_TREE)
    try:
        procend.ask_agent_process_to_end(proc)
        _assert_ended(reader, pids)
    finally:
        _cleanup(proc, pids)


def test_an_unrelated_process_is_untouched(tmp_path, no_group_numbers):
    """Every ending entry, on a live tree and on an exited one: a process
    outside the tree — its own group, the test's sibling — lives on."""
    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
    try:
        for end in (procend.end_process_tree, procend.ask_agent_process_to_end,
                    procend.end_unless_exited):
            (tmp_path / end.__name__).mkdir()
            proc, reader, pids = _start(tmp_path / end.__name__, then="exit")
            try:
                proc.wait(timeout=TREE_END_S)
                end(proc)
                procend.release_exited(proc)
                _assert_ended(reader, pids)
            finally:
                _cleanup(proc, pids)
            assert bystander.poll() is None, f"{end.__name__} ended a bystander"
    finally:
        bystander.kill()
        bystander.wait(timeout=TREE_END_S)


def test_sigchld_ignored_changes_nothing(tmp_path, no_group_numbers):
    """SIGCHLD=SIG_IGN auto-reaps every child the moment it exits — the
    leader and the keeper included. Nothing here waits on a zombie, so the
    ownership and the ending stay what they are."""
    previous = signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    proc = None
    pids = []
    try:
        proc, reader, pids = _start(tmp_path)
        assert procend.owns(proc)
        with stopclock.ending(time.monotonic()):
            procend.end_process_tree(proc)
        _assert_ended(reader, pids)
    finally:
        signal.signal(signal.SIGCHLD, previous)
        if proc is not None:
            _cleanup(proc, pids)


def test_a_dead_keeper_still_ends_the_leader(tmp_path):
    """A keeper killed from outside takes the group's ending with it; the
    leader is still ended, by its own Popen."""
    proc, reader, pids = _start(tmp_path)
    try:
        keeper = procend._trees[proc]._keeper
        keeper.kill()
        keeper.wait(timeout=TREE_END_S)
        with stopclock.ending(time.monotonic()):
            procend.end_process_tree(proc)
        assert proc.wait(timeout=TREE_END_S) is not None
    finally:
        _cleanup(proc, pids)


# The run: starts a stubborn tree on its own stdout and leaves AT ONCE
# (`os._exit`) without ending it — the run killed from outside.
_RUN = ("import os, sys, time\n"
        "from llm_loop import procend\n"
        "marker, tree = sys.argv[1], sys.argv[2]\n"
        "procend.start_tree([sys.executable, '-c', tree, marker, 'sleep'],"
        " stdout=sys.stdout)\n"
        "while not os.path.exists(marker):\n"
        "    time.sleep(0.02)\n"
        "os._exit(0)\n")


def test_the_owner_s_death_ends_its_tree(tmp_path):
    """The keeper's pipe closes with its owner: what Windows' kill-on-close
    does for a job."""
    source = str(Path(procend.__file__).resolve().parents[1])
    marker = tmp_path / "tree-up"
    run = subprocess.Popen([sys.executable, "-c", _RUN, str(marker), _STUBBORN_TREE],
                           stdout=subprocess.PIPE,
                           env=dict(os.environ, PYTHONPATH=source))
    pids = []
    try:
        reader = threading.Thread(target=run.stdout.read, daemon=True)
        reader.start()
        reader.join(START_S + TREE_END_S)
        if marker.exists():
            pids = [int(pid) for pid in marker.read_text().split()]
        assert pids, "the tree never started"
        assert not reader.is_alive(), (
            f"a member outlived its owner; alive: {[p for p in pids if _alive(p)]}")
    finally:
        _cleanup(run, pids)
