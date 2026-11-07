"""`protocol._reap`: the server's end goes through `llm_loop.procend`, skips a
child that already exited, and never raises out of the `finally` it runs in."""

import subprocess
import sys

from llm_agents.codex import protocol


class _Dead:
    """A handle whose every OS call fails: the process is long gone and the
    OS has reclaimed it (Windows raises OSError from `wait` then). `pid` -1
    names no process, so `taskkill` fails and the escalation reaches the
    handle."""

    pid = -1

    def __init__(self, exited=None):
        self.exited = exited
        self.calls = []

    def poll(self):
        self.calls.append("poll")
        return self.exited

    def terminate(self):
        self.calls.append("terminate")
        raise OSError("gone")

    def kill(self):
        self.calls.append("kill")
        raise OSError("gone")

    def wait(self, timeout=None):
        self.calls.append("wait")
        raise OSError("gone")


def test_reap_ends_a_live_child_through_procend(monkeypatch):
    ended = []
    real = protocol.end_process_tree
    monkeypatch.setattr(protocol, "end_process_tree",
                        lambda proc: (ended.append(proc), real(proc)))
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        protocol._reap(child)
        # Bounded by procend: ask + wait + kill + wait, REAP_GRACE_S each.
        assert child.wait(timeout=4 * protocol.REAP_GRACE_S + 5) is not None
        assert ended == [child]
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_reap_leaves_an_exited_child_alone(monkeypatch):
    ended = []
    monkeypatch.setattr(protocol, "end_process_tree", ended.append)
    done = _Dead(exited=0)
    protocol._reap(done)
    assert ended == [] and done.calls == ["poll"]


def test_reap_never_raises_on_a_handle_the_os_took_back():
    dead = _Dead()
    protocol._reap(dead)  # every call below raises OSError; none escapes
    assert dead.calls[:2] == ["poll", "terminate"]
    assert "wait" in dead.calls
