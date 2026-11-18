"""The core lease client (`llm_agents.cores`) against a fake counter.

The wire is replaced (`connect`/`transport_of`), so nothing here needs the
optional `extras/agent-counter` executable; the round trips against a built
one belong to whoever builds it, and to `AgentCounter.exe --selftest`.
"""

import os
import sys
from pathlib import Path

import pytest

from llm_agents import agent_counter, cores

REGISTRY_CPP = Path(__file__).resolve().parents[1] / "extras" / "agent-counter" / "Registry.cpp"

windows_only = pytest.mark.skipif(os.name != "nt", reason="the broker speaks a Windows named pipe")


def test_the_default_wait_is_the_same_figure_in_both_halves():
    assert f"kDefaultWaitS = {cores.DEFAULT_WAIT_S}" in REGISTRY_CPP.read_text(encoding="utf-8")


class FakeCounter:
    """A transport playing the counter: `answers` are handed out one poll at a
    time (None = nothing arrived yet), `cores.Closed` ends the stream."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.sent = []
        self.closed = False

    def send(self, message):
        self.sent.append(message)

    def poll(self):
        if not self.answers:
            return None
        answer = self.answers.pop(0)
        if answer is cores.Closed:
            raise cores.Closed("end of stream")
        return answer

    def close(self):
        self.closed = True


@pytest.fixture
def wire(monkeypatch):
    """`lease`/`run` options routing to one FakeCounter; lines said are kept."""
    monkeypatch.delenv(cores.LEASE_ENV, raising=False)
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "1")
    state = {"lines": []}

    def options(counter, handle=object()):
        state["counter"] = counter
        return {"connect": lambda complain: handle, "transport_of": lambda _: counter,
                "say": state["lines"].append}

    state["options"] = options
    return state


def _grant(cores_, total=16, **extra):
    return dict({"op": "grant", "cores": cores_, "total": total, "free": cores_,
                 "foreign": 0.0, "forced": False, "waited_s": 0.0}, **extra)


@windows_only
def test_a_grant_reaches_the_child_and_its_exit_code_comes_back(wire, tmp_path):
    counter = FakeCounter(None, _grant(6))
    seen = tmp_path / "seen"
    code = cores.run([sys.executable, "-c",
                      f"import os, sys; open({str(seen)!r}, 'w').write(os.environ['{cores.LEASE_ENV}']); "
                      f"sys.exit(7)"],
                     want=8, minimum=2, label="suite", wait_s=30, **wire["options"](counter))
    assert code == 7
    assert seen.read_text() == "6"
    job = counter.sent[0].pop("job")
    assert job.startswith("Local\\llm-cores-"), "the lease names the job the child runs in"
    assert counter.sent == [{"op": "lease", "want": 8, "min": 2, "label": "suite", "wait_s": 30}]
    assert counter.closed, "the lease is returned once the child is done"
    assert wire["lines"] == ["6 of 16 for suite (asked 8, min 2; free 6, foreign 0.0)"]


_IN_JOB = """
import ctypes, sys
from ctypes import wintypes
k32 = ctypes.WinDLL("kernel32")
k32.OpenJobObjectW.restype = wintypes.HANDLE
k32.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
job = k32.OpenJobObjectW(0x0004, False, sys.argv[1])
inside = wintypes.BOOL(False)
k32.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
k32.IsProcessInJob(wintypes.HANDLE(-1), job, ctypes.byref(inside))
sys.exit(0 if job and inside.value else 9)
"""


@windows_only
def test_the_child_runs_inside_the_named_job(wire):
    """The job is what the broker measures and what kills the child with the
    wrapper; a child started outside it gets neither."""
    job = cores.Job.create()
    counter = FakeCounter(_grant(2))
    code = cores.run([sys.executable, "-c", _IN_JOB, job.name], want=2, make_job=lambda say: job,
                     **wire["options"](counter))
    assert code == 0, "the child did not find itself in the job"
    assert counter.sent[0]["job"] == job.name


_LEAVES_A_GRANDCHILD = """
import subprocess, sys
grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "w").write(str(grandchild.pid))
"""

# A kill-on-close exit measured 0.0053 s (claude_agent/test_claude_agent_protocol.py,
# 2026-09-08): 10 s to die; a survivor is judged alive after 1 s.
_DIES_MS = 10_000
_SURVIVES_MS = 1_000
_WAIT_TIMEOUT = 258


def _grandchild(pid_file):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x0001, False, int(pid_file.read_text()))
    assert handle, (f"the grandchild is gone already "
                    f"({ctypes.WinError(ctypes.get_last_error())})")
    return kernel, handle


def _reap(grandchild):
    kernel, handle = grandchild
    if kernel.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT:
        kernel.TerminateProcess(handle, 1)
    kernel.CloseHandle(handle)


_ROOT_ENDS = {"exits 0": "", "fails": "sys.exit(3)",
              "is terminated": "import os; os.kill(os.getpid(), 9)"}


@windows_only
@pytest.mark.parametrize("keep", [False, True], ids=["default", "keep_descendants"])
@pytest.mark.parametrize("end", list(_ROOT_ENDS))
def test_what_a_root_leaves_running_is_kept_only_when_opted_in(wire, tmp_path, monkeypatch,
                                                               end, keep):
    """Default: a test runner's stray worker or a crashed root's descendants
    die with the job. Opted in (a build): MSBuild's reusable nodes and
    `mspdbsrv` outlive it, whatever its exit code (`run`'s docstring)."""
    pid_file = tmp_path / "grandchild.pid"
    opened = []
    real_wait = cores._wait

    def wait(child, job, say):
        answer = real_wait(child, job, say)
        opened.append(_grandchild(pid_file))  # held open: its pid cannot be reused
        return answer
    monkeypatch.setattr(cores, "_wait", wait)
    cores.run([sys.executable, "-c", _LEAVES_A_GRANDCHILD + _ROOT_ENDS[end], str(pid_file)],
              want=2, keep_descendants=keep, **wire["options"](FakeCounter(_grant(2))))
    kernel, handle = opened[0]
    try:
        waited = kernel.WaitForSingleObject(handle, _SURVIVES_MS if keep else _DIES_MS)
        if keep:
            assert waited == _WAIT_TIMEOUT, "an opted-in run killed what its root left running"
        else:
            assert waited == 0, "a default run left its root's descendants running"
    finally:
        _reap(opened[0])


@windows_only
def test_a_run_that_does_not_end_normally_ends_the_whole_tree(wire, tmp_path, monkeypatch):
    pid_file = tmp_path / "grandchild.pid"
    opened = []

    def failing_wait(child, job, say):
        child.wait()
        opened.append(_grandchild(pid_file))  # held open: its pid cannot be reused
        raise RuntimeError("the wrapper failed while the tree ran")
    monkeypatch.setattr(cores, "_wait", failing_wait)
    with pytest.raises(RuntimeError):
        cores.run([sys.executable, "-c", _LEAVES_A_GRANDCHILD, str(pid_file)], want=2,
                  keep_descendants=True, **wire["options"](FakeCounter(_grant(2))))
    grandchild = opened[0]
    try:
        kernel, handle = grandchild
        assert kernel.WaitForSingleObject(handle, _DIES_MS) == 0, \
            "a run that failed left the child's tree running"
    finally:
        _reap(grandchild)


@windows_only
def test_the_command_line_opts_in_to_keeping_descendants(monkeypatch):
    seen = []
    monkeypatch.setattr(cores, "run", lambda *a, **k: seen.append(k["keep_descendants"]) or 0)
    cores.main(["run", "--want", "2", "--", "x"])
    cores.main(["run", "--want", "2", "--keep-descendants", "--", "x"])
    assert seen == [False, True]


@windows_only
@pytest.mark.parametrize("script, disarmed", [([3], ["disarm", "close"]),
                                              ([KeyboardInterrupt, KeyboardInterrupt, 1],
                                               ["terminate", "close"])],
                         ids=["normal exit", "ended by a second Ctrl+C"])
def test_only_a_child_that_exits_on_its_own_disarms_the_job(wire, script, disarmed):
    calls = []

    class FakeJob:
        name = "Local\\fake"

        def start(self, argv, env, popen):
            return _Child(script)

        def disarm(self):
            calls.append("disarm")

        def terminate(self):
            calls.append("terminate")

        def close(self):
            calls.append("close")

    cores.run(["x"], want=2, make_job=lambda say: FakeJob(), keep_descendants=True,
              **wire["options"](FakeCounter(_grant(2))))
    assert calls == disarmed


@windows_only
def test_a_job_the_counter_cannot_open_is_said(wire):
    counter = FakeCounter(_grant(2, measured=False))
    with cores.lease(2, 1, "blind", job="Local\\x", **wire["options"](counter)) as granted:
        assert granted == 2
    assert wire["lines"] == ["2 of 16 for blind (asked 2, min 1; free 2, foreign 0.0); its load is "
                             "not measured (the counter could not open its job)"]


@windows_only
def test_the_lease_is_held_for_the_childs_whole_life(wire):
    counter = FakeCounter(_grant(4))
    with cores.lease(4, 1, "held", **wire["options"](counter)) as granted:
        assert granted == 4 and not counter.closed
    assert counter.closed


@windows_only
def test_a_queued_lease_says_it_waits_then_takes_the_grant(wire):
    counter = FakeCounter({"op": "queued", "ahead": 1, "free": 1, "total": 16}, None, None,
                          _grant(3))
    with cores.lease(4, 3, "waiter", wait_s=60, **wire["options"](counter)) as granted:
        assert granted == 3
    assert wire["lines"][0] == "waiting for 3 core(s) for waiter (free 1 of 16, 1 ahead), up to 60 s"
    assert len(wire["lines"]) == 2


@windows_only
def test_a_forced_grant_says_it_runs_over_the_free_cores(wire):
    """Over the FREE cores, not "the machine": 2 forced beside 1 free on 8 is
    an over-commit of the budget, which another run's load may explain."""
    counter = FakeCounter(_grant(2, total=8, free=1, forced=True, waited_s=300.2))
    with cores.lease(8, 2, "late", **wire["options"](counter)) as granted:
        assert granted == 2
    assert wire["lines"] == ["waited 300 s; late runs with its min 2 anyway, over the 1 free of 8"]


@windows_only
def test_a_failed_negotiation_closes_the_pipe_before_the_child_runs(wire):
    """A request the counter still queues would otherwise be granted LATE,
    holding cores for a run already going on without them."""
    counter = FakeCounter({"op": "error", "error": "bad-lease"})
    closed_at_start = []

    def popen(argv, **kwargs):
        closed_at_start.append(counter.closed)
        return _Child([0])

    assert cores.run(["x"], want=2, popen=popen, make_job=lambda say: None,
                     **wire["options"](counter)) == 0
    assert closed_at_start == [True]


class _Child:
    """A child whose `wait` raises or returns from a script, one per call."""

    def __init__(self, script):
        self.script = list(script)
        self.killed = 0

    def wait(self):
        step = self.script.pop(0)
        if step is KeyboardInterrupt:
            raise KeyboardInterrupt
        return step

    def kill(self):
        self.killed += 1


@windows_only
def test_ctrl_c_while_queued_starts_nothing_and_holds_nothing(wire):
    class Interrupting(FakeCounter):
        def poll(self):
            raise KeyboardInterrupt

    counter = Interrupting()
    started = []
    code = cores.run(["x"], want=2, popen=lambda *a, **k: started.append(a),
                     make_job=lambda say: None, **wire["options"](counter))
    assert code == cores.INTERRUPTED_EXIT and not started and counter.closed
    assert wire["lines"][-1] == "interrupted while waiting for cores; x did not start"


@windows_only
def test_repeated_ctrl_c_ends_the_child_before_the_lease_goes(wire):
    """The first Ctrl+C is the child's; the second ends it; the wait goes on
    through a third, and the lease is returned only once the child is reaped."""
    counter = FakeCounter(_grant(2))
    child = _Child([KeyboardInterrupt, KeyboardInterrupt, KeyboardInterrupt, 1])
    open_while_waiting = []
    original_wait = child.wait

    def wait():
        open_while_waiting.append(not counter.closed)
        return original_wait()

    child.wait = wait
    code = cores.run(["x"], want=2, popen=lambda *a, **k: child, make_job=lambda say: None,
                     **wire["options"](counter))
    assert code == 1 and child.killed == 1
    assert open_while_waiting == [True] * 4 and counter.closed


@windows_only
@pytest.mark.parametrize("answers, reason", [
    (({"op": "error", "error": "unknown-op", "got": "lease"},),
     "the counter refused the lease (unknown-op, lease)"),
    ((cores.Closed,), "the counter closed the connection without a grant (a build without leases?)"),
    (({"op": "grant", "cores": 0},), "unexpected answer from the counter: {'op': 'grant', 'cores': 0}"),
])
def test_an_old_or_refusing_counter_is_no_broker_and_one_line(wire, tmp_path, answers, reason):
    """A counter without `lease` (refusing by name, or a pre-v2 build cutting
    the pipe) leaves the child its own defaults: LLM_CORES stays unset."""
    counter = FakeCounter(*answers)
    seen = tmp_path / "seen"
    code = cores.run([sys.executable, "-c",
                      f"import os; open({str(seen)!r}, 'w').write(os.environ.get('{cores.LEASE_ENV}', 'unset'))"],
                     want=4, label="old", **wire["options"](counter))
    assert code == 0 and seen.read_text() == "unset"
    assert wire["lines"] == [f"no core broker ({reason}); old runs with its own defaults "
                             f"({os.cpu_count()} cores on this machine)"]


@windows_only
def test_a_hung_counter_costs_its_deadline_then_falls_back(wire, monkeypatch):
    clock = iter(range(0, 1000, 10))
    counter = FakeCounter()
    options = wire["options"](counter)
    monkeypatch.setattr(cores, "negotiate", lambda *a, **k: _negotiate(*a, **k, clock=lambda: next(clock),
                                                                         sleep=lambda s: None))
    with cores.lease(4, 1, "hung", wait_s=20, **options) as granted:
        assert granted is None
    assert wire["lines"][0].startswith(f"no core broker (no grant within {20 + cores.ANSWER_SLACK_S:.0f} s)")


_negotiate = cores.negotiate


def test_without_a_counter_the_run_goes_on_with_one_line(wire, monkeypatch, tmp_path):
    """Not built: `_open_or_start` complains once through the lease's sink,
    and that reason is the lease's one line."""
    monkeypatch.setattr(agent_counter, "_connect", lambda: None)
    monkeypatch.setattr(agent_counter, "counter_exe", lambda: None)
    monkeypatch.delenv(agent_counter.EXE_ENV, raising=False)
    lines = []
    with cores.lease(4, 1, "unbuilt", say=lines.append) as granted:
        assert granted is None
    if os.name == "nt":
        assert lines == [f"no core broker (no counter executable (the host policy names none)); "
                         f"unbuilt runs with its own defaults ({os.cpu_count()} cores on this machine)"]
    else:
        assert len(lines) == 1 and "Windows named pipe" in lines[0]


def test_a_disabled_counter_never_touches_the_wire(monkeypatch):
    monkeypatch.delenv(cores.LEASE_ENV, raising=False)
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "0")

    def explode(complain):
        raise AssertionError("a disabled broker opened the pipe")

    lines = []
    with cores.lease(4, 1, "off", say=lines.append, connect=explode) as granted:
        assert granted is None
    assert len(lines) == 1 and lines[0].startswith("no core broker (")


def test_a_nested_run_takes_the_outer_lease_without_a_new_one(monkeypatch):
    monkeypatch.setenv(cores.LEASE_ENV, "3")

    def explode(complain):
        raise AssertionError("a nested lease asked the broker again")

    lines = []
    with cores.lease(8, 1, "inner", say=lines.append, connect=explode) as granted:
        assert granted == 3
    assert lines == ["inside a lease of 3: inner takes 3 of it without a new one "
                     "(inner runs side by side share it)"]
    # Side by side, each inner run is offered the same outer K: sharing it is
    # the caller's division (module docstring), and the line says so.
    with cores.lease(2, 1, "small", say=lines.append, connect=explode) as granted:
        assert granted == 2


@windows_only
def test_the_child_exit_code_is_forwarded_without_a_broker(monkeypatch):
    monkeypatch.delenv(cores.LEASE_ENV, raising=False)
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "0")
    assert cores.run([sys.executable, "-c", "raise SystemExit(5)"], want=2, say=lambda line: None) == 5


def test_a_body_exception_propagates_and_still_returns_the_lease(wire):
    if os.name != "nt":
        pytest.skip("the broker speaks a Windows named pipe")
    counter = FakeCounter(_grant(2))
    with pytest.raises(RuntimeError):
        with cores.lease(2, 1, "boom", **wire["options"](counter)):
            raise RuntimeError("the body failed")
    assert counter.closed


@pytest.mark.parametrize("argv", [
    ["run", "--want", "0", "--", "x"],
    ["run", "--want", "2", "--min", "3", "--", "x"],
    ["run", "--want", "2"],
])
def test_the_command_line_refuses_an_impossible_lease(argv, capsys):
    with pytest.raises(SystemExit) as exit_:
        cores.main(argv)
    assert exit_.value.code == 2
