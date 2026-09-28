"""One owner per terminal resource: the workers' console lines and the pinned rows.

Three layers, pinned in that order:

  * `ownership.OwnerThread` itself — order, drain, close, the bounded queue,
    the idle hook and what a failing call costs;
  * the parallel runner's console lines — every write on the one owner thread,
    never two at once, all of them on screen before the run reports;
  * the status line — while the painter runs, nobody else renders or writes
    the terminal or touches the mode stack; everybody else's paint is a
    request, and a key, a resize or a `disable` is a call posted to it.
"""

import sys
import threading
import time

import pytest

from llm_loop import ownership, parallel, termio
from llm_loop import statusline as sl
from llm_loop.breakpoints import Breakpoints

from _runfixtures import MemListDriver, isolated_run, par_args
from _termfixtures import KeysByHand, RecordingTerminal

# Upper bound on every wait below, and on every painter stall a pin arms (so a
# pin that fails by hanging still ends). Each wait is a handshake a healthy run
# completes at once, so only a broken owner ever gets near it. 0.49 s for this
# whole file, measured 2026-09-25 — the bound is many times that.
WAIT_S = 10.0


@pytest.fixture(autouse=True)
def _isolated_run(tmp_path, monkeypatch):
    with isolated_run(monkeypatch, tmp_path):
        yield


# --- the owner itself -----------------------------------------------------------


def test_calls_from_many_posters_run_on_one_thread_in_each_posters_order():
    owner = ownership.OwnerThread("pin-owner").start()
    ran = []

    def record(poster, n):
        ran.append((threading.current_thread().name, poster, n))

    def post_fifty(poster):
        for n in range(50):
            owner.post(record, poster, n)

    posters = [threading.Thread(target=post_fifty, args=(k,)) for k in range(4)]
    for thread in posters:
        thread.start()
    for thread in posters:
        thread.join(WAIT_S)
    assert owner.close(WAIT_S)

    assert len(ran) == 200
    assert {name for name, _poster, _n in ran} == {"pin-owner"}
    for poster in range(4):
        assert [n for _name, p, n in ran if p == poster] == list(range(50))


def test_drain_waits_for_everything_posted_before_it():
    owner = ownership.OwnerThread("pin-owner").start()
    entered, release = threading.Event(), threading.Event()
    ran = []

    def blocked():
        entered.set()
        release.wait(WAIT_S)

    owner.post(blocked)
    owner.post(ran.append, "after")
    assert entered.wait(WAIT_S)
    assert not owner.drain(timeout=0.05), "drain returned past a call still running"
    release.set()
    assert owner.drain(WAIT_S)
    assert ran == ["after"]
    assert owner.close(WAIT_S)


def test_close_runs_the_backlog_then_hands_the_resource_back():
    owner = ownership.OwnerThread("pin-owner").start()
    ran = []
    for n in range(20):
        owner.post(lambda n=n: ran.append((threading.current_thread().name, n)))

    assert owner.close(WAIT_S)
    assert ran == [("pin-owner", n) for n in range(20)]

    # No owner any more: the call is the caller's, and so is its exception.
    owner.post(lambda: ran.append((threading.current_thread().name, "late")))
    assert ran[-1] == (threading.current_thread().name, "late")
    with pytest.raises(ValueError):
        owner.post(int, "not a number")


def test_a_failing_call_costs_only_itself_and_is_reported_once(capsys):
    owner = ownership.OwnerThread("pin-owner").start()
    ran = []

    def refuse():
        raise BrokenPipeError("closed")

    owner.post(refuse)
    owner.post(refuse)
    owner.post(ran.append, "still running")
    assert owner.close(WAIT_S)

    assert ran == ["still running"]
    assert owner.failures == 2
    assert capsys.readouterr().err.count("pin-owner: BrokenPipeError: closed") == 1


def test_a_full_queue_makes_the_poster_wait_instead_of_growing():
    owner = ownership.OwnerThread("pin-owner", maxsize=1).start()
    entered, release = threading.Event(), threading.Event()
    ran = []

    def blocked():
        entered.set()
        release.wait(WAIT_S)

    owner.post(blocked)
    assert entered.wait(WAIT_S)          # off the queue, holding the owner
    owner.post(ran.append, 2)            # fills the one slot
    third_posted = threading.Event()

    def post_third():
        owner.post(ran.append, 3)
        third_posted.set()

    threading.Thread(target=post_third, daemon=True).start()
    assert not third_posted.wait(0.2), "a post went past a full queue"
    release.set()
    assert third_posted.wait(WAIT_S)
    assert owner.close(WAIT_S)
    assert ran == [2, 3]


class _Stall:
    """A posted call that holds the owner until released: a console nobody reads.

    Releases itself after WAIT_S, so a pin that fails by hanging still ends.
    """

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread = None

    def __call__(self, *_args):
        self.thread = threading.current_thread()
        self.entered.set()
        self.release.wait(WAIT_S)


def _returns_within(seconds, call):
    """(returned in time, its answer) — for a call that may hang the old way."""
    answer = []
    thread = threading.Thread(target=lambda: answer.append(call()), daemon=True)
    thread.start()
    thread.join(seconds)
    return not thread.is_alive(), (answer[0] if answer else None)


# What a bounded close/drain may take past its own 0.1 s timeout before the pin
# calls it unbounded. 0.10 s measured for both 2026-09-25; the unbounded version
# waits the whole WAIT_S for the stall to give up, so the two cannot be confused.
BOUND_SLACK_S = 2.0


@pytest.mark.parametrize("api", ["close", "drain"])
def test_close_and_drain_keep_their_timeout_over_a_full_queue_and_a_stuck_call(api):
    """A stuck resource may keep its backlog; it may not keep the caller.

    Both used to put their marker into the queue under the state lock and
    before their timed wait, so with the queue full the put blocked until the
    resource came back — `close(timeout=0.1)` waited as long as the console did.
    """
    owner = ownership.OwnerThread("pin-owner", maxsize=1).start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    owner.post(lambda: None)                  # fills the one slot
    try:
        returned, answer = _returns_within(
            0.1 + BOUND_SLACK_S, lambda: getattr(owner, api)(timeout=0.1))
        assert returned, f"{api}(timeout=0.1) waited for the stuck call"
        assert answer is False
    finally:
        stall.release.set()
    assert owner.close(WAIT_S)


def test_a_post_during_close_joins_the_owner_instead_of_running_beside_it():
    """Closing is not closed: the owner is still writing, so it still owns.

    A post that found the owner "closed" used to run inline on its caller —
    while the owner was still inside its backlog. A worker that outlives
    `INTERRUPT_JOIN_TIMEOUT_S` is exactly such a late producer.
    """
    owner = ownership.OwnerThread("pin-owner").start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    assert not owner.close(timeout=0.05)
    ran = []
    late = threading.Thread(
        target=lambda: owner.post(
            lambda: ran.append(threading.current_thread().name)),
        name="late-producer")
    late.start()
    late.join(WAIT_S)
    assert ran == [], "a post during close ran on its own thread, beside the owner"
    stall.release.set()
    assert owner.close(WAIT_S)
    assert ran == ["pin-owner"]


def test_opening_again_during_a_timed_out_close_keeps_the_one_owner():
    """A restart must not start a second owner over the same resource."""
    owner = ownership.OwnerThread("pin-owner").start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    assert not owner.close(timeout=0.05)

    owner.start()
    ran_on = []
    owner.post(lambda: ran_on.append(threading.current_thread()))
    stall.release.set()
    assert owner.drain(WAIT_S)
    assert ran_on == [stall.thread], "the reopened owner is a second thread"
    assert owner.close(WAIT_S)


def test_an_owner_that_handed_back_leaves_the_next_owner_open():
    """A restart right after a normal hand-back keeps its new owner.

    The old thread hands back in one locked step and re-checks in another on
    its way out; a `start()` between the two opens a NEW owner, which the old
    thread must not take for its own and close. Staged by releasing the lock
    right after the hand-back (a `wait` on the condition does exactly that)
    until the restart has happened.
    """
    owner = ownership.OwnerThread("pin-owner")
    first_hand_back = [True]
    real_close_locked = owner._close_locked

    def hand_back_then_let_a_restart_in():
        real_close_locked()
        if first_hand_back[0] and threading.current_thread().name == "pin-owner":
            first_hand_back[0] = False
            restart_may_go.set()
            owner._changed.wait_for(lambda: owner._state == ownership._OPEN,
                                    WAIT_S)

    restart_may_go = threading.Event()
    owner._close_locked = hand_back_then_let_a_restart_in
    owner.start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)          # owner is outside its state lock
    assert not owner.close(timeout=0)         # the hand-back runs on the owner
    stall.release.set()
    assert restart_may_go.wait(WAIT_S)
    owner.start()
    with owner._changed:                      # a fresh start notifies nobody
        owner._changed.notify_all()
    old = [t for t in threading.enumerate() if t.name == "pin-owner"]
    for thread in old:
        if thread is not owner._thread:
            thread.join(WAIT_S)

    ran_on = []
    owner.post(lambda: ran_on.append(threading.current_thread()))
    assert owner.drain(WAIT_S)
    assert ran_on and ran_on[0] is not threading.current_thread(), \
        "the old owner's exit closed the new one: the post ran on its caller"
    assert owner.close(WAIT_S)


def test_close_asked_again_answers_for_the_thread_it_left_running():
    owner = ownership.OwnerThread("pin-owner").start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)

    assert not owner.close(timeout=0.05)
    assert not owner.close(timeout=0.05), \
        "a repeated close reported the owner gone while it was still running"
    stall.release.set()
    assert owner.close(WAIT_S)
    assert owner.close(0)                     # and stays answered


def test_a_call_that_raises_system_exit_does_not_end_the_owner(capsys):
    """A dead owner is a queue nobody empties: posters would block for ever."""
    owner = ownership.OwnerThread("pin-owner", maxsize=1).start()
    ran = []
    owner.post(sys.exit, 3)

    def post_three():
        for n in range(3):
            owner.post(lambda n=n: ran.append((threading.current_thread().name, n)))

    returned, _ = _returns_within(WAIT_S, post_three)
    assert returned, "posting into the owner blocked after a call raised SystemExit"
    assert owner.close(WAIT_S)
    assert ran == [("pin-owner", n) for n in range(3)]
    assert owner.failures == 1
    assert "pin-owner: SystemExit: 3" in capsys.readouterr().err


# The owner's death is staged on purpose; its traceback is the fixture's.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_an_owner_that_dies_anyway_hands_the_resource_back():
    """The second net under the one above: whatever gets past `_invoke`.

    Staged by replacing `_invoke` itself, since nothing a posted call can raise
    gets past the real one any more.
    """
    owner = ownership.OwnerThread("pin-owner", maxsize=1)

    def die(call, args):
        raise SystemExit("the owner itself died")

    owner._invoke = die
    owner.start()
    thread = owner._thread
    owner.post(lambda: None)
    thread.join(WAIT_S)
    assert not thread.is_alive()
    ran = []

    def post_three():
        for n in range(3):
            owner.post(ran.append, n)

    returned, _ = _returns_within(WAIT_S, post_three)
    assert returned, "posts queued into an owner that had died"
    assert ran == [0, 1, 2]


def test_a_call_that_posts_from_the_owner_with_the_queue_full_runs_inline():
    """The owner cannot wait for room in a queue only it empties."""
    owner = ownership.OwnerThread("pin-owner", maxsize=1).start()
    entered, full = threading.Event(), threading.Event()
    ran = []

    def nests():
        entered.set()
        full.wait(WAIT_S)
        owner.post(ran.append, "nested")
        ran.append("outer done")

    owner.post(nests)
    assert entered.wait(WAIT_S)
    owner.post(ran.append, "queued")          # the one slot
    full.set()
    returned, closed = _returns_within(WAIT_S, lambda: owner.close(WAIT_S))
    assert returned and closed, "the owner deadlocked on its own post"
    assert ran == ["nested", "outer done", "queued"]


class _Idle:
    """An `idle` hook that answers `delays` in turn (then None) and records
    which thread called it and what had run by then."""

    def __init__(self, ran, *delays):
        self.ran = ran
        self.delays = list(delays)
        self.calls = []
        self.called = threading.Event()

    def __call__(self):
        self.calls.append((threading.current_thread().name, list(self.ran)))
        self.called.set()
        return self.delays.pop(0) if self.delays else None


def _wait_for(condition):
    deadline = time.monotonic() + WAIT_S
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    return condition()


def test_idle_runs_on_the_owner_once_the_queue_is_empty_then_on_its_own_delay():
    """Due after posted calls — never between two of them — and again after
    the delay it asked for; None waits for the next post."""
    ran = []
    idle = _Idle(ran, 0.01, 0.01)
    owner = ownership.OwnerThread("pin-owner", idle=idle).start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    for n in range(3):
        owner.post(ran.append, n)
    stall.release.set()
    # 1 after the batch, 2 and 3 on the delays it returned, then None.
    assert _wait_for(lambda: len(idle.calls) >= 3)
    assert idle.calls[0] == ("pin-owner", [0, 1, 2]), \
        "idle ran before the queue was empty"
    time.sleep(0.1)
    assert len(idle.calls) == 3, "idle ran again after answering None"
    owner.post(ran.append, 3)
    assert _wait_for(lambda: len(idle.calls) == 4)
    assert {name for name, _ran in idle.calls} == {"pin-owner"}
    assert owner.close(WAIT_S)


def test_idle_is_not_due_before_the_first_post_nor_while_closing():
    """What the hook works on is set up by a post and handed back by the
    backlog `close` runs: an idle pass on either side would touch a resource
    that is not (or no longer) there."""
    ran = []
    idle = _Idle(ran, 0.0, 0.0, 0.0)
    owner = ownership.OwnerThread("pin-owner", idle=idle).start()
    assert not idle.called.wait(0.1), "idle ran before anything was posted"
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    owner.post(ran.append, "last")
    assert not owner.close(timeout=0.05)
    stall.release.set()
    assert owner.close(WAIT_S)
    assert ran == ["last"]
    assert idle.calls == [], "idle ran on an owner that was closing"


def test_a_failing_idle_is_reported_and_due_again_after_the_next_post(capsys):
    ran = []

    def idle():
        ran.append("idle")
        raise BrokenPipeError("closed")

    owner = ownership.OwnerThread("pin-owner", idle=idle).start()
    owner.post(ran.append, 1)
    assert _wait_for(lambda: ran == [1, "idle"])
    owner.post(ran.append, 2)
    assert _wait_for(lambda: ran == [1, "idle", 2, "idle"])
    assert owner.close(WAIT_S)
    assert owner.failures == 2
    assert capsys.readouterr().err.count("pin-owner: BrokenPipeError: closed") == 1


def test_idle_is_not_due_in_a_reopened_window_before_its_first_post():
    """Reopened while closing, the owner still runs the last window's backlog
    first — and neither that backlog nor a delay the hook answered in the
    last window makes it due in the new one."""
    ran = []
    idle = _Idle(ran, 0.5)                 # due again 0.5 s after its first pass
    owner = ownership.OwnerThread("pin-owner", idle=idle).start()
    owner.post(ran.append, "a")
    assert _wait_for(lambda: len(idle.calls) == 1)
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    owner.post(ran.append, "old")          # last window's, behind the stall
    assert not owner.close(timeout=0.05)
    owner.start()                          # reopened while closing
    stall.release.set()
    assert owner.drain(WAIT_S)
    time.sleep(0.8)                        # past the 0.5 s the hook asked for
    assert len(idle.calls) == 1, "idle ran in the new window before its first post"
    owner.post(ran.append, "new")
    assert _wait_for(lambda: len(idle.calls) == 2)
    assert idle.calls[1] == ("pin-owner", ["a", "old", "new"])
    assert owner.close(WAIT_S)


@pytest.mark.parametrize("answer, failure", [(float("inf"), None),
                                             (10 ** 30, None),
                                             (float("nan"), "TypeError"),
                                             ("soon", "TypeError")])
def test_an_idle_answer_that_is_not_a_delay_cannot_end_the_owner(
        answer, failure, capsys):
    """`inf` reached `Condition.wait` as an OverflowError and ended the owner,
    after which every post ran on its caller. Too long a delay is clamped to
    IDLE_DELAY_MAX; what is not a delay at all is the hook's failure."""
    answered = threading.Event()

    def idle():
        answered.set()
        return answer

    owner = ownership.OwnerThread("pin-owner", idle=idle).start()
    owner.post(lambda: None)
    assert answered.wait(WAIT_S)
    time.sleep(0.05)                       # for the owner to act on the answer
    ran_on = []
    owner.post(lambda: ran_on.append(threading.current_thread().name))
    assert owner.drain(WAIT_S)
    assert ran_on == ["pin-owner"], "the owner died of its idle hook's answer"
    assert owner.close(WAIT_S)
    err = capsys.readouterr().err
    if failure is None:
        assert owner.failures == 0 and err == ""
    else:
        assert owner.failures >= 1
        assert f"pin-owner: {failure}: idle answered" in err


def test_post_with_a_timeout_gives_up_on_a_full_queue_and_queues_nothing():
    owner = ownership.OwnerThread("pin-owner", maxsize=1).start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    ran = []
    assert owner.post(ran.append, 1, timeout=0.1) is True    # the one slot
    try:
        returned, answer = _returns_within(
            0.1 + BOUND_SLACK_S, lambda: owner.post(ran.append, 2, timeout=0.1))
        assert returned, "post(timeout=0.1) waited for the stuck call"
        assert answer is False
    finally:
        stall.release.set()
    assert owner.close(WAIT_S)
    assert ran == [1], "a post that gave up was queued anyway"


def test_close_hands_back_past_a_full_queue_on_the_owner_after_the_backlog():
    """The hand-back may not wait for room behind a stuck resource.

    StatusApp.stop() used to `post` its release before `close(timeout)`: with
    the painter stuck and the queue full, that post blocked for as long as
    the terminal did, and the bound after it was never reached.
    """
    owner = ownership.OwnerThread("pin-owner", maxsize=1).start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    ran = []
    owner.post(lambda: ran.append(("queued", threading.current_thread().name)))
    try:
        returned, closed = _returns_within(
            0.1 + BOUND_SLACK_S, lambda: owner.close(
                timeout=0.1, final=lambda: ran.append(
                    ("final", threading.current_thread().name))))
        assert returned, "close(final=) waited for room in a full queue"
        assert closed is False and ran == []
    finally:
        stall.release.set()
    assert owner.close(WAIT_S)
    assert ran == [("queued", "pin-owner"), ("final", "pin-owner")]
    # No owner: `final` is the caller's, like a post.
    owner.close(final=lambda: ran.append(("late", threading.current_thread().name)))
    assert ran[-1] == ("late", threading.current_thread().name)


def test_start_sets_up_past_a_full_queue_behind_the_last_window():
    """The opening call joins a reopened owner behind the last window's
    backlog — and like the hand-back, never waits for room there."""
    owner = ownership.OwnerThread("pin-owner", maxsize=1).start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    ran = []
    owner.post(ran.append, "old")          # the one slot
    assert not owner.close(timeout=0.05)
    try:
        returned, _ = _returns_within(
            BOUND_SLACK_S, lambda: owner.start(first=lambda: ran.append("first")))
        assert returned, "start(first=) waited for room in a full queue"
    finally:
        stall.release.set()
    owner.post(ran.append, "new")
    assert owner.drain(WAIT_S)
    assert ran == ["old", "first", "new"]
    assert owner.close(WAIT_S)


# --- the parallel runner's console lines ----------------------------------------


class _WatchedConsole:
    """Stands in for `print_markup`: who wrote, and whether two ever overlapped.

    Also writes the plain line to stdout, so the run's own closing report (a
    plain `print`) lands in the same stream and the order of the two can be read.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.inside = 0
        self.max_inside = 0
        self.writers = set()
        self.lines = []

    def __call__(self, plain, markup):
        with self._lock:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
            self.writers.add(threading.current_thread().name)
        time.sleep(0.001)       # the window a second writer would need
        sys.stdout.write(plain + "\n")
        with self._lock:
            self.lines.append(plain)
            self.inside -= 1


def _chatty_jobs(jobs: int, lines: int):
    """A `run_job` stand-in whose workers all talk at once."""
    together = threading.Barrier(jobs, timeout=WAIT_S)

    def run_job(job_id, command, mailbox=None):
        together.wait()
        out = parallel.job_lines(job_id)
        for n in range(lines):
            out.line(f"{command.label} line {n}")
        return 0, 0.0, 0.01

    return run_job


def _run(tmp_path, items, jobs):
    return parallel.run_parallel(
        MemListDriver(items), par_args(tmp_path, jobs=jobs),
        app_name="pytest-output-owner", setup_logging=False,
        wait_on_start=False)


def test_worker_lines_are_written_by_one_thread_in_order_before_the_report(
        tmp_path, monkeypatch, capsys):
    console = _WatchedConsole()
    monkeypatch.setattr(parallel, "print_markup", console)
    monkeypatch.setattr(parallel, "run_job", _chatty_jobs(3, 20))

    result = _run(tmp_path, [f"products/f{i}.md" for i in range(3)], jobs=3)

    assert result.completed == 3
    assert console.writers == {"console-lines"}
    assert console.max_inside == 1, "two console writes overlapped"
    for i in range(3):
        mine = [line for line in console.lines if f"f{i}.md line " in line]
        assert [int(line.rsplit(" ", 1)[1]) for line in mine] == list(range(20))
    out = capsys.readouterr().out
    assert out.rindex(" line 19") < out.index("Processed 3 file(s)"), \
        "a worker line was still unwritten when the run reported"


@pytest.mark.filterwarnings("error::pytest.PytestUnhandledThreadExceptionWarning")
def test_a_console_that_refuses_every_line_costs_the_lines_not_the_run(
        tmp_path, monkeypatch, capsys):
    """The failure is the console's, so it may not end any worker.

    A worker used to write its own lines, and a closed pipe under one of them
    unwound it mid-file (see `test_parallel_termination`). The filter above is
    half the pin: a worker that dies of the refusal is a red test.
    """
    def refuse(plain, markup):
        raise BrokenPipeError("stdout closed")

    monkeypatch.setattr(parallel, "print_markup", refuse)
    monkeypatch.setattr(parallel, "run_job", _chatty_jobs(2, 5))

    result = _run(tmp_path, [f"products/f{i}.md" for i in range(4)], jobs=2)

    assert result.completed == 4
    assert capsys.readouterr().err.count(
        "console-lines: BrokenPipeError: stdout closed") == 1


def test_ctrl_c_over_a_stuck_console_still_stops_the_workers_and_reports(
        tmp_path, monkeypatch, capsys):
    """The interrupt may not wait for the console — and neither may the report.

    The Ctrl+C branch used to post its announcement BEFORE signalling anyone,
    and `run_parallel` closed the console with no timeout: with the queue full
    and the console stuck, Ctrl+C set nothing and the run never reached its
    epilogue. Staged with a one-slot queue so "full" is one line away.
    """
    owner = ownership.OwnerThread("console-lines", maxsize=1)
    monkeypatch.setattr(parallel, "_console", owner)
    monkeypatch.setattr(parallel, "INTERRUPT_JOIN_TIMEOUT_S", 0.1)
    monkeypatch.setattr(parallel, "CONSOLE_CLOSE_TIMEOUT_S", 0.1)
    stall = _Stall()
    monkeypatch.setattr(parallel, "print_markup", lambda plain, markup: stall())

    def run_job(job_id, command, mailbox=None):
        # The worker's own lines may fill the queue first; these make sure.
        out = parallel.job_lines(job_id)
        for n in range(3):
            out.line(f"line {n}")
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", run_job)
    runs = []
    real_shared = parallel.Shared

    class _Recorded(real_shared):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            runs.append(self)

    monkeypatch.setattr(parallel, "Shared", _Recorded)

    def interrupt(threads):
        # One line stuck in the console, one waiting in the one slot: full.
        deadline = time.monotonic() + WAIT_S
        while owner.backlog < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert owner.backlog == 2
        raise KeyboardInterrupt

    monkeypatch.setattr(parallel, "join_workers", interrupt)
    exit_codes = []

    def interrupted_run():
        try:
            _run(tmp_path, ["products/a.md"], jobs=1)
        except SystemExit as exc:
            exit_codes.append(exc.code)

    runner = threading.Thread(target=interrupted_run, daemon=True)
    try:
        runner.start()
        runner.join(WAIT_S)
        assert not runner.is_alive(), "Ctrl+C waited for a stuck console"
    finally:
        stall.release.set()
        owner.close(WAIT_S)

    assert exit_codes == [130], "the interrupted run never reached its epilogue"
    assert runs[0].stop.is_set(), "the workers were never told to stop"
    captured = capsys.readouterr()
    # No room in the queue: the line is deferred to the report, not dropped.
    assert parallel.INTERRUPT_ANNOUNCEMENT.strip() in captured.out
    assert "console-lines: 2 line(s) still unwritten" in captured.err


# --- the status line's painter --------------------------------------------------


def _paint_log():
    """A screenless terminal that records which thread wrote it (see
    `RecordingTerminal`), as wide as the Resize the key pin sends."""
    return RecordingTerminal(columns=100)


class _RecordingMode(sl.Mode):
    """Consumes every key and remembers which thread handled it."""

    def __init__(self, app):
        super().__init__(app)
        self.handled_on = []

    def handle(self, event):
        self.handled_on.append((threading.current_thread().name, event))
        return True


def test_while_started_only_the_painter_writes_the_terminal():
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    with app:
        # start() paints its first frame itself, before the painter exists.
        del terminal.writers[:]

        def feed(k):
            for n in range(20):
                app.note(f"note {k}-{n}")

        feeders = [threading.Thread(target=feed, args=(k,), name=f"feeder{k}")
                   for k in range(4)]
        for thread in feeders:
            thread.start()
        for thread in feeders:
            thread.join(WAIT_S)
        app.note("the LAST note")
        deadline = time.monotonic() + WAIT_S
        while "the LAST note" not in " ".join(terminal.frames.get(
                timeout=max(0, deadline - time.monotonic()))):
            pass
        writers = set(terminal.writers)

    assert writers == {sl.PAINTER_THREAD_NAME}


def test_without_a_painter_the_caller_paints():
    """Before start() nobody owns the terminal, and a title is due at once."""
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource())

    app.update(phase="running")

    assert terminal.writers == [threading.current_thread().name]


def test_while_started_keys_and_resizes_are_handled_on_the_painter():
    """The mode stack and the region are the painter's, like the terminal.

    A key used to be handled on the key reader's thread — mutating the modes
    while the painter read them — and a Resize re-pinned the region from there.
    """
    terminal, keys = _paint_log(), KeysByHand()
    app = sl.StatusApp(terminal=terminal, input_source=keys, refresh=60)
    recorder = _RecordingMode(app)
    app.push_mode(recorder)
    with app:
        del terminal.writers[:]       # start() pins and paints on its own thread
        keys.handler(termio.Key("x"))
        keys.handler(termio.Resize(100, 30))
        # From this thread it is handed over too, and waited for — after the
        # two posted before it, so everything above has run once it returns.
        app.handle_event(termio.Key("y"))
        handled = list(recorder.handled_on)
        writers = set(terminal.writers)

    assert handled == [(sl.PAINTER_THREAD_NAME, termio.Key("x")),
                       (sl.PAINTER_THREAD_NAME, termio.Key("y"))]
    assert writers == {sl.PAINTER_THREAD_NAME}, \
        "a key or a resize wrote the terminal off the painter"


def test_disabling_from_another_thread_is_carried_out_by_the_painter():
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    with app:
        app.disable()
        released_by = list(terminal.releases)

    assert released_by == [sl.PAINTER_THREAD_NAME]
    assert isinstance(app.terminal, termio.NullTerminal)


def test_stop_leaves_the_release_to_a_painter_stuck_in_its_frame(monkeypatch):
    """stop() gives up joining a stuck painter; it may not write beside it.

    It used to release the terminal from its own thread right after the join
    timed out — while the painter was still inside its frame, which then went
    on writing into a region that no longer existed.
    """
    monkeypatch.setattr(sl, "PAINTER_JOIN_SECONDS", 0.05)
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    app.start()
    painter = app._painter._thread
    terminal.arm_stall(WAIT_S)
    app.update(iteration=1)
    assert terminal.stalled.wait(WAIT_S)
    try:
        app.stop()
        assert painter.is_alive()
        assert terminal.releases == [], "stop() released under a live painter"
    finally:
        terminal.unstall.set()
    painter.join(WAIT_S)
    assert not painter.is_alive()
    assert terminal.releases == [sl.PAINTER_THREAD_NAME], \
        "the painter did not release the terminal on its way out"


def test_a_restart_after_a_timed_out_stop_keeps_the_one_painter(monkeypatch):
    """A restart reopens the painter the last stop() gave up on — the one
    restart policy, `ownership.OwnerThread.start` — and writes nothing beside it.

    It used to start a SECOND painter and pin the new region from its own
    thread while the old one was still inside its frame: that frame then
    landed over the new region, and with the write stuck inside the terminal's
    lock start() itself hung on it. (Before that, the one shared stop event was
    cleared and the old painter revived next to the new one.) Kept on, the old
    frame, the old release and the new region run in the order they were asked.
    """
    monkeypatch.setattr(sl, "PAINTER_JOIN_SECONDS", 0.05)
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    app.start()
    old = app._painter._thread
    terminal.arm_stall(WAIT_S)
    app.update(iteration=1)
    assert terminal.stalled.wait(WAIT_S)
    try:
        app.stop()
        assert old.is_alive()
        del terminal.writers[:]
        returned, _ = _returns_within(0.05 + BOUND_SLACK_S, app.start)
        assert returned, "start() waited for the stuck frame"
        assert terminal.writers == [], \
            "the restart wrote the terminal beside the painter still in its frame"
    finally:
        terminal.unstall.set()
    try:
        # Waited for, behind the old frame, the old release and the new region.
        app.handle_event(termio.Resize(terminal.columns, terminal.lines))
        app.update(iteration=2)
        deadline = time.monotonic() + WAIT_S
        while not any("iter 2" in row for row in terminal.frames.get(
                timeout=max(0, deadline - time.monotonic()))):
            pass
        assert app._painter._thread is old, "the restart started a second painter"
        assert set(terminal.writers) == {sl.PAINTER_THREAD_NAME}
        # The old stop's release ran before the new region was pinned.
        assert terminal.releases == [sl.PAINTER_THREAD_NAME] and terminal.active
    finally:
        app.stop()


def test_a_note_is_stamped_and_expired_by_the_painter(monkeypatch):
    """The note's clock has one writer: the painter, which is what expires it.

    `note()` used to write the timestamp from whichever thread called it while
    the painter read and reset it.
    """
    monkeypatch.setattr(sl, "NOTE_TTL", 0.05)
    app = sl.StatusApp(terminal=_paint_log(), input_source=termio.NullInputSource(),
                       refresh=0.01)
    stamped_on = []
    real_stamp = app._stamp_note

    def spy(stamp):
        stamped_on.append(threading.current_thread().name)
        real_stamp(stamp)

    app._stamp_note = spy
    with app:
        app.note("short-lived")
        deadline = time.monotonic() + WAIT_S
        while app.status.note and time.monotonic() < deadline:
            time.sleep(0.01)
        left = app.status.note

    assert left == "", "the note never expired"
    assert stamped_on == [sl.PAINTER_THREAD_NAME]


def test_a_paste_tail_posted_behind_its_enter_is_discarded_with_it():
    """Keys are posted, so the tail of a paste can be queued before its Enter
    is handled — and must go the way the reader's own discard sends the rest.

    The reader used to run the handler itself, so Enter's `discard_pending`
    stopped it mid-chunk; now the keys it already read sit in the painter's
    mailbox, and a pasted "…\\rs" would otherwise stop the run.
    """
    points = Breakpoints(lambda: "cleanup")
    terminal = _paint_log()
    reader = termio.TerminalInput()     # never started: its feed is driven here
    app = sl.StatusApp(terminal=terminal, input_source=reader, refresh=60)
    app.register_action(sl.BreakpointAction(points))
    with app:
        app.handle_event(termio.Key("b"))
        terminal.arm_stall(WAIT_S)
        app.update(iteration=1)
        assert terminal.stalled.wait(WAIT_S)     # the painter is held in a frame
        try:
            for char in "cleanup\rspm":
                reader._emit(app._handle_input, char)
        finally:
            terminal.unstall.set()
        app.handle_event(termio.Resize(100, 30))  # after everything posted above
        mode = app.mode

    assert points.names == ("cleanup",)
    assert isinstance(mode, sl.NormalMode)
    assert not app.stop_requested_here and not app.paused


def test_a_stuck_painter_with_a_full_queue_holds_neither_stop_nor_the_key_reader(
        monkeypatch, capsys):
    """Every wait on the painter keeps its bound, the wait for room included.

    stop() used to post its release before its bounded close: with the frame
    stuck and the queue full of keys, the post blocked for as long as the
    terminal did. So did the key reader's own post — and on Windows the
    reader is what turns Ctrl+C into an interrupt.
    """
    monkeypatch.setattr(sl, "PAINTER_JOIN_SECONDS", 0.05)
    monkeypatch.setattr(sl, "POSTED_CALL_WAIT_SECONDS", 0.05)
    terminal, keys = _paint_log(), KeysByHand()
    app = sl.StatusApp(terminal=terminal, input_source=keys, refresh=60)
    app.start()
    app._painter._maxsize = 3              # "full" a few keys away
    painter = app._painter._thread
    terminal.arm_stall(WAIT_S)
    app.update(iteration=1)
    assert terminal.stalled.wait(WAIT_S)   # a frame, not a queued call: empty
    try:
        def type_eight():
            for _ in range(8):
                keys.handler(termio.Key("x"))

        returned, _ = _returns_within(BOUND_SLACK_S, type_eight)
        assert returned, "the key reader waited for room behind a stuck frame"
        assert app._keys_dropped == 5
        returned, _ = _returns_within(
            0.05 + BOUND_SLACK_S, lambda: app.handle_event(termio.Key("y")))
        assert returned, "handle_event() waited for room behind a stuck frame"
        returned, _ = _returns_within(0.05 + BOUND_SLACK_S, app.stop)
        assert returned, "stop() waited for room behind a stuck frame"
        assert terminal.releases == [], "stop() released under a live painter"
    finally:
        terminal.unstall.set()
    painter.join(WAIT_S)
    assert not painter.is_alive()
    assert terminal.releases == [sl.PAINTER_THREAD_NAME], \
        "the release queued past the bound never ran"
    err = capsys.readouterr().err
    assert "handle_event (not queued, dropped) not done within" in err
    assert "the last frame and the release not done within" in err
    assert "5 key(s) dropped so far" in err


def test_a_resize_posted_behind_a_timed_out_stop_does_not_re_pin_the_region(
        monkeypatch):
    """What is posted after stop() still runs on the painter, after the
    release — and a Resize there used to reserve the region again, with
    nobody left to release it."""
    monkeypatch.setattr(sl, "PAINTER_JOIN_SECONDS", 0.05)
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    app.start()
    painter = app._painter._thread
    terminal.arm_stall(WAIT_S)
    app.update(iteration=1)
    assert terminal.stalled.wait(WAIT_S)
    resize = threading.Thread(
        target=app.handle_event,
        args=(termio.Resize(terminal.columns, terminal.lines),), daemon=True)
    try:
        app.stop()
        resize.start()
        # Queued behind the release: the release and the resize.
        assert _wait_for(lambda: app._painter.backlog == 2)
    finally:
        terminal.unstall.set()
    resize.join(WAIT_S)
    painter.join(WAIT_S)
    assert terminal.releases == [sl.PAINTER_THREAD_NAME]
    assert not terminal.active, "a Resize run after the release re-pinned the region"


def test_ctrl_c_while_start_waits_for_its_first_frame_puts_the_terminal_back():
    """`with app:` runs no `__exit__` for a start() that raised.

    A Ctrl+C in start()'s wait for the first frame left the painter open and
    the region pinned, with the signal restore not yet installed either.
    """
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    real_drain = app._painter.drain
    restore_installed = []

    def interrupted(timeout=None):
        real_drain(WAIT_S)         # the region is pinned: there is something to undo
        restore_installed.append(app._atexit_registered)
        raise KeyboardInterrupt

    app._painter.drain = interrupted
    with pytest.raises(KeyboardInterrupt):
        app.start()
    assert restore_installed == [True], \
        "a signal during start()'s wait found no restore installed"
    assert _wait_for(lambda: terminal.releases), "the region was never released"
    assert terminal.releases == [sl.PAINTER_THREAD_NAME]
    assert not terminal.active
    assert not app._atexit_registered


def test_a_burst_behind_a_stuck_frame_is_one_request_and_one_frame_of_its_end():
    """Every paint request asks for "the state as it is now", so one in flight
    is the whole queue, and the frame after the stall shows the burst's end."""
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    with app:
        terminal.arm_stall(WAIT_S)
        app.update(iteration=1)
        assert terminal.stalled.wait(WAIT_S)
        before = len(terminal.painted)
        try:
            def feed(k):
                for n in range(200):
                    app.update(iteration=k * 1000 + n)

            feeders = [threading.Thread(target=feed, args=(k,)) for k in range(4)]
            for thread in feeders:
                thread.start()
            for thread in feeders:
                thread.join(WAIT_S)
            app.update(iteration=999999)
            queued = app._painter.backlog
        finally:
            terminal.unstall.set()
        deadline = time.monotonic() + WAIT_S
        while not any("iter 999999" in row for row in terminal.frames.get(
                timeout=max(0, deadline - time.monotonic()))):
            pass
        frames = len(terminal.painted) - before
        flag = app._frame_posted

    assert queued == 1, f"801 updates queued {queued} frame requests"
    assert frames == 2, f"the stuck frame and one more expected, got {frames}"
    assert flag is False


def test_a_frame_request_lost_before_the_queue_does_not_stop_the_frames():
    """The one-in-flight flag, set by a request that never got queued (its
    poster interrupted in between), used to silence every later paint."""
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    with app:
        assert app._painter.drain(WAIT_S)
        app._frame_posted = True           # as that poster left it
        app.update(iteration=7)
        deadline = time.monotonic() + WAIT_S
        while not any("iter 7" in row for row in terminal.frames.get(
                timeout=max(0, deadline - time.monotonic()))):
            pass


class _StartStopInput(termio.NullInputSource):
    """Remembers the order it was started and stopped in."""

    def __init__(self):
        self.events = []

    def start(self, handler):
        self.events.append("start")

    def stop(self):
        self.events.append("stop")


def test_a_region_refused_behind_a_stuck_restart_leaves_no_key_reader(monkeypatch):
    """A refused region means no keys, as it always did at start() — also when
    the refusal lands after start() has given up waiting for it and started
    the reader."""
    monkeypatch.setattr(sl, "PAINTER_JOIN_SECONDS", 0.05)
    terminal, keys = _paint_log(), _StartStopInput()
    app = sl.StatusApp(terminal=terminal, input_source=keys, refresh=60)
    app.start()
    terminal.arm_stall(WAIT_S)
    app.update(iteration=1)
    assert terminal.stalled.wait(WAIT_S)
    try:
        app.stop()
        terminal.lines = 4                 # no room for the region any more
        app.start()                        # returns with the region undecided
    finally:
        terminal.unstall.set()
    try:
        assert _wait_for(lambda: isinstance(app.terminal, termio.NullTerminal))
        assert _wait_for(lambda: keys.events == ["start", "stop", "start", "stop"]), \
            f"the reader outlived the refused region: {keys.events}"
    finally:
        app.stop()
