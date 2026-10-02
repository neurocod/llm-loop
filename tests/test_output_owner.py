"""One owner per terminal resource: the workers' console lines and the pinned rows.

Three layers, pinned in that order:

  * `ownership.OwnerThread` itself — order, drain, close, the bounded queue,
    the idle hook and what a failing call costs;
  * the parallel runner's console lines — every write on the one owner thread,
    never two at once, all of them on screen before the run reports; the lines
    of the threads the runner does not own (the usage gate, the pusher) reach
    that thread through `console.route_through`;
  * the status line — while the painter runs, nobody else renders or writes
    the terminal or touches the mode stack; everybody else's paint is a
    request, and a key, a resize or a `disable` is a call posted to it.
"""

import subprocess
import sys
import threading
import time

import pytest

from llm_loop import console, gitpush, ownership, parallel, termio, usage
from llm_loop import statusline as sl
from llm_loop.breakpoints import Breakpoints
from llm_loop.limits import LimitPolicy, SessionLimit

from _runfixtures import (MemListDriver, isolated_run, par_args,
                          record_exit_pushes)
from _termfixtures import KeysByHand, RecordingTerminal

# Upper bound on every wait below but the frame waits (those take
# `_termfixtures.FRAME_WAIT_S`), and on every painter stall a pin arms (so a
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


def _stuck_and_full(name):
    """(owner, stall): an owner held by `stall`, its one queue slot taken —
    a console that has stopped, until `stall.release` is set."""
    owner = ownership.OwnerThread(name, maxsize=1).start()
    stall = _Stall()
    owner.post(stall)
    assert stall.entered.wait(WAIT_S)
    owner.post(lambda: None)                  # fills the one slot
    return owner, stall


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
    owner, stall = _stuck_and_full("pin-owner")
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


class _HeldWake(threading.Condition):
    """The owner's condition, able to hold the owner inside a wait.

    While `hold` is set, the owner's wait does not return — woken or timed out
    — until the test clears it, the lock released all along. That stages "a
    close and a reopen both landed while the owner slept" without a race: a
    real `Condition.wait` is free to return that late. `asleep` is set as the
    owner enters a wait, under the lock, so a test that sees it and then takes
    the lock knows the owner is inside that wait.
    """

    def __init__(self, lock):
        super().__init__(lock)
        self.hold = threading.Event()
        self.asleep = threading.Event()

    def wait(self, timeout=None):
        if threading.current_thread().name != "pin-owner":
            return super().wait(timeout)
        self.asleep.set()
        woke = super().wait(timeout)
        deadline = time.monotonic() + WAIT_S
        while self.hold.is_set() and time.monotonic() < deadline:
            super().wait(0.005)
        return woke


def _sleeping_owner(delay):
    """An open owner asleep after one post and one idle pass that answered
    `delay`, held there (see `_HeldWake`); (owner, its condition, ran, idle
    calls)."""
    ran, calls, idled = [], [], threading.Event()

    def idle():
        calls.append(list(ran))
        if len(calls) == 1:
            held.hold.set()             # every wake from here on waits for us
            held.asleep.clear()         # and the next wait is the one we want
            idled.set()
        return delay

    owner = ownership.OwnerThread("pin-owner", idle=idle)
    owner._changed = held = _HeldWake(owner._lock)
    owner.start()
    owner.post(ran.append, "a")
    # The owner may already sleep awaiting this post, and that sleep sets
    # `asleep` too (seen on windows-latest CI): only after the idle pass has
    # cleared it is `asleep` the sleep we want.
    assert idled.wait(WAIT_S)
    assert held.asleep.wait(WAIT_S)
    return owner, held, ran, calls


def test_a_reopen_with_a_first_call_while_the_owner_sleeps_makes_idle_due():
    """Close and reopen both land in one sleep, and the new window's `first`
    is the owner's next call: it makes idle due like any call of the window.

    The owner used to ask which window it was in once per call, before its
    wait: it ran `first` still counting the last window, marked idle due,
    then noticed the new window and dropped that — no idle until a post.
    """
    owner, held, ran, calls = _sleeping_owner(None)
    assert not owner.close(timeout=0)
    owner.start(first=lambda: ran.append("first"))
    held.hold.clear()
    assert _wait_for(lambda: len(calls) == 2), \
        "idle was not due after the reopened window's first call"
    assert calls[1] == ["a", "first"]
    assert owner.close(WAIT_S)


# How long past the old window's idle delay the pin below watches for a pass
# that must not come. Under the fix nothing ever comes, so no value can turn it
# red; it only has to outlast the 0.2 s delay for the unfixed owner to show.
_STALE_DELAY_S = 0.2
_WATCH_S = 0.6


def test_a_reopen_while_the_owner_sleeps_drops_the_last_windows_idle_delay():
    """Close and reopen with nothing posted, both in one sleep: the delay the
    hook answered in the last window does not make it due in this one.

    The owner used to wake, find the queue still empty and go on waiting for
    the old delay — then run idle in a window nothing had been posted in.
    """
    owner, held, ran, calls = _sleeping_owner(_STALE_DELAY_S)
    assert not owner.close(timeout=0)
    owner.start()
    held.hold.clear()
    time.sleep(_WATCH_S)
    assert len(calls) == 1, "idle ran in the new window before its first post"
    owner.post(ran.append, "new")
    assert _wait_for(lambda: len(calls) >= 2)
    assert calls[1] == ["a", "new"]
    assert owner.close(WAIT_S)


@pytest.mark.parametrize("answer, failure", [(float("inf"), None),
                                             (10 ** 30, None),
                                             (10 ** 1000, None),
                                             (float("nan"), "TypeError"),
                                             ("soon", "TypeError")],
                         ids=["inf", "1e30", "1e1000", "nan", "str"])
def test_an_idle_answer_that_is_not_a_delay_cannot_end_the_owner(
        answer, failure, capsys):
    """`inf` reached `Condition.wait` as an OverflowError and ended the owner,
    after which every post ran on its caller; `10**1000` did the same one step
    earlier, in `float()`, outside the net a posted call runs in. Too long a
    delay is clamped to IDLE_DELAY_MAX; what is not a delay at all is the
    hook's failure."""
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
    assert console._route is None, "the interrupted run left the console routed"


# --- the lines of threads the runner does not own: console.route_through --------


class _ThreadedStdout:
    """A stdout that remembers which thread made each write.

    Installed by `_as_plain_stdout`, with rich switched off, so a console line
    is one write of its plain copy — which is what is read here.
    """

    encoding = "utf-8"

    def __init__(self):
        self._lock = threading.Lock()
        self.writes = []

    def write(self, text):
        with self._lock:
            self.writes.append((threading.current_thread().name, text))
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False

    def lines(self):
        """(thread, text) of every write that is a line, not a bare newline —
        its own newline dropped (a `console` line is one write, newline
        included; a bare `print` makes two). Blind to a line split by another
        write: that is what `screen` is for."""
        with self._lock:
            return [(name, text[:-1] if text.endswith("\n") else text)
                    for name, text in self.writes if text != "\n"]

    def screen(self):
        """Everything written, in order: what the terminal shows."""
        with self._lock:
            return "".join(text for _name, text in self.writes)


def _as_plain_stdout(monkeypatch, out):
    """`out` installed as `sys.stdout`, rich off; returns it."""
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(console, "RICH_AVAILABLE", False)
    return out


class _OnePushGit:
    """`gitpush.subprocess` for a pusher that pushes exactly once, when let.

    `rev-list` answers one commit ahead until the push has happened and none
    after it, so the pump's later turns find nothing to push; `git push` waits
    for `let_push`, which is what puts the pusher's line behind the worker's.
    """

    PIPE = subprocess.PIPE
    STDOUT = subprocess.STDOUT
    TimeoutExpired = subprocess.TimeoutExpired
    CompletedProcess = subprocess.CompletedProcess

    def __init__(self):
        self.let_push = threading.Event()
        self.pushed = False

    def Popen(self, argv, **kwargs):
        # The pump's checks carry the run's `PushAbort`, and a git call with
        # one is started through `Popen` (see `gitpush._run_git`).
        return _OnePushProcess(self, argv)

    def run(self, argv, **kwargs):
        if tuple(argv)[:2] == ("git", "push"):
            assert self.let_push.wait(WAIT_S), "the pin never let the pusher push"
            self.pushed = True
            return subprocess.CompletedProcess(argv, 0, stdout="")
        return subprocess.CompletedProcess(argv, 0,
                                           stdout="0" if self.pushed else "1")


class _OnePushProcess:
    """What `_OnePushGit.Popen` starts: `communicate` makes the call."""

    def __init__(self, git, argv):
        self._git = git
        self._argv = argv
        self.returncode = None

    def communicate(self, timeout=None):
        done = self._git.run(self._argv)
        self.returncode = done.returncode
        return done.stdout, None

    def kill(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _OverTheCeiling:
    """A usage source whose session is over any ceiling the pin sets."""

    def get_usage(self, cache_value=True):
        return usage.parse_usage({"five_hour": {"utilization": 90.0}})

    def invalidate(self):
        pass


def test_quota_and_pusher_lines_are_written_by_the_owner_after_queued_worker_lines(
        tmp_path, monkeypatch):
    """The gate and the pusher print through `console`, not through `_console`.

    Both used to write on their own threads while worker lines were still
    queued, so a quota pause or a push could appear ahead of lines said before
    it, and between two writes of the owner. Staged with the owner held by a
    stalled write: whatever is queued behind it is written only once it is
    released, so a line written EARLIER than the worker lines, or by any other
    thread, was not routed.

    The gate is the real `LimitPolicy`, over its ceiling and told to stop at
    once, so it prints all three of its kinds of line: the percentage reading
    (`print_percents`) and the two plain ones of the hold (`print_line`). Their
    own order is part of the pin — routing only the first would reorder them.
    """
    out = _as_plain_stdout(monkeypatch, _ThreadedStdout())
    record_exit_pushes(monkeypatch)
    git = _OnePushGit()
    monkeypatch.setattr(gitpush, "subprocess", git)
    monkeypatch.setattr(parallel, "PUSH_PUMP_INTERVAL_S", 0.01)
    stall = _Stall()
    reached = []

    def run_job(job_id, command, mailbox=None):
        parallel._console.post(stall)
        assert stall.entered.wait(WAIT_S)
        try:
            lines = parallel.job_lines(job_id)
            for n in range(3):
                lines.line(f"worker line {n}")
            LimitPolicy([SessionLimit(5)]).check_and_wait(
                _OverTheCeiling(), time.time(), should_stop=lambda: True)
            git.let_push.set()
            # The stall, three worker lines, three gate lines, one push line —
            # a line that is not routed never arrives, and is asserted on below.
            reached.append(_wait_for(lambda: parallel._console.backlog >= 8))
        finally:
            stall.release.set()
        return 0, 0.0, 0.01

    monkeypatch.setattr(parallel, "run_job", run_job)

    result = parallel.run_parallel(
        MemListDriver(["products/a.md"]),
        par_args(tmp_path, jobs=1, git_push="after_new_commits"),
        app_name="pytest-output-owner", setup_logging=False,
        wait_on_start=False)
    console.print_markup("after the run", "after the run")

    assert result.completed == 1
    lines = out.lines()
    markers = ["worker line 0", "worker line 1", "worker line 2",
               "usage: 90%", "⏳ Over usage limit", "⏹ Stop requested",
               "git push: done"]
    found = []
    for marker in markers:
        hits = [(i, name) for i, (name, text) in enumerate(lines)
                if marker in text]
        assert len(hits) == 1, f"{marker!r} written {len(hits)} time(s): {lines}"
        found.append(hits[0])
    assert {name for _i, name in found} == {"console-lines"}, \
        f"a line was written off the owner: {list(zip(markers, found))}"
    assert [i for i, _name in found] == sorted(i for i, _name in found), \
        f"the lines were written out of the order they were said: {lines}"
    assert reached == [True]
    # The phase over, the route is gone and a line is its caller's again.
    assert lines[-1] == (threading.current_thread().name, "after the run")
    assert console._route is None


def test_a_route_whose_queue_stays_full_writes_the_line_itself_and_says_so(
        monkeypatch, capsys):
    """The blocking policy: a poster waits `post_timeout` for room, no longer.

    Past it the line is written on the poster's own thread — out of order, and
    counted on stderr when the window closes — rather than dropped, or waited
    for as long as the console stays stuck. Staged with a one-slot queue that
    a stalled write keeps full.
    """
    out = _as_plain_stdout(monkeypatch, _ThreadedStdout())
    owner, stall = _stuck_and_full("pin-console")
    try:
        with console.route_through(owner, post_timeout=0.05):
            returned, _ = _returns_within(
                0.05 + BOUND_SLACK_S,
                lambda: console.print_markup("past a full queue", "x"))
    finally:
        stall.release.set()
        assert owner.close(WAIT_S)

    assert returned, "a poster waited for room past its post_timeout"
    written = out.lines()
    assert [text for _name, text in written] == ["past a full queue"]
    assert written[0][0] != "pin-console", "the line waited for the owner"
    assert "pin-console: 1 line(s) written directly" in capsys.readouterr().err


class _StallsTheOwnersWrite(_ThreadedStdout):
    """Holds the first write made on `owner`'s thread — after recording it —
    until `release`: a console that froze in the middle of the owner's line."""

    def __init__(self, owner):
        super().__init__()
        self._owner = owner
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, text):
        written = super().write(text)
        if (threading.current_thread().name == self._owner
                and not self.entered.is_set()):
            self.entered.set()
            self.release.wait(WAIT_S)
        return written


def test_a_line_written_past_a_stuck_route_never_lands_inside_the_owners(
        monkeypatch, capsys):
    """A direct write lands between two of the owner's lines, not inside one.

    `print` writes a line's text and its newline in two calls; with the owner
    frozen between them, the poster's line used to land in the gap and the
    screen read "queueddirect" followed by two newlines.
    """
    out = _as_plain_stdout(monkeypatch, _StallsTheOwnersWrite("pin-console"))
    owner = ownership.OwnerThread("pin-console", maxsize=1).start()
    try:
        with console.route_through(owner, post_timeout=0.05):
            console.print_line("queued")
            assert out.entered.wait(WAIT_S)
            owner.post(lambda: None)            # the one slot: full
            poster = threading.Thread(target=console.print_line,
                                      args=("direct",), name="pin-poster")
            poster.start()
            poster.join(WAIT_S)
            assert not poster.is_alive(), "the direct write waited for the owner"
    finally:
        out.release.set()
        assert owner.close(WAIT_S)

    assert out.screen() == "queued\ndirect\n"
    assert "pin-console: 1 line(s) written directly" in capsys.readouterr().err


def test_past_one_timeout_a_route_stops_waiting_for_room_until_there_is_some(
        monkeypatch, capsys):
    """A stuck console costs a route ONE `post_timeout`, not one per line.

    The gate prints several lines under `usage_lock`; each used to wait the
    whole timeout before writing itself. Past the first, a poster takes only
    room that is free at once — and waits again once a post has found some.
    """
    out = _as_plain_stdout(monkeypatch, _ThreadedStdout())
    owner, stall = _stuck_and_full("pin-console")
    waits = []
    real_post = owner.post

    def post(call, *args, timeout=None):
        waits.append(timeout)
        return real_post(call, *args, timeout=timeout)

    monkeypatch.setattr(owner, "post", post)
    try:
        with console.route_through(owner, post_timeout=0.05):
            for n in range(3):
                console.print_line(f"stuck {n}")
            stall.release.set()
            assert owner.drain(WAIT_S)
            console.print_line("room again")
            assert owner.drain(WAIT_S)          # room for the next, surely
            console.print_line("waited for again")
    finally:
        stall.release.set()
        assert owner.close(WAIT_S)

    assert waits == [0.05, 0, 0, 0, 0.05]
    assert [text for _name, text in out.lines()] == [
        "stuck 0", "stuck 1", "stuck 2", "room again", "waited for again"]
    assert "pin-console: 3 line(s) written directly" in capsys.readouterr().err


class _PostHeldUntilTheWindowCloses:
    """An owner whose post waits for `go`, then finds no room."""

    name = "pin-console"
    owns_current_thread = False

    def __init__(self):
        self.waiting = threading.Event()
        self.go = threading.Event()

    def post(self, call, *args, timeout=None):
        self.waiting.set()
        self.go.wait(WAIT_S)
        return False


def test_a_line_written_directly_after_the_window_closed_reports_itself(
        monkeypatch, capsys):
    """The close reports what was written directly by then; a poster still
    waiting at that moment is not in its count, and says so itself."""
    out = _as_plain_stdout(monkeypatch, _ThreadedStdout())
    owner = _PostHeldUntilTheWindowCloses()
    poster = threading.Thread(target=console.print_line, args=("late",),
                              name="pin-poster")
    with console.route_through(owner, post_timeout=WAIT_S):
        poster.start()
        assert owner.waiting.wait(WAIT_S)
    at_close = capsys.readouterr().err
    owner.go.set()
    poster.join(WAIT_S)

    assert at_close == "", "the close counted a line not yet written"
    assert out.lines() == [("pin-poster", "late")]
    assert ("pin-console: 1 line written directly, out of order, after the "
            "window closed") in capsys.readouterr().err


def test_one_route_at_a_time():
    owner = ownership.OwnerThread("pin-console")
    with console.route_through(owner, post_timeout=1.0):
        with pytest.raises(RuntimeError, match="already routed"):
            with console.route_through(owner, post_timeout=1.0):
                pass
        assert console._route is not None, "the refusal removed the open route"
    assert console._route is None


def test_a_phase_that_raises_leaves_the_console_unrouted(tmp_path, monkeypatch):
    """Whatever ends the worker phase early — a second Ctrl+C, a failing
    status line — the route is closed on the way out, not left behind to send
    the next run's lines to an owner nobody runs."""
    monkeypatch.setattr(parallel, "run_job",
                        lambda job_id, command, mailbox=None: (0, 0.0, 0.01))

    def blow_up(threads):
        # Joined first: a worker left running writes its first line after the
        # test, into whichever test's stdout is installed by then.
        for thread in threads:
            thread.join(WAIT_S)
        raise RuntimeError("the phase failed")

    monkeypatch.setattr(parallel, "join_workers", blow_up)

    with pytest.raises(RuntimeError, match="the phase failed"):
        _run(tmp_path, ["products/a.md"], jobs=1)

    assert console._route is None


# --- the status line's painter --------------------------------------------------


def _paint_log():
    """A screenless terminal that records which thread wrote it (see
    `RecordingTerminal`), as wide as the Resize the key pin sends."""
    return RecordingTerminal(columns=100)


def _hold_a_frame(app, terminal):
    """Return once the painter is held inside a frame: a terminal write that
    does not come back until `terminal.unstall` is set."""
    terminal.arm_stall(WAIT_S)
    app.update(iteration=1)
    assert terminal.stalled.wait(WAIT_S)


def _shows(text):
    """A `RecordingTerminal.wait_for_frame` predicate: `text` in any row. Its
    `what` names the wait in the failure message."""
    def shows(frame):
        return any(text in row for row in frame)

    shows.what = f"a row containing {text!r}"
    return shows


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
        terminal.wait_for_frame(_shows("the LAST note"))
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
    # push_mode() named the window on this thread: nobody owned the terminal
    # yet. From start() on, the region is pinned and painted by the painter.
    del terminal.writers[:]
    with app:
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


def test_the_painters_owner_only_calls_refuse_another_thread():
    """`app.painter` is public: calling its terminal writes directly from a
    worker must fail loudly, not write beside a live frame."""
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    with app:
        assert app.painter.drain(WAIT_S)
        for name, call in (("disable", app.painter.disable),
                           ("resize", app.painter.resize),
                           ("reserve", lambda: app.painter.reserve(3))):
            with pytest.raises(RuntimeError, match=rf"Painter\.{name}\(\)"):
                call()
        with pytest.raises(AttributeError):
            app.painter.terminal = termio.NullTerminal()
        assert terminal.releases == []
        assert app.terminal is terminal


def test_the_painter_calls_the_apps_methods_as_they_are_now():
    """A patch of `app.render` after construction reaches the painter."""
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    app.render = lambda width=None, now=None: ["patched render"]
    with app:
        terminal.wait_for_frame(_shows("patched render"))


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
    painter = app.painter.thread
    _hold_a_frame(app, terminal)
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
    old = app.painter.thread
    _hold_a_frame(app, terminal)
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
        terminal.wait_for_frame(_shows("iter 2"))
        assert app.painter.thread is old, "the restart started a second painter"
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
        _hold_a_frame(app, terminal)
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
    monkeypatch.setattr(sl, "PAINTER_QUEUE_MAXSIZE", 3)  # "full" a few keys away
    terminal, keys = _paint_log(), KeysByHand()
    app = sl.StatusApp(terminal=terminal, input_source=keys, refresh=60)
    app.start()
    painter = app.painter.thread
    _hold_a_frame(app, terminal)           # a frame, not a queued call: empty
    try:
        def type_eight():
            for _ in range(8):
                keys.handler(termio.Key("x"))

        returned, _ = _returns_within(BOUND_SLACK_S, type_eight)
        assert returned, "the key reader waited for room behind a stuck frame"
        assert app.painter.keys_dropped == 5
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
    painter = app.painter.thread
    _hold_a_frame(app, terminal)
    resize = threading.Thread(
        target=app.handle_event,
        args=(termio.Resize(terminal.columns, terminal.lines),), daemon=True)
    try:
        app.stop()
        resize.start()
        # Queued behind the release: the release and the resize.
        assert _wait_for(lambda: app.painter.backlog == 2)
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
    real_drain = app.painter.drain
    restore_installed = []

    def interrupted(timeout=None):
        real_drain(WAIT_S)         # the region is pinned: there is something to undo
        restore_installed.append(app._atexit_registered)
        raise KeyboardInterrupt

    app.painter.drain = interrupted
    with pytest.raises(KeyboardInterrupt):
        app.start()
    assert restore_installed == [True], \
        "a signal during start()'s wait found no restore installed"
    assert _wait_for(lambda: terminal.releases), "the region was never released"
    assert terminal.releases == [sl.PAINTER_THREAD_NAME]
    assert not terminal.active
    assert not app._atexit_registered


def test_ctrl_c_in_a_service_stop_still_stops_the_rest_and_releases_the_region():
    """Every step of stop() runs past an interrupted one, then it re-raises.

    A Ctrl+C in a service's join used to skip the later services, the key
    reader and the painter's close: the region stayed pinned with the emergency
    restore already removed, and a second stop() did nothing.
    """
    class Interrupted:
        def start(self):
            pass

        def stop(self):
            raise KeyboardInterrupt

    class Recorded:
        stopped = False

        def start(self):
            pass

        def stop(self):
            self.stopped = True

    class Keys(termio.NullInputSource):
        stopped = False

        def stop(self):
            self.stopped = True

    terminal = _paint_log()
    keys = Keys()
    app = sl.StatusApp(terminal=terminal, input_source=keys, refresh=60)
    app.add_service(Interrupted())
    later = app.add_service(Recorded())
    app.start()
    painter = app.painter.thread
    with pytest.raises(KeyboardInterrupt):
        app.stop()
    painter.join(WAIT_S)
    assert later.stopped, "a service after the interrupted one was never stopped"
    assert keys.stopped, "the key reader was never stopped"
    assert terminal.releases == [sl.PAINTER_THREAD_NAME], "the region was never released"
    assert not terminal.active
    assert not painter.is_alive(), "the painter was never closed"
    assert not app._atexit_registered


def test_a_burst_behind_a_stuck_frame_is_one_request_and_one_frame_of_its_end():
    """Every paint request asks for "the state as it is now", so one in flight
    is the whole queue, and the frame after the stall shows the burst's end."""
    terminal = _paint_log()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource(),
                       refresh=60)
    with app:
        _hold_a_frame(app, terminal)
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
            queued = app.painter.backlog
        finally:
            terminal.unstall.set()
        terminal.wait_for_frame(_shows("iter 999999"))
        frames = len(terminal.painted) - before
        flag = app.painter.frame_requested

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
        assert app.painter.drain(WAIT_S)
        # As that poster left it: a state no public call leaves behind.
        app.painter._frame_posted = True
        app.update(iteration=7)
        terminal.wait_for_frame(_shows("iter 7"))


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
    _hold_a_frame(app, terminal)
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
