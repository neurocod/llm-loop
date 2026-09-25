"""One owner per terminal resource: the workers' console lines and the pinned rows.

Three layers, pinned in that order:

  * `ownership.OwnerThread` itself — order, drain, close, the bounded queue and
    what a failing call costs;
  * the parallel runner's console lines — every write on the one owner thread,
    never two at once, all of them on screen before the run reports;
  * the status line — while the painter runs, nobody else renders or writes
    the terminal; everybody else's paint is a request.
"""

import io
import queue
import sys
import threading
import time

import pytest

from llm_loop import ownership, parallel, projectroot, termio
from llm_loop import statusline as sl

from _runfixtures import MemListDriver, par_args

# Upper bound on every wait below; each one is a handshake that a healthy run
# completes at once, so only a broken owner ever gets near it. 0.49 s for this
# whole file, measured 2026-09-25 — the bound is many times that.
WAIT_S = 10.0


@pytest.fixture(autouse=True)
def _restore_run_state():
    """The runner tees the streams and moves the project root; put both back."""
    out, err = sys.stdout, sys.stderr
    root = projectroot.project_dir()
    yield
    sys.stdout, sys.stderr = out, err
    projectroot.set_project_root(root)


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


class _PaintLog(termio.Terminal):
    """A terminal with no screen that remembers which thread wrote to it."""

    def __init__(self):
        super().__init__(io.StringIO())
        self.writers = []
        self.frames = queue.Queue()

    def size(self):
        return (100, 30)

    def set_title(self, text, *, reassert=False):
        self.writers.append(threading.current_thread().name)
        return super().set_title(text, reassert=reassert)

    def paint(self, lines, *, reassert=False):
        self.writers.append(threading.current_thread().name)
        result = super().paint(lines, reassert=reassert)
        self.frames.put(" ".join(lines))
        return result


def test_while_started_only_the_painter_writes_the_terminal():
    terminal = _PaintLog()
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
        while "the LAST note" not in terminal.frames.get(
                timeout=max(0, deadline - time.monotonic())):
            pass
        writers = set(terminal.writers)

    assert writers == {"statusline-paint"}


def test_without_a_painter_the_caller_paints():
    """Before start() nobody owns the terminal, and a title is due at once."""
    terminal = _PaintLog()
    app = sl.StatusApp(terminal=terminal, input_source=termio.NullInputSource())

    app.update(phase="running")

    assert terminal.writers == [threading.current_thread().name]
