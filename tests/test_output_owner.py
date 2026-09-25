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
