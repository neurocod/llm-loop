"""Starting and ending a child process tree: the escalation every owner of a
child shares.

A leaf module on purpose — standard library and the leaves `stopclock` and
`win_job` only — so both the turn runner (`providers`) and the quota server
(`codex_usage`, which `providers` imports) can reach it without an import
running against that direction. Each caller keeps only its own prologue
(which pipe to close first, whether the child gets an EOF to exit on) and
hands the rest to `end_process_tree`.

A tree started with `start_tree` has a lifetime the OS owns, and its ending
acts on that identity, never on the numeric pid, so it can be cut to the
ending's stop budget (`stopclock.within`) without handing anything to a
reaper that outlives the wait:

  * Windows: the child is started SUSPENDED in a job object of its own with
    KILL_ON_JOB_CLOSE (`win_job.start`), so every descendant is in the job
    from its first instruction. `TerminateJobObject` ends the whole tree at
    once and needs no wait; the handle closing — `close`, or Windows closing
    it when this process dies in any way — ends what is left.
  * POSIX: the child leads a session of its own (`start_new_session`), and
    `killpg` reaches every descendant still in it. The group id is the
    leader's pid, and that number stays ours exactly as long as the leader is
    not reaped: so the leader's exit is awaited WITHOUT reaping it (a pidfd,
    `os.pidfd_open`, Linux 5.3+; else `waitid(WNOWAIT)`), the group gets its
    SIGKILL while the zombie still pins the number, and only then is the
    leader reaped.

A nested job is fine: a run inside a job of its own (`llm_agents.cores` hold,
a worker's `WorkerJob`, a bridge's `ChildJob`) puts the child in both, and
the outer job's limits are kill-on-close and breakaway-OK, neither of which
refuses the nesting. CREATE_BREAKAWAY_FROM_JOB is therefore not asked for —
it would take the child OUT of the outer job, whose closing must still end
it. A job Windows will not make or join leaves the child unowned, started
again plainly (it never ran: it was suspended); its ending is the pid-based
sequence below.
"""

import os
import select
import signal
import subprocess
import threading
import time
import weakref

from . import stopclock, win_job

# How long a provider child gets after being told to end before it is killed
# outright, and the ceiling on the `taskkill` call itself. The bound is what
# keeps the reaping from becoming a second hang inside the guard that exists to
# end the first one: the parallel runner calls it from a thread that is ALREADY
# dying, so nothing above it is left to notice a wait that never returns.
REAP_GRACE_S = 2.0

# How often `waitid(WNOWAIT)` is polled where no pidfd can be waited on.
_EXIT_POLL_S = 0.05


class _JobTree:
    """A Windows child's job: the tree's identity, independent of any pid."""

    def __init__(self, handle: int) -> None:
        self._handle = handle
        self._lock = threading.Lock()

    def ask(self, proc) -> None:
        """End every process in the job. There is nothing to ask a console
        child WITH on Windows (`ask_agent_process_to_end`), so the ask is the
        kill — of the tree, at once."""
        with self._lock:
            if self._handle:
                win_job.kernel32().TerminateJobObject(self._handle, 1)

    def end(self, proc) -> None:
        self.ask(proc)
        try:
            proc.wait(timeout=stopclock.within(REAP_GRACE_S))
        except (OSError, subprocess.TimeoutExpired):
            pass  # reclaimed, or not yet collected: the job is ended either way
        self.close()

    def exited(self, proc) -> None:
        """The leader is gone; whatever it left in the job goes with the job."""
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._handle:
                win_job.kernel32().CloseHandle(self._handle)
                self._handle = None


class _GroupTree:
    """A POSIX child's process group, with the leader's pidfd where there is
    one. The group id is safe to signal only while the leader is unreaped
    (`proc.returncode is None`): a reaped leader frees the number."""

    def __init__(self, pid: int, pidfd) -> None:
        self._pid = pid
        self._pidfd = pidfd
        self._lock = threading.Lock()

    def _signal(self, proc, sig) -> None:
        # A reaping thread can still win the race between this check and the
        # call; the group id then stays pinned by any member left in it, and
        # an empty group's number is reused only after a pid wrap-around.
        if proc.returncode is not None:
            return
        try:
            os.killpg(self._pid, sig)
        except OSError:
            pass  # the group is gone

    def ask(self, proc) -> None:
        self._signal(proc, signal.SIGTERM)

    def _await_exit(self, timeout: float):
        """True once the leader has exited, False on the timeout, None where
        no wait leaves it unreaped. Never reaps."""
        with self._lock:
            pidfd = self._pidfd
        if pidfd is not None:
            try:
                ready, _, _ = select.select([pidfd], [], [], timeout)
                return bool(ready)
            except (OSError, ValueError):
                pass  # closed under us: fall back to waitid
        if not (hasattr(os, "waitid") and hasattr(os, "WNOWAIT")):
            return None
        deadline = time.monotonic() + timeout
        while True:
            try:
                if os.waitid(os.P_PID, self._pid,
                             os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
                    return True
            except ChildProcessError:
                return True  # reaped by someone else: exited
            except OSError:
                return None
            left = deadline - time.monotonic()
            if left <= 0:
                return False
            time.sleep(min(_EXIT_POLL_S, left))

    def end(self, proc) -> None:
        self.ask(proc)
        exited = None
        if proc.returncode is None:
            exited = self._await_exit(stopclock.within(REAP_GRACE_S))
        if exited is None:
            # No unreaping wait: the reaping one, and SIGKILL only while the
            # leader is still unreaped — descendants left behind by a leader
            # that exited in the grace are not reached.
            try:
                proc.wait(timeout=stopclock.within(REAP_GRACE_S))
            except OSError:
                return
            except subprocess.TimeoutExpired:
                self._signal(proc, signal.SIGKILL)
        else:
            # The leader is unreaped (a zombie at most), so the group id is
            # still ours: the rest of the group — and the leader, if the grace
            # ran out — get their SIGKILL now.
            self._signal(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=stopclock.within(REAP_GRACE_S))
        except (OSError, subprocess.TimeoutExpired):
            pass
        self.close()

    def exited(self, proc) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            pidfd, self._pidfd = self._pidfd, None
        if pidfd is not None:
            try:
                os.close(pidfd)
            except OSError:
                pass


# Process (or a wrapper standing for it, `share_tree`) -> its tree. Weak, so
# a dropped Popen takes its entry with it; the tree's handle is closed by the
# finalizer `_own` registers.
_trees: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _own(proc, tree) -> None:
    _trees[proc] = tree
    finalizer = weakref.finalize(proc, tree.close)
    # Not at interpreter exit: the job's closing would kill a quota server
    # before its owner's own exit hook closes it politely; the OS closes the
    # handle after every hook has run anyway.
    finalizer.atexit = False


def owns(proc) -> bool:
    """Whether `proc` was started by `start_tree` with an OS-owned lifetime."""
    try:
        return proc in _trees
    except TypeError:
        return False  # not weak-referenceable: nobody's tree


def share_tree(wrapper, proc) -> None:
    """Let `wrapper` — an object standing in for `proc` with the same
    `poll`/`wait`/`returncode` — be ended as `proc`'s tree."""
    tree = _trees.get(proc)
    if tree is not None:
        _trees[wrapper] = tree


def start_tree(argv, **options):
    """`subprocess.Popen(argv, **options)` whose tree's lifetime the OS owns
    (see the module docstring); the caller's own `creationflags` are kept.

    A stand-in `Popen` (a test's) without an OS process behind it — no
    `_handle` on Windows, no int `pid` on POSIX — is returned unowned."""
    if os.name == "nt":
        return _start_in_job(argv, options)
    options.setdefault("start_new_session", True)
    proc = subprocess.Popen(argv, **options)
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or not options["start_new_session"]:
        return proc
    pidfd = None
    if hasattr(os, "pidfd_open"):
        try:
            pidfd = os.pidfd_open(pid)
        except OSError:
            pidfd = None  # a kernel before 5.3: `waitid(WNOWAIT)` instead
    _own(proc, _GroupTree(pid, pidfd))
    return proc


def _start_in_job(argv, options):
    try:
        handle = win_job.create(None, win_job.KILL_ON_JOB_CLOSE)
    except OSError:
        return subprocess.Popen(argv, **options)
    suspended = dict(options, creationflags=options.get("creationflags", 0)
                     | win_job.CREATE_SUSPENDED)
    try:
        proc = subprocess.Popen(argv, **suspended)
    except BaseException:
        win_job.kernel32().CloseHandle(handle)
        raise
    if getattr(proc, "_handle", None) is None:
        win_job.kernel32().CloseHandle(handle)
        return proc
    try:
        win_job.adopt(handle, proc)  # kills and reaps the child on a refusal
    except OSError:
        win_job.kernel32().CloseHandle(handle)
        return subprocess.Popen(argv, **options)
    except BaseException:
        win_job.kernel32().CloseHandle(handle)
        raise
    _own(proc, _JobTree(handle))
    return proc


def ask_agent_process_to_end(proc) -> None:
    """Aim the ending at the provider CLI, not at the shim standing in front of it.

    On Windows the handle we hold is usually `cmd.exe`: `runtime_argv` resolves
    the provider to an npm `.cmd` shim and CreateProcess runs a batch file
    through the interpreter, so the CLI itself is a GRANDCHILD. TerminateProcess
    on the shim leaves it running, still holding the stdout handle it inherited
    and still printing over whatever the terminal does next — which is the whole
    symptom being fixed, so the Windows branch has to reach the tree: the job
    of a `start_tree` child, `taskkill /T` of any other. On POSIX an npm bin is
    the executable itself (a shebang script), so the handle IS the provider;
    a `start_tree` child's whole group is asked.

    ASYMMETRY WORTH KNOWING, because the two halves do NOT offer the same deal:
    POSIX gets a real request — SIGTERM, which a CLI can catch and use to close
    its session store — and only then, after `REAP_GRACE_S`, the kill. Windows
    gets no such step, because there is nothing to ask WITH: `taskkill` without
    `/F` posts WM_CLOSE, which a windowless console process never receives, so
    the polite spelling would do nothing at all and the child would be killed
    two seconds later regardless. `/F` (and `TerminateJobObject`) is therefore
    not impatience — it is the only thing that ends the tree there, and the
    cost (a session store torn mid-write) is charged on Windows whichever
    spelling is used.
    """
    tree = _trees.get(proc) if owns(proc) else None
    if tree is not None:
        tree.ask(proc)
        return
    if os.name == "nt":
        try:
            # Not capped by the stop budget: this is the act that ends a tree
            # no job holds, and cutting it short orphans the descendants it
            # has not reached yet. Only a child `start_tree` could not put in
            # a job comes here.
            done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                  capture_output=True, timeout=REAP_GRACE_S)
            if done.returncode == 0:
                return
            # Non-zero means the tree was NOT ended (no such pid, access
            # denied): fall through rather than return, or the caller's wait
            # would be a two-second pause on the way to the same `kill`.
        except (OSError, subprocess.SubprocessError):
            pass  # no taskkill, or it hung — fall back to the handle we hold
    try:
        proc.terminate()
    except OSError:
        pass  # it died between the poll and here; `wait` below collects it


def end_unless_exited(proc) -> None:
    """`end_process_tree`, unless the child has already exited.

    `poll()` is guarded too: on Windows it raises `OSError` on a handle the OS
    has already reclaimed, and then there is nothing left to end. An exited
    `start_tree` child's job is closed, ending what it left in it. Never raises
    `OSError`, like the escalation it fronts.
    """
    try:
        exited = proc.poll() is not None
    except OSError:
        exited = True
    if not exited:
        end_process_tree(proc)
        return
    tree = _trees.get(proc) if owns(proc) else None
    if tree is not None:
        tree.exited(proc)


def end_process_tree(proc) -> None:
    """Ask the child (tree) to end, wait, kill, wait — never raises.

    Every wait takes what is left of the ending's stop budget
    (`stopclock.within`), each at most `REAP_GRACE_S`; a spent budget ends a
    `start_tree` child's tree without waiting at all (module docstring). A
    child started otherwise keeps the pid-based ask (`taskkill /T`, its own
    `REAP_GRACE_S` limit) before its budgeted waits. `OSError` from `wait`
    means the OS has already reclaimed the handle (Windows raises it there),
    so there is nothing left to end; `KeyboardInterrupt` is NOT absorbed here
    — that is the caller's call (`providers.reap_agent_process`).
    """
    tree = _trees.get(proc) if owns(proc) else None
    if tree is not None:
        tree.end(proc)
        return
    ask_agent_process_to_end(proc)
    try:
        proc.wait(timeout=stopclock.within(REAP_GRACE_S))
    except OSError:
        return
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass  # gone between the wait and here — `wait` below collects it
        try:
            proc.wait(timeout=stopclock.within(REAP_GRACE_S))
        except OSError:
            return
        except subprocess.TimeoutExpired:
            # Unkillable (a stuck kernel-mode handle) — deliberately not waited
            # on any longer. In the parallel runner the thread is dying either
            # way, and hanging here would take the rest of the fleet with it,
            # which is the one thing this whole seam exists to prevent.
            pass
