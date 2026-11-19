"""Starting and ending a child process tree: the escalation every owner of a
child shares.

A leaf module on purpose — standard library and the leaves `stopclock` and
`win_job` only — so both the turn runner (`providers`) and the quota server
(`codex_usage`, which `providers` imports) can reach it without an import
running against that direction. Each caller keeps only its own prologue
(which pipe to close first, whether the child gets an EOF to exit on) and
hands the rest to `end_process_tree`.

On Windows a tree started with `start_tree` has a lifetime the OS owns: the
child is started SUSPENDED in a job object of its own with KILL_ON_JOB_CLOSE
(`win_job.start`), so every descendant is in the job from its first
instruction. `TerminateJobObject` ends the whole tree at once by that
identity, never by a numeric pid, so its ending can be cut to the ending's
stop budget (`stopclock.within`) with nothing handed to a reaper that
outlives the wait; the handle closing — `close`, or Windows closing it when
this process dies in any way — ends what is left.

A nested job is fine: a run inside a job of its own (`llm_agents.cores` hold,
a worker's `WorkerJob`, a bridge's `ChildJob`) puts the child in both, and
the outer job's limits are kill-on-close and breakaway-OK, neither of which
refuses the nesting. CREATE_BREAKAWAY_FROM_JOB is therefore not asked for —
it would take the child OUT of the outer job, whose closing must still end
it. A job Windows will not make or join leaves the child unowned, started
again plainly (it never ran: it was suspended).

POSIX, and an unowned Windows child, get the pid-based sequence of
`end_process_tree`, which is NOT bounded by the stop budget (see there).
"""

import os
import subprocess
import threading
import weakref

from . import stopclock, win_job

# How long a provider child gets after being told to end before it is killed
# outright, and the ceiling on the `taskkill` call itself. The bound is what
# keeps the reaping from becoming a second hang inside the guard that exists to
# end the first one: the parallel runner calls it from a thread that is ALREADY
# dying, so nothing above it is left to notice a wait that never returns.
REAP_GRACE_S = 2.0


class _JobTree:
    """A Windows child's job: the tree's identity, independent of any pid.

    `ask` is reentrant, because it is a Ctrl+C hook (`providers.ended_on_press`)
    and `ctrlc.Interrupt.press` runs hooks from the SIGINT handler — on the
    main thread, between any two bytecodes of whatever it was doing, an `ask`
    or a `close` of this very tree included. So the lock is an RLock (a plain
    one held by the interrupted call deadlocks the handler), and `close`
    detaches the handle BEFORE closing it: a handler running inside `close`
    sees either the open handle or None, never a closed number Windows may
    already have handed to another object."""

    def __init__(self, handle: int) -> None:
        self._handle = handle
        self._lock = threading.RLock()

    def ask(self, proc) -> None:
        """End every process in the job. There is nothing to ask a console
        child WITH on Windows (`ask_agent_process_to_end`), so the ask is the
        kill — of the tree, at once."""
        with self._lock:
            handle = self._handle
            if handle:
                win_job.kernel32().TerminateJobObject(handle, 1)

    def end(self, proc) -> None:
        self.ask(proc)
        try:
            proc.wait(timeout=stopclock.within(REAP_GRACE_S))
        except (OSError, subprocess.TimeoutExpired):
            pass  # reclaimed, or not yet collected: the job is ended either way
        self.close()

    def close(self) -> None:
        """Close the job, ending whatever is left in it. Idempotent."""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle:
                win_job.kernel32().CloseHandle(handle)


# Process (or a wrapper standing for it, `share_tree`) -> its job. Weak, so a
# dropped Popen takes its entry with it; the handle is closed by the
# finalizer `_own` registers.
_trees: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _own(proc, tree: _JobTree) -> None:
    _trees[proc] = tree
    finalizer = weakref.finalize(proc, tree.close)
    # Not at interpreter exit: the job's closing would kill a quota server
    # before its owner's own exit hook closes it politely; the OS closes the
    # handle after every hook has run anyway.
    finalizer.atexit = False


def _tree(proc):
    try:
        return _trees.get(proc)
    except TypeError:
        return None  # not weak-referenceable: nobody's tree


def owns(proc) -> bool:
    """Whether `proc` was started by `start_tree` in a job of its own."""
    return _tree(proc) is not None


def share_tree(wrapper, proc) -> None:
    """Let `wrapper` — an object standing in for `proc` with the same
    `poll`/`wait`/`returncode` — be ended as `proc`'s tree."""
    tree = _tree(proc)
    if tree is not None:
        _trees[wrapper] = tree


def release_exited(proc) -> None:
    """After `proc` has exited: close its job, ending what it left in it (a
    descendant still holding its stdout). Nothing for an unowned child."""
    tree = _tree(proc)
    if tree is not None:
        tree.close()


def start_tree(argv, **options):
    """`subprocess.Popen(argv, **options)`, on Windows in a job of its own
    (see the module docstring); the caller's own `creationflags` are kept.
    POSIX: a plain `Popen`.

    A stand-in `Popen` (a test's) without an OS handle behind it is returned
    unowned."""
    if os.name != "nt":
        return subprocess.Popen(argv, **options)
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
    the executable itself (a shebang script), so the handle IS the provider and
    SIGTERM lands where it is aimed.

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
    tree = _tree(proc)
    if tree is not None:
        tree.ask(proc)
        return
    if os.name == "nt":
        try:
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
    `start_tree` child's job is closed (`release_exited`). Never raises
    `OSError`, like the escalation it fronts.
    """
    try:
        exited = proc.poll() is not None
    except OSError:
        exited = True
    if exited:
        release_exited(proc)
    else:
        end_process_tree(proc)


def end_process_tree(proc) -> None:
    """Ask the child (tree) to end, wait, kill, wait — never raises.

    A `start_tree` child (Windows): its job is terminated, and the one wait
    for the child takes what is left of the ending's stop budget
    (`stopclock.within`, at most `REAP_GRACE_S`); a spent budget does not wait.

    Any other child — every POSIX one — gets the pid-based sequence: at most
    `REAP_GRACE_S` for the ask's `taskkill` plus two waits of `REAP_GRACE_S`
    each, NOT bounded by the stop budget, 6 s per tree past it at worst.
    Cutting it short hands the tree to something that outlives the wait — a
    daemon thread the exiting interpreter freezes, or a reaper working from
    the numeric pid, which can kill an unrelated process once the pid is
    reused. POSIX has no job to own the tree by; a process group would need
    the group id pinned (the leader unreaped) until `killpg`, and that needs
    exclusive reaping across every `poll()`/`wait()` caller of the child
    (`streamrender`, `parallel`, `codex_usage`), defined behaviour under
    SIGCHLD=SIG_IGN, and a fallback where neither `os.pidfd_open` nor
    `os.waitid` exists (macOS before Python 3.13) — not built.

    `OSError` from `wait` means the OS has already reclaimed the handle
    (Windows raises it there), so there is nothing left to end;
    `KeyboardInterrupt` is NOT absorbed here — that is the caller's call
    (`providers.reap_agent_process`).
    """
    tree = _tree(proc)
    if tree is not None:
        tree.end(proc)
        return
    ask_agent_process_to_end(proc)
    try:
        proc.wait(timeout=REAP_GRACE_S)
    except OSError:
        return
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass  # gone between the wait and here — `wait` below collects it
        try:
            proc.wait(timeout=REAP_GRACE_S)
        except OSError:
            return
        except subprocess.TimeoutExpired:
            # Unkillable (a stuck kernel-mode handle) — deliberately not waited
            # on any longer. In the parallel runner the thread is dying either
            # way, and hanging here would take the rest of the fleet with it,
            # which is the one thing this whole seam exists to prevent.
            pass
