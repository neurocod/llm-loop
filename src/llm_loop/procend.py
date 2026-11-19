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

On POSIX a tree started with `start_tree` lives in a process group of its
own, and the group is signalled only by a KEEPER: a `/bin/sh` that leads
the group, ignores SIGINT/SIGTERM/SIGHUP and reads a pipe only this process
holds the write end of. A line on the pipe makes the keeper send SIGTERM
(and SIGCONT, so a member stopped by SIGTTIN hears it) to its own group;
the pipe's EOF — `close`, or the OS closing it when this process dies in any
way — makes it send SIGKILL. `kill(0, ...)` from inside the group names the
group by the keeper's own membership, never by a number, so the group's
ending needs no pinned id and no exclusive reaping: any `poll()`/`wait()` on
the provider (`streamrender`, `parallel`, `codex_usage`, the runners' own)
may reap the leader at any time, and SIGCHLD=SIG_IGN changes nothing (the
keeper's death only loses the group kill, see `_GroupTree`). No pidfd, no
`os.waitid`.

Python 3.11 or newer only (`POSIX_GROUPS`): joining the keeper's group
needs `setpgid` between fork and exec, which is `Popen(process_group=)`
from 3.11 and before it only `preexec_fn` — Python code in a forked child
of a threaded runner, which can deadlock on a lock another thread held at
the fork. 3.9 and 3.10 (both past end of life) keep the pre-2496 ending: a
plain `Popen` in the runner's own group, the pid-based sequence outside the
stop budget, a descendant outliving the leader left on its pipe, and no
ending on the owner's death.

What the group costs, said aloud: the tree is a new process GROUP in the
runner's session — `start_new_session` would drop the controlling terminal
as well, and is not used. A group that is not the terminal's foreground one
no longer hears the terminal's Ctrl+C (nor Ctrl+Z or Ctrl+\\), just as the
Windows child under CREATE_NO_WINDOW does not; the run's press ends the turn
through `providers.ended_on_press`, whose hook asks the tree to end (the
keeper's SIGTERM) and arms `end_after_grace` — the reader may be blocked on
a child that prints nothing, so the escalation cannot wait for the stream
to notice the press. A member that reads the terminal is stopped by
SIGTTIN where it used to read the operator's keys; the ending's SIGKILL ends
a stopped member too. A descendant that leaves the group itself (`setsid`,
`setpgid`) leaves the ending with it — what Windows' job would still hold.

An unowned child — a job Windows will not make, a keeper POSIX cannot start,
a stand-in `Popen` — gets the pid-based sequence of `end_process_tree`.
"""

import os
import subprocess
import sys
import threading
import weakref

from . import stopclock, win_job

# Whether `start_tree` owns a POSIX tree on this Python (module docstring).
POSIX_GROUPS = os.name != "nt" and sys.version_info >= (3, 11)

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

    def close(self, wait: bool = True) -> None:
        """Close the job, ending whatever is left in it. Idempotent; `wait`
        is `_GroupTree.close`'s — closing a handle has nothing to wait for."""
        with self._lock:
            handle, self._handle = self._handle, None
            if handle:
                win_job.kernel32().CloseHandle(handle)


# The keeper (see the module docstring). Every `kill` names group 0 — the
# keeper's own — and the keeper ignores the polite signals it sends, so it
# is still there for the next line and for the EOF. `read` fails on the EOF
# (and on a partial last line), which ends the loop into the SIGKILL.
_KEEPER = ("trap '' INT TERM HUP\n"
           "while read line; do kill -s TERM 0; kill -s CONT 0; done\n"
           "kill -s KILL 0\n")

# The real class, taken before any test stands a fake in for
# `subprocess.Popen`: the keeper is always a real process, and a provider
# `Popen` that is not one of these is a stand-in left unowned.
_Popen = subprocess.Popen

def _reap_later(keeper) -> None:
    """Reap a keeper whose EOF was not waited for (a finalizer, a spent
    budget) as soon as it exits, on a daemon thread of its own: a blocking
    `wait` on that one pid — it signals nothing — so no keeper stays a
    zombie until some later launch. A keeper exits within milliseconds of
    its EOF; one that will not (stopped from outside) keeps its thread, and
    the interpreter's exit does not wait for it."""
    def reap():
        try:
            keeper.wait()
        except OSError:
            pass  # reaped already: SIGCHLD=SIG_IGN
    threading.Thread(target=reap, name="procend-keeper-reap", daemon=True).start()


class _GroupTree:
    """A POSIX child's group, ended through its keeper's pipe (module
    docstring). Reentrant for the same reason as `_JobTree`, and `close`
    detaches the keeper before closing its pipe the same way.

    A keeper that died (killed from outside) takes the group's ending with
    it: a write to its pipe fails, and the ending falls back to the leader's
    own `Popen`, which signals it only while unreaped."""

    def __init__(self, keeper) -> None:
        self._keeper = keeper
        self._lock = threading.RLock()

    def _tell(self, line: bytes) -> bool:
        """Write `line` to the keeper; False when there is none to hear it."""
        with self._lock:
            keeper = self._keeper
            if keeper is None:
                return False
            try:
                os.write(keeper.stdin.fileno(), line)
                return True
            except (OSError, ValueError):
                return False  # the keeper is gone, or its pipe already closed

    def ask(self, proc) -> None:
        """SIGTERM (then SIGCONT) to every process in the group; without a
        live keeper, SIGTERM to the leader alone."""
        if not self._tell(b"\n") and proc is not None:
            _signal_leader(proc, kill=False)

    def end(self, proc) -> None:
        self.ask(proc)
        if _waited(proc):
            self.close()
            return
        self.close()  # the EOF: SIGKILL to the group
        if _waited(proc):
            return
        # No keeper heard the EOF (or the kill is not through yet): the
        # leader by its own Popen, so a provider never outlives the ending.
        _signal_leader(proc, kill=True)
        _waited(proc)

    def close(self, wait: bool = True) -> None:
        """Close the keeper's pipe, ending whatever is left in the group.
        Idempotent. `wait` reaps the keeper within the stop budget; a
        finalizer passes False. What is not reaped here, `_reap_later`
        reaps."""
        with self._lock:
            keeper, self._keeper = self._keeper, None
            if keeper is None:
                return
            try:
                keeper.stdin.close()
            except OSError:
                pass
        if wait:
            try:
                keeper.wait(timeout=stopclock.within(REAP_GRACE_S))
                return
            except OSError:
                return  # reaped already: SIGCHLD=SIG_IGN
            except subprocess.TimeoutExpired:
                pass
        _reap_later(keeper)


def _waited(proc) -> bool:
    """Whether `proc` exited within what the stop budget leaves of the grace;
    True also for a handle the OS has already reclaimed."""
    try:
        proc.wait(timeout=stopclock.within(REAP_GRACE_S))
        return True
    except OSError:
        return True
    except subprocess.TimeoutExpired:
        return False


def _signal_leader(proc, kill: bool) -> None:
    try:
        proc.kill() if kill else proc.terminate()
    except OSError:
        pass  # it died between the poll and here


def _start_group(argv, options):
    """`start_tree`'s POSIX half: the keeper first, then the child in its
    group. A caller's own group or session options leave the child
    unowned, as does a Python before 3.11 (`POSIX_GROUPS`) or a keeper
    `/bin/sh` will not start."""
    if (not POSIX_GROUPS or {"preexec_fn", "process_group"} & options.keys()
            or options.get("start_new_session")):
        return subprocess.Popen(argv, **options)
    try:
        keeper = _Popen(["/bin/sh", "-c", _KEEPER], stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        bufsize=0, process_group=0)
    except (OSError, subprocess.SubprocessError):
        return subprocess.Popen(argv, **options)
    tree = _GroupTree(keeper)
    try:
        proc = subprocess.Popen(argv, **dict(options, process_group=keeper.pid))
    except BaseException:
        tree.close()
        raise
    if not isinstance(proc, _Popen):
        tree.close()
        return proc
    _own(proc, tree)
    return proc


# Process (or a wrapper standing for it, `share_tree`) -> its job. Weak, so a
# dropped Popen takes its entry with it; the handle is closed by the
# finalizer `_own` registers.
_trees: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _own(proc, tree) -> None:
    _trees[proc] = tree
    # Not waiting: a finalizer runs inside whatever the collector interrupted.
    finalizer = weakref.finalize(proc, tree.close, False)
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
    """Whether `proc` was started by `start_tree` in a job (Windows) or a
    kept group (POSIX) of its own."""
    return _tree(proc) is not None


def share_tree(wrapper, proc) -> None:
    """Let `wrapper` — an object standing in for `proc` with the same
    `poll`/`wait`/`returncode` — be ended as `proc`'s tree."""
    tree = _tree(proc)
    if tree is not None:
        _trees[wrapper] = tree


def release_exited(proc) -> None:
    """After `proc` has exited: close its job or its keeper's pipe, ending
    what it left (a descendant still holding its stdout). Nothing for an
    unowned child."""
    tree = _tree(proc)
    if tree is not None:
        tree.close()


def start_tree(argv, **options):
    """`subprocess.Popen(argv, **options)`, on Windows in a job of its own,
    on POSIX in a kept process group of its own (see the module docstring);
    the caller's own `creationflags` are kept.

    A stand-in `Popen` (a test's) without an OS process behind it is
    returned unowned."""
    if os.name != "nt":
        return _start_group(argv, options)
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
    a `start_tree` child's SIGTERM goes to its whole group (its keeper's),
    any other's to the child alone.

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


# Children `end_after_grace` is already escalating: one watchdog per child,
# however many presses (and a `wake=True` hook runs twice per press).
_escalating: "weakref.WeakSet" = weakref.WeakSet()
_escalating_lock = threading.Lock()


def end_after_grace(proc) -> None:
    """After an ask nothing else will follow up — a Ctrl+C hook, whose turn
    reader may be blocked on a child that prints nothing and ignores
    SIGTERM, so the stream never comes round to notice the press — end the
    tree once `REAP_GRACE_S` (capped by the caller's stop budget) has
    passed without the child exiting. Its killing closes the child's
    stdout, which is what unblocks the reader.

    From a daemon thread, which only ever acts by identity on an owned
    tree (the job, the keeper's EOF) — a thread the exiting interpreter
    freezes leaves nothing half-done that the OS will not finish (the
    handle's or the pipe's closing). An unowned child gets `Popen.kill`,
    which signals only an unreaped child. Never raises; once per child.
    """
    try:
        with _escalating_lock:
            if proc in _escalating:
                return
            _escalating.add(proc)
    except TypeError:
        pass  # not weak-referenceable: armed every time, which is harmless
    deadline = stopclock.deadline()

    def escalate():
        try:
            with stopclock.ending(deadline):
                if _waited(proc):
                    return
                tree = _tree(proc)
                if tree is not None:
                    tree.close()
                else:
                    _signal_leader(proc, kill=True)
        except Exception:  # noqa: BLE001 - a watchdog must not die loudly
            pass

    threading.Thread(target=escalate, name="procend-press-escalation",
                     daemon=True).start()


def end_unless_exited(proc) -> None:
    """`end_process_tree`, unless the child has already exited.

    `poll()` is guarded too: on Windows it raises `OSError` on a handle the OS
    has already reclaimed, and then there is nothing left to end. An exited
    `start_tree` child's tree is closed (`release_exited`). Never raises
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

    A `start_tree` child: its job is terminated (Windows), or its group is
    asked and, past the grace, killed through the keeper (POSIX); every wait
    for the child takes what is left of the ending's stop budget
    (`stopclock.within`, at most `REAP_GRACE_S`), and a spent budget does
    not wait. What is not waited for still lands: the job's termination, the
    keeper's SIGKILL on its pipe's EOF — neither needs this process alive.

    Any other child gets the pid-based sequence: at most `REAP_GRACE_S` for
    the ask's `taskkill` plus two waits of `REAP_GRACE_S` each, NOT bounded
    by the stop budget, 6 s per tree past it at worst. Cutting it short
    hands the tree to something that outlives the wait — a daemon thread the
    exiting interpreter freezes, or a reaper working from the numeric pid,
    which can kill an unrelated process once the pid is reused.

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
