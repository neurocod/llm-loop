"""Ending a child process tree: the escalation every owner of a child shares.

A leaf module on purpose — standard library and the leaf `stopclock` only — so both the turn runner
(`providers`) and the quota server (`codex_usage`, which `providers` imports)
can reach it without an import running against that direction. Each caller
keeps only its own prologue (which pipe to close first, whether the child gets
an EOF to exit on) and hands the rest to `end_process_tree`.
"""

import os
import subprocess
import sys

from . import stopclock

# How long a provider child gets after being told to end before it is killed
# outright, and the ceiling on the `taskkill` call itself. The bound is what
# keeps the reaping from becoming a second hang inside the guard that exists to
# end the first one: the parallel runner calls it from a thread that is ALREADY
# dying, so nothing above it is left to notice a wait that never returns.
REAP_GRACE_S = 2.0
# Inside an ending the caller waits for no more than its budget has left, and
# the tree is still ended whole (`end_process_tree` says how).


def ask_agent_process_to_end(proc) -> None:
    """Aim the ending at the provider CLI, not at the shim standing in front of it.

    On Windows the handle we hold is usually `cmd.exe`: `runtime_argv` resolves
    the provider to an npm `.cmd` shim and CreateProcess runs a batch file
    through the interpreter, so the CLI itself is a GRANDCHILD. TerminateProcess
    on the shim leaves it running, still holding the stdout handle it inherited
    and still printing over whatever the terminal does next — which is the whole
    symptom being fixed, so the Windows branch has to reach the tree
    (`taskkill /T`). On POSIX an npm bin is the executable itself (a shebang
    script), so the handle IS the provider and SIGTERM lands where it is aimed.

    ASYMMETRY WORTH KNOWING, because the two halves do NOT offer the same deal:
    POSIX gets a real request — SIGTERM, which a CLI can catch and use to close
    its session store — and only then, after `REAP_GRACE_S`, the kill. Windows
    gets no such step, because there is nothing to ask WITH: `taskkill` without
    `/F` posts WM_CLOSE, which a windowless console process never receives, so
    the polite spelling would do nothing at all and the child would be killed
    two seconds later regardless. `/F` is therefore not impatience — it is the
    only thing that ends the tree there, and the cost (a session store torn
    mid-write) is charged on Windows whichever spelling is used.
    """
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
    has already reclaimed, and then there is nothing left to end. Never raises
    `OSError`, like the escalation it fronts.
    """
    try:
        if proc.poll() is not None:
            return
    except OSError:
        return
    end_process_tree(proc)


def end_process_tree(proc) -> None:
    """Ask the child (tree) to end, wait, kill, wait — bounded, never raises.

    At most `REAP_GRACE_S` for the ask's `taskkill` plus two waits of
    `REAP_GRACE_S` each. `OSError` from `wait` means the OS has already
    reclaimed the handle (Windows raises it there), so there is nothing left to
    end; `KeyboardInterrupt` is NOT absorbed here — that is the caller's call
    (`providers.reap_agent_process`).

    Inside an ending (`stopclock`) the ending is handed to a REAPER PROCESS
    (`_start_reaper`) before this returns, and the caller then waits for the
    child no longer than the budget has left. The steps cannot be shortened
    in place: `taskkill /T` finds the tree through the root's parent links, so
    killing the root before taskkill has walked them orphans the descendants
    still holding the child's pipes, and a `taskkill` given a fraction of its
    limit is killed mid-walk. Nor can a thread own them: a daemon the
    interpreter exits under — at once, on a spent budget — may not have
    started anything yet. A process the OS has created runs to its end
    whatever this one does next, so the tree is ended however soon the run
    leaves.
    """
    remaining = stopclock.left()
    if remaining is None:
        _end_tree(proc)
        return
    if not _start_reaper(proc):
        # No reaper could be started: end the tree here, unbudgeted — a
        # stop that overruns beats a tree left running.
        _end_tree(proc)
        return
    try:
        proc.wait(timeout=remaining)
    except (OSError, subprocess.TimeoutExpired):
        pass


# The reaper on POSIX: the same ask, grace and kill as `_end_tree`, by pid,
# in a process of its own session so the run's exit does not take it along.
_POSIX_REAPER = """\
import os, signal, sys, time
pid, grace = int(sys.argv[1]), float(sys.argv[2])
try:
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        os.kill(pid, 0)
        time.sleep(0.05)
    os.kill(pid, signal.SIGKILL)
except OSError:
    pass
"""


def _start_reaper(proc) -> bool:
    """Start a process that ends `proc`'s tree on its own; True once started.

    On Windows that process is `taskkill /F /T` itself — forced already, so
    there is no grace to escalate from — never timed and never followed by a
    kill of the root, which is what would race its walk of the tree. On
    POSIX a small Python reaper runs `_end_tree`'s SIGTERM, grace, SIGKILL.
    Either is left to finish unwatched; its exit status is not read.
    """
    if os.name == "nt":
        argv = ["taskkill", "/F", "/T", "/PID", str(proc.pid)]
        options = {}
    else:
        argv = [sys.executable, "-c", _POSIX_REAPER, str(proc.pid),
                str(REAP_GRACE_S)]
        options = {"start_new_session": True}
    try:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, **options)
    except OSError:
        return False
    return True


def _end_tree(proc) -> None:
    """The body of `end_process_tree`, every step on its own limit."""
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
