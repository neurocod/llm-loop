"""Ending a child process tree: the escalation every owner of a child shares.

A leaf module on purpose — standard library only — so both the turn runner
(`providers`) and the quota server (`codex_usage`, which `providers` imports)
can reach it without an import running against that direction. Each caller
keeps only its own prologue (which pipe to close first, whether the child gets
an EOF to exit on) and hands the rest to `end_process_tree`.
"""

import os
import subprocess

# How long a provider child gets after being told to end before it is killed
# outright, and the ceiling on the `taskkill` call itself. The bound is what
# keeps the reaping from becoming a second hang inside the guard that exists to
# end the first one: the parallel runner calls it from a thread that is ALREADY
# dying, so nothing above it is left to notice a wait that never returns.
REAP_GRACE_S = 2.0


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


def end_process_tree(proc) -> None:
    """Ask the child (tree) to end, wait, kill, wait — bounded, never raises.

    At most `REAP_GRACE_S` for the ask's `taskkill` plus two waits of
    `REAP_GRACE_S` each. `OSError` from `wait` means the OS has already
    reclaimed the handle (Windows raises it there), so there is nothing left to
    end; `KeyboardInterrupt` is NOT absorbed here — that is the caller's call
    (`providers.reap_agent_process`).
    """
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
