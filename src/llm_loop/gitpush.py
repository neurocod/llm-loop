"""When a loop pushes what it has committed, and the two git calls that decide.

The policy is the same for both runners and belongs to neither: the sequential
loop applies it at the top of every iteration, the parallel one from its own
timer thread, and BOTH end with `final_git_push`. It lived in `cyclecore` only
because `git_push` had to print, and printing lived there too — the moment the
terminal moved to `console`/`termio` the cut became clean, and a module that had
to import a runner back stopped being one.

THE REPOSITORY IS A PARAMETER, NOT AN AMBIENT FACT. Every call here takes the
directory to run git in. The old spelling read the runner's own project-root
global directly (`cyclecore.PROJECT_DIR`, a name that no longer exists anywhere
— the root is `projectroot` now), which is what tied the policy to the runner:
a caller could not push a
repository the sequential runner did not happen to be pointed at, and the module
could not be tested without setting a runner's global. The engine is meant to be
vendored under a host project whose root is NOT the process cwd (see
`projectroot.set_project_root`), so "wherever git happens to land" is a wrong
answer, not merely an unspecified one.

The root has since become a leaf of its own (`projectroot`), so reading it here
would no longer cost an import of a runner — and the parameter stays anyway.
The reason was never the import: a git push is about A REPOSITORY, and this
module has no business assuming which one.
"""

from enum import Enum
import subprocess
import threading
import time
from typing import Callable, Optional

from .console import LINES, print_done, print_error


class GitPushPolicy(Enum):
    """When the loop should run `git push` between iterations.

    Checked at the start of every iteration (see ``maybe_git_push``):

      * ``NONE``            — never push automatically.
      * ``AFTER_NEW_COMMITS`` — push whenever HEAD is ahead of its upstream
        (i.e. there are local commits that haven't been pushed yet).
      * ``EACH_HOUR``       — push at most once per hour, and only when there is
        something to push.
    """
    NONE = "none"
    AFTER_NEW_COMMITS = "after_new_commits"
    EACH_HOUR = "each_hour"


# Default push policy. Override on the command line with --git-push.
GIT_PUSH_POLICY = GitPushPolicy.EACH_HOUR

# What this knob is called on the pinned row: each runner registers a
# `statusline.Setting` under this label.
# A constant because `statusline.colorize` anchors on it to find the VALUE it has
# to light up: the policy words are ordinary English ("none"), so the label is
# what tells a mode word from a file called none.md on a job row. Renaming the
# knob here therefore moves the colouring with it instead of silently losing it.
GIT_PUSH_SETTING = "git-push"

# EACH_HOUR cadence: push no more often than this many seconds.
GIT_PUSH_INTERVAL = 3600  # seconds — one hour


class PushAbort:
    """A Ctrl+C's hold on a push: once `set` has returned, no git child of it starts.

    A plain Event could not say that. The push is on another thread (the
    parallel runner's pusher), so "is it set?" and "start git" are two steps,
    and a `set` landing between them — while the push printed its announcement,
    say — let a `git push` start after the operator had abandoned it. Here the
    check and the child's start are one step under `_lock`, and `set` takes the
    same lock: it waits out a start in progress (a process spawn, milliseconds,
    never the git call itself), and every start after it is refused.

    A child started before `set` is not stopped from here — see
    `final_git_push`.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._set = False

    def set(self) -> None:
        with self._lock:
            self._set = True

    def is_set(self) -> bool:
        return self._set

    def start(self, spawn: Callable[[], "subprocess.Popen"]
              ) -> "Optional[subprocess.Popen]":
        """`spawn()`'s process, or None — and nothing spawned — once set."""
        with self._lock:
            if self._set:
                return None
            return spawn()


def _run_git(argv, cwd: str, timeout: float,
             abort: Optional[PushAbort] = None
             ) -> "Optional[subprocess.CompletedProcess]":
    """`subprocess.run` of one git call, or None when `abort` refused its start.

    Without `abort` it IS `subprocess.run`. With one the child is started under
    the abort's lock (`PushAbort.start`) and then waited for outside it, the
    way `subprocess.run` does: killed on its timeout (TimeoutExpired raised
    on) and on any exception out of the wait.
    """
    options = dict(cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                   text=True, encoding="utf-8", errors="replace")
    if abort is None:
        return subprocess.run(argv, timeout=timeout, **options)
    proc = abort.start(lambda: subprocess.Popen(argv, **options))
    if proc is None:
        return None
    with proc:
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        except BaseException:
            proc.kill()
            raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout=out)


def git_unpushed_count(cwd: str, *, abort: Optional[PushAbort] = None
                       ) -> Optional[int]:
    """Number of local commits ahead of the upstream branch (HEAD not yet pushed).

    Returns the count, or None if it can't be determined (no upstream configured,
    git missing, not a repo, …) — in which case callers treat a push as worth
    attempting rather than silently skipping. None too when `abort` refused to
    start git; the caller asks `abort` to tell the two apart.
    """
    try:
        proc = _run_git(["git", "rev-list", "--count", "@{u}..HEAD"], cwd, 30,
                        abort)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc is None or proc.returncode != 0:
        return None
    try:
        return int((proc.stdout or "").strip())
    except ValueError:
        return None


def git_push(cwd: str, *, abort: Optional[PushAbort] = None) -> bool:
    """Run `git push` in `cwd`, printing the outcome. Returns True on success.

    False, and silent, when `abort` refused to start it: whoever set it says why.
    """
    try:
        proc = _run_git(["git", "push"], cwd, 300, abort)
    except FileNotFoundError:
        print_error("  · git push skipped: 'git' not found on PATH.")
        return False
    except subprocess.TimeoutExpired:
        print_error("  · git push timed out.")
        return False
    if proc is None:
        return False
    if proc.returncode == 0:
        print_done("  · git push: done.")
        return True
    LINES.fitted(f"  · git push failed (exit {proc.returncode}): ",
                 proc.stdout or "", "bold red")
    return False


def final_git_push(policy: GitPushPolicy, cwd: str, *,
                   abort: Optional[PushAbort] = None) -> None:
    """Push whatever is still local on the way out of a run.

    Regardless of the EACH_HOUR cadence: the run is ending, so work must not be
    left only on the local branch. A no-op for NONE (never auto-push), and the
    "nothing to push" case says so rather than staying silent, because a run that
    printed nothing about pushing reads like a run whose push failed.

    THIS FUNCTION DOES NOT EXCLUDE ANYTHING, and that is the answer to "who owns
    the mutual exclusion of the exit push": the caller does. Both runners end
    with this call and only one of them has threads: the parallel one has it
    made by the same thread that makes its periodic pushes (see
    `runlifecycle.close_run`), and the sequential one has nothing to exclude. A
    function that pushes A REPOSITORY has no business knowing whether its caller
    is threaded (the same argument that made `cwd` a parameter; see the header).

    `abort` (a `PushAbort`, or None): once it is set, the push starts no
    further git and returns quietly — the caller that set it says why. Asked
    AT each child's start, under the abort's lock (`PushAbort.start`), not
    before the step that leads to it: the announcement printed between the
    count and the push can block on a stalled console, and a Ctrl+C heard
    meanwhile must still stop the push. A git call already running is not
    stopped from here: a Ctrl+C typed at the terminal reaches the git child
    too (one console on Windows, one foreground process group on POSIX), and
    anything else is bounded by the call's own subprocess timeout.
    """
    if policy == GitPushPolicy.NONE:
        return
    count = git_unpushed_count(cwd, abort=abort)
    if abort is not None and abort.is_set():
        return
    if count is None or count > 0:
        print("  · final git push on exit…")
        git_push(cwd, abort=abort)
    else:
        print("  · final git push: nothing to push.")


def maybe_git_push(policy: GitPushPolicy, last_push: float, cwd: str) -> float:
    """Apply the GitPushPolicy at the start of an iteration.

    `last_push` is the epoch time of the previous push attempt (0.0 if never).
    Returns the updated `last_push` so the caller can carry it to the next
    iteration. A no-op for NONE; pushes when commits are pending for
    AFTER_NEW_COMMITS; for EACH_HOUR pushes pending commits at most once an hour.
    """
    if policy == GitPushPolicy.NONE:
        return last_push

    if policy == GitPushPolicy.AFTER_NEW_COMMITS:
        count = git_unpushed_count(cwd)
        if count is None or count > 0:
            if git_push(cwd):
                return time.time()
        return last_push

    if policy == GitPushPolicy.EACH_HOUR:
        now = time.time()
        if now - last_push < GIT_PUSH_INTERVAL:
            return last_push
        # An hour has passed — push if there is anything to push, and reset the
        # timer either way so we re-check at most once per hour.
        count = git_unpushed_count(cwd)
        if count is None or count > 0:
            git_push(cwd)
        return now

    return last_push
