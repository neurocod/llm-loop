"""
drivers.py - reusable Driver implementations for the autonomous Claude-CLI loop.

These are the two task shapes the loop was originally built around, generalised
so a host project supplies its own paths, prompts and models instead of them
being hard-coded module constants:

  * StateFileDriver - a state machine: read the first line of a state file each
    iteration and run a fixed prompt against it, stopping on an `error` state.
  * ListFileDriver  - a work queue: hand out the files listed in a list file one
    at a time (at random by default, to spread evenly across a grouped list; set
    `pick_order = "list"` to walk it top to bottom instead), striking each out
    once its command succeeds; stop when the list is empty.

Both resolve their relative paths against cyclecore.project_dir() (the project
root, set from --project-dir / cwd), so the same code drives any host project.
"""

import os
import random
from typing import Optional

from . import cyclecore
from .cyclecore import ClaudeCommand, Driver, LoopStop


def _abs_in_project(path: str) -> str:
    """Resolve a path against the project root; absolute paths pass through."""
    return path if os.path.isabs(path) else os.path.join(
        cyclecore.project_dir(), path)


# --- state-machine driver ------------------------------------------------------

class StateFileDriver(Driver):
    """Read the first line of a state file each iteration and run a fixed prompt.

    The loop never ends on its own (the state machine is meant to run forever);
    it stops only on an `error` state — which aborts the run with exit code 1 —
    or via the usual stop file / Ctrl+C / --max-runs.

    Configure via class attributes on a subclass:
      state_file     relative (to the project root) or absolute path to the state
                     file whose first line is the current state.
      error_token    substring (case-insensitive) in the state line that aborts
                     the run for human intervention. Default: "error".
      app_name / prog / description — the entry-point labels; app_name and prog
                     default to the wrapper's filename (see Driver), so usually
                     only description (if any) is worth setting.

    Override methods to customise behaviour:
      prompt()  -> the instruction sent every iteration (default:
                   "Follow the instructions in <state_file>").
      model()   -> pin a model, or vary it by state (read self.first_line()
                   inside). Default: "" — the CLI's own configured model.

    Entry point: ``MyStateDriver.main()``.
    """

    state_file: str = "products/currentState.md"
    error_token: str = "error"

    def _state_path(self) -> str:
        return _abs_in_project(self.state_file)

    def first_line(self) -> str:
        """First line of the state file (empty string if the file is missing)."""
        try:
            with open(self._state_path(), "r", encoding="utf-8") as f:
                return f.readline().strip()
        except FileNotFoundError:
            return ""

    def prompt(self) -> str:
        """The instruction sent every iteration. Override to change the playbook."""
        return f"Follow the instructions in {self.state_file}"

    def next_command(self) -> Optional[ClaudeCommand]:
        state = self.first_line()
        # State-machine contract: on error we do not proceed.
        if self.error_token.lower() in state.lower():
            raise LoopStop(
                f"State '{state}' — error detected. Stopping, "
                f"human intervention required.",
                exit_code=1,
            )
        label = state or f"{self.state_file} not found"
        return ClaudeCommand(self.prompt(), self.model(), label)

    def final_summary(self) -> Optional[str]:
        return f"Final state: {self.first_line()}"


# --- list / work-queue driver --------------------------------------------------

def _read_list_lines(list_path: str) -> list:
    """All raw lines of the list file (without trailing newlines).

    Returns [] if the file is missing, so an absent/emptied list just ends the
    loop cleanly. utf-8-sig strips a leading BOM so it never corrupts the first
    path into a relative-looking name.
    """
    try:
        with open(list_path, "r", encoding="utf-8-sig") as f:
            return [ln.rstrip("\n") for ln in f]
    except FileNotFoundError:
        return []


def _write_list_lines(list_path: str, lines: list) -> None:
    """Rewrite the list file: newline-joined, trailing newline iff non-empty."""
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        if lines:
            f.write("\n")


class ListFileDriver(Driver):
    """Hand out files from a list file one at a time, striking each out of the
    list once its command succeeds.

    Progress lives in the list file itself (done paths are removed), so the run
    is idempotent: stop any time and relaunch — it picks up whatever paths are
    still listed. `next_command` re-reads the list, hands one still-pending line
    to `pick()`, and returns None — ending the loop — once nothing is pending. A
    failed iteration leaves its line in the list (on_success is what strikes it),
    so it gets retried on some later iteration.

    Configure via class attributes on a subclass:
      list_file      relative (to the project root) or absolute path to the list.
      target_suffix  output sibling suffix, e.g. ".ru.md"; lines already ending
                     in it are skipped as already-done / self-referential.
      source_ext     source extension replaced by target_suffix when deriving the
                     target path (default ".md": foo.md -> foo<suffix>).
      pick_order     which pending line goes out next: "random" (default) or
                     "list" (top to bottom). See pick().
      app_name / prog / description — the entry-point labels; app_name and prog
                     default to the wrapper's filename (see Driver), so usually
                     only description (if any) is worth setting.
      jobs           default worker count for .main_parallel() (None -> the
                     engine default unless -j/--jobs is passed). Set it as a class
                     attribute or `self.jobs = N` in __init__ to bake in a default;
                     an explicit -j/--jobs on the CLI always wins.

    Override `prompt(source_abs, target_abs)` (required — it builds the per-file
    instruction) and `model()` to pin/vary the model (default: "" — the CLI's own
    configured model; mechanical work like translation often warrants a cheaper
    one).

    Entry points: ``MyListDriver.main()`` (sequential), or
    ``MyListDriver.main_parallel()`` (N concurrent `claude` workers).
    """

    list_file: str = "products/list.md"
    target_suffix: str = ".ru.md"
    source_ext: str = ".md"
    # Which pending line goes out next — see pick(). "random" is the default
    # because it spreads a run evenly across a category-grouped list.
    pick_order: str = "random"
    PICK_ORDERS = ("random", "list")
    # Default concurrent-worker count for .main_parallel(); None means "defer to
    # the CLI -j/--jobs, else the engine default". A subclass can pin it.
    jobs: Optional[int] = None

    def __init__(self, jobs: Optional[int] = None):
        self._current_line: Optional[str] = None  # raw list line being processed
        if jobs is not None:
            self.jobs = jobs
        # Validated once, at construction: a typo here would otherwise pass
        # silently as "random" and quietly ignore what the wrapper asked for.
        if self.pick_order not in self.PICK_ORDERS:
            raise ValueError(
                f"{type(self).__name__}.pick_order={self.pick_order!r} — "
                f"expected one of {', '.join(map(repr, self.PICK_ORDERS))}")

    def prompt(self, source: str, target: str) -> str:
        """The per-file instruction (receives absolute paths). Override this — it
        is the one project-specific piece a list driver must supply."""
        raise NotImplementedError(
            "ListFileDriver subclasses must override prompt(source, target)")

    def list_path(self) -> str:
        return _abs_in_project(self.list_file)

    def is_pending(self, line: str) -> bool:
        """A list line that still names a file to process (skips blanks, `#`
        comments, and any path already pointing at a target_suffix output)."""
        s = line.strip()
        if not s or s.startswith("#"):
            return False
        if s.lower().endswith(self.target_suffix.lower()):
            return False
        return True

    def target_path(self, source: str) -> str:
        """`<name><target_suffix>` derived from the source path."""
        if source.lower().endswith(self.source_ext.lower()):
            return source[: -len(self.source_ext)] + self.target_suffix
        return source + self.target_suffix

    def pending_lines(self) -> list:
        """All raw list lines still naming work to do (re-read each call)."""
        return [ln for ln in _read_list_lines(self.list_path())
                if self.is_pending(ln)]

    def command_for(self, line: str) -> ClaudeCommand:
        """Build the command for one raw list line, without touching driver
        state — safe to call from any thread (used by the parallel runner)."""
        source = line.strip()
        source_abs = _abs_in_project(source)
        target_abs = self.target_path(source_abs)
        label = os.path.basename(source_abs)
        return ClaudeCommand(self.prompt(source_abs, target_abs),
                             self.model(), label)

    def strike(self, line: str) -> bool:
        """Remove the first list entry exactly matching `line`; rewrite the file.

        Returns True if a line was removed, False if it was already gone (e.g.
        hand-edited out). The list's single source of truth — both runners go
        through here so the file always keeps the same shape.
        """
        list_path = self.list_path()
        lines = _read_list_lines(list_path)
        for i, ln in enumerate(lines):
            if ln == line:
                del lines[i]
                break
        else:
            return False
        _write_list_lines(list_path, lines)
        return True

    def pick(self, pending: list) -> str:
        """Choose the next line to hand out from the still-pending ones.

        Both runners go through here — the sequential one below and the parallel
        runner's claim() — so `pick_order` (and any override of this method)
        governs the whole family. The two orders answer different questions:

          "random" — spread the run evenly across a grouped list. The head would
                     otherwise drain one category at a time, so an interrupted
                     run leaves a lopsided sample of the catalogue.
          "list"   — walk the list top to bottom, i.e. the order it was written
                     in. Reach for it when the list is deliberately ordered
                     (hand-picked priorities first) or when a run has to be
                     reproducible.

        Must be thread-safe: the parallel runner calls it under its own lock
        with the driver shared across workers.
        """
        return pending[0] if self.pick_order == "list" else random.choice(pending)

    def next_command(self) -> Optional[ClaudeCommand]:
        pending = self.pending_lines()
        if not pending:
            self._current_line = None
            return None
        self._current_line = self.pick(pending)
        return self.command_for(self._current_line)

    def on_success(self, returncode: int) -> None:
        """Remove the just-processed path from the list (the list is the source of
        truth for what is left to do)."""
        if self._current_line is None:
            return
        if self.strike(self._current_line):
            self._current_line = None

    def final_summary(self) -> Optional[str]:
        remaining = len(self.pending_lines())
        if remaining == 0:
            return f"All items done — {self.list_file} is empty."
        return f"{remaining} item(s) still pending in {self.list_file}."

    @classmethod
    def main_parallel(cls, argv=None, jobs: Optional[int] = None) -> None:
        """Parse the parallel CLI (adds -j/--jobs) and drain the list with N
        concurrent `claude` workers over a fresh instance.

        The parallel counterpart of Driver.main(): a wrapper for the concurrent
        runner is just a subclass calling this. Imported lazily so drivers.py and
        parallel.py don't form an import cycle.

        Worker-count precedence (first that is set wins): an explicit -j/--jobs on
        the command line, the `jobs=` argument here, the driver's `jobs` attribute
        (class attribute or `self.jobs` set in __init__), then the engine default.
        """
        from .parallel import run_parallel
        from .parallel import parse_args as parse_parallel_args
        args = parse_parallel_args(argv, prog=cls.resolved_prog(),
                                   description=cls.description)
        driver = cls()
        if jobs is not None:
            driver.jobs = jobs
        run_parallel(driver, args, app_name=cls.resolved_app_name())
