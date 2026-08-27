"""What one unit of work IS, and what a task must answer to produce them.

This is the contract between a host project's wrapper and the engine: a
``Driver`` subclass says what to do next, an ``AgentCommand`` is one such answer,
``build_agent_argv`` turns it into the selected provider's command line, and
``LoopStop`` is how a driver aborts the whole run. Both runners EXECUTE that
contract; neither owns it, which is why it is not in either of them.

It lived in `cyclecore` (the sequential runner) for the usual reason a thing
lives where it was first written, and the bill was paid by everyone else:
`drivers.py` imported the sequential loop for three names and used none of it,
and so did the parallel runner. A wrapper that never runs `run_loop` still had to
import it to name the base class it subclasses.

``Driver.main`` reaches the sequential runner through a LOCAL import, exactly as
``ListFileDriver.main_parallel`` already reaches the parallel one: the entry
points are the one place the contract has to know a runner, and a module-level
import in either direction would put the cycle back.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import NamedTuple, Optional

from . import projectroot
from . import stopchannel
from .providers import build_agent_argv as provider_argv


class AgentCommand(NamedTuple):
    """One unit of work for the loop: the prompt to send, the model to use, and a
    short label shown in the iteration header. Drivers build these in
    next_command(); build_agent_argv() turns one into the provider's full argv.

    An empty `model` means "no --model flag": the selected provider then uses
    its configured default, which is the common case.
    """
    prompt: str
    model: str = ""
    label: str = ""
    provider: str = ""
    sandbox_mode: str = ""


ClaudeCommand = AgentCommand


def build_agent_argv(command: AgentCommand, provider: Optional[str] = None) -> list:
    """Full provider command line for one unit of work."""
    provider = provider or command.provider or "claude"
    return provider_argv(command, provider, projectroot.project_dir())


def build_claude_argv(command: ClaudeCommand) -> list:
    """Full `claude` command line for one ClaudeCommand.

    The flags are identical for every task; only the prompt and the model vary,
    so this is the single place those two are spliced into the otherwise fixed
    argv (stream-json + partial messages so the loop can render work live). An
    empty `command.model` omits --model entirely, letting the CLI pick its own
    configured default.
    """
    return build_agent_argv(command, "claude")


class LoopStop(Exception):
    """Raised by a Driver to abort the whole run (not a normal completion).

    `exit_code` is the process exit status: non-zero for an error stop that needs
    a human (the loop sys.exit()s immediately, skipping the final push), 0 for a
    clean stop. `message` is printed before exiting.
    """

    def __init__(self, message: str, exit_code: int = 0):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class Driver:
    """What the generic loop needs from a task. Subclass and override.

    A Driver is customised two ways, both declarative:

      * class attributes for the labels the entry points use — ``app_name``
        (names the rotating mirror log) and ``prog`` (the --help program name)
        both default to None and are then derived from the invoked script's
        filename (runTranslate.py -> "runTranslate" / "runTranslate.py"), so a
        typical wrapper need not set them at all; ``description`` (the --help
        description) is free prose, so set it if you want one; ``limit_policy``
        (a limits.LimitPolicy) picks the usage-limit specialisation, defaulting
        to a day/night session rule when unset; ``sandbox_mode`` selects an
        explicit Codex sandbox for this driver's commands. Override any of them
        on your subclass to pin an explicit value.
      * methods for behaviour — ``next_command()``, ``model()``, ``on_success()``,
        ``final_summary()``. Override the ones you need; the rest keep their
        default.

    The loop owns all the scaffolding (stop file, git push, usage limits,
    --max-runs, streaming render); the Driver only decides *what work to
    do*. A project wrapper is then just::

        class MyDriver(StateFileDriver):
            state_file = "products/currentState.md"
            app_name   = "runCycle"

            def prompt(self):
                return f"Follow the instructions in {self.state_file}"

        if __name__ == "__main__":
            MyDriver.main()

    ``main()`` parses the shared CLI and hands a fresh instance to run_loop(); the
    subclass never touches parse_args / run_loop by hand.
    """

    # --- labels used by the entry points (override on the subclass) -----------
    # None => derive from the invoked script's filename (see resolved_app_name /
    # resolved_prog); set an explicit string to override.
    app_name: Optional[str] = None      # names the rotating mirror log file
    prog: Optional[str] = None          # --help program name
    description: Optional[str] = None   # --help description (None = generic)
    provider: str = "claude"            # may be overridden by --codex
    # Optional explicit Codex sandbox for every command this driver builds.
    # Claude ignores it. Keeping this on the driver lets one trusted workflow
    # opt into a broader boundary without changing every shared-loop invocation.
    sandbox_mode: str = ""

    # --- usage-limit specialisation (declarative, like the labels above) ------
    # A limits.LimitPolicy picking which quota(s) to gate on and at what
    # ceiling; None => the engine's default (a day/night session rule, see
    # limits.default_policy). Set it as a class attribute to specialise, e.g.
    #   limit_policy = LimitPolicy([SessionLimit(80)])            # flat session
    #   limit_policy = LimitPolicy([WeeklyLimit(90)])             # weekly cap
    #   limit_policy = LimitPolicy([DayNightLimit(), WeeklyLimit(90)])  # composite
    # LimitPolicy/rules are stateless, so a shared default instance is safe here.
    limit_policy = None

    @classmethod
    def resolved_app_name(cls) -> str:
        """The mirror-log label: ``app_name`` if set, else the invoked script's
        filename stem (runTranslate.py -> "runTranslate"). Deriving it keeps each
        entry point on its own log without the wrapper having to spell it out; the
        fallback covers odd argv[0] values (e.g. ``-c``)."""
        if cls.app_name:
            return cls.app_name
        return Path(sys.argv[0]).stem or "runCycle"

    @classmethod
    def resolved_prog(cls) -> str:
        """The --help program name: ``prog`` if set, else the invoked script's
        basename (runTranslate.py -> "runTranslate.py")."""
        if cls.prog:
            return cls.prog
        return os.path.basename(sys.argv[0]) or "runCycle.py"

    @classmethod
    def add_cli_options(cls, parser: argparse.ArgumentParser) -> None:
        """Add this wrapper's own options to the shared --help. Default: none.

        Called by main() and main_parallel() with the parser that is about to
        run, so an entry point can document (or genuinely add) options the
        engine knows nothing about. The usual case is a MODE switch — one the
        wrapper must read out of argv itself, because it decides which of the
        two parsers runs at all, and which therefore can never be a plain option
        of either. Undocumented, such a flag exists only in prose, and `--help`
        answers "there is no such option" to a user who is looking straight at
        the one they want.

        Both entry points call it, so a flag spelled the same in both modes is
        documented in both --helps from one override. Build the option strings
        from the same constants the wrapper's argv scan uses; a second spelling
        typed out here is a spelling that can drift.
        """

    def next_command(self) -> Optional[AgentCommand]:
        """The command to run this iteration, or None when work is exhausted and
        the loop should stop normally. May raise LoopStop to abort the run."""
        raise NotImplementedError

    def model(self) -> str:
        """The selected provider's model for this iteration.

        Called by next_command() implementations to fill in AgentCommand.model.
        The default returns "" — no --model flag, so the provider CLI uses its
        own configured model. Override this (the single model knob) to pin a
        specific model, pick a cheaper/faster one for mechanical work (e.g. a
        list driver translating files needs less than the main state machine), or
        vary the model per iteration (read whatever state you like inside).
        """
        return ""

    def pending_total(self) -> Optional[int]:
        """How many units of work are still waiting, or None when unknowable.

        This is the summary row's denominator, and it is asked of the driver
        because only the driver knows what its queue is: a list file's pending
        lines, a folder of requests, rows in a table. Re-read on every call (not
        cached) — the first answer is latched as the invocation's baseline and
        every later one moves the counter, so a queue that grows or shrinks under
        the run stays honestly described.

        None means "no total to report": the row then counts bare iterations,
        which is the right answer for a state machine that is meant to run
        forever, and the wrong one for anything with a finish line — a run whose
        row reads `iter 1` with no `/N` is usually a driver that forgot this.

        Report it in the unit an ITERATION works through, where you can: the row
        clamps the total against `--max-runs`, which counts iterations, so a
        driver whose one iteration clears several items (the kit-promotion pass
        empties its whole requests folder in one) reads `iter 3/3` under
        `--max-runs 3` while a dozen items are still waiting. Only the display
        is affected — no cap, gate or queue decision reads this — and only when
        such a driver is given a cap, which is why the honest count of what is
        left is still the better answer for it.
        """
        return None

    def on_success(self, returncode: int) -> None:
        """Called after an iteration whose provider CLI exited 0 — record progress
        here (mark a file done, advance a cursor). Default: nothing to do."""

    # --- the two boundaries of one unit of work -------------------------------
    # `on_success` records what the driver's OWN queue did; these two are about
    # the world around the run — a folder another agent writes into, a lock, a
    # budget — and they are where a driver gets to ask for the run to hand
    # control back to whoever launched it.
    #
    # Both boundaries exist because they answer different questions, and a host
    # is expected to override at most the one it needs:
    #
    #   * `item_started` sees the command that is about to be sent, so it can
    #     record the attempt or notice a condition BEFORE paying for a turn. It
    #     cannot cancel that turn — the item is already claimed and the runner
    #     is about to launch it — so a pause asked for here still lets this item
    #     run to the end;
    #   * `item_finished` sees the outcome, and is the cheaper boundary to watch
    #     the world on: whatever an iteration produced is on disk by the time it
    #     is called, so a wrapper reacting to what the agents themselves write
    #     (a request file, a lock, a report) wants this one.
    #
    # Returning a short reason from either asks the runner to STOP HANDING OUT
    # NEW WORK and return `RunStopReason.DRIVER_PAUSE` once whatever is in
    # flight has finished — the parallel runner therefore winds the whole fleet
    # down without cancelling a single turn, exactly as the `s` key does.
    # That is deliberately not the same as the two endings a driver already had:
    # `next_command() -> None` says the queue is empty, and `LoopStop` aborts
    # the invocation. A pause says "this runner call is over, ask me again" —
    # which is what a wrapper alternating two kinds of run needs, and what it
    # otherwise has to fake by slicing the work into fixed-size batches.
    #
    # THREAD SAFETY: the parallel runner calls both from its worker threads,
    # several at once. Keep them cheap and re-entrant (a listdir, a counter
    # under a lock); anything that mutates driver state has to guard it.

    def item_started(self, command: AgentCommand) -> Optional[str]:
        """About to send `command` to the provider. Default: nothing to do.

        Return a short reason to ask the run to hand control back once the work
        in flight (this item included) has finished; None to carry on.
        """
        return None

    def item_finished(self, command: AgentCommand,
                      returncode: int) -> Optional[str]:
        """`command` returned `returncode` and its outcome has been recorded.

        Called for failures as well as successes, and after `on_success` when
        there was one, so the driver's own queue is already up to date here.
        Return a short reason to ask the run to hand control back once the work
        still in flight has finished; None to carry on.
        """
        return None

    def final_summary(self) -> Optional[str]:
        """An optional closing line printed on the way out (after the final
        git push). Return None for no summary."""
        return None

    @classmethod
    def main(cls, argv=None) -> stopchannel.RunResult:
        """Parse the shared CLI and run the sequential loop over a fresh instance.

        This is the whole body of a project wrapper: subclass, override the
        methods you need, then ``if __name__ == "__main__": MyDriver.main()``.
        ``prog`` labels the --help text and ``app_name`` names the log; both are
        taken from the (sub)class or derived from the script filename when unset,
        and ``description`` is the (optional) --help blurb.

        Imported inside the method, like ``ListFileDriver.main_parallel``'s reach
        for the parallel runner: the runner imports this module for the contract,
        so the entry point can only borrow it back locally.
        """
        from .cyclecore import parse_args, run_loop
        args = parse_args(argv, prog=cls.resolved_prog(),
                          description=cls.description,
                          extra_options=cls.add_cli_options)
        return run_loop(cls(), args, app_name=cls.resolved_app_name())
