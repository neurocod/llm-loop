"""
Example wrapper: drive a state machine from a state file - currentState.md or another

This is the StateFileDriver pattern. Each iteration reads the first line of a
state file and runs a fixed prompt against it; on an `error` state the loop stops
for human intervention, on a `done` / `complete` state it ends cleanly, otherwise it
runs forever (until the stop file, --max-runs, or Ctrl+C). The state lives in files
in your project, so each provider call starts
with fresh context and picks up where the last one left off — the canonical
"Ralph" pattern.

A typical setup: `currentState.md` names the current phase, and your prompt tells
the agent to follow a playbook (e.g. read a TODO list, do one item, update the
state). Override `model()` to pin or vary the model by state if you like; leave it
alone to let the selected CLI use its own configured model.

Copy this into your host project root (next to the `tools/llm-loop` submodule),
adjust the class attributes, and run `python runCycle.py`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools", "llm-loop", "src"))

from llm_loop import StateFileDriver

STATE_FILE_REL = "currentState.md"


class CycleDriver(StateFileDriver):
    state_file = STATE_FILE_REL
    description = f"Autonomous loop driving Claude or Codex per {STATE_FILE_REL}."

    def prompt(self) -> str:
        """Tell each fresh agent process where its state-machine playbook is."""
        return f"Follow the instructions in {self.state_file}"

    def model(self) -> str:
        """Explicit step providers override --provider/--codex. A provider alone
        uses its CLI's configured model; provider/model pins a specific model.
        Return "" instead to use the launch provider and its configured model.
        """
        if self.state_name() == "implementation":
            return "claude/opus"
        return "codex"


if __name__ == "__main__":
    CycleDriver.main()
