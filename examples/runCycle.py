"""
Example wrapper: drive a state machine from a state file - currentState.md or another

This is the StateFileDriver pattern. Each iteration reads the first line of a
state file and runs a fixed prompt against it; on an `error` state the loop stops
for human intervention, otherwise it runs forever (until the stop file, --max-runs, or
Ctrl+C). The state lives in files in your project, so each `claude` call starts
with fresh context and picks up where the last one left off — the canonical
"Ralph" pattern.

A typical setup: `currentState.md` names the current phase, and your prompt tells
claude to follow a playbook (e.g. read a TODO list, do one item, update the
state). Override `model()` to pin or vary the model by state if you like; leave it
alone to let the `claude` CLI use its own configured model.

Copy this into your host project root (next to the `tools/claude-loop` submodule),
adjust the class attributes, and run `python runCycle.py`.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tools", "claude-loop"))

from claude_loop import StateFileDriver

STATE_FILE_REL = "currentState.md"


class CycleDriver(StateFileDriver):
    state_file = STATE_FILE_REL
    description = f"Autonomous loop driving the Claude CLI per {STATE_FILE_REL}."

    def model(self) -> str:
        """Choose the model from the current state. Override as needed, or delete
        this method to let the CLI use its own configured model."""
        if "implementation" in self.first_line().lower():
            return "opus"
        return "opus"


if __name__ == "__main__":
    CycleDriver.main()
