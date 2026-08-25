"""The ready-made drivers leave project-specific instructions to wrappers."""

import pytest

from llm_loop import StateFileDriver


def test_state_file_driver_requires_a_prompt_override():
    class MissingPrompt(StateFileDriver):
        def first_line(self):
            return "Current state: implementation"

    with pytest.raises(
            NotImplementedError,
            match=r"StateFileDriver subclasses must override prompt\(\)"):
        MissingPrompt().next_command()


def test_state_file_driver_accepts_a_prompt_override():
    class ProjectDriver(StateFileDriver):
        def first_line(self):
            return "Current state: implementation"

        def prompt(self):
            return f"Follow the instructions in {self.state_file}"

    command = ProjectDriver().next_command()

    assert command.prompt == (
        "Follow the instructions in products/currentState.md")
