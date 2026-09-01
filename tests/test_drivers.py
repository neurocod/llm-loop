"""The ready-made drivers leave project-specific instructions to wrappers."""

import pytest

from llm_loop import LoopStop, StateFileDriver


def driver_at(line):
    """A StateFileDriver whose state file's first line is `line`."""
    class Fixed(StateFileDriver):
        def first_line(self):
            return line

        def prompt(self):
            return "go"

    return Fixed()


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


def test_state_file_driver_stops_cleanly_on_done():
    with pytest.raises(LoopStop) as stop:
        driver_at("Current state: done").next_command()

    assert stop.value.exit_code == 0
    assert "finished" in stop.value.message


@pytest.mark.parametrize("line", [
    "Current state: complete",
    "Current state: Done — Phase 5 reached",
    "done",                       # no label at all
    "Current state: COMPLETE.",
])
def test_state_file_driver_reads_the_finish_word_at_the_start(line):
    with pytest.raises(LoopStop) as stop:
        driver_at(line).next_command()

    assert stop.value.exit_code == 0


@pytest.mark.parametrize("line", [
    "Current state: review complete",   # a step named with the word, not the state
    "Current state: done-list",         # a different word that starts the same way
    "Current state: completed",
    "Current state: implementation",
])
def test_state_file_driver_keeps_going_when_the_word_is_not_the_state(line):
    command = driver_at(line).next_command()

    assert command.prompt == "go"


def test_state_file_driver_error_wins_over_done():
    with pytest.raises(LoopStop) as stop:
        driver_at("Current state: done, with error").next_command()

    assert stop.value.exit_code == 1


def test_state_file_driver_done_tokens_are_a_class_knob():
    class Finished(StateFileDriver):
        done_tokens = ("finished",)

        def first_line(self):
            return "Current state: finished"

        def prompt(self):
            return "go"

    with pytest.raises(LoopStop) as stop:
        Finished().next_command()
    assert stop.value.exit_code == 0

    assert driver_at("Current state: finished").next_command().prompt == "go"
