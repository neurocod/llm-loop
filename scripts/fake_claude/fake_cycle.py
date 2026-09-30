"""A real sequential runner driving the fake `claude` beside it: a free loop for
debugging the console.

    python tools/llm-loop/scripts/fake_claude/fake_cycle.py [runner options]

Everything but the agent is the real thing — status line, key reader, painter,
live-note channel, mirror log, exit record — so pressing `m` and typing a note
here walks the same path as in a paid run. The agent is `fake_claude.py`, found
through PATH exactly as the real CLI is (`providers.runtime_argv`), which is
why this script puts its own directory first on PATH and changes nothing else.

The project root is a scratch directory of its own (a state file and nothing
else), so no git push, no commit and no stop file ever touch a real project.
The key trace (`LLM_LOOP_KEYTRACE`) is on unless the caller set it; the run's
banner prints where the diagnostics go. While it runs, from another terminal:

    py-spy dump --pid <pid printed at start> --native

Tune the fake with FAKE_CLAUDE_SECONDS / FAKE_CLAUDE_DELTA_MS (see
fake_claude.py). Stop it as any run: `s`, or Ctrl+C.
"""

import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "src"))

from llm_loop import StateFileDriver  # noqa: E402
from llm_loop import diaglog  # noqa: E402


class FakeCycleDriver(StateFileDriver):
    state_file = "state.md"
    description = "Sequential runner against the fake claude (no model, no cost)."

    def prompt(self) -> str:
        return "Fake prompt: stream text until FAKE_CLAUDE_SECONDS pass."

    def model(self) -> str:
        return "claude"


def main() -> None:
    os.environ["PATH"] = HERE + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault(diaglog.KEYTRACE_ENV, "1")
    project = os.path.join(tempfile.gettempdir(), "llm-loop-fake-cycle")
    os.makedirs(project, exist_ok=True)
    with open(os.path.join(project, "state.md"), "w", encoding="utf-8") as f:
        f.write("Current state: fake\n")
    print(f"fake cycle: pid {os.getpid()}, project {project}")
    argv = sys.argv[1:]
    if "--project-dir" not in argv and "-C" not in argv:
        argv = ["--project-dir", project] + argv
    if "--git-push" not in argv and "-g" not in argv:
        argv = ["--git-push", "none"] + argv
    FakeCycleDriver.main(argv)


if __name__ == "__main__":
    main()
