# agent-tools

Small standalone scripts for LLM agents working in a repository: standard
library only, Python 3.9 or newer, no import of `llm_loop`. Copy one file or
the whole directory; each script's docstring says what it is for and what it
refuses to do, and `--help` prints its options.

They exist because the obvious one-liners are the ones an agent harness keeps
stopping on. A shell gate refuses `while ...; do sleep 1; done` and `sed -i`,
and a helper the agent writes into a scratch directory asks the human for
permission on every run. A reviewed script the human allows once does the same
job without the prompt.

| script | does |
| --- | --- |
| `follow_file.py` | prints a growing file until a pattern shows up or a timeout |
| `wait_file_appears.py` | waits until a file exists (optionally: matches a regex) |
| `wait_file_gone.py` | waits until a file disappears |
| `move_lines.py` | cuts whole-line ranges out of one file and appends them to another, byte-exact |

A host project that wants its own defaults (for example a `follow_file.py`
preset for its test runner's end marker) keeps a small wrapper script of its
own that passes them as arguments, rather than editing these copies.
