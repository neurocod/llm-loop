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
| `cpu_load.py` | the machine's CPU load, the heaviest processes and a process tree's own share, once or as a stream (Windows) |
| `find_refs.py` | whole-repository text search for a name, submodules and untracked files included |
| `rename_gate.py` | the same search as a gate after a rename: any hit left exits 1 (needs `find_refs.py` beside it) |
| `agent_context_watch.py` | reads how full a Claude Code subagent's context is, or waits until it reaches a mark |
| `ci_failures.py` | a commit's GitHub Actions runs (every workflow) or one run: failed jobs and their pytest failures, each once with the jobs it hit, also while a run is going (needs `gh`) |

A host project that wants its own defaults (for example a `follow_file.py`
preset for its test runner's end marker, the `--orphan-image`s of
`cpu_load.py`, or the `--aside`/`--exclude` paths of `find_refs.py`) keeps a
small wrapper script of its own that passes them as arguments, rather than
editing these copies.
