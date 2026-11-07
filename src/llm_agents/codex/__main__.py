"""`python -m llm_agents.codex <command>`, or `python <src>/llm_agents/codex`.

Without a host this runs under the empty policy and says so on stderr
(`host_policy.announce_empty`). A project with rules of its own does not run
this file: its entry point calls `cli.main(policy=..., root=...)` itself.

Handed a directory, Python runs this file with THAT directory on `sys.path`
rather than `src/`, so the package imports nothing; the one line below puts
`src/` first. FIRST, not appended: another copy reachable through PYTHONPATH
or site-packages would otherwise win and run its revision instead of this one.
Not the PEP 366 spelling (assigning `__package__`): 3.12+ then warns on every
run, and a wrapper whose output a model reads must print nothing of its own.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from llm_agents.codex.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    sys.exit(main())
