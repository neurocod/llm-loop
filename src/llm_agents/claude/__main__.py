"""`python -m llm_agents.claude <command>`, or `python <src>/llm_agents/claude`.

Without a host this runs under the empty policy and says so on stderr. A
project with rules of its own calls `cli.main(policy=..., root=...,
entry=...)` from its own entry point; the detached worker is started through
that `entry` (`worker.launch`), so it runs under the same host. The path line
below has the reason of `llm_agents/codex/__main__.py`.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from llm_agents.claude.cli import main
else:
    from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
