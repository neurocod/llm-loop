"""Codex compatibility entry point for the shared optional screen counter."""
from .. import agent_counter

DISABLE_ENV = "CODEX_AGENTS_COUNTER"


def announce(label: str):
    return agent_counter.announce(label, agent_type="codex", disable_env=DISABLE_ENV)
