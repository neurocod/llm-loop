"""Named Claude workers; see cli.build_parser for the command interface.

Each submitted message is a turn. A detached worker drains the durable queue,
starting Claude's stream-json CLI once per turn and resuming its session for
followups. Idle agents consume no processes. Messages sent during a turn are
queued, not injected into the active model request. Only result events become
answers; intermediate tool output remains in per-turn logs.

Run it as `python -m llm_agents.claude <command>`; a project with rules of its
own calls `cli.main(policy=..., root=..., entry=...)` from an entry point of
its own (`llm_agents.host_policy` says what a host supplies).
"""
