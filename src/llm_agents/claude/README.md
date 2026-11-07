# `llm_agents.claude` — Claude Code as a named worker

The other half of [`llm_agents.codex`](../codex/README.md): this is how Codex
has Claude review its commits (the reasoning for two independent opinions is
there).

Recipe (Claude Code 2.1.259+; name the exact commit SHAs in the task file):

```text
python -m llm_agents.claude spawn_agent review-name --message-file review_prompt.txt
python -m llm_agents.claude wait_agent review-name --turn 1 --until-done
python -m llm_agents.claude send_message review-name --message-file review_followup.txt
python -m llm_agents.claude wait_agent review-name --turn 2 --until-done
python -m llm_agents.claude list_agents
python -m llm_agents.claude interrupt_agent review-name
```

A project with rules of its own runs its own entry point instead, which calls
`cli.main(policy=..., root=..., entry=...)`; `entry` is the path the detached
worker and the printed wait command are started by, so they run under the
same host. The `review` profile needs the host's gate hook
(`HostPolicy.review`) and refuses to start without one.

The other commands and options: `python -m llm_agents.claude --help`. Tests:
`tests/claude_agent/` of this repository.
