"""Named Codex and Claude workers driven the way a subagent is driven.

`llm_agents.codex` speaks Codex's app-server protocol, `llm_agents.claude`
Claude Code's stream-json CLI; both keep a registry of named agents, print
only the final answer, and run a command a sandboxed worker asks for outside
the sandbox only when a judge lets it out. See `codex/README.md` and
`claude/README.md` beside this file.

What a project adds — its command tables, protected paths, briefing lines,
review hook — arrives as a `host_policy.HostPolicy` handed to `cli.main`;
without one the bridges run with the empty policy and say so.

The modules beside the two bridges are what both share: `host_policy`,
`answer_contract` (the `## Coverage` contract and its follow-up),
`child_job` (a Windows Job Object that ends a child's tree with its owner),
`agent_counter` (the client of an optional on-screen counter) and `gitlib`
(the git runner). `git` is the git kit for several agents in one repository
(a worktree pool, a pathspec-exact commit helper, the commit-approval store),
which the Codex bridge's commit judge reads its rules from.

A separate top-level package from `llm_loop`, which it imports and which never
imports it: `llm_loop`'s lazy front door stays untouched.
"""
