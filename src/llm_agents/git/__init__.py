"""The git kit for several agents at work in one repository (see README.md).

`worktree` keeps a pool of standing worktree slots beside the main checkout and
lands their branches; `commit` commits exactly the named paths, whoever else
staged what; `check_gitlinks`, `branch_overlap` and `union_conflicts` are the
small checks and repairs around them; `slots` is the pool's naming rule and
`commit_identity` the commit-approval store, both shared with the Codex
bridge's commit judge (`llm_agents.codex.command_commit`).

What a project adds arrives explicitly: a `commit.CommitHost` and a
`worktree.PoolHost`, handed to each script's `main` by the project's own entry
point. Importing this package imports none of the modules.
"""
