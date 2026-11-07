# llm_agents.git — a git kit for several agents in one repository

Several agents writing one repository at once trip over git in a few
predictable ways: a commit sweeps up hunks another agent staged, two commits
collide on `.git/index.lock`, two writers share one working tree, a landed
submodule commit is nowhere a fresh clone can fetch it from. This package is
the set of scripts that keeps that from happening. Standard library only,
Python 3.9+.

| Module | What it does |
|---|---|
| `worktree.py` | A pool of standing worktree **slots** beside the main checkout (`<repo>-wt1`, ... on branches `wt1`, ...): `status`, `new`, `equalize`, `land`, `next`, `release`, `drop`, and Claude Code's `WorktreeCreate`/`WorktreeRemove` hooks, which hand each `isolation: "worktree"` agent a free slot and lock it for the task. |
| `commit.py` | Commits exactly the named paths, whatever else is staged; message from a file or `-m` paragraphs, never re-quoted; waits out a held index lock; `-C <dir>` to commit in a slot from anywhere. |
| `check_gitlinks.py` | Refuses a gitlink without its `.gitmodules` stanza (and the reverse); `commit.py` runs it on what it is about to commit. |
| `branch_overlap.py` | Which files two diverged branches both touched, and which of ours the other side moved or deleted. |
| `union_conflicts.py` | Resolves conflict hunks by keeping both sides, in a chosen order. |
| `slots.py` | The pool's naming rule, shared with the Codex bridge's commit judge. |
| `commit_identity.py` | The approval store that binds a sandboxed worker's commit to the checkout its judge approved. |

## Running

```
python -m llm_agents.git.worktree status
python -m llm_agents.git.commit -F msg.txt src/a.py docs/b.md
python -m llm_agents.git.commit -C ../repo-wt1 -m "Subject" -m "Body" src/a.py
```

Run this way the scripts use the empty host and say so on stderr.

## A host's rules

A project hands its rules to `main` from a small entry point of its own:

- `commit.CommitHost`: names never committed (a loop's stop file), checks that
  refuse a commit before staging and again before committing, the wording of
  `-C`;
- `worktree.PoolHost`: how the project spells the command, the submodules a
  new slot gets, directories only the main checkout fills (`SharedDir`,
  linked into every slot), the dependency install step, notes and help texts.

```python
from llm_agents.git import worktree

POOL = worktree.PoolHost(command="python tools/worktree.py",
                         default_submodules=("vendor/lib",))
raise SystemExit(worktree.main(POOL))
```

## Platforms

Shared directories are directory junctions on Windows (no administrator
rights needed) and symlinks elsewhere. Before git removes a slot, `drop`
detaches every link in it that points outside -- the host's or anybody
else's -- because Git for Windows follows a junction while removing.
The index-lock wait recognises both POSIX and Windows wording.
