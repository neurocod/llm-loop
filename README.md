# llm-loop

**Tools for running coding agents unattended — for hours, several at once, in
one repository.** Python 3.9+, no required third-party package (`rich` is
optional); one code path on Windows, Linux and macOS, and CI runs all three.

An agent CLI left alone fails in predictable places: it fills its context window
and forgets the plan; it runs out of quota mid-task; two of them stage each
other's files; one of them stops on a permission prompt nobody is watching. Each
part here closes one of those gaps, and each stands on its own — vendor the
repository and use the part you need.

| Part | Closes | Where |
|---|---|---|
| **The loop** | one long task, fresh context per step, quota-aware | [`llm_loop`](src/llm_loop/README.md) |
| **Agent bridges** | a Codex or Claude worker driven like a subagent; a review by the *other* model | [`llm_agents.codex`](src/llm_agents/codex/README.md), [`llm_agents.claude`](src/llm_agents/claude/README.md) |
| **Git kit** | several agents committing in one repository: a worktree pool, an exact commit | [`llm_agents.git`](src/llm_agents/git/README.md) |
| **Host rules** | what a project adds to the kit and the bridges, in one small module | [`examples/host_rules.py`](examples/host_rules.py) |
| **Test support** | pinning a host's rules against the same fake Codex wire the core is pinned on | [`llm_agents.codex.testing`](src/llm_agents/codex/testing/__init__.py) |
| **ask-user-gate** | a Claude Code hook that refuses the shell shapes that stop a session on a prompt | [`claude-plugin/ask-user-gate`](claude-plugin/ask-user-gate/README.md) |
| **agent-tools** | the small waits and edits an agent harness keeps asking permission for | [`agent-tools`](agent-tools/README.md) |
| **agent-counter** | optional: an on-screen count of running agents (Windows, Qt) | [`extras/agent-counter`](extras/agent-counter/README.md) |

A few words the sections below lean on:

- **host** — the project that uses this repository's tools; what it adds to
  them (`PoolHost`, `CommitHost`, `HostPolicy`) are its *host rules*. A tool
  run without any runs on the **empty host policy**: neutral defaults, nothing
  project-specific, said once on stderr.
- **vendor** — take the repository as a git submodule and put its `src/` on
  `sys.path`, instead of pip-installing it; the route the examples assume.
- **worker** — a Codex or Claude CLI process a bridge starts for one task,
  inside a sandbox; a **judge** is the function that decides whether one of
  its commands may run outside that sandbox.
- **gate** — a check that has to pass before a step goes on: a test suite, or
  a hook that refuses a command before it runs.
- **slot** — one standing worktree of the pool (`<repo>-wt1` on branch `wt1`),
  used by one agent at a time.

## The loop — `llm_loop`

Geoffrey Huntley's [Ralph](https://ghuntley.com/ralph/) is `while :; do cat
PROMPT.md | claude ; done`; this is that loop made survivable overnight. Each
iteration starts a headless CLI (`claude` or `codex`) with a fresh context
window, progress lives in files and git, and before every iteration the loop
reads the account's real quota and waits under a ceiling you set. A `Driver`
subclass of some fifteen lines says what one iteration does: a state machine
(`StateFileDriver`) or a work queue drained one item — or `-j N` items — at a
time (`ListFileDriver`).

```bash
git submodule add https://github.com/neurocod/llm-loop.git tools/llm-loop
cp tools/llm-loop/examples/runFileList.py .   # then edit the constants
python runFileList.py --dry-run               # the commands it would run
python runFileList.py --codex --max-runs 5
```

Per-iteration model choice, the pinned status line and its keys (`s` stop, `p`
hold, `m` a note to the running agent), the stop file, push policy, Ctrl+C and
the usage-limit rules are the loop's own manual:
[`src/llm_loop/README.md`](src/llm_loop/README.md). The playbook this
repository's own loop runs on is
[`examples/currentState.md`](examples/currentState.md).

## Agent bridges — `llm_agents.codex`, `llm_agents.claude`

Run Codex CLI or Claude Code as a **named worker** from a script or from another
agent: `spawn` an agent with a task, `send` a follow-up on the same thread,
read its `result`, `logs` and `status`. The turn itself stays in the agent's own
log; only the answer reaches the caller's context. Both bridges speak the CLI's
own protocol (Codex's app-server, Claude Code's stream-json), sandbox the worker,
and let a command out of the sandbox only when a judge says so.

The reason to have both: **a review by the other model finds what yours
missed.** One commit reviewed by Codex and by a Claude subagent gave 8 and 11
findings with 4 in common (measured 2026-08-24) — half of each list was visible
to one model alone. So Claude's commits get a Codex review and Codex's a Claude
review.

The Codex reviewer runs on an empty host, straight from the checkout:

```bash
python -m llm_agents.codex spawn review-1 --role reviewer --sandbox read-only --prompt-file review.txt
python -m llm_agents.codex result review-1
```

The Claude reviewer does not: its `review` profile (the default of `spawn`)
reuses the project's ask-user-gate hook and refuses to start without one
(`HostPolicy.review`; `python -m llm_agents.claude spawn --help` names the
profiles). So it runs through the project's own entry point, a script that
hands the bridge a policy with a `ReviewProfile` — `claude_agent_main` in
[`examples/host_rules.py`](examples/host_rules.py) is one:

```bash
python tools/claude_agent spawn review-2 --message-file review.txt
python tools/claude_agent wait review-2 --until-done
```

Run bare, a bridge uses the **empty host policy** and says so on stderr:
nothing leaves the sandbox unless the call itself asks (`--escalate`,
`--gates-from-prompt`). A project hands its rules to `cli.main(policy=...,
root=..., entry=...)` from an entry point of its own — see *Host rules* below.
The reasoning behind the sandboxes, how to cut a task so the worker returns one
result, and the Windows sandbox's traps are in the package READMEs.

## Git kit — `llm_agents.git`

Several agents writing one repository trip over git in a few predictable ways:
a commit sweeps up hunks another agent staged, two commits collide on
`index.lock`, two writers share one working tree. The kit is the set of scripts
that keeps that from happening:

- `worktree` — a pool of standing worktree **slots** beside the main checkout
  (`<repo>-wt1`, `-wt2`, … on branches `wt1`, …), with `new`, `equalize`,
  `land`, `drop`, and Claude Code's `WorktreeCreate`/`WorktreeRemove` hooks
  that hand each `isolation: "worktree"` agent a free slot;
- `commit` — commits exactly the named paths whatever else is staged, waits out
  a held index lock, and reads its message from a file so no shell re-quotes it.

```bash
python -m llm_agents.git.worktree status
python -m llm_agents.git.worktree new            # creates <repo>-wt1 on branch wt1
python -m llm_agents.git.commit -F msg.txt src/a.py docs/b.md
python -m llm_agents.git.commit -C ../repo-wt1 -m "Subject" -m "Body" src/a.py
```

The defaults are neutral: the main branch is `main`, the remote `origin`, and
**nothing pushes unless asked** (`--push` on the command, or a host that opts in
with `PoolHost.push_by_default`). The fields a project sets — its submodules,
shared directories, install step, branch names — are `worktree.PoolHost` and
`commit.CommitHost`; the kit's [README](src/llm_agents/git/README.md) lists the
other modules.

## Host rules — `examples/host_rules.py`

The kit and the bridges know no project. What one adds — the file never
committed, the submodules a slot gets, the paths no agent writes, the commit
command a worker is told, the gate hook a review reuses — is a `PoolHost`, a
`CommitHost` and a `HostPolicy` (`llm_agents.host_policy`) in one module of
the project's own, `tools/host_rules.py`, which puts the vendored `src/` on
`sys.path` itself. Each entry point beside it is then two lines:

```python
# tools/worktree.py
from host_rules import worktree_main      # your copy of examples/host_rules.py

raise SystemExit(worktree_main())
```

[`examples/host_rules.py`](examples/host_rules.py) is a neutral example with
every field a downstream project usually sets, the review profile included,
and the four entry points as functions; `tests/test_examples_host_rules.py`
copies it into that layout and runs each entry point there.

## Test support for a host — `llm_agents.codex.testing`

A host's judge tables deserve pins of their own, against a tree of their own;
what a host should not rewrite is how a judge is asked, how a fixture
repository and a pool slot are laid out, and the stand-in `codex app-server`
the core's own tests run against. The package exports exactly that:
`worker_judge`, `pool_slot`, `commit_judge` and the rest of
[`__init__.py`](src/llm_agents/codex/testing/__init__.py); `wire` is the
offline app-server (a real child process on real pipes, no network, no
`codex`); `fixtures` the pytest fixtures a turn test registers by importing
them. The core's own test host, `tests/codex_agent/core_policy.py`, is a small
`HostPolicy` built the way a project's is.

## ask-user-gate — a Claude Code plugin

A `PreToolUse` hook that refuses the shell commands which would park the
session on a permission dialog — a `cd … && …` chain, a backgrounded `&`, a
heredoc, `sed -i`, a `sleep` loop, a command over the analyser's length limit —
and names the replacement for each. It ships `replace_in_file.py` (a checked
`sed -i`) and `try_patch.py` (mutate, run, restore from a `finally`; with
`--expect-fail` the way to prove a test can fail), plus a C++ twin for machines
that call the hook a hundred thousand times a month.

```text
/plugin marketplace add neurocod/llm-loop
/plugin install ask-user-gate@neurocod
```

```bash
python claude-plugin/ask-user-gate/hooks/ask_user_gate.py --self-test
python claude-plugin/ask-user-gate/bin/try_patch.py --selftest
```

Why each shape is refused, how to update an installed copy, and the C++ port:
[`claude-plugin/ask-user-gate/README.md`](claude-plugin/ask-user-gate/README.md).

## agent-tools

Standalone scripts (no import of the packages) for the one-liners an agent
harness keeps stopping on: wait for a file to appear or vanish, follow a growing
log until a pattern shows, move line ranges between files byte-exactly, read the
machine's CPU load, search a whole repository for a name after a rename.

```bash
python agent-tools/follow_file.py --help
python agent-tools/rename_gate.py old_name     # exit 1 while any hit is left
```

[`agent-tools/README.md`](agent-tools/README.md) has the table.

## agent-counter — optional, Windows, Qt

An on-screen badge with the number of running agents and the provider's
weekly-limit remainder. Not part of the pip package (Qt is no dependency of
this project), never built on another OS, and absent it the bridges show
nothing and never fail a turn. Build and wiring:
[`extras/agent-counter/README.md`](extras/agent-counter/README.md).

## Requirements and CI

Python 3.9 or newer and no required third-party package: every import across
`src/` is from the standard library, except `rich`, which the loop uses for live
Markdown rendering when it is installed and does without when it is not
(`pyproject.toml` says why the floor is 3.9). The loop and the bridges need the
CLI they drive — `claude` or `codex` — installed and signed in;
the tests need neither (the Codex tests run against the stand-in wire, the Claude
tests against patched launches).

CI (`.github/workflows/tests.yml`) runs the suite and the gate's self-tests on
Linux, Windows and macOS for Python 3.9 through 3.13. The same matrix on your
machine, one uv venv per Python:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
python scripts/ci_local.py 3.9 3.13     # a subset; no argument runs them all
```

## Layout

```
src/llm_loop/        the loop engine                        — src/llm_loop/README.md
src/llm_agents/      the bridges and what they share:
  codex/             Codex CLI as a named worker              — codex/README.md
  codex/testing/     the fake app-server wire, for a host's tests
  claude/            Claude Code as a named worker            — claude/README.md
  git/               the worktree pool and the commit helper  — git/README.md
  host_policy.py     what a project adds, handed over explicitly
  answer_contract.py the `## Coverage` contract and its follow-up
  child_job.py       a Windows Job Object that ends a child's tree with its owner
  agent_counter.py   the client of the optional counter
  gitlib.py          the git runner the kit and the bridges share
examples/            the loop wrappers, the playbook, host_rules.py
claude-plugin/       ask-user-gate: the hook, its bin/ tools, the C++ port
agent-tools/         standalone scripts for agents
extras/              optional native add-ons outside the pip package
scripts/             ci_local.py; fake_claude/; collect_codex_thread.py
ideas/               study material, never shipped
```

`src/` holds the two packages and nothing else, so a host that puts it on
`sys.path` inherits exactly two names (`tests/test_vendoring_footprint.py`).

## Install

Vendoring is the intended route — it pins the engine to a commit your project
controls, and a wrapper needs no install step:

```bash
git submodule add https://github.com/neurocod/llm-loop.git tools/llm-loop
```

It is also a normal PEP 621 project: `pip install
git+https://github.com/neurocod/llm-loop.git`. Note that `pip install llm-loop`
does **not** get you this project — that name on PyPI belongs to an unrelated
2023 package. MIT license.
