# `llm_agents.codex` — Codex CLI as a named worker

`python -m llm_agents.codex <command>` runs Codex CLI as a NAMED worker and
prints only the answer you picked: the turn itself (one measured review: 540 s,
12 commands, 1.08M input tokens) stays in `.codex-agents/<name>/events.jsonl`,
and only the findings reach the caller's context. Commands: `spawn` (create an
agent and give it its first task), `send` (continue its thread), `result`,
`logs` (one line per agent action), `status`, `drop`; each has `--help`.

The package: `protocol` — the wire, `registry` — the state, `gates` — gates
read from the prompt, `events` — the log and event shapes, `turn` — one turn,
`subagents` — its sub-agents and their outcome, `subagent_wait` — waiting for
them, `sandbox` — permissions, `commands` — command parsing and the worker's
allowlist, `service` — the service on the far end, `cli` — commands and
output, `paths` — the leaf the others stand on.

The tests need no network: the real `codex` is never started, and a stand-in
app-server answers on the wire (`tests/codex_agent/` of this repository).

## A host's rules

The core knows no project. A project's command tables, protected paths, gate
directory, briefing lines and root arrive as a `llm_agents.host_policy.HostPolicy`
that the project's own entry point hands to `cli.main(policy=..., root=...)`;
the registry `.codex-agents` lives under that root. Run without one, the
bridge says so on stderr and runs with the empty policy: nothing leaves the
sandbox unless that same call asked for it (`--escalate`,
`--gates-from-prompt`).

## A second opinion from the other model

Have Codex review the commits Claude wrote, and Claude (`llm_agents.claude`)
the ones Codex wrote. One commit reviewed by Codex and by a Claude subagent
gave 8 and 11 findings with 4 in common (measured 2026-08-24): half of what
each found was visible to it alone. A piece worth reviewing is worth both.

## Recipe

Start a worker or a reviewer with `spawn --role worker|reviewer`; the prompt
holds only the task: the item text quoted, the gates, and what not to touch.
An answer missing the role's last heading is followed up once, on the same
thread, by the wrapper itself (`turn._follow_up_on_missing`); `--expect=`
turns that off.

A worker's report file is `sandbox.report_path`; the answer is chosen by
`turn._read_report`.

Write the prompt to a file and pass `--prompt-file`; run reviews with
`--sandbox read-only`, so the agent cannot edit the tree you edit in parallel.
Background work is an ordinary background command: when it reports done, read
`result <name>`; while the turn runs, `status` prints
`RUNNING since … (N cmd, M edits)` and `logs <name>` what the agent is doing.
Sub-agents still alive when the main thread answers are waited for up to
`subagents.SUB_AGENT_GRACE_S`, then the main thread gets one more turn about
them (their results, or a request to restart the interrupted ones); the worst
case of one attempt is spelled out at `turn.DEFAULT_TURN_TIMEOUT_S`. The
summary, `status` and `logs` name the interrupted ones — for a worker, those
are edits in the tree its report does not cover.

**What the agent knows without the prompt.** A repository's
`.codex/config.toml` can load its agent instructions as a project doc
(`project_doc_fallback_filenames`), and the sandbox facts arrive through
`developerInstructions` (`sandbox._sandbox_facts`). "Read the instructions
file first" in a prompt makes the agent re-read both by hand (measured
2026-09-13: three workers, ~2 commands and ~40 KB each) — leave it out. What
does belong in the prompt: the whole task text (quoted, not "take it from that
directory": a worker pointed at a directory reads more of it than it needs),
the gates with their commands, and the task's limits.

**A drop on the service side heals itself; do not `send` by hand.**
"stream disconnected before completion: Unable to verify model access" after
five server reconnects is an OpenAI failure (on 2026-09-13 it took down three
workers within ninety seconds); the thread and the edits in the tree are
intact, so the wrapper resends the turn ON THE SAME thread, asking it to carry
on — `turn.MAX_SERVICE_RETRIES` times, with a pause
(`turn.SERVICE_RETRY_PAUSES_S`) and an API reachability check before each
attempt; what counts as a service failure is `service.is_service_failure`.
Waiting for the network: `service.is_network_wait`,
`service.NETWORK_WAIT_GAP_S`, `service.MAX_NETWORK_WAIT_S`,
`service.NetworkWait.deadline`. A retry shows in the turn's summary, in
`status` and in `logs`; if the service never answers, the turn is still
returned as a refusal with the reason in its text. The price: `--timeout` is
the budget of ONE attempt, not of the whole call.

## How to cut a task

Codex returns as its final message the LAST thing it did: everything else
goes into intermediate notes that only `logs` shows. Hence three rules,
measured on one wave of 2026-09-13 (11 pieces, 14 workers and 9 reviewers):

- **A worker gets one result per spawn, not one item.** An item with two
  independent families is two spawns: the first analysed the first family and
  stopped, the second did both but wrote its report as a "read-only audit".
  A shared gate does not make them one piece; one diff does.
- **Reading is a step of the edit, not an item of the report.** "First compare
  by reading and write down the differences" was read by three workers of four
  as a question of its own, answered in the final message with the edits
  undone. The wording that works: "move X to Y; if the comparison shows a
  seam, stop and name it" — stopping is the exception, editing the norm.
- **A reviewer gets one commit and up to 5–6 questions, not one spawn per
  question.** The answer arrives for the last one or two questions; the
  follow-up by the question numbers (`answer_contract.coverage_expect`) brings
  the rest in one more turn (+~100 s), while a spawn per question would pay
  for re-reading the commit (~250k input tokens) each time.

What not to tell a worker: a hash it should "check" — three recomputed a data
hash from a test fixture and reported a phantom mismatch; name the gate (the
suite) instead. And `git mv` — in a linked worktree the index is outside the
writable root; the wrapper says so itself.

## Sandboxes

The price of `read-only`: the agent has NO writable directory at all, the
temp directory included, so `pytest` with a `tmp_path` fixture does not start
— a reviewer runs only the non-writing subset and says so at the end
(measured 2026-08-24). Nothing inside `read-only` fixes that: in the protocol
`SandboxPolicy.readOnly` has no writable root (only `networkAccess`), and the
one mode with a writable root, `workspaceWrite`, writes the ROOT OF THE
WORKING DIRECTORY.

So a reviewer that has to run tests gets **`--sandbox scratch-write`**: the
same `workspaceWrite`, but the working directory is a temporary one
(`%TEMP%/codex_agent/<name>`), not the repository. The tree stays readable and
NOT writable (`echo > <repo>\...` → "Access is denied"); the agent writes only
into its scratch. Its path is the first line `spawn` prints.

One trap of that mode is measured and told to the agent automatically
(through `developerInstructions`): pytest's shared root
`%TEMP%/pytest-of-<user>` belongs to the user OUTSIDE the sandbox, and
`tmp_path` fails on it with `PermissionError [WinError 5]` — every test that
asks for `tmp_path` errors in setup (32 tests of 51 on 2026-08-24). A
`--basetemp` inside the working directory cures it, and the same gate goes
green in the sandbox — provided the repository has a root `conftest.py` that
undoes Python ≥ 3.13's owner-only mode for `os.mkdir(path, 0o700)` under
Codex (without it pytest's directory is unreadable to its own creator). The
agent that checked this ran under `scratch-write` itself; it also confirmed
that writes into the checkout fail, reads work, and a write through a
junction it had made to the project fails too.

**One more directory: `spawn --writable DIR`** (repeatable): an existing
directory the agent under `workspace-write`/`scratch-write` may write besides
its own roots; the reasoning and the refusals are at
`sandbox.check_writable_roots`.

**The mode is chosen ONCE — in `spawn` — and `send` does not change it** (no
flag on `send`, on purpose: `thread_params` resends the same registry record on
resume, so `status` and the running agent cannot disagree about its rights).
So "more rights" for an existing agent means a new `spawn`; a taken name
starts a fresh agent by default.

**An editing agent needs no flag: `workspace-write` is the default.** The care
is not with the flag but with the TREE: it edits exactly the root of its
working directory, so either hand it the whole tree (and do not edit it
meanwhile) or give it a directory of ITS own with `--cwd`. Without the flag
the worktree of the process's CURRENT directory is used, not the checkout the
script lives in. Its second writable place is its scratch
(`sandbox.scratch_root`, the same `paths.scratch_dir` as `scratch-write`): it
is told to keep baselines and measuring scripts there, and `spawn` prints it
first. Without it workers put those into the tree (measured 2026-09-13). A
linked worktree's admin directory sits in the main checkout's `.git`, outside
the working root, so `git add`/`git commit`/`git mv` from the sandbox fail; a
worker in a pool slot commits by escalating the host's commit command (judge:
`command_commit._commit`, briefing: `sandbox._worktree_fact`) and moves files
with `Move-Item`.

## One task, one agent

The model and reasoning-effort defaults are `cli.build_parser`'s; the model a
turn actually ran on is in its `events.jsonl`. The window is 258 400 tokens,
and one review of a medium commit already takes a third of it; `send` re-reads
the whole thread (measured: 19.7k → 41.4k input on a trivial second message).
So `send` is for follow-ups of the same task, and the next task is a NEW
`spawn`. No need to invent names: `spawn` with a taken name starts a fresh
agent by default (the old thread is not lost — the first line prints it, and
its `events.jsonl`/`last.md` move to timestamped files beside the new ones).
`--keep` restores the old behaviour, a refusal. The turn summary shows
`ctx 89k/258k (34%)` — the window's OCCUPANCY (`last`), not the spend
(`1082k in`, the sum over the turn's requests); from 60 % on the summary
itself says to start a new agent.

## The Windows sandbox

**Gates do not start inside the sandbox, and a flag fixes that.** The
sandbox's restricted token can CREATE a named pipe but not OPEN its client end
(Win32 error 5), which is how libuv connects a child's stdio: every Node child
with a pipe dies with `spawn EPERM`, so `npm run typecheck` and `npm test`
(Vite runs `exec` while loading its config) do not start inside at all.
Measured 2026-09-13 by two probes (`codex sandbox` and an agent under
`scratch-write`); `windows.sandbox = "elevated"` is no way out — it raises a
UAC prompt nobody is there to confirm. Every agent is told this
(`sandbox._sandbox_facts`), so it does not burn its turn on retries.

The way out is escalation: the agent runs a command OUTSIDE the sandbox on
request (`approvalPolicy: on-request`, judged by `sandbox.escalation_judge`).
A **worker** (`--role worker`, `workspace-write`) needs no flag: the commands
of its host's allowlist (`WorkerCommands.shapes`, judged by
`commands.worker_command_refusal`) are granted, and what the host reserves for
online tasks only with `spawn --allow-online-probes`. `--escalate '<regex>'`
ADDS to the list for a worker; for everyone else (a reviewer, an agent without
a role) it is still the only grant (`sandbox.escalation_pattern`). Every
decision is written to `events.jsonl` with its reason, and a refusal shows in
`logs`. An edit (apply_patch) that failed in the sandbox is retried outside it
by `sandbox.file_change_judge`; a refusal is named by `logs` and the turn
summary. `--gates-from-prompt` (`gates.py`) still runs the gates in the
wrapper after the turn. With no role and no flags the policy is `never`.
Deleting without a regex is granted to any agent only for
`sandbox.CLEANUP_DIRS` (`sandbox.cleanup_targets`).

`commands.py` is the public facade and the one text of the trust model. The
judge's map: `command_shell.py` — shell and words; `command_paths.py` and
`command_arguments.py` — paths, arguments and context; `command_node.py` —
npm/node/npx; `command_python.py` — Python; `command_commit.py` — the commit
script; `command_dispatch.py` — the dispatcher.

`workspace-write` breaks when a directory in the repository is owned by a
`CodexSandbox*` user (the symptom: the agent silently cannot run ANY command —
"command runner failed" and no `commandExecution` in the stream). The
diagnosis is in `~/.codex/.sandbox/sandbox.<date>.log`, not in `codex doctor`,
which shows the LAST provisioning from anywhere and so is usually green.
Recreating that directory as yourself cures it.
