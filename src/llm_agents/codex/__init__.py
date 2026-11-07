"""Drive Codex CLI as a named worker, the way a subagent is driven.

Run it as `python -m llm_agents.codex <command>`; a project with rules of its
own calls `cli.main(policy=..., root=...)` from an entry point of its own
(`llm_agents.host_policy` says what a host supplies).

Why this exists: `codex -p "..."` prints its whole transcript to stdout, so a
caller that reads that output pays for every intermediate step in its own
context. This wrapper speaks Codex's app-server JSON-RPC protocol instead
(`codex app-server --stdio`, see `codex app-server generate-json-schema`), keeps
the event stream in a log file, and prints ONLY the agent's final message plus a
one-line summary.

NO DAEMON, deliberately. A named agent is a Codex thread id in the registry
below, and Codex itself persists threads on disk; `thread/resume` reopens one in
a fresh app-server process. So every command here is short-lived, nothing has to
be reaped later, and a crash loses at most one turn. The cost is a ~1-2 s
app-server start per message, and that `turn/steer` (inject text into a RUNNING
turn) is out of reach — steering needs the process that owns the turn. If that
is ever wanted, the shared daemon (`codex app-server daemon start` plus
`codex app-server proxy`) is the seam to build it on, not a daemon of our own.

Background work is the caller's job, not this package's: run `spawn`/`send` as a
background shell command and the harness reports its completion, which is the
same shape as waiting on a subagent.

The package seams, one module each:

  * `protocol.py` — the wire: the app-server process, JSON-RPC, approvals;
  * `events.py`   — shared decoding of completed app-server items;
  * `registry.py` — the state: names, thread ids, the lock they share;
  * `store.py`    — the other half of it: each agent's event log and answer;
  * `turn.py`     — turn execution, events, retries, follow-ups and logging;
  * `sandbox.py`  — permissions, workspaces, briefings and escalation;
  * `cli.py`      — the commands and everything they print;
  * `service.py`  — the service on the far end: whose fault a dropped turn was,
                    and whether OpenAI answers at all;
  * `paths.py`    — the leaf the others stand on.

This file deliberately re-exports NOTHING; import from the module that owns the
name (`from llm_agents.codex.turn import run_turn`). A root that re-exports gives
every name two spellings, and the second one is a trap rather than a
convenience: `from x import y` copies the binding, so a test that patches
`llm_agents.codex.run_turn` leaves `cli.run_turn` on the original and passes while
the real code runs — here, spending a real Codex turn to do it. The rule for
patching follows from the same fact: name the module that LOOKS THE NAME UP (the
consumer), not the one that defines it.

`llm_loop.wire` owns the app-server stream's vocabulary this package reads
(which events are the main thread's and which a sub-agent's); it is imported
by name, so whoever runs this package puts the matching `llm_loop` on the path
(one `src/` holds both).
"""
