"""The command line and everything it prints.

The caller of this script is usually a model, so the rendering rules here are
context-budget rules: one summary line, then the answer, and `logs` boiling a
JSON-RPC stream down to one line per action.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from llm_loop.wire import (
    codex_app_event_thread, codex_app_from_sub_agent, codex_app_learn_main_thread,
)
from .. import host_policy
from ..answer_contract import followed_up_note, incomplete_note

from .events import (
    END_META, FOLLOWUP_GATES, FOLLOWUP_META, GATE_META, REPORT_META, RETRY_META,
    SUB_AGENTS_CUT, SUB_AGENTS_FOLLOW_UP, SUB_AGENTS_META, SUB_AGENTS_WAITING,
    TURN_META, NETWORK_WAIT_META,
    TurnTrace, completed_item, edit_decision, follow_up_kind, item_work, text_of,
)
from .gates import GatesError, required_gates, summary as gate_summary
from .protocol import APPROVAL_META, N_TURN_COMPLETED
from .paths import agent_name, entered, invocation_root, scratch_dir
from .registry import AgentRecord, Registry, RegistryError
from .store import AgentStore, free_archive_stamp
from .roles import ROLES, expected_answer
from .sandbox import (
    SANDBOX_MODES, SCRATCH_WRITE, check_online_probes, check_role_sandbox,
    check_writable_roots, scratch_root,
    report_path, workspace_of,
)
from .service import (
    NETWORK_WAIT_GAP_S, network_wait_note, reconnecting_note, retried_note,
)
from .subagents import (
    DONE_TURN, RESTART_TURN, SUB_AGENT_GRACE_S, SUB_AGENT_INTERRUPT_S,
)
from .turn import (
    DEFAULT_TURN_TIMEOUT_S, MAX_SERVICE_RETRIES,
    TURN_STARTUP_GRACE_S, TurnOutcome, run_turn,
)


# --- output ---------------------------------------------------------------


def _thousands(value: int) -> str:
    return f"{value / 1000:.1f}k" if value >= 1000 else str(value)


# Above this share of the model's window, another message on the same thread is
# a bad deal: resuming re-sends the whole thread, so the next turn starts from
# here and grows. Measured doubling on trivial follow-ups (19.7k -> 41.4k,
# 39.6k -> 82.8k input), against a 258.4k window on gpt-5.6-sol — roughly a
# quarter of what this script's own caller has. Past the mark, a fresh `spawn`
# costs less than a `send`, and the run says so rather than leaving the caller
# to discover it as a truncated thread.
CONTEXT_PRESSURE_SHARE = 0.6

# `status` without `--all` lists only agents touched this recently, in days:
# a review thread older than this is one nobody resumes (see `cmd_status`).
STATUS_RECENT_DAYS = 3


def _context_share(outcome: TurnOutcome) -> str:
    """The thread's occupancy of the model window, as the summary shows it."""
    if not outcome.context_window:
        return _thousands(outcome.context_tokens)
    share = outcome.context_tokens / outcome.context_window
    return (f"{_thousands(outcome.context_tokens)}/"
            f"{_thousands(outcome.context_window)} ({share:.0%})")


def _context_pressure(outcome: TurnOutcome) -> str:
    if not outcome.context_window or not outcome.context_tokens:
        return ""
    share = outcome.context_tokens / outcome.context_window
    if share < CONTEXT_PRESSURE_SHARE:
        return ""
    return (f"thread fills {share:.0%} of the model's window — give the next "
            f"task to a new `spawn`, not `send` to this one")


def print_outcome(name: str, outcome: TurnOutcome, *, quiet: bool) -> int:
    """Print the compact summary plus the stored answer; return an exit code.

    The summary is one line on purpose: the caller of this script is usually a
    model, and every line it prints is context spent on something other than the
    answer.
    """
    out = sys.stdout
    # A red wrapper gate fails the call: the answer is the worker's claim,
    # the gates are what the wrapper saw (`gates`).
    ok = (outcome.status == "completed" and not outcome.error
          and all(result.passed for result in outcome.gates))
    if not quiet:
        parts = [f"{outcome.duration_s:.0f}s"]
        if outcome.report_source:
            parts.append(outcome.report_source)
        if outcome.report_note:
            parts.append(outcome.report_note)
        if outcome.commands:
            parts.append(f"{outcome.commands} cmd")
        if outcome.file_changes:
            parts.append(f"{outcome.file_changes} edits")
        if outcome.network_wait_s:
            parts.append(network_wait_note(outcome.network_wait_s))
        if outcome.service_retries:
            # Named in the summary because it explains both halves of what the
            # caller sees: minutes of wall-clock that bought nothing, and a
            # turn whose beginning was spoken by an attempt the caller never
            # saw — the same agent on the same thread, earlier.
            parts.append(retried_note(outcome.service_retries))
        if outcome.follow_ups:
            # Explain the extra turn, whether its answer replaces or extends.
            parts.append(followed_up_note(outcome.expect, outcome.follow_ups))
        if outcome.gates:
            parts.append("gates " + gate_summary(outcome.gates,
                                                 outcome.gate_follow_ups))
        if outcome.gate_note:
            parts.append(outcome.gate_note)
        if outcome.declined_edits:
            parts.append(f"{len(outcome.declined_edits)} edit(s) declined by "
                         f"the wrapper")
        subs = outcome.sub_agents
        if subs.late:
            asked = {DONE_TURN: ", then gave their results to the main thread",
                     RESTART_TURN: ", then asked the main thread to restart "
                                   "the cut ones"}.get(subs.turn, "")
            parts.append(f"waited {subs.wait_s:.0f}s for "
                         f"{len(subs.late)} sub-agent(s){asked}")
        if outcome.input_tokens or outcome.output_tokens:
            parts.append(f"{_thousands(outcome.input_tokens)} in / "
                         f"{_thousands(outcome.output_tokens)} out")
        if outcome.context_tokens:
            parts.append(f"ctx {_context_share(outcome)}")
        status = outcome.status or ("completed" if ok else "unknown")
        out.write(f"[codex:{name}] {status} · {' · '.join(parts)}\n")
        out.write(f"[codex:{name}] thread {outcome.thread_id or '?'} · "
                  f"log {outcome.log_path}\n\n")
        crowded = _context_pressure(outcome)
        if crowded:
            out.write(f"[codex:{name}] {crowded}\n\n")
    if outcome.text:
        out.write(outcome.text.rstrip() + "\n")
    elif outcome.sub_agents.block:
        # Not stored as the answer (`SubAgentAccount.under_answer`), but the
        # caller is owed it all the same.
        out.write(outcome.sub_agents.block)
    if outcome.expect_met is False:
        # Printed even with --quiet, like the error, but unlike the error it
        # leaves the exit code alone: the summary's "the first answer lacked"
        # says why a follow-up was sent, not that it closed the gap.
        out.write(f"\n[codex:{name}] "
                  f"{incomplete_note(outcome.expect, outcome.follow_ups)}\n")
    for entry in outcome.declined_edits:
        # Even with --quiet, exit code alone: the agent heard "rejected by
        # user", so its report may say a person refused.
        out.write(f"\n[codex:{name}] {edit_decision(entry)}\n")
    cut = outcome.sub_agents.cut
    if cut:
        # Even with --quiet, and the exit code left alone like the line
        # above: the answer is the main thread's and stands, but a worker's
        # tree may hold edits it does not describe (the block under the
        # answer names the files).
        redone = all(fate.told for fate in cut) and not outcome.sub_agents.note
        out.write(f"\n[codex:{name}] warning: {len(cut)} sub-agent(s) cut while "
                  f"still running: "
                  + ", ".join(f"{fate.path} ({len(fate.files)} file(s))"
                              for fate in cut)
                  + (" — the main thread was asked to restart them or finish "
                     "their work\n" if redone else
                     " — check the tree for edits the answer does not "
                     "describe\n"))
    if outcome.error:
        out.write(f"\n[codex:{name}] error: {outcome.error}\n")
    out.flush()
    return 0 if ok else 1


# --- commands -------------------------------------------------------------


def _read_prompt(args: argparse.Namespace) -> str:
    if getattr(args, "prompt_file", None):
        return Path(args.prompt_file).read_text(encoding="utf-8")
    prompt = " ".join(args.prompt) if isinstance(args.prompt, list) else args.prompt
    if prompt == "-":
        return sys.stdin.read()
    if not prompt:
        raise SystemExit("empty prompt: pass text, `-` for stdin, or --prompt-file")
    return prompt


def _archive_scratch(name: str) -> None:
    """Move a `scratch-write` agent's previous workspace aside, keeping it.

    A fresh agent must start in an EMPTY workspace: the name is reusable by
    design, so without this a new `rev` opens in the last `rev`'s working
    directory and inherits its half-written scripts, its `pytest-tmp`, its
    stale config — which is worse than the same problem in a log, because the
    agent will read and run what it finds there.

    Moved rather than deleted, for the same reason `AgentStore.archive` moves:
    what the previous run left behind is often why a new one is being started.
    Only `spawn` calls it — a `send` continues the same agent, and clearing its
    workspace mid-task would delete work it is about to use.
    """
    workspace = scratch_dir(name)
    if not workspace.exists() or not any(workspace.iterdir()):
        return
    def target(stamp: str) -> Path:
        return workspace.with_name(f"{workspace.name}.{stamp}")
    workspace.replace(target(free_archive_stamp(lambda s: [target(s)])))


def _expect(role: str, explicit: Optional[str], prompt: str) -> str:
    """`--expect` for this turn, refused before anything is claimed or sent."""
    try:
        return expected_answer(role, explicit, prompt)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def _remember_turn(registry: Registry, record: AgentRecord,
                   outcome: TurnOutcome) -> None:
    """Record what the turn changed about the agent — nothing more.

    A turn that never started is not a turn: counting it (and stamping a new
    last-turn time) made `status` claim work that produced no answer.
    """
    if outcome.thread_id and outcome.thread_id != record.thread_id:
        record.thread_id = outcome.thread_id
    if outcome.status == "completed":
        record.turns += 1
        record.last_turn = time.strftime("%Y-%m-%d %H:%M:%S")
    registry.put(record)


def cmd_spawn(args: argparse.Namespace) -> int:
    registry = Registry()
    # The prompt is read FIRST because everything below it destroys something:
    # `reserve` overwrites the record and `AgentStore.archive` moves the
    # answer aside. A typo in `--prompt-file` used to cost a working agent —
    # thread id gone, `last.md` archived, `send` refusing it as "no thread
    # recorded" — for a turn that then never ran.
    prompt = _read_prompt(args)
    try:
        gate_commands = required_gates(prompt) if args.gates_from_prompt else []
    except GatesError as exc:
        raise SystemExit(str(exc)) from None
    expect = _expect(args.role, args.expect, prompt)
    project = Path(args.cwd).resolve() if args.cwd else invocation_root()
    if not project.is_dir():
        # Before the name is claimed, and before anything is created: a mistyped
        # --cwd used to reach the turn and have the directory made for it.
        raise SystemExit(f"--cwd {project} is not a directory")
    record = AgentRecord(
        name=args.name,
        cwd=str(project),
        model=args.model or "",
        sandbox=args.sandbox,
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
        escalate=args.escalate or "",
        writable=[str(Path(root).resolve()) for root in args.writable],
        role=args.role,
        gates=gate_commands,
        allow_online_probes=args.allow_online_probes,
    )
    # Same moment as the `--cwd` check, for the same reason: a root that does
    # not exist, or a mode the flag cannot widen, is refused before the name
    # is claimed and anything is archived.
    check_role_sandbox(record)
    check_writable_roots(record)
    check_online_probes(record)
    # Claim the name BEFORE the turn, and store the thread id as soon as the
    # server hands it over: both windows used to stay open for the whole turn,
    # which is minutes (see Registry.reserve and run_turn's `on_thread`).
    displaced = registry.reserve(record, replace=not args.keep)
    if displaced:
        AgentStore(args.name).archive()
        if not args.quiet:
            sys.stdout.write(
                f"[codex:{args.name}] replaced thread {displaced} "
                f"(still listed by `codex agents`)\n")
    if record.sandbox == SCRATCH_WRITE:
        _archive_scratch(args.name)
        if not args.quiet:
            # Said once, because anything this agent leaves on disk lands there
            # and nowhere else — and unlike the log, that path is not in the
            # summary.
            sys.stdout.write(f"[codex:{args.name}] scratch workspace "
                             f"{workspace_of(record)}\n")
    elif scratch_root(record) is not None:
        # The same directory as a SECOND root beside the project
        # (`sandbox.scratch_root`); archived on a reused name for the same reason
        # a scratch workspace is — the previous agent's baselines are not this
        # one's, and it would read them as its own.
        _archive_scratch(args.name)
        if not args.quiet:
            sys.stdout.write(f"[codex:{args.name}] scratch directory "
                             f"{scratch_root(record)} (writable, outside the "
                             f"project; the tree is the other root)\n")
    if record.writable and not args.quiet:
        sys.stdout.write(f"[codex:{args.name}] also writable: "
                         f"{', '.join(record.writable)}\n")
    report = report_path(record)
    if report is not None and not args.quiet:
        sys.stdout.write(f"[codex:{args.name}] report {report}\n")

    def remember_thread(thread_id: str) -> None:
        """Store the id the moment the server hands it over, not at the end.

        A named function rather than a lambda because the storing is two steps,
        and the helper that used to squeeze them into one expression was named
        for the thread REPLACEMENT `spawn` performs a few lines above
        (`reserve(replace=True)` plus the archive) — which this has nothing to
        do with, and which cost a reader the trace either way.
        """
        record.thread_id = thread_id
        registry.put(record)

    outcome = run_turn(record=record, prompt=prompt, resume=False,
                       timeout=args.timeout, effort=args.effort,
                       expect=expect, on_thread=remember_thread)
    _remember_turn(registry, record, outcome)
    return print_outcome(args.name, outcome, quiet=args.quiet)


def cmd_send(args: argparse.Namespace) -> int:
    registry = Registry()
    record = registry.get(args.name)
    if record is None:
        raise SystemExit(f"no agent named {args.name!r}; `spawn` one first "
                         f"(see `status` for the list)")
    if not record.thread_id:
        # Without this, `send` quietly fell through to `thread/start`: a brand
        # new thread with none of the agent's history, answering as if it had
        # context, while the empty id stayed empty and the next message did it
        # again.
        raise SystemExit(
            f"agent {args.name!r} has no thread recorded — its first turn never "
            f"reached the server. Use `spawn {args.name} ...` to start it over.")
    prompt = _read_prompt(args)
    expect = _expect(record.role, args.expect, prompt)
    _touch(registry, record)
    report = report_path(record)
    if report is not None and not args.quiet:
        sys.stdout.write(f"[codex:{args.name}] report {report}\n")
    outcome = run_turn(record=record, prompt=prompt, resume=True,
                       timeout=args.timeout, effort=args.effort, expect=expect)
    _remember_turn(registry, record, outcome)
    return print_outcome(args.name, outcome, quiet=args.quiet)


def cmd_result(args: argparse.Namespace) -> int:
    answer = AgentStore(args.name).read_answer()
    if answer is None:
        raise SystemExit(f"no recorded answer for {args.name!r}")
    sys.stdout.write(answer)
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    """Show what an agent DID, one line per action.

    The raw event log is JSON-RPC traffic, tens of thousands of lines for a long
    turn — unreadable for the caller and ruinous to paste anywhere. This renders
    the few item kinds that say what happened: commands run, files touched,
    messages produced.
    """
    store = AgentStore(args.name)
    if not store.has_events():
        raise SystemExit(f"no log for {args.name!r}")
    if args.child:
        return _child_messages(store, args.child)
    lines: list[str] = []
    # A sub-agent's lines are prefixed with its `agentPath`: unmarked, its
    # `final_answer` read as the main thread's in the middle of the turn.
    main = ""
    sub_paths: dict[str, str] = {}  # thread -> agentPath, from spawn records
    for message in store.events():
        meta = message.get("_meta")
        if meta == TURN_META:
            lines.append(f"--- turn: {str(message.get('prompt', ''))[:100]}")
            main = ""
            continue
        if meta == END_META:
            lines.append(f"--- end: {message.get('status', '?')}")
            continue
        if meta == NETWORK_WAIT_META:
            note = (reconnecting_note(str(message.get("since") or ""),
                                      int(message.get("tries") or 0))
                    if message.get("active") else
                    network_wait_note(float(message.get("waited_s") or 0)))
            lines.append(f"--- {note}")
            continue
        if meta == REPORT_META:
            differs = " (differs from the message)" if message.get("differs") else ""
            lines.append(f"--- report: {message.get('source', '?')}{differs}")
            continue
        if meta == FOLLOWUP_META:
            if follow_up_kind(message) == FOLLOWUP_GATES:
                lines.append(f"--- follow-up: red gates "
                             f"{', '.join(message.get('gates') or ['?'])}")
            else:
                lines.append(f"--- follow-up: the answer lacked "
                             f"/{message.get('expect', '?')}/")
            continue
        if meta == GATE_META:
            if message.get("running"):
                lines.append(f"--- gate: {message.get('command', '?')} "
                             f"(cwd {message.get('cwd', '?')})")
            else:
                lines.append(f"--- gate: {message.get('command', '?')} -> "
                             f"exit {message.get('exit')} "
                             f"({message.get('duration_s', '?')}s)")
            continue
        if meta == SUB_AGENTS_META:
            lines += _sub_agents_lines(message)
            continue
        if meta == APPROVAL_META:
            if message.get("kind") == "fileChange":
                lines.append(f"--- {edit_decision(message)}")
            elif message.get("decision") != "accept":
                # A granted command shows as its own `$` line; a refused one
                # never runs, so this is its only trace.
                reason = (f": {message['reason']}" if message.get("reason")
                          else "")
                command = " ".join(str(message.get("command") or "").split())
                lines.append(f"--- escalation declined by the wrapper{reason} "
                             f"— {command[:160]}")
            continue
        if meta == RETRY_META:
            # Two lines per slot: the wait, then what the check decided. The
            # first is the one an operator reads while it is happening.
            head = (f"--- retry {message.get('attempt', '?')}/"
                    f"{message.get('of', '?')} after a service drop "
                    f"({message.get('reason', '?')}): ")
            if message.get("waiting"):
                lines.append(head + f"waiting {message.get('paused_s', '?')}s")
            else:
                lines.append(head + f"paused {message.get('paused_s', '?')}s, api "
                             + ("reachable" if message.get("reachable") else
                                "unreachable — not resent"))
            continue
        main = codex_app_learn_main_thread(message, main)
        item = completed_item(message)
        if item is None:
            continue
        mark = ""
        if codex_app_from_sub_agent(message, main):
            mark = f"[{sub_paths.get(codex_app_event_thread(message), 'sub-agent')}] "
        kind = item.get("type")
        if kind == "commandExecution":
            code = item.get("exitCode")
            command = " ".join(str(item.get("command") or "").split())
            lines.append(f"{mark}$ {command[:160]}"
                         + (f"  -> exit {code}" if code not in (0, None) else ""))
        elif kind == "fileChange":
            changes = item.get("changes") or []
            # A declined or failed edit wrote nothing, and says so.
            state = ("" if item.get("status") in (None, "completed")
                     else f" [{item.get('status')}]")
            for change in changes:
                kind_name = (change.get("kind") or {}).get("type", "?")
                lines.append(f"{mark}~ {kind_name} {change.get('path')}{state}")
            # `events.item_work` counts a fileChange with no list as one edit,
            # and it is the counter `status` and the summary line both read.
            # Printing nothing for that shape made `logs` show two lines for
            # the turn `status` called "3 edits" (review 2026-09-13).
            for _ in range(item_work(item)[1] - len(changes)):
                lines.append(f"{mark}~ edit (the server named no file)")
        elif kind == "mcpToolCall":
            lines.append(f"{mark}@ {item.get('server')}/{item.get('tool')}")
        elif kind == "subAgentActivity" and item.get("kind") in (
                "started", "completed", "interrupted"):
            # A parent's record of its child, on whichever thread the parent
            # is: a child spawning its own records it on the child's thread,
            # and labels learnt from the main thread's alone printed the
            # grandchild as "[sub-agent]". The path is the child's, so the
            # line needs no mark. Left out: "interacted", a message between
            # parent and child that either side records — a child's record
            # names its PARENT (`/root` for the main thread).
            path = str(item.get("agentPath") or "?")
            if item.get("kind") == "started":
                sub_paths[str(item.get("agentThreadId") or "")] = path
            lines.append(f"--- sub-agent {path} {item.get('kind')}")
        elif kind == "agentMessage":
            # The agent's own progress notes are the one line kind whose
            # tail carries the point ("...checking which state it reads"),
            # so they get more room than a command line.
            first = (text_of(item).strip().splitlines() or [""])[0]
            lines.append(f"{mark}= {first[:400]}")
    tail = lines[-args.tail:] if args.tail > 0 else lines
    sys.stdout.write("\n".join(tail) + "\n")
    return 0


def _child_messages(store: AgentStore, path: str) -> int:
    """`logs --child PATH`: every message of the sub-agent at `path`, WHOLE
    — where the capped quotes of `subagents` send their reader
    (`subagents.full_text_command`); the line-per-action view keeps only a
    message's first line. A message the server carried only inside the
    turn's verdict (`turn/completed` items) is printed too, once."""
    threads: set[str] = set()
    printed: set[tuple[str, str]] = set()
    out: list[str] = []
    for message in store.events():
        item = completed_item(message)
        if item is not None and item.get("type") == "subAgentActivity" and (
                item.get("kind") == "started" and item.get("agentPath") == path):
            threads.add(str(item.get("agentThreadId") or ""))
            continue
        thread = codex_app_event_thread(message)
        if thread not in threads:
            continue
        params = message.get("params") or {}
        texts = []
        if item is not None and item.get("type") == "agentMessage":
            texts.append(text_of(item))
        elif message.get("method") == N_TURN_COMPLETED:
            texts += [text_of(done) for done in (params.get("turn") or {}).get(
                "items") or [] if isinstance(done, dict)
                and done.get("type") == "agentMessage"]
        for text in texts:
            if text and (thread, text) not in printed:
                printed.add((thread, text))
                out.append(f"--- message of {path}:\n{text.rstrip()}\n")
    if not out:
        raise SystemExit(f"no messages of {path!r} in the log of {store.name!r}")
    sys.stdout.write("\n".join(out))
    return 0


def _sub_agents_lines(message: dict) -> list[str]:
    """`logs` lines for one `SUB_AGENTS_META` line
    (`subagent_wait.SubAgentWait`)."""
    phase = message.get("phase")
    if phase == SUB_AGENTS_WAITING:
        return [f"--- sub-agents still running after the main answer: "
                f"{', '.join(message.get('open') or ['?'])} (waiting up to "
                f"{message.get('grace_s', '?')}s)"]
    if phase == SUB_AGENTS_CUT:
        return [f"--- sub-agents interrupted, still running at the deadline: "
                f"{', '.join(message.get('cut') or ['?'])}"]
    if phase == SUB_AGENTS_FOLLOW_UP:
        finished = ", ".join(message.get("finished") or []) or "none"
        if message.get("kind") == RESTART_TURN:
            return [f"--- asking the main thread to restart or finish the cut "
                    f"sub-agents: {', '.join(message.get('cut') or ['?'])} "
                    f"(finished: {finished})"]
        return [f"--- sub-agents finished: {finished}; their results sent to "
                f"the main thread"]
    labels = {("finished", True): "finished after the main answer; result "
                                  "passed to the main thread",
              ("finished", False): "finished after the main answer; UNREAD by "
                                   "the main thread (quoted under the answer)",
              ("cut", True): "CUT: interrupted; the main thread was asked to "
                             "restart it",
              ("cut", False): "CUT: interrupted, still running"}
    lines = []
    for state in ("finished", "cut"):
        for child in message.get(state) or []:
            files = child.get("files") or []
            restarted = " (restarted after a cut)" if child.get("restarted") else ""
            verdict = (f" [{child['status']}]" if child.get("status") not in
                       (None, "", "completed") and state == "finished" else "")
            lines.append(f"--- sub-agent {child.get('path', '?')}{restarted} "
                         f"{labels[state, bool(child.get('told'))]}{verdict}"
                         + (f"; touched {', '.join(files)}" if files else ""))
    if message.get("note"):
        lines.append(f"--- {message['note']}")
    return lines


def _touched_at(record: AgentRecord) -> Optional[float]:
    """When a turn was last started on the agent, as epoch seconds — None when
    the record carries no usable date (written before the fields existed).

    The LATEST stamp decides, and only it: a latest stamp in a format this
    cannot read means "unknown", not "fall through to the creation date and
    call the agent old" (review 2026-09-12).
    """
    stamp = record.touched or record.last_turn or record.created
    try:
        return time.mktime(time.strptime(stamp, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return None


def _turn_in_progress(store: AgentStore, trace: Optional[TurnTrace]) -> str:
    """The `status` suffix for an agent whose last turn has not ended.

    Three workers running their first turn showed as `0 turns · last -`, the
    same line as an agent that never started (2026-09-13) — the registry only
    learns of a turn when it completes. The log knows earlier: its opening
    line is written before the server starts and its closing line on every
    exit, so a turn with the first and not the second is in progress — unless
    the process that ran it was killed (TaskStop, a reboot), which writes
    nothing. That case is told apart by AGE: the log is appended on every
    event, so one silent for longer than the turn's own timeout belongs to a
    process that is gone, since a live one would have timed out and closed it
    — plus `turn.TURN_STARTUP_GRACE_S`, the silence a turn is entitled to
    before that timeout is even armed — and the sub-agent grace it announced
    (`subagents.SUB_AGENT_GRACE_S`), plus the heartbeat allowance of an
    announced network wait (`service.NETWORK_WAIT_GAP_S`).
    """
    if trace is None or trace.ended:
        return ""
    work = f"{trace.commands} cmd, {trace.file_changes} edits"
    if trace.waiting_sub_agents:
        # "Up to": the waiting line counts the children open at the main
        # answer; some may have finished since.
        work += (f", waiting for up to {trace.waiting_sub_agents} "
                 f"sub-agent(s)")
    if trace.retries:
        work += f", {retried_note(trace.retries)}"
    if trace.reconnecting_since:
        work += f", {reconnecting_note(trace.reconnecting_since, trace.reconnecting_tries)}"
    if trace.follow_ups:
        work += f", {trace.follow_ups} follow-up"
    if trace.gate_follow_ups:
        work += f", {trace.gate_follow_ups} gate follow-up"
    silent_s = store.events_silent_s()
    network_grace = NETWORK_WAIT_GAP_S if trace.reconnecting_since else 0.0
    if silent_s > ((trace.timeout_s or DEFAULT_TURN_TIMEOUT_S)
                   + TURN_STARTUP_GRACE_S + trace.sub_agent_grace_s + network_grace):
        return (f" · turn started {trace.started} never ended ({work}; the "
                f"process was killed?)")
    return f" · RUNNING since {trace.started} ({work})"


def _touch(registry: Registry, record: AgentRecord) -> None:
    """A turn is about to start on this agent: say so before it runs, so a
    turn that fails or times out still counts as recent activity."""
    record.touched = time.strftime("%Y-%m-%d %H:%M:%S")
    registry.put(record)


def cmd_status(args: argparse.Namespace) -> int:
    """List agents — by default only the ones touched in the last `--days`.

    The registry is append-only in practice (names are reused, never dropped),
    so after two weeks of reviews a bare `status` printed 116 lines, all but
    two of them agents whose threads nobody would ever `send` to again
    (measured 2026-09-12). The caller is usually a model, and those lines are
    context spent on nothing; the hidden count and `--all` keep the full list
    one flag away. A record with no date is shown: unknown age is not old age.
    """
    agents = Registry().load()
    if not agents:
        sys.stdout.write("no codex agents\n")
        return 0
    names: Iterable[str] = [args.name] if args.name else sorted(agents)
    # `--days 0` would hide every dated agent (nothing is newer than now), so
    # zero means "no window" — the same as `--all`.
    cutoff = (None if args.name or args.all or args.days <= 0
              else time.time() - args.days * 86400)
    hidden = 0
    idle = 0
    for name in names:
        record = agents.get(name)
        if record is None:
            sys.stdout.write(f"{name}: unknown\n")
            continue
        touched = _touched_at(record)
        if cutoff is not None and touched is not None and touched < cutoff:
            hidden += 1
            continue
        store = AgentStore(name)
        trace = store.last_turn()
        progress = _turn_in_progress(store, trace)
        if args.running and not progress:
            # An orchestrator polling three workers read 25 lines per poll, 22
            # of them yesterday's reviewers inside the recent window (measured
            # 2026-09-13); the window is about what may be resumed, this flag
            # is about what is happening now.
            idle += 1
            continue
        # The sandbox shown is the one the agent will actually get, from the
        # record itself rather than from the request the turn builds: reading it
        # back out of `thread_params(record)["sandbox"]` was right but made a
        # protocol key rename break the status line.
        widened = (f" +{len(record.writable)} writable" if record.writable
                   else "")
        role = f" · role {record.role}" if record.role else ""
        if record.gates:
            role += f" · {len(record.gates)} gates"
        if record.allow_online_probes:
            role += " · " + host_policy.active().commands.reserved.online_label
        if trace is not None and trace.cut_sub_agents:
            # Kept apart from `progress`, which `--running` filters on: an
            # ended turn is not running — but it is named on every line until
            # the next turn, since the tree may hold the cut children's edits.
            progress += (f" · last turn cut {len(trace.cut_sub_agents)} "
                         f"sub-agent(s): {', '.join(trace.cut_sub_agents)}")
        report = report_path(record)
        report_token = f" · report={report}" if report is not None else ""
        sys.stdout.write(
            f"{name}: thread {record.thread_id or '?'} · {record.turns} turns · "
            f"{record.effective_sandbox}{widened}{role} · last {record.last_turn or '-'}"
            f"{progress}{report_token}\n")
    if idle:
        sys.stdout.write(f"({idle} idle agent(s) not listed; `status` shows "
                         f"them)\n")
    if hidden:
        sys.stdout.write(f"({hidden} agent(s) idle for over {args.days:g} day(s) "
                         f"not listed; `status --all` shows them)\n")
    return 0


def cmd_drop(args: argparse.Namespace) -> int:
    if Registry().drop(args.name):
        sys.stdout.write(f"dropped {args.name}\n")
        return 0
    sys.stdout.write(f"no agent named {args.name}\n")
    return 1


def build_parser() -> argparse.ArgumentParser:
    policy = host_policy.active()
    parser = argparse.ArgumentParser(
        prog="codex_agent",
        description="Run Codex CLI as a named worker; print only its answer.")
    subs = parser.add_subparsers(dest="command", required=True)

    def add_turn_flags(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("prompt", nargs="*", help="prompt text, or `-` for stdin")
        sub.add_argument("--prompt-file", help="read the prompt from this file")
        sub.add_argument("--timeout", type=float, default=DEFAULT_TURN_TIMEOUT_S,
                         help="seconds before an ATTEMPT is abandoned; a turn "
                              "the service drops is resent with the same bound, "
                              "so the wall-clock worst case is this times "
                              f"{MAX_SERVICE_RETRIES + 1} plus the pauses; an "
                              "attempt whose sub-agents outlive the answer "
                              "adds up to "
                              f"{SUB_AGENT_GRACE_S:.0f}s of waiting, one more "
                              "of this for the turn sent about them, and "
                              f"2x{SUB_AGENT_INTERRUPT_S:.0f}s of interrupts")
        # User decision (2026-11-07): default to medium; the orchestrator
        # assigns heavy tasks to Opus instead of raising Codex's default.
        # Effort is per turn, not persisted in AgentRecord: send without the
        # flag uses the current default, even after an explicit high turn.
        sub.add_argument("--effort", default="medium",
                         help="reasoning effort override for this turn "
                              "(default: %(default)s)")
        # A reviewer given five questions answers one and stops (three
        # "cover the remaining N" resends in the wave of 2026-09-13, one more
        # the day after); the operator's fix was one `send` naming the gap.
        # This is that `send`, sent by the run when the answer does not
        # match — `turn._follow_up_on_missing`.
        sub.add_argument("--expect", default=None, metavar="REGEX",
                         help="the answer must match this regex (multiline); "
                              "one that does not is followed up once on the "
                              "same thread, e.g. '^## Coverage'; defaults to "
                              "the agent role's last answer heading; "
                              "--expect= (empty) disables the follow-up")
        sub.add_argument("--quiet", action="store_true",
                         help="print only the answer, no summary lines")

    spawn = subs.add_parser("spawn", help="create a named agent and run its first turn")
    spawn.add_argument("name", type=agent_name)
    spawn.add_argument(
        "--cwd",
        help="working root for the agent (default: current Git worktree, or cwd)")
    # Pin the worker independently of the user's Codex CLI configuration.
    spawn.add_argument("--model", default="gpt-6-astra",
                       help="model override (default: %(default)s)")
    spawn.add_argument("--role", choices=sorted(ROLES), default="",
                       help="append worker or reviewer instructions and default "
                            "answer heading (preserved by send)")
    # `workspace-write` keeps an agent's writes inside this checkout, which is
    # what makes running several of them survivable.
    #
    # It is also the setting that breaks first on Windows, and the symptom does
    # not name its cause: the agent silently cannot run ANY command ("command
    # runner failed", no commandExecution item in the stream at all). That is
    # the sandbox failing to provision, and the reason is a path in the repo
    # owned by a Codex sandbox user instead of by you — provisioning must place
    # a deny-ACE on the repo's `.agents` directory, cannot change a DACL it does
    # not own, and reports the whole setup as failed (`helper_unknown_error`).
    # Diagnosed 2026-08-24; fixed by deleting and recreating that directory.
    # `codex doctor` records the LAST provisioning attempt, so a run from any
    # other directory overwrites the failure with "complete" — check it by
    # running `codex sandbox --permission-profile ":workspace" -C <repo> --
    # cmd.exe /c "echo ok"` from inside the repo instead, and read
    # `~/.codex/.sandbox/sandbox.<date>.log` for the path that was refused.
    # Run that check from PowerShell, not Git Bash: MSYS path conversion turns
    # the bare `/c` into a path, so cmd.exe opens an interactive shell and the
    # check reads exactly like a broken sandbox (measured 2026-08-24).
    #
    # `read-only` is the better choice for review work: the agent reads the tree
    # and answers in text, so it cannot collide with edits being made in the
    # same worktree by somebody else. Its price is that NOTHING is writable —
    # not even a temp directory — so a reviewer under it cannot run a test suite
    # and has to take the gate's result on trust. `scratch-write` is that
    # trade-off without the price; see `sandbox.SANDBOX_MODES`.
    spawn.add_argument("--sandbox", default="workspace-write",
                       choices=sorted(SANDBOX_MODES),
                       help="permissions for this agent (default: workspace-write). "
                            "read-only for review beside your own edits; "
                            "scratch-write for a reviewer that must run tests — "
                            "same unwritable project, plus a scratch directory")
    # Reusing a name starts a FRESH agent by default, because that is what the
    # caller almost always wants: Codex's window is about a quarter of this
    # script's caller's, so one task per agent is the working rule, and refusing
    # the name would only push the caller into inventing `review2`, `review3`.
    # Nothing is lost by it — the displaced thread stays in Codex's own history
    # (`codex agents`), the run prints its id, and the previous log and answer
    # are archived beside the new ones rather than appended to or deleted.
    # The sandbox cannot run this repo's npm gates (a Node child with piped
    # stdio is refused — `sandbox._sandbox_facts`). This names the commands that
    # may escape it: the agent asks, and only a match is granted. A whole-turn
    # `danger-full-access` is the alternative, and it drops the sandbox for
    # everything; see `sandbox.escalation_judge`.
    spawn.add_argument("--escalate", default="", metavar="REGEX",
                       help="commands matching this regex may run OUTSIDE the "
                            "sandbox when the agent asks (e.g. "
                            r"'^npm (test|run typecheck)\b'); everything else "
                            "is still declined. A --role worker escalates the "
                            f"command shapes of {policy.commands.shapes_name} "
                            "without it; given, the regex adds to them")
    spawn.add_argument("--allow-online-probes", action="store_true",
                       help=policy.commands.reserved.online_help)
    # One more directory, by name: the protocol's `workspaceWrite` policy takes
    # a list of roots (`sandbox.turn_sandbox_policy`), so "may this agent write
    # to X as well" is a flag rather than a wider mode. Existing directories
    # only, and only for the two modes built on `workspaceWrite`.
    spawn.add_argument("--writable", action="append", default=[], metavar="DIR",
                       help="an existing directory the agent may also write to "
                            "(repeatable; workspace-write and scratch-write "
                            "only)")
    spawn.add_argument("--gates-from-prompt", action="store_true",
                       help="run the commands of the fenced block under the "
                            "prompt's `## Gates` after every turn, outside the "
                            "sandbox (cwd: gates.gate_cwd), print the results "
                            "under the "
                            "answer, send one follow-up on a red gate; the "
                            "same commands may also escalate")
    spawn.add_argument("--keep", action="store_true",
                       help="refuse if this name is taken, instead of starting "
                            "a fresh agent over it")
    add_turn_flags(spawn)
    spawn.set_defaults(func=cmd_spawn)

    send = subs.add_parser("send", help="continue an existing agent's thread")
    send.add_argument("name", type=agent_name)
    add_turn_flags(send)
    send.set_defaults(func=cmd_send)

    result = subs.add_parser("result", help="print an agent's last answer")
    result.add_argument("name", type=agent_name)
    result.set_defaults(func=cmd_result)

    logs = subs.add_parser("logs", help="what the agent did, one line per action")
    logs.add_argument("name", type=agent_name)
    logs.add_argument("--tail", type=int, default=40,
                      help="last N lines (0 for all)")
    logs.add_argument("--child", metavar="AGENT_PATH", default="",
                      help="print every message of the sub-agent at this path "
                           "(e.g. /root/kid) in full, instead of the actions")
    logs.set_defaults(func=cmd_logs)

    status = subs.add_parser("status", help="list agents (or one)")
    status.add_argument("name", nargs="?")
    status.add_argument("--all", action="store_true",
                        help="list every agent, not only the recently active")
    status.add_argument("--days", type=float, default=STATUS_RECENT_DAYS,
                        help=f"how recent 'recently active' is "
                             f"(default {STATUS_RECENT_DAYS}; 0 = no window)")
    status.add_argument("--running", action="store_true",
                        help="only agents with a turn in progress (or one "
                             "whose process looks killed)")
    status.set_defaults(func=cmd_status)

    drop = subs.add_parser("drop", help="forget an agent (its Codex thread stays)")
    drop.add_argument("name", type=agent_name)
    drop.set_defaults(func=cmd_drop)
    return parser


def main(argv: Optional[list[str]] = None, *,
         policy: Optional[host_policy.HostPolicy] = None,
         root: Optional[Path] = None, entry: Optional[Path] = None) -> int:
    """The command line. A host entry point passes its `policy` and `root`
    (its own `__main__.py`), and as `entry` the path `python <entry>` runs
    it by, which the hints naming the bridge print (`paths.bridge_command`;
    default `python -m llm_agents.codex`); all three hold for this call only
    (`host_policy.installed`, `paths.entered`). Without a policy the call
    runs under `EMPTY`, says so on stderr, and honours only what it was
    asked for itself (`explicit_grants`)."""
    if hasattr(sys.stdout, "reconfigure"):
        # Windows consoles default to a legacy code page, and an answer full of
        # box-drawing or non-ASCII text would die on a UnicodeEncodeError after
        # the model work was already paid for.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    if policy is not None and root is None:
        raise ValueError("a host policy comes with the host root it names")
    with host_policy.installed(policy or host_policy.EMPTY, root), entered(entry):
        host_policy.announce_empty("codex_agent", EMPTY_STILL)
        args = build_parser().parse_args(argv)
        host_policy.grant(explicit_grants(args))
        try:
            return args.func(args)
        except RegistryError as exc:
            # A broken registry is the operator's to fix, and the message must
            # name the file rather than arrive as a traceback from a write.
            sys.stderr.write(f"codex_agent: {exc}\n")
            return 2


# What a call without a host still honours (`host_policy.announce_empty`).
EMPTY_STILL = ("no command leaves the sandbox by shape, cleanup or edit retry, "
               "recorded gates do not run and briefings carry no host rules; "
               "honoured when passed in this call: spawn --escalate REGEX, "
               "spawn --gates-from-prompt")


def explicit_grants(args: argparse.Namespace) -> set[str]:
    """What this call's caller asked for itself (`host_policy.EXPLICIT_GRANTS`):
    only `spawn` takes the flags; a `send` replays the record's."""
    if args.command != "spawn":
        return set()
    return ({"escalate"} if args.escalate else set()) | (
        {"gates"} if args.gates_from_prompt else set())
