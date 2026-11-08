"""Read an agent's event log, and decode what its lines mean.

Two jobs that belong together: the shapes the app-server sends (`completed_item`,
`text_of`, `item_work`) and the reading of the `events.jsonl` those shapes were
appended to (`iter_log`, `last_turn_trace`) — one module so that the writer
(`turn.run_turn`) has a single counterpart, and the readers (`cli.cmd_logs`,
`cli.cmd_status`) share one answer about what a line says.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llm_loop.wire import codex_app_from_sub_agent, codex_app_learn_main_thread

from .protocol import N_ITEM_COMPLETED, N_TURN_COMPLETED

# The bookkeeping lines `turn.run_turn` writes into `events.jsonl` around the
# server's own traffic, keyed by `_meta`. `TURN_META` opens a turn (prompt,
# time, timeout); `END_META` closes it with the outcome's status; `RETRY_META`
# stands between two attempts of ONE turn that the service dropped, and records
# what was waited and what the reachability check said. Their names live here
# because both the writer (`turn`) and the readers (`last_turn_trace`,
# `cli.cmd_logs`) spell them, and a turn whose end is spelled differently reads
# as one that never finished.
TURN_META = "turn"
END_META = "end"
RETRY_META = "retry"
# Reconnect heartbeats and recovery/final accounting from TurnRunner.
NETWORK_WAIT_META = "network_wait"
# Stands between an answer and the follow-up turn sent on the same thread; its
# `kind` says why (`follow_up_kind`).
FOLLOWUP_META = "followup"
# The answer lacked what the caller's `--expect` named
# (`turn._follow_up_on_missing`); the line carries `expect`.
FOLLOWUP_EXPECT = "expect"
# A wrapper gate was red (`turn._check_gates`); the line carries `gates`.
FOLLOWUP_GATES = "gates"
_FOLLOWUP_KINDS = (FOLLOWUP_EXPECT, FOLLOWUP_GATES)
# One per caller turn: selected answer source and file/message disagreement.
REPORT_META = "report"
# Two per gate the wrapper runs after the turn (`gates.run_gates`): one as it
# starts (`running`), one with its exit code.
GATE_META = "gate"
# Sub-agents still running when the main thread's turn completed
# (`subagent_wait.await_main_and_sub_agents`); `phase` says which moment:
# `SUB_AGENTS_WAITING` opens the wait (`open`: their paths, `grace_s`),
# `SUB_AGENTS_CUT` is the moment `cut` were sent `turn/interrupt`,
# `SUB_AGENTS_FOLLOW_UP` stands before the one turn sent to the main thread
# about them (`kind`: `subagents.DONE_TURN` / `RESTART_TURN`, with the
# `finished` and `cut` paths it carries), and `SUB_AGENTS_END` closes the
# attempt's account (`cut` and `finished`: one `{path, thread, files, told,
# status, restarted}` per child; `waited_s`, `turn`, `note`). Waiting lasts
# from the first line to the last, across any main turn in between.
SUB_AGENTS_META = "subagents"
SUB_AGENTS_WAITING = "waiting"
SUB_AGENTS_CUT = "cut"
SUB_AGENTS_FOLLOW_UP = "follow_up"
SUB_AGENTS_END = "end"
# The attempt's thread opener (`turn._run_turn_logged`): `OPENER_THREAD_OPEN`
# is written just before `thread/start` or `thread/resume` and carries the
# extra silence that request may take (`grace_s`); `OPENER_DONE` follows it on
# every exit. Only between the two is the log entitled to that extra silence
# (`cli._turn_in_progress`) — a grace granted to every phase reported a worker
# that died mid-turn ten minutes late.
OPENER_META = "opener"
OPENER_THREAD_OPEN = "thread_open"
OPENER_DONE = "done"


def follow_up_kind(message: dict) -> str:
    """Why a `FOLLOWUP_META` line's follow-up was sent: `FOLLOWUP_EXPECT` or
    `FOLLOWUP_GATES`. One reader for `cli.cmd_logs` and `last_turn_trace`, so a
    third kind is added here and in the writer, not guessed from keys twice.

    A line written before `kind` existed is told apart by its `gates` list —
    only the gate follow-up ever wrote one. A `kind` this reader does not know
    raises instead: a third kind left out of `_FOLLOWUP_KINDS` would otherwise
    be counted silently as one of the two."""
    if "kind" not in message:
        return FOLLOWUP_GATES if message.get("gates") else FOLLOWUP_EXPECT
    kind = message["kind"]
    if kind not in _FOLLOWUP_KINDS:
        raise ValueError(f"unknown follow-up kind {kind!r} "
                         f"(known: {', '.join(_FOLLOWUP_KINDS)})")
    return kind


def completed_item(message: dict) -> Optional[dict]:
    """Return an `item/completed` payload, or None for another event."""
    if message.get("method") != N_ITEM_COMPLETED:
        return None
    params = message.get("params") or {}
    return params.get("item") or {}


def text_of(item: dict) -> str:
    """The visible text of an item, whichever shape it arrived in.

    An `agentMessage` carries a flat `text` (schema bundle of codex-cli
    0.149.0), not a content list; a reader that expects only the list turns
    a real answer into "the agent had nothing to say".
    """
    text = item.get("text")
    if isinstance(text, str) and text:
        return text
    content = item.get("content")
    if isinstance(content, list):
        parts = [str(part.get("text") or "") for part in content
                 if isinstance(part, dict) and part.get("type") == "text"]
        joined = "\n".join(p for p in parts if p)
        if joined:
            return joined
    message = item.get("message")
    return message if isinstance(message, str) else ""


def item_work(item: dict) -> tuple[int, int]:
    """How much work one completed item stands for: (commands, file changes).

    One function because two readers of the same stream count it: `TurnRunner`
    while the turn runs (its numbers reach the summary line) and
    `last_turn_trace` re-reading the log afterwards (its numbers reach
    `status`). The two are shown to the same caller about the same turn, so a
    second spelling of "a fileChange item is as many edits as it lists" is a
    way for `RUNNING … (3 edits)` and `3 cmd, 5 edits` to disagree.
    """
    kind = item.get("type")
    if kind == "commandExecution":
        return 1, 0
    if kind == "fileChange":
        # A fileChange with an empty (or missing) `changes` list is still one
        # edit: the server sends that shape, and counting zero would show a
        # turn that rewrote files as one that touched nothing.
        return 0, len(item.get("changes") or []) or 1
    return 0, 0


def edit_decision(entry: dict) -> str:
    """One line for a fileChange approval the wrapper answered
    (`protocol.AppServer._answer_file_change`), shared by `cli.cmd_logs` and
    the turn summary. "Declined by the wrapper" is the point: the agent saw
    "rejected by user", and the caller must not go looking for a user."""
    verdict = ("retried outside the sandbox (granted by the wrapper)"
               if entry.get("decision") == "accept"
               else "declined by the wrapper")
    paths = ", ".join(str(path) for path in entry.get("paths") or []) or "?"
    reason = f": {entry['reason']}" if entry.get("reason") else ""
    asked = (f" (sandbox: {entry['sandbox_reason']})"
             if entry.get("sandbox_reason") else "")
    return f"edit {verdict}{reason} — {paths}{asked}"


def changed_paths(item: dict) -> list[str]:
    """The files a `fileChange` item names, in order; [] for another item or
    a change that names none (counted all the same by `item_work`)."""
    if item.get("type") != "fileChange":
        return []
    return [str(change.get("path")) for change in item.get("changes") or []
            if isinstance(change, dict) and change.get("path")]


@dataclass
class TurnTrace:
    """What the log says about an agent's LAST turn — enough for `status` to
    tell a turn in progress from one that ended, and how much it did so far."""
    started: str = ""        # the `at` stamp of the opening line
    timeout_s: float = 0.0   # the bound that turn was started with, 0 if unknown
    ended: bool = False
    status: str = ""
    commands: int = 0
    file_changes: int = 0
    # Attempts this turn was resent after the service dropped it — the ones
    # that were actually SENT, so a slot the reachability check refused does
    # not read as work the agent redid.
    retries: int = 0
    reconnecting_since: str = ""
    reconnecting_tries: int = 0
    # Follow-up turns sent because the answer lacked what `--expect` named —
    # the same count as `TurnOutcome.follow_ups`.
    follow_ups: int = 0
    # Follow-up turns sent because a wrapper gate was red — counted apart, as
    # `TurnOutcome.gate_follow_ups` is, so `status` and the summary line agree.
    gate_follow_ups: int = 0
    # Whether the wrapper is in its sub-agent phase (from the waiting line to
    # the closing one), at most how many it waits for, and how long that wait
    # may last: a log may stay silent for it on top of the turn's timeout.
    sub_agent_phase: bool = False
    waiting_sub_agents: int = 0
    sub_agent_grace_s: float = 0.0
    # Paths of the sub-agents this turn interrupted (`SubAgentAccount.cut`).
    cut_sub_agents: tuple[str, ...] = ()
    # The extra silence the pending thread opener may take (`OPENER_META`);
    # 0 outside it.
    opener_grace_s: float = 0.0


def iter_log(path: Path, *, start: int = 0):
    """Every parseable line of an events log from byte `start` on, in order.

    A torn line (the writer is mid-`write` when a reader arrives, or `start`
    landed inside a line) is skipped, not raised — including one torn inside a
    multi-byte character, which `errors="replace"` turns into a line
    `json.loads` refuses rather than a `UnicodeDecodeError` out of `status`
    (prompts here are non-ASCII; review 2026-09-13). The replace is what the
    narrow `except` below rests on: caught as `ValueError` — which
    `UnicodeDecodeError` also is — a strict decode would have read as a
    skippable line and hidden that this ever mattered.

    Opened in binary because `start` is a byte offset: text-mode `seek` accepts
    only opaque cookies its own `tell` returned.
    """
    with path.open("rb") as handle:
        if start:
            handle.seek(start)
        for raw in handle:
            try:
                yield json.loads(raw.decode("utf-8", "replace"))
            except json.JSONDecodeError:
                continue


# How much of a log's tail `_last_turn_offset` reads at a time, doubling until
# it finds a turn's opening line or reaches the start of the file. A turn's own
# records sit between that line and the end of the log, so one chunk covers an
# ordinary turn; a long one costs a doubling or two, and every doubling does
# step back into earlier turns — what the tail read buys is that the reading
# stops growing with the LOG, not that nothing before the last turn is ever
# touched. Both errors are cheap — too small buys another read, too large reads
# bytes nobody needed — so this is a round number rather than a measured one.
#
# What it saves: `status` calls `last_turn_trace` once per listed agent, and a
# log keeps every turn its agent ever ran. Measured 2026-09-13 (Windows, warm
# cache, 42 MB log of 120k event lines): reading it front-to-back took 0.178 s
# per agent, the tail read 0.0007 s. The point is less the ratio than the
# shape — the full parse grows with the log, the tail read does not.
_TAIL_CHUNK_BYTES = 256 * 1024


def _last_turn_offset(path: Path) -> int:
    """Byte offset of the last `TURN_META` line in `path` — a start for
    `iter_log` that skips the turns nobody asked about.

    0 when the log has no such line: `iter_log` then parses the whole file,
    having already read it here — the one shape where this costs rather than
    saves, and the reason it is worth keeping a log's opening line."""
    size = path.stat().st_size
    window = _TAIL_CHUNK_BYTES
    while True:
        start = max(0, size - window)
        with path.open("rb") as handle:
            handle.seek(start)
            chunk = handle.read(size - start)
        # Away from the file's start the window opens mid-line; drop that
        # fragment (and the whole window when it holds no line end at all). A
        # turn line the cut split is found by the next, wider window, because
        # losing it leaves this one with nothing to return.
        pos = 0
        if start:
            newline = chunk.find(b"\n")
            pos = len(chunk) if newline < 0 else newline + 1
        found = -1
        while pos < len(chunk):
            end = chunk.find(b"\n", pos)
            if end < 0:
                end = len(chunk)
            line = chunk[pos:end]
            # `b'"_meta"'` is the cheap filter: of a log full of server
            # traffic, only the few bookkeeping lines carry that key at all.
            if b'"_meta"' in line and _is_turn_line(line):
                found = start + pos
            pos = end + 1
        if found >= 0:
            return found
        if start == 0:
            return 0
        window *= 2


def _is_turn_line(line: bytes) -> bool:
    # Narrow, for `iter_log`'s reason: a `UnicodeDecodeError` swallowed here
    # would be a log the tail scan silently refuses to find a turn in.
    try:
        return json.loads(line.decode("utf-8", "replace")).get("_meta") == TURN_META
    except (json.JSONDecodeError, AttributeError):
        return False


def last_turn_trace(path: Path) -> Optional[TurnTrace]:
    """The trace of the last turn in `path`, or None without a log or a turn.

    A turn ends by its `END_META` line, written by `run_turn` whatever the
    outcome (completed, failed, timed out, aborted before the server started).
    A log written before that line existed ends at the server's own
    `turn/completed` instead, so an OLD finished turn is not reported as
    running; an old turn that failed or timed out has no end at all and reads
    as in progress — `cli._turn_in_progress` ages it out by the log's mtime.
    """
    if not path.exists():
        return None
    trace: Optional[TurnTrace] = None
    main = ""  # this turn's main thread (`codex_app_learn_main_thread`)
    # From the last opening line, not from the top: everything before it
    # belongs to turns this function would only overwrite (`_TAIL_CHUNK_BYTES`).
    for message in iter_log(path, start=_last_turn_offset(path)):
        meta = message.get("_meta")
        if meta == TURN_META:
            trace = TurnTrace(started=str(message.get("at") or ""),
                              timeout_s=float(message.get("timeout") or 0))
            main = ""
            continue
        main = codex_app_learn_main_thread(message, main)
        if trace is None:
            continue
        if meta == NETWORK_WAIT_META:
            trace.reconnecting_since = (str(message.get("since") or "")
                                        if message.get("active") else "")
            trace.reconnecting_tries = int(message.get("tries") or 0)
            continue
        if meta == END_META:
            trace.ended = True
            trace.status = str(message.get("status") or "")
            continue
        if meta == OPENER_META:
            trace.opener_grace_s = (float(message.get("grace_s") or 0)
                                    if message.get("phase") == OPENER_THREAD_OPEN
                                    else 0.0)
            continue
        if meta == RETRY_META:
            # Each attempt opens its own thread: nothing of the last
            # attempt's opener carries over into the pause before the next.
            trace.opener_grace_s = 0.0
            # A retry REOPENS the turn: the dropped attempt already wrote its
            # own `turn/completed`, and leaving that verdict standing would let
            # `status` report a turn as failed while its next attempt is
            # running. The closing line is written once, after the last
            # attempt. `reachable` false means the slot was spent waiting and
            # nothing was resent.
            trace.retries += bool(message.get("reachable"))
            trace.ended = False
            trace.status = ""
            continue
        if meta == FOLLOWUP_META:
            # Same reopening, for the same reason: the first answer's
            # `turn/completed` is on the log and the agent is busy again.
            if follow_up_kind(message) == FOLLOWUP_GATES:
                trace.gate_follow_ups += 1
            else:
                trace.follow_ups += 1
            trace.ended = False
            trace.status = ""
            continue
        if meta == GATE_META:
            # The wrapper is running a gate after the server's
            # `turn/completed`: the turn is not over until `END_META`.
            trace.ended = False
            trace.status = ""
            continue
        if meta == SUB_AGENTS_META:
            phase = message.get("phase")
            if phase == SUB_AGENTS_END:
                trace.sub_agent_phase = False
                trace.waiting_sub_agents = 0
                trace.cut_sub_agents += tuple(
                    str(cut.get("path") or "?")
                    for cut in message.get("cut") or [])
                continue
            # The main thread's `turn/completed` above is not the end of this
            # turn, and no main `turn/completed` is until the closing line.
            trace.sub_agent_phase = True
            trace.ended = False
            trace.status = ""
            if phase == SUB_AGENTS_WAITING:
                trace.waiting_sub_agents = len(message.get("open") or [])
                trace.sub_agent_grace_s += float(message.get("grace_s") or 0)
            else:
                # Cut, or the turn about them sent: no longer waiting.
                trace.waiting_sub_agents = 0
            continue
        if message.get("method") == N_TURN_COMPLETED:
            if codex_app_from_sub_agent(message, main):
                # A sub-agent finishing is not this turn ending (it is still
                # activity: its items below count like the main thread's).
                continue
            # A main turn inside the sub-agent phase (a continuation, or the
            # turn sent about the children) ends nothing: the wrapper may
            # still wait, interrupt or send.
            trace.ended = not trace.sub_agent_phase
            turn = (message.get("params") or {}).get("turn") or {}
            trace.status = (str(turn.get("status") or "completed")
                            if trace.ended else "")
            continue
        item = completed_item(message)
        if item is None:
            continue
        commands, file_changes = item_work(item)
        trace.commands += commands
        trace.file_changes += file_changes
    return trace
