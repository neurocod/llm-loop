"""Sub-agents of a turn's main thread: which still run, what they touched,
what the main thread has been told about them.

Bookkeeping only, fed event by event by `turn.TurnRunner`, and the account of
them an outcome carries (`SubAgentAccount`); waiting for them, interrupting
them and the turns sent about them are `subagent_wait`'s, which reads the
bounds below at call time — tests shorten them by patching this module.
"""

from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass, field
from typing import Optional

from .events import text_of, changed_paths
from .paths import bridge_command

# How long the wrapper keeps the app-server open after the MAIN thread's turn
# completed while sub-agents it (or they) spawned still have a turn open. User
# decision 2026-10-24: the standard 600 s, as Claude CLI waits for its own.
#
# Why wait at all: closing the server kills every thread on it, and the
# children's results were lost three times by 2026-10-24
# (reviewer rev-a's log line 708, `rev-b` 821, `rev-c` 475) — the
# main thread had answered after a short `wait_agent` timeout (10 s, 60 s; the
# model chooses it) or without waiting. Codex does not wake the main thread
# when a child finishes later (`wake_on_agent_messages` is off by default in
# rust-v0.164.0), so the wrapper sends it a turn itself (`DONE_TURN`).
#
# One deadline per attempt, counted from the first main answer that left
# children running, and enforced whatever the main thread does meanwhile: a
# main turn that continues by itself does not extend it (a review reproduced
# 1000 s of waiting under a 600 s grace when it did). What a shift costs:
# shorter cuts children that were going to finish (a worker's half-made
# edits, a reviewer's findings); longer only delays a caller whose child
# hangs. The `SUB_AGENTS_META` waiting line names it, so `status` does not
# call the silent log a killed process.
SUB_AGENT_GRACE_S = 600.0

# How long interrupted children get to report their aborted turn before they
# are written off as cut: Codex itself gives every thread 10 s to stop when
# its stdin closes (rust-v0.164.0), so waiting longer buys nothing. A child
# that ignores the interrupt costs exactly this, once per cut.
SUB_AGENT_INTERRUPT_S = 10.0

# The two turns the wrapper may send the main thread about its children, at
# most ONE per attempt: `DONE_TURN` when every child that outlived the answer
# finished, `RESTART_TURN` when some had to be cut (user decision 2026-10-24:
# "the worker can track it and restart its sub-agent"). Their names are the
# `kind` of the `SUB_AGENTS_FOLLOW_UP` log line.
DONE_TURN = "done"
RESTART_TURN = "restart"

# Both carry the children's results IN THE TEXT rather than trusting the main
# thread's inbox: Codex emits a child's terminal event before it routes the
# result (core/src/session/mod.rs:2522, agent/control/completion.rs), so a
# turn sent on that event may start before the result is queued, and a
# grandchild's result goes to its direct parent only — never to the main
# thread. `{report}` is the worker's report clause, `{children}` the list.
DONE_PROMPT = (
    "The sub-agents below finished only after your final answer, so you have "
    "not used their results yet; they are quoted here. Take them into account "
    "and give your COMPLETE final answer again, in the format the task asked "
    "for{report}. Do not start new sub-agents for this.\n\n{children}"
)
RESTART_PROMPT = (
    "Some sub-agents you started were still running when the wrapper's wait "
    "for them ran out, and were interrupted: their work is unfinished, and "
    "the edits they made so far are in the tree. For each interrupted one "
    "below, if its work is still needed, restart it (and wait for it with "
    "`wait_agent` before answering) or finish that work yourself. Then give "
    "your COMPLETE final answer again, in the format the task asked "
    "for{report}. Sub-agents still running when this turn ends are "
    "interrupted without another wait.\n\n{children}"
)

# How much of one child's message the turn about it (and the account under
# the answer) quotes, and how much all of them together may take. Codex
# rejects a `turn/start` whose text exceeds 1 048 576 characters
# (`MAX_USER_INPUT_TEXT_CHARS`, rust-v0.164.0), and a rejected turn loses the
# results it carried (`subagent_wait._sub_agent_turn` keeps the first answer); 20 000
# characters is several pages of findings, and the total leaves the limit a
# fivefold margin for the prompt around them. The rest is printed by
# `full_text_command`.
ANSWER_CHARS = 20_000
PROMPT_CHARS = 200_000

# Heading of the account under the answer (`render_block`); worded to hold
# when there is no answer above it at all.
BLOCK_HEADING = "### Sub-agents still running when the main turn ended"


@dataclass
class SubAgentFate:
    """A sub-agent still running when a main turn ended: what became of it."""
    path: str
    thread_id: str
    # "finished" (its turn ended while the wrapper waited, on its own) or
    # "cut" (interrupted: the wait ran out, or the main turn ended otherwise).
    state: str = "finished"
    files: list[str] = field(default_factory=list)
    # Whether a turn the wrapper sent the main thread — one that COMPLETED —
    # carried it: its result, or the news of its cut. Its last message,
    # quoted to the caller when nothing did.
    told: bool = False
    answer: str = ""
    # Its last turn's own verdict ("completed", "failed", "interrupted"; ""
    # when none came) and error: a message of a turn that did not complete
    # is partial, and is labelled so.
    status: str = ""
    error: str = ""
    # Cut once and then given a new turn — the main thread restarting it.
    restarted: bool = False


@dataclass
class SubAgentAccount:
    """What the sub-agents came to: one attempt's (`turn.TurnOutcome.
    sub_agents`, filled by `subagent_wait`), or a caller turn's once its
    attempts are folded (`fold`)."""
    # Sub-agents still running when a main turn ended, each with its fate
    # (`SubAgentTracker.fates`).
    late: list[SubAgentFate] = field(default_factory=list)
    # Wall-clock spent after a main answer because of them.
    wait_s: float = 0.0
    # Which turn was sent the main thread about them (`DONE_TURN` /
    # `RESTART_TURN`, "" for none).
    turn: str = ""
    # Why that turn's (or a continuation's) answer is not the one kept, when
    # it failed.
    note: str = ""
    # Their account when there is no answer text to put it under
    # (`under_answer`).
    block: str = ""
    # The turn about the children's own message when it wrote no report and
    # the first, complete report was put back: joined under that report
    # (`turn._read_report`).
    reply: str = ""

    @property
    def cut(self) -> list[SubAgentFate]:
        """The late sub-agents that were interrupted: for a worker, edits in
        the tree that the answer may not describe."""
        return [fate for fate in self.late if fate.state == "cut"]

    @classmethod
    def fold(cls, attempts: list["SubAgentAccount"]) -> "SubAgentAccount":
        """One caller turn's account from its attempts' (oldest first), each
        field by its own rule in `_FOLD`. A field without a rule raises
        rather than being dropped: the caller turn's outcome is a NEW
        account, so a field nobody folds would read as empty."""
        folded = {}
        for item in dataclasses.fields(cls):
            rule = _FOLD.get(item.name)
            if rule is None:
                raise TypeError(f"SubAgentAccount.{item.name} has no rule "
                                f"in subagents._FOLD")
            folded[item.name] = rule([getattr(attempt, item.name)
                                      for attempt in attempts])
        return cls(**folded)

    def under_answer(self, text: str, agent: str) -> str:
        """`text` with the account (`render_block`) under it — after the
        gates block, because it describes the tree as the children left it.

        With no answer text the block stays in `block`, which
        `cli.print_outcome` prints: as the text it would be stored as the
        agent's answer, and `last.md` must never trade a real answer of an
        earlier turn for a list of children (`turn.run_turn`'s guard)."""
        if not self.late and not self.note:
            return text
        block = render_block(self.late, self.note, agent)
        if text:
            return text.rstrip() + "\n\n" + block
        self.block = block
        return text


def _last_set(values: list[str]) -> str:
    """The last attempt's value that is set, or ""."""
    return next((value for value in reversed(values) if value), "")


# How `SubAgentAccount.fold` folds each field over the attempts. Every field
# has a line here, checked below at import: a field folded by nobody was the
# way a new one got lost in the hand-written fold this replaced.
_FOLD = {
    "late": lambda values: [fate for value in values for fate in value],
    "wait_s": sum,
    # The last attempt that sent one: a later attempt without one (a
    # follow-up) does not unsay the turn the children got.
    "turn": _last_set,
    "note": _last_set,
    # Rendered after the fold (`under_answer`); no attempt sets it.
    "block": lambda values: "",
    # The last attempt's own: `turn._read_report` has already joined each
    # attempt's reply under that attempt's report.
    "reply": lambda values: values[-1] if values else "",
}

assert set(_FOLD) == {item.name for item in dataclasses.fields(SubAgentAccount)}


class _Phase(enum.Enum):
    """Where a child's current generation stands."""
    NEW = "new"  # no turn opened yet
    RUNNING = "running"
    # Its turn ended inside a main turn, or before any main answer.
    ENDED = "ended"
    # Its turn ended while the main thread was idle on the attempt's
    # deadline, so no main turn read its result.
    ENDED_IDLE = "ended_idle"
    # Interrupted (`SubAgentTracker.mark_cut`), whatever it ended as before.
    CUT = "cut"


@dataclass
class _Child:
    path: str = "?"
    # The open turn's id; "" while the child is known only from its parent's
    # `subAgentActivity started` record, which can precede its `turn/started`
    # by long enough for the main turn to complete in between (rev-a: lines
    # 52 and 57, 36 ms).
    turn_id: str = ""
    phase: _Phase = _Phase.NEW
    # Turn ids that reached `turn/completed`: a late completion of an older
    # turn closes nothing and says nothing, and a `turn/started` replayed for
    # one opens nothing.
    ended: set[str] = field(default_factory=set)
    files: list[str] = field(default_factory=list)
    # Of the current turn only: reset when the child opens again.
    answer: str = ""
    status: str = ""
    error: str = ""
    # The current generation — set when the child (re)opens. `late`: it runs
    # on the attempt's deadline — open when a main turn completed, or opened
    # while the main thread was idle (a grandchild spawned during the wait).
    # A child opened inside a main turn belongs to that turn instead: the
    # restart turn resuming a cut child must not see it interrupted at once
    # against the deadline that cut it. `told`: a COMPLETED turn the wrapper
    # sent carried it.
    late: bool = False
    told: bool = False
    # Across generations: it was cut and then reopened.
    restarted: bool = False


def full_text_command(agent: str, path: str) -> str:
    """The command that prints a child's messages whole (`cli.cmd_logs
    --child`): where a capped quote sends its reader, spelled the way the
    host runs the bridge (`paths.bridge_command`)."""
    return f"`{bridge_command()} logs {agent} --child {path}`"


def _capped(text: str, limit: int, agent: str, path: str) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return (text[:limit] + f"\n[... {len(text) - limit} more characters: "
            f"{full_text_command(agent, path)} prints them]")


def _verdict(child) -> str:
    """How a child's last turn ended, for a heading: "" when it completed."""
    if child.status in ("", "completed"):
        return ""
    return f" ({child.status}" + (f": {child.error}" if child.error else "") + ")"


class SubAgentTracker:
    """Every thread but the main one, keyed by thread id — grandchildren
    included, since a grandchild's spawn and turns are announced on the same
    connection, on its own and its parent's threads."""

    def __init__(self) -> None:
        self._children: dict[str, _Child] = {}
        self.main_running = False

    def _child(self, thread: str) -> _Child:
        return self._children.setdefault(thread, _Child())

    def _open(self, child: _Child, turn_id: str) -> None:
        if child.phase is not _Phase.RUNNING:
            child.late = not self.main_running
            child.told = False
            child.answer = child.status = child.error = ""
            if child.phase is _Phase.CUT:
                child.restarted = True
        elif turn_id and child.turn_id and turn_id != child.turn_id:
            # A newer turn on a thread whose older one never completed: its
            # words are the newer turn's from here on.
            child.answer = child.status = child.error = ""
        child.phase = _Phase.RUNNING
        child.turn_id = turn_id

    # --- events

    def on_record(self, item: dict) -> None:
        """A parent's `subAgentActivity` record (on the parent's thread).

        `started` opens the child: its `turn/started` may not have arrived
        yet. The other kinds close nothing: `interrupted` is an interrupt
        SUBMITTED, not one carried out, and `completed` (id
        `subagent-completed-<turnId>`) duplicates the child's own
        `turn/completed`, the one signal sent whatever the turn's status."""
        if item.get("type") != "subAgentActivity" or item.get("kind") != "started":
            return
        thread = str(item.get("agentThreadId") or "")
        if not thread:
            return
        child = self._child(thread)
        child.path = str(item.get("agentPath") or "?")
        if (child.phase not in (_Phase.RUNNING, _Phase.CUT)
                and not child.ended):
            self._open(child, "")

    def on_turn_started(self, thread: str, turn_id: str) -> None:
        child = self._child(thread)
        if turn_id and turn_id in child.ended:
            return
        self._open(child, turn_id)

    def on_turn_completed(self, thread: str, turn: dict) -> None:
        """Only the child's CURRENT turn speaks: an older turn's completion
        (or a repeat) closes nothing and its payload is ignored."""
        child = self._child(thread)
        turn_id = str(turn.get("id") or "")
        if turn_id and turn_id in child.ended:
            return
        if turn_id:
            child.ended.add(turn_id)
        running = child.phase is _Phase.RUNNING
        if not (turn_id == child.turn_id or (running and not child.turn_id)):
            return
        for done_item in turn.get("items") or []:
            if isinstance(done_item, dict) and done_item.get(
                    "type") == "agentMessage" and text_of(done_item):
                child.answer = text_of(done_item)
        child.status = str(turn.get("status") or "completed")
        error = turn.get("error")
        child.error = str((error.get("message") if isinstance(error, dict)
                           else error) or "")
        if running:
            child.phase = (_Phase.ENDED_IDLE
                           if child.late and not self.main_running
                           else _Phase.ENDED)

    def on_item(self, thread: str, turn_id: str, item: dict) -> None:
        """A completed item on a child's own thread (`turn_id` as the event
        declares it, "" when it does not): only its current turn's message
        is its answer; every edit counts."""
        child = self._child(thread)
        if (item.get("type") == "agentMessage" and text_of(item)
                and (not turn_id or not child.turn_id or turn_id == child.turn_id)):
            child.answer = text_of(item)
        if item.get("type") == "fileChange":
            child.files += (changed_paths(item)
                            or ["(an edit the server named no file for)"])

    def main_started(self) -> None:
        self.main_running = True

    def main_completed(self) -> None:
        self.main_running = False
        for child in self._children.values():
            if child.phase is _Phase.RUNNING:
                child.late = True

    # --- questions

    def open_threads(self, *, late_only: bool = False) -> dict[str, str]:
        """thread -> open turn id ("" before its `turn/started`)."""
        return {thread: child.turn_id for thread, child in self._children.items()
                if child.phase is _Phase.RUNNING
                and (child.late or not late_only)}

    def is_open(self, thread: str) -> bool:
        child = self._children.get(thread)
        return child is not None and child.phase is _Phase.RUNNING

    def path(self, thread: str) -> str:
        child = self._children.get(thread)
        return child.path if child is not None else "?"

    def mark_cut(self, threads) -> None:
        """After `turn/interrupt` and its drain: a child still open (it
        ignored the interrupt) or whose turn ended other than `completed` is
        cut; one that completed on its own meanwhile finished."""
        for thread in threads:
            child = self._child(thread)
            if child.phase is _Phase.RUNNING or child.status != "completed":
                child.phase = _Phase.CUT

    def news(self) -> tuple[list[str], list[str]]:
        """(finished, cut) threads whose fate no completed turn of the
        wrapper's has carried yet. Finished means finished while the main
        thread was idle: one that ended inside a main turn is quoted to the
        caller instead (`fates`), not sent back."""
        def untold(phase: _Phase) -> list[str]:
            return [thread for thread, child in self._children.items()
                    if child.phase is phase and not child.told]

        return untold(_Phase.ENDED_IDLE), untold(_Phase.CUT)

    def turn_prompt(self, report_clause: str, agent: str
                    ) -> Optional[tuple[str, str, list[str], list[str]]]:
        """The one turn to send the main thread about its children: (kind,
        text, finished threads, cut threads), or None when there is no news.
        Marks nothing: `tell` does, once the turn has completed."""
        finished, cut = self.news()
        if not finished and not cut:
            return None
        kind = RESTART_TURN if cut else DONE_TURN
        parts, room = [], PROMPT_CHARS
        for state, threads in (("Interrupted", cut), ("Finished", finished)):
            for thread in threads:
                child = self._children[thread]
                files = ", ".join(child.files) if child.files else "none"
                verdict = "" if child.phase is _Phase.CUT else _verdict(child)
                head = (f"### {state}{verdict}: {child.path}\n"
                        f"Files it changed: {files}\n")
                if not child.answer.strip():
                    answer = "(no message)"
                elif room <= 0:
                    answer = (f"(omitted: the quoted messages reached "
                              f"{PROMPT_CHARS} characters; "
                              f"{full_text_command(agent, child.path)} "
                              f"prints it)")
                else:
                    answer = _capped(child.answer, min(ANSWER_CHARS, room),
                                     agent, child.path)
                    room -= len(answer)
                label = ("Its last message:" if child.status == "completed"
                         else "Its last message (PARTIAL — its turn did not "
                              "complete):")
                parts.append(f"{head}{label}\n{answer}\n")
        template = RESTART_PROMPT if kind == RESTART_TURN else DONE_PROMPT
        return (kind, template.format(report=report_clause,
                                      children="\n".join(parts)),
                finished, cut)

    def tell(self, threads) -> None:
        """The turn that carried `threads` completed. A child the main thread
        reopened during it (a restart) is a new generation, not the one told
        about."""
        for thread in threads:
            child = self._child(thread)
            if child.phase in (_Phase.CUT, _Phase.ENDED_IDLE):
                child.told = True

    def fates(self) -> list[SubAgentFate]:
        """Every child that outlived a main answer, was cut, or was restarted
        after a cut, in spawn order; one that ran and ended inside the main
        turn is not news."""
        return [SubAgentFate(path=child.path, thread_id=thread,
                             state=("cut" if child.phase is _Phase.CUT
                                    else "finished"),
                             files=list(child.files), told=child.told,
                             answer="" if child.told else child.answer,
                             status=child.status, error=child.error,
                             restarted=child.restarted)
                for thread, child in self._children.items()
                if child.late or child.phase is _Phase.CUT or child.restarted]


def render_block(fates: list[SubAgentFate], note: str, agent: str) -> str:
    """The account put under the answer (`SubAgentAccount.under_answer`), with the
    note of a turn about them that failed — `result` must say it too."""
    lines = [BLOCK_HEADING]
    for fate in fates:
        touched = (f"; touched {', '.join(fate.files)}" if fate.files
                   else "; touched no files")
        again = " after its restart" if fate.restarted else ""
        if fate.state == "cut" and fate.told:
            lines.append(f"- {fate.path}: CUT{again} — interrupted while still "
                         f"running; the main thread was asked to restart it "
                         f"or finish its work{touched}.")
        elif fate.state == "cut":
            lines.append(f"- {fate.path}: CUT{again} — interrupted while "
                         f"still running{touched}. Its edits are in the tree, "
                         f"and no answer of the main thread accounts for "
                         f"them.")
        elif fate.told:
            lines.append(f"- {fate.path}: finished after the main answer; its "
                         f"result was passed to the main thread{touched}.")
        else:
            how = ("restarted by the main thread after its cut, ended inside "
                   "its turn" if fate.restarted else
                   "finished after the main answer, UNREAD by the main thread")
            verdict = _verdict(fate)
            lines.append(f"- {fate.path}: {how}{verdict}{touched}."
                         + (" Its answer" + (" (PARTIAL)" if verdict else "")
                            + ":" if fate.answer.strip()
                            else " It left no message."))
            lines += ["  > " + line for line in
                      _capped(fate.answer, ANSWER_CHARS, agent,
                              fate.path).splitlines()]
    if note:
        lines.append(f"Note: {note}")
    return "\n".join(lines) + "\n"
