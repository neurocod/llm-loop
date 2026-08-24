"""operator.py - notes typed by the human at the console, on their way to the agent.

The loop is autonomous, not unattended: someone watches it work, and what they
see often deserves a sentence ("that folder already has a research.md", "use the
kit's lathe instead of rolling your own"). Without a way in, that sentence costs
a stop, an edit and a relaunch - so it usually goes unsaid.

This module is the mailbox between the console and the run: the status line puts
text IN (from its key-reader thread) and the runners take it OUT (on the main
thread), which is what keeps the UI and the transport from having to know about
each other. The TRANSPORT knows nothing about terminals or providers — a note is
a string here, framed but never rendered — and the one exception is deliberate
and one-directional: `report_undelivered_notes` at the bottom is the closing
word about a mailbox that ends a run still holding notes, so it prints, through
`console` like everything else with something to say. It carries no knowledge
back the other way; nothing above it may.

Two deliveries, and the difference is only WHEN the agent sees the note:

  * live - an iteration is in flight and its provider process accepts more user
    messages on stdin (`AgentChannel`). The note reaches the model at its next
    turn boundary, mid-iteration. Measured against `claude` 2.1.238: a note sent
    while a tool call was running was picked up ~1.8 s later, after that call
    finished, and the turn continued rather than restarting.
  * queued - no such channel right now (between iterations, a provider without
    the transport, `--no-live-messages`). The note waits and is appended to the
    NEXT iteration's prompt by `append_notes`.

Both are framed before the agent sees them (`frame_live` / `append_notes`) and
the framing is load-bearing, not decoration. A bare imperative arriving mid-turn
reads exactly like a prompt injection - in a probe of the raw transport, a note
that landed without context was refused as one - so the frame says who is
speaking and how much authority the sentence carries.
"""

import json
import threading
from typing import List, NamedTuple, Optional, Tuple

from . import wire
from .console import print_error

__all__ = [
    "AgentChannel",
    "ChannelError",
    "Delivery",
    "Mailbox",
    "NullChannel",
    "append_notes",
    "frame_live",
    "report_undelivered_notes",
    "user_message_line",
]


# How many delivered-but-unacknowledged notes are remembered for the replay echo
# (see Mailbox.claim_echo). A bound, because a provider that never replays would
# otherwise grow the list for the length of an eight-hour run.
MAX_AWAITING_ECHO = 20


# The frame around a note delivered INTO a running turn. It answers the three
# questions the model would otherwise have to guess at: where the text came
# from (a human, not the task file), what weight it carries (guidance), and
# whether the current task is cancelled by it (no, unless the note says so).
# The last clause is not politeness: in the probe behind this module a note
# reading "stop reading files" was obeyed to the letter and the rest of the
# iteration's plan was abandoned, which is right when meant and expensive when
# not.
LIVE_NOTE_HEADER = (
    "[Operator note - typed at the console by the human running this loop while "
    "you work. It is guidance about the task you are already on: carry on with "
    "it unless the note itself says otherwise.]")

# The same thing for notes that missed their turn and ride the next prompt.
# Placed at the END of that prompt: it is an addendum to the task above it, and
# a reader (human or model) meets it after the task it qualifies.
QUEUED_NOTES_HEADER = (
    "Operator notes\n"
    "The human running this loop typed the following at the console before this "
    "iteration started. Treat them as guidance on top of the task above; they do "
    "not replace it unless they say so.")


def frame_live(text: str) -> str:
    """One note as the running agent should see it."""
    return f"{LIVE_NOTE_HEADER}\n\n{text}"


def append_notes(prompt: str, notes: List[str]) -> str:
    """`prompt` with the queued notes appended as a closing section.

    Returns the prompt unchanged when there is nothing to append, so callers can
    apply it unconditionally.
    """
    if not notes:
        return prompt
    body = "\n".join(f"- {note}" for note in notes)
    return f"{prompt.rstrip()}\n\n{QUEUED_NOTES_HEADER}\n\n{body}\n"


def user_message_line(text: str) -> str:
    """One `--input-format stream-json` user message, newline-terminated.

    The wire format the provider CLI reads: one JSON object per line, shaped
    like an API message. `ensure_ascii` is left on so the line survives a stdin
    pipe with any encoding.

    The SHAPE is `wire.user_message`, not spelled here: the CLI replays this note
    back on the stream the renderers read, where `Mailbox.claim_echo` has to
    recognise it again — one fact with two ends, so one home.
    """
    return json.dumps(wire.user_message(text)) + "\n"


class ChannelError(RuntimeError):
    """The live channel could not take the note (closed, or the pipe broke)."""


class AgentChannel:
    """The stdin half of a provider process that reads user messages while it works.

    Thread-safe by construction: the console thread writes to it, the main
    thread closes it when the turn's `result` event arrives, and a note that
    loses that race is refused rather than written into a pipe nobody is
    reading any more.

    A `mailbox` is attached on construction and detached on close, because
    those two are one pair: a mailbox still holding a closed channel would keep
    offering the console a pipe nobody reads. Left None, the channel exists only
    to be closed — which every process of a streaming-input CLI needs, whether
    or not anybody is listening (see `providers.note_channel`).
    """

    def __init__(self, stream, mailbox: Optional["Mailbox"] = None):
        self._stream = stream
        self._lock = threading.Lock()
        self._closed = False
        self._mailbox = mailbox
        if mailbox is not None:
            mailbox.attach(self)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def send(self, text: str) -> None:
        """Write one user message. Raises ChannelError if it did not go out."""
        with self._lock:
            if self._closed:
                raise ChannelError("the turn is over")
            try:
                self._stream.write(user_message_line(text))
                self._stream.flush()
            except (OSError, ValueError) as exc:
                # ValueError: the file object was closed under us. Both mean the
                # same thing to the caller - the note did not reach the agent.
                self._closed = True
                raise ChannelError(str(exc)) from exc

    def close(self) -> None:
        """Close the pipe: for a streaming-input CLI this is what ends the session.

        Safe to call twice; never raises, because it runs on the way out of a
        run that may already be failing for another reason. Detaches first, so
        the window in which the console can still reach a closing pipe is as
        small as it can be — and a note that lands inside it is refused and
        queued rather than lost.
        """
        mailbox, self._mailbox = self._mailbox, None
        if mailbox is not None:
            mailbox.detach()
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass


class NullChannel:
    """No pipe to talk through, and closing it is a no-op.

    Exists so a runner can hold "the channel for this process" unconditionally:
    the alternative is an `if channel is None` before every use, which is
    exactly the guard that once tied closing the pipe to whether anyone wanted
    to write to it.
    """

    closed = True

    def send(self, text: str) -> None:
        raise ChannelError("this run has no live channel")

    def close(self) -> None:
        return None


class Delivery(NamedTuple):
    """What became of one submitted note - the status line shows `message`."""

    live: bool          # went straight into the turn in flight
    queued: int         # notes now waiting for the next iteration
    error: str = ""     # why the live attempt failed, when it did

    @property
    def message(self) -> str:
        if self.live:
            return "note sent to the agent"
        if self.error:
            return (f"live delivery failed ({self.error}) - queued for the next "
                    f"iteration ({self.queued} waiting)")
        return f"queued for the next iteration ({self.queued} waiting)"


class Mailbox:
    """Where a note waits: in the running turn if there is one, else in the queue.

    One per run. Every method is safe to call from any thread; `submit` is the
    console's, `attach`/`detach`/`take_queued`/`claim_echo` are the runner's.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._queued: List[str] = []
        self._channel: Optional[AgentChannel] = None
        # (what was written, what the human typed) for notes whose replay echo
        # has not come back yet - see claim_echo.
        self._awaiting: List[Tuple[str, str]] = []

    # --- the runner's side -------------------------------------------------

    def attach(self, channel: AgentChannel) -> None:
        """A turn is in flight and takes live notes."""
        with self._lock:
            self._channel = channel

    def detach(self) -> None:
        """The turn is over; further notes go to the queue."""
        with self._lock:
            self._channel = None

    def take_queued(self) -> List[str]:
        """Drain the queue - the caller is about to put these into a prompt."""
        with self._lock:
            queued, self._queued = self._queued, []
            return queued

    def splice(self, prompt: str) -> Tuple[str, List[str]]:
        """(`prompt` with the queued notes in it, the notes that went in).

        Both runners do this at the same moment - after the driver names the
        next command and BEFORE the status line records its prompt, so the
        prompt shown is the prompt sent. That ordering is policy, and policy
        that lives at two call sites is policy only one of them will follow the
        next time it changes; what legitimately differs between them - how a
        note is announced - stays with the caller.
        """
        notes = self.take_queued()
        return append_notes(prompt, notes), notes

    def claim_echo(self, replayed: str) -> Optional[str]:
        """The human's text for a replayed user message, or None if it is not ours.

        `--replay-user-messages` sends every message we write back on the event
        stream, including the iteration's own prompt. Matching against what this
        mailbox actually sent is what separates "the note landed" from "the task
        prompt is being echoed back", without the renderer having to know the
        difference.
        """
        with self._lock:
            for index, (framed, original) in enumerate(self._awaiting):
                if framed == replayed:
                    del self._awaiting[index]
                    return original
            return None

    # --- the console's side ------------------------------------------------

    @property
    def queued_count(self) -> int:
        with self._lock:
            return len(self._queued)

    def submit(self, text: str) -> Optional[Delivery]:
        """Deliver a note now if a turn is in flight, else queue it.

        None for an empty note, so the caller can hand raw input straight in.
        """
        text = (text or "").strip()
        if not text:
            return None
        with self._lock:
            channel = self._channel
        if channel is not None:
            framed = frame_live(text)
            # Recorded BEFORE the write, and unrecorded if the write fails. The
            # replay comes back on the RUNNER's thread, which can be inside
            # claim_echo while this one is still between two statements: an echo
            # that arrives before its entry exists matches nothing and is
            # dropped, so the note lands and the log never says so.
            with self._lock:
                self._awaiting.append((framed, text))
                # Keep the newest: a mailbox this far behind is one whose
                # replays are not coming back at all, and then the entry worth
                # holding is the one whose note was typed most recently.
                del self._awaiting[:-MAX_AWAITING_ECHO]
            try:
                channel.send(framed)
            except ChannelError as exc:
                with self._lock:
                    if (framed, text) in self._awaiting:
                        self._awaiting.remove((framed, text))
                return self._queue(text, error=str(exc))
            return Delivery(live=True, queued=0)
        return self._queue(text)

    def _queue(self, text: str, error: str = "") -> Delivery:
        with self._lock:
            self._queued.append(text)
            return Delivery(live=False, queued=len(self._queued), error=error)


def report_undelivered_notes(mailbox) -> None:
    """Say so when the run ends holding notes nobody ever saw.

    A queued note promises "the next iteration" — and there is not always a next
    one (the list drained, --max-runs, a stop request, a quota pause that
    outlasts the run). Whoever typed it during the last iteration would
    otherwise have no way to learn it was never delivered, and the text itself
    is printed so it can be pasted into the next run rather than retyped.

    Both runners end with this call and neither owns it, so it is here — with
    the mailbox it is about — rather than in whichever loop happened to grow it
    first. Same shape as `gitpush.git_push`, and it stayed behind for the same
    reason that one did: it had to print, and printing used to live in the
    sequential runner too.
    """
    if mailbox is None:
        return
    notes = mailbox.take_queued()
    if not notes:
        return
    print_error(f"\n  ⚠ the run ended holding {len(notes)} undelivered operator "
                f"note(s) — there was no next iteration to carry them:")
    for note in notes:
        print_error(f"      {note}")
