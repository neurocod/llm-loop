"""One provider run, rendered live into the terminal — the sequential runner's half.

Everything about turning ONE stream of provider events into what a person
watching a run sees: start the CLI, read its JSONL, print each event in the
style of interactive mode, and hand back the exit code. The words the stream is
made of are `wire`, shared with the other renderer; the SHAPES of the printed
lines are `compactline` and `console`, shared with it too. What is genuinely
this module's is the single-stream state those two cannot have: which content
block text is flowing into, the live Markdown renderer it flows through, the
session cost already reported, and the latch holding the run's own rate-limit
verdict.

That state is why this is a module rather than a pair of functions on the loop:
it is module-global by nature, it is only correct with ONE stream in flight, and
`run_loop` is the caller that has exactly one. `parallel.run_job` renders the
same stream WITHOUT any of it — N at once, one atomic line per event — which is
the same reason its rate-limit verdict is a local there and a latch here.

The loop that OWNS the run (when to start another iteration, when to pause on a
quota, when to stop) is `cyclecore`, and it reaches these through its own module
globals so a test can replace them: 19 pins across six files patch
`cyclecore.run_claude_streaming` / `cyclecore.run_agent_streaming`, and they
still bite because the loop calls the name it imported rather than reaching
through this module.
"""

import json
import sys
from typing import Optional

from . import compactline, projectroot, wire
from .console import (
    LINES,
    MarkdownStream,
    print_done,
    print_error,
    print_note,
    render_markdown_block,
)
from .providers import (
    note_channel,
    provider_spec,
    reap_agent_process,
    start_agent_process,
)
from .usage import RateLimitEvent, rate_limit_event_from

# Streaming print state: the single content-block index text is currently flowing
# into (assistant replies stream one text block at a time), plus its live renderer.
_active_text_index = None
_md_stream = MarkdownStream()

# The session cost already reported by earlier `result` events of the process
# now streaming. Reset by run_agent_streaming; see the result branch for why the
# figure has to be differenced at all.
_turn_cost_base = 0.0


# --- the free half of the limit machinery: the run's own rate-limit events -----
#
# What such an event IS — `RateLimitEvent`, `rate_limit_event_from`, and the
# labels the CLI gives each quota — is `usage`, next to everything else known
# about a quota, because the PARALLEL runner parses the same events out of the
# same stream and had to import a runner to name them.
#
# THE LATCH BELOW STAYS WITH THE RENDERER, and that is a decision rather than a
# leftover. `_last_rate_limit_event` answers "the verdict of the run that just
# finished", which is a question only a caller with ONE run in flight can ask.
# This renderer is that caller by construction — it already keeps module-global
# streaming state that no second concurrent stream could share, which is why
# `parallel.run_job` prints its own compact lines instead of reusing it, and
# `run_agent_streaming` clears the latch on entry so the answer describes this
# run and not the previous one. The parallel runner has N streams at once and
# keeps each job's verdict as a LOCAL in `run_job`, which is the only correct
# shape there. Moving the latch into a module both runners import would publish
# an address that cannot be right for one of them — a process-global "the last
# verdict" is meaningless when ten runs are in flight — so the vocabulary is
# shared and the latch is not.

# The last verdict seen by run_claude_streaming, cleared when a run starts — so it
# always describes the run that just finished, never the one before it.
_last_rate_limit_event = None


def last_rate_limit_event() -> Optional[RateLimitEvent]:
    """The rate-limit verdict of the most recent run_claude_streaming call (None
    if that run streamed no rate_limit_event)."""
    return _last_rate_limit_event


def _render_claude_event(ev: dict, partial: bool, mailbox=None) -> None:
    """Print a single stream-json event in the style of interactive mode.

    partial=True — --include-partial-messages is enabled: we print text from the
    deltas (`stream_event`), and from the final `assistant` we take only the tool
    calls, so as not to duplicate already-printed text.

    `mailbox` turns the replay of an operator note (`--replay-user-messages`)
    into a console line. Rendered here rather than where the note was typed
    because this is the main thread, between two events — the one place a line
    can be printed without cutting into the live Markdown block — and because
    the replay is the CLI confirming delivery, not the console assuming it.
    """
    et = wire.event_type(ev)

    if wire.is_session_start(ev):
        print(f"  · session started (model {wire.session_model(ev)})")
        return

    if et == wire.RATE_LIMIT_EVENT:
        global _last_rate_limit_event
        _last_rate_limit_event = rate_limit_event_from(ev)
        # "allowed" is the normal case and would be one more line per run; the
        # two states that mean something get shown.
        if _last_rate_limit_event.status != "allowed":
            print_error(f"  ⚠ rate limit: {_last_rate_limit_event.describe()}")
        return

    # Streaming deltas (Anthropic streaming events, wrapped in stream_event).
    if et == wire.STREAM_EVENT:
        global _active_text_index
        inner = wire.stream_inner(ev)
        it = wire.event_type(inner)
        if it == wire.CONTENT_BLOCK_START:
            if wire.stream_block_is_text(inner):
                _active_text_index = wire.stream_block_index(inner)
                _md_stream.start()
        elif it == wire.CONTENT_BLOCK_DELTA:
            text = wire.stream_text_delta(inner)
            if (text is not None
                    and wire.stream_block_index(inner) == _active_text_index):
                _md_stream.feed(text)
        elif it == wire.CONTENT_BLOCK_STOP:
            if wire.stream_block_index(inner) == _active_text_index:
                _md_stream.stop()  # finalize the Markdown render / line
                _active_text_index = None
        return

    if et == wire.ASSISTANT:
        for block in wire.message_blocks(ev):
            bt = wire.block_type(block)
            if bt == wire.BLOCK_TEXT:
                if partial:
                    continue  # already printed streaming from the deltas
                render_markdown_block(wire.block_text(block))
            elif bt == wire.BLOCK_TOOL_USE:
                LINES.tool_use(wire.tool_use_name(block),
                               wire.tool_use_input(block))
        return

    if et == wire.USER:
        for block in wire.message_blocks(ev):
            if wire.block_type(block) == wire.BLOCK_TEXT and mailbox is not None:
                note = mailbox.claim_echo(wire.block_text(block))
                if note is not None:
                    print_note(note)
                continue
            if wire.block_type(block) != wire.BLOCK_TOOL_RESULT:
                continue
            is_err = wire.tool_result_failed(block)
            mark = "✗" if is_err else "✓"
            line = compactline.short(
                wire.tool_result_text(block),
                LINES.budget(compactline.mark_line_head(mark)))
            if line:
                LINES.mark(mark, "red" if is_err else "green", line)
        return

    if et == wire.RESULT:
        global _turn_cost_base
        cost = wire.result_cost(ev)
        if cost is not None:
            # `total_cost_usd` is the SESSION's running total, not this turn's
            # (measured: two trivial turns in one process reported $0.2015 then
            # $0.2204, while their durations were 2243 ms and 1991 ms — the
            # second figure is the first plus $0.019, not a second $0.2). A
            # process emits more than one `result` whenever a note typed late is
            # answered as its own turn, and `report_costs` sums these lines, so
            # each line shows what its turn ADDED.
            cost, _turn_cost_base = cost - _turn_cost_base, cost
        dur = wire.result_duration_ms(ev)
        bits = []
        if dur is not None:
            bits.append(f"{dur / 1000:.1f} c")
        if cost is not None:
            bits.append(f"${cost:.4f}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        if wire.result_failed(ev):
            print_error(f"  ⚠ result: {wire.result_subtype(ev)}{suffix}")
        else:
            print_done(f"  · done{suffix}")
        return


def _render_codex_event(ev: dict, mailbox=None) -> None:
    """Render one normalised Codex event."""
    event_type = wire.event_type(ev)
    item = wire.codex_item(ev)
    item_type = wire.codex_item_type(item)

    if event_type == wire.THREAD_STARTED:
        print(f"  · session started (thread {wire.codex_thread_id(ev)})")
        return

    if event_type == wire.ITEM_COMPLETED and item_type == wire.AGENT_MESSAGE:
        render_markdown_block(wire.codex_message_text(item))
        return

    if event_type == wire.ITEM_COMPLETED and item_type == wire.USER_MESSAGE:
        if mailbox is not None:
            note = mailbox.claim_echo(wire.codex_user_message_text(item))
            if note is not None:
                print_note(note)
        return

    if event_type == wire.ITEM_STARTED and item_type == wire.COMMAND_EXECUTION:
        command = compactline.short(
            wire.codex_command(item),
            LINES.budget(compactline.tool_line_head("Bash")))
        LINES.tool("Bash", command)
        return

    if event_type == wire.ITEM_COMPLETED and item_type == wire.COMMAND_EXECUTION:
        exit_code = wire.codex_exit_code(item)
        mark = "✓" if exit_code in (None, 0) else "✗"
        # The head is measured from what this line will actually print: there is
        # no "exit N: " when the provider reported no code, and no " — " when
        # there is no output to put after it. Measuring both unconditionally
        # left the line eleven columns short of the width it had been given.
        code_head = f"exit {exit_code}: " if exit_code is not None else ""
        separator = (" — " if compactline.collapse(
            wire.codex_output(item)) else "")
        command, output = compactline.fit_two(
            LINES.budget(f"{compactline.mark_line_head(mark)}"
                         f"{code_head}{separator}"),
            wire.codex_command(item),
            wire.codex_output(item))
        detail = f"{code_head}{command}"
        if output:
            detail += f"{separator}{output}"
        LINES.mark(mark, "green" if exit_code in (None, 0) else "red", detail)
        return

    if event_type == wire.ITEM_COMPLETED and item_type == wire.FILE_CHANGE:
        paths = wire.codex_changed_paths(item)
        LINES.tool("Edit", ", ".join(paths) or "file changes applied")
        return

    if event_type == wire.TURN_COMPLETED:
        counts = wire.codex_token_counts(ev)
        bits = []
        if counts is not None:
            tokens_in, cached, tokens_out = counts
            bits.append(f"input {tokens_in}")
            if cached:
                bits.append(f"cached {cached}")
            bits.append(f"output {tokens_out}")
        suffix = f" (tokens: {', '.join(bits)})" if bits else ""
        print_done(f"  · done{suffix}")
        return

    if event_type in (wire.ERROR, wire.TURN_FAILED):
        LINES.fitted("  ⚠ result: ", wire.codex_error(ev), "bold red")


def run_agent_streaming(cmd: list, provider: str, raw: bool,
                        partial: bool = True, prompt: str = "",
                        mailbox=None) -> int:
    """Run one provider CLI, parse its JSONL and render progress live.

    Claude's rate-limit verdict, if streamed, is left in
    ``last_rate_limit_event()``. Codex limits are queried separately through
    its app-server before and after turns.

    `mailbox` is this run's `operator.Mailbox`: while the turn is in flight it
    holds the process's stdin, so a note typed at the console reaches the agent
    mid-iteration. The channel is closed on the turn's `result` event and again
    in the `finally` — closing stdin is what ends a streaming-input session, so
    an unclosed pipe is a run that never returns, and the redundancy is
    deliberate: a crash that swallows `result` must not be able to hang the loop.

    Everything from `start_agent_process` down to `proc.wait()` runs with a
    child process alive, and `wait()` is the only exit that reaps it: the
    console write behind each rendered line, `note_channel`'s close and a
    decoder error on the child's own stream all raise past it. Hence the
    `finally` — `reap_agent_process` is what ends the CLI, and it is the whole
    ending: the `except KeyboardInterrupt` below deliberately no longer calls
    `proc.terminate()` itself, because on Windows that aims at the npm `.cmd`
    shim and leaves the CLI — a GRANDCHILD — running with the terminal's stdout
    in hand.
    """
    global _last_rate_limit_event, _turn_cost_base
    _last_rate_limit_event = None
    _turn_cost_base = 0.0     # a new process starts a new cost total
    spec = provider_spec(provider)
    try:
        proc = start_agent_process(cmd, provider, prompt, projectroot.project_dir())
    except FileNotFoundError:
        print(f"Executable {spec.executable!r} not found. "
              f"Is {spec.display_name} installed and on PATH?")
        sys.exit(2)

    provider_failed = False
    try:
        with note_channel(proc, provider, mailbox) as channel:
            for line in proc.stdout:
                line = line.rstrip("\n")
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    # non-JSON line (e.g. CLI diagnostics) — print it as is
                    print(line)
                    continue
                if not isinstance(ev, dict):
                    # A JSON scalar/array is diagnostic output, not a JSONL event.
                    print(line)
                    continue
                event_type = wire.event_type(ev)
                if (provider == "claude" and event_type == wire.RESULT
                        and wire.result_failed(ev)):
                    provider_failed = True
                if ((provider == "claude" and event_type == wire.RESULT)
                        or (provider == "codex" and event_type in (
                            wire.TURN_COMPLETED, wire.TURN_FAILED))):
                    # Before rendering, so the console cannot take a note for a
                    # turn that has already reported its ending.
                    channel.close()
                if raw:
                    print(line)
                    continue
                if provider == "claude":
                    _render_claude_event(ev, partial, mailbox)
                else:
                    _render_codex_event(ev, mailbox)
                    provider_failed = provider_failed or event_type in (
                        wire.ERROR, wire.TURN_FAILED)
        # Outside the `with`, so the pipe is already closed: waiting on a
        # process whose stdin is still open is the hang this whole seam exists
        # to prevent.
        returncode = proc.wait()
        return 1 if returncode == 0 and provider_failed else returncode
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
        sys.exit(130)
    finally:
        reap_agent_process(proc)


def run_claude_streaming(cmd: list, raw: bool, partial: bool,
                         prompt: str = "", mailbox=None) -> int:
    """Backward-compatible Claude-only wrapper.

    `prompt` is required whenever the live-message transport is on — the argv
    then carries no task, only flags.
    """
    return run_agent_streaming(cmd, "claude", raw, partial, prompt=prompt,
                               mailbox=mailbox)
