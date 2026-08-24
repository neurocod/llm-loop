"""
cyclecore.py - reusable engine behind autonomous Claude/Codex CLI loops.

This module holds everything that is *not* specific to one particular task:
command-line parsing, stream-json rendering with the run's own rate-limit
verdict picked out of that stream, and the generic `run_loop()` that ties it
together. Reading a quota and pausing on it are `usage`/`limits`, and when a run
pushes what it has committed is `gitpush` — both runners apply those and neither
owns them. What the run PRINTS is `console` — with the rotating mirror log, which
is the second copy of every printed line and therefore belongs to the printer;
what `--cost` reads back OUT of that log is still here, because the lines it
parses are emitted here. What is asked of a run from OUTSIDE it —
the `s` key and the `stop` sentinel, the `p` key's hold, and the reason a
runner reports on the way out — is `stopchannel`, a module of its own, because
the parallel runner and a host wrapper speak that vocabulary too and neither
should have to import the sequential loop to do so.

The only thing this engine does **not** decide is *what work to do each
iteration* — that is supplied by a `Driver` (see below), so it drives both:

  * runCycle.py    — a state-machine driver reading products/currentState.md;
  * runTranslate.py — a list driver translating files from products/list.md.

A Driver answers three questions for the loop, via three hooks:

  * next_command()  -> AgentCommand | None
        The command to run this iteration, or None when there is no more work
        and the loop should stop normally. It may raise LoopStop to abort the
        whole run (e.g. an error state that needs a human).
  * on_success(rc)  -> None
        Called after an iteration whose provider CLI exited 0 — the place to record
        progress (mark a file done, advance a cursor, …). Default: no-op.
  * final_summary() -> str | None
        A closing line printed on the way out (e.g. "Final state: …"). Optional.

Everything below is lifted verbatim from the original single-file runCycle.py,
with the few state-specific pieces (which file to read, which prompt to send,
which model to pick) factored out into the Driver protocol.

Claude token-limit handling is driven by the account's real usage figures rather than
guessed from error counts, in two layers:

  * proactive — before each iteration, and again immediately after any non-zero
    Claude exit, the loop asks the Driver's `limit_policy` for the current
    quota percentages and pauses if any watched one is at/over its ceiling. The
    reading lives in usage.py (UsageSource, an HTTP GET of the usage endpoint)
    and the pausing policy in limits.py (LimitPolicy and the ready-made
    SessionLimit / DayNightLimit / WeeklyLimit rules).
  * reactive — every Claude run streams its own rate-limit verdict, which this module
    picks out of the stream it is already parsing (see RateLimitEvent): a
    "rejected" means the wall was hit, and the loop waits that quota out even if
    the proactive reading was unavailable.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, NamedTuple, Optional, Union

from . import (compactline, console, exitlog, operator, projectroot, providers,
               stopchannel, textwidth)
# What the run PRINTS, and the mirror log that is the second copy of it, are
# `console` (see its header for why those are one module). The line helpers are
# imported by name because this module calls them on nearly every path; the
# log's own names are reached through the module instead — `LOG_DIR`, the
# handler and the tee are configured and replaced, and a second binding here
# would be a second address for a test or a wrapper to miss.
from .console import (
    LINES,
    _fmt_clock,
    _fmt_left,
    _fmt_moment,
    _MarkdownStream,
    _render_markdown_block,
    print_done,
    print_error,
    print_note,
    print_percents,
)
# The vocabulary of stopping and pausing is `stopchannel`, its own module,
# because both runners and a host wrapper speak it and none of them should have
# to import the sequential runner to do so.
#
# Reached through the module, never `from .stopchannel import …`, and that is
# mechanics rather than style: `STOP_FILE` moves when --project-dir does, and a
# name imported here would freeze at the launch directory; a stop function
# imported here would be a SECOND address for it, so a test (or a wrapper) that
# replaced `stopchannel.pause_requested` would change what the parallel runner
# does and not what this one does. One address, one thing to patch.
from .providers import (
    build_agent_argv as provider_argv,
    note_channel,
    prompt_on_stdin,
    provider_spec,
    reap_agent_process,
    set_live_messages,
    start_agent_process,
    usage_source_for,
)
# The git-push policy is `gitpush`, its own module, for the same reason as the
# two above: both runners apply it and neither owns it. Imported by name because
# these are the spellings both runners and the host wrappers already use, and
# `GitPushPolicy` is part of the package's public surface (see __init__).
from .gitpush import (
    GIT_PUSH_POLICY,
    GIT_PUSH_SETTING,
    GitPushPolicy,
    git_push,
    git_unpushed_count,
    maybe_git_push,
)
# The length of the window a token-limited run waits out. In `usage`, with the
# rest of what is known about a quota, so the limit rules can use it without
# importing this runner.
from .usage import CLAUDE_SESSION_DURATION

# The usage-limit policy (which quota to gate on, what ceiling to allow,
# when to pause) lives in limits.py / usage.py, chosen per project via a Driver's
# `limit_policy` attribute — see Driver and run_loop.


# The Windows console is often cp1252 — switch output to UTF-8 so we can print Cyrillic.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

# The project root lives in `projectroot`, its own leaf module, and is reached
# through it: `projectroot.project_dir()`. Not re-exported here on purpose —
# this runner is a CONSUMER of the root, not its home, and a `from … import`
# would put a second address on the same fact, which is what the split removed
# (see that module's header for the two mirrors it deleted). The package's front
# door still spells them `llm_loop.project_dir` / `set_project_root` /
# `find_project_root`, so an embedder's address is unchanged.


# Per-run cost accounting parsed straight back out of the mirror log: every run's
# first iteration logs a "=== Iteration 1 ===" header (see run_loop) and every
# successful iteration logs a "done (… c, $…)" line (see _render_event's "result"
# branch). Summing the dollar figures between headers reconstructs per-run spend
# with no extra bookkeeping. The patterns live here, next to the code that emits
# both lines, so they can't drift apart. Note "=== Iteration 1 ===" matches only
# iteration 1 (the "1 ===" boundary rules out "11", "12", …), so each match is a
# genuine run boundary.
_SESSION_RE = re.compile(r"=== Iteration 1 ===")
_COST_RE = re.compile(r"done \(\s*[\d.]+ c,\s*\$([\d.]+)\)")


def report_costs(app_name: str = "runCycle",
                 path: Optional[Union[str, Path]] = None) -> None:
    """Print per-session (per-run) cost totals parsed from the mirror log, then
    exit — the standalone counterpart reached via the --cost flag.

    A "session" is one run of the loop, delimited by its "=== Iteration 1 ==="
    header; within it every "done (… c, $…)" line contributes its dollar cost. We
    print a line per session, a grand total, and how full the log is against the
    rotation limit (LOG_MAX_BYTES). With no `path`, the log is resolved via
    log_file_path(app_name), so --cost reports on the very log this entry point
    writes — under the project root already chosen by --project-dir.

    `path` (the --cost-log flag) names a log this entry point does NOT write:
    a rotated backup (`<app>-<project>.log.1`) or a copy taken elsewhere. It is
    the one case app_name cannot reach, since rotation renames files out from
    under log_file_path.
    """
    path = Path(path) if path else console.log_file_path(app_name)
    # Always name the log we are reading, so an empty report is unambiguous
    # (right file, no data) rather than looking like a silent failure.
    print(f"Reading mirror log: {path}")
    sessions = []  # list of (header, total_cost, count)
    header = None
    total = 0.0
    count = 0

    def flush():
        if header is not None:
            sessions.append((header, total, count))

    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if _SESSION_RE.search(line):
                    flush()
                    header = line.strip()
                    total = 0.0
                    count = 0
                else:
                    m = _COST_RE.search(line)
                    if m and header is not None:
                        total += float(m.group(1))
                        count += 1
    except FileNotFoundError:
        print(f"No mirror log at {path} yet — nothing to report.")
        return
    flush()

    grand = 0.0
    grand_count = 0
    for i, (h, t, c) in enumerate(sessions, 1):
        print(f"Session {i}: {c} costs, ${t:.4f}  | {h}")
        grand += t
        grand_count += c

    print("-" * 60)
    print(f"TOTAL: {len(sessions)} sessions, {grand_count} costs, ${grand:.4f}")
    if not sessions:
        # The log exists but held no run boundaries / cost lines. Point at the
        # likely cause rather than leaving a bare zero.
        print("  (log has no '=== Iteration 1 ===' / '· done (… c, $…)' lines — "
              "no completed billed iterations recorded here)")

    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    limit = console.LOG_MAX_BYTES
    pct = size / limit * 100 if limit else 0.0
    print(f"LOG: {size / 1024 / 1024:.2f} / {limit / 1024 / 1024:.0f} MB "
          f"({pct:.1f}% full, rotates at 100%)")
    print(f"     {path}")


class ConsumedByWrapperAction(argparse.Action):
    """An option this parser only DOCUMENTS — the wrapper reads it out of argv
    itself, before parsing (see Driver.add_cli_options).

    Reaching the parser therefore means the wrapper's own scan did not match
    what was typed. argparse would otherwise accept it (an abbreviation like
    `--grow` for `--grow-kit` is one this parser resolves and that scan does
    not), store it in a namespace nobody reads, and run the DEFAULT mode — a
    flag that appears to work and silently does nothing. Saying so is the whole
    job of this action.

    Declare a value-taking switch with `nargs=1, metavar="N"` so --help shows
    its argument; the default `nargs=0` documents a bare flag.
    """

    def __init__(self, option_strings, dest, nargs=0, **kwargs):
        super().__init__(option_strings, dest, nargs=nargs,
                         default=argparse.SUPPRESS, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # `option_string` is the canonical spelling, not what was typed — which
        # is the point: it names the option the abbreviation resolved to, and
        # the one to type instead.
        parser.error(
            f"{option_string} never reached the argv scan that reads it, "
            f"which runs before this parser. Spell it out in full; "
            f"abbreviations are resolved here and are invisible there. (If it "
            f"WAS spelled in full, that scan and this help have drifted "
            f"apart.)")


def parse_args(argv=None, *, prog: str = "runCycle.py",
               description: Optional[str] = None,
               extra_options: Optional[Callable[[argparse.ArgumentParser],
                                                None]] = None
               ) -> argparse.Namespace:
    """Command-line interface shared by every entry point. Every long option has
    a single-letter alias.

    `prog`/`description` let each entry script label its own --help text while
    reusing the exact same option set (so there is no duplicated argument code).

    `extra_options` is handed the parser once the shared options are on it, and
    is how a wrapper documents the flags IT consumes before this parser ever
    runs (a mode switch that decides which parser to use cannot be one of this
    parser's options). Passing the parser rather than a block of help text is
    what keeps such a flag formatted, aligned and grouped like every other
    option — and lets the hook add a real, parsed option when the flag is not
    one the wrapper strips.
    """
    p = argparse.ArgumentParser(
        prog=prog,
        description=description or "Autonomous loop driving an LLM CLI.",
    )
    # Most option names are kept in sync with continuous_claude.py (kebab-case,
    # and --max-runs for the iteration cap). The former spellings (--max,
    # --startIn) stay on as accepted aliases so existing invocations keep
    # working.
    p.add_argument("-m", "--max-runs", "--max", dest="max", type=int, default=None,
                   metavar="N",
                   help="stop after N iterations (default: run forever); "
                        "--max is a deprecated alias")
    p.add_argument("--codex", action="store_const", const="codex",
                   dest="provider", default=None,
                   help="run Codex CLI instead of the Driver's default provider")
    p.add_argument("-d", "--dry-run", action="store_true",
                   help="only print the commands, don't run the LLM CLI")
    p.add_argument("-c", "--cost", action="store_true",
                   help="print per-session cost totals from the mirror log and "
                        "exit (no loop is run)")
    p.add_argument("--cost-log", dest="cost_log", metavar="LOG",
                   help="report on this log file instead of this entry point's "
                        "own — a rotated backup (<app>-<project>.log.1) or a "
                        "copy; implies --cost")
    # No -r short flag: -r is --review-prompt in continuous_claude.py, so it is
    # left free here rather than reused for --raw.
    p.add_argument("--raw", action="store_true",
                   help="print raw JSON events (for debugging)")
    p.add_argument("-s", "--start-in", "--startIn", dest="start_in", metavar="DURATION",
                   help="wait this long before starting the loop, e.g. 29m, 1h30m")
    p.add_argument("-g", "--git-push", dest="git_push",
                   choices=[pol.value for pol in GitPushPolicy],
                   default=GIT_PUSH_POLICY.value,
                   help="when to `git push` at the start of each iteration: "
                        "none | after_new_commits | each_hour "
                        f"(default: {GIT_PUSH_POLICY.value})")
    p.add_argument("-C", "--project-dir", dest="project_dir", metavar="DIR",
                   default=None,
                   help="project root: cwd for git/provider CLI, base for the stop "
                        "file and the Driver's relative paths "
                        "(default: the current working directory)")
    # No short alias: this is a rescue hatch for an odd terminal, not a knob to
    # reach for. LLM_LOOP_STATUSLINE=0 does the same without editing a
    # command line, and no TTY disables it by itself.
    p.add_argument("--no-statusline", dest="no_statusline", action="store_true",
                   help="do not pin the interactive status rows at the bottom of "
                        "the terminal (same as LLM_LOOP_STATUSLINE=0)")
    # Same shape of rescue hatch as --no-statusline, and for the same reason: it
    # turns off a transport, not a feature. A note typed with the `m` key still
    # reaches the agent — with the next iteration's prompt instead of the one
    # already running.
    p.add_argument("--no-live-messages", dest="no_live_messages",
                   action="store_true",
                   help="do not keep the agent's stdin open for notes typed "
                        "during an iteration; they wait for the next prompt "
                        f"instead (same as {providers.LIVE_MESSAGES_ENV}=0)")
    if extra_options is not None:
        extra_options(p)
    return p.parse_args(argv)


def parse_duration(text: str) -> float:
    """Parse a duration like '29m', '1h', '90s', '1h30m' into seconds.

    A bare number is treated as minutes ('29' == '29m'). Raises ValueError on
    anything it can't make sense of.
    """
    text = text.strip().lower()
    if not text:
        raise ValueError("empty duration")
    if text.isdigit():  # bare number — minutes
        return int(text) * 60

    units = {"h": 3600, "m": 60, "s": 1}
    total = 0.0
    matched = False
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)\s*([hms])", text):
        total += float(value) * units[unit]
        matched = True
    if not matched:
        raise ValueError(f"cannot parse duration: {text!r}")
    return total


class AgentCommand(NamedTuple):
    """One unit of work for the loop: the prompt to send, the model to use, and a
    short label shown in the iteration header. Drivers build these in
    next_command(); build_agent_argv() turns one into the provider's full argv.

    An empty `model` means "no --model flag": the selected provider then uses
    its configured default, which is the common case.
    """
    prompt: str
    model: str = ""
    label: str = ""
    provider: str = ""
    sandbox_mode: str = ""


ClaudeCommand = AgentCommand


def build_agent_argv(command: AgentCommand, provider: Optional[str] = None) -> list:
    """Full provider command line for one unit of work."""
    provider = provider or command.provider or "claude"
    return provider_argv(command, provider, projectroot.project_dir())


def build_claude_argv(command: ClaudeCommand) -> list:
    """Full `claude` command line for one ClaudeCommand.

    The flags are identical for every task; only the prompt and the model vary,
    so this is the single place those two are spliced into the otherwise fixed
    argv (stream-json + partial messages so the loop can render work live). An
    empty `command.model` omits --model entirely, letting the CLI pick its own
    configured default.
    """
    return build_agent_argv(command, "claude")


def report_undelivered_notes(mailbox) -> None:
    """Say so when the run ends holding notes nobody ever saw.

    A queued note promises "the next iteration" — and there is not always a next
    one (the list drained, --max-runs, a stop request, a quota pause that
    outlasts the run). Whoever typed it during the last iteration would
    otherwise have no way to learn it was never delivered, and the text itself
    is printed so it can be pasted into the next run rather than retyped.
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


# Streaming print state: the single content-block index text is currently flowing
# into (assistant replies stream one text block at a time), plus its live renderer.
_active_text_index = None
_md_stream = _MarkdownStream()

# The session cost already reported by earlier `result` events of the process
# now streaming. Reset by run_agent_streaming; see the result branch for why the
# figure has to be differenced at all.
_turn_cost_base = 0.0


# --- the free half of the limit machinery: the run's own rate-limit events -----
#
# Every `claude` run streams a line of its own, built from the ratelimit headers
# the API already returned to it:
#
#   {"type":"rate_limit_event","rate_limit_info":{"status":"allowed",
#     "resetsAt":1786807200,"rateLimitType":"five_hour", …}}
#
# Reading it costs nothing — that stream is parsed anyway — and unlike a queried
# figure it cannot be stale or unavailable: it is the wire's own verdict on the
# request that just went out. It carries no percentage, so it cannot drive a
# ceiling; it is the backstop *under* the proactive check in limits.py. "rejected"
# means this run hit the wall, and `resetsAt` says when that quota comes back.

# Quota id -> the name the CLI itself uses for it in limit messages.
RATE_LIMIT_LABELS = {
    "five_hour": "session limit",
    "seven_day": "weekly limit",
    "seven_day_opus": "Opus limit",
    "seven_day_sonnet": "Sonnet limit",
    "seven_day_overage_included": "usage-credit limit",
}


class RateLimitEvent(NamedTuple):
    """One rate_limit_event: which quota it is about, how it stands, when it resets.

    `status` is the API's own verdict — "allowed", "allowed_warning" (close to
    the wall) or "rejected" (refused). `resets_at` is epoch seconds, or None when
    the event carried no reset time.
    """
    status: str
    limit_type: str
    resets_at: Optional[float]

    @property
    def label(self) -> str:
        return RATE_LIMIT_LABELS.get(self.limit_type, self.limit_type or "limit")

    def describe(self) -> str:
        when = f", resets {_fmt_moment(self.resets_at)}" if self.resets_at else ""
        return f"{self.label} {self.status}{when}"


def rate_limit_event_from(ev: dict) -> Optional[RateLimitEvent]:
    """The RateLimitEvent carried by a stream-json event, or None if it isn't one."""
    if ev.get("type") != "rate_limit_event":
        return None
    info = ev.get("rate_limit_info") or {}
    resets = info.get("resetsAt")
    return RateLimitEvent(
        status=str(info.get("status") or "unknown"),
        limit_type=str(info.get("rateLimitType") or ""),
        resets_at=float(resets) if isinstance(resets, (int, float)) else None,
    )


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
    et = ev.get("type")

    if et == "system" and ev.get("subtype") == "init":
        model = ev.get("model", "?")
        print(f"  · session started (model {model})")
        return

    if et == "rate_limit_event":
        global _last_rate_limit_event
        _last_rate_limit_event = rate_limit_event_from(ev)
        # "allowed" is the normal case and would be one more line per run; the
        # two states that mean something get shown.
        if _last_rate_limit_event.status != "allowed":
            print_error(f"  ⚠ rate limit: {_last_rate_limit_event.describe()}")
        return

    # Streaming deltas (Anthropic streaming events, wrapped in stream_event).
    if et == "stream_event":
        global _active_text_index
        inner = ev.get("event", {})
        it = inner.get("type")
        if it == "content_block_start":
            if inner.get("content_block", {}).get("type") == "text":
                _active_text_index = inner.get("index")
                _md_stream.start()
        elif it == "content_block_delta":
            d = inner.get("delta", {})
            if d.get("type") == "text_delta" and inner.get("index") == _active_text_index:
                _md_stream.feed(d.get("text", ""))
        elif it == "content_block_stop":
            if inner.get("index") == _active_text_index:
                _md_stream.stop()  # finalize the Markdown render / line
                _active_text_index = None
        return

    if et == "assistant":
        for block in ev.get("message", {}).get("content", []):
            bt = block.get("type")
            if bt == "text":
                if partial:
                    continue  # already printed streaming from the deltas
                _render_markdown_block(block.get("text", ""))
            elif bt == "tool_use":
                LINES.tool_use(block.get("name", "?"),
                               block.get("input", {}) or {})
        return

    if et == "user":
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") == "text" and mailbox is not None:
                note = mailbox.claim_echo(block.get("text", ""))
                if note is not None:
                    print_note(note)
                continue
            if block.get("type") != "tool_result":
                continue
            content = block.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            is_err = block.get("is_error")
            mark = "✗" if is_err else "✓"
            line = compactline.short(
                content, LINES.budget(compactline.mark_line_head(mark)))
            if line:
                LINES.mark(mark, "red" if is_err else "green", line)
        return

    if et == "result":
        global _turn_cost_base
        cost = ev.get("total_cost_usd")
        if cost is not None:
            # `total_cost_usd` is the SESSION's running total, not this turn's
            # (measured: two trivial turns in one process reported $0.2015 then
            # $0.2204, while their durations were 2243 ms and 1991 ms — the
            # second figure is the first plus $0.019, not a second $0.2). A
            # process emits more than one `result` whenever a note typed late is
            # answered as its own turn, and `report_costs` sums these lines, so
            # each line shows what its turn ADDED.
            cost, _turn_cost_base = cost - _turn_cost_base, cost
        dur = ev.get("duration_ms")
        bits = []
        if dur is not None:
            bits.append(f"{dur / 1000:.1f} c")
        if cost is not None:
            bits.append(f"${cost:.4f}")
        suffix = f" ({', '.join(bits)})" if bits else ""
        if ev.get("subtype") != "success" or ev.get("is_error"):
            print_error(f"  ⚠ result: {ev.get('subtype', 'error')}{suffix}")
        else:
            print_done(f"  · done{suffix}")
        return


def _render_codex_event(ev: dict) -> None:
    """Render one event from ``codex exec --json``."""
    event_type = ev.get("type")
    item = ev.get("item") or {}
    item_type = item.get("type")

    if event_type == "thread.started":
        thread_id = ev.get("thread_id") or "?"
        print(f"  · session started (thread {thread_id})")
        return

    if event_type == "item.completed" and item_type == "agent_message":
        _render_markdown_block(str(item.get("text") or ""))
        return

    if event_type == "item.started" and item_type == "command_execution":
        command = compactline.short(
            compactline.undouble_backslashes(str(item.get("command", ""))),
            LINES.budget(compactline.tool_line_head("Bash")))
        LINES.tool("Bash", command)
        return

    if event_type == "item.completed" and item_type == "command_execution":
        exit_code = item.get("exit_code")
        mark = "✓" if exit_code in (None, 0) else "✗"
        # The head is measured from what this line will actually print: there is
        # no "exit N: " when the provider reported no code, and no " — " when
        # there is no output to put after it. Measuring both unconditionally
        # left the line eleven columns short of the width it had been given.
        code_head = f"exit {exit_code}: " if exit_code is not None else ""
        separator = (" — " if compactline.collapse(
            item.get("aggregated_output", "")) else "")
        command, output = compactline.fit_two(
            LINES.budget(f"{compactline.mark_line_head(mark)}"
                         f"{code_head}{separator}"),
            compactline.undouble_backslashes(str(item.get("command", ""))),
            item.get("aggregated_output", ""))
        detail = f"{code_head}{command}"
        if output:
            detail += f"{separator}{output}"
        LINES.mark(mark, "green" if exit_code in (None, 0) else "red", detail)
        return

    if event_type == "item.completed" and item_type == "file_change":
        changes = item.get("changes") or []
        paths = [str(change.get("path")) for change in changes
                 if isinstance(change, dict) and change.get("path")]
        LINES.tool("Edit", ", ".join(paths) or "file changes applied")
        return

    if event_type == "turn.completed":
        usage = ev.get("usage") or {}
        bits = []
        if usage:
            bits.append(f"input {usage.get('input_tokens', 0)}")
            cached = usage.get("cached_input_tokens", 0)
            if cached:
                bits.append(f"cached {cached}")
            bits.append(f"output {usage.get('output_tokens', 0)}")
        suffix = f" (tokens: {', '.join(bits)})" if bits else ""
        print_done(f"  · done{suffix}")
        return

    if event_type in ("error", "turn.failed"):
        error = ev.get("message") or ev.get("error") or item.get("error") or ev
        LINES.fitted("  ⚠ result: ", error, "bold red")


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
                if raw:
                    print(line)
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
                if provider == "claude":
                    # Before rendering, so the console cannot take a note for a
                    # turn that has already reported its result.
                    if ev.get("type") == "result":
                        channel.close()
                    _render_claude_event(ev, partial, mailbox)
                else:
                    _render_codex_event(ev)
                    provider_failed = provider_failed or ev.get("type") in (
                        "error", "turn.failed")
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


def _count_down_to(target_ts: float, should_stop=None) -> bool:
    """Idle until `target_ts`, printing what is left about once a minute.

    The body every timed wait shares; they differ only in the lines they print
    around it, which is why this holds none of them. Returns True when it left
    early because `should_stop()` asked it to. Ctrl+C ends the process from
    here, as it always did.
    """
    try:
        while True:
            now = time.time()
            remaining = target_ts - now
            if remaining <= 0:
                return False
            print(f"    … {_fmt_left(remaining)} left (now {_fmt_clock(now)})",
                  flush=True)
            if stopchannel.sleep_unless(min(remaining, 60), should_stop):
                return True
    except KeyboardInterrupt:
        print("\nWait interrupted by user (Ctrl+C).")
        sys.exit(130)


def wait_until(target_ts: float, reason: str = None, should_stop=None) -> bool:
    """Sleep until wall-clock time reaches target_ts, printing a periodic countdown.

    Used after a probable token-limit error, or once the LimitPolicy decides the
    account's real usage figures leave no room: we idle until the 5-hour session
    window should have refreshed. `reason` overrides the default opening line.
    Ctrl+C interrupts the wait and stops the script.

    `should_stop` is the run's stop channels (see `stopchannel.sleep_unless`):
    a hold that can last hours must end the moment a human asks it to, and
    returns True when that is why it returned. Without one the wait runs to
    `target_ts` as before.
    """
    if reason is None:
        reason = ("Looks like the token limit is exhausted. Waiting until "
                  f"{_fmt_clock(target_ts)} (until the 5-hour session window refreshes)…")
    print(f"  ⏳ {reason}")
    if _count_down_to(target_ts, should_stop):
        print("  ⏹ Stop requested — leaving the wait.")
        return True
    print("  ▶ The session window should have refreshed — continuing the loop.")
    return False


def wait_before_start(spec: str) -> None:
    """Idle for the duration given by --start-in before the loop begins.

    Lets you launch the script and walk away; work kicks off after the delay.
    Ctrl+C interrupts the wait and stops the script.

    The only wait that runs its clock out (no `should_stop`), because there is
    nothing here to stop yet: this is before the status line exists, so there is
    no `s` key to press, and a sentinel that appears meanwhile was waited out
    just above and is honoured at the first iteration boundary anyway.
    """
    try:
        seconds = parse_duration(spec)
    except ValueError as e:
        print(f"Invalid --start-in value {spec!r}: {e}")
        sys.exit(2)
    if seconds <= 0:
        return
    target_ts = time.time() + seconds
    print(f"  ⏳ --start-in {spec}: waiting until {_fmt_clock(target_ts)} before starting…")
    _count_down_to(target_ts)
    print("  ▶ Starting the loop.")


class RunSettings:
    """The script's own knobs, held in one MUTABLE object the loop re-reads.

    Plain locals froze these at startup, which made "edit the limits while the
    run goes" (the status line's `l` key) impossible without touching the loop
    body. The loop now reads this object at every iteration boundary, so moving
    `--max-runs` from 40 to 60 mid-run takes effect at the next boundary and
    nothing else has to change.
    """

    def __init__(self, *, max_runs: Optional[int] = None,
                 git_push: "GitPushPolicy" = None):
        self.max_runs = max_runs
        self.git_push = git_push or GitPushPolicy(GIT_PUSH_POLICY)


def _script_settings(run_settings: RunSettings, statusline) -> "SettingsRegistry":
    """The script's knobs as a SettingsRegistry — the display AND edit surface.

    One registry is the single source of truth for both the pinned row
    (`status_entries()`) and the reproducing command line (`overrides()`), so a
    figure on screen can never disagree with the flag that would reproduce it;
    the flags are checked against `cmdline.FLAG_ALIASES` at registration.
    `statusline` is passed in because `run_loop` imports it lazily and hands it
    down — see the note there. It was once a cycle break, and is no longer one.
    """
    registry = statusline.SettingsRegistry()
    registry.add(statusline.NumberSetting(
        "max-runs", "--max-runs",
        lambda: run_settings.max_runs,
        lambda value: setattr(run_settings, "max_runs",
                              None if value is None else int(value)),
        minimum=1,
        # Editable and reproducible, but not a field of its own: the counter
        # already ends in this number (`iter 11/40`), or in the list's size when
        # that is the smaller of the two — see InvocationProgress.summary_fields.
        # Off the row it also stops printing `max-runs off` for every run that
        # never set one.
        show_in_status=False))
    registry.add(statusline.Setting(
        GIT_PUSH_SETTING, "--git-push",
        lambda: run_settings.git_push.value,
        lambda value: setattr(run_settings, "git_push", GitPushPolicy(value))))
    return registry


class LoopStop(Exception):
    """Raised by a Driver to abort the whole run (not a normal completion).

    `exit_code` is the process exit status: non-zero for an error stop that needs
    a human (the loop sys.exit()s immediately, skipping the final push), 0 for a
    clean stop. `message` is printed before exiting.
    """

    def __init__(self, message: str, exit_code: int = 0):
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


class Driver:
    """What the generic loop needs from a task. Subclass and override.

    A Driver is customised two ways, both declarative:

      * class attributes for the labels the entry points use — ``app_name``
        (names the rotating mirror log) and ``prog`` (the --help program name)
        both default to None and are then derived from the invoked script's
        filename (runTranslate.py -> "runTranslate" / "runTranslate.py"), so a
        typical wrapper need not set them at all; ``description`` (the --help
        description) is free prose, so set it if you want one; ``limit_policy``
        (a limits.LimitPolicy) picks the usage-limit specialisation, defaulting
        to a day/night session rule when unset; ``sandbox_mode`` selects an
        explicit Codex sandbox for this driver's commands. Override any of them
        on your subclass to pin an explicit value.
      * methods for behaviour — ``next_command()``, ``model()``, ``on_success()``,
        ``final_summary()``. Override the ones you need; the rest keep their
        default.

    The loop owns all the scaffolding (stop file, git push, usage limits,
    --max-runs, streaming render); the Driver only decides *what work to
    do*. A project wrapper is then just::

        class MyDriver(StateFileDriver):
            state_file = "products/currentState.md"
            app_name   = "runCycle"

        if __name__ == "__main__":
            MyDriver.main()

    ``main()`` parses the shared CLI and hands a fresh instance to run_loop(); the
    subclass never touches parse_args / run_loop by hand.
    """

    # --- labels used by the entry points (override on the subclass) -----------
    # None => derive from the invoked script's filename (see resolved_app_name /
    # resolved_prog); set an explicit string to override.
    app_name: Optional[str] = None      # names the rotating mirror log file
    prog: Optional[str] = None          # --help program name
    description: Optional[str] = None   # --help description (None = generic)
    provider: str = "claude"            # may be overridden by --codex
    # Optional explicit Codex sandbox for every command this driver builds.
    # Claude ignores it. Keeping this on the driver lets one trusted workflow
    # opt into a broader boundary without changing every shared-loop invocation.
    sandbox_mode: str = ""

    # --- usage-limit specialisation (declarative, like the labels above) ------
    # A limits.LimitPolicy picking which quota(s) to gate on and at what
    # ceiling; None => the engine's default (a day/night session rule, see
    # limits.default_policy). Set it as a class attribute to specialise, e.g.
    #   limit_policy = LimitPolicy([SessionLimit(80)])            # flat session
    #   limit_policy = LimitPolicy([WeeklyLimit(90)])             # weekly cap
    #   limit_policy = LimitPolicy([DayNightLimit(), WeeklyLimit(90)])  # composite
    # LimitPolicy/rules are stateless, so a shared default instance is safe here.
    limit_policy = None

    @classmethod
    def resolved_app_name(cls) -> str:
        """The mirror-log label: ``app_name`` if set, else the invoked script's
        filename stem (runTranslate.py -> "runTranslate"). Deriving it keeps each
        entry point on its own log without the wrapper having to spell it out; the
        fallback covers odd argv[0] values (e.g. ``-c``)."""
        if cls.app_name:
            return cls.app_name
        return Path(sys.argv[0]).stem or "runCycle"

    @classmethod
    def resolved_prog(cls) -> str:
        """The --help program name: ``prog`` if set, else the invoked script's
        basename (runTranslate.py -> "runTranslate.py")."""
        if cls.prog:
            return cls.prog
        return os.path.basename(sys.argv[0]) or "runCycle.py"

    @classmethod
    def add_cli_options(cls, parser: argparse.ArgumentParser) -> None:
        """Add this wrapper's own options to the shared --help. Default: none.

        Called by main() and main_parallel() with the parser that is about to
        run, so an entry point can document (or genuinely add) options the
        engine knows nothing about. The usual case is a MODE switch — one the
        wrapper must read out of argv itself, because it decides which of the
        two parsers runs at all, and which therefore can never be a plain option
        of either. Undocumented, such a flag exists only in prose, and `--help`
        answers "there is no such option" to a user who is looking straight at
        the one they want.

        Both entry points call it, so a flag spelled the same in both modes is
        documented in both --helps from one override. Build the option strings
        from the same constants the wrapper's argv scan uses; a second spelling
        typed out here is a spelling that can drift.
        """

    def next_command(self) -> Optional[AgentCommand]:
        """The command to run this iteration, or None when work is exhausted and
        the loop should stop normally. May raise LoopStop to abort the run."""
        raise NotImplementedError

    def model(self) -> str:
        """The selected provider's model for this iteration.

        Called by next_command() implementations to fill in AgentCommand.model.
        The default returns "" — no --model flag, so the provider CLI uses its
        own configured model. Override this (the single model knob) to pin a
        specific model, pick a cheaper/faster one for mechanical work (e.g. a
        list driver translating files needs less than the main state machine), or
        vary the model per iteration (read whatever state you like inside).
        """
        return ""

    def pending_total(self) -> Optional[int]:
        """How many units of work are still waiting, or None when unknowable.

        This is the summary row's denominator, and it is asked of the driver
        because only the driver knows what its queue is: a list file's pending
        lines, a folder of requests, rows in a table. Re-read on every call (not
        cached) — the first answer is latched as the invocation's baseline and
        every later one moves the counter, so a queue that grows or shrinks under
        the run stays honestly described.

        None means "no total to report": the row then counts bare iterations,
        which is the right answer for a state machine that is meant to run
        forever, and the wrong one for anything with a finish line — a run whose
        row reads `iter 1` with no `/N` is usually a driver that forgot this.

        Report it in the unit an ITERATION works through, where you can: the row
        clamps the total against `--max-runs`, which counts iterations, so a
        driver whose one iteration clears several items (the kit-promotion pass
        empties its whole requests folder in one) reads `iter 3/3` under
        `--max-runs 3` while a dozen items are still waiting. Only the display
        is affected — no cap, gate or queue decision reads this — and only when
        such a driver is given a cap, which is why the honest count of what is
        left is still the better answer for it.
        """
        return None

    def on_success(self, returncode: int) -> None:
        """Called after an iteration whose provider CLI exited 0 — record progress
        here (mark a file done, advance a cursor). Default: nothing to do."""

    def final_summary(self) -> Optional[str]:
        """An optional closing line printed on the way out (after the final
        git push). Return None for no summary."""
        return None

    @classmethod
    def main(cls, argv=None) -> stopchannel.RunResult:
        """Parse the shared CLI and run the sequential loop over a fresh instance.

        This is the whole body of a project wrapper: subclass, override the
        methods you need, then ``if __name__ == "__main__": MyDriver.main()``.
        ``prog`` labels the --help text and ``app_name`` names the log; both are
        taken from the (sub)class or derived from the script filename when unset,
        and ``description`` is the (optional) --help blurb.
        """
        args = parse_args(argv, prog=cls.resolved_prog(),
                          description=cls.description,
                          extra_options=cls.add_cli_options)
        return run_loop(cls(), args, app_name=cls.resolved_app_name())


@stopchannel.stop_file_lifecycle()
def run_loop(driver: Driver, args: argparse.Namespace,
             app_name: str = "runCycle", *, setup_logging: bool = True,
             wait_on_start: bool = True, progress=None) -> stopchannel.RunResult:
    """Drive the selected provider per `driver`, with all shared lifecycle
    machinery. This is the former runCycle.main(), generalised: the only thing
    that changed is that "read currentState.md and pick a prompt" became
    `driver.next_command()`, and the closing "Final state" line became
    `driver.final_summary()`.

    `progress` is the whole invocation's InvocationProgress, for a wrapper that
    makes several runner calls in one process (see run_parallel); left None, this
    call is the invocation and owns its own figures.
    """
    # The usage-limit query/parse (UsageSource) and pausing policy (LimitPolicy)
    # live in their own modules. The import is local for history, not necessity:
    # it broke a cycle until the git-push policy and the session-window constant
    # moved out of here, and neither `limits` nor `statusline` imports this
    # module any more. Kept local because hoisting it changes what a bare
    # `import llm_loop.cyclecore` drags in, which is a different question.
    from . import limits, statusline

    owns_progress = progress is None
    if owns_progress:
        progress = statusline.InvocationProgress(max_items=args.max)

    provider = getattr(args, "provider", None) or driver.provider
    spec = provider_spec(provider)
    driver.provider = provider
    # The live knobs (see RunSettings): read at each iteration boundary, never
    # snapshotted into locals, so the status line's editor can move them mid-run.
    run_settings = RunSettings(
        max_runs=args.max,                          # None = no limit
        git_push=GitPushPolicy(args.git_push))      # when to `git push` each iteration
    # When a finite iteration cap is given (-m/--max-runs) the run is short and
    # bounded on purpose, so the usage-limit machinery (the LimitPolicy
    # pause-on-limit logic) is skipped — we just run the requested iterations
    # without ever waiting out a window. Decided once, from the value the run was
    # LAUNCHED with: it also governs whether this run talks to the usage endpoint
    # at all, which is a property of the invocation, not of the current cap.
    ignore_usage_limits = (args.max is not None
                           or not spec.supports_usage_limits)
    dry_run = args.dry_run
    raw = args.raw
    start_in = args.start_in      # e.g. "29m" — delay before the loop starts
    # Decided per invocation, before the first argv is built: the transport is
    # what --no-live-messages turns off, and both the argv and the process's
    # stdin have to agree about it. Set in BOTH directions — a wrapper that
    # calls two runners in one process (see runGenerateModels' periodic mode)
    # would otherwise have the first `--no-live-messages` phase decide the
    # transport for every phase after it.
    set_live_messages(not getattr(args, "no_live_messages", False))

    # Anchor every project-relative operation (git/provider cwd, the stop file, the
    # log name, the Driver's paths) to the chosen root before anything reads it.
    projectroot.set_project_root(getattr(args, "project_dir", None))

    # --cost: report per-run spend from the mirror log and exit, without touching
    # the loop, the tee, git, or the usage gate. Done here (after the root is
    # set, so the log path resolves) rather than in the loop body proper.
    # --cost-log implies --cost: naming a log to read and getting a loop run
    # instead would be a silent misfire, and there is nothing else it could mean.
    cost_log = getattr(args, "cost_log", None)
    if getattr(args, "cost", False) or cost_log:
        report_costs(app_name, cost_log)
        return stopchannel.RunResult(stopchannel.RunStopReason.NO_WORK)

    # Mirror all screen output into a rotating log file under the home dir —
    # except for a dry run, which is a preview and not a run: its output would
    # otherwise displace real runs' records out of the shared rotating log (a
    # preview once pushed ~26 MB through it, and the failure it was launched to
    # explain rotated off the end). Said on screen so the missing log is visible
    # rather than mysterious.
    if setup_logging and not dry_run:
        logger = console._setup_file_logging(app_name)
        sys.stdout = console._TeeToLog(sys.stdout, logger)
        sys.stderr = console._TeeToLog(sys.stderr, logger)
    if not dry_run:
        # After the tee, so the report of a run that vanished lands in the very
        # log whose abrupt end it explains. Idempotent per process: the periodic
        # wrapper calls this runner repeatedly and keeps one record.
        exitlog.begin(app_name, console.LOG_DIR,
                      os.path.basename(projectroot.project_dir()))
    print(f"  · project root: {projectroot.project_dir()}")
    if dry_run:
        print(f"  · dry run: nothing is mirrored to "
              f"{console.log_file_path(app_name)}")
    else:
        print(f"  · logging to {console.log_file_path(app_name)}")
    print(f"  · provider: {spec.display_name}")
    if not console._RICH_AVAILABLE:
        print("  · Markdown rendering is off (the 'rich' library is missing). "
              "Enable it with:")
        print(f"      {sys.executable} -m pip install rich")

    # A stop request pending from another run: wait it out rather than consume
    # it, so this launch starts on a clean sentinel instead of stopping on its
    # first iteration boundary. Before --start-in: the point is to begin as soon
    # as the brake is off, not to burn the delay while it is still on.
    if not dry_run and wait_on_start:
        stopchannel.wait_for_stop_file_clear()

    if start_in and not dry_run:
        wait_before_start(start_in)

    session_start = time.time()   # start of the current 5-hour session window
    consecutive_errors = 0        # reset to 0 after any successful iteration
    usage_source = usage_source_for(provider)
    limit_policy = None
    if usage_source is not None:
        limit_policy = driver.limit_policy or limits.default_policy(provider)
    last_git_push = 0.0           # epoch time of the last `git push` (0 = never)
    print(f"  · git push policy: {run_settings.git_push.value}")
    if ignore_usage_limits:
        print(f"  · usage limit policy: disabled (bounded run, "
              f"--max {run_settings.max_runs})")
    else:
        print_percents(f"  · usage limit policy: {limit_policy.describe()}")

    # Bookend the run with a usage snapshot (the policy's watched quotas) so each
    # run records where it started; the matching end-of-run snapshot is below.
    if not dry_run and usage_source is not None:
        limit_policy.log_snapshot(usage_source, "at start (iteration 1)")

    # The pinned status area. A Job is the unit of display in both runners, so
    # the sequential loop is a run with exactly one Job — no branch anywhere in
    # the status line separates it from `-j N`. Disabled it is a Null object, so
    # every call below stays a no-op and the loop behaves exactly as before.
    settings = _script_settings(run_settings, statusline)
    # A driver that knows how much work it has is the source of truth for it:
    # the summary row then counts items FINISHED out of that total, not
    # iterations, so a retried item is not progress and a preflight that strikes
    # finished ones is. Asked of the driver (Driver.pending_total) rather than
    # read off a list file, so a queue that is not a list — the kit-promotion
    # pass draining its requests folder — gets a denominator too instead of
    # counting bare iterations into a row that reads "iter 1" forever.
    tracked_total = driver.pending_total()
    if tracked_total is not None:
        progress.track_total(tracked_total)
    # One mailbox for the whole run: the console writes to it, the loop below
    # empties it into the next prompt, and run_agent_streaming lends it the
    # running turn's stdin. A dry run gets none — there is no agent to talk to.
    mailbox = None if dry_run else operator.Mailbox()
    app = statusline.StatusApp(
        # From the invocation's pool: a wrapper's next runner call resumes this
        # row instead of starting a fresh Job at iteration 1.
        status=statusline.LoopStatus(jobs=progress.jobs(1)),
        settings=settings,
        messages=mailbox,
        enabled=not dry_run and not getattr(args, "no_statusline", False))
    app.update(
        provider=provider,
        **progress.summary_fields(),
        # Only a list driver has a pick order; read it defensively so any other
        # Driver simply reports no `rand` marker.
        random_order=str(getattr(driver, "pick_order", "")) == "random",
        script_limits=settings.status_entries(),
    )

    iteration = 0
    completed = 0
    paused_since = 0.0            # start of the pause being held (0.0 = none)
    stop_reason = stopchannel.RunStopReason.NO_WORK
    stop_file_noted = False       # dry-run: report the sentinel once, not per iteration
    dry_run_prompt_shown = False  # dry-run: show job 1's prompt once, not per pass

    def stop_pending() -> bool:
        """Is either stop channel asking for this run right now?

        Handed to every hold that can outlast an iteration (the usage gate, the
        post-refusal wait). It only reports — the loop head is the single place
        that decides what a request means, cancel grace included.
        """
        return stopchannel.pending_stop(app) is not None

    with app:
        if usage_source is not None:
            # Inside `with`, not before it: push_quotas is silent until start()
            # has marked the app enabled, so priming it earlier left requirement
            # #1 — the provider's live figures — blank until the first limit
            # check (i.e. for the whole run when the checks are skipped). The
            # reading itself is already paid for by the start-of-run snapshot
            # above, so this costs no round-trip.
            statusline.push_quotas(app, usage_source, limit_policy)
            if not ignore_usage_limits:
                # Only for a run that is allowed to talk to the usage endpoint at
                # all: the poll forces a FRESH reading (cache_value=False), so on
                # a bounded `-m N` run — whose whole point is not to touch the
                # usage machinery — it would be an unsolicited HTTP GET every
                # interval (a codex app-server call every 300 s).
                app.add_service(statusline.QuotaRefresher(
                    app, usage_source, limit_policy, provider=provider))
        while True:
            # The caps are read LIVE (see RunSettings) and republished here, so an
            # edit made while the run is going takes effect at this boundary and
            # is what the pinned row shows. Under a wrapper the invocation's cap
            # is the wrapper's to set — this call's cap only sizes one batch.
            if owns_progress:
                progress.max_items = run_settings.max_runs
            app.update(**progress.summary_fields(),
                       script_limits=settings.status_entries())
            pending = stopchannel.pending_stop(app)
            if pending is stopchannel.StopSource.FILE and dry_run:
                # The sentinel is removed only after the outer application has
                # finished cleanup. A dry run must not claim it. `-d` is routinely
                # used to preview commands while a real loop is running — and that is
                # exactly when a
                # stop request is pending — so removing it here would silently cancel
                # someone else's stop, and the loop it was meant to halt would run on.
                # Report it and leave it for whoever it was written for. (There is
                # no key branch to write here: a dry run's status line is disabled,
                # so `s` is never even read.)
                if not stop_file_noted:
                    print("Stop file present — a real run would have waited for "
                          "it at startup, and stops here if it appears mid-run. "
                          "Left in place (a dry run never consumes it).")
                    stop_file_noted = True
            elif pending is not None and stopchannel.confirm_stop_request(app):
                # Reason first, line second: `commit_stop` hands the line back
                # instead of writing it (the order the parallel runner spends a
                # lock on — see there). What this branch gets out of that order
                # is narrower than what the fleet gets, and the difference is
                # worth naming here rather than inheriting the fleet's promise.
                # A refused write cannot lose the SENTINEL: `commit_stop` marked
                # it for cleanup before returning, and `stop_file_lifecycle`'s
                # finally still removes it while the exception unwinds. It does
                # lose the REASON: `stop_reason` is a local, nothing around this
                # loop catches, and a raising print leaves run_loop with no
                # RunResult at all (exitlog then prints "reason not recorded").
                # Left that way on purpose — one thread, no other worker to
                # mislead, so the exception IS this run's ending instead of a
                # wrong answer about why it ended.
                stop_reason, announcement = stopchannel.commit_stop(app, pending)
                print(announcement)
                app.update(phase="stopping")
                break
            # Cancelled inside the interactive grace — carry on with no trace.

            # Git push policy: evaluated at the start of every iteration.
            if not dry_run:
                last_git_push = maybe_git_push(run_settings.git_push,
                                               last_git_push, projectroot.project_dir())

            max_runs = run_settings.max_runs
            if max_runs is not None and iteration >= max_runs:
                print(f"Iteration limit reached (--max-runs {max_runs}). Stopping.")
                stop_reason = stopchannel.RunStopReason.LIMIT_REACHED
                break

            # The `p` key's hold. AFTER the cap check, so a run that has no
            # iteration left to hold back ends instead of standing paused for a
            # boundary that will never come; BEFORE `driver.next_command()`,
            # which is what the key is for — the state file, the queue and the
            # tree are all quiet while it holds, so an edit made now is what the
            # next iteration reads.
            if stopchannel.pause_requested(app):
                if not paused_since:
                    # Announced once per pause, not once per pass: a stop
                    # requested and then withdrawn inside the hold sends the
                    # loop back through here, and that is the same pause.
                    paused_since = time.time()
                    print(f"\n  {statusline.PAUSE_GLYPH} Paused — press p to "
                          f"resume, s to stop, m to queue a note for the next "
                          f"iteration.")
                    exitlog.note(phase="paused (p key)", iterations=iteration,
                                 completed=completed)
                stopchannel.wait_while_paused(app, should_stop=stop_pending)
                # Back to the head rather than on: a stop pressed during the hold
                # is the head's to act on (with its cancel grace), and the caps
                # and quotas are re-read there.
                continue
            if paused_since:
                # Only here, where the loop is actually going on to work: a hold
                # the stop channels ended never reaches this line, and "pause
                # released" is not what happened to a run that is stopping.
                print(f"  ▶ Pause released after "
                      f"{statusline.format_elapsed(time.time() - paused_since)}.")
                paused_since = 0.0

            # Proactive limit check: read the real Current-session usage from the
            # account and pause cleanly between iterations if it is already at/over
            # the threshold, instead of running an iteration that would hit the wall.
            if not dry_run and not ignore_usage_limits:
                app.update(phase="waiting")
                paused, session_start = limit_policy.check_and_wait(
                    usage_source, session_start, should_stop=stop_pending)
                # The check just paid for a usage reading; publishing it here is
                # what puts the provider's live limits on the status row without
                # a second HTTP round-trip (the UsageSource cache serves it).
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if paused:
                    consecutive_errors = 0  # fresh window — start counting errors anew
                if stop_pending() or stopchannel.pause_requested(app):
                    # Asked to stop or to hold while parked on the limit. Back to
                    # the loop head, which owns both decisions (a key press still
                    # gets its cancel grace, a stop file still ends the run at
                    # once) — deciding here would be a second, divergent copy.
                    #
                    # The pause half is not symmetry: the gate is the longest
                    # hold in the engine, so it is exactly where somebody watching
                    # a run that is doing nothing reaches for `p`. Without this
                    # re-read the flag was set, the gate did not watch it, and the
                    # window opening started a full iteration under a row that
                    # already said PAUSED.
                    continue

            # Ask the driver what to do next. None => no more work (stop cleanly);
            # LoopStop => abort the run (e.g. an error state needing a human).
            try:
                command = driver.next_command()
            except LoopStop as stop:
                print(stop.message)
                if stop.exit_code:
                    exitlog.set_reason(
                        f"the driver stopped the run (exit {stop.exit_code}): "
                        f"{stop.message.splitlines()[0]}")
                    sys.exit(stop.exit_code)
                stop_reason = stopchannel.RunStopReason.DRIVER_STOP
                break
            if command is None:
                print("No more work — stopping.")
                stop_reason = stopchannel.RunStopReason.NO_WORK
                break

            # Notes typed while nothing was running (or while the transport was
            # off) ride this prompt — see Mailbox.splice for the ordering.
            if mailbox is not None:
                spliced, notes = mailbox.splice(command.prompt)
                if notes:
                    command = command._replace(prompt=spliced)
                    for note in notes:
                        print_note(note)

            iteration += 1
            state_label = command.label or "(no label)"
            # Show the model this iteration will use right in the header, so the
            # per-iteration model is visible up front (an empty command.model means
            # no --model flag — the CLI falls back to its own configured default).
            model_label = command.model or "cli default"
            # Same three calls the parallel workers make, on this run's one Job:
            # the Job clock times THIS iteration, the run clock (latched once)
            # times the whole run.
            started_at = time.time()
            app.mark_run_started(started_at)
            # The Job bumps its own counter (no `iteration=`): the local counter
            # restarts with every runner call, and pinning the row to it is what
            # kept a periodic run's job rows at 1.
            app.job(1).start(item=state_label, model=command.model,
                             prompt=command.prompt, now=started_at)
            if tracked_total is None:
                progress.note_iteration()
            app.update(**progress.summary_fields(), phase="running")
            # What a post-mortem needs from a run that never got to write an
            # ending: which item it was on when it stopped existing.
            exitlog.note(phase=f"iteration {iteration} — {state_label}",
                         iterations=iteration, completed=completed)
            # Through the module, unlike the line helpers above: this is the one
            # printed line the run must be able to READ BACK (`report_costs`
            # parses "=== Iteration 1 ===" as a run boundary), and the pins that
            # capture printed lines replace `console.print_markup`. A binding of
            # our own here would be a third address none of them reaches, so the
            # header would sail past every one of them uncaptured.
            console.print_markup(
                f"\n=== Iteration {iteration} === [{state_label} · {model_label}]",
                f"\n[bold cyan]=== Iteration {iteration} ===[/] "
                f"[dim]\\[{state_label} · {model_label}][/]",
            )

            cmd = build_agent_argv(command, provider)
            if dry_run:
                print("DRY-RUN:", " ".join(cmd))
                if prompt_on_stdin(provider):
                    # The argv above is complete but not self-contained: the
                    # prompt travels on stdin, so the preview has to show it
                    # separately or it shows a command with no task in it.
                    print("STDIN:", command.prompt)
                if not dry_run_prompt_shown:
                    # The argv line above is what will be executed, but for
                    # claude the whole prompt sits inside one joined `-p …`
                    # token and is unreadable — and reading the prompt is what a
                    # dry run is for. Printed once, for job 1.
                    dry_run_prompt_shown = True
                    print(statusline.format_prompt_block(
                        job_id=1, label=state_label, prompt=command.prompt,
                        width=textwidth.screen_width()))
                # looping forever in dry-run is pointless — nothing is actually done,
                # so the driver would keep handing back the same first unit of work.
                if run_settings.max_runs is None:
                    print("(dry-run without --max-runs: running a single iteration and exiting)")
                    stop_reason = stopchannel.RunStopReason.DRY_RUN
                    break
                continue

            if provider == "claude":
                returncode = run_claude_streaming(
                    cmd, raw, partial=True, prompt=command.prompt,
                    mailbox=mailbox)
            else:
                returncode = run_agent_streaming(
                    cmd, provider, raw, partial=False, prompt=command.prompt)
            app.job(1).finish()
            app.update(phase="idle")

            if returncode == 0:
                consecutive_errors = 0
                completed += 1
                driver.on_success(returncode)
                if tracked_total is not None:
                    # on_success recorded the item, so the driver's own count now
                    # says how far the invocation has got.
                    remaining = driver.pending_total()
                    if remaining is not None:
                        progress.note_remaining(remaining)
                        app.update(**progress.summary_fields())

            # The backstop under the proactive check (see RateLimitEvent): this run's
            # own verdict from the wire. A refusal needs no figure and no query to be
            # trusted, so it is honoured whatever the usage report said — including
            # when the report was unavailable, which is the case this exists for.
            # Checked for both outcomes: a run refused on its last turn may still have
            # exited 0 with its work recorded above.
            refusal = last_rate_limit_event() if provider == "claude" else None
            if (not ignore_usage_limits and refusal is not None
                    and refusal.status == "rejected"):
                # +5s so we come back after the reset, not exactly on it.
                target_ts = (refusal.resets_at
                             or time.time() + CLAUDE_SESSION_DURATION) + 5
                app.update(phase="paused")
                wait_until(target_ts,
                           reason=f"Hit the {refusal.label} — this run was refused. "
                                  f"Waiting until {_fmt_moment(target_ts)} for that "
                                  f"window to refresh…",
                           should_stop=stop_pending)
                usage_source.invalidate()  # the figures behind the refusal are stale
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if refusal.limit_type == "five_hour":
                    session_start = time.time()
                consecutive_errors = 0  # fresh window — start counting errors anew
                continue

            if returncode == 0:
                continue

            # Non-zero exit — the cause is ambiguous (a network blip / one-off CLI
            # hiccup, or the session's token limit). Rather than guessing from a
            # second consecutive error, ask the account directly: read the usage
            # figures right away and let the real Current-session percentage decide.
            consecutive_errors += 1
            elapsed = time.time() - session_start
            print_error(f"{spec.display_name} exited with code {returncode} "
                        f"(error #{consecutive_errors} in a row).")

            if not ignore_usage_limits:
                app.update(phase="waiting")
                paused, session_start = limit_policy.check_and_wait(
                    usage_source, session_start, note=" (checked after error)",
                    should_stop=stop_pending)
                statusline.push_quotas(app, usage_source, limit_policy)
                app.update(phase="idle")
                if paused:
                    consecutive_errors = 0  # fresh window — start counting errors anew
                    continue
                if stop_pending():
                    # Kept separate from the reset above: a request that is
                    # withdrawn inside the grace must not have zeroed the
                    # five-errors-in-a-row brake on its way past, or a genuinely
                    # broken provider gets to loop for free.
                    continue

            # Session is under the limit — this was a transient failure, not token
            # exhaustion. Retry, but don't spin forever if something is truly broken.
            if consecutive_errors < 5:
                if usage_source is None:
                    print("  ↻ Provider quota status is unavailable — retrying immediately.")
                else:
                    print("  ↻ Session under the allowed limit — likely transient. "
                          "Retrying immediately.")
                continue
            else:
                quota_state = ("with provider quota status unavailable"
                               if usage_source is None
                               else "with the session under the allowed limit")
                print(f"  ⚠ {consecutive_errors} errors in a row {quota_state} after "
                      f"{int(elapsed // 60)} min. Stopping.")
                exitlog.set_reason(
                    f"{consecutive_errors} provider errors in a row "
                    f"(last exit code {returncode})",
                    iterations=iteration, completed=completed)
                sys.exit(returncode)

    # Final push: regardless of the EACH_HOUR cadence, push any pending commits
    # on the way out so work isn't left only on the local branch — unless the
    # policy is NONE (never auto-push).
    if not dry_run and run_settings.git_push != GitPushPolicy.NONE:
        count = git_unpushed_count(projectroot.project_dir())
        if count is None or count > 0:
            print("  · final git push on exit…")
            git_push(projectroot.project_dir())
        else:
            print("  · final git push: nothing to push.")

    # End-of-run usage snapshot (the policy's watched quotas), mirroring the one
    # logged before iteration 1 — so each run records where it finished. Forced
    # fresh (cache_value=False) so it reflects the true post-run state rather
    # than a possibly-recent cached reading from the last limit check.
    if not dry_run and usage_source is not None:
        limit_policy.log_snapshot(usage_source, "at end (after last cycle)",
                                  cache_value=False)

    report_undelivered_notes(mailbox)

    # Closing line, if the driver has one (e.g. "Final state: …").
    summary = driver.final_summary()
    if summary:
        print(f"\n{summary}")
    # The reason this runner returned. Recorded rather than printed: a wrapper
    # may call several runners, and the `=== run ended: … ===` line belongs to
    # the process, so the last reason set wins and exitlog prints it on exit.
    exitlog.set_reason(
        stopchannel.STOP_REASON_TEXT.get(stop_reason, stop_reason.value),
        iterations=iteration, completed=completed)
    return stopchannel.RunResult(stop_reason, iteration, completed)
