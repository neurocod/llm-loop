"""The provider stream-json vocabulary, written once.

Both runners parse the SAME stream. The sequential one renders it live into a
terminal (`streamrender`), the parallel one renders one atomic line per key
event so N workers can print at once (`parallel.run_job`), and two more modules
touch the same wire from the edges: `operator.user_message_line` WRITES a user
message onto it, and `usage.rate_limit_event_from` reads one event's payload out
of it. The SHAPES of the printed lines were made common long ago
(`compactline`), but the dispatch and the key names were not — measured
2026-08-24 over both renderer bodies, 32 of the wire's literals were spelled in
`cyclecore` and in `parallel` both, `agent_message` and `total_cost_usd` and
`cached_input_tokens` among them. A new field of the stream therefore had to be
added in two places, and nothing said so: the second renderer simply went on
printing what it had always printed.

So the words live here and nowhere else, and `tests/test_wire_vocabulary.py`
keeps that true by reading THIS module's own `VOCABULARY` and failing on any
package module that spells one of them itself.

Two kinds of name are declared, and the difference is not stylistic:

  * a CONSTANT for every value the stream is dispatched ON — event types, item
    types, block types, the subtypes. Those are the words a reader greps for;
  * a FUNCTION for every field the stream is read THROUGH. The key names behind
    them ("message", "content", "text", "usage", …) are ordinary English, so
    they are deliberately NOT constants — policing `"text"` package-wide would
    report a dozen innocent strings. Kept inside these bodies they exist once
    anyway, which is the property that matters.

What is NOT here: the tool-INPUT schema (`Bash`'s `command`, `Grep`'s `path`,
…). That is a layer deeper — the tools' own vocabulary rather than the
stream's — and it already lives in one place, `compactline.describe_tool`.
"""

from typing import Any, Optional

from . import compactline

# --- Claude stream-json: event types --------------------------------------
SYSTEM = "system"
SUBTYPE_INIT = "init"
ASSISTANT = "assistant"
USER = "user"
RESULT = "result"
STREAM_EVENT = "stream_event"
RATE_LIMIT_EVENT = "rate_limit_event"
SUBTYPE_SUCCESS = "success"

# --- Claude stream-json: content-block types ------------------------------
BLOCK_TEXT = "text"
BLOCK_TOOL_USE = "tool_use"
BLOCK_TOOL_RESULT = "tool_result"
BLOCK_IS_ERROR = "is_error"

# --- Claude stream-json: the partial-message deltas -----------------------
CONTENT_BLOCK_START = "content_block_start"
CONTENT_BLOCK_DELTA = "content_block_delta"
CONTENT_BLOCK_STOP = "content_block_stop"
TEXT_DELTA = "text_delta"

# --- Claude stream-json: the result event's figures -----------------------
TOTAL_COST_USD = "total_cost_usd"
DURATION_MS = "duration_ms"

# --- Codex events normalised from `exec --json` or app-server -------------
THREAD_STARTED = "thread.started"
ITEM_STARTED = "item.started"
ITEM_COMPLETED = "item.completed"
TURN_COMPLETED = "turn.completed"
TURN_FAILED = "turn.failed"
ERROR = "error"

# --- Codex: item types ----------------------------------------------------
AGENT_MESSAGE = "agent_message"
COMMAND_EXECUTION = "command_execution"
FILE_CHANGE = "file_change"
USER_MESSAGE = "user_message"

# --- Codex app-server JSON-RPC methods and item types ---------------------
APP_INITIALIZE = "initialize"
APP_INITIALIZED = "initialized"
APP_THREAD_START = "thread/start"
APP_THREAD_STARTED = "thread/started"
APP_TURN_START = "turn/start"
APP_TURN_STEER = "turn/steer"
APP_ITEM_STARTED = "item/started"
APP_ITEM_COMPLETED = "item/completed"
APP_TURN_COMPLETED = "turn/completed"
APP_TOKEN_USAGE_UPDATED = "thread/tokenUsage/updated"
APP_RATE_LIMITS_READ = "account/rateLimits/read"
APP_ERROR = "error"
APP_AGENT_MESSAGE = "agentMessage"
APP_COMMAND_EXECUTION = "commandExecution"
APP_FILE_CHANGE = "fileChange"
APP_USER_MESSAGE = "userMessage"

# --- Codex: the fields worth naming ---------------------------------------
EXIT_CODE = "exit_code"
AGGREGATED_OUTPUT = "aggregated_output"
INPUT_TOKENS = "input_tokens"
CACHED_INPUT_TOKENS = "cached_input_tokens"
OUTPUT_TOKENS = "output_tokens"


# Three of the words above are ordinary English, and three modules that have
# nothing to do with this stream spell them for their own reasons: `codex_usage`
# parses the app-server's JSON-RPC envelope, whose keys are `result` and `error`;
# `drivers.StateFileDriver.error_token` is a word looked for in a state FILE; and
# `providers` passes `text=True` to `subprocess.Popen`, which is about decoding a
# pipe. Naming those sites here rather than dropping the three words from the
# guarded set is what keeps a THIRD copy of `et == "result"` a violation. The
# gate also fails on an entry that has gone stale, so this cannot quietly turn
# into an amnesty list: an exemption survives only while its module still says
# the word.
SPELLED_ELSEWHERE = {
    "codex_usage": frozenset({RESULT, ERROR}),
    "drivers": frozenset({ERROR}),
    "providers": frozenset({BLOCK_TEXT}),
}

# Every word this module owns — the set the gate polices. Derived from the
# module's own globals rather than typed out again: a constant added above joins
# it by existing, which is the only way a list like this stays true.
VOCABULARY = frozenset(
    value for name, value in list(globals().items())
    if name.isupper() and isinstance(value, str)
)


# --- reading one event ----------------------------------------------------

def event_type(ev: dict) -> Optional[str]:
    """The event's own type — the value every dispatch below switches on."""
    return ev.get("type")


def session_model(ev: dict) -> str:
    """The model named by Claude's `system`/`init` event."""
    return ev.get("model", "?")


def is_session_start(ev: dict) -> bool:
    """True for the one event that opens a Claude session."""
    return event_type(ev) == SYSTEM and ev.get("subtype") == SUBTYPE_INIT


# --- assistant / user content blocks --------------------------------------

def message_blocks(ev: dict) -> list:
    """The content blocks of an `assistant` or `user` event.

    `content` is a list of typed blocks on most events, but a bare STRING on a
    `user` event the CLI writes itself — the echo of a note typed into a running
    turn arrives as `{"message": {"content": "…"}}`. Both renderers iterate what
    this returns, and over a string that iteration yields characters, so the
    string is wrapped here into the one text block it means. Anything else
    (None, a number) has no blocks in it.
    """
    content = ev.get("message", {}).get("content", [])
    if isinstance(content, str):
        return [{"type": BLOCK_TEXT, "text": content}]
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def block_type(block: dict) -> Optional[str]:
    """A content block's type — `text`, `tool_use`, `tool_result`, …"""
    return block.get("type")


def block_text(block: dict) -> str:
    """A text block's text."""
    return block.get("text", "")


def tool_use_name(block: dict) -> str:
    """The tool a `tool_use` block calls."""
    return block.get("name", "?")


def tool_use_input(block: dict) -> dict:
    """A `tool_use` block's arguments (never None — an absent one is empty)."""
    return block.get("input", {}) or {}


def tool_result_failed(block: dict) -> bool:
    """Whether a `tool_result` block reports an error."""
    return bool(block.get(BLOCK_IS_ERROR))


def tool_result_text(block: dict) -> str:
    """A `tool_result` block's content as one string.

    The field is a string on some turns and a list of typed parts on others, and
    both renderers flattened it the same way; the shape is the wire's, so the
    flattening is this module's.
    """
    content = block.get("content", "")
    if isinstance(content, list):
        return " ".join(part.get(BLOCK_TEXT, "") for part in content
                        if isinstance(part, dict))
    return content


# --- the partial-message deltas (Claude, --include-partial-messages) -------

def stream_inner(ev: dict) -> dict:
    """The Anthropic streaming event wrapped inside a `stream_event`."""
    return ev.get("event", {})


def stream_block_index(inner: dict) -> Any:
    """Which content block this delta belongs to."""
    return inner.get("index")


def stream_block_is_text(inner: dict) -> bool:
    """Whether a `content_block_start` opened a TEXT block."""
    return inner.get("content_block", {}).get("type") == BLOCK_TEXT


def stream_text_delta(inner: dict) -> Optional[str]:
    """The text carried by a `content_block_delta`, or None if it carries none.

    None and "" are different answers: a delta of another kind has no text at
    all, while a text delta may legitimately carry an empty string.
    """
    delta = inner.get("delta", {})
    if delta.get("type") != TEXT_DELTA:
        return None
    return delta.get(BLOCK_TEXT, "")


# --- the result event -----------------------------------------------------

def result_cost(ev: dict, default: Any = None) -> Any:
    """The session's running cost total as of this `result`, or `default`.

    A running TOTAL, not this turn's share — see the caller that differences it.
    The default is the caller's for the same reason as `codex_exit_code`'s: the
    parallel runner passes the figure it already has, because a second `result`
    without one must not erase the first.
    """
    return ev.get(TOTAL_COST_USD, default)


def result_duration_ms(ev: dict) -> Optional[float]:
    """This turn's duration in milliseconds (None if absent)."""
    return ev.get(DURATION_MS)


def result_subtype(ev: dict) -> str:
    """How the turn ended, as the wire words it."""
    return ev.get("subtype", ERROR)


def result_failed(ev: dict) -> bool:
    """Whether this `result` reports a failed turn."""
    return ev.get("subtype") != SUBTYPE_SUCCESS or bool(ev.get(BLOCK_IS_ERROR))


# --- Codex ----------------------------------------------------------------

def codex_item(ev: dict) -> dict:
    """The item an `item.*` event is about (never None — an absent one is empty)."""
    return ev.get("item") or {}


def codex_item_type(item: dict) -> Optional[str]:
    """What kind of item it is — `agent_message`, `command_execution`, …"""
    return item.get("type")


def codex_thread_id(ev: dict) -> str:
    """The thread a `thread.started` event opened."""
    return ev.get("thread_id") or "?"


def codex_message_text(item: dict) -> str:
    """An `agent_message` item's text."""
    return str(item.get(BLOCK_TEXT) or "")


def codex_user_message_text(item: dict) -> str:
    """Text replayed by an app-server ``userMessage`` item."""
    return str(item.get(BLOCK_TEXT) or "")


def codex_command(item: dict) -> str:
    """A `command_execution` item's command line, ready to print.

    Un-doubling the backslashes is part of reading the field rather than of
    printing it: the provider escapes them into the JSON string, and every
    consumer wanted them back.
    """
    return compactline.undouble_backslashes(str(item.get("command", "")))


def codex_exit_code(item: dict, default: Any = None) -> Any:
    """A finished `command_execution`'s exit code, or `default` if it carried none.

    The two renderers want different absences and both are deliberate: the live
    one asks for None so it can drop the whole "exit N: " prefix, the compact one
    asks for "" so its fixed head keeps an empty slot. Same read, so one place.
    """
    return item.get(EXIT_CODE, default)


def codex_output(item: dict) -> str:
    """A finished `command_execution`'s captured output."""
    return item.get(AGGREGATED_OUTPUT, "")


def codex_changed_paths(item: dict) -> list:
    """The paths a `file_change` item touched."""
    changes = item.get("changes") or []
    return [str(change.get("path")) for change in changes
            if isinstance(change, dict) and change.get("path")]


def codex_token_counts(ev: dict) -> Optional[tuple]:
    """`turn.completed`'s (input, cached, output) token counts, or None.

    None means the turn reported no usage block at all, which both renderers
    treat as "print no token line" — as opposed to a block of zeroes, which is a
    turn that really did use nothing.
    """
    usage = ev.get("usage") or {}
    if not usage:
        return None
    return (usage.get(INPUT_TOKENS, 0),
            usage.get(CACHED_INPUT_TOKENS, 0),
            usage.get(OUTPUT_TOKENS, 0))


def codex_error(ev: dict) -> Any:
    """What an `error` / `turn.failed` event says went wrong.

    Falls back to the event itself, because a provider that fails in a way this
    list does not cover must still print SOMETHING. The two renderers spelled
    this chain differently — the sequential one looked in the item as well, the
    parallel one did not — and the longer chain is the one kept: an item's own
    error is more specific than the whole event dumped as a dict.
    """
    item = codex_item(ev)
    return ev.get("message") or ev.get(ERROR) or item.get(ERROR) or ev


# --- Codex app-server -----------------------------------------------------

def _rpc(method: str, request_id: Optional[int] = None,
         params: Optional[dict] = None) -> dict:
    message = {"method": method}
    if request_id is not None:
        message["id"] = request_id
    if params is not None:
        message["params"] = params
    return message


def codex_app_initialize(request_id: int) -> dict:
    return _rpc(APP_INITIALIZE, request_id, {"clientInfo": {
        "name": "llm_loop",
        "title": "llm-loop",
        "version": "1.0.0",
    }})


def codex_app_initialized() -> dict:
    return _rpc(APP_INITIALIZED, params={})


def codex_app_rate_limits_read(request_id: int) -> dict:
    return _rpc(APP_RATE_LIMITS_READ, request_id)


def codex_app_thread_start(request_id: int, project_dir: str, model: str,
                           sandbox_mode: str) -> dict:
    explicit_sandbox = bool(sandbox_mode)
    params = {
        "cwd": project_dir,
        "model": model or None,
        "sandbox": sandbox_mode or "workspace-write",
        # This is the app-server spelling of `exec --approve-for-me`: requests
        # stay enabled, but their reviewer is Codex's automatic reviewer.
        "approvalPolicy": "never" if explicit_sandbox else "on-request",
        "approvalsReviewer": None if explicit_sandbox else "auto_review",
    }
    return _rpc(APP_THREAD_START, request_id, params)


def codex_app_turn_start(request_id: int, thread_id: str, prompt: str) -> dict:
    return _rpc(APP_TURN_START, request_id, {
        "threadId": thread_id,
        "input": [{"type": BLOCK_TEXT, "text": prompt}],
    })


def codex_app_turn_steer(request_id: int, thread_id: str, turn_id: str,
                         text: str) -> dict:
    return _rpc(APP_TURN_STEER, request_id, {
        "threadId": thread_id,
        "expectedTurnId": turn_id,
        "input": [{"type": BLOCK_TEXT, "text": text}],
    })


def codex_rpc_response_id(message: dict) -> Optional[int]:
    value = message.get("id")
    return value if isinstance(value, int) else None


def codex_rpc_result(message: dict) -> dict:
    value = message.get("result")
    return value if isinstance(value, dict) else {}


def codex_rpc_error(message: dict) -> str:
    value = message.get(ERROR)
    if not value:
        return ""
    if isinstance(value, dict):
        return str(value.get("message") or value)
    return str(value)


def codex_app_thread_id(result: dict) -> str:
    thread = result.get("thread") or {}
    return str(thread.get("id") or "")


def codex_app_turn_id(result: dict) -> str:
    turn = result.get("turn") or {}
    return str(turn.get("id") or "")


def codex_app_notification_belongs_to(message: dict, thread_id: str,
                                      turn_id: str) -> bool:
    """Whether a notification belongs to the app-server turn we started.

    One app-server connection also reports work done by collaboration agents.
    Their item and completion notifications use different thread/turn IDs; if
    those reach the provider-neutral stream, a child's ``turn/completed`` looks
    exactly like completion of the root job.  Require every ID declared by the
    notification to match.  ``thread/started`` has no turn scope; every other
    notification consumed by this adapter must also declare the expected turn,
    so malformed completion events cannot finish the job either.
    """
    params = message.get("params") or {}
    if not isinstance(params, dict):
        return False
    thread = params.get("thread") or {}
    turn = params.get("turn") or {}
    thread_ids = [params.get("threadId")]
    turn_ids = [params.get("turnId")]
    if isinstance(thread, dict):
        thread_ids.append(thread.get("id"))
    if isinstance(turn, dict):
        turn_ids.append(turn.get("id"))
    thread_ids = [str(value) for value in thread_ids if value]
    turn_ids = [str(value) for value in turn_ids if value]
    thread_matches = (bool(thread_ids)
                      and all(value == thread_id for value in thread_ids))
    if message.get("method") == APP_THREAD_STARTED:
        return thread_matches
    return (thread_matches and bool(turn_ids)
            and all(value == turn_id for value in turn_ids))


def codex_app_token_usage(message: dict) -> Optional[dict]:
    """The latest turn token counts, in the normalised ``exec --json`` shape."""
    if message.get("method") != APP_TOKEN_USAGE_UPDATED:
        return None
    params = message.get("params") or {}
    usage = (params.get("tokenUsage") or {}).get("last") or {}
    if not usage:
        return {}
    return {
        INPUT_TOKENS: usage.get("inputTokens", 0),
        CACHED_INPUT_TOKENS: usage.get("cachedInputTokens", 0),
        OUTPUT_TOKENS: usage.get("outputTokens", 0),
    }


def _codex_app_text(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get(BLOCK_TEXT) or "")
        for part in content
        if isinstance(part, dict) and part.get("type") == BLOCK_TEXT
    )


def _codex_app_item(item: Any) -> dict:
    if not isinstance(item, dict):
        return {}
    normalised = dict(item)
    item_type = item.get("type")
    if item_type == APP_AGENT_MESSAGE:
        normalised["type"] = AGENT_MESSAGE
    elif item_type == APP_COMMAND_EXECUTION:
        normalised["type"] = COMMAND_EXECUTION
        normalised[EXIT_CODE] = item.get("exitCode")
        normalised[AGGREGATED_OUTPUT] = item.get("aggregatedOutput") or ""
    elif item_type == APP_FILE_CHANGE:
        normalised["type"] = FILE_CHANGE
    elif item_type == APP_USER_MESSAGE:
        normalised["type"] = USER_MESSAGE
        normalised[BLOCK_TEXT] = _codex_app_text(item.get("content"))
    return normalised


def codex_app_event(message: dict, usage: Optional[dict] = None) -> Optional[dict]:
    """Translate one app-server notification to the existing Codex event wire.

    Keeping this adapter at the wire boundary lets both renderers and the old
    result parsers continue to consume exactly one provider-neutral Codex shape.
    JSON-RPC responses and notifications with no ``exec --json`` counterpart
    return ``None``.
    """
    method = message.get("method")
    params = message.get("params") or {}
    if method == APP_THREAD_STARTED:
        thread = params.get("thread") or {}
        return {"type": THREAD_STARTED,
                "thread_id": str(thread.get("id") or "?")}
    if method in (APP_ITEM_STARTED, APP_ITEM_COMPLETED):
        return {
            "type": ITEM_STARTED if method == APP_ITEM_STARTED else ITEM_COMPLETED,
            "item": _codex_app_item(params.get("item")),
        }
    if method == APP_ERROR:
        error = params.get(ERROR) or {}
        detail = error.get("message") if isinstance(error, dict) else error
        return {"type": ERROR, "message": str(detail or params)}
    if method == APP_TURN_COMPLETED:
        turn = params.get("turn") or {}
        status = turn.get("status")
        if status != "completed":
            error = turn.get(ERROR) or {}
            detail = error.get("message") if isinstance(error, dict) else error
            return {"type": TURN_FAILED,
                    ERROR: str(detail or status or "turn failed")}
        event = {"type": TURN_COMPLETED}
        if usage:
            event["usage"] = usage
        return event
    return None


# --- writing one event ----------------------------------------------------

def user_message(text: str) -> dict:
    """One `--input-format stream-json` user message, as a dict.

    The WRITE half of the `user` event `message_blocks` reads back: a note typed
    at the console goes out in this shape and the CLI replays it in that one, so
    the two halves are one fact and belong in one file. `operator` serialises it.
    """
    return {
        "type": USER,
        "message": {"role": "user",
                    "content": [{"type": BLOCK_TEXT, "text": text}]},
    }
