"""The wire: one `codex app-server --stdio` process, spoken to over JSON-RPC.

Everything here is about the protocol and the child process that speaks it — no
registry, no CLI. The names and reply shapes were taken from the schema bundle
that `codex app-server generate-json-schema` prints, not guessed: a wrong one is
accepted in silence and shows up as a turn that never completes.

The transport itself — spawn, line-delimited JSON-RPC, id correlation and the
ending — is `llm_loop.appserver`, shared with the loop's provider process; this
module adds what an unattended agent needs on top: approvals, the edit judge's
bookkeeping, and the names of the thread and turn methods.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Callable, Optional

from llm_loop import appserver
from llm_loop.appserver import STARTUP_TIMEOUT_S
from llm_loop.procend import REAP_GRACE_S

# --- protocol names -------------------------------------------------------
# Spelled out rather than inlined because a typo in any of them shows up as a
# turn that never completes (the server simply ignores an unknown method and we
# wait for a notification that will not arrive), which is the most expensive
# failure mode this script has.

M_INITIALIZE = "initialize"
M_INITIALIZED = "initialized"
M_THREAD_START = "thread/start"
M_THREAD_RESUME = "thread/resume"
M_THREAD_NAME_SET = "thread/name/set"
M_TURN_START = "turn/start"
# `{threadId, turnId}` (rust-v0.164.0 `TurnInterruptParams`); the server
# answers once the turn has aborted, just BEFORE emitting its `turn/completed`
# (app-server bespoke_event_handling.rs), and an empty `turnId` interrupts a
# thread still starting up. Interrupting the main thread leaves its sub-agents
# running — each child thread is interrupted by its own id.
M_TURN_INTERRUPT = "turn/interrupt"

N_THREAD_STARTED = "thread/started"
N_TURN_STARTED = "turn/started"
N_ITEM_STARTED = "item/started"
N_ITEM_COMPLETED = "item/completed"
N_TURN_COMPLETED = "turn/completed"
N_TOKEN_USAGE = "thread/tokenUsage/updated"
N_ERROR = "error"
# Account-wide, unprompted, every 10–20 s of a turn; its params are the
# `account/rateLimits/read` result (`{"rateLimits": {...}}`).
N_RATE_LIMITS = "account/rateLimits/updated"

# Deltas are the bulk of the stream and we never render them: the final text
# arrives whole in `item/completed`. Opting out at `initialize` means the server
# does not send them at all, which is cheaper than filtering our side.
NOISY_NOTIFICATIONS = [
    "item/agentMessage/delta",
    "item/reasoning/textDelta",
    "item/reasoning/summaryTextDelta",
    "item/reasoning/summaryPartAdded",
    "item/commandExecution/outputDelta",
    "item/fileChange/outputDelta",
    "item/plan/delta",
    "command/exec/outputDelta",
    "process/outputDelta",
]

# `STARTUP_TIMEOUT_S` (imported above) bounds each handshake and opener
# request; `REAP_GRACE_S` bounds each step of ending the server.


# What to reply to each server-initiated request, per method. THREE DIFFERENT
# SHAPES live here, which is why this is a table and not one constant — all
# three were verified against the schema bundle from
# `codex app-server generate-json-schema` (codex-cli 0.149.0), because each
# wrong shape fails silently in its own way:
#
#   * the current turn/start API answers with `decision`, one of
#     accept / acceptForSession / decline / cancel — `decline` lets the turn
#     continue without the action, `cancel` would end the turn outright;
#   * `item/permissions/requestApproval` has NO decision field at all: the reply
#     is a granted profile, and an empty one IS the refusal;
#   * the legacy pair (execCommandApproval / applyPatchApproval, reached only by
#     the old SendUserMessage APIs) uses snake_case ReviewDecision values, where
#     refusal is a `denied` object carrying a rejection string — "decline" is
#     not a member of that enum.
M_COMMAND_APPROVAL = "item/commandExecution/requestApproval"
# `{threadId, turnId, itemId, reason, grantRoot}` (rust-v0.164.0
# `FileChangeRequestApprovalParams`): NO file list — the files are on the
# item's own `item/started`, which the server sends first (a worker's
# events.jsonl, lines 422 and 424, 2026-11-03).
M_FILE_CHANGE_APPROVAL = "item/fileChange/requestApproval"

# The `_meta` of the line each judged approval writes to `events.jsonl`.
APPROVAL_META = "approval"

# A fileChange approval has no row: it is always judged and logged
# (`AppServer._answer_file_change`), in the same `decision` shape.
APPROVAL_ANSWERS: dict[str, dict] = {
    M_COMMAND_APPROVAL: {"decision": "decline"},
    "item/permissions/requestApproval": {"permissions": {}},
    "item/tool/requestUserInput": {"answers": {}},
    "execCommandApproval": {
        "decision": {"denied": {"rejection": "unattended codex_agent run"}}},
    "applyPatchApproval": {
        "decision": {"denied": {"rejection": "unattended codex_agent run"}}},
}


def file_change_paths(item: dict) -> list[str]:
    """Every file a fileChange item writes: each change's `path` and, for a
    move, its `move_path` (the destination is written too)."""
    paths: list[str] = []
    for change in item.get("changes") or []:
        if not isinstance(change, dict):
            continue
        if change.get("path"):
            paths.append(str(change["path"]))
        kind = change.get("kind")
        moved = kind.get("move_path") if isinstance(kind, dict) else None
        if moved:
            paths.append(str(moved))
    return paths


class _Declined:
    """A "no" the wire gives itself, with its reason: no judge installed, or
    a judge that raised. Falsy, like a declining `sandbox.Verdict`."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def __bool__(self) -> bool:
        return False


def _judged(judge: Callable, request: object) -> object:
    """`judge(request)`, or a decline when it raises: the server blocks on the
    answer, so a judge's bug must cost one refusal, not the turn."""
    try:
        return judge(request)
    except Exception as exc:  # noqa: BLE001 — any bug is a "no", never a crash
        return _Declined(f"the wrapper's judge failed: {type(exc).__name__}: "
                         f"{exc}")


def _no_edits(_paths: list[str]) -> _Declined:
    """The edit judge of an AppServer given none."""
    return _Declined("this agent's edits never leave the sandbox")


def _reason_of(verdict: object) -> dict:
    """`{"reason": ...}` when the judge said why (`sandbox.Verdict`), else {}."""
    reason = getattr(verdict, "reason", "")
    return {"reason": reason} if reason else {}


# --- process plumbing -----------------------------------------------------


def _codex_executable() -> str:
    exe = shutil.which("codex")
    if not exe:
        raise SystemExit(
            "codex CLI not found on PATH. Install it (npm i -g @openai/codex) "
            "or put it on PATH before using this script."
        )
    return exe


def _server_argv() -> list[str]:
    """The command line of the server: on Windows `codex` on PATH is the npm
    shim `codex.cmd`, whose tree `appserver.Connection.close` ends."""
    return [_codex_executable(), "app-server", "--stdio"]


class AppServer:
    """One `codex app-server --stdio` process, spoken to over JSON-RPC.

    The reading, the waits and their timeouts are `appserver.Connection`'s;
    this class decides what every message read means to an unattended agent.
    """

    def __init__(self, cwd: Path, log: Optional[Callable[[dict], None]] = None,
                 approve: Optional[Callable[[dict], bool]] = None,
                 approve_files: Optional[Callable[[list[str]], bool]] = None):
        self._log = log or (lambda _msg: None)
        # The exceptions to "refuse everything": a judge for command
        # approvals, given the request's params, saying whether THIS command
        # may run outside the sandbox, and one for edits, given the files the
        # edit touches (`sandbox.file_change_judge`). None means nothing may.
        # A judge may return a `sandbox.Verdict`, whose `reason` is logged.
        self._approve = approve
        self._approve_files = approve_files or _no_edits
        # itemId -> files of every fileChange item started and not completed:
        # the approval request names the item, not its files.
        self._file_change_paths: dict[str, list[str]] = {}
        # One entry per fileChange approval answered, as logged — the turn
        # summary reads the declined ones (`turn._run_turn_logged`).
        self.file_decisions: list[dict] = []
        self._connection = appserver.Connection(appserver.spawn(
            _server_argv(), cwd=str(cwd),
            env={**os.environ, **getattr(approve, "environment", {})},
        ), first_id=1)

    @property
    def stderr_lines(self) -> list[str]:
        return self._connection.stderr_lines

    @property
    def finished(self) -> bool:
        """Whether the server's stdout has reached EOF (`Connection.take`)."""
        return self._connection.finished

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        self._connection.send(appserver.message(method, params))

    def request(self, method: str, params: Optional[dict] = None,
                timeout: float = STARTUP_TIMEOUT_S,
                on_event: Optional[Callable[[dict], None]] = None) -> dict:
        """Send a request and return its result, feeding events to `on_event`.

        Notifications and server-initiated requests keep arriving while we
        wait, and each is routed as it comes (`Connection.request`)."""
        return self._connection.request(
            appserver.message(method, params), timeout,
            on_other=lambda item: self._received(item, on_event))

    def send_request(self, method: str, params: Optional[dict] = None) -> int:
        """Send a request without waiting for its answer; return its id.

        For requests whose outcome is an event the caller pumps for anyway
        (`turn/interrupt`: the reply comes as the turn aborts, just before its
        `turn/completed`, which is what the caller waits for): the reply is
        then one of the stale responses `_dispatch` logs and drops."""
        return self._connection.send_request(appserver.message(method, params))

    def pump(self, deadline: float, on_event: Callable[[dict], None]) -> None:
        """Read exactly one incoming message and route it."""
        self._received(self._connection.take(deadline, "turn"), on_event)

    def _received(self, item: appserver.Item,
                  on_event: Optional[Callable[[dict], None]]) -> None:
        """Route one item read: a line that is not a JSON object is logged as
        `{"_raw": ...}`, a blank one dropped."""
        if isinstance(item, str):
            line = item.strip()
            if not line:
                return
            item = {"_raw": line}
        self._dispatch(item, on_event)

    def _send(self, message: dict) -> None:
        self._connection.send(message)

    def _dispatch(self, message: dict, on_event: Optional[Callable[[dict], None]]) -> None:
        self._log(message)
        if "method" not in message:
            return  # a stale response to a request we no longer wait on
        self._track_file_change(message)
        if message.get("id") is not None:
            self._answer_server_request(message)
            return
        if on_event is not None:
            on_event(message)

    def _answer_server_request(self, message: dict) -> None:
        """Answer a server-initiated request so the turn can move on.

        Approvals are decided by the sandbox and approval policy chosen at
        `thread/start`, not here: this runs unattended, so nobody is present to
        say yes. Refusing is therefore the only honest answer — but it must be
        SPELLED right, and the spelling differs per method (see
        APPROVAL_ANSWERS). An unanswered request stalls the turn until the
        timeout; a wrongly-shaped answer is a protocol error the server may
        treat the same way.

        Anything not in the table gets a JSON-RPC "method not found" rather than
        an invented result: a made-up payload can be accepted as a real answer
        and change what the agent does, while an error response cannot.

        The two "yes" answers this can give come from the caller's judges, each
        seeing the request rather than a flag: a COMMAND when `approve` says so
        for that command (the Windows sandbox cannot run this repo's npm gates
        — a Node child with piped stdio is refused — so a command's way out
        is judged one by one: `sandbox.escalation_judge`), and an EDIT that
        failed inside when `approve_files` says so for its files
        (`_answer_file_change`, `sandbox.file_change_judge`). Every other
        request is refused in its own shape.
        """
        method = str(message.get("method") or "")
        answer = APPROVAL_ANSWERS.get(method)
        params = message.get("params") or {}
        if method == M_COMMAND_APPROVAL and self._approve is not None:
            verdict = _judged(self._approve, params)
            if verdict:
                answer = {"decision": "accept"}
            self._log({"_meta": APPROVAL_META, "command": params.get("command"),
                       "decision": answer["decision"],
                       **_reason_of(verdict)})
        elif method == M_FILE_CHANGE_APPROVAL:
            answer = self._answer_file_change(params)
        if answer is None:
            self._send({"jsonrpc": "2.0", "id": message["id"], "error": {
                "code": -32601,
                "message": f"{method} is not handled by an unattended client",
            }})
            return
        self._send({"jsonrpc": "2.0", "id": message["id"], "result": answer})

    def _track_file_change(self, message: dict) -> None:
        method = message.get("method")
        if method not in (N_ITEM_STARTED, N_ITEM_COMPLETED):
            return
        item = (message.get("params") or {}).get("item") or {}
        if item.get("type") != "fileChange":
            return
        item_id = str(item.get("id") or "")
        if method == N_ITEM_STARTED:
            self._file_change_paths[item_id] = file_change_paths(item)
        else:
            self._file_change_paths.pop(item_id, None)

    def _answer_file_change(self, params: dict) -> dict:
        """Judge an edit that failed inside the sandbox; log every decision.

        Logged even with no judge installed, because the agent hears a decline
        as "rejected by user" and the log is where the orchestrator finds out
        it was the wrapper.

        Only ONE edit is ever granted, and only one whose whole change list
        the wrapper saw on its `item/started`: a request for an item it never
        saw start is declined, not judged by the `grantRoot` that came with it
        (review 2026-11-03: an unseen item carrying `grantRoot=<workspace>`
        was accepted as "every path inside the roots"). A `grantRoot` asks to
        write under that root for the rest of the SESSION, which is not one
        edit; Codex core sends none today (rust-v0.164.0), so any is declined
        with its reason rather than judged as a path."""
        item_id = str(params.get("itemId") or "")
        paths = list(self._file_change_paths.get(item_id, []))
        grant_root = params.get("grantRoot")
        if grant_root is not None:
            verdict: object = False
            reason = {"reason": f"asks for grantRoot {grant_root}: the wrapper "
                                f"grants one edit's files, never a root for "
                                f"the session"}
        elif item_id not in self._file_change_paths:
            verdict = False
            reason = {"reason": "the wrapper did not see which files this "
                                "change touches (no item/started for it)"}
        else:
            verdict = _judged(self._approve_files, paths)
            reason = _reason_of(verdict)
        answer = {"decision": "accept" if verdict else "decline"}
        entry = {"_meta": APPROVAL_META, "kind": "fileChange", "item": item_id,
                 "thread": params.get("threadId"), "paths": paths,
                 **({"grantRoot": grant_root} if grant_root is not None else {}),
                 "decision": answer["decision"], **reason,
                 "sandbox_reason": params.get("reason")}
        self.file_decisions.append(entry)
        self._log(entry)
        return answer

    def close(self) -> None:
        self._connection.close()
        cleanup = getattr(self._approve, "close", None)
        if cleanup is not None:
            cleanup()
