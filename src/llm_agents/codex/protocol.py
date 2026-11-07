"""The wire: one `codex app-server --stdio` process, spoken to over JSON-RPC.

Everything here is about the protocol and the child process that speaks it — no
registry, no CLI. The names and reply shapes were taken from the schema bundle
that `codex app-server generate-json-schema` prints, not guessed: a wrong one is
accepted in silence and shows up as a turn that never completes.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from llm_loop.procend import REAP_GRACE_S, end_process_tree

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

# The handshake and the thread/turn openers are answered in well under a second
# by a healthy server; a separate, short bound keeps a misconfigured CLI from
# looking like a slow model.
STARTUP_TIMEOUT_S = 60.0

# `REAP_GRACE_S` (imported above) bounds each step of ending the server.


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


def _reap(proc: subprocess.Popen) -> None:
    """End a child, aiming past the npm shim on Windows.

    On Windows `codex` on PATH is `codex.cmd`, so the handle we hold is
    `cmd.exe` and the real CLI is a GRANDCHILD: `terminate()` would leave it
    running, still holding the pipes. `taskkill /T` reaches the tree. The
    escalation — ask, wait, kill, wait, each bounded by `REAP_GRACE_S` — is
    `llm_loop.procend.end_process_tree`, the one copy every owner of a
    provider child shares; this keeps only its own prologue: a child that has
    already exited is left alone.

    Nothing here may raise: it runs from `finally` blocks, where an exception
    would replace the one being unwound (`end_process_tree` absorbs every
    `OSError`; a `KeyboardInterrupt` still propagates, as it always did).
    """
    if proc.poll() is not None:
        return
    end_process_tree(proc)


class AppServer:
    """One `codex app-server --stdio` process, spoken to over JSON-RPC.

    Reading happens on a worker thread feeding a queue, so every wait in this
    class can carry a timeout. A blocking `readline()` cannot: a server that
    dies mid-turn, or a model that stalls, would hang the caller forever with no
    way to report which of the two happened.
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
        self._next_id = 0
        self._proc = subprocess.Popen(
            [_codex_executable(), "app-server", "--stdio"],
            cwd=str(cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**os.environ, **getattr(approve, "environment", {})},
        )
        self._inbox: "queue.Queue[Optional[dict]]" = queue.Queue()
        self.stderr_lines: list[str] = []
        self.finished = False  # the child's stdout reached EOF
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    # -- plumbing

    def _pump_stdout(self) -> None:
        try:
            for line in self._proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._inbox.put(json.loads(line))
                except ValueError:
                    self._inbox.put({"_raw": line})
        except (OSError, ValueError):
            pass
        finally:
            self._inbox.put(None)  # end of stream sentinel

    def _pump_stderr(self) -> None:
        try:
            for line in self._proc.stderr:
                line = line.rstrip()
                if line:
                    self.stderr_lines.append(line)
        except (OSError, ValueError):
            pass

    def _send(self, message: dict) -> None:
        self._proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._proc.stdin.flush()

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: Optional[dict] = None,
                timeout: float = STARTUP_TIMEOUT_S,
                on_event: Optional[Callable[[dict], None]] = None) -> dict:
        """Send a request and return its result, feeding events to `on_event`.

        Notifications and server-initiated requests keep arriving while we wait,
        so the wait cannot simply discard traffic: the server BLOCKS on an
        approval request until it is answered, and a discarded one is a turn
        that never ends.
        """
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id,
                    "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        while True:
            message = self._take(deadline, method)
            if message.get("id") == request_id and "method" not in message:
                error = message.get("error")
                if error:
                    detail = error.get("message") if isinstance(error, dict) else error
                    raise RuntimeError(f"{method} failed: {detail}")
                result = message.get("result")
                return result if isinstance(result, dict) else {}
            self._dispatch(message, on_event)

    def send_request(self, method: str, params: Optional[dict] = None) -> int:
        """Send a request without waiting for its answer; return its id.

        For requests whose outcome is an event the caller pumps for anyway
        (`turn/interrupt`: the reply comes as the turn aborts, just before its
        `turn/completed`, which is what the caller waits for): the reply is
        then one of the stale responses `_dispatch` logs and drops."""
        self._next_id += 1
        self._send({"jsonrpc": "2.0", "id": self._next_id,
                    "method": method, "params": params or {}})
        return self._next_id

    def pump(self, deadline: float, on_event: Callable[[dict], None]) -> None:
        """Read exactly one incoming message and route it."""
        self._dispatch(self._take(deadline, "turn"), on_event)

    def _take(self, deadline: float, what: str) -> dict:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            message = self._inbox.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"codex app-server did not answer {what} in time")
        if message is None:
            # The sentinel arrives once. Record that the stream is over so a
            # caller that catches this failure (the best-effort name/set) can
            # tell "the server rejected my request" from "the server is gone",
            # and put the sentinel back for whoever waits next.
            self.finished = True
            self._inbox.put(None)
            tail = "; ".join(self.stderr_lines[-3:])
            raise RuntimeError(
                f"codex app-server closed the connection during {what}"
                + (f" (stderr: {tail})" if tail else ""))
        return message

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
        try:
            self._proc.stdin.close()
        except (OSError, ValueError):
            pass
        _reap(self._proc)
        cleanup = getattr(self._approve, "close", None)
        if cleanup is not None:
            cleanup()
