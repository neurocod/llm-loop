"""Line-delimited JSON-RPC over a child's stdio: the one transport every
`codex app-server` client here is built on.

Three clients speak this wire for different reasons — the loop's provider
process (`providers`), whose stdout a renderer streams, the named Codex worker
(`llm_agents.codex.protocol`), which answers approvals and waits on
sub-agents, and the loop's quota reader (`codex_usage`), which keeps a private
server for rate-limit reads — and each used to carry its own spawn, line
parser, response wait and shutdown. A fix in one never reached the others: the
loop's startup waited with no bound at all, and two of them took any message
carrying the awaited id for the reply, a server-initiated request included.
What is common lives here; what a client needs on top (approvals and threads,
steering and event conversion, restarts and caching) stays with the client.

Reading happens on a pump thread feeding a queue, so every wait can carry a
deadline: a blocking `readline()` cannot, and a server that dies mid-turn or a
model that stalls would hang the caller with no way to say which of the two it
was. The pump also keeps the child's stdout drained, so the child never blocks
writing into a pipe nobody reads.
"""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from typing import Callable, Optional, Union

from . import wire
from .procend import end_unless_exited

# The bound on each startup request that starts nothing of its own: the
# handshake (`initialize`, timed from the spawn, so the CLI's own start through
# the npm shim is in it), `thread/name/set` and `turn/start` (answered once the
# turn is accepted, before the model says anything). A separate, short bound
# keeps a misconfigured CLI from looking like a slow model.
#
# Measured on codex-cli 0.167.0, 2026-11-08, 5 live runs from this repo's
# root: spawn + initialize 0.406-0.458 s, turn/start 0.027-0.047 s; codex-cli
# 0.159.2 took up to 1.463 s cold for spawn + handshake (2026-10-01,
# `codex_usage.APP_SERVER_MAX_AGE`). 60 s is 40x the slowest of those: a
# loaded machine or a slow disk fits, a CLI that never answers is still told
# apart from a turn's own 30-minute budget.
STARTUP_TIMEOUT_S = 60.0

# The bound on `thread/start` and `thread/resume`, kept OUT of
# `STARTUP_TIMEOUT_S` because opening a thread waits for the MCP servers the
# user's config marks `required = true`, each up to its own
# `startup_timeout_sec` (default 30 s, rust-v0.164.0
# `rmcp_client::DEFAULT_STARTUP_TIMEOUT`; the user may raise it — Codex's own
# test of this wait sets 120 s). Servers that are not required start in the
# background: with one enabled (`node_repl`), thread/start took 0.203-0.436 s
# (same 5 runs, 2026-11-08). The bound is what a hung server costs, so it errs
# long: 600 s is 20x the default and 5x that test's 120 s, and a user who
# needs more than that has an MCP server to fix, not a client to wait.
THREAD_OPEN_TIMEOUT_S = 600.0

# How often a wait without a deadline wakes: a blocking `Queue.get()` on
# Windows does not return to the interpreter for a Ctrl+C until it is fed.
_IDLE_WAKE_S = 0.5

# What the queue holds: a decoded JSON object, or the line exactly as read
# (a diagnostic, a JSON value that is not an object, a blank line).
Item = Union[dict, str]

_EOF = object()


class Closed(RuntimeError):
    """The server's stdout ended before the awaited message arrived."""


class InputClosed(OSError):
    """A write after `close_input`: nothing more can reach the server."""


class RpcError(RuntimeError):
    """The server answered a request with a JSON-RPC error."""

    def __init__(self, method: str, detail: str) -> None:
        super().__init__(f"{method} failed: {detail}")
        self.method = method
        self.detail = detail


def spawn(argv: list, *, cwd: Optional[str] = None, env: Optional[dict] = None,
          merge_stderr: bool = False, **options) -> subprocess.Popen:
    """Start an app-server child with UTF-8 text pipes on all three streams.

    `merge_stderr` folds stderr into stdout, where it reaches the reader as raw
    lines (the loop's renderer shows them); otherwise `Connection` drains it
    into `stderr_lines` for the error that reports the server's death.
    `options` go to `Popen` as given (`creationflags`)."""
    return subprocess.Popen(
        argv, cwd=cwd, env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1, **options)


def encode(message: dict) -> str:
    """One message as the line the server reads."""
    return json.dumps(message, separators=(",", ":")) + "\n"


def decode(line: str) -> Item:
    """The object a line carries, or the line itself when it carries none."""
    try:
        message = json.loads(line)
    except (TypeError, ValueError):
        return line
    return message if isinstance(message, dict) else line


def response_id(item: Item) -> Optional[int]:
    """The id a RESPONSE answers, or None for anything else.

    A server-initiated request carries an id too, from the server's own
    counter, so the id alone is not enough: a request whose id happens to equal
    one of ours is a question to answer, not our reply."""
    if not isinstance(item, dict) or "method" in item:
        return None
    return wire.codex_rpc_response_id(item)


def message(method: str, params: Optional[dict] = None) -> dict:
    """A request or notification body, without the id `Connection` assigns."""
    return {"jsonrpc": "2.0", "method": method, "params": params or {}}


class Pending:
    """One request on the wire, settled by whoever reads its response."""

    def __init__(self, request_id: int, method: str) -> None:
        self.id = request_id
        self.method = method
        self._settled = threading.Event()
        self._response: Optional[dict] = None
        self._failure = ""

    @property
    def done(self) -> bool:
        return self._settled.is_set()

    def wait(self, timeout: float) -> bool:
        return self._settled.wait(timeout)

    def settle(self, response: Optional[dict], failure: str = "") -> None:
        self._response = response
        self._failure = failure
        self._settled.set()

    @property
    def failure(self) -> str:
        """Why the request failed ("" when it did not): the server's error, or
        the reason the connection gave up on it (`Connection.fail_pending`)."""
        if self._failure:
            return self._failure
        return wire.codex_rpc_error(self._response or {})

    def result(self) -> dict:
        """The response's result; `RpcError` when the request failed."""
        failure = self.failure
        if failure:
            raise RpcError(self.method, failure)
        return wire.codex_rpc_result(self._response or {})


class Connection:
    """One app-server child, spoken to over line-delimited JSON-RPC.

    Takes a started process (`spawn`, or a stand-in with the same pipes) and
    owns its reading from here on: stdout through a pump thread into a queue
    (`take`), stderr — when it is a pipe of its own — into `stderr_lines`.
    Request ids count up from `first_id`; a response settles the `Pending` of
    its id (`resolve`), whichever thread reads it. Writes are serialized, so a
    second thread may send while the first one reads.
    """

    def __init__(self, proc, *, first_id: int = 0) -> None:
        self.proc = proc
        self.stderr_lines: list[str] = []
        self.finished = False  # the child's stdout reached EOF
        self.input_closed = False
        self._next_id = first_id
        self._pending: dict[int, Pending] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._inbox: "queue.Queue[object]" = queue.Queue()
        threading.Thread(target=self._pump_stdout, name="app-server-stdout",
                         daemon=True).start()
        if getattr(proc, "stderr", None) is not None:
            threading.Thread(target=self._pump_stderr, name="app-server-stderr",
                             daemon=True).start()

    # -- reading

    def _pump_stdout(self) -> None:
        stdout = self.proc.stdout
        try:
            for line in stdout:
                self._inbox.put(decode(line))
        except (OSError, ValueError):
            pass
        finally:
            self._inbox.put(_EOF)
            # The read end is closed HERE, by the thread that reads it: closing
            # it from another thread while this one is inside `readline` blocks
            # on the reader's lock until the child writes again or dies (19.0 s
            # measured 2026-10-02 against a grandchild holding the pipe).
            try:
                stdout.close()
            except (OSError, ValueError):
                pass

    def _pump_stderr(self) -> None:
        try:
            for line in self.proc.stderr:
                line = line.rstrip()
                if line:
                    self.stderr_lines.append(line)
        except (OSError, ValueError):
            pass

    def take(self, deadline: Optional[float], what: str) -> Item:
        """The next item read; `TimeoutError` past `deadline` (None waits for
        as long as the stream lasts), `Closed` once the stream has ended.

        The end-of-stream sentinel is put back each time it is read, and
        `finished` set, so a caller that catches the failure (a best-effort
        request) can tell "the server rejected my request" from "the server is
        gone", and whoever waits next sees the end too."""
        while True:
            if deadline is None:
                timeout = _IDLE_WAKE_S
            else:
                timeout = max(0.0, deadline - time.monotonic())
            try:
                item = self._inbox.get(timeout=timeout)
                break
            except queue.Empty:
                if deadline is not None:
                    raise TimeoutError(
                        f"codex app-server did not answer {what} in time") from None
        if item is _EOF:
            self.finished = True
            self._inbox.put(_EOF)
            tail = "; ".join(self.stderr_lines[-3:])
            raise Closed(f"codex app-server closed the connection during {what}"
                         + (f" (stderr: {tail})" if tail else ""))
        return item  # type: ignore[return-value]

    def resolve(self, item: Item) -> bool:
        """Settle the request `item` responds to; False when it is not the
        response of a request still awaited (a stale one included)."""
        request_id = response_id(item)
        if request_id is None:
            return False
        with self._pending_lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            return False
        pending.settle(item)  # type: ignore[arg-type]
        return True

    # -- writing

    def send(self, body: dict) -> None:
        """Write one message; `InputClosed` after `close_input`."""
        with self._write_lock:
            if self.input_closed:
                raise InputClosed("the server's input is closed")
            self.proc.stdin.write(encode(body))
            self.proc.stdin.flush()

    @property
    def next_id(self) -> int:
        """The id the next request will carry: an owner that outlives this
        connection continues its numbering from here (`codex_usage`)."""
        with self._pending_lock:
            return self._next_id

    def _number(self, body: dict) -> dict:
        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
        return {**body, "id": request_id}

    def send_request(self, body: dict) -> int:
        """Send a request nobody waits on; return its id. Its response is then
        a stale one: `resolve` declines it and the reader handles it as any
        other message."""
        numbered = self._number(body)
        self.send(numbered)
        return numbered["id"]

    def start_request(self, body: dict) -> Pending:
        """Send a request and return what its response will settle."""
        numbered = self._number(body)
        pending = Pending(numbered["id"], str(body.get("method") or ""))
        with self._pending_lock:
            self._pending[pending.id] = pending
        try:
            self.send(numbered)
        except BaseException:
            self.forget(pending)
            raise
        return pending

    def forget(self, pending: Pending) -> None:
        """Stop awaiting `pending`: a response arriving later is stale."""
        with self._pending_lock:
            self._pending.pop(pending.id, None)

    def request(self, body: dict, timeout: float = STARTUP_TIMEOUT_S,
                on_other: Optional[Callable[[Item], None]] = None) -> dict:
        """Send a request, read until its response, and return the result.

        Everything read meanwhile goes to `on_other`, in order, rather than
        being discarded: the server BLOCKS on a request of its own until it is
        answered, and a notification read here is one the caller would
        otherwise never see."""
        pending = self.start_request(body)
        deadline = time.monotonic() + timeout
        try:
            while not pending.done:
                item = self.take(deadline, pending.method)
                if not self.resolve(item) and on_other is not None:
                    on_other(item)
        finally:
            self.forget(pending)
        return pending.result()

    def fail_pending(self, reason: str) -> None:
        """Settle every awaited request with `reason`: no response will come."""
        with self._pending_lock:
            pending, self._pending = self._pending, {}
        for entry in pending.values():
            entry.settle(None, reason)

    # -- ending

    def close_input(self) -> None:
        """Close the server's stdin — its signal to exit. Idempotent."""
        with self._write_lock:
            if self.input_closed:
                return
            self.input_closed = True
            try:
                self.proc.stdin.close()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        """Close stdin, then end the child's tree unless it has exited.

        Nothing here may raise: it runs from `finally` blocks, where an
        exception would replace the one being unwound. stdout is left to the
        pump, which closes it at EOF (see `_pump_stdout`)."""
        self.close_input()
        end_unless_exited(self.proc)
