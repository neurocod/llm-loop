"""Read Codex account rate limits through the official app-server protocol.

``codex app-server`` exposes a JSONL/JSON-RPC method named
``account/rateLimits/read``.  This module performs the required initialization
handshake, requests that report, and maps its short and long windows onto the
shared ``Usage`` shape consumed by ``limits.py``.

The app-server process uses the Codex CLI's existing authentication.  No model
turn is started and no prompt tokens are consumed.  Query failures degrade to
an empty snapshot so a temporary CLI/auth problem does not make the loop crash.
"""

import atexit
import json
import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, List, Optional

from . import wire
from .console import print_line
from .usage import EMPTY_READING, EMPTY_USAGE, Usage, UsageReading, summary_line

# The bound on one request — the handshake of a fresh server included. A wait
# that runs out kills the server; the next query starts another.
APP_SERVER_TIMEOUT = 15.0
USAGE_CACHE_TTL = 30.0
LONG_WINDOW_MINUTES = 24 * 60
# A quota server older than this is replaced at its next query, so a process
# that has drifted (a refreshed login, a CLI upgraded underneath the run) is
# not consulted for the whole of a multi-day run. A restart costs one
# handshake: on Codex CLI 0.159.2 a one-shot query took 0.610–1.463 s (median
# 0.696 s) against 0.419–0.589 s (median 0.486 s) for a read through a live
# server (5 + 5 runs, measured 2026-10-01), so an hourly restart is noise.
APP_SERVER_MAX_AGE = 60 * 60.0

# Why not the official `codex app-server daemon` + `codex app-server proxy`
# (checked on codex-cli 0.160.0, 2026-10-02): the daemon is ONE per user
# (`~/.codex/app-server-control/app-server-control.sock`), started by `daemon
# start` and outliving whoever started it. The loop would either share it with
# the user's IDE and other loops — and could then neither restart it on a hang
# nor stop it on exit without breaking them (`daemon update`/`restart` "may
# interrupt running work") — or leave it running after the run, which is the
# orphan this source must not leave. And `proxy` is itself a `codex` process
# per connection, so reaching the daemon through it still pays a CLI start per
# query unless the proxy is kept alive — which is the private process below
# with an extra hop. Hence a private `codex app-server` owned by the source.


def _reading(entry, *, reached: bool = False) -> UsageReading:
    if not isinstance(entry, dict):
        return EMPTY_READING
    percent = entry.get("usedPercent")
    percent = float(percent) if isinstance(percent, (int, float)) else None
    if reached:
        percent = max(100.0, percent or 0.0)
    reset = entry.get("resetsAt")
    reset_ts = float(reset) if isinstance(reset, (int, float)) else None
    return UsageReading(percent, reset_ts)


def _prefer_higher(left: UsageReading, right: UsageReading) -> UsageReading:
    """Conservatively keep the more-used reading when two windows share a slot."""
    if left.percent is None:
        return right
    if right.percent is None:
        return left
    return right if right.percent > left.percent else left


def parse_rate_limits(data: dict) -> Usage:
    """Map one app-server ``account/rateLimits/read`` result to shared quotas.

    Codex calls the returned windows ``primary`` and ``secondary`` rather than
    assigning fixed meanings to them.  Their duration is authoritative: a
    window shorter than one day is the session reading, while a day-or-longer
    window is the weekly/long-term reading.  This also handles plans like the
    current weekly-only plan, where ``primary`` itself is seven days.
    """
    if not isinstance(data, dict):
        return EMPTY_USAGE
    bucket = data.get("rateLimits")
    if not isinstance(bucket, dict):
        return EMPTY_USAGE

    reached_type = str(bucket.get("rateLimitReachedType") or "").lower()
    session = EMPTY_READING
    week = EMPTY_READING
    for name in ("primary", "secondary"):
        entry = bucket.get(name)
        if not isinstance(entry, dict):
            continue
        duration = entry.get("windowDurationMins")
        is_long = (isinstance(duration, (int, float))
                   and duration >= LONG_WINDOW_MINUTES)
        # Older/partial servers may omit the duration.  Preserve the traditional
        # primary=session, secondary=long-window ordering in that case.
        quota = "week" if is_long or (duration is None and name == "secondary") else "session"
        reading = _reading(entry, reached=name in reached_type)
        if quota == "week":
            week = _prefer_higher(week, reading)
        else:
            session = _prefer_higher(session, reading)

    if bucket.get("spendControlReached"):
        # A spend control is a hard account wall even when the service omitted
        # the usual percentage.  Attach it to an available window so the normal
        # policy waits/rechecks instead of repeatedly starting doomed turns.
        if week.percent is not None:
            week = UsageReading(max(100.0, week.percent), week.reset_ts)
        else:
            session = UsageReading(max(100.0, session.percent or 0.0),
                                   session.reset_ts)

    summary = []
    if session.percent is not None:
        summary.append(summary_line("Current session", session))
    if week.percent is not None:
        summary.append(summary_line("Current week (all models)", week))
    return Usage(session, week, EMPTY_READING, summary)


class _ServerClosed(RuntimeError):
    """The quota server's stdout ended (EOF) before the awaited reply."""


class _ServerError(RuntimeError):
    """The quota server answered with an error: it is alive and kept."""


_EOF = object()


class _QuotaServer:
    """One live `codex app-server` and the thread pumping its stdout.

    The pump turns the blocking `readline` into a queue the caller can wait on
    with a deadline — the only way to bound a read from a pipe that is
    portable to Windows. stderr is merged into stdout; lines that are not JSON
    are skipped.
    """

    def __init__(self, argv: List[str]):
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )
        self.started = time.monotonic()
        self._lines: "queue.Queue[object]" = queue.Queue()
        self._pump = threading.Thread(target=self._read_stdout,
                                      name="codex-quota-stdout", daemon=True)
        self._pump.start()

    def _read_stdout(self) -> None:
        try:
            for line in self.proc.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass                    # stdout closed under us by `stop`
        finally:
            self._lines.put(_EOF)

    def write(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()

    def read_response(self, request_id: int, deadline: float) -> dict:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("codex app-server request timed out")
            try:
                line = self._lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError("codex app-server request timed out") from None
            if line is _EOF:
                self._lines.put(_EOF)   # every later read sees it too
                raise _ServerClosed("codex app-server closed before replying")
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(message, dict) or message.get("id") != request_id:
                continue
            if "error" in message:
                error = message.get("error") or {}
                detail = error.get("message") if isinstance(error, dict) else error
                raise _ServerError(str(detail or "unknown app-server error"))
            return message.get("result") or {}

    def stop(self) -> None:
        """Close stdin and reap; escalate to ending the tree, then kill.

        On Windows `proc` is usually the npm shim's `cmd.exe` and the CLI a
        grandchild, so the escalation goes through the turn process's
        tree-aware `ask_agent_process_to_end` (`taskkill /T`), not the
        shim-only `terminate`. A grandchild that survives anyway keeps the
        stdout write end: the pump then stays in `readline`, and closing
        stdout would block on the lock that read holds (19.0 s measured
        2026-10-02 against a 20 s grandchild) — so the stream is left to the
        daemon pump instead.
        """
        # Imported here: `providers` imports this module.
        from .providers import REAP_GRACE_S, ask_agent_process_to_end
        proc = self.proc
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=REAP_GRACE_S)
        except subprocess.TimeoutExpired:
            ask_agent_process_to_end(proc)
            try:
                proc.wait(timeout=REAP_GRACE_S)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proc.kill()
                    proc.wait(timeout=REAP_GRACE_S)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        except OSError:
            pass
        self._pump.join(timeout=REAP_GRACE_S)
        if not self._pump.is_alive():
            try:
                proc.stdout.close()
            except OSError:
                pass


# Every source that may own a live server, so the endings that never reach
# `runlifecycle.close_run` (see its docstring) still reap them at interpreter
# exit. Strong, not weak: a source an unwinding exception dropped without
# `close` leaves its server and pump running, and a weak entry would vanish
# before the hook that exists to reap exactly that server. `close` removes it.
_open_sources: "set[CodexUsageSource]" = set()


@atexit.register
def _close_open_sources() -> None:
    for source in list(_open_sources):
        source.close()


def _no_figures(reason) -> None:
    print_line(f"  · no Codex usage figures: {reason}")


def _default_argv() -> List[str]:
    return [shutil.which("codex") or "codex", "app-server"]


class CodexUsageSource:
    """Query and cache Codex rate limits without starting a model turn.

    Keeps one private quota server (`_QuotaServer`) for its lifetime rather
    than starting one per cache miss. It is never the turn's app-server: that
    process's stdout belongs to the renderer, while the quota gate and the
    status line's `QuotaRefresher` read from other threads at any time.

    Queries are serialized by one lock, cache check included, so a gate that
    waited behind the refresher's read takes the figures that read produced.
    Request ids grow monotonically for the source's whole life, across server
    restarts, so a late reply to an abandoned request can never answer a newer
    one. A timeout kills the server; EOF or a broken pipe on a reused server is
    retried once on a fresh one (the server died while idle); a server older
    than `max_age` is replaced. `close` stops the server for good — a query
    after it reads nothing rather than starting a process nobody will reap.
    """

    def __init__(self, cache_ttl: float = USAGE_CACHE_TTL,
                 timeout: float = APP_SERVER_TIMEOUT,
                 max_age: float = APP_SERVER_MAX_AGE,
                 argv: Optional[Callable[[], List[str]]] = None):
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self.max_age = max_age
        self._argv = argv or _default_argv
        self._cached: Optional[Usage] = None
        self._cached_ts = 0.0
        self._lock = threading.Lock()
        self._server: Optional[_QuotaServer] = None
        self._next_id = 0
        self._closed = False
        _open_sources.add(self)

    def _request_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    def _drop_server(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.stop()

    def _live_server(self, deadline: float) -> _QuotaServer:
        """The current server, after a max-age replacement or a first start."""
        server = self._server
        if server is not None and (server.proc.poll() is not None or
                                   time.monotonic() - server.started >= self.max_age):
            self._drop_server()
            server = None
        if server is None:
            server = _QuotaServer(self._argv())
            self._server = server
            try:
                request_id = self._request_id()
                server.write(wire.codex_app_initialize(request_id))
                server.read_response(request_id, deadline)
                server.write(wire.codex_app_initialized())
            except BaseException:
                self._drop_server()
                raise
        return server

    def _query_locked(self) -> Optional[dict]:
        deadline = time.monotonic() + self.timeout
        for attempt in (1, 2):
            previous = self._server
            try:
                server = self._live_server(deadline)
                # A server started by this very call (first, dead or over-age)
                # gets no retry: only one that died while idle does.
                fresh = server is not previous
            except (FileNotFoundError, PermissionError) as exc:
                return _no_figures(f"could not start 'codex app-server' ({exc})")
            except (OSError, RuntimeError, ValueError) as exc:
                return _no_figures(exc)
            try:
                request_id = self._request_id()
                server.write(wire.codex_app_rate_limits_read(request_id))
                return server.read_response(request_id, deadline)
            except (BrokenPipeError, _ServerClosed) as exc:
                self._drop_server()
                if fresh or attempt == 2:
                    return _no_figures(exc)
                # A reused server that died while idle: one fresh retry.
            except _ServerError as exc:
                return _no_figures(exc)
            except (OSError, ValueError) as exc:
                self._drop_server()   # timeout or pipe trouble: restart
                return _no_figures(exc)
        return None

    def get_usage(self, cache_value: bool = True) -> Usage:
        with self._lock:
            now = time.time()
            if (cache_value and self._cached is not None
                    and now - self._cached_ts < self.cache_ttl):
                return self._cached
            if self._closed:
                # A status-line poll racing the epilogue keeps the last figures.
                return self._cached or EMPTY_USAGE
            data = self._query_locked()
            if data is None:
                return EMPTY_USAGE
            snapshot = parse_rate_limits(data)
            self._cached = snapshot
            self._cached_ts = now
            return snapshot

    def invalidate(self) -> None:
        with self._lock:
            self._cached = None
            self._cached_ts = 0.0

    def close(self) -> None:
        """Stop the quota server; later queries read nothing. Idempotent.

        Waits for a query in flight (bounded by `timeout`) rather than
        killing the process under it.
        """
        with self._lock:
            self._closed = True
            self._drop_server()
        _open_sources.discard(self)
