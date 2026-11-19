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
import shutil
import subprocess
import threading
import time
from typing import Callable, List, Optional

from . import appserver, stopclock, wire
from .console import print_line
from .procend import REAP_GRACE_S, end_process_tree
from .usage import EMPTY_READING, EMPTY_USAGE, Usage, UsageReading, summary_line

# The bound on one request — the handshake of a fresh server included. A wait
# that runs out kills the server; the next query starts another. Inside an
# ending it is capped by what the ending's budget has left (`stopclock`), the
# wait for a query in flight on another thread included.
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


class _QuotaServer:
    """One live `codex app-server`, read through `appserver.Connection`.

    stderr is merged into stdout; whatever is not the awaited response — a
    line that is not JSON, a notification, a request of the server's own,
    even one carrying the awaited id — is skipped. The transport's failures
    are translated into the ones `CodexUsageSource` acts on, with the words
    the status line has always printed.
    """

    def __init__(self, argv: List[str], first_id: int):
        self.connection = appserver.Connection(
            appserver.spawn(argv, merge_stderr=True), first_id=first_id)
        self.proc = self.connection.proc
        self.started = time.monotonic()

    @property
    def next_id(self) -> int:
        return self.connection.next_id

    def send(self, message: dict) -> None:
        self.connection.send(message)

    def request(self, body: dict, deadline: float) -> dict:
        """Send `body` (numbered by the connection) and return its result."""
        try:
            return self.connection.request(body, deadline - time.monotonic())
        except TimeoutError:
            raise TimeoutError("codex app-server request timed out") from None
        except appserver.Closed:
            raise _ServerClosed("codex app-server closed before replying") from None
        except appserver.RpcError as exc:
            raise _ServerError(exc.detail or "unknown app-server error") from None

    def stop(self) -> None:
        """Close stdin and give the server `REAP_GRACE_S` to exit on it; then
        end the tree.

        On Windows `proc` is usually the npm shim's `cmd.exe` and the CLI a
        grandchild, so the escalation goes through the shared tree-aware
        `procend.end_process_tree` (`taskkill /T`), not the shim-only
        `terminate`. stdout is the connection's pump's to close
        (`appserver.Connection._pump_stdout`).
        """
        proc = self.proc
        self.connection.close_input()
        try:
            proc.wait(timeout=stopclock.within(REAP_GRACE_S))
        except subprocess.TimeoutExpired:
            end_process_tree(proc)
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
        # Under the deadline of the ending that already gave this close up,
        # if one did: interpreter exit is that ending's last step, not a new
        # unbounded wait for the query it would not wait for.
        with stopclock.ending(source._close_deadline):
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
        # Guards the close mark against the lock's release (`_release`).
        self._handoff = threading.Lock()
        # Stops a holder has begun and not finished, and their end's signal:
        # the source leaves `_open_sources` only once this is back to zero.
        self._stopping = 0
        self._settled = threading.Condition(self._handoff)
        # The deadline of the ending whose `close` could not wait, kept for
        # the exit hook's close; None when no close has had to give up.
        self._close_deadline: Optional[float] = None
        self._server: Optional[_QuotaServer] = None
        self._next_id = 0
        self._closed = False
        _open_sources.add(self)

    def _request(self, server: _QuotaServer, body: dict, deadline: float) -> dict:
        """`server.request`, the source's id counter kept in step with it."""
        try:
            return server.request(body, deadline)
        finally:
            self._next_id = server.next_id

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
            server = _QuotaServer(self._argv(), first_id=self._next_id)
            self._server = server
            try:
                self._request(server, wire.codex_app_initialize(None), deadline)
                server.send(wire.codex_app_initialized())
            except BaseException:
                self._drop_server()
                raise
        return server

    def _query_locked(self) -> Optional[dict]:
        if stopclock.spent():
            # No time left to start a server, let alone hear it answer.
            return _no_figures("the stop budget is spent")
        deadline = time.monotonic() + stopclock.within(self.timeout)
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
                return self._request(
                    server, wire.codex_app_rate_limits_read(None), deadline)
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

    def _acquire(self) -> bool:
        """Take the lock — blocking outside an ending, inside one for at most
        what its budget has left: the holder is a query on another thread
        (the status line's refresher), bounded only by its own `timeout`."""
        wait = stopclock.within(None)
        if wait is None:
            return self._lock.acquire()
        return self._lock.acquire(timeout=wait)

    def _release(self) -> None:
        """Release the lock — and, once the source is closed, take its server
        and stop it: EVERY path that held the lock comes here, so a `close`
        that could not wait for the holder hands the stop to it.

        The hand-off is atomic: `close` marks the source closed under
        `_handoff`, and the holder reads that mark and lets the lock go under
        it too. Either the holder sees the mark and stops the server, or it
        had let go before the mark and `close` then gets the lock itself;
        nothing slips between a check and a release.
        """
        with self._handoff:
            closed = self._closed
            server = None
            if closed and self._server is not None:
                server, self._server = self._server, None
                # Counted BEFORE the lock goes: a close that gets the lock
                # next must see a stop still under way, not "no server".
                self._stopping += 1
            self._lock.release()
        if server is None:
            return
        try:
            server.stop()
        finally:
            with self._handoff:
                self._stopping -= 1
                self._settled.notify_all()
                if self._stopping == 0:
                    _open_sources.discard(self)

    def get_usage(self, cache_value: bool = True) -> Usage:
        if not self._acquire():
            _no_figures("the stop budget ran out waiting for a query in flight")
            return self._cached or EMPTY_USAGE
        try:
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
        finally:
            self._release()

    def invalidate(self) -> None:
        self._lock.acquire()
        try:
            self._cached = None
            self._cached_ts = 0.0
        finally:
            self._release()

    def close(self) -> None:
        """Stop the quota server; later queries read nothing. Idempotent.

        Waits for a query in flight (bounded by `timeout`) rather than
        killing the process under it — inside an ending, for no longer than
        its budget has left. A close that cannot wait leaves the stop to the
        holder (`_release`) and keeps the ending's deadline, so the exit hook
        that closes it again does not wait past that ending either. Should
        the holder be a daemon the exit freezes first, the server still ends:
        its stdin closes with this process, and it exits on EOF
        (`_QuotaServer.stop`).
        """
        with self._handoff:
            self._closed = True
        if not self._acquire():
            self._close_deadline = stopclock.deadline()
            return
        self._release()
        # A stop a holder began is not done because the server is detached:
        # wait for it (within the ending), and leave the registry to it.
        with self._handoff:
            settled = self._settled.wait_for(lambda: self._stopping == 0,
                                             stopclock.within(None))
            if settled:
                _open_sources.discard(self)
        if not settled:
            self._close_deadline = stopclock.deadline()
