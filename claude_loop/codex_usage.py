"""Read Codex account rate limits through the official app-server protocol.

``codex app-server`` exposes a JSONL/JSON-RPC method named
``account/rateLimits/read``.  This module performs the required initialization
handshake, requests that report, and maps its short and long windows onto the
shared ``Usage`` shape consumed by ``limits.py``.

The app-server process uses the Codex CLI's existing authentication.  No model
turn is started and no prompt tokens are consumed.  Query failures degrade to
an empty snapshot so a temporary CLI/auth problem does not make the loop crash.
"""

import json
import shutil
import subprocess
import threading
import time
from typing import Optional

from .usage import Usage, UsageReading, _EMPTY_READING, _EMPTY_USAGE, _summary_line

APP_SERVER_TIMEOUT = 15.0
USAGE_CACHE_TTL = 30.0
LONG_WINDOW_MINUTES = 24 * 60


def _reading(entry, *, reached: bool = False) -> UsageReading:
    if not isinstance(entry, dict):
        return _EMPTY_READING
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
        return _EMPTY_USAGE
    bucket = data.get("rateLimits")
    if not isinstance(bucket, dict):
        return _EMPTY_USAGE

    reached_type = str(bucket.get("rateLimitReachedType") or "").lower()
    session = _EMPTY_READING
    week = _EMPTY_READING
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
        summary.append(_summary_line("Current session", session))
    if week.percent is not None:
        summary.append(_summary_line("Current week (all models)", week))
    return Usage(session, week, _EMPTY_READING, summary)


class CodexUsageSource:
    """Query and cache Codex rate limits without starting a model turn."""

    def __init__(self, cache_ttl: float = USAGE_CACHE_TTL,
                 timeout: float = APP_SERVER_TIMEOUT):
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._cached: Optional[Usage] = None
        self._cached_ts = 0.0

    @staticmethod
    def _write(proc, message: dict) -> None:
        proc.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        proc.stdin.flush()

    @staticmethod
    def _read_response(proc, request_id: int, timed_out: threading.Event) -> dict:
        for line in proc.stdout:
            try:
                message = json.loads(line)
            except (TypeError, ValueError):
                continue
            if message.get("id") != request_id:
                continue
            if "error" in message:
                error = message.get("error") or {}
                detail = error.get("message") if isinstance(error, dict) else error
                raise RuntimeError(str(detail or "unknown app-server error"))
            return message.get("result") or {}
        if timed_out.is_set():
            raise TimeoutError("codex app-server request timed out")
        raise RuntimeError("codex app-server closed before replying")

    def query_rate_limits_json(self) -> Optional[dict]:
        executable = shutil.which("codex") or "codex"
        try:
            proc = subprocess.Popen(
                [executable, "app-server"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1,
            )
        except (FileNotFoundError, OSError) as exc:
            print(f"  · no Codex usage figures: could not start 'codex app-server' ({exc})")
            return None

        timed_out = threading.Event()

        def kill_on_timeout():
            timed_out.set()
            try:
                proc.kill()
            except OSError:
                pass

        timer = threading.Timer(self.timeout, kill_on_timeout)
        timer.daemon = True
        timer.start()
        try:
            self._write(proc, {
                "method": "initialize",
                "id": 0,
                "params": {"clientInfo": {
                    "name": "claude_loop",
                    "title": "claude-loop",
                    "version": "1.0.0",
                }},
            })
            self._read_response(proc, 0, timed_out)
            self._write(proc, {"method": "initialized", "params": {}})
            self._write(proc, {"method": "account/rateLimits/read", "id": 1})
            return self._read_response(proc, 1, timed_out)
        except (BrokenPipeError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
            print(f"  · no Codex usage figures: {exc}")
            return None
        finally:
            timer.cancel()
            try:
                proc.stdin.close()
                proc.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                try:
                    if proc.poll() is None:
                        proc.terminate()
                    proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        proc.kill()
                    except OSError:
                        pass

    def get_usage(self, cache_value: bool = True) -> Usage:
        now = time.time()
        if (cache_value and self._cached is not None
                and now - self._cached_ts < self.cache_ttl):
            return self._cached
        data = self.query_rate_limits_json()
        if data is None:
            return _EMPTY_USAGE
        snapshot = parse_rate_limits(data)
        self._cached = snapshot
        self._cached_ts = now
        return snapshot

    def invalidate(self) -> None:
        self._cached = None
        self._cached_ts = 0.0
