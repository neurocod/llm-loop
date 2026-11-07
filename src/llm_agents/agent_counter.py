"""Telling the on-screen counter that this process is a running agent.

A background agent is invisible: the turn takes minutes, the chat that launched
it has moved on, and nothing on screen says work is still happening. `AgentCounter`
(a badge with provider icons and counts at the top, `AgentCounter/`) answers that, and this
module is the half that reports.

The report is the CONNECTION, not a message. This process opens the counter's
named pipe when the turn starts and holds it until the turn ends; the count is
the number of open connections. That choice is the whole design, and it buys
three things the obvious alternatives do not:

  * no goodbye to trust. `taskkill /F` (which this package itself uses), an
    unhandled exception, a hung turn killed by hand — every one of them closes
    the handle, because the kernel closes it, and the counter sees a broken pipe;
  * no pid to watch, and so no pid-reuse race and no polling timer;
  * no files. Nothing is written anywhere, so nothing is left behind to go stale
    and nothing has to be swept up after a crash.

What it costs, stated plainly: if the counter process itself is killed while
agents are running, they are not re-counted — after its hello this side speaks
only to repeat it with a new weekly-limit figure (`Beacon`), and a heartbeat
would need a thread it does not otherwise have. The badge is click-through and
not in the taskbar, so that means someone went looking for it in a task manager.

Nothing here may break a turn. Every failure is swallowed; the only trace is one
line on stderr, once per process, because a counter that silently never counts is
worse than no counter at all.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from . import host_policy


# The pipe name, spelled a SECOND time in `AgentCounter/Registry.cpp`. A
# disagreement is invisible at runtime — this side finds no server, starts one,
# and still cannot reach it — so `test_beacon_pipe_name_matches_the_counter`
# reads both files and refuses a mismatch. Change one, change the other.
PIPE_PREFIX = "multiskill-agents.v1."

# Set to 0 to keep the counter out of the way (the test suite does exactly that:
# a gate run must not put a window on the operator's screen).
DISABLE_ENV = "AGENT_COUNTER_ENABLED"

# Mirrored by Registry.cpp::normalizedAgentType; the shared gate compares both.
KNOWN_AGENT_TYPES = ("claude", "codex")

# Where the counter is looked for: this variable, else the host's
# `HostPolicy.counter_exes` under its root, most-preferred first. An unbuilt
# exe is not an error: this package works fine without a badge.
EXE_ENV = "AGENT_COUNTER_EXE"

# How long to keep trying to reach a counter we just started. Qt's start-up plus
# the pipe going up is well under a second; the rest is headroom for a machine
# busy with the very agents being counted.
SPAWN_WAIT_S = 3.0
RETRY_STEP_S = 0.05

# ERROR_PIPE_BUSY is NOT "nobody is there": the pipe exists and every instance
# is momentarily taken, and Windows prescribes waiting for one. Read as absence
# it starts a second counter, which can only lose the claim and exit — and if
# the real counter stays busy past the retry window, the turn goes uncounted for
# its whole life. So it is waited out where it happens.
WIN_ERROR_PIPE_BUSY = 231
BUSY_WAIT_S = 1.0

# Mirrors the counter's own cap, and not for tidiness: the write below is
# synchronous on a blocking pipe handle, so a label big enough to fill the pipe
# buffer would wait for the counter to read it — forever, if the counter is
# wedged. Bounded here, one hello can never exceed one buffer, and the only
# blocking call on a turn's path cannot block.
MAX_LABEL_CHARS = 120

# The hello's optional weekly-limit REMAINDER (0..100, whole percent), read by
# `Registry::handleLine` / `weekLeftFrom` in AgentCounter/Registry.cpp.
WEEK_LEFT_FIELD = "week_left"

_complained = False
# Provider types already told "no weekly figure" in this process: a worker
# runs turn after turn, and a field the provider dropped for good would
# otherwise add the same line to its log once per turn.
_week_complained: set[str] = set()


def _fold_user(raw: str) -> str:
    """The user name as it appears in the pipe name.

    ASCII-only and crude on purpose: `Registry.cpp` reproduces this character
    for character, and anything richer (Python's Unicode-aware `lower` against
    Qt's) could differ on a name neither author will ever type.
    """
    kept = [ch.lower() if "A" <= ch <= "Z" else ch
            for ch in raw
            if ("A" <= ch <= "Z") or ("a" <= ch <= "z") or ("0" <= ch <= "9")]
    return "".join(kept)[:32] or "anon"


def pipe_path() -> str:
    return r"\\.\pipe\%s%s" % (PIPE_PREFIX, _fold_user(os.environ.get("USERNAME", "")))


def counter_exe() -> Optional[Path]:
    override = os.environ.get(EXE_ENV)
    if override:
        path = Path(override)
        return path if path.is_file() else None
    for path in host_policy.counter_paths():
        if path.is_file():
            return path
    return None


def _complain(reason: str) -> None:
    """Say once, on stderr, that the badge will not be showing this agent.

    Once because a turn prints a report the operator reads, and a repeated
    diagnostic about a decoration would bury it. Not never, because a mismatched
    pipe name or a counter that refuses to start is exactly the failure this
    module cannot detect any other way.
    """
    global _complained
    if _complained:
        return
    _complained = True
    try:
        print(f"agent_counter: agent counter unavailable ({reason})", file=sys.stderr)
    except Exception:  # noqa: BLE001
        # Reporting must not become the failure it reports. This runs from
        # inside `announce`'s own except block, so a closed or broken stderr
        # here would raise straight past it and into the turn.
        pass


def _complain_week(agent_type: str, why: str) -> None:
    """One line, once per provider per process, when a turn ended without a
    weekly figure: the badge then shows the count alone, and this line is
    what tells a missing figure apart from a counter that drew nothing."""
    if agent_type in _week_complained:
        return
    _week_complained.add(agent_type)
    try:
        print(f"agent_counter: no weekly-limit figure for {agent_type} ({why}); "
              f"the badge shows the count alone", file=sys.stderr)
    except Exception:  # noqa: BLE001 - see `_complain`
        pass


def _line(message: dict) -> bytes:
    return json.dumps(message).encode("utf-8") + b"\n"


class Beacon:
    """What `announce` yields: the running agent's way to tell the badge how
    much of its provider's WEEKLY limit is left.

    The figure travels as the same hello repeated with `week_left` added — not
    as a new op, because a counter built before the field drops a client on an
    unknown op and would stop counting this agent (Registry.cpp, handleLine).

    Repeating a write on this blocking handle cannot stall the turn the way a
    single oversized hello could (`MAX_LABEL_CHARS`): Qt re-posts its pipe read
    from a thread-pool callback with no buffer cap (qwindowspipereader.cpp,
    `waitCallback`), so even a counter whose GUI thread is busy keeps draining.
    And only a CHANGED figure is written: Codex repeats its report every
    10–20 s (26 times in one turn of `.codex-agents/f1942`), and a turn should
    write a few lines, not one per provider notification.

    A disabled or unreachable counter yields a Beacon with no handle, on which
    every call is a no-op, so callers report without asking first.
    """

    def __init__(self, handle: Optional[object] = None, hello: Optional[dict] = None,
                 agent_type: str = "unknown") -> None:
        self._handle = handle
        self._hello = hello or {}
        self._agent_type = agent_type
        # `report_week_used` may run on a provider-stream thread while the
        # turn's own thread closes the handle in `announce`.
        self._lock = threading.Lock()
        self._sent: Optional[int] = None
        self._figures = 0
        self._why = ""

    def report_week_used(self, used: Optional[float], why: str = "") -> None:
        """The provider's weekly utilisation in PERCENT (both providers report
        it used, not left), or None with `why` when its report had no figure.

        The remainder is floored — the badge never shows more left than there
        is — after rounding away float noise, since a fraction times 100 can
        land a hair OVER a whole number: Claude's 0.55 becomes
        55.00000000000001, and a bare floor shows 44 % left instead of 45.
        A report without a figure changes nothing: a weekly remainder moves
        slowly, and one empty report must not blank a figure already shown.
        """
        if self._handle is None:
            return
        if (not isinstance(used, (int, float)) or isinstance(used, bool)
                or not math.isfinite(used)):
            self._why = why or "the provider's report carried no weekly figure"
            return
        left = min(100, max(0, math.floor(round(100.0 - float(used), 6))))
        with self._lock:
            self._figures += 1
            if self._handle is None or left == self._sent:
                return
            try:
                self._handle.write(_line(dict(self._hello, **{WEEK_LEFT_FIELD: left})))
                self._sent = left
            except Exception as exc:  # noqa: BLE001 - a decoration must not fail a turn
                _complain(str(exc))
                self._close_locked()

    def _close_locked(self) -> None:
        if self._handle is not None:
            try:
                self._handle.close()
            except OSError:
                pass
            self._handle = None

    def close(self) -> None:
        """Release the registration; say once if no weekly figure ever came.

        Only for the providers that do report one: an untyped sender has no
        source, and saying so every turn would be noise about nothing."""
        with self._lock:
            live = self._handle is not None
            self._close_locked()
        if live and self._figures == 0 and self._agent_type in KNOWN_AGENT_TYPES:
            _complain_week(self._agent_type,
                           self._why or "the provider stream carried none this turn")


def _connect() -> Optional[object]:
    """Open the counter's pipe, or None if nothing is listening.

    "Busy" and "absent" are answered differently on purpose — see
    WIN_ERROR_PIPE_BUSY. Only a pipe that is not there at all returns None
    quickly; a busy one is waited for, because starting a rival counter cannot
    help and losing the registration is the cost of getting it wrong.
    """
    deadline = time.monotonic() + BUSY_WAIT_S
    while True:
        try:
            return open(pipe_path(), "r+b", buffering=0)
        except FileNotFoundError:
            return None
        except OSError as exc:
            if getattr(exc, "winerror", None) != WIN_ERROR_PIPE_BUSY:
                raise
            if time.monotonic() >= deadline:
                return None
            time.sleep(RETRY_STEP_S)


def _spawn(exe: Path) -> None:
    # Detached and window-less: the counter outlives this turn on purpose (the
    # next agent should find it already up), and inheriting this process's
    # handles would keep the pipe of a finished turn open.
    flags = 0
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW
    subprocess.Popen([str(exe)], creationflags=flags, close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL)


def _open_or_start() -> Optional[object]:
    handle = _connect()
    if handle is not None:
        return handle
    exe = counter_exe()
    if exe is None:
        named = host_policy.counter_paths()
        _complain(f"{named[0].name} is not built" if named else
                  "no counter executable (the host policy names none)")
        return None
    try:
        _spawn(exe)
    except OSError as exc:
        _complain(f"cannot start {exe}: {exc}")
        return None
    deadline = time.monotonic() + SPAWN_WAIT_S
    while time.monotonic() < deadline:
        time.sleep(RETRY_STEP_S)
        handle = _connect()
        if handle is not None:
            return handle
    _complain(f"started {exe.name} but could not reach it on {pipe_path()}")
    return None


@contextmanager
def announce(label: str, *, agent_type: str = "unknown", disable_env: str | None = None) -> Iterator[Beacon]:
    """Count this process as one running agent for the length of the block.

    Counter setup/cleanup never fails the body; exceptions from the body propagate.
    AGENT_COUNTER_ENABLED=0 disables both providers; disable_env adds a provider
    switch such as CODEX_AGENTS_COUNTER or CLAUDE_AGENTS_COUNTER.
    agent_type is explicit wire metadata, independent of the human label.
    Unspecified/unrecognized types remain visible as an unbranded group.
    The yielded `Beacon` carries the weekly-limit figure, if the caller has one.
    """
    if os.name != "nt" or any(os.environ.get(key, "").strip() == "0"
                              for key in (DISABLE_ENV, disable_env) if key):
        yield Beacon()
        return
    handle = None
    hello = {"op": "hello", "label": label[:MAX_LABEL_CHARS],
             "agent_type": agent_type if agent_type in KNOWN_AGENT_TYPES else "unknown"}
    try:
        handle = _open_or_start()
        if handle is not None:
            handle.write(_line(hello))
    except Exception as exc:  # noqa: BLE001 - see the docstring: nothing propagates
        _complain(str(exc))
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass
            handle = None
    beacon = Beacon(handle, hello, hello["agent_type"])
    try:
        yield beacon
    finally:
        # Closing is what decrements the count. It happens here for a tidy exit
        # and in the kernel for every other kind, which is the point of holding
        # a handle rather than sending a message.
        beacon.close()
