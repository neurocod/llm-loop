"""When the SERVICE dropped a turn, and whether it is reachable right now.

Two questions that belong together because one is only asked to answer the
other: `turn.run_turn` resends a turn the service dropped, and it asks here both
"was that the service's fault" (`is_service_failure`) and "is there any point
trying again" (`api_reachable`). Nothing here imports a sibling module, so it
stays below `turn` and `events` in the package.

The other half of that policy — how long to wait, how many times, and what the
resent turn says — is `turn.SERVICE_RETRY_PAUSES_S` and its neighbours, because
those are bounded by the turn's own timeout and read by the loop that runs it.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

# The closed set of `codexErrorInfo` values, read from the schema bundle
# `codex app-server generate-json-schema --out <dir>` (`CodexErrorInfo` in
# `codex_app_server_protocol.v2.schemas.json`, codex-cli 2026-09-13) rather than
# guessed. Two spellings share it: thirteen are bare strings, five are
# single-key objects (four carrying an `httpStatusCode`, `activeTurnNotSteerable`
# a `turnKind`) — hence `error_kind`. All eighteen are pinned by the gate, so a
# member added upstream is a red test rather than a failure silently reclassed
# as "not ours to retry".
#
# In THIS set are the failures that are the service's own and that a resend of
# the same thread can survive. Deliberately NOT in it, because a resend either
# cannot help or makes things worse:
#
#   * `contextWindowExceeded`, `sessionBudgetExceeded` — the thread is already
#     too big, and a resume sends all of it again;
#   * `usageLimitExceeded`, `rateLimitExceeded` — an account limit, not an
#     outage: it outlives the minutes this retries over, and each attempt costs
#     a full thread resend (1.2-1.7M input tokens on a worker, measured);
#   * `unauthorized`, `badRequest`, `activeTurnNotSteerable` — our request or
#     our credentials, identical on the next try;
#   * `cyberPolicy`, `misalignmentPolicyViolation` — a refusal, not a fault;
#   * `sandboxError`, `threadRollbackFailed` — local, and this machine's
#     sandbox is exactly as broken a minute later.
#
# The weakest member is `serverOverloaded`, and this repo's own logs say so:
# `reviewBfix` was told "Selected model is at capacity" at 04:06:14 and again,
# identically, 50 s later — inside what is now the first pause, and on the same
# model, since `sandbox.thread_params` resends `record.model`. It stays in the set
# because the second slot is three minutes out and capacity does come back, but
# it is the one kind whose two retries can be expected to buy nothing.
SERVICE_FAILURE_KINDS = frozenset({
    "serverOverloaded",                 # "Selected model is at capacity", 4x in the logs
    "internalServerError",
    "httpConnectionFailed",
    "responseStreamConnectionFailed",
    "responseStreamDisconnected",
    "responseTooManyFailedAttempts",
})

# What the 2026-09-13 outage actually put on the wire, and the reason this is a
# text match at all. While the server was still reconnecting, its errors carried
# `{"responseStreamDisconnected": …}` and `willRetry: true`; the FINAL error —
# the one that fails the turn — degraded to the catch-all `"other"` with the
# reconnect count gone and the sentence kept:
#
#   "stream disconnected before completion: Unable to verify model access right
#    now. Please retry."
#
# So the only thing separating that from any other `other` is the message. The
# match is on this fragment rather than the whole sentence because the tail
# after the colon is the upstream's wording of the day; the tests'
# `LOGGED_DROP` pins the whole line as it was logged.
SERVICE_FAILURE_TEXTS = ("stream disconnected",)


def retried_note(count: int) -> str:
    """The one wording for "this turn was resent N times".

    One function because three places tell the same operator the same fact —
    the summary line of a finished turn, the `status` line of a running one,
    and the failure message itself — and they were drifting apart the moment
    they were written (`retried 2× after service drop` against `retried 2x
    after a service drop`). A caller grepping either spelling would miss the
    others.
    """
    return f"retried {count}× after a service drop"


def error_kind(error: object) -> str:
    """The `codexErrorInfo` of a `TurnError` as one token, or "".

    Both wire spellings collapse here: `"serverOverloaded"` and
    `{"responseStreamDisconnected": {"httpStatusCode": null}}` become the same
    kind of string, so callers match against one set instead of two.
    """
    if not isinstance(error, dict):
        return ""
    info = error.get("codexErrorInfo")
    if isinstance(info, str):
        return info
    if isinstance(info, dict) and len(info) == 1:
        return next(iter(info))
    return ""


def is_service_failure(kind: str, message: str) -> bool:
    """Was this failure the service's, rather than the task's or the sandbox's?

    The question a retry hangs on, so it answers "no" to everything it does not
    recognise: a wrong "no" costs the operator one `send` typed by hand (what
    they do today), a wrong "yes" burns a whole thread resend on a failure that
    will repeat.
    """
    if kind in SERVICE_FAILURE_KINDS:
        return True
    lowered = message.lower()
    return kind == "other" and any(text in lowered for text in SERVICE_FAILURE_TEXTS)


def is_network_wait(params: dict) -> bool:
    """A retrying disconnected stream explicitly waiting for the local network.

    `Reconnecting... 5/5` alone is also emitted for service faults; a 429 is
    another retry the CLI owns. Neither suspends the attempt's work budget.
    Shape measured in worker-c/events.jsonl, 2026-11-07.
    """
    error = params.get("error") or {}
    return (bool(params.get("willRetry"))
            and error_kind(error) == "responseStreamDisconnected"
            and "waiting for network" in str(error.get("message", "")).lower())


# All seven .codex-agents/*/events.jsonl with actual network waits, scanned
# 2026-11-07 (218 notifications): rev-e, rev-d, worker-a, worker-b,
# worker-c, worker-d, worker-f. Largest heartbeat gap: rev-d, 182.785 s;
# last heartbeat to progress: rev-d 77.717 s, worker-f 75.573 s,
# worker-b 68.559 s. Round UP twice the largest of both measurements.
# This bounds credited silence, never the lifetime of the attempt itself.
NETWORK_WAIT_GAP_S = 366.0
# A separate caller-wide wall-clock allowance, not a work timeout. Two hours
# covers over twice the measured 31-minute outage; resends share this ceiling.
MAX_NETWORK_WAIT_S = 2 * 60 * 60.0
# No child-only outage duration has been measured. Conservatively allow two
# full freshness windows (2 * 366 = 732 s, from the measurements above) across
# the run; child heartbeats must not turn a working main thread's budget into
# the two-hour main-outage allowance. Main-thread waits keep their own credit.
MAX_CHILD_NETWORK_WAIT_S = 2 * NETWORK_WAIT_GAP_S
# Credit is summed as differences of monotonic readings, which loses ULPs: a
# wait that met the ceiling summed to 99.99999999999999 of 100 (ubuntu CI
# 2026-11-08). Short of the clamp, `deadline` skipped the terminal exhaustion
# and its `now + remaining` rounded to `now`: a RETRYABLE timeout instead.
# A microsecond is far over that drift and far under any real wait.
CEILING_SLACK_S = 1e-6


class NetworkWaitTimeout(TimeoutError):
    """An outage outliving the work budget, or its total allowance exhausted."""

    def __init__(self, seconds: float, *, exhausted: bool) -> None:
        self.exhausted = exhausted
        suffix = "wait limit reached" if exhausted else "attempt budget expired"
        super().__init__(f"no network for {seconds / 60:.1f} min ({suffix})")


@dataclass
class NetworkWait:
    """One monotonic wait account shared by all threads of this run.

    Progress or a non-network error clears only its own thread's wait;
    telemetry proves no recovery. Stale waits stop earning credit, but their
    thread ids remain pending until recovery, to classify an eventual timeout.
    `total_s` includes earlier attempts; child-only credit is capped separately.
    `started` and `tries` describe only the current continuous wait.
    No credit extends beyond the last heartbeat
    plus NETWORK_WAIT_GAP_S, even when the event loop wakes up late.
    """
    total_s: float = 0.0
    started: Optional[float] = None
    last: float = 0.0
    tries: int = 0
    since: str = ""
    pending: dict[str, float] = field(default_factory=dict)
    main_thread: str = ""
    child_s: float = 0.0
    credited_at: Optional[float] = None

    def seconds(self, now: float) -> float:
        current = (max(0.0, min(now, self.last + NETWORK_WAIT_GAP_S) - self.credited_at)
                   if self.credited_at is not None else 0.0)
        if self.main_thread not in self.pending:
            current = min(current, max(0.0, MAX_CHILD_NETWORK_WAIT_S - self.child_s))
        total = self.total_s + current
        return MAX_NETWORK_WAIT_S if total >= MAX_NETWORK_WAIT_S - CEILING_SLACK_S else total

    def _accrue(self, now: float) -> None:
        credited = self.seconds(now)
        if self.main_thread not in self.pending:
            self.child_s += credited - self.total_s
        self.total_s = credited
        if self.started is not None:
            self.credited_at = now

    def stop(self, now: float) -> None:
        self._accrue(now)
        self.started = None
        self.credited_at = None

    def observe(self, message: dict, now: float, stamp: str,
                main_thread: str = "") -> bool:
        """Fold a run's event; return whether the visible state changed."""
        self.main_thread = main_thread or self.main_thread
        # Settle the previous interval before an event changes which threads
        # are waiting, so recovering the main thread cannot retract its credit.
        self._accrue(now)
        method = str(message.get("method") or "")
        thread = str((message.get("params") or {}).get("threadId") or main_thread)
        if method == "error" and is_network_wait(message.get("params") or {}):
            if self.started is not None and now - self.last > NETWORK_WAIT_GAP_S:
                self.stop(now)
            if self.started is None:
                self.started, self.since, self.tries = now, stamp, 0
                self.credited_at = now
            self.last = now
            self.pending[thread] = now
            self.tries += 1
            return True
        if thread in self.pending and (
                method == "error" or method.startswith(("item/", "turn/"))):
            del self.pending[thread]
            if not self.pending:
                self.stop(now)
            else:
                latest = max(self.pending.values())
                if now >= latest + NETWORK_WAIT_GAP_S:
                    # The recovering thread kept this interval fresh until
                    # now. Do not backdate its credit to an older thread.
                    self.stop(now)
                self.last = latest
            return True
        return False

    def deadline(self, base: float, baseline_s: float, now: float) -> float:
        """The ONE deadline calculation for both the loop and Queue.get.

        While reconnects are fresh, project credit to the next heartbeat
        boundary: an empty queue must not expire on the unextended deadline.
        On progress, only time actually waited remains credited. The ceiling
        is terminal while still waiting. A missing heartbeat freezes credit;
        it is retryable only if the main thread is pending when its budget
        runs out. A child-only wait gives bounded credit, never a resend.
        """
        spent = self.seconds(now)
        main_waiting = self.main_thread in self.pending
        if spent >= MAX_NETWORK_WAIT_S and self.started is not None and main_waiting:
            raise NetworkWaitTimeout(spent, exhausted=True)
        if self.started is not None:
            if now >= self.last + NETWORK_WAIT_GAP_S:
                self.stop(now)
        extended = base + spent - baseline_s
        if now >= extended and main_waiting:
            raise NetworkWaitTimeout(spent, exhausted=False)
        if self.started is not None:
            remaining = MAX_NETWORK_WAIT_S - spent
            if not main_waiting:
                child_spent = self.child_s + spent - self.total_s
                remaining = min(remaining, max(0.0, MAX_CHILD_NETWORK_WAIT_S - child_spent))
            if remaining <= 0:
                return extended
            fresh_until = min(self.last + NETWORK_WAIT_GAP_S,
                              now + remaining)
            return min(extended + fresh_until - now, fresh_until)
        return extended


def reconnecting_note(since: str, tries: int) -> str:
    return f"RECONNECTING since {since[11:16]} ({tries} tries)"


def network_wait_note(seconds: float) -> str:
    return f"waited {seconds / 60:.1f} min for network"


# What is asked before each retry, and why an unauthenticated GET is enough: a
# reply of ANY kind proves the machine can reach OpenAI, and that is the whole
# question — entitlement is what the turn itself is about to test. Measured
# 2026-09-13 with no credentials of any sort in the request: this URL answers
# 405 in 0.11-0.19 s, `https://api.openai.com/v1/models` answers 401 in 0.24 s.
#
# THIS host because the Codex CLI here authenticates as ChatGPT (`auth_mode` in
# `~/.codex/auth.json`), so this is the host its turns actually talk to. A
# setup running on an API key talks to `api.openai.com` instead, and this
# constant is then the one line to move.
#
# What the check is NOT: a detector of the outage it retries over. On
# 2026-09-13 the stream dropped while the endpoint plainly answered — five
# reconnects reached it and failed. It exists for the cheaper case the user
# named: the machine's own link is down, so the attempt is not worth making.
API_PROBE_URL = "https://chatgpt.com/backend-api/codex/responses"

# Long enough that a slow-but-alive endpoint is not called dead, short beside
# the pause it follows (`turn.SERVICE_RETRY_PAUSES_S`, 60 s at the shortest).
API_PROBE_TIMEOUT_S = 10.0


def api_reachable(url: str = API_PROBE_URL,
                  timeout: float = API_PROBE_TIMEOUT_S) -> bool:
    """True when OpenAI answered anything at all — 401 and 405 included.

    A 5xx is the one HTTP answer that counts as unreachable: it is the service
    saying it cannot serve, which is what the caller is waiting out. Everything
    below the HTTP layer (DNS, TLS, a refused connection, the timeout) is a
    plain "no".
    """
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, method="GET"), timeout=timeout) as reply:
            return int(getattr(reply, "status", 0) or 0) < 500
    except urllib.error.HTTPError as answered:
        # A subclass of URLError, so it must be caught first: an HTTP status IS
        # an answer, and an unauthenticated probe expects one in the 400s.
        return answered.code < 500
    except (urllib.error.URLError, OSError, ValueError):
        return False
