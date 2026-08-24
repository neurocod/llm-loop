"""
usage.py - the "what does the account say" layer for Claude usage limits.

This module owns everything about *reading* the plan's utilisation: the HTTP
round-trip, caching its response, and turning it into the three quota figures the
loop gates on. It knows nothing about *policy* (which quota to gate on, what
ceiling to allow, when to pause) — that lives in limits.py, which consumes the
`Usage` snapshots this module produces.

The figures come from `GET /api/oauth/usage` — the same endpoint the CLI's own
`/usage` panel and the status line are built from — authenticated with the OAuth
token the CLI already keeps on disk. The response is machine-readable:

    {"five_hour": {"utilization": 9.0, "resets_at": "2026-08-15T15:19:59+00:00"},
     "seven_day": {"utilization": 40.0, "resets_at": "2026-08-19T12:59:59+00:00"},
     "seven_day_sonnet": null, ...}

Each entry becomes a `UsageReading` (percent + reset epoch); the three together,
plus summary lines rendered for the log, make a `Usage` snapshot.

**Why not `claude -p "/usage"`** (what this used to do): that is not a local
command — it starts a whole session and lets the model read the panel and retell
it in prose, which we then had to parse with regexes. Measured on opus: $0.33,
17.4 s and 1174 output tokens *per check* — i.e. the measurement was spending a
noticeable slice of the very budget it measured, and the parse broke on any
rewording. This costs no tokens and ~0.3 s.

The endpoint is undocumented, so treat a failure as "no figures" rather than a
fatal error: the loop degrades to the free backstop instead, i.e. the
`rate_limit_event` the CLI emits on its own stream — which reports a hard
refusal even when this reading is missing, and which is `RateLimitEvent` below,
here for the same reason `CLAUDE_SESSION_DURATION` is.

`EMPTY_READING`, `EMPTY_USAGE` and `summary_line` are public here because
`codex_usage` builds the same snapshots from its own protocol and needs them;
`_` in this package means "this file's business" and nothing wider — see
tests/test_package_privacy.py.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import NamedTuple, Optional

# The one thing this module prints rather than computes: a reset MOMENT in the
# same wording every other line of the run uses for one. `_fmt_reset` below is
# not that wording — it is the "Aug 15, 6:19pm" of the usage summary lines, whose
# exact text is a contract with LimitPolicy.log_snapshot — so the verdict's
# `describe()` borrows the console's spelling rather than growing a third.
from .console import fmt_moment
# One event of the provider stream reaches this module — the rate-limit verdict —
# and the word the stream calls it by lives with the rest of that vocabulary.
from . import wire

# The usage endpoint, and the OAuth credentials the CLI stores for its own calls.
# ANTHROPIC_BASE_URL / CLAUDE_CONFIG_DIR are the CLI's own environment overrides,
# honoured here so a redirected install keeps working.
USAGE_URL_PATH = "/api/oauth/usage"
DEFAULT_BASE_URL = "https://api.anthropic.com"
CREDENTIALS_FILENAME = ".credentials.json"
HTTP_TIMEOUT = 10        # seconds — the CLI uses 5s; a little slack for a busy box
HTTP_ATTEMPTS = 2        # one retry: a dropped connection shouldn't blind the gate

# The reading is cheap (no tokens, ~0.3s), so the cache exists only to collapse
# the several reads that happen at one instant (limit check + log snapshot) into
# one request, not to ration the query.
USAGE_CACHE_TTL = 30  # seconds

# How long a Claude session window lasts. The one number both users of it need:
# a run that hit a token limit waits out that window before retrying, and a
# policy with no reset time in the report falls back to "one window from now".
#
# Here rather than in a runner because it is a fact about the QUOTA, and this is
# the module that reads quotas — it was the last thing `limits` had to import a
# runner for, and that import is what made the limit rules unusable without the
# sequential loop. The +3s is an unmeasured safety margin against waking exactly
# on the boundary; it moved here verbatim and nothing here justifies the number.
CLAUDE_SESSION_DURATION = 5 * 60 * 60 + 3  # 5 hours, in seconds


class UsageReading(NamedTuple):
    """One quota from the usage report: its utilisation % and reset time.

    `percent` is None when the quota was absent or null in the response (an
    unused / not-applicable quota, e.g. the Sonnet-only week on a plan without
    one). `reset_ts` is the epoch time the quota's window resets (None if the
    response carried no reset time).
    """
    percent: Optional[float]
    reset_ts: Optional[float]


class Quota(NamedTuple):
    """One of the windows an account is metered on — the *description* of a
    quota, not a reading of it.

    Everything that needs to speak about a quota (the parser, the log lines, the
    rules in limits.py, the status line) names it through this table rather than
    with a string of its own, so the four namings cannot drift apart:

      field  — the `Usage` attribute holding its reading, and the name a
               LimitRule's `quota` selects it by.
      key    — the response key it is parsed from.
      label  — what its summary line starts with (a contract: LimitPolicy
               picks its lines by matching a rule's `label` against them).
      short  — the status line's abbreviation; a pinned row has no room for prose.
      always — pinned on the status line even with no figure to show. True for
               the windows every plan has, so the reader always sees both;
               false for the ones a plan may simply not have.
    """
    field: str
    key: str
    label: str
    short: str
    always: bool


QUOTAS = (
    Quota("session", "five_hour", "Current session", "session", True),
    Quota("week_all", "seven_day", "Current week (all models)", "week", True),
    Quota("week_sonnet", "seven_day_sonnet", "Current week (Sonnet only)",
          "week/sonnet", False),
)
QUOTA_BY_FIELD = {quota.field: quota for quota in QUOTAS}


# --- the other source of the same knowledge: the wire's own verdict ------------
#
# Every `claude` run streams a line of its own, built from the ratelimit headers
# the API already returned to it:
#
#   {"type":"rate_limit_event","rate_limit_info":{"status":"allowed",
#     "resetsAt":1786807200,"rateLimitType":"five_hour", …}}
#
# Reading it costs nothing — that stream is parsed anyway — and unlike the
# queried figures above it cannot be stale or unavailable: it is the wire's own
# verdict on the request that just went out. It carries no percentage, so it
# cannot drive a ceiling; it is the backstop *under* the proactive check in
# limits.py. "rejected" means this run hit the wall, and `resetsAt` says when
# that quota comes back.
#
# Here rather than in a runner for the reason `CLAUDE_SESSION_DURATION` is: it is
# a fact about a QUOTA, and BOTH runners read it — the sequential one out of the
# stream it renders, the parallel one out of each worker's stream — so living in
# either made the other import a loop it does not run. What does NOT live here is
# the latch remembering the last verdict: that is single-stream state, and it
# stays with the single-stream renderer, behind
# `streamrender.last_rate_limit_event`.

# Quota id -> the name the CLI itself uses for it in limit messages.
#
# A fifth naming of the windows `Quota` above describes, and deliberately not
# folded into it: this table is keyed by the ids that appear on the WIRE, and it
# carries two the usage report has no entry for at all ("seven_day_opus",
# "seven_day_overage_included"). Merging it would mean inventing report keys and
# status-line abbreviations for windows the report never mentions. Adjacency is
# the point instead — a sixth window arriving on the wire is now visibly a
# question about both tables rather than about whichever file was open.
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
        when = f", resets {fmt_moment(self.resets_at)}" if self.resets_at else ""
        return f"{self.label} {self.status}{when}"


def rate_limit_event_from(ev: dict) -> Optional[RateLimitEvent]:
    """The RateLimitEvent carried by a stream-json event, or None if it isn't one.

    The envelope's word is `wire`'s (one home for every literal the provider
    stream is dispatched on); the PAYLOAD's keys are this module's, because what
    a quota verdict is made of is a fact about a quota.
    """
    if wire.event_type(ev) != wire.RATE_LIMIT_EVENT:
        return None
    info = ev.get("rate_limit_info") or {}
    resets = info.get("resetsAt")
    return RateLimitEvent(
        status=str(info.get("status") or "unknown"),
        limit_type=str(info.get("rateLimitType") or ""),
        resets_at=float(resets) if isinstance(resets, (int, float)) else None,
    )


class Usage(NamedTuple):
    """A full parsed usage snapshot: the three quota readings plus summary lines
    rendered for the log. The readings are the QUOTAS above, in that order:

      * session     — the ~5-hour window ("five_hour").
      * week_all    — the weekly all-models window ("seven_day").
      * week_sonnet — the weekly Sonnet-only window ("seven_day_sonnet").

    `summary_lines` keeps the historical "Current session: 9% used · resets …"
    wording: LimitPolicy.log_snapshot picks the lines it wants by matching a
    rule's `label` against their start, so the labels are a contract, not prose.
    """
    session: UsageReading
    week_all: UsageReading
    week_sonnet: UsageReading
    summary_lines: list

    def readings(self) -> tuple:
        """((Quota, UsageReading), …) for every quota, in report order.

        The one way to walk a snapshot without hard-coding its field names, so a
        fourth window would reach the log and the status line by being added to
        QUOTAS alone.
        """
        return tuple((quota, getattr(self, quota.field)) for quota in QUOTAS)


EMPTY_READING = UsageReading(None, None)
EMPTY_USAGE = Usage(EMPTY_READING, EMPTY_READING, EMPTY_READING, [])


def _config_dir() -> str:
    """The CLI's config directory (~/.claude, or CLAUDE_CONFIG_DIR)."""
    return (os.environ.get("CLAUDE_CONFIG_DIR")
            or os.path.join(os.path.expanduser("~"), ".claude"))


def oauth_token() -> Optional[str]:
    """The OAuth access token to authenticate the usage request with.

    Read fresh on every call rather than cached: the CLI rewrites the credentials
    file whenever it refreshes the token, and this loop runs the CLI constantly —
    so re-reading is what keeps a long run authenticated for free.
    """
    env = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env:
        return env.strip()
    path = os.path.join(_config_dir(), CREDENTIALS_FILENAME)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        print(f"  · no usage figures: {path} not found (is this machine signed "
              f"in with an OAuth login?)")
        return None
    except (OSError, ValueError) as e:
        print(f"  · no usage figures: could not read {path} ({e})")
        return None
    token = (data.get("claudeAiOauth") or {}).get("accessToken")
    if not token:
        print(f"  · no usage figures: no OAuth access token in {path}")
        return None
    return token


def _iso_to_ts(value) -> Optional[float]:
    """Epoch seconds for an ISO-8601 timestamp from the report, or None.

    The endpoint returns offset-aware ISO strings ("…+00:00"); a bare "Z" suffix
    is normalised first because datetime.fromisoformat only learned it in 3.11.
    """
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _reading_from(entry) -> UsageReading:
    """Parse one quota object ({"utilization": …, "resets_at": …}) into a reading.

    A quota that does not apply to the plan comes back as JSON null, and an
    entitlement that exists but was never touched can carry a null utilisation —
    both mean "no figure", which is what an all-None reading says.
    """
    if not isinstance(entry, dict):
        return EMPTY_READING
    percent = entry.get("utilization")
    percent = float(percent) if isinstance(percent, (int, float)) else None
    return UsageReading(percent, _iso_to_ts(entry.get("resets_at")))


def _fmt_percent(percent: float) -> str:
    """"9%" / "9.5%" — drop the decimal point when the figure is a whole number."""
    return f"{percent:.0f}%" if float(percent).is_integer() else f"{percent:.1f}%"


def _fmt_reset(ts: float) -> str:
    """"Aug 15, 6:19pm" in local time — built field by field because the obvious
    strftime forms for an unpadded day/hour (%-d, %-I) are not portable to
    Windows."""
    dt = datetime.fromtimestamp(ts)
    hour = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"{dt.strftime('%b')} {dt.day}, {hour}:{dt.minute:02d}{ampm}"


def summary_line(label: str, reading: UsageReading) -> str:
    """One "Current session: 9% used · resets Aug 15, 6:19pm" line for the log."""
    line = f"{label}: {_fmt_percent(reading.percent)} used"
    if reading.reset_ts is not None:
        line += f" · resets {_fmt_reset(reading.reset_ts)}"
    return line


def parse_usage(data: dict) -> Usage:
    """Turn a raw /api/oauth/usage response into a Usage snapshot.

    Quotas that are absent or null come back as all-None readings and contribute
    no summary line, so a plan without a Sonnet-only week simply has nothing to
    say about it (and a rule watching it never fires).
    """
    if not isinstance(data, dict):
        return EMPTY_USAGE
    readings = {}
    summary = []
    for quota in QUOTAS:
        reading = _reading_from(data.get(quota.key))
        readings[quota.field] = reading
        if reading.percent is not None:
            summary.append(summary_line(quota.label, reading))
    return Usage(readings["session"], readings["week_all"],
                 readings["week_sonnet"], summary)


class UsageSource:
    """Queries and caches the usage endpoint; hands out parsed Usage snapshots.

    The read-only counterpart to a LimitPolicy: the policy asks this for the
    current figures and decides what to do. One response is cached for
    `cache_ttl` seconds and reused by every reader (the limit check and the
    bookend snapshots share it).
    """

    def __init__(self, cache_ttl: float = USAGE_CACHE_TTL):
        self.cache_ttl = cache_ttl
        self._cached: Optional[Usage] = None
        self._cached_ts: float = 0.0

    def _url(self) -> str:
        base = os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL
        return base.rstrip("/") + USAGE_URL_PATH

    def query_usage_json(self) -> Optional[dict]:
        """GET the usage report, or None (with a printed reason) on failure.

        A 401 means the stored token went stale; no refresh is attempted here —
        writing the credentials file from under a running CLI is not worth the
        race, and every `claude` run the loop makes refreshes it anyway, so the
        next check picks the new token up by itself.
        """
        token = oauth_token()
        if not token:
            return None
        request = urllib.request.Request(self._url(), headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "anthropic-beta": "oauth-2025-04-20",
            "User-Agent": "llm-loop",
        })
        last = ""
        for attempt in range(HTTP_ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    print("  · no usage figures: the stored OAuth token was "
                          "rejected (401) — it should refresh on the next "
                          "`claude` run.")
                    return None
                last = f"HTTP {e.code} {e.reason}"
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
                last = str(e)
            if attempt + 1 < HTTP_ATTEMPTS:
                time.sleep(1)
        print(f"  · no usage figures: {self._url()} did not answer ({last})")
        return None

    def get_usage(self, cache_value: bool = True) -> Usage:
        """Return the current Usage snapshot, reusing a cached reading when fresh.

        With `cache_value` True (default) a snapshot younger than `cache_ttl` is
        returned without a new request; otherwise (or once stale) the endpoint is
        queried and the result cached. Pass `cache_value=False` to force a fresh
        reading. A failed query is not cached, so the next call retries instead of
        being stuck on an all-None snapshot.
        """
        now = time.time()
        if (cache_value and self._cached is not None
                and now - self._cached_ts < self.cache_ttl):
            return self._cached
        data = self.query_usage_json()
        if data is None:
            return EMPTY_USAGE
        usage = parse_usage(data)
        self._cached = usage
        self._cached_ts = now
        return usage

    def invalidate(self) -> None:
        """Drop the cached snapshot (e.g. after waiting out a window, when the old
        percentages are no longer meaningful)."""
        self._cached = None
        self._cached_ts = 0.0
