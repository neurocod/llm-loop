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
`rate_limit_event` the CLI emits on its own stream (see cyclecore.RateLimitEvent),
which reports a hard refusal even when this reading is missing.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import NamedTuple, Optional

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


class UsageReading(NamedTuple):
    """One quota from the usage report: its utilisation % and reset time.

    `percent` is None when the quota was absent or null in the response (an
    unused / not-applicable quota, e.g. the Sonnet-only week on a plan without
    one). `reset_ts` is the epoch time the quota's window resets (None if the
    response carried no reset time).
    """
    percent: Optional[float]
    reset_ts: Optional[float]


class Usage(NamedTuple):
    """A full parsed usage snapshot: the three quota readings plus summary lines
    rendered for the log.

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


_EMPTY_READING = UsageReading(None, None)
_EMPTY_USAGE = Usage(_EMPTY_READING, _EMPTY_READING, _EMPTY_READING, [])

# Response key -> the label its summary line starts with. The labels must keep
# matching the rules' `label` attributes in limits.py (see Usage.summary_lines).
_QUOTAS = (
    ("five_hour", "Current session"),
    ("seven_day", "Current week (all models)"),
    ("seven_day_sonnet", "Current week (Sonnet only)"),
)


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
        return _EMPTY_READING
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


def _summary_line(label: str, reading: UsageReading) -> str:
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
        return _EMPTY_USAGE
    readings = {}
    summary = []
    for key, label in _QUOTAS:
        reading = _reading_from(data.get(key))
        readings[key] = reading
        if reading.percent is not None:
            summary.append(_summary_line(label, reading))
    return Usage(readings["five_hour"], readings["seven_day"],
                 readings["seven_day_sonnet"], summary)


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
            return _EMPTY_USAGE
        usage = parse_usage(data)
        self._cached = usage
        self._cached_ts = now
        return usage

    def invalidate(self) -> None:
        """Drop the cached snapshot (e.g. after waiting out a window, when the old
        percentages are no longer meaningful)."""
        self._cached = None
        self._cached_ts = 0.0
