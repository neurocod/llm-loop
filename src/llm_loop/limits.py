"""
limits.py - the "what do we do about usage" layer: the pausing policy.

usage.py reads the account's quota figures into a `Usage` snapshot; this module
decides what to do with them. A host project picks a *limit specialisation* by
setting one class attribute on its Driver::

    class MyDriver(StateFileDriver):
        limit_policy = LimitPolicy([SessionLimit(80)])          # flat session cap
        # or  LimitPolicy([DayNightLimit()])                    # the default
        # or  LimitPolicy([WeeklyLimit(90)])                    # weekly cap
        # or  LimitPolicy([DayNightLimit(), WeeklyLimit(90)])   # composite

A `LimitPolicy` holds one or more *rules*; the loop pauses whenever **any** rule
is over its ceiling (so a composite protects the weekly cap even while the
session still has budget). Three ready-made rules cover the common cases:

  * SessionLimit(limit)   — a flat ceiling on the ~5-hour "Current session"
    figure. Simple and predictable: pause at `limit`% and wait out the window.
  * DayNightLimit(...)    — the smart session rule (the engine default). Its base
    ceiling depends on the time of day — at night the leftover session budget is
    just wasted while we sleep, so we may run almost to the wall; in the daytime
    we keep a reserve — and near the window's reset the usable ceiling climbs
    toward 100%, since budget that can't be spent before the reset is safe to
    release (see _dynamic_ceiling).
  * WeeklyLimit(limit)    — a flat ceiling on the weekly figure ("all models",
    or Sonnet-only), waiting out the week-long window when hit.

Write your own rule by subclassing LimitRule (set `quota`/`label`, implement
`ceiling`; override `status` to change what it contributes to the status line).

A policy decides where the loop PAUSES, not what the status line shows: the
pinned row lists the provider's own windows either way (see
statusline.quota_rows), and a rule only adds its half — its ceiling — next to the
window it watches.
"""

import sys
import time
from datetime import datetime
from typing import Optional

from .console import _fmt_clock, _fmt_left, print_percents
from .cyclecore import CLAUDE_SESSION_DURATION
from .stopchannel import sleep_unless
from .usage import QUOTA_BY_FIELD, Usage, UsageReading

# Default ceilings for the ready-made rules (all overridable per instance).
NIGHT_USAGE_LIMIT = 95          # % — overnight, when leftover session budget is wasted
DAY_USAGE_LIMIT = 95            # % — daytime, keep a reserve for other work
NIGHT_RESET_DEADLINE_HOUR = 10  # the "morning" boundary: 10:00 local time
WEEKLY_USAGE_LIMIT = 90         # % — default weekly ceiling

# Empirically, on this plan and these typical tasks, the session budget is spent
# at roughly this rate per minute of active work. Near the end of a window unused
# budget would be wasted anyway, and we physically cannot overspend it: with T
# minutes left we can burn at most USAGE_RATE_PER_MIN * T %. So DayNightLimit lets
# the usable ceiling rise toward 100% as the window winds down — see
# _dynamic_ceiling().
USAGE_RATE_PER_MIN = 1.5        # % of the session budget spent per minute of work


def _dynamic_ceiling(base: float, reset_ts: Optional[float], now: float,
                     rate: float) -> float:
    """Usable ceiling (%) given a base limit and how close the window is to reset.

    The base keeps a reserve for other work, but that reserve is only worth
    protecting while the window lasts: with `minutes` left we can spend at most
    rate * minutes more, so any usage above 100 - rate * minutes can never be
    reached before the reset and is safe to release. The ceiling is therefore
    max(base, 100 - rate * minutes), capped at 100 — it equals `base` until the
    window is within (100 - base) / rate minutes of its reset, then climbs toward
    100%. With no known reset time the base is returned unchanged.
    """
    if reset_ts is None:
        return float(base)
    minutes = max(0.0, (reset_ts - now) / 60.0)
    return min(100.0, max(float(base), 100.0 - rate * minutes))


class LimitRule:
    """One quota + its ceiling policy. Subclass and override; set `quota` to the
    Usage field this rule watches and `label` to its human name.

      quota   — the name of the window watched: a `field` from usage.QUOTAS
                ("session" | "week_all" | "week_sonnet"), which picks the
                UsageReading out of a snapshot.
      label   — shown in status/snapshot lines (e.g. "Current session").
      ceiling(reading, now) — the allowed % right now (>= it means "pause").
      status(reading, now)  — this rule's one-line contribution to the pinned
                status line, appended to the provider's own figures.

    Rules are read-only and stateless (all mutable run state lives in the
    UsageSource cache), so a single default instance is safe to share as a class
    attribute across Drivers.
    """

    quota: str = "session"
    label: str = "limit"

    def reading(self, usage: Usage) -> UsageReading:
        """The UsageReading this rule watches, selected by `quota`."""
        return getattr(usage, self.quota)

    def ceiling(self, reading: UsageReading, now: float) -> float:
        """The allowed usage % right now; usage at/above it triggers a pause."""
        raise NotImplementedError

    def status(self, reading: UsageReading, now: float) -> str:
        """What this rule adds to the status line's row for its quota.

        The row itself belongs to the provider — the percentage and the time to
        reset are facts about the account and are shown whether or not any rule
        watches that window. This is the policy's own half of it, so the default
        is the single number a rule contributes: the ceiling in force right now
        (which for DayNightLimit moves as the window winds down). Override to say
        more; return "" to add nothing.
        """
        return f"ceil {self.ceiling(reading, now):.0f}%"

    def describe(self) -> str:
        """One-line human summary of this rule's quota and ceiling policy, shown
        in the loop preamble so a run records exactly what it is gated on."""
        raise NotImplementedError


class SessionLimit(LimitRule):
    """A flat ceiling on the ~5-hour "Current session" figure.

    The simplest session rule: pause once the session hits `limit`% and wait out
    the window. No day/night awareness and no near-reset climb — use DayNightLimit
    for those. "Total session limit" behaviour.
    """

    quota = "session"
    label = QUOTA_BY_FIELD["session"].label

    def __init__(self, limit: float = DAY_USAGE_LIMIT):
        self.limit = float(limit)

    def ceiling(self, reading: UsageReading, now: float) -> float:
        return self.limit

    def describe(self) -> str:
        return f"{self.label}: flat ceiling {self.limit:.0f}%"


class DayNightLimit(LimitRule):
    """The smart session rule (engine default): a day/night base ceiling that also
    climbs toward 100% as the session window nears its reset.

    Rule for the base: if the current local time is in the small hours (before
    `deadline_hour`:00) *and* the session resets no later than `deadline_hour`:00
    (it refreshes in the morning while we're likely still asleep), allow up to
    `night`%; otherwise cap at `day`%. The near-reset climb (_dynamic_ceiling)
    then reclaims budget that couldn't be spent before the reset anyway.
    """

    quota = "session"
    label = QUOTA_BY_FIELD["session"].label

    def __init__(self, *, day: float = DAY_USAGE_LIMIT,
                 night: float = NIGHT_USAGE_LIMIT,
                 deadline_hour: int = NIGHT_RESET_DEADLINE_HOUR,
                 rate_per_min: float = USAGE_RATE_PER_MIN):
        self.day = float(day)
        self.night = float(night)
        self.deadline_hour = deadline_hour
        self.rate_per_min = rate_per_min

    def _base(self, reset_ts: Optional[float], now: float) -> float:
        """The day/night base ceiling before the near-reset climb is applied."""
        now_dt = datetime.fromtimestamp(now)
        after_midnight = now_dt.hour < self.deadline_hour
        resets_by_morning = False
        if reset_ts is not None:
            reset_dt = datetime.fromtimestamp(reset_ts)
            resets_by_morning = (
                reset_dt.hour * 60 + reset_dt.minute
                <= self.deadline_hour * 60
            )
        if after_midnight and resets_by_morning:
            return self.night
        return self.day

    def ceiling(self, reading: UsageReading, now: float) -> float:
        base = self._base(reading.reset_ts, now)
        return _dynamic_ceiling(base, reading.reset_ts, now, self.rate_per_min)

    def describe(self) -> str:
        return (f"{self.label}: day {self.day:.0f}% / night {self.night:.0f}% "
                f"(night before {self.deadline_hour:02d}:00 when the window "
                f"resets by then), climbing toward 100% near reset")


class WeeklyLimit(LimitRule):
    """A flat ceiling on the weekly figure — "all models" by default, or the
    Sonnet-only quota with `sonnet_only=True`.

    The weekly window is a week long, so there is no day/night rule and no
    near-reset climb (the last few minutes of a week are negligible): pause at
    `limit`% and wait out the window. In a composite this guards against the
    weekly cap killing a run while the session still has budget.
    """

    def __init__(self, limit: float = WEEKLY_USAGE_LIMIT, *,
                 sonnet_only: bool = False):
        self.limit = float(limit)
        self.quota = "week_sonnet" if sonnet_only else "week_all"
        self.label = QUOTA_BY_FIELD[self.quota].label

    def ceiling(self, reading: UsageReading, now: float) -> float:
        return self.limit

    def describe(self) -> str:
        return f"{self.label}: flat ceiling {self.limit:.0f}%"


class LimitPolicy:
    """One or more LimitRules; the loop pauses when **any** of them is exceeded.

    A single rule is the common case (`LimitPolicy([DayNightLimit()])`); several
    make a composite (`LimitPolicy([DayNightLimit(), WeeklyLimit(90)])`) that
    holds until *every* rule is back under its ceiling — so whichever quota is the
    binding constraint at the time governs the pause.
    """

    def __init__(self, rules):
        self.rules = list(rules)

    def describe(self) -> str:
        """Human summary of every rule, joined for the loop preamble. The loop
        pauses when *any* rule is exceeded, so the rules are joined with 'or'."""
        parts = [r.describe() for r in self.rules]
        if not parts:
            return "no rules (never pauses on usage)"
        if len(parts) == 1:
            return parts[0]
        return "pause when any of — " + " | ".join(parts)

    # --- inspection helpers ----------------------------------------------------

    def rule_for(self, quota: str) -> Optional[LimitRule]:
        """The rule watching `quota` (a usage.QUOTAS field), or None for a window
        this policy does not gate on.

        The status line asks this per quota: the provider's figures are pinned
        either way, and a rule that is there adds its own part next to them. A
        lookup rather than a flag on the rules, so the display never has to be
        told about a new rule — it just finds it.
        """
        for rule in self.rules:
            if rule.quota == quota:
                return rule
        return None

    def _status(self, usage: Usage, now: float) -> list:
        """[(rule, reading, ceiling)] for every rule at time `now`."""
        return [(r, r.reading(usage), r.ceiling(r.reading(usage), now))
                for r in self.rules]

    @staticmethod
    def _violations(status: list) -> list:
        """The (rule, reading, ceiling) entries whose usage is at/over the ceiling."""
        return [(r, rd, c) for r, rd, c in status
                if rd.percent is not None and rd.percent >= c]

    # --- the public entry point ------------------------------------------------

    def check_and_wait(self, source, session_start: float, note: str = "",
                       cache_value: bool = True, should_stop=None) -> tuple:
        """Read the usage figures; if any rule is at/over its ceiling, pause until
        they all clear (a window reset, or — for DayNightLimit — the ceiling rising
        above the usage as the reset nears).

        Returns (paused, session_start). `session_start` is refreshed to now only
        when the *session* window actually reset, so callers can reset their
        per-session bookkeeping; otherwise it is returned unchanged.

        `should_stop` is the caller's stop channels (see stopchannel.sleep_unless).
        The pause here is the longest hold in the engine — hours, with every
        worker parked in it — so it must be abandonable: the caller re-reads its
        own stop state after this returns and decides what to do about it.
        """
        usage = source.get_usage(cache_value)
        now = time.time()
        status = self._status(usage, now)

        for r, rd, c in status:
            if rd.percent is None:
                print(f"  · {r.label}: no figure in the usage report{note}")
            else:
                # How long the window still has to run: the other half of what a
                # percentage means (10% with four hours left is a different
                # situation from 10% with ten minutes left), and the quantity a
                # DayNightLimit ceiling is itself computed from.
                left = (f", {_fmt_left(rd.reset_ts - now)} left"
                        if rd.reset_ts is not None else " now")
                print_percents(f"  · {r.label} usage: {rd.percent:.0f}% "
                               f"(ceiling {c:.0f}%{left}){note}")

        if not self._violations(status):
            return False, session_start
        return self._wait(source, session_start, should_stop)

    def _wait(self, source, session_start: float, should_stop=None) -> tuple:
        """Hold while any rule is over its ceiling.

        The usage percentages are frozen for the duration (we are not burning
        tokens while paused), so we keep the snapshot taken on entry and only
        re-query after a window resets. Each minute the clock advances, which
        lets a DayNightLimit ceiling climb; we resume the moment no rule is
        violated, re-reading fresh figures whenever a window refreshes.

        Returns (True, session_start): session_start bumped to now if the session
        window reset during the wait, else unchanged.

        A pending stop (`should_stop`) ends the hold immediately. It is reported
        as a pause like any other, because that is what it was — what to do next
        is the caller's decision, and it reads its own stop channels for that.
        """
        usage = source.get_usage()  # snapshot, frozen until a window refreshes
        labels = ", ".join(r.label for r, _, _ in
                           self._violations(self._status(usage, time.time())))
        print(f"  ⏳ Over usage limit on: {labels} — holding until it clears or "
              f"the window resets…")
        try:
            while True:
                if should_stop is not None and should_stop():
                    print("  ⏹ Stop requested while over the usage limit — "
                          "leaving the wait without resuming.")
                    return True, session_start
                now = time.time()
                status = self._status(usage, now)
                violated = self._violations(status)
                if not violated:
                    # About to resume; force a fresh reading on the next check.
                    source.invalidate()
                    print(f"  ▶ Back under all usage limits (now {_fmt_clock(now)}) "
                          f"— resuming.")
                    return True, session_start

                # When will the next window refresh? Watch the soonest reset among
                # the violated rules; if none carry a reset time, fall back to a
                # full session-window wait.
                resets = [rd.reset_ts for _, rd, _ in violated
                          if rd.reset_ts is not None]
                next_reset = min(resets) if resets else now + CLAUDE_SESSION_DURATION

                if now >= next_reset:
                    # A window refreshed — the frozen percentages are stale.
                    print(f"  ▶ A usage window reset (now {_fmt_clock(now)}) — "
                          f"re-checking with fresh figures.")
                    # If it was the session window, restart the session clock.
                    session = usage.session
                    if session.reset_ts is not None and now >= session.reset_ts:
                        session_start = now
                    source.invalidate()
                    usage = source.get_usage()
                    continue

                over = ", ".join(
                    f"{r.label} {rd.percent:.0f}% ≥ {c:.0f}%"
                    for r, rd, c in violated)
                print_percents(f"    … {over}; {_fmt_left(next_reset - now)} "
                               f"to next reset (now {_fmt_clock(now)})")
                sleep_unless(min(next_reset - now, 60), should_stop)
        except KeyboardInterrupt:
            print("\nWait interrupted by user (Ctrl+C).")
            sys.exit(130)

    def log_snapshot(self, source, label: str = "",
                     cache_value: bool = True) -> None:
        """Log the usage figures this policy watches (its rules' quota lines).

        Called at the run's bookends so each run records where it started and
        finished against exactly the limits it is gated on — a weekly-only policy
        logs only the weekly line, a composite logs both. Picks its lines out of
        `Usage.summary_lines` by matching each rule's `label` against their start.
        """
        usage = source.get_usage(cache_value)
        head = f"  · usage {label}".rstrip() + ":"
        wanted = {r.label.lower() for r in self.rules}
        lines = [ln for ln in usage.summary_lines
                 if any(ln.lower().startswith(w) for w in wanted)]
        if not lines:
            print(f"{head} (no matching figures in the usage report)")
            return
        print(head)
        for ln in lines:
            print_percents(f"      {ln}")


def default_policy(provider: str = "claude") -> LimitPolicy:
    """The provider-aware default limit specialisation.

    Claude retains its historical smart session rule.  Codex plans can expose
    a short session window, a weekly window, or only the weekly window, so its
    default watches both shared slots and simply ignores an absent reading.
    """
    if provider == "codex":
        return LimitPolicy([DayNightLimit(), WeeklyLimit()])
    return LimitPolicy([DayNightLimit()])
