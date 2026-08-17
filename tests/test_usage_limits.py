"""The two layers that keep a run inside its quota.

Proactive: usage.py turns the account's usage report into the three readings the
policy gates on. The labels its summary lines start with are a contract with
limits.py (LimitPolicy.log_snapshot matches a rule's `label` against them), so
they are asserted here rather than left to prose.

Reactive: every `claude` run streams its own rate-limit verdict, and a "rejected"
must park the loop until that quota resets — even when the proactive reading was
unavailable, which is the case the backstop exists for. That path is the one that
only ever runs when the budget is already spent, i.e. the one nobody exercises by
hand, so it is pinned with a fake run instead.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_loop import cyclecore, usage
from llm_loop.cyclecore import ClaudeCommand, Driver, RateLimitEvent
from llm_loop.limits import DayNightLimit, LimitPolicy, WeeklyLimit


# A response like the endpoint's, trimmed to the quotas the engine reads. The
# Sonnet-only week is null: a quota the plan does not have is absent, not zero.
SAMPLE = {
    "five_hour": {"utilization": 9.0,
                  "resets_at": "2026-08-15T15:19:59.700784+00:00"},
    "seven_day": {"utilization": 40.5,
                  "resets_at": "2026-08-19T12:59:59.700808+00:00"},
    "seven_day_sonnet": None,
}


def _iso_in(seconds: float) -> str:
    """An ISO reset time `seconds` from now — the readings a test asserts about
    have to move with the clock, or the test expires on a fixed date."""
    return datetime.fromtimestamp(time.time() + seconds, timezone.utc).isoformat()


class _StubSource:
    """A UsageSource answering from a payload instead of the network."""

    def __init__(self, payload=None):
        self.invalidated = 0
        self.payload = payload if payload is not None else {
            "five_hour": {"utilization": 9.0, "resets_at": _iso_in(2 * 3600)},
            "seven_day": {"utilization": 40.5, "resets_at": _iso_in(4 * 86400)},
            "seven_day_sonnet": None,
        }

    def get_usage(self, cache_value=True):
        return usage.parse_usage(self.payload)

    def invalidate(self):
        self.invalidated += 1


@pytest.fixture(autouse=True)
def _restore_streams():
    """run_loop tees sys.stdout/stderr into its log and never puts them back."""
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


# -- the usage report -----------------------------------------------------------

def test_parse_reads_percent_and_reset():
    u = usage.parse_usage(SAMPLE)
    assert u.session.percent == 9.0
    assert u.week_all.percent == 40.5
    assert u.session.reset_ts == pytest.approx(1786807199.7, abs=1)
    assert u.week_all.reset_ts == pytest.approx(1787144399.7, abs=1)


def test_absent_quota_is_no_figure_not_zero():
    """A plan without a Sonnet-only week must read as "no figure": a 0% would let
    a WeeklyLimit(sonnet_only=True) claim the budget is untouched."""
    u = usage.parse_usage(SAMPLE)
    assert u.week_sonnet == usage.UsageReading(None, None)
    assert not any(ln.startswith("Current week (Sonnet") for ln in u.summary_lines)


def test_summary_lines_match_the_rule_labels():
    """log_snapshot picks its lines by matching a rule's `label` against their
    start — so the wording is an interface, not decoration."""
    u = usage.parse_usage(SAMPLE)
    for rule in (DayNightLimit(), WeeklyLimit()):
        assert any(ln.lower().startswith(rule.label.lower())
                   for ln in u.summary_lines), rule.label
    assert u.summary_lines[0].startswith("Current session: 9% used · resets ")
    assert u.summary_lines[1].startswith("Current week (all models): 40.5% used")


def test_malformed_report_reads_as_empty():
    for bad in (None, [], {}, {"five_hour": "nope"}, {"five_hour": {}}):
        u = usage.parse_usage(bad)
        assert u.session.percent is None
        assert u.summary_lines == []


def test_iso_z_suffix_is_accepted():
    """datetime.fromisoformat only learned the bare "Z" in 3.11."""
    assert usage._iso_to_ts("2026-08-15T15:19:59Z") == pytest.approx(
        usage._iso_to_ts("2026-08-15T15:19:59+00:00"))
    assert usage._iso_to_ts("not a time") is None
    assert usage._iso_to_ts(None) is None


def test_a_failed_query_is_not_cached(monkeypatch):
    """A blind reading must not stick: the next check has to try again rather
    than run on an all-None snapshot that never pauses anything."""
    answers = [None, SAMPLE]
    source = usage.UsageSource()
    monkeypatch.setattr(source, "query_usage_json", lambda: answers.pop(0))
    assert source.get_usage().session.percent is None
    assert source.get_usage().session.percent == 9.0
    assert answers == []


@pytest.mark.parametrize("seconds,expected", [
    (4 * 86400 + 3 * 3600 + 59 * 60, "4d3h"),
    (2 * 86400, "2d"),                 # a zero smaller unit is dropped
    (3 * 3600 + 24 * 60, "3h24m"),
    (3 * 3600 + 4 * 60, "3h4m"),
    (3600, "1h"),
    (24 * 60 + 59, "24m"),
    (30, "<1m"),                       # never "0m": the wait is not over yet
    (-5, "<1m"),
])
def test_time_left_reads_as_a_quantity(seconds, expected):
    assert cyclecore._fmt_left(seconds) == expected


def test_the_status_line_says_how_long_the_window_has_left(capsys):
    """A percentage alone is half the picture — 9% with four hours left and 9%
    with ten minutes left call for opposite decisions, and it is the quantity the
    DayNightLimit ceiling is computed from."""
    source = _StubSource()
    LimitPolicy([DayNightLimit()]).check_and_wait(source, time.time())
    line = capsys.readouterr().out
    assert "Current session usage: 9% (ceiling " in line
    assert " left)" in line


def test_a_reading_without_a_reset_time_still_prints(capsys):
    """No reset time in the report — the ceiling line must not lose the ceiling."""
    source = _StubSource({"five_hour": {"utilization": 9.0}})
    LimitPolicy([DayNightLimit()]).check_and_wait(source, time.time())
    out = capsys.readouterr().out
    assert "Current session usage: 9% (ceiling 95% now)" in out


# -- the rate_limit_event backstop ---------------------------------------------

def test_event_parse():
    ev = {"type": "rate_limit_event", "rate_limit_info": {
        "status": "rejected", "resetsAt": 1786807200, "rateLimitType": "five_hour"}}
    rl = cyclecore.rate_limit_event_from(ev)
    assert (rl.status, rl.limit_type, rl.resets_at) == (
        "rejected", "five_hour", 1786807200.0)
    assert rl.label == "session limit"
    assert cyclecore.rate_limit_event_from({"type": "result"}) is None
    # A verdict with fields missing still parses — it must not throw mid-stream.
    bare = cyclecore.rate_limit_event_from({"type": "rate_limit_event"})
    assert bare.resets_at is None and bare.status == "unknown"


class _OneShotDriver(Driver):
    """Serves one command, then reports the work exhausted."""

    def __init__(self):
        self.served = 0
        self.succeeded = 0
        self.limit_policy = _NeverPauses()

    def next_command(self):
        if self.served:
            return None
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")

    def on_success(self, rc):
        self.succeeded += 1


class _NeverPauses:
    """A LimitPolicy whose proactive check always says "plenty left" — so a pause
    in these tests can only have come from the reactive backstop."""

    def describe(self):
        return "stub"

    def log_snapshot(self, *args, **kwargs):
        pass

    def check_and_wait(self, source, session_start, note="", cache_value=True):
        return False, session_start


def _args(project_dir):
    ns = type("NS", (), {})()
    ns.max = None          # None keeps the limit machinery on (a bounded run skips it)
    ns.dry_run = False
    ns.raw = False
    ns.start_in = None
    ns.max_strike = None
    ns.git_push = "none"
    ns.project_dir = project_dir
    ns.cost = False
    return ns


def _run_with_verdict(tmp_path, monkeypatch, verdict):
    """Run one iteration whose fake `claude` streams `verdict`; return the
    wait_until targets the loop asked for."""
    waits = []
    monkeypatch.setattr(usage, "UsageSource", lambda *a, **k: _StubSource())
    monkeypatch.setattr(cyclecore, "wait_until",
                        lambda ts, reason=None: waits.append(ts))

    def fake_run(cmd, raw, partial):
        cyclecore._last_rate_limit_event = verdict
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", fake_run)
    driver = _OneShotDriver()
    cyclecore.run_loop(driver, _args(str(tmp_path)), app_name="pytest-usage")
    return driver, waits


def test_a_refusal_parks_the_loop_until_that_quota_resets(tmp_path, monkeypatch):
    resets = time.time() + 1800
    driver, waits = _run_with_verdict(
        tmp_path, monkeypatch, RateLimitEvent("rejected", "five_hour", resets))
    assert len(waits) == 1, "a refused run did not park the loop"
    assert waits[0] == pytest.approx(resets + 5, abs=0.1)
    # The iteration still counted: a run refused on its last turn may have
    # finished its work first, and dropping that would redo it after the wait.
    assert driver.succeeded == 1


def test_a_weekly_refusal_waits_out_the_week_not_the_session(tmp_path, monkeypatch):
    """The wait follows the quota that actually refused — waking after five hours
    into a weekly wall would just burn the next request the same way."""
    resets = time.time() + 3 * 86400
    _, waits = _run_with_verdict(
        tmp_path, monkeypatch, RateLimitEvent("rejected", "seven_day", resets))
    assert waits[0] == pytest.approx(resets + 5, abs=0.1)


def test_a_refusal_without_a_reset_time_waits_out_a_session(tmp_path, monkeypatch):
    before = time.time()
    _, waits = _run_with_verdict(
        tmp_path, monkeypatch, RateLimitEvent("rejected", "five_hour", None))
    assert waits[0] >= before + cyclecore.CLAUDE_SESSION_DURATION


@pytest.mark.parametrize("verdict", [
    None,
    RateLimitEvent("allowed", "five_hour", time.time() + 1800),
    RateLimitEvent("allowed_warning", "five_hour", time.time() + 1800),
])
def test_anything_short_of_a_refusal_runs_on(tmp_path, monkeypatch, verdict):
    """Only "rejected" is a wall. A warning is worth printing, not stopping for."""
    driver, waits = _run_with_verdict(tmp_path, monkeypatch, verdict)
    assert waits == []
    assert driver.succeeded == 1


def test_the_verdict_does_not_outlive_its_run(monkeypatch):
    """run_claude_streaming clears it on entry — otherwise the run after a refusal
    inherits the refusal and parks the loop a second time for nothing."""
    cyclecore._last_rate_limit_event = RateLimitEvent(
        "rejected", "five_hour", time.time())

    def boom(*a, **k):
        raise FileNotFoundError

    monkeypatch.setattr(cyclecore.subprocess, "Popen", boom)
    with pytest.raises(SystemExit):
        cyclecore.run_claude_streaming(["claude"], raw=False, partial=True)
    assert cyclecore.last_rate_limit_event() is None
