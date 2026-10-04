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

import time
from datetime import datetime, timezone

import pytest

from llm_loop import (console, cyclecore, parallel, providers, runlifecycle,
                      streamrender, usage)
from llm_loop.usage import RateLimitEvent
from llm_loop.agentwork import ClaudeCommand, Driver
from llm_loop.limits import DayNightLimit, LimitPolicy, SessionLimit, WeeklyLimit

from _runfixtures import MemListDriver, StubPolicy, isolated_run, par_args, seq_args


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
def _isolated_run(tmp_path, monkeypatch):
    with isolated_run(monkeypatch, tmp_path):
        yield


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


def test_the_quota_table_is_the_one_naming_of_a_window():
    """Parser key, Usage field, log label and status-line abbreviation come from
    one row of usage.QUOTAS, so the four cannot drift apart."""
    u = usage.parse_usage(SAMPLE)
    readings = u.readings()

    assert [q.field for q, _ in readings] == ["session", "week_all", "week_sonnet"]
    assert [q.short for q, _ in readings] == ["session", "week", "week/sonnet"]
    assert [r for _, r in readings] == [u.session, u.week_all, u.week_sonnet]
    # The windows every plan has stay on the status line even without a figure;
    # a plan-specific one is only shown when there is something to show.
    assert [q.always for q, _ in readings] == [True, True, False]
    for quota in usage.QUOTAS:
        assert usage.QUOTA_BY_FIELD[quota.field] is quota
        assert getattr(u, quota.field) is not None


def test_a_rule_selects_its_reading_and_labels_itself_from_that_table():
    u = usage.parse_usage(SAMPLE)

    assert DayNightLimit().reading(u) == u.session
    assert WeeklyLimit().reading(u) == u.week_all
    assert WeeklyLimit(sonnet_only=True).reading(u) == u.week_sonnet
    assert DayNightLimit().label == usage.QUOTA_BY_FIELD["session"].label
    assert WeeklyLimit().label == usage.QUOTA_BY_FIELD["week_all"].label


def test_the_policy_answers_which_rule_watches_a_window():
    """What the status line asks to decide whether it has a policy half to show
    for a window — the provider's own half is shown either way."""
    policy = LimitPolicy([DayNightLimit(), WeeklyLimit(90)])

    assert isinstance(policy.rule_for("session"), DayNightLimit)
    assert isinstance(policy.rule_for("week_all"), WeeklyLimit)
    assert policy.rule_for("week_sonnet") is None
    assert LimitPolicy([]).rule_for("session") is None


def test_a_rules_status_is_its_live_ceiling_by_default():
    """The default contribution is the one number a rule adds — and for
    DayNightLimit it moves with the window, exactly as the gate does."""
    now = time.time()
    u = usage.parse_usage({"five_hour": {"utilization": 9.0,
                                         "resets_at": _iso_in(4 * 3600)}})
    rule = DayNightLimit(day=80, night=80)
    reading = rule.reading(u)

    assert rule.status(reading, now) == "ceil 80%"
    assert WeeklyLimit(90).status(u.week_all, now) == "ceil 90%"
    # 10 minutes from the reset the ceiling has climbed; the row says so too.
    near = reading.reset_ts - 600
    assert rule.status(reading, near) == f"ceil {rule.ceiling(reading, near):.0f}%"
    assert rule.status(reading, near) != "ceil 80%"


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
    assert console.fmt_left(seconds) == expected


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


@pytest.mark.parametrize("sonnet_only", [False, True])
@pytest.mark.parametrize("limit", [0, 100])
def test_zero_weekly_limit_does_not_wait_at_full_usage(monkeypatch, capsys, sonnet_only, limit):
    field = "seven_day_sonnet" if sonnet_only else "seven_day"
    source = _StubSource({field: {"utilization": 100}})
    rule = WeeklyLimit(limit, sonnet_only=sonnet_only)
    policy = LimitPolicy([rule])
    monkeypatch.setattr(policy, "_wait", lambda *args: pytest.fail("disabled weekly limit waited"))
    assert policy.check_and_wait(source, 123) == (False, 123)
    assert "ceiling N/A" in capsys.readouterr().out
    assert "no ceiling" in policy.describe()


@pytest.mark.parametrize("limit", [0, 100])
@pytest.mark.parametrize("smart", [False, True])
def test_explicit_unlimited_session_does_not_wait_at_full_usage(
        monkeypatch, capsys, limit, smart):
    source = _StubSource({"five_hour": {"utilization": 100,
                                          "resets_at": _iso_in(2 * 3600)}})
    rule = DayNightLimit(day=limit, night=limit) if smart else SessionLimit(limit)
    policy = LimitPolicy([rule])
    monkeypatch.setattr(policy, "_wait", lambda *a: pytest.fail("unlimited session waited"))
    assert policy.check_and_wait(source, 123) == (False, 123)
    assert "ceiling N/A" in capsys.readouterr().out
    assert "no ceiling" in policy.describe()


@pytest.mark.parametrize("rule", [SessionLimit(99), WeeklyLimit(99),
                                  DayNightLimit(day=99, night=99)])
def test_a_configured_99_still_pauses_at_its_ceiling(rule):
    now = datetime(2026, 10, 4, 12).timestamp()
    below = usage.parse_usage({"five_hour": {"utilization": 98.9},
                               "seven_day": {"utilization": 98.9}})
    at_limit = usage.parse_usage({"five_hour": {"utilization": 99},
                                  "seven_day": {"utilization": 99}})
    policy = LimitPolicy([rule])
    assert not policy._violations(policy._status(below, now))
    assert [r for r, _, _ in policy._violations(policy._status(at_limit, now))] == [rule]


def test_dynamic_session_ceiling_reaching_100_keeps_the_usage_gate():
    now = datetime(2026, 10, 4, 12).timestamp()
    rule = DayNightLimit(day=96, night=96)
    reading = usage.UsageReading(100, now)
    assert rule.ceiling(reading, now) == 100
    assert LimitPolicy._violations([(rule, reading, rule.ceiling(reading, now))])


@pytest.mark.parametrize("night_unlimited", [False, True])
def test_day_night_unlimited_applies_only_to_the_selected_base(night_unlimited):
    morning = datetime(2026, 10, 4, 9).timestamp()
    night = datetime(2026, 10, 4, 2).timestamp()
    daytime = datetime(2026, 10, 4, 12).timestamp()
    reading = usage.UsageReading(100, morning)
    rule = DayNightLimit(day=96 if night_unlimited else 100,
                         night=100 if night_unlimited else 96)
    assert rule.ceiling(reading, night) == (float("inf") if night_unlimited else 96)
    assert rule.ceiling(usage.UsageReading(100, None), daytime) == (
        96 if night_unlimited else float("inf"))


# -- the rate_limit_event backstop ---------------------------------------------

def test_event_parse():
    ev = {"type": "rate_limit_event", "rate_limit_info": {
        "status": "rejected", "resetsAt": 1786807200, "rateLimitType": "five_hour"}}
    rl = usage.rate_limit_event_from(ev)
    assert (rl.status, rl.limit_type, rl.resets_at) == (
        "rejected", "five_hour", 1786807200.0)
    assert rl.label == "session limit"
    assert usage.rate_limit_event_from({"type": "result"}) is None
    # A verdict with fields missing still parses — it must not throw mid-stream.
    bare = usage.rate_limit_event_from({"type": "rate_limit_event"})
    assert bare.resets_at is None and bare.status == "unknown"


class _TwoShotDriver(Driver):
    """Serve a second command so a first-turn refusal gates real pending work."""

    def __init__(self):
        self.served = 0
        self.succeeded = 0
        # The proactive check always says "plenty left" — so a pause in these
        # tests can only have come from the reactive backstop.
        self.limit_policy = StubPolicy()

    def next_command(self):
        if self.served >= 2:
            return None
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")

    def on_success(self, rc):
        self.succeeded += 1


@pytest.mark.parametrize("runner", ["sequential", "parallel"])
def test_new_codex_tasks_start_after_both_quotas_reach_100(
        tmp_path, monkeypatch, runner):
    source = _StubSource({"five_hour": {"utilization": 100},
                          "seven_day": {"utilization": 100}})
    policy = LimitPolicy([DayNightLimit(day=100, night=100), WeeklyLimit(100)])
    monkeypatch.setattr(policy, "_wait", lambda *a: pytest.fail("100% blocked new tasks"))
    monkeypatch.setattr(runlifecycle, "usage_source_for", lambda provider: source)
    monkeypatch.setattr(cyclecore, "run_agent_streaming", lambda *a, **k: 0)
    monkeypatch.setattr(parallel, "run_job", lambda *a, **k: (0, 0.0, 0.01))
    if runner == "sequential":
        driver = _TwoShotDriver()
        driver.limit_policy = policy
        result = cyclecore.run_loop(
            driver, seq_args(tmp_path, provider="codex", no_statusline=True),
            app_name="pytest-usage", wait_on_start=False)
    else:
        driver = MemListDriver(["products/a.md", "products/b.md"])
        driver.limit_policy = policy
        result = parallel.run_parallel(
            driver, par_args(tmp_path, provider="codex", jobs=2, ignore_usage=False,
                             no_statusline=True),
            app_name="pytest-usage", wait_on_start=False)
    assert result.completed == 2


def _run_with_verdict(tmp_path, monkeypatch, verdict):
    """Run two iterations whose first fake `claude` streams `verdict`; return the
    wait_until targets the loop asked for."""
    waits = []
    monkeypatch.setattr(usage, "UsageSource", lambda *a, **k: _StubSource())
    monkeypatch.setattr(cyclecore, "wait_until",
                        lambda ts, reason=None, should_stop=None:
                            waits.append(ts))

    def fake_run(cmd, raw, partial, prompt="", mailbox=None):
        streamrender._last_rate_limit_event = verdict if driver.served == 1 else None
        return 0

    monkeypatch.setattr(cyclecore, "run_claude_streaming", fake_run)
    driver = _TwoShotDriver()
    # No --max-runs: a bounded run skips the limit machinery this is about.
    cyclecore.run_loop(driver, seq_args(tmp_path), app_name="pytest-usage")
    return driver, waits


def test_a_refusal_parks_the_loop_until_that_quota_resets(tmp_path, monkeypatch):
    resets = time.time() + 1800
    driver, waits = _run_with_verdict(
        tmp_path, monkeypatch, RateLimitEvent("rejected", "five_hour", resets))
    assert len(waits) == 1, "a refused run did not park the loop"
    assert waits[0] == pytest.approx(resets + 5, abs=0.1)
    # The iteration still counted: a run refused on its last turn may have
    # finished its work first, and dropping that would redo it after the wait.
    assert driver.succeeded == 2


def test_a_weekly_refusal_waits_out_the_week_not_the_session(tmp_path, monkeypatch):
    """The wait follows the quota that actually refused — waking after five hours
    into a weekly wall would just burn the next request the same way."""
    resets = time.time() + 3 * 86400
    _, waits = _run_with_verdict(
        tmp_path, monkeypatch, RateLimitEvent("rejected", "seven_day", resets))
    assert waits[0] == pytest.approx(resets + 5, abs=0.1)


# The length of the session window, spelled here INDEPENDENTLY of the engine's
# `usage.CLAUDE_SESSION_DURATION`, because "five hours" is not this test's guess:
# it is what the quota id on the wire — "five_hour", the very string the verdict
# below carries — names. The engine's constant adds an unmeasured +3 s margin and
# the loop adds +5 s more, both of which the tolerance below absorbs.
#
# It has to be an independent spelling. Read from the engine, the assertion
# compares the code's answer against the code's own input and passes for ANY
# value of it: with `CLAUDE_SESSION_DURATION` set to one second the old form was
# still green, so it could catch a wrong FORMULA (waiting for something other
# than a session) and never a wrong NUMBER. That was the whole defect.
SESSION_WINDOW_S = 5 * 60 * 60
# Slack above the window: the two margins (3 s + 5 s) plus whatever the run took.
# Wide enough never to flake, far too narrow to hide a window of the wrong size.
SESSION_WINDOW_SLACK_S = 60


def test_a_refusal_without_a_reset_time_waits_out_a_session(tmp_path, monkeypatch):
    before = time.time()
    _, waits = _run_with_verdict(
        tmp_path, monkeypatch, RateLimitEvent("rejected", "five_hour", None))
    assert waits[0] >= before + SESSION_WINDOW_S, \
        "a refusal with no reset time woke up before the session window was out"
    assert waits[0] <= before + SESSION_WINDOW_S + SESSION_WINDOW_SLACK_S, \
        "the wait is longer than one session window — the quota is back by then"


@pytest.mark.parametrize("verdict", [
    None,
    RateLimitEvent("allowed", "five_hour", time.time() + 1800),
    RateLimitEvent("allowed_warning", "five_hour", time.time() + 1800),
])
def test_anything_short_of_a_refusal_runs_on(tmp_path, monkeypatch, verdict):
    """Only "rejected" is a wall. A warning is worth printing, not stopping for."""
    driver, waits = _run_with_verdict(tmp_path, monkeypatch, verdict)
    assert waits == []
    assert driver.succeeded == 2


def test_the_verdict_does_not_outlive_its_run(monkeypatch):
    """run_claude_streaming clears it on entry — otherwise the run after a refusal
    inherits the refusal and parks the loop a second time for nothing."""
    streamrender._last_rate_limit_event = RateLimitEvent(
        "rejected", "five_hour", time.time())

    def boom(*a, **k):
        raise FileNotFoundError

    # `providers`, because that is the module that launches the CLI. This used
    # to say `cyclecore.subprocess`, which worked only because a module object
    # is shared process-wide — an address that outlived cyclecore's own use of
    # `subprocess` and pointed at nothing this test is about.
    monkeypatch.setattr(providers.subprocess, "Popen", boom)
    with pytest.raises(SystemExit):
        cyclecore.run_claude_streaming(["claude"], raw=False, partial=True)
    assert cyclecore.last_rate_limit_event() is None
