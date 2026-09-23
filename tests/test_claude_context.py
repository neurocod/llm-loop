"""Claude occupancy uses the latest main request, including both cache buckets."""

import pytest

from llm_loop import statusline as sl, wire


MODEL = "claude-opus-5-5"


def request(tokens=2, created=8246, cached=32160, model=MODEL):
    # Trimmed from a real CLI review stream, measured 2026-09-24.
    return {"type": "assistant", "parent_tool_use_id": None, "message": {
        "model": model, "usage": {
            "input_tokens": tokens, "cache_creation_input_tokens": created,
            "cache_read_input_tokens": cached, "output_tokens": 3}}}


def capacity():
    return {"type": "result", "usage": {"input_tokens": 9_000_000},
            "modelUsage": {
                "claude-haiku-test": {"contextWindow": 200_000},
                MODEL + "[1m]": {"canonicalModel": MODEL,
                                  "contextWindow": 1_000_000}}}


def job_with_usage():
    job = sl.Job(model="opus")
    job.observe_claude_event({"type": "system", "subtype": "init",
                              "model": MODEL + "[1m]"})
    job.observe_claude_event(request())
    return job


def test_live_tokens_then_reported_capacity_and_duplicate_snapshots():
    job = job_with_usage()
    assert job.context_label() == "ctx 40k"
    assert job.context_window is None  # A routing tag is not a measurement.
    job.observe_claude_event(request())
    assert job.context_tokens == 40_408
    job.observe_claude_event(capacity())
    assert job.snapshot().context_label() == "ctx 40k/1M (4%)"
    assert job.model_label() == MODEL + "[1m]"
    job.observe_claude_event(request(2, 0, 9998))
    assert job.context_label() == "ctx 10k/1M (1%)"


def test_message_start_updates_but_output_deltas_do_not_change_occupancy():
    job = sl.Job()
    job.observe_claude_event({"type": "stream_event", "event": {
        "type": "message_start", "message": request()["message"]}})
    assert job.context_tokens == 40_408
    assert job.model_label() == MODEL
    job.observe_claude_event({"type": "stream_event", "event": {
        "type": "message_delta", "usage": {"output_tokens": 900_000}}})
    event = request()
    event["message"]["usage"] = {"output_tokens": 900_000}
    job.observe_claude_event(event)
    assert job.context_tokens == 40_408


@pytest.mark.parametrize("streaming", [False, True])
def test_cli_synthetic_errors_do_not_replace_the_real_model_or_usage(streaming):
    job = job_with_usage()
    job.observe_claude_event(capacity())
    event = request(0, 0, 0, "<synthetic>")
    event["isApiErrorMessage"] = True
    if streaming:
        event = {"type": "stream_event", "event": {
            "type": "message_start", "message": event["message"]}}
    job.observe_claude_event(event)
    job.observe_claude_event(capacity())
    assert job.model_label() == MODEL + "[1m]"
    assert job.context_tokens == 40_408
    assert job.context_label() == "ctx 40k/1M (4%)"


@pytest.mark.parametrize("event", [
    request(900_000, 0, 0, "child-model"), capacity(),
    {"type": "system", "subtype": "init", "model": "child-model"},
    {"type": "system", "subtype": "compact_boundary"},
    {"type": "stream_event", "event": {
        "type": "message_start", "message": request(900_000)["message"]}},
])
def test_subagent_events_cannot_change_the_main_context(event):
    job = job_with_usage()
    job.observe_claude_event(dict(event, parent_tool_use_id="child-tool"))
    assert job.context_tokens == 40_408
    assert job.context_window is None
    assert job.resolved_model == MODEL + "[1m]"


@pytest.mark.parametrize("event", [
    {"type": "system", "subtype": "compact_boundary",
     "compact_metadata": {"pre_tokens": 900_000}},
    {"type": "system", "subtype": "status", "status": "compacting"},
])
def test_compaction_clears_occupancy_until_next_request(event):
    job = job_with_usage()
    job.observe_claude_event(capacity())
    job.observe_claude_event(event)
    assert job.context_label() == ""
    job.observe_claude_event(capacity())
    assert job.context_label() == ""
    job.observe_claude_event(request(1000, 0, 0))
    assert job.context_tokens == 1000
    assert job.context_window == 1_000_000


@pytest.mark.parametrize("value", [None, True, -1, "2", 2.5])
@pytest.mark.parametrize("field", [
    "input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"])
def test_invalid_input_counters_do_not_invent_a_reading(field, value):
    job = job_with_usage()
    event = request()
    event["message"]["usage"][field] = value
    job.observe_claude_event(event)
    assert job.context_tokens is None


def test_missing_cache_fields_and_zero_input_are_supported():
    job = job_with_usage()
    event = request()
    event["message"]["usage"] = {"input_tokens": 0}
    job.observe_claude_event(event)
    job.observe_claude_event(capacity())
    assert job.context_label() == "ctx 0/1M (0%)"


@pytest.mark.parametrize("event", [
    {"type": "system", "subtype": "init", "model": "new-model"},
    request(1, 0, 0, "new-model"),
])
def test_model_switch_forgets_previous_capacity(event):
    job = job_with_usage()
    job.observe_claude_event(capacity())
    job.observe_claude_event(event)
    assert job.resolved_model == "new-model"
    assert job.context_window is None
    assert job.context_tokens == (1 if event["type"] == "assistant" else None)


@pytest.mark.parametrize("action", ["start", "select"])
def test_next_iteration_forgets_claude_context(action):
    job = job_with_usage()
    job.observe_claude_event(capacity())
    getattr(job, action)(model="next")
    assert job.context_label() == ""
    assert job.context_window is None


def test_capacity_does_not_take_a_lone_auxiliary_model_or_ambiguous_match():
    event = {"type": "result", "modelUsage": {
        "auxiliary-model": {"contextWindow": 200_000}}}
    assert wire.result_context_window(event, MODEL) is None
    event = capacity()
    assert wire.result_context_window(event, MODEL) == 1_000_000
    event["modelUsage"]["other-route"] = {
        "canonicalModel": MODEL, "contextWindow": 200_000}
    assert wire.result_context_window(event, MODEL) is None
    assert wire.result_context_window(event, MODEL + "[1m]") == 1_000_000


@pytest.mark.parametrize("event", [
    {"type": "assistant", "message": None},
    {"type": "assistant", "message": {"usage": "bad"}},
    {"type": "stream_event", "event": None},
    {"type": "stream_event", "event": {"type": "message_start", "message": []}},
])
def test_malformed_events_leave_the_last_reading_alone(event):
    job = job_with_usage()
    job.observe_claude_event(event)
    assert job.context_tokens == 40_408
