"""Context measurements use the latest main request, not billing accumulators."""
import pytest

from llm_agents.claude.context import ContextTracker


def assistant(tokens=2, cache_create=2858, cache_read=80164, **extra):
    return {"type": "assistant", "parent_tool_use_id": None, "message": {
        "id": "same-message", "model": "claude-opus-5", "usage": {
            "input_tokens": tokens, "cache_creation_input_tokens": cache_create,
            "cache_read_input_tokens": cache_read, "output_tokens": 5951}}, **extra}


def result():
    return {"type": "result", "usage": {"input_tokens": 715678}, "modelUsage": {
        "claude-haiku-4-5": {"canonicalModel": "claude-haiku-4-5", "contextWindow": 200000},
        "claude-opus-5[1m]": {"canonicalModel": "claude-opus-5", "contextWindow": 1000000}}}


def tracker():
    value = ContextTracker()
    value.observe({"type": "system", "subtype": "init", "model": "claude-opus-5[1m]"})
    return value


def test_cached_input_matches_real_review_and_ignores_billing_totals():
    value = tracker()
    value.observe(assistant())
    value.observe(result())
    assert value.snapshot() == {"model": "claude-opus-5[1m]", "input_tokens": 83024,
                                "context_window": 1000000, "used_percent": 8.3024,
                                "warning": False}


def test_unknown_capacity_is_not_inferred_from_1m_model_name():
    value = tracker()
    value.observe(assistant(tokens=700000))
    assert value.snapshot()["context_window"] is None
    assert value.snapshot()["used_percent"] is None
    assert not value.snapshot()["warning"]


@pytest.mark.parametrize("tokens,warning", [(449999, False), (450000, True), (450001, True)])
def test_warning_threshold_is_inclusive_and_not_rounded(tokens, warning):
    value = tracker()
    value.observe(assistant(tokens=tokens, cache_create=0, cache_read=0))
    value.observe(result())
    assert value.snapshot()["warning"] is warning


def test_partial_duplicates_replace_instead_of_accumulating():
    value = tracker()
    event = assistant()
    assert value.observe(event) is not None
    assert value.observe(event) is None
    event["message"]["usage"] = {"output_tokens": 1000000}
    assert value.observe(event) is None
    assert value.snapshot()["input_tokens"] == 83024
    value.observe(assistant(tokens=12, cache_create=0, cache_read=0))
    assert value.snapshot()["input_tokens"] == 12


def test_subagent_events_cannot_replace_model_usage_or_capacity():
    value = tracker()
    value.observe(assistant())
    child = assistant(tokens=900000, parent_tool_use_id="tool-child")
    child["message"]["model"] = "claude-haiku-4-5"
    assert value.observe(child) is None
    child_result = result() | {"parent_tool_use_id": "tool-child"}
    assert value.observe(child_result) is None
    assert value.snapshot()["context_window"] is None
    value.observe(result())
    assert value.snapshot()["context_window"] == 1000000


@pytest.mark.parametrize("event", [
    {"type": "system", "subtype": "compact_boundary", "compact_metadata": {"pre_tokens": 900000}},
    {"type": "system", "subtype": "status", "status": "compacting"},
])
def test_compaction_clears_warning_until_next_main_request(event):
    value = tracker()
    value.observe(assistant(tokens=450000, cache_create=0, cache_read=0))
    value.observe(result())
    assert value.snapshot()["warning"]
    value.observe(event)
    assert value.snapshot()["input_tokens"] is None
    assert value.snapshot()["used_percent"] is None
    assert not value.snapshot()["warning"]
    value.observe(result())
    assert value.snapshot()["input_tokens"] is None
    value.observe(assistant())
    assert value.snapshot()["used_percent"] == 8.3024


def test_previous_turn_capacity_supports_live_updates_and_model_change_resets_it():
    previous = tracker()
    previous.observe(assistant())
    previous.observe(result())
    value = ContextTracker(previous.snapshot())
    value.observe({"type": "system", "subtype": "init", "model": "claude-opus-5[1m]"})
    value.observe(assistant(tokens=450000, cache_create=0, cache_read=0))
    assert value.snapshot()["warning"]
    value.observe({"type": "system", "subtype": "init", "model": "claude-sonnet-5"})
    assert value.snapshot()["context_window"] is None
    assert value.snapshot()["input_tokens"] is None


def test_canonical_match_and_ambiguous_capacities():
    value = ContextTracker()
    value.observe(assistant())
    value.observe(result())
    assert value.snapshot()["context_window"] == 1000000
    ambiguous = result()
    ambiguous["modelUsage"]["other-routing-key"] = {
        "canonicalModel": "claude-opus-5", "contextWindow": 200000}
    value.observe(ambiguous)
    assert value.snapshot()["context_window"] is None


def test_exact_configuration_wins_over_other_matching_canonical_windows():
    value = tracker()
    data = result()
    data["modelUsage"]["claude-opus-5"] = {"canonicalModel": "claude-opus-5", "contextWindow": 200000}
    value.observe(data)
    assert value.snapshot()["context_window"] == 1000000


def test_stream_message_start_updates_without_waiting_for_full_assistant_block():
    value = tracker()
    value.observe({"type": "stream_event", "event": {
        "type": "message_start", "message": assistant()["message"]}})
    assert value.snapshot()["input_tokens"] == 83024
    assert value.observe({"type": "stream_event", "event": {
        "type": "message_delta", "usage": {"output_tokens": 99}}}) is None


def test_auxiliary_only_result_and_zero_input_do_not_invent_usage():
    value = tracker()
    value.observe({"type": "result", "usage": {"input_tokens": 999999}, "modelUsage": {
        "claude-haiku-4-5": {"canonicalModel": "claude-haiku-4-5", "contextWindow": 200000}}})
    assert value.snapshot()["context_window"] is None
    assert value.snapshot()["input_tokens"] is None
    value.observe(assistant(tokens=0, cache_create=0, cache_read=0))
    value.observe(result())
    assert value.snapshot()["used_percent"] == 0
    assert not value.snapshot()["warning"]
