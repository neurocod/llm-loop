"""One logical turn: a run plus the `--expect` follow-ups, merged into one outcome."""
from __future__ import annotations

from pathlib import Path
from time import monotonic

from ..answer_contract import (
    EXPECT_FOLLOWUP_PROMPT, expect_follow_up, expect_verdict, follow_up_failed,
    join_answers, needs_expect_follow_up)

from .protocol import run_turn, save_outcome
from .state import Store


def add_usage(first: dict, followed: dict) -> dict:
    """Two attempts' `result.usage`, summed; either one missing makes it `{}`.

    The counters are per CLI invocation, so they add up: measured 2026-09-28
    over 69 resumed turn pairs in `.claude-agents/`, output tokens fell from
    one turn to the next by up to ×33 (`review-developer-flight` 48416 → 1442).
    Nested counters recurse, lists (`iterations`) concatenate, labels
    (`service_tier`, `speed`) keep the follow-up's. A missing measurement
    stays missing rather than passing one attempt off as the total.
    """
    if not first or not followed:
        return {}
    merged = dict(followed)
    for key, value in first.items():
        other = followed.get(key)
        if key not in followed:
            merged[key] = value
        elif isinstance(value, dict) and isinstance(other, dict):
            merged[key] = add_usage(value, other)
        elif isinstance(value, list) and isinstance(other, list):
            merged[key] = [*value, *other]
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (value, other)):
            merged[key] = value + other
    return merged


def _completed(outcome: dict) -> bool:
    """A turn with a result to judge: completed and without an error."""
    return outcome.get("status") == "completed" and not outcome.get("error")


def merge_follow_up(first: dict, followed: dict, follow_up_dir: Path) -> dict:
    """One logical turn's outcome from its first attempt and its follow-up.

    The answers are joined by `join_answers`; a failed follow-up keeps the
    first answer and reports its own status, with the error naming its log
    directory. `log_dir` stays the
    first attempt's, the directory `Logs:` points at.
    """
    expect = first["expect"]
    merged = followed | {
        "expect": expect, "follow_ups": first["follow_ups"] + 1, "log_dir": first.get("log_dir"),
        "follow_up_log_dirs": [*first.get("follow_up_log_dirs", []), str(follow_up_dir)],
        "permission_denials": [*first.get("permission_denials", []),
                               *followed.get("permission_denials", [])],
        "usage": add_usage(first.get("usage") or {}, followed.get("usage") or {}),
        # NOT summed: on a resumed session the CLI reports the session's
        # running total. Same 2026-09-28 measurement: every pair since
        # 2026-09-19 reported more than the turn before even where output
        # shrank ×10 (`review-g` 6398 → 651 tokens,
        # $0.78 → $0.87), so the follow-up's figure already includes the
        # first attempt. Earlier pairs were per invocation: an older CLI
        # under-reports here, it never bills the first attempt twice.
        "total_cost_usd": followed.get("total_cost_usd")}
    # Wall time of both invocations; a missing figure stays missing.
    if first.get("duration_s") is not None and followed.get("duration_s") is not None:
        merged["duration_s"] = round(first["duration_s"] + followed["duration_s"], 6)
    if not _completed(followed):
        merged["answer"] = first.get("answer", "")
        merged["error"] = follow_up_failed(
            expect_follow_up(expect), followed.get("error") or followed.get("status"),
            logs=follow_up_dir)
    else:
        merged["answer"] = join_answers(first.get("answer", ""), followed.get("answer") or "")
    return merged


def answer_turn(store: Store, name: str, owner: str, turn: dict, directory: Path,
                on_week_used=None) -> dict:
    """Run a turn, then follow it up in the same session while the answer
    lacks what the turn's `expect` names — `codex_agent`'s `--expect`, with the
    same prompt and the same single follow-up (`answer_contract`).

    Only a completed answer is judged: a failed turn is short of a result, not
    of a section. The two attempts become one outcome in `merge_follow_up`. A
    completed turn records `expect_met`, so an answer still incomplete after
    the follow-up never reads as a clean success. A cancel request, or a
    `turn_timeout` already spent, skips the follow-up; the first answer stands.

    `--turn-timeout` bounds the logical turn, not each invocation: both share
    one deadline. The merged result is rewritten into the turn's own
    artifacts (`save_outcome`), the directory `Logs:` points at.

    `on_week_used` goes to every invocation: the follow-up's stream reports
    the weekly figure too (`protocol.run_turn`).
    """
    def cancelled() -> bool:
        return store.cancelled(name, owner)

    timeout = store.get(name)["options"].get("turn_timeout")
    deadline = monotonic() + timeout if timeout else None

    def run(turn: dict, directory: Path) -> dict:
        return run_turn(store.get(name), turn, directory, cancelled,
                        on_session=lambda session: store.session_ready(name, owner, session),
                        on_context=lambda context: store.context_ready(name, owner, context),
                        on_week_used=on_week_used, deadline=deadline)

    outcome = run(turn, directory)
    expect = turn.get("expect") or ""
    if not expect:
        return outcome
    outcome |= {"expect": expect, "follow_ups": 0}
    while (needs_expect_follow_up(expect, outcome.get("answer") or "",
                                  outcome["follow_ups"], completed=_completed(outcome))
           and not cancelled() and (deadline is None or monotonic() < deadline)):
        # Sibling of the turn's own log directory, so `status` activity
        # (`turn-*/events.jsonl`) keeps seeing the follow-up's stream.
        follow_up_dir = directory.with_name(
            f"{directory.name}-follow-up-{outcome['follow_ups'] + 1}")
        followed = run({"turn": turn["turn"], "resume": True,
                        "prompt": EXPECT_FOLLOWUP_PROMPT.format(pattern=expect)},
                       follow_up_dir)
        outcome = merge_follow_up(outcome, followed, follow_up_dir)
    verdict = expect_verdict(expect, outcome.get("answer") or "",
                             completed=_completed(outcome))
    if verdict is not None:
        outcome["expect_met"] = verdict
    if outcome["follow_ups"]:
        save_outcome(directory, outcome)
    return outcome
