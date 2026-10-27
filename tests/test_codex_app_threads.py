"""Which app-server notifications are the main thread's, which a sub-agent's,
and which belong to the one turn a provider run started.

One table for every reader of the stream (`wire.codex_app_from_sub_agent`,
`wire.codex_app_in_turn`, `wire.codex_app_learn_main_thread`): the provider
adapter here and hosts that drive the app-server themselves. The shapes are
trimmed from real app-server logs of 2026-10 — ids shortened, the keys that
declare a thread or turn kept as the server sent them.
"""

import pytest

from llm_loop import wire

MAIN, CHILD = "th-main", "th-child"


def _note(method, **params):
    return {"method": method, "params": params}


@pytest.mark.parametrize("message, main, sub", [
    # A declared thread decides once the main one is known.
    (_note("item/completed", threadId=CHILD, turnId="t-child", item={}),
     MAIN, True),
    (_note("item/completed", threadId=MAIN, turnId="t-main", item={}),
     MAIN, False),
    (_note("thread/tokenUsage/updated", threadId=CHILD, turnId="t-child"),
     MAIN, True),
    # Children of older Codex versions carry no `rootTurnId`: only the thread
    # tells them apart, so a reader that does not know the main thread yet
    # cannot (`codex_app_learn_main_thread` is why it learns it first).
    (_note("turn/started", threadId=CHILD, turn={"id": "t-child"}),
     MAIN, True),
    (_note("turn/started", threadId=CHILD, turn={"id": "t-child"}),
     "", False),
    # A child's turn names the main turn as its root — no state needed.
    (_note("turn/started", threadId=CHILD,
           turn={"id": "t-child", "rootTurnId": "t-main"}), "", True),
    (_note("turn/completed", turn={"id": "t-child", "rootTurnId": "t-main"}),
     MAIN, True),
    (_note("turn/completed", threadId=MAIN,
           turn={"id": "t-main", "rootTurnId": "t-main"}), "", False),
    # The main thread named outright is the main thread, whatever its turn's
    # root says: read as a child's, its `turn/completed` would go unread.
    (_note("turn/completed", threadId=MAIN,
           turn={"id": "t-main2", "rootTurnId": "t-main"}), MAIN, False),
    # A thread that names a parent is a child even before ours is known.
    (_note("thread/started", thread={"id": CHILD, "parentThreadId": MAIN,
                                     "forkedFromId": None}), "", True),
    (_note("thread/started", thread={"id": MAIN, "parentThreadId": None,
                                     "forkedFromId": None}), "", False),
    # Nothing declared: account-wide notifications, and test stand-ins.
    (_note("account/rateLimits/updated"), MAIN, False),
    (_note("turn/completed", turn={}), MAIN, False),
    ({"method": "turn/completed", "params": None}, MAIN, False),
    ({"method": "turn/completed", "params": ["not", "a", "dict"]}, MAIN, False),
])
def test_which_notifications_are_a_sub_agents(message, main, sub):
    assert wire.codex_app_from_sub_agent(message, main) is sub


@pytest.mark.parametrize("message, ours", [
    (_note("item/completed", threadId=MAIN, turnId="t-main", item={}), True),
    (_note("item/completed", threadId=MAIN, turnId="t-old", item={}), False),
    (_note("turn/completed", threadId=MAIN,
           turn={"id": "t-main", "status": "completed"}), True),
    (_note("turn/completed", threadId=MAIN,
           turn={"id": "t-next", "rootTurnId": "t-next"}), False),
    # Every declared turn must be ours, and one must be declared: a malformed
    # completion cannot finish the job.
    (_note("turn/completed", threadId=MAIN, turn={"status": "completed"}),
     False),
    (_note("item/completed", threadId=MAIN, turnId="t-main",
           turn={"id": "t-other"}), False),
    (_note("thread/status/changed", threadId=MAIN), False),
    (_note("account/rateLimits/updated"), False),
    # A turn named without a thread is still that turn.
    (_note("item/completed", turnId="t-main", item={}), True),
    # `thread/started` has no turn.
    (_note("thread/started", thread={"id": MAIN, "parentThreadId": None}),
     True),
    ({"method": "item/completed", "params": None}, False),
])
def test_which_main_thread_notifications_are_the_turn_we_started(message, ours):
    assert wire.codex_app_in_turn(message, "t-main") is ours


def test_a_log_reader_learns_the_main_thread_from_its_first_start():
    """A resumed thread's log holds no `thread/started` and no responses; the
    main turn's `turn/started` names it, and a child's start after it does
    not replace it — with a root mark or without one."""
    log = [
        _note("account/rateLimits/updated"),
        _note("item/completed", threadId=MAIN, turnId="t-main", item={}),
        # A child marked as one is skipped even before the main is known.
        _note("turn/started", threadId=CHILD,
              turn={"id": "t-child", "rootTurnId": "t-main"}),
        _note("turn/started", threadId=MAIN,
              turn={"id": "t-main", "rootTurnId": "t-main"}),
        _note("turn/started", threadId="th-old-child", turn={"id": "t-x"}),
    ]
    main, seen = "", []
    for message in log:
        main = wire.codex_app_learn_main_thread(message, main)
        seen.append(main)
    assert seen == ["", "", "", MAIN, MAIN]


def test_a_fresh_threads_log_learns_it_from_thread_started():
    main = wire.codex_app_learn_main_thread(
        _note("thread/started", thread={"id": MAIN, "parentThreadId": None}),
        "")
    assert main == MAIN
    assert wire.codex_app_learn_main_thread(
        _note("thread/started", thread={"id": CHILD, "parentThreadId": MAIN}),
        "") == ""


@pytest.mark.parametrize("message, thread", [
    (_note("item/completed", threadId=MAIN), MAIN),
    (_note("thread/started", thread={"id": MAIN}), MAIN),
    (_note("account/updated"), ""),
    ({"method": "x", "params": "text"}, ""),
])
def test_the_declared_thread(message, thread):
    assert wire.codex_app_event_thread(message) == thread
