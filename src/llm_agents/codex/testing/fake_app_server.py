"""Stand-in codex app-server for the offline wire tests."""
import json
import os
import sys
import time

SCENARIO = sys.argv[1]
WORK = sys.argv[2]
# The sentence a real drop carried, handed over rather than spelled again here:
# one string, so rewording the pin cannot leave the stand-in replaying a shape
# OpenAI never sent.
DROP_MESSAGE = sys.argv[3]
NETWORK_FIXTURE, NETWORK_TIMES = json.loads(sys.argv[4])

sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

RECEIVED = os.path.join(WORK, "received.jsonl")


def out(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def read():
    while True:
        line = sys.stdin.readline()
        if not line:
            return None
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        with open(RECEIVED, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(msg) + "\n")
        return msg


def note(method, params=None):
    out({"jsonrpc": "2.0", "method": method, "params": params or {}})


def item(payload):
    note("item/completed", {"item": payload})


def ask(request_id, method, params=None):
    """Server-initiated request; blocks until the client answers it."""
    out({"jsonrpc": "2.0", "id": request_id, "method": method,
         "params": params or {}})
    while True:
        msg = read()
        if msg is None:
            sys.exit(0)
        if msg.get("id") == request_id and "method" not in msg:
            return msg


def wait_for(path, limit=10.0):
    deadline = time.monotonic() + limit
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.02)
    return False


def attempt_number():
    """Which attempt at this turn is running, counted on disk.

    A service-drop retry starts a FRESH server process, so an in-memory
    counter would say "1" every time.
    """
    path = os.path.join(WORK, "attempts.txt")
    seen = 0
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            seen = int(handle.read() or 0)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(str(seen + 1))
    return seen + 1


def usage(total_in, cached, out, occupied):
    """Report token usage the way the server does — CUMULATIVELY OVER THE
    THREAD, so a resumed turn carries on from the dropped one's numbers rather
    than restarting at zero. The pairs the callers pass are the real ones,
    lifted from the outage log either side of its resume."""
    note("thread/tokenUsage/updated", {"tokenUsage": {
        "total": {"inputTokens": total_in, "cachedInputTokens": cached,
                  "outputTokens": out},
        "last": {"totalTokens": occupied}, "modelContextWindow": 258400}})


def fail(message, info, reconnect=False):
    """Fail the turn the way the 2026-09-13 outage did: five reconnects with
    `willRetry` and the kind in its object spelling, then the terminal error
    with the kind degraded to `"other"`, then the failed `turn/completed`."""
    if reconnect:
        note("error", {"error": {
            "message": "Reconnecting... 5/5",
            "codexErrorInfo": {"responseStreamDisconnected":
                               {"httpStatusCode": None}},
            "additionalDetails": message, "misalignment": None},
            "willRetry": True, "threadId": "th-fake", "turnId": "t1"})
    error = {"message": message, "codexErrorInfo": info,
             "additionalDetails": None, "misalignment": None}
    note("error", {"error": error, "willRetry": False,
                   "threadId": "th-fake", "turnId": "t1"})
    note("turn/completed", {"threadId": "th-fake", "turn": {
        "id": "t1", "items": [], "itemsView": "notLoaded", "status": "failed",
        "error": error, "durationMs": 179014}})


def sub_agent_turn(main_answer, drop=False):
    """Codex splitting a review: the main thread spawns a child, the child
    finishes FIRST with its subtask's answer, and only then does the main
    thread answer and complete. Shapes (and the three ids' relations) from
    `.codex-agents/review-a/events.jsonl`, lines 48-255 (2026-10-24):
    every notification declares its `threadId`, a child's turn carries the
    main turn as `rootTurnId`, and each thread reports its own cumulative
    token usage. `drop`: the service drops the main thread instead of letting
    it answer, after the children are done."""
    main = {"threadId": "th-fake", "turnId": "t-main"}
    child = {"threadId": "th-child", "turnId": "t-child"}
    late = {"threadId": "th-late", "turnId": "t-late"}
    note("turn/started", {"threadId": "th-fake", "turn": {
        "id": "t-main", "rootTurnId": "t-main", "status": "inProgress"}})
    note("item/completed", {**main, "item": {
        "type": "subAgentActivity", "kind": "started",
        "agentThreadId": "th-child", "agentPath": "/root/client_review"}})
    note("turn/started", {"threadId": "th-child", "turn": {
        "id": "t-child", "rootTurnId": "t-main", "status": "inProgress"}})
    # A second child, whose message arrives only AFTER the main thread's
    # final answer: routed as a main-thread item it would replace that answer
    # (a child that speaks first is overwritten by the main answer either way).
    note("item/completed", {**main, "item": {
        "type": "subAgentActivity", "kind": "started",
        "agentThreadId": "th-late", "agentPath": "/root/late_check"}})
    note("turn/started", {"threadId": "th-late", "turn": {
        "id": "t-late", "rootTurnId": "t-main", "status": "inProgress"}})
    note("item/completed", {**child, "item": {
        "type": "commandExecution", "command": "git show a", "exitCode": 0}})
    note("item/completed", {**child, "item": {
        "type": "agentMessage", "phase": "final_answer",
        "text": "## Findings\nnone in the client portion"}})
    # Not in that log: a child's terminal error, in the shape of
    # LOGGED_DROP — the main thread then completes the turn regardless.
    note("error", {**child, "error": {"message": "child stream broke",
                                      "codexErrorInfo": "other"},
                   "willRetry": False})
    # The main thread's last reading comes BEFORE the child's (lines 248
    # and 251), so a child reading taken as occupancy would be the last.
    note("thread/tokenUsage/updated", {**main, "tokenUsage": {
        "total": {"inputTokens": 501566, "cachedInputTokens": 435584,
                  "outputTokens": 2163},
        "last": {"totalTokens": 79835}, "modelContextWindow": 258400}})
    note("thread/tokenUsage/updated", {**child, "tokenUsage": {
        "total": {"inputTokens": 433471, "cachedInputTokens": 382336,
                  "outputTokens": 1837},
        "last": {"totalTokens": 85661}, "modelContextWindow": 258400}})
    note("item/completed", {**main, "item": {
        "type": "subAgentActivity", "kind": "completed",
        "agentThreadId": "th-child", "agentPath": "/root/client_review"}})
    note("turn/completed", {"threadId": "th-child", "turn": {
        "id": "t-child", "rootTurnId": "t-main", "status": "completed"}})
    note("item/completed", {**main, "item": {
        "type": "commandExecution", "command": "git show b", "exitCode": 0}})
    if drop:
        note("item/completed", {**main, "item": {
            "type": "subAgentActivity", "kind": "completed",
            "agentThreadId": "th-late", "agentPath": "/root/late_check"}})
        note("turn/completed", {"threadId": "th-late", "turn": {
            "id": "t-late", "rootTurnId": "t-main", "status": "completed"}})
        fail(DROP_MESSAGE, "other", reconnect=True)
        return
    note("item/completed", {**main, "item": {
        "type": "agentMessage", "phase": "final_answer", "text": main_answer}})
    note("item/completed", {**late, "item": {
        "type": "agentMessage", "phase": "final_answer",
        "text": "## Findings\nthe late child's subtask"}})
    note("item/completed", {**main, "item": {
        "type": "subAgentActivity", "kind": "completed",
        "agentThreadId": "th-late", "agentPath": "/root/late_check"}})
    note("turn/completed", {"threadId": "th-late", "turn": {
        "id": "t-late", "rootTurnId": "t-main", "status": "completed"}})
    note("turn/completed", {"threadId": "th-fake", "turn": {
        "id": "t-main", "rootTurnId": "t-main", "status": "completed"}})


# The scenarios where a child outlives the main answer (`live_child_turn`).
LIVE_CHILD_SCENARIOS = (
    "sub-agent-live", "sub-agent-gap", "sub-agent-stale", "sub-agent-stuck",
    "sub-agent-continue", "sub-agent-busy-continue", "sub-agent-late-root",
    "sub-agent-fail", "sub-agent-mixed", "sub-agent-deaf", "sub-agent-mute",
    "subrep-live", "subrep-fail", "subrep-timeout", "subrep-silent",
    "sub-agent-revive",
    "sub-agent-idle-grand", "sub-agent-done-fail", "sub-agent-done-reject",
    "sub-agent-restart-timeout", "sub-agent-restart-spawn",
    "sub-agent-child-fail", "sub-agent-drain-done", "sub-agent-continue-fail",
    "sub-agent-long", "subrep-silent-gatefail")
# A child message past `subagents.ANSWER_CHARS`, with a mark at its very end.
LONG_ANSWER = "L" * 30000 + "TAIL-MARK"
# Children that never finish on their own: cut at the deadline.
STUCK_SCENARIOS = ("sub-agent-stuck", "sub-agent-deaf", "sub-agent-mute",
                   "sub-agent-revive", "sub-agent-restart-timeout",
                   "sub-agent-restart-spawn", "sub-agent-drain-done")
# What the last `turn/start` asked, for `wrapper_turn`.
TURN_TEXT = ""
# Names a turn the wrapper sends about its children may carry: results,
# paths and files of the scenarios below.
MARKERS = ("the kid's result", "the grand's result", "A's result",
           "/root/kid/grand", "/root/b", "kid.py", "grand.py", "a.py", "b.py",
           "restart it")


# The main thread's turns, in the children's shapes below.
def root_turn(turn_id):
    child_turn("th-fake", turn_id)


def root_says(text):
    child_says("th-fake", text)


def root_done(turn_id):
    child_done("th-fake", turn_id)


def spawn(parent, thread, path, turn_id=""):
    """The parent's `subAgentActivity started` record (on the PARENT's
    thread, as a grandchild's is on its parent's), then — unless `turn_id`
    is withheld — the child's own `turn/started` (no `rootTurnId`, as in
    reviewer rev-a's log)."""
    note("item/completed", {"threadId": parent, "item": {
        "type": "subAgentActivity", "kind": "started",
        "agentThreadId": thread, "agentPath": path}})
    if turn_id:
        child_turn(thread, turn_id)


def child_turn(thread, turn_id):
    note("turn/started", {"threadId": thread, "turn": {
        "id": turn_id, "status": "inProgress"}})


def child_edit(thread, path):
    note("item/completed", {"threadId": thread, "item": {
        "type": "fileChange", "changes": [
            {"path": path, "kind": {"type": "update"}}]}})


def child_says(thread, text):
    note("item/completed", {"threadId": thread, "item": {
        "type": "agentMessage", "phase": "final_answer", "text": text}})


def child_done(thread, turn_id, text="", status="completed", error=None,
               verdict_text=""):
    """`verdict_text`: a message carried only inside the verdict's
    `turn.items`, as the real server repeats the final one there."""
    if text:
        child_says(thread, text)
    turn = {"id": turn_id, "status": status}
    if error:
        turn["error"] = {"message": error}
    if verdict_text:
        turn["items"] = [{"type": "agentMessage", "text": verdict_text}]
    note("turn/completed", {"threadId": thread, "turn": turn})


def wrapper_turn(text=None):
    """A turn the wrapper sent about the children. The stand-in has NO
    inbox: its answer names only what that turn's PROMPT carried, so a
    result the prompt left out is one the main thread never saw — the case
    of a grandchild (delivered to its parent only) and of a child whose
    result Codex routes after its terminal event."""
    seen = [marker for marker in MARKERS if marker in TURN_TEXT]
    root_turn("t-wrap")
    root_says(text or ("## Findings\nsaw: " + (", ".join(seen) or "nothing")))
    root_done("t-wrap")


def live_child_turn():
    """The main thread answers while a child still runs — the shape of
    reviewer rev-a's log line 708. Turns after the first are the
    wrapper's (`wrapper_turn`), except where a scenario says otherwise."""
    report = os.environ.get("FAKE_REPORT_PATH", "")
    attempt = attempt_number()
    if attempt > 1:
        if SCENARIO == "subrep-silent-gatefail" and attempt > 2:
            # The gate follow-up, which fails.
            root_turn("t-gate")
            fail("You've hit your usage limit.", "usageLimitExceeded")
        elif SCENARIO == "sub-agent-mute":
            root_turn("t-wrap")
            root_done("t-wrap")
        elif SCENARIO == "sub-agent-revive":
            # The restart turn resumes the cut child on its OWN thread (Codex
            # v2 `followup_task`: a new turn there) and waits for it.
            root_turn("t-wrap")
            child_turn("th-kid", "t-kid2")
            child_edit("th-kid", "kid2.py")
            child_done("th-kid", "t-kid2", "the kid's second result")
            root_says("## Findings\nrestarted the kid")
            root_done("t-wrap")
        elif SCENARIO == "sub-agent-restart-spawn":
            # ... or starts a new child, and answers before it is done.
            root_turn("t-wrap")
            spawn("th-fake", "th-new", "/root/new", "t-new")
            child_edit("th-new", "new.py")
            root_says("## Findings\nstarted a new one")
            root_done("t-wrap")
        elif SCENARIO in ("sub-agent-restart-timeout", "sub-agent-done-reject"):
            root_turn("t-wrap")  # and never completes (rejected: never runs)
        elif SCENARIO == "sub-agent-done-fail":
            root_turn("t-wrap")
            fail("You've hit your usage limit.", "usageLimitExceeded")
        elif SCENARIO in ("subrep-silent", "subrep-silent-gatefail"):
            # Completes, answering in the message only: no new report.
            wrapper_turn("merged the kid's result in the message only")
        elif SCENARIO == "subrep-timeout":
            # Keeps rewriting the report until the wrapper stops the server.
            root_turn("t-wrap")
            for _ in range(200):
                with open(report, "w", encoding="utf-8", newline="") as handle:
                    handle.write("## Report\n### Done\npartial")
                time.sleep(0.05)
        elif SCENARIO.startswith("subrep-"):
            with open(os.path.join(WORK, "report_at_wrapper.txt"), "w",
                      encoding="utf-8") as handle:
                handle.write(str(os.path.exists(report)))
            with open(report, "w", encoding="utf-8", newline="") as handle:
                handle.write("## Report\n### Done\npartial" if
                             SCENARIO == "subrep-fail" else
                             "## Report\n### Done\nmerged the kid's result\n"
                             "### Not done / open questions\nnone\n")
            if SCENARIO == "subrep-fail":
                root_turn("t-wrap")
                fail("You've hit your usage limit.", "usageLimitExceeded")
            else:
                wrapper_turn("## Report\n### Not done / open questions\nnone")
        else:
            wrapper_turn()
        return
    if SCENARIO.startswith("subrep-"):
        with open(report, "w", encoding="utf-8", newline="") as handle:
            handle.write("## Report\n### Done\nfirst\n"
                         "### Not done / open questions\nnone\n")
    root_turn("t-main")
    if SCENARIO == "sub-agent-gap":
        # The record alone: the main turn completes before the child's
        # `turn/started` (rev-a: lines 52 and 57, 36 ms apart).
        spawn("th-fake", "th-kid", "/root/kid")
    elif SCENARIO == "sub-agent-mixed":
        spawn("th-fake", "th-a", "/root/a", "t-a")
        child_edit("th-a", "a.py")
        spawn("th-fake", "th-b", "/root/b", "t-b")
        child_edit("th-b", "b.py")
    elif SCENARIO == "sub-agent-stale":
        spawn("th-fake", "th-kid", "/root/kid", "t-kid1")
        child_turn("th-kid", "t-kid2")
        child_says("th-kid", "kid2 progress")
    else:
        spawn("th-fake", "th-kid", "/root/kid", "t-kid")
        child_edit("th-kid", "kid.py")
    if SCENARIO == "sub-agent-stuck":
        spawn("th-kid", "th-grand", "/root/kid/grand", "t-grand")
        child_edit("th-grand", "grand.py")
    if SCENARIO == "sub-agent-late-root":
        return  # the main turn never completes; see `on_interrupt`
    if SCENARIO == "sub-agent-fail":
        fail("You've hit your usage limit.", "usageLimitExceeded")
        return
    if SCENARIO.startswith("subrep-"):
        root_says("## Report\n### Not done / open questions\nnone")
    elif SCENARIO != "sub-agent-mute":
        root_says("## Findings\nwithout the kid")
    root_done("t-main")
    if SCENARIO in STUCK_SCENARIOS:
        return
    time.sleep(0.2)
    if SCENARIO == "sub-agent-idle-grand":
        # The child spawns a grandchild while the main thread is idle, then
        # finishes; the grandchild finishes later still.
        spawn("th-kid", "th-grand", "/root/kid/grand", "t-grand")
        child_done("th-kid", "t-kid", "the kid's result")
        time.sleep(0.3)
        child_done("th-grand", "t-grand", "the grand's result")
    elif SCENARIO == "sub-agent-continue-fail":
        # A continuation the service drops, with the child still running.
        root_turn("t-main2")
        fail(DROP_MESSAGE, "other", reconnect=True)
    elif SCENARIO == "sub-agent-child-fail":
        child_done("th-kid", "t-kid", "half-done notes", status="failed",
                   error="child broke")
    elif SCENARIO == "sub-agent-gap":
        child_turn("th-kid", "t-kid")
        child_edit("th-kid", "kid.py")
        child_done("th-kid", "t-kid", "the kid's result")
    elif SCENARIO == "sub-agent-stale":
        # The OLDER turn's late completion, with words of its own.
        child_done("th-kid", "t-kid1", verdict_text="stale text")
    elif SCENARIO == "sub-agent-mixed":
        child_done("th-a", "t-a", "A's result")
    elif SCENARIO in ("sub-agent-continue", "sub-agent-long"):
        root_turn("t-main2")
        child_done("th-kid", "t-kid", LONG_ANSWER if SCENARIO == "sub-agent-long"
                   else "the kid's result")
        root_says("## Findings\ncontinued")
        root_done("t-main2")
    elif SCENARIO == "sub-agent-busy-continue":
        # A continuation that streams for 1.5 s with the child still open.
        root_turn("t-main2")
        for _ in range(60):
            note("thread/tokenUsage/updated", {"threadId": "th-fake", "tokenUsage": {
                "total": {"inputTokens": 10, "cachedInputTokens": 0,
                          "outputTokens": 1}}})
            time.sleep(0.025)
        root_says("## Findings\ncontinued")
        root_done("t-main2")
    else:
        child_done("th-kid", "t-kid", "the kid's result")


def on_interrupt(request_id, params):
    """As the app-server does (bespoke_event_handling.rs): the reply as the
    turn aborts, then its `turn/completed`. "-deaf" never stops; "-late-root"
    lets the never-finished main turn fail the way a service drop does first."""
    out({"jsonrpc": "2.0", "id": request_id, "result": {}})
    if SCENARIO == "sub-agent-deaf":
        return
    if SCENARIO == "sub-agent-late-root":
        fail(DROP_MESSAGE, "other", reconnect=True)
    if SCENARIO == "sub-agent-drain-done":
        # It had just finished on its own: the verdict says completed.
        child_done(params.get("threadId"), params.get("turnId"),
                   "finished just in time")
        return
    note("turn/completed", {"threadId": params.get("threadId"), "turn": {
        "id": params.get("turnId"), "status": "interrupted"}})


def run_turn():
    if SCENARIO in ("loop-turn", "id-clash"):
        # The loop's provider stream reads only notifications that name its
        # turn (`wire.codex_app_in_turn`), as the real server's all do.
        scoped = {"threadId": "th-fake", "turnId": "t1"}
        note("item/completed", {**scoped, "item": {
            "id": "m1", "type": "agentMessage", "text": "fake answer"}})
        note("turn/completed", {"threadId": "th-fake", "turn": {
            "id": "t1", "status": "completed"}})
        return
    if SCENARIO in LIVE_CHILD_SCENARIOS:
        live_child_turn()
        return
    if SCENARIO.startswith("report-"):
        attempt = attempt_number()
        report = os.environ["FAKE_REPORT_PATH"]
        message = "## Report\n### Not done / open questions\nnone"
        text = "## Report\n### Done\nEdited café.\n### Not done / open questions\nnone\n"
        write = True
        if SCENARIO == "report-empty":
            text = " \n"
        elif SCENARIO == "report-directory":
            os.makedirs(report, exist_ok=True)
            write = False
        elif SCENARIO == "report-same":
            text = message
        elif SCENARIO == "report-newline":
            text = message + "\n"
        elif SCENARIO == "report-absent":
            write = False
        elif SCENARIO == "report-stale":
            write = attempt == 1
        elif SCENARIO in ("report-partial", "report-fallback", "report-failed",
                          "report-silent", "report-file-only", "report-repeat",
                          "report-fallback-then-fail") and attempt == 1:
            text = "## Report\n### Done\nEdited café."
        elif SCENARIO == "report-repeat":
            write = False
            message = "## Report\n### Done\nEdited café."
        elif SCENARIO in ("report-silent", "report-file-only"):
            message = ""
            write = SCENARIO == "report-file-only"
            text = "## Report\n### Done\nEdited café."
        elif SCENARIO in ("report-fallback", "report-failed",
                          "report-fallback-then-fail"):
            write = False
        elif SCENARIO == "report-gate-regress" and attempt == 2:
            # The gate follow-up replaces a complete report with one word.
            text = "fixed"
        elif SCENARIO == "report-missing" and attempt == 1:
            write = False
            message = "Finished editing."
        elif SCENARIO == "report-retry" and attempt > 1:
            write = False
        if write:
            encoding = {"report-utf8-bom": "utf-8-sig",
                        "report-utf16": "utf-16",
                        "report-cp1252": "cp1252"}.get(SCENARIO, "utf-8")
            with open(report, "w", encoding=encoding, newline="") as handle:
                handle.write(text)
        if SCENARIO == "report-retry" and attempt == 1:
            fail(DROP_MESSAGE, "other", reconnect=True)
            return
        if SCENARIO == "report-failed" and attempt > 1:
            fail("You've hit your usage limit.", "usageLimitExceeded")
            return
        item({"type": "agentMessage", "text": message})
        # "-then-fail": the third (gate follow-up) turn says something and
        # then fails, so its partial message must not take the kept answer's
        # provenance with it.
        if SCENARIO == "report-terminal-failure" or (
                SCENARIO == "report-fallback-then-fail" and attempt > 2):
            fail("You've hit your usage limit.", "usageLimitExceeded")
            return
        note("turn/completed", {"turn": {"status": "completed"}})
        return
    if SCENARIO in ("sub-agent-follow-up", "sub-agent-retry"):
        # The split review's main answer is short of its Coverage, or the
        # service drops the main thread after the child is done; the next
        # attempt on the same thread runs no child, and the thread's own
        # cumulative total grows past the first attempt's 501 566.
        if attempt_number() == 1:
            sub_agent_turn("## Findings\n1. server half",
                           drop=SCENARIO == "sub-agent-retry")
            return
        note("thread/tokenUsage/updated", {
            "threadId": "th-fake", "turnId": "t-main2", "tokenUsage": {
                "total": {"inputTokens": 560000, "cachedInputTokens": 480000,
                          "outputTokens": 2500},
                "last": {"totalTokens": 81000}, "modelContextWindow": 258400}})
        note("item/completed", {"threadId": "th-fake", "item": {
            "type": "agentMessage", "text": "## Coverage\nall of them"}})
        note("turn/completed", {"threadId": "th-fake", "turn": {
            "id": "t-main2", "rootTurnId": "t-main2", "status": "completed"}})
        return
    if SCENARIO == "sub-agent":
        sub_agent_turn("## Findings\n1. server half\n## Coverage\nall of them")
        return
    if SCENARIO == "die":
        sys.exit(1)  # mid-turn death: the client must see EOF, not a stall
    if SCENARIO.startswith("network-"):
        attempt = attempt_number()
        if SCENARIO in ("network-child-stale", "network-child-heartbeats"):
            if attempt == 1:
                origin = NETWORK_TIMES[0]
                out({"method": "item/started", "params": {"threadId": "th-fake"},
                     "emittedAtMs": origin})
                NETWORK_FIXTURE["params"]["threadId"] = "th-child"
                for elapsed in range(10, 8000, 60):
                    if elapsed == 10 or SCENARIO == "network-child-heartbeats":
                        out({**NETWORK_FIXTURE, "emittedAtMs": origin + elapsed * 1000})
                    out({"method": "item/started", "params": {"threadId": "th-fake"},
                         "emittedAtMs": origin + elapsed * 1000})
            root_says("main kept working")
            root_done("t1")
            return
        if SCENARIO == "network-rev-d":
            # rev-d emittedAtMs: 1792073187371, 1792073370156,
            # then first progress at 1792073447873 (2026-10-15).
            origin = NETWORK_TIMES[0]
            out({"method": "thread/tokenUsage/updated", "params": {},
                 "emittedAtMs": origin})
            for elapsed in (187371, 370156):
                NETWORK_FIXTURE["params"]["threadId"] = "th-fake"
                out({**NETWORK_FIXTURE, "emittedAtMs": origin + elapsed})
            out({"method": "item/started", "params": {"threadId": "th-fake"},
                 "emittedAtMs": origin + 447873})
            root_says("recovered without a resend")
            root_done("t1")
            return
        if SCENARIO == "network-complete-at-ceiling":
            NETWORK_FIXTURE["params"]["threadId"] = "th-fake"
            out(NETWORK_FIXTURE)
            # Final message and completion are delivered at the ceiling.
            out({"method": "item/completed", "emittedAtMs": NETWORK_TIMES[0] + 100000,
                 "params": {"threadId": "th-fake", "item": {
                     "type": "agentMessage", "text": "complete answer"}}})
            root_done("t1")
            return
        if attempt == 1 or SCENARIO == "network-ceiling-across-retries":
            event = NETWORK_FIXTURE
            event["params"]["threadId"] = (
                "th-child" if SCENARIO == "network-child-only" else "th-fake")
            if SCENARIO == "network-429":
                event["params"]["error"]["codexErrorInfo"] = "rateLimitExceeded"
            elif SCENARIO == "network-stream-drop":
                event["params"]["error"]["message"] = "Reconnecting... 5/5"
            for stamp in NETWORK_TIMES:
                event["emittedAtMs"] = stamp
                out(event)
                if SCENARIO == "network-child-only":
                    # Main-thread progress does not recover the child's stream.
                    note("item/started", {"threadId": "th-fake"})
                if SCENARIO in ("network-stale", "network-ceiling-across-retries"):
                    # Staleness alone leaves work time; only the second tick
                    # exhausts the 100 s work budget plus credited silence.
                    out({"method": "thread/tokenUsage/updated", "params": {},
                         "emittedAtMs": stamp + 367000})
                    out({"method": "thread/tokenUsage/updated", "params": {},
                         "emittedAtMs": stamp + 467000})
                    return
            if SCENARIO == "network-child-only":
                note("turn/completed", {"threadId": "th-child",
                                        "turn": {"status": "completed"}})
        item({"type": "agentMessage", "text": "network recovered"})
        note("turn/completed", {"turn": {"status": "completed"}})
        return
    if SCENARIO in ("drop-once", "drop-always", "refuse", "drop-then-refuse"):
        attempt = attempt_number()
        if SCENARIO == "refuse" or (SCENARIO == "drop-then-refuse" and attempt > 1):
            fail("You've hit your usage limit.", "usageLimitExceeded")
            return
        if SCENARIO == "drop-always" or attempt == 1:
            # Work done before the stream died: the summary must keep it.
            item({"type": "commandExecution", "command": "git diff --stat",
                  "exitCode": 0})
            item({"type": "fileChange", "changes": [
                {"path": "half.py", "kind": {"type": "update"}}]})
            usage(400578, 354560, 2869, 60754)
            fail(DROP_MESSAGE, "other", reconnect=True)
            return
        item({"type": "agentMessage", "text": "picked up after the drop"})
        usage(467645, 415232, 3075, 67273)
        note("turn/completed", {"turn": {"status": "completed"}})
        return
    if SCENARIO in ("partial", "partial-twice", "partial-twice-then-refuse",
                    "partial-then-silent"):
        # A reviewer that answers one of the questions and stops; asked again
        # on the same thread it adds the rest without repeating itself — or,
        # "partial-twice", adds more findings and still no Coverage; the
        # "-then-refuse" form then fails the third (gate follow-up) turn;
        # "partial-then-silent" completes the follow-up with no message.
        attempt = attempt_number()
        if SCENARIO == "partial-then-silent" and attempt > 1:
            note("turn/completed", {"turn": {"status": "completed"}})
            return
        if SCENARIO == "partial-twice-then-refuse" and attempt > 2:
            fail("You've hit your usage limit.", "usageLimitExceeded")
            return
        if attempt == 1:
            item({"type": "commandExecution", "command": "git show a",
                  "exitCode": 0})
            item({"type": "agentMessage", "text": "## Findings\n1. one thing"})
        else:
            item({"type": "commandExecution", "command": "git show b",
                  "exitCode": 0})
            item({"type": "agentMessage", "text": "## Coverage\nall of them"
                  if SCENARIO == "partial" else "2. another thing"})
        note("turn/completed", {"turn": {"status": "completed"}})
        return
    if SCENARIO == "approvals":
        ask(9001, "item/permissions/requestApproval", {"itemId": "i1"})
        ask(9002, "item/commandExecution/requestApproval", {"itemId": "i2"})
        ask(9003, "execCommandApproval", {"callId": "c1"})
        ask(9004, "totally/unknown/request", {})
        ask(9005, "item/fileChange/requestApproval", {"itemId": "i9"})
        item({"type": "agentMessage", "text": "approvals handled"})
    elif SCENARIO == "file-approvals":
        # An edit that failed inside the sandbox, asked to be retried outside
        # it: the request names the item, its `item/started` names the files.
        def change(item_id, *paths, move=None):
            note("item/started", {"item": {
                "type": "fileChange", "id": item_id, "changes": [
                    {"path": path, "kind": {"type": "update", "move_path": move}}
                    for path in paths]}})

        def retry(request_id, item_id):
            ask(request_id, "item/fileChange/requestApproval",
                {"itemId": item_id, "threadId": "th-fake",
                 "reason": "command failed; retry without sandbox?"})

        inside = os.path.join(WORK, "src", "a.py")
        outside = os.path.join(os.path.dirname(WORK), "elsewhere.py")
        change("f1", inside)
        retry(9201, "f1")
        change("f2", inside, outside)
        retry(9202, "f2")
        item({"type": "fileChange", "id": "f2", "status": "declined",
              "changes": [{"path": outside, "kind": {"type": "update"}}]})
        change("f3", os.path.join(WORK, ".git", "config"))
        retry(9203, "f3")
        # Two files under the host's protected paths, named by the test
        # (`FAKE_PROTECTED_EDITS`, a JSON list of two relative paths).
        protected = json.loads(os.environ.get(
            "FAKE_PROTECTED_EDITS", '["vault/note.md", "vault/0001.md"]'))
        change("f4", os.path.join(WORK, *protected[0].split("/")))
        retry(9204, "f4")
        change("f5", os.path.join(WORK, *protected[1].split("/")))
        retry(9205, "f5")
        retry(9206, "never-started")
        change("f7", inside, move=outside)
        retry(9207, "f7")
        change("f8", os.path.join(os.environ["FAKE_SCRATCH"], "m.txt"))
        retry(9208, "f8")
        # A session-wide root, on a seen item and on one never started.
        change("f9", inside)
        ask(9209, "item/fileChange/requestApproval",
            {"itemId": "f9", "threadId": "th-fake", "grantRoot": WORK})
        ask(9210, "item/fileChange/requestApproval",
            {"itemId": "ghost", "threadId": "th-fake", "grantRoot": WORK})
        item({"type": "agentMessage", "text": "file approvals handled"})
    elif SCENARIO == "escalate":
        ask(9101, "item/commandExecution/requestApproval",
            {"itemId": "i3", "command": "npm test", "cwd": WORK})
        ask(9102, "item/commandExecution/requestApproval",
            {"itemId": "i4", "command": "rm -rf /", "cwd": WORK})
        ask(9103, "item/commandExecution/requestApproval",
            {"itemId": "i5", "command": "npm test",
             "cwd": os.path.dirname(WORK)})
        ask(9104, "item/commandExecution/requestApproval",
            {"itemId": "i6", "command": "npm test", "cwd": WORK,
             "kind": "writeStdin"})
        ask(9105, "item/commandExecution/requestApproval",
            {"itemId": "i7", "cwd": WORK, "command":
             '"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\'
             'powershell.exe" -Command \'npm run typecheck\''})
        ask(9106, "item/commandExecution/requestApproval",
            {"itemId": "i8", "cwd": WORK, "command":
             '"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\'
             'powershell.exe" -Command \'node -e "console.log(1)"\''})
        item({"type": "agentMessage", "text": "escalation handled"})
    elif SCENARIO == "commit":
        # A worker in a pool slot commits its own work through commit.py.
        slot = os.environ["FAKE_SLOT"]
        approval_store = json.loads(os.environ["CODEX_COMMIT_APPROVAL_STORE"])
        assert approval_store["workspace"] == slot
        assert os.path.isdir(approval_store["directory"])
        message = os.path.join(os.environ["FAKE_SCRATCH"], "msg.txt")
        elsewhere = os.path.join(os.path.dirname(WORK), "elsewhere", "msg.txt")
        wrap = ('"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\'
                'powershell.exe" -Command \'{}\'')
        for request_id, words in (
                (9301, f"python tools/git/commit.py -F {message} src/a.ts"),
                (9302, f"python tools/git/commit.py -F {message} ."),
                (9303, f"python tools/git/commit.py -F {elsewhere} src/a.ts"),
                (9304, f"python tools/git/commit.py --amend -F {message} src/a.ts")):
            ask(request_id, "item/commandExecution/requestApproval",
                {"itemId": f"c{request_id}", "cwd": slot,
                 "command": wrap.format(words)})
        item({"type": "agentMessage", "text": "commit handled"})
    elif SCENARIO == "rate-limits":
        # The account's weekly figure, the shape of `.codex-agents/fix-a`
        # line 31: repeated unchanged, then changed, then a report with only
        # a five-hour window (no weekly figure at all).
        for window, used in ((10080, 82.5), (10080, 82.5), (10080, 90), (300, 5)):
            note("account/rateLimits/updated", {"rateLimits": {
                "limitId": "codex", "primary": {
                    "usedPercent": used, "windowDurationMins": window,
                    "resetsAt": 1794568554}, "secondary": None}})
        item({"type": "agentMessage", "text": "fake answer"})
    elif SCENARIO == "silent":
        pass
    elif SCENARIO == "callback":
        seen = wait_for(os.path.join(WORK, "thread_seen"))
        item({"type": "agentMessage",
              "text": "callback:" + ("yes" if seen else "no")})
    else:
        item({"type": "commandExecution", "command": "git status",
              "exitCode": 0})
        item({"type": "fileChange", "changes": [
            {"path": "a.py", "kind": {"type": "update"}},
            {"path": "b.py", "kind": {"type": "add"}}]})
        item({"type": "agentMessage", "text": "fake answer"})
        note("thread/tokenUsage/updated", {"tokenUsage": {"total": {
            "inputTokens": 120, "cachedInputTokens": 20, "outputTokens": 8}}})
    note("turn/completed", {"turn": {"status": "completed"}})


def main():
    global TURN_TEXT
    while True:
        msg = read()
        if msg is None:
            return
        method = msg.get("method")
        request_id = msg.get("id")
        if method == "initialize":
            if SCENARIO == "mute-initialize":
                continue  # a CLI that starts and never answers
            out({"jsonrpc": "2.0", "id": request_id,
                 "result": {"userAgent": "fake-app-server"}})
        elif method in ("thread/start", "thread/resume"):
            if SCENARIO == "id-clash":
                # A request of the server's own, numbered by ITS counter,
                # that happens to carry the id the client awaits; asked
                # without waiting, as a client may leave it unanswered.
                out({"jsonrpc": "2.0", "id": request_id,
                     "method": "item/tool/requestUserInput", "params": {}})
            note("thread/started", {"thread": {"id": "th-fake"}})
            out({"jsonrpc": "2.0", "id": request_id,
                 "result": {"thread": {"id": "th-fake"}}})
        elif method == "thread/name/set":
            out({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif method == "turn/start":
            # Whether each extra writable root exists AT THIS MOMENT — the
            # agent starts writing right after this reply, so a root created
            # later is one it could not use.
            if SCENARIO == "sub-agent-done-reject" and os.path.exists(
                    os.path.join(WORK, "attempts.txt")):
                # The turn about the children is refused outright (as an
                # input over Codex's size limit is).
                with open(os.path.join(WORK, "rejected.txt"), "w") as handle:
                    handle.write("1")
                out({"jsonrpc": "2.0", "id": request_id, "error": {
                    "code": -32600, "message": "input too large"}})
                continue
            policy = (msg.get("params") or {}).get("sandboxPolicy") or {}
            roots = policy.get("writableRoots") or []
            with open(os.path.join(WORK, "roots_at_turn_start.json"), "w",
                      encoding="utf-8") as handle:
                json.dump({root: os.path.isdir(root) for root in roots}, handle)
            out({"jsonrpc": "2.0", "id": request_id,
                 "result": {"turn": {"id": "t1", "status": "inProgress"}}})
            TURN_TEXT =" ".join(str(part.get("text") or "") for part in
                                 (msg.get("params") or {}).get("input") or [])
            run_turn()
        elif method == "turn/interrupt":
            on_interrupt(request_id, msg.get("params") or {})
        elif request_id is not None:
            out({"jsonrpc": "2.0", "id": request_id, "result": {}})


main()
