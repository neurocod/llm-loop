"""The reviewer answer contract both agent wrappers brief and enforce.

`llm_agents.codex` and `llm_agents.claude` review the same commits side by
side, and the orchestrator merges both finding lists before one fixer. A
review on 2026-09-13 found the two wrappers asking for different skeletons:
Claude was never asked for missed siblings or a verdict, so the merge was done
by hand. One text here keeps the two answers mergeable; each
wrapper adds only what its own transport and permissions need.
"""

import re
from typing import Optional

# Measured 2026-09-13 on rev_panerun: five questions were answered in progress
# notes, only the last reached the final message, and a Coverage follow-up
# claimed all five answered. Require both answers and Coverage in the final
# message, the only part the orchestrator reads, so progress cannot stand in.
REVIEWER_BRIEFING = """Only your final message is read — progress notes are discarded. Answer every numbered question of the task in the final message; a question with nothing wrong gets the line `N: nothing found`.
Review the commit(s) named in the task, not the working tree: someone else may be editing it while you work.
Take the gate results the task reports as given, and run a gate yourself only when the task asks you to.
Your final answer must use these headings in order; say "none" under a heading rather than omitting it.
## Findings
Number findings; each needs file:line, what is wrong, why it matters, a suggested fix, and severity blocker / should / nit.
## Missed siblings
## Verdict
Choose ship / ship with fixes / do not ship, with one sentence saying why.
## Coverage
One line per numbered question of the task: answered / nothing found."""


# A numbered question of a review task: a line starting with `N.` — the shape
# every reviewer prompt of the 2026-09-13 waves used.
_NUMBERED_QUESTION = re.compile(r"^(\d+)\.\s", re.MULTILINE)


def coverage_expect(prompt: str) -> str:
    """The reviewer's default `--expect`: `## Coverage` naming EVERY question.

    A heading alone is not enough. Measured 2026-09-13 on `rev_lotrules`: the
    final message answered questions 4 and 5, its `## Coverage` honestly
    listed "4: answered / 5: answered", the heading matched and no follow-up
    fired — the contract had made the omission visible without closing it.
    So when the prompt carries numbered questions the expectation is the
    heading followed by a line for each number in order (`(?ms)` — `^` at
    line starts, `.` across lines); a prompt without them keeps the bare
    heading, which is what a `send` follow-up prompt has.
    """
    numbers = sorted({int(n) for n in _NUMBERED_QUESTION.findall(prompt)})
    if not numbers:
        return "^## Coverage"
    return "(?ms)^## Coverage" + "".join(f".*^{n}:" for n in numbers)


# What the follow-up turn says when the answer lacked what `--expect` named.
# One message on the same thread, like the operator's "cover the remaining N"
# that three reviewers needed on 2026-09-13 and a fourth on the day after
# (a 15-commit breakdown answered with one of its five questions): the thread
# has the task and its own first answer, so the follow-up names only the gap
# and forbids repetition — the two answers are joined into one.
#
# It says "finish the work" before "complete the answer" because a WORKER's
# missing section usually means unfinished work, not an unfinished write-up:
# measured 2026-09-13 on `topology`, whose turn ended after two edits with the
# gates unrun, and which answered the first wording of this prompt with a
# report saying "NOT RUN; interrupted" — twice, once per turn — rather than
# running them.
FINISH_TASK_CLAUSE = (
    "If the task itself is unfinished — gates not run, edits pending, a "
    "question unanswered — finish it first; then "
)

EXPECT_FOLLOWUP_PROMPT = (
    "Your answer is incomplete: nothing in it matches /{pattern}/, which the "
    f"task asked for. {FINISH_TASK_CLAUSE}give the complete "
    "answer in ONE message, adding what is missing without repeating what you "
    "already wrote.\n"
)

# One follow-up, not a loop: measured need was one message every time, and an
# answer that lacks the section after being told so is one the caller has to
# read, not one to keep prodding at the price of a re-sent thread each turn.
MAX_EXPECT_FOLLOWUPS = 1


# The helpers below are called by BOTH wrappers; the texts among them
# (`followed_up_note`, `incomplete_note`, `follow_up_failed`) have one wording
# each, because the orchestrator merging a Codex and a Claude review of one
# commit searches both outputs for the same substring (the two once had
# drifted into "1 follow-up (the first answer lacked …)" and "Followed up 1x:
# the first answer lacked …", and the failure text had grown `(logs: …)` on
# one side only).


def followed_up_note(expect: str, follow_ups: int) -> str:
    """Why a turn spent extra provider calls: its first answer lacked `expect`.

    Says nothing about the final answer — `incomplete_note` does, when it
    still lacks the section.
    """
    return f"{follow_ups} follow-up(s): the first answer lacked /{expect}/"


def incomplete_note(expect: str, follow_ups: int) -> str:
    """The line both wrappers print when a completed answer still misses
    `expect` after its follow-ups; the status stays completed, so the answer
    is kept.
    """
    return f"Incomplete: the answer still lacks /{expect}/ after {follow_ups} follow-up(s)"


def expect_follow_up(expect: str) -> str:
    """The name of the follow-up `expect` triggers, as `follow_up_failed` says it."""
    return f"follow-up for /{expect}/"


def follow_up_failed(what: str, reason: str, logs: object = None) -> str:
    """The error of a turn whose follow-up `what` did not complete.

    `what` is `expect_follow_up(expect)` or another follow-up's name (Codex's
    gate follow-up); `logs` names the follow-up's own log directory where the
    wrapper keeps one apart from the turn's. The first answer survives, and
    the text says so: the status alone reads as a turn that produced nothing.
    """
    where = f" (logs: {logs})" if logs is not None else ""
    return f"{what} failed: {reason}{where}; the first answer is kept"


def join_answers(first: str, followed: str) -> str:
    """One answer from the first attempt and its follow-up, oldest first.

    Joined, not replaced, because the follow-up is told not to repeat what it
    already wrote; an empty follow-up or one that repeated the first answer
    verbatim adds nothing and leaves the first as it was.
    """
    if not followed or followed == first:
        return first
    return first.rstrip() + "\n\n" + followed.lstrip()


def checked_expect(expect: str) -> str:
    """`expect` itself once it compiles; ValueError naming the flag otherwise.

    Called at the CLI boundary, before anything is spawned or queued: an
    invalid regex otherwise surfaces as `re.error` from `answers` only after
    the first turn has been paid for.
    """
    try:
        re.compile(expect)
    except re.error as exc:
        raise ValueError(f"Invalid --expect regex {expect!r}: {exc}") from None
    return expect


def answers(expect: str, text: str) -> bool:
    """Whether `text` satisfies `expect`; an empty expectation always does.

    Both wrappers test with MULTILINE so a bare `^## Heading` matches a line
    anywhere in the answer, not only its first character.
    """
    return not expect or re.search(expect, text, re.MULTILINE) is not None


# The `--expect` follow-up rule, one copy for both wrappers' loops (two
# copies had drifted twice — `expect_met` judged by one wrapper only,
# and `MAX_EXPECT_FOLLOWUPS` read as "always N" by one and "up to N" by the
# other). The loops keep only their transport and their data model.
#
# Only a COMPLETED answer — finished without an error — is judged: a failed or
# timed-out turn is short of a result, not of a section, and prodding it would
# spend a resend on it.


def needs_expect_follow_up(expect: str, text: str, sent: int, *,
                           completed: bool) -> bool:
    """Whether the answer `text` gets another `--expect` follow-up, `sent`
    having been sent so far by this loop: a completed turn whose answer lacks
    a non-empty `expect`, up to MAX_EXPECT_FOLLOWUPS.

    `sent` is the calling loop's own count. Codex's gate path runs the loop a
    second time over the replacement answer, with a count of its own.
    """
    return (bool(expect) and completed and sent < MAX_EXPECT_FOLLOWUPS
            and not answers(expect, text))


def expect_verdict(expect: str, text: str, *, completed: bool) -> Optional[bool]:
    """`expect_met` for the last answer, re-judged after the follow-ups: None
    when there is nothing to judge (no `expect`, or a turn that did not
    complete), so the wrapper records no verdict. Without it two answers both
    lacking the section read as completed with "the first answer lacked",
    i.e. as a gap closed."""
    if not expect or not completed:
        return None
    return answers(expect, text)
