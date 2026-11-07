"""Task-independent role briefings and their answer contracts.

Measured in the wave of 2026-09-13: seven prompts repeated about fifteen lines
each, and the copies drifted; one worker was told to use git mv in a slot
that cannot. Centralize those rules here; permissions remain in sandbox.py.
The worker's report path is told once, immediately before the report skeleton.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..answer_contract import REVIEWER_BRIEFING, checked_expect, coverage_expect
from ..host_policy import active as host_policy


# Keep the report skeleton with the worker rules: the seven copied prompts in
# the 2026-09-13 wave drifted, so callers should supply tasks, not this contract.
WORKER_REPORT = """Use these headings, in this order, and nothing after them; under Gates give each command and its result line.
## Report
### Done
### Files changed
### Gates
### Missed siblings
### Not done / open questions"""


def _worker_briefing(worktree_briefed: bool, scratch: Optional[Path],
                     report: Optional[Path] = None) -> str:
    """Add worker rules once, with pytest files outside the project tree.

    Tell the report path once, immediately before the skeleton.

    Measured 2026-09-13: a worker ran pytest under its scratch root, then
    cleanup_targets refused deletion because it only grants workspace
    deletions; the worker reported "blocked by policy". Scratch basetemp needs
    no deletion. Full access has no scratch root, so use a directory under
    %TEMP%. Only an actually supplied worktree fact suppresses commit/move
    rules; a worker was told git mv in a slot that cannot during that wave.
    """
    lines = []
    if not worktree_briefed:
        lines.append("You cannot commit; the orchestrator commits from your report.")
        lines.append("Move files with plain filesystem commands and list every move in your report.")
    lines.append("Edit the project tree in place; other people's changes in `git status` are not yours — leave them.")
    lines.append("Do not launch Claude or other agents for review; the orchestrator runs the cross-model review. Your own sub-agents are fine.")
    if scratch is not None:
        lines.append(f"pytest: pass `--basetemp {scratch / 'pytest-tmp'}` "
                     "(a directory under your scratch directory), so nothing needs deleting afterwards.")
    else:
        lines.append("pytest: pass `--basetemp` under `%TEMP%`, so nothing needs deleting afterwards.")
    # The host's own rules for a worker (`HostPolicy.worker_lines`).
    lines.extend(host_policy().worker_lines)
    # Measured 2026-09-13 on `topology`, `lotrules`, `sections`: a worker
    # whose task opened with "compare / prove by reading" delivered the
    # reading as its final message — edits pending or absent, gates unrun —
    # and one called itself a "read-only audit". Each was sent back with a
    # `send`; the rule is stated up front so the first turn is the whole turn.
    lines.append("You are the one who EDITS: a turn that ends with the task's edits not made "
                 "or its gates not run is sent back — reading and comparing are the first "
                 "step of the task, not a deliverable. Do the edits, run every gate the "
                 "task lists, then report.")
    lines.append('Say "none" under a heading rather than omitting it.')
    if report is not None:
        lines.append(f"Write the complete report to {report} as UTF-8 on every turn; "
                     "the wrapper reads that file as your answer. Your final message "
                     "is only a fallback when the file is missing or empty.")
    else:
        lines.append("There is no report file for this mode (full access has no "
                     "scratch root); put the complete report in your final message.")
    lines.append(WORKER_REPORT)
    return "\n".join(lines)


def _reviewer_briefing(worktree_briefed: bool, scratch: Optional[Path],
                       report: Optional[Path] = None) -> str:
    """Use one final-message contract regardless of the reviewer's sandbox.

    The rev_panerun loss of four answers on 2026-09-13 was about where answers
    were written, not permissions. Accept the same environment inputs as the
    worker factory so the role table dispatches without a parallel if-chain.
    The text is shared with `claude_agent --profile review`
    (`tools._shared.answer_contract`), so both reviews of one commit merge.
    """
    return REVIEWER_BRIEFING


@dataclass(frozen=True)
class Role:
    """Pair dispatch and expectation so CLI choices cannot outgrow dispatch.

    The 2026-09-13 review found separate choices and an if-chain that could
    reject a role only after reserving its name and archiving the old log.
    Every table entry now supplies both pieces by construction.
    """

    briefing: Callable[[bool, Optional[Path], Optional[Path]], str]
    # The default `--expect` for a turn whose prompt is the argument: a role
    # may read the prompt to decide what a complete answer looks like.
    expect: Callable[[str], str]


# One table supplies argparse choices, briefing dispatch and answer contracts.
# Use the LAST heading: the 2026-09-13 review found that ^## Report accepted a
# worker answer truncated after Done. Coverage also catches the rev_panerun
# final-message omission measured that day; a first heading cannot do that.
ROLES = {
    "worker": Role(briefing=_worker_briefing, expect=lambda prompt: "^### Not done"),
    "reviewer": Role(briefing=_reviewer_briefing, expect=coverage_expect),
}


def role_briefing(role: str, *, worktree_briefed: bool = False,
                  scratch: Optional[Path] = None,
                  report: Optional[Path] = None) -> str:
    """Dispatch through the same table that admits roles at the CLI boundary.

    The 2026-09-13 review's late unknown-role exit came from a second list of
    roles in an if-chain. Table dispatch removes that drift; an omitted role
    still adds nothing for existing callers and persisted records.
    """
    return ROLES[role].briefing(worktree_briefed, scratch, report) if role else ""


def expected_answer(role: str, explicit: Optional[str], prompt: str = "") -> str:
    """Resolve --expect for both spawn and send from the persisted role.

    An explicit flag wins, including an empty string to disable follow-up.
    Only an omitted flag (None) falls back to the role's default, computed
    from this turn's prompt (`Role.expect`): the LAST heading — the
    2026-09-13 review found that the first heading accepted truncated
    reports — or, for a reviewer, that heading plus every numbered question
    (`coverage_expect`). No role preserves the previous empty expectation.
    Keeping this at the CLI boundary leaves the turn runner independent of
    roles; an explicit regex that does not compile raises ValueError there
    (`checked_expect`).
    """
    if explicit is not None:
        return checked_expect(explicit)
    return ROLES[role].expect(prompt) if role else ""
