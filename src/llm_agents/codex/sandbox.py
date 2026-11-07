"""Permissions: sandbox modes, writable roots, agent briefings and escalation.

Owns workspace preparation, the thread's options (`thread_params`: the rights
the thread is opened with, and the model, resent beside them on every resume),
shell unwrapping and the cleanup grammar. Turn execution, event handling,
retries and logging live in turn.py; this module does not run the app-server
or consume its event stream.
"""

from __future__ import annotations

import os
import re
import shlex
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .commands import (  # noqa: F401 — split_shell/unwrap_shell re-exported
    CommandContext, path_refusal, pool_slot, resolve_path, shapes_text,
    split_shell, unwrap_shell, worker_command_refusal,
)
from .file_renewal import renew_file  # re-exported for edit-judge callers
from .gates import escalate_pattern
from .paths import host_root, scratch_dir
from .registry import AgentRecord
from .roles import role_briefing
from .command_commit import _git_admin
from ..commit_identity import ENV as COMMIT_APPROVAL_ENV
from ..host_policy import EMPTY as EMPTY_POLICY
from ..host_policy import active as host_policy
from ..host_policy import explicit as explicit_grants


def _protected_text() -> str:
    """The host's protected path runs as briefings name them."""
    return ", ".join(f"`{'/'.join(part)}/`"
                     for part in host_policy().protected_parts)


# Our sandbox vocabulary, and the protocol mode each one is made of. Three of
# the four are the protocol's own (`SandboxMode` in the schema bundle); the
# fourth is a POLICY we assemble, because the protocol cannot express it:
# `SandboxPolicy.readOnly` carries a `networkAccess` flag and nothing else — no
# writable root of any kind — so an agent that must run a test suite cannot have
# one under `read-only`, and the only writable-anything mode is
# `workspaceWrite`, which makes its workspace ROOT writable.
#
# `scratch-write` is therefore `workspace-write` whose root is a scratch
# directory outside the project (`paths.scratch_dir`): the project stays
# readable and unwritable, while the agent gets somewhere to work. Measured
# 2026-08-24 from inside the sandbox, by an agent running under this very mode:
# the repo's whole gate passes (51 of 51 that day), reads of the checkout work,
# a write into it is refused with "Access is denied" — and so is a write through
# a junction the agent plants in its own workspace and points at the project.
SCRATCH_WRITE = "scratch-write"

SANDBOX_MODES = {
    "read-only": "read-only",
    "workspace-write": "workspace-write",
    "danger-full-access": "danger-full-access",
    SCRATCH_WRITE: "workspace-write",
}


def workspace_of(record: AgentRecord) -> Path:
    """Where the agent's shell starts — which under `workspace-write` is also
    what it may write to (plus the second root of `scratch_root`), hence the
    whole of `scratch-write`."""
    if record.effective_sandbox == SCRATCH_WRITE:
        return scratch_dir(record.name)
    return Path(record.cwd or host_root())


def scratch_root(record: AgentRecord) -> Optional[Path]:
    """A writable directory OUTSIDE the tree for a `workspace-write` agent.

    Measured 2026-09-13, two workers in their first hour: an agent whose only
    writable root is the checkout puts its throwaway files INTO the checkout —
    a `scratch/` folder for a baseline it wanted to diff, a `_measure.mjs`
    beside the sources for a measuring script — and `git status`
    then shows them beside the real edit for the orchestrator to sort out at
    commit time. The protocol allows more: `turn/start` takes a `sandboxPolicy`
    whose `workspaceWrite` form lists extra `writableRoots`, so the directory
    `scratch-write` uses as its whole workspace (`paths.scratch_dir`) is handed
    to a `workspace-write` agent as a SECOND root, and the briefing names it.

    The briefing is the working half. Measured 2026-09-13 by a test agent
    under this mode with the root on the wire: its cwd, this directory AND a
    bare `$env:TEMP\\<file>` were all writable — Codex's `workspaceWrite`
    admits the temp directory by default (`excludeTmpdirEnvVar: false`), so
    the agents that littered the tree had the RIGHT to write here all along
    and lacked only the name. The root is still sent, so the promise in the
    briefing does not rest on a default that a config line can switch off.

    Only that mode: `scratch-write` already works inside the directory,
    `read-only` has no writable root to add to, and `danger-full-access`
    needs none.
    """
    if record.effective_sandbox == "workspace-write":
        return scratch_dir(record.name)
    return None


def report_path(record: AgentRecord) -> Optional[Path]:
    """The worker's answer file, shared by spawn, briefing and turn reader.

    Only workers with a scratch root use files. Full-access workers have no
    scratch root and keep the final-message fallback, explicitly stated in
    their briefing. Reviewers retain their message and coverage contract.
    """
    scratch = scratch_root(record)
    return scratch / "report.md" if record.role == "worker" and scratch else None


def turn_sandbox_policy(record: AgentRecord) -> Optional[dict]:
    """The per-turn `sandboxPolicy`, sent only when the thread's `sandbox`
    MODE cannot say it: `thread/start` takes a mode string and no roots,
    `turn/start` takes a policy object with them (schema bundle of codex-cli
    0.154.0, `WorkspaceWriteSandboxPolicy`). The object is sent whole, so it
    repeats the network choice `.codex/config.toml` makes for this mode
    (`[sandbox_workspace_write] network_access = true`) rather than leaving a
    field for the configured policy to fill in.

    `record.writable` (`spawn --writable DIR`) rides in the same list, after
    the scratch root: the operator's answer to "may this agent have one more
    directory" (asked 2026-09-13), and the protocol's only door for it — the
    mode string cannot name a directory. `check_writable_roots` says where
    the flag is meaningless."""
    roots: list[str] = []
    scratch = scratch_root(record)
    if scratch is not None:
        roots.append(str(scratch))
    roots.extend(record.writable)
    if not roots:
        return None
    return {"type": "workspaceWrite", "writableRoots": roots,
            "networkAccess": True}


def check_role_sandbox(record: AgentRecord) -> None:
    """Refuse worker instructions when the project cannot be edited.

    The 2026-09-13 role review found that workers were told to edit the tree
    even under read-only and scratch-write, where that contradicts the sandbox
    briefing. Refuse at spawn before reserving the name or archiving its logs;
    reviewers can work under every mode because they need no project writes.
    """
    if record.role == "worker" and record.effective_sandbox not in (
            "workspace-write", "danger-full-access"):
        raise SystemExit(
            f"--role worker needs workspace-write or danger-full-access; "
            f"{record.name!r} is {record.effective_sandbox}")


def check_writable_roots(record: AgentRecord) -> None:
    """Refuse a `--writable` that could not mean what it says.

    It widens a `workspaceWrite` policy and nothing else: `read-only` has no
    root list to add to and `danger-full-access` needs none, so under either
    the flag would be stored, printed by `status`, and do nothing. Each root
    must EXIST — like `--cwd`, a mistyped one is the operator's typo, and
    creating it would put an empty directory wherever the typo pointed. Under
    `scratch-write` a root that contains the project would make the project
    writable, the hole `_refuse_project_inside_workspace` closes for the
    workspace itself. Called by `spawn` before the name is claimed and by
    `prepare_workspace` on every turn, so a `send` on a record whose root has
    since vanished is refused the same way.
    """
    if not record.writable:
        return
    if SANDBOX_MODES.get(record.effective_sandbox) != "workspace-write":
        raise SystemExit(
            f"--writable needs a workspace-write or scratch-write agent; "
            f"{record.name!r} is {record.effective_sandbox}")
    for root in record.writable:
        if not Path(root).is_dir():
            raise SystemExit(
                f"--writable {root} is not a directory (nothing was created "
                f"there)")
    if record.effective_sandbox == SCRATCH_WRITE:
        project = Path(record.cwd or host_root()).resolve()
        for root in record.writable:
            resolved = Path(root).resolve()
            if project == resolved or resolved in project.parents:
                raise SystemExit(
                    f"scratch-write would make the project writable: "
                    f"--writable {root} contains {project}")


def check_online_probes(record: AgentRecord) -> None:
    """Refuse `--allow-online-probes` where nothing reads it: only a
    sandboxed worker's allowlist (`opens_commands`) hands the bit to the
    host's judges, so on any other agent the flag would be stored, printed
    by `status`, and
    do nothing (review 2026-11-03) — the same refusal as `--writable`'s.
    Called by `spawn` before the name is claimed and on every turn."""
    if record.allow_online_probes and not opens_commands(record):
        raise SystemExit(
            f"--allow-online-probes needs --role worker under workspace-write; "
            f"{record.name!r} is role {record.role or 'none'} under "
            f"{record.effective_sandbox}")


def _sandbox_facts(scratch: Optional[Path] = None,
                   extra: Iterable[str] = (),
                   edit_roots: Optional[Iterable[Path]] = None) -> str:
    """What every sandboxed agent has to be told about running things here.

    Measured 2026-09-12 by an agent under this sandbox: a Node child process
    spawned with PIPED stdio is refused with `spawn EPERM` (errno -4048) — any
    pipe on stdin, stdout or stderr, `shell` or not — while `stdio: 'inherit'`
    and `'ignore'` spawns succeed and `node <script>` itself runs. That is
    every npm gate of this repo: `npm run typecheck` (the runner pipes 28
    jobs), `npm test` (Vite `exec`s `net use` at startup, and `exec` always
    pipes) — both die before doing anything, and two workers that day spent
    a dozen commands each retrying them. Told once, up front, so the retry
    is not paid again.

    `scratch` is the second writable root of a `workspace-write` agent
    (`scratch_root`): named here because an agent that is not told it has one
    puts its baselines and measuring scripts into the tree. `extra` are the
    operator's `--writable` roots, named for the same reason: a right the
    agent has not been told about is one it will not use.

    `edit_roots` is given when the agent may ask to leave the sandbox
    (`approvalPolicy: on-request`): an edit that fails inside is then retried
    outside by `file_change_judge`'s say, and the agent is told who says it
    (`_edit_escalation_fact`).
    """
    facts = (
        "Running things in this sandbox: a Node child process with PIPED stdio "
        "is refused (`spawn EPERM`), so `npm test`, `npm run typecheck`, vitest "
        "and every script that spawns with default stdio fail at startup. Do "
        "not retry them or look for a workaround. `node <script>` runs, and so "
        "does a direct `node node_modules/typescript/bin/tsc --noEmit -p "
        "tsconfig.json`. In your final answer name every gate you could not "
        "run; the orchestrator runs them outside the sandbox."
    )
    if scratch is not None:
        facts += (
            f"\nTemporary files — baselines to diff against, measuring scripts, "
            f"captured outputs — go to {scratch}, a writable directory OUTSIDE "
            f"the project that nobody commits. Do not create them inside the "
            f"project tree: everything you leave there shows up in `git status` "
            f"next to your real edit."
        )
    granted = list(extra)
    if granted:
        facts += (
            f"\nYou may also write to {', '.join(granted)} — granted by the "
            f"operator for this task."
        )
    if edit_roots is not None:
        facts += _edit_escalation_fact(edit_roots)
    return facts


def _edit_escalation_fact(roots: Iterable[Path]) -> str:
    """What an agent that may ask to leave the sandbox is told about its
    EDITS: who answers when apply_patch fails inside, and what to do on a
    "no". A worker read the wrapper's decline as "patch rejected by user",
    stopped and asked a person who was not there (2026-11-03)."""
    named = ", ".join(str(root) for root in roots) or "none"
    protected = _protected_text()
    return (
        f"\nWhen an edit fails inside the sandbox and is retried without it, "
        f"the wrapper decides, not a person — nobody is there to ask: it is "
        f"granted when every file lies inside your writable roots ({named}) "
        f"and none under {protected}, and declined otherwise. A declined edit "
        f"(\"rejected\") is the wrapper's answer with its reason in the "
        f"wrapper's log, not a human decision: do not stop or ask for "
        f"permission — re-read the file and retry the edit, or write the file "
        f"anew, and if it still fails name the file in your final answer. A "
        f"file you created that later refuses writes inside the sandbox "
        f"(UnauthorizedAccessException on Set-Content/Remove-Item) has a "
        f"stale access list, not a lock: edit it with apply_patch — the "
        f"wrapper's retry outside the sandbox also renews the file, after "
        f"which sandboxed writes to it work again — rather than retrying the "
        f"same shell write."
    )


def _worktree_fact(workspace: Path,
                   commit_roots: Optional[list[Path]] = None,
                   *, model: str = "") -> str:
    """Told to an agent whose workspace is a LINKED git worktree.

    Such a checkout keeps its index and HEAD under the main checkout's `.git`,
    outside every writable root the agent has, so any git command that writes
    fails on `index.lock` — `git mv` included. Measured 2026-09-13: a worker in
    pool slot whose task said "move with git mv" spent the turn asking whether
    it might move files another way, and moved nothing. A linked worktree is
    recognised by its `.git` being a FILE (`gitdir: ...`), not a directory.

    `commit_roots` (the writable roots outside the tree, scratch first) is
    given when the agent may commit here — a worker that escalates by
    allowlist in a pool slot, on a host that names how (`commit_briefed`):
    it commits through the host's commit command outside the sandbox
    (`command_commit._commit` judges tools/git/commit.py), with its message
    file in one of these roots, since commit.py deletes it; the words are
    the host's `HostPolicy.commit_briefing`.

    `model` is the spawn's persisted AgentRecord.model, also used on resume;
    the orchestrator's model must not receive credit for the worker's commit.
    Old records with no model leave its name for the worker to fill in.
    """
    if not (workspace / ".git").is_file():
        return ""
    head = (
        "\nThis checkout is a linked git worktree: its index lives in the main "
        "repository's `.git`, outside your writable roots, so git commands "
        "that WRITE (`git mv`, `git add`, `git commit`, `git stash`, "
        "`git checkout -- <file>`) fail with an `index.lock` error. Do not "
        "retry or ask: move and create files with plain filesystem commands "
        "(`Move-Item`, `New-Item`), keep using read-only git (`git status`, "
        "`git diff`, `git log`, `git show`), and ")
    if commit_roots is None:
        return head + ("list every move in your final answer — the "
                       "orchestrator stages and commits.")
    scratch, *writable = commit_roots
    where = f"your scratch directory ({scratch})"
    if writable:
        where += (" or a --writable directory ("
                  + ", ".join(str(root) for root in writable) + ")")
    return head + host_policy().commit_briefing(where, model)


def commit_briefed(record: AgentRecord) -> bool:
    """Whether the agent is told to commit itself (`_worktree_fact`): a
    worker that escalates by allowlist (`opens_commands`) in a pool slot
    (`commands.pool_slot`) — exactly where `command_commit._commit` grants it
    — on a host that says how (`HostPolicy.commit_briefing`)."""
    return (host_policy().commit_briefing is not None and opens_commands(record)
            and pool_slot(workspace_of(record).resolve()))


def _escalation_briefing(pattern: str) -> str:
    """Told to an agent with an `escalate` rule, after `_sandbox_facts`: the
    gates it was just told not to retry ARE runnable — outside the sandbox,
    by asking. Without this it would take the facts at their word. The
    cleanup grant is named only where it applies (`_automatic`)."""
    text = (
        f"\nEXCEPTION: commands matching the regex {pattern!r} are allowed to "
        f"run outside the sandbox. Run them with escalated permissions "
        f"(request the escalation; it is granted automatically for exactly "
        f"these commands and declined for everything else) and read their "
        f"output as usual. Run them from your working directory or below it."
    )
    if not _automatic():
        return text
    return text + (
        f" Deleting your own {', '.join(sorted(CLEANUP_DIRS))} inside your "
        f"working directory is granted the same way — as ONE command with a "
        f"literal relative path (`Remove-Item -Recurse -Force .pytest-tmp`), "
        f"not chained, not through a variable; other deletions are not."
    )


def _worker_escalation_briefing(record: AgentRecord, extra: str) -> str:
    """Told to a worker that escalates by allowlist (`opens_commands`), in
    place of `_escalation_briefing`: the shapes it may run outside, in the
    words of the host's `WorkerCommands.shapes`, and what is declined — the
    core's refusals, then the host's (`Reserved.text`). The agent never
    sees a decline's reason — Codex reports "rejected" — so the list is told
    up front."""
    policy = host_policy()
    protected = _protected_text()
    text = (
        f"\nEXCEPTION: what the sandbox refuses (the npm gates, vitest, a Node "
        f"child with pipes) you may run OUTSIDE it by requesting escalated "
        f"permissions, when the command is one of these shapes: "
        f"{shapes_text(policy)}. The wrapper grants such a request automatically "
        f"when its working directory is inside {workspace_of(record)} and "
        f"every path it names lies inside your writable roots and outside "
        f"{protected}; a chain of granted commands with `;` or `&&`, `2>&1`, "
        f"`> <file in your roots>` and a pipe into `Select-Object -Last N` or "
        f"`Select-String` are read through. It declines everything else: git "
        f"and every cmdlet (`Set-Content`, `Copy-Item`, `Remove-Item`"
        + (f" but your own {', '.join(sorted(CLEANUP_DIRS))}" if _automatic() else "")
        + f"), `node -e`, `python -c`, "
        f"{policy.commands.reserved.text(record)}, and any `$`, backtick, `&`, "
        f"parentheses, braces, `@`, `,` or wildcard outside quotes. A declined "
        f"escalation is that rule, not a person's answer: do not ask for "
        f"permission — run the step inside the sandbox, do it another way, or "
        f"name it in your final answer for the orchestrator."
    )
    if extra:
        text += (f" Also granted for this task: commands matching the regex "
                 f"{extra!r}.")
    return text


def escalation_briefing(record: AgentRecord) -> str:
    """The escalation paragraph `thread_params` closes the instructions
    with, or "" when the agent may not ask (`asks_approval`)."""
    escalate = escalation_pattern(record)
    if opens_commands(record):
        return _worker_escalation_briefing(record, escalate)
    return _escalation_briefing(escalate) if escalate else ""


def _scratch_briefing(project: str, extra: Iterable[str] = (),
                      edit_roots: Optional[Iterable[Path]] = None) -> str:
    """What a `scratch-write` agent has to be told, because its cwd is not the
    project: where the project is, that it cannot write there, and the one
    operational fact that decides whether it can run a Python test suite at all.

    Measured: pytest's shared temp root (`%TEMP%/pytest-of-<user>`) already
    exists and belongs to the user OUTSIDE the sandbox, so `os.scandir` on it
    raises `PermissionError: [WinError 5]` and every test wanting a `tmp_path`
    errors in setup — 32 of 51 on 2026-08-24. With `--basetemp` inside the
    writable workspace the same suite is green. Sent as `developerInstructions`
    rather than glued onto the caller's prompt: it is a fact about the
    environment, not about the task.
    """
    return (
        f"Your working directory is a scratch directory, and it is the ONLY "
        f"place you may write.\n"
        f"The project you are working on is at {project}. You can read all of "
        f"it, you cannot write any of it, and its instructions for you (if it "
        f"has any) are in AGENTS.md or CLAUDE.md there — read that before "
        f"anything else.\n"
        f"Running a Python test suite: pytest's shared temp root belongs to a "
        f"user outside your sandbox and every `tmp_path` fixture fails on it, "
        f"so pass a basetemp inside your working directory — "
        f"`python -m pytest <tests> -q --basetemp=./pytest-tmp`.\n"
        + _sandbox_facts(extra=extra, edit_roots=edit_roots)
    )


def _refuse_project_inside_workspace(project: Path, workspace: Path) -> None:
    """`scratch-write` promises one thing; this is the only thing that keeps it.

    The mode is `workspace-write` aimed elsewhere, so "the project is not
    writable" is true for exactly as long as the project is not inside the
    workspace. Point `--cwd` at the scratch directory (or anywhere under it) and
    the agent is simply a `workspace-write` agent, while `status`, the flag and
    the briefing all still say the project is read-only.

    Compared RESOLVED, because a junction or symlink is a second way in: a link
    inside the scratch root that targets the project is writable through the
    link, and the unresolved strings look unrelated. (A junction planted by the
    agent mid-turn is out of reach of any check here — that one the sandbox
    itself refuses, verified 2026-08-24: writing through such a link was denied.)
    """
    project = project.resolve()
    workspace = workspace.resolve()
    if project == workspace or workspace in project.parents:
        raise SystemExit(
            f"scratch-write would make the project writable: {project} is "
            f"inside the agent's own workspace {workspace}. Point --cwd at the "
            f"project you want reviewed, or use --sandbox workspace-write and "
            f"say so.")


# Directories an agent may delete un-sandboxed without naming them in its
# `--escalate` regex: the pytest basetemp the briefing tells it to create, and
# pytest's own cache. Under `approvalPolicy: on-request` Codex asks before a
# `Remove-Item -Recurse` even inside the writable workspace, the judge declined
# it as a non-gate, and the worker reported "rejected by user" and left the
# directory behind (2026-09-13; user decision: grant at least this). Basenames
# only, matched after resolving the target under the agent's workspace — the
# grant runs with the user's full rights, so a path outside it never qualifies.
CLEANUP_DIRS = frozenset({".pytest-tmp", "pytest-tmp", ".pytest_cache"})

_DELETE_COMMAND = re.compile(
    r"^\s*(?P<verb>Remove-Item|ri|rm|rmdir|rd|del|erase)\s+(?P<rest>.+)$",
    re.IGNORECASE)
# cmd.exe writes its switches with a slash, and only for its own verbs. Without
# this the judge read `/s` in `rmdir /s /q .pytest-tmp` as a TARGET — a path
# resolving to `C:\s`, outside the workspace — and declined the whole command,
# which is how an agent working under `cmd` was refused the cleanup the
# briefing had just told it to do (review 2026-09-13). Kept to the switches
# that delete: no `/a:<attrs>`, whose value is a second token to misread.
#
# The VERB does not decide this, the SHELL does (`split_shell`): all four of
# these words are aliases of `Remove-Item` under PowerShell, where `/s` is the
# path `C:\s` and `rmdir /s /q .pytest-tmp` deletes `C:\s`, `C:\q` and the
# basetemp. Judged by the verb, that command walked out of the sandbox with a
# grant (review 2026-09-13, reproduced against the live judge) — which is the
# whole difference between a cleanup and an un-sandboxed delete of two paths
# nobody named. cmd also accepts its switches glued together (`/s/q`), so a
# slash token is a run of switches rather than one.
_CMD_VERBS = frozenset({"del", "erase", "rmdir", "rd"})
_CMD_FLAGS = frozenset({"/s", "/q", "/f"})
_CMD_SWITCH_RUN = re.compile(r"/[^/]*")
# The ONLY flags a granted deletion may carry — an allowlist, not "skip
# whatever starts with a dash": review 2026-09-13 showed `-Recurse:(./x.ps1)`
# is an expression PowerShell evaluates, so an unknown flag is a "no".
_BARE_FLAGS = frozenset({
    "-recurse", "-force", "-confirm:$false", "-recurse:$true", "-force:$true",
    "-path", "-literalpath", "-lp", "-r", "-rf", "-fr", "-f", "--",
})
# ...and the two whose VALUE (the next token) is not a path.
_VALUED_FLAGS = frozenset({"-erroraction", "-ea"})
# A literal path and nothing else: no `$`/`~`/`%` (PowerShell and cmd expand
# them BEFORE the delete sees the string — `$env:USERPROFILE\.pytest-tmp` read
# as a workspace-relative literal is an out-of-workspace delete, review
# 2026-09-13), no `,` (a PowerShell array: `src,./.pytest-tmp` deletes both),
# no wildcards, no quotes or brackets or shell punctuation of any kind.
#
# A QUOTED token may hold spaces and nothing else besides: quoting is the only
# reason a path with a space arrives as one token at all, and splitting it on
# whitespace (as this parser did until review 2026-09-13) turned
# `'./my dir/.pytest-tmp'` into two targets that are neither a cleanup dir nor
# inside the workspace — the agent was refused a deletion it had spelled
# correctly.
_LITERAL_PATH = re.compile(r"^[A-Za-z0-9_.\\/:-]+$")
_QUOTED_PATH = re.compile(r"^[A-Za-z0-9_.\\/: -]+$")
_COMMAND_SEPARATORS = re.compile(r"[;&|`\n]|\$\(")


def cleanup_targets(command: str, cwd: Path, workspace: Path) -> bool:
    """True when `command` deletes ONLY entries from `CLEANUP_DIRS` inside
    `workspace`: one deletion command (no chaining, no substitution), only
    allowlisted flags — PowerShell's `-` ones, and cmd's `/` ones when cmd is
    the shell that will run one of its own verbs (`_CMD_FLAGS`) — every target a
    literal path token without wildcards that resolves under the workspace with
    an allowed basename. Anything the parser does not understand is a "no" — the
    cost of a wrong "no" is one stray directory, the cost of a wrong "yes" is an
    un-sandboxed delete."""
    shell, inner = split_shell(command)
    if _COMMAND_SEPARATORS.search(inner):
        return False
    match = _DELETE_COMMAND.match(inner)
    if match is None:
        return False
    # Under anything but cmd — PowerShell, or no wrapper at all — a leading
    # slash is the start of a path, and a root-relative one (`/src/repo/…`
    # with the cwd on C:) is a path the workspace check then judges like any
    # other.
    slash_is_switch = shell == "cmd" and match.group("verb").lower() in _CMD_VERBS
    try:
        # Quote-aware, because a quoted path may hold spaces; `posix=False`
        # keeps Windows backslashes as backslashes rather than as escapes.
        tokens = shlex.split(match.group("rest"), posix=False)
    except ValueError:
        return False  # unbalanced quotes: not a command this parser can read
    targets: list[str] = []
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        lowered = token.lower()
        if slash_is_switch and lowered.startswith("/"):
            if all(switch in _CMD_FLAGS
                   for switch in _CMD_SWITCH_RUN.findall(lowered)):
                continue
            return False  # an unknown switch, or `/a:r` with a value in it
        if lowered.startswith("-"):
            if lowered in _VALUED_FLAGS:
                skip_next = True
                continue
            if lowered in _BARE_FLAGS:
                continue
            return False  # an unknown flag, or a flag carrying an expression
        quoted = len(token) > 2 and token[0] == token[-1] and token[0] in "'\""
        target = token[1:-1] if quoted else token
        if not (_QUOTED_PATH if quoted else _LITERAL_PATH).match(target):
            return False
        targets.append(target)
    if not targets:
        return False
    workspace = workspace.resolve()
    for target in targets:
        resolved = (cwd / target).resolve() if not Path(target).is_absolute() \
            else Path(target).resolve()
        if resolved.name not in CLEANUP_DIRS:
            return False
        try:
            resolved.relative_to(workspace)
        except ValueError:
            return False
        if resolved == workspace or _holds_link(resolved):
            return False
    return True


def _holds_link(folder: Path) -> bool:
    """Whether a tree to be deleted holds a junction or symlink: Windows
    PowerShell 5.1's `Remove-Item -Recurse` deletes THROUGH a junction, so a
    `.pytest-tmp` with a link inside empties wherever the link points
    (review 2026-11-03). Unreadable counts as holding one; a target that is
    not a directory (absent included) holds nothing."""
    if not folder.is_dir():
        return False
    try:
        for top, dirs, files in os.walk(folder, followlinks=False,
                                        onerror=_raise):
            for name in (*dirs, *files):
                entry = os.path.join(top, name)
                if os.path.islink(entry) or os.path.isjunction(entry):
                    return True
    except OSError:
        return True
    return False


def _raise(error: OSError) -> None:
    raise error


def escalation_pattern(record: AgentRecord) -> str:
    """The regex an escalation request is judged by: the operator's
    `--escalate` and the recorded gates (`gates.escalate_pattern`), either one
    enough. One function, because the judge, the approval policy and the
    briefing must agree on it — a briefing that names a rule the judge does
    not apply sends the agent into refusals it was told would not happen.

    For a worker (`opens_commands`) both are ADDITIONS to the allowlist
    (`commands.worker_command_refusal`), never a narrowing: `--escalate` is
    the orchestrator granting one odd command for one spawn, and a recorded
    gate is a line the orchestrator wrote under `## Gates`. With nothing left
    that narrows, neither can widen a narrowing (the denylist era's
    `--escalate` narrowed, and a recorded gate widened past it — a Codex
    review of the allowlist, 2026-11-03).

    Without a host (`host_policy.EMPTY`) each half counts only when the
    caller asked for it in this invocation (`host_policy.explicit`): a
    recorded `--escalate` or gate list replayed by a later `send` is not
    an ask."""
    escalate = record.escalate
    derived = escalate_pattern(record.gates)
    if not _automatic():
        granted = explicit_grants()
        escalate = escalate if "escalate" in granted else ""
        derived = derived if "gates" in granted else ""
    if escalate and derived:
        return f"(?:{escalate})|(?:{derived})"
    return escalate or derived


def _automatic() -> bool:
    """Whether the wrapper may grant on its own account — cleanup deletes,
    edit retries, recorded gates: only under a host policy. Under `EMPTY`
    nothing leaves the sandbox but what this invocation's caller named."""
    return host_policy() is not EMPTY_POLICY


def opens_commands(record: AgentRecord) -> bool:
    """Whether the agent escalates by the worker ALLOWLIST (the host's
    `WorkerCommands.shapes`) rather than by `--escalate` alone.

    Operator's decision 2026-11-03: "freer access for the agents
    the harness starts". A worker escalated only what `--escalate` named and
    stumbled on every command left out of it; a denylist that granted the
    rest could not be made safe (both reviews), so a sandboxed worker
    escalates every command of a known shape (`worker_command_refusal`).
    `--escalate` stays the rule for every other agent (a reviewer's
    read-only sandbox stays read-only) and adds to a worker's
    (`escalation_pattern`)."""
    return (record.role == "worker"
            and record.effective_sandbox == "workspace-write")


def asks_approval(record: AgentRecord) -> bool:
    """`approvalPolicy: on-request` — the agent may ask to leave the sandbox,
    and `escalation_judge` / `file_change_judge` answer."""
    return bool(escalation_pattern(record)) or opens_commands(record)


def escalation_judge(record: AgentRecord) -> Optional[Callable[[dict], Verdict]]:
    """The "yes" an unattended run can give: a command may run OUTSIDE the
    sandbox when the agent asks — a worker's when it is of a known shape
    (`commands.worker_command_refusal`), anyone's when it matches the
    `escalate` regex (`escalation_pattern`).

    Why it exists: the Windows sandbox cannot start a Node child with piped
    stdio (`_sandbox_facts`), so `npm test` and `npm run typecheck` — the two
    gates every code piece needs — cannot run inside it at all. Rather than
    dropping the sandbox for the whole turn (`danger-full-access`) or having
    the orchestrator run the gates after the fact, the escaping commands are
    judged one by one. A command runs with the user's full rights once it
    escapes, so the regex is the caller's promise that what it matches is a
    gate and not a shell — and for a worker (`opens_commands`) the allowlist
    is the promise, the regex adding to it.

    The judge reads the REQUEST — its `command` and `cwd` — not a flag: the
    command must match, must be a command (not stdin to a running one), and
    must run inside the agent's own workspace, so an escaped `npm test` cannot
    be pointed at some other tree.
    """
    escalate = escalation_pattern(record)
    worker = opens_commands(record)
    if not escalate and not worker:
        return None
    pattern = re.compile(escalate) if escalate else None
    workspace = workspace_of(record).resolve()
    roots = tuple(root.resolve() for root in writable_roots(record))
    scratch = scratch_root(record)
    # Git's admin directory is outside a slot worker's writable roots. A
    # scratch file would let the process swapping a junction forge approval.
    commit_store = (tempfile.TemporaryDirectory(prefix="commit-approvals-",
                                               dir=_git_admin(workspace))
                    if worker and pool_slot(workspace) else None)
    policy = host_policy()
    context = CommandContext(
        workspace=workspace, roots=roots,
        scratch=scratch.resolve() if scratch is not None else None,
        allow_online_probes=record.allow_online_probes,
        commit_approvals=Path(commit_store.name) if commit_store else None,
        policy=policy, host_flags=policy.commands.flags(roots))
    decline_tail = worker_decline_tail()
    automatic = _automatic()

    def judge(params: dict) -> Verdict:
        if params.get("kind", "command") != "command":
            return Verdict(False, "not a command (input to a running one)")
        command = params.get("command")
        if not isinstance(command, str):
            return Verdict(False, "the request names no command")
        try:
            cwd = Path(str(params.get("cwd") or workspace)).resolve()
            cwd.relative_to(workspace)
        except (OSError, ValueError):
            return Verdict(False, f"cwd {params.get('cwd')} is outside your "
                                  f"working directory {workspace}")
        refusal = (worker_command_refusal(command, cwd, context)
                   if worker else "")
        if worker and not refusal:
            return Verdict(True, "a worker command shape")
        if pattern is not None and pattern.search(unwrap_shell(command)):
            return Verdict(True, "matches the escalate rule")
        if automatic and cleanup_targets(command, cwd, workspace):
            return Verdict(True, "deletes the agent's own pytest directories")
        if worker:
            return Verdict(False, f"{refusal}; {decline_tail}")
        return Verdict(False, f"does not match the escalate rule /{escalate}/")

    if commit_store is not None:
        judge.environment = {COMMIT_APPROVAL_ENV: json.dumps({
            "workspace": str(workspace), "directory": commit_store.name})}
        judge.close = commit_store.cleanup
    return judge


def worker_decline_tail() -> str:
    """How a worker's decline ends in the log: where the shapes live (as the
    host names its table), and what the agent is told to do instead
    (`_worker_escalation_briefing`)."""
    return (f"a worker escalates only {host_policy().commands.shapes_name} "
            f"— run it inside the sandbox or ask the orchestrator")


@dataclass(frozen=True)
class Verdict:
    """A judge's answer and the reason it is logged with (`events.jsonl`,
    `logs`, the turn summary). Truthy when granted, so a caller asking only
    "yes or no" reads it as a bool."""
    granted: bool
    reason: str = ""

    def __bool__(self) -> bool:
        return self.granted


def writable_roots(record: AgentRecord) -> list[Path]:
    """Every directory the agent's sandbox lets it write: its workspace, the
    scratch root of `workspace-write` and the operator's `--writable` roots.
    Empty under `read-only` (nothing is writable) and `danger-full-access`
    (no sandbox, so nothing ever asks to leave it)."""
    if SANDBOX_MODES.get(record.effective_sandbox) != "workspace-write":
        return []
    roots = [workspace_of(record)]
    scratch = scratch_root(record)
    if scratch is not None:
        roots.append(scratch)
    roots.extend(Path(root) for root in record.writable)
    return roots


def file_change_judge(record: AgentRecord) -> Callable[[list[str]], Verdict]:
    """Whether an edit that failed INSIDE the sandbox may be retried outside it.

    Codex asks this (`item/fileChange/requestApproval`, reason "command failed;
    retry without sandbox?") when apply_patch could not write, and the request
    names no file — the paths come from the item's `item/started`
    (`protocol.AppServer`). Declining every such request, as the wrapper did
    until 2026-11-03, reached the agent as "patch rejected by user": a
    worker stopped its fix and asked a person who was not there (its
    events.jsonl, lines 422-427). Operator's decision that day: grant it when
    every path lies inside the agent's own writable roots (`writable_roots`)
    and none in the host's `protected_parts`; otherwise decline with the
    reason named.

    Each path is judged by `commands.path_refusal`, the same rule a worker's
    command arguments meet: resolved (a junction inside the tree that points
    elsewhere is elsewhere), `\\\\?\\` taken off, alternate data streams and
    drive-relative paths refused. The request's own `grantRoot` and an item
    the wrapper never saw start are refused before this
    (`protocol.AppServer._answer_file_change`).

    A granted edit's stale files are renewed first (`renew_file`): the
    likeliest reason the edit failed inside is a stale DACL, and renewing it
    lets the agent's next in-sandbox write succeed instead of asking again.
    A file kept stale is named in the reason with why.
    """
    roots = [root.resolve() for root in writable_roots(record)]
    base = workspace_of(record)
    protected = host_policy().protected_parts
    automatic = _automatic()

    def judge(paths: list[str]) -> Verdict:
        if not automatic:
            return Verdict(False, "no host policy: a failed edit is not "
                                  "retried outside the sandbox")
        if not roots:
            return Verdict(False, f"a {record.effective_sandbox} agent has no "
                                  f"writable root")
        if not paths:
            return Verdict(False, "the wrapper did not see which files this "
                                  "change touches")
        resolved_paths: list[Path] = []
        for raw in paths:
            why = path_refusal(raw, base, roots, protected)
            if why:
                return Verdict(False, why)
            resolved_paths.append(resolve_path(raw, base)[1])
        renewed = 0
        kept: list[str] = []
        for path in resolved_paths:
            done, why = renew_file(path)
            renewed += done
            if why:
                kept.append(f"{path.name} ({why})")
        return Verdict(True, "every path is inside the writable roots"
                       + (f"; renewed {renewed} file(s)" if renewed else "")
                       + (f"; kept stale: {', '.join(kept)}" if kept else ""))

    return judge


# Told to every agent, whatever its sandbox or role: any of them may spawn
# sub-agents. The main threads that lost their children's work by 2026-10-24
# answered after a 10 s or 60 s `wait_agent` timeout of their own choosing, or
# without waiting (`subagents.SUB_AGENT_GRACE_S` has the logs); nothing in
# Codex's own prompts says to wait. The wrapper's grace is the backstop, this
# the cheaper fix at the source. The restart clause is the user's decision of
# 2026-10-24 ("the worker can track it and restart its sub-agent"), the same
# request `subagents.RESTART_PROMPT` makes after a cut.
SUB_AGENT_FACT = (
    "If you start sub-agents, wait for every one of them to finish (use "
    "`wait_agent` with long timeouts, again until all are done) BEFORE your "
    "final answer, and build that answer on their results: after your turn "
    "completes the session is kept open only for a limited time, then every "
    "sub-agent still running is interrupted. If a sub-agent of yours was "
    "interrupted and its work is still needed, restart it."
)


def thread_params(record: AgentRecord) -> dict:
    """Thread options, sent when starting AND when resuming.

    `approvalPolicy: never` goes with an explicit sandbox: unattended, there is
    nobody to ask, so the sandbox alone decides what the agent may do and an
    approval request would only stall the turn
    (see `protocol.AppServer._answer_server_request`).
    With an `escalate` rule, and for a sandboxed worker (`asks_approval`),
    the policy is `on-request` instead: the agent may ask, and
    `escalation_judge` / `file_change_judge` are the ones who answer.

    Resuming repeats them rather than inheriting whatever the stored thread was
    opened with, so `status` and the running agent cannot disagree about what it
    is allowed to do — the registry is the single answer to that question, and
    `AgentRecord.effective_sandbox` is where it answers.
    """
    policy = record.effective_sandbox
    mode = SANDBOX_MODES.get(policy)
    if mode is None:
        # Reached only through a hand-edited registry; argparse refuses the rest.
        # Named rather than passed through, because the server's answer to an
        # unknown sandbox would arrive as a failed turn with no hint of why.
        raise SystemExit(
            f"agent {record.name!r} has an unknown sandbox {policy!r}; "
            f"expected one of {', '.join(sorted(SANDBOX_MODES))}")
    workspace = workspace_of(record)
    asks = asks_approval(record)
    params: dict[str, Any] = {
        "cwd": str(workspace),
        "sandbox": mode,
        "approvalPolicy": "on-request" if asks else "never",
    }
    worktree_fact = ""
    # Edits that fail inside are retried outside only under `on-request`.
    # (and only where the wrapper may grant on its own: `_automatic`).
    edit_roots = writable_roots(record) if asks and _automatic() else None
    if policy == SCRATCH_WRITE:
        project = Path(record.cwd or host_root())
        _refuse_project_inside_workspace(project, workspace)
        params["developerInstructions"] = _scratch_briefing(
            str(project), record.writable, edit_roots)
    elif policy != "danger-full-access":
        # The sandbox's own limits apply to every sandboxed agent, not only
        # the one whose cwd is elsewhere.
        worktree_fact = _worktree_fact(
            workspace, writable_roots(record)[1:] if commit_briefed(record)
            else None, model=record.model)
        params["developerInstructions"] = (
            _sandbox_facts(scratch_root(record), record.writable, edit_roots)
            + worktree_fact)
    if record.role:
        briefing = role_briefing(
            record.role, worktree_briefed=bool(worktree_fact),
            scratch=scratch_root(record), report=report_path(record))
        existing = params.get("developerInstructions", "")
        params["developerInstructions"] = existing + ("\n\n" if existing else "") + briefing
    existing = params.get("developerInstructions", "")
    params["developerInstructions"] = (
        existing + ("\n\n" if existing else "") + SUB_AGENT_FACT)
    # Escalation qualifies every preceding briefing, including a role under
    # full access where no sandbox briefing created developerInstructions.
    if asks:
        params["developerInstructions"] = (
            params.get("developerInstructions", "")
            + escalation_briefing(record))
    if record.model:
        params["model"] = record.model
    return params


def prepare_workspace(record: AgentRecord) -> tuple[Path, Optional[dict]]:
    """Prepare the writable roots before starting the app-server."""
    # The server process starts where the agent will work. A `scratch-write`
    # workspace is OURS and gets created; anything else is the user's
    # directory and must already exist — creating it instead put a stray
    # directory into the project on a mistyped `--cwd`, under `read-only` of
    # all modes, before a sandbox that forbids exactly that had even started.
    workspace = workspace_of(record)
    if record.effective_sandbox == SCRATCH_WRITE:
        workspace.mkdir(parents=True, exist_ok=True)
    elif not workspace.is_dir():
        # SystemExit, like the sandbox check above: this is the operator's
        # typo, and it deserves the message rather than a traceback out of
        # Popen or a "failed turn" line that blames the model.
        raise SystemExit(
            f"agent {record.name!r} works in {workspace}, which does not "
            f"exist (nothing was created there — check --cwd)")
    # The second root is OURS, like a scratch workspace, and is created here
    # for the same reason: a root that does not exist is one the agent cannot
    # write to, and the briefing has just told it to. The operator's roots
    # are NOT ours to create — `check_writable_roots` says why.
    scratch = scratch_root(record)
    if scratch is not None:
        scratch.mkdir(parents=True, exist_ok=True)
    # Both checks run on EVERY turn, not only at `spawn`: a record persisted
    # by an older version, or hand-edited, reaches `send` without passing
    # `cmd_spawn`, and a worker resumed under `read-only` would be briefed to
    # edit a tree it cannot write (found in review).
    check_role_sandbox(record)
    check_writable_roots(record)
    check_online_probes(record)
    return workspace, turn_sandbox_policy(record)
