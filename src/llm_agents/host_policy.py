"""What a project adds to the agent bridges: its rules, handed over explicitly.

`llm_agents.codex` and `llm_agents.claude` are project-agnostic: shell parsing, path
checks, the approval stores, the registries and the turn machinery know no
project. Everything that names one project's files — the commands a worker may
run outside the sandbox (`WorkerCommands`), the gates' working directory, the
extra briefing lines, the commit command, the review profile's gate
(`ReviewProfile`), the badge's executable — is a `HostPolicy`, and the
checkout the bridges keep their state in is the host ROOT.

Both arrive from the host's entry point (a small `__main__.py` of the host's
own that imports its rules module) as arguments of the bridge's `main`, which
installs them for that one invocation (`installed`); nothing here looks for a
policy. A bridge run without one runs with `EMPTY` and says so
(`announce_empty`): nothing runs outside the sandbox unless the caller asked
for it in that same invocation (`explicit`), gates run in the project root,
briefings carry no host lines, and a review refuses to start. Silence there
would be the costly kind of fallback: a run that quietly lost its rules looks
exactly like one that had them.

The installed policy is process state, read at use (`active`, `root`): a
judge, a briefing and a gate run built in one invocation must agree on one
policy, and threading it through every call of the turn machinery would add
a parameter to functions that never read it.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional


def _project_root(command: str, project: Path) -> Path:
    return project


def _no_flags(roots: tuple[Path, ...]) -> frozenset[str]:
    return frozenset()


def _nothing_more(record: Any) -> str:
    return "every other program"


# Git's own store: what no agent writes on ANY host, so the empty policy keeps
# it (operator's decision 2026-11-03, which put each host's own runs beside it).
GIT_STORE = ((".git",),)


@dataclass(frozen=True)
class Reserved:
    """What a worker's allowlist holds back, as the briefing lists it
    (`text(record)`, after the core's own git/cmdlet/`node -e` refusals),
    and the record bit `spawn --allow-online-probes` that releases the
    online part: its help text and how `status` shows it set."""
    text: Callable[[Any], str] = _nothing_more
    online_help: str = ("let a worker escalate what the host policy reserves "
                        "for online tasks (this host reserves nothing)")
    online_label: str = "online commands allowed"


@dataclass(frozen=True, eq=False)
class WorkerCommands:
    """codex_agent: what a sandboxed worker may run outside the sandbox.

    `shapes` are `command_dispatch.Shape`s; `npm_scripts` the scripts `npm
    test`/`npm run` may start (`command_node.NpmScript`, a key ending in `:`
    a family); each table travels with the name its refusals give it, so a
    host's logged refusal texts stay word for word. `script_refusal` judges
    a node/tsx script beyond its arguments' paths, (script, args, ctx,
    judge_argv) -> refusal or ""; `flags(roots)` are the host's own facts the
    judges read off `CommandContext.host_flags`."""
    shapes: tuple = ()
    shapes_name: str = "the host policy's command shapes"
    npm_scripts: Mapping[str, Any] = field(default_factory=dict)
    npm_name: str = "the host policy's npm scripts"
    script_refusal: Optional[Callable[..., str]] = None
    flags: Callable[[tuple[Path, ...]], frozenset[str]] = _no_flags
    reserved: Reserved = Reserved()


@dataclass(frozen=True)
class ReviewProfile:
    """claude_agent's review profile: where the host keeps the gate hook the
    review reuses, all relative to the root. `settings` is the project
    settings file whose PreToolUse entry names the hook, `gate` the
    ask-user-gate directory (`hooks/ask_user_gate.exe`, its `cpp/` sources),
    `submodule` a submodule the reviewer may read with `git -C` templates
    ("" for none), `note` what the host says about other checkouts' commits."""
    settings: str
    gate: str
    submodule: str = ""
    note: str = ""


@dataclass(frozen=True, eq=False)
class HostPolicy:
    """One host's rules for the bridges; every field's default is the empty
    policy's answer. Compared by identity (`eq=False`): `EMPTY` is the value
    callers test for with `is`."""
    commands: WorkerCommands = WorkerCommands()
    # Path runs (whole components, case-folded) no agent writes.
    protected_parts: tuple[tuple[str, ...], ...] = GIT_STORE
    # Where a `## Gates` line runs: (command, project root) -> cwd.
    gate_cwd: Callable[[str, Path], Path] = _project_root
    # Extra lines of the worker's role briefing, before the edit rule.
    worker_lines: tuple[str, ...] = ()
    # How a worker in a pool slot is told to commit: (where the message file
    # goes, the worker's model) -> text. None: workers are not told to commit.
    commit_briefing: Optional[Callable[[str, str], str]] = None
    # How a refusal of the commit judge names the host's stand-in of the git
    # kit's commit script and its worktree pool (`codex.command_commit`).
    commit_script: str = "commit.py"
    pool_script: str = "llm_agents.git.worktree"
    # None: the review profile refuses to start (`llm_agents.claude.protocol`).
    review: Optional[ReviewProfile] = None
    # The on-screen counter's executable, relative to the root, most
    # preferred first; none for the empty policy.
    counter_exes: tuple[str, ...] = ()


EMPTY = HostPolicy()

# What the caller of THIS invocation asked for itself (`grant`): under
# `EMPTY` the only grants that leave the sandbox. "escalate": `spawn
# --escalate`; "gates": `spawn --gates-from-prompt`.
EXPLICIT_GRANTS = frozenset({"escalate", "gates"})

_installed: dict[str, Any] = {"policy": EMPTY, "root": None,
                              "explicit": frozenset()}


def active() -> HostPolicy:
    return _installed["policy"]


def root() -> Optional[Path]:
    """The installed host root, or None when no host named one."""
    return _installed["root"]


def explicit() -> frozenset[str]:
    """The `EXPLICIT_GRANTS` the caller of this invocation asked for."""
    return _installed["explicit"]


def grant(names: Iterable[str]) -> None:
    """Record what this invocation's caller asked for explicitly; lasts as
    long as the enclosing `installed` block."""
    names = frozenset(names)
    unknown = names - EXPLICIT_GRANTS
    if unknown:
        raise ValueError(f"unknown explicit grant(s): {sorted(unknown)}")
    _installed["explicit"] = _installed["explicit"] | names


def counter_paths() -> list[Path]:
    """The installed policy's counter executables, absolute."""
    base = root()
    if base is None:
        return []
    return [base / relative for relative in active().counter_exes]


def announce_empty(program: str, still: str) -> None:
    """Say, when no host policy is installed, that `program` runs without
    one; `still` names what the caller may still ask for itself."""
    if active() is not EMPTY:
        return
    sys.stderr.write(f"{program}: no host policy given: {still}; state is "
                     f"kept under the caller's checkout\n")


@contextmanager
def installed(policy: HostPolicy, at: Optional[Path] = None) -> Iterator[None]:
    """`policy` and root `at` (None: no root, as when no host named one) for
    the length of a block, with no explicit grants yet; then the previous
    host back, whatever the block raised. One bridge invocation is one
    block (`cli.main`), so a host never outlives the call that named it."""
    before = dict(_installed)
    _installed.update(policy=policy,
                      root=Path(at).resolve() if at is not None else None,
                      explicit=frozenset())
    try:
        yield
    finally:
        _installed.clear()
        _installed.update(before)
