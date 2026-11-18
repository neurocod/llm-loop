"""The git subprocess runner the bridges (and a host's own git tools) share.

One runner rather than a `subprocess.run(["git", ...])` per caller, so the
hardening below — UTF-8 path decoding, no credential prompt on an automatic
push, a bounded push — is written once and every caller keeps its own
defaults through `functools.partial`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union


class Fail(SystemExit):
    """Exit 1 with `error: <message>` on stderr (printed by the interpreter, once)."""

    def __init__(self, message: str):
        super().__init__(f"error: {message}")


# Bound on an automatic `git push`, s: a push of ~100 commits took ~5 s on
# 2026-09-06, so this is well over twenty times the measured cost — enough for a
# slow link, short enough that an auth stall does not eat a hook's whole budget
# (`terminal_prompt=False` is the other half of that guarantee).
PUSH_TIMEOUT_S = 120


def trusted_git(roots: tuple[Path, ...], path: Optional[str] = None) -> tuple[Optional[Path], str]:
    """The git executable the bridge's own PATH names first, or None and why.

    Absolute PATH entries only (a relative one is the cwd in disguise), and
    on Windows `git.exe` only: a `git.cmd`/`git.bat` is a script anyone with
    the folder may rewrite. Refused when it lies inside one of `roots` (the
    worker's writable roots): that git would be the worker's program."""
    names = ("git.exe",) if os.name == "nt" else ("git",)
    for folder in (path if path is not None else os.environ.get("PATH", "")).split(os.pathsep):
        folder = folder.strip().strip('"')
        if not folder or not os.path.isabs(folder):
            continue
        for name in names:
            candidate = Path(folder) / name
            if candidate.is_file():
                resolved = candidate.resolve()
                if any(resolved == root or root in resolved.parents for root in roots):
                    return None, f"the first git on PATH, {resolved}, lies in a writable root"
                return resolved, ""
    return None, "no git executable on the bridge's PATH"


# Set for a pinned git and everything it starts (hooks, filters, what they
# start). On Windows a program started by bare name (CreateProcess, cmd,
# Node's libuv: a hook's `execFileSync("git")`) is looked up in the current
# directory before the PATH unless this variable is set; a hook's cwd is the
# committed tree's root, the worker's to write (review of 2432 reproduced a
# planted git.exe running from a real pre-commit hook). Ignored elsewhere.
NO_CWD_LOOKUP = {"NoDefaultCurrentDirectoryInExePath": "1"}

# Configuration forced on every pinned git, after (and so over) every file's:
# the settings that start a program while git only answers a question or
# stages a file. A belt: what actually keeps the worker's configuration out
# is the binding (`Repository`). No `diff.external=`: git takes the empty
# value for a program name and every patch-producing diff dies with 128
# (measured 2026-11-18, Git for Windows 2.53); a diff the kit runs passes
# `--no-ext-diff` instead.
FORCED = ("core.fsmonitor=false", "core.editor=true", "sequence.editor=true")


@dataclass(frozen=True)
class Repository:
    """A repository named to git explicitly (`--git-dir`, `--work-tree`),
    never discovered.

    Discovery walks up from the cwd to the first `.git` and takes that
    repository's config — the place a program git starts is named
    (`core.fsmonitor`, `filter.<x>.clean` with `info/attributes`,
    `diff.external`, `core.hooksPath`). Under Codex's `workspace-write` the
    sandbox write-protects only the TOP-LEVEL `.git` of each writable root
    (codex-rs `protocol/src/permissions.rs`,
    `default_read_only_subpaths_for_writable_root`: `<root>/.git` and the
    gitdir it points at; `windows-sandbox-rs/src/allow.rs` denies exactly
    that path), so a submodule's `<slot>/<sub>/.git`, or a `.git` planted in
    any folder of the slot, is the worker's to point at a repository of its
    own (read 2026-11-18 at codex 0.166.0). A git bound to a `Repository`
    whose `git_dir` lies outside every writable root reads only system,
    global and that repository's own config (and the files they include) —
    none of them the worker's to write."""
    git_dir: Path
    work_tree: Path

    def arguments(self) -> list[str]:
        return ["--git-dir", str(self.git_dir), "--work-tree", str(self.work_tree)]


@dataclass(frozen=True)
class Pinned:
    """One git invocation fixed outside a sandbox: the absolute executable
    (`trusted_git`), the hooks folder, `FORCED` plus the caller's own
    settings, and — for every command that touches a repository — the
    `Repository` it is bound to."""
    executable: Path
    hooks: Path
    repository: Optional[Repository] = None
    extra: tuple[str, ...] = ()

    def argv(self, args: list[str]) -> list[str]:
        settings = (f"core.hooksPath={self.hooks}", *FORCED, *self.extra)
        bound = self.repository.arguments() if self.repository is not None else []
        return [str(self.executable),
                *(part for setting in settings for part in ("-c", setting)),
                *bound, *args]

    @staticmethod
    def environment(base: Optional[dict] = None) -> dict[str, str]:
        """`base` (the process's) without any `GIT_*` (GIT_DIR, GIT_WORK_TREE,
        GIT_CONFIG_*, GIT_EXTERNAL_DIFF, GIT_EXEC_PATH... each redirects or
        widens git), then `NO_CWD_LOOKUP`, no editor and no prompt: nobody
        sits at a pinned git's terminal."""
        env = {key: value for key, value in (os.environ if base is None else base).items()
               if not key.upper().startswith("GIT_")}
        env.update(NO_CWD_LOOKUP, GIT_EDITOR="true", GIT_SEQUENCE_EDITOR="true",
                   GIT_TERMINAL_PROMPT="0")
        return env


# The git a broker process runs instead of `git` by name (`pin`).
_pinned: Optional[Pinned] = None


def pin(pinned: Pinned) -> None:
    """Run every later `git` of this process as `pinned`.

    For a process started for one job outside a sandbox (the commit broker,
    `llm_agents.git.commit_broker`): `git` by name is whatever comes first
    in the search order, the inherited `GIT_*` redirect or widen what it
    does, and a repository found by discovery may be the worker's.
    Process-wide on purpose: the commit script, its gitlink check and the
    host's checks all run git through `git` below, and none of them may slip
    past the pin. `NO_CWD_LOOKUP` goes into this process's own environment
    too: a host check may start programs of its own."""
    global _pinned
    _pinned = pinned
    os.environ.update(NO_CWD_LOOKUP)


def _program() -> list[str]:
    return ["git"] if _pinned is None else _pinned.argv([])


def git(args: list[str], *, cwd: Union[str, Path, None] = None,
        check: bool = True, timeout: Optional[float] = None,
        capture_output: bool = True, text: bool = True,
        encoding: Optional[str] = "utf-8", errors: Optional[str] = None,
        terminal_prompt: Optional[bool] = None,
        failure: Literal["raise", "exit", "command"] = "raise",
        ) -> subprocess.CompletedProcess:
    """Run Git arguments (without the executable) and return CompletedProcess.

    `terminal_prompt=False` overlays GIT_TERMINAL_PROMPT=0 on the inherited
    environment: automatic pushes must not ask for credentials. Combined with
    `timeout=PUSH_TIMEOUT_S` (120 s; pushing about 100 commits took about 5 s,
    measured 2026-09-06), this prevents an authentication stall from eating a
    hook's budget and aborting an agent's start. None preserves the inherited
    environment and the existing unbounded timeout of other callers.

    UTF-8 decodes Git's path bytes independently of the Windows locale: cp1252
    can raise UnicodeDecodeError on Cyrillic filenames in a subprocess reader
    thread, leaving stdout as None. `encoding=None` retains the locale codec;
    `errors="replace"` gives tolerant decoding to callers that only scan the
    output. `text=False, encoding=None` keeps paths and blobs as bytes, without
    newline translation or decoding.

    `check=False` returns failed commands for the caller to inspect. Otherwise
    `failure` picks the CLI contract: "raise" uses subprocess's
    CalledProcessError, "exit" raises Fail with command, cwd and captured output,
    and "command" replays output to stderr before SystemExit. "exit" also turns
    TimeoutExpired into Fail even when check=False, so a timed-out automatic
    push can report a warning while keeping its local result. Capture can be
    disabled for commands whose output belongs directly on the console.
    """
    env = None if terminal_prompt is None else {
        **os.environ, "GIT_TERMINAL_PROMPT": "1" if terminal_prompt else "0",
    }
    if _pinned is not None:
        env = Pinned.environment()
        if terminal_prompt:
            env["GIT_TERMINAL_PROMPT"] = "1"
    try:
        proc = subprocess.run(
            [*_program(), *args], cwd=str(cwd) if cwd else None, env=env,
            capture_output=capture_output, text=text, encoding=encoding,
            errors=errors, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if failure == "exit":
            raise Fail(f"git {' '.join(args)} exceeded {timeout} s")
        raise
    if check and proc.returncode != 0:
        # `or ""`: an uncaptured run (`capture_output=False`) has both streams
        # as None, and a diagnosis must not turn into a TypeError on the way
        # out (review 2026-09-13).
        if failure == "exit":
            raise Fail(f"git {' '.join(args)} failed in {cwd or os.getcwd()}:\n"
                       f"{proc.stdout or ''}{proc.stderr or ''}".rstrip())
        if failure == "command":
            sys.stderr.write((proc.stdout or "") + (proc.stderr or ""))
            raise SystemExit(f"command failed ({proc.returncode}): git {' '.join(args)}")
        proc.check_returncode()
    return proc
