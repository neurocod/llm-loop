"""Test support for a host's judge tests: the mechanisms, without any tables.

A host that binds the core's judges to tables of its own (`HostPolicy`) pins
them against a tree of its own; what it should not have to rewrite is how a
judge is asked, how a fixture repository and a pool slot are laid out, and
how a junction is made. Those live here, used by this repository's core tests
and importable by a host's; every function takes the files it writes, so the
host keeps only its data. `monkeypatch` is pytest's fixture, passed in: the
module itself imports nothing outside the standard library.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from . import sandbox
from .registry import AgentRecord


class CodexWrap:
    """Codex's PowerShell wrapper around a command, quoted the way Codex
    quotes it — POSIX-style (`sandbox.split_shell`) — with `.format` so a
    row reads like the template it replaced."""

    PREFIX = ('"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" '
              "-Command ")

    def format(self, command: str) -> str:
        return self.PREFIX + shlex.quote(command)


PS = CodexWrap()


def _judge(tmp_path: Path, monkeypatch: Any, cwd: Path, fields: dict) -> Callable:
    """A sandboxed worker's escalation judge over `cwd`, its scratch under
    `tmp_path` (a role of "worker" unless `fields` names one)."""
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    fields.setdefault("role", "worker")
    return sandbox.escalation_judge(AgentRecord(
        name="w", cwd=str(cwd), sandbox="workspace-write", **fields))


def worker_judge(tmp_path: Path, monkeypatch: Any, **fields):
    """(work, ask): a worker's judge over an empty `tmp_path/ws`, asked
    through Codex's PowerShell wrapper; `ask(command, cwd=None, wrap=True)`,
    and `wrap=False` hands the command over as it is."""
    work = tmp_path / "ws"
    work.mkdir(exist_ok=True)
    judge = _judge(tmp_path, monkeypatch, work, fields)

    def ask(command: str, cwd: Optional[Path] = None, wrap: bool = True):
        return judge({"command": PS.format(command) if wrap else command,
                      "cwd": str(cwd or work)})

    return work, ask


def junction(link: Path, target: Path) -> None:
    """A directory junction (no privilege needed, unlike a symlink); Windows only."""
    import _winapi
    _winapi.CreateJunction(str(target), str(link))


def write_tree(folder: Path, files: Mapping[str, str]) -> None:
    """Each relative path of `files` under `folder`, with its text."""
    for name, text in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def worker_tree(tmp_path: Path, monkeypatch: Any, files: Mapping[str, str], **fields):
    """`worker_judge` over a tree holding `files`, and (on Windows) a
    junction `link` in it pointing OUT, at `tmp_path/outside`."""
    work, ask = worker_judge(tmp_path, monkeypatch, **fields)
    write_tree(work, files)
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    (outside / "a.txt").write_text("", encoding="utf-8")
    if os.name == "nt" and not (work / "link").exists():
        junction(work / "link", outside)
    return work, ask


def git_in(where: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git in a fixture tree; a failure names git's own stderr, so a red
    setup reads as its cause rather than as a bare exit status.

    `core.longpaths`: a nested submodule's objects sit under
    `<main>/.git/worktrees/<slot>/modules/<sub>/modules/<nested>/objects/..`,
    which under a long `--basetemp` passes Windows' 260-character MAX_PATH
    and the clone dies with "Filename too long" (measured 2026-11-07: an
    object path of 260 characters under an agent's scratch basetemp; a
    reviewer with a short basetemp passed 8/8). The flag reaches the clone
    git spawns through GIT_CONFIG_PARAMETERS."""
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.longpaths=true", *args], cwd=where,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise AssertionError(f"git {' '.join(map(str, args))} in {where} exited "
                             f"{done.returncode}:\n{done.stderr}{done.stdout}")
    return done


def pool_slot(tmp_path: Path, files: Mapping[str, str], deleted: Iterable[str] = (),
              name: str = "main-wt1", gitdir: Optional[str] = None) -> Path:
    """A pool slot as git lays one out: `<main>-wtN` is a linked worktree,
    its `.git` a file naming `<main>/.git/worktrees/<name>`, over `files`
    committed in `<main>`; the slot then `git rm`s each of `deleted`. With
    `gitdir`, a `.git` file naming it over the same files, and no repository
    behind it. A second call adds another slot of the same main."""
    main = tmp_path / "main"
    slot = tmp_path / name
    if gitdir is not None:
        write_tree(slot, files)
        (slot / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
        return slot
    if not (main / ".git").is_dir():
        write_tree(main, files)
        git_in(main, "init", "--quiet")
        git_in(main, "add", "--", *files)
        git_in(main, "commit", "--quiet", "--no-verify", "-m", "fixture")
    git_in(main, "worktree", "add", "--quiet", "--detach", str(slot))
    for path in deleted:
        git_in(slot, "rm", "--quiet", path)
    return slot


def add_submodule(tmp_path: Path, slot: Path, path: str, name: str) -> tuple[Path, Path]:
    """A one-commit repository `tmp_path/source` added to `slot` at `path`
    under submodule `name` and checked out: (its checkout, the source)."""
    source = tmp_path / "source"
    source.mkdir()
    git_in(source, "init", "--quiet")
    (source / "work.txt").write_text("initial\n", encoding="utf-8")
    git_in(source, "add", "work.txt")
    git_in(source, "commit", "--quiet", "-m", "Initial")
    git_in(slot, "-c", "protocol.file.allow=always", "submodule", "add",
           "--name", name, str(source), path)
    return slot / path, source


def commit_judge(tmp_path: Path, monkeypatch: Any, slot: Path, **fields):
    """A worker's judge over the pool slot `slot`, asked through Codex's
    wrapper; `ask.environment` is what the judge would hand the escalated
    command (the commit-approval store), {} when it hands nothing."""
    judge = _judge(tmp_path, monkeypatch, slot, fields)

    def ask(command: str, cwd: Optional[Path] = None):
        return judge({"command": PS.format(command), "cwd": str(cwd or slot)})

    ask.environment = getattr(judge, "environment", {})
    return ask
