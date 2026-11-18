"""The core judge tests' data (`core_policy.CORE_POLICY`): a worker tree and
a pool slot with no project's names in them. The mechanisms are
`llm_agents.codex.testing`, shared with any host's own judge tests."""

import json
import os
import shutil
import time
from functools import partial
from pathlib import Path

from llm_agents.codex import testing
from llm_agents.codex.testing import commit_judge, git_in, junction, worker_judge  # noqa: F401

# Every file the core host's entries and the npm judges look for, the
# protected `vault/`, and `sub/` with a package.json of its own.
_TREE = {
    "package.json": json.dumps({"scripts": {
        "test": "vitest run", "check:a": "tool src/a.ts", "dev": "vite"}}),
    "sub/package.json": json.dumps({"scripts": {"test": "vitest run"}}),
    "scripts/plain.mjs": "",
    "node_modules/tsx/dist/loader.mjs": "", "node_modules/typescript/bin/tsc": "",
    "node_modules/vitest/vitest.mjs": "", "node_modules/vite/bin/vite.js": "",
    "node_modules/.bin/vitest.cmd": "", "node_modules/.bin/tsc.cmd": "",
    "node_modules/.bin/tsx.cmd": "",
    "bin/try_patch.py": "", "bin/edit.py": "", "bin/read.py": "", "bin/gen.py": "",
    "bin/agents/start.py": "", "bin/team_launch_all.py": "", "bin/other.py": "",
    "tools/git/commit.py": "",
    "src/a.ts": "", "vault/key.txt": "", ".git/config": "",
}

worker_tree = partial(testing.worker_tree, files=_TREE)

# What the pool slot's main repository tracks: `vault/` at the root and
# below a folder, a declared submodule, and a protected file the slot then
# deletes (`git rm`).
SUBMODULE = "lib/vendor"
_SLOT_FILES = {
    "src/a.ts": "", "tools/git/commit.py": "", "vault/k.txt": "",
    "app/vault/y.txt": "", "app/src/b.ts": "", "docs/vault/z.txt": "",
    ".gitmodules": f'[submodule "vendor"]\n\tpath = {SUBMODULE}\n\turl = ./vendor\n'}


def pool_slot(tmp_path, name="main-wt1"):
    """A linked worktree `<main>-wtN` of a repository holding `_SLOT_FILES`."""
    return testing.pool_slot(tmp_path, _SLOT_FILES, ["docs/vault/z.txt"], name)


def slot_with_submodule(tmp_path):
    """`pool_slot` with its declared submodule added and checked out."""
    slot = pool_slot(tmp_path)
    checkout, source = testing.add_submodule(tmp_path, slot, SUBMODULE, "vendor")
    return slot, checkout, source


def scratch_message(tmp_path: Path, text: str = "Slot work\n") -> Path:
    """`<tmp_path>/scratch/msg.txt` holding `text`: the judge reads a `-F`
    file when it grants the commit (`command_commit._message_snapshot`)."""
    message = tmp_path / "scratch" / "msg.txt"
    message.parent.mkdir(parents=True, exist_ok=True)
    message.write_text(text, encoding="utf-8")
    return message


def planted_gitdir(tmp_path: Path, sub: Path, marker: Path) -> Path:
    """A copy of `sub`'s repository where the worker could write it (its
    scratch) whose config names a program as fsmonitor and as the `evil`
    clean filter; the program appends to `marker` and passes its input
    through."""
    script = tmp_path / "scratch" / "planted.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_bytes(f"#!/bin/sh\necho ran >> '{marker.as_posix()}'\ncat\n".encode("utf-8"))
    script.chmod(0o755)
    real = git_in(sub, "rev-parse", "--absolute-git-dir").stdout.strip()
    planted = tmp_path / "scratch" / "planted.git"
    shutil.copytree(real, planted)
    for key, value in (("core.worktree", sub.as_posix()), ("core.fsmonitor", script.as_posix()),
                       ("filter.evil.clean", script.as_posix())):
        git_in(tmp_path, "config", "--file", str(planted / "config"), key, value)
    return planted


def point(git_file: Path, gitdir: Path) -> None:
    # Git marks the file hidden on Windows; opening it with 'w' is refused.
    with git_file.open("r+", encoding="utf-8") as stream:
        stream.write(f"gitdir: {gitdir}\n")
        stream.truncate()


def nested_filter(tmp_path: Path, sub: Path) -> Path:
    """Review 2 of 2432: `sub/.git` pointed at `planted_gitdir`, every file
    of `sub` filtered `evil`, and its tracked `work.txt` touched (same size,
    new mtime), so the next git that asks whether `sub` is dirty runs the
    planted filter. The marker the filter writes."""
    marker = tmp_path / "nested-filter.txt"
    point(sub / ".git", planted_gitdir(tmp_path, sub, marker))
    (sub / ".gitattributes").write_text("* filter=evil\n", encoding="utf-8")
    later = time.time() + 10
    os.utime(sub / "work.txt", (later, later))
    return marker
