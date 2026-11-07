"""Fixtures for the core's own judge tests (`core_policy.CORE_POLICY`): a
worker tree and a pool slot with no project's names in them."""

import json
import os
import subprocess

from llm_agents.codex import sandbox
from llm_agents.codex.registry import AgentRecord
from codex_wrap_support import _PS


def worker_judge(tmp_path, monkeypatch, **fields):
    """A sandboxed worker's judge, asked through Codex's PowerShell wrapper."""
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    work = tmp_path / "ws"
    work.mkdir(exist_ok=True)
    fields.setdefault("role", "worker")
    judge = sandbox.escalation_judge(AgentRecord(
        name="w", cwd=str(work), sandbox="workspace-write", **fields))

    def ask(command, cwd=None):
        return judge({"command": _PS.format(command), "cwd": str(cwd or work)})

    return work, ask


def junction(link, target):
    """A directory junction (no privilege needed, unlike a symlink)."""
    import _winapi
    _winapi.CreateJunction(str(target), str(link))


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


def worker_tree(tmp_path, monkeypatch, **fields):
    """`worker_judge` over `_TREE`, and a junction `link` pointing OUT."""
    work, ask = worker_judge(tmp_path, monkeypatch, **fields)
    for name, text in _TREE.items():
        path = work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir(exist_ok=True)
    (outside / "a.txt").write_text("", encoding="utf-8")
    if os.name == "nt" and not (work / "link").exists():
        junction(work / "link", outside)
    return work, ask


# What the pool slot's main repository tracks: `vault/` at the root and
# below a folder, a declared submodule, and a protected file the slot then
# deletes (`git rm`).
SUBMODULE = "lib/vendor"
_SLOT_FILES = {
    "src/a.ts": "", "tools/git/commit.py": "", "vault/k.txt": "",
    "app/vault/y.txt": "", "app/src/b.ts": "", "docs/vault/z.txt": "",
    ".gitmodules": f'[submodule "vendor"]\n\tpath = {SUBMODULE}\n\turl = ./vendor\n'}


def git_in(where, *args):
    """git in a fixture tree; a failure names git's stderr. `core.longpaths`:
    a nested submodule's objects under a long basetemp pass MAX_PATH (the
    host copy, `codex_sandbox_support._git_in`, has the measurement)."""
    done = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                           "-c", "core.longpaths=true", *args], cwd=where,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    if done.returncode != 0:
        raise AssertionError(f"git {' '.join(map(str, args))} in {where} exited "
                             f"{done.returncode}:\n{done.stderr}{done.stdout}")
    return done


def pool_slot(tmp_path, name="main-wt1"):
    """A linked worktree `<main>-wtN` of a repository holding `_SLOT_FILES`."""
    main = tmp_path / "main"
    slot = tmp_path / name
    if not (main / ".git").is_dir():
        for file, text in _SLOT_FILES.items():
            (main / file).parent.mkdir(parents=True, exist_ok=True)
            (main / file).write_text(text, encoding="utf-8")
        git_in(main, "init", "--quiet")
        git_in(main, "add", "--", *_SLOT_FILES)
        git_in(main, "commit", "--quiet", "--no-verify", "-m", "fixture")
    git_in(main, "worktree", "add", "--quiet", "--detach", str(slot))
    git_in(slot, "rm", "--quiet", "docs/vault/z.txt")
    return slot


def slot_with_submodule(tmp_path):
    """`pool_slot` with its declared submodule added and checked out."""
    slot = pool_slot(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    git_in(source, "init", "--quiet")
    (source / "work.txt").write_text("initial\n", encoding="utf-8")
    git_in(source, "add", "work.txt")
    git_in(source, "commit", "--quiet", "-m", "Initial")
    git_in(slot, "-c", "protocol.file.allow=always", "submodule", "add",
           "--name", "vendor", str(source), SUBMODULE)
    return slot, slot / SUBMODULE, source


def commit_judge(tmp_path, monkeypatch, slot, **fields):
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    fields.setdefault("role", "worker")
    judge = sandbox.escalation_judge(AgentRecord(
        name="w", cwd=str(slot), sandbox="workspace-write", **fields))

    def ask(command, cwd=None):
        return judge({"command": _PS.format(command), "cwd": str(cwd or slot)})

    return ask
