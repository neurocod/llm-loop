"""The core judge tests' data (`core_policy.CORE_POLICY`): a worker tree and
a pool slot with no project's names in them. The mechanisms are
`llm_agents.codex.testing`, shared with any host's own judge tests."""

import json
from functools import partial

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
