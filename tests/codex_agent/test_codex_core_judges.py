"""The core's deep judges under its own test host (`core_policy.CORE_POLICY`):
pytest's delete guards, try_patch, npm/npx/node, the slot commit and its
submodule rules, junction paths, the worker briefing skeleton and the fake
wire. A host's own tables are pinned in the host's repository; these hold
for any host binding the same judges."""

from test_codex_agent import _no_badge  # noqa: F401 — see _no_badge docstring
import json
import os

import pytest

from test_codex_agent import _answers, _first_request, _run_fake_turn
from llm_agents.codex import roles, sandbox
from llm_agents.codex.registry import AgentRecord
from llm_agents.codex.store import AgentStore
from core_judge_support import (
    SUBMODULE, commit_judge, git_in, junction, pool_slot, slot_with_submodule,
    worker_judge, worker_tree)

windows = pytest.mark.skipif(os.name != "nt", reason="Windows junction")


def _declined(ask, command, reason, **where):
    verdict = ask(command, **where)
    assert not verdict and reason in verdict.reason, (command, verdict.reason)
    assert sandbox.worker_decline_tail() in verdict.reason, command


def _granted(ask, command, **where):
    verdict = ask(command, **where)
    assert verdict and verdict.reason == "a worker command shape", (command, verdict.reason)


# --- pytest -----------------------------------------------------------------

@pytest.mark.parametrize("command, reason", [
    ("python -m pytest --basetemp src", "pytest deletes"),
    ("python -m pytest --basetemp=.", "a writable root, which pytest deletes"),
    ("python -m pytest --basetemp ..", "outside your writable roots"),
    ("python -m pytest -p bin.read", "loads a module"),
    ("python -m pytest -qp bin.read", "loads a module"),
    ("python -m pytest -o addopts=--basetemp=..", "only the ini keys"),
    ("python -m pytest -oaddopts=--basetemp=..", "only the ini keys"),
    ("python -m pytest -qo addopts=--basetemp=..", "only the ini keys"),
    ("python -m pytest --override-ini=addopts=-pbin.read", "only the ini keys"),
    ("python -m pytest -o cache_dir", "only the ini keys"),
    ("python -m pytest -o cache_dir=../out", "outside your writable roots"),
    ("python -m pytest '@args.txt'", "reads arguments from a file"),
    ("python -m bin.read", "only pytest"),
    ("python -c 1", "python option -c"),
])
def test_pytest_refuses_what_deletes_or_loads_unread_code(tmp_path, monkeypatch,
                                                          command, reason):
    _, ask = worker_tree(tmp_path, monkeypatch)
    _declined(ask, command, reason)


@pytest.mark.parametrize("command", [
    "python -m pytest -q",
    "python -m pytest --basetemp {scratch}",
    "python -m pytest --basetemp .pytest-tmp",
    "python -m pytest --basetemp new-dir",
    "python -m pytest -o cache_dir=.cache -o log_cli=true -pno:cacheprovider",
    "py -3 -m pytest -qvx",
])
def test_pytest_grants_a_new_or_scratch_basetemp_and_listed_ini_keys(
        tmp_path, monkeypatch, command):
    _, ask = worker_tree(tmp_path, monkeypatch)
    (tmp_path / "scratch").mkdir(exist_ok=True)
    _granted(ask, command.format(scratch=tmp_path / "scratch" / "pt"))


# --- try_patch and the entry table -------------------------------------------

@pytest.mark.parametrize("command, reason", [
    ("python bin/try_patch.py --file src/a.ts --old a --new b -- git push",
     "`git` is not a worker command shape"),
    ("python bin/try_patch.py --file src/a.ts --old a --new b git push",
     "`git` is not a worker command shape"),
    ("python bin/try_patch.py --file src/a.ts --old a --new b --cw sub -- npm test",
     "try_patch option --cw"),
    ("python bin/try_patch.py --selftest", "try_patch option --selftest"),
    ("python bin/try_patch.py --file ../x.ts --old a --new b -- npm test",
     "outside your writable roots"),
    ("python bin/try_patch.py --file vault/key.txt --old a --new b -- npm test", "vault/"),
    # Judged where it runs: sub/package.json has no `check:a`.
    ("python bin/try_patch.py --file src/a.ts --old a --new b --cwd=sub -- npm run check:a",
     "has no npm script 'check:a'"),
    ("python bin/try_patch.py --file src/a.ts --old a --new b -- "
     "python -m pytest -o addopts=--basetemp=..", "only the ini keys"),
    ("python bin/agents/start.py", "starts agents"),
    ("python bin/team_launch_all.py", "launches"),
    ("python bin/other.py", "is not on the core test entrypoints"),
    ("python bin/gen.py", "run it with --check"),
    ("python bin/gen.py --check --w", "--w: a generator is granted only"),
])
def test_try_patch_and_the_entries_judge_what_they_run(tmp_path, monkeypatch,
                                                       command, reason):
    _, ask = worker_tree(tmp_path, monkeypatch)
    _declined(ask, command, reason)


@pytest.mark.parametrize("command, where", [
    ("python bin/try_patch.py --file src/a.ts --old a --new b -- npm test", ""),
    ("python bin/try_patch.py --file src/a.ts --old a --new b --cwd=sub -- npm test", ""),
    ("python bin/try_patch.py --file=src/a.ts --remove a --expect-fail "
     "--reject-exit 5 -- npm run check:a", ""),
    ("python BIN/Try_Patch.py --file src/a.ts --old a --new b -- npm test", ""),
    ("python ../bin/try_patch.py --file ../src/a.ts --old a --new b -- npm test", "sub"),
    ("python bin/edit.py src/a.ts", ""),
    ("python bin/gen.py --check --stdout", ""),
])
def test_try_patch_and_the_entries_grant_their_shapes(tmp_path, monkeypatch, command, where):
    work, ask = worker_tree(tmp_path, monkeypatch)
    _granted(ask, command, cwd=work / where)


# --- npm, npx, node ----------------------------------------------------------

def test_npm_npx_node_judge_bodies_hooks_and_installs(tmp_path, monkeypatch):
    """Listing a script is not trusting its body: hooks npm runs unasked, and
    every body, are judged as worker commands (`command_node._npm_body_refusal`)."""
    work, ask = worker_tree(tmp_path, monkeypatch)
    hooks, bodies = work / "hooks", work / "bodies"
    hooks.mkdir()
    bodies.mkdir()
    (hooks / "package.json").write_text(json.dumps({"scripts": {
        "test": "vitest run", "pretest": "git reset --hard",
        "check:a": "tool a", "postcheck:a": "node ../scripts/plain.mjs"}}))
    (bodies / "package.json").write_text(json.dumps({"scripts": {
        "check:git": "git reset --hard", "check:install": "npm install",
        "check:redirect": "tool a > C:/outside/x.txt", "check:loop": "npm run check:loop",
        "check:ok": "tool a && npm test", "test": "tool b", "dev": "vite"}}))
    declined = {
        ("npm test", hooks): "pretest, which npm runs around test",
        ("npm run check:a", hooks): "postcheck:a, which npm runs around",
        ("npm run dev", bodies): "not on the core test npm scripts",
        ("npm run check:git", bodies): "`git` is not a worker command shape",
        ("npm run check:install", bodies): "npm install",
        ("npm run check:redirect", bodies): "a pipe or redirect in its body",
        ("npm run check:loop", bodies): "nested deeper than",
        ("npm test | Out-File x", work): "pipe into Out-File",
        ("npm test --script-shell=evil.exe", work): "npm option --script-shell",
        ("npm exec x", work): "npm exec",
        ("npm test -- 'a&b'", work): "cmd shims",
        ("npx vitest run -t 'a|b'", work): "cmd shims",
        ("npx --yes cowsay", work): "npx option --yes",
        ("npx tsc --noEmit", bodies): "npx would download it",
        ("npx vitest", work): "watches",
        ("npx vitest run --watch", work): "never exits",
        ("npx tsc '@compile.args'", work): "a response file",
        ("node node_modules/typescript/bin/tsc '@compile.args'", work): "a response file",
        ("node node_modules/vite/bin/vite.js", work): "only vitest's and TypeScript's",
        ("node node_modules/vitest/vitest.mjs", work): "watches",
        ("node -e 1", work): "node option -e",
        ("node --import evil.mjs scripts/plain.mjs", work): "--import",
    }
    for (command, cwd), reason in declined.items():
        _declined(ask, command, reason, cwd=cwd)
    loader = (work / "node_modules/tsx/dist/loader.mjs").as_uri()
    for command, cwd in [("npm test", work), ("npm run check:a", work),
                         ("npm --prefix sub test", work), ("npm run check:ok", bodies),
                         ("npx --no-install tsc --noEmit", bodies),
                         ("npx vitest run src -t 'one name|other name'", work),
                         ("npx tsx scripts/plain.mjs --x 1", work),
                         (f"node --import {loader} scripts/plain.mjs", work),
                         ("node --check scripts/plain.mjs", work),
                         ("node node_modules/vitest/vitest.mjs run --root .", work)]:
        _granted(ask, command, cwd=cwd)


# --- paths -------------------------------------------------------------------

@windows
@pytest.mark.parametrize("command", [
    "tool link/a.txt", "python bin/edit.py link",
    "python bin/try_patch.py --file link/a.txt --old a --new b -- npm test",
])
def test_a_junction_out_of_the_tree_is_outside_the_roots(tmp_path, monkeypatch, command):
    _, ask = worker_tree(tmp_path, monkeypatch)
    _declined(ask, command, "outside your writable roots")


@pytest.mark.parametrize("command, reason", [
    ("tool vault/key.txt", "vault/"), ("tool VAULT/key.txt", "vault/"),
    ("tool .git/config", ".git/"), ("tool .git./config", ".git/"),
    ("tool ../outside/a.txt", "outside your writable roots"),
])
def test_protected_and_outside_paths_are_refused(tmp_path, monkeypatch, command, reason):
    _, ask = worker_tree(tmp_path, monkeypatch)
    _declined(ask, command, reason)


# --- the slot commit ---------------------------------------------------------

def test_a_worker_commits_named_paths_in_its_pool_slot(tmp_path, monkeypatch):
    slot = pool_slot(tmp_path)
    extra = tmp_path / "pad"
    extra.mkdir()
    ask = commit_judge(tmp_path, monkeypatch, slot, writable=[str(extra)])
    message = tmp_path / "scratch" / "msg.txt"
    for command in [f"python tools/git/commit.py -F {message} src/a.ts",
                    f"python tools/git/commit.py --file {extra / 'm.txt'} src/a.ts src/gone.ts",
                    "python tools/git/commit.py -m 'subject' -m '- bullet' src/a.ts",
                    f"python tools/git/commit.py -F {message} -- src/a.ts",
                    f"python tools/git/commit.py -F {message} app/src tools/git"]:
        _granted(ask, command)
    assert ask(f"python ../tools/git/commit.py -C .. -F {message} src/a.ts", cwd=slot / "src")
    sibling = pool_slot(tmp_path, "main-wt2")
    untracked = slot / "notes" / "vault" / "u.txt"
    untracked.parent.mkdir(parents=True)
    untracked.write_text("", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    for command, reason in {
        f"python tools/git/commit.py -F {elsewhere / 'msg.txt'} src/a.ts":
            "outside your writable roots",
        "python tools/git/commit.py -F src/a.ts app/src/b.ts": "outside the tree",
        f"python tools/git/commit.py -C {sibling} -F {message} src/a.ts":
            "outside your writable roots",
        f"python tools/git/commit.py -F {message} vault/k.txt": "vault/",
        f"python tools/git/commit.py -F {message} .": "the whole tree",
        f"python tools/git/commit.py -F {message} app": "holds app/vault/y.txt",
        f"python tools/git/commit.py -F {message} docs": "holds docs/vault/z.txt",
        f"python tools/git/commit.py -F {message} notes": "holds notes/vault/u.txt",
        f"python tools/git/commit.py -F {message} lib": f"submodule {SUBMODULE}",
        f"python tools/git/commit.py -F {message} ':/'": "pathspec magic",
        f"python tools/git/commit.py -F {message} 'src/*.ts'": "wildcard",
        f"python tools/git/commit.py --amend -F {message} src/a.ts": "commit.py option --amend",
        f"python tools/git/commit.py -F{message} src/a.ts": "glued",
        f"python tools/git/commit.py src/a.ts -F {message}": "options go before the paths",
        f"python tools/git/commit.py -F {message}": "no paths",
        "python tools/git/commit.py -m -x src/a.ts": "-m with no value",
    }.items():
        _declined(ask, command, reason)


def test_commit_is_declined_outside_a_pool_slot(tmp_path, monkeypatch):
    work = pool_slot(tmp_path, "review-tree")
    ask = commit_judge(tmp_path, monkeypatch, work)
    _declined(ask, f"python tools/git/commit.py -F {tmp_path / 'scratch' / 'm.txt'} src/a.ts",
              "only in a pool slot")


def test_commit_in_the_slot_s_own_declared_submodule_only(tmp_path, monkeypatch):
    slot, sub, source = slot_with_submodule(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    msg = tmp_path / "scratch" / "msg.txt"
    for target in (f"-C {sub}", f"--cwd={SUBMODULE}"):
        verdict = ask(f"python tools/git/commit.py {target} -F {msg} work.txt")
        assert verdict, verdict.reason
    _declined(ask, f"python tools/git/commit.py -C {sub} -F {msg} .", "whole tree")
    (sub / "src" / "vault").mkdir(parents=True)
    (sub / "src" / "vault" / "x.txt").write_text("", encoding="utf-8")
    _declined(ask, f"python tools/git/commit.py -C {sub} -F {msg} src", "holds src/vault/x.txt")
    plain = slot / "unlisted"
    plain.mkdir()
    git_in(plain, "init", "--quiet")
    _declined(ask, f"python tools/git/commit.py -C {plain} -F {msg} work.txt",
              "not a declared submodule")
    other = pool_slot(tmp_path, "main-wt2")
    git_in(other, "-c", "protocol.file.allow=always", "submodule", "add",
           "--name", "vendor", str(source), SUBMODULE)
    _declined(ask, f"python tools/git/commit.py -C {other / SUBMODULE} -F {msg} work.txt",
              "another slot's submodule")


# --- the worker briefing -----------------------------------------------------

def test_the_worker_briefing_skeleton_and_the_host_s_lines(tmp_path):
    told = roles.role_briefing("worker")
    assert told.count("## Report") == 1
    assert "## Report\n### Done\n### Files changed\n### Gates\n" in told
    assert "### Missed siblings\n### Not done / open questions" in told
    assert "Use these headings, in this order, and nothing after them" in told
    assert "You cannot commit" in told
    assert "Core test host line." in told
    record = AgentRecord(name="w", cwd=str(tmp_path), sandbox="workspace-write", role="worker")
    instructions = sandbox.thread_params(record)["developerInstructions"]
    assert sandbox.shapes_text() in instructions and "`tool <paths>`" in sandbox.shapes_text()
    assert "`.git/`, `vault/`" in instructions


# --- the fake wire -----------------------------------------------------------

def _approvals():
    return {line["item"]: line for line in AgentStore("w").events()
            if line.get("_meta") == "approval"}


def test_edits_retried_outside_are_judged_on_the_wire(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: scratch)
    monkeypatch.setenv("FAKE_SCRATCH", str(scratch))
    outcome = _run_fake_turn(tmp_path, monkeypatch, "file-approvals",
                             sandbox="workspace-write", timeout=15.0)
    assert outcome.text == "file approvals handled"
    decision = {request: _answers(tmp_path)[request]["result"]["decision"]
                for request in (9201, 9202, 9203, 9206, 9209, 9210)}
    assert decision == {9201: "accept", 9202: "decline", 9203: "decline",
                        9206: "decline", 9209: "decline", 9210: "decline"}
    logged = _approvals()
    assert logged["f1"]["decision"] == "accept"
    assert "outside your writable roots" in logged["f2"]["reason"]
    assert ".git/" in logged["f3"]["reason"]
    assert "did not see which files" in logged["never-started"]["reason"]
    assert "grantRoot" in logged["f9"]["reason"]
    assert "grantRoot" in logged["ghost"]["reason"]


def test_commands_are_judged_on_the_wire(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    (tmp_path / "package.json").write_text(json.dumps({"scripts": {
        "test": "vitest run", "typecheck": "tool a"}}))
    outcome = _run_fake_turn(tmp_path, monkeypatch, "escalate", timeout=15.0,
                             sandbox="workspace-write", role="worker", expect="")
    assert outcome.text.startswith("escalation handled")
    answers = _answers(tmp_path)
    assert {request: answers[request]["result"]["decision"]
            for request in range(9101, 9107)} == {
        9101: "accept", 9102: "decline", 9103: "decline", 9104: "decline",
        9105: "decline", 9106: "decline"}
    reasons = [line.get("reason") or "" for line in AgentStore("w").events()
               if line.get("_meta") == "approval"]
    assert any("not on the core test npm scripts" in reason for reason in reasons)
    assert any("node option -e" in reason for reason in reasons)
    assert any("`rm` is not a worker command shape" in reason for reason in reasons)


def test_a_commit_is_judged_on_the_wire(tmp_path, monkeypatch):
    slot = pool_slot(tmp_path)
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    monkeypatch.setenv("FAKE_SLOT", str(slot))
    monkeypatch.setenv("FAKE_SCRATCH", str(tmp_path / "scratch"))
    outcome = _run_fake_turn(tmp_path, monkeypatch, "commit", timeout=15.0,
                             sandbox="workspace-write", role="worker", expect="", cwd=slot)
    assert outcome.text.startswith("commit handled")
    answers = _answers(tmp_path)
    assert {request: answers[request]["result"]["decision"]
            for request in range(9301, 9305)} == {
        9301: "accept", 9302: "decline", 9303: "decline", 9304: "decline"}
    briefing = _first_request(tmp_path, "thread/start")["params"]["developerInstructions"]
    assert "commit with tools/git/commit.py, the message file in your scratch" in briefing
