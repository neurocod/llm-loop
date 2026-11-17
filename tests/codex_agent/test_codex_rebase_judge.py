"""The slot rebase grant (`command_rebase._git`) under the core's test host,
whose main branch is `trunk` (`core_policy.CORE_POLICY`): the shapes a
worker in a pool slot may escalate to bring its own slot up to the main
branch and resolve a conflict, the ones refused, and two real conflicts run
through the granted commands with the environment the judge hands them."""

from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401 — see no_badge's docstring
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from llm_agents import host_policy
from llm_agents.codex import sandbox
from llm_agents.codex.registry import AgentRecord
from core_judge_support import commit_judge, git_in, pool_slot, slot_with_submodule
from core_policy import CORE_POLICY

# A rebase of a two-commit fixture: well under a second when measured
# (2026-11-18); the bound only keeps an editor that did open from holding
# the suite.
_GIT_TIMEOUT_S = 60.0


def _declined(ask, command, reason, **where):
    verdict = ask(command, **where)
    assert not verdict and reason in verdict.reason, (command, verdict.reason)
    assert sandbox.worker_decline_tail() in verdict.reason, command


def _granted(ask, command, **where):
    verdict = ask(command, **where)
    assert verdict and verdict.reason == "a worker command shape", (command, verdict.reason)


def _with_trunk(tmp_path):
    """A clean pool slot (its fixture deletion committed) and its main
    repository with a `trunk` branch at the slot's base."""
    slot = pool_slot(tmp_path)
    git_in(slot, "commit", "--quiet", "-m", "fixture deletion")
    git_in(tmp_path / "main", "branch", "trunk")
    return slot


def _run(slot: Path, *args: str, environment=None) -> subprocess.CompletedProcess:
    """A granted command as the escalation would run it: git in the slot
    with the judge's environment, an identity for the commits it replays."""
    env = {key: value for key, value in os.environ.items()
           if key not in ("GIT_EDITOR", "GIT_SEQUENCE_EDITOR")}
    env.update(GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid",
               **(environment or {}))
    return subprocess.run(["git", "-C", str(slot), *args], env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=_GIT_TIMEOUT_S)


def _conflict(tmp_path, *, delete: bool):
    """A slot whose commit and trunk's disagree on one file, mid-rebase: an
    edit against an edit, or (`delete`) the slot's edit against trunk's
    delete. The slot's commit also adds `src/new.ts`, so it stays non-empty
    whichever side the worker keeps. The main repository's `core.editor` is
    `false`: `--continue` succeeds only when the judge's GIT_EDITOR wins."""
    slot = _with_trunk(tmp_path)
    main = tmp_path / "main"
    git_in(main, "config", "core.editor", "false")
    (slot / "app/src/b.ts").write_text("slot\n", encoding="utf-8")
    (slot / "src/new.ts").write_text("new\n", encoding="utf-8")
    git_in(slot, "add", "app/src/b.ts", "src/new.ts")
    git_in(slot, "commit", "--quiet", "-m", "slot work")
    git_in(main, "checkout", "--quiet", "trunk")
    if delete:
        git_in(main, "rm", "--quiet", "app/src/b.ts")
    else:
        (main / "app/src/b.ts").write_text("trunk\n", encoding="utf-8")
        git_in(main, "add", "app/src/b.ts")
    git_in(main, "commit", "--quiet", "-m", "trunk work")
    started = _run(slot, "rebase", "trunk")
    assert started.returncode != 0 and "CONFLICT" in started.stdout + started.stderr, (
        started.stdout + started.stderr)
    return slot


def test_a_worker_rebases_its_own_slot_onto_the_main_branch(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    for command in [f"git -C {slot} rebase trunk", "git rebase trunk",
                    f"git -C {slot} rebase --continue", f"git -C {slot} rebase --abort",
                    f"git -C {slot} rebase --skip", f"git -C {slot} status --short",
                    f"git -C {slot} diff --name-only --diff-filter=U",
                    f"git -C {slot} ls-files -u", f"git.exe -C {slot} rebase trunk"]:
        _granted(ask, command)
    _granted(ask, "git -C .. rebase trunk", cwd=slot / "src")
    assert ask.environment["GIT_EDITOR"] == "true"


def test_rebase_shapes_beyond_the_grant_are_declined(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    sibling = pool_slot(tmp_path, "main-wt2")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    ask = commit_judge(tmp_path, monkeypatch, slot, writable=[str(tmp_path)])
    for command, reason in {
        f"git -C {slot} rebase master": "only `rebase trunk`",
        f"git -C {slot} rebase -i trunk": "only `rebase trunk`",
        f"git -C {slot} rebase --interactive trunk": "only `rebase trunk`",
        f"git -C {slot} rebase --onto trunk HEAD": "only `rebase trunk`",
        f"git -C {slot} rebase --exec x trunk": "only `rebase trunk`",
        f"git -C {slot} rebase -x x trunk": "only `rebase trunk`",
        f"git -C {slot} rebase -X ours trunk": "only `rebase trunk`",
        f"git -C {slot} rebase --autostash trunk": "only `rebase trunk`",
        f"git -C {slot} rebase": "only `rebase trunk`",
        f"git -C {slot} rebase --continue --no-verify": "only `rebase trunk`",
        f"git -c core.editor=x -C {slot} rebase --continue": "no global options",
        f"git -C {slot} -c sequence.editor=x rebase trunk": "no global options",
        f"git -C {sibling} rebase trunk": "another slot's tree",
        f"git -C {tmp_path / 'main'} rebase trunk": "the main checkout",
        f"git -C {elsewhere} rebase trunk": "outside your own slot",
        f"git -C {slot / 'src'} rebase trunk": "name your slot's root",
        f"git -C {slot} push": "escalates only `git rebase`",
        f"git -C {slot} commit -m x": "escalates only `git rebase`",
        f"git -C {slot} reset --hard trunk": "escalates only `git rebase`",
        f"git -C {slot} add src/a.ts": "only to resolve a conflict",
        f"git -C {slot} rm src/a.ts": "only to resolve a conflict",
        f"git -C {slot} diff --output=x": "only --cached",
        f"git -C {slot} diff --ext-diff": "only --cached",
        f"git -C {slot} status ../elsewhere": "not in your slot",
        "git -C": "with no value",
    }.items():
        _declined(ask, command, reason)
    (slot / "src/a.ts").write_text("dirty\n", encoding="utf-8")
    _declined(ask, f"git -C {slot} rebase trunk", "uncommitted changes in src/a.ts")


def test_rebase_is_declined_outside_a_pool_slot(tmp_path, monkeypatch):
    work = pool_slot(tmp_path, "review-tree")
    ask = commit_judge(tmp_path, monkeypatch, work)
    _declined(ask, f"git -C {work} rebase trunk", "only its own pool slot")


def test_rebase_needs_a_host_that_names_its_main_branch(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    with host_policy.installed(replace(CORE_POLICY, main_branch="")):
        ask = commit_judge(tmp_path, monkeypatch, slot)
        _declined(ask, f"git -C {slot} rebase trunk", "names no main branch")
        record = AgentRecord(name="w", cwd=str(slot), sandbox="workspace-write",
                             role="worker")
        assert "rebase trunk" not in sandbox._worktree_fact(slot, [tmp_path / "scratch"])
        assert sandbox.commit_briefed(record)


def test_rebase_inside_a_submodule_is_declined(tmp_path, monkeypatch):
    """A slot's submodule sits on the gitlink's commit and its main branch is
    not the pool's; the orchestrator moves it (`_target_refusal`)."""
    slot, sub, _ = slot_with_submodule(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    _declined(ask, f"git -C {sub} rebase trunk", "a submodule")
    _declined(ask, "git rebase trunk", "a submodule", cwd=sub)


def test_a_worker_resolves_an_edit_conflict_with_add(tmp_path, monkeypatch):
    slot = _conflict(tmp_path, delete=False)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    for command, reason in {
        f"git -C {slot} add src/a.ts": "not an unmerged path",
        f"git -C {slot} add src/new.ts": "not an unmerged path",
        f"git -C {slot} add app/src": "not an unmerged path",
        f"git -C {slot} add .": "the whole tree",
        f"git -C {slot} add -A": "no options",
        f"git -C {slot} add -f app/src/b.ts": "no options",
        f"git -C {slot} add 'app/src/*.ts'": "wildcard",
        f"git -C {slot} add lib/vendor": "submodule",
        f"git -C {slot} rm -r app/src/b.ts": "no options but --cached",
        f"git -C {slot} add": "with no paths",
    }.items():
        _declined(ask, command, reason)
    _granted(ask, f"git -C {slot} add app/src/b.ts")
    _granted(ask, f"git -C {slot} add -- app/src/b.ts")
    _granted(ask, "git add b.ts", cwd=slot / "app/src")
    (slot / "app/src/b.ts").write_text("resolved\n", encoding="utf-8")
    assert _run(slot, "add", "app/src/b.ts", environment=ask.environment).returncode == 0
    refused = _run(slot, "rebase", "--continue")
    assert refused.returncode != 0, "core.editor=false should fail without the judge's env"
    done = _run(slot, "rebase", "--continue", environment=ask.environment)
    assert done.returncode == 0, done.stdout + done.stderr
    assert (slot / "app/src/b.ts").read_text(encoding="utf-8") == "resolved\n"
    _declined(ask, f"git -C {slot} add app/src/b.ts", "only to resolve a conflict")


def test_a_worker_resolves_a_modify_delete_conflict_with_rm(tmp_path, monkeypatch):
    slot = _conflict(tmp_path, delete=True)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    _declined(ask, f"git -C {slot} rm src/a.ts", "not an unmerged path")
    _granted(ask, f"git -C {slot} rm app/src/b.ts")
    _granted(ask, f"git -C {slot} rm --cached app/src/b.ts")
    removed = _run(slot, "rm", "--quiet", "app/src/b.ts", environment=ask.environment)
    assert removed.returncode == 0, removed.stdout + removed.stderr
    done = _run(slot, "rebase", "--continue", environment=ask.environment)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (slot / "app/src/b.ts").exists() and (slot / "src/new.ts").is_file()


def test_a_slot_worker_is_told_the_rebase_shapes(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    told = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(slot), sandbox="workspace-write",
        role="worker"))["developerInstructions"]
    spelled = slot.resolve().as_posix()
    assert f"`git -C {spelled} rebase trunk`" in told
    assert f"`git -C {spelled} rm <paths>` for a modify/delete conflict" in told
    assert "rebase --continue` (no editor opens)" in told
    assert "no `-i`, `--onto`, `--exec`, `-c`" in told
    reviewer = sandbox.thread_params(AgentRecord(
        name="r", cwd=str(slot), sandbox="read-only",
        role="reviewer"))["developerInstructions"]
    assert "rebase trunk" not in reviewer


@pytest.mark.parametrize("step", ["--continue", "--abort", "--skip"])
def test_rebase_steps_take_nothing_more(tmp_path, monkeypatch, step):
    slot = _with_trunk(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    _declined(ask, f"git -C {slot} rebase {step} trunk", "only `rebase trunk`")
