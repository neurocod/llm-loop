"""The rebase broker (`llm_agents.git.rebase`) and its judge
(`command_rebase._rebase_broker`) under the core's test host, whose main
branch is `trunk` (`core_policy.CORE_POLICY`).

The judge tests ask the escalation judge; the effect tests run the granted
command the way production does — the bridge's environment plus what the
judge hands every escalated command (`ask.environment`), the broker file of
this checkout — and assert what happened in the repository: hooks did not
run, other branches did not move, a resumed rebase ran no exec line."""

from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401 — see no_badge's docstring
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llm_agents import gitlib, host_policy
from llm_agents.codex import sandbox
from llm_agents.codex.command_rebase import BROKER
from llm_agents.codex.registry import AgentRecord
from llm_agents.git import rebase as broker
from llm_agents.codex.testing import PS
from core_judge_support import (commit_judge, git_in, junction, nested_filter, pool_slot,
                                slot_with_submodule)
from core_policy import CORE_POLICY

# A broker step on a two-commit fixture: under 2 s measured 2026-11-18 (a
# Python start plus a few git calls); the bound only keeps a hung git or an
# editor that did open from holding the suite.
_TIMEOUT_S = 120.0
RUN = f"python -I {BROKER}"
windows = pytest.mark.skipif(os.name != "nt", reason="Windows junctions")


def _declined(ask, command, reason, **where):
    verdict = ask(command, **where)
    assert not verdict and reason in verdict.reason, (command, verdict.reason)
    assert sandbox.worker_decline_tail() in verdict.reason, command


def _granted(ask, command, **where):
    verdict = ask(command, **where)
    assert verdict and verdict.reason == "a worker command shape", (command, verdict.reason)


def _broker(ask, slot: Path, *args: str, environment=None) -> subprocess.CompletedProcess:
    """The granted `python <BROKER> <args>` as the escalation runs it: the
    bridge's environment plus the judge's, from the slot."""
    env = {**os.environ, **(environment or ask.environment)}
    return subprocess.run([sys.executable, "-I", str(BROKER), *args], cwd=slot, env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=_TIMEOUT_S)


def _plain_git(slot: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(slot), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=_TIMEOUT_S,
                          env={**os.environ, "GIT_EDITOR": "true"})


def _with_trunk(tmp_path, *, branch: bool = False):
    """A clean pool slot (its fixture deletion committed) and its main
    repository with an identity and a `trunk` branch at the slot's base;
    with `branch`, the slot on a branch of its own (`wt1`)."""
    slot = pool_slot(tmp_path)
    main = tmp_path / "main"
    git_in(main, "config", "user.name", "t")
    git_in(main, "config", "user.email", "t@example.invalid")
    git_in(main, "branch", "trunk")
    if branch:
        git_in(slot, "switch", "--quiet", "-c", "wt1")
    git_in(slot, "commit", "--quiet", "-m", "fixture deletion")
    return slot


def _trunk_commit(tmp_path, change) -> None:
    """One commit on trunk, made in the main checkout by `change(main)`."""
    main = tmp_path / "main"
    git_in(main, "checkout", "--quiet", "trunk")
    change(main)
    git_in(main, "commit", "--quiet", "-am", "trunk work")


def _conflict(tmp_path, monkeypatch, *, delete: bool):
    """(slot, ask) mid-rebase started by the broker: the slot's edit of
    app/src/b.ts against trunk's edit or (`delete`) trunk's delete. The
    slot's commit also adds src/new.ts, so it stays non-empty whichever
    side is kept. The main repository's `core.editor` is `false`: a
    `continue` that opened an editor would fail."""
    slot = _with_trunk(tmp_path)
    git_in(tmp_path / "main", "config", "core.editor", "false")
    (slot / "app/src/b.ts").write_text("slot\n", encoding="utf-8")
    (slot / "src/new.ts").write_text("new\n", encoding="utf-8")
    git_in(slot, "add", "app/src/b.ts", "src/new.ts")
    git_in(slot, "commit", "--quiet", "-m", "slot work")

    def change(main):
        if delete:
            git_in(main, "rm", "--quiet", "app/src/b.ts")
        else:
            (main / "app/src/b.ts").write_text("trunk\n", encoding="utf-8")
    _trunk_commit(tmp_path, change)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    _granted(ask, f"{RUN} start")
    started = _broker(ask, slot, "start")
    assert started.returncode != 0 and "CONFLICT" in started.stdout + started.stderr, (
        started.stdout + started.stderr)
    return slot, ask


# --- the judge ---------------------------------------------------------------

def test_a_slot_worker_may_run_the_broker_and_not_git(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    for command in [f"{RUN} start", f"{RUN} status", f"{RUN} diff --name-only",
                    f"{RUN} unmerged", f"python -I -B {BROKER} status",
                    f"python -I {BROKER.as_posix()} status",
                    *([f"python -I {str(BROKER).upper()} status"] if os.name == "nt" else [])]:
        _granted(ask, command)
    _granted(ask, f"{RUN} status", cwd=slot / "src")
    grant = broker.Grant.from_env(ask.environment[broker.ENV])
    assert grant.workspace == slot.resolve() and grant.main == "trunk"
    assert grant.git.is_absolute() and grant.git.is_file()
    assert grant.hooks.is_dir() and not any(grant.hooks.iterdir())
    for root in (slot, tmp_path / "scratch"):
        assert root.resolve() not in grant.hooks.parents
    for command, reason in {
        f"git -C {slot} rebase trunk": "`git` is not a worker command shape",
        "git status": "`git` is not a worker command shape",
        f"{RUN} rebase": "no operation 'rebase'",
        f"{RUN}": "with no operation",
        f"{RUN} start trunk": "start takes nothing",
        f"{RUN} start -i": "start takes nothing",
        f"{RUN} continue": "no rebase in progress",
        f"{RUN} abort": "no rebase in progress",
        f"{RUN} add src/a.ts": "only to resolve a conflict",
        f"{RUN} rm src/a.ts": "only to resolve a conflict",
        f"{RUN} diff --output=x": "only --cached",
        f"{RUN} diff --ext-diff": "only --cached",
        f"{RUN} status x": "status takes nothing",
    }.items():
        _declined(ask, command, reason)
    (slot / "src/a.ts").write_text("dirty\n", encoding="utf-8")
    _declined(ask, f"{RUN} start", "uncommitted changes in src/a.ts")


def test_the_broker_is_declined_outside_a_pool_slot(tmp_path, monkeypatch):
    work = pool_slot(tmp_path, "review-tree")
    ask = commit_judge(tmp_path, monkeypatch, work)
    _declined(ask, f"{RUN} start", "only its own pool slot")
    assert broker.ENV not in ask.environment


def test_the_broker_needs_a_host_that_names_its_main_branch(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    with host_policy.installed(replace(CORE_POLICY, main_branch="")):
        ask = commit_judge(tmp_path, monkeypatch, slot)
        _declined(ask, f"{RUN} start", "names no main branch")
        assert broker.ENV not in ask.environment
        assert "start` with escalated" not in sandbox._worktree_fact(
            slot, [tmp_path / "scratch"])


def test_a_git_inside_the_writable_roots_is_never_the_trusted_one(tmp_path, monkeypatch):
    """The bridge pins git from its own PATH (`gitlib.trusted_git`): absolute
    entries only, `git.exe` only on Windows, and never one inside the
    worker's roots — the judge then grants no broker at all."""
    slot = _with_trunk(tmp_path)
    name = "git.exe" if os.name == "nt" else "git"
    (slot / name).write_text("", encoding="utf-8")
    (slot / "git.cmd").write_text("", encoding="utf-8")
    real = Path(gitlib.trusted_git(())[0]).parent
    assert gitlib.trusted_git((slot.resolve(),), f"{slot}{os.pathsep}{real}")[0] is None
    assert gitlib.trusted_git((), f".{os.pathsep}{real}")[0] == real / name
    if os.name == "nt":
        shim = tmp_path / "shim"
        shim.mkdir()
        (shim / "git.cmd").write_text("", encoding="utf-8")
        assert gitlib.trusted_git((), f"{shim}{os.pathsep}{real}")[0] == real / name
    monkeypatch.setenv("PATH", f"{slot}{os.pathsep}{os.environ['PATH']}")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    # No grant at all; the launch is refused first, on the same PATH entry
    # (`command_brokers.trusted_python`).
    assert broker.ENV not in ask.environment
    _declined(ask, f"{RUN} start", "lies in your writable roots")


def test_a_slot_worker_is_told_the_broker_operations(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    told = sandbox.thread_params(AgentRecord(
        name="w", cwd=str(slot), sandbox="workspace-write",
        role="worker"))["developerInstructions"]
    run = f"python -I {BROKER.as_posix()}"
    for step in ("start", "unmerged", "add <paths>", "rm <paths>", "continue",
                 "abort", "skip"):
        assert f"`{run} {step}`" in told, step
    assert "`git rebase` itself is declined" in told
    reviewer = sandbox.thread_params(AgentRecord(
        name="r", cwd=str(slot), sandbox="read-only",
        role="reviewer"))["developerInstructions"]
    assert run not in reviewer


# --- effects -----------------------------------------------------------------

def _hook_fixture(tmp_path):
    """A slot one commit behind trunk whose repository runs `.githooks` (a
    hooks folder inside the tree), with untracked hooks the worker planted: each
    appends its name to `marker`."""
    slot = _with_trunk(tmp_path, branch=True)
    (slot / "src/new.ts").write_text("new\n", encoding="utf-8")
    git_in(slot, "add", "src/new.ts")
    git_in(slot, "commit", "--quiet", "-m", "slot work")
    _trunk_commit(tmp_path, lambda main: (main / "src/a.ts").write_text("t\n", encoding="utf-8"))
    git_in(tmp_path / "main", "config", "core.hooksPath", ".githooks")
    marker = tmp_path / "marker.txt"
    for hook in ("pre-rebase", "post-checkout", "post-rewrite", "post-commit"):
        path = slot / ".githooks" / hook
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"#!/bin/sh\necho {hook} >> '{marker.as_posix()}'\n",
                        encoding="utf-8", newline="\n")
        path.chmod(0o755)
    return slot, marker


def test_the_broker_runs_no_hook_the_worker_planted(tmp_path, monkeypatch):
    slot, marker = _hook_fixture(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    done = _broker(ask, slot, "start")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (slot / "src/a.ts").read_text(encoding="utf-8") == "t\n"
    assert not marker.exists(), marker.read_text(encoding="utf-8")
    # The control: the same hooks DO run under a plain git in this fixture.
    _plain_git(slot, "checkout", "--quiet", "--detach")
    assert "post-checkout" in marker.read_text(encoding="utf-8")


def test_the_broker_runs_the_pinned_git_not_one_from_the_cwd(tmp_path, monkeypatch):
    """`git.cmd`/`git.bat` in the slot (the cwd) never run, and the git the
    broker starts is the grant's: a recording wrapper named there is the one
    that runs."""
    slot, _ = _hook_fixture(tmp_path)
    planted = tmp_path / "planted.txt"
    for name in ("git.cmd", "git.bat"):
        (slot / name).write_text(f"@echo planted>>\"{planted}\"\n", encoding="utf-8")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    assert _broker(ask, slot, "status").returncode == 0
    assert not planted.exists()
    if os.name != "nt":
        return
    grant = broker.Grant.from_env(ask.environment[broker.ENV])
    log = tmp_path / "pinned.txt"
    wrapper = tmp_path / "pinned-git.cmd"
    wrapper.write_text(f"@echo %*>>\"{log}\"\n@\"{grant.git}\" %*\n", encoding="utf-8")
    pinned = {broker.ENV: replace(grant, git=wrapper).to_env()}
    done = _broker(ask, slot, "status", environment=pinned)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "core.hooksPath=" in log.read_text(encoding="utf-8")
    # Bound to the registered admin, never discovered through the slot's `.git`.
    assert grant.admin == (tmp_path / "main/.git/worktrees/main-wt1").resolve()
    assert all(f"--git-dir {grant.admin}" in line
               for line in log.read_text(encoding="utf-8").splitlines())


def test_status_and_diff_run_no_filter_a_submodule_s_git_file_names(tmp_path, monkeypatch):
    """Review 2 of 2432: the binding holds for the broker's git, not for the
    git its `status`/`diff` start per submodule, which discovers the
    worker-writable `<sub>/.git` and ran the clean filter its config names."""
    slot, sub, _ = slot_with_submodule(tmp_path)
    git_in(tmp_path / "main", "branch", "trunk")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    marker = nested_filter(tmp_path, sub)
    for request in (["status"], ["diff"], ["diff", "--stat"], ["diff", "--name-only"]):
        _granted(ask, f"{RUN} {' '.join(request)}")
        done = _broker(ask, slot, *request)
        assert done.returncode == 0, (request, done.stdout + done.stderr)
        assert not marker.exists(), request
    _plain_git(slot, "diff", "--name-only")
    assert marker.exists()  # the control: a plain diff runs it


def test_the_broker_ignores_update_refs_and_autostash_config(tmp_path, monkeypatch):
    slot, _ = _hook_fixture(tmp_path)
    git_in(slot, "branch", "side")
    main = tmp_path / "main"
    git_in(main, "config", "rebase.updateRefs", "true")
    git_in(main, "config", "rebase.autoStash", "true")
    side = git_in(slot, "rev-parse", "side").stdout
    ask = commit_judge(tmp_path, monkeypatch, slot)
    (slot / "src/a.ts").write_text("dirty\n", encoding="utf-8")
    refused = _broker(ask, slot, "start")
    assert refused.returncode == 1 and "uncommitted changes" in refused.stderr
    assert (slot / "src/a.ts").read_text(encoding="utf-8") == "dirty\n"
    git_in(slot, "checkout", "--", "src/a.ts")
    done = _broker(ask, slot, "start")
    assert done.returncode == 0, done.stdout + done.stderr
    assert git_in(slot, "rev-parse", "side").stdout == side
    # The control: plain git with that config moves `side`.
    git_in(slot, "reset", "--quiet", "--hard", "ORIG_HEAD")
    assert _plain_git(slot, "rebase", "trunk").returncode == 0
    assert git_in(slot, "rev-parse", "side").stdout != side


def test_continue_resumes_only_a_plain_rebase_the_broker_started(tmp_path, monkeypatch):
    slot, ask = _conflict(tmp_path, monkeypatch, delete=False)
    state = Path(git_in(slot, "rev-parse", "--path-format=absolute",
                        "--git-path", "rebase-merge").stdout.strip())
    marker = tmp_path / "exec.txt"
    todo = state / "git-rebase-todo"
    original = todo.read_text(encoding="utf-8")
    todo.write_text(original + f"exec echo x > '{marker.as_posix()}'\n", encoding="utf-8")
    _declined(ask, f"{RUN} continue", "only picks resume here")
    _declined(ask, f"{RUN} add app/src/b.ts", "only picks resume here")
    assert _broker(ask, slot, "continue").returncode == 1 and not marker.exists()
    todo.write_text(original, encoding="utf-8")
    (state / broker.MARKER).unlink()
    _declined(ask, f"{RUN} continue", "not started by this broker")
    _declined(ask, f"{RUN} skip", "not started by this broker")
    _granted(ask, f"{RUN} abort")
    assert _broker(ask, slot, "abort").returncode == 0
    assert not state.exists()


def test_a_worker_resolves_an_edit_conflict_with_add(tmp_path, monkeypatch):
    slot, ask = _conflict(tmp_path, monkeypatch, delete=False)
    for command, reason in {
        f"{RUN} add src/a.ts": "not an unmerged path",
        f"{RUN} add src/new.ts": "not an unmerged path",
        f"{RUN} add app/src": "not an unmerged path",
        f"{RUN} add .": "relative to the slot root",
        f"{RUN} add {slot / 'app/src/b.ts'}": "relative to the slot root",
        f"{RUN} add ./app/src/b.ts": "relative to the slot root",
        f"{RUN} add 'app/src/*.ts'": "a wildcard",
        f"{RUN} add -A": "not an unmerged path",
        f"{RUN} rm -r app/src/b.ts": "not an unmerged path",
        f"{RUN} add": "with no paths",
    }.items():
        _declined(ask, command, reason)
    _granted(ask, f"{RUN} add app/src/b.ts")
    _granted(ask, f"{RUN} add app\\src\\b.ts")
    (slot / "app/src/b.ts").write_text("resolved\n", encoding="utf-8")
    added = _broker(ask, slot, "add", "app/src/b.ts")
    assert added.returncode == 0, added.stdout + added.stderr
    done = _broker(ask, slot, "continue")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (slot / "app/src/b.ts").read_text(encoding="utf-8") == "resolved\n"
    assert git_in(slot, "log", "-1", "--format=%s").stdout.strip() == "slot work"
    _declined(ask, f"{RUN} add app/src/b.ts", "only to resolve a conflict")


def test_a_worker_resolves_a_modify_delete_conflict_with_rm(tmp_path, monkeypatch):
    slot, ask = _conflict(tmp_path, monkeypatch, delete=True)
    _declined(ask, f"{RUN} rm src/a.ts", "not an unmerged path")
    _granted(ask, f"{RUN} rm app/src/b.ts")
    _granted(ask, f"{RUN} rm --cached app/src/b.ts")
    removed = _broker(ask, slot, "rm", "app/src/b.ts")
    assert removed.returncode == 0, removed.stdout + removed.stderr
    done = _broker(ask, slot, "continue")
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (slot / "app/src/b.ts").exists() and (slot / "src/new.ts").is_file()


def test_a_symlink_alias_of_an_unmerged_file_is_refused(tmp_path, monkeypatch):
    """The judge compares the literal index name: an alias that RESOLVES to
    the unmerged file is not it (git would stage the link)."""
    slot, ask = _conflict(tmp_path, monkeypatch, delete=False)
    try:
        os.symlink(slot / "app/src/b.ts", slot / "app/src/alias.ts")
    except OSError:
        pytest.skip("no symlink privilege")
    _declined(ask, f"{RUN} add app/src/alias.ts", "not an unmerged path")
    (slot / "app/src/b.ts").unlink()
    os.symlink(slot / "src/a.ts", slot / "app/src/b.ts")
    _declined(ask, f"{RUN} add app/src/b.ts", "is a link or junction")


@windows
def test_an_unmerged_path_behind_a_junction_is_refused(tmp_path, monkeypatch):
    slot, ask = _conflict(tmp_path, monkeypatch, delete=False)
    elsewhere = tmp_path / "elsewhere"
    (slot / "app/src").rename(elsewhere)
    junction(slot / "app/src", elsewhere)
    _declined(ask, f"{RUN} add app/src/b.ts", "is a link or junction")
    assert _broker(ask, slot, "add", "app/src/b.ts").returncode == 1


def test_an_unmerged_path_replaced_by_a_folder_is_refused(tmp_path, monkeypatch):
    slot, ask = _conflict(tmp_path, monkeypatch, delete=False)
    (slot / "app/src/b.ts").unlink()
    (slot / "app/src/b.ts").mkdir()
    (slot / "app/src/b.ts/inner.ts").write_text("", encoding="utf-8")
    _declined(ask, f"{RUN} add app/src/b.ts", "not a regular file")
    _declined(ask, f"{RUN} add app/src/b.ts/inner.ts", "not an unmerged path")


def test_a_gitlink_conflict_is_refused_whatever_gitmodules_says(tmp_path, monkeypatch):
    """Both sides move the submodule's pointer to diverging commits: the
    unmerged entry has mode 160000, refused even after the worker deletes
    the `.gitmodules` stanza (`rebase._path_refusal`)."""
    slot, sub, _ = slot_with_submodule(tmp_path)
    main = tmp_path / "main"
    git_in(main, "config", "user.name", "t")
    git_in(main, "config", "user.email", "t@example.invalid")
    git_in(slot, "commit", "--quiet", "-am", "submodule")
    git_in(main, "branch", "trunk", git_in(slot, "rev-parse", "HEAD").stdout.strip())
    base = git_in(sub, "rev-parse", "HEAD").stdout.strip()
    tips = []
    for text in ("one\n", "two\n"):
        git_in(sub, "checkout", "--quiet", "--detach", base)
        (sub / "work.txt").write_text(text, encoding="utf-8")
        git_in(sub, "commit", "--quiet", "-am", text.strip())
        tips.append(git_in(sub, "rev-parse", "HEAD").stdout.strip())
    gitlink = "lib/vendor"
    git_in(slot, "update-index", "--cacheinfo", f"160000,{tips[0]},{gitlink}")
    git_in(slot, "commit", "--quiet", "-m", "slot pointer")
    git_in(main, "checkout", "--quiet", "trunk")
    git_in(main, "update-index", "--cacheinfo", f"160000,{tips[1]},{gitlink}")
    git_in(main, "commit", "--quiet", "-m", "trunk pointer")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    started = _broker(ask, slot, "start")
    assert started.returncode != 0, started.stdout + started.stderr
    modes = git_in(slot, "ls-files", "-u", "--", gitlink).stdout
    assert modes.startswith("160000"), modes
    _declined(ask, f"{RUN} add {gitlink}", "submodule's pointer")
    (slot / ".gitmodules").write_text("", encoding="utf-8")
    _declined(ask, f"{RUN} add {gitlink}", "submodule's pointer")
    environment = json.loads(ask.environment[broker.ENV])
    assert Path(environment["workspace"]) == slot.resolve()


# --- the production launch ----------------------------------------------------

def _launch(ask, slot: Path, command: str, extra=None) -> subprocess.CompletedProcess:
    """`command` started the way production does: Codex hands PowerShell the
    command as one `-Command` argument (the approval logs show it quoted,
    `PS.format`), with the bridge's environment plus the judge's
    (`protocol.AppServer`). The grant itself is asked separately, on the
    same words."""
    shell = PS.PREFIX.split('"')[1]
    env = {**os.environ, **ask.environment, **(extra or {})}
    return subprocess.run([shell, "-Command", command], cwd=slot, env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=_TIMEOUT_S)


@windows
def test_the_granted_line_starts_isolated_whatever_pythonpath_holds(tmp_path, monkeypatch):
    """A `sitecustomize.py` on an inherited PYTHONPATH in a writable folder
    runs before any check of the broker's — unless Python starts isolated,
    which the judge requires (`broker_launch_refusal`)."""
    slot = _with_trunk(tmp_path)
    planted = slot / "planted"
    planted.mkdir()
    marker = tmp_path / "sitecustomize.txt"
    (planted / "sitecustomize.py").write_text(
        f"open({str(marker)!r}, 'a').write('ran')\n", encoding="utf-8")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    poisoned = {"PYTHONPATH": str(planted)}
    granted = f"{RUN} status"
    _granted(ask, granted)
    done = _launch(ask, slot, granted, poisoned)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith("##"), done.stdout  # git status ran, not an echo
    assert not marker.exists()
    # The control: the same line without -I runs the planted file — and is
    # what the judge refuses.
    bare = f"python {BROKER} status"
    _declined(ask, bare, "starts only isolated")
    _declined(ask, f"python -E {BROKER} status", "starts only isolated")
    control = _launch(ask, slot, bare, poisoned)
    assert marker.exists(), control.stdout + control.stderr


def test_a_dotted_spelling_of_the_broker_is_refused(tmp_path, monkeypatch):
    """`..`, `.` and doubled separators read as the broker only lexically;
    what they open depends on the links along them (`_spelled` has no
    normpath)."""
    slot = _with_trunk(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    folder, name, sep = str(BROKER.parent), BROKER.name, os.sep
    for spelled in (f"{folder}{sep}..{sep}{BROKER.parent.name}{sep}{name}",
                    f"{folder}{sep}.{sep}{name}", f"{folder}{sep}{sep}{name}",
                    f"{BROKER.parent.as_posix()}/../{BROKER.parent.name}/{name}"):
        _declined(ask, f"python -I {spelled} status", "spell the rebase broker exactly")


@pytest.mark.skipif(os.name == "nt", reason="POSIX walks `..` through a symlink")
def test_a_symlink_and_dotdot_alias_of_the_broker_is_refused(tmp_path, monkeypatch):
    slot = _with_trunk(tmp_path)
    os.symlink(tmp_path / "scratch", slot / "j", target_is_directory=True)
    climb = "../" * len((slot / "j").parts[1:])
    word = f"{slot}/j/{climb}{str(BROKER).lstrip('/')}"
    assert os.path.normpath(word) == str(BROKER)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    _declined(ask, f"python -I {word} status", "spell the rebase broker exactly")


@windows
def test_an_alias_of_the_broker_through_the_tree_is_refused(tmp_path, monkeypatch):
    """A junction in the worker's tree to the bridge's folder resolves to the
    broker now, and could be swapped for the worker's own file after the
    grant: only the literal canonical spelling is the broker."""
    slot = _with_trunk(tmp_path)
    junction(slot / "bridge", BROKER.parent)
    alias = slot / "bridge" / BROKER.name
    assert alias.resolve() == BROKER
    ask = commit_judge(tmp_path, monkeypatch, slot)
    for spelled in (alias, Path("bridge") / BROKER.name):
        _declined(ask, f"python -I {spelled} status", "outside your writable roots")
    _granted(ask, f"{RUN} status")
