"""The commit broker (`llm_agents.git.commit_broker`), its launch rules
(`codex.command_brokers`) and the cmd refusal, under the core's test host.

The judge tests ask the escalation judge; the effect tests run the granted
command the way production does — the bridge's environment plus what the
judge hands every escalated command (`ask.environment`), the broker file of
this checkout — and assert what happened: which hooks ran, which git ran,
what was committed. Each "did not run" has its control showing that the
same fixture runs it without the broker."""

from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401 — see no_badge's docstring
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llm_agents import host_policy
from llm_agents.codex import command_brokers, sandbox
from llm_agents.codex.command_commit import BROKER
from llm_agents.codex.testing import PS
from llm_agents.git import commit_broker
from llm_agents.git.commit_identity import ENV as APPROVALS
from core_judge_support import commit_judge, git_in, pool_slot, slot_with_submodule, worker_judge
from core_policy import CORE_POLICY

# A commit on a fixture slot: about 1 s measured 2026-11-18 (a Python start
# plus a dozen git calls); the bound only keeps a hung git or hook from
# holding the suite.
_TIMEOUT_S = 120.0
RUN = f"python -I {BROKER}"
windows = pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell and .cmd lookup")
# The hooks the 2026-11-18 probe saw a granted commit run from the worker's
# hooks folder.
PLANTED = ("pre-commit", "prepare-commit-msg", "commit-msg", "post-commit",
           "post-index-change", "reference-transaction")


def _declined(ask, command, reason, **where):
    verdict = ask(command, **where)
    assert not verdict and reason in verdict.reason, (command, verdict.reason)


def _granted(ask, command, **where):
    verdict = ask(command, **where)
    assert verdict and verdict.reason == "a worker command shape", (command, verdict.reason)


def _hook(folder: Path, name: str, body: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8", newline="\n")
    path.chmod(0o755)


def _slot(tmp_path):
    """A clean pool slot whose main repository has an identity and runs
    `.githooks` — a hooks folder of the tree the worker writes — holding
    every `PLANTED` hook, each appending its name to the marker."""
    slot = pool_slot(tmp_path)
    main = tmp_path / "main"
    git_in(main, "config", "user.name", "t")
    git_in(main, "config", "user.email", "t@example.invalid")
    git_in(slot, "commit", "--quiet", "-m", "fixture deletion")
    git_in(main, "config", "core.hooksPath", ".githooks")
    marker = tmp_path / "planted.txt"
    for name in PLANTED:
        _hook(slot / ".githooks", name, f"echo {name} >> '{marker.as_posix()}'")
    return slot, marker


def _host(tmp_path, **fields):
    """The core test host with its root at `tmp_path/host`, holding a hooks
    folder whose commit-msg adds a `Host-Hook: ran` trailer."""
    root = tmp_path / "host"
    _hook(root / "hooks", "commit-msg", 'echo "Host-Hook: ran" >> "$1"')
    return host_policy.installed(replace(CORE_POLICY, **fields), at=root)


def _message(tmp_path, text="Slot work\n") -> Path:
    message = tmp_path / "scratch" / "msg.txt"
    message.parent.mkdir(parents=True, exist_ok=True)
    message.write_text(text, encoding="utf-8")
    return message


def _commit(ask, cwd: Path, *args: str, environment=None) -> subprocess.CompletedProcess:
    """The granted `python -I <BROKER> <args>` as the escalation runs it."""
    env = {**os.environ, **(ask.environment if environment is None else environment)}
    return subprocess.run([sys.executable, "-I", str(BROKER), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=_TIMEOUT_S)


def _launch(ask, cwd: Path, command: str) -> subprocess.CompletedProcess:
    """`command` as Codex starts it: PowerShell `-Command`, the bridge's
    environment plus the judge's."""
    shell = PS.PREFIX.split('"')[1]
    return subprocess.run([shell, "-Command", command], cwd=cwd,
                          env={**os.environ, **ask.environment}, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=_TIMEOUT_S)


def _head_message(repo: Path) -> str:
    return git_in(repo, "log", "-1", "--format=%B").stdout


# --- the judge ---------------------------------------------------------------

def test_a_slot_worker_commits_only_through_the_isolated_broker(tmp_path, monkeypatch):
    slot, _ = _slot(tmp_path)
    copy = slot / "src" / "llm_agents" / "git" / "commit_broker.py"
    copy.parent.mkdir(parents=True)
    copy.write_text("", encoding="utf-8")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    message = tmp_path / "scratch" / "msg.txt"
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    _granted(ask, f"python -I -B {BROKER.as_posix()} -F {message} src/a.ts")
    grant = commit_broker.Grant.from_env(ask.environment[commit_broker.ENV])
    assert grant.git.is_absolute() and grant.git.is_file()
    assert grant.hooks == grant.no_hooks and not any(grant.no_hooks.iterdir())
    for root in (slot, tmp_path / "scratch"):
        assert root.resolve() not in grant.no_hooks.parents
    for command, reason in {
        f"python {BROKER} -F {message} src/a.ts": "starts only isolated",
        f"py -3 -I {BROKER} -F {message} src/a.ts": "not through `py`",
        # the slot's copy of the broker is the worker's to edit
        f"python -I {copy} -F {message} src/a.ts": "not on the core test entrypoints",
        f"{RUN} --amend -F {message} src/a.ts": "commit.py option --amend",
    }.items():
        _declined(ask, command, reason)


def test_the_broker_needs_a_host_that_lets_workers_commit(tmp_path, monkeypatch):
    slot, _ = _slot(tmp_path)
    with host_policy.installed(replace(CORE_POLICY, commit_briefing=None)):
        ask = commit_judge(tmp_path, monkeypatch, slot)
        _declined(ask, f"{RUN} -F {tmp_path / 'scratch/m.txt'} src/a.ts",
                  "does not let workers commit")
        assert commit_broker.ENV not in ask.environment


def test_a_cmd_wrapped_escalation_is_refused(tmp_path, monkeypatch):
    """cmd looks a program up in the working directory before the PATH: a
    `python.cmd` the worker left there would run (item 2432)."""
    _, ask = worker_judge(tmp_path, monkeypatch)
    for command in ('C:\\WINDOWS\\system32\\cmd.exe /d /s /c "python -m pytest"',
                    'cmd.exe /c "npm test"'):
        verdict = ask(command, wrap=False)
        assert not verdict and "wrapped in cmd" in verdict.reason, verdict.reason
    assert ask("python -m pytest")


def test_python_is_pinned_on_the_path(tmp_path):
    """`trusted_python`: any relative PATH entry or entry inside a writable
    root refuses, wherever it stands; the first hit is the interpreter."""
    real = Path(sys.executable).parent
    root = tmp_path / "root"
    root.mkdir()
    name = "python" if os.name == "nt" else Path(sys.executable).name
    found, why = command_brokers.trusted_python(name, (root,), f"{real}")
    assert found is not None and not why, why
    for path, reason in ((f".{os.pathsep}{real}", "is relative"),
                         (f"{real}{os.pathsep}bin", "is relative"),
                         (f"{real}{os.pathsep}{root}", "lies in your writable roots"),
                         (f"{root}{os.pathsep}{real}", "lies in your writable roots"),
                         (str(tmp_path / "nothing"), f"no `{name}`")):
        found, why = command_brokers.trusted_python(name, (root,), path)
        assert found is None and reason in why, (path, why)


@windows
def test_a_script_beside_python_exe_is_not_pinned(tmp_path):
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "python.exe").write_bytes(b"")
    assert command_brokers.trusted_python("python", (), str(shim))[0] == shim / "python.exe"
    (shim / "python.cmd").write_text("", encoding="utf-8")
    found, why = command_brokers.trusted_python("python", (), str(shim))
    assert found is None and "only a lone python.exe" in why, why


def test_a_python_path_entry_in_the_roots_refuses_the_launch(tmp_path, monkeypatch):
    slot, _ = _slot(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path / 'scratch'}{os.pathsep}{os.environ['PATH']}")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    _declined(ask, f"{RUN} -F {tmp_path / 'scratch/m.txt'} src/a.ts",
              "lies in your writable roots: a `python` planted there would run")


# --- effects -----------------------------------------------------------------

def test_the_commit_runs_the_host_s_hooks_not_the_planted_ones(tmp_path, monkeypatch):
    slot, marker = _slot(tmp_path)
    with _host(tmp_path, commit_hooks="hooks"):
        ask = commit_judge(tmp_path, monkeypatch, slot)
    grant = commit_broker.Grant.from_env(ask.environment[commit_broker.ENV])
    assert grant.hooks == (tmp_path / "host" / "hooks").resolve()
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    done = _commit(ask, slot, "-F", str(message), "src/a.ts")
    assert done.returncode == 0, done.stdout + done.stderr
    assert git_in(slot, "show", "HEAD:src/a.ts").stdout == "work\n"
    assert "Host-Hook: ran" in _head_message(slot)
    assert not marker.exists(), marker.read_text(encoding="utf-8")
    # The control: the same planted hooks DO run under a plain commit here.
    git_in(slot, "commit", "--quiet", "--allow-empty", "-m", "control")
    assert "pre-commit" in marker.read_text(encoding="utf-8")


def test_a_submodule_commit_runs_no_hooks(tmp_path, monkeypatch):
    slot, sub, _ = slot_with_submodule(tmp_path)
    git_in(sub, "config", "user.name", "t")
    git_in(sub, "config", "user.email", "t@example.invalid")
    with _host(tmp_path, commit_hooks="hooks"):
        ask = commit_judge(tmp_path, monkeypatch, slot)
    (sub / "work.txt").write_text("sub work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -C {sub} -F {message} work.txt")
    done = _commit(ask, slot, "-C", str(sub), "-F", str(message), "work.txt")
    assert done.returncode == 0, done.stdout + done.stderr
    assert _head_message(sub).strip() == "Slot work"


def test_the_commit_runs_the_host_s_rules_from_its_root(tmp_path, monkeypatch):
    slot, _ = _slot(tmp_path)
    rules = tmp_path / "host" / "rules.py"
    rules.parent.mkdir(parents=True)
    rules.write_text("from llm_agents.git.commit import CommitHost\n"
                     "RULES = CommitHost(blocked_names=frozenset({'a.ts'}))\n",
                     encoding="utf-8")
    with _host(tmp_path, commit_host="rules.py:RULES"):
        ask = commit_judge(tmp_path, monkeypatch, slot)
    assert commit_broker.Grant.from_env(
        ask.environment[commit_broker.ENV]).host == f"{rules.resolve()}:RULES"
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    done = _commit(ask, slot, "-F", str(message), "src/a.ts")
    assert done.returncode == 1 and "refusing to commit sentinel" in done.stderr, (
        done.stdout + done.stderr)


def test_the_broker_commits_only_an_approved_invocation(tmp_path, monkeypatch):
    slot, _ = _slot(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    alone = {key: value for key, value in ask.environment.items() if key != APPROVALS}
    done = _commit(ask, slot, "-F", str(message), "src/a.ts", environment=alone)
    assert done.returncode == 2 and "no approval store" in done.stderr, done.stderr
    unasked = _commit(ask, slot, "-F", str(message), "src/a.ts")
    assert unasked.returncode == 1 and "no readable approval" in unasked.stderr
    nothing = _commit(ask, slot, "-F", str(message), "src/a.ts", environment={})
    assert nothing.returncode == 2 and commit_broker.ENV in nothing.stderr
    assert git_in(slot, "log", "-1", "--format=%s").stdout.strip() == "fixture deletion"


@windows
def test_the_granted_line_runs_the_pinned_git_and_python(tmp_path, monkeypatch):
    """Through PowerShell as Codex starts it: a `git.cmd`/`python.cmd` in the
    slot (the cwd) never runs, and the git the broker starts is the grant's
    — a recording wrapper named there is the one that runs, with the forced
    hooks folder on its command line."""
    slot, marker = _slot(tmp_path)
    planted = tmp_path / "cwd-planted.txt"
    for name in ("git.cmd", "git.bat", "python.cmd", "python.bat"):
        (slot / name).write_text(f"@echo {name}>>\"{planted}\"\n", encoding="utf-8")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    command = f"{RUN} -F {message} src/a.ts"
    _granted(ask, command)
    done = _launch(ask, slot, command)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not planted.exists() and not marker.exists()
    grant = commit_broker.Grant.from_env(ask.environment[commit_broker.ENV])
    log = tmp_path / "pinned.txt"
    wrapper = tmp_path / "pinned-git.cmd"
    wrapper.write_text(f"@echo %*>>\"{log}\"\n@\"{grant.git}\" %*\n", encoding="utf-8")
    (slot / "src/a.ts").write_text("more\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, command)
    pinned = {**ask.environment,
              commit_broker.ENV: replace(grant, git=wrapper).to_env()}
    again = _commit(ask, slot, "-F", str(message), "src/a.ts", environment=pinned)
    assert again.returncode == 0, again.stdout + again.stderr
    assert f"core.hooksPath={grant.no_hooks}" in log.read_text(encoding="utf-8")
    assert not planted.exists()
