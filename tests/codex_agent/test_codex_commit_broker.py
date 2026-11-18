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
from llm_agents.codex.testing import PS, junction
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


def test_the_cmd_refusal_is_final_over_the_escalate_rule(tmp_path, monkeypatch):
    """The escalate regex reads the command unwrapped: without the final
    refusal it would grant `cmd /c "npm test"`, and cmd would run an npm.cmd
    planted in the cwd (review of 2432). PowerShell's spelling still passes."""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"^npm test$")
    for command in ('C:\\WINDOWS\\system32\\cmd.exe /d /s /c "npm test"',
                    'cmd.exe /c "npm test"'):
        verdict = ask(command, wrap=False)
        assert not verdict and "wrapped in cmd" in verdict.reason, verdict.reason
    verdict = ask("npm test")
    assert verdict and verdict.reason == "matches the escalate rule", verdict.reason


def test_the_broker_starts_only_as_python_typed_plainly(tmp_path, monkeypatch):
    """`argv_refusal` drops a program suffix to find the shape; the broker
    launch judges the word as typed: `python.cmd` is a script PowerShell
    finds on the PATH, not the interpreter `trusted_python` pinned."""
    slot, _ = _slot(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    message = tmp_path / "scratch" / "msg.txt"
    for program in ("python.cmd", "python.bat", "python.ps1", "python.exe",
                    "python3.cmd"):
        _declined(ask, f"{program} -I {BROKER} -F {message} src/a.ts",
                  f"not through `{program}`")
    _granted(ask, f"Python -I {BROKER} -F {message} src/a.ts")


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
def test_a_python_path_entry_through_a_junction_is_not_pinned(tmp_path):
    """A junction resolves to the real Python at the judge and is the
    worker's to swap before PowerShell walks the PATH: the entry is judged
    as spelled (review of 2432). Outside the writable roots too — the link
    itself is the risk — and through a parent folder as well as the entry."""
    real = Path(sys.executable).parent
    link = tmp_path / "python-link"
    junction(link, real)
    parent = tmp_path / "parent-link"
    junction(parent, real.parent)
    for path in (str(link), str(parent / real.name)):
        found, why = command_brokers.trusted_python("python", (), path)
        assert found is None and "link or junction" in why, (path, why)
    dotted = f"{real}{os.sep}..{os.sep}{real.name}"
    found, why = command_brokers.trusted_python("python", (), dotted)
    assert found is None and "`.`/`..` segments" in why, why
    # The control: the same folder, spelled plainly (a trailing separator too).
    for path in (str(real), f"{real}{os.sep}"):
        assert command_brokers.trusted_python("python", (), path)[0] is not None, path


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


def test_the_hooks_run_without_a_cwd_program_lookup(tmp_path, monkeypatch):
    """`gitlib.NO_CWD_LOOKUP` reaches the hooks even when the bridge's
    environment lacks it: on Windows a hook's bare-name start (Node's
    `execFileSync("git")`) would otherwise try the slot root first."""
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    slot, _ = _slot(tmp_path)
    seen = tmp_path / "seen.txt"
    # Windows names are case-blind and Python's os.environ upper-cases them;
    # the hook's sh is case-sensitive, so it reads both spellings.
    with _host(tmp_path, commit_hooks="hooks"):
        _hook(tmp_path / "host" / "hooks", "pre-commit",
              'echo "[$NoDefaultCurrentDirectoryInExePath$NODEFAULTCURRENTDIRECTORYINEXEPATH]"'
              f" > '{seen.as_posix()}'")
        ask = commit_judge(tmp_path, monkeypatch, slot)
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    done = _commit(ask, slot, "-F", str(message), "src/a.ts")
    assert done.returncode == 0, done.stdout + done.stderr
    assert seen.read_text(encoding="utf-8").strip() == "[1]"


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


# --- the repository is named, never discovered (item 2432, 2435) -------------

def _planted_gitdir(tmp_path: Path, sub: Path, marker: Path) -> Path:
    """A copy of `sub`'s repository where the worker could write it (its
    scratch) whose config names a program as fsmonitor and as the `evil`
    clean filter; the program appends to `marker` and passes its input
    through."""
    import shutil
    script = tmp_path / "scratch" / "planted.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(f"#!/bin/sh\necho ran >> '{marker.as_posix()}'\ncat\n",
                      encoding="utf-8", newline="\n")
    script.chmod(0o755)
    real = git_in(sub, "rev-parse", "--absolute-git-dir").stdout.strip()
    planted = tmp_path / "scratch" / "planted.git"
    shutil.copytree(real, planted)
    for key, value in (("core.worktree", sub.as_posix()), ("core.fsmonitor", script.as_posix()),
                       ("filter.evil.clean", script.as_posix())):
        git_in(tmp_path, "config", "--file", str(planted / "config"), key, value)
    return planted


def _point(git_file: Path, gitdir: Path) -> None:
    # Git marks the file hidden on Windows; opening it with 'w' is refused.
    with git_file.open("r+", encoding="utf-8") as stream:
        stream.write(f"gitdir: {gitdir}\n")
        stream.truncate()


def test_the_judge_never_asks_git_through_a_submodule_s_git_file(tmp_path, monkeypatch):
    """Review A: `<sub>/.git` is writable under Codex's sandbox (only each
    root's top-level `.git` is protected); flipped to a worker repository
    between the ownership check and the folder listing, a discovering
    `ls-files` started that repository's fsmonitor with the user's rights."""
    from llm_agents.codex import command_commit
    slot, sub, _ = slot_with_submodule(tmp_path)
    (sub / "src").mkdir()
    (sub / "src/b.py").write_text("", encoding="utf-8")
    marker = tmp_path / "fsmonitor.txt"
    planted = _planted_gitdir(tmp_path, sub, marker)
    listed = command_commit._commit_path_refusal

    def flipped_first(*args, **kwargs):
        _point(sub / ".git", planted)
        return listed(*args, **kwargs)

    ask = commit_judge(tmp_path, monkeypatch, slot)
    monkeypatch.setattr(command_commit, "_commit_path_refusal", flipped_first)
    verdict = ask(f"{RUN} -C {sub} -F {_message(tmp_path)} src")
    assert not marker.exists(), verdict.reason
    git_in(sub, "ls-files", "-co", "--exclude-standard")
    assert marker.exists()  # the control: discovery through the flip runs it


def test_the_approvals_live_in_the_registered_admin_not_the_git_file_s(tmp_path, monkeypatch):
    """Review B: a slot `.git` naming the worker's own `<main>/.git/
    worktrees/<slot>` (it passes `pool_slot`) must not move the approval
    store or the empty hooks folder where the worker could forge a ticket
    or plant a hook; both stay in the admin the MAIN clone registers."""
    slot, _ = _slot(tmp_path)
    fake = tmp_path / "scratch" / "main" / ".git" / "worktrees" / "main-wt1"
    fake.mkdir(parents=True)
    _point(slot / ".git", fake)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    import json
    store = Path(json.loads(ask.environment[APPROVALS])["directory"])
    real = (tmp_path / "main" / ".git" / "worktrees" / "main-wt1").resolve()
    assert real in store.parents, store
    assert real in commit_broker.Grant.from_env(
        ask.environment[commit_broker.ENV]).no_hooks.parents
    assert not any(fake.iterdir())
    # And the commit itself lands in the registered repository.
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    done = _commit(ask, slot, "-F", str(message), "src/a.ts")
    assert done.returncode == 0, done.stdout + done.stderr
    head = subprocess.run(["git", "--git-dir", str(real), "log", "-1", "--format=%s"],
                          capture_output=True, text=True, check=True).stdout
    assert head.strip() == "Slot work"


def test_no_broker_over_an_admin_directory_the_worker_may_write(tmp_path, monkeypatch):
    """Review B: a writable root holding the main clone's `.git` makes the
    registered admin the worker's too; no store, no grant."""
    slot, _ = _slot(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot, writable=[str(tmp_path / "main")])
    assert APPROVALS not in ask.environment
    _declined(ask, f"{RUN} -F {_message(tmp_path)} src/a.ts", "lies in your writable roots")


@windows
def test_every_git_after_the_ticket_is_bound_to_its_repository(tmp_path, monkeypatch):
    """Review C: the post-commit `log` and `status` (a status runs the clean
    filter `.gitattributes` names, from the config of whatever repository it
    finds) ran after the last identity check, discovering through the
    submodule's writable `.git`. A recording wrapper as the pinned git shows
    every command after the ticket carrying the approved `--git-dir`."""
    slot, sub, _ = slot_with_submodule(tmp_path)
    git_in(sub, "config", "user.name", "t")
    git_in(sub, "config", "user.email", "t@example.invalid")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    (sub / "work.txt").write_text("sub work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -C {sub} -F {message} work.txt")
    grant = commit_broker.Grant.from_env(ask.environment[commit_broker.ENV])
    log = tmp_path / "pinned.txt"
    wrapper = tmp_path / "pinned-git.cmd"
    wrapper.write_text(f"@echo %*>>\"{log}\"\n@\"{grant.git}\" %*\n", encoding="utf-8")
    pinned = {**ask.environment, commit_broker.ENV: replace(grant, git=wrapper).to_env()}
    done = _commit(ask, slot, "-C", str(sub), "-F", str(message), "work.txt",
                   environment=pinned)
    assert done.returncode == 0, done.stdout + done.stderr
    admin = (tmp_path / "main/.git/worktrees/main-wt1/modules/vendor").resolve()
    lines = log.read_text(encoding="utf-8").splitlines()
    assert any(" status " in line for line in lines), lines
    unbound = [line for line in lines if f"--git-dir {admin}" not in line]
    assert not unbound, unbound
