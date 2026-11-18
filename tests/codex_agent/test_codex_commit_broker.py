"""The commit broker (`llm_agents.git.commit_broker`), its launch rules
(`codex.command_brokers`) and the cmd refusal, under the core's test host.

The judge tests ask the escalation judge; the effect tests run the granted
command the way production does — the bridge's environment plus what the
judge hands every escalated command (`ask.environment`), the broker file of
this checkout — and assert what happened: which hooks ran, which git ran,
what was committed. Each "did not run" has its control showing that the
same fixture runs it without the broker."""

from llm_agents.codex.testing.fixtures import bridge_path, no_badge  # noqa: F401 — see their docstrings
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
from core_judge_support import (commit_judge, git_in, nested_filter, planted_gitdir, point,
                                pool_slot, slot_with_submodule, worker_judge)
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
    path.write_bytes(f"#!/bin/sh\n{body}\n".encode("utf-8"))
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
    message = _message(tmp_path)
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


def test_a_cmd_inside_powershell_is_refused_over_the_escalate_rule(tmp_path, monkeypatch):
    """Review 2 of 2432: the final refusal saw only Codex's wrapper; a `cmd`
    the worker typed inside PowerShell's reached the unanchored escalate
    rule as `cmd /c npm test` and was granted. (The rule now matches each
    command whole, so the arguments it grants are its own words.)"""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"npm test( .*)?")
    for command in ("cmd /c npm test", "cmd.exe /c 'npm test'", "& cmd /k npm test",
                    "C:\\Windows\\System32\\CMD.EXE /d /c npm test",
                    "powershell -Command 'cmd /c npm test'"):
        verdict = ask(command)
        assert not verdict, (command, verdict.reason)
    for command in ("npm test", "npm test -- tools/cmd/x.test.ts"):
        verdict = ask(command)
        assert verdict and verdict.reason == "matches the escalate rule", (command, verdict.reason)


@pytest.mark.parametrize("role", ["worker", "reviewer"])
def test_an_indirect_invocation_is_refused_over_the_escalate_rule(tmp_path, monkeypatch, role):
    """Round 3 of 2432 (Codex): `& $env:ComSpec /c 'npm test'` names no
    `cmd` word, so the final cmd refusal missed it and an unanchored rule
    granted cmd running whatever npm.cmd the cwd holds. The rule now reads
    literal command lines only (`command_escalate.escalate_refusal`)."""
    _, ask = worker_judge(tmp_path, monkeypatch, escalate=r"npm test", role=role)
    for command in ("& $env:ComSpec /c 'npm test'", "$c = 'npm test'; & $c",
                    "Invoke-Expression 'npm test'", "iex 'npm test'",
                    "Start-Process npm -ArgumentList 'npm test'",
                    "pwsh -Command 'npm test'", "bash -c 'npm test'",
                    "echo x | iex 'npm test'", ". ./setup.ps1; npm test",
                    "& (Get-Command npm) test; npm test"):
        verdict = ask(command)
        assert not verdict, (command, verdict.reason)
    for command in ("npm test", "npm test 2>&1 | Select-Object -Last 40"):
        verdict = ask(command)
        assert verdict and verdict.reason == "matches the escalate rule", (command, verdict.reason)


def test_the_broker_starts_only_as_python_typed_plainly(tmp_path, monkeypatch):
    """`argv_refusal` drops a program suffix to find the shape; the broker
    launch judges the word as typed: `python.cmd` is a script PowerShell
    finds on the PATH, not the interpreter `trusted_python` pinned."""
    slot, _ = _slot(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    message = _message(tmp_path)
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


def test_a_doubled_separator_inside_a_python_path_entry_is_one_separator(tmp_path):
    """Only a LEADING doubled separator (a UNC share, POSIX `//`) refuses:
    inside an entry it names no link (a Windows runner's `C:\\\\ghcup\\bin`)."""
    real = Path(sys.executable).parent.resolve()
    name = "python" if os.name == "nt" else Path(sys.executable).name
    doubled = f"{real.parent}{os.sep}{os.sep}{real.name}"
    found, why = command_brokers.trusted_python(name, (), doubled)
    assert found is not None and not why, (doubled, why)
    leading = f"{os.sep}{os.sep}{str(real).lstrip(os.sep)}"
    found, why = command_brokers.trusted_python(name, (), leading)
    assert found is None and "starts with a doubled separator" in why, (leading, why)


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0,
                    reason="POSIX symlinks under a folder the user may not write")
def test_a_python_path_link_only_its_user_could_not_repoint_is_pinned(tmp_path):
    """`/sbin -> usr/sbin` under a root-owned `/` (every usr-merged Linux):
    with the user's rights the worker cannot repoint it, so the PATH entry
    through it pins; the same link in a folder the user writes refuses, and
    so does a fixed link whose target passes a writable one."""
    real = Path(sys.executable).parent.resolve()
    name = Path(sys.executable).name
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    (fixed / "bin").symlink_to(real)
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "bin").symlink_to(real)
    (fixed / "via").symlink_to(loose / "bin")
    fixed.chmod(0o555)
    try:
        found, why = command_brokers.trusted_python(name, (), str(fixed / "bin"))
        assert found is not None and not why, why
        for entry in (loose / "bin", fixed / "via"):
            found, why = command_brokers.trusted_python(name, (), str(entry))
            assert found is None and "link or junction" in why, (entry, why)
    finally:
        fixed.chmod(0o755)


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


def test_the_sandbox_temp_folder_counts_as_a_writable_root(tmp_path, monkeypatch):
    """Review 2 of 2432: the worker may write its temp folder beside its
    roots, so a `python` on the PATH there is the worker's. That folder is
    the scratch root the policy grants, never the bridge's own TEMP/TMP
    (2470: Codex's grant of those is excluded)."""
    from llm_agents.codex import paths
    from llm_agents.codex.registry import AgentRecord
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(tmp_path / "t1"))
    writer = AgentRecord(name="w", cwd=str(tmp_path), sandbox="workspace-write")
    assert sandbox.sandbox_temp_dirs(writer) == [paths.scratch_dir("w")]
    assert sandbox.sandbox_temp_dirs(replace(writer, sandbox="read-only")) == []
    slot, _ = _slot(tmp_path)
    (tmp_path / "temp").mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", f"{tmp_path / 'temp'}{os.pathsep}{os.environ['PATH']}")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    _declined(ask, f"{RUN} -F {_message(tmp_path)} src/a.ts",
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
    planted = planted_gitdir(tmp_path, sub, marker)
    listed = command_commit._commit_path_refusal

    def flipped_first(*args, **kwargs):
        point(sub / ".git", planted)
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
    point(slot / ".git", fake)
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


def test_no_broker_over_a_writable_root_inside_the_slot_s_repository(tmp_path, monkeypatch):
    """Review 2 of 2432: the other direction — a `--writable` folder inside
    the main clone's `.git` (a submodule's admin, its config) is the
    worker's to write; no store, no grant."""
    slot, _ = _slot(tmp_path)
    inside = tmp_path / "main/.git/worktrees/main-wt1/modules/vendor"
    inside.mkdir(parents=True)
    ask = commit_judge(tmp_path, monkeypatch, slot, writable=[str(inside)])
    assert APPROVALS not in ask.environment
    _declined(ask, f"{RUN} -F {_message(tmp_path)} src/a.ts", "lies in the slot's repository")


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


@windows
def test_the_commit_takes_the_approved_message_not_a_re_pointed_file(tmp_path, monkeypatch):
    """Review 2 of 2432: after the approval the worker swaps its message
    folder for a junction to a folder outside its writable roots. The
    broker commits the text the judge read, and neither reads nor deletes
    the file the junction now reaches."""
    import shutil
    slot, _ = _slot(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    folder = tmp_path / "scratch" / "messages"
    folder.mkdir(parents=True)
    note = folder / "note.txt"
    note.write_text("Approved text\n", encoding="utf-8")
    _granted(ask, f"{RUN} -F {note} src/a.ts")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "note.txt").write_text("The user's file\n", encoding="utf-8")
    shutil.rmtree(folder)
    junction(folder, outside)
    done = _commit(ask, slot, "-F", str(note), "src/a.ts")
    assert done.returncode == 0, done.stdout + done.stderr
    assert _head_message(slot).strip() == "Approved text"
    assert (outside / "note.txt").read_text(encoding="utf-8") == "The user's file\n"
    assert list(outside.iterdir()) == [outside / "note.txt"]


def test_a_commit_runs_no_filter_a_submodule_s_git_file_names(tmp_path, monkeypatch):
    """Review 2 of 2432: binding the outer git does not bind the git its
    `status` starts per submodule, which discovers `<sub>/.git` — the
    worker's to point at a repository with a clean filter of its own. A
    commit of an unrelated slot file must not run it."""
    slot, sub, _ = slot_with_submodule(tmp_path)
    main = tmp_path / "main"
    git_in(main, "config", "user.name", "t")
    git_in(main, "config", "user.email", "t@example.invalid")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    marker = nested_filter(tmp_path, sub)
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    done = _commit(ask, slot, "-F", str(message), "src/a.ts")
    assert done.returncode == 0, done.stdout + done.stderr
    assert git_in(slot, "show", "HEAD:src/a.ts").stdout == "work\n"
    assert not marker.exists(), done.stdout + done.stderr
    git_in(slot, "status", "--short")
    assert marker.exists()  # the control: a plain status runs it


def test_a_commit_of_an_unchanged_path_runs_no_submodule_filter(tmp_path, monkeypatch):
    """Round 3 of 2432: git answers "nothing to commit" with a full status,
    `-F` or not, and a worker's `.gitmodules` `ignore = none` beats the
    forced `diff.ignoreSubmodules=all` there. The broker refuses before git
    runs (`commit.nothing_staged`)."""
    slot, sub, _ = slot_with_submodule(tmp_path)
    main = tmp_path / "main"
    git_in(main, "config", "user.name", "t")
    git_in(main, "config", "user.email", "t@example.invalid")
    git_in(slot, "commit", "--quiet", "-m", "fixture")
    ask = commit_judge(tmp_path, monkeypatch, slot)
    git_in(slot, "config", "--file", ".gitmodules", "submodule.vendor.ignore", "none")
    git_in(slot, "commit", "--quiet", "-m", "worker's gitmodules", "--", ".gitmodules")
    marker = nested_filter(tmp_path, sub)
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    done = _commit(ask, slot, "-F", str(message), "src/a.ts")
    assert done.returncode == 1 and "nothing to commit" in done.stderr, done.stdout + done.stderr
    assert not marker.exists(), done.stdout + done.stderr
    # The control: the forced settings alone do not stop git's own answer.
    plain = subprocess.run(
        ["git", "-c", "diff.ignoreSubmodules=all", "-c", "commit.status=false", "-C", str(slot),
         "commit", "-m", "x", "--", "src/a.ts"], capture_output=True, text=True, timeout=_TIMEOUT_S)
    assert plain.returncode == 1, plain.stdout + plain.stderr
    assert marker.exists()


def test_the_message_reaches_git_on_stdin_not_through_a_temp_file(tmp_path, monkeypatch):
    """Round 3 of 2432: the broker wrote the approved message to a file in
    %TEMP% — a root the worker writes — and closed it before git read it,
    so a racing worker could change the committed text. The pinned git is
    a recording wrapper: `commit -F -`, and no file appears in TEMP."""
    if os.name != "nt":
        pytest.skip("a .cmd wrapper")
    slot, _ = _slot(tmp_path)
    ask = commit_judge(tmp_path, monkeypatch, slot)
    grant = commit_broker.Grant.from_env(ask.environment[commit_broker.ENV])
    log = tmp_path / "pinned.txt"
    wrapper = tmp_path / "pinned-git.cmd"
    wrapper.write_text(f"@echo %*>>\"{log}\"\n@\"{grant.git}\" %*\n", encoding="utf-8")
    temp = tmp_path / "broker-temp"
    temp.mkdir()
    env = {**ask.environment, commit_broker.ENV: replace(grant, git=wrapper).to_env(),
           "TEMP": str(temp), "TMP": str(temp)}
    (slot / "src/a.ts").write_text("work\n", encoding="utf-8")
    message = _message(tmp_path)
    _granted(ask, f"{RUN} -F {message} src/a.ts")
    done = _commit(ask, slot, "-F", str(message), "src/a.ts", environment=env)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _head_message(slot).strip() == "Slot work"
    commits = [line for line in log.read_text(encoding="utf-8").splitlines()
               if " commit " in f" {line} "]
    assert commits and all(" -F - " in line for line in commits), commits
    assert list(temp.iterdir()) == []
