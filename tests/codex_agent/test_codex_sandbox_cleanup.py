"""Sandbox cleanup scenarios."""

from test_codex_agent import _no_badge  # noqa: F401 — see _no_badge docstring
from pathlib import PurePath
from llm_agents.codex import sandbox
from llm_agents.codex.registry import AgentRecord
from llm_agents.codex.testing import PS as _PS


def test_the_judge_lets_an_agent_delete_its_own_pytest_basetemp(tmp_path):
    """Under `on-request` Codex asks before `Remove-Item -Recurse` even inside
    the writable workspace; a judge that only knew the gate regex declined it
    and the worker left `.pytest-tmp` behind (2026-09-13). The grant is
    narrow: one deletion command, literal targets that resolve under the
    workspace with an allowed basename, and nothing else.

    Half the rows are about ONE question — is a `/token` a switch or a path —
    because the same words mean different things under cmd and under
    PowerShell, and the answer decides whether `rmdir /s /q .pytest-tmp` is a
    cleanup or an un-sandboxed delete of `C:\\s` and `C:\\q`."""
    work = tmp_path / "ws"
    work.mkdir()
    (work / ".pytest-tmp").mkdir()
    judge = sandbox.escalation_judge(AgentRecord(
        name="w", cwd=str(work), sandbox="workspace-write", escalate="^npm test"))
    ps = _PS

    def ask(command, cwd=None):
        return judge({"command": command, "cwd": str(cwd or work)})

    assert ask(ps.format("Remove-Item -Recurse -Force .pytest-tmp"))
    assert ask(ps.format("Remove-Item -LiteralPath .pytest-tmp -Recurse -Force "
                         "-ErrorAction SilentlyContinue -Confirm:$false"))
    assert ask(ps.format(f"Remove-Item -Recurse {work / '.pytest-tmp'}"))
    assert ask("rm -rf .pytest-tmp .pytest_cache")
    assert ask(ps.format("Remove-Item -Recurse -Force pytest-tmp"), cwd=work / "app")
    # Not granted: another directory, a wildcard, a chain, the workspace itself,
    # a path that escapes it, a cwd outside it, no target at all.
    assert not ask(ps.format("Remove-Item -Recurse -Force src"))
    assert not ask(ps.format("Remove-Item -Recurse -Force .pytest-tmp*"))
    assert not ask(ps.format("Remove-Item -Recurse .pytest-tmp; npm run build"))
    assert not ask(ps.format("Remove-Item -Recurse .pytest-tmp && echo ok"))
    assert not ask(ps.format("Remove-Item -Recurse -Force ."))
    assert not ask(ps.format(f"Remove-Item -Recurse {tmp_path / '.pytest-tmp'}"))
    assert not ask(ps.format("Remove-Item -Recurse .pytest-tmp"), cwd=tmp_path)
    assert not ask(ps.format("Remove-Item -Recurse -Force"))
    assert not ask(ps.format("Get-ChildItem .pytest-tmp"))
    # Review 2026-09-13: what PowerShell expands or evaluates before the delete
    # sees it — arrays, variables, expressions in a flag — is not a literal.
    assert not ask(ps.format("Remove-Item -Recurse src,./.pytest-tmp"))
    assert not ask(ps.format("Remove-Item -Recurse .pytest-tmp,./.pytest_cache"))
    assert not ask(ps.format('Remove-Item -Recurse "$HOME/.pytest-tmp"'))
    assert not ask(ps.format("Remove-Item -Recurse $env:USERPROFILE\\.pytest-tmp"))
    assert not ask(ps.format("Remove-Item -Recurse ~\\.pytest-tmp"))
    assert not ask(ps.format("Remove-Item -Recurse ${env:APPDATA}\\.pytest_cache"))
    assert not ask(ps.format("Remove-Item .pytest-tmp -Recurse:(./payload.ps1)"))
    assert not ask(ps.format("Remove-Item -Recurse -Include * .pytest-tmp"))
    assert not ask(ps.format("Remove-Item -Recurse -WhatIf:$false .pytest-tmp"))
    assert not ask("rm -rf %USERPROFILE%\\.pytest-tmp")
    # cmd spells its switches with a slash: `/s` used to be read as a target
    # (a path resolving to `C:\s`) and took the whole command down with it.
    cmd = 'C:\\WINDOWS\\system32\\cmd.exe /d /s /c "{}"'
    assert ask(cmd.format("rmdir /s /q .pytest-tmp"))
    assert ask(cmd.format("rd /S /Q .pytest_cache"))
    assert ask(cmd.format("del /s /q /f pytest-tmp"))
    assert ask(cmd.format("erase /s /q pytest-tmp"))
    assert ask(cmd.format("rmdir /s/q .pytest-tmp"))  # cmd takes them glued too
    # A switch is only a switch for cmd's own verbs, and only a known one.
    assert not ask(cmd.format("rmdir /s /q src"))
    assert not ask(cmd.format("rmdir /x /q .pytest-tmp"))
    assert not ask(cmd.format("rmdir /s/x .pytest-tmp"))
    assert not ask(cmd.format("del /a:r .pytest-tmp"))
    assert not ask(cmd.format("rmdir /s /q %TEMP%\\.pytest-tmp"))
    assert not ask(cmd.format("del /s C:\\"))
    # ...and it is the SHELL that says so, not the verb: all four of cmd's
    # delete words are aliases of `Remove-Item` under PowerShell, where `/s` is
    # the path `C:\s`. Judged by the verb, each of these walked out of the
    # sandbox with a grant to delete two paths nobody named (review
    # 2026-09-13).
    for slashed in ("rmdir /s /q .pytest-tmp", "rd /s /q .pytest-tmp",
                    "del /s .pytest-tmp", "erase /s .pytest-tmp"):
        assert not ask(ps.format(slashed)), slashed
        assert ask(cmd.format(slashed)), slashed
    assert not ask(ps.format("Remove-Item /s .pytest-tmp"))
    # Under PowerShell a leading slash is the start of a PATH, and a
    # root-relative one on the cwd's own drive is an ordinary target: the
    # parent judge granted it, and reading it as a switch refused it.
    rooted = "/" + PurePath(work.resolve()).as_posix().split(":/", 1)[-1].lstrip("/")
    assert ask(ps.format(f"rmdir -Recurse -Force {rooted}/.pytest-tmp"))
    assert not ask(ps.format(f"rmdir -Recurse -Force {rooted}"))
    # A path with a space arrives quoted, and quoting is the only reason it
    # arrives as one token: split on whitespace it became two targets that are
    # neither a cleanup dir nor inside the workspace.
    (work / "my dir").mkdir()
    assert ask(ps.format("Remove-Item -Recurse -Force './my dir/.pytest-tmp'"))
    assert ask(ps.format('Remove-Item -Recurse -Force ".\\my dir\\.pytest-tmp"'))
    assert not ask(ps.format("Remove-Item -Recurse -Force './my dir/src'"))
    assert not ask(ps.format("Remove-Item -Recurse -Force './my dir/.pytest-tmp"))
    # A workspace that is itself named like a cleanup dir is never a target.
    nested = tmp_path / ".pytest-tmp"
    nested.mkdir()
    judge_nested = sandbox.escalation_judge(AgentRecord(
        name="n", cwd=str(nested), sandbox="workspace-write", escalate="^npm test"))
    assert not judge_nested({"command": ps.format("Remove-Item -Recurse -Force ."),
                             "cwd": str(nested)})
    assert not judge_nested({"command": ps.format(f"Remove-Item -Recurse {nested}"),
                             "cwd": str(nested)})
    # The gate regex still works beside it.
    assert ask(ps.format("npm test"))
    assert "pytest-tmp" in sandbox._escalation_briefing("^npm test")
