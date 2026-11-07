"""Throwaway repositories and a CLI runner for the git kit's tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"


def git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.name=Kit Test", "-c", "user.email=kit@example.invalid",
         "-c", "commit.gpgsign=false", *args], cwd=repo, text=True,
        encoding="utf-8", capture_output=True, check=True)


def out(repo: Path, *args: str) -> str:
    return git(repo, *args).stdout.strip()


def repo(tmp_path: Path, name: str = "repo") -> Path:
    path = tmp_path / name
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.name", "Commit Helper Test")
    git(path, "config", "user.email", "commit-helper@example.invalid")
    return path


def kit_env(**extra: str) -> dict:
    """The environment of a kit script run as `python -m`: the package's
    `src/` first, an inherited commit approval dropped (these repositories
    belong to no worker)."""
    env = {**os.environ, **extra}
    env.pop("CODEX_COMMIT_APPROVAL_STORE", None)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])])
    env.update(extra)
    return env


def run_kit(module: str, *args: str, cwd: Path, env: dict | None = None
            ) -> subprocess.CompletedProcess:
    """`python -m llm_agents.git.<module> <args>` in `cwd`."""
    return subprocess.run(
        [sys.executable, "-m", f"llm_agents.git.{module}", *args], cwd=cwd,
        text=True, encoding="utf-8", capture_output=True,
        env=env if env is not None else kit_env())
