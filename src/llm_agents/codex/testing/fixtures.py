"""pytest fixtures for tests that run a Codex turn; the one module of
`llm_agents.codex.testing` that needs pytest.

A fixture is registered by importing it into the test module that uses it
(`from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401`).
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from ... import agent_counter
from .. import command_brokers, paths, sandbox
from ..registry import AgentRecord


@pytest.fixture(autouse=True)
def no_badge(monkeypatch):
    """Keep the on-screen counter out of every test that runs a turn.

    `run_turn` announces itself to `AgentCounter`, which would start a window on
    the operator's screen once per turn test. Autouse rather than per-test: the
    announcement is inside `run_turn`, so every future test that runs a turn
    inherits it without having to know the badge exists. The tests that are
    ABOUT the beacon undo it deliberately.

    Each test module that runs a turn must import this fixture to register it
    with pytest. The import LOOKS unused (2026-09-13), but removing it allows
    turn tests to open real badge windows. Keep registration explicit rather
    than widening its scope to every test through conftest.py.
    """
    monkeypatch.setenv(agent_counter.DISABLE_ENV, "0")
    monkeypatch.delenv("CODEX_AGENTS_COUNTER", raising=False)


def bridge_path_folders(shim: Path | None = None) -> list[str]:
    """The folders of `bridge_path`: `shim` first (off Windows), then where
    this Python, git and (off Windows) `sh`, `cat` and `env` lie, each
    resolved, so no entry passes through a link; on Windows the system
    folders PowerShell runs from."""
    programs = [sys.executable, shutil.which("git")]
    if os.name == "nt":
        system = Path(os.environ.get("SystemRoot", r"C:\Windows"))
        extra = [system / "System32", system, system / "System32" / "WindowsPowerShell" / "v1.0"]
    else:
        programs += [shutil.which(name) for name in ("sh", "cat", "env")]
        extra = []
    folders = ([shim] if shim is not None else []) + [
        Path(program).parent.resolve() for program in programs if program] + extra
    return list(dict.fromkeys(str(folder) for folder in folders))


def _python_shim() -> Optional[Path]:
    """Off Windows, a folder holding `python` and `python3` as plain scripts
    that exec this interpreter, removed at exit; None on Windows. Made at
    import so that test modules can spell `BROKER_PYTHON` in constants."""
    if os.name == "nt":
        return None
    folder = Path(tempfile.mkdtemp(prefix="bridge-python-")).resolve()
    atexit.register(shutil.rmtree, folder, True)
    for name in ("python", "python3"):
        script = folder / name
        script.write_bytes(f"#!/bin/sh\nexec '{sys.executable}' \"$@\"\n".encode("utf-8"))
        script.chmod(0o755)
    return folder


_SHIM = _python_shim()
# The word a broker launch starts with under `bridge_path`
# (`command_brokers.launch_word`): `python` on Windows, the shim's absolute
# `python3` elsewhere.
BROKER_PYTHON = "python" if _SHIM is None else str(_SHIM / "python3")


@pytest.fixture(autouse=True)
def bridge_path(monkeypatch):
    """PATH = `bridge_path_folders()` for every test of a module that imports
    this fixture.

    The broker launch judge (`command_brokers.trusted_python`) judges EVERY
    PATH entry of the bridge, and a verdict read from the test machine's own
    PATH tests that machine's image, not the judge: a linked tool cache on a
    macOS runner refused every broker grant there (CI 2026-11-18). A test
    about one PATH entry prepends it to this PATH.

    Off Windows the broker needs an interpreter root alone may change
    (`command_brokers._root_chain`), which no CI runner's Python is and no
    test can make: the shim, its scripts and the folders above it stand in
    for root's (`_fixed` patched for those paths only), the way the chain
    tests stand their own trees in."""
    monkeypatch.setenv("PATH", os.pathsep.join(bridge_path_folders(_SHIM)))
    monkeypatch.setenv("FAKE_BROKER_PYTHON", BROKER_PYTHON)
    if _SHIM is None:
        return
    original = command_brokers._fixed
    trusted = {_SHIM, *_SHIM.parents}

    def fixed(path: Path, status: os.stat_result) -> bool:
        return path in trusted or path.parent == _SHIM or original(path, status)

    monkeypatch.setattr(command_brokers, "_fixed", fixed)


@pytest.fixture
def worker_report(tmp_path, monkeypatch):
    """(record, report path): a workspace-write worker "w" in `tmp_path`, its
    scratch under it, and the path its report file goes to handed to the
    stand-in as FAKE_REPORT_PATH."""
    monkeypatch.setenv(paths.STATE_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(sandbox, "scratch_dir", lambda name: tmp_path / "scratch")
    record = AgentRecord(name="w", cwd=str(tmp_path),
                         sandbox="workspace-write", role="worker")
    report = sandbox.report_path(record)
    monkeypatch.setenv("FAKE_REPORT_PATH", str(report))
    return record, report
