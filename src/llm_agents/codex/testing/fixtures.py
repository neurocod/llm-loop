"""pytest fixtures for tests that run a Codex turn; the one module of
`llm_agents.codex.testing` that needs pytest.

A fixture is registered by importing it into the test module that uses it
(`from llm_agents.codex.testing.fixtures import no_badge  # noqa: F401`).
"""

from __future__ import annotations

import pytest

from ... import agent_counter
from .. import paths, sandbox
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
