"""Keep real runner locks local to each test, including concurrent test suites."""

import atexit

import pytest

from llm_loop import scriptlock


@pytest.fixture(autouse=True)
def isolate_launch_decision(tmp_path, monkeypatch):
    launches = {}
    monkeypatch.setattr(scriptlock, 'LOCK_DIR', tmp_path / 'script-locks')
    monkeypatch.setattr(scriptlock, '_launches', launches)
    yield
    for lock in launches.values():
        if lock is not None:
            atexit.unregister(lock.close)
            lock.close()
