"""Suite-wide guards: real runner locks local to each test, and no exit record
outliving the test that opened it."""

import atexit

import pytest

from llm_loop import exitlog, scriptlock

_RECORD_BEFORE = pytest.StashKey()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    item.stash[_RECORD_BEFORE] = exitlog.current()
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Fail the test that leaves a run's exit record behind, and close it.

    A run that is not dry opens a process-wide exit record (`exitlog.begin`) and
    registers its closing line with `atexit`. A test that ran one without
    `_runfixtures.isolated_run` left it open: the record file sat in the REAL
    log dir under the home directory, every later unisolated run inherited it
    (`begin` is idempotent per process), and the suite ended with a stray
    `=== run ended: … ===` line naming whichever test got there first — measured,
    `test_git_push` alone produced one. Checked here, per test, so a leak names
    its test instead of surfacing at interpreter exit.

    A hook around teardown rather than an autouse fixture: the check is only
    true once EVERY fixture is down, `monkeypatch` included (it is what puts an
    isolated test's `exitlog._record` back), and autouse fixtures are not set up
    in declaration order — measured, a guard fixture declared first here was
    still torn down before `monkeypatch`.
    """
    result = yield
    before = item.stash.get(_RECORD_BEFORE, None)
    leaked = exitlog.current()
    if leaked is not before:
        exitlog._record = before
        if leaked is not None:
            atexit.unregister(leaked.finish)
            leaked.finish("leaked by a test")
        pytest.fail(f"{item.nodeid} left a run's exit record open — run it "
                    f"inside `_runfixtures.isolated_run`", pytrace=False)
    return result


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
