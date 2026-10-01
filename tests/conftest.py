"""Suite-wide guards: real runner locks local to each test, and nothing a staged
run leaves in the process — an exit record, a moved project root, a replaced
stream, an open console route — outliving the test that left it."""

import atexit
import sys

import pytest

from llm_loop import console, exitlog, projectroot, scriptlock

from _runfixtures import finish_record

# `pytester` is what pins the guard below (test_leak_guard): the guard is only
# worth its name if a test that leaks is shown failing under it.
pytest_plugins = ["pytester"]

_BEFORE = pytest.StashKey()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_setup(item):
    """Take the process state the test must hand back, and refuse an open record.

    The record is compared to None, never to what an earlier test left: `begin`
    is idempotent per process, so once one leak is taken as the baseline every
    later run REUSES that record and each further leak compares equal to it —
    measured, a leak after a teardown that raised went unreported. A record open
    here was left outside any test's teardown (an import at collection, a session
    fixture); it is closed and this test fails naming it.
    """
    stale = finish_record("leaked before a test began")
    item.stash[_BEFORE] = (projectroot.project_dir(), sys.stdout, sys.stderr,
                           console._route)
    if stale is not None:
        exitlog._record = None
        pytest.fail(f"an exit record was already open when {item.nodeid} "
                    f"began: {stale.path}", pytrace=False)
    return (yield)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item, nextitem):
    """Fail the test that leaves a staged run's process-wide state behind, and
    put that state back.

    A run that is not dry opens a process-wide exit record (`exitlog.begin`),
    registers its closing line with `atexit`, raises the tee over `sys.stdout` /
    `sys.stderr`, and every run — a dry one too — anchors the project root. A
    test that ran one without `_runfixtures.isolated_run` left all of it: the
    record file sat in the REAL log dir under the home directory, every later
    unisolated run inherited it (`begin` is idempotent per process), and the
    suite ended with a stray `=== run ended: … ===` line naming whichever test
    got there first — measured, `test_git_push` alone produced one; a moved root
    points the next test's stop file, mirror log and git at the leaking test's
    tmp dir. Checked here, per test, so a leak names its test instead of
    surfacing at interpreter exit or as another test's failure.

    The streams are compared for `-s` runs: under capture, pytest's own suspend
    between phases already puts `sys.stdout` back, which hides such a leak
    rather than curing it.

    A hook around teardown rather than an autouse fixture: the check is only
    true once EVERY fixture is down, `monkeypatch` included (it is what puts an
    isolated test's `exitlog._record` back), and autouse fixtures are not set up
    in declaration order — measured, a guard fixture declared first here was
    still torn down before `monkeypatch`. The state is put back even when a
    fixture's teardown raised: skipped then, the leak would become the next
    test's starting state.
    """
    try:
        result = yield
    except BaseException as exc:
        leaks = _put_back(item)
        if leaks and hasattr(exc, "add_note"):      # Python 3.11+
            exc.add_note(_leak_message(item, leaks))
        raise
    leaks = _put_back(item)
    if leaks:
        pytest.fail(_leak_message(item, leaks), pytrace=False)
    return result


def _put_back(item) -> list:
    """Restore what the test changed; return a description of each change."""
    leaks = []
    if finish_record("leaked by a test") is not None:
        exitlog._record = None
        leaks.append("an exit record open")
    root, out, err, route = item.stash.get(_BEFORE, (None, None, None, None))
    if root is not None and projectroot.project_dir() != root:
        leaks.append(f"the project root moved to {projectroot.project_dir()}")
        projectroot.set_project_root(root)
    if out is not None and sys.stdout is not out:
        leaks.append(f"sys.stdout replaced by {sys.stdout!r}")
        sys.stdout = out
    if err is not None and sys.stderr is not err:
        leaks.append(f"sys.stderr replaced by {sys.stderr!r}")
        sys.stderr = err
    # Every later run would refuse to open its own (`console.route_through`).
    if console._route is not route:
        leaks.append(f"the console routed through {console._route.owner.name!r}"
                     if console._route is not None else "the console route removed")
        console._route = route
    return leaks


def _leak_message(item, leaks) -> str:
    return (f"{item.nodeid} left {'; '.join(leaks)} — run it inside "
            f"`_runfixtures.isolated_run`")


@pytest.fixture(autouse=True)
def isolate_launch_decision(tmp_path, monkeypatch):
    launches = {}
    monkeypatch.setattr(scriptlock, 'LOCK_DIR', tmp_path / 'script-locks')
    monkeypatch.setattr(scriptlock, '_launches', launches)
    monkeypatch.setattr(scriptlock, '_launch_decision', None)
    yield
    for lock in launches.values():
        if lock is not None:
            atexit.unregister(lock.close)
            lock.close()
