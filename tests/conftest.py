"""Suite-wide guards: real runner locks local to each test, and nothing a staged
run leaves in the process — an exit record, a moved project root, a replaced
stream, an open console route — outliving the test that left it."""

import atexit
import sys
from pathlib import Path

import pytest

from llm_agents import host_policy
from llm_loop import console, exitlog, limits, projectroot, scriptlock

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


_TESTS = Path(__file__).parent
_CORE_TESTS = _TESTS / "codex_agent"


@pytest.fixture(autouse=True)
def _core_host(request):
    """Every core test (`codex_agent/`) runs under the core's own test host
    (`core_policy`); a test about the empty policy installs `EMPTY` itself.

    Here, scoped by path, and not as an autouse fixture in
    `codex_agent/conftest.py`: pytest 9.1 binds a subdirectory conftest's
    fixtures to the FIRST `Dir` node of that directory, and a FILE argument
    lying directly in an ancestor directory, given between two arguments
    below it (`codex_agent/a.py test_b.py codex_agent/c.py`), re-collects
    that ancestor without deduplication (`handle_dupes=False`,
    `_pytest/main.py:968-973` in pytest 9.1.1, `Session.collect`), so
    `codex_agent/` comes back as a new `Dir` node that gets none of them.
    Arguments that merely alternate between sibling directories reuse the
    cached nodes and are harmless. Measured 2026-11-08: under that order
    every test of the second node ran under `EMPTY`, and the one that needed
    the host's gates (`..._survives_a_failed_gate_follow_up`) read
    `gate_follow_ups == 0` with "gates not run: no host policy". `tests/`
    itself is re-collected only by a file argument directly in the project
    root; `pytest_collection_modifyitems` below refuses either case.
    """
    if not request.path.is_relative_to(_CORE_TESTS):
        yield
        return
    from core_policy import CORE_POLICY  # on `pythonpath` in pyproject.toml
    with host_policy.installed(CORE_POLICY):
        yield


_SUITE_FIXTURES = ("isolate_process_policies", "isolate_launch_decision")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    """Refuse a run in which an item under `tests/` lost this file's autouse
    fixtures — the re-collected `Dir` node of `_core_host`'s docstring.

    A hook, because hooks are found by the conftest's path, not through the
    `Dir` node that lost the fixtures, so the re-collection cannot hide it.
    Unrefused, the loss reads as a test's own failure (or, worse, a pass under
    the wrong policy). One set lookup per item.
    """
    missing = []
    for item in items:
        names = getattr(item, "fixturenames", None)
        path = getattr(item, "path", None)
        if names is None or path is None or not path.is_relative_to(_TESTS):
            continue
        need = (*_SUITE_FIXTURES, "_core_host") if path.is_relative_to(
            _CORE_TESTS) else _SUITE_FIXTURES
        if not set(need) <= set(names):
            missing.append(f"{item.nodeid} (lacks "
                           f"{', '.join(n for n in need if n not in names)})")
    if missing:
        raise pytest.UsageError(
            f"{len(missing)} test(s) collected without tests/conftest.py's "
            f"autouse fixtures: a file argument between two paths below its "
            f"directory re-collected a Dir node; reorder or split the "
            f"arguments: " + "; ".join(missing[:3]))


@pytest.fixture
def status_row(monkeypatch):
    """Factory of the handle the runners bind (`StatusApp.job`), on a disabled
    app: with no painter its posts run at once, so `snapshot()` reads each
    call's effect. `model`, when given, is the row's selector (`update`).

    Strict where the app is lenient: `Painter.apply_state` records a raising
    state call and goes on, so a mutator that raised would leave the row as it
    was and a "nothing changed" assertion green. Here that record fails the
    call (and, if a runner swallowed it, the test at teardown). Watched through
    `statusline`'s own `diaglog` name, so a test that patches `diaglog.record`
    itself still sees every record.
    """
    from llm_loop import diaglog, statusline

    raised = []

    class _StrictDiaglog:
        def __getattr__(self, name):
            return getattr(diaglog, name)

        def record(self, where, *args, **kwargs):
            diaglog.record(where, *args, **kwargs)
            if where == "Painter state call raised":
                raised.append((args, kwargs.get("exc")))
                raise AssertionError(f"a status-row state call raised: {args}"
                                     ) from kwargs.get("exc")

    monkeypatch.setattr(statusline, "diaglog", _StrictDiaglog())

    def make(model=None):
        job = statusline.StatusApp(enabled=False).job(1)
        if model is not None:
            job.update(model=model)
        return job

    yield make
    assert not raised, f"status-row state call(s) raised: {raised}"


@pytest.fixture(autouse=True)
def isolate_process_policies(monkeypatch):
    """Each test settles its own accounts' policies (`limits.process_policy`)."""
    monkeypatch.setattr(limits, "_PROCESS_POLICIES", {})


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
