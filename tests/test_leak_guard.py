"""The suite's own guard (`conftest.py`) fails the test that leaks, every time.

Each pin runs a small suite under a copy of the real `conftest.py` and reads its
outcomes. The guard is the only thing standing between a staged run and the
operator's real log dir, so a guard that stops catching leaks has to go red
somewhere — it cannot pin itself from inside the suite it guards.
"""

from pathlib import Path

_CONFTEST = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")

_LEAKY = '''
import io
import sys

import pytest

from llm_loop import exitlog, projectroot


@pytest.fixture
def teardown_breaks():
    yield
    raise RuntimeError("a fixture's teardown broke")


def test_leaks_a_record_and_its_teardown_raises(teardown_breaks, tmp_path):
    exitlog.begin("pytest-guard", tmp_path, "first")


def test_leaks_nothing():
    pass


def test_leaks_a_record_plainly(tmp_path):
    exitlog.begin("pytest-guard", tmp_path, "second")


def test_moves_the_project_root(tmp_path):
    projectroot.set_project_root(str(tmp_path))


def test_replaces_stdout():
    sys.stdout = io.StringIO()
'''


def _run(pytester, source):
    pytester.makeconftest(_CONFTEST)
    pytester.makepyfile(test_inner=source)
    # `-s`: under capture pytest itself puts `sys.stdout` back between phases,
    # and the stream pin would pass whatever the guard did.
    return pytester.runpytest("-s", "-p", "no:cacheprovider")


def test_each_leak_fails_its_own_test_even_after_a_teardown_that_raised(
        pytester):
    """The leak after the raising teardown is the one that used to go missing:
    the raise skipped the check, the record stayed open, and the next leak's
    run reused it (`exitlog.begin` is idempotent per process), so it compared
    equal to the "before" the guard had taken."""
    result = _run(pytester, _LEAKY)

    result.assert_outcomes(passed=5, errors=4)
    result.stdout.fnmatch_lines_random([
        "*ERROR at teardown of test_leaks_a_record_and_its_teardown_raises*",
        "*ERROR at teardown of test_leaks_a_record_plainly*",
        "*ERROR at teardown of test_moves_the_project_root*",
        "*ERROR at teardown of test_replaces_stdout*",
        "*test_leaks_a_record_plainly left an exit record open*",
        "*test_moves_the_project_root left the project root moved to*",
        "*test_replaces_stdout left sys.stdout replaced by*",
    ])


def test_a_record_open_before_a_test_begins_fails_that_test(pytester):
    """Opened outside any test's protocol (here: at import, during collection),
    so no teardown saw it; setup refuses it instead of taking it as the
    baseline."""
    result = _run(pytester, '''
from llm_loop import exitlog

exitlog.begin("pytest-guard", ".", "at-import")     # the pytester dir: cwd


def test_innocent():
    pass
''')

    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines([
        "*an exit record was already open when test_inner.py::test_innocent*"])
