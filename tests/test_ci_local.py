"""scripts/ci_local.py must read the real workflow, not a remembered copy of it.

Its parser is deliberately not YAML: it knows the two shapes tests.yml uses.
These pins are what turns an edit to the workflow that leaves those shapes into
a red test here, rather than a local run that quietly covers less than CI.
"""

import importlib.util
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "ci_local", os.path.join(ROOT, "scripts", "ci_local.py"))
ci_local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_local)


def test_the_real_workflow_parses_to_the_declared_floor_and_both_test_steps():
    with open(ci_local.WORKFLOW, encoding="utf-8") as handle:
        plan = ci_local.parse_workflow(handle.read())
    assert plan.versions[0] == "3.9"  # pyproject's requires-python floor
    commands = [c for step in plan.steps for c in step.commands]
    assert any("pip install" in c for c in commands)
    assert any("pytest" in c for c in commands)
    # The `|` block: both plugin self-tests, one command each.
    assert any(c.endswith("ask_user_gate.py --self-test") for c in commands)
    assert any(c.endswith("try_patch.py --selftest") for c in commands)


def test_inline_and_block_runs_keep_their_order_and_names():
    plan = ci_local.parse_workflow(
        "    strategy:\n"
        "      matrix:\n"
        "        python-version: [\"3.9\", '3.10', 3.11]\n"
        "    steps:\n"
        "      - uses: actions/checkout@v4\n"
        "      - name: One\n"
        "        run: echo one\n"
        "      - name: Two\n"
        "        run: |\n"
        "          echo a\n"
        "\n"
        "          echo b\n"
        "      - name: Setup\n"
        "        uses: actions/setup-python@v5\n"
        "      - run: echo three\n")
    assert plan.versions == ["3.9", "3.10", "3.11"]
    assert plan.steps == [ci_local.Step("One", ["echo one"]),
                          ci_local.Step("Two", ["echo a", "echo b"]),
                          ci_local.Step("", ["echo three"])]


def test_a_workflow_it_cannot_read_is_an_error_not_an_empty_matrix():
    with pytest.raises(ValueError):
        ci_local.parse_workflow("steps:\n  - run: pytest\n")
    with pytest.raises(ValueError):
        ci_local.parse_workflow("python-version: [3.9]\n")
