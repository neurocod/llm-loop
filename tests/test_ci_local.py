"""scripts/ci_local.py must run what CI runs, or refuse -- never something else.

Its reader is deliberately not YAML: it models a subset, and every shape
outside it must raise before anything runs. These pins cover the real workflow,
the shapes it accepts (with their exact text), one refusal per shape it
refuses, and that a failing step is reported as a failure.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "ci_local", os.path.join(ROOT, "scripts", "ci_local.py"))
ci_local = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci_local)

HEAD = ("on:\n"
        "  push:\n"
        "jobs:\n"
        "  pytest:\n"
        "    runs-on: ${{ matrix.os }}\n"
        "    strategy:\n"
        "      matrix:\n"
        "        python-version: [\"3.9\", '3.10']\n"
        "    steps:\n")


def workflow(steps):
    return HEAD + steps


def test_the_real_workflow_parses_to_the_declared_floor_and_every_step():
    with open(ci_local.WORKFLOW, encoding="utf-8") as handle:
        plan = ci_local.parse_workflow(handle.read())
    assert plan.versions[0] == "3.9"  # pyproject's requires-python floor
    as_tuples = [tuple(int(part) for part in v.split(".")) for v in plan.versions]
    assert as_tuples == sorted(set(as_tuples)) and len(as_tuples) > 1
    scripts = [step.script for step in plan.steps]
    assert scripts == ['python -m pip install -e ".[test]"',
                       "python -m pytest -q",
                       "python claude-plugin/ask-user-gate/hooks/ask_user_gate.py"
                       " --self-test",
                       "python claude-plugin/ask-user-gate/bin/try_patch.py"
                       " --selftest"]


def test_inline_and_block_runs_keep_their_text_order_and_names():
    plan = ci_local.parse_workflow(workflow(
        "      - uses: actions/checkout@v4\n"
        "      - uses: actions/setup-python@v5\n"
        "        with:\n"
        "          python-version: ${{ matrix.python-version }}\n"
        "      - name: One\n"
        "        run: echo one\n"
        "      - name: Two\n"
        "        run: |\n"
        "          # a shell comment, part of the script\n"
        "          if true; then\n"
        "            echo a\n"
        "          fi\n"
        "\n"
        "          echo b\n"
        "\n"
        "      - name: Three\n"
        "        run: |-\n"
        "          echo c\n"
        "      - run: echo four\n"))
    assert plan.versions == ["3.9", "3.10"]
    assert plan.steps == [
        ci_local.Step("One", "echo one"),
        ci_local.Step("Two", "# a shell comment, part of the script\n"
                             "if true; then\n  echo a\nfi\n\necho b\n"),
        ci_local.Step("Three", "echo c"),
        ci_local.Step("", "echo four")]


def test_a_dash_run_block_ends_at_its_sibling_key():
    # The block is indented past `run`, not past the `-`: `name:` is a key of
    # the same step, not a line of the script.
    plan = ci_local.parse_workflow(workflow(
        "      - run: |\n"
        "          echo a\n"
        "        name: Named after\n"))
    assert plan.steps == [ci_local.Step("Named after", "echo a\n")]


# id -> (the steps, a fragment of the refusal naming the reason). The fragment
# keeps each case honest: a refusal for some other reason would pass without it.
REFUSED_STEP_SHAPES = {
    "folded block": ("      - run: >\n          echo a\n          echo b\n", "'>'"),
    "folded strip": ("      - run: >-\n          echo a\n", "'>-'"),
    "keep chomping": ("      - run: |+\n          echo a\n", "'|+'"),
    "indentation indicator": ("      - run: |2\n          echo a\n", "'|2'"),
    "indicator and strip": ("      - run: |-2\n          echo a\n", "'|-2'"),
    "comment after header": ("      - run: | # note\n          echo a\n", "'| # note'"),
    "double-quoted": ("      - run: \"echo a\"\n", "quoted"),
    "single-quoted": ("      - run: 'echo a'\n", "quoted"),
    "inline comment": ("      - run: echo a # note\n", "comment"),
    "plain continued": ("      - run: echo a\n          && echo b\n", "continued"),
    "colon space": ("      - run: echo a: b\n", "`: `"),
    "expression": ("      - run: echo ${{ matrix.os }}\n", "expressions"),
    "step if": ("      - run: echo a\n        if: always()\n", ": if not modelled"),
    "step shell": ("      - run: echo a\n        shell: bash\n", ": shell not"),
    "step working-directory": ("      - run: echo a\n        working-directory: x\n",
                               ": working-directory not"),
    "step env": ("      - run: echo a\n        env:\n          A: b\n", ": env not"),
    "step continue-on-error": ("      - run: echo a\n        continue-on-error: true\n",
                               ": continue-on-error not"),
    "unknown action": ("      - uses: some/action@v1\n        with:\n          run: echo a\n",
                       "some/action"),
    "checkout with inputs": ("      - uses: actions/checkout@v4\n        with:\n"
                             "          path: x\n", "actions/checkout"),
    "setup-python other input": ("      - uses: actions/setup-python@v5\n        with:\n"
                                 "          python-version: '3.12'\n",
                                 "actions/setup-python"),
    "neither uses nor run": ("      - name: Nothing\n", "neither"),
    "tab indentation": ("      - run: |\n\t  echo a\n", "tab"),
    "anchor": ("      - run: &a echo a\n", "does not model"),
    "same-column sequence": ("    - run: echo a\n", "not a `key: value`"),
}


@pytest.mark.parametrize("steps, reason", list(REFUSED_STEP_SHAPES.values()),
                         ids=list(REFUSED_STEP_SHAPES))
def test_a_step_shape_it_does_not_model_is_refused(steps, reason):
    with pytest.raises(ci_local.WorkflowShapeError) as refused:
        ci_local.parse_workflow(workflow(steps))
    assert reason in str(refused.value)


RUN = "      - run: echo a\n"
REFUSED_WORKFLOW_SHAPES = {
    "second job": (HEAD + RUN + "  other:\n    runs-on: x\n    steps:\n" + RUN,
                   "exactly one job"),
    "workflow defaults": ("defaults:\n  run:\n    shell: bash\n" + HEAD + RUN,
                          "the workflow: defaults"),
    "workflow env": ("env:\n  A: b\n" + HEAD + RUN, "the workflow: env"),
    "job env": (HEAD.replace("    runs-on:", "    env:\n      A: b\n    runs-on:") + RUN,
                "job pytest: env"),
    "matrix include": (HEAD.replace("    steps:\n", "        include:\n"
                                    "          - python-version: '3.14'\n"
                                    "    steps:\n") + RUN, "matrix: include"),
    "matrix exclude": (HEAD.replace("    steps:\n", "        exclude:\n"
                                    "          - python-version: '3.9'\n"
                                    "    steps:\n") + RUN, "matrix: exclude"),
    "unquoted version": (HEAD.replace("[\"3.9\", '3.10']", "[3.9, 3.10]") + RUN,
                         "must be quoted"),
    "block-list versions": (HEAD.replace(" [\"3.9\", '3.10']",
                                         "\n          - '3.9'") + RUN, "flow list"),
    "no versions": (HEAD.replace("        python-version: [\"3.9\", '3.10']\n",
                                 "        os: [x]\n") + RUN, "no python-version"),
    "no run steps": (HEAD + "      - uses: actions/checkout@v4\n", "no `run:` steps"),
}


@pytest.mark.parametrize("text, reason", list(REFUSED_WORKFLOW_SHAPES.values()),
                         ids=list(REFUSED_WORKFLOW_SHAPES))
def test_a_workflow_shape_it_does_not_model_is_refused(text, reason):
    with pytest.raises(ci_local.WorkflowShapeError) as refused:
        ci_local.parse_workflow(text)
    assert reason in str(refused.value)


def _run_steps(steps, tmp_path):
    """run_steps with this interpreter first on PATH; (its result, the log)."""
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    log_path = tmp_path / "log.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        failed = ci_local.run_steps(steps, env, log, cwd=tmp_path)
    return failed, log_path.read_text(encoding="utf-8")


def test_a_failing_line_fails_its_step_even_when_a_later_one_passes(tmp_path):
    # `fail; echo done` is green under a plain `sh -c` and under a line-by-line
    # runner that ignores `;`. CI's shells fail it -- bash -e at the first
    # command, pwsh through its `exit $LASTEXITCODE` (echo sets none) -- and
    # the next step never runs.
    failing = 'python -c "raise SystemExit(3)"; echo done'
    marker = tmp_path / "ran"
    steps = [ci_local.Step("Fails", failing),
             ci_local.Step("After", f'python -c "open(r\'{marker}\', \'w\')"')]
    failed, log = _run_steps(steps, tmp_path)
    assert failed is not None and failed.startswith("Fails"), log
    assert not marker.exists()


def test_a_multi_line_step_is_one_script(tmp_path):
    # A variable set on one line is seen on the next: one shell per step.
    script = ('$v = "7"\npython -c "import sys; sys.exit(int(sys.argv[1]) - 7)" $v\n'
              if os.name == "nt" else
              'v=7\npython -c "import sys; sys.exit(int(sys.argv[1]) - 7)" "$v"\n')
    failed, log = _run_steps([ci_local.Step("Carry", script)], tmp_path)
    assert failed is None, log
