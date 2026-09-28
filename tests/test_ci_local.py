"""scripts/ci_local.py must run what CI runs, or refuse -- never something else.

Its reader is deliberately not YAML: it models a subset, and every shape
outside it must raise before anything runs. These pins cover the real workflow,
the shapes it accepts (with their exact text), one refusal per shape it
refuses, and that a failing step is reported as a failure. Around the steps:
an interrupted run takes its whole process tree down before it returns, the
per-version lock, the scrubbed environment, the interpreter check (its python
faked, so no uv runs here) and the venv root's key.
"""

import _thread
import importlib.util
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Cancellation: the whole tree dies before the venv lock can be released.
# ---------------------------------------------------------------------------

# A grandchild that ignores SIGINT and appends a beat every 20 ms, its pid on
# the first line; bounded so that a failed run leaves nothing forever.
_BEATING = ("import os, signal, sys, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "with open(sys.argv[1], 'a') as out:\n"
            "    out.write(f'{os.getpid()}\\n')\n"
            "    for _ in range(3000):\n"
            "        out.write('.')\n"
            "        out.flush()\n"
            "        time.sleep(0.02)\n"
            "    out.write('end')\n")
# The direct child, standing in for the step's shell: it only waits on it.
_PARENT = ("import subprocess, sys\n"
           "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]]).wait()\n")
# Until the first beat: two python starts. Measured 0.045-0.093 s (2026-09-28,
# Windows, 3.14 and a 3.9 venv, 5 runs each); 20 s leaves room for a loaded
# machine.
_FIRST_BEAT_TIMEOUT = 20.0
# Long enough for a surviving grandchild to beat ~25 times.
_QUIET = 0.5


def _beats(path):
    try:
        return path.read_text(encoding="utf-8").count(".")
    except OSError:
        return 0


def _kill_leftover(path):
    try:
        pid = int(path.read_text(encoding="utf-8").split("\n", 1)[0])
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass


def test_an_interrupted_run_kills_the_whole_tree_before_it_returns(tmp_path):
    """Ctrl+C in run_tree: the grandchild is gone when the exception arrives.

    subprocess.run killed only the direct child, so the grandchild -- a python
    under the step's shell -- kept running while exclusive() released the venv
    lock, and a waiting run could `--clear` the venv under it.
    """
    beat = tmp_path / "beat.txt"

    def interrupt_on_first_beat():
        deadline = time.monotonic() + _FIRST_BEAT_TIMEOUT
        while not _beats(beat) and time.monotonic() < deadline:
            time.sleep(0.01)
        _thread.interrupt_main()

    threading.Thread(target=interrupt_on_first_beat, daemon=True).start()
    try:
        with pytest.raises(KeyboardInterrupt):
            ci_local.run_tree([sys.executable, "-c", _PARENT, _BEATING, str(beat)],
                              stdin=subprocess.DEVNULL)
        assert _beats(beat), "the grandchild never started"
        # An interrupt held until the tree exited by itself would pass the
        # check below; the tree must have been cut short.
        assert "end" not in beat.read_text(encoding="utf-8"), \
            "the interrupt was not seen until the step ended"
        after_kill = _beats(beat)
        time.sleep(_QUIET)
        assert _beats(beat) == after_kill, "the grandchild outlived run_tree"
    finally:
        _kill_leftover(beat)


def test_run_tree_returns_the_exit_code():
    assert ci_local.run_tree([sys.executable, "-c", "raise SystemExit(5)"]) == 5


# ---------------------------------------------------------------------------
# The per-version lock.
# ---------------------------------------------------------------------------

def test_exclusive_holds_the_lock_and_releases_it_on_an_exception(tmp_path):
    """A second handle -- another run, to both lock APIs -- is refused while
    the body runs and served after it raised. (The release is the handle's
    close as much as the explicit unlock: dropping the unlock alone passes.)"""
    path = tmp_path / "v.lock"
    with open(path, "a+b") as other:
        with pytest.raises(RuntimeError):
            with ci_local.exclusive(path):
                with pytest.raises(OSError):
                    ci_local._lock(other)
                raise RuntimeError("the body failed")
        ci_local._lock(other)
        ci_local._lock(other, unlock=True)


def test_exclusive_waits_for_the_holder_and_says_so_once(tmp_path, monkeypatch,
                                                          capsys):
    path = tmp_path / "v.lock"
    naps = []
    with open(path, "a+b") as holder:
        ci_local._lock(holder)

        def nap(seconds):
            naps.append(seconds)
            if len(naps) == 2:
                ci_local._lock(holder, unlock=True)

        monkeypatch.setattr(ci_local.time, "sleep", nap)
        with ci_local.exclusive(path):
            with pytest.raises(OSError):
                ci_local._lock(holder)
    assert len(naps) == 2
    assert capsys.readouterr().out.count("waiting for") == 1


# ---------------------------------------------------------------------------
# The steps' environment and interpreter.
# ---------------------------------------------------------------------------

def test_step_env_scrubs_the_callers_python_setup(monkeypatch, tmp_path):
    caller = {"PYTHONPATH": "p", "PYTHONHOME": "h", "PIP_INDEX_URL": "i",
              "PYTEST_ADDOPTS": "-x", "VIRTUAL_ENV": "elsewhere",
              "CONDA_PREFIX": "c", "CI_LOCAL_KEPT": "1"}
    for key, value in caller.items():
        monkeypatch.setenv(key, value)
    venv = tmp_path / "venv"
    env = ci_local.step_env(venv)
    # Named here, not read from _SCRUBBED: a prefix dropped there must fail.
    leaked = [key for key in caller if key in env
              and key not in ("VIRTUAL_ENV", "CI_LOCAL_KEPT")]
    assert leaked == []
    assert env["VIRTUAL_ENV"] == str(venv)
    assert env["CI_LOCAL_KEPT"] == "1"
    assert env["PATH"].split(os.pathsep)[0] == str(ci_local.venv_bin(venv))


def _fake_programs(directory, *names):
    """Empty executables that shutil.which finds under these names."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = directory / (name + ".exe" if os.name == "nt" else name)
        path.write_bytes(b"")
        path.chmod(0o755)
    return directory


def _check(venv, path_dirs, steps, monkeypatch, prefix=None, code=0):
    """check_interpreters, the venv python faked: (its answer, calls, log)."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        out = "Python 3.9.25 (fake)" if argv[1] == "-VV" else f"{prefix}\n"
        return subprocess.CompletedProcess(argv, code, out, "")

    monkeypatch.setattr(ci_local.subprocess, "run", fake_run)
    env = {"PATH": os.pathsep.join(str(d) for d in path_dirs)}
    log = []

    class Log:
        write = log.append

    wrong = ci_local.check_interpreters(venv, env, steps, Log())
    return wrong, calls, "".join(log)


PLAIN = [ci_local.Step("Test", "python -m pytest")]


def test_a_python_missing_from_the_venv_is_refused(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    ci_local.venv_bin(venv).mkdir(parents=True)
    system = _fake_programs(tmp_path / "system", "python")
    wrong, calls, _ = _check(venv, [ci_local.venv_bin(venv), system], PLAIN,
                             monkeypatch)
    assert wrong and "`python` resolves to" in wrong and not calls


def test_python3_is_checked_only_when_a_step_uses_it(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    bin_dir = _fake_programs(ci_local.venv_bin(venv), "python")
    system = _fake_programs(tmp_path / "system", "python3")
    wrong, _, _ = _check(venv, [bin_dir, system],
                         [ci_local.Step("Test", "python3 -m pytest")], monkeypatch,
                         prefix=venv)
    assert wrong and "`python3` resolves to" in wrong
    wrong, _, _ = _check(venv, [bin_dir, system], PLAIN, monkeypatch, prefix=venv)
    assert wrong is None


def test_a_venv_python_with_another_prefix_is_refused(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    bin_dir = _fake_programs(ci_local.venv_bin(venv), "python")
    wrong, _, _ = _check(venv, [bin_dir], PLAIN, monkeypatch,
                         prefix=tmp_path / "elsewhere")
    assert wrong and "sys.prefix" in wrong
    wrong, _, _ = _check(venv, [bin_dir], PLAIN, monkeypatch, prefix=venv, code=1)
    assert wrong and "sys.prefix" in wrong


def test_the_venv_python_is_accepted_and_logged(tmp_path, monkeypatch):
    venv = tmp_path / "venv"
    bin_dir = _fake_programs(ci_local.venv_bin(venv), "python")
    wrong, calls, log = _check(venv, [bin_dir], PLAIN, monkeypatch, prefix=venv)
    assert wrong is None
    assert [argv[1] for argv in calls] == ["-c", "-VV"]
    assert "interpreter: Python 3.9.25 (fake)" in log


# ---------------------------------------------------------------------------
# The venv root's key.
# ---------------------------------------------------------------------------

def test_the_venv_root_folds_case_only_on_windows():
    assert (ci_local.venv_root(Path("/Foo"), windows=False)
            != ci_local.venv_root(Path("/foo"), windows=False))
    assert (ci_local.venv_root(Path("C:/Foo"), windows=True)
            == ci_local.venv_root(Path("c:/foo"), windows=True))
