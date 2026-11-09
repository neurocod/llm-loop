"""agent-tools/ci_failures.py: the log split, the pytest blocks, the dedup
across a matrix and every exit code -- on captured text, without gh or a
network. The pytest text is trimmed from `gh run view 42537657856 --log-failed`
(neurocod/llm-loop, 2026-11-09) unless a fixture says it is made up."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import ci_failures as cf

MAC = "pytest (macos-latest, 3.12)"
UBU = "pytest (ubuntu-latest, 3.13)"
WIN = "pytest (windows-latest, 3.12)"
OLD = "pytest (ubuntu-latest, 3.9)"


def prefixed(job: str, body: str, bom: bool = True) -> str:
    """`body` as gh prints it: `<job>\\tUNKNOWN STEP\\t<timestamp> <line>`, the
    first line's timestamp behind a BOM (both as in the live log)."""
    out = []
    for i, line in enumerate(body.split("\n")):
        mark = "\ufeff" if bom and i == 0 else ""
        out.append(f"{job}\tUNKNOWN STEP\t{mark}2026-11-09T00:01:25.{1546560 + i:07d}Z {line}")
    return "\n".join(out)


# The cpu_load red, macOS 3.12 rendering (carets, no `<string>` source).
CPU_312 = r"""=================================== FAILURES ===================================
_____________ test_stream_exits_0_when_its_reader_closes_the_pipe ______________

    def test_stream_exits_0_when_its_reader_closes_the_pipe():
        child = subprocess.Popen([sys.executable, "-c", ENDLESS_STREAM], cwd=TOOLS,
                                 env={**os.environ, "PYTHONPATH": str(TOOLS)},
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = child.communicate(timeout=STREAM_CHILD_S)
>               pytest.fail(f"stream child printed nothing{'' if lines else ' in time'}, "
                            f"exit {child.returncode}:\n{err.decode(errors='replace')}")
E                           Failed: stream child printed nothing, exit 1:
E                           Traceback (most recent call last):
E                             File "<string>", line 15, in <module>
E                             File "{WS}agent-tools/cpu_load.py", line 498, in main
E                               args = build_parser().parse_args()
E                                      ^^^^^^^^^^^^^^
E                             File "{PY}argparse.py", line 1820, in __init__
E                               self.add_argument(
E                             File "{PY}argparse.py", line 172, in __init__
E                               import shutil
E                             File "{PY}shutil.py", line 44, in <module>
E                               import _winapi
E                           ModuleNotFoundError: No module named '_winapi'

tests/test_cpu_load.py:460: Failed"""

# The same red, ubuntu 3.13 rendering: the `-c` source shown, a multi-line
# call's every line, `~` carets, other stdlib line numbers.
CPU_313 = r"""=================================== FAILURES ===================================
_____________ test_stream_exits_0_when_its_reader_closes_the_pipe ______________

    def test_stream_exits_0_when_its_reader_closes_the_pipe():
        child = subprocess.Popen([sys.executable, "-c", ENDLESS_STREAM], cwd=TOOLS,
                                 env={**os.environ, "PYTHONPATH": str(TOOLS)},
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            _, err = child.communicate(timeout=STREAM_CHILD_S)
>               pytest.fail(f"stream child printed nothing{'' if lines else ' in time'}, "
                            f"exit {child.returncode}:\n{err.decode(errors='replace')}")
E                           Failed: stream child printed nothing, exit 1:
E                           Traceback (most recent call last):
E                             File "<string>", line 15, in <module>
E                               sys.exit(cpu_load.main())
E                                        ~~~~~~~~~~~~~^^
E                             File "/home/runner/work/llm-loop/llm-loop/agent-tools/cpu_load.py", line 498, in main
E                               args = build_parser().parse_args()
E                                      ~~~~~~~~~~~~^^
E                             File "/opt/hostedtoolcache/Python/3.13.18/x64/lib/python3.13/argparse.py", line 1841, in __init__
E                               self.add_argument(
E                               ~~~~~~~~~~~~~~~~~^
E                                   default_prefix+'h', default_prefix*2+'help',
E                                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
E                             File "/opt/hostedtoolcache/Python/3.13.18/x64/lib/python3.13/argparse.py", line 170, in __init__
E                               import shutil
E                             File "/opt/hostedtoolcache/Python/3.13.18/x64/lib/python3.13/shutil.py", line 43, in <module>
E                               import _winapi
E                           ModuleNotFoundError: No module named '_winapi'

tests/test_cpu_load.py:460: Failed"""

SUMMARY_CPU = r"""=========================== short test summary info ============================
FAILED tests/test_cpu_load.py::test_stream_exits_0_when_its_reader_closes_the_pipe - Failed: stream child printed nothing, exit 1:
Traceback (most recent call last):
  File "<string>", line 15, in <module>
ModuleNotFoundError: No module named '_winapi'"""

CODEX_BLOCK = r"""_ test_child_wait_with_busy_main_times_out_without_resend[network-child-heartbeats-732.0] _

tmp_path = PosixPath('/private/var/folders/h5/t85nd63d71x27_8gl5gw3l_w0000gn/T/pytest-of-runner/pytest-0/test_child_wait_with_busy_main1')
monkeypatch = <_pytest.monkeypatch.MonkeyPatch object at 0x1059b1f70>
scenario = 'network-child-heartbeats', credit = 732.0

>       assert outcome.network_wait_s == credit
E       AssertionError: assert 731.9999999999999 == 732.0

tests/codex_agent/test_codex_agent.py:302: AssertionError"""

SUMMARY_CODEX = ("FAILED tests/codex_agent/test_codex_agent.py::test_child_wait_with_busy_main_times_out_"
                 "without_resend[network-child-heartbeats-732.0] - AssertionError: assert "
                 "731.9999999999999 == 732.0")

TAIL_OF_JOB = r"""1 failed, 2404 passed, 79 skipped, 27 subtests passed in 152.79s (0:02:32)
##[error]Process completed with exit code 1.
Post job cleanup.
[command]/usr/bin/git version
git version 2.56.0"""

MAC_PATHS = {"{WS}": "/Users/runner/work/llm-loop/llm-loop/",
             "{PY}": "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/"}
# Made up: the macOS text with Windows runner paths, as a Windows job prints them.
WIN_PATHS = {"{WS}": "D:\\a\\llm-loop\\llm-loop\\",
             "{PY}": "C:\\hostedtoolcache\\windows\\Python\\3.12.10\\x64\\Lib\\"}


def fill(text: str, paths: dict) -> str:
    for k, v in paths.items():
        text = text.replace(k, v)
    if "\\" in paths.get("{WS}", ""):
        text = text.replace("agent-tools/cpu_load.py", "agent-tools\\cpu_load.py")
        text = text.replace("tests/test_cpu_load.py:460", "tests\\test_cpu_load.py:460")
    return text


MAC_BODY = "\n".join([
    "Current runner version: '2.338.2'",
    "........................................................................ [ 97%]",
    CPU_312 + "\n" + CODEX_BLOCK,
    SUMMARY_CPU, SUMMARY_CODEX,
    TAIL_OF_JOB.replace("1 failed", "2 failed"),
])
MAC_BODY = fill(MAC_BODY, MAC_PATHS)
UBU_BODY = "\n".join(["Current runner version: '2.338.2'", CPU_313, SUMMARY_CPU, TAIL_OF_JOB])
WIN_BODY = fill("\n".join([CPU_312, SUMMARY_CPU, TAIL_OF_JOB]), WIN_PATHS)
# Made up: a job whose install step failed, so no pytest ran.
OLD_BODY = "\n".join(["##[group]Run pip install -e .[test]", "Collecting pytest>=8"]
                     + [f"  Downloading pkg{i}.whl ({i}.2 MB)" for i in range(50)]
                     + ["ERROR: Could not find a version that satisfies the requirement pytest>=8",
                        "##[error]Process completed with exit code 1.",
                        "Post job cleanup.", "[command]/usr/bin/git version"])

LOG = "\n".join(prefixed(j, b) for j, b in [(MAC, MAC_BODY), (UBU, UBU_BODY), (WIN, WIN_BODY),
                                             (OLD, OLD_BODY)])


def job(name, conclusion="failure", step="Test", status="completed", job_id=None):
    steps = [{"name": "Install", "conclusion": "success"},
             {"name": step, "conclusion": conclusion},
             {"name": "Post Run actions/checkout@v4", "conclusion": "success"}]
    out = {"name": name, "status": status, "conclusion": conclusion if status == "completed" else "",
           "steps": steps}
    if job_id is not None:
        out["databaseId"] = job_id
    return out


def own_log(lines) -> str:
    """A job's own log as the API serves it: timestamped lines, a BOM first."""
    return "\ufeff" + "\n".join(f"2026-11-09T00:00:00.{i:07d}Z {ln}" for i, ln in enumerate(lines))


def run_payload(status="completed", conclusion="failure", jobs=None):
    """What `gh run view --json` answers for run 42."""
    return {"databaseId": 42, "workflowName": "tests", "headSha": "0ca0abdb" + "0" * 32,
            "status": status, "conclusion": conclusion if status == "completed" else "",
            "url": "https://github.com/o/r/actions/runs/42", "displayTitle": "a commit", "attempt": 1,
            "jobs": jobs if jobs is not None else [
                job("pytest (ubuntu-latest, 3.10)", "success"), job(MAC), job(UBU), job(WIN),
                job(OLD, step="Install")]}


def units_of(body_job):
    lines = cf.split_log(LOG)[body_job]
    return cf.failure_units([cf.clean(ln) for ln in lines], cf.TAIL_LINES)


# ------------------------------------------------------------ parsing


def test_split_log_puts_every_line_under_its_job_and_keeps_the_timestamp():
    jobs = cf.split_log(LOG)
    assert list(jobs) == [MAC, UBU, WIN, OLD]
    assert len(jobs[UBU]) == len(UBU_BODY.split("\n"))
    assert jobs[UBU][0].startswith("\ufeff2026-11-09T00:01:25.")


def test_split_log_handles_crlf_and_a_line_without_the_prefix():
    text = prefixed(MAC, "one\ntwo").replace("\n", "\r\n") + "\r\nstray continuation\r\n"
    assert [cf.clean(ln) for ln in cf.split_log(text)[MAC]] == ["one", "two", "stray continuation"]


def test_clean_strips_timestamp_bom_and_colour():
    raw = "\ufeff2026-11-09T00:00:32.7638521Z \x1b[31mFAILED\x1b[0m tests/x.py::t"
    assert cf.clean(raw) == "FAILED tests/x.py::t"


def test_blocks_entries_and_outcome_with_ansi_and_crlf():
    colour = CPU_313.replace("FAILURES", "\x1b[1mFAILURES\x1b[0m").replace("E   ", "\x1b[31mE   \x1b[0m", 1)
    text = prefixed(UBU, "\n".join([colour, SUMMARY_CPU, TAIL_OF_JOB])).replace("\n", "\r\n")
    lines = [cf.clean(ln) for ln in cf.split_log(text)[UBU]]
    blocks, entries, outcomes, end = cf.parse_pytest(lines)
    assert [t for t, _ in blocks] == ["test_stream_exits_0_when_its_reader_closes_the_pipe"]
    body = blocks[0][1]
    assert body[0].startswith("    def test_stream")          # leading blank dropped
    assert body[-1] == "tests/test_cpu_load.py:460: Failed"
    assert not any("\x1b" in ln or "\r" in ln for ln in body)
    assert len(entries) == 1 and entries[0][-1] == "ModuleNotFoundError: No module named '_winapi'"
    assert outcomes == ["1 failed, 2404 passed, 79 skipped, 27 subtests passed in 152.79s (0:02:32)"]
    assert lines[end].startswith("1 failed, 2404 passed")


def test_chained_traceback_separator_is_not_a_block_header():
    lines = ["=== FAILURES ===", "___ test_a ___", "first", "_ _ _ _ _ _ _ _ _ _", "second",
             "=== short test summary info ===", "FAILED tests/x.py::test_a - boom"]
    blocks, _, _, _ = cf.parse_pytest(lines)
    assert blocks == [("test_a", ["first", "_ _ _ _ _ _ _ _ _ _", "second"])]


def test_summary_entry_finds_its_block_by_class_and_by_error_title():
    lines = ["=== ERRORS ===", "___ ERROR at setup of TestK.test_b ___", "setup boom",
             "=== FAILURES ===", "___ TestK.test_a ___", "assert 0",
             "=== short test summary info ===",
             "FAILED tests/x.py::TestK::test_a - assert 0",
             "ERROR tests/x.py::TestK::test_b - setup boom"]
    units, _ = cf.failure_units(lines, 10)
    assert [(u["headline"], u["block"]) for u in units] == [
        ("FAILED tests/x.py::TestK::test_a", ["assert 0"]),
        ("ERROR tests/x.py::TestK::test_b", ["setup boom"])]


def test_failed_in_call_and_error_in_teardown_keep_their_own_blocks():
    # pytest prints ERRORS before FAILURES; the summary lists FAILED first.
    lines = ["=== ERRORS ===", "___ ERROR at teardown of test_x ___", "teardown boom",
             "___ ERROR collecting tests/y.py ___", "ImportError: no y",
             "=== FAILURES ===", "___ test_x ___", "assert 1 == 2",
             "=== short test summary info ===",
             "FAILED tests/x.py::test_x - assert 1 == 2",
             "ERROR tests/x.py::test_x - teardown boom",
             "ERROR tests/y.py - ImportError: no y"]
    units, _ = cf.failure_units(lines, 10)
    assert [(u["headline"], u["block"]) for u in units] == [
        ("FAILED tests/x.py::test_x", ["assert 1 == 2"]),
        ("ERROR tests/x.py::test_x", ["teardown boom"]),
        ("ERROR tests/y.py", ["ImportError: no y"])]


def test_pytest_lines_printed_by_a_test_stay_in_its_block():
    # Made up: test_a runs pytest in a child and its captured output carries
    # the child's banner and outcome line; test_b's block follows.
    lines = ["=== FAILURES ===", "___ test_a ___", "assert child ok",
             "----- Captured stdout call -----", "===== x =====", "1 failed in 0.10s",
             "___ test_b ___", "assert 3 == 4",
             "=== short test summary info ===",
             "FAILED tests/x.py::test_a - assert child ok",
             "FAILED tests/x.py::test_b - assert 3 == 4",
             "2 failed, 7 passed in 1.50s"]
    units, outcomes = cf.failure_units(lines, 10)
    assert [(u["headline"], u["block"]) for u in units] == [
        ("FAILED tests/x.py::test_a", ["assert child ok", "----- Captured stdout call -----",
                                       "===== x =====", "1 failed in 0.10s"]),
        ("FAILED tests/x.py::test_b", ["assert 3 == 4"])]
    assert outcomes == ["2 failed, 7 passed in 1.50s"]


# A test that runs pytest in a child and prints its output: pytest 9.2.0's
# real text (2026-11-09, a scratch outer/child pair), paths shortened. The
# child's session banner, FAILURES, block, warnings, summary and outcome are
# all inside test_child_session_is_green's captured stdout.
NESTED = r"""=================================== FAILURES ===================================
_________________________ test_child_session_is_green __________________________

    def test_child_session_is_green():
        proc = subprocess.run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", str(CHILD)],
                              capture_output=True, text=True, cwd=CHILD)
        print(proc.stdout)
>       assert proc.returncode == 0
E       AssertionError: assert 1 == 0
E        +  where 1 = CompletedProcess(args=['/usr/bin/python3', '-m', 'pytest', ...], returncode=1, stdout='...').returncode

tests/test_outer.py:12: AssertionError
----------------------------- Captured stdout call -----------------------------
============================= test session starts ==============================
platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.6.0
rootdir: /tmp/pytest-of-runner/pytest-0/child
collected 2 items

test_inner.py .F                                                         [100%]

=================================== FAILURES ===================================
_______________________________ test_inner_fails _______________________________

    def test_inner_fails():
>       assert 2 + 2 == 5
E       assert (2 + 2) == 5

test_inner.py:6: AssertionError
=============================== warnings summary ===============================
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED test_inner.py::test_inner_fails - assert (2 + 2) == 5
=================== 1 failed, 1 passed, 1 warning in 0.09s =====================

_____________________________ test_after_the_child _____________________________

    def test_after_the_child():
>       assert "outer" == "other"
E       AssertionError: assert 'outer' == 'other'

tests/test_outer.py:16: AssertionError
=========================== short test summary info ============================
FAILED tests/test_outer.py::test_child_session_is_green - AssertionError: assert 1 == 0
FAILED tests/test_outer.py::test_after_the_child - AssertionError: assert 'outer' == 'other'
========================= 2 failed, 5 passed in 0.84s ==========================
##[error]Process completed with exit code 1."""


def test_a_child_pytest_session_in_captured_output_stays_in_its_block():
    units, outcomes = cf.failure_units(NESTED.split("\n"), 10)
    assert [u["headline"] for u in units] == ["FAILED tests/test_outer.py::test_child_session_is_green",
                                              "FAILED tests/test_outer.py::test_after_the_child"]
    child = units[0]["block"]
    assert child[-1] == "=================== 1 failed, 1 passed, 1 warning in 0.09s ====================="
    assert "FAILED test_inner.py::test_inner_fails - assert (2 + 2) == 5" in child
    assert units[1]["block"][-1] == "tests/test_outer.py:16: AssertionError"
    assert outcomes == ["2 failed, 5 passed in 0.84s"]


def test_a_child_session_without_its_outcome_closes_nothing():
    # Made up: the child died after its banner; the outer run's own outcome
    # must not be taken for the child's, or test_after_the_child would vanish.
    lines = NESTED.split("\n")
    cut = lines[:lines.index("platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.6.0")]
    cut += ["Killed"] + lines[lines.index("_____________________________ test_after_the_child _____________________________"):]
    units, outcomes = cf.failure_units(cut, 10)
    assert [u["headline"] for u in units] == ["FAILED tests/test_outer.py::test_child_session_is_green",
                                              "FAILED tests/test_outer.py::test_after_the_child"]
    assert units[0]["block"][-1] == "Killed"
    assert outcomes == ["2 failed, 5 passed in 0.84s"]


def two_invocations(printed: list) -> list:
    """Made up: two pytest runs in one step; test_a of the first prints the
    lines `printed` (a banner shaped like a child's) and nothing else."""
    return (["=== FAILURES ===", "___ test_a ___", "E   assert 0", "tests/x.py:2: AssertionError",
             "----- Captured stdout call -----", *printed,
             "___ test_b ___", "E   assert 3 == 4", "tests/x.py:5: AssertionError",
             "=== short test summary info ===",
             "FAILED tests/x.py::test_a - assert 0", "FAILED tests/x.py::test_b - assert 3 == 4",
             "=== 2 failed in 0.10s ===",
             "=== test session starts ===", "platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.6.0",
             "collected 1 item", "", "tests/y.py F",
             "=== FAILURES ===", "___ test_c ___", "E   assert 0", "tests/y.py:2: AssertionError",
             "=== short test summary info ===", "FAILED tests/y.py::test_c - assert 0",
             "=== 1 failed in 0.05s ===", "##[error]Process completed with exit code 1."])


@pytest.mark.parametrize("printed", [
    ["=== test session starts ==="],
    # With a header: test_b's title before any FAILURES of a child is the outer run's.
    ["=== test session starts ===", "platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.6.0"],
    # A child that died in its FAILURES: the summary naming test_a itself is the outer run's.
    ["=== test session starts ===", "platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.6.0",
     "collected 1 item", "=== FAILURES ===", "___ test_inner ___", "Killed"],
])
def test_a_printed_session_banner_hides_no_outer_failure(printed):
    units, outcomes = cf.failure_units(two_invocations(printed), 10)
    assert [u["headline"] for u in units][:3] == ["FAILED tests/x.py::test_a", "FAILED tests/x.py::test_b",
                                                  "FAILED tests/y.py::test_c"]
    assert units[1]["block"] == ["E   assert 3 == 4", "tests/x.py:5: AssertionError"]
    assert outcomes == ["2 failed in 0.10s", "1 failed in 0.05s"]


@pytest.mark.parametrize("printed, titles", [
    (["=== test session starts ==="], ["test_a"]),
    # With a header, but test_b's title comes before any FAILURES of a child.
    (["=== test session starts ===", "platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.6.0",
      "___ test_b ___", "E   assert 3 == 4"], ["test_a", "test_b"]),
])
def test_a_printed_banner_without_a_sessions_shape_takes_no_outcome(printed, titles):
    # Made up: `-rN` (no summary entry to tell the outer run by), test_a prints
    # a banner; the first run's outcome is not a child's, and a second run follows.
    lines = ["=== FAILURES ===", "___ test_a ___", "E   assert 0", "----- Captured stdout call -----", *printed,
             "=== 2 failed in 0.10s ===",
             "=== test session starts ===", "platform linux -- Python 3.13.7, pytest-9.2.0, pluggy-1.6.0",
             "collected 1 item", "=== 1 passed in 0.05s ===", "##[error]Process completed with exit code 1."]
    units, outcomes = cf.failure_units(lines, 10)
    assert [u["headline"] for u in units] == titles
    assert outcomes == ["2 failed in 0.10s", "1 passed in 0.05s"]


def test_quiet_outcome_line_right_after_the_blocks_is_the_outcome():
    # `-q -rN`: no summary, the unbannered outcome line ends the output.
    lines = ["=== FAILURES ===", "___ test_a ___", "assert 0", "",
             "1 failed, 3 passed in 0.50s", "##[error]Process completed with exit code 1."]
    units, outcomes = cf.failure_units(lines, 10)
    assert [(u["headline"], u["block"]) for u in units] == [("test_a", ["assert 0"])]
    assert outcomes == ["1 failed, 3 passed in 0.50s"]


def test_stopping_after_line_does_not_join_the_last_summary_entry():
    lines = ["=== short test summary info ===", "FAILED tests/x.py::t - boom", "  more of boom",
             "!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!",
             "1 failed in 0.20s"]
    units, outcomes = cf.failure_units(lines, 10)
    assert [(u["headline"], u["block"]) for u in units] == [("FAILED tests/x.py::t - boom", ["  more of boom"])]
    assert outcomes == ["1 failed in 0.20s"]


# Made up: a red pytest step, then a later step that failed too.
DEPLOY_AFTER = ["=== FAILURES ===", "___ test_a ___", "assert 0",
                "=== short test summary info ===", "FAILED tests/x.py::test_a - assert 0",
                "1 failed, 3 passed in 0.50s", "##[error]Process completed with exit code 1.",
                "##[group]Run ./deploy.sh", "deploy: permission denied", "##[endgroup]",
                "##[error]Process completed with exit code 2.", "Post job cleanup."]


def test_a_failure_after_pytest_is_reported_too():
    units, _ = cf.failure_units(DEPLOY_AFTER, 10)
    assert [u["kind"] for u in units] == ["pytest", "tail"]
    assert "deploy: permission denied" in units[1]["block"]
    assert units[1]["block"][-1] == "##[error]Process completed with exit code 2."


@pytest.mark.parametrize("after", [
    ["##[error]The action has timed out."],
    ["Fatal Python error: Segmentation fault", "##[error]Process completed with exit code 139."],
])
def test_a_crash_or_timeout_after_pytest_is_reported_too(after):
    lines = DEPLOY_AFTER[:6] + after
    units, _ = cf.failure_units(lines, 10)
    assert units[-1]["kind"] == "tail" and units[-1]["block"][-1] == after[-1]


def test_a_job_with_two_failed_steps_gets_its_tail_even_without_more_log():
    units, _ = cf.failure_units(DEPLOY_AFTER[:7], 10, failed_steps=2)
    assert [u["kind"] for u in units] == ["pytest", "tail"] and "2 failed steps" in units[1]["headline"]
    units, _ = cf.failure_units(DEPLOY_AFTER[:7], 10, failed_steps=1)
    assert [u["kind"] for u in units] == ["pytest"]


def test_clean_strips_osc_sequences_ended_by_st():
    assert cf.clean("\x1b]8;;https://x.test/\x1b\\link\x1b]8;;\x1b\\ text") == "link text"


def test_job_without_pytest_keeps_its_tail_up_to_the_runner_error():
    units, outcomes = units_of(OLD)
    assert outcomes == [] and len(units) == 1
    unit = units[0]
    assert unit["kind"] == "tail" and "no pytest failures" in unit["headline"]
    assert len(unit["block"]) == cf.TAIL_LINES
    assert unit["block"][-1] == "##[error]Process completed with exit code 1."
    assert unit["block"][-2].startswith("ERROR: Could not find a version")
    assert "Post job cleanup." not in unit["block"]


# ------------------------------------------------------------ dedup


def test_one_failure_on_three_platforms_is_one_group_shown_as_the_first_job():
    per_job = [(name, units_of(name)[0]) for name in (MAC, UBU, WIN)]
    groups = cf.group_failures(per_job)
    assert [g["jobs"] for g in groups] == [[MAC, UBU, WIN], [MAC]]
    cpu = groups[0]
    assert cpu["headline"] == "FAILED tests/test_cpu_load.py::test_stream_exits_0_when_its_reader_closes_the_pipe"
    assert cpu["shownFrom"] == MAC
    assert any("/Users/runner/work/llm-loop/llm-loop/agent-tools/cpu_load.py" in ln for ln in cpu["block"])


def test_a_different_message_or_parameter_is_another_group():
    base = "E       AssertionError: assert 731.9999999999999 == 732.0"
    a = {"kind": "pytest", "headline": "FAILED t.py::t[a]", "block": [base]}
    b = {"kind": "pytest", "headline": "FAILED t.py::t[b]", "block": [base]}
    c = {"kind": "pytest", "headline": "FAILED t.py::t[a]", "block": [base.replace("731", "730")]}
    groups = cf.group_failures([("j1", [a]), ("j2", [b]), ("j3", [c])])
    assert len(groups) == 3


@pytest.mark.parametrize("left, right", [
    ("File \"/Users/runner/work/r/r/a.py\", line 3", "File \"D:\\a\\r\\r\\a.py\", line 3"),
    ("tmp_path = PosixPath('/tmp/pytest-of-runner/pytest-0/t0')",
     "tmp_path = WindowsPath('C:/Users/runneradmin/AppData/Local/Temp/pytest-of-runneradmin/pytest-3/t0')"),
    ("<MonkeyPatch object at 0x1059b1f70>", "<MonkeyPatch object at 0x0000015F50CC64A0>"),
    ("E     1 failed, 3 passed in 1.25s", "E     1 failed, 3 passed in 3.50s (0:00:03)"),
    ("tests\\x.py:74: AssertionError", "tests/x.py:74: AssertionError"),
    ("p = '/tmp/tmpab12cd_9/out'", "p = '/tmp/tmpzz99yy00/out'"),
])
def test_normalize_takes_out_per_job_noise(left, right):
    assert cf.normalize(left) == cf.normalize(right)


def clock_block(args: str, then: str = "") -> str:
    """A block whose first entry has the argument line(s) `args`."""
    return "\n".join([args, "", "    def test_x(network_clock):", ">       assert network_clock == []",
                      "E       AssertionError", "", "tests/x.py:3: AssertionError", then])


# Run 42525390878, ubuntu 3.10 and macOS 3.12: the one red of 3 jobs, apart
# only by the fixture's wall clock.
CLOCK_A = ("network_clock = [' · RUNNING since 2026-11-08 21:28:59 (0 cmd, 0 edits, RECONNECTING since 21:28 "
           "(1 tries))', ' · RUNNING since 2026-11-08 21:28:59 (0 cmd, 0 edits, RECONNECTING since 21:28 (2 tries))']")
CLOCK_B = CLOCK_A.replace("21:28:59", "21:28:57")


@pytest.mark.parametrize("left, right", [
    (clock_block(CLOCK_A), clock_block(CLOCK_B)),
    (clock_block("tmp_path = PosixPath('/tmp/t0')\n" + CLOCK_A), clock_block("tmp_path = PosixPath('/tmp/t0')\n" + CLOCK_B)),
    # The head of a later entry, after pytest's `_ _ _` separator.
    (clock_block("x = 1") + "\n_ _ _ _ _ _ _ _\n\n" + CLOCK_A, clock_block("x = 1") + "\n_ _ _ _ _ _ _ _\n\n" + CLOCK_B),
])
def test_clock_times_in_an_entrys_object_args_are_noise(left, right):
    assert cf.normalize(left) == cf.normalize(right)


@pytest.mark.parametrize("left, right", [
    # A plain literal argument is a value under test.
    (clock_block("at = '12:30'"), clock_block("at = '12:45'")),
    (clock_block("state = [1], at = '12:30'"), clock_block("state = [1], at = '12:45'")),
    # Captured output after the traceback, even shaped as `name = value`.
    (clock_block("x = 1", "----- Captured stdout call -----\nactual = '12:30'"),
     clock_block("x = 1", "----- Captured stdout call -----\nactual = '12:45'")),
    (clock_block("x = 1", "----- Captured stdout call -----\nactual = ['since 12:30']"),
     clock_block("x = 1", "----- Captured stdout call -----\nactual = ['since 12:45']")),
    # A separator or a chained-exception line in captured output opens no entry.
    (clock_block("x = 1", "----- Captured stdout call -----\n_ _ _ _ _ _\nactual = ['12:30']"),
     clock_block("x = 1", "----- Captured stdout call -----\n_ _ _ _ _ _\nactual = ['12:45']")),
    (clock_block("x = 1", "----- Captured stderr call -----\n"
                 "During handling of the above exception, another exception occurred:\nactual = ['12:30']"),
     clock_block("x = 1", "----- Captured stderr call -----\n"
                 "During handling of the above exception, another exception occurred:\nactual = ['12:45']")),
    # An `E` line of the verdict.
    (clock_block(CLOCK_A).replace("E       AssertionError", "E       assert '21:28' == '21:29'"),
     clock_block(CLOCK_A).replace("E       AssertionError", "E       assert '21:28' == '21:30'")),
])
def test_clock_times_elsewhere_tell_failures_apart(left, right):
    assert cf.normalize(left) != cf.normalize(right)


@pytest.mark.parametrize("left, right", [
    ("assert 1 == 2", "assert 1 == 3"),
    ("tests/x.py:74: E", "tests/x.py:75: E"),
    ("E   assert elapsed < 2 s", "E   assert elapsed < 9 s"),
    ("E   assert flags == 0x20", "E   assert flags == 0x30"),
    ("E   assert at == '12:30'", "E   assert at == '12:45'"),
    ("E   at = '12:30'", "E   at = '12:45'"),
    ("at = '12:30'", "at = '12:45'"),
    ("    assert at == '12:30'", "    assert at == '12:45'"),
    ("E   assert s == 'a\\b'", "E   assert s == 'a/b'"),
    # A plain frame's `E` lines are pytest's verdict, not the frame's source.
    ('  File "a.py", line 3, in f\nE   AssertionError: 1', '  File "a.py", line 3, in f\nE   AssertionError: 2'),
])
def test_normalize_keeps_what_tells_failures_apart(left, right):
    assert cf.normalize(left) != cf.normalize(right)


@pytest.mark.parametrize("left, right", [
    ("t[5s]", "t[10s]"), ("test_hex[0x10]", "test_hex[0x20]"), ("t[12:30]", "t[12:45]"),
])
def test_a_node_id_is_compared_exactly(left, right):
    a = {"kind": "pytest", "headline": f"FAILED tests/x.py::{left}", "block": ["E   assert 0"]}
    b = {"kind": "pytest", "headline": f"FAILED tests/x.py::{right}", "block": ["E   assert 0"]}
    c = {"kind": "pytest", "headline": f"FAILED tests/x.py::{left} - took 5s", "block": []}
    d = {"kind": "pytest", "headline": f"FAILED tests/x.py::{right} - took 5s", "block": []}
    assert len(cf.group_failures([("j1", [a]), ("j2", [b]), ("j3", [c]), ("j4", [d])])) == 4


def test_short_names_drop_the_shared_prefix():
    assert cf.short_names([MAC, UBU]) == {MAC: "macos-latest 3.12", UBU: "ubuntu-latest 3.13"}
    assert cf.short_names(["build", MAC]) == {"build": "build", MAC: MAC}


# ------------------------------------------------------------ repo and sha


@pytest.mark.parametrize("url, repo", [
    ("https://github.com/neurocod/llm-loop.git", "neurocod/llm-loop"),
    ("https://github.com/neurocod/llm-loop", "neurocod/llm-loop"),
    ("git@github.com:neurocod/llm-loop.git", "neurocod/llm-loop"),
    ("ssh://git@github.com/neurocod/llm-loop.git\n", "neurocod/llm-loop"),
    ("https://gitlab.com/a/b.git", None),
])
def test_repo_from_remote(url, repo):
    assert cf.repo_from_remote(url) == repo


class Fake:
    """`subprocess.run` stand-in: the first rule whose prefix starts the
    argv's words after the executable answers; a list answers in turn."""

    def __init__(self, rules):
        self.rules, self.calls, self.kwargs = rules, [], []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        self.kwargs.append(kwargs)
        words = " ".join(str(a) for a in argv[1:])
        for prefix, answer in self.rules:
            if words.startswith(prefix):
                if isinstance(answer, list):
                    answer = answer.pop(0) if len(answer) > 1 else answer[0]
                if isinstance(answer, BaseException):
                    raise answer
                code, out, err = answer
                sink = kwargs.get("stdout")
                if hasattr(sink, "write"):  # stdout into a file: gh's logs
                    sink.write(out.encode("utf-8"))
                    out = None
                return subprocess.CompletedProcess(argv, code, out, err)
        raise AssertionError(f"unexpected call {argv}")


def test_sha_and_repo_come_from_git_in_the_cwd(monkeypatch):
    fake = Fake([("rev-parse HEAD", (0, "a" * 40 + "\n", "")),
                 ("rev-parse --verify --quiet 0ca0abdb^{commit}", (0, "b" * 40 + "\n", "")),
                 ("remote get-url origin", (0, "git@github.com:o/r.git\n", ""))])
    monkeypatch.setattr(subprocess, "run", fake)
    assert cf.derive_sha(None, ".") == "a" * 40
    assert cf.derive_sha("0ca0abdb", ".") == "b" * 40
    assert cf.derive_sha("C" * 40, ".") == "c" * 40
    assert cf.derive_repo(".") == "o/r"


def test_a_too_short_sha_is_a_usage_error(monkeypatch):
    fake = Fake([("rev-parse", (0, "e" * 40 + "\n", ""))])
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(cf.Failure) as exc:
        cf.derive_sha("0ca0ab", ".")
    assert exc.value.code == cf.EXIT_USAGE and fake.calls == []


def test_a_short_sha_git_does_not_know_stays_a_prefix(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Fake([("rev-parse", (1, "", ""))]))
    assert cf.derive_sha("0CA0ABD", ".") == "0ca0abd"
    with pytest.raises(cf.Failure) as exc:
        cf.derive_sha("main~2", ".")
    assert exc.value.code == cf.EXIT_USAGE


def test_no_git_is_a_usage_error_not_a_guess(monkeypatch):
    monkeypatch.setattr(subprocess, "run", Fake([("", (128, "", "fatal: not a git repository"))]))
    with pytest.raises(cf.Failure) as exc:
        cf.derive_sha(None, ".")
    assert exc.value.code == cf.EXIT_USAGE
    with pytest.raises(cf.Failure) as exc:
        cf.derive_repo(".")
    assert exc.value.code == cf.EXIT_USAGE


# ------------------------------------------------------------ exit codes


@pytest.fixture
def gh(tmp_path):
    path = tmp_path / "gh.exe"
    path.write_text("")
    return str(path)


def main(monkeypatch, rules, argv):
    fake = Fake(rules)
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(cf.time, "sleep", lambda s: None)
    return cf.main(argv), fake


AUTH = ("auth status", (0, "Logged in", ""))
LOG_FAILED = ("run view 42 --log-failed", (0, LOG, ""))


def view(run):
    return ("run view 42 --json", (0, json.dumps(run), ""))


def run42(gh, *extra):
    """The command line judging run 42 of o/r with the fake gh."""
    return ["--gh", gh, "--repo", "o/r", "--run", "42", *(str(a) for a in extra)]


def test_red_run_prints_jobs_groups_and_raw_logs(monkeypatch, capsys, gh, tmp_path):
    raw = tmp_path / "raw"
    code, fake = main(monkeypatch, [AUTH, view(run_payload()), LOG_FAILED], run42(gh, "--raw-dir", raw))
    out = capsys.readouterr().out
    assert code == cf.EXIT_RED
    assert "pytest (ubuntu-latest, 3.10): success\n" in out
    assert f"{MAC}: failure @ Test -- 2 failed, 2404 passed" in out
    assert f"{OLD}: failure @ Install\n" in out
    assert "3 distinct failure(s) in 4 failed job(s)" in out
    assert ("[1] FAILED tests/test_cpu_load.py::test_stream_exits_0_when_its_reader_closes_the_pipe"
            " -- macos-latest 3.12, ubuntu-latest 3.13, windows-latest 3.12") in out
    assert "no pytest failures in this job's log" in out
    assert sorted(os.listdir(raw)) == sorted(
        ["pytest_macos-latest_3.12.log", "pytest_ubuntu-latest_3.13.log",
         "pytest_windows-latest_3.12.log", "pytest_ubuntu-latest_3.9.log", "report.txt"])
    assert (raw / "pytest_ubuntu-latest_3.13.log").read_text(encoding="utf-8").startswith("\ufeff2026-")
    assert all("--repo" in c and "o/r" in c for c in fake.calls[1:])


def test_max_lines_cuts_the_blocks_and_names_the_full_report(monkeypatch, capsys, gh, tmp_path):
    raw = tmp_path / "raw"
    code, _ = main(monkeypatch, [AUTH, view(run_payload()), LOG_FAILED],
                   run42(gh, "--raw-dir", raw, "--max-lines", 30))
    out = capsys.readouterr().out.splitlines()
    report = raw / "report.txt"
    full = report.read_text(encoding="utf-8").splitlines()
    assert code == cf.EXIT_RED and len(out) == 30
    assert out[:28] == full[:28]
    assert out[-2].startswith("... failure blocks cut at ") and out[-1] == f"full report: {report}"
    assert "ModuleNotFoundError: No module named '_winapi'" in report.read_text(encoding="utf-8")


# Made up: run 42 with 30 failures, each with a 20-line block, then run 7 with one.
MANY = "\n".join(["=== FAILURES ==="]
                 + [ln for i in range(30) for ln in [f"___ test_{i} ___"] + [f"line {k} of {i}" for k in range(20)]]
                 + ["=== short test summary info ==="] + [f"FAILED tests/x.py::test_{i} - boom {i}" for i in range(30)]
                 + ["30 failed in 1.00s", "##[error]Process completed with exit code 1."])


@pytest.mark.parametrize("max_lines", [1, 80])
def test_every_runs_index_is_printed_whatever_the_cap(monkeypatch, capsys, gh, tmp_path, max_lines):
    runs = [listed(42, created="2026-11-08T11:00:00Z"), listed(7, workflow="lint")]
    code, _ = main(monkeypatch, [AUTH, runs_rule(runs), view_of(42, jobs=[job(MAC)]),
                                 ("run view 42 --log-failed", (0, prefixed(MAC, MANY), "")),
                                 view_of(7, jobs=[job("lint", step="ruff")]),
                                 ("run view 7 --log-failed", (0, prefixed("lint", "E501 too long\n"
                                  "##[error]Process completed with exit code 1."), ""))],
                   ["--gh", gh, *by_sha("--raw-dir", tmp_path, "--max-lines", max_lines)])
    out = capsys.readouterr().out
    assert code == cf.EXIT_RED
    for i in range(30):
        assert f"] FAILED tests/x.py::test_{i} -- " in out
    lint = "  [1] no pytest failures in this job's log; its last lines -- lint"
    assert lint in out
    if max_lines > 1:  # the blocks follow every index
        assert out.index(lint) < out.index("line 0 of 0")
    assert out.rstrip().endswith(f"full report: {tmp_path / 'report.txt'}")


def test_json_output(monkeypatch, capsys, gh, tmp_path):
    code, _ = main(monkeypatch, [AUTH, view(run_payload()), LOG_FAILED], run42(gh, "--raw-dir", tmp_path, "--json"))
    data = json.loads(capsys.readouterr().out)
    assert code == data["exit"] == cf.EXIT_RED
    # The shape the module docstring documents.
    assert data["schema"] == cf.JSON_SCHEMA == 1
    assert set(data) == {"schema", "exit", "rawDir", "report", "runs"}
    [run] = data["runs"]
    assert set(run) == {"run", "exit", "note", "jobs", "failures"}
    assert set(run["jobs"][0]) == {"name", "status", "conclusion", "failedSteps", "outcome", "rawLog"}
    assert set(run["failures"][0]) == {"kind", "headline", "jobs", "shownFrom", "block", "savedLog"}
    assert run["exit"] == cf.EXIT_RED
    assert [f["jobs"] for f in run["failures"]][0] == [MAC, UBU, WIN]
    assert {j["name"]: j["failedSteps"] for j in run["jobs"]}[OLD] == ["Install"]


def test_green_run_exits_0_without_fetching_logs(monkeypatch, capsys, gh, tmp_path):
    green = run_payload(conclusion="success", jobs=[job(MAC, "success")])
    code, fake = main(monkeypatch, [AUTH, view(green)], run42(gh, "--raw-dir", tmp_path))
    assert code == cf.EXIT_GREEN
    assert not any("--log-failed" in c for c in fake.calls)
    assert f"{MAC}: success" in capsys.readouterr().out


WORKFLOW_IDS = {"tests": 336505049, "lint": 336505050}


def listed(run_id, workflow="tests", created="2026-11-08T10:00:00Z", event="push", sha="d" * 40,
           workflow_id=None, path=None):
    """One run as `gh api .../actions/runs --jq RUNS_JQ` lists it."""
    return {"id": run_id, "workflow_id": workflow_id or WORKFLOW_IDS.get(workflow, 1),
            "name": workflow, "path": path or f".github/workflows/{workflow}.yml", "event": event,
            "head_sha": sha, "created_at": created}


def runs_rule(runs, sha="d" * 40, page=1, total=None):
    """The fake gh's answer to page `page` of the commit's run listing."""
    body = {"total_count": len(runs) if total is None else total, "workflow_runs": runs}
    return (f"api repos/o/r/actions/runs?head_sha={sha}&per_page=100&page={page} ", (0, json.dumps(body), ""))


def by_sha(*extra):
    return ["--repo", "o/r", "--sha", "d" * 40, *(str(a) for a in extra)]


def view_of(run_id, **kw):
    run = run_payload(**kw)
    run["databaseId"] = run_id
    return (f"run view {run_id} --json", (0, json.dumps(run), ""))


GREEN = {"conclusion": "success", "jobs": [job(MAC, "success")]}


def test_newest_run_for_the_sha_is_chosen(monkeypatch, capsys, gh, tmp_path):
    runs = [listed(41), listed(42, created="2026-11-08T11:00:00Z")]
    code, fake = main(monkeypatch, [AUTH, runs_rule(runs), view_of(42, **GREEN)],
                      ["--gh", gh, *by_sha("--raw-dir", tmp_path)])
    assert code == cf.EXIT_GREEN
    assert not any("41" in c for c in fake.calls)  # no argument is the older run's id


def test_every_page_of_the_commit_runs_is_read(monkeypatch, capsys, gh, tmp_path):
    # The red run is on page 2: a first-page-only listing would judge green.
    page1 = [listed(100 + i, created=f"2026-11-08T11:{i:02d}:00Z") for i in range(100)]
    page2 = [listed(7, workflow="lint", created="2026-11-08T09:00:00Z")]
    code, fake = main(monkeypatch, [AUTH, runs_rule(page1, total=101), runs_rule(page2, page=2, total=101),
                                    view_of(199, **GREEN), view_of(7, jobs=[job("lint", step="ruff")]),
                                    ("run view 7 --log-failed", (0, prefixed("lint", "E501 too long"), ""))],
                      ["--gh", gh, *by_sha("--raw-dir", tmp_path)])
    assert code == cf.EXIT_RED and "E501 too long" in capsys.readouterr().out


@pytest.mark.parametrize("pages", [
    [(100, 1500)] * 10,           # over the 1000 the API serves for a head_sha query
    [(100, 150), (0, 150)],       # a page came back empty before the stated count
    [(100, 101), (0, 100)],       # a run deleted while paging: the oldest was never read
    [(100, 101), (2, 102)],       # a run created while paging
])
def test_a_listing_that_cannot_be_shown_complete_is_never_green(monkeypatch, capsys, gh, tmp_path, pages):
    rules, made = [AUTH], 0
    for page, (count, total) in enumerate(pages, 1):
        runs = [listed(1000 + made + i, created="2026-11-08T10:00:00Z") for i in range(count)]
        made += count
        rules.append(runs_rule(runs, page=page, total=total))
    rules.append(("run view", (0, json.dumps(run_payload(**GREEN)), "")))
    code, fake = main(monkeypatch, rules, ["--gh", gh, *by_sha("--raw-dir", tmp_path)])
    assert code == cf.EXIT_INCOMPLETE and "a red one may be missing" in capsys.readouterr().err
    assert not any("view" in c for c in fake.calls)
    assert sum("api" in c for c in fake.calls) == len(pages)


def test_two_workflows_with_one_display_name_are_both_judged(monkeypatch, capsys, gh, tmp_path):
    runs = [listed(42, created="2026-11-08T11:00:00Z", workflow_id=1, path=".github/workflows/a.yml"),
            listed(7, created="2026-11-08T09:00:00Z", workflow_id=2, path=".github/workflows/b.yml")]
    code, _ = main(monkeypatch, [AUTH, runs_rule(runs), view_of(42, **GREEN),
                                 view_of(7, jobs=[job("lint", step="ruff")]),
                                 ("run view 7 --log-failed", (0, prefixed("lint", "E501 too long"), ""))],
                   ["--gh", gh, *by_sha("--raw-dir", tmp_path)])
    assert code == cf.EXIT_RED and "2 runs for commit" in capsys.readouterr().out


@pytest.mark.parametrize("workflow", ["tests", "tests.yml", ".github/workflows/tests.yml", "336505049"])
def test_workflow_by_name_file_or_id(monkeypatch, capsys, gh, tmp_path, workflow):
    runs = [listed(42), listed(7, workflow="lint", created="2026-11-08T11:00:00Z")]
    code, fake = main(monkeypatch, [AUTH, runs_rule(runs), view_of(42, **GREEN)],
                      ["--gh", gh, *by_sha("--workflow", workflow, "--raw-dir", tmp_path)])
    assert code == cf.EXIT_GREEN and not any("run view 7" in " ".join(c) for c in fake.calls)


def test_a_prefix_git_does_not_know_is_resolved_by_the_repository(monkeypatch, capsys, gh, tmp_path):
    full = "0ca0abd" + "1" * 33
    code, fake = main(monkeypatch, [AUTH, ("rev-parse", (1, "", "")),
                                    ("api repos/o/r/commits/0ca0abd --jq {sha}", (0, json.dumps({"sha": full}), "")),
                                    runs_rule([listed(42, sha=full)], sha=full), view_of(42, **GREEN)],
                      ["--gh", gh, "--repo", "o/r", "--sha", "0ca0abd", "--raw-dir", str(tmp_path)])
    assert code == cf.EXIT_GREEN


def test_every_workflow_and_event_of_the_commit_is_judged(monkeypatch, capsys, gh, tmp_path):
    runs = [listed(41), listed(42, created="2026-11-08T11:00:00Z"),
            listed(7, workflow="lint", created="2026-11-08T09:00:00Z"),
            listed(9, event="pull_request", created="2026-11-08T08:00:00Z")]
    code, fake = main(monkeypatch, [AUTH, runs_rule(runs), view_of(42, **GREEN), view_of(7, **GREEN),
                                    view_of(9, status="in_progress", jobs=[job(UBU, status="in_progress")])],
                      ["--gh", gh, *by_sha("--raw-dir", tmp_path)])
    out = capsys.readouterr().out
    assert code == cf.EXIT_RUNNING
    assert "3 runs for commit 0ca0abdb:" in out
    for run_id in (42, 7, 9):
        assert f"run {run_id} " in out
    assert "run is in_progress; pass --wait" in out


def test_one_red_run_of_the_commit_makes_it_red(monkeypatch, capsys, gh, tmp_path):
    runs = [listed(42, created="2026-11-08T11:00:00Z"), listed(7, workflow="lint")]
    code, _ = main(monkeypatch, [AUTH, runs_rule(runs), view_of(42, **GREEN),
                                 view_of(7, jobs=[job("lint", step="ruff")]),
                                 ("run view 7 --log-failed", (0, prefixed("lint", "E501 too long\n"
                                  "##[error]Process completed with exit code 1."), ""))],
                   ["--gh", gh, *by_sha("--raw-dir", tmp_path / "raw")])
    out = capsys.readouterr().out
    assert code == cf.EXIT_RED and "E501 too long" in out and "lint: failure @ ruff" in out
    assert sorted(os.listdir(tmp_path / "raw")) == ["7_lint.log", "report.txt"]


def test_a_sha_prefix_matching_no_commit_or_two_is_not_found(monkeypatch, capsys, gh):
    stderr = "gh: No commit found for SHA: 0ca0abd (HTTP 422)"
    code, fake = main(monkeypatch, [AUTH, ("rev-parse", (1, "", "")),
                                    ("api repos/o/r/commits/0ca0abd", (1, "", stderr))],
                      ["--gh", gh, "--repo", "o/r", "--sha", "0ca0abd"])
    assert code == cf.EXIT_NOT_FOUND and "more than one" in capsys.readouterr().err
    assert not any("view" in c or "actions/runs" in " ".join(c) for c in fake.calls)


def test_no_run_for_the_sha_exits_not_found(monkeypatch, capsys, gh):
    code, _ = main(monkeypatch, [AUTH, runs_rule([])], ["--gh", gh, *by_sha()])
    assert code == cf.EXIT_NOT_FOUND
    assert "not pushed yet?" in capsys.readouterr().err


def test_missing_run_id_exits_not_found(monkeypatch, gh):
    stderr = "failed to get run: HTTP 404: Not Found (https://api.github.com/repos/o/r/actions/runs/42)"
    code, _ = main(monkeypatch, [AUTH, ("run view 42", (1, "", stderr))], run42(gh))
    assert code == cf.EXIT_NOT_FOUND


def test_running_run_without_wait_exits_running(monkeypatch, capsys, gh):
    running = run_payload(status="in_progress", jobs=[job(MAC, status="in_progress")])
    code, _ = main(monkeypatch, [AUTH, view(running)], run42(gh))
    out = capsys.readouterr().out
    assert code == cf.EXIT_RUNNING
    assert "run is in_progress; pass --wait" in out and f"{MAC}: in_progress" in out


def test_wait_that_times_out_exits_running(monkeypatch, gh):
    running = run_payload(status="queued", jobs=[])
    clock = Clock()
    monkeypatch.setattr(cf.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(subprocess, "run", Fake([AUTH, view(running)]))
    monkeypatch.setattr(cf.time, "sleep", clock.sleep)
    assert cf.main(run42(gh, "--wait", "--timeout", 10)) == cf.EXIT_RUNNING


def test_wait_polls_until_the_run_completes(monkeypatch, gh, tmp_path):
    answers = [(0, json.dumps(run_payload(status="in_progress", jobs=[])), ""),
               (0, json.dumps(run_payload(status="in_progress", jobs=[])), ""),
               (0, json.dumps(run_payload(conclusion="success", jobs=[job(MAC, "success")])), "")]
    code, fake = main(monkeypatch, [AUTH, ("run view 42 --json", answers)],
                      run42(gh, "--wait", "--raw-dir", tmp_path))
    assert code == cf.EXIT_GREEN
    assert sum("--json" in c for c in fake.calls) == 3


def test_gh_not_installed_exits_no_gh(monkeypatch, capsys, tmp_path):
    code, fake = main(monkeypatch, [], ["--gh", str(tmp_path / "absent.exe"), "--repo", "o/r", "--run", "1"])
    assert code == cf.EXIT_NO_GH and fake.calls == []
    assert "https://cli.github.com/" in capsys.readouterr().err


def test_gh_found_nowhere_exits_no_gh(monkeypatch):
    monkeypatch.setattr(cf.shutil, "which", lambda name: None)
    monkeypatch.setattr(cf.os.path, "isfile", lambda p: False)
    assert cf.find_gh(None) is None
    code, _ = main(monkeypatch, [], ["--repo", "o/r", "--run", "1"])
    assert code == cf.EXIT_NO_GH


def test_gh_logged_out_exits_no_gh(monkeypatch, capsys, gh):
    code, fake = main(monkeypatch, [("auth status", (1, "", "You are not logged into any GitHub hosts."))],
                      run42(gh))
    assert code == cf.EXIT_NO_GH and len(fake.calls) == 1
    assert "gh auth login" in capsys.readouterr().err


@pytest.mark.parametrize("answer", [
    (1, "", "HTTP 502: Bad Gateway"),
    (0, "not json", ""),
    subprocess.TimeoutExpired(["gh"], 120),
])
def test_other_gh_failures_exit_gh_error(monkeypatch, gh, answer):
    code, _ = main(monkeypatch, [AUTH, ("run view 42", answer)], run42(gh))
    assert code == cf.EXIT_GH_ERROR


def test_no_repo_and_no_remote_exits_usage(monkeypatch, capsys, gh):
    code, _ = main(monkeypatch, [("remote get-url origin", (2, "", "error: No such remote 'origin'"))],
                   ["--gh", gh, "--run", "42"])
    assert code == cf.EXIT_USAGE
    assert "--repo" in capsys.readouterr().err


def test_exit_codes_are_distinct():
    codes = [cf.EXIT_GREEN, cf.EXIT_RED, cf.EXIT_USAGE, cf.EXIT_NOT_FOUND, cf.EXIT_RUNNING,
             cf.EXIT_NO_GH, cf.EXIT_GH_ERROR, cf.EXIT_INCOMPLETE, cf.EXIT_INTERNAL]
    assert len(set(codes)) == len(codes)


# ------------------------------------------------------------ green needs complete data


def any_log():
    """Rules answering every log call: an empty --log-failed and, for each
    job, its own log of one runner error line."""
    return [("run view 42 --log-failed", (0, "", "")),
            ("api repos/o/r/actions/jobs/", (0, "2026-11-09T00:00:00.0000000Z ##[error]boom\n", ""))]


def jobs_of(*conclusions):
    """Jobs j0, j1, ... with these conclusions; `in_progress` is a status."""
    return [job(f"j{i}", "" if c == "in_progress" else c,
                status="in_progress" if c == "in_progress" else "completed", job_id=1000 + i)
            for i, c in enumerate(conclusions)]


@pytest.mark.parametrize("conclusion, jobs, expected", [
    ("success", jobs_of("success"), "GREEN"),
    ("success", jobs_of("success", "skipped", "neutral"), "GREEN"),
    ("success", [], "INCOMPLETE"),
    ("success", None, "INCOMPLETE"),                    # no "jobs" key at all
    ("success", jobs_of("skipped"), "INCOMPLETE"),      # nothing ran
    ("success", jobs_of("success", ""), "INCOMPLETE"),  # a job without a conclusion
    ("success", jobs_of("success", "in_progress"), "INCOMPLETE"),
    ("success", jobs_of("success", "failure"), "RED"),  # a failed job beats the run's word
    ("success", jobs_of("cancelled"), "RED"),
    ("success", jobs_of("timed_out"), "RED"),
    ("success", jobs_of("startup_failure"), "RED"),
    ("success", jobs_of("action_required"), "RED"),
    ("failure", jobs_of("failure"), "RED"),
    ("failure", [], "RED"),
    ("cancelled", jobs_of("success", "cancelled"), "RED"),
    ("timed_out", jobs_of("success"), "RED"),
    ("startup_failure", [], "RED"),
    ("action_required", [], "INCOMPLETE"),
    ("neutral", jobs_of("neutral"), "INCOMPLETE"),
    ("skipped", jobs_of("skipped"), "INCOMPLETE"),
    ("stale", jobs_of("success"), "INCOMPLETE"),
    ("", jobs_of("success"), "INCOMPLETE"),
])
def test_conclusion_matrix(monkeypatch, capsys, gh, tmp_path, conclusion, jobs, expected):
    run = run_payload(conclusion=conclusion, jobs=jobs if jobs is not None else [])
    if jobs is None:
        del run["jobs"]
    code, _ = main(monkeypatch, [AUTH, view(run)] + any_log(), run42(gh, "--raw-dir", tmp_path))
    # By name: the module of 713cfed had no EXIT_INCOMPLETE, and must still collect.
    assert code == getattr(cf, f"EXIT_{expected}")


# ------------------------------------------------------------ what a report must not lose


def test_uncapped_report_keeps_the_whole_headline(monkeypatch, capsys, gh, tmp_path):
    message = "AssertionError: " + "x" * 400
    body = "\n".join(["=== short test summary info ===", f"FAILED tests/x.py::t - {message}",
                      "1 failed in 0.10s", "##[error]Process completed with exit code 1."])
    code, _ = main(monkeypatch, [AUTH, view(run_payload(jobs=[job(MAC)])),
                                 ("run view 42 --log-failed", (0, prefixed(MAC, body), ""))],
                   run42(gh, "--raw-dir", tmp_path))
    out = capsys.readouterr().out
    assert code == cf.EXIT_RED
    assert message not in out and "x" * 200 + "..." in out
    assert message in (tmp_path / "report.txt").read_text(encoding="utf-8")


def test_the_index_names_every_failure_before_the_cap(monkeypatch, capsys, gh, tmp_path):
    code, _ = main(monkeypatch, [AUTH, view(run_payload()), LOG_FAILED],
                   run42(gh, "--raw-dir", tmp_path, "--max-lines", 14))
    out = capsys.readouterr().out
    assert code == cf.EXIT_RED
    assert "  [1] FAILED tests/test_cpu_load.py::" in out
    assert "  [2] FAILED tests/codex_agent/test_codex_agent.py::" in out
    assert "  [3] no pytest failures in this job's log; its last lines -- ubuntu-latest 3.9" in out


def test_a_cancelled_job_missing_from_log_failed_shows_its_own_log(monkeypatch, capsys, gh, tmp_path):
    cancelled = job(UBU, "cancelled", job_id=2002)
    own = own_log(["Running tests", "##[error]The operation was canceled.", "Post job cleanup."])
    gone = job(WIN, "timed_out", job_id=2003)
    stderr = "HTTP 410: Gone (https://api.github.com/repos/o/r/actions/jobs/2003/logs)"
    rules = [AUTH, view(run_payload(jobs=[job(MAC), cancelled, gone])),
             ("run view 42 --log-failed", (0, prefixed(MAC, TAIL_OF_JOB), "")),
             ("api repos/o/r/actions/jobs/2002/logs", (0, own, "")),
             ("api repos/o/r/actions/jobs/2003/logs", (1, "", stderr))]
    code, fake = main(monkeypatch, rules, run42(gh, "--raw-dir", tmp_path))
    out = capsys.readouterr().out
    assert code == cf.EXIT_RED
    assert "##[error]The operation was canceled." in out and "own log" in out
    assert "HTTP 410: Gone" in out
    assert (tmp_path / "pytest_ubuntu-latest_3.13.log").read_text(encoding="utf-8") == own


def test_a_running_run_shows_the_failures_already_known(monkeypatch, capsys, gh, tmp_path):
    running = run_payload(status="in_progress", jobs=[job(MAC, job_id=1001), job(UBU, status="in_progress")])
    own = own_log(MAC_BODY.split("\n"))
    code, fake = main(monkeypatch, [AUTH, view(running), ("api repos/o/r/actions/jobs/1001/logs", (0, own, ""))],
                      run42(gh, "--raw-dir", tmp_path))
    out = capsys.readouterr().out
    assert code == cf.EXIT_RUNNING
    assert "run is in_progress; pass --wait" in out
    assert "[1] FAILED tests/test_cpu_load.py::test_stream_exits_0_when_its_reader_closes_the_pipe" in out
    assert f"{MAC}: failure @ Test -- 2 failed, 2404 passed" in out
    assert not any("--log-failed" in c for c in fake.calls)


@pytest.fixture
def long_dir(tmp_path):
    """A raw dir of at least 150 characters: with the ~110 of the message
    before it, a path in the headline would pass HEADLINE_CHARS (240). Not
    much longer: a Windows path over 260 characters needs long paths enabled."""
    path = tmp_path / ("long_" + "d" * max(0, 150 - len(str(tmp_path))))
    assert len(str(path)) >= 150
    return path


def test_a_log_over_the_size_limit_is_named_not_parsed(monkeypatch, capsys, gh, long_dir):
    monkeypatch.setattr(cf, "MAX_LOG_BYTES", 100)
    code, _ = main(monkeypatch, [AUTH, view(run_payload()), LOG_FAILED], run42(gh, "--raw-dir", long_dir))
    out = capsys.readouterr().out.splitlines()
    saved = long_dir / "log-failed.txt"
    assert code == cf.EXIT_RED
    assert any("over the 100-byte limit (MAX_LOG_BYTES)" in ln for ln in out)
    assert f"      saved log: {saved}" in out    # in the index, whole, on its own line
    assert saved.read_text(encoding="utf-8") == LOG


def test_a_job_log_over_the_limit_keeps_its_own_path(monkeypatch, capsys, gh, long_dir):
    monkeypatch.setattr(cf, "MAX_LOG_BYTES", 100)
    jobs = [job(UBU, "cancelled", job_id=2002), job(WIN, "cancelled", job_id=2003)]
    big = own_log(["x" * 200, "##[error]The operation was canceled."])
    rules = [AUTH, view(run_payload(jobs=jobs)), ("run view 42 --log-failed", (0, "", "")),
             ("api repos/o/r/actions/jobs/", (0, big, ""))]
    code, _ = main(monkeypatch, rules, run42(gh, "--raw-dir", long_dir, "--json"))
    data = json.loads(capsys.readouterr().out)
    assert code == cf.EXIT_RED
    assert sorted(f["savedLog"] for f in data["runs"][0]["failures"]) == sorted(
        str(long_dir / n) for n in ("pytest_ubuntu-latest_3.13.log", "pytest_windows-latest_3.12.log"))


def test_a_multi_step_failure_names_every_step(monkeypatch, capsys, gh, tmp_path):
    two = job(MAC)
    two["steps"].append({"name": "Deploy", "conclusion": "failure"})
    log = prefixed(MAC, "\n".join(DEPLOY_AFTER))
    code, _ = main(monkeypatch, [AUTH, view(run_payload(jobs=[two])), ("run view 42 --log-failed", (0, log, ""))],
                   run42(gh, "--raw-dir", tmp_path))
    out = capsys.readouterr().out
    assert code == cf.EXIT_RED
    assert f"{MAC}: failure @ Test, Deploy" in out and "deploy: permission denied" in out


# ------------------------------------------------------------ --timeout, crashes, gh


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def sleep(self, s):
        self.now += s


@pytest.mark.parametrize("timeout, bounds", [(30, []), (45, [15]), (100, [70, 40, 10])])
def test_wait_never_outlives_the_timeout(monkeypatch, gh, timeout, bounds):
    clock = Clock()
    fake = Fake([AUTH, view(run_payload(status="in_progress", jobs=[]))])
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(cf.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cf.time, "sleep", clock.sleep)
    code = cf.main(run42(gh, "--wait", "--timeout", timeout))
    assert code == cf.EXIT_RUNNING
    views = [kw["timeout"] for argv, kw in zip(fake.calls, fake.kwargs) if "view" in argv]
    assert views[0] == min(cf.GH_TIMEOUT_S, timeout) and views[1:] == bounds
    assert clock.now <= 1000.0 + timeout


class SlowGh(Fake):
    """A Fake whose every call takes `cost` seconds of `clock`, or times out
    (as subprocess.run does) when its timeout is shorter."""

    def __init__(self, rules, clock, cost):
        super().__init__(rules)
        self.clock, self.cost, self.left = clock, cost, []

    def __call__(self, argv, **kwargs):
        self.left.append((kwargs.get("timeout"), self.deadline - self.clock.now))
        if kwargs.get("timeout") is not None and kwargs["timeout"] < self.cost:
            self.calls.append(argv)
            self.clock.now += kwargs["timeout"]
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        self.clock.now += self.cost
        return super().__call__(argv, **kwargs)


@pytest.mark.parametrize("timeout, cost, expected, said", [
    # auth and the view eat the deadline: the view itself is cut short.
    (50, 30, cf.EXIT_GH_ERROR, "--timeout ran out during gh run view 42"),
    # auth and the view fit; the download is cut short: red stays red, and
    # the report says why the failures' text is missing.
    (100, 45, cf.EXIT_RED, "--timeout ran out during gh run view 42 --log-failed"),
])
def test_one_deadline_bounds_every_gh_call(monkeypatch, capsys, gh, tmp_path, timeout, cost, expected, said):
    clock = Clock()
    rules = [AUTH, view(run_payload()), ("run view 42 --log-failed", (0, prefixed(MAC, "x"), ""))]
    fake = SlowGh(rules, clock, cost)
    fake.deadline = 1000.0 + timeout
    monkeypatch.setattr(subprocess, "run", fake)
    monkeypatch.setattr(cf.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(cf.time, "sleep", clock.sleep)
    code = cf.main(run42(gh, "--timeout", timeout, "--raw-dir", tmp_path))
    captured = capsys.readouterr()
    assert code == expected and code != cf.EXIT_GREEN
    assert said in captured.out + captured.err
    assert all(t <= left for t, left in fake.left)
    assert clock.now <= 1000.0 + timeout


def test_an_unexpected_exception_is_an_internal_error_not_red(monkeypatch, capsys):
    def boom(args):
        raise KeyError("jobs")
    monkeypatch.setattr(cf, "answer", boom)
    code = cf.main(["--repo", "o/r", "--run", "1"])
    err = capsys.readouterr().err
    assert code == cf.EXIT_INTERNAL and code != cf.EXIT_RED
    assert "Traceback" in err and "KeyError" in err


def test_gh_given_by_name_is_looked_up_on_path(monkeypatch):
    monkeypatch.setattr(cf.shutil, "which", lambda name: "/usr/local/bin/gh2" if name == "gh2" else None)
    assert cf.find_gh("gh2") == "/usr/local/bin/gh2"
    assert cf.find_gh("absent-gh") is None


def test_auth_is_checked_for_github_com(monkeypatch, gh, tmp_path):
    green = run_payload(conclusion="success", jobs=[job(MAC, "success")])
    _, fake = main(monkeypatch, [AUTH, view(green)], run42(gh, "--raw-dir", tmp_path))
    assert fake.calls[0][1:] == ["auth", "status", "--hostname", "github.com"]


@pytest.mark.parametrize("answer, said", [
    (OSError("[WinError 2] cannot find the file"), "cannot start"),
    (subprocess.TimeoutExpired(["gh"], 120), "did not answer within 120 s"),
])
def test_gh_start_failure_and_timeout_are_told_apart(monkeypatch, capsys, gh, answer, said):
    code, _ = main(monkeypatch, [AUTH, ("run view 42", answer)], run42(gh))
    assert code == cf.EXIT_GH_ERROR and said in capsys.readouterr().err


def test_no_access_to_the_repository_is_said(monkeypatch, capsys, gh):
    stderr = "HTTP 403: Resource not accessible by integration (https://api.github.com/repos/o/r/actions/runs/42)"
    code, _ = main(monkeypatch, [AUTH, ("run view 42", (1, "", stderr))], run42(gh))
    assert code == cf.EXIT_NO_GH and "the gh login cannot read o/r" in capsys.readouterr().err


def test_default_raw_dir_is_named_by_the_run_and_printed(monkeypatch, capsys, gh, tmp_path):
    temp = tmp_path / "temp"
    temp.mkdir()
    monkeypatch.setattr(cf.tempfile, "tempdir", str(temp))
    code, _ = main(monkeypatch, [AUTH, view(run_payload()), LOG_FAILED], run42(gh))
    [made] = os.listdir(temp)
    assert code == cf.EXIT_RED and made.startswith("ci_failures_42_")
    assert str(temp / made) in capsys.readouterr().out


def test_script_runs_standalone_with_help():
    script = Path(cf.__file__)
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "--wait" in proc.stdout
