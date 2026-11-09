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


def job(name, conclusion="failure", step="Test", status="completed"):
    steps = [{"name": "Install", "conclusion": "success"},
             {"name": step, "conclusion": conclusion if conclusion != "success" else "success"},
             {"name": "Post Run actions/checkout@v4", "conclusion": "success"}]
    return {"name": name, "status": status, "conclusion": conclusion if status == "completed" else "",
            "steps": steps}


def run_json(status="completed", conclusion="failure", jobs=None):
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
    blocks, entries, outcomes = cf.parse_pytest(lines)
    assert [t for t, _ in blocks] == ["test_stream_exits_0_when_its_reader_closes_the_pipe"]
    body = blocks[0][1]
    assert body[0].startswith("    def test_stream")          # leading blank dropped
    assert body[-1] == "tests/test_cpu_load.py:460: Failed"
    assert not any("\x1b" in ln or "\r" in ln for ln in body)
    assert len(entries) == 1 and entries[0][-1] == "ModuleNotFoundError: No module named '_winapi'"
    assert outcomes == ["1 failed, 2404 passed, 79 skipped, 27 subtests passed in 152.79s (0:02:32)"]


def test_chained_traceback_separator_is_not_a_block_header():
    lines = ["=== FAILURES ===", "___ test_a ___", "first", "_ _ _ _ _ _ _ _ _ _", "second",
             "=== short test summary info ===", "FAILED tests/x.py::test_a - boom"]
    blocks, _, _ = cf.parse_pytest(lines)
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
    ("since 2026-11-08 23:58:13, at 23:58", "since 2026-11-09 01:02:03, at 01:02"),
    ("took 1.25s", "took 3.5 s"),
    ("tests\\x.py:74: AssertionError", "tests/x.py:74: AssertionError"),
])
def test_normalize_takes_out_per_job_noise(left, right):
    assert cf.normalize(left) == cf.normalize(right)


def test_normalize_keeps_what_tells_failures_apart():
    assert cf.normalize("assert 1 == 2") != cf.normalize("assert 1 == 3")
    assert cf.normalize("tests/x.py:74: E") != cf.normalize("tests/x.py:75: E")


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
        self.rules, self.calls = rules, []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        words = " ".join(str(a) for a in argv[1:])
        for prefix, answer in self.rules:
            if words.startswith(prefix):
                if isinstance(answer, list):
                    answer = answer.pop(0) if len(answer) > 1 else answer[0]
                if isinstance(answer, BaseException):
                    raise answer
                code, out, err = answer
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


def view(run):
    return ("run view 42 --json", (0, json.dumps(run), ""))


def test_red_run_prints_jobs_groups_and_raw_logs(monkeypatch, capsys, gh, tmp_path):
    raw = tmp_path / "raw"
    code, fake = main(monkeypatch, [AUTH, view(run_json()), ("run view 42 --log-failed", (0, LOG, ""))],
                      ["--gh", gh, "--repo", "o/r", "--run", "42", "--raw-dir", str(raw)])
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


def test_max_lines_cuts_the_print_and_names_the_full_report(monkeypatch, capsys, gh, tmp_path):
    raw = tmp_path / "raw"
    code, _ = main(monkeypatch, [AUTH, view(run_json()), ("run view 42 --log-failed", (0, LOG, ""))],
                   ["--gh", gh, "--repo", "o/r", "--run", "42", "--raw-dir", str(raw), "--max-lines", "10"])
    out = capsys.readouterr().out.splitlines()
    report = raw / "report.txt"
    full = report.read_text(encoding="utf-8").splitlines()
    assert code == cf.EXIT_RED and len(out) == 10
    assert out[:9] == full[:9]
    assert out[-1].startswith("... report cut at 9 of ") and out[-1].endswith(f"full report: {report}")
    assert "ModuleNotFoundError: No module named '_winapi'" in report.read_text(encoding="utf-8")


def test_json_output(monkeypatch, capsys, gh, tmp_path):
    code, _ = main(monkeypatch, [AUTH, view(run_json()), ("run view 42 --log-failed", (0, LOG, ""))],
                   ["--gh", gh, "--repo", "o/r", "--run", "42", "--raw-dir", str(tmp_path), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert code == data["exit"] == cf.EXIT_RED
    assert [f["jobs"] for f in data["failures"]][0] == [MAC, UBU, WIN]
    assert {j["name"]: j["failedStep"] for j in data["jobs"]}[OLD] == "Install"


def test_green_run_exits_0_without_fetching_logs(monkeypatch, capsys, gh, tmp_path):
    green = run_json(conclusion="success", jobs=[job(MAC, "success")])
    code, fake = main(monkeypatch, [AUTH, view(green)],
                      ["--gh", gh, "--repo", "o/r", "--run", "42", "--raw-dir", str(tmp_path)])
    assert code == cf.EXIT_GREEN
    assert not any("--log-failed" in c for c in fake.calls)
    assert f"{MAC}: success" in capsys.readouterr().out


def test_newest_run_for_the_sha_is_chosen(monkeypatch, capsys, gh, tmp_path):
    runs = [{"databaseId": 41, "workflowName": "tests", "createdAt": "2026-11-08T10:00:00Z"},
            {"databaseId": 42, "workflowName": "tests", "createdAt": "2026-11-08T11:00:00Z"},
            {"databaseId": 7, "workflowName": "lint", "createdAt": "2026-11-08T09:00:00Z",
             "conclusion": "success"}]
    green = run_json(conclusion="success", jobs=[job(MAC, "success")])
    code, fake = main(monkeypatch, [AUTH, ("run list", (0, json.dumps(runs), "")), view(green)],
                      ["--gh", gh, "--repo", "o/r", "--sha", "d" * 40, "--raw-dir", str(tmp_path)])
    assert code == cf.EXIT_GREEN
    listing = next(c for c in fake.calls if "list" in c)
    assert listing[listing.index("--commit") + 1] == "d" * 40
    assert "also for this commit: run 7 'lint': success" in capsys.readouterr().out


def test_no_run_for_the_sha_exits_not_found(monkeypatch, capsys, gh):
    code, _ = main(monkeypatch, [AUTH, ("run list", (0, "[]", ""))],
                   ["--gh", gh, "--repo", "o/r", "--sha", "d" * 40])
    assert code == cf.EXIT_NOT_FOUND
    assert "not pushed yet?" in capsys.readouterr().err


def test_missing_run_id_exits_not_found(monkeypatch, gh):
    stderr = "failed to get run: HTTP 404: Not Found (https://api.github.com/repos/o/r/actions/runs/42)"
    code, _ = main(monkeypatch, [AUTH, ("run view 42", (1, "", stderr))],
                   ["--gh", gh, "--repo", "o/r", "--run", "42"])
    assert code == cf.EXIT_NOT_FOUND


def test_running_run_without_wait_exits_running(monkeypatch, capsys, gh):
    running = run_json(status="in_progress", jobs=[job(MAC, status="in_progress")])
    code, _ = main(monkeypatch, [AUTH, view(running)], ["--gh", gh, "--repo", "o/r", "--run", "42"])
    out = capsys.readouterr().out
    assert code == cf.EXIT_RUNNING
    assert "run is in_progress; pass --wait" in out and f"{MAC}: in_progress" in out


def test_wait_that_times_out_exits_running(monkeypatch, gh):
    running = run_json(status="queued", jobs=[])
    code, _ = main(monkeypatch, [AUTH, view(running)],
                   ["--gh", gh, "--repo", "o/r", "--run", "42", "--wait", "--timeout", "0"])
    assert code == cf.EXIT_RUNNING


def test_wait_polls_until_the_run_completes(monkeypatch, gh, tmp_path):
    answers = [(0, json.dumps(run_json(status="in_progress", jobs=[])), ""),
               (0, json.dumps(run_json(status="in_progress", jobs=[])), ""),
               (0, json.dumps(run_json(conclusion="success", jobs=[job(MAC, "success")])), "")]
    code, fake = main(monkeypatch, [AUTH, ("run view 42 --json", answers)],
                      ["--gh", gh, "--repo", "o/r", "--run", "42", "--wait", "--raw-dir", str(tmp_path)])
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
                      ["--gh", gh, "--repo", "o/r", "--run", "42"])
    assert code == cf.EXIT_NO_GH and len(fake.calls) == 1
    assert "gh auth login" in capsys.readouterr().err


@pytest.mark.parametrize("answer", [
    (1, "", "HTTP 502: Bad Gateway"),
    (0, "not json", ""),
    subprocess.TimeoutExpired(["gh"], 120),
])
def test_other_gh_failures_exit_gh_error(monkeypatch, gh, answer):
    code, _ = main(monkeypatch, [AUTH, ("run view 42", answer)], ["--gh", gh, "--repo", "o/r", "--run", "42"])
    assert code == cf.EXIT_GH_ERROR


def test_no_repo_and_no_remote_exits_usage(monkeypatch, capsys, gh):
    code, _ = main(monkeypatch, [("remote get-url origin", (2, "", "error: No such remote 'origin'"))],
                   ["--gh", gh, "--run", "42"])
    assert code == cf.EXIT_USAGE
    assert "--repo" in capsys.readouterr().err


def test_exit_codes_are_distinct():
    codes = [cf.EXIT_GREEN, cf.EXIT_RED, cf.EXIT_USAGE, cf.EXIT_NOT_FOUND, cf.EXIT_RUNNING,
             cf.EXIT_NO_GH, cf.EXIT_GH_ERROR]
    assert len(set(codes)) == len(codes)


def test_script_runs_standalone_with_help():
    script = Path(cf.__file__)
    proc = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert proc.returncode == 0 and "--wait" in proc.stdout
