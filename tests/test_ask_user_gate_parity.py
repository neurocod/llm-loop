"""The ask-user-gate plugin's two implementations must not drift apart.

Until this file existed, `cpp/parity_check.py` was named in a README and run by
whoever remembered. That is a convention, not a mechanism: the live gate on a
machine that opted into the binary is a gitignored `.exe`, so editing
`hooks/ask_user_gate.py` leaves a stale binary guarding the session and nothing
says so -- not `git status`, not a test run. Here, a `pytest` does.

The parity case is skipped where the binary was never built, which is every CI
runner and every fresh clone. A binary older than its source FAILS it instead:
parity_check refuses to compare yesterday's port (newer_sources there). The
script's own `--self-test` is not skipped: it needs nothing but the checkout,
and it is the half that every plugin install actually runs.
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
import threading
import time

import pytest

PLUGIN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "claude-plugin", "ask-user-gate")
SCRIPT = os.path.join(PLUGIN, "hooks", "ask_user_gate.py")
PARITY = os.path.join(PLUGIN, "cpp", "parity_check.py")
BINARY = os.path.join(PLUGIN, "hooks",
                      "ask_user_gate.exe" if os.name == "nt" else "ask_user_gate")


# _run's timeouts, so a hang fails the test that hung instead of the whole
# suite. One launch of the script: `--self-test` 0.8 s alone, and the slowest
# single launch inside two concurrent parity runs 12.1 s and 15.0 s (measured
# 2026-09-29, parity_check.LAUNCH_TIMEOUT). The whole parity_check: 62.8 s
# alone, 108 s and 120 s beside a second run (2026-09-29, 8 jobs); at
# PARITY_JOBS=1 it took 506-772 s beside a second suite (2026-09-28).
SHORT_TIMEOUT = 60
PARITY_TIMEOUT = 1800


def _run(argv, timeout=SHORT_TIMEOUT):
    # stdin closed, not pytest's: whatever this starts never waits on a payload.
    return subprocess.run([sys.executable] + argv, capture_output=True,
                          stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


def test_reference_self_test():
    """The scanner, the wiring and both branches of the path resolver."""
    result = _run([SCRIPT, "--self-test"])
    assert result.returncode == 0, result.stdout + result.stderr


def test_reference_check_file_it_cannot_read_says_so(tmp_path):
    """The message, and no traceback; the exit code is an ARGV_CASES row."""
    result = _run([SCRIPT, "--check-file", str(tmp_path / "missing.txt")])
    assert result.returncode == 2, result.stdout + result.stderr
    assert "cannot read" in result.stderr
    assert "Traceback" not in result.stderr


def _parity_module():
    """parity_check, loaded by path at collection (ARGV_CASES parametrizes).

    As a script it puts hooks/ first on sys.path and imports ask_user_gate by
    name; both are undone here, so the rest of the session resolves those names
    as it would without this file. The module keeps its own reference.
    """
    saved_path = list(sys.path)
    saved_module = sys.modules.get("ask_user_gate")
    try:
        spec = importlib.util.spec_from_file_location("parity_check_ref", PARITY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
        if saved_module is None:
            sys.modules.pop("ask_user_gate", None)
        else:
            sys.modules["ask_user_gate"] = saved_module
    return module


_PARITY = _parity_module()


@pytest.mark.parametrize("argv, code, text", _PARITY.ARGV_CASES,
                         ids=[ascii(case[0]) for case in _PARITY.ARGV_CASES])
def test_reference_command_line(argv, code, text):
    """The reference half of parity_check's ARGV_CASES, binary or not.

    That script compares the two halves only where the port was built; this
    holds the reference to the table's own answers everywhere else, through
    the same argv_verdict, so both read one output the same way.
    """
    # The table's rule, checked here rather than trusted to its comment: a
    # verdict with no text would pass on a crash (exit 1) or on hook mode's
    # silent pass-through (exit 0).
    assert code == 2 or text is not None, "a verdict row must name its text"
    got_code, got_text = _PARITY.argv_verdict([sys.executable, SCRIPT], argv)
    assert not got_text.startswith("<stderr>"), got_text
    assert got_code == code, got_text
    if text is not None:
        assert _PARITY.normalise(got_text) == _PARITY.normalise(text)


# A child that never answers and never reads stdin. 30 s and not forever: a
# regression that loses the timeout is a red after 30 s, not a stalled suite.
SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


@pytest.mark.parametrize("kind", ["check", "argv", "hook"])
def test_a_launch_that_never_answers_is_a_verdict_naming_its_argv(
        kind, monkeypatch, tmp_path):
    """A hung half fails its case by name instead of hanging the suite.

    The hook launch hands the sleeper 1 MB it never reads: the timeout covers
    feeding stdin as well (launch's docstring). That half bites only where
    communicate() writes its input in the calling thread -- Windows through
    3.13, where a pipe held the launch until the sleeper exited (31 s,
    measured 2026-09-29 on 3.13.7); 3.14 passes either way.
    """
    monkeypatch.setattr(_PARITY, "LAUNCH_TIMEOUT", 2)
    start = time.monotonic()
    if kind == "check":
        code, text = _PARITY.check_verdict(SLEEPER, "ls", "bash", "Bash",
                                           str(tmp_path))
        # The --check-file path is a fresh temporary name.
        launched = SLEEPER + ["--check-file", "PATH", "--shell", "bash",
                              "--tool", "Bash", "--platform", "windows"]
    elif kind == "argv":
        code, text = _PARITY.argv_verdict(SLEEPER, ["--check", "ls"])
        launched = SLEEPER + ["--platform", "windows", "--check", "ls"]
    else:
        code, text = _PARITY.hook_verdict(SLEEPER, b"x" * (1 << 20))
        launched = SLEEPER
    # 2.4-3.7 s measured 2026-09-29; the sleeper's own 30 s is the failure.
    elapsed = time.monotonic() - start
    assert elapsed < 20, f"the 2 s timeout took {elapsed:.1f} s"
    assert code == _PARITY.TIMED_OUT, text
    assert not isinstance(code, int), "a hang must not look like a return code"
    expected = re.escape(f"<no answer in 2 s from {ascii(launched)}>")
    assert re.fullmatch(expected.replace("PATH", "[^']+"), text), text


# What _stub_main passes as --exe: any existing file will do, and naming one
# skips the stale-binary check, which only reads the default binary.
GATE = [sys.executable]


def _stub_main(monkeypatch, launch, jobs):
    """parity_check.main() with every launch replaced by `launch`."""
    monkeypatch.setattr(_PARITY, "launch", launch)
    monkeypatch.setattr(sys, "argv", ["parity_check.py", "--exe", GATE[0],
                                      "--jobs", str(jobs)])
    return _PARITY.main()


def _quiet(argv, payload=None):
    """A launch both halves answer alike: exit 0, no output."""
    return subprocess.CompletedProcess(argv, 0, b"", b"")


def test_a_hung_hook_and_self_test_are_reported_by_name(monkeypatch, capsys):
    """The CRASH line and the self-test line for a launch that timed out."""
    def launch(argv, payload=None):
        if argv == GATE and payload == b"[1,2,3]":
            return None
        if argv == GATE + ["--self-test"]:
            return None
        return _quiet(argv)

    assert _stub_main(monkeypatch, launch, jobs=4) == 1
    out, err = capsys.readouterr()
    timeout = _PARITY.LAUNCH_TIMEOUT
    assert (f"CRASH [hook] top-level array: c++ <no answer in {timeout} s "
            f"from {ascii(GATE)}>\n") in err
    assert (f"c++ --self-test: <no answer in {timeout} s from "
            f"{ascii(GATE + ['--self-test'])}>\n") in out
    assert "python --self-test: \n" in out


def test_a_crashed_hook_names_each_code(monkeypatch, capsys):
    def launch(argv, payload=None):
        if payload == b"[1,2,3]":
            return subprocess.CompletedProcess(argv, 3 if argv == GATE else 1,
                                               b"", b"")
        return _quiet(argv)

    assert _stub_main(monkeypatch, launch, jobs=4) == 1
    assert ("CRASH [hook] top-level array: python exited 1, c++ exited 3\n"
            in capsys.readouterr().err)


def test_a_failing_case_stops_the_run_instead_of_draining_it(monkeypatch):
    """An exception in a worker surfaces now, not after ~630 queued launches.

    The first launch answers with bytes check_verdict cannot decode -- with
    two workers it is one half of the first case, which main() reads first --
    and the rest take 0.1 s each for the first 40 and nothing after, so a pool
    that drains its queue is a red in ~2 s. The bound is loose on purpose: the
    main thread may be slow to wake, and only the drain (every launch) is the
    defect.
    """
    lock = threading.Lock()
    calls = []

    def launch(argv, payload=None):
        with lock:
            calls.append(argv)
            count = len(calls)
        if count == 1:
            return subprocess.CompletedProcess(argv, 0, b"\xff\xfe", b"")
        if count <= 40:
            time.sleep(0.1)
        return _quiet(argv)

    with pytest.raises(UnicodeDecodeError):
        _stub_main(monkeypatch, launch, jobs=2)
    assert len(calls) <= 20, f"{len(calls)} launches ran in all"


def test_ctrl_c_kills_the_children_in_flight(monkeypatch):
    """KeyboardInterrupt leaves main() without waiting its children out.

    Real children (SLEEPER, 30 s) through the real launch; both halves of the
    first case -- the future main() reads first, whichever worker ran it --
    raise KeyboardInterrupt, as result() hands it to the main thread, once the
    third worker has started its child. Waiting that out would take 30 s;
    killing it took 2.2 s and 4.5 s (measured 2026-09-29), hence the 10. After the 12th launch the stub starts no
    child at all, so a regression costs one SLEEPER, not the corpus.
    """
    real = _PARITY.launch
    first = _PARITY.reference.SELF_TEST_CASES[0][0]
    lock = threading.Lock()
    calls = []

    def launch(argv, payload=None):
        with lock:
            calls.append(argv)
            count = len(calls)
        if "--check-file" in argv:
            with open(argv[argv.index("--check-file") + 1],
                      encoding="utf-8", newline="") as handle:
                if handle.read() == first:
                    time.sleep(2)  # the third worker's Popen has returned
                    raise KeyboardInterrupt
        if count > 12:
            return _quiet(argv)
        return real(SLEEPER)

    start = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        _stub_main(monkeypatch, launch, jobs=3)
    elapsed = time.monotonic() - start
    assert elapsed < 10, f"main() took {elapsed:.1f} s to stop"
    # The abort is over once main() is: the next launch runs.
    monkeypatch.setattr(_PARITY, "launch", real)
    result = _PARITY.launch([sys.executable, "-c", "print('ok')"])
    assert result is not None and result.stdout.strip() == b"ok"


def test_jobs_come_from_the_flag_then_the_environment_then_the_cpu_count():
    resolve = _PARITY.resolve_jobs
    assert resolve(None, {}) == min(_PARITY.DEFAULT_JOBS, os.cpu_count() or 1)
    assert resolve(None, {"PARITY_JOBS": "3"}) == 3
    assert resolve(5, {"PARITY_JOBS": "3"}) == 5
    # Refused, never read as "unset": the operator asked for something.
    for raw in ("0", "", "abc", " 3", "-2", "2.5", "٣"):
        with pytest.raises(ValueError, match="PARITY_JOBS"):
            resolve(None, {"PARITY_JOBS": raw})
    with pytest.raises(ValueError, match="--jobs"):
        resolve(0, {})


def test_run_closes_stdin_of_what_it_starts():
    """_run hands its child the null device, not the test process's stdin.

    fd 0 is made a pipe nobody writes to or closes, so a child that inherited
    it would wait on it until the timeout; with stdin closed it reads nothing
    and exits at once.
    """
    read_end, write_end = os.pipe()
    saved = os.dup(0)
    try:
        os.dup2(read_end, 0)
        result = _run(["-c", "import sys; print(repr(sys.stdin.read()))"],
                      timeout=20)
    finally:
        os.dup2(saved, 0)
        for descriptor in (saved, read_end, write_end):
            os.close(descriptor)
    assert result.stdout.strip() == "''", result.stdout + result.stderr


def test_port_options_are_the_reference_options():
    """The port's kOptions is a hand copy of the reference's option strings.

    Read from the source, so it bites on every checkout: the behavioural pin
    (parity_check's per-option ARGV_CASES rows) needs the binary, and an option
    added to the script alone would otherwise stay unknown to the port -- a
    usage error there, a verdict here -- until someone built it.
    """
    with open(os.path.join(PLUGIN, "cpp", "ask_user_gate.cpp"),
              encoding="utf-8") as handle:
        source = handle.read()
    block = re.search(r"constexpr CliOption kOptions\[\] = \{(.*?)\n\};", source,
                      re.DOTALL)
    assert block, "kOptions initializer not found in ask_user_gate.cpp"
    # What the compiler reads: a commented-out entry is no entry.
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", block.group(1), flags=re.DOTALL)
    entry = r'\{\s*"([^"\\]+)"\s*,\s*Arity::(\w+)\s*\}'
    # Every character is an entry, a comma or whitespace: an entry spelled
    # some other way fails here instead of dropping out of the comparison.
    leftover = re.sub(r"[\s,]+", " ", re.sub(entry, "", body)).strip()
    assert not leftover, (f"kOptions holds something other than "
                          f"{{\"name\", Arity::X}} entries: {leftover!r}")
    entries = re.findall(entry, body)
    names = [name for name, _ in entries]
    # The port takes the FIRST match, and a second copy of a name makes every
    # prefix of it ambiguous there -- a dict here would swallow it.
    duplicates = sorted({name for name in names if names.count(name) > 1})
    assert not duplicates, f"kOptions repeats {duplicates}"
    arities = {arity for _, arity in entries}
    assert arities <= {"None", "Value"}, (
        f"kOptions uses arities {sorted(arities - {'None', 'Value'})} this "
        f"test does not know; teach it what they take")
    port = {name: arity == "Value" for name, arity in entries}
    assert port == _PARITY.reference_options()


def _gate_module():
    spec = importlib.util.spec_from_file_location("ask_user_gate_ref", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_parser_without_help_refuses_a_dash_equals_token():
    """The stand-in's tuple width comes from a probe of its own.

    It was copied from a `--hel` lookup on the parser in use, which is empty
    without --help: `-= x` then raised IndexError instead of a usage error.
    """
    gate = _gate_module()
    parser = gate._Parser(add_help=False, exit_on_error=False)
    parser.add_argument("--check")
    parser.add_argument("--tool")
    with pytest.raises(argparse.ArgumentError, match="ambiguous option"):
        parser.parse_args(["-= x"])
    # A flag, so not --check's value.
    with pytest.raises(argparse.ArgumentError, match="expected one argument"):
        parser.parse_args(["--check", "-= x"])


def test_reference_parser_refuses_allow_abbrev_false():
    """Its rules are abbreviation rules; with abbreviations off, newer argparse
    reads `-= x` as a value, which the stand-in would contradict."""
    with pytest.raises(ValueError, match="allow_abbrev=False"):
        _gate_module()._Parser(allow_abbrev=False)


def test_reference_parser_refuses_an_option_tuple_layout_it_never_measured(
        monkeypatch):
    """A 5-field tuple must stop the parse loudly, not be padded with None."""
    gate = _gate_module()
    real = argparse.ArgumentParser._get_option_tuples

    def five_fields(self, option_string):
        return [match + (None,) for match in real(self, option_string)]

    monkeypatch.setattr(argparse.ArgumentParser, "_get_option_tuples",
                        five_fields)
    with pytest.raises(RuntimeError, match="teach _option_tuple_fields"):
        gate._option_tuple_fields()
    parser = gate._Parser(exit_on_error=False)
    parser.add_argument("--check")
    parser.add_argument("--tool")
    with pytest.raises(RuntimeError, match="teach _option_tuple_fields"):
        parser.parse_args(["--check", "-= x"])


@pytest.mark.skipif(not os.path.isfile(BINARY),
                    reason="the C++ gate is an opt-in build; see cpp/build.py")
def test_port_agrees_with_the_reference():
    """Verdict, exit code and refusal text, over the CLI and over hook mode.

    Failure here is not always "the port is wrong": a rule edited in the script
    alone fails it too, and that is the point -- whichever half moved, the pair
    stopped being one gate. PARITY_JOBS in the environment sets its pool size.
    """
    result = _run([PARITY], timeout=PARITY_TIMEOUT)
    assert result.returncode == 0, result.stdout + result.stderr
