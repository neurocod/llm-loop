"""Run the GitHub CI matrix on this machine: every Python version, every step.

The matrix and the commands are read from .github/workflows/tests.yml rather
than restated here, so a version or a step added to the workflow is run locally
without touching this file. Only the `os` axis is not reproducible: the local
one is the one you get, and each step runs in the shell CI gives that OS (see
step_command).

Each version gets its own venv, made by uv (https://docs.astral.sh/uv/). uv is
what keeps this from disturbing the machine: a Python it has to download lands
in uv's own directory (`uv python dir`), is not put on PATH, not registered with
the `py` launcher, and is removed by `uv python uninstall <version>`. A version
already installed is used as the venv's base and left untouched.

    python scripts/ci_local.py                 # the whole matrix
    python scripts/ci_local.py 3.9 3.12        # a subset
    python scripts/ci_local.py --list          # the plan, without running it
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, NoReturn, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"


def venv_root(checkout: Path, windows: bool = os.name == "nt") -> Path:
    """Where one checkout's venvs live.

    Outside the checkout on purpose: a venv inside it would be walked by every
    tool that walks the tree, and would be one more thing .gitignore must name.
    Keyed on the checkout: two worktrees of this repository run at once, and a
    shared venv is `uv venv --clear`ed by one under the other's tests, and holds
    an editable install of whichever checkout ran `pip install -e` last. Case
    is folded only where the filesystem folds it: on POSIX `/Foo` and `/foo`
    are two checkouts.
    """
    key = str(checkout).lower() if windows else str(checkout)
    return (Path.home() / ".cache" / "llm-loop-ci"
            / hashlib.sha256(key.encode()).hexdigest()[:12])


VENVS = venv_root(ROOT)


class WorkflowShapeError(ValueError):
    """The workflow uses a shape this reader does not model; nothing was run."""


def _refuse(why: str, line: Optional[int] = None) -> NoReturn:
    where = WORKFLOW.name if line is None else f"{WORKFLOW.name}:{line}"
    raise WorkflowShapeError(f"{where}: {why}")


class Step(NamedTuple):
    name: str
    script: str


class Plan(NamedTuple):
    versions: List[str]
    steps: List[Step]


# ---------------------------------------------------------------------------
# The workflow reader.
#
# Not a YAML parser -- the standard library has none, and this repository takes
# no dependency for a dev script. It is a reader of a SUBSET: block mappings and
# sequences by indentation, plain one-line scalars, flow lists of quoted
# strings, and `|` / `|-` literal blocks. Anything else -- folded or indented
# block headers, quoted or multi-line scalars, anchors, tags, tabs -- is refused
# with its line number, and so is every workflow key that would change what a
# step does locally without this script modelling it (see plan_from). Refusing
# before anything runs is the whole contract: a local run that quietly covers
# less than CI, or runs something else, reads as a green CI.
# ---------------------------------------------------------------------------

_KEY = re.compile(r"([A-Za-z0-9_][A-Za-z0-9_.-]*):(?:[ ]+(.*))?$")
# Characters that open something other than a plain scalar at a value's start.
_NOT_PLAIN = set("\"'&*!%@`{}|>?,[]#")


class Scalar(NamedTuple):
    text: str   # a plain scalar's raw text (comment included), or a block's content
    line: int   # 1-based, for the refusal message
    block: bool


class _Reader:
    def __init__(self, text: str):
        self.lines = text.replace("\r\n", "\n").split("\n")
        self.index = 0

    def fail(self, line: int, why: str) -> NoReturn:
        _refuse(why, line)

    def indent(self, number: int) -> int:
        line = self.lines[number]
        width = len(line) - len(line.lstrip(" "))
        if line[width:width + 1] == "\t":
            self.fail(number + 1, "a tab in the indentation")
        return width

    def next_significant(self) -> Optional[int]:
        number = self.index
        while number < len(self.lines):
            stripped = self.lines[number].strip()
            if stripped and not stripped.startswith("#"):
                return number
            number += 1
        return None

    def node(self, parent: int):
        """The block node under a key at column `parent`, or None."""
        number = self.next_significant()
        if number is None or self.indent(number) <= parent:
            return None
        column = self.indent(number)
        text = self.lines[number][column:]
        if text == "-" or text.startswith("- "):
            return self.sequence(column)
        return self.mapping(column)

    def mapping(self, column: int, first: Optional[Tuple[int, str]] = None) -> dict:
        result: dict = {}
        while True:
            if first is not None:
                number, text = first
                first = None
            else:
                number = self.next_significant()
                if number is None or self.indent(number) < column:
                    return result
                if self.indent(number) > column:
                    self.fail(number + 1, "unexpected indentation")
                text = self.lines[number][column:]
            match = _KEY.match(text)
            if not match:
                self.fail(number + 1, f"not a `key: value` line: {text.strip()!r}")
            key = match.group(1)
            if key in result:
                self.fail(number + 1, f"duplicate key {key!r}")
            self.index = number + 1
            result[key] = self.value(match.group(2) or "", column, number)

    def sequence(self, column: int) -> list:
        items = []
        while True:
            number = self.next_significant()
            if number is None or self.indent(number) < column:
                return items
            text = self.lines[number][column:]
            if self.indent(number) > column or not (text == "-" or text.startswith("- ")):
                self.fail(number + 1, "expected a `- ` sequence item")
            rest = text[1:]
            body = rest.lstrip(" ")
            self.index = number + 1
            if not body or body.startswith("#"):
                items.append(self.node(column))
            elif _KEY.match(body):
                items.append(self.mapping(column + 1 + len(rest) - len(body),
                                          first=(number, body)))
            else:
                items.append(self.value(body, column, number))

    def value(self, text: str, column: int, number: int):
        """What follows `key:` on line `number`, the key being at `column`."""
        if not text or text.startswith("#"):
            return self.node(column)
        if text[0] in "|>":
            return self.block(text, column, number)
        if text[0] in _NOT_PLAIN - set("\"'["):
            self.fail(number + 1, f"YAML syntax this reader does not model: {text!r}")
        following = self.next_significant()
        if following is not None and self.indent(following) > column:
            self.fail(following + 1, "a scalar continued on the next line")
        return Scalar(text, number + 1, False)

    def block(self, header: str, column: int, number: int) -> Scalar:
        # `|` keeps the final newline, `|-` drops it. Folding (`>`), keep (`|+`)
        # and explicit indentation (`|2`) change the text a shell receives, and a
        # comment after the header is one more thing to get wrong: refused.
        if header not in ("|", "|-"):
            self.fail(number + 1, f"block scalar header {header!r}: only `|` and "
                                  f"`|-` are modelled")
        body = []
        cursor = self.index
        while cursor < len(self.lines):
            line = self.lines[cursor]
            if line.strip() and self.indent(cursor) <= column:
                break
            body.append((cursor, line))
            cursor += 1
        self.index = cursor
        while body and not body[-1][1].strip():
            body.pop()
        if not body:
            return Scalar("", number + 1, True)
        width = self.indent(next(n for n, line in body if line.strip()))
        out = []
        for line_number, line in body:
            if line.strip() and self.indent(line_number) < width:
                self.fail(line_number + 1, "less indented than the block's first line")
            out.append(line[width:])
        return Scalar("\n".join(out) + ("\n" if header == "|" else ""), number + 1, True)


def parse_yaml_subset(text: str) -> dict:
    reader = _Reader(text)
    number = reader.next_significant()
    if number is None:
        _refuse("empty")
    if reader.indent(number):
        reader.fail(number + 1, "the top level is indented")
    document = reader.mapping(0)
    left = reader.next_significant()
    if left is not None:
        reader.fail(left + 1, "unexpected text after the document")
    return document


def _plain(node, where: str) -> str:
    """A plain one-line scalar, its trailing comment removed."""
    if not isinstance(node, Scalar) or node.block:
        _refuse(f"{where}: expected a one-line value")
    text = node.text
    if text[0] in "\"'[":
        _refuse(f"{where}: a quoted or flow value is not modelled here", node.line)
    # In a plain scalar ` #` starts a comment even inside what looks like
    # quotes, and `: ` is an error; neither is guessed at.
    if ": " in text or text.endswith(":"):
        _refuse(f"{where}: `: ` in a plain value", node.line)
    return re.sub(r"\s+#.*$", "", text).strip()


def _only(mapping: dict, allowed: set, where: str) -> None:
    extra = [key for key in mapping if key not in allowed]
    if extra:
        _refuse(f"{where}: {', '.join(extra)} not modelled by ci_local.py -- "
                f"teach it, or it would run something other than CI")


_FLOW_OF_QUOTED = re.compile(
    r"""\[\s*((?:"[^"\\]*"|'[^']*')(?:\s*,\s*(?:"[^"\\]*"|'[^']*'))*)\s*,?\s*\]"""
    r"""\s*(?:#.*)?$""")


def _versions(node) -> List[str]:
    if not isinstance(node, Scalar) or node.block or not node.text.startswith("["):
        _refuse("matrix.python-version must be a one-line flow list `[...]`")
    match = _FLOW_OF_QUOTED.match(node.text)
    if not match:
        # Unquoted items are floats to YAML: `3.10` would be CI's 3.1.
        _refuse("every python-version must be quoted (YAML reads 3.10 as the "
                "float 3.1)", node.line)
    versions = [item[1:-1] for item in
                re.findall(r""""[^"\\]*"|'[^']*'""", match.group(1))]
    bad = [v for v in versions if not re.fullmatch(r"[0-9][0-9A-Za-z.+-]*", v)]
    if bad or len(set(versions)) != len(versions):
        _refuse(f"python-version items {versions} are not plain distinct versions",
                node.line)
    return versions


def _step(node, index: int) -> Optional[Step]:
    where = f"step {index + 1}"
    if not isinstance(node, dict):
        _refuse(f"{where} is not a mapping")
    name = _plain(node["name"], where) if "name" in node else ""
    where = f"step {index + 1} ({name})" if name else where
    if "uses" in node:
        _only(node, {"name", "uses", "with"}, where)
        action = _plain(node["uses"], where).split("@")[0]
        if action == "actions/checkout" and "with" not in node:
            return None  # the checkout is this one
        if (action == "actions/setup-python" and isinstance(node.get("with"), dict)
                and set(node["with"]) == {"python-version"}
                and isinstance(node["with"]["python-version"], Scalar)
                and node["with"]["python-version"].text.strip()
                == "${{ matrix.python-version }}"):
            return None  # the venv is this one
        _refuse(f"{where}: `uses: {_plain(node['uses'], where)}` (with its `with:`) "
                f"is not an action ci_local.py reproduces")
    if "run" not in node:
        _refuse(f"{where}: neither `uses` nor `run`")
    # shell, working-directory, env, if, continue-on-error all change what the
    # step does or whether it counts.
    _only(node, {"name", "run"}, where)
    run = node["run"]
    if isinstance(run, Scalar) and run.block:
        script = run.text
    else:
        script = _plain(run, where)
        if isinstance(run, Scalar) and re.search(r"\s#", run.text):
            _refuse(f"{where}: a comment after an inline `run:` -- use a `|` block",
                    run.line)
    if "${{" in script:
        _refuse(f"{where}: `${{{{ ... }}}}` expressions are not evaluated by "
                f"ci_local.py")
    if not script.strip():
        _refuse(f"{where}: an empty `run:`")
    return Step(name, script)


def parse_workflow(text: str) -> Plan:
    """The matrix's python-version list and the `run:` steps of its one job.

    Raises WorkflowShapeError on any shape it does not model (see the reader
    above), so a workflow that outgrows it fails here instead of running half
    a matrix, or a different one.
    """
    document = parse_yaml_subset(text)
    _only(document, {"name", "run-name", "on", "jobs", "permissions", "concurrency"},
          "the workflow")
    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or len(jobs) != 1:
        _refuse("exactly one job is modelled")
    (job_name, job), = jobs.items()
    if not isinstance(job, dict):
        _refuse(f"job {job_name} is not a mapping")
    _only(job, {"name", "runs-on", "strategy", "steps", "timeout-minutes", "permissions"},
          f"job {job_name}")
    strategy = job.get("strategy")
    matrix = strategy.get("matrix") if isinstance(strategy, dict) else None
    if not isinstance(matrix, dict):
        _refuse(f"job {job_name} has no strategy.matrix")
    _only(strategy, {"fail-fast", "max-parallel", "matrix"}, f"job {job_name} strategy")
    _only(matrix, {"os", "python-version"}, f"job {job_name} matrix")
    if "python-version" not in matrix:
        _refuse("the matrix has no python-version")
    versions = _versions(matrix["python-version"])
    raw_steps = job.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        _refuse(f"job {job_name} has no steps")
    steps = [step for step in (_step(node, i) for i, node in enumerate(raw_steps))
             if step is not None]
    if not steps:
        _refuse("no `run:` steps")
    return Plan(versions, steps)


# ---------------------------------------------------------------------------
# Running it.
# ---------------------------------------------------------------------------

def venv_bin(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


SCRIPT_SUFFIX = ".ps1" if os.name == "nt" else ".sh"


def step_command(script_path: Path) -> List[str]:
    """The argv CI runs a `run:` step with (no `shell:`) on this OS; the text
    it wraps the step in is wrap_script's. From the runner, not guessed: the
    default shell and its arguments (https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#defaultsrunshell)
    and the PowerShell wrapper, ScriptHandlerHelpers.FixUpScriptContents in
    https://github.com/actions/runner/blob/main/src/Runner.Worker/Handlers/ScriptHandlerHelpers.cs.

    POSIX: `bash -e {0}` (sh -e without bash) -- the whole script, stopping at
    the first failing line, but a failure inside a pipeline is not one.
    Windows: pwsh (the runner falls back to Windows PowerShell without it),
    `$ErrorActionPreference = 'stop'` before the script and `exit
    $LASTEXITCODE` after it: a failing NATIVE command does not stop the
    script, and only the last one's exit code is the step's.
    """
    if os.name == "nt":
        shell = shutil.which("pwsh") or shutil.which("powershell")
        if not shell:
            raise RuntimeError("neither pwsh nor powershell is on PATH")
        quoted = str(script_path).replace("'", "''")
        return [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                "-command", f". '{quoted}'"]
    bash = shutil.which("bash")
    return [bash, "-e", str(script_path)] if bash else ["sh", "-e", str(script_path)]


def wrap_script(script: str) -> str:
    if SCRIPT_SUFFIX == ".ps1":
        return ("$ErrorActionPreference = 'stop'\n" + script + "\n"
                + "if ((Test-Path -LiteralPath variable:\\LASTEXITCODE)) "
                  "{ exit $LASTEXITCODE }\n")
    return script


# How long a killed tree may take to leave its Windows job before the kill is
# reported as incomplete. The job of a python child and grandchild measured
# empty on the first query after TerminateJobObject (under 1 ms, 5 runs,
# 2026-09-28); the bound only matters when something is wrong.
TREE_EXIT_TIMEOUT = 30.0

# run_tree polls in slices of this, sleeping in between, rather than calling
# Popen.wait. On Windows an unbounded wait() blocks in WaitForSingleObject and
# a Ctrl+C is not seen until the step ends by itself; wait(timeout) is worse on
# 3.9: a Ctrl+C landing after the slice ran out makes its interrupt handler wait
# a NEGATIVE time, which WaitForSingleObject reads as nearly forever (measured:
# the interrupt was held until the step exited). time.sleep() is interruptible.
_WAIT_SLICE = 0.1

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    _kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _kernel32.QueryInformationJobObject.argtypes = (
        wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD))
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)

    class _JobAccounting(ctypes.Structure):
        # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
        _fields_ = [("TotalUserTime", ctypes.c_int64),
                    ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64),
                    ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD),
                    ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD),
                    ("TotalTerminatedProcesses", wintypes.DWORD)]

    _JOB_BASIC_ACCOUNTING = 1
    _PROCESS_TERMINATE_AND_SET_QUOTA = 0x0001 | 0x0100

    def _win_check(ok) -> None:
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    class _ProcessTree:
        """A job object: every descendant of the step joins it, including one
        whose parent has already exited, and it can be counted until empty.

        The process is assigned right after CreateProcess returns, not
        created suspended: a child spawned in that window would escape, and a
        shell (or uv) is still loading its runtime then.
        No kill-on-close: a normal step's leftovers outlive the step, as they
        do on a runner.
        """
        POPEN: Dict[str, object] = {}

        def __init__(self) -> None:
            self.job = _kernel32.CreateJobObjectW(None, None)
            _win_check(self.job)

        def adopt(self, process: subprocess.Popen) -> None:
            handle = _kernel32.OpenProcess(_PROCESS_TERMINATE_AND_SET_QUOTA, False,
                                           process.pid)
            _win_check(handle)
            try:
                _win_check(_kernel32.AssignProcessToJobObject(self.job, handle))
            finally:
                _kernel32.CloseHandle(handle)

        def active(self) -> int:
            info = _JobAccounting()
            _win_check(_kernel32.QueryInformationJobObject(
                self.job, _JOB_BASIC_ACCOUNTING, ctypes.byref(info),
                ctypes.sizeof(info), None))
            return info.ActiveProcesses

        def kill(self, process: subprocess.Popen) -> None:
            _kernel32.TerminateJobObject(self.job, 1)
            process.kill()  # in case adopt() never ran or failed
            process.wait()
            deadline = time.monotonic() + TREE_EXIT_TIMEOUT
            while self.active() and time.monotonic() < deadline:
                time.sleep(0.05)
            if self.active():
                print(f"ci_local: {self.active()} process(es) of the interrupted "
                      f"step still alive after {TREE_EXIT_TIMEOUT:.0f} s",
                      file=sys.stderr)

        def close(self) -> None:
            _kernel32.CloseHandle(self.job)
else:
    import signal

    class _ProcessTree:
        """A session of its own: the step's shell leads a process group that
        every descendant inherits unless it makes a session of its own.

        SIGKILL to the group stops every member from running any more code at
        once, so nothing of it writes to the venv after kill() returns; a
        member left a zombie is reaped by init, not waited for here.
        """
        POPEN: Dict[str, object] = {"start_new_session": True}

        def adopt(self, process: subprocess.Popen) -> None:
            pass  # start_new_session made its group

        def kill(self, process: subprocess.Popen) -> None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.kill()
            process.wait()

        def close(self) -> None:
            pass


def run_tree(argv: List[str], **popen) -> int:
    """subprocess.run(argv, **popen).returncode, for the whole process TREE.

    subprocess.run kills only its direct child when interrupted: a step's
    shell dies, the python under it (one that ignores SIGINT, or a
    grandchild) keeps running, and the caller's `exclusive()` then releases
    the venv lock while it still runs in the venv -- the next waiting run
    `--clear`s the venv under it. Here any exception, Ctrl+C included, kills
    every descendant and waits for them BEFORE it propagates, so the lock is
    released over a dead tree.
    """
    tree = _ProcessTree()
    try:
        process = subprocess.Popen(argv, **tree.POPEN, **popen)
        try:
            tree.adopt(process)
            while process.poll() is None:
                time.sleep(_WAIT_SLICE)
            return process.returncode
        except BaseException:
            tree.kill(process)
            raise
    finally:
        tree.close()


def run_steps(steps: List[Step], env: Dict[str, str], log, cwd: Path = ROOT
              ) -> Optional[str]:
    """None when every step passed, else the name of the first that failed.
    Each step is ONE script in one shell, as in CI: `cd`, variables and
    multi-line constructs carry from line to line, and a failure counts
    exactly when CI's shell would count it."""
    with tempfile.TemporaryDirectory(prefix="ci_local-") as scratch:
        for number, step in enumerate(steps, 1):
            label = step.name or f"step {number}"
            path = Path(scratch) / f"step{number}{SCRIPT_SUFFIX}"
            path.write_text(wrap_script(step.script), encoding="utf-8")
            argv = step_command(path)
            log.write(f"\n== {label}\n$ {' '.join(argv)}\n{step.script.rstrip()}\n--\n")
            log.flush()
            code = run_tree(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT)
            if code:
                log.write(f"-- exit {code}\n")
                return f"{label} (exit {code})"
    return None


# Nothing of the caller's Python setup reaches the steps: CI has none of it, and
# PYTHONPATH or PYTEST_ADDOPTS alone can make a local run test other code.
_SCRUBBED = ("PYTHON", "PIP_", "PYTEST_", "VIRTUAL_ENV", "CONDA_")


def step_env(venv: Path) -> Dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith(_SCRUBBED)}
    env["VIRTUAL_ENV"] = str(venv)
    env["PATH"] = str(venv_bin(venv)) + os.pathsep + env.get("PATH", "")
    return env


def check_interpreters(venv: Path, env: Dict[str, str], steps: List[Step], log
                       ) -> Optional[str]:
    """Why the steps would not run the venv's Python, or None.

    The venv is on PATH first, but a name it lacks falls through to the
    system: a Windows venv has no python3.exe, so a `python3` step would run
    whatever python3 the machine has and be reported as this version.
    """
    for name in ("python", "python3"):
        if name == "python3" and not any(re.search(r"\bpython3\b", s.script)
                                         for s in steps):
            continue
        found = shutil.which(name, path=env["PATH"])
        # The directory, not the file: a POSIX venv's python is a symlink out.
        if not found or Path(found).parent.resolve() != venv_bin(venv).resolve():
            return f"`{name}` resolves to {found}, not into {venv}"
    python = str(venv_bin(venv) / "python")
    result = subprocess.run([python, "-c", "import sys; print(sys.prefix)"],
                            env=env, capture_output=True, text=True)
    if result.returncode or Path(result.stdout.strip()).resolve() != venv.resolve():
        return f"the venv's python reports sys.prefix {result.stdout.strip()!r}"
    version = subprocess.run([python, "-VV"], env=env, capture_output=True, text=True)
    log.write(f"interpreter: {version.stdout.strip()}  ({python})\n")
    return None


if os.name == "nt":
    import msvcrt

    def _lock(handle, unlock: bool = False) -> None:
        """Raises OSError when another process holds it."""
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
else:
    import fcntl

    def _lock(handle, unlock: bool = False) -> None:
        """Raises OSError when another process holds it."""
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN if unlock
                    else fcntl.LOCK_EX | fcntl.LOCK_NB)


@contextlib.contextmanager
def exclusive(path: Path) -> Iterator[None]:
    """Hold `path` locked for the whole create/install/test of one version, so
    a second run of the same checkout waits instead of clearing the venv
    under the first."""
    with path.open("a+b") as handle:
        told = False
        while True:
            try:
                _lock(handle)
                break
            except OSError:
                if not told:
                    print(f"   waiting for {path} (another ci_local run)", flush=True)
                    told = True
                time.sleep(2)
        try:
            yield
        finally:
            _lock(handle, unlock=True)


def run_version(version: str, steps: List[Step], log_path: Path) -> Optional[str]:
    """None when every step passed, else what failed first."""
    venv = VENVS / version
    with exclusive(VENVS / f"{version}.lock"), \
            log_path.open("w", encoding="utf-8", errors="replace") as log:
        # --seed puts pip in the venv, so the workflow's own
        # `python -m pip install -e ".[test]"` runs as written.
        create = ["uv", "venv", "--clear", "--seed", "--python", version, str(venv)]
        log.write("$ " + " ".join(create) + "\n")
        log.flush()
        if run_tree(create, stdout=log, stderr=subprocess.STDOUT):
            return "create the venv"
        env = step_env(venv)
        wrong = check_interpreters(venv, env, steps, log)
        if wrong:
            log.write(wrong + "\n")
            return wrong
        return run_steps(steps, env, log)


def tail(path: Path, count: int = 25) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join("    " + line for line in lines[-count:])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the GitHub CI matrix locally, one uv venv per Python.")
    parser.add_argument("versions", nargs="*",
                        help="a subset of the workflow's python-version list")
    parser.add_argument("--list", action="store_true",
                        help="print the versions and steps, run nothing")
    options = parser.parse_args(argv)

    try:
        plan = parse_workflow(WORKFLOW.read_text(encoding="utf-8"))
    except WorkflowShapeError as error:
        print(f"ci_local: {error}", file=sys.stderr)
        return 2
    unknown = [v for v in options.versions if v not in plan.versions]
    if unknown:
        parser.error(f"not in the workflow matrix {plan.versions}: {unknown}")
    versions = options.versions or plan.versions

    if options.list:
        print("versions:", " ".join(versions))
        print("shell:", " ".join(step_command(Path("{0}"))))
        for number, step in enumerate(plan.steps, 1):
            print(f"{step.name or f'step {number}'}:")
            for line in step.script.rstrip("\n").split("\n"):
                print(f"  {line}")
        return 0

    if not shutil.which("uv"):
        print("ci_local: uv is not on PATH; install it first "
              "(https://docs.astral.sh/uv/getting-started/installation/)",
              file=sys.stderr)
        return 2

    VENVS.mkdir(parents=True, exist_ok=True)
    # One version at a time: the suites spawn processes and measure time, and
    # five of them sharing the CPU is not the machine CI gives each job.
    failures = []
    for version in versions:
        log_path = VENVS / f"{version}.log"
        started = time.monotonic()
        print(f"== {version} ...", flush=True)
        failed = run_version(version, plan.steps, log_path)
        took = f"{time.monotonic() - started:.0f} s"
        if failed:
            failures.append(version)
            print(f"   FAIL in {took} at {failed}\n   log: {log_path}\n{tail(log_path)}")
        else:
            print(f"   ok in {took}  (log: {log_path})")
    print("all passed" if not failures else f"FAILED: {' '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
