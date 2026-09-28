"""Run the GitHub CI matrix on this machine: every Python version, every step.

The matrix and the commands are read from .github/workflows/tests.yml rather
than restated here, so a version or a step added to the workflow is run locally
without touching this file. Only the `os` axis is not reproducible: the local
one is the one you get.

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
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, NamedTuple, Optional

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
# Outside the checkout on purpose: a venv inside it would be walked by every
# tool that walks the tree, and would be one more thing .gitignore must name.
VENVS = Path.home() / ".cache" / "llm-loop-ci"


class Step(NamedTuple):
    name: str
    commands: List[str]


class Plan(NamedTuple):
    versions: List[str]
    steps: List[Step]


def parse_workflow(text: str) -> Plan:
    """The matrix's python-version list and the `run:` steps, in order.

    Not a YAML parser -- the standard library has none, and this repository
    takes no dependency for a dev script. It reads exactly the shapes
    tests.yml uses: a flow list `python-version: [...]`, and `run:` either
    inline or as a `|` block. A workflow that outgrows them fails here loudly
    (no versions, or no steps) instead of running half a matrix.
    """
    match = re.search(r"^\s*python-version:\s*\[(.*?)\]", text, re.MULTILINE)
    if not match:
        raise ValueError("no `python-version: [...]` list in the workflow")
    versions = [item.strip().strip("\"'") for item in match.group(1).split(",")
                if item.strip()]

    steps: List[Step] = []
    lines = text.splitlines()
    name = ""
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if re.match(r"\s*-\s", line):
            name = ""  # a new step: a `uses:` one leaves no name behind
        named = re.match(r"\s*-\s*name:\s*(.+)", line)
        if named:
            name = named.group(1).strip()
            continue
        run = re.match(r"(\s*)(?:-\s*)?run:\s*(.*)", line)
        if not run:
            continue
        indent, value = len(run.group(1)), run.group(2).strip()
        if value not in ("|", "|-", ">", ">-"):
            steps.append(Step(name, [value]))
        else:
            block = []
            while index < len(lines) and (not lines[index].strip()
                                          or len(lines[index]) - len(lines[index].lstrip()) > indent):
                if lines[index].strip():
                    block.append(lines[index].strip())
                index += 1
            steps.append(Step(name, block))
    if not versions or not steps:
        raise ValueError("the workflow parsed to no versions or no steps")
    return Plan(versions, steps)


def venv_bin(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def run_logged(command: str, env: dict, log) -> int:
    log.write(f"\n$ {command}\n")
    log.flush()
    # shell=True: the workflow's lines are shell lines (quoting, `.[test]`).
    return subprocess.run(command, shell=True, cwd=ROOT, env=env,
                          stdout=log, stderr=subprocess.STDOUT).returncode


def run_version(version: str, steps: List[Step], log_path: Path) -> Optional[str]:
    """None when every command passed, else the step that failed first.

    A `|` block is run line by line and stops at the first failure, which is
    stricter than CI's Windows shell (pwsh there reports only the LAST line's
    exit code) and equal to its bash.
    """
    venv = VENVS / version
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        # --seed puts pip in the venv, so the workflow's own
        # `python -m pip install -e ".[test]"` runs as written.
        create = ["uv", "venv", "--clear", "--seed", "--python", version, str(venv)]
        log.write("$ " + " ".join(create) + "\n")
        log.flush()
        if subprocess.run(create, stdout=log, stderr=subprocess.STDOUT).returncode:
            return "create the venv"
        env = dict(os.environ)
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = str(venv_bin(venv)) + os.pathsep + env.get("PATH", "")
        env.pop("PYTHONHOME", None)
        for step in steps:
            for command in step.commands:
                if run_logged(command, env, log):
                    return f"{step.name or 'step'}: {command}"
    return None


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

    plan = parse_workflow(WORKFLOW.read_text(encoding="utf-8"))
    unknown = [v for v in options.versions if v not in plan.versions]
    if unknown:
        parser.error(f"not in the workflow matrix {plan.versions}: {unknown}")
    versions = options.versions or plan.versions

    if options.list:
        print("versions:", " ".join(versions))
        for step in plan.steps:
            print(f"{step.name or 'step'}:")
            for command in step.commands:
                print(f"  {command}")
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
