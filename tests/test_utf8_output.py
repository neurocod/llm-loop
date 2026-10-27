"""Pin that both runners can print a non-ASCII project root to a narrow stream.

A Windows console or a redirected stdout is often cp1252, and the first line
every run prints is the project root (`runlifecycle.begin_run`) — a Cyrillic
folder there was a `UnicodeEncodeError` before the run had done anything. The
UTF-8 switch used to be a side effect of importing `cyclecore`, which the
eager front door ran for every runner; once the front door went lazy the
parallel runner (which never imports `cyclecore`) lost it. The switch now
lives in the package's `__init__`, which runs before anything in the package.

Each run is a fresh interpreter: inside the suite the streams are pytest's
captures and some earlier test has already imported every module, so neither
the encoding nor the import order here would say anything.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# The runner a script below starts: a driver with nothing to do, so the run
# prints its header and returns — the header is the part under test. The limit
# policy is stubbed so a sequential run never asks a provider for its quota.
_SCRIPT = """\
import sys
from llm_loop import {base}

class _Stub:
    def describe(self):
        return "stub"
    def log_snapshot(self, *args, **kwargs):
        pass
    def check_and_wait(self, source, session_start, *args, **kwargs):
        return False, session_start

class Idle({base}):
    limit_policy = _Stub()
    def prompt(self, source, target):
        return "unused"
    def pending_lines(self):
        return []
    def next_command(self):
        return None

Idle.{entry}(sys.argv[1:])
"""

_RUNNERS = {
    "parallel": ("ListFileDriver", "main_parallel", ["--jobs", "1",
                                                     "--ignore-usage"]),
    "sequential": ("Driver", "main", []),
}


@pytest.mark.parametrize("mode", sorted(_RUNNERS))
def test_a_cyrillic_project_root_prints_through_a_cp1252_stdout(mode, tmp_path):
    base, entry, extra = _RUNNERS[mode]
    root = tmp_path / "проект"
    root.mkdir()
    env = dict(os.environ, PYTHONPATH=str(SRC), PYTHONIOENCODING="cp1252")
    env.pop("PYTHONUTF8", None)
    done = subprocess.run(
        [sys.executable, "-c", _SCRIPT.format(base=base, entry=entry),
         "--dry-run", "--no-statusline", "--git-push", "none",
         "--project-dir", str(root), *extra],
        # 0.39 s measured 2026-10-27; generous for a loaded machine
        env=env, cwd=str(tmp_path), capture_output=True, timeout=60)

    stderr = done.stderr.decode("utf-8", errors="replace")
    assert done.returncode == 0, stderr
    assert str(root) in done.stdout.decode("utf-8"), stderr
