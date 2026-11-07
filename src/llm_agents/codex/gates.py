"""Gates the WRAPPER runs after a worker's turn: parsed from the prompt, run
outside the sandbox, reported under the answer (`spawn --gates-from-prompt`).

Why the wrapper and not the worker: in the wave of 2026-09-13 three workers of
fourteen (`topology`, `lotrules`, `sections`) ended their turn with the gates
unrun and a report saying "NOT RUN; interrupted", and two more (`entrycount`,
`indoorlight`) named gates in their report that nothing showed they ran. The
orchestrator ran every one of them itself. The commands were already in the
prompt, under `## Gates`; this reads them from there, so a worker's claims
about gates stop mattering and the one question left to it is the edit.

The same list also feeds the escalation regex (`escalate_pattern`), which the
orchestrator used to copy out of the prompt by hand — and forgot a line of:
`rename_gate.py` was refused on the first worker it was needed by.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..host_policy import active as host_policy
from .events import GATE_META
from .protocol import _reap

# A heading named Gates, at any level; the section runs to the next heading of
# the same level or higher. Trailing words are allowed ("## Gates (all green
# today)") because a prompt author writes them.
_GATES_HEADING = re.compile(r"^(#{1,6})\s+Gates\b", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s")
_FENCE = re.compile(r"^\s{0,3}(```|~~~)")

# Gates run under cmd.exe (`run_gate`, `shell=True`), while prompts and
# handoffs are often written in PowerShell: `$env:TEMP` would be a literal
# directory name there, and a backtick a literal character. Refused rather
# than run wrongly.
_POWERSHELL_SYNTAX = re.compile(r"[$`]")

# How much of a red gate's output travels: into the follow-up prompt and into
# the block under the answer. A budget, not a measurement — the reader of both
# is a model, and a failing test run's cause sits in its last screenful
# (vitest and pytest both print the summary and the first failure's assertion
# last). Too few lines costs the worker a rerun to see the cause; too many
# costs every reader of `result` the context.
GATE_TAIL_LINES = 40

# The follow-up sent when a gate is red. One, on the same thread, like
# `turn._follow_up_on_missing`: the thread has the task and the edit, so the
# message names only what the wrapper saw. It asks for the whole report again
# because the first one's `### Gates` describes a tree that has since changed.
GATE_FOLLOWUP_PROMPT = (
    "The wrapper ran the task's gates outside the sandbox after your turn, and "
    "{failed} of {total} failed. Their output is below. Fix the cause in the "
    "project tree (do not weaken or skip a gate), then give the COMPLETE "
    "report again{report_clause}; the wrapper reruns every gate after this "
    "turn and reports their results itself.\n\n{details}\n"
)


class GatesError(ValueError):
    """The prompt promised gates (`--gates-from-prompt`) and holds none."""


@dataclass
class GateRun:
    command: str
    cwd: str
    # None when the command never finished (timeout) or never started.
    exit_code: Optional[int]
    output: str
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


GateRunner = Callable[[str, Path, float], GateRun]


def prompt_gates(prompt: str) -> list[str]:
    """Every command in a fenced block under a `Gates` heading, in order.

    One command per non-empty line; a line starting with `#` is a comment.
    Only FENCED lines count: bullets and inline code under the heading are
    prose about the gates, and guessing which backticked span is a command is
    how a wrapper would run half a sentence outside the sandbox.
    """
    return [command for section in _gate_sections(prompt) for command in section]


def _gate_sections(prompt: str) -> list[list[str]]:
    """The fenced commands of each `Gates` section that has any."""
    sections: list[list[str]] = []
    level = 0          # the open Gates section's heading level, 0 outside one
    fenced = False
    for line in prompt.splitlines():
        if _FENCE.match(line):
            fenced = not fenced
            continue
        if fenced:
            if level and line.strip() and not line.strip().startswith("#"):
                sections[-1].append(line.strip())
            continue
        heading = _HEADING.match(line)
        if heading is None:
            continue
        gates = _GATES_HEADING.match(line)
        if gates is not None:
            level = len(gates.group(1))
            sections.append([])
        elif level and len(heading.group(1)) <= level:
            level = 0
    return [section for section in sections if section]


def required_gates(prompt: str) -> list[str]:
    """`prompt_gates`, refusing a prompt that has none — the flag asked for
    them, and a spawn that silently runs zero gates reports a clean result for
    work nobody checked.

    Also refused: a SECOND `Gates` section with commands (a report template
    quoted in the prompt — "under `### Gates` list ..." — whose fenced example
    would otherwise run with full rights), and a line in PowerShell syntax
    (`_POWERSHELL_SYNTAX`)."""
    sections = _gate_sections(prompt)
    if not sections:
        raise GatesError(
            "--gates-from-prompt: the prompt has no fenced command block under "
            "a `## Gates` heading")
    if len(sections) > 1:
        raise GatesError(
            f"--gates-from-prompt: {len(sections)} `Gates` sections hold "
            f"commands; keep the gates in one and rename the others")
    for command in sections[0]:
        if _POWERSHELL_SYNTAX.search(command):
            raise GatesError(
                f"--gates-from-prompt: `{command}` uses `$` or a backtick; gates "
                f"run under cmd.exe, write the line without PowerShell syntax")
    return sections[0]


def _command_pattern(command: str) -> str:
    """One gate as a regex over the command the agent will type: the prompt's
    line EXACTLY, tokens separated by spaces or tabs, a path separator either
    slash (an agent on Windows types `tools\\check\\...` as often as the
    prompt's `tools/check/...`).

    No extra arguments: once a command matches it runs with the user's full
    rights, and literal arguments are an escape of their own — `npm test
    --script-shell=<exe>` runs a program the agent wrote, `--prefix <dir>`
    points it at another tree, pytest's `-p <module>` loads agent code (review
    2026-09-30). The wrapper runs the gates itself after the turn, so a worker
    denied a variant loses nothing but its own preview. No `\\s` either: it
    matches a line break, and `npm test\\n<cmd>` is two commands.
    """
    tokens = []
    for token in command.split():
        tokens.append(r"[/\\]".join(re.escape(part)
                                     for part in re.split(r"[/\\]", token)))
    return r"[ \t]+".join(tokens)


def escalate_pattern(commands: list[str]) -> str:
    """The `--escalate` regex these gates need, anchored at both ends (`\\Z`,
    since `$` also matches before a final newline)."""
    if not commands:
        return ""
    return "^(?:" + "|".join(_command_pattern(c) for c in commands) + r")[ \t]*\Z"


def gate_cwd(command: str, project: Path) -> Path:
    """Where one gate line runs: the host's answer (`HostPolicy.gate_cwd`),
    the project root under the empty policy."""
    return host_policy().gate_cwd(command, project)


def run_gate(command: str, cwd: Path, timeout: float) -> GateRun:
    """Run one gate through the shell, outside every sandbox.

    Output goes to a temporary FILE, not a pipe: `npm` on Windows is a `.cmd`
    shim, so the process we hold is `cmd.exe`, and a grandchild left running
    by a timeout keeps a pipe open — reading it to EOF would hang the wrapper
    on exactly the gate it was bounding. A file has no EOF to wait for.
    """
    started = time.monotonic()
    if not cwd.is_dir():
        return GateRun(command, str(cwd), None,
                       f"working directory {cwd} does not exist")
    with tempfile.TemporaryFile() as sink:
        try:
            proc = subprocess.Popen(
                command, shell=True, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=sink, stderr=subprocess.STDOUT,
                **({"creationflags": subprocess.CREATE_NO_WINDOW}
                   if os.name == "nt" else {}))
        except OSError as exc:
            return GateRun(command, str(cwd), None, f"could not start: {exc}",
                           time.monotonic() - started)
        note = ""
        code: Optional[int] = None
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            note = f"\n[wrapper] timed out after {timeout:g}s"
        finally:
            # Any exit but a finished wait reaps the tree — a Ctrl+C included:
            # the gate's window is hidden (`CREATE_NO_WINDOW`), so the console
            # interrupt never reaches it, and a test or bake left running keeps
            # editing the shared tree after the wrapper is gone.
            if code is None:
                _reap(proc)
        sink.seek(0)
        output = sink.read().decode("utf-8", "replace") + note
    return GateRun(command, str(cwd), code, output, time.monotonic() - started)


def run_gates(commands: list[str], project: Path, budget: float,
              runner: GateRunner,
              log: Callable[[dict], None]) -> list[GateRun]:
    """Run every gate, even after a red one: the follow-up names all failures
    at once, so the worker is not sent back once per gate.

    The gates SHARE `budget` seconds, each getting what the ones before it
    left; a gate reached with none left is reported as not run. Per-gate
    timeouts made one `spawn` last up to `(2N + 2)` turn timeouts, a bound the
    caller's `--timeout` no longer described (review 2026-09-30)."""
    results = []
    deadline = time.monotonic() + budget
    for command in commands:
        cwd = gate_cwd(command, project)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            result = GateRun(command, str(cwd), None,
                             f"[wrapper] not run: the gate budget of "
                             f"{budget:g}s was spent")
            log({"_meta": GATE_META, "command": command, "cwd": str(cwd),
                 "exit": None, "duration_s": 0.0, "skipped": True,
                 "at": time.strftime("%Y-%m-%d %H:%M:%S")})
            results.append(result)
            continue
        # A line before each gate as well as after: a gate may run for the
        # turn's whole timeout, and `status` judges a silent log (`events.
        # last_turn_trace` reopens the turn on this line).
        log({"_meta": GATE_META, "command": command, "cwd": str(cwd),
             "running": True, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        result = runner(command, cwd, remaining)
        log({"_meta": GATE_META, "command": command, "cwd": result.cwd,
             "exit": result.exit_code, "duration_s": round(result.duration_s, 1),
             "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        results.append(result)
    return results


def tail(output: str, lines: int = GATE_TAIL_LINES) -> str:
    return "\n".join(output.rstrip().splitlines()[-lines:])


def _status(result: GateRun) -> str:
    code = "no exit (timed out or not started)" if result.exit_code is None \
        else f"exit {result.exit_code}"
    return f"`{result.command}` (cwd {result.cwd}): {code}, {result.duration_s:.0f}s"


def failure_details(results: list[GateRun]) -> str:
    parts = []
    for result in results:
        if result.passed:
            continue
        parts.append(f"{_status(result)}\n```\n{tail(result.output)}\n```")
    return "\n\n".join(parts)


def follow_up_prompt(results: list[GateRun], report: Optional[Path]) -> str:
    failed = [r for r in results if not r.passed]
    clause = (f", written as UTF-8 to {report} (the wrapper reads that file)"
              if report is not None else " in your final message")
    return GATE_FOLLOWUP_PROMPT.format(
        failed=len(failed), total=len(results), report_clause=clause,
        details=failure_details(results))


def summary(results: list[GateRun], follow_ups: int) -> str:
    """"N/M green[, after K gate follow-up]" — the one spelling of the gate
    count, read by both the block heading (`render`) and the call's summary
    line (`cli.print_outcome`), so a format change cannot reach only one."""
    green = sum(r.passed for r in results)
    text = f"{green}/{len(results)} green"
    if follow_ups:
        text += f", after {follow_ups} gate follow-up"
    return text


def render(results: list[GateRun], follow_ups: int) -> str:
    """The block printed under the answer: what the WRAPPER saw, one line per
    gate, and the output tail of each red one."""
    lines = [f"## Wrapper gates ({summary(results, follow_ups)})"]
    lines.extend(f"- {'PASS' if r.passed else 'FAIL'} {_status(r)}"
                 for r in results)
    details = failure_details(results)
    if details:
        lines.append("")
        lines.append(details)
    return "\n".join(lines)
