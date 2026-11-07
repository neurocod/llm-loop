"""Claude print/SDK transport, independent of the worker registry.

The CLI's result is the turn boundary. Process exit 0 alone is insufficient:
budget/turn limits and API failures can arrive as result.is_error. Conversely,
a successful result followed by a failing process is retained but not reported
as a successful turn. Never expose assistant/thinking/tool events as answers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import threading
import time
from typing import Callable
import uuid

from functools import partial

from .. import host_policy
from ..answer_contract import REVIEWER_BRIEFING
from ..child_job import ChildJob
from ..gitlib import git

# Outside a work tree `git` answers non-zero, and that is an answer here (the
# cwd stands in), not a failure to report — so the returncode is read rather
# than raised on.
git = partial(git, check=False)


def invocation_root() -> Path:
    """The Git worktree containing the cwd, or the cwd itself."""
    try:
        result = git(["rev-parse", "--show-toplevel"])
        if result.returncode == 0:
            return Path(result.stdout.strip()).resolve()
    except OSError:
        pass
    return Path.cwd().resolve()

from .context import ContextTracker


REVIEW_TOOLS = "Read,Glob,Grep,Bash"
# Claude Code's own read-only classifier is the main policy. Live dontAsk run
# 2026-11-03 (two smoke agents): it approves `git show HEAD
# --stat`, `git diff HEAD~1 HEAD --stat`, `git status --short`, `git log
# --oneline -3`, `git grep -n -C 3 ...`, `git grep -n -A 22 ...` and pipes into
# head with no rule here; it refuses any `git -C <dir>`, an attached context
# count (`-A22`, `-B2`, `-C30`), `show --output=FILE`, `show --ext-diff`. Older
# logs (126 reviewer events.jsonl: 735 Bash calls, 72 denied) add `$`/`$?`,
# redirects and `cd`; 12 of those denials were `git -C <submodule> grep`,
# and the earlier unterminated `git -C <submodule> show *` carried 27 calls.
# So rules exist only for the host's submodule (`ReviewProfile.submodule`), as
# templates the briefing spells out.
# A rule matches the literal command text and '*' spans options as well, so
# every template ends option parsing BEFORE the wildcard. Trust model: a
# cooperative reviewer, guarded against accidents, not an attacker; still the
# classifier refuses --output=FILE (the one way to write a file), --ext-diff /
# --textconv (run configured programs) and worktree grep -O (runs a program)
# in the main tree, and the submodule must not be the weaker door, so the old
# `show *` is not restored. Cost: no option after the hash (2 of the 27 used
# --stat, hence its own form; `HASH -- path` still works after the marker),
# and log is limited by a range (HASH~10..HASH), not -N.
# Slot commits share the main tree's object store: no slot -C paths.
# https://code.claude.com/docs/en/permissions#wildcard-patterns
REVIEW_RULES = (
    "show --end-of-options",
    "show --stat --end-of-options",
    "log --end-of-options",
    "grep -n -E --end-of-options",
    "grep -n -E -C 30 --end-of-options",
)


class ReviewUnavailable(RuntimeError):
    """The review profile cannot start: no host names its gate hook."""


def review_profile() -> host_policy.ReviewProfile:
    """The host's review profile, or `ReviewUnavailable` naming why there is
    none: without the host's gate hook a review would run with only Claude's
    own classifier, which is not what this profile promises."""
    profile = host_policy.active().review
    if profile is None:
        raise ReviewUnavailable(
            "the review profile needs the host's gate hook (HostPolicy.review) "
            "and this run has no host policy; run through the host's entry "
            "point or use another --profile")
    return profile


def review_allow(submodule: str) -> list[str]:
    """The profile's tool grants: the read tools, and the `git -C <submodule>`
    templates (`REVIEW_RULES`) when the host names a submodule."""
    rules = [f"Bash(git -C {submodule} {rule} *)" for rule in REVIEW_RULES] if submodule else []
    return ["Read", "Glob", "Grep"] + rules


# The answer skeleton is `codex_agent --role reviewer`'s, verbatim: the
# orchestrator merges both reviews of one commit, and this profile used to ask
# for neither missed siblings nor a verdict. Only what this profile's tool
# policy implies is added in front of it.
def review_instructions() -> str:
    """The review profile's system-prompt addition; what the host says about
    reading other checkouts' commits (`ReviewProfile.note`) and its
    submodule's templates (`ReviewProfile.submodule`) sit between the grep
    rule and the shared skeleton. Without a host, neither."""
    profile = host_policy.active().review
    note = profile.note if profile else ""
    submodule = profile.submodule if profile else ""
    prefix = f"git -C {submodule}"
    templates = (
        f"The {submodule} submodule "
        "takes only these templates, typed as shown up to the marker, with no option "
        "after `--end-of-options` (a path after `--` is fine): "
        f"`{prefix} show --end-of-options HASH` (`show --stat --end-of-options HASH` "
        f"for the file list), `{prefix} log --end-of-options HASH~10..HASH`, "
        f"`{prefix} grep -n -E --end-of-options PATTERN HASH -- PATH`, "
        f"`{prefix} grep -n -E -C 30 --end-of-options PATTERN HASH -- PATH`. "
    ) if submodule else ""
    return (
        "You are a delegated code reviewer. Read the exact hashes the caller gives. "
        "Plain read-only git from the repository root is approved automatically: "
        "`git show HASH` (HASH:path for a whole committed file, --stat after it is fine), "
        "`git log`, `git diff A B`, `git grep -n -E PATTERN HASH -- PATH`, `git ls-tree`, "
        "`git ls-files`, `git status`. Give a grep context count as a separate word "
        "(`-C 30`, `-A 22`); an attached one (`-C30`, `-B2`) is refused. "
        + (note + " " if note else "") + templates +
        "These are refused: any `$` (also a regex end anchor) or `$?`, redirects into "
        "files, `cd`, temporary files. Pipes into head or grep are fine. "
        "Do not edit files, commit, or delegate "
        "further. Distinguish confirmed defects from plausible risks, and state "
        "which checks were actually run. Your tools cannot run tests, builds or "
        "scripts: report a gate the task asks you to run as not run.\n" + REVIEWER_BRIEFING
    )


def _root() -> Path:
    """The host root (`host_policy.installed`), else the caller's checkout."""
    return host_policy.root() or invocation_root()


def review_settings() -> str:
    """Reuse the project's compiled gate without its broad Bash permissions."""
    profile = review_profile()
    project = json.loads((_root() / profile.settings).read_text(encoding="utf-8"))
    command_text = f'"$CLAUDE_PROJECT_DIR/{profile.gate}/hooks/ask_user_gate.exe"'
    expected = {"matcher": "Bash|PowerShell|Monitor", "hooks": [
        {"type": "command", "command": command_text}]}
    # A project setting also grants unrestricted Bash. Copy only the known
    # deny-only hook; fail if its entry changes rather than importing grants.
    if expected not in project["hooks"]["PreToolUse"]:
        raise ValueError("Project settings no longer contain the compiled ask_user_gate hook")
    # CLAUDE_PROJECT_DIR follows the child cwd, which may be another worktree.
    # Bind the hook to the executable verified by verify_review_gate().
    expected["hooks"][0]["command"] = f'"{review_gate_path().as_posix()}"'
    return json.dumps({"hooks": {"PreToolUse": [expected]}})


def review_gate_path() -> Path:
    return _root() / review_profile().gate / "hooks" / "ask_user_gate.exe"


def verify_review_gate() -> None:
    """Fail before Claude starts if its configured gate cannot run; without
    a host, `ReviewUnavailable` (`review_profile`)."""
    gate_root = _root() / review_profile().gate
    gate = review_gate_path()
    build = (gate_root / "cpp" / "build.py").relative_to(_root()).as_posix()
    if not gate.is_file():
        raise FileNotFoundError(f"Review gate is missing: {gate}; build it with "
                                f"python {build}")
    cpp = gate_root / "cpp"
    sources = [gate_root / "hooks" / "ask_user_gate.py"]
    # Keep the compiled-source set aligned with parity_check.newer_sources.
    sources.extend(path for path in cpp.iterdir() if path.suffix in {".cpp", ".inc", ".h", ".hpp"})
    sources.extend(path for path in (cpp / "third_party").rglob("*") if path.is_file())
    if any(source.stat().st_mtime > gate.stat().st_mtime for source in sources):
        raise RuntimeError(f"Review gate is older than its source: {gate}; rebuild it with "
                           f"python {build}")


def command(agent: dict, resume: bool) -> list[str]:
    """Review is a tool permission policy, not an OS filesystem sandbox.

    Ignore ambient permission grants and MCP servers for this profile; a broad
    user Bash allow rule would otherwise silently widen the review policy.
    Claude's built-in read-only command classification also remains active.
    Authentication is still loaded by the CLI (do not use --bare with OAuth).
    """
    executable = shutil.which("claude")
    if not executable:
        raise FileNotFoundError("claude executable not found on PATH")
    opts = agent["options"]
    argv = [executable, "-p", "--input-format", "stream-json",
            "--output-format", "stream-json", "--verbose",
            "--permission-mode", "dontAsk", "--permission-prompts", "none",
            "--resume" if resume else "--session-id", agent["session_id"]]
    if opts.get("model"):
        argv += ["--model", opts["model"]]
    if opts.get("effort"):
        argv += ["--effort", opts["effort"]]
    if opts.get("profile", "review") == "review":
        argv += ["--tools", REVIEW_TOOLS, "--allowedTools",
                 *review_allow(review_profile().submodule),
                 "--setting-sources", "", "--settings", review_settings(),
                 "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                 "--disable-slash-commands", "--append-system-prompt", review_instructions()]
    return argv


def child_environment(*, review: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    # A delegated process is a separate session, even when its caller is Claude.
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    env.pop("CLAUDE_AGENT_REVIEW_GATE", None)
    if review:
        # The compiled gate suppresses an interactive escape hint in dontAsk.
        env["CLAUDE_AGENT_REVIEW_GATE"] = "1"
    env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    return env


def terminate_tree(proc: subprocess.Popen) -> None:
    """Terminate only the child owned by this live Popen, never a persisted PID."""
    if os.name == "nt":
        if proc.poll() is not None:
            return
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if proc.poll() is None:
        proc.kill()
    proc.wait()


RATE_LIMIT_EVENT = "rate_limit_event"


def week_used_percent(event: dict) -> tuple[float | None, str]:
    """The account's weekly utilisation in percent from a `rate_limit_event`,
    or (None, why) when the event does not carry it.

    The CLI writes one such event per turn, near its start (6 of 6 sampled
    turns in `.claude-agents/`, 2026-11-06), shaped
    `{"rate_limit_info": {..., "unifiedWindows": {"seven_day":
    {"utilization": 0.18, "resetsAt": 1794402000, "status": "allowed"}}}}`.
    `utilization` is a FRACTION there (0.81 stood beside
    `"surpassedThreshold": 0.75`), unlike the percent of `/api/oauth/usage`.

    Read off the stream rather than from that endpoint (`llm_loop.usage`):
    the stream is parsed anyway, needs no OAuth token, and puts no request on
    the agent's start path. Undocumented, so a missing field is a `why`, not
    an error — `agent_counter.Beacon` turns it into one log line.
    """
    info = event.get("rate_limit_info")
    windows = info.get("unifiedWindows") if isinstance(info, dict) else None
    week = windows.get("seven_day") if isinstance(windows, dict) else None
    used = week.get("utilization") if isinstance(week, dict) else None
    if isinstance(used, (int, float)) and not isinstance(used, bool):
        return float(used) * 100.0, ""
    return None, f"{RATE_LIMIT_EVENT} without unifiedWindows.seven_day.utilization"


def run_turn(agent: dict, turn: dict, directory: Path,
             cancelled: Callable[[], bool], *,
             on_session: Callable[[str], None] | None = None,
             on_context: Callable[[dict], None] | None = None,
             on_week_used: Callable[[float | None, str], None] | None = None,
             deadline: float | None = None) -> dict:
    """`deadline` is an absolute `time.monotonic()` bound for a caller that
    spends one logical turn on several invocations (the worker's follow-up):
    the later call gets what is left of `turn_timeout`, not a fresh one.
    Omitted, the bound is `turn_timeout` from this call's start.

    `on_week_used(percent, why)` receives each `rate_limit_event`'s weekly
    figure (`week_used_percent`), for the on-screen counter."""
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result = None
    failure = ""
    status = "failed"
    session_id = ""
    proc = None
    reader = None
    writer = None
    write_errors = []
    job = None
    events: queue.Queue = queue.Queue()
    context = ContextTracker(agent.get("context"))
    opts = agent["options"]
    timeout = opts.get("turn_timeout")  # No model deadline unless requested.
    if deadline is None and timeout:
        deadline = started + timeout
    with (directory / "events.jsonl").open("w", encoding="utf-8") as log, \
            (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            if opts.get("profile", "review") == "review":
                verify_review_gate()
            job = ChildJob()
            proc = subprocess.Popen(
                command(agent, turn["resume"]), cwd=agent["cwd"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                env=child_environment(review=opts.get("profile", "review") == "review"),
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
                   else {"start_new_session": True}))
            job.assign(proc)

            def read_stream():
                try:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        events.put(line)
                finally:
                    events.put(None)

            reader = threading.Thread(target=read_stream, daemon=True)
            reader.start()
            envelope = {"type": "user", "uuid": str(uuid.uuid4()),
                        "session_id": agent["session_id"], "parent_tool_use_id": None,
                        "message": {"role": "user", "content": turn["prompt"]}}
            def write_message():
                try:
                    proc.stdin.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                    proc.stdin.close()
                except (OSError, ValueError) as exc:
                    write_errors.append(exc)

            # A brief can exceed pipe capacity while Claude is still starting.
            # Keep cancellation/deadline polling independent of stdin draining.
            writer = threading.Thread(target=write_message, daemon=True)
            writer.start()
            eof = False
            reaped_descendants = False
            while not eof or proc.poll() is None:
                if proc.poll() is not None and not reaped_descendants:
                    # A finished CLI can leave a tool holding stdout open.
                    # Release descendants before waiting for the reader's EOF,
                    # otherwise the default unlimited turn never completes.
                    terminate_tree(proc)
                    job.close()
                    reaped_descendants = True
                if cancelled():
                    status, failure = "interrupted", "Interrupted by caller"
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    failure = f"Turn exceeded caller deadline ({timeout or 0:g}s)"
                    break
                if write_errors:
                    failure = f"Could not deliver message: {write_errors[0]}"
                    break
                try:
                    line = events.get(timeout=0.1)
                except queue.Empty:
                    continue
                if line is None:
                    eof = True
                    continue
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue  # Preserve diagnostics in the raw log.
                if not isinstance(event, dict):
                    continue
                snapshot = context.observe(event)
                if snapshot is not None and on_context:
                    on_context(snapshot)
                if on_week_used and event.get("type") == RATE_LIMIT_EVENT:
                    on_week_used(*week_used_percent(event))
                if event.get("session_id") and event["session_id"] != session_id:
                    session_id = event["session_id"]
                    if on_session:
                        on_session(session_id)
                if event.get("type") == "result":
                    result = event
            if not failure:
                code = proc.wait()
                if result is None:
                    failure = f"Claude exited {code} without a result event; see stderr.log"
                elif result.get("is_error") or result.get("subtype") != "success":
                    details = result.get("errors") or result.get("result") or result.get("subtype")
                    failure = str(details)
                elif code:
                    failure = f"Claude returned a result but exited {code}; see stderr.log"
                else:
                    status = "completed"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
        finally:
            if proc is not None:
                terminate_tree(proc)
                if job:
                    job.close()
                if writer:
                    writer.join()
                if reader:
                    reader.join()
                for stream in (proc.stdin, proc.stdout):
                    if stream and not stream.closed:
                        stream.close()
            elif job:
                job.close()
    outcome = {"status": status, "answer": (result or {}).get("result", ""),
               "error": failure, "session_id": session_id,
               "duration_s": round(time.monotonic() - started, 3),
               "usage": (result or {}).get("usage", {}),
               "total_cost_usd": (result or {}).get("total_cost_usd"),
               "permission_denials": (result or {}).get("permission_denials", []),
               "context": context.snapshot(),
               "log_dir": str(directory)}
    save_outcome(directory, outcome)
    return outcome


def save_outcome(directory: Path, outcome: dict) -> None:
    """A turn's artifacts in its log directory: `result.json` and `answer.md`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(
        json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "answer.md").write_text(outcome.get("answer") or "", encoding="utf-8")
