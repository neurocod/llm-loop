#!/usr/bin/env python3
"""Continuous Claude - run an AI coding agent (Claude Code or Codex CLI) in a loop
with automatic commit / pull-request / merge management.

Python port of continuous_claude.sh v0.24.7
(https://github.com/AnandChowdhary/continuous-claude), Copyright (c) Anand
Chowdhary, MIT License. This file is a derivative work of that script and
carries its notice; the rest of this repository is Copyright (c) 2026 neurocod,
also MIT.

Not part of the llm_loop library - see ideas/README.md.

Differences from the Bash original:
- No external `jq` dependency (native JSON parsing).
- Self-update is removed: the `update` command prints a notice, and
  --auto-update / --disable-updates are accepted as no-ops for CLI compatibility.
- Rate/error/cost sliding-window logs live in memory instead of temp files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

VERSION = "0.1.0"  # port of continuous_claude.sh v0.24.7

CLAUDE_DEFAULT_FLAGS = [
    "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose",
]
CODEX_DEFAULT_FLAGS = [
    "--json", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check",
]

RATE_LIMIT_WINDOW_SECONDS = 3600
RATE_LIMIT_DEFAULT_BACKOFF = 300
PR_CHECKS_POLL_SECONDS = 10
PR_CHECKS_MAX_POLLS = 180  # 180 * 10s = 30 minutes

# ---------------------------------------------------------------------------
# Prompt templates (ported verbatim from the Bash script)
# ---------------------------------------------------------------------------

PROMPT_COMMIT_MESSAGE = (
    "Please review all uncommitted changes in the git repository (both modified and "
    "new files). Write a commit message with: (1) a short one-line summary, (2) two "
    "newlines, (3) then a detailed explanation. Do not include any footers or metadata "
    "like 'Generated with Claude Code' or 'Co-Authored-By'. Feel free to look at the "
    "last few commits to get a sense of the commit message style for consistency. "
    "First run 'git add .' to stage all changes including new untracked files, then "
    "commit using 'git commit -m \"your message\"' (don't push, just commit, no need "
    "to ask for confirmation)."
)

PROMPT_WORKFLOW_CONTEXT = """## CONTINUOUS WORKFLOW CONTEXT

This is part of a continuous development loop where work happens incrementally across multiple iterations. You might run once, then a human developer might make changes, then you run again, and so on. This could happen daily or on any schedule.

**Important**: You don't need to complete the entire goal in one iteration. Just make meaningful progress on one thing, then leave clear notes for the next iteration (human or AI). Think of it as a relay race where you're passing the baton.

**Do NOT commit or push changes** - The automation will handle committing and pushing your changes after you finish. Just focus on making the code changes.

**Project Completion Signal**: If you determine that not just your current task but the ENTIRE project goal is fully complete (nothing more to be done on the overall goal), only include the exact phrase "COMPLETION_SIGNAL_PLACEHOLDER" in your response. Only use this when absolutely certain that the whole project is finished, not just your individual task. We will stop working on this project when multiple developers independently determine that the project is complete.

## PRIMARY GOAL"""

PROMPT_NOTES_UPDATE_EXISTING = (
    "Update the `{notes_file}` file with relevant context for the next iteration. "
    "Add new notes and remove outdated information to keep it current and useful."
)
PROMPT_NOTES_CREATE_NEW = (
    "Create a `{notes_file}` file with relevant context and instructions for the next iteration."
)
PROMPT_NOTES_GUIDELINES = """

This file helps coordinate work across iterations (both human and AI developers). It should:

- Contain relevant context and instructions for the next iteration
- Stay concise and actionable (like a notes file, not a detailed report)
- Help the next developer understand what to do next

The file should NOT include:
- Lists of completed work or full reports
- Information that can be discovered by running tests/coverage
- Unnecessary details"""

PROMPT_KNOWLEDGE_UPDATE_EXISTING = (
    "Update the `{knowledge_file}` file with durable project knowledge learned during this iteration."
)
PROMPT_KNOWLEDGE_CREATE_NEW = (
    "Create a `{knowledge_file}` file with durable project knowledge learned during this iteration."
)
PROMPT_KNOWLEDGE_GUIDELINES = """

This file is long-lived project memory for future AI and human developers. It should:

- Capture reusable conventions, commands, architecture decisions, pitfalls, and style preferences
- Stay laconic and information dense
- Avoid per-iteration status logs, completed-work summaries, and facts that are easy to rediscover"""

PROMPT_REVIEWER_CONTEXT = """## CODE REVIEW CONTEXT

You are performing a review pass on changes just made by another developer. This is NOT a new feature implementation - you are reviewing and validating existing changes using the instructions given below by the user. Feel free to use git commands to see what changes were made if it's helpful to you.

**Do NOT commit or push changes** - The automation will handle committing and pushing your changes after you finish. Just focus on validating and fixing any issues."""

PROMPT_DEFAULT_REVIEWER = (
    "Review the currently changed files on this branch before I ship. Look at the diff "
    "and read everything that changed. Run the test suite, typecheck, lint, formatter, "
    "etc., whatever is available, and fix anything that fails. Invoke the /simplify "
    "skill on the changed files to dedupe, extract clean abstractions where patterns "
    "repeat, and tighten naming, but don't over-abstract. Then start the dev server if "
    "any, and drive the app with real tooling, like a browser test similar to the "
    "agent-browser CLI or whatever else is relevant to this project. Screenshot "
    "surfaces you touched, click through the golden path and edge cases, and watch the "
    "dev server logs and browser console for warnings or errors where relevant. Report "
    "back with what changed, what you simplified, test results, and a "
    "screenshot-backed walkthrough, and flag anything you couldn't verify. No need to "
    "commit or push."
)

PROMPT_CI_FIX_CONTEXT = """## CI FAILURE FIX CONTEXT

You are analyzing and fixing a CI/CD failure for a pull request.

**Your task:**
1. Inspect the failed CI workflow using the commands below
2. Analyze the error logs to understand what went wrong
3. Make the necessary code changes to fix the issue
4. Stage and commit your changes (they will be pushed to update the PR)

**Commands to inspect CI failures:**
- `gh run list --status failure --limit 3` - List recent failed runs
- `gh run view <RUN_ID> --log-failed` - View failed job logs (shorter output)
- `gh run view <RUN_ID> --log` - View full logs for a specific run

**Important:**
- Focus only on fixing the CI failure, not adding new features
- Make minimal changes necessary to pass CI
- If the failure seems unfixable (e.g., flaky test, infrastructure issue), explain why in your response"""

PROMPT_COMMENT_REVIEW_CONTEXT = """## PR COMMENT REVIEW CONTEXT

You are addressing review comments on a pull request.

**Your task:**
1. Use `gh api repos/{owner}/{repo}/pulls/{pr}/comments` to read inline code review comments
2. Use `gh api repos/{owner}/{repo}/issues/{pr}/comments` to read PR-level comments
3. Analyze each comment and determine if it requires code changes
4. Make the necessary code changes to address the feedback
5. Stage, commit, AND PUSH your changes with a clear commit message describing what comments you addressed

**Important:**
- Focus only on addressing the review comments, not adding new features
- Make minimal changes necessary to address the feedback
- If a comment is just informational or a question, no code changes are needed for it"""

# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def log(message: str = "") -> None:
    """All progress output goes to stderr, like the Bash original."""
    print(message, file=sys.stderr, flush=True)


def utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def parse_duration(duration_str: str) -> int | None:
    """Parse "2h", "30m", "1h30m", "90s" into seconds. None on error."""
    remaining = re.sub(r"\s+", "", duration_str or "")
    if not remaining:
        return None
    total = 0
    for pattern, factor in ((r"(\d+)[hH]", 3600), (r"(\d+)[mM]", 60), (r"(\d+)[sS]", 1)):
        match = re.search(pattern, remaining)
        if match:
            total += int(match.group(1)) * factor
            remaining = remaining.replace(match.group(0), "", 1)
    if remaining or total == 0:
        return None
    return total


def format_duration(seconds: int | float | None) -> str:
    seconds = int(seconds or 0)
    if seconds == 0:
        return "0s"
    hours, minutes, secs = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    result = ""
    if hours:
        result += f"{hours}h"
    if minutes:
        result += f"{minutes}m"
    if secs or not result:
        result += f"{secs}s"
    return result


def truncate(text: str, limit: int = 1000) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def relpath(path: str, cwd: str) -> str:
    for prefix in (cwd + "/", cwd + os.sep):
        if path.startswith(prefix):
            return path[len(prefix):]
    return "." if path == cwd else path


def strip_cwd(text: str, cwd: str) -> str:
    return text.replace(cwd + "/", "").replace(cwd + os.sep, "")


def tail_lines(text: str, count: int = 80) -> str:
    return "\n".join((text or "").splitlines()[-count:])


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

EXE: dict[str, str] = {}  # resolved executable paths, filled by validate_requirements


def resolve_exe(name: str) -> str:
    return EXE.get(name) or shutil.which(name) or name


def run_cmd(args: list[str], **kwargs) -> tuple[int, str, str]:
    """Run a command capturing stdout/stderr. Returns (code, stdout, stderr)."""
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            stdin=subprocess.DEVNULL, **kwargs
        )
    except OSError as exc:
        return 127, "", str(exc)
    return proc.returncode, proc.stdout, proc.stderr


def git(*args: str) -> tuple[int, str, str]:
    return run_cmd([resolve_exe("git"), *args])


def git_ok(*args: str) -> bool:
    return git(*args)[0] == 0


def gh(*args: str) -> tuple[int, str, str]:
    return run_cmd([resolve_exe("gh"), *args])


def in_git_repo() -> bool:
    return git_ok("rev-parse", "--git-dir")


def current_branch() -> str:
    code, out, _ = git("rev-parse", "--abbrev-ref", "HEAD")
    return out.strip() if code == 0 and out.strip() else "main"


def untracked_files() -> str:
    return git("ls-files", "--others", "--exclude-standard")[1].strip()


def has_uncommitted_changes() -> bool:
    if not git_ok("diff", "--quiet", "--ignore-submodules=dirty"):
        return True
    if not git_ok("diff", "--cached", "--quiet", "--ignore-submodules=dirty"):
        return True
    return bool(untracked_files())


def repo_has_pending_changes() -> bool:
    return in_git_repo() and has_uncommitted_changes()


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    command: str
    display_name: str
    install_url: str
    default_flags: list[str]


PROVIDERS = {
    "claude": ProviderSpec(
        "claude", "claude", "Claude Code", "https://claude.ai/code", CLAUDE_DEFAULT_FLAGS
    ),
    "codex": ProviderSpec(
        "codex", "codex", "Codex CLI",
        "https://help.openai.com/en/articles/11096431", CODEX_DEFAULT_FLAGS,
    ),
}


CLAUDE_TOOL_EMOJI = {
    "Read": "📖", "Write": "✍️", "Edit": "✏️", "MultiEdit": "✏️", "Bash": "💻",
    "Glob": "📁", "Grep": "🔎", "Task": "📋", "NotebookEdit": "📓",
    "AskUserQuestion": "❓", "Skill": "⚡", "SlashCommand": "⚡",
    "TaskOutput": "📤", "BashOutput": "📤", "KillShell": "🛑",
    "ExitPlanMode": "🗺️", "EnterPlanMode": "🗺️",
}


def claude_tool_emoji(name: str) -> str:
    if name in CLAUDE_TOOL_EMOJI:
        return CLAUDE_TOOL_EMOJI[name]
    if name.startswith(("WebFetch",)):
        return "🌍"
    if name.startswith(("WebSearch",)):
        return "🔍"
    if re.search(r"Todo|TaskCreate|TaskUpdate|TaskList|TaskGet", name, re.IGNORECASE):
        return "📝"
    if name.startswith("mcp__"):
        return "🔌"
    return "🛠️"


def claude_tool_detail(name: str, inp: dict, cwd: str) -> str:
    """Human-readable one-line summary of a tool_use event (port of the jq filter)."""
    def path_of(key: str = "file_path") -> str:
        return relpath(str(inp.get(key, "")), cwd)

    try:
        if name == "Bash":
            command = strip_cwd(str(inp.get("command", "")), cwd)
            return truncate(command.split("\n")[0])
        if name == "Read":
            detail = path_of()
            if inp.get("offset"):
                detail += f" (line {inp['offset']})"
            return detail
        if name in ("Write", "Edit", "MultiEdit"):
            return path_of()
        if name == "Glob":
            detail = str(inp.get("pattern", ""))
            if inp.get("path"):
                detail += " in " + relpath(str(inp["path"]), cwd)
            return detail
        if name == "Grep":
            detail = f'"{inp.get("pattern", "")}"'
            if inp.get("path"):
                detail += " in " + relpath(str(inp["path"]), cwd)
            if inp.get("glob"):
                detail += f" ({inp['glob']})"
            return detail
        if name.startswith("WebFetch"):
            return f"{inp.get('url', '')} → {truncate(str(inp.get('prompt', '')))}"
        if name.startswith("WebSearch"):
            detail = f'"{inp.get("query", "")}"'
            if inp.get("allowed_domains"):
                detail += " (domains: " + ", ".join(inp["allowed_domains"]) + ")"
            return detail
        if name == "Task":
            return f"[{inp.get('subagent_type', 'agent')}] {inp.get('description', '')}"
        if name == "NotebookEdit":
            return f"{relpath(str(inp.get('notebook_path', '')), cwd)} [{inp.get('edit_mode', 'replace')}]"
        if name == "AskUserQuestion":
            questions = inp.get("questions") or [{}]
            return truncate(str(questions[0].get("question", "")))
        if name in ("Skill", "SlashCommand"):
            detail = "/" + str(inp.get("skill") or inp.get("command") or "")
            if inp.get("args"):
                detail += " " + str(inp["args"])
            return detail
        if re.search(r"TodoWrite", name, re.IGNORECASE):
            todos = inp.get("todos") or []
            in_progress = [t.get("content") or t.get("activeForm") for t in todos
                           if t.get("status") == "in_progress"]
            first = todos[0].get("content") or todos[0].get("activeForm", "") if todos else ""
            return truncate(str(in_progress[0] if in_progress else first or ""))
        if re.search(r"TaskCreate", name, re.IGNORECASE):
            return str(inp.get("subject") or inp.get("description") or "")
        if re.search(r"TaskUpdate", name, re.IGNORECASE):
            return f"#{inp.get('taskId', '')} → {inp.get('status', 'update')}"
        if re.search(r"TaskList|TaskGet", name, re.IGNORECASE):
            return f"#{inp['taskId']}" if inp.get("taskId") else ""
        if name in ("TaskOutput", "BashOutput"):
            return "id:" + str(inp.get("task_id") or inp.get("bash_id") or "")
        if name == "KillShell":
            return "id:" + str(inp.get("shell_id", ""))
        if name in ("ExitPlanMode", "EnterPlanMode"):
            return ""
        if name.startswith("mcp__"):
            return "/".join(name.split("__")[1:]) or name
    except (TypeError, KeyError, IndexError, AttributeError):
        pass
    return name


def render_claude_event(event: dict, cwd: str) -> tuple[list[str], list[str]]:
    """Returns (assistant text lines, tool-use display lines)."""
    texts: list[str] = []
    tools: list[str] = []
    if event.get("type") != "assistant":
        return texts, tools
    for item in (event.get("message") or {}).get("content") or []:
        if item.get("type") == "text" and item.get("text"):
            texts.extend(item["text"].splitlines())
        elif item.get("type") == "tool_use":
            name = item.get("name", "unknown")
            detail = claude_tool_detail(name, item.get("input") or {}, cwd)
            tools.append(f"{claude_tool_emoji(name)} {detail or name}")
    return texts, tools


def render_codex_event(event: dict, cwd: str) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    tools: list[str] = []
    item = event.get("item") or {}
    etype = event.get("type")
    if etype == "item.completed" and item.get("type") == "agent_message" and item.get("text"):
        texts.extend(item["text"].splitlines())
    elif etype == "item.started" and item.get("type") == "command_execution":
        command = strip_cwd(str(item.get("command", "")), cwd)
        tools.append("💻 " + truncate(command.split("\n")[0]))
    elif etype == "item.completed" and item.get("type") == "command_execution":
        command = strip_cwd(str(item.get("command", "")), cwd)
        tools.append(f"📤 exit {item.get('exit_code', '')}: " + truncate(command.split("\n")[0]))
    elif etype == "item.completed" and item.get("path") is not None:
        tools.append("🛠️ " + relpath(str(item.get("path", "")), cwd))
    return texts, tools


def parse_agent_result(provider: str, events: list[dict], invalid_json: bool) -> str:
    """Returns "success" or an error type string (claude_error, codex_error, ...)."""
    if invalid_json or not events:
        return "invalid_json"
    if provider == "claude":
        if events[-1].get("is_error") is True:
            return "claude_error"
    elif provider == "codex":
        if any(e.get("type") in ("error", "turn.failed") for e in events):
            return "codex_error"
        if not any(e.get("type") == "turn.completed" for e in events):
            return "codex_incomplete"
    return "success"


def extract_agent_result_text(provider: str, events: list[dict]) -> str:
    if not events:
        return ""
    if provider == "claude":
        return str(events[-1].get("result") or "")
    if provider == "codex":
        parts = [
            (e.get("item") or {}).get("text") or ""
            for e in events
            if e.get("type") == "item.completed"
            and (e.get("item") or {}).get("type") == "agent_message"
        ]
        return "\n".join(parts)
    return ""


def extract_agent_error_text(provider: str, events: list[dict]) -> str:
    if not events:
        return ""
    if provider == "claude":
        last = events[-1]
        if last.get("is_error") is True:
            return str(last.get("result") or last.get("error") or "Unknown error")
        return ""
    if provider == "codex":
        errors = [
            str(e.get("message") or e.get("error") or e)
            for e in events
            if e.get("type") in ("error", "turn.failed")
        ]
        return errors[-1] if errors else ""
    return ""


def extract_agent_cost(provider: str, events: list[dict], cfg: "Config") -> float | None:
    if not events:
        return None
    if provider == "claude":
        cost = events[-1].get("total_cost_usd")
        return float(cost) if cost is not None else None
    if provider == "codex":
        if cfg.codex_input_cost is None or cfg.codex_output_cost is None:
            return None
        usage = _last_codex_usage(events)
        if not usage:
            return None
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if input_tokens is None or output_tokens is None:
            return None
        cached = usage.get("cached_input_tokens") or 0
        cached_rate = (
            cfg.codex_cached_input_cost
            if cfg.codex_cached_input_cost is not None
            else cfg.codex_input_cost
        )
        uncached = max(input_tokens - cached, 0)
        return (
            uncached * cfg.codex_input_cost
            + cached * cached_rate
            + output_tokens * cfg.codex_output_cost
        ) / 1_000_000
    return None


def _last_codex_usage(events: list[dict]) -> dict | None:
    completed = [e for e in events if e.get("type") == "turn.completed"]
    return (completed[-1].get("usage") or None) if completed else None


def extract_agent_usage_summary(provider: str, events: list[dict]) -> str:
    if provider != "codex":
        return ""
    usage = _last_codex_usage(events)
    if not usage:
        return ""
    return (
        f"Tokens: input {usage.get('input_tokens', 0)}"
        f", cached input {usage.get('cached_input_tokens', 0)}"
        f", output {usage.get('output_tokens', 0)}"
    )


# ---------------------------------------------------------------------------
# Rate limiting and rate-limit error detection
# ---------------------------------------------------------------------------


class RateWindow:
    """Sliding-window log of (timestamp, value) pairs (in-memory)."""

    def __init__(self, window_seconds: int = RATE_LIMIT_WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self.entries: deque[tuple[float, float]] = deque()

    def prune(self, now: float | None = None) -> None:
        cutoff = (now if now is not None else time.time()) - self.window_seconds
        while self.entries and self.entries[0][0] < cutoff:
            self.entries.popleft()

    def add(self, value: float = 0.0, now: float | None = None) -> None:
        now = now if now is not None else time.time()
        self.prune(now)
        self.entries.append((now, value))

    def count(self) -> int:
        self.prune()
        return len(self.entries)

    def total(self) -> float:
        self.prune()
        return sum(value for _, value in self.entries)

    def oldest(self) -> float | None:
        self.prune()
        return self.entries[0][0] if self.entries else None


def seconds_until_time_today_or_tomorrow(hour: int, minute: int = 0,
                                         timezone: str | None = None) -> int:
    tz = None
    if timezone:
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(timezone)
        except Exception:
            tz = None
    now = datetime.now(tz)
    now_seconds = now.hour * 3600 + now.minute * 60 + now.second
    target_seconds = (hour % 24) * 3600 + minute * 60
    wait = target_seconds - now_seconds
    if wait <= 0:
        wait += 86400
    return wait


def parse_reset_time_wait_seconds(text: str) -> int | None:
    """Parse "resets at 5pm (America/Chicago)" style hints into a wait in seconds."""
    flat = (text or "").replace("\n", " ")
    match = re.search(
        r"resets(?:\s+at)?\s+(\d{1,2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*(?:\(([^)]*)\))?",
        flat,
    )
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = (match.group(3) or "").lower()
    timezone = match.group(4) or None
    if ampm == "pm" and hour < 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    return seconds_until_time_today_or_tomorrow(hour, minute, timezone)


RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit_error", "too many requests", "429",
    "overloaded_error", "temporarily overloaded", "limit reached",
)


def detect_rate_limit_wait_seconds(text: str) -> int | None:
    """Return a wait in seconds if the text looks like a rate-limit error, else None."""
    lower = (text or "").lower()
    if not any(marker in lower for marker in RATE_LIMIT_MARKERS):
        return None

    reset_wait = parse_reset_time_wait_seconds(text)
    if reset_wait is not None:
        return reset_wait

    retry_after = re.search(r"retry[-_ ]?after[^0-9]{0,20}(\d+)", lower)
    if retry_after:
        return int(retry_after.group(1))

    wait_match = re.search(
        r"(?:try again|retry|wait)[^0-9]{0,20}(\d+)\s*(seconds?|secs?|s|minutes?|mins?|m)\b",
        lower,
    )
    if wait_match:
        amount = int(wait_match.group(1))
        unit = wait_match.group(2)
        return amount * 60 if unit.startswith("m") else amount

    return RATE_LIMIT_DEFAULT_BACKOFF


COMPLETION_HEURISTIC_PHRASES = (
    "all scoped tasks complete", "all requested tasks complete",
    "all tasks complete", "nothing left to do", "no remaining work",
)


def detect_positive_completion_heuristic(result_text: str) -> bool:
    normalized = (result_text or "").lower()
    return any(phrase in normalized for phrase in COMPLETION_HEURISTIC_PHRASES)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    prompt: str = ""
    provider: str = "claude"
    review_provider: str = ""
    max_runs: int | None = None
    max_cost: float | None = None
    max_duration: int | None = None  # seconds
    enable_commits: bool = True
    disable_branches: bool = False
    git_branch_prefix: str = "continuous-claude/"
    merge_strategy: str = "squash"
    github_owner: str = ""
    github_repo: str = ""
    notes_file: str = "SHARED_TASK_NOTES.md"
    knowledge_file: str = ""
    worktree_name: str = ""
    worktree_base_dir: str = "../continuous-claude-worktrees"
    cleanup_worktree: bool = False
    list_worktrees: bool = False
    dry_run: bool = False
    completion_signal: str = "CONTINUOUS_CLAUDE_PROJECT_COMPLETE"
    completion_threshold: int = 3
    stall_threshold: int | None = None
    max_calls_per_hour: int | None = None
    error_threshold: int = 3
    review_prompt: str = ""
    ci_retry_enabled: bool = True
    ci_retry_max_attempts: int = 1
    comment_review_enabled: bool = True
    comment_review_max_attempts: int = 1
    command_retry_max_attempts: int = 3
    command_retry_base_delay: int = 5
    codex_input_cost: float | None = None
    codex_output_cost: float | None = None
    codex_cached_input_cost: float | None = None
    extra_agent_flags: list[str] = field(default_factory=list)

    @property
    def effective_review_provider(self) -> str:
        return self.review_provider or self.provider


def fail(message: str) -> None:
    log(f"❌ Error: {message}")
    sys.exit(1)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="continuous-claude", add_help=True, allow_abbrev=False,
        description="Continuous Claude - Run Claude Code iteratively with automatic PR management",
    )
    add = parser.add_argument
    add("-v", "--version", action="version", version=f"continuous-claude version {VERSION}")
    add("-p", "--prompt", default="")
    add("-m", "--max-runs", default=None)
    add("--provider", default=os.environ.get("CONTINUOUS_CLAUDE_PROVIDER", "claude"))
    add("--review-provider", default="")
    add("--max-cost", default=None)
    add("--max-duration", default=None)
    add("--codex-input-cost-per-million", dest="codex_input_cost",
        default=os.environ.get("CODEX_INPUT_COST_PER_MILLION") or None)
    add("--codex-output-cost-per-million", dest="codex_output_cost",
        default=os.environ.get("CODEX_OUTPUT_COST_PER_MILLION") or None)
    add("--codex-cached-input-cost-per-million", dest="codex_cached_input_cost",
        default=os.environ.get("CODEX_CACHED_INPUT_COST_PER_MILLION") or None)
    add("--owner", dest="github_owner", default="")
    add("--repo", dest="github_repo", default="")
    add("--git-branch-prefix", default="continuous-claude/")
    add("--merge-strategy", default="squash")
    add("--notes-file", default="SHARED_TASK_NOTES.md")
    add("--knowledge-file", default="")
    add("--disable-commits", dest="enable_commits", action="store_false")
    add("--disable-branches", action="store_true")
    add("--worktree", dest="worktree_name", default="")
    add("--worktree-base-dir", default="../continuous-claude-worktrees")
    add("--cleanup-worktree", action="store_true")
    add("--list-worktrees", action="store_true")
    add("--dry-run", action="store_true")
    add("--completion-signal", default="CONTINUOUS_CLAUDE_PROJECT_COMPLETE")
    add("--completion-threshold", default="3")
    add("--stall-threshold", default=None)
    add("--max-calls-per-hour", default=None)
    add("--error-threshold", default="3")
    add("-r", "--review-prompt", nargs="?", const=PROMPT_DEFAULT_REVIEWER, default="")
    add("--disable-ci-retry", dest="ci_retry_enabled", action="store_false")
    add("--ci-retry-max", default="1")
    add("--disable-comment-review", dest="comment_review_enabled", action="store_false")
    add("--comment-review-max", default="1")
    add("--command-retry-max", default="3")
    add("--command-retry-base-delay", default="5")
    # Self-update flags kept as accepted no-ops for compatibility with the Bash CLI.
    add("--auto-update", action="store_true", help=argparse.SUPPRESS)
    add("--disable-updates", action="store_true", help=argparse.SUPPRESS)
    return parser


def positive_int(value: str, flag: str) -> int:
    if not re.fullmatch(r"\d+", value or "") or int(value) < 1:
        fail(f"{flag} must be a positive integer")
    return int(value)


def is_non_negative_number(value: str) -> bool:
    return bool(re.fullmatch(r"\d+\.?\d*", value or ""))


def is_positive_number(value: str) -> bool:
    return is_non_negative_number(value) and float(value) > 0


def parse_config(argv: list[str]) -> Config:
    # Everything after a literal "--" is forwarded to the provider CLI verbatim.
    passthrough: list[str] = []
    if "--" in argv:
        split_at = argv.index("--")
        passthrough = argv[split_at + 1:]
        argv = argv[:split_at]

    args, unknown = build_arg_parser().parse_known_args(argv)
    cfg = Config(
        prompt=args.prompt,
        provider=args.provider,
        review_provider=args.review_provider,
        enable_commits=args.enable_commits,
        disable_branches=args.disable_branches,
        git_branch_prefix=args.git_branch_prefix,
        merge_strategy=args.merge_strategy,
        github_owner=args.github_owner,
        github_repo=args.github_repo,
        notes_file=args.notes_file,
        knowledge_file=args.knowledge_file,
        worktree_name=args.worktree_name,
        worktree_base_dir=args.worktree_base_dir,
        cleanup_worktree=args.cleanup_worktree,
        list_worktrees=args.list_worktrees,
        dry_run=args.dry_run,
        completion_signal=args.completion_signal,
        review_prompt=args.review_prompt,
        ci_retry_enabled=args.ci_retry_enabled,
        comment_review_enabled=args.comment_review_enabled,
        extra_agent_flags=[*unknown, *passthrough],
    )
    if cfg.review_prompt == "":
        # Explicit `-r ""` means "use the default reviewer prompt" in the Bash CLI,
        # but only when -r was actually passed; argparse default is also "".
        if "-r" in argv or any(a.startswith("--review-prompt") for a in argv):
            cfg.review_prompt = PROMPT_DEFAULT_REVIEWER

    if cfg.list_worktrees:
        # Listing worktrees needs no prompt, limits, or GitHub configuration.
        return cfg

    if not cfg.prompt:
        log("❌ Error: Prompt is required. Use -p to provide a prompt.")
        log("Run 'continuous-claude --help' for usage information.")
        sys.exit(1)

    if not re.fullmatch(r"claude|codex", cfg.provider):
        fail("--provider must be one of: claude, codex")
    if cfg.review_provider and not re.fullmatch(r"claude|codex", cfg.review_provider):
        fail("--review-provider must be one of: claude, codex")

    if args.max_runs is None and args.max_cost is None and args.max_duration is None:
        log("❌ Error: Either --max-runs, --max-cost, or --max-duration is required.")
        log("Run 'continuous-claude --help' for usage information.")
        sys.exit(1)

    if args.max_runs is not None:
        if not re.fullmatch(r"\d+", args.max_runs):
            fail("--max-runs must be a non-negative integer")
        cfg.max_runs = int(args.max_runs)

    if args.max_cost is not None:
        if not is_positive_number(args.max_cost):
            fail("--max-cost must be a positive number")
        cfg.max_cost = float(args.max_cost)

    if cfg.dry_run and cfg.max_runs is None and cfg.max_cost is not None and args.max_duration is None:
        cfg.max_runs = 1

    for attr, flag in (
        ("codex_input_cost", "--codex-input-cost-per-million"),
        ("codex_output_cost", "--codex-output-cost-per-million"),
    ):
        value = getattr(args, attr)
        if value is not None:
            if not is_positive_number(value):
                fail(f"{flag} must be a positive number")
            setattr(cfg, attr, float(value))
    if args.codex_cached_input_cost is not None:
        if not is_non_negative_number(args.codex_cached_input_cost):
            fail("--codex-cached-input-cost-per-million must be a non-negative number")
        cfg.codex_cached_input_cost = float(args.codex_cached_input_cost)

    uses_codex = cfg.provider == "codex" or (
        bool(cfg.review_prompt) and cfg.effective_review_provider == "codex"
    )
    if uses_codex and cfg.max_cost is not None:
        if cfg.codex_input_cost is None or cfg.codex_output_cost is None:
            fail(
                "Codex CLI does not report USD cost. Use --codex-input-cost-per-million "
                "and --codex-output-cost-per-million with --max-cost."
            )

    if args.max_duration is not None:
        seconds = parse_duration(args.max_duration)
        if seconds is None:
            fail("--max-duration must be a valid duration (e.g., '2h', '30m', '1h30m', '90s')")
        cfg.max_duration = seconds

    if cfg.merge_strategy not in ("squash", "merge", "rebase"):
        fail("--merge-strategy must be one of: squash, merge, rebase")

    cfg.completion_threshold = positive_int(args.completion_threshold, "--completion-threshold")
    if args.stall_threshold is not None:
        cfg.stall_threshold = positive_int(args.stall_threshold, "--stall-threshold")
    if args.max_calls_per_hour is not None:
        cfg.max_calls_per_hour = positive_int(args.max_calls_per_hour, "--max-calls-per-hour")
    cfg.error_threshold = positive_int(args.error_threshold, "--error-threshold")
    cfg.ci_retry_max_attempts = positive_int(args.ci_retry_max, "--ci-retry-max")
    cfg.comment_review_max_attempts = positive_int(args.comment_review_max, "--comment-review-max")
    cfg.command_retry_max_attempts = positive_int(args.command_retry_max, "--command-retry-max")
    if not re.fullmatch(r"\d+", args.command_retry_base_delay or ""):
        fail("--command-retry-base-delay must be a non-negative integer")
    cfg.command_retry_base_delay = int(args.command_retry_base_delay)

    if cfg.enable_commits:
        if not cfg.github_owner or not cfg.github_repo:
            detected = detect_github_repo()
            if detected:
                cfg.github_owner = cfg.github_owner or detected[0]
                cfg.github_repo = cfg.github_repo or detected[1]
        if not cfg.github_owner:
            fail(
                "GitHub owner is required. Use --owner to provide the owner, "
                "or run from a git repository with a GitHub remote."
            )
        if not cfg.github_repo:
            fail(
                "GitHub repo is required. Use --repo to provide the repo, "
                "or run from a git repository with a GitHub remote."
            )

    return cfg


def detect_github_repo() -> tuple[str, str] | None:
    if not in_git_repo():
        return None
    code, out, _ = git("remote", "get-url", "origin")
    if code != 0:
        return None
    remote_url = out.strip()
    match = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+)", remote_url) or re.fullmatch(
        r"git@github\.com:([^/]+)/([^/]+)", remote_url
    )
    if not match:
        return None
    owner, repo = match.group(1), match.group(2)
    repo = repo.removesuffix(".git")
    return (owner, repo) if owner and repo else None


def validate_requirements(cfg: Config) -> None:
    EXE["git"] = shutil.which("git") or "git"

    for provider_name in {cfg.provider} | (
        {cfg.effective_review_provider} if cfg.review_prompt else set()
    ):
        spec = PROVIDERS[provider_name]
        found = shutil.which(spec.command)
        if not found:
            fail(f"{spec.display_name} is not installed: {spec.install_url}")
        EXE[spec.command] = found

    if cfg.enable_commits:
        gh_path = shutil.which("gh")
        if not gh_path:
            fail("GitHub CLI (gh) is not installed: https://cli.github.com")
        EXE["gh"] = gh_path
        if gh("auth", "status")[0] != 0:
            fail("GitHub CLI is not authenticated. Run 'gh auth login' first.")


# ---------------------------------------------------------------------------
# Worktrees
# ---------------------------------------------------------------------------


def list_worktrees_and_exit() -> None:
    if not in_git_repo():
        fail("Not in a git repository")
    print("📋 Active Git Worktrees:")
    print()
    code, out, err = git("worktree", "list")
    if code != 0:
        fail(f"Failed to list worktrees: {err.strip()}")
    print(out.rstrip())
    sys.exit(0)


def worktree_path_for(cfg: Config) -> Path:
    path = Path(cfg.worktree_base_dir) / cfg.worktree_name
    if not path.is_absolute():
        main_repo_dir = git("rev-parse", "--show-toplevel")[1].strip()
        if main_repo_dir:
            path = Path(main_repo_dir) / path
    return path


def setup_worktree(cfg: Config) -> None:
    if not cfg.worktree_name:
        return
    if not in_git_repo():
        fail("Not in a git repository. Worktrees require a git repository.")

    path = worktree_path_for(cfg)
    branch = current_branch()

    if path.is_dir():
        log(f"🌿 Worktree '{cfg.worktree_name}' already exists at: {path}")
        log("📂 Switching to worktree directory...")
        os.chdir(path)
        log(f"📥 Pulling latest changes from {branch}...")
        if not git_ok("pull", "origin", branch):
            log("⚠️  Warning: Failed to pull latest changes (continuing anyway)")
    else:
        log(f"🌿 Creating new worktree '{cfg.worktree_name}' at: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        code, out, err = git("worktree", "add", str(path), branch)
        if code != 0:
            log(out.rstrip())
            log(err.rstrip())
            fail("Failed to create worktree")
        log("📂 Switching to worktree directory...")
        os.chdir(path)

    log(f"✅ Worktree '{cfg.worktree_name}' ready at: {path}")


def cleanup_worktree(cfg: Config) -> None:
    if not cfg.worktree_name or not cfg.cleanup_worktree:
        return
    if not in_git_repo():
        return

    path = worktree_path_for(cfg)
    log("")
    log(f"🗑️  Cleaning up worktree '{cfg.worktree_name}'...")

    common_dir = git("rev-parse", "--git-common-dir")[1].strip()
    if common_dir:
        main_repo = Path(common_dir).parent
        if main_repo.is_dir():
            os.chdir(main_repo)

    if git_ok("worktree", "remove", str(path), "--force"):
        log("✅ Worktree removed successfully")
    else:
        log("⚠️  Warning: Failed to remove worktree (may need manual cleanup)")
        log(f"   You can manually remove it with: git worktree remove {path} --force")


# ---------------------------------------------------------------------------
# GitHub PR helpers
# ---------------------------------------------------------------------------


def extract_pr_number(pr_output: str) -> int | None:
    match = re.search(r"(?:pull/|#)(\d+)", pr_output or "")
    return int(match.group(1)) if match else None


def wait_for_pr_checks(pr_number: int, owner: str, repo: str, display: str) -> bool:
    prev_state: tuple | None = None
    waiting_message_printed = False

    for poll in range(PR_CHECKS_MAX_POLLS):
        no_checks_configured = False
        code, out, err = gh(
            "pr", "checks", str(pr_number), "--repo", f"{owner}/{repo}",
            "--json", "state,bucket",
        )
        combined = out + err
        if code != 0:
            if "no checks" in combined:
                no_checks_configured = True
                checks = []
            else:
                log(f"⚠️  {display} Failed to get PR checks status: {combined.strip()}")
                return False
        else:
            try:
                checks = json.loads(out or "[]")
            except json.JSONDecodeError:
                checks = []

        check_count = len(checks)
        all_completed = not (not no_checks_configured and check_count == 0)
        all_success = True
        pending_count = success_count = failed_count = 0
        for check in checks:
            bucket = check.get("bucket") or "pending"
            if bucket == "pending":
                all_completed = False
                pending_count += 1
            elif bucket == "fail":
                all_success = False
                failed_count += 1
            else:
                success_count += 1

        code, out, err = gh(
            "pr", "view", str(pr_number), "--repo", f"{owner}/{repo}",
            "--json", "reviewDecision,reviewRequests",
        )
        if code != 0:
            log(f"⚠️  {display} Failed to get PR review status: {(out + err).strip()}")
            return False
        try:
            pr_info = json.loads(out)
        except json.JSONDecodeError:
            pr_info = {}
        review_decision = pr_info.get("reviewDecision") or None
        review_requests_count = len(pr_info.get("reviewRequests") or [])

        reviews_pending = review_decision == "REVIEW_REQUIRED" or review_requests_count > 0
        if review_decision:
            review_status = review_decision
        elif review_requests_count > 0:
            review_status = f"{review_requests_count} review(s) requested"
        else:
            review_status = "None"

        state = (check_count, success_count, pending_count, failed_count,
                 review_status, no_checks_configured)
        state_changed = state != prev_state

        if state_changed:
            log("")
            log(f"🔍 {display} Checking PR status (iteration {poll + 1}/{PR_CHECKS_MAX_POLLS})...")
            if no_checks_configured:
                log("   📊 No checks configured")
            else:
                log(f"   📊 Found {check_count} check(s)")
            if check_count > 0:
                log(f"   🟢 {success_count}    🟡 {pending_count}    🔴 {failed_count}")
            log(f"   👁️  Review status: {review_status}")
            prev_state = state

        if check_count == 0 and not no_checks_configured:
            if poll < 18:
                if not waiting_message_printed:
                    sys.stderr.write("⏳ Waiting for checks to start... (will timeout after 3 minutes) ")
                    waiting_message_printed = True
                sys.stderr.write(".")
                sys.stderr.flush()
                time.sleep(PR_CHECKS_POLL_SECONDS)
                continue
            log("")
            log("   ⚠️  No checks found after waiting, proceeding without checks")
            all_completed = True
            all_success = True
        elif waiting_message_printed:
            log("")
            waiting_message_printed = False

        if all_completed and all_success and not reviews_pending:
            if review_decision == "APPROVED" or (
                not review_decision and review_requests_count == 0
            ):
                log(f"✅ {display} All PR checks and reviews passed")
                return True

        if all_completed and all_success and reviews_pending and state_changed:
            log("   ✅ All checks passed, but waiting for review...")

        if all_completed and not all_success:
            log(f"❌ {display} PR checks failed")
            return False

        if review_decision == "CHANGES_REQUESTED":
            log(f"❌ {display} PR has changes requested in review")
            return False

        waiting_items = []
        if not all_completed:
            waiting_items.append("checks to complete")
        if reviews_pending:
            waiting_items.append("code review")
        if waiting_items and state_changed:
            log(f"⏳ Waiting for: {', '.join(waiting_items)}")

        time.sleep(PR_CHECKS_POLL_SECONDS)

    log(f"⏱️  {display} Timeout waiting for PR checks and reviews (30 minutes)")
    return False


def check_pr_comments(pr_number: int, owner: str, repo: str, display: str,
                      since: str | None = None) -> bool:
    def api_count(endpoint: str, filter_since: bool) -> int:
        code, out, _ = gh("api", endpoint)
        if code != 0:
            return 0
        try:
            comments = json.loads(out)
        except json.JSONDecodeError:
            return 0
        if filter_since and since:
            comments = [c for c in comments if c.get("created_at", "") > since]
        return len(comments)

    review_comments = api_count(f"repos/{owner}/{repo}/pulls/{pr_number}/comments", True)
    issue_endpoint = f"repos/{owner}/{repo}/issues/{pr_number}/comments"
    if since:
        issue_endpoint += f"?since={since}"
    issue_comments = api_count(issue_endpoint, False)

    total = review_comments + issue_comments
    if total > 0:
        log(f"💬 {display} Found {total} comment(s) on PR #{pr_number} "
            f"({review_comments} inline, {issue_comments} general)")
        return True
    log(f"✅ {display} No comments found on PR #{pr_number}")
    return False


def get_failed_run_id(pr_number: int, owner: str, repo: str) -> str | None:
    code, out, _ = gh(
        "pr", "view", str(pr_number), "--repo", f"{owner}/{repo}",
        "--json", "headRefOid", "--jq", ".headRefOid",
    )
    head_sha = out.strip()
    if code != 0 or not head_sha:
        return None
    code, out, _ = gh(
        "run", "list", "--repo", f"{owner}/{repo}", "--commit", head_sha,
        "--status", "failure", "--limit", "1",
        "--json", "databaseId", "--jq", ".[0].databaseId",
    )
    run_id = out.strip()
    return run_id if code == 0 and run_id and run_id != "null" else None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class ContinuousLoop:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.error_count = 0
        self.extra_iterations = 0
        self.successful_iterations = 0
        self.total_cost = 0.0
        self.completion_signal_count = 0
        self.iteration = 1
        self.start_time: float | None = None
        self.error_text = ""  # replaces the Bash ERROR_LOG temp file
        self.calls_window = RateWindow()
        self.errors_window = RateWindow()
        self.cost_window = RateWindow()

    # -- rate tracking ------------------------------------------------------

    def rate_limit_window_stats(self) -> str:
        return (
            f"calls {self.calls_window.count()}/hr"
            f", errors {self.errors_window.count()}/hr"
            f", cost ${self.cost_window.total():.3f}/hr"
        )

    def record_agent_call(self, label: str = "agent call") -> None:
        if self.cfg.dry_run:
            return
        self.calls_window.prune()
        if self.cfg.max_calls_per_hour is not None:
            if self.calls_window.count() >= self.cfg.max_calls_per_hour:
                oldest = self.calls_window.oldest()
                if oldest is not None:
                    wait = int(oldest + RATE_LIMIT_WINDOW_SECONDS - time.time())
                    if wait > 0:
                        log(f"⏱ {label} throttled for {format_duration(wait)} "
                            f"(limit {self.cfg.max_calls_per_hour}/hr; "
                            f"{self.rate_limit_window_stats()})")
                        time.sleep(wait)
        self.calls_window.add()

    def record_rate_error(self) -> None:
        self.errors_window.add()

    def record_rate_cost(self, cost: float | None) -> None:
        if cost is not None:
            self.cost_window.add(cost)

    def add_cost(self, cost: float | None, label: str, display: str) -> None:
        if cost is None:
            return
        log(f"💰 {display} {label}: ${cost:.3f}")
        self.total_cost += cost
        self.record_rate_cost(cost)
        log(f"   Running total: ${self.total_cost:.3f}")

    # -- agent execution ----------------------------------------------------

    def run_agent_iteration(self, prompt: str, display: str,
                            provider: str | None = None) -> tuple[int, list[dict], bool]:
        """Run one streaming provider invocation.

        Returns (exit_code, parsed events, saw_invalid_json). Human-readable
        progress is printed to stderr live; stderr of the agent is captured
        into self.error_text.
        """
        cfg = self.cfg
        provider = provider or cfg.provider
        spec = PROVIDERS[provider]
        self.record_agent_call(f"{display} agent call")
        self.error_text = ""

        if cfg.dry_run:
            log(f"🤖 (DRY RUN) Would run {spec.display_name} with prompt: {prompt}")
            if provider == "claude":
                events = [{"type": "result", "is_error": False,
                           "result": "This is a simulated response from Claude Code."}]
            else:
                events = [
                    {"type": "item.completed", "item": {
                        "type": "agent_message",
                        "text": "This is a simulated response from Codex CLI."}},
                    {"type": "turn.completed", "usage": {
                        "input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}},
                ]
            return 0, events, False

        cwd = os.getcwd()
        if provider == "claude":
            cmd = [resolve_exe("claude"), "-p", prompt,
                   *spec.default_flags, *cfg.extra_agent_flags]
        else:
            cmd = [resolve_exe("codex"), "exec", *spec.default_flags,
                   "-C", cwd, *cfg.extra_agent_flags, prompt]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            )
        except OSError as exc:
            self.error_text = f"Failed to start {spec.display_name}: {exc}"
            log(self.error_text)
            return 127, [], False

        stderr_lines: list[str] = []

        def pump_stderr() -> None:
            assert proc.stderr is not None
            for line in proc.stderr:
                stderr_lines.append(line)
                sys.stderr.write(line)
                sys.stderr.flush()

        stderr_thread = threading.Thread(target=pump_stderr, daemon=True)
        stderr_thread.start()

        events: list[dict] = []
        invalid_json = False
        render = render_claude_event if provider == "claude" else render_codex_event
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                invalid_json = True
                continue
            if not isinstance(event, dict):
                invalid_json = True
                continue
            events.append(event)
            texts, tools = render(event, cwd)
            for text_line in texts:
                log(f"   {display} 💬 {text_line}")
            for tool_line in tools:
                log(f"   {display} {tool_line}")

        exit_code = proc.wait()
        stderr_thread.join(timeout=10)

        self.error_text = "".join(stderr_lines)
        if exit_code != 0 and not self.error_text.strip():
            json_error = extract_agent_error_text(provider, events)
            if json_error:
                self.error_text = json_error
                log(json_error)
            else:
                self.error_text = (
                    f"{spec.display_name} exited with code {exit_code} but produced no error output\n"
                    "\n"
                    "This usually means:\n"
                    f"  - {spec.display_name} crashed or failed to start\n"
                    "  - An authentication or permission issue occurred\n"
                    "  - The command arguments are invalid\n"
                    "\n"
                    "Try running this command directly to see the full error:\n"
                    f"  {subprocess.list2cmdline(cmd)}"
                )

        return exit_code, events, invalid_json

    def run_agent_prompt_quiet(self, prompt: str, mode: str = "git") -> tuple[bool, str]:
        """Run the agent non-streaming for auxiliary tasks (e.g. committing)."""
        cfg = self.cfg
        self.record_agent_call("agent prompt")
        if cfg.provider == "claude":
            allowed_tools = "Bash,Read" if mode == "setup" else "Bash(git)"
            cmd = [resolve_exe("claude"), "-p", prompt, "--allowedTools", allowed_tools,
                   "--dangerously-skip-permissions", *cfg.extra_agent_flags]
        elif cfg.provider == "codex":
            cmd = [resolve_exe("codex"), "exec", *CODEX_DEFAULT_FLAGS,
                   "-C", os.getcwd(), *cfg.extra_agent_flags, prompt]
        else:
            return False, f"Unsupported provider: {cfg.provider}"
        code, out, err = run_cmd(cmd)
        return code == 0, (out + err).strip()

    def run_with_command_retry(self, label: str, fn) -> tuple[bool, str]:
        """Retry a transient command with exponential backoff.

        fn is a zero-argument callable returning (ok, output).
        """
        delay = self.cfg.command_retry_base_delay
        output = ""
        for attempt in range(1, self.cfg.command_retry_max_attempts + 1):
            ok, output = fn()
            if ok:
                return True, output
            if attempt >= self.cfg.command_retry_max_attempts:
                break
            log(f"⚠️  {label} failed (attempt {attempt}/{self.cfg.command_retry_max_attempts}): {output}")
            log(f"⏳ Retrying {label} in {delay}s...")
            time.sleep(delay)
            delay *= 2
        return False, output

    # -- error / stall handling ---------------------------------------------

    def get_recent_failure_details(self, fallback: str = "") -> str:
        if self.error_text.strip():
            return tail_lines(self.error_text)
        if fallback:
            return tail_lines(fallback)
        return "No diagnostics captured."

    def maybe_sleep_for_rate_limit(self, display: str, error_type: str,
                                   details: str = "") -> bool:
        error_details = self.get_recent_failure_details(details)
        wait = detect_rate_limit_wait_seconds(error_details)
        if wait is None:
            return False
        log(f"⏱ {display} Rate limit detected in {error_type}; throttled for "
            f"{format_duration(wait)} ({self.rate_limit_window_stats()})")
        time.sleep(wait)
        self.error_count = 0
        return True

    def append_stall_summary(self, display: str, reason: str, details: str = "") -> None:
        notes_path = Path(self.cfg.notes_file)
        if notes_path.parent != Path("."):
            notes_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = "\n".join(
            "    " + line for line in self.get_recent_failure_details(details).splitlines()
        )
        with notes_path.open("a", encoding="utf-8") as notes:
            notes.write(
                "\n"
                f"## Health pause - {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
                "\n"
                f"- Iteration: {display}\n"
                f"- Consecutive failures: {self.error_count}\n"
                f"- Reason: {reason}\n"
                "\n"
                "Recent diagnostics:\n"
                f"{diagnostics}\n"
                "\n"
                "Next step: Inspect the failure, fix the project or adjust the prompt, "
                "then rerun Continuous Claude.\n"
            )

    def maybe_handle_stall_threshold(self, display: str, reason: str,
                                     details: str = "") -> None:
        cfg = self.cfg
        if cfg.stall_threshold is not None and self.error_count >= cfg.stall_threshold:
            self.append_stall_summary(display, reason, details)
            log(f"⏸️  {display} Health stall threshold reached "
                f"({self.error_count}/{cfg.stall_threshold} consecutive failures)")
            log(f"📝 {display} Wrote stall diagnostics to {cfg.notes_file}")
            if sys.stdin.isatty():
                log("Press Enter after human intervention to continue, or Ctrl+C to exit.")
                try:
                    input()
                except EOFError:
                    sys.exit(1)
                self.error_count = 0
                return
            log(f"❌ {display} Non-interactive shell detected; exiting so a human can intervene.")
            sys.exit(1)

        if cfg.stall_threshold is None and self.error_count >= cfg.error_threshold:
            log(f"❌ Fatal: {cfg.error_threshold} consecutive errors occurred. Exiting.")
            sys.exit(1)

    def handle_iteration_error(self, display: str, error_type: str,
                               error_output: str) -> None:
        self.error_count += 1
        self.extra_iterations += 1
        self.record_rate_error()

        log("")
        if error_type == "exit_code":
            log(f"❌ {display} Error occurred ({self.error_count} consecutive errors):")
            log("")
            if self.error_text.strip():
                log("Error details:")
                log(self.error_text.rstrip())
            else:
                log("No error details captured")
        elif error_type == "invalid_json":
            log(f"❌ {display} Error: Invalid JSON response ({self.error_count} consecutive errors):")
            log("")
            log(error_output)
        elif error_type in ("claude_error", "codex_error", "codex_incomplete"):
            provider_label = "Claude Code" if error_type == "claude_error" else "Codex CLI"
            if error_type == "codex_incomplete":
                log(f"❌ {display} Error: Codex CLI response did not include a completed "
                    f"turn ({self.error_count} consecutive errors)")
            else:
                log(f"❌ {display} Error in {provider_label} response "
                    f"({self.error_count} consecutive errors):")
            log("")
            log(error_output)
        log("")

        if self.maybe_sleep_for_rate_limit(display, error_type, error_output):
            return
        self.maybe_handle_stall_threshold(display, error_type, error_output)

    # -- git / PR workflow ---------------------------------------------------

    def create_iteration_branch(self, display: str, iteration_num: int) -> str | None:
        """Returns the branch name, "" when not in a git repo, None on failure."""
        cfg = self.cfg
        if not in_git_repo():
            return ""

        branch = current_branch()
        if branch.startswith(cfg.git_branch_prefix):
            log(f"⚠️  {display} Already on iteration branch: {branch}")
            if not git_ok("checkout", "main"):
                return None

        date_str = time.strftime("%Y-%m-%d")
        branch_name = f"{cfg.git_branch_prefix}iteration-{iteration_num}/{date_str}-{secrets.token_hex(4)}"
        log(f"🌿 {display} Creating branch: {branch_name}")

        if cfg.dry_run:
            log(f"   (DRY RUN) Would create branch {branch_name}")
            return branch_name

        if not git_ok("checkout", "-b", branch_name):
            log(f"⚠️  {display} Failed to create branch")
            return None
        return branch_name

    def cleanup_branch(self, branch_name: str, main_branch: str) -> None:
        if branch_name and in_git_repo():
            git("checkout", main_branch)
            git("branch", "-D", branch_name)

    def commit_via_agent(self, display: str) -> bool:
        ok, _ = self.run_with_command_retry(
            f"{display} commit command",
            lambda: self.run_agent_prompt_quiet(PROMPT_COMMIT_MESSAGE),
        )
        return ok

    def continuous_claude_commit(self, display: str, branch_name: str,
                                 main_branch: str) -> bool:
        cfg = self.cfg
        if not in_git_repo():
            return True

        uncommitted = has_uncommitted_changes()

        # The agent is instructed not to commit, but if it does anyway the branch
        # can be ahead of main while the worktree is clean. Treat those commits as
        # changes so we still push the branch and create the PR.
        code, out, _ = git("rev-list", "--count", f"{main_branch}..{branch_name}")
        commits_ahead = int(out.strip()) if code == 0 and out.strip().isdigit() else 0

        if not uncommitted and commits_ahead == 0:
            log(f"🫙 {display} No changes detected, cleaning up branch...")
            git("checkout", main_branch)
            git("branch", "-D", branch_name)
            return True

        if cfg.dry_run:
            log(f"💬 {display} (DRY RUN) Would commit changes...")
            log(f"📦 {display} (DRY RUN) Changes committed on branch: {branch_name}")
            log(f"📤 {display} (DRY RUN) Would push branch...")
            log(f"🔨 {display} (DRY RUN) Would create pull request...")
            log(f"✅ {display} (DRY RUN) PR merged: <commit title would appear here>")
            return True

        if uncommitted:
            log(f"💬 {display} Committing changes...")
            if not self.commit_via_agent(display):
                log(f"⚠️  {display} Failed to commit changes")
                git("checkout", main_branch)
                return False
            if has_uncommitted_changes():
                log(f"⚠️  {display} Commit command ran but changes still present "
                    "(uncommitted or untracked files remain)")
                git("checkout", main_branch)
                return False
            log(f"📦 {display} Changes committed on branch: {branch_name}")
        else:
            log(f"📦 {display} Changes already committed on branch: {branch_name} "
                f"({commits_ahead} commit(s) ahead)")

        commit_message = git("log", "-1", "--format=%B", branch_name)[1]
        message_lines = commit_message.splitlines()
        commit_title = message_lines[0] if message_lines else ""
        commit_body = "\n".join(message_lines[3:])

        log(f"📤 {display} Pushing branch...")
        ok, push_output = self.run_with_command_retry(
            f"{display} push branch",
            lambda: _combined(git("push", "-u", "origin", branch_name)),
        )
        if not ok:
            log(f"⚠️  {display} Failed to push branch: {push_output}")
            git("checkout", main_branch)
            return False

        log(f"🔨 {display} Creating pull request...")
        ok, pr_output = self.run_with_command_retry(
            f"{display} create PR",
            lambda: _combined(gh(
                "pr", "create", "--repo", f"{cfg.github_owner}/{cfg.github_repo}",
                "--title", commit_title, "--body", commit_body, "--base", main_branch,
            )),
        )
        if not ok:
            log(f"⚠️  {display} Failed to create PR: {pr_output}")
            git("checkout", main_branch)
            return False

        pr_number = extract_pr_number(pr_output)
        if pr_number is None:
            log(f"⚠️  {display} Failed to extract PR number from: {pr_output}")
            git("checkout", main_branch)
            return False

        owner, repo = cfg.github_owner, cfg.github_repo

        def close_pr_and_cleanup(reason: str) -> None:
            log(reason)
            gh("pr", "close", str(pr_number), "--repo", f"{owner}/{repo}", "--delete-branch")
            log(f"🗑️  {display} Cleaning up local branch: {branch_name}")
            self.cleanup_branch(branch_name, main_branch)

        log(f"🔍 {display} PR #{pr_number} created, waiting 5 seconds for GitHub to set up...")
        time.sleep(5)
        if not wait_for_pr_checks(pr_number, owner, repo, display):
            if cfg.ci_retry_enabled:
                log(f"🔧 {display} CI checks failed, attempting automatic fix...")
                if self.attempt_ci_fix_and_recheck(pr_number, branch_name, display):
                    log(f"🎉 {display} CI fix successful!")
                else:
                    close_pr_and_cleanup(
                        f"⚠️  {display} CI fix unsuccessful, closing PR and deleting remote branch..."
                    )
                    return False
            else:
                close_pr_and_cleanup(
                    f"⚠️  {display} PR checks failed or timed out, closing PR and "
                    "deleting remote branch..."
                )
                return False

        if cfg.comment_review_enabled:
            if check_pr_comments(pr_number, owner, repo, display):
                log(f"💬 {display} PR has review comments, attempting to address them...")
                if not self.attempt_comment_fix_and_recheck(pr_number, branch_name, display):
                    close_pr_and_cleanup(f"⚠️  {display} Failed to address PR comments, closing PR...")
                    return False

        if not self.merge_pr_and_cleanup(pr_number, branch_name, display, main_branch):
            code, out, _ = gh("pr", "view", str(pr_number), "--repo", f"{owner}/{repo}",
                              "--json", "state", "--jq", ".state")
            pr_state = out.strip() if code == 0 else "UNKNOWN"
            if pr_state == "OPEN":
                close_pr_and_cleanup(
                    f"⚠️  {display} Failed to merge PR, closing it and deleting remote branch..."
                )
            else:
                log(f"⚠️  {display} PR was merged but cleanup failed")
                log(f"🗑️  {display} Cleaning up local branch: {branch_name}")
                self.cleanup_branch(branch_name, main_branch)
            return False

        log(f"✅ {display} PR #{pr_number} merged: {commit_title}")
        if not git_ok("checkout", main_branch):
            log(f"⚠️  {display} Failed to checkout {main_branch}")
            return False
        return True

    def merge_pr_and_cleanup(self, pr_number: int, branch_name: str,
                             display: str, main_branch: str) -> bool:
        cfg = self.cfg
        owner, repo = cfg.github_owner, cfg.github_repo

        log(f"🔄 {display} Updating branch with latest from main...")
        code, out, err = gh("pr", "update-branch", str(pr_number), "--repo", f"{owner}/{repo}")
        update_output = (out + err).strip()
        if code == 0:
            log(f"📥 {display} Branch updated, re-checking PR status...")
            if not wait_for_pr_checks(pr_number, owner, repo, display):
                log(f"❌ {display} PR checks failed after branch update")
                return False
        elif re.search(r"already up-to-date|is up to date", update_output, re.IGNORECASE):
            log(f"✅ {display} Branch already up-to-date")
        else:
            log(f"⚠️  {display} Branch update failed: {update_output}")
            return False

        merge_flag = {"squash": "--squash", "merge": "--merge", "rebase": "--rebase"}[cfg.merge_strategy]
        log(f"🔀 {display} Merging PR #{pr_number} with strategy: {cfg.merge_strategy}...")
        code, out, err = gh("pr", "merge", str(pr_number), "--repo", f"{owner}/{repo}", merge_flag)
        merge_output = (out + err).strip()
        if code != 0:
            log(f"⚠️  {display} Failed to merge PR: {merge_output}")
            if re.search(
                r"upgrade to github pro|make this repository public|http 403"
                r"|status code 403|resource not accessible",
                merge_output, re.IGNORECASE,
            ):
                log("   GitHub reported an API or plan restriction. This is not a merge "
                    "queue failure; check repository visibility, branch protection/ruleset "
                    "availability, and your GitHub plan.")
            return False

        log(f"📥 {display} Pulling latest from main...")
        if not git_ok("checkout", main_branch):
            log(f"⚠️  {display} Failed to checkout {main_branch}")
            return False
        if not git_ok("pull", "origin", main_branch):
            log(f"⚠️  {display} Failed to pull from {main_branch}")
            return False

        log(f"🗑️  {display} Deleting local branch: {branch_name}")
        git("branch", "-d", branch_name)
        return True

    def commit_on_current_branch(self, display: str) -> bool:
        if not in_git_repo():
            return True
        if not has_uncommitted_changes():
            log(f"ℹ️  {display} No changes to commit")
            return True
        if self.cfg.dry_run:
            log(f"💬 {display} (DRY RUN) Would commit changes on current branch...")
            return True

        log(f"💬 {display} Committing changes on current branch...")
        if not self.commit_via_agent(display):
            log(f"⚠️  {display} Failed to commit changes")
            return False
        if has_uncommitted_changes():
            log(f"⚠️  {display} Commit command ran but changes still present")
            return False
        commit_title = git("log", "-1", "--format=%s")[1].strip()
        log(f"✅ {display} Committed: {commit_title}")
        return True

    # -- reviewer / CI-fix / comment-fix passes -------------------------------

    def _run_auxiliary_agent_pass(self, display: str, prompt: str, cost_label: str,
                                  provider: str | None = None) -> bool:
        provider = provider or self.cfg.provider
        exit_code, events, invalid_json = self.run_agent_iteration(prompt, display, provider)
        if exit_code != 0:
            log(f"❌ {display} {cost_label} failed with exit code: {exit_code}")
            return False
        parse_result = parse_agent_result(provider, events, invalid_json)
        if parse_result != "success":
            log(f"❌ {display} {cost_label} returned error: {parse_result}")
            return False
        self.add_cost(extract_agent_cost(provider, events, self.cfg),
                      f"{cost_label} cost", display)
        usage = extract_agent_usage_summary(provider, events)
        if usage:
            log(f"   {usage}")
        return True

    def run_reviewer_iteration(self, display: str) -> bool:
        provider = self.cfg.effective_review_provider
        log(f"🔍 {display} Running reviewer pass with {PROVIDERS[provider].display_name}...")
        full_prompt = (
            f"{PROMPT_REVIEWER_CONTEXT}\n\n## USER REVIEW INSTRUCTIONS\n\n{self.cfg.review_prompt}"
        )
        if not self._run_auxiliary_agent_pass(display, full_prompt, "Reviewer", provider):
            return False
        log(f"✅ {display} Reviewer pass completed")
        return True

    def run_ci_fix_iteration(self, display: str, pr_number: int, branch_name: str,
                             retry_attempt: int) -> bool:
        cfg = self.cfg
        log(f"🔧 {display} Attempting to fix CI failure "
            f"(attempt {retry_attempt}/{cfg.ci_retry_max_attempts})...")
        failed_run_id = get_failed_run_id(pr_number, cfg.github_owner, cfg.github_repo)

        prompt = (
            f"{PROMPT_CI_FIX_CONTEXT}\n\n## CURRENT CONTEXT\n\n"
            f"- Repository: {cfg.github_owner}/{cfg.github_repo}\n"
            f"- PR Number: #{pr_number}\n"
            f"- Branch: {branch_name}"
        )
        if failed_run_id:
            prompt += (
                f"\n- Failed Run ID: {failed_run_id} "
                f"(use this with `gh run view {failed_run_id} --log-failed`)"
            )
        prompt += (
            "\n\n## INSTRUCTIONS\n\n"
            "1. Start by running `gh run list --status failure --limit 3` to see recent failures\n"
            "2. Then use `gh run view <RUN_ID> --log-failed` to see the error details\n"
            "3. Analyze what went wrong and fix it\n"
            "4. After making changes, stage, commit, AND PUSH them with a clear commit "
            "message describing the fix\n"
            "5. You MUST push the changes to trigger a new CI run"
        )

        if not self._run_auxiliary_agent_pass(display, prompt, "CI fix"):
            return False
        log(f"✅ {display} CI fix iteration completed, checking CI status...")
        return True

    def attempt_ci_fix_and_recheck(self, pr_number: int, branch_name: str,
                                   display: str) -> bool:
        cfg = self.cfg
        for attempt in range(1, cfg.ci_retry_max_attempts + 1):
            if not self.run_ci_fix_iteration(display, pr_number, branch_name, attempt):
                log(f"⚠️  {display} CI fix attempt {attempt} failed")
                continue
            time.sleep(5)
            log(f"🔍 {display} Waiting for CI checks after fix...")
            if wait_for_pr_checks(pr_number, cfg.github_owner, cfg.github_repo, display):
                log(f"✅ {display} CI checks passed after fix!")
                return True
            log(f"⚠️  {display} CI still failing after fix attempt {attempt}")
        log(f"❌ {display} All CI fix attempts exhausted")
        return False

    def run_comment_fix_iteration(self, display: str, pr_number: int,
                                  branch_name: str, retry_attempt: int) -> bool:
        cfg = self.cfg
        log(f"💬 {display} Attempting to address PR comments "
            f"(attempt {retry_attempt}/{cfg.comment_review_max_attempts})...")
        prompt = (
            f"{PROMPT_COMMENT_REVIEW_CONTEXT}\n\n## CURRENT CONTEXT\n\n"
            f"- Repository: {cfg.github_owner}/{cfg.github_repo}\n"
            f"- PR Number: #{pr_number}\n"
            f"- Branch: {branch_name}\n\n"
            "## INSTRUCTIONS\n\n"
            "1. Start by reading inline review comments: "
            f"`gh api repos/{cfg.github_owner}/{cfg.github_repo}/pulls/{pr_number}/comments`\n"
            "2. Also read PR-level comments: "
            f"`gh api repos/{cfg.github_owner}/{cfg.github_repo}/issues/{pr_number}/comments`\n"
            "3. Analyze each comment and determine what code changes are needed\n"
            "4. Make the necessary changes to address the feedback\n"
            "5. After making changes, stage, commit, AND PUSH them with a clear commit "
            "message describing what comments you addressed\n"
            "6. You MUST push the changes to update the PR"
        )
        if not self._run_auxiliary_agent_pass(display, prompt, "Comment review"):
            return False
        log(f"✅ {display} Comment review iteration completed")
        return True

    def attempt_comment_fix_and_recheck(self, pr_number: int, branch_name: str,
                                        display: str) -> bool:
        cfg = self.cfg
        for attempt in range(1, cfg.comment_review_max_attempts + 1):
            if not self.run_comment_fix_iteration(display, pr_number, branch_name, attempt):
                log(f"⚠️  {display} Comment review attempt {attempt} failed, proceeding to merge")
                return True
            time.sleep(5)
            log(f"🔍 {display} Waiting for CI checks after comment fixes...")
            if wait_for_pr_checks(pr_number, cfg.github_owner, cfg.github_repo, display):
                log(f"✅ {display} CI still green after addressing comments!")
                return True
            log(f"⚠️  {display} CI failed after comment review attempt {attempt}")
        log(f"❌ {display} CI broken after addressing comments")
        return False

    # -- prompt assembly ------------------------------------------------------

    def build_enhanced_prompt(self) -> str:
        cfg = self.cfg
        context = PROMPT_WORKFLOW_CONTEXT.replace(
            "COMPLETION_SIGNAL_PLACEHOLDER", cfg.completion_signal
        )
        prompt = f"{context}\n\n{cfg.prompt}\n\n"

        notes_path = Path(cfg.notes_file)
        notes_exist = notes_path.is_file()
        if notes_exist:
            notes_content = notes_path.read_text(encoding="utf-8", errors="replace")
            prompt += (
                "## CONTEXT FROM PREVIOUS ITERATION\n\n"
                f"The following is from {cfg.notes_file}, maintained by previous "
                f"iterations to provide context:\n\n{notes_content}\n\n"
            )

        knowledge_path = Path(cfg.knowledge_file) if cfg.knowledge_file else None
        if knowledge_path and knowledge_path.is_file():
            knowledge_content = knowledge_path.read_text(encoding="utf-8", errors="replace")
            prompt += (
                "## DURABLE PROJECT KNOWLEDGE\n\n"
                f"The following is from {cfg.knowledge_file}, maintained across "
                f"iterations as long-lived project knowledge:\n\n{knowledge_content}\n\n"
            )

        prompt += "## ITERATION NOTES\n\n"
        notes_template = PROMPT_NOTES_UPDATE_EXISTING if notes_exist else PROMPT_NOTES_CREATE_NEW
        prompt += notes_template.format(notes_file=cfg.notes_file)
        prompt += PROMPT_NOTES_GUIDELINES

        if cfg.knowledge_file:
            prompt += "\n\n## DURABLE KNOWLEDGE RECORDING\n\n"
            knowledge_template = (
                PROMPT_KNOWLEDGE_UPDATE_EXISTING
                if knowledge_path and knowledge_path.is_file()
                else PROMPT_KNOWLEDGE_CREATE_NEW
            )
            prompt += knowledge_template.format(knowledge_file=cfg.knowledge_file)
            prompt += PROMPT_KNOWLEDGE_GUIDELINES

        return prompt

    # -- iteration orchestration ----------------------------------------------

    def get_iteration_display(self, iteration_num: int) -> str:
        max_runs = self.cfg.max_runs or 0
        if max_runs == 0:
            return f"({iteration_num})"
        return f"({iteration_num}/{max_runs + self.extra_iterations})"

    def handle_iteration_success(self, display: str, events: list[dict],
                                 branch_name: str, main_branch: str) -> bool:
        cfg = self.cfg
        result_text = extract_agent_result_text(cfg.provider, events)
        explicit_completion = bool(result_text) and cfg.completion_signal in result_text

        cost = extract_agent_cost(cfg.provider, events, cfg)
        if cost is not None:
            log("")
        self.add_cost(cost, "Iteration cost", display)
        usage = extract_agent_usage_summary(cfg.provider, events)
        if usage:
            log(f"   {usage}")

        log(f"✅ {display} Work completed")
        if cfg.enable_commits:
            if cfg.disable_branches:
                committed = self.commit_on_current_branch(display)
                failure_reason = "Commit failed"
            else:
                committed = self.continuous_claude_commit(display, branch_name, main_branch)
                failure_reason = "PR workflow failed"
            if not committed:
                self.error_count += 1
                self.extra_iterations += 1
                self.record_rate_error()
                log(f"❌ {display} {failure_reason} ({self.error_count} consecutive errors)")
                if self.maybe_sleep_for_rate_limit(display, failure_reason.lower()):
                    return False
                self.maybe_handle_stall_threshold(display, failure_reason.lower())
                return False
        else:
            log(f"⏭️  {display} Skipping commits (--disable-commits flag set)")
            self.cleanup_branch(branch_name, main_branch)

        if explicit_completion:
            self.completion_signal_count += 1
            log("")
            log(f"🎯 {display} Completion signal detected "
                f"({self.completion_signal_count}/{cfg.completion_threshold})")
        elif detect_positive_completion_heuristic(result_text) and not repo_has_pending_changes():
            self.completion_signal_count += 1
            log("")
            log(f"🩺 {display} Positive completion heuristic detected "
                f"({self.completion_signal_count}/{cfg.completion_threshold})")
        else:
            if self.completion_signal_count > 0:
                log("")
                log(f"🔄 {display} Completion signal not found, resetting counter")
            self.completion_signal_count = 0

        self.error_count = 0
        if self.extra_iterations > 0:
            self.extra_iterations -= 1
        self.successful_iterations += 1
        return True

    def execute_single_iteration(self, iteration_num: int) -> bool:
        cfg = self.cfg
        display = self.get_iteration_display(iteration_num)
        log(f"🔄 {display} Starting iteration...")

        main_branch = current_branch()
        branch_name = ""
        if cfg.enable_commits and not cfg.disable_branches:
            branch_name = self.create_iteration_branch(display, iteration_num)
            if branch_name is None:
                if in_git_repo():
                    log(f"❌ {display} Failed to create branch")
                    self.handle_iteration_error(display, "exit_code", "")
                    return False
                branch_name = ""

        enhanced_prompt = self.build_enhanced_prompt()
        log(f"🤖 {display} Running {PROVIDERS[cfg.provider].display_name}...")

        exit_code, events, invalid_json = self.run_agent_iteration(enhanced_prompt, display)
        if exit_code != 0:
            log("")
            log(f"⚠️  {PROVIDERS[cfg.provider].display_name} command failed with "
                f"exit code: {exit_code}")
            self.cleanup_branch(branch_name, main_branch)
            self.handle_iteration_error(display, "exit_code", "")
            return False

        parse_result = parse_agent_result(cfg.provider, events, invalid_json)
        if parse_result != "success":
            self.cleanup_branch(branch_name, main_branch)
            error_output = extract_agent_error_text(cfg.provider, events) or json.dumps(
                events[-1] if events else {}, ensure_ascii=False
            )
            self.handle_iteration_error(display, parse_result, error_output)
            return False

        if cfg.review_prompt:
            if not self.run_reviewer_iteration(display):
                log(f"❌ {display} Reviewer failed, aborting iteration")
                self.cleanup_branch(branch_name, main_branch)
                self.error_count += 1
                self.extra_iterations += 1
                self.record_rate_error()
                if self.maybe_sleep_for_rate_limit(display, "reviewer failed"):
                    return False
                self.maybe_handle_stall_threshold(display, "reviewer failed")
                return False

        return self.handle_iteration_success(display, events, branch_name, main_branch)

    def should_continue(self) -> bool:
        cfg = self.cfg
        should = (
            cfg.max_runs is None
            or cfg.max_runs == 0
            or self.successful_iterations < cfg.max_runs
        )

        if cfg.max_cost is not None and self.total_cost >= cfg.max_cost:
            should = False

        if cfg.max_duration is not None and self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= cfg.max_duration:
                log("")
                log(f"⏱️  Maximum duration reached ({format_duration(elapsed)})")
                should = False

        if cfg.max_runs and self.successful_iterations >= cfg.max_runs:
            should = False

        if self.completion_signal_count >= cfg.completion_threshold:
            log("")
            log(f"🎉 Project completion signal detected {self.completion_signal_count} "
                "times consecutively!")
            should = False

        return should

    def main_loop(self) -> None:
        if self.cfg.max_duration is not None:
            self.start_time = time.time()
        while self.should_continue():
            self.execute_single_iteration(self.iteration)
            time.sleep(1)
            self.iteration += 1

    def show_completion_summary(self) -> None:
        cfg = self.cfg
        elapsed_msg = ""
        if self.start_time is not None:
            elapsed_msg = f" (elapsed: {format_duration(time.time() - self.start_time)})"

        if self.completion_signal_count >= cfg.completion_threshold:
            if self.total_cost > 0:
                print(f"✨ Project completed! Detected completion signal "
                      f"{self.completion_signal_count} times in a row. "
                      f"Total cost: ${self.total_cost:.3f}{elapsed_msg}")
            else:
                print(f"✨ Project completed! Detected completion signal "
                      f"{self.completion_signal_count} times in a row.{elapsed_msg}")
        elif cfg.max_runs or cfg.max_cost is not None or cfg.max_duration is not None:
            if self.total_cost > 0:
                print(f"🎉 Done with total cost: ${self.total_cost:.3f}{elapsed_msg}")
            else:
                print(f"🎉 Done{elapsed_msg}")


def _combined(result: tuple[int, str, str]) -> tuple[bool, str]:
    code, out, err = result
    return code == 0, (out + err).strip()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    utf8_console()
    argv = list(sys.argv[1:] if argv is None else argv)

    if argv and argv[0] == "update":
        log("ℹ️  Self-update is not supported in the Python port; "
            "pull the latest version from the source repository instead.")
        return 0

    cfg = parse_config(argv)
    if cfg.list_worktrees:
        EXE["git"] = shutil.which("git") or "git"
        list_worktrees_and_exit()

    validate_requirements(cfg)

    setup_worktree(cfg)

    loop = ContinuousLoop(cfg)
    try:
        loop.main_loop()
        loop.show_completion_summary()
    except KeyboardInterrupt:
        log("")
        log("🛑 Interrupted.")
        return 130
    finally:
        cleanup_worktree(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
