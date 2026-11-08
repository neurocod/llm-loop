"""Subagent-shaped commands; ordinary output excludes provider transcripts."""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
from pathlib import Path
import sqlite3
import sys
import time

from .. import host_policy
from ..answer_contract import (
    checked_expect, coverage_expect, followed_up_note, incomplete_note)

from .protocol import invocation_root
from .state import Store
from .worker import entered, entry_point, launch, work

# The model a spawned Claude runs on when `--model` names none (see the flag).
DEFAULT_MODEL = "opus"

# User-requested progress reminder, not a task deadline: long read-only reviews
# may leave the working tree unchanged. The waiter and worker keep running.
WAIT_REMINDER_SECONDS = 30 * 60

# Recovery cadence, not a task timeout: retry once per minute so an abandoned
# running turn cannot strand an unlimited waiter. The worker's OS lock keeps
# retries from starting a second provider while the original worker is alive.
WAIT_RECOVERY_SECONDS = 60


def read_message(args) -> str:
    if args.message_file:
        message = Path(args.message_file).read_text(encoding="utf-8-sig")
    elif args.message == "-":
        message = sys.stdin.read()
    else:
        message = args.message or ""
    if not message.strip():
        raise ValueError("Message is empty; use --message or --message-file")
    return message


def expected_answer(profile: str, explicit: str | None, prompt: str) -> str:
    """Resolve `--expect` at the CLI boundary, per turn, from the persisted profile.

    Mirrors `llm_agents.codex.roles.expected_answer`: an explicit flag wins, the
    empty string included; an omitted one takes the review contract's
    `coverage_expect` of THIS turn's message. `configured` is a free-form
    profile with no answer contract, so it expects nothing by default.
    """
    return checked_expect(explicit if explicit is not None else (
        coverage_expect(prompt) if profile == "review" else ""))


def seconds(value: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("Expected finite non-negative seconds")
    return value


class _ExactParser(argparse.ArgumentParser):
    """A parser, and subparsers (built with the parent's class), that refuse
    abbreviated long options: `spawn w --turn 5` was read as `--turn-timeout
    5`, `--turn` being a flag of `wait`/`result`/`logs` only."""

    def __init__(self, *args, allow_abbrev: bool = False, **kwargs) -> None:
        super().__init__(*args, allow_abbrev=allow_abbrev, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = _ExactParser(description=__doc__, epilog=(
        "spawn returns immediately. send queues a followup in the same Claude "
        "session. wait returns only the final answer; an expired wait leaves "
        "the worker running and exits successfully with pending status. "
        "Use wait --until-done in one background command/session for a final result. "
        "Use --json on commands for structured output. "
        "Exit codes: 0 successful command/completed turn (including expired "
        "wait), 1 failed/interrupted turn or command error, 2 result still pending. "
        "Requires Claude Code 2.1.259 or newer (permission-prompts)."))
    parser.add_argument("--state-dir", type=Path,
                        help="Registry/log root (default CLAUDE_AGENTS_DIR or checkout/.claude-agents)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, aliases in [("spawn", ["spawn_agent"]),
                          ("send", ["send_message", "followup_task"])]:
        cmd = sub.add_parser(name, aliases=aliases)
        cmd.set_defaults(action=name)
        cmd.add_argument("name")
        message = cmd.add_mutually_exclusive_group(required=True)
        message.add_argument("--message", "--prompt", help="Task text, or '-' for stdin")
        message.add_argument("--message-file", "--prompt-file", type=Path)
        if name == "spawn":
            cmd.add_argument("--cwd", type=Path)
            # opus unless named (operator's choice, 2026-11-03): left to the
            # CLI, 49 of 69 reviews ran on sonnet (the init events kept in
            # .claude-agents/, counted from 2026-10-02 on).
            cmd.add_argument("--model", default=DEFAULT_MODEL,
                             help=f"Default {DEFAULT_MODEL}; name another to override")
            cmd.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
            cmd.add_argument("--profile", choices=["review", "configured"], default="review",
                             help="review: read/git tools; configured: existing permissions/hooks, can edit cwd")
            cmd.add_argument("--turn-timeout", type=seconds,
                             help="Optional caller deadline per turn; default unlimited, 0 disables")
        cmd.add_argument("--expect", default=None, metavar="REGEX",
                         help="The answer must match this regex (multiline); one that does not "
                              "is followed up once in the same session. Default for the review "
                              "profile: ## Coverage with a line per numbered question of the "
                              "message; none for configured. --expect= (empty) disables it")
        cmd.add_argument("--wait", type=seconds, metavar="SECONDS",
                         help="Wait briefly for this submitted turn before returning")
        cmd.add_argument("--json", action="store_true")
    for name, aliases in [("wait", ["wait_agent"]), ("result", []),
                          ("interrupt", ["interrupt_agent"])]:
        cmd = sub.add_parser(name, aliases=aliases)
        cmd.set_defaults(action=name)
        cmd.add_argument("name")
        if name != "interrupt":
            cmd.add_argument("--turn", type=int, help="Turn returned by spawn/send (default latest)")
        if name == "wait":
            waiting = cmd.add_mutually_exclusive_group()
            waiting.add_argument("--timeout", type=seconds, default=90,
                                 help="Wait window in seconds, not a task deadline (default 90)")
            waiting.add_argument("--until-done", dest="timeout", action="store_const", const=None,
                                 help="Wait without a deadline; remind on stderr every 30 minutes to check progress")
        cmd.add_argument("--json", action="store_true")
    cmd = sub.add_parser("status", aliases=["list_agents"],
                         help="List status/activity; recover abandoned workers with pending work")
    cmd.set_defaults(action="status")
    cmd.add_argument("name", nargs="?")
    cmd.add_argument("--json", action="store_true")
    cmd = sub.add_parser("logs")
    cmd.set_defaults(action="logs")
    cmd.add_argument("name")
    cmd.add_argument("--turn", type=int)
    cmd.add_argument("--json", action="store_true")
    cmd = sub.add_parser("_worker")
    cmd.set_defaults(action="_worker")
    cmd.add_argument("name")
    return parser


def brief(agent: dict, store: Store) -> dict:
    result = {key: agent.get(key) for key in
              ("name", "status", "turn", "completed_turn", "session_id", "cwd", "error", "updated", "context")}
    # Stream mtime is cheap and includes thinking/retry activity without reading
    # or exposing those messages. Queued followups may not have a log yet.
    logs = (store.root / agent["name"]).glob("turn-*/events.jsonl")
    last_event = max((path.stat().st_mtime for path in logs), default=None)
    result["last_activity_s"] = round(max(0, time.time() - last_event), 1) if last_event else None
    return result


def turn_result(store: Store, name: str, turn: int | None = None) -> dict:
    result = store.result(name, turn)
    result.pop("prompt", None)
    defaults = {"answer": "", "error": "", "duration_s": None,
                "usage": {}, "total_cost_usd": None, "permission_denials": [],
                "context": store.get(name)["context"] if result["status"] in {"queued", "running"} else None,
                "session_id": store.get(name)["session_id"],
                "log_dir": str(store.root / result["name"] / f"turn-{result['turn']:04d}")}
    return defaults | result


def wait_result(store: Store, name: str, turn: int | None, timeout: float | None) -> dict:
    result = turn_result(store, name, turn)
    launch(store, name)
    started = time.monotonic()
    deadline = None if timeout is None else started + timeout
    next_reminder = started + WAIT_REMINDER_SECONDS
    next_recovery = started + WAIT_RECOVERY_SECONDS
    while result["status"] in {"queued", "running"} and (deadline is None or time.monotonic() < deadline):
        time.sleep(1 if deadline is None else min(0.1, max(0, deadline - time.monotonic())))
        result = turn_result(store, name, result["turn"])
        if deadline is None and result["status"] in {"queued", "running"}:
            now = time.monotonic()
            if now >= next_recovery:
                try:
                    launch(store, name)
                except OSError as exc:
                    # Recovery is optional while the original worker may
                    # still be alive. A launch failure must not discard its result.
                    print(f"[claude:{name}] Recovery check failed: {exc}; waiting continues.",
                          file=sys.stderr, flush=True)
                next_recovery = now + WAIT_RECOVERY_SECONDS
            if now >= next_reminder:
                print(f"[claude:{name}] Progress reminder: turn {result['turn']} is still "
                      f"{result['status']} after {(now - started) / 60:.0f} minutes of waiting. "
                      "Check recent log activity and task progress; inspect working-tree changes "
                      "if edits are expected. Read-only reviews may leave the tree unchanged. "
                      f"Waiting continues. Logs: {result['log_dir']}", file=sys.stderr, flush=True)
                next_reminder = now + WAIT_REMINDER_SECONDS
    return result


def result_code(result: dict, *, waiting: bool) -> int:
    if result["status"] in {"queued", "running"}:
        return 0 if waiting else 2
    return 0 if result["status"] == "completed" else 1


def emit(value, structured: bool, *, wait_hint: Path | None = None) -> None:
    if structured:
        print(json.dumps(value, ensure_ascii=False))
        return
    if isinstance(value, list):
        for item in value:
            emit(item, False)
        return
    activity = value.get("last_activity_s")
    suffix = f" · stream active {activity:g}s ago" if activity is not None else ""
    context = value.get("context") or {}
    if context.get("input_tokens") is not None:
        window = context.get("context_window")
        suffix += f" · ctx {context['input_tokens']:,}/{window if window else '?'}"
        if context.get("used_percent") is not None:
            suffix += f" ({context['used_percent']:.1f}%)"
    print(f"[claude:{value['name']}] {value['status']} · turn {value['turn']}{suffix}")
    if context.get("warning"):
        print("Warning: context is at least 45% full. Give new large tasks to a "
              "fresh spawn_agent; keep send_message for this task's followups.")
    if value.get("answer"):
        print(value["answer"].rstrip())
    if value.get("follow_ups"):
        print(followed_up_note(value.get("expect"), value["follow_ups"]))
    if value.get("expect_met") is False:
        print(incomplete_note(value.get("expect"), value.get("follow_ups", 0)))
    if value.get("error"):
        print(f"Error: {value['error']}")
    if value.get("permission_denials"):
        print(f"Tool requests denied: {len(value['permission_denials'])}; see logs")
    if value.get("log_dir"):
        print(f"Logs: {value['log_dir']}")
    for directory in value.get("follow_up_log_dirs") or []:
        print(f"Follow-up logs: {directory}")
    if wait_hint is not None and value["status"] in {"queued", "running"}:
        # PowerShell needs a call operator before a quoted executable and
        # literal single quotes for paths containing '$' or backticks.
        quote = (lambda s: "'" + s.replace("'", "''") + "'") if os.name == "nt" else shlex.quote
        launcher = ("& " if os.name == "nt" else "") + quote(sys.executable)
        script = quote(str(entry_point()))
        registry = quote(str(wait_hint.resolve()))
        print(f"Wait once: {launcher} {script} --state-dir {registry} wait_agent "
              f"--turn {value['turn']} --until-done -- {value['name']}")
        print("Run this in one background command/session. "
              "If the runner returns a session ID, resume that session instead of starting another wait. "
              "Chat completion notifications depend on the runner, not this CLI.")


# What a call without a host still does (`host_policy.announce_empty`).
EMPTY_STILL = ("the review profile refuses to start (it needs the host's gate "
               "hook), other profiles run unchanged")


def main(argv=None, *, policy=None, root=None, entry=None) -> int:
    """The command line. A host entry point passes its `policy` and `root`,
    and as `entry` the path `python <entry>` runs it by, which starts the
    detached worker and is what the wait hint prints (default: this
    package's directory); all three hold for this call only
    (`host_policy.installed`, `worker.entered`). Without a policy the call
    runs under `EMPTY` and says so on stderr."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if policy is not None and root is None:
        raise ValueError("a host policy comes with the host root it names")
    with host_policy.installed(policy or host_policy.EMPTY, root), entered(entry):
        host_policy.announce_empty("claude_agent", EMPTY_STILL)
        return _run(build_parser().parse_args(argv))


def _run(args) -> int:
    try:
        root = args.state_dir or os.environ.get("CLAUDE_AGENTS_DIR")
        store = Store(Path(root).resolve() if root else invocation_root() / ".claude-agents")
        if args.action == "_worker":
            work(store, args.name)
            return 0
        if args.action == "spawn":
            message = read_message(args)
            cwd = args.cwd.resolve() if args.cwd else invocation_root()
            if not cwd.is_dir():
                raise ValueError(f"Not a directory: {cwd}")
            options = {key: getattr(args, key) for key in
                       ("model", "effort", "profile", "turn_timeout")}
            agent = store.create(args.name, str(cwd), options, message,
                                 expected_answer(args.profile, args.expect, message))
            launch(store, agent["name"])
            if args.wait is not None:
                result = wait_result(store, agent["name"], agent["turn"], args.wait)
                emit(result, args.json, wait_hint=store.root)
                return result_code(result, waiting=True)
            emit(brief(agent, store), args.json, wait_hint=store.root)
        elif args.action == "send":
            message = read_message(args)
            profile = store.get(args.name)["options"].get("profile", "review")
            turn = store.enqueue(args.name, message,
                                 expected_answer(profile, args.expect, message))
            launch(store, args.name)
            if args.wait is not None:
                result = wait_result(store, args.name, turn, args.wait)
                emit(result, args.json, wait_hint=store.root)
                return result_code(result, waiting=True)
            value = brief(store.get(args.name), store)
            value.update(turn=turn, status="queued")
            emit(value, args.json, wait_hint=store.root)
        elif args.action == "status":
            agents = [store.get(args.name)] if args.name else store.list()
            for agent in agents:
                if agent["status"] in {"queued", "running"} or agent["owner"]:
                    launch(store, agent["name"])
            emit([brief(agent, store) for agent in agents], args.json)
        elif args.action == "interrupt":
            store.interrupt(args.name)
            # Recover a crashed owner too; active owners observe the flag.
            launch(store, args.name)
            value = brief(store.get(args.name), store)
            if store.get(args.name)["owner"]:
                value["status"] = "interrupt_requested"
            emit(value, args.json)
        elif args.action in {"result", "wait", "logs"}:
            result = (wait_result(store, args.name, args.turn, args.timeout)
                      if args.action == "wait" else turn_result(store, args.name, args.turn))
            if args.action == "logs":
                directory = store.root / result["name"] / f"turn-{result['turn']:04d}"
                result = {"name": result["name"], "turn": result["turn"],
                          "status": result["status"], "log_dir": str(directory),
                          "follow_up_log_dirs": result.get("follow_up_log_dirs") or []}
            emit(result, args.json, wait_hint=store.root if args.action != "logs" else None)
            if args.action != "logs":
                return result_code(result, waiting=args.action == "wait")
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, sqlite3.Error) as exc:
        message = str(exc.args[0]) if isinstance(exc, KeyError) else str(exc)
        if getattr(args, "json", False):
            print(json.dumps({"status": "error", "error": message}, ensure_ascii=False))
        else:
            print(f"claude_agent: {message}", file=sys.stderr)
        return 1
