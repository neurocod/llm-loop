"""Print a compact, chronological view of one Codex rollout JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def find_rollout(value: str, sessions_root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_file():
        return candidate

    matches = list(sessions_root.rglob(f"*{value}*.jsonl"))
    if not matches:
        raise SystemExit(f"No rollout matching {value!r} under {sessions_root}")
    if len(matches) > 1:
        names = "\n".join(f"  {path}" for path in matches)
        raise SystemExit(f"More than one rollout matches {value!r}:\n{names}")
    return matches[0]


def text_parts(content: Any) -> Iterable[str]:
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            yield text


def flatten(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def compact(text: str, limit: int) -> str:
    text = " ".join(text.replace("\x00", "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def format_event(record: dict[str, Any], limit: int) -> str | None:
    timestamp = record.get("timestamp", "?")
    record_type = record.get("type")
    payload = record.get("payload") or {}

    if record_type == "session_meta":
        session_id = payload.get("session_id") or payload.get("id")
        return f"{timestamp} SESSION {session_id} cwd={payload.get('cwd')}"

    if record_type == "event_msg":
        event_type = payload.get("type")
        if event_type == "token_count":
            return None
        if event_type == "agent_message":
            return f"{timestamp} FINAL {compact(str(payload.get('message', '')), limit)}"
        interesting = {
            "task_started",
            "task_complete",
            "task_failed",
            "turn_aborted",
            "sub_agent_activity",
            "context_compacted",
        }
        if event_type in interesting:
            return f"{timestamp} EVENT {event_type} {compact(flatten(payload), limit)}"
        return None

    if record_type != "response_item":
        return None

    item_type = payload.get("type")
    if item_type == "message":
        role = str(payload.get("role", "?")).upper()
        texts = list(text_parts(payload.get("content")))
        if role == "DEVELOPER":
            return None
        return f"{timestamp} {role} {compact(' '.join(texts), limit)}"
    if item_type in {"function_call", "custom_tool_call"}:
        namespace = payload.get("namespace")
        name = payload.get("name", "?")
        qualified = f"{namespace}.{name}" if namespace else str(name)
        args = payload.get("arguments", payload.get("input", ""))
        return f"{timestamp} CALL {qualified} {compact(flatten(args), limit)}"
    if item_type in {"function_call_output", "custom_tool_call_output"}:
        output = payload.get("output", "")
        return f"{timestamp} RESULT {payload.get('call_id', '?')} {compact(flatten(output), limit)}"
    return None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("rollout", help="JSONL path or unique session/thread ID fragment")
    parser.add_argument("--limit", type=int, default=500, help="maximum characters per event")
    parser.add_argument(
        "--sessions-root",
        type=Path,
        default=Path.home() / ".codex" / "sessions",
    )
    args = parser.parse_args()

    path = find_rollout(args.rollout, args.sessions_root)
    print(path)
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                print(f"line {line_number}: invalid JSON: {error}")
                continue
            rendered = format_event(record, args.limit)
            if rendered:
                print(f"{line_number}: {rendered}")


if __name__ == "__main__":
    main()