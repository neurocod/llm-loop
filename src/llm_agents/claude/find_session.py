"""Find Claude Code sessions by a fragment of a prompt typed into them.

    python -m llm_agents.claude.find_session "fix the parser" [--days 7] [--project SUBSTR]

Scans the transcripts under ~/.claude/projects (all projects unless --project is
given), newest match first. Each hit prints the id for `claude --resume <id>`, the
local time of the latest matching prompt and that prompt collapsed to one line,
then the session's last turn: the final human prompt and the final assistant text.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECTS = Path.home() / ".claude" / "projects"
# The auto-generated recap that opens a session continued after a context overflow is recorded as a user prompt.
CONTINUATION = "This session is being continued"


def prompt_text(rec):
    """Text of a prompt typed by the human, or None for tool results, meta and slash-command noise."""
    if rec.get("type") != "user" or rec.get("isMeta") or rec.get("isSidechain"):
        return None
    origin = rec.get("origin")
    if origin is not None and origin.get("kind") != "human":
        return None
    content = rec.get("message", {}).get("content")
    if isinstance(content, list):
        content = "\n".join(b.get("text", "") for b in content if b.get("type") == "text")
    if not isinstance(content, str) or not content.strip() or content.lstrip().startswith(("<", CONTINUATION)):
        return None
    return content


def parse_time(stamp):
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).astimezone()


def reply_text(rec):
    """Text blocks of a main-thread assistant record, or None when it carries none (tool calls only)."""
    if rec.get("type") != "assistant" or rec.get("isSidechain"):
        return None
    content = rec.get("message", {}).get("content")
    if not isinstance(content, list):
        return None
    text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
    return text or None


def scan(path, needle):
    """(match, last_prompt, last_reply) of one transcript, each a (time, text) or None.

    match is the last prompt containing needle; None for it means the transcript does not qualify.
    """
    match = prompt = reply = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"user"' not in line and '"assistant"' not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if not rec.get("timestamp"):
                continue
            text = prompt_text(rec)
            if text is not None:
                prompt = (parse_time(rec["timestamp"]), text)
                if needle in text.lower():
                    match = prompt
                continue
            text = reply_text(rec)
            if text is not None:
                reply = (parse_time(rec["timestamp"]), text)
    return match, prompt, reply


def oneline(text, limit):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit] + "…" if limit and len(text) > limit else text


MATCH = "1;30;43"  # bold black on yellow: the matched fragment
DATE = "36"  # cyan: timestamps


def paint(text, sgr, enabled):
    return f"\x1b[{sgr}m{text}\x1b[0m" if enabled else text


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False, description=__doc__.split("\n")[0])
    ap.add_argument("text", help="fragment of the prompt (case-insensitive)")
    ap.add_argument("--days", type=float, default=7, help="only transcripts modified within N days")
    ap.add_argument("--project", default="", help="only project dirs whose name contains this (case-insensitive)")
    ap.add_argument("--max-chars", type=int, default=0, help="truncate the printed prompt (0 = whole)")
    ap.add_argument("--last-chars", type=int, default=300, help="truncate the last prompt/reply (0 = whole)")
    ap.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                    help="highlight the match and the dates with ANSI colors (auto: only on a terminal)")
    ap.add_argument("--no-last", action="store_true", help="do not print the session's last turn")
    args = ap.parse_args()

    if not PROJECTS.is_dir():
        sys.exit(f"no transcripts at {PROJECTS}")
    needle = args.text.lower()
    cutoff = time.time() - args.days * 86400
    hits = []
    for proj in PROJECTS.iterdir():
        if not proj.is_dir() or args.project.lower() not in proj.name.lower():
            continue
        for path in proj.glob("*.jsonl"):  # subagent transcripts live deeper and are not resumable
            if os.path.getmtime(path) < cutoff:
                continue
            match, prompt, reply = scan(path, needle)
            if match:
                hits.append((match[0], path.stem, proj.name, match[1], prompt, reply))
    hits.sort(key=lambda h: h[0], reverse=True)
    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty() and "NO_COLOR" not in os.environ)
    if color:
        os.system("")  # makes the Windows console interpret ANSI escapes
    # The collapsed text has single spaces, so any whitespace run in the needle matches one.
    found = re.compile(r"\s+".join(map(re.escape, needle.split())), re.IGNORECASE)

    def show(text, limit):
        text = oneline(text, limit)
        return found.sub(lambda m: paint(m.group(), MATCH, color), text) if color else text

    for when, sid, proj, text, prompt, reply in hits:
        print(f"{sid}  {paint(f'{when:%Y-%m-%d %H:%M}', DATE, color)}  [{proj}]\n    {show(text, args.max_chars)}")
        if not args.no_last:
            for label, turn in (("last prompt", prompt), ("last reply", reply)):
                if turn:
                    print(f"    -> {label} {paint(f'{turn[0]:%m-%d %H:%M}', DATE, color)}: {show(turn[1], args.last_chars)}")
        print()
    if not hits:
        print("nothing found", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
