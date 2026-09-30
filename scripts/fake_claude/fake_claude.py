"""A stand-in for `claude -p ... --output-format stream-json`: no model, no cost.

It speaks just enough of the wire (`llm_loop.wire`) for a runner to render it —
`system`/`init`, partial-message text deltas, an `assistant` message, a tool
call with its result, the `result` event — and, when started with
`--input-format stream-json`, it reads the prompt and every later note from
stdin and replays each note back as a `user` event, as `--replay-user-messages`
does. That is the whole path a note typed with the `m` key travels.

What it is for: reproducing console trouble (the `m` editor freezing mid-note,
a stuck painter) against a live runner without spending a turn, while
`py-spy dump --pid <runner pid>` and the diagnostics log (`llm_loop.diaglog`)
watch. `fake_cycle.py` beside it drives a runner at this script.

Knobs, all environment variables so the runner's argv stays the real one:

  FAKE_CLAUDE_SECONDS   how long one turn streams (default 300)
  FAKE_CLAUDE_DELTA_MS  pause between two text deltas (default 40) — lower it
                        to press the terminal harder
  FAKE_CLAUDE_TOOL_EVERY  a tool call every N deltas (default 60)
  FAKE_CLAUDE_TOOL_SECONDS  make the FIRST tool call a silent foreground
                        child running this long (default 0: none) — the agent
                        waiting on a script, with nothing on the stream
  FAKE_CLAUDE_TOOL_CMD  that child: a preset (`sleep`, `conin`, `powershell`,
                        `bash` — see TOOL_PRESETS) or a command line with `{s}`
"""

import json
import os
import sys
import threading
import time

SESSION = "fake-session-0001"
MODEL = "fake-model"
WORDS = ("the painter owns the terminal while the reader posts keys and the "
         "agent streams text into the same console ").split()


_emit_lock = threading.Lock()


def emit(event: dict) -> None:
    """One event line. Two threads emit (the stream and the note replay), and a
    line must never be cut by the other's."""
    with _emit_lock:
        sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def stream_event(inner: dict) -> dict:
    return {"type": "stream_event", "event": inner, "session_id": SESSION}


def usage() -> dict:
    return {"input_tokens": 1000, "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0, "output_tokens": 10}


def read_notes(first_line: threading.Event, done: threading.Event) -> None:
    """stdin: the prompt first, then notes until the runner closes the pipe."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if not first_line.is_set():
            first_line.set()
            continue
        try:
            message = json.loads(line)
        except ValueError:
            emit({"type": "system", "subtype": "fake_bad_input", "line": line})
            continue
        # --replay-user-messages: the note comes back on the event stream, its
        # content flattened to a bare string as the real CLI does
        # (`wire.message_blocks` handles exactly that shape).
        content = message.get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(block.get("text", "") for block in content
                              if isinstance(block, dict))
        emit({"type": "user", "session_id": SESSION,
              "message": {"role": "user", "content": content}})
    done.set()


# Presets for FAKE_CLAUDE_TOOL_CMD; `{s}` is FAKE_CLAUDE_TOOL_SECONDS. `sleep`
# only sits on the console; `conin` reads the console's input queue the way a
# prompting program does, which is the key-stealing case to rule in or out.
TOOL_PRESETS = {
    "sleep": '"{py}" -c "import time; time.sleep({s})"',
    # One line: the command goes through cmd.exe, which ends it at a newline.
    "conin": ('"{py}" -c "import msvcrt, time; t = time.time() + {s}; '
              '[print(repr(msvcrt.getwch()), flush=True) if msvcrt.kbhit() '
              'else time.sleep(0.05) '
              'for _ in iter(lambda: time.time() < t, False)]"'),
    "powershell": 'powershell -NoProfile -NonInteractive -Command "Start-Sleep {s}"',
    "bash": 'bash -c "sleep {s}"',
}


def tool_command(seconds: float) -> str:
    raw = os.environ.get("FAKE_CLAUDE_TOOL_CMD", "sleep")
    template = TOOL_PRESETS.get(raw, raw)
    return template.format(py=sys.executable, s=seconds)


def run_long_tool(command: str) -> str:
    """Run it as a tool would: a child that inherits this process's console
    (the runner's own), stdin closed, output captured — and nothing on the
    event stream until it ends, like a foreground script the agent waits on."""
    import subprocess

    done = subprocess.run(command, shell=True, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, errors="replace")
    return (done.stdout + done.stderr).strip() or f"exit {done.returncode}"


def main() -> int:
    # The real CLI speaks UTF-8 on its pipes and the runner decodes UTF-8; a
    # Windows pipe defaults to the ANSI code page, which cannot encode a
    # Cyrillic note replayed back (UnicodeEncodeError, seen 2026-09-30).
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    live = "--input-format" in argv and "stream-json" in argv
    seconds = float(os.environ.get("FAKE_CLAUDE_SECONDS", "300"))
    delta_s = float(os.environ.get("FAKE_CLAUDE_DELTA_MS", "40")) / 1000
    tool_every = int(os.environ.get("FAKE_CLAUDE_TOOL_EVERY", "60"))
    tool_seconds = float(os.environ.get("FAKE_CLAUDE_TOOL_SECONDS", "0"))

    got_prompt, stdin_closed = threading.Event(), threading.Event()
    if live:
        threading.Thread(target=read_notes, args=(got_prompt, stdin_closed),
                         daemon=True).start()
        got_prompt.wait(10)
    else:
        stdin_closed.set()

    emit({"type": "system", "subtype": "init", "session_id": SESSION,
          "model": MODEL, "cwd": os.getcwd(), "tools": []})
    started = time.monotonic()
    deltas, message_no = 0, 0
    while time.monotonic() - started < seconds:
        message_no += 1
        emit(stream_event({"type": "message_start", "message": {
            "model": MODEL, "usage": usage()}}))
        emit(stream_event({"type": "content_block_start", "index": 0,
                           "content_block": {"type": "text", "text": ""}}))
        text = []
        for _ in range(tool_every):
            word = WORDS[deltas % len(WORDS)] + " "
            deltas += 1
            text.append(word)
            emit(stream_event({"type": "content_block_delta", "index": 0,
                               "delta": {"type": "text_delta", "text": word}}))
            time.sleep(delta_s)
            if time.monotonic() - started >= seconds:
                break
        emit(stream_event({"type": "content_block_stop", "index": 0}))
        tool_id = f"toolu_fake_{message_no}"
        long_tool = message_no == 1 and tool_seconds > 0
        command = (tool_command(tool_seconds) if long_tool
                   else f"echo fake step {message_no}")
        emit({"type": "assistant", "session_id": SESSION, "message": {
            "model": MODEL, "usage": usage(), "content": [
                {"type": "text", "text": "".join(text)},
                {"type": "tool_use", "id": tool_id, "name": "Bash",
                 "input": {"command": command,
                           "description": "Fake tool call"}}]}})
        output = run_long_tool(command) if long_tool else f"fake step {message_no}"
        emit({"type": "user", "session_id": SESSION, "message": {
            "role": "user", "content": [
                {"type": "tool_result", "tool_use_id": tool_id,
                 "content": output}]}})

    emit({"type": "result", "subtype": "success", "is_error": False,
          "session_id": SESSION, "duration_ms": int(seconds * 1000),
          "num_turns": message_no, "total_cost_usd": 0.0,
          "result": "fake turn finished"})
    # With stdin open the runner closes it at `result`; like the real CLI,
    # exit once it has. Bounded, so a runner that never closes is visible.
    stdin_closed.wait(30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
