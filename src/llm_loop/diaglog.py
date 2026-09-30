"""The status line's quiet failures, written where somebody can read them later.

The status line is built to survive its own faults: a key reader that raises
ends quietly (`termio.TerminalInput._guard`), a console write that raises marks
the terminal failed and every later frame returns early (`termio.Terminal._write`),
a frame that raises swaps the terminal for a NullTerminal for good
(`statusline.Painter._disable`), a key the painter has no room for is dropped
(`Painter.post_key`). Each is the right call for the RUN, and each leaves the
operator looking at a screen that simply stops changing — the `m` editor
freezing a few words into a note, reported more than once, with nothing on
screen or in the mirror log to say which of them it was. So every such branch
names itself here.

Written only while this process holds an exit record that has not finished
(`exitlog.current()`), and beside it: a dry run, a pin driving an app by hand and
a test that did not stage a run write nothing, and a staged test's record — so
this file too — lives in its tmp dir (`_runfixtures.isolated_run`). One file per
app and project, like the mirror log; every line carries the pid, since
concurrent runs share it.

`LLM_LOOP_KEYTRACE=1` adds one line per stage a key passes — read, posted (or
dropped), applied, painted — so the last line before a freeze says which stage
the keys stopped at. Off by default: it is a line per keystroke and per frame.

The callers are the key reader and the painter, the very threads under
suspicion, and `post_key` promises the reader never waits (Ctrl+C arrives as a
key on Windows). So a caller only formats its line and puts it on a
`queue.SimpleQueue` — no lock, no file I/O, safe from a signal handler — and a
daemon writer thread does the writing. The price: the lines of a process killed
outright may not reach the file; `atexit` drains what it can, bounded.

Best-effort throughout: a diagnostic that raises would be the very kind of
failure it is here to report.
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Optional

from . import exitlog

KEYTRACE_ENV = "LLM_LOOP_KEYTRACE"
DIAG_SUFFIX = ".diag.log"
# The key trace is a line per keystroke and per frame (4 a second while idle);
# past this the file is started over, keeping one previous generation (.1).
MAX_BYTES = 5 * 1024 * 1024
# How long `flush` (and so interpreter exit) waits for the writer. A line is a
# few hundred bytes appended to an open file; only a stalled disk gets near it.
FLUSH_WAIT_S = 1.0

_lines: "queue.SimpleQueue" = queue.SimpleQueue()
_writer: Optional[threading.Thread] = None
_writer_started = threading.Lock()


def keytrace_enabled() -> bool:
    return os.environ.get(KEYTRACE_ENV, "") not in ("", "0", "false", "no")


def log_path() -> Optional[Path]:
    """Where the lines go: beside this process's live exit record, or nowhere."""
    record = exitlog.current()
    if record is None or record.finished:
        return None
    # `<app>-<project>.<pid>.run.json` -> `<app>-<project>.diag.log`. Cut from
    # the right: the project is a directory name and may itself hold dots.
    name = record.path.name
    if name.endswith(exitlog.RECORD_SUFFIX):
        name = name[:-len(exitlog.RECORD_SUFFIX)]
    stem = name.rsplit(".", 1)[0]
    return record.path.with_name(stem + DIAG_SUFFIX)


def record(where: str, text: str = "",
           exc: Optional[BaseException] = None) -> None:
    """One line (plus a traceback for `exc`) naming the branch that was taken."""
    try:
        path = log_path()
        if path is None:
            return
        now = time.time()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
        line = (f"{stamp}.{int(now * 1000) % 1000:03d} pid={os.getpid()} "
                f"[{threading.current_thread().name}] {where}")
        if text:
            line += f": {text}"
        if exc is not None:
            line += "\n" + "".join(traceback.format_exception(
                type(exc), exc, exc.__traceback__)).rstrip()
        _lines.put((path, line))
        _ensure_writer()
    except Exception:
        pass


def trace(where: str, text: str = "") -> None:
    """`record`, only while `LLM_LOOP_KEYTRACE` is on."""
    if keytrace_enabled():
        record(where, text)


def console_snapshot() -> str:
    """Who shares this console, and its input/output modes (Windows; "" else).

    A Windows console's input buffer is ONE queue for every process attached to
    it: the agent's descendants (a shell, the script it runs) inherit the
    console even though the agent's own stdio are pipes. One of them reading
    CONIN$, or switching the input mode, takes keys the status line never sees
    — so the list and the modes are what tells that apart from a stuck reader.
    """
    if os.name != "nt":
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        pids = (wintypes.DWORD * 64)()
        count = kernel32.GetConsoleProcessList(pids, 64)
        names = []
        for pid in list(pids)[:min(count, 64)]:
            names.append(f"{pid}:{_process_name(kernel32, pid)}")
        modes = []
        for label, std in (("in", -10), ("out", -11)):
            mode = wintypes.DWORD()
            handle = kernel32.GetStdHandle(std)
            ok = kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            modes.append(f"{label}=0x{mode.value:04x}" if ok else f"{label}=?")
        return f"{count} process(es) [{' '.join(names)}] {' '.join(modes)}"
    except Exception as exc:
        return f"console snapshot failed: {exc!r}"


def _process_name(kernel32, pid: int) -> str:
    import ctypes
    from ctypes import wintypes

    handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    if not handle:
        return "?"
    try:
        buffer = ctypes.create_unicode_buffer(512)
        size = wintypes.DWORD(512)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buffer,
                                               ctypes.byref(size)):
            return os.path.basename(buffer.value)
        return "?"
    finally:
        kernel32.CloseHandle(handle)


def flush(timeout: float = FLUSH_WAIT_S) -> bool:
    """Wait until every line put so far is written; False past `timeout`."""
    if _writer is None:
        return True
    done = threading.Event()
    _lines.put(done)
    return done.wait(timeout)


def _ensure_writer() -> None:
    global _writer
    if _writer is not None:
        return
    # Non-blocking: a caller that loses the race leaves its line queued for
    # the writer the winner is starting.
    if not _writer_started.acquire(blocking=False):
        return
    try:
        if _writer is None:
            thread = threading.Thread(target=_write_lines, name="diaglog-writer",
                                      daemon=True)
            thread.start()
            _writer = thread
            atexit.register(flush)
    finally:
        _writer_started.release()


def _write_lines() -> None:
    files = {}
    while True:
        item = _lines.get()
        if isinstance(item, threading.Event):
            item.set()
            continue
        path, line = item
        try:
            out = files.get(path)
            if out is not None and out.tell() > MAX_BYTES:
                out.close()
                out = None
                os.replace(path, path.with_name(path.name + ".1"))
            if out is None:
                out = files[path] = open(path, "a", encoding="utf-8",
                                         errors="replace")
            out.write(line + "\n")
            out.flush()
        except Exception:
            files.pop(path, None)
