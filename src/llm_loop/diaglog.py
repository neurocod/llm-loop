"""The status line's quiet failures, written where somebody can read them later.

The status line is built to survive its own faults: a key reader that raises
ends quietly (`termio.TerminalInput._guard`), a frame that raises swaps the
terminal for a NullTerminal for good (`statusline.Painter._disable`), a key the
painter has no room for is dropped (`Painter.post_key`). Each is the right call
for the RUN, and each leaves the operator looking at a screen that simply stops
changing — the `m` editor freezing a few words into a note, reported more than
once, with nothing on screen or in the mirror log to say which of them it was.
So every such branch names itself here.

Written only while this process holds an exit record (`exitlog.current()`), and
beside it: a dry run, a pin driving an app by hand and a test that did not stage
a run write nothing, and a staged test's record — so this file too — lives in its
tmp dir (`_runfixtures.isolated_run`). One file per app and project, like the
mirror log; every line carries the pid, since concurrent runs share it.

`LLM_LOOP_KEYTRACE=1` adds one line per stage a key passes — read, posted (or
dropped), applied, painted — so the last line before a freeze says which stage
the keys stopped at. Off by default: it is a line per keystroke and per frame.

Best-effort throughout: a diagnostic that raises would be the very kind of
failure it is here to report.
"""

from __future__ import annotations

import os
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

_lock = threading.Lock()


def keytrace_enabled() -> bool:
    return os.environ.get(KEYTRACE_ENV, "") not in ("", "0", "false", "no")


def log_path() -> Optional[Path]:
    """Where the lines go: beside this process's exit record, or nowhere."""
    record = exitlog.current()
    if record is None:
        return None
    # `<app>-<project>.<pid>.run.json` -> `<app>-<project>.diag.log`
    stem = record.path.name.split(".", 1)[0]
    return record.path.with_name(stem + DIAG_SUFFIX)


def record(where: str, text: str = "",
           exc: Optional[BaseException] = None) -> None:
    """One line (plus a traceback for `exc`) naming the branch that was taken."""
    path = log_path()
    if path is None:
        return
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    millis = int(time.time() * 1000) % 1000
    line = (f"{stamp}.{millis:03d} pid={os.getpid()} "
            f"[{threading.current_thread().name}] {where}")
    if text:
        line += f": {text}"
    if exc is not None:
        line += "\n" + "".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)).rstrip()
    try:
        with _lock:
            try:
                if path.stat().st_size > MAX_BYTES:
                    os.replace(path, path.with_name(path.name + ".1"))
            except OSError:
                pass
            with open(path, "a", encoding="utf-8", errors="replace") as out:
                out.write(line + "\n")
    except Exception:
        pass


def trace(where: str, text: str = "") -> None:
    """`record`, only while `LLM_LOOP_KEYTRACE` is on."""
    if keytrace_enabled():
        record(where, text)
