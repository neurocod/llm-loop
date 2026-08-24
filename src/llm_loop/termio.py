"""termio.py - the terminal itself: the pinned region's bytes, and the keys back.

The front end under `statusline`. It knows nothing about what a status row means
— no `LoopStatus`, no `Row`, no `Segment` reaches it — and `statusline` in turn
never writes an escape or reads a key of its own. That is the whole seam: one
side decides WHAT the pinned area says, this one moves the bytes.

Two halves, and they are not the same question:

  * out — `Terminal` reserves the region (DECSTBM), repaints its rows, names the
    window and gives all of it back; `NullTerminal` is the Null Object every
    failure path falls to, and `terminal_for` decides between them once.
  * in — `InputSource`/`TerminalInput` read keys on a daemon thread and
    `_EscapeDecoder` turns a platform's spelling (CSI on POSIX, msvcrt scan
    codes on Windows) into one symbolic `Key`, so nothing above this module
    learns which platform it is on.

The one thing it imports from the package is `console.real_stream`, and for
the reason stated there: `sys.stdout` is the mirror-log tee, and cursor-movement
bytes in the log corrupt the run record `--cost` parses.
"""

import os
import shutil
import sys
import threading
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from . import console

__all__ = [
    "CSI_KEYS",
    "ENV_FLAG",
    "InputEvent",
    "InputSource",
    "Key",
    "MIN_COLUMNS",
    "MIN_LINES",
    "Mouse",
    "NullInputSource",
    "NullTerminal",
    "Resize",
    "TITLE_RESET",
    "TITLE_SET",
    "Terminal",
    "TerminalInput",
    "WINDOWS_KEYS",
    "decode_escape",
    "terminal_for",
]

# Environment kill switch, honoured next to --no-statusline so a hostile terminal
# can be worked around without editing any command line.
ENV_FLAG = "LLM_LOOP_STATUSLINE"

# Below these the reserved region would eat the visible scrollback, so we simply
# do not pin anything.
MIN_COLUMNS = 40
MIN_LINES = 8

# OSC 0: sets the window title AND the icon/tab name, so one escape covers both
# a tab strip and a taskbar button. BEL-terminated rather than ST — every
# emulator this runs under accepts BEL, some older ones only accept BEL.
TITLE_SET = "\x1b]0;{}\x07"
# The empty title is the documented way back: Windows Terminal (and Tabby, and
# the usual POSIX emulators) fall back to the profile's own title, which is
# nearer to "as we found it" than any string we could invent.
TITLE_RESET = TITLE_SET.format("")


# --- terminal ------------------------------------------------------------------


def _enable_windows_vt(stream) -> bool:
    """Turn on ENABLE_VIRTUAL_TERMINAL_PROCESSING; True if escapes will work.

    Rich has usually done this already, but the status line must not depend on
    rich being installed. A console that refuses simply gets no status line.
    """
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


class Terminal:
    """The reserved bottom region: bytes only, no idea what a row means.

    Mechanics: a DECSTBM scroll region shrinks the scrolling area by `rows`, so
    ordinary output keeps scrolling above untouched. Repaints save/restore the
    cursor around absolute moves, which is also what heals the rows after rich's
    Live has been moving the cursor around for a streamed Markdown block.
    """

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else console.real_stream()
        self._lock = threading.Lock()
        self._rows = 0
        self._size = (0, 0)
        self._title = None    # last title written (None = we never set one)
        self._released = False  # torn down: writes no more title (see set_title)
        self.failed = False

    @property
    def active(self) -> bool:
        return self._rows > 0 and not self.failed

    def size(self) -> Tuple[int, int]:
        """(columns, lines) of the terminal right now."""
        size = shutil.get_terminal_size(fallback=(80, 24))
        return size.columns, size.lines

    def _write(self, text: str) -> bool:
        try:
            self._stream.write(text)
            self._stream.flush()
            return True
        except Exception:
            # A closed/odd stream must not take the loop down with it.
            self.failed = True
            return False

    def reserve(self, rows: int) -> bool:
        """Pin `rows` lines at the bottom. Safe to call again to resize."""
        if self.failed or rows <= 0:
            return False
        columns, lines = self.size()
        if columns < MIN_COLUMNS or lines < MIN_LINES or rows >= lines - 1:
            return False
        with self._lock:
            # Pinning rows is a run starting here, so the title ban lifts: a
            # Terminal that was released and is now reserved again is in use.
            self._released = False
            # Scroll only for the rows we are *adding*: re-establishing the
            # region on every resize must not walk the output up the screen.
            grown = max(0, rows - self._rows)
            self._rows = rows
            self._size = (columns, lines)
            first = lines - rows + 1
            return self._write(
                "\n" * grown
                + f"\x1b[{first};1H\x1b[0J"        # clear the whole reserved area
                + f"\x1b[1;{lines - rows}r"        # DECSTBM: shrink the scroll area
                + f"\x1b[{lines - rows};1H")       # park at the bottom of it

    def set_title(self, text: str, *, reassert: bool = False) -> bool:
        """Name the window/tab. Repeats are dropped, so this is cheap to call
        on every repaint — and an emulator that ignores OSC 0 simply swallows a
        few bytes that never reach the scrollback.

        Not gated on `active`, because a title has no geometry behind it: it is
        written as soon as the run has something to say, before any region is
        pinned. It IS refused once `release()` has run, and that is not
        symmetry for its own sake — nothing would ever clear a name written
        after the teardown. Two threads reach here after it routinely: a worker
        still finishing its job past a parallel run's Ctrl+C, and the quota
        refresher whose 1 s join timed out. Both used to leave the dead run's
        name on the window for good.

        `reassert` writes even when the text has not changed, and the periodic
        repaint passes it for the same reason it re-states the scroll region
        (see `paint`): the name is not ours alone. Anything the run launches can
        take it — an escape from a provider CLI or a pager, and on Windows a
        child that calls SetConsoleTitle, which no escape of ours is answering.
        Without a re-assert the dedupe below then holds the window on somebody
        else's name for the rest of the run, because as far as we know we
        already wrote ours.
        """
        if self.failed or self._released:
            return False
        with self._lock:
            # Compared UNDER the lock, with the write: two threads painting
            # different states could otherwise leave the newer text recorded and
            # the older one on screen, and every later repaint would then skip
            # the correction as a duplicate.
            if text == self._title and not reassert:
                return False
            self._title = text
            return self._write(TITLE_SET.format(text))

    def paint(self, lines: Sequence[str], *, reassert: bool = False) -> bool:
        if not self.active:
            return False
        with self._lock:
            columns, total = self._size
            first = total - self._rows + 1
            out = ["\x1b7"]  # save cursor (DECSC) — output above must not move
            if reassert:
                # The region is not sticky: an `ESC[r`, an alt-screen switch or a
                # full reset from anything downstream (a provider CLI, a pager)
                # un-pins it for good, and every absolute row write below would
                # then scroll away with the output. Re-stating it costs one short
                # escape on the periodic repaint; DECSTBM homes the cursor, which
                # the DECRC at the end undoes.
                out.append(f"\x1b[1;{total - self._rows}r")
            for i in range(self._rows):
                text = lines[i] if i < len(lines) else ""
                out.append(f"\x1b[{first + i};1H\x1b[2K{text}")
            out.append("\x1b8")  # restore cursor (DECRC)
            return self._write("".join(out))

    def release(self) -> None:
        """Reset the scroll region and the title, clear the rows, leave a clean
        cursor.

        The title is put back BEFORE the early return: a run that never managed
        to pin a row may still have named the window, and a finished run whose
        name still says "iter 7/40" is worse than no name at all.
        """
        with self._lock:
            if self._title is not None:
                self._title = None
                self._write(TITLE_RESET)
            self._released = True
            if self._rows <= 0:
                return
            _columns, total = self._size
            first = total - self._rows + 1
            out = ["\x1b7"]
            for i in range(self._rows):
                out.append(f"\x1b[{first + i};1H\x1b[2K")
            out.append("\x1b8")
            out.append("\x1b[r")             # full-height scroll region again
            out.append(f"\x1b[{first};1H")   # sit on the first freed line
            self._rows = 0
            self._write("".join(out))


class NullTerminal(Terminal):
    """Null Object: chosen when there is no TTY, when disabled, or after any
    error — so the controller has no `if enabled:` scattered through it."""

    def __init__(self):
        self._rows = 0
        self.failed = False

    @property
    def active(self) -> bool:
        return False

    def size(self):
        return (0, 0)

    def reserve(self, rows):
        return False

    def set_title(self, text, *, reassert=False):
        return False

    def paint(self, lines, *, reassert=False):
        return False

    def release(self):
        return None


def terminal_for(stream=None, *, enabled: bool = True) -> Terminal:
    """A real Terminal when pinning rows is safe here, else a NullTerminal."""
    if not enabled or os.environ.get(ENV_FLAG) == "0":
        return NullTerminal()
    stream = stream if stream is not None else console.real_stream()
    try:
        if not stream.isatty():
            return NullTerminal()
    except Exception:
        return NullTerminal()
    if not _enable_windows_vt(stream):
        return NullTerminal()
    size = shutil.get_terminal_size(fallback=(0, 0))
    if size.columns < MIN_COLUMNS or size.lines < MIN_LINES:
        return NullTerminal()
    return Terminal(stream)


# --- input ---------------------------------------------------------------------


class InputEvent:
    """Something the user did. Subclasses carry the payload."""


@dataclass
class Key(InputEvent):
    char: str


@dataclass
class Mouse(InputEvent):
    """An SGR mouse report. Produced from wave 4 on; the type exists now so the
    Mode/Action dispatch already speaks it."""

    button: int
    x: int
    y: int
    pressed: bool = True


@dataclass
class Resize(InputEvent):
    columns: int
    lines: int


# Named keys. A `Key.char` longer than one character is one of these symbolic
# names, so a Mode can ask for "left" without knowing whether the terminal spelt
# it as a CSI sequence (POSIX) or a two-byte scan code (Windows).
CSI_KEYS = {
    "A": "up", "B": "down", "C": "right", "D": "left", "H": "home", "F": "end",
    "1~": "home", "3~": "delete", "4~": "end", "5~": "pgup", "6~": "pgdn",
    "7~": "home", "8~": "end",
    # Modified arrows: `ESC [ 1 ; <mod> <letter>`, where the modifier is
    # 1+bitmask (2 shift, 4 alt, 8 ctrl). Ctrl (5) and Alt (3) both mean
    # word-wise here — which one a terminal actually sends is not the typist's
    # choice, and no other meaning is on offer in a one-line editor.
    "1;5C": "wordright", "1;5D": "wordleft",
    "1;3C": "wordright", "1;3D": "wordleft",
    "1;5H": "home", "1;5F": "end",
}
# msvcrt reports these as a '\x00'/'\xe0' lead byte plus a scan code. The
# word-wise codes are the console's own (Ctrl+Left 0x73 's', Ctrl+Right 0x74
# 't', Ctrl+Home 0x77 'w', Ctrl+End 0x75 'u'); the lead byte is what keeps them
# from colliding with those letters.
WINDOWS_KEYS = {
    "H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end",
    "I": "pgup", "Q": "pgdn", "S": "delete",
    "s": "wordleft", "t": "wordright", "w": "home", "u": "end",
    "\x93": "delete",                       # Ctrl+Delete
}
_WINDOWS_LEAD = ("\x00", "\xe0")


def decode_escape(sequence: str) -> Optional[InputEvent]:
    """A CSI sequence -> an InputEvent, or None to drop it.

    The single seam for escape-driven input: wave 4 turns
    `ESC [ < b ; x ; y M|m` into a Mouse here and nothing else changes. Unknown
    sequences are dropped rather than delivered as a burst of Keys, which would
    look to the user like random keypresses.
    """
    body = sequence[2:]                       # strip the leading "ESC ["
    name = CSI_KEYS.get(body)
    return Key(name) if name else None


class InputSource:
    """Produces InputEvents on a daemon thread."""

    def start(self, handler: Callable[[InputEvent], None]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def restore_tty(self) -> None:
        """Undo any terminal mode this source set. Nothing to undo by default.

        Separate from `stop()` because it must be callable from a signal handler
        or from atexit, where the reader thread's own `finally` will never run.
        """
        return None


class NullInputSource(InputSource):
    """No TTY (or disabled): yields nothing, so no key is ever read."""

    def start(self, handler):
        return None

    def stop(self):
        return None


class _EscapeDecoder:
    """Assembles keystrokes into InputEvents: plain chars, CSI sequences (POSIX)
    and msvcrt scan codes (Windows), so both platforms deliver the same
    symbolic `Key("left")` / `Key("pgup")`.

    A bare ESC is only known to be a bare ESC once nothing follows it, hence
    `flush()` — the reader calls it when its poll times out. Wave 2's edit mode
    needs Esc to mean cancel, so the distinction is not academic.
    """

    def __init__(self, scan_codes: bool = os.name == "nt"):
        # `scan_codes` off on POSIX: there a NUL is a normal (if odd) character,
        # and treating it as a lead byte would swallow the key after it.
        self.scan_codes = scan_codes
        self._buf = ""
        self._high = ""

    def feed(self, char: str) -> List[InputEvent]:
        if self._high:
            char, self._high = self._combine(self._high, char), ""
            if not char:
                return []
        elif "\ud800" <= char <= "\udbff":
            # msvcrt.getwch() hands over UTF-16 code UNITS, so anything outside
            # the BMP (an emoji in a pasted line) arrives as two lone surrogates.
            # Each half is unprintable on its own, so passing them through drops
            # the character silently — join them here, where the platform quirk
            # already lives, and every consumer keeps seeing one printable Key.
            self._high = char
            return []
        if self._buf in _WINDOWS_LEAD and self._buf:   # lead byte + scan code
            name = WINDOWS_KEYS.get(char)
            self._buf = ""
            return [Key(name)] if name else []
        if not self._buf:
            if self.scan_codes and char in _WINDOWS_LEAD:
                self._buf = char
                return []
            if char == "\x1b":
                self._buf = char
                return []
            return [Key(char)]
        self._buf += char
        if len(self._buf) == 2 and char not in "[O":
            buf, self._buf = self._buf, ""
            return [Key("\x1b"), Key(buf[1])]
        if len(self._buf) >= 3 and "@" <= char <= "~":
            buf, self._buf = self._buf, ""
            event = decode_escape(buf)
            return [event] if event is not None else []
        return []

    @staticmethod
    def _combine(high: str, low: str) -> str:
        """One character from a UTF-16 surrogate pair, or "" if it was not one."""
        try:
            return (high + low).encode("utf-16", "surrogatepass").decode("utf-16")
        except UnicodeError:
            return ""

    def flush(self) -> List[InputEvent]:
        self._high = ""      # a half pair that never completed is not a keypress
        if self._buf == "\x1b":
            self._buf = ""
            return [Key("\x1b")]
        return []


class TerminalInput(InputSource):
    """Raw-ish key reader: msvcrt on Windows, termios+select on POSIX.

    cbreak, never raw: ISIG stays on so Ctrl+C keeps behaving exactly as it does
    today. On Windows msvcrt hands '\\x03' over instead, so it is turned back
    into a KeyboardInterrupt on the main thread.
    """

    poll_seconds = 0.05

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._decoder = _EscapeDecoder()
        # (fd, saved termios attrs) while the tty is in cbreak, else None.
        self._tty_state: Optional[Tuple[int, object]] = None

    def usable(self) -> bool:
        try:
            return bool(self._stream) and self._stream.isatty()
        except Exception:
            return False

    def start(self, handler):
        if self._thread is not None or not self.usable():
            return
        self._stop.clear()
        target = self._run_windows if os.name == "nt" else self._run_posix
        self._thread = threading.Thread(target=self._guard, args=(target, handler),
                                        name="statusline-input", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)
        self.restore_tty()   # the thread normally did it; make sure of it here

    def restore_tty(self):
        """Take the tty back out of cbreak, from any thread, at most once."""
        state, self._tty_state = self._tty_state, None
        if state is None:
            return
        fd, saved = state
        try:
            import termios

            termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        except Exception:
            pass

    def _guard(self, target, handler):
        try:
            target(handler)
        except Exception:
            return  # a broken reader costs the keys, never the run

    def _emit(self, handler, char: str) -> None:
        if char == "\x03":  # Ctrl+C on the Windows path: keep the usual meaning
            import _thread

            _thread.interrupt_main()
            return
        for event in self._decoder.feed(char):
            handler(event)

    def _run_windows(self, handler):
        import msvcrt

        while not self._stop.is_set():
            if msvcrt.kbhit():
                self._emit(handler, msvcrt.getwch())
                continue
            for event in self._decoder.flush():
                handler(event)
            self._stop.wait(self.poll_seconds)

    def _run_posix(self, handler):
        import codecs
        import select
        import termios
        import tty

        fd = self._stream.fileno()
        saved = termios.tcgetattr(fd)
        # INCREMENTAL, not a plain .decode() per read: a paste arrives as a byte
        # stream cut at arbitrary offsets, and a read that ends mid-character
        # (every non-ASCII one is 2-4 bytes) would decode to U+FFFD and eat the
        # start of the next character with it. Keeping the decoder across reads
        # is what makes a pasted Russian sentence survive a chunk boundary.
        utf8 = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            tty.setcbreak(fd)  # cbreak, not raw: ISIG (Ctrl+C) stays enabled
            # Published so a signal handler / atexit can undo it: this thread is
            # a daemon, and a signal kills it without running the finally below.
            self._tty_state = (fd, saved)
            while not self._stop.is_set():
                ready, _, _ = select.select([fd], [], [], self.poll_seconds)
                if not ready:
                    for event in self._decoder.flush():
                        handler(event)
                    continue
                for char in utf8.decode(os.read(fd, 1024)):
                    self._emit(handler, char)
        finally:
            # Guaranteed restore: a terminal left in cbreak is unusable.
            self.restore_tty()
