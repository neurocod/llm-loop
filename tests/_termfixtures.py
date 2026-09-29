"""Terminal and key-source doubles for the status-line pins.

Imported by name (`from _termfixtures import ...`), like `_runfixtures`. One copy
of each, because they stand in for a contract — `termio.InputSource.start`,
`Terminal.reserve`/`paint`/`release` — and a copy nobody updated when that
contract moved went on passing against the old one.
"""

import io
import queue
import re
import threading
import time

from llm_loop import termio
from llm_loop.statusline import PAINTER_THREAD_NAME

# A frame is recorded as the rows a reader sees, so a pin can compare text.
_SGR = re.compile(r"\x1b\[[0-9;]*m")

# How long `RecordingTerminal.wait_for_frame` waits for the frame a pin expects.
# A healthy painter shows it within a frame or two, so only a broken one gets
# near this: the longest of the 21 waits over three runs of the eight painter
# and runner test files was 0.047 s, measured 2026-09-29. Kept at the larger of
# the two budgets it replaces (10 s and 15 s), since a red that means no
# defect costs more than a slow one that does.
FRAME_WAIT_S = 15.0


class KeysByHand(termio.NullInputSource):
    """An input source whose handler the test calls: the key reader's seat."""

    def start(self, handler):
        self.handler = handler


class LiveTerminal(termio.Terminal):
    """Active from reserve() on, with no screen behind it.

    What the pins using it are about happens once the app is enabled — the
    parallel runner's stop grace, `push_quotas` — which a NullTerminal (what a
    test process without a tty gets: off always, and short-circuited by
    `StatusApp.start()`) cannot express, and a real Terminal needs a tty for.
    Paints nothing, and answers as the real one would: a frame is accepted
    only while reserved, and after release() no title is written.
    """

    def __init__(self):
        super().__init__(stream=io.StringIO())
        self._on = False

    @property
    def active(self):
        return self._on

    def size(self):
        return (120, 30)

    def reserve(self, rows):
        self._on = True
        self._released = False
        return True

    def paint(self, lines, *, reassert=False):
        return self._on

    def release(self):
        self._on = False
        self._released = True


class RecordingTerminal(termio.Terminal):
    """A real Terminal writing into a StringIO, that remembers what it was asked.

    * `frames` — a queue of the frames painted, each a list of rows with the
      colour codes stripped: for a pin waiting on the painter thread. Only the
      frames the real `paint` accepted — one refused (no region reserved) never
      reached a screen, so a pin must not read it as shown;
      A pin waits on it through `wait_for_frame`;
    * `painted` — the same frames as a list, for a pin reading them after;
    * `writers` — the thread behind every reserve, release, title and paint
      call, refused or not, and `releases` the thread behind each release alone;
    * `arm_stall(limit)` — the painter thread's next frame hangs until `unstall`
      is set (or `limit` seconds pass, so a pin that fails by hanging still
      ends): a terminal write that does not come back. `stalled` is set once
      the painter is held.

    Its size is `columns` x `lines`; set `columns` mid-test to stage a resize.
    The real class does the writing, so its geometry rules (`MIN_COLUMNS`, a
    region smaller than the screen) apply here as they do on a console.
    """

    def __init__(self, columns=80, lines=30):
        super().__init__(io.StringIO())
        self.columns = columns
        self.lines = lines
        self.frames = queue.Queue()
        self.painted = []
        self.writers = []
        self.releases = []
        self._stall_limit = None        # seconds; set = armed
        self.stalled = threading.Event()
        self.unstall = threading.Event()

    def size(self):
        return self.columns, self.lines

    def arm_stall(self, limit):
        self._stall_limit = limit

    def wait_for_frame(self, predicate, timeout=FRAME_WAIT_S):
        """Return the first accepted frame `predicate(frame)` holds for.

        Takes every frame before it off `frames`, so a second call waits for a
        LATER frame. Past `timeout` the pin fails here, naming the wait.
        """
        deadline = time.monotonic() + timeout
        while True:
            try:
                frame = self.frames.get(
                    timeout=max(0, deadline - time.monotonic()))
            except queue.Empty:
                raise AssertionError(
                    f"no painted frame matched within {timeout:g} s") from None
            if predicate(frame):
                return frame

    def reserve(self, rows):
        self.writers.append(threading.current_thread().name)
        return super().reserve(rows)

    def release(self):
        self.writers.append(threading.current_thread().name)
        self.releases.append(threading.current_thread().name)
        return super().release()

    def set_title(self, text, *, reassert=False):
        self.writers.append(threading.current_thread().name)
        return super().set_title(text, reassert=reassert)

    def paint(self, lines, *, reassert=False):
        self.writers.append(threading.current_thread().name)
        limit = self._stall_limit
        if (limit is not None
                and threading.current_thread().name == PAINTER_THREAD_NAME):
            self._stall_limit = None
            self.stalled.set()
            self.unstall.wait(limit)
        accepted = super().paint(lines, reassert=reassert)
        if accepted:
            frame = [_SGR.sub("", line) for line in lines]
            self.painted.append(frame)
            self.frames.put(frame)
        return accepted
