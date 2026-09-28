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

from llm_loop import termio

# A frame is recorded as the rows a reader sees, so a pin can compare text.
_SGR = re.compile(r"\x1b\[[0-9;]*m")

# How long an armed stall holds the painter if the pin never releases it, so a
# pin that fails by hanging still ends. Matches the waits of the pins using it.
STALL_LIMIT_S = 10.0


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
    Paints nothing and says it did.
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
        return True

    def paint(self, lines, *, reassert=False):
        return True

    def release(self):
        self._on = False


class RecordingTerminal(termio.Terminal):
    """A real Terminal writing into a StringIO, that remembers what it was asked.

    * `frames` — a queue of the frames painted, each a list of rows with the
      colour codes stripped: for a pin waiting on the painter thread;
    * `painted` — the same frames as a list, for a pin reading them after;
    * `writers` — the thread behind every reserve, release, title and paint,
      and `releases` the thread behind each release alone;
    * `arm_stall()` — the painter thread's next frame hangs until `unstall` is
      set (or `STALL_LIMIT_S` passes): a terminal write that does not come back.
      `stalled` is set once the painter is held.

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
        self._armed = threading.Event()
        self.stalled = threading.Event()
        self.unstall = threading.Event()

    def size(self):
        return self.columns, self.lines

    def arm_stall(self):
        self._armed.set()

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
        if (self._armed.is_set()
                and threading.current_thread().name == "statusline-paint"):
            self._armed.clear()
            self.stalled.set()
            self.unstall.wait(STALL_LIMIT_S)
        result = super().paint(lines, reassert=reassert)
        frame = [_SGR.sub("", line) for line in lines]
        self.painted.append(frame)
        self.frames.put(frame)
        return result
