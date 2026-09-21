"""Run-local state breakpoints shared by the console and sequential runner."""

import threading
from typing import Callable, Optional, Tuple


class Breakpoints:
    """Stop before running a named state, after the in-flight agent finishes.

    The console adds names; the runner checks them at iteration boundaries and
    during idle waits. Names match the whole state, case-insensitively like
    StateFileDriver.state_name(). Nothing is persisted across runner calls.
    """

    def __init__(self, state_name: Callable[[], str]):
        self._state_name = state_name
        self._names = []
        self._lock = threading.Lock()

    @property
    def names(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._names)

    def add(self, text: str) -> Tuple[str, ...]:
        """Append nonempty, trimmed pipe-separated names; return the full list."""
        names = [part.strip() for part in text.split("|") if part.strip()]
        with self._lock:
            self._names.extend(names)
            return tuple(self._names)

    def reached(self) -> Optional[str]:
        names = self.names
        if not names:
            return None
        state = self._state_name().strip().lower()
        return next((name for name in names if name.lower() == state), None)
