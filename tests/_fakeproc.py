"""The plumbing of a stand-in app-server process shared by the transport and
quota-source tests: stdin is recorded, stdout replays JSON lines.

How the process exits is each test's own (`returncode`, `terminate`, ...),
so it is left to the subclass."""

import io
import json


class RecordedInput(io.StringIO):
    """stdin that survives `close`: the test reads what was sent afterwards,
    and `was_closed` says whether the client closed it."""

    was_closed = False

    def close(self):
        self.was_closed = True


class JsonLinesProcess:
    """stdout replays `items` — a dict as one JSON line, a str as written."""

    def __init__(self, *items):
        self.stdin = RecordedInput()
        self.stdout = io.StringIO("".join(
            item if isinstance(item, str) else json.dumps(item) + "\n"
            for item in items))

    def sent(self):
        """Every JSON line the client wrote to stdin, in order."""
        return [json.loads(line) for line in self.stdin.getvalue().splitlines()]
