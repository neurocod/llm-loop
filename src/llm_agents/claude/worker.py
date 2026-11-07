"""Detached queue consumer with an OS-released exclusive ownership lock."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional
import uuid

from .. import agent_counter

from .follow_up import answer_turn
from .protocol import child_environment
from .state import Store


# What `python <entry>` starts: this package's directory unless the host's
# entry point named its own (`cli.main(entry=...)`), so that the detached
# worker and the printed wait command run under the host that spawned them.
_entry: dict = {"path": None}


def entry_point() -> Path:
    """The directory or script `launch` runs with this interpreter."""
    return _entry["path"] or Path(__file__).resolve().parent


@contextmanager
def entered(path: Optional[Path]):
    """`entry_point()` is `path` (None: this package) for one block."""
    before = _entry["path"]
    _entry["path"] = Path(path).resolve() if path is not None else None
    try:
        yield
    finally:
        _entry["path"] = before


@contextmanager
def worker_lock(path: Path):
    """The lock's lifetime is the process, not a stale PID or a wall clock lease."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def launch(store: Store, name: str) -> None:
    agent = store.get(name)
    directory = store.root / agent["name"]
    directory.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, str(entry_point()),
            "--state-dir", str(store.root.resolve()), "_worker", agent["name"]]
    with (directory / "worker.log").open("ab") as log:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         cwd=agent["cwd"], env=child_environment(), close_fds=True,
                         **({"creationflags": subprocess.CREATE_NO_WINDOW |
                             subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                            else {"start_new_session": True}))


def work(store: Store, name: str) -> None:
    agent = store.get(name)
    name = agent["name"]
    owner = str(uuid.uuid4())
    with worker_lock(store.root / name / "worker.lock") as acquired:
        if not acquired:
            return
        store.recover(name)
        if store.claim(name, owner):
            try:
                while (turn := store.next_turn(name, owner)) is not None:
                    # Only the lock-owning worker's active turn is counted.
                    # The badge is outside Claude's child job, so it can serve
                    # both providers and outlive this particular review.
                    label = f"claude:{name}@{Path(agent['cwd']).name}"
                    with agent_counter.announce(label, agent_type="claude",
                                                disable_env="CLAUDE_AGENTS_COUNTER") as counter:
                        outcome = answer_turn(store, name, owner, turn,
                                              store.root / name / f"turn-{turn['turn']:04d}",
                                              on_week_used=counter.report_week_used)
                    store.finish(name, owner, turn["turn"], outcome)
            except BaseException as exc:
                store.fail_worker(name, owner, f"{type(exc).__name__}: {exc}")
                print(f"Worker failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    # Enqueue may have raced the final next_turn and a duplicate launcher may
    # have seen our lock. Inspect only AFTER releasing it to close that window.
    if store.get(name)["status"] == "queued":
        launch(store, name)
