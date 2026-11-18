"""Tie a child process tree's life to this Python process on Windows.

The job is set to KILL_ON_JOB_CLOSE, so the started process and every
descendant it starts die when the last job handle closes -- `close()`, or
Windows closing it when this process ends in any way, `TerminateProcess`
included, which no `finally` survives.

`start` creates the child suspended and resumes it only once it is in the job
(`win_job.start`), so no descendant can be started outside it, however early.

The job handle is deliberately non-inheritable: a child holding it would keep
its own job alive. Ordinary cancellation still uses the live Popen handle.
Off Windows `ChildJob` is a no-op: `handle` stays None and `start` is a plain
`Popen`.
"""
from __future__ import annotations

import os
import subprocess
from typing import Sequence

from . import win_job


class ChildJob:
    def __init__(self):
        self.handle = None
        if os.name == "nt":
            self.handle = win_job.create(None, win_job.KILL_ON_JOB_CLOSE)

    def start(self, argv: Sequence[str], popen=None, **options):
        """`popen(argv, **options)` inside the job; a child that cannot be put
        there is killed and the OSError raised. `popen` (None:
        `subprocess.Popen`) exists for tests."""
        if not self.handle:
            return (popen or subprocess.Popen)(argv, **options)
        return win_job.start(self.handle, argv, popen, **options)

    def close(self) -> None:
        if self.handle:
            win_job.kernel32().CloseHandle(self.handle)
            self.handle = None
