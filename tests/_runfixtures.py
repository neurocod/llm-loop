"""What the runner pins hand `run_loop` / `run_parallel`: arguments, drivers, a
quota policy, the two "provably elsewhere" project roots — and `isolated_run`,
which keeps what such a run leaves in the process inside the test.

Imported by name (`from _runfixtures import ...`), never through an autouse
`conftest.py`: a pin that reads a helper it can see explains itself, and one that
leans on an invisible fixture does not.

The argument builders are the reason this module exists. `runlifecycle.begin_run`
reads the whole command-line contract, so a namespace written out by hand in each
test file broke as a set — eleven of them — every time the contract grew an
attribute, and the copies had already drifted apart on which attributes they
carried. These take the namespace from the runner's own parser instead, so a new
option reaches every pin the day it reaches `clispec`.
"""

import argparse
import atexit
import logging
import os
import sys
import threading
from contextlib import contextmanager

from llm_loop import (clispec, console, exitlog, projectroot, providers,
                      runlifecycle)
from llm_loop.agentwork import ClaudeCommand, Driver
from llm_loop.drivers import ListFileDriver


@contextmanager
def isolated_run(monkeypatch, tmp_path):
    """Keep what a staged run leaves in the PROCESS inside the test that ran it.

    A run that is not dry opens an exit record and raises the tee, and both are
    process-wide: the record sits in `console.LOG_DIR` — the operator's real log
    dir under the home directory — with its closing line registered on `atexit`,
    and `begin` is idempotent per process, so the next unisolated run inherits
    it. Inside this block the log dir is `tmp_path / "logs"` (yielded) and the
    record slot starts empty; on the way out the record is finished, and the
    streams the tee wrapped, the project root the run anchored, the mirror
    logger and the live-message transport `begin_run` set are put back.

    Every mirror handler is closed on the way in AND out, whatever `app_name`
    the run used, so a caller cannot name the wrong one: a mirror left open
    holds its file, which on Windows keeps the test's directory from being
    removed.

    The mirror loggers' `propagate` and level are put back too
    (`_logger_settings`): `setup_file_logging` turns propagation off, and pytest
    hangs its capture handler on every non-propagating logger it finds at the
    start of each phase, so a logger left that way is captured in every later
    test that never ran it. Handler lists are NOT restored wholesale: pytest's
    handler of the phase the snapshot was taken in is gone by the teardown, and
    putting it back would leave a stale capture handler behind.

    Called explicitly, from a thin autouse fixture in each file that stages runs;
    `conftest.py` fails any test that leaves a record open, the root moved or
    a stream replaced without it.
    """
    logs = tmp_path / "logs"
    monkeypatch.setattr(console, "LOG_DIR", logs)
    monkeypatch.setattr(console, "_FILE_LOGGER", console._FILE_LOGGER)
    monkeypatch.setattr(providers, "_LIVE_MESSAGES", providers._LIVE_MESSAGES)
    monkeypatch.setattr(exitlog, "_record", None)
    root = projectroot.project_dir()
    streams = sys.stdout, sys.stderr
    settings = _logger_settings()
    _close_mirror_handlers()
    try:
        yield logs
    finally:
        finish_record()
        sys.stdout, sys.stderr = streams
        projectroot.set_project_root(root)
        _close_mirror_handlers()
        _restore_logger_settings(settings)


def finish_record(reason=None):
    """Finish the process's open exit record, if any, and take its closing line
    off `atexit` — else it would print again at interpreter exit for a run that
    ended long ago. Returns the record it finished (None when there was none);
    `exitlog._record` is left for the caller, who knows what belongs there."""
    record = exitlog.current()
    if record is not None:
        atexit.unregister(record.finish)
        record.finish(reason)
    return record


def _registered_loggers():
    """Every logger in the registry, read as it is: `logging.getLogger` on a
    placeholder name would create one."""
    return [logger for logger in list(logging.Logger.manager.loggerDict.values())
            if isinstance(logger, logging.Logger)]


def _close_mirror_handlers():
    """Close and detach every mirror handler (`console._MirrorLogHandler`), on
    whichever logger holds it."""
    for logger in _registered_loggers():
        for handler in [handler for handler in logger.handlers
                        if isinstance(handler, console._MirrorLogHandler)]:
            handler.close()
            logger.removeHandler(handler)


# The loggers `console.setup_file_logging` configures: "runCycle.<app_name>".
_MIRROR_LOGGER_PREFIX = "runCycle."


def _logger_settings() -> dict:
    """`propagate` and level of each mirror logger, by name (`isolated_run`)."""
    return {logger.name: (logger.propagate, logger.level)
            for logger in _registered_loggers()
            if logger.name.startswith(_MIRROR_LOGGER_PREFIX)}


def _restore_logger_settings(settings: dict) -> None:
    """Put `_logger_settings`' snapshot back; a mirror logger created since gets
    a new logger's own (propagating, NOTSET), since the registry cannot forget
    it."""
    for logger in _registered_loggers():
        if logger.name.startswith(_MIRROR_LOGGER_PREFIX):
            logger.propagate, level = settings.get(logger.name,
                                                   (True, logging.NOTSET))
            logger.setLevel(level)


def capture_run_context(monkeypatch) -> dict:
    """Hand the pin the run's own `RunContext` as `made["ctx"]`, once the
    shared prologue (`runlifecycle.begin_run`) has built it.

    The seam for a pin that edits a knob mid-run: `ctx.registry` is the one
    registry whose setters write the run's `RunSettings`, so an edit through it
    is an edit the runner has to see. Wrapped rather than replaced, so the run
    goes through the real prologue.
    """
    made = {}
    real_begin_run = runlifecycle.begin_run     # captured before the patch

    def _begin_run(*args, **kwargs):
        made["ctx"] = real_begin_run(*args, **kwargs)
        return made["ctx"]

    monkeypatch.setattr(runlifecycle, "begin_run", _begin_run)
    return made


def record_exit_pushes(monkeypatch) -> list:
    """Record every call to the EXIT push, and stop it reaching git.

    Returns the list the calls land in, as `(policy, project_dir)` pairs.
    Replaced on `runlifecycle`, not on `gitpush`: the epilogue imported the name,
    so that is the binding its call resolves — patching the owner would leave the
    real push running and the recorder empty.
    """
    calls = []
    monkeypatch.setattr(runlifecycle, "final_git_push",
                        lambda policy, project_dir, abort=None: calls.append(
                            (policy, project_dir)))
    return calls


class StubSource:
    """Stands in for a UsageSource without an endpoint behind it.

    Only the two calls a closing run makes are answered; the status area is a
    Null object under `--no-statusline`, so nothing else reaches for it.
    """

    def get_usage(self):
        return None

    def invalidate(self):
        pass


def _parsed(mode: str, project_dir, fields: dict) -> argparse.Namespace:
    """`mode`'s parser run on an empty argv, then `project_dir` and `fields` set.

    A field the parser does not declare is refused rather than set: the runners
    read exactly what the parser produces, so a misspelt override would build an
    attribute no code reads and leave the pin testing the default it meant to
    change.
    """
    args = clispec.build_parser(mode, prog="pytest").parse_args([])
    unknown = sorted(set(fields) - set(vars(args)))
    if unknown:
        raise TypeError(f"the {mode} parser declares no {', '.join(unknown)}")
    args.project_dir = str(project_dir)
    for name, value in fields.items():
        setattr(args, name, value)
    return args


def seq_args(project_dir, **fields) -> argparse.Namespace:
    """Arguments of a sequential run, as `cyclecore.parse_args` would build them.

    One departure from the command line's defaults: `git_push` is "none". The
    parser's default policy pushes, and the loop pushes at the top of every pass,
    so a pin that did not name a policy would run `git push` against its tmp dir —
    harmless there and a real push the day a tmp dir sits inside a repository
    with an upstream. A pin about pushing names its policy.
    """
    fields.setdefault("git_push", "none")
    return _parsed(clispec.SEQUENTIAL, project_dir, fields)


def par_args(project_dir, *, jobs: int, **fields) -> argparse.Namespace:
    """Arguments of a parallel run, as `parallel.parse_args` would build them.

    `jobs` is required: the parser's default leaves the width to the driver, and
    through it to `clispec.DEFAULT_JOBS` workers, which no pin here is about.
    Besides `git_push` (see `seq_args`), `ignore_usage` defaults on — a pin that
    wants the closing usage snapshot turns it off and supplies a source itself.
    """
    fields.setdefault("git_push", "none")
    fields.setdefault("ignore_usage", True)
    return _parsed(clispec.PARALLEL, project_dir, dict(fields, jobs=jobs))


class StubPolicy:
    """A LimitPolicy that never reads the usage report and never pauses.

    `log_snapshot` RECORDS the call instead of doing nothing: the closing
    snapshot is one of the steps `test_abnormal_exit_epilogue` pins, and a stub
    that swallowed it would leave that step unpinned while looking pinned.
    `logged` holds each call whole — `(source, label, cache_value)`, for the pins
    about which source was read and whether fresh — and `snapshots` the labels
    alone, for the pins about which snapshots were taken.
    """

    def __init__(self):
        self.logged = []

    @property
    def snapshots(self):
        return [label for _source, label, _cache_value in self.logged]

    def describe(self):
        return "stub"

    def log_snapshot(self, source, label="", cache_value=True):
        self.logged.append((source, label, cache_value))

    def check_and_wait(self, source, session_start, note="",
                       cache_value=True, should_stop=None):
        return False, session_start


class NoWorkDriver(Driver):
    """A driver whose queue is empty from the start."""

    def __init__(self):
        self.limit_policy = StubPolicy()

    def next_command(self):
        return None


class OneShotDriver(NoWorkDriver):
    """Hands out a single command, then reports the work exhausted.

    `served` counts the commands handed out, which is how a pin tells "the loop
    ran" from "the loop reported and returned".
    """

    def __init__(self):
        super().__init__()
        self.served = 0

    def next_command(self):
        if self.served:
            return None
        self.served += 1
        return ClaudeCommand("do the thing", "", "the-thing")


class MemListDriver(ListFileDriver):
    """ListFileDriver backed by an in-memory list (no files, no real provider).

    Walked in list order so a pin can name the item that goes first, and locked
    because the parallel runner claims and strikes from several workers.

    A parallel pin needs at least one item: a run with nothing pending reports
    "nothing to do" and returns before its exit push, its closing snapshot and
    everything else a pin could look at.
    """

    target_suffix = ".out.md"
    pick_order = "list"

    def __init__(self, items):
        super().__init__()
        self._items = list(items)
        self._lock = threading.Lock()
        self.limit_policy = StubPolicy()

    def prompt(self, source, target):
        return f"do {os.path.basename(source)}"

    def pending_lines(self):
        with self._lock:
            return list(self._items)

    def strike(self, line):
        with self._lock:
            if line in self._items:
                self._items.remove(line)
                return True
            return False


def root_not_cwd(tmp_path) -> str:
    """A project root that is provably not the directory this process stands in.

    For the pins that follow a root through a handover (which directory git runs
    in): on a machine where pytest happened to run from tmp_path they would pass
    while proving nothing, so that case fails here instead.
    """
    root = os.path.abspath(str(tmp_path))
    assert os.path.normcase(root) != os.path.normcase(os.getcwd()), \
        "the project root and the process cwd are the same directory, so these " \
        "pins cannot tell a handed-over root from an ambient one"
    return root


def root_named_unlike_cwd(tmp_path) -> str:
    """A project root whose FOLDER NAME is provably not this process's own.

    Stronger than `root_not_cwd`, and not interchangeable with it: the mirror
    log's file name carries only the project folder's basename, so two different
    directories with one name are exactly what a pin about that name cannot tell
    apart.
    """
    project = tmp_path / "some-project"
    project.mkdir(exist_ok=True)
    root = os.path.abspath(str(project))
    assert os.path.normcase(os.path.basename(root)) != \
        os.path.normcase(os.path.basename(os.getcwd())), \
        "the project folder and the process cwd share a name, so this pin " \
        "cannot tell an anchored root from an ambient one"
    return root
