"""Lease CPU cores from the counter for one heavy child process.

    python -m llm_agents.cores run --want 8 --min 2 -- npm test
    python -m llm_agents.cores status
    python -m llm_agents.cores hold --want 3 [--min M] [--wait S] [--label L]
    python -m llm_agents.cores budget [N|off]

A test suite, a compiler and a probe batch started side by side each take
every core, none knowing of the others, and a timeout under that load stops
meaning a defect. `run` asks the core broker inside `AgentCounter`
(`extras/agent-counter/CoreBroker.h` has the rules: what "free" is, first
come first served, a waiter's patience) for up to `--want` cores, at least
`--min`, and starts the child with the answer K in `LLM_CORES` (`LEASE_ENV`).
Reading it is the child's business — or its wrapper's: this module knows no
test runner's or compiler's flags.

The lease is the CONNECTION, exactly as an agent's registration is
(`agent_counter`'s docstring): it is held for the child's whole life and
returned when this process exits, however it exits — a `taskkill /F` of this
wrapper closes the handle in the kernel, and the broker hands the cores to
the next waiter. There is no "release" message to forget. And the child goes
with it: `run` puts it in a kill-on-close job object (`Job`), so a wrapper
killed without `/T` does not leave a child burning K cores nobody holds.
The job closes armed after a normal exit too, unless the caller opts in to
keeping what a build leaves behind (`run`'s `keep_descendants`,
`--keep-descendants`).

Nested runs do not lease again: an inner `run` (one whose environment already
carries `LEASE_ENV`) takes min(want, outer K) of the OUTER lease. Several
inner runs started side by side therefore SHARE those K cores — the caller
that starts them in parallel divides K among them, as it would have divided
the machine. Leasing them from the broker under the outer budget would need
the outer wrapper to serve a sub-broker of its own while it waits on its
child, for a pattern no caller has.

Nothing here may fail the run it wraps. Without a broker — the counter not
built, `AGENT_COUNTER_ENABLED=0`, not Windows, a counter that refuses or
does not answer — the child runs exactly as it would have without this
wrapper (`LLM_CORES` left as it was, so its own defaults, all cores), and one
line on stderr says why. Its exit code is returned unchanged, always.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from llm_loop import win_job

from . import agent_counter, host_policy

# The variable the child reads its lease from: a whole number >= 1.
LEASE_ENV = "LLM_CORES"

# How long a request below its `min` waits before the broker grants `min`
# anyway, marked forced (CoreBroker::settle). The counter's own default for a
# request without `wait_s` is the same figure (Registry.cpp, kDefaultWaitS).
# Five minutes: longer than the suite-sized runs it waits behind usually
# take, short enough that a gate stuck behind a neighbour's hour-long batch
# runs over-committed instead of looking hung.
DEFAULT_WAIT_S = 300.0

# The broker answers a lease within its own tick (1 s) once it may grant; the
# client gives up on an answer only past the patience it asked for plus this,
# so a hung counter costs a run this much and then the no-broker fallback —
# never a wait for ever.
ANSWER_SLACK_S = 15.0
POLL_S = 0.05

WIN_ERROR_BROKEN_PIPE = 109

# The exit code of a run interrupted (Ctrl+C) before its child started: the
# shell convention for SIGINT, 128 + 2.
INTERRUPTED_EXIT = 130


class Job:
    """A Windows job object holding the leased child and its descendants.

    Two jobs in one: KILL_ON_JOB_CLOSE ends the child when the last handle
    goes — this wrapper's, and the broker's, which opens the job by `name`
    (sent in the lease) to read its CPU time and so tell the lease's own load
    from foreign load (`CoreBroker::addSample`). BREAKAWAY_OK lets a process
    meant to outlive the run leave it (`agent_counter._spawn`: a counter
    started from inside a leased run must not die with it).

    The child is started SUSPENDED and resumed once it is in the job
    (`win_job.start`), so no grandchild can be spawned outside it in between.
    """

    def __init__(self, name: str, handle: int) -> None:
        self.name = name
        self._handle = handle

    @classmethod
    def create(cls) -> "Job":
        """A new named job; OSError when Windows will not make one."""
        # Session-local and unguessable enough: the broker runs as the same
        # user in the same session, and only ever opens it for a query.
        name = f"Local\\llm-cores-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        return cls(name, win_job.create(name, win_job.KILL_ON_JOB_CLOSE | win_job.BREAKAWAY_OK))

    def start(self, argv: Sequence[str], env: dict, popen=subprocess.Popen):
        """Start `argv` inside the job. OSError from the start itself; a child
        that cannot be put in the job is killed and the error raised, so the
        caller can start it plainly instead."""
        return win_job.start(self._handle, argv, popen, env=env)

    def disarm(self) -> None:
        """Clear KILL_ON_JOB_CLOSE, keeping BREAKAWAY_OK, so closing the job
        afterwards leaves what still runs in it alone. For a child that ended
        NORMALLY and left processes running on purpose: MSBuild's reusable
        nodes, or an `mspdbsrv.exe` another slot's build may be talking to —
        killing those with the job breaks that build. OSError when Windows
        refuses (the job then stays armed)."""
        win_job.set_limits(self._handle, win_job.BREAKAWAY_OK)

    def terminate(self, code: int = INTERRUPTED_EXIT) -> None:
        win_job.kernel32().TerminateJobObject(self._handle, code)

    def close(self) -> None:
        if self._handle:
            win_job.kernel32().CloseHandle(self._handle)
            self._handle = 0


class Closed(Exception):
    """The counter closed the connection."""


class PipeTransport:
    """Line-delimited JSON over the counter's pipe handle, read WITHOUT blocking.

    A blocking `read` on the handle cannot be given a timeout, and reading on a
    helper thread is worse: Windows serializes synchronous I/O on one handle, so
    a read parked on a thread would also block the close at the end. So the
    pipe is peeked (`PeekNamedPipe`) and only what is already there is read.
    """

    def __init__(self, handle) -> None:
        self._handle = handle
        self._pending = b""

    def send(self, message: dict) -> None:
        self._handle.write(json.dumps(message).encode("utf-8") + b"\n")

    def _available(self) -> int:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        avail = wintypes.DWORD(0)
        os_handle = msvcrt.get_osfhandle(self._handle.fileno())
        ok = ctypes.windll.kernel32.PeekNamedPipe(
            wintypes.HANDLE(os_handle), None, 0, None, ctypes.byref(avail), None)
        if not ok:
            raise Closed(f"peek failed (winerror {ctypes.GetLastError()})")
        return avail.value

    def poll(self) -> Optional[dict]:
        """One complete line as a dict, or None if none has arrived yet."""
        if b"\n" not in self._pending:
            count = self._available()
            if count:
                data = self._handle.read(count)
                if not data:
                    raise Closed("end of stream")
                self._pending += data
        if b"\n" not in self._pending:
            return None
        line, self._pending = self._pending.split(b"\n", 1)
        message = json.loads(line)
        if not isinstance(message, dict):
            raise ValueError(f"not a JSON object: {line!r}")
        return message

    def close(self) -> None:
        try:
            self._handle.close()
        except OSError:
            pass


def _say(line: str) -> None:
    try:
        print(f"cores: {line}", file=sys.stderr, flush=True)
    except Exception:  # noqa: BLE001 - a report must not become the failure
        pass


def _machine_cores() -> int:
    return os.cpu_count() or 1


def negotiate(transport, want: int, minimum: int, label: str, wait_s: float, *,
              job: Optional[str] = None,
              say: Callable[[str], None] = _say,
              clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep) -> tuple[Optional[int], str]:
    """Ask for the lease and wait for its answer: (K, line) on a grant,
    (None, reason) when there is no lease to be had. A "queued" notice is
    said at once — the run is about to sit silent for up to `wait_s`.
    `job`: the name of the job object the child will run in (`Job`)."""
    request = {"op": "lease", "want": want, "min": minimum, "label": label, "wait_s": wait_s}
    if job is not None:
        request["job"] = job
    transport.send(request)
    deadline = clock() + wait_s + ANSWER_SLACK_S
    while True:
        try:
            message = transport.poll()
        except Closed:
            return None, "the counter closed the connection without a grant (a build without leases?)"
        if message is None:
            if clock() >= deadline:
                return None, f"no grant within {wait_s + ANSWER_SLACK_S:.0f} s"
            sleep(POLL_S)
            continue
        op = message.get("op")
        if op == "queued":
            say(f"waiting for {minimum} core(s) for {label} (free {message.get('free')} of "
                f"{message.get('total')}, {message.get('ahead')} ahead), up to {wait_s:.0f} s")
            continue
        if op == "error":
            got = f", {message['got']}" if "got" in message else ""
            return None, f"the counter refused the lease ({message.get('error')}{got})"
        cores = message.get("cores")
        if op != "grant" or not isinstance(cores, int) or isinstance(cores, bool) or cores < 1:
            return None, f"unexpected answer from the counter: {message!r}"
        total = message.get("total")
        # The broker forces only a head whose min does not fit (`settle`), so
        # a forced grant is always over the FREE cores — not necessarily over
        # the machine: the shortfall may be someone else's load.
        unmeasured = ("; its load is not measured (the counter could not open its job)"
                      if job is not None and message.get("measured") is False else "")
        if message.get("forced"):
            return cores, (f"waited {message.get('waited_s', 0):.0f} s; {label} runs with its "
                           f"min {cores} anyway, over the {message.get('free')} free of "
                           f"{total}{unmeasured}")
        return cores, (f"{cores} of {total} for {label} (asked {want}, min {minimum}; "
                       f"free {message.get('free')}, foreign {message.get('foreign')}){unmeasured}")


def _inherited() -> Optional[int]:
    raw = os.environ.get(LEASE_ENV, "").strip()
    return int(raw) if raw.isdigit() and int(raw) >= 1 else None


@contextmanager
def lease(want: int, minimum: int = 1, label: str = "", wait_s: float = DEFAULT_WAIT_S, *,
          job: Optional[str] = None,
          say: Callable[[str], None] = _say,
          connect: Optional[Callable[[Callable[[str], None]], Optional[object]]] = None,
          transport_of: Callable[[object], object] = PipeTransport) -> Iterator[Optional[int]]:
    """Hold a lease for the block: yields K, or None for "no broker, use the
    machine's defaults". Says exactly one line, plus a "waiting" one if queued.

    Inside a lease already (`LEASE_ENV` set by an outer `run`), no new one is
    asked: the outer lease already counts these cores, and a second would
    count them twice. The inner run takes min(want, outer) — of the outer's
    K, shared with any inner run beside it (the module docstring).

    `job`: the job object the block's work runs in (`Job.name`), so the
    broker measures its load; without one the lease is assumed to burn all
    it holds. A Ctrl+C while waiting for the grant returns nothing and
    propagates: there is no lease, and nothing has started.

    `connect(complain)` returns the counter's handle or None after one
    complaint (default: `agent_counter._open_or_start`, which starts the
    counter if nobody listens); `transport_of` wraps it. Both exist for tests.
    """
    label = (label or "lease")[:agent_counter.MAX_LABEL_CHARS]
    outer = _inherited()
    if outer is not None:
        cores = min(want, outer)
        say(f"inside a lease of {outer}: {label} takes {cores} of it without a new one "
            f"(inner runs side by side share it)")
        yield cores
        return

    def fallback(reason: str) -> str:
        return (f"no core broker ({reason}); {label} runs with its own defaults "
                f"({_machine_cores()} cores on this machine)")

    if os.name != "nt":
        say(fallback("the broker speaks a Windows named pipe"))
        yield None
        return
    if os.environ.get(agent_counter.DISABLE_ENV, "").strip() == "0":
        say(fallback(f"{agent_counter.DISABLE_ENV}=0"))
        yield None
        return
    reasons: list[str] = []
    transport = None
    cores: Optional[int] = None
    try:
        handle = (connect or agent_counter._open_or_start)(reasons.append)
        if handle is None:
            say(fallback(reasons[0] if reasons else "no counter"))
        else:
            transport = transport_of(handle)
            cores, line = negotiate(transport, want, minimum, label, wait_s, job=job, say=say)
            say(line if cores is not None else fallback(line))
    except KeyboardInterrupt:
        if transport is not None:
            transport.close()
        raise
    except Exception as exc:  # noqa: BLE001 - see the docstring: the run goes on
        cores = None
        say(fallback(f"{type(exc).__name__}: {exc}"))
    if cores is None and transport is not None:
        # No grant, so nothing to hold — and a request the broker still
        # queues would be granted LATE, holding cores for a run that is
        # already going on without them.
        transport.close()
        transport = None
    try:
        yield cores
    finally:
        # Returning the lease is this close and nothing else.
        if transport is not None:
            transport.close()


def _create_job(say: Callable[[str], None]) -> Optional[Job]:
    """The run's job, or None (one line) when Windows will not make one."""
    if os.name != "nt" or _inherited() is not None:
        return None  # nested: the child is in the outer run's job already
    try:
        return Job.create()
    except OSError as exc:
        say(f"no job object for the child ({exc}): it is not killed with this wrapper, "
            f"and the broker cannot measure its load")
        return None


def _wait(child, job: Optional[Job], say: Callable[[str], None]) -> tuple[int, bool]:
    """(The child's exit code, whether THIS wrapper left it alone — no second
    Ctrl+C ended it; a crash or a foreign kill also reads True), through any
    number of Ctrl+C. The first is the child's to answer (the console
    delivered it there too); a second ends it — the whole job — and the wait
    goes on until it has been reaped, because returning first would release
    the lease under a child still running."""
    interrupts = 0
    while True:
        try:
            return child.wait(), interrupts < 2
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts == 1:
                say("interrupted: waiting for the child to stop (Ctrl+C again ends it)")
                continue
            if interrupts == 2:
                say("interrupted again: ending the child")
                if job is not None:
                    job.terminate()
                else:
                    child.kill()


def run(command: Sequence[str], want: int, minimum: int = 1, label: str = "",
        wait_s: float = DEFAULT_WAIT_S, *, popen=subprocess.Popen,
        make_job: Optional[Callable[[Callable[[str], None]], Optional[Job]]] = None,
        keep_descendants: bool = False, **lease_options) -> int:
    """Run `command` under a lease; return its exit code unchanged, or
    INTERRUPTED_EXIT for a Ctrl+C before it started. `popen`/`make_job`
    exist for tests.

    By default the job closes ARMED however the child ends: what it left
    running dies with it, so a test runner's stray worker or a crashed
    root's descendants do not keep burning cores nobody holds.
    `keep_descendants` (opt-in, for a build) disarms it (`Job.disarm`) once
    the child has exited without this wrapper ending it — with any exit
    code: MSBuild's reusable nodes and `mspdbsrv` serve the next build
    whether this one compiled or not. The price, named: a build root killed
    from outside reads the same as a failed one, and its leftovers are kept.
    """
    label = label or " ".join(command)
    say = lease_options.get("say", _say)
    job = (make_job or _create_job)(say)
    try:
        try:
            with lease(want, minimum, label, wait_s, job=job.name if job else None,
                       **lease_options) as cores:
                env = dict(os.environ)
                if cores is not None:
                    env[LEASE_ENV] = str(cores)
                # `npm` is `npm.cmd` on Windows: CreateProcess needs the full name.
                argv = [shutil.which(command[0]) or command[0], *command[1:]]
                # The child must NOT inherit the pipe handle (Python's handles
                # are non-inheritable, PEP 446, and nothing here passes it on):
                # a child holding it would keep the lease alive after this
                # wrapper is killed.
                child = None
                if job is not None and cores is not None:
                    try:
                        child = job.start(argv, env, popen)
                    except OSError as exc:
                        say(f"cannot put {command[0]} in a job ({exc}); started plainly")
                if child is None:
                    try:
                        child = popen(argv, env=env)
                    except OSError as exc:
                        say(f"cannot start {command[0]}: {exc}")
                        return 127
                code, left_alone = _wait(child, job if cores is not None else None, say)
                if job is not None and keep_descendants and left_alone:
                    try:
                        job.disarm()
                    except OSError as exc:
                        say(f"the job of {label} could not be disarmed ({exc}): what it "
                            f"left running ends now")
                return code
        except KeyboardInterrupt:
            say(f"interrupted while waiting for cores; {label} did not start")
            return INTERRUPTED_EXIT
    finally:
        if job is not None:
            job.close()


def hold(want: int, minimum: int = 1, label: str = "hold", wait_s: float = DEFAULT_WAIT_S, *,
         stdin=None, stdout=None, **lease_options) -> int:
    """Hold a lease until `stdin` reaches end of file, for a client that is not
    Python and cannot keep the broker's pipe itself (a node script): the lease
    is taken in THIS process, ONE stdout line `LLM_CORES=<K>` answers it
    (`LLM_CORES=` without a lease; the `cores: ...` line on stderr says why),
    and it is held until the parent closes the other end of the pipe — which
    is the parent's exit, however it exits. No job object: the work is the
    parent's, not this process's, so the broker counts the lease as burning
    all it holds.

    From the answer on, SIGINT is ignored: the lease's end is the parent's
    (stdin's EOF), not the console's — a Ctrl+C reaches this process too, and
    returning the lease on it would hand the cores away while the parent still
    tears down. A Ctrl+C while still waiting for the grant propagates, as in
    `lease`. `stdin` (bytes) / `stdout` default to this process's own;
    `lease_options` go to `lease` (tests)."""
    stdin = sys.stdin.buffer if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    previous = None
    try:
        with lease(want, minimum, label, wait_s, **lease_options) as granted:
            print(f"{LEASE_ENV}={granted or ''}", file=stdout, flush=True)
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stdin.read()
            except OSError:
                pass
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
    return 0


def status() -> int:
    """Print the counter's core figures as JSON; never starts a counter."""
    if os.name != "nt":
        _say("no counter off Windows")
        return 1
    handle = agent_counter._connect()
    if handle is None:
        _say("no counter is running")
        return 1
    transport = PipeTransport(handle)
    try:
        transport.send({"op": "status"})
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            message = transport.poll()
            if message is not None:
                print(json.dumps(message.get("cores", {}), indent=1))
                return 0 if "cores" in message else 1
            time.sleep(POLL_S)
        _say("the counter did not answer")
        return 1
    except (Closed, OSError, ValueError) as exc:
        _say(f"status failed: {exc}")
        return 1
    finally:
        transport.close()


def _ask(transport, message: dict, *, clock: Callable[[], float] = time.monotonic,
         sleep: Callable[[float], None] = time.sleep) -> Optional[dict]:
    """Send one line and return the first answer, or None within 5 s."""
    transport.send(message)
    deadline = clock() + 5.0
    while clock() < deadline:
        answer = transport.poll()
        if answer is not None:
            return answer
        sleep(POLL_S)
    return None


def budget(cores: Optional[int], *, change: bool = True,
           connect: Optional[Callable[[], Optional[object]]] = None,
           transport_of: Callable[[object], object] = PipeTransport,
           say: Callable[[str], None] = _say, out=None) -> int:
    """Set (`cores` >= 1), lift (`cores` None) or, with `change=False`, only
    read the broker's budget: a ceiling on all leases together
    (`CoreBroker::setBudget` has the rule). Prints one line, returns 0 when
    the counter answered.

    Setting one STARTS a counter if none runs — the budget lives in that
    process, and the counter does not retire while one is set. Lifting or
    reading never starts one: no counter means no budget."""
    out = sys.stdout if out is None else out
    if connect is not None:
        handle = connect()
    elif os.name != "nt":
        say("no counter off Windows: no budget")
        return 1
    elif change and cores is not None:
        reasons: list[str] = []
        handle = agent_counter._open_or_start(reasons.append)
        if handle is None:
            say(f"no counter to hold the budget ({reasons[0] if reasons else 'none started'})")
            return 1
    else:
        handle = agent_counter._connect()
    if handle is None:
        print("budget: none (no counter is running)", file=out, flush=True)
        return 0
    transport = transport_of(handle)
    try:
        message: dict = {"op": "budget"}
        if change:
            message["cores"] = cores
        answer = _ask(transport, message)
    except (Closed, OSError, ValueError) as exc:
        say(f"budget failed: {exc}")
        return 1
    finally:
        transport.close()
    if answer is None:
        say("the counter did not answer the budget")
        return 1
    if answer.get("op") == "error":
        if answer.get("error") == "unknown-op":
            say("the running counter predates budgets: let it retire or `AgentCounter --quit` it, "
                "then ask again")
        else:
            say(f"the counter refused the budget ({answer.get('error')}: {answer.get('detail', '')})")
        return 1
    if answer.get("op") != "budget":
        say(f"unexpected answer from the counter: {answer!r}")
        return 1
    held = answer.get("budget")
    print(f"budget: {held if held is not None else 'none'}; leases share {answer.get('total')} "
          f"of {answer.get('machine')} cores, {answer.get('leased')} leased now", file=out, flush=True)
    return 0


def _budget_arg(raw: str) -> Optional[int]:
    if raw == "off":
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise argparse.ArgumentTypeError("a whole number of cores >= 1, or off")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m llm_agents.cores", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    go = sub.add_parser("run", help="run a command under a core lease")
    go.add_argument("--want", type=int, required=True, help="cores the command can use")
    go.add_argument("--min", type=int, default=1, dest="minimum",
                    help="fewer than this is not worth running with: wait for it (default 1)")
    go.add_argument("--label", default="", help="the name the badge's status shows (default: the command)")
    go.add_argument("--wait", type=float, default=DEFAULT_WAIT_S, dest="wait_s",
                    help=f"seconds to wait for --min before running with it anyway (default {DEFAULT_WAIT_S:.0f})")
    go.add_argument("--keep-descendants", action="store_true",
                    help="a build: what the command leaves running once it has exited "
                         "(MSBuild nodes, mspdbsrv) is kept, not killed with the job")
    go.add_argument("command", nargs=argparse.REMAINDER, help="-- then the command")
    sub.add_parser("status", help="print the broker's leases, queue and load as JSON")
    keep = sub.add_parser("hold", help="hold a core lease for a non-Python parent until stdin closes")
    keep.add_argument("--want", type=int, required=True, help="cores the parent can use")
    keep.add_argument("--min", type=int, default=1, dest="minimum",
                      help="fewer than this is not worth running with: wait for it (default 1)")
    keep.add_argument("--label", default="hold", help="the name the badge's status shows")
    keep.add_argument("--wait", type=float, default=DEFAULT_WAIT_S, dest="wait_s",
                      help=f"seconds to wait for --min before holding it anyway (default {DEFAULT_WAIT_S:.0f})")
    cap = sub.add_parser("budget", help="cap all leases together at N cores, or lift the cap (off); "
                                        "without an argument print it")
    # Parsed in `main`, not by `type=`: Python before 3.11 runs the type on
    # an absent optional positional's default too, so no default can mean
    # "not given" there.
    cap.add_argument("cores", nargs="?", default=None, help="N >= 1, or off")
    return parser


def main(argv: Optional[Sequence[str]] = None, *, policy: Optional[host_policy.HostPolicy] = None,
         root: Optional[Path] = None) -> int:
    """`policy`/`root`: a host's entry point names its counter executable
    this way (`HostPolicy.counter_exes`), as it does for the bridges; without
    one the counter is found through `AGENT_COUNTER_EXE`, or only if running."""
    parser = _parser()
    args = parser.parse_args(argv)
    with host_policy.installed(policy or host_policy.EMPTY, root):
        if args.action == "status":
            return status()
        if args.action == "budget":
            if args.cores is None:
                return budget(None, change=False)
            try:
                return budget(_budget_arg(args.cores))
            except argparse.ArgumentTypeError as exc:
                parser.error(f"budget: {exc}")
        if args.action == "hold":
            if args.want < 1 or not 1 <= args.minimum <= args.want or args.wait_s < 0:
                parser.error("hold: need --want >= 1, 1 <= --min <= --want, --wait >= 0")
            return hold(args.want, args.minimum, args.label, args.wait_s)
        command = list(args.command)
        if command and command[0] == "--":
            command = command[1:]
        if not command:
            parser.error("run: no command after --")
        if args.want < 1 or not 1 <= args.minimum <= args.want or args.wait_s < 0:
            parser.error("run: need --want >= 1, 1 <= --min <= --want, --wait >= 0")
        return run(command, args.want, args.minimum, args.label, args.wait_s,
                   keep_descendants=args.keep_descendants)


if __name__ == "__main__":
    raise SystemExit(main())
