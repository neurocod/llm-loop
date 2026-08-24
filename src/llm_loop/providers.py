"""LLM provider adapters used by the shared loop engine.

The loop lifecycle is provider-neutral. This module owns the small part that is
not: executable names, non-interactive flags, and command-line construction.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import os
import shutil
import subprocess
from typing import Optional, Protocol

from . import operator
from .operator import user_message_line


class AgentCommandLike(Protocol):
    prompt: str
    model: str
    sandbox_mode: str


# Whether a claude iteration is started with its stdin open for more user
# messages (`--input-format stream-json`), which is what lets the console send
# the agent a note mid-iteration. Off, the prompt travels in argv exactly as it
# did before and a note waits for the next iteration instead.
#
# A module-level switch rather than a parameter because it is a property of the
# RUN, and both runners plus every wrapper build their argv through the two
# functions below; the environment variable is the same escape hatch shape as
# LLM_LOOP_STATUSLINE, for a terminal or a CLI build where the transport
# misbehaves.
LIVE_MESSAGES_ENV = "LLM_LOOP_LIVE_MESSAGES"
_LIVE_MESSAGES = os.environ.get(LIVE_MESSAGES_ENV, "1") not in ("0", "false", "no")


def set_live_messages(enabled: bool) -> None:
    """Turn the streaming-input transport on or off for this process."""
    global _LIVE_MESSAGES
    _LIVE_MESSAGES = bool(enabled)


def live_messages_enabled(provider: str) -> bool:
    """Whether `provider`'s process will accept notes while it works.

    Only claude has the transport; codex's `exec --json` reads one prompt from a
    stdin it then needs closed, so a second message has nowhere to go.
    """
    return _LIVE_MESSAGES and provider == "claude"


def prompt_on_stdin(provider: str) -> bool:
    """Whether this provider's prompt travels on stdin rather than in argv.

    True for codex always, and for claude whenever the live-message transport is
    on - `--input-format stream-json` ignores an argv prompt (measured: the
    session did not start until the first stdin message arrived), so the two go
    together.
    """
    return provider == "codex" or live_messages_enabled(provider)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    display_name: str
    executable: str
    supports_usage_limits: bool


PROVIDERS = {
    "claude": ProviderSpec("claude", "Claude Code", "claude", True),
    "codex": ProviderSpec("codex", "Codex CLI", "codex", True),
}
PROVIDER_NAMES = tuple(PROVIDERS)


def provider_spec(name: str) -> ProviderSpec:
    try:
        return PROVIDERS[name]
    except KeyError as exc:
        expected = ", ".join(PROVIDER_NAMES)
        raise ValueError(
            f"Unknown LLM provider {name!r}; expected one of: {expected}"
        ) from exc


def usage_source_for(provider: str):
    """Return the selected provider's quota source."""
    spec = provider_spec(provider)
    if not spec.supports_usage_limits:
        return None
    # Imported here rather than at the top: a run talks to exactly one provider,
    # and the loser of that choice costs nothing — `codex_usage` starts no
    # process on import, but it does drag in the whole app-server client for a
    # run that will never call it. Both modules sit below this one (they import
    # nothing from the package but each other), so this is a cost, not a cycle.
    if provider == "claude":
        from .usage import UsageSource
        return UsageSource()
    if provider == "codex":
        from .codex_usage import CodexUsageSource
        return CodexUsageSource()
    raise NotImplementedError(f"No usage source adapter for {spec.display_name}")


def runtime_argv(argv: list[str], provider: str) -> list[str]:
    """Resolve a provider shim (notably npm's ``codex.cmd`` on Windows)."""
    spec = provider_spec(provider)
    executable = shutil.which(spec.executable) or spec.executable
    return [executable, *argv[1:]]


def start_agent_process(argv: list[str], provider: str, prompt: str,
                        project_dir: str):
    """Start a provider CLI with its provider-specific prompt transport.

    Three shapes, and which one applies is `prompt_on_stdin`'s answer:

      * claude with live messages - the prompt is the first JSON line on a stdin
        that STAYS OPEN, so the console can send more user messages while the
        turn runs. A pipe left open that way is a session that never ends, so
        every caller must hand the process to ``note_channel`` below rather than
        arrange the closing itself.
      * claude without them - the prompt sits in argv, stdin is inherited, and
        the process exits on its own when the turn is done.
      * codex - the prompt is read from a stdin closed immediately after: this
        preserves non-interactive JSONL mode while avoiding Windows command-line
        length limits, especially when the executable is an npm ``codex.CMD`` shim.
    """
    kwargs = {
        "cwd": project_dir,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "bufsize": 1,
    }
    on_stdin = prompt_on_stdin(provider)
    keep_open = live_messages_enabled(provider)
    if on_stdin:
        kwargs["stdin"] = subprocess.PIPE

    proc = subprocess.Popen(runtime_argv(argv, provider), **kwargs)
    if on_stdin:
        payload = user_message_line(prompt) if keep_open else prompt
        try:
            proc.stdin.write(payload)
            if keep_open:
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            # The CLI may reject its flags before reading stdin. Its stdout/stderr
            # still carries the useful diagnostic, so let the renderer show it.
            pass
        finally:
            if not keep_open:
                try:
                    proc.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
    return proc


@contextmanager
def note_channel(proc, provider: str, mailbox: Optional[object] = None):
    """The live-note pipe of one process, closed however the caller leaves.

    The other half of `start_agent_process`: it decided to leave stdin open, so
    the same module owns the closing rather than lending the obligation to a
    runner. Both runners had their own copy of this before, and the copies were
    not the same one - the parallel one closed the pipe only when a mailbox
    existed, which `-j 2+` never has, so its workers started CLIs they could
    never end (the turn finished in 2.7 s; the call returned when the process
    was killed at 90 s).

    Yields something with `.close()` in every case, including "this run has no
    transport" - so a caller closing at the turn's `result` event needs no
    None-check, and the guard that caused the bug has nowhere to come back.

    The caller must still close before `proc.wait()`: waiting on a process whose
    stdin is open is the same hang seen from the other end. The context manager
    covers the paths a caller cannot - an exception, KeyboardInterrupt, a stream
    that ends without ever reporting a result.
    """
    if live_messages_enabled(provider) and proc.stdin is not None:
        channel = operator.AgentChannel(proc.stdin, mailbox)
    else:
        channel = operator.NullChannel()
    try:
        yield channel
    finally:
        channel.close()


# How long a provider child gets after being told to end before it is killed
# outright, and the ceiling on the `taskkill` call itself. The bound is what
# keeps the reaping from becoming a second hang inside the guard that exists to
# end the first one: the parallel runner calls it from a thread that is ALREADY
# dying, so nothing above it is left to notice a wait that never returns.
REAP_GRACE_S = 2.0


def _ask_agent_process_to_end(proc) -> None:
    """Aim the ending at the provider CLI, not at the shim standing in front of it.

    On Windows the handle we hold is usually `cmd.exe`: `runtime_argv` resolves
    the provider to an npm `.cmd` shim and CreateProcess runs a batch file
    through the interpreter, so the CLI itself is a GRANDCHILD. TerminateProcess
    on the shim leaves it running, still holding the stdout handle it inherited
    and still printing over whatever the terminal does next — which is the whole
    symptom being fixed, so the Windows branch has to reach the tree
    (`taskkill /T`). On POSIX an npm bin is the executable itself (a shebang
    script), so the handle IS the provider and SIGTERM lands where it is aimed.

    ASYMMETRY WORTH KNOWING, because the two halves do NOT offer the same deal:
    POSIX gets a real request — SIGTERM, which a CLI can catch and use to close
    its session store — and only then, after `REAP_GRACE_S`, the kill. Windows
    gets no such step, because there is nothing to ask WITH: `taskkill` without
    `/F` posts WM_CLOSE, which a windowless console process never receives, so
    the polite spelling would do nothing at all and the child would be killed
    two seconds later regardless. `/F` is therefore not impatience — it is the
    only thing that ends the tree there, and the cost (a session store torn
    mid-write) is charged on Windows whichever spelling is used.
    """
    if os.name == "nt":
        try:
            done = subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                                  capture_output=True, timeout=REAP_GRACE_S)
            if done.returncode == 0:
                return
            # Non-zero means the tree was NOT ended (no such pid, access
            # denied): fall through rather than return, or the caller's wait
            # would be a two-second pause on the way to the same `kill`.
        except (OSError, subprocess.SubprocessError):
            pass  # no taskkill, or it hung — fall back to the handle we hold
    try:
        proc.terminate()
    except OSError:
        pass  # it died between the poll and here; `wait` below collects it


def reap_agent_process(proc) -> None:
    """End and collect a provider child whose reader is not coming back.

    Both runners have exactly one reaping exit — the `proc.wait()` after the
    stream ends — and every step above it can raise while the CLI runs: the
    console write behind each rendered line, `note_channel`'s close, a decoder
    error on the child's own stream, a Ctrl+C. Without this the exception
    unwinds past `wait()` and the provider is simply left ALIVE — nothing has
    asked it to stop, and it goes on writing into the same terminal, over the
    output of whatever the run does next.

    Here rather than in either runner because the obligation belongs to whoever
    started the process, and that is `start_agent_process` above: the sequential
    runner used to spell its half as a bare `proc.terminate()` under
    `except KeyboardInterrupt`, which on Windows aims at the shim and leaves
    exactly the child being complained about (see `_ask_agent_process_to_end`).

    Safe on a child that has already exited (only the pipe is closed) and
    bounded on every other path: told to end, then killed if it will not be.

    NOTHING HERE MAY RAISE. It runs from a `finally`, so an exception escaping
    this function REPLACES the one being unwound — the `BrokenPipeError` that
    the whole seam exists to survive would reach the caller as a
    `PermissionError` from the reaper, and the guard above it would report the
    wrong cause.
    """
    # Before the poll, not after it: the read end is a descriptor the caller
    # would otherwise leak until GC even on the common path (an exception raised
    # while printing the LAST lines, after the CLI has already exited), and a
    # child blocked writing into a pipe nobody will read again cannot act on
    # anything it is told until that pipe is gone.
    if proc.stdout is not None:
        try:
            proc.stdout.close()
        except OSError:
            pass
    if proc.poll() is not None:
        return
    _ask_agent_process_to_end(proc)
    try:
        proc.wait(timeout=REAP_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            pass  # gone between the wait and here — `wait` below collects it
        try:
            proc.wait(timeout=REAP_GRACE_S)
        except subprocess.TimeoutExpired:
            # Unkillable (a stuck kernel-mode handle) — deliberately not waited
            # on any longer. In the parallel runner the thread is dying either
            # way, and hanging here would take the rest of the fleet with it,
            # which is the one thing this whole seam exists to prevent.
            pass


def build_agent_argv(command: AgentCommandLike, provider: str,
                     project_dir: str) -> list[str]:
    """Build one unattended JSONL-producing provider invocation."""
    spec = provider_spec(provider)
    if provider == "claude":
        argv = [spec.executable, "-p"]
        if not live_messages_enabled(provider):
            argv.append(command.prompt)
        if command.model:
            argv += ["--model", command.model]
        argv += [
            "--permission-mode", "acceptEdits",
            "--allowedTools", "Bash Edit Write Read Glob Grep WebFetch WebSearch",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        if live_messages_enabled(provider):
            argv += [
                "--input-format", "stream-json",
                # The delivery receipt: every message we write comes back on the
                # event stream, which is how a note reaches the console log and
                # the run record at the moment the agent actually got it, rather
                # than at the moment the console thread wrote to a pipe.
                "--replay-user-messages",
            ]
        return argv

    argv = [
        spec.executable,
        "exec",
        "--json",
    ]
    sandbox_mode = getattr(command, "sandbox_mode", "")
    if sandbox_mode:
        # --approve-for-me itself selects workspace-write and cannot be combined
        # with an explicit sandbox. An unattended explicit mode also needs an
        # explicit non-interactive approval policy.
        argv += [
            "--sandbox", sandbox_mode,
            "--config", 'approval_policy="never"',
        ]
    else:
        # --approve-for-me already selects the workspace-write sandbox. Current
        # Codex CLI versions reject combining it with an explicit --sandbox.
        argv.append("--approve-for-me")
    argv += [
        "--skip-git-repo-check",
        "-C", project_dir,
    ]
    if command.model:
        argv += ["--model", command.model]
    argv.append("-")
    return argv
