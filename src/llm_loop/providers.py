"""LLM provider adapters used by the shared loop engine.

The loop lifecycle is provider-neutral. This module owns the small part that is
not: executable names, non-interactive flags, and command-line construction.
"""

from dataclasses import dataclass
import os
import shutil
import subprocess
from typing import Protocol

from .operator import user_message_line


class AgentCommandLike(Protocol):
    prompt: str
    model: str


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
    if provider == "claude":
        # Lazy import avoids providers -> usage -> cyclecore -> providers while
        # the package is being initialized.
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
        turn runs. Whoever opened it must close it (see
        ``cyclecore.run_agent_streaming``): a streaming-input CLI keeps its
        session alive until stdin reaches EOF, so an unclosed pipe is a run that
        never ends.
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
        # --approve-for-me already selects the workspace-write sandbox. Current
        # Codex CLI versions reject combining it with an explicit --sandbox.
        "--approve-for-me",
        "--skip-git-repo-check",
        "-C", project_dir,
    ]
    if command.model:
        argv += ["--model", command.model]
    argv.append("-")
    return argv
