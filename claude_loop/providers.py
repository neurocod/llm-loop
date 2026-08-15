"""LLM provider adapters used by the shared loop engine.

The loop lifecycle is provider-neutral. This module owns the small part that is
not: executable names, non-interactive flags, and command-line construction.
"""

from dataclasses import dataclass
import shutil
from typing import Protocol


class AgentCommandLike(Protocol):
    prompt: str
    model: str


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


def build_agent_argv(command: AgentCommandLike, provider: str,
                     project_dir: str) -> list[str]:
    """Build one unattended JSONL-producing provider invocation."""
    spec = provider_spec(provider)
    if provider == "claude":
        argv = [spec.executable, "-p", command.prompt]
        if command.model:
            argv += ["--model", command.model]
        argv += [
            "--permission-mode", "acceptEdits",
            "--allowedTools", "Bash Edit Write Read Glob Grep WebFetch WebSearch",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]
        return argv

    argv = [
        spec.executable,
        "exec",
        "--json",
        "--sandbox", "workspace-write",
        "--approve-for-me",
        "--skip-git-repo-check",
        "-C", project_dir,
    ]
    if command.model:
        argv += ["--model", command.model]
    argv.append(command.prompt)
    return argv
