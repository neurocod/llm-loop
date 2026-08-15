"""LLM provider adapters used by the shared loop engine.

The loop lifecycle is provider-neutral. This module owns the small part that is
not: executable names, non-interactive flags, and command-line construction.
"""

from dataclasses import dataclass
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
    # Codex quota retrieval is intentionally a separate follow-up: do not let a
    # Codex run accidentally consult Claude's credentials or quota endpoint.
    "codex": ProviderSpec("codex", "Codex CLI", "codex", False),
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
    """Return the provider's quota source, or None while it has no adapter."""
    spec = provider_spec(provider)
    if not spec.supports_usage_limits:
        return None
    if provider == "claude":
        # Lazy import avoids providers -> usage -> cyclecore -> providers while
        # the package is being initialized.
        from .usage import UsageSource
        return UsageSource()
    raise NotImplementedError(f"No usage source adapter for {spec.display_name}")


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
