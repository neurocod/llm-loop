"""
claude_loop - a reusable engine for autonomous Claude/Codex CLI loops.

Vendor this package as a git submodule under a host project, then write a thin
wrapper in the project that subclasses a Driver — supplying the project-specific
bits (which file to read, which prompt to send, which model to use) as class
attributes / overridden methods — and calls its `.main()`:

    # runTranslate.py (thin wrapper in the host project root)
    from claude_loop import ListFileDriver

    class TranslateDriver(ListFileDriver):
        list_file     = "products/list.md"
        target_suffix = ".ru.md"

        def model(self):
            return "" if self.provider == "codex" else "sonnet"

        def prompt(self, source, target):
            return f"Translate {source} to Russian, write {target}, keep search: verbatim ..."

    if __name__ == "__main__":
        TranslateDriver.main()          # or .main_parallel() for N concurrent workers

The engine anchors every project-relative operation (git/agent cwd, the stop
file, the Driver's relative paths) to the project root — the current working
directory by default, or --project-dir / set_project_root(). So the code can
live in a submodule subdirectory while still driving the host project's repo.

Pick a usage-limit specialisation with a Driver's `limit_policy` attribute, e.g.
`LimitPolicy([SessionLimit(80)])`, `LimitPolicy([WeeklyLimit(90)])`, or a
composite `LimitPolicy([DayNightLimit(), WeeklyLimit(90)])`; unset defaults to a
day/night session rule for Claude and a session-plus-weekly policy for Codex.

See cyclecore for the engine and the Driver protocol, drivers for the two ready
made Drivers, parallel for the concurrent list runner, usage/codex_usage for the
provider quota query layers, and limits for the pausing policy.
"""

from . import cmdline
from .cyclecore import (
    AgentCommand,
    ClaudeCommand,
    Driver,
    GitPushPolicy,
    LoopStop,
    RateLimitEvent,
    RunResult,
    RunStopReason,
    build_agent_argv,
    build_claude_argv,
    confirm_stop_request,
    find_project_root,
    last_rate_limit_event,
    log_file_path,
    parse_args,
    parse_duration,
    project_dir,
    rate_limit_event_from,
    report_costs,
    run_loop,
    run_agent_streaming,
    set_project_root,
)
from .providers import (
    PROVIDER_NAMES,
    PROVIDERS,
    ProviderSpec,
    provider_spec,
    runtime_argv,
    usage_source_for,
)
from .codex_usage import CodexUsageSource, parse_rate_limits
from .usage import Usage, UsageReading, UsageSource, oauth_token, parse_usage
from .limits import (
    DayNightLimit,
    LimitPolicy,
    LimitRule,
    SessionLimit,
    WeeklyLimit,
    default_policy,
)
from .statusline import (
    Action,
    InvocationProgress,
    Job,
    LoopStatus,
    Mode,
    NumberSetting,
    PercentSetting,
    QuotaRefresher,
    Setting,
    SettingsRegistry,
    StatusApp,
    push_quotas,
    quota_rows,
    render_rows,
)
from .drivers import ListFileDriver, StateFileDriver
from .parallel import run_parallel
from .parallel import parse_args as parse_parallel_args
from .notifications import (
    SettingsError,
    completion_sound_enabled,
    play_completion_sound,
    settings_path,
)

__all__ = [
    "Action",
    "AgentCommand",
    "ClaudeCommand",
    "CodexUsageSource",
    "DayNightLimit",
    "Driver",
    "GitPushPolicy",
    "InvocationProgress",
    "Job",
    "LimitPolicy",
    "LimitRule",
    "ListFileDriver",
    "LoopStatus",
    "LoopStop",
    "Mode",
    "NumberSetting",
    "PROVIDERS",
    "PROVIDER_NAMES",
    "PercentSetting",
    "ProviderSpec",
    "QuotaRefresher",
    "RateLimitEvent",
    "RunResult",
    "RunStopReason",
    "SessionLimit",
    "Setting",
    "SettingsError",
    "SettingsRegistry",
    "StateFileDriver",
    "StatusApp",
    "Usage",
    "UsageReading",
    "UsageSource",
    "WeeklyLimit",
    "build_agent_argv",
    "build_claude_argv",
    "cmdline",
    "completion_sound_enabled",
    "confirm_stop_request",
    "default_policy",
    "find_project_root",
    "last_rate_limit_event",
    "log_file_path",
    "oauth_token",
    "parse_args",
    "parse_parallel_args",
    "parse_duration",
    "parse_rate_limits",
    "parse_usage",
    "play_completion_sound",
    "project_dir",
    "provider_spec",
    "push_quotas",
    "quota_rows",
    "rate_limit_event_from",
    "render_rows",
    "report_costs",
    "run_agent_streaming",
    "run_loop",
    "run_parallel",
    "runtime_argv",
    "set_project_root",
    "settings_path",
    "usage_source_for",
]
