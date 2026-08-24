"""
llm_loop - a reusable engine for autonomous Claude/Codex CLI loops.

Vendor this package as a git submodule under a host project, then write a thin
wrapper in the project that subclasses a Driver — supplying the project-specific
bits (which file to read, which prompt to send, which model to use) as class
attributes / overridden methods — and calls its `.main()`:

    # runTranslate.py (thin wrapper in the host project root)
    from llm_loop import ListFileDriver

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

See agentwork for what a unit of work is and the Driver protocol you subclass,
cyclecore for the sequential engine, streamrender for one provider run shown
live and wire for the words that stream is made of,
drivers for the two ready made Drivers,
parallel for the concurrent list runner, projectroot for the root itself,
usage/codex_usage for the provider quota query layers, and limits for the
pausing policy.
"""

# The single source of truth for the version: pyproject.toml reads it back out
# of here (`[tool.setuptools.dynamic] version = {attr = "llm_loop.__version__"}`),
# so bumping this line is the whole release step. Keep it a plain literal -
# the build backend parses it without importing the package, which only works
# while the right-hand side stays a constant.
__version__ = "0.1.0"

from . import cmdline
from . import exitlog
from . import stopchannel
from .cyclecore import (
    ConsumedByWrapperAction,
    parse_args,
    parse_duration,
    report_costs,
    run_loop,
)
# Same story as every move below: rendering one provider stream — and the
# single-stream latch that goes with it — left cyclecore for `streamrender`, and
# the front door is unchanged on purpose. An embedder driving its own turn asks
# the PACKAGE to run one, not whichever module currently prints it.
from .streamrender import last_rate_limit_event, run_agent_streaming
# Same story as the four below: the vocabulary of WORK — what a unit of it is,
# how it becomes an argv, and the Driver protocol a wrapper subclasses — moved
# out of cyclecore into `agentwork`, and the front door is unchanged on purpose.
# A wrapper names the PACKAGE's `Driver`, not the runner it happens to drive;
# most wrappers here drive the parallel one and never call `run_loop` at all.
from .agentwork import (
    AgentCommand,
    ClaudeCommand,
    Driver,
    LoopStop,
    build_agent_argv,
    build_claude_argv,
)
# The project root moved out of cyclecore into `projectroot`, a leaf below both
# runners and below the two modules that used to keep MIRRORS of it (see that
# module's header). Same story as the three below: the front door is unchanged
# on purpose — an embedder asks the PACKAGE where the project is, not whichever
# module currently holds the global. `cyclecore.project_dir` deliberately no
# longer resolves, so there is exactly one internal address for it too.
from .projectroot import find_project_root, project_dir, set_project_root
# Same story as the stop vocabulary below: the terminal front end and the mirror
# log it writes moved out of cyclecore into `console`, and the front door is
# unchanged on purpose — an embedder asks the PACKAGE where its log is.
from .console import log_file_path
# And again for the git-push policy, which moved out into `gitpush`: a wrapper
# that names a policy names the PACKAGE's, not the runner it happens to drive.
from .gitpush import GitPushPolicy
# The stop/pause vocabulary moved out of cyclecore into its own module; the
# front door is unchanged on purpose, since an embedder asks the PACKAGE for
# `pending_stop`, not the module that used to hold it.
from .stopchannel import (
    REQUESTED_STOP_REASONS,
    RunResult,
    RunStopReason,
    StopSource,
    commit_stop,
    confirm_stop_request,
    latched_stop,
    pause_requested,
    pending_stop,
    stop_file_for,
    wait_while_paused,
)
from .providers import (
    PROVIDER_NAMES,
    PROVIDERS,
    ProviderSpec,
    provider_spec,
    runtime_argv,
    usage_source_for,
)
from .operator import Mailbox
from .codex_usage import CodexUsageSource, parse_rate_limits
# `RateLimitEvent`/`rate_limit_event_from` are here and not with the renderer
# that shows them: they say what a quota's wire verdict IS, which is a fact about
# a quota. Only `last_rate_limit_event` above stays `streamrender`'s — it is that
# renderer's single-stream latch, not vocabulary (see its block there).
from .usage import (
    RateLimitEvent,
    Usage,
    UsageReading,
    UsageSource,
    oauth_token,
    parse_usage,
    rate_limit_event_from,
)
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
    "ConsumedByWrapperAction",
    "DayNightLimit",
    "Driver",
    "GitPushPolicy",
    "InvocationProgress",
    "Job",
    "LimitPolicy",
    "LimitRule",
    "ListFileDriver",
    "LoopStatus",
    # The only valid argument to run_agent_streaming's and StatusApp's
    # `mailbox=`/`messages=`, so an embedder building its own front end needs it
    # from the front door rather than from llm_loop.operator.
    "Mailbox",
    "LoopStop",
    "Mode",
    "NumberSetting",
    "PROVIDERS",
    "PROVIDER_NAMES",
    "PercentSetting",
    "ProviderSpec",
    "QuotaRefresher",
    "REQUESTED_STOP_REASONS",
    "RateLimitEvent",
    "RunResult",
    "RunStopReason",
    "SessionLimit",
    "Setting",
    "SettingsError",
    "SettingsRegistry",
    "StateFileDriver",
    "StatusApp",
    "StopSource",
    "Usage",
    "UsageReading",
    "UsageSource",
    "WeeklyLimit",
    "build_agent_argv",
    "build_claude_argv",
    "cmdline",
    "commit_stop",
    "completion_sound_enabled",
    "confirm_stop_request",
    "default_policy",
    "exitlog",
    "find_project_root",
    "last_rate_limit_event",
    "latched_stop",
    "log_file_path",
    "oauth_token",
    "parse_args",
    "parse_parallel_args",
    "parse_duration",
    "parse_rate_limits",
    "parse_usage",
    "pause_requested",
    "pending_stop",
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
    "stop_file_for",
    "stopchannel",
    "usage_source_for",
    "wait_while_paused",
]
