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
usage/codex_usage for the provider quota query layers, limits for the
pausing policy, and modeswitch for a wrapper's own mode switches (the flags
that choose between the two runners' parsers).
"""

import importlib as _importlib

# The single source of truth for the version: pyproject.toml reads it back out
# of here (`[tool.setuptools.dynamic] version = {attr = "llm_loop.__version__"}`),
# so bumping this line is the whole release step. Keep it a plain literal -
# the build backend parses it without importing the package, which only works
# while the right-hand side stays a constant.
__version__ = "0.1.0"

# The front door is LAZY (PEP 562): a name below is imported from its home
# module on first access, not when the package is. Every submodule import runs
# this file first, so an eager front door made `import llm_loop.wire` — a leaf a
# tool reads for one classifier — pay for the whole engine, rich's Markdown
# renderer and markdown_it included: 0.64 s of import on 2026-10-27, which
# `tools/codex_agent` paid on every command, `status` included. Pinned by
# tests/test_lazy_front_door.py. Spellings are unchanged: `from llm_loop import
# Driver`, `llm_loop.run_loop`, `from llm_loop import *` and `llm_loop.cyclecore`
# (any submodule) all resolve on demand.
#
# name -> (home module, attribute there); attribute None = the module itself.
_FRONT_DOOR = {}


def _home(module, *names):
    for name in names:
        _FRONT_DOOR[name] = (module, name)


for _module in ("cmdline", "exitlog", "modeswitch", "stopchannel"):
    _FRONT_DOOR[_module] = (_module, None)
_home("modeswitch", "ModeSwitch")
_home("cyclecore", "is_report", "parse_args", "run_loop")
# Same story as the moves below: parsing a --start-in value moved out of
# cyclecore into `clispec`, beside the `type=` check that now runs it inside the
# parser, and the front door is unchanged on purpose.
_home("clispec", "parse_duration")
# Same story again: reading a run's cost back out of the mirror log moved from
# cyclecore into `costlog`, with the two lines it reads, and the front door is
# unchanged on purpose.
_home("costlog", "report_costs")
# Same story as every move below: rendering one provider stream — and the
# single-stream latch that goes with it — left cyclecore for `streamrender`, and
# the front door is unchanged on purpose. An embedder driving its own turn asks
# the PACKAGE to run one, not whichever module currently prints it.
_home("streamrender", "last_rate_limit_event", "run_agent_streaming")
# Same story as the four below: the vocabulary of WORK — what a unit of it is,
# how it becomes an argv, and the Driver protocol a wrapper subclasses — moved
# out of cyclecore into `agentwork`, and the front door is unchanged on purpose.
# A wrapper names the PACKAGE's `Driver`, not the runner it happens to drive;
# most wrappers here drive the parallel one and never call `run_loop` at all.
_home("agentwork", "AgentCommand", "ClaudeCommand", "Driver", "LoopStop",
      "build_agent_argv", "build_claude_argv")
# The project root moved out of cyclecore into `projectroot`, a leaf below both
# runners and below the two modules that used to keep MIRRORS of it (see that
# module's header). Same story as the three below: the front door is unchanged
# on purpose — an embedder asks the PACKAGE where the project is, not whichever
# module currently holds the global. `cyclecore.project_dir` deliberately no
# longer resolves, so there is exactly one internal address for it too.
_home("projectroot", "find_project_root", "project_dir", "set_project_root")
# Same story as the stop vocabulary below: the terminal front end and the mirror
# log it writes moved out of cyclecore into `console`, and the front door is
# unchanged on purpose — an embedder asks the PACKAGE where its log is.
_home("console", "log_file_path")
# And again for the git-push policy, which moved out into `gitpush`: a wrapper
# that names a policy names the PACKAGE's, not the runner it happens to drive.
_home("gitpush", "GitPushPolicy")
# The stop/pause vocabulary moved out of cyclecore into its own module; the
# front door is unchanged on purpose, since an embedder asks the PACKAGE for
# `pending_stop`, not the module that used to hold it.
_home("stopchannel", "REQUESTED_STOP_REASONS", "RunResult", "RunStopReason",
      "StopSource", "commit_stop", "confirm_stop_request", "latched_stop",
      "pause_requested", "pending_stop", "stop_file_for", "wait_while_paused")
_home("providers", "PROVIDER_NAMES", "PROVIDERS", "ProviderSpec", "provider_spec",
      "runtime_argv", "usage_source_for")
# `Mailbox` is the only valid argument to run_agent_streaming's and StatusApp's
# `mailbox=`/`messages=`, so an embedder building its own front end needs it
# from the front door rather than from llm_loop.operator.
_home("operator", "Mailbox", "MailboxSet")
_home("codex_usage", "CodexUsageSource", "parse_rate_limits")
# `RateLimitEvent`/`rate_limit_event_from` are here and not with the renderer
# that shows them: they say what a quota's wire verdict IS, which is a fact about
# a quota. Only `last_rate_limit_event` above stays `streamrender`'s — it is that
# renderer's single-stream latch, not vocabulary (see its block there).
_home("usage", "RateLimitEvent", "Usage", "UsageReading", "UsageSource",
      "oauth_token", "parse_usage", "rate_limit_event_from")
_home("limits", "DayNightLimit", "LimitPolicy", "LimitRule", "SessionLimit",
      "WeeklyLimit", "default_policy")
_home("statusline", "Action", "InvocationProgress", "Job", "LoopStatus", "Mode",
      "NumberSetting", "PercentSetting", "QuotaRefresher", "Setting",
      "SettingsRegistry", "StatusApp", "push_quotas", "quota_rows", "render_rows")
_home("drivers", "ListFileDriver", "StateFileDriver")
_home("parallel", "run_parallel")
_FRONT_DOOR["parse_parallel_args"] = ("parallel", "parse_args")
_home("notifications", "SettingsError", "completion_sound_enabled",
      "play_completion_sound", "settings_path")
del _module, _home

__all__ = sorted(_FRONT_DOOR)


def __getattr__(name):
    home = _FRONT_DOOR.get(name)
    if home is not None:
        module = _importlib.import_module(f".{home[0]}", __name__)
        value = module if home[1] is None else getattr(module, home[1])
        globals()[name] = value      # resolved once; later reads skip this hook
        return value
    # Not a front-door name: `llm_loop.cyclecore` after a bare `import llm_loop`
    # resolved while the front door was eager, so any submodule still does.
    try:
        return _importlib.import_module(f".{name}", __name__)
    except ModuleNotFoundError as error:
        if error.name != f"{__name__}.{name}":
            raise                    # the submodule exists; ITS import failed
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
