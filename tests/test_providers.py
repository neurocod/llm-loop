import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_loop import codex_usage, cyclecore, limits, parallel, providers
from llm_loop.cyclecore import AgentCommand, Driver
from llm_loop.providers import (build_agent_argv, provider_spec,
                                   runtime_argv, start_agent_process,
                                   usage_source_for)


@pytest.fixture(autouse=True)
def _restore_streams():
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


def test_claude_argv_keeps_existing_contract():
    argv = build_agent_argv(AgentCommand("work", "opus"), "claude", "/repo")
    assert argv[:5] == ["claude", "-p", "work", "--model", "opus"]
    assert "--output-format" in argv
    assert "stream-json" in argv


def test_codex_argv_is_non_interactive_jsonl():
    argv = build_agent_argv(AgentCommand("work", "gpt-test"), "codex", "/repo")
    assert argv[:3] == ["codex", "exec", "--json"]
    assert "--approve-for-me" in argv
    assert "--sandbox" not in argv
    assert argv[argv.index("-C") + 1] == "/repo"
    assert argv[argv.index("--model") + 1] == "gpt-test"
    assert "work" not in argv
    assert argv[-1] == "-"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        provider_spec("gemini")


def test_codex_weekly_only_window_is_not_misclassified_as_a_session():
    snapshot = codex_usage.parse_rate_limits({"rateLimits": {
        "primary": {
            "usedPercent": 16,
            "windowDurationMins": 7 * 24 * 60,
            "resetsAt": 1787231258,
        },
        "secondary": None,
    }})
    assert snapshot.session.percent is None
    assert snapshot.week_all.percent == 16
    assert snapshot.week_all.reset_ts == 1787231258
    assert snapshot.summary_lines[0].startswith(
        "Current week (all models): 16% used")


def test_codex_short_and_long_windows_map_to_both_policy_slots():
    snapshot = codex_usage.parse_rate_limits({"rateLimits": {
        "primary": {
            "usedPercent": 40,
            "windowDurationMins": 300,
            "resetsAt": 1000,
        },
        "secondary": {
            "usedPercent": 80,
            "windowDurationMins": 10080,
            "resetsAt": 2000,
        },
    }})
    assert snapshot.session.percent == 40
    assert snapshot.session.reset_ts == 1000
    assert snapshot.week_all.percent == 80
    assert snapshot.week_all.reset_ts == 2000


def test_codex_reached_window_is_treated_as_full():
    snapshot = codex_usage.parse_rate_limits({"rateLimits": {
        "primary": {"usedPercent": 42, "windowDurationMins": 300},
        "rateLimitReachedType": "primary",
    }})
    assert snapshot.session.percent == 100


def test_codex_default_policy_watches_session_and_week():
    policy = limits.default_policy("codex")
    assert [rule.label for rule in policy.rules] == [
        "Current session", "Current week (all models)"]


class _FakeAppServer:
    class _Input(io.StringIO):
        def close(self):
            pass

    def __init__(self, lines):
        self.stdin = self._Input()
        self.stdout = io.StringIO("".join(json.dumps(line) + "\n" for line in lines))
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def test_codex_usage_source_uses_app_server_and_caches(monkeypatch):
    created = []

    def fake_popen(argv, **kwargs):
        proc = _FakeAppServer([
            {"id": 0, "result": {"userAgent": "test"}},
            {"id": 1, "result": {"rateLimits": {
                "primary": {"usedPercent": 16,
                            "windowDurationMins": 10080,
                            "resetsAt": 1787231258},
                "secondary": None,
            }}},
        ])
        created.append((argv, proc))
        return proc

    monkeypatch.setattr(codex_usage.shutil, "which", lambda name: "codex.CMD")
    monkeypatch.setattr(codex_usage.subprocess, "Popen", fake_popen)
    source = codex_usage.CodexUsageSource()
    assert source.get_usage().week_all.percent == 16
    assert source.get_usage().week_all.percent == 16
    assert len(created) == 1
    argv, proc = created[0]
    assert argv == ["codex.CMD", "app-server"]
    methods = [json.loads(line)["method"]
               for line in proc.stdin.getvalue().splitlines()]
    assert methods == ["initialize", "initialized", "account/rateLimits/read"]


def test_codex_provider_constructs_its_own_usage_source():
    assert isinstance(usage_source_for("codex"), codex_usage.CodexUsageSource)


def test_runtime_argv_resolves_windows_command_shim(monkeypatch):
    monkeypatch.setattr("llm_loop.providers.shutil.which",
                        lambda executable: "C:/npm/codex.CMD")
    assert runtime_argv(["codex", "exec", "prompt"], "codex") == [
        "C:/npm/codex.CMD", "exec", "prompt"]


class _PromptSink:
    def __init__(self):
        self.text = ""
        self.closed = False

    def write(self, value):
        self.text += value
        return len(value)

    def close(self):
        self.closed = True


class _FakeAgentProcess:
    def __init__(self, has_stdin=True, stdout="", returncode=0):
        self.stdin = _PromptSink() if has_stdin else None
        self.stdout = io.StringIO(stdout)
        self.returncode = returncode

    def wait(self):
        return self.returncode


def test_codex_process_receives_prompt_via_closed_stdin(monkeypatch):
    created = []

    def fake_popen(argv, **kwargs):
        proc = _FakeAgentProcess(has_stdin=kwargs.get("stdin") is subprocess.PIPE)
        created.append((argv, kwargs, proc))
        return proc

    monkeypatch.setattr(providers.subprocess, "Popen", fake_popen)
    prompt = "x" * 50_000

    proc = start_agent_process(
        ["codex", "exec", "--json", "-"], "codex", prompt, "/repo")

    argv, kwargs, _ = created[0]
    assert argv[-1] == "-"
    assert kwargs["stdin"] is subprocess.PIPE
    assert proc.stdin.text == prompt
    assert proc.stdin.closed


def test_claude_process_keeps_prompt_out_of_stdin(monkeypatch):
    created = []

    def fake_popen(argv, **kwargs):
        proc = _FakeAgentProcess(has_stdin=False)
        created.append((argv, kwargs, proc))
        return proc

    monkeypatch.setattr(providers.subprocess, "Popen", fake_popen)
    argv = ["claude", "-p", "work", "--output-format", "stream-json"]

    proc = start_agent_process(argv, "claude", "work", "/repo")

    launched, kwargs, _ = created[0]
    assert launched[1:] == argv[1:]
    assert "stdin" not in kwargs
    assert proc.stdin is None


def test_sequential_codex_runner_forwards_prompt_to_stdin_transport(monkeypatch):
    calls = []

    def fake_start(argv, provider, prompt, project_dir):
        calls.append((argv, provider, prompt, project_dir))
        return _FakeAgentProcess(has_stdin=False)

    monkeypatch.setattr(cyclecore, "start_agent_process", fake_start)

    assert cyclecore.run_agent_streaming(
        ["codex", "exec", "--json", "-"], "codex", False,
        prompt="sequential prompt",
    ) == 0
    assert calls[0][1:3] == ("codex", "sequential prompt")


def test_parallel_codex_runner_forwards_prompt_to_stdin_transport(monkeypatch):
    calls = []

    def fake_start(argv, provider, prompt, project_dir):
        calls.append((argv, provider, prompt, project_dir))
        return _FakeAgentProcess(has_stdin=False)

    monkeypatch.setattr(parallel, "start_agent_process", fake_start)
    command = AgentCommand("parallel prompt", "gpt-test", "job", "codex")

    assert parallel.run_job(1, command)[0] == 0
    assert calls[0][0][-1] == "-"
    assert calls[0][1:3] == ("codex", "parallel prompt")


def test_parallel_job_tag_remains_visible_in_rich_markup():
    rich_markup = pytest.importorskip("rich.markup")
    plain, markup = parallel._job_tag(3)

    assert rich_markup.render(markup).plain == plain


@pytest.mark.parametrize("line", ["0", "null", "true", '"text"', "[]", "[1]"])
def test_sequential_runner_treats_non_object_json_as_diagnostic(
        monkeypatch, capsys, line):
    monkeypatch.setattr(
        cyclecore, "start_agent_process",
        lambda *args: _FakeAgentProcess(has_stdin=False, stdout=f"{line}\n"),
    )

    assert cyclecore.run_agent_streaming(
        ["codex", "exec", "--json", "-"], "codex", False,
    ) == 0
    assert capsys.readouterr().out.strip() == line


@pytest.mark.parametrize("line", ["0", "null", "true", '"text"', "[]", "[1]"])
def test_parallel_runner_ignores_non_object_json(monkeypatch, line):
    monkeypatch.setattr(
        parallel, "start_agent_process",
        lambda *args: _FakeAgentProcess(has_stdin=False, stdout=f"{line}\n"),
    )

    command = AgentCommand("parallel prompt", "gpt-test", "job", "codex")
    assert parallel.run_job(1, command) == (0, None, None)


def test_codex_events_render_message_commands_and_usage(capsys):
    cyclecore._render_codex_event({
        "type": "item.started",
        "item": {"type": "command_execution", "command": "pytest -q"},
    })
    cyclecore._render_codex_event({
        "type": "item.completed",
        "item": {"type": "agent_message", "text": "Finished cleanly."},
    })
    cyclecore._render_codex_event({
        "type": "turn.completed",
        "usage": {"input_tokens": 12, "cached_input_tokens": 3,
                  "output_tokens": 4},
    })
    output = capsys.readouterr().out
    assert "pytest -q" in output
    assert "Finished cleanly." in output
    assert "input 12" in output
    assert "cached 3" in output
    assert "output 4" in output


class _OneShotCodexDriver(Driver):
    provider = "codex"

    def __init__(self):
        self.served = False

    def next_command(self):
        if self.served:
            return None
        self.served = True
        return AgentCommand("do the thing", "", "thing", self.provider)


def test_codex_dry_run_uses_codex_policy_not_claude_usage_source(
        tmp_path, monkeypatch, capsys):
    from llm_loop import usage

    def forbidden_usage_source():
        raise AssertionError("Codex must not read Claude usage")

    monkeypatch.setattr(usage, "UsageSource", forbidden_usage_source)
    args = cyclecore.parse_args([
        "--codex", "--dry-run", "--git-push", "none",
        "--project-dir", str(tmp_path),
    ])
    cyclecore.run_loop(_OneShotCodexDriver(), args, app_name="pytest-provider")
    output = capsys.readouterr().out
    assert "provider: Codex CLI" in output
    assert "Current session" in output
    assert "Current week (all models)" in output
    assert "DRY-RUN: codex exec --json" in output


# --- a failed parallel job must say why ----------------------------------------
#
# The child's stderr is merged into its stdout, so a provider that dies with a
# plain-text message says so on lines the compact renderer skips as non-JSON.
# Dropping them is how a job printed `✗ … (exit 1)` with no cause anywhere.

def _codex_job(monkeypatch, stdout, returncode):
    monkeypatch.setattr(
        parallel, "start_agent_process",
        lambda *args: _FakeAgentProcess(
            has_stdin=False, stdout=stdout, returncode=returncode))
    return parallel.run_job(4, AgentCommand("p", "gpt-test", "job", "codex"))


def test_a_failed_job_surfaces_the_providers_plain_text_diagnostics(
        monkeypatch, capsys):
    rc, _cost, _dur = _codex_job(
        monkeypatch,
        "error: unexpected argument '--approve-for-me' found\n"
        "try 'codex exec --help'\n",
        returncode=1)

    out = capsys.readouterr().out
    assert rc == 1
    assert "[job 4]" in out                      # attributed to the failing job
    assert "unexpected argument" in out
    assert "try 'codex exec --help'" in out


def test_a_successful_job_stays_quiet_about_skipped_lines(monkeypatch, capsys):
    rc, _cost, _dur = _codex_job(
        monkeypatch,
        "npm warn deprecated something@1.0.0\n"
        '{"type": "turn.completed", "usage": {"input_tokens": 1}}\n',
        returncode=0)

    out = capsys.readouterr().out
    assert rc == 0
    assert "npm warn" not in out                 # compact mode stays compact


def test_a_reported_provider_error_is_not_repeated_as_a_tail(monkeypatch, capsys):
    """turn.failed already printed the cause live; repeating it is noise."""
    rc, _cost, _dur = _codex_job(
        monkeypatch,
        "npm warn deprecated something@1.0.0\n"
        '{"type": "turn.failed", "message": "model refused the turn"}\n',
        returncode=0)

    out = capsys.readouterr().out
    assert rc == 1                               # mapped from the failure event
    assert out.count("model refused the turn") == 1
    assert "npm warn" not in out


def test_the_kept_tail_is_bounded_and_truncated(monkeypatch, capsys):
    """A chatty CLI must not be able to grow a worker's memory."""
    noise = "".join(f"line {i} {'x' * 400}\n" for i in range(50))
    _codex_job(monkeypatch, noise, returncode=1)

    out = capsys.readouterr().out
    kept = [ln for ln in out.splitlines() if "line " in ln]
    assert len(kept) == parallel.FAILURE_TAIL_LINES
    assert "line 49" in out and "line 44" not in out   # the LAST few lines
    assert all(len(ln) < 300 for ln in kept)           # each one _short-ened


# --- command lines are shown the way they can be copied ------------------------
#
# Codex doubles every backslash in the `command` it reports, so a Windows path
# reached the screen as C:\\WINDOWS\\... and had to be hand-edited after copying.

def test_a_doubled_windows_path_is_halved_for_display():
    assert cyclecore.undouble_backslashes(
        r"powershell.exe -Command 'D:\\g\\3d-research\\x'"
    ) == r"powershell.exe -Command 'D:\g\3d-research\x'"


def test_a_doubled_unc_path_keeps_its_leading_pair():
    assert cyclecore.undouble_backslashes(
        r"dir \\\\server\\share") == r"dir \\server\share"


def test_a_string_with_an_odd_run_is_left_completely_alone():
    raw = r"grep 'a\b' \\host\c"      # runs of 1, 2, 1: not uniformly doubled
    assert cyclecore.undouble_backslashes(raw) == raw


def test_a_string_without_backslashes_is_untouched():
    assert cyclecore.undouble_backslashes("pytest -q") == "pytest -q"


def test_both_renderers_use_the_helper_on_command_execution(monkeypatch, capsys):
    item = {"type": "command_execution", "command": r"cd D:\\g\\loop && ls",
            "exit_code": 0}
    cyclecore._render_codex_event({"type": "item.completed", "item": item})
    _codex_job(monkeypatch,
               json.dumps({"type": "item.completed", "item": item}) + "\n",
               returncode=0)

    out = capsys.readouterr().out
    assert out.count(r"cd D:\g\loop && ls") == 2
    assert r"D:\\g" not in out


def test_the_claude_bash_tool_line_uses_the_helper():
    assert cyclecore._describe_tool("Bash", {"command": r"type C:\\a\\b.txt"}) \
        == r"$ type C:\a\b.txt"
