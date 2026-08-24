import io
import json
import os
import subprocess
import sys

import pytest

from llm_loop import (codex_usage, compactline, console, cyclecore, limits,
                      parallel, providers, textwidth)
from llm_loop.agentwork import AgentCommand, Driver
from llm_loop.providers import (build_agent_argv, provider_spec,
                                   runtime_argv, start_agent_process,
                                   usage_source_for)


@pytest.fixture(autouse=True)
def _restore_streams():
    out, err = sys.stdout, sys.stderr
    yield
    sys.stdout, sys.stderr = out, err


@pytest.fixture
def no_live_messages():
    """The pre-operator-notes transport: prompt in argv, stdin inherited."""
    previous = providers.live_messages_enabled("claude")
    providers.set_live_messages(False)
    yield
    providers.set_live_messages(previous)


def test_claude_argv_keeps_existing_contract(no_live_messages):
    argv = build_agent_argv(AgentCommand("work", "opus"), "claude", "/repo")
    assert argv[:5] == ["claude", "-p", "work", "--model", "opus"]
    assert "--output-format" in argv
    assert "stream-json" in argv
    assert "--input-format" not in argv


def test_codex_argv_is_non_interactive_jsonl():
    argv = build_agent_argv(AgentCommand("work", "gpt-test"), "codex", "/repo")
    assert argv[:3] == ["codex", "exec", "--json"]
    assert "--approve-for-me" in argv
    assert "--sandbox" not in argv
    assert argv[argv.index("-C") + 1] == "/repo"
    assert argv[argv.index("--model") + 1] == "gpt-test"
    assert "work" not in argv
    assert argv[-1] == "-"


def test_codex_argv_can_select_a_workflow_specific_sandbox():
    command = AgentCommand("work", "gpt-test",
                           sandbox_mode="danger-full-access")
    argv = build_agent_argv(command, "codex", "/repo")

    assert "--approve-for-me" not in argv
    assert argv[argv.index("--sandbox") + 1] == "danger-full-access"
    assert argv[argv.index("--config") + 1] == 'approval_policy="never"'


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

    def poll(self):
        # Always "already exited": this fake's stdout is a fixed string, so it
        # models a process whose stream has ended, never one still running.
        # Both runners ask before reaping (see `providers.reap_agent_process`),
        # and a fake that answered None would have the reaper signalling a pid
        # nobody owns.
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


def test_claude_process_keeps_prompt_out_of_stdin(monkeypatch, no_live_messages):
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


@pytest.mark.parametrize("shape,args", [
    ("line", ("$ grep '[job 3]' *.md",)),
    ("tool", ("Grep", "[job 3] in products")),
])
def test_a_bracket_in_a_worker_line_reaches_the_screen(monkeypatch, shape, args):
    """rich reads '[' as the start of a style tag and drops what follows it
    without complaining, so the screen quietly lost text the log still had."""
    rich_markup = pytest.importorskip("rich.markup")
    printed = []
    monkeypatch.setattr(parallel, "print_markup",
                        lambda plain, markup: printed.append((plain, markup)))

    getattr(parallel.job_lines(3), shape)(*args)

    plain, markup = printed[0]
    assert "[job 3]" in plain                     # twice: the tag and the text
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


# --- the sequential runner owes its child an ending too --------------------------

# The fake provider writes one line and then simply stays alive. Long enough
# that "still running when the runner returned" cannot be a race with a child
# about to exit by itself — the question this pin asks is whether anything
# ENDED it.
_ORPHAN_LIFETIME_S = 120

# The text of that line. A non-JSON line is printed straight through by
# `run_agent_streaming`, so this is what lets the stub stdout fail on the ONE
# write that happens with a child process running.
_KILLS_THE_WRITER = "boom"

_FAKE_PROVIDER_SRC = (
    "import sys, time\n"
    "sys.stdout.write(%r)\n"
    "sys.stdout.flush()\n"
    "time.sleep(%d)\n"
) % (_KILLS_THE_WRITER + "\n", _ORPHAN_LIFETIME_S)

# How long a child gets to be dead once the runner has returned. The reaping is
# synchronous inside `run_agent_streaming`, so a healthy call has already
# collected it and this bound is never approached; it is here so the FAILING
# case reports an orphan instead of blocking for `_ORPHAN_LIFETIME_S`.
REAP_WAIT_S = 5.0


class _StdoutThatDies:
    """A terminal that goes away mid-line, as a closed pipe does."""

    def write(self, text):
        if _KILLS_THE_WRITER in text:
            raise BrokenPipeError("the terminal went away mid-line")
        return len(text)

    def flush(self):
        pass


def _outlived_the_runner(proc, timeout=REAP_WAIT_S) -> bool:
    """Is `proc` still running `timeout` after the call that started it returned?

    Always leaves the child dead: a pin that fails must not also leak the very
    process it is complaining about into the rest of the suite.
    """
    try:
        proc.wait(timeout=timeout)
        return False
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=REAP_WAIT_S)
        return True


def test_a_dying_sequential_turn_does_not_leave_the_provider_running(monkeypatch):
    """`run_agent_streaming` that raises must not walk away from a live child.

    Its `proc.wait()` is the only reaping exit, and every write to the terminal
    sits above it — the sequential runner used to guard only `KeyboardInterrupt`,
    and even that guard called `proc.terminate()`, which on Windows aims at the
    npm `.cmd` shim and leaves the CLI (a grandchild) running.

    A real subprocess rather than a stub, because the defect is exactly the
    thing a stub does not have — an OS process that outlives the function that
    started it.
    """
    children = []

    def fake_provider(argv, provider, prompt, project_dir):
        proc = subprocess.Popen(
            [sys.executable, "-c", _FAKE_PROVIDER_SRC],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", bufsize=1)
        children.append(proc)
        return proc

    monkeypatch.setattr(cyclecore, "start_agent_process", fake_provider)
    monkeypatch.setattr(sys, "stdout", _StdoutThatDies())

    with pytest.raises(BrokenPipeError):
        cyclecore.run_agent_streaming(
            ["codex", "exec", "--json", "-"], "codex", False)

    assert children, "the fake provider was never started — the pin proved nothing"
    assert not _outlived_the_runner(children[0]), \
        "run_agent_streaming unwound past a live provider child: orphaned, not reaped"


def test_a_failed_prompt_handover_does_not_leave_the_provider_running(monkeypatch):
    """The one stretch neither runner's `finally` can cover.

    Both runners guard from `start_agent_process`'s RETURN onwards, so an
    exception between `Popen` and that return leaves a live CLI with no reaper
    anywhere — nobody has the handle yet. Building the stdin payload is inside
    that stretch, and it can raise on its own (a pipe closed under us, a payload
    that will not encode).

    A real subprocess again, and a real failure at the real seam: the launcher
    is asked for the live-message transport, and the call that builds its first
    message raises.
    """
    started = []
    real_popen = subprocess.Popen

    def watch_popen(argv, **kwargs):
        proc = real_popen(argv, **kwargs)
        started.append(proc)
        return proc

    def boom(prompt):
        raise ValueError("the payload could not be encoded")

    previous = providers.live_messages_enabled("claude")
    providers.set_live_messages(True)
    try:
        monkeypatch.setattr(
            providers, "runtime_argv",
            lambda argv, provider: [sys.executable, "-c", _FAKE_PROVIDER_SRC])
        monkeypatch.setattr(providers, "user_message_line", boom)
        monkeypatch.setattr(providers.subprocess, "Popen", watch_popen)

        with pytest.raises(ValueError):
            start_agent_process(["claude", "-p"], "claude", "work", os.getcwd())
    finally:
        providers.set_live_messages(previous)

    assert started, "no process was started — the pin proved nothing"
    assert not _outlived_the_runner(started[0]), \
        "start_agent_process raised past a live provider child: orphaned, not reaped"


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
    # The width is pinned, because the tail is cut to the terminal: run
    # in a 500-column window this same assertion failed, which is the test
    # depending on the developer's terminal rather than on the code.
    _terminal(monkeypatch, 200)
    noise = "".join(f"line {i} {'x' * 400}\n" for i in range(50))
    _codex_job(monkeypatch, noise, returncode=1)

    out = capsys.readouterr().out
    kept = [ln for ln in out.splitlines() if "line " in ln]
    assert len(kept) == parallel.FAILURE_TAIL_LINES
    assert "line 49" in out and "line 44" not in out   # the LAST few lines
    assert all(len(ln) < 300 for ln in kept)           # each one cut to the row


# --- command lines are shown the way they can be copied ------------------------
#
# Codex doubles every backslash in the `command` it reports, so a Windows path
# reached the screen as C:\\WINDOWS\\... and had to be hand-edited after copying.

def test_a_doubled_windows_path_is_halved_for_display():
    assert compactline.undouble_backslashes(
        r"powershell.exe -Command 'D:\\g\\3d-research\\x'"
    ) == r"powershell.exe -Command 'D:\g\3d-research\x'"


def test_a_doubled_unc_path_keeps_its_leading_pair():
    assert compactline.undouble_backslashes(
        r"dir \\\\server\\share") == r"dir \\server\share"


def test_a_string_with_an_odd_run_is_left_completely_alone():
    raw = r"grep 'a\b' \\host\c"      # runs of 1, 2, 1: not uniformly doubled
    assert compactline.undouble_backslashes(raw) == raw


def test_a_string_without_backslashes_is_untouched():
    assert compactline.undouble_backslashes("pytest -q") == "pytest -q"


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
    assert compactline.describe_tool("Bash", {"command": r"type C:\\a\\b.txt"}) \
        == r"$ type C:\a\b.txt"


# --- one event, one line, the width of the terminal ---------------------------
#
# Each renderer used to cut its variable field at a figure of its own: 200
# characters for a Claude tool call, 160 or 140 for the same command coming from
# codex. How much of a command reached the screen therefore depended on the
# provider rather than on the screen — a wide terminal showed a halved command
# beside a third of a blank row, a narrow one wrapped anyway. All of them now
# ask `textwidth.line_budget`, whose own arithmetic is pinned in
# test_textwidth.py; what follows is that the renderers use it, prefix
# included.

LONG_COMMAND = "echo " + "x" * 500


@pytest.fixture
def plain_lines(monkeypatch):
    """Every rendered line, as the PLAIN copy that goes to the mirror log.

    Collected here rather than from capsys because with rich installed the
    styled copy is wrapped at rich's own console width (80 when stdout is not a
    terminal), so the capture would show that wrapping instead of our cut.

    Both renderers reach their sink through a `compactline.LineWriter` that
    resolves this name per line, which is what makes patching it here work at
    all — a writer built around the function object would print past the capture
    and leave every assertion below reading an empty list.
    """
    lines = []

    def record(plain, _markup):
        lines.append(plain)

    monkeypatch.setattr(console, "print_markup", record)
    monkeypatch.setattr(parallel, "print_markup", record)   # imported by name
    return lines


def _terminal(monkeypatch, columns):
    """Pretend the terminal is `columns` wide; return what one line may fill.

    Only meaningful above `LEGACY_LINE_COLUMNS`, below which a line is allowed
    to wrap rather than record less (see the narrow-terminal test).
    """
    monkeypatch.setattr(textwidth.shutil, "get_terminal_size",
                        lambda fallback=(80, 24): os.terminal_size((columns, 30)))
    return columns - 1          # the column left for the terminal to wrap on


def _bash_tool_use(command):
    return {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": command}}]}}


def test_a_tool_line_is_cut_to_the_terminal_not_to_two_hundred(
        monkeypatch, plain_lines):
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_claude_event(_bash_tool_use(LONG_COMMAND), True)

    line, = plain_lines
    assert line.endswith("…")                        # it was cut
    assert textwidth.cell_width(line) == fits       # ...to the terminal
    assert len(line) > 200                           # ...not to the old figure


def test_the_same_command_is_cut_the_same_from_either_provider(
        monkeypatch, plain_lines):
    """200 for claude and 160 for codex was the provider deciding the width."""
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_claude_event(_bash_tool_use(LONG_COMMAND), True)
    cyclecore._render_codex_event({"type": "item.started", "item": {
        "type": "command_execution", "command": LONG_COMMAND}})

    claude_line, codex_line = plain_lines
    assert textwidth.cell_width(claude_line) == fits
    assert textwidth.cell_width(codex_line) == fits


def test_a_worker_line_leaves_room_for_its_job_tag(monkeypatch, plain_lines):
    """`[job 4] ` is furniture too: cut to a bare line, a worker's text would
    overflow by exactly the tag."""
    fits = _terminal(monkeypatch, 300)

    _codex_job(monkeypatch, json.dumps({"type": "item.started", "item": {
        "type": "command_execution", "command": LONG_COMMAND}}) + "\n",
        returncode=0)

    line, = plain_lines
    assert line.startswith("[job 4] ")
    assert textwidth.cell_width(line) == fits


def test_a_worker_tool_line_leaves_room_for_its_job_tag(
        monkeypatch, plain_lines):
    fits = _terminal(monkeypatch, 300)
    monkeypatch.setattr(
        parallel, "start_agent_process",
        lambda *args: _FakeAgentProcess(
            has_stdin=False,
            stdout=json.dumps(_bash_tool_use(LONG_COMMAND)) + "\n"))

    parallel.run_job(4, AgentCommand("p", "opus", "job", "claude"))

    line, = plain_lines
    assert line.startswith("[job 4]   ⚙ Bash: $ ")
    assert textwidth.cell_width(line) == fits


def test_a_narrow_terminal_still_records_what_the_old_limits_did(
        monkeypatch, plain_lines):
    """The mirror log reads these lines too, so a small window is allowed to
    wrap them rather than shrink what the run recorded."""
    _terminal(monkeypatch, 40)

    cyclecore._render_claude_event(_bash_tool_use(LONG_COMMAND), True)

    line, = plain_lines
    assert len(line) >= 200         # the old fixed figure, wrapped as it was


def test_a_wide_glyph_body_is_cut_by_cells_not_characters(
        monkeypatch, plain_lines):
    """The budget is columns; spending it in characters put a 299-character
    line 580 columns wide on the screen."""
    pytest.importorskip("rich.cells")
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_claude_event(_bash_tool_use("漢" * 500), True)

    line, = plain_lines
    assert textwidth.cell_width(line) == fits
    assert len(line) < 200          # half as many characters as columns


def test_a_printer_coerces_a_detail_that_is_not_a_string(plain_lines):
    """Both printers take whatever their caller has. Joined with `+`, a number
    raised TypeError from inside the renderer — which ends the sequential run,
    and in a worker escapes the thread while it still holds a claimed item."""
    console.LINES.tool("Read", 123)
    parallel.job_lines(4).tool("Read", 123)

    assert plain_lines == ["  ⚙ Read: 123", "[job 4]   ⚙ Read: 123"]


@pytest.mark.parametrize("name,tool_input,expected", [
    ("Read", {"file_path": 123}, "  ⚙ Read: 123"),
    ("Skill", {"skill": ["deploy"]}, "  ⚙ Skill: ['deploy']"),
    ("Grep", {"pattern": 7}, "  ⚙ Grep: 7"),
    ("Write", {"file_path": None}, "  ⚙ Write"),      # nothing to say, no ': '
])
def test_a_tool_input_that_is_not_a_string_still_prints(
        monkeypatch, plain_lines, name, tool_input, expected):
    """End to end over the same values, since these are whatever the provider's
    JSON held — `null` included, which used to print the word None."""
    _terminal(monkeypatch, 300)

    cyclecore._render_claude_event({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": name, "input": tool_input}]}}, True)

    assert plain_lines == [expected]


@pytest.mark.parametrize("name,tool_input", [
    ("Read", {"file_path": "p" * 5000}),
    ("Write", {"notebook_path": "n" * 5000}),
    ("Grep", {"pattern": "p" * 5000, "path": "products"}),
    ("Skill", {"skill": "s" * 5000}),
    ("Task", {"description": "d" * 5000}),
    ("WebFetch", {"url": "u" * 5000}),
])
def test_every_tool_argument_is_cut_like_a_long_command(
        monkeypatch, plain_lines, name, tool_input):
    """Only Bash and Task were bounded; a Grep pattern printed a 517-cell line
    and a path is exactly as unbounded as a command."""
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_claude_event({"type": "assistant", "message": {"content": [
        {"type": "tool_use", "name": name, "input": tool_input}]}}, True)

    line, = plain_lines
    assert textwidth.cell_width(line) == fits


# --- the codex line that carries TWO variable fields --------------------------
#
# `✗ exit 1: <command> — <output>` has to fit both in one budget. Capping the
# command at half of it and giving the output the rest left half the row blank
# whenever there was no output — the very defect the width change is about.

def _codex_completed(command, output="", exit_code=1):
    item = {"type": "command_execution", "command": command,
            "aggregated_output": output}
    if exit_code is not None:
        item["exit_code"] = exit_code
    return {"type": "item.completed", "item": item}


def test_a_command_with_no_output_gets_the_whole_line(monkeypatch, plain_lines):
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_codex_event(_codex_completed(LONG_COMMAND))

    line, = plain_lines
    assert textwidth.cell_width(line) == fits


def test_a_short_output_lends_its_room_to_the_command(monkeypatch, plain_lines):
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_codex_event(_codex_completed(LONG_COMMAND, "not found"))

    line, = plain_lines
    assert textwidth.cell_width(line) == fits
    assert line.endswith(" — not found")     # kept whole, and the command took
    assert "…" in line                       # the rest of the row


def test_two_long_fields_share_the_line_evenly(monkeypatch, plain_lines):
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_codex_event(
        _codex_completed(LONG_COMMAND, "y" * 500))

    line, = plain_lines
    command, output = line.split(" — ")
    assert textwidth.cell_width(line) == fits
    assert command.endswith("…") and output.endswith("…")   # both were cut
    assert abs(len(command) - len(output)) < 30             # ...about evenly


def test_a_missing_exit_code_does_not_shrink_the_line(monkeypatch, plain_lines):
    """The head is measured from what is printed: no code, no `exit N: `."""
    fits = _terminal(monkeypatch, 300)

    cyclecore._render_codex_event(_codex_completed(LONG_COMMAND,
                                                   exit_code=None))

    line, = plain_lines
    assert "exit" not in line
    assert textwidth.cell_width(line) == fits


# --- the other copy of the same line: the one with the colour in it -----------
#
# `LineWriter` builds every line twice — the plain copy the mirror log records
# and the markup copy the screen renders. Everything above reads the plain one,
# so the styling was pinned by nothing at all: deleting the style wrapper from
# `line`, the gear and name tags from `tool` and the mark tag from `mark` left
# the suite green while every verdict, error and tool call went out grey.
#
# These read the markup copy THROUGH rich rather than matching tags in it. What
# a pin owes the reader here is that the text arrives styled and that the styling
# lands on the right piece of the line — not how the tag was spelled, which a
# rewrite is free to change without breaking anything a human sees.


@pytest.fixture
def line_pairs(monkeypatch):
    """Both copies of every rendered line, as `(plain, markup)` pairs."""
    pairs = []
    monkeypatch.setattr(console, "print_markup",
                        lambda plain, markup: pairs.append((plain, markup)))
    return pairs


def _style_over(markup, part):
    """The rich style covering `part` of the rendered line; "" when unstyled."""
    rich_markup = pytest.importorskip("rich.markup")
    text = rich_markup.render(markup)
    start = text.plain.index(part)
    end = start + len(part)
    return " ".join(str(span.style) for span in text.spans
                    if span.start <= start and end <= span.end)


def test_a_styled_line_reaches_the_screen_in_its_style(line_pairs):
    """Worker verdicts, the bold-red error lines and the stop/pause notices are
    all this one shape carrying a style."""
    console.LINES.line("run stopped: stop file", style="bold red")

    plain, markup = line_pairs[0]
    assert plain == "run stopped: stop file"        # the log copy stays plain
    assert _style_over(markup, plain) == "bold red"


def test_a_tool_line_keeps_its_coloured_glyph_and_bold_name(line_pairs):
    console.LINES.tool("Bash", "$ pytest -q")

    _, markup = line_pairs[0]
    assert _style_over(markup, "⚙")                  # the glyph is coloured
    assert "bold" in _style_over(markup, "Bash")     # the name is picked out
    assert _style_over(markup, "$ pytest -q") == ""  # the detail is not


def test_a_mark_line_styles_the_glyph_and_not_the_body(line_pairs):
    """The ✓/✗ is the outcome, so it carries the colour; the body beside it is a
    command and its output, which a style would only make harder to read."""
    console.LINES.mark("✗", "red", "exit 1: pytest -q")

    _, markup = line_pairs[0]
    assert _style_over(markup, "✗") == "red"
    assert _style_over(markup, "exit 1: pytest -q") == ""
