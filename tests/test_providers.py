import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_loop import cyclecore
from claude_loop.cyclecore import AgentCommand, Driver
from claude_loop.providers import build_agent_argv, provider_spec


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
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"
    assert "--approve-for-me" in argv
    assert argv[argv.index("-C") + 1] == "/repo"
    assert argv[argv.index("--model") + 1] == "gpt-test"
    assert argv[-1] == "work"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        provider_spec("gemini")


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


def test_codex_dry_run_does_not_construct_claude_usage_source(
        tmp_path, monkeypatch, capsys):
    from claude_loop import usage

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
    assert "usage limit policy: unavailable for Codex CLI (provider stub)" in output
    assert "DRY-RUN: codex exec --json" in output
