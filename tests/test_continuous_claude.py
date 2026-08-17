"""Tests for continuous_claude.py (port of the bats suite plus port-specific coverage)."""

import pytest


import continuous_claude as cc


# -- duration parsing ---------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("2h", 7200),
    ("30m", 1800),
    ("90s", 90),
    ("1h30m", 5400),
    ("1h30m45s", 5445),
    ("2H", 7200),
    (" 1h 30m ", 5400),
])
def test_parse_duration_valid(text, expected):
    assert cc.parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "abc", "1x", "h", "0s", "1h2x", None])
def test_parse_duration_invalid(text):
    assert cc.parse_duration(text) is None


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"),
    (None, "0s"),
    (45, "45s"),
    (60, "1m"),
    (3661, "1h1m1s"),
    (7200, "2h"),
    (5400, "1h30m"),
])
def test_format_duration(seconds, expected):
    assert cc.format_duration(seconds) == expected


# -- rate-limit detection -----------------------------------------------------

def test_rate_limit_not_detected_in_plain_error():
    assert cc.detect_rate_limit_wait_seconds("some unrelated failure") is None


def test_rate_limit_retry_after():
    text = "Error 429 Too Many Requests. Retry-After: 42"
    assert cc.detect_rate_limit_wait_seconds(text) == 42


def test_rate_limit_try_again_minutes():
    text = "rate limit exceeded, please try again in 2 minutes"
    assert cc.detect_rate_limit_wait_seconds(text) == 120


def test_rate_limit_try_again_seconds():
    text = "too many requests, wait 30 seconds"
    assert cc.detect_rate_limit_wait_seconds(text) == 30


def test_rate_limit_default_backoff():
    text = "rate_limit_error: something opaque"
    assert cc.detect_rate_limit_wait_seconds(text) == cc.RATE_LIMIT_DEFAULT_BACKOFF


def test_rate_limit_resets_at_is_bounded():
    wait = cc.detect_rate_limit_wait_seconds("Limit reached. Your limit resets at 5pm.")
    assert wait is not None and 0 < wait <= 86400


def test_parse_reset_time_with_minutes_and_tz():
    wait = cc.parse_reset_time_wait_seconds("resets at 11:30pm (America/Chicago)")
    assert wait is not None and 0 < wait <= 86400


# -- completion heuristic -----------------------------------------------------

def test_completion_heuristic_positive():
    assert cc.detect_positive_completion_heuristic("All tasks complete. Nothing else needed.")
    assert cc.detect_positive_completion_heuristic("there is NO REMAINING WORK")


def test_completion_heuristic_negative():
    assert not cc.detect_positive_completion_heuristic("made progress on the parser")
    assert not cc.detect_positive_completion_heuristic("")


# -- PR number extraction -----------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("https://github.com/o/r/pull/893", 893),
    ("Created PR #12 successfully", 12),
    ("no pr here", None),
])
def test_extract_pr_number(text, expected):
    assert cc.extract_pr_number(text) == expected


# -- provider result parsing --------------------------------------------------

def test_parse_claude_success():
    events = [{"type": "result", "is_error": False, "result": "done"}]
    assert cc.parse_agent_result("claude", events, False) == "success"


def test_parse_claude_error():
    events = [{"type": "result", "is_error": True, "result": "boom"}]
    assert cc.parse_agent_result("claude", events, False) == "claude_error"


def test_parse_invalid_json():
    assert cc.parse_agent_result("claude", [], False) == "invalid_json"
    assert cc.parse_agent_result("claude", [{"a": 1}], True) == "invalid_json"


def test_parse_codex_incomplete_and_error():
    message = {"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}
    assert cc.parse_agent_result("codex", [message], False) == "codex_incomplete"
    assert cc.parse_agent_result(
        "codex", [message, {"type": "turn.failed", "error": "x"}], False
    ) == "codex_error"
    assert cc.parse_agent_result(
        "codex", [message, {"type": "turn.completed", "usage": {}}], False
    ) == "success"


def test_extract_result_text():
    claude_events = [{"type": "result", "result": "final text"}]
    assert cc.extract_agent_result_text("claude", claude_events) == "final text"
    codex_events = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "a"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "b"}},
        {"type": "turn.completed", "usage": {}},
    ]
    assert cc.extract_agent_result_text("codex", codex_events) == "a\nb"


# -- cost extraction ----------------------------------------------------------

def make_cfg(**overrides):
    cfg = cc.Config(prompt="x", max_runs=1)
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def test_claude_cost():
    events = [{"type": "result", "total_cost_usd": 0.042}]
    assert cc.extract_agent_cost("claude", events, make_cfg()) == pytest.approx(0.042)
    assert cc.extract_agent_cost("claude", [{"type": "result"}], make_cfg()) is None


def test_codex_cost_requires_rates():
    events = [{"type": "turn.completed",
               "usage": {"input_tokens": 1000, "cached_input_tokens": 0, "output_tokens": 100}}]
    assert cc.extract_agent_cost("codex", events, make_cfg()) is None


def test_codex_cost_with_rates():
    events = [{"type": "turn.completed",
               "usage": {"input_tokens": 2_000_000, "cached_input_tokens": 1_000_000,
                          "output_tokens": 500_000}}]
    cfg = make_cfg(codex_input_cost=1.25, codex_output_cost=10.0, codex_cached_input_cost=0.125)
    # 1M uncached * 1.25 + 1M cached * 0.125 + 0.5M output * 10.0 = 1.25 + 0.125 + 5.0
    assert cc.extract_agent_cost("codex", events, cfg) == pytest.approx(6.375)


def test_codex_usage_summary():
    events = [{"type": "turn.completed",
               "usage": {"input_tokens": 10, "cached_input_tokens": 2, "output_tokens": 3}}]
    assert cc.extract_agent_usage_summary("codex", events) == (
        "Tokens: input 10, cached input 2, output 3"
    )
    assert cc.extract_agent_usage_summary("claude", events) == ""


# -- tool-use rendering -------------------------------------------------------

def test_claude_tool_detail_bash_strips_cwd_and_truncates():
    detail = cc.claude_tool_detail("Bash", {"command": "/repo/x.sh --flag\nsecond line"}, "/repo")
    assert detail == "x.sh --flag"
    long_cmd = "a" * 1500
    assert cc.claude_tool_detail("Bash", {"command": long_cmd}, "/repo").endswith("...")


def test_claude_tool_detail_read_with_offset():
    detail = cc.claude_tool_detail("Read", {"file_path": "/repo/src/a.py", "offset": 10}, "/repo")
    assert detail == "src/a.py (line 10)"


def test_claude_tool_detail_mcp_name():
    assert cc.claude_tool_detail("mcp__server__do_thing", {}, "/repo") == "server/do_thing"
    assert cc.claude_tool_emoji("mcp__server__do_thing") == "🔌"


def test_render_codex_command_events():
    started = {"type": "item.started",
               "item": {"type": "command_execution", "command": "/repo/run.sh"}}
    texts, tools = cc.render_codex_event(started, "/repo")
    assert not texts and tools == ["💻 run.sh"]
    completed = {"type": "item.completed",
                 "item": {"type": "command_execution", "command": "ls", "exit_code": 0}}
    _, tools = cc.render_codex_event(completed, "/repo")
    assert tools == ["📤 exit 0: ls"]


# -- config validation --------------------------------------------------------

def test_config_requires_prompt():
    with pytest.raises(SystemExit):
        cc.parse_config(["-m", "5", "--disable-commits"])


def test_config_requires_a_limit():
    with pytest.raises(SystemExit):
        cc.parse_config(["-p", "do things", "--disable-commits"])


def test_config_rejects_bad_provider():
    with pytest.raises(SystemExit):
        cc.parse_config(["-p", "x", "-m", "1", "--provider", "gemini", "--disable-commits"])


def test_config_rejects_bad_merge_strategy():
    with pytest.raises(SystemExit):
        cc.parse_config(["-p", "x", "-m", "1", "--merge-strategy", "fast-forward",
                         "--disable-commits"])


def test_config_codex_max_cost_requires_rates():
    with pytest.raises(SystemExit):
        cc.parse_config(["-p", "x", "--provider", "codex", "--max-cost", "5",
                         "--disable-commits"])


def test_config_parses_limits_and_forwards_unknown_flags():
    cfg = cc.parse_config([
        "-p", "goal", "-m", "5", "--max-cost", "2.50", "--max-duration", "1h30m",
        "--disable-commits", "--model", "claude-haiku-4-5",
        "--", "--extra-provider-flag",
    ])
    assert cfg.prompt == "goal"
    assert cfg.max_runs == 5
    assert cfg.max_cost == pytest.approx(2.5)
    assert cfg.max_duration == 5400
    assert not cfg.enable_commits
    assert cfg.extra_agent_flags == ["--model", "claude-haiku-4-5", "--extra-provider-flag"]


def test_config_review_prompt_default_when_bare_r():
    cfg = cc.parse_config(["-p", "x", "-m", "1", "--disable-commits", "-r"])
    assert cfg.review_prompt == cc.PROMPT_DEFAULT_REVIEWER
    cfg = cc.parse_config(["-p", "x", "-m", "1", "--disable-commits", "-r", "run tests"])
    assert cfg.review_prompt == "run tests"
    cfg = cc.parse_config(["-p", "x", "-m", "1", "--disable-commits"])
    assert cfg.review_prompt == ""


def test_config_dry_run_with_cost_only_gets_one_run():
    cfg = cc.parse_config(["-p", "x", "--max-cost", "5", "--dry-run", "--disable-commits"])
    assert cfg.max_runs == 1


def test_config_list_worktrees_needs_no_prompt():
    cfg = cc.parse_config(["--list-worktrees"])
    assert cfg.list_worktrees


# -- rate window --------------------------------------------------------------

def test_rate_window_prunes_and_sums():
    window = cc.RateWindow(window_seconds=100)
    now = 1_000_000.0
    window.add(1.0, now=now - 200)  # outside window
    window.add(2.0, now=now - 50)
    window.add(3.0, now=now - 10)
    window.prune(now)
    assert len(window.entries) == 2
    assert sum(v for _, v in window.entries) == pytest.approx(5.0)


# -- prompt assembly ----------------------------------------------------------

def test_enhanced_prompt_contains_goal_and_signal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = make_cfg(prompt="build the thing", completion_signal="DONE_SIGNAL",
                   enable_commits=False)
    loop = cc.ContinuousLoop(cfg)
    prompt = loop.build_enhanced_prompt()
    assert "build the thing" in prompt
    assert "DONE_SIGNAL" in prompt
    assert "COMPLETION_SIGNAL_PLACEHOLDER" not in prompt
    assert "Create a `SHARED_TASK_NOTES.md` file" in prompt


def test_enhanced_prompt_includes_existing_notes_and_knowledge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "SHARED_TASK_NOTES.md").write_text("previous notes here", encoding="utf-8")
    (tmp_path / "KNOW.md").write_text("durable knowledge here", encoding="utf-8")
    cfg = make_cfg(prompt="goal", knowledge_file="KNOW.md", enable_commits=False)
    loop = cc.ContinuousLoop(cfg)
    prompt = loop.build_enhanced_prompt()
    assert "previous notes here" in prompt
    assert "durable knowledge here" in prompt
    assert "Update the `SHARED_TASK_NOTES.md` file" in prompt
    assert "Update the `KNOW.md` file" in prompt


# -- iteration display --------------------------------------------------------

def test_iteration_display():
    loop = cc.ContinuousLoop(make_cfg(max_runs=5))
    assert loop.get_iteration_display(1) == "(1/5)"
    loop.extra_iterations = 2
    assert loop.get_iteration_display(3) == "(3/7)"
    loop_inf = cc.ContinuousLoop(make_cfg(max_runs=0))
    assert loop_inf.get_iteration_display(4) == "(4)"


# -- stop conditions ----------------------------------------------------------

def test_should_continue_max_runs():
    loop = cc.ContinuousLoop(make_cfg(max_runs=2))
    assert loop.should_continue()
    loop.successful_iterations = 2
    assert not loop.should_continue()


def test_should_continue_max_cost():
    loop = cc.ContinuousLoop(make_cfg(max_runs=0, max_cost=1.0))
    assert loop.should_continue()
    loop.total_cost = 1.0
    assert not loop.should_continue()


def test_should_continue_completion_threshold():
    loop = cc.ContinuousLoop(make_cfg(max_runs=0, completion_threshold=2))
    loop.completion_signal_count = 2
    assert not loop.should_continue()
