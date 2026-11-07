"""Codex's shell wrapper read back into the command it wraps."""

from test_codex_agent import _no_badge  # noqa: F401 — see _no_badge docstring
import pytest
from llm_agents.codex import sandbox


@pytest.mark.parametrize("command, expected", [
    pytest.param("\"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe\" "
        "-Command 'try { [Console]::OutputEncoding=[System.Text.Encoding]::UTF8 } "
        "catch {}\nnpm test -- test/hash.test.ts'", "npm test -- test/hash.test.ts", id='powershell-encoding'),
    pytest.param("cmd.exe /d /s /c \"npm test\"", "npm test", id='cmd-wrapper'),
    pytest.param("npm test", "npm test", id='unwrapped'),
    pytest.param("powershell -Command 'echo '\"'\"'hi'\"'\"''", "echo 'hi'", id='posix-quotes'),
    pytest.param('"C:\\\\WINDOWS\\\\powershell.exe" -Command "python tools/try_patch.py '
        "--file C:\\\\w\\\\x.ts --old 'if (\"'!a) b;'\"' -- npm test\"", "python tools/try_patch.py --file C:\\w\\x.ts --old 'if (!a) b;' -- npm test", id='escaped-bang'),
    pytest.param('C:\\WINDOWS\\system32\\cmd.exe /d /s /c "npm test"', "npm test", id='raw-shell-path'),
])
def test_codex_shell_wrapper_words(command, expected):
    assert sandbox.unwrap_shell(command) == expected
