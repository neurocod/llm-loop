import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claude_loop.notifications import (
    SettingsError,
    completion_sound_enabled,
    settings_path,
)
from claude_loop import notifications


def test_settings_path_uses_appdata_on_windows(tmp_path):
    assert settings_path(
        environ={"APPDATA": str(tmp_path)}, platform="win32",
    ) == tmp_path / "claude-loop" / "settings.json"


def test_settings_path_honours_explicit_override(tmp_path):
    target = tmp_path / "custom.json"
    assert settings_path(
        environ={"CLAUDE_LOOP_SETTINGS": str(target)}, platform="win32",
    ) == target


def test_settings_path_uses_xdg_then_home(tmp_path):
    assert settings_path(
        environ={"XDG_CONFIG_HOME": str(tmp_path)}, platform="linux",
    ) == tmp_path / "claude-loop" / "settings.json"
    assert settings_path(
        environ={}, platform="linux", home=tmp_path,
    ) == tmp_path / ".config" / "claude-loop" / "settings.json"


def test_completion_sound_is_disabled_without_settings(tmp_path):
    assert completion_sound_enabled(tmp_path / "missing.json") is False


def test_completion_sound_requires_explicit_true(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"completion_sound": True}), encoding="utf-8")

    assert completion_sound_enabled(path) is True


@pytest.mark.parametrize("contents", [
    "[]",
    "not json",
    '{"completion_sound": 1}',
    '{"completion_sound": true, "other": NaN}',
])
def test_invalid_settings_are_reported(tmp_path, contents):
    path = tmp_path / "settings.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(SettingsError):
        completion_sound_enabled(path)


def test_windows_completion_sound_uses_native_message_beep(monkeypatch):
    calls = []
    fake_winsound = SimpleNamespace(
        MB_ICONASTERISK=64,
        MessageBeep=lambda kind: calls.append(kind),
    )
    monkeypatch.setattr(notifications.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "winsound", fake_winsound)

    notifications.play_completion_sound()

    assert calls == [64]


def test_completion_sound_cannot_fail_when_output_streams_are_broken(monkeypatch):
    class BrokenStream:
        def write(self, _text):
            raise BrokenPipeError("closed")

        def flush(self):
            raise BrokenPipeError("closed")

    monkeypatch.setattr(notifications.sys, "platform", "linux")
    monkeypatch.setattr(notifications.sys, "stdout", BrokenStream())
    monkeypatch.setattr(notifications.sys, "stderr", BrokenStream())

    notifications.play_completion_sound()
