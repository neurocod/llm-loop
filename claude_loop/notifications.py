"""Opt-in user settings and completion sounds for unattended loop wrappers."""

import json
import os
import sys
from pathlib import Path
from typing import Mapping, Optional


SETTINGS_ENV = "CLAUDE_LOOP_SETTINGS"
SETTINGS_FILENAME = "settings.json"
COMPLETION_SOUND_KEY = "completion_sound"


class SettingsError(ValueError):
    """A present claude-loop settings file is malformed."""


def _reject_json_constant(value: str):
    raise ValueError(f"non-standard JSON value {value}")


def settings_path(*, environ: Optional[Mapping[str, str]] = None,
                  platform: Optional[str] = None,
                  home: Optional[Path] = None) -> Path:
    """Return the per-user settings path, with an environment override."""
    env = os.environ if environ is None else environ
    override = env.get(SETTINGS_ENV)
    if override:
        return Path(override).expanduser()

    active_platform = sys.platform if platform is None else platform
    if active_platform == "win32" and env.get("APPDATA"):
        config_home = Path(env["APPDATA"])
    elif env.get("XDG_CONFIG_HOME"):
        config_home = Path(env["XDG_CONFIG_HOME"])
    else:
        config_home = (Path.home() if home is None else Path(home)) / ".config"
    return config_home / "claude-loop" / SETTINGS_FILENAME


def load_settings(path: Optional[Path] = None) -> dict:
    """Read optional settings; an absent file is the all-disabled default."""
    resolved = settings_path() if path is None else Path(path)
    try:
        raw = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeError) as exc:
        raise SettingsError(f"cannot read {resolved}: {exc}") from exc

    try:
        data = json.loads(raw, parse_constant=_reject_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise SettingsError(f"invalid JSON in {resolved}: {exc}") from exc
    if not isinstance(data, dict):
        raise SettingsError(f"{resolved} must contain a JSON object")
    return data


def completion_sound_enabled(path: Optional[Path] = None) -> bool:
    """Whether the opt-in completion sound setting is exactly true."""
    data = load_settings(path)
    value = data.get(COMPLETION_SOUND_KEY, False)
    if not isinstance(value, bool):
        resolved = settings_path() if path is None else Path(path)
        raise SettingsError(
            f"{COMPLETION_SOUND_KEY!r} in {resolved} must be true or false")
    return value


def play_completion_sound() -> None:
    """Play one best-effort native completion sound without failing the run."""
    try:
        if sys.platform == "win32":
            import winsound
            winsound.MessageBeep(winsound.MB_ICONASTERISK)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception as exc:
        try:
            print(f"warning: completion sound failed: {exc}", file=sys.stderr)
        except Exception:
            pass
