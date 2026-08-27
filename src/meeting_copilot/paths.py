"""Central place for all filesystem paths so every module agrees on layout.

Handles both a source checkout and a PyInstaller-frozen .app: bundled,
read-only assets (configs/) resolve relative to `sys._MEIPASS` when frozen;
mutable per-user state (data/, logs/, .env) moves to
~/Library/Application Support/meeting-copilot so a read-only .app bundle
never needs to write inside itself.
"""

import os
import sys
from pathlib import Path

_FROZEN = getattr(sys, "frozen", False)


def _bundle_root() -> Path:
    if _FROZEN:
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def _user_data_root() -> Path:
    if _FROZEN:
        return Path.home() / "Library" / "Application Support" / "meeting-copilot"
    return _bundle_root()


PROJECT_ROOT = _bundle_root()
CONFIGS_DIR = PROJECT_ROOT / "configs"

_USER_ROOT = _user_data_root()
DATA_DIR = _USER_ROOT / "data"
LOGS_DIR = _USER_ROOT / "logs"

SETTINGS_FILE = CONFIGS_DIR / os.environ.get("MEETING_COPILOT_SETTINGS_FILE", "settings.yaml")
LOGGING_FILE = CONFIGS_DIR / "logging.yaml"
TOPICS_FILE = CONFIGS_DIR / "topics.yaml"
ENV_FILE = _USER_ROOT / ".env"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
