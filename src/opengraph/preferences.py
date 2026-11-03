"""Per-user preferences (theme, input mode, last folder, window size) in a JSON file.

The file lives in the platform's configuration folder, or in ``$OPENGRAPH_CONFIG_DIR``
when that is set (tests use it to stay out of the real profile). A missing or broken
file just means default preferences.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from opengraph.json_types import JSONValue, is_string_keyed_dict

log = logging.getLogger(__name__)

FILE_NAME = "preferences.json"


def config_dir() -> Path:
    override = os.environ.get("OPENGRAPH_CONFIG_DIR")
    if override:
        return Path(override)
    app_data = os.environ.get("FLET_APP_STORAGE_DATA")  # set in the Android and iOS apps
    if app_data:
        return Path(app_data)
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home()) / "OpenGraph"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "OpenGraph"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "opengraph"


class Preferences:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config_dir() / FILE_NAME
        self._values: dict[str, object] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        if is_string_keyed_dict(data):
            self._values = data

    def get_str(self, key: str, default: str) -> str:
        value = self._values.get(key)
        return value if isinstance(value, str) else default

    def get_bool(self, key: str, default: bool) -> bool:
        value = self._values.get(key)
        return value if isinstance(value, bool) else default

    def get_float(self, key: str) -> float | None:
        value = self._values.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)

    def set(self, key: str, value: JSONValue) -> None:
        if self._values.get(key) == value:
            return
        self._values[key] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._values, indent=2), encoding="utf-8")
        except OSError as exc:
            log.warning("Could not save preferences to %s: %s", self.path, exc)
