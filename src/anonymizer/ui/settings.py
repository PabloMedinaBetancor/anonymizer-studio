"""The user's settings: validated, kept in the user folder, never in demo mode."""

from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any, Dict, Optional

import anonymizer

DEFAULTS: Dict[str, Any] = {
    "language": None,  # None until the person chooses one: the page then follows the browser's language ("en" is a choice like any other)
    "faces": True,
    "plates": True,
    "thoroughness": "standard",
    "detection": "careful",
    "gap_fill": "camera",
    "blur": "gaussian",
    "strength": 1.0,
    "extra_margin": 0.0,
    "remove_gps": True,
    "default_destination": os.path.join(os.path.expanduser("~"), "Anonymizer"),
    "threads": 0,
}
CHOICES = {
    "thoroughness": (
        "standard",
    ),  # "maximum" is a value of the Python API, not offered by the Studio (not repeatable on video, no better than standard)
    "detection": ("fast", "careful"),  # Deep is an experimental value of the Python API, not offered by the Studio
    "gap_fill": ("camera", "straight", "hold"),
    "blur": ("gaussian", "pixelate", "solid"),
}
RANGES = {"strength": (1.0, 2.0), "extra_margin": (0.0, 0.5), "threads": (0, 64)}
BOOLEANS = ("faces", "plates", "remove_gps")
OPTION_FIELDS = (
    "faces",
    "plates",
    "thoroughness",
    "detection",
    "gap_fill",
    "blur",
    "strength",
    "extra_margin",
    "remove_gps",
)


class InvalidSetting(ValueError):
    def __init__(self, field: str, detail: str):
        super().__init__(f"{field}: {detail}")
        self.field = field
        self.detail = detail


def folder_path(text: str) -> str:
    """The folder a person typed, as a full path: `~` is their home, and a path that does not start at the root is taken from their home
    folder (it used to be taken from wherever the server happened to be started, and the files were written there)."""
    expanded = os.path.expanduser(text.strip() or "~")
    if not os.path.isabs(expanded) and not (
        len(expanded) > 1 and expanded[1] == ":"
    ):  # a Windows drive letter is already absolute
        expanded = os.path.join(os.path.expanduser("~"), expanded)
    return os.path.abspath(expanded)


def user_folder() -> str:
    if sys.platform == "win32":
        return os.path.join(os.environ.get("APPDATA") or os.path.expanduser("~"), "AnonymizerStudio")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "anonymizer-studio")


def validate(changes: Dict[str, Any], languages: Optional[list] = None) -> Dict[str, Any]:
    """The checked and normalised version of `changes`; raises InvalidSetting naming the field."""
    clean: Dict[str, Any] = {}
    for key, value in changes.items():
        if key not in DEFAULTS:
            raise InvalidSetting(key, "unknown setting")
        if key in BOOLEANS:
            if not isinstance(value, bool):
                raise InvalidSetting(key, "must be true or false")
        elif key in CHOICES:
            if value not in CHOICES[key]:
                raise InvalidSetting(key, f"must be one of {CHOICES[key]}")
        elif key in RANGES:
            low, high = RANGES[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
                raise InvalidSetting(key, f"must be a number from {low} to {high}")
            value = int(value) if key == "threads" else float(value)
        elif key == "language":
            if not isinstance(value, str) or (languages is not None and value not in languages):
                raise InvalidSetting(key, "unknown language")
        elif key == "default_destination":
            if not isinstance(value, str) or not value.strip() or len(value) > 4096 or any(ord(c) < 32 for c in value):
                raise InvalidSetting(
                    key, "must be a folder path"
                )  # a NUL in it failed every upload's export, much later and as "engine crashed"
            value = folder_path(value)
        clean[key] = value
    return clean


class Settings:
    """The current values. With `folder` they are loaded from settings.json there and saved to it on every change;
    without, they live in memory (demo mode, tests)."""

    def __init__(self, folder: Optional[str] = None, languages: Optional[list] = None):
        self.folder = folder
        self.languages = languages
        self._lock = threading.Lock()
        self.save_failed = False  # the folder cannot be written: the settings then last as long as the server does
        self._values = dict(DEFAULTS)
        if folder:
            self._load()

    def _path(self) -> str:
        return os.path.join(self.folder, "settings.json")

    def _load(self) -> None:
        try:
            with open(self._path(), encoding="utf-8") as f:
                stored = json.load(f)
        except (OSError, ValueError, RecursionError):
            return  # no file yet, or one we cannot read: the defaults stay
        if not isinstance(stored, dict):
            return  # valid JSON that is not a set of settings (a list, null, a number): not to be trusted, and not to stop the Studio from starting
        merged = dict(self._values)
        for key, value in stored.items():
            if key not in DEFAULTS:
                continue
            try:
                merged.update(validate({key: value}, self.languages))
            except InvalidSetting:
                continue  # one value that is no longer valid costs that setting, not the person's others
        if not (merged["faces"] or merged["plates"]):
            # nothing to look for is not a state to start in
            merged["faces"], merged["plates"] = DEFAULTS["faces"], DEFAULTS["plates"]
        self._values = merged

    def get(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def update(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        clean = validate(changes, self.languages)
        with self._lock:
            if not dict(self._values, **clean)["faces"] and not dict(self._values, **clean)["plates"]:
                raise InvalidSetting("faces", "choose faces, plates or both")
            self._values.update(clean)
            values = dict(self._values)
        if self.folder:
            try:
                os.makedirs(self.folder, exist_ok=True)
                tmp = self._path() + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(values, f, indent=1)
                os.replace(tmp, self._path())
                self.save_failed = False
            except (
                OSError
            ):  # a read-only home or a locked profile: the change still applies, and the person is told once
                self.save_failed = True
        return values

    def options(self, **override: Any) -> "anonymizer.Options":
        """The settings as anonymizer.Options, with `override` on top."""
        values = {k: v for k, v in self.get().items() if k in OPTION_FIELDS}
        values.update(validate(override))
        if not (values["faces"] or values["plates"]):
            raise InvalidSetting("faces", "choose faces, plates or both")
        return anonymizer.Options(**values)
