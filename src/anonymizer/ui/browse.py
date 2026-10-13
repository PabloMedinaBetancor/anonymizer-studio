"""Listing folders for the Import dialog: the browser cannot give disk paths, so the server lists them."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import anonymizer

from . import intake

MAX_ENTRIES = 2000
ROOT_FOLDERS = ("Desktop", "Documents", "Downloads", "Videos", "Pictures", "Movies")


def kind_of(name: str) -> str:
    """ "video", "photo" or "unsupported" (listed greyed, never silently skipped) from the extension; "other" for the rest."""
    ext = os.path.splitext(name)[1].lower()
    if ext in anonymizer.VIDEO_EXTENSIONS:
        return "video"
    if ext in anonymizer.PHOTO_EXTENSIONS:
        return "photo"
    if ext in anonymizer.UNSUPPORTED_EXTENSIONS:
        return "unsupported"
    return "other"


def roots() -> List[Dict[str, str]]:
    home = os.path.expanduser("~")
    found = [{"name": "Home", "path": home}]
    for name in ROOT_FOLDERS:
        path = os.path.join(home, name)
        if os.path.isdir(path):
            found.append({"name": name, "path": path})
    if sys.platform == "win32":
        found += [{"name": f"{c}:", "path": f"{c}:\\"} for c in "CDEFGHIJKLMNOPQRSTUVWXYZ" if os.path.isdir(f"{c}:\\")]
    return found


def count_entries(path: str) -> Optional[int]:
    try:
        return sum(1 for e in os.scandir(path) if not e.name.startswith("."))
    except OSError:
        return None


def listing(path: Optional[str]) -> Dict[str, Any]:
    """The folders and the media files directly inside `path` (the roots when there is no path). Raises NotADirectoryError
    for something that is not a folder and PermissionError for one that cannot be read."""
    if not path:
        return {"path": None, "parent": None, "roots": roots(), "entries": [], "truncated": False}
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise NotADirectoryError(path)
    folders: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    for entry in os.scandir(path):  # PermissionError goes to the caller
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                folder = {"name": entry.name, "kind": "dir"}
                items = count_entries(entry.path)
                if items is not None:
                    folder["items"] = items
                folders.append(folder)
                continue
            kind = kind_of(entry.name)
            if kind == "other":
                continue
            stat = entry.stat()
        except OSError:
            continue  # vanished or unreadable while listing: it is not offered
        item = {"name": entry.name, "kind": kind, "size_bytes": stat.st_size, "mtime": int(stat.st_mtime)}
        if kind == "unsupported":
            item["reason"] = intake.unsupported_code(entry.name)  # shown greyed, with why
        files.append(item)
    folders.sort(key=lambda e: e["name"].lower())
    files.sort(key=lambda e: e["name"].lower())
    entries = folders + files
    parent = os.path.dirname(path)
    return {
        "path": path,
        "parent": parent if parent != path else None,
        "roots": roots(),
        "entries": entries[:MAX_ENTRIES],
        "truncated": len(entries) > MAX_ENTRIES,
    }
