"""Preferences that survive a restart.

One JSON file per user, in the application data folder (``%APPDATA%/FACET``
on Windows, ``~/.config/FACET`` elsewhere), read on start and written on
every change. A plain file rather than QSettings: it is Qt-free, it can be
read by a person, and a test points it elsewhere with the
``FACET_PREFS_DIR`` environment variable instead of writing into the real
registry (which QSettings does silently).

Everything in it is a convenience -- the theme, the explorer's options,
the recent files, the dock layout, whether the last session reopens --
and ``forget_everything`` removes the file, after which the application
starts as it did the first time.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FILE_NAME = "prefs.json"
MAX_RECENT = 12


def folder() -> Path:
    """Where the preferences live; created on first write."""
    override = os.environ.get("FACET_PREFS_DIR")
    if override:
        return Path(override)
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "FACET"


def path() -> Path:
    return folder() / FILE_NAME


def load() -> dict:
    """The whole preferences dictionary; empty when there is none or it is
    unreadable (a damaged file is not an error at start-up)."""
    try:
        with open(path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save(data: dict) -> Path | None:
    """Write the dictionary; None when the folder cannot be written (a
    read-only profile), which the application treats as 'remember nothing'."""
    try:
        folder().mkdir(parents=True, exist_ok=True)
        tmp = path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str),
                       encoding="utf-8")
        tmp.replace(path())
        return path()
    except OSError:
        return None


def get(key: str, default: Any = None) -> Any:
    return load().get(key, default)


def set(key: str, value: Any) -> None:            # noqa: A001 - the natural name
    data = load()
    data[key] = value
    save(data)


def update(values: dict) -> None:
    data = load()
    data.update(values)
    save(data)


def forget_everything() -> bool:
    """Remove the file. True when something was removed."""
    try:
        path().unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        return False


# -- recent files ---------------------------------------------------------------

def recent_files() -> list[str]:
    files = get("recent_files", [])
    return [str(f) for f in files if isinstance(f, str)] if isinstance(files, list) else []


def add_recent(file: str | Path) -> list[str]:
    """Put ``file`` first, once, and keep the list to MAX_RECENT."""
    text = str(file)
    files = [f for f in recent_files() if os.path.normcase(f) != os.path.normcase(text)]
    files.insert(0, text)
    files = files[:MAX_RECENT]
    set("recent_files", files)
    return files


def forget_recent() -> None:
    set("recent_files", [])


# -- named layouts -----------------------------------------------------------------

def layouts() -> dict[str, str]:
    value = get("layouts", {})
    return {str(k): str(v) for k, v in value.items()} if isinstance(value, dict) else {}


def save_layout(name: str, state_b64: str) -> None:
    current = layouts()
    current[name] = state_b64
    set("layouts", current)


def delete_layout(name: str) -> None:
    current = layouts()
    current.pop(name, None)
    set("layouts", current)
