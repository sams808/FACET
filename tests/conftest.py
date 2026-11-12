"""Shared test helpers.

One thing lives here, and it is here because eight test files needed it and
none of them had it.
"""
from __future__ import annotations

import pytest


def dispose(*widgets) -> None:
    """Close widgets and actually destroy them.

    ``close()`` only hides a window, and ``deleteLater()`` schedules its
    destruction for the next time the event loop delivers a DeferredDelete
    event -- which ``processEvents()`` does **not** do at the top level. So a
    fixture that closed its window left every widget of it alive for the whole
    session, and anything that restyles the QApplication then has to re-polish
    all of them.

    Measured on the main window, which is 1968 widgets: building and closing it
    five times left 11808 widgets alive and took a single application-wide
    restyle from 0.31 s to 92.59 s. With the deferred deletes delivered, zero
    widgets are left and the restyle is immediate. In the suite that was 499 s
    in one test of tests/test_window.py.
    """
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication

    for widget in widgets:
        if widget is None:
            continue
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass                    # already destroyed by its parent
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


def windows_look() -> bool:
    """True when widgets measure as a Windows user sees them.

    The sizes the layout tests state in pixels -- a page no wider than its
    470 px column, a window that can be made 730 px tall -- are those of the
    Windows style with the Segoe UI font, which is what FACET is drawn for.
    They hold on the native Windows platform. On the offscreen platform Qt
    falls back to the Fusion style and, on Windows, to a font database with
    no fonts in it (a short label measures 360 px); on Linux and macOS the
    interface font is wider and taller. There the same assertions measure
    the font, not the layout.

    A test asks this before it asserts a pixel size. Everything else it
    asserts runs everywhere. Needs the QApplication to exist.
    """
    import sys

    from PySide6.QtGui import QGuiApplication

    return sys.platform == "win32" and QGuiApplication.platformName() == "windows"


def has_vector_marks(svg_text: str) -> bool:
    """True when an SVG holds drawn geometry rather than an embedded picture.

    Which element Qt writes for a stroked line depends on its version: Qt 6.9
    writes ``<path>``, Qt 6.12 writes ``<polyline>`` (and ``<polygon>`` or
    ``<circle>`` for closed shapes). A test that asked for ``<path>`` alone
    passed on one and failed on the other with the same curve in the file.
    """
    return any(tag in svg_text for tag in
               ("<path", "<polyline", "<polygon", "<line", "<circle", "<ellipse"))


# ---------------------------------------------------------------------------
# finding the reference structures
# ---------------------------------------------------------------------------
#
# The tests used to name files by path. That made them hostage to how the
# collection happens to be arranged: reorganising it into folders by chemical
# system, and keeping one copy of each structure under its most descriptive
# name, silently skipped a quarter of the suite. A database code does not move,
# so the tests ask for that instead.

import os as _os0
from pathlib import Path as _Path

REFERENCE_DIR = _Path(__file__).resolve().parent / "data" / "reference"

#: The whole collection of CIF files the sweeping tests read ("for every
#: structure of the collection ..."). It is on one machine and is not public;
#: FACET_REFERENCE_CIFS names a copy elsewhere. Where the folder is absent
#: those tests skip.
COLLECTION = _Path(_os0.environ.get(
    "FACET_REFERENCE_CIFS", r"C:\Users\samso\Desktop\WSU_work\XRD\cif"))

_CIF_ROOT = None
_BY_CODE: dict[str, object] = {}
_BY_NAME: dict[str, object] = {}


def _index() -> dict:
    """Every CIF under the reference collection, by the code in its name.

    Built once. `_duplicates` is skipped: it holds the copies the collection
    deliberately does not use, and a test that picked one up would be reading a
    file nobody else reads.
    """
    global _CIF_ROOT
    if _BY_CODE:
        return _BY_CODE
    from pathlib import Path

    import re

    # The structures the tests name one by one travel with the repository
    # (tests/data/reference, all from the Crystallography Open Database), so
    # a checkout anywhere runs them, and runs the same bytes everywhere. They
    # are indexed first: a name or a code found there is never looked for in
    # the collection.
    for path in sorted(REFERENCE_DIR.glob("*.cif")):
        _BY_NAME.setdefault(path.name, path)
        for code in re.findall(r"\d{6,8}", path.stem):
            _BY_CODE.setdefault(code, path)

    # then the collection, where there is one (COLLECTION, above)
    _CIF_ROOT = Path(COLLECTION)
    if not _CIF_ROOT.is_dir():
        return _BY_CODE

    for path in _CIF_ROOT.rglob("*.cif"):
        if "_duplicates" in path.parts:
            continue
        _BY_NAME.setdefault(path.name, path)
        for code in re.findall(r"\d{6,8}", path.stem):
            _BY_CODE.setdefault(code, path)
    return _BY_CODE


def sample_cif(code: str, name_hint: str = ""):
    """The reference structure with this name, or failing that this code.

    The name is tried first because a code is not always unique: COD 1010622 is
    two settings of BiI3, held as two files, and a lookup by code alone hands
    back the same one twice. The code is the fallback, for a file that has been
    renamed or refiled.

    Returns a Path, or one that does not exist so that the caller's own skipif
    fires with its usual message.
    """
    from pathlib import Path

    _index()
    if name_hint and name_hint in _BY_NAME:
        return _BY_NAME[name_hint]
    found = _BY_CODE.get(str(code))
    if found is not None:
        return found
    return Path(_CIF_ROOT or ".") / (name_hint or f"{code}.cif")


# Preferences live in a JSON file per user (facet.core.prefs). The suite
# points them at a folder of its own, so no test reads or writes what the
# person running it has chosen -- the lesson of the QSettings tests that
# once rewrote a real registry.
import os as _os
import tempfile as _tempfile

_PREFS_DIR = _tempfile.mkdtemp(prefix="facet_prefs_")
_os.environ["FACET_PREFS_DIR"] = _PREFS_DIR


@pytest.fixture(autouse=True)
def _isolated_prefs(tmp_path_factory):
    """Each test starts with an empty preferences file of its own, in a
    folder of its own (not inside tmp_path, which some tests expect to be
    exactly what they wrote)."""
    folder = tmp_path_factory.mktemp("prefs")
    previous = _os.environ.get("FACET_PREFS_DIR")
    _os.environ["FACET_PREFS_DIR"] = str(folder)
    try:
        yield folder
    finally:
        _os.environ["FACET_PREFS_DIR"] = previous or _PREFS_DIR
