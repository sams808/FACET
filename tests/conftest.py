"""Shared test helpers.

One thing lives here, and it is here because eight test files needed it and
none of them had it.
"""
from __future__ import annotations


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


# ---------------------------------------------------------------------------
# finding the reference structures
# ---------------------------------------------------------------------------
#
# The tests used to name files by path. That made them hostage to how the
# collection happens to be arranged: reorganising it into folders by chemical
# system, and keeping one copy of each structure under its most descriptive
# name, silently skipped a quarter of the suite. A database code does not move,
# so the tests ask for that instead.

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

    _CIF_ROOT = Path(r"C:\Users\samso\Desktop\WSU_work\XRD\cif")
    if not _CIF_ROOT.is_dir():
        return _BY_CODE
    import re

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
