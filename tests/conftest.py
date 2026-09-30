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
