"""What the window remembers between runs, and the rest of the window
batch: recent files, named layouts, the message log, the threshold as an
undo step, the change log, the problem report, the keyboard.

The preferences file is isolated per test by conftest (FACET_PREFS_DIR), so
nothing here touches the person's own choices.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import dispose, sample_cif

from facet.core import prefs

SAMPLE = sample_cif("1526458", "1526458_Bi2O3.cif")


# -- the store -------------------------------------------------------------------

def test_prefs_round_trip_and_forget(tmp_path):
    assert prefs.load() == {}
    prefs.set("theme", "Dark")
    prefs.update({"explorer_axis": "valence", "n": 3})
    assert prefs.get("theme") == "Dark" and prefs.get("n") == 3
    assert prefs.path().exists()
    assert prefs.forget_everything() is True
    assert prefs.load() == {} and prefs.forget_everything() is False


def test_recent_files_are_unique_newest_first_and_capped():
    for i in range(prefs.MAX_RECENT + 3):
        prefs.add_recent(f"C:/x/file{i}.cif")
    files = prefs.recent_files()
    assert len(files) == prefs.MAX_RECENT
    assert files[0].endswith(f"file{prefs.MAX_RECENT + 2}.cif")
    prefs.add_recent("c:/X/FILE3.CIF")           # the same file, another case
    assert sum(1 for f in prefs.recent_files() if f.lower().endswith("file3.cif")) == 1
    assert prefs.recent_files()[0].lower().endswith("file3.cif")
    prefs.forget_recent()
    assert prefs.recent_files() == []


def test_a_damaged_prefs_file_reads_as_empty():
    prefs.folder().mkdir(parents=True, exist_ok=True)
    prefs.path().write_text("{not json", encoding="utf-8")
    assert prefs.load() == {}
    prefs.set("a", 1)
    assert prefs.get("a") == 1


def test_layouts_are_named():
    prefs.save_layout("figure", "AAAA")
    prefs.save_layout("analysis", "BBBB")
    assert prefs.layouts() == {"figure": "AAAA", "analysis": "BBBB"}
    prefs.delete_layout("figure")
    assert list(prefs.layouts()) == ["analysis"]


# -- the window ------------------------------------------------------------------

@pytest.fixture
def window(qapp):
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    yield win
    win.close()
    dispose(win)


def _rows(win, title):
    bar = next(a for a in win.menuBar().actions()
               if a.text().replace("&", "") == title)
    return [a.text().replace("&", "") for a in bar.menu().actions()
            if not a.isSeparator()]


def test_the_menus_carry_the_new_items_within_twelve_rows(window):
    for title in ("File", "Edit", "View", "Help"):
        assert len(_rows(window, title)) <= 12, title
    assert "Open recent" in _rows(window, "File")
    assert "Preferences…" in _rows(window, "Edit")
    assert "Dock layouts" in _rows(window, "View")
    assert {"What changed", "Report a problem…"} <= set(_rows(window, "Help"))


def test_opening_a_file_puts_it_in_recent(window, qapp):
    window.load_many([str(SAMPLE)])
    qapp.processEvents()
    assert prefs.recent_files()[0].endswith(SAMPLE.name)
    window._fill_recent_menu()
    texts = [a.text() for a in window.recent_menu.actions() if not a.isSeparator()]
    assert texts[0] == SAMPLE.name and texts[-1] == "Forget the list"


def test_closing_writes_the_preferences_and_reopening_reads_them(qapp):
    prefs.set("tour_offered", True)
    from facet.ui import preview
    win = preview.PreviewWindow()
    win.load_many([str(SAMPLE)])
    qapp.processEvents()
    win.explorer_axis_actions["valence"].trigger()
    win.explorer_option_actions["show_bars"].setChecked(False)
    win.running_sum_column.setChecked(False)
    win.close()
    dispose(win)
    data = json.loads(prefs.path().read_text(encoding="utf-8"))
    assert data["explorer_axis"] == "valence"
    assert data["explorer_show_bars"] is False
    assert data["running_sum_column"] is False
    assert data["window_state"] and data["window_geometry"]

    again = preview.PreviewWindow()
    try:
        assert again.explorer.axis == "valence"
        assert not again.explorer.show_bars
        assert not again.running_sum_column.isChecked()
    finally:
        again.close()
        dispose(again)


def test_remember_layout_off_leaves_the_defaults(qapp):
    prefs.update({"tour_offered": True, "remember_layout": False,
                  "explorer_axis": "valence"})
    from facet.ui import preview
    win = preview.PreviewWindow()
    try:
        assert win.explorer.axis == "distance"
        win.close()
        assert "window_state" not in prefs.load()
    finally:
        dispose(win)


def test_the_last_session_reopens_when_asked(qapp):
    prefs.update({"tour_offered": True, "restore_session": True})
    from facet.ui import preview
    win = preview.PreviewWindow()
    win.load_many([str(SAMPLE)])
    qapp.processEvents()
    win.close()
    dispose(win)
    saved = prefs.get("last_session")
    assert saved and Path(saved).exists()
    again = preview.PreviewWindow()
    try:
        for _ in range(20):
            qapp.processEvents()
            if len(again.project) == 1:
                break
        assert len(again.project) == 1
        assert again.project.entries[0].name == SAMPLE.name
    finally:
        again.close()
        dispose(again)


def test_named_layouts_save_apply_and_reset(window, qapp):
    window.show()
    qapp.processEvents()
    window.save_layout("smoke")
    assert "smoke" in prefs.layouts()
    assert window.apply_layout("smoke") is True
    assert window.apply_layout("nothing") is False
    window.messages_dock.show()
    window._reset_layout()
    qapp.processEvents()
    window._fill_layout_menu()
    texts = [a.text().replace("&", "") for a in window.layout_menu.actions()
             if not a.isSeparator()]
    assert "Save this layout…" in texts and "smoke" in texts
    assert any(t.startswith("Forget smoke") for t in texts)


def test_status_messages_are_logged_with_a_time(window):
    window.statusBar().showMessage("one", 1000)
    window.statusBar().showMessage("two", 1000)
    stamps = [s for s, _ in window.messages()]
    texts = [t for _, t in window.messages()]
    assert texts[-2:] == ["one", "two"]
    assert all(len(s) == 8 and s[2] == ":" for s in stamps)
    assert "two" in window.messages_text.toPlainText()


def test_a_threshold_drag_becomes_one_undo_step(window, qapp):
    window.load_many([str(SAMPLE)])
    qapp.processEvents()
    assert not window.undo_action.isEnabled()
    window.explorer.set_threshold(0.2, emit=True)
    window.explorer.set_threshold(0.25, emit=True)
    window._push_threshold_history()
    assert window.undo_action.isEnabled()
    assert "0.2500" in window.undo_action.text()
    window.undo_action.trigger()
    qapp.processEvents()
    assert window.project.v_bond == pytest.approx(0.075)


def test_the_problem_report_goes_to_the_clipboard(window, qapp):
    from PySide6.QtWidgets import QApplication, QMessageBox
    window.statusBar().showMessage("something happened", 1000)
    shown = []
    original = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **k: shown.append(a) or QMessageBox.Ok)
    try:
        window._report_problem()
    finally:
        QMessageBox.information = original
    text = QApplication.clipboard().text()
    assert "FACET" in text and "something happened" in text
    assert shown


def test_the_change_log_opens_from_the_bundled_file(window, qapp):
    window._show_changelog()
    qapp.processEvents()
    dialog = getattr(window, "_changelog_dialog", None)
    assert dialog is not None and dialog.isVisible()
    assert "What changed" in dialog.browser.toPlainText()
    dialog.close()


def test_the_first_run_offers_the_tour_once(qapp):
    from PySide6.QtWidgets import QMessageBox
    from facet.ui import preview
    asked = []
    original = QMessageBox.question
    QMessageBox.question = staticmethod(lambda *a, **k: asked.append(a) or QMessageBox.No)
    try:
        import time
        win = preview.PreviewWindow(offer_tour=True)
        end = time.monotonic() + 5
        while time.monotonic() < end and not asked:
            qapp.processEvents()
            time.sleep(0.05)
        win.close()
        dispose(win)
        assert asked and prefs.get("tour_offered") is True
        again = preview.PreviewWindow(offer_tour=True)
        end = time.monotonic() + 1.5
        while time.monotonic() < end:
            qapp.processEvents()
            time.sleep(0.05)
        again.close()
        dispose(again)
        assert len(asked) == 1
    finally:
        QMessageBox.question = original


def test_every_shortcut_is_unique_and_escape_closes_the_dialogs(window, qapp):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction
    from PySide6.QtTest import QTest
    seen = {}
    for action in window.findChildren(QAction):
        key = action.shortcut().toString()
        if key:
            assert key not in seen, (key, action.text(), seen[key])
            seen[key] = action.text()
    from facet.ui.preferences_dialog import PreferencesDialog
    from facet.ui.report_dialog import ReportDialog
    for dialog in (PreferencesDialog(window), ReportDialog(window)):
        dialog.show()
        qapp.processEvents()
        QTest.keyClick(dialog, Qt.Key_Escape)
        qapp.processEvents()
        assert not dialog.isVisible()
        dispose(dialog)
