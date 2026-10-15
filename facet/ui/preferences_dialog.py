"""Edit > Preferences…: the few things the application remembers between
runs, in one place, with a button that makes it forget them all.

What is remembered lives in ``facet.core.prefs`` (a JSON file per user).
This dialog reads it, shows it, and writes it back on OK; nothing here
changes a number in any analysis.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import prefs


class PreferencesDialog(QDialog):
    """Remember the session, offer the tour, the explorer's defaults."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setMinimumWidth(460)
        self.forgot = False

        layout = QVBoxLayout(self)
        where = QLabel(f"Kept in {prefs.path()}")
        where.setWordWrap(True)
        where.setStyleSheet("color: #777; font-size: 11px;")
        layout.addWidget(where)

        form = QFormLayout()
        self.restore_session = QCheckBox(
            "Reopen the last session at start")
        self.restore_session.setToolTip(
            "The files, the threshold, the styles and the view of the last "
            "session come back when the application starts with no file.")
        self.restore_session.setChecked(bool(prefs.get("restore_session", False)))
        form.addRow("", self.restore_session)

        self.remember_layout = QCheckBox("Remember the window layout and theme")
        self.remember_layout.setToolTip(
            "The docks, the splitters, the window size and the theme come "
            "back as they were left.")
        self.remember_layout.setChecked(bool(prefs.get("remember_layout", True)))
        form.addRow("", self.remember_layout)

        self.offer_tour = QCheckBox("Offer Tutorial 1 when the application starts empty")
        self.offer_tour.setChecked(bool(prefs.get("offer_tour", True)))
        form.addRow("", self.offer_tour)

        self.explorer_axis = QComboBox()
        self.explorer_axis.addItems(["distance", "bond valence"])
        self.explorer_axis.setCurrentIndex(
            0 if prefs.get("explorer_axis", "distance") == "distance" else 1)
        self.explorer_axis.setToolTip(
            "The horizontal axis the cutoff explorer opens with.")
        form.addRow("Explorer axis at start", self.explorer_axis)
        layout.addLayout(form)

        forget = QPushButton("Forget everything…")
        forget.setToolTip("Remove the preferences file: the recent files, the "
                          "layouts, the theme, every choice above. The "
                          "application then starts as it did the first time.")
        forget.clicked.connect(self._forget)
        layout.addWidget(forget)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict:
        return {
            "restore_session": self.restore_session.isChecked(),
            "remember_layout": self.remember_layout.isChecked(),
            "offer_tour": self.offer_tour.isChecked(),
            "explorer_axis": ("distance" if self.explorer_axis.currentIndex() == 0
                              else "valence"),
        }

    def accept(self) -> None:
        if not self.forgot:
            prefs.update(self.values())
        super().accept()

    def _forget(self) -> None:
        answer = QMessageBox.question(
            self, "Forget everything",
            "Remove every remembered preference, the recent files and the "
            "saved layouts?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer == QMessageBox.Yes:
            prefs.forget_everything()
            self.forgot = True
            self.restore_session.setChecked(False)
            self.remember_layout.setChecked(True)
            self.offer_tour.setChecked(True)
            self.explorer_axis.setCurrentIndex(0)
