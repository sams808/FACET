"""The loaded-structures panel.

A checkbox per structure controls whether it is drawn; the selection controls
which one the analysis panel and the cutoff explorer report on. Those are two
different things and are deliberately separate: comparing four polymorphs means
seeing all four while reading one.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core import theme as theme_mod


def _dot(rgb) -> QIcon:
    pix = QPixmap(12, 12)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    c = QColor()
    c.setRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    p.setBrush(c)
    p.setPen(Qt.NoPen)
    p.drawEllipse(1, 1, 10, 10)
    p.end()
    return QIcon(pix)


class StructureList(QWidget):
    """Loaded structures: what is drawn, and what is being read."""

    activeChanged = Signal(int)
    visibilityChanged = Signal()
    overlayChanged = Signal(bool, float)
    removeRequested = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = None
        self._loading = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemChanged.connect(self._on_check)
        layout.addWidget(self.list, 1)

        self.overlay = QCheckBox("Show all at once")
        self.overlay.setToolTip(
            "Draw every ticked structure together. The selected one still "
            "drives the analysis panel.")
        self.overlay.toggled.connect(self._emit_overlay)
        layout.addWidget(self.overlay)

        row = QHBoxLayout()
        row.addWidget(QLabel("Spacing"))
        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.0, 200.0)
        self.spacing.setSingleStep(2.0)
        self.spacing.setSuffix(" Å")
        self.spacing.setToolTip(
            "0 superimposes the structures about a common origin, which is "
            "what comparing two refinements of one phase wants. Above 0 lays "
            "them out in a row.")
        self.spacing.valueChanged.connect(self._emit_overlay)
        row.addWidget(self.spacing, 1)
        layout.addLayout(row)

        buttons = QHBoxLayout()
        self.all_on = QPushButton("All")
        self.all_off = QPushButton("None")
        self.remove = QPushButton("Remove")
        self.all_on.clicked.connect(lambda: self._set_all(True))
        self.all_off.clicked.connect(lambda: self._set_all(False))
        self.remove.clicked.connect(
            lambda: self.removeRequested.emit(self.list.currentRow()))
        for b in (self.all_on, self.all_off, self.remove):
            buttons.addWidget(b)
        layout.addLayout(buttons)

    # -- state -------------------------------------------------------------
    def set_project(self, project) -> None:
        self.project = project
        self.refresh()

    def refresh(self) -> None:
        if self.project is None:
            return
        self._loading = True
        self.list.clear()
        cycle = theme_mod.SITE_CYCLE
        for i, entry in enumerate(self.project.entries):
            item = QListWidgetItem(entry.name)
            item.setToolTip(entry.label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if entry.visible else Qt.Unchecked)
            item.setIcon(_dot(cycle[entry.color_key % len(cycle)]))
            font = QFont(item.font())
            font.setBold(i == self.project.active)
            item.setFont(font)
            self.list.addItem(item)
        if self.project.active is not None:
            self.list.setCurrentRow(self.project.active)
        self.overlay.setChecked(self.project.overlay)
        self.spacing.setValue(self.project.overlay_spacing)
        self.spacing.setEnabled(self.project.overlay)
        self._loading = False

    # -- events ------------------------------------------------------------
    def _on_row(self, row: int) -> None:
        if self._loading or self.project is None or row < 0:
            return
        self.activeChanged.emit(row)

    def _on_check(self, item: QListWidgetItem) -> None:
        if self._loading or self.project is None:
            return
        row = self.list.row(item)
        self.project.toggle_visible(row, item.checkState() == Qt.Checked)
        self.visibilityChanged.emit()

    def _set_all(self, visible: bool) -> None:
        if self.project is None:
            return
        self.project.show_all(visible)
        self.refresh()
        self.visibilityChanged.emit()

    def _emit_overlay(self) -> None:
        if self._loading:
            return
        self.spacing.setEnabled(self.overlay.isChecked())
        self.overlayChanged.emit(self.overlay.isChecked(), self.spacing.value())
