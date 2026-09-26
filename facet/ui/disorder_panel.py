"""Choosing a disorder configuration.

A CIF may describe a structure that is not one structure: where a region of the
cell has two alternatives, the file lists both and tags them. Only one exists in
any unit cell.

This panel says what the file declared, shows how close the alternatives sit to
each other, and lets you pick. Every alternative is offered, including "all at
once" -- which is what a file's own authors drew and the only way to see the
overlap -- but a configuration is chosen by default, because drawing them
together puts atoms a fraction of an angstrom apart and makes every coordination
number in the structure wrong.

The panel does not say which alternative is right, or whether a disorder model is
the right description of the material.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import disorder as disorder_mod
from . import chrome


class DisorderPanel(QWidget):
    """One chooser per disorder assembly, plus what overlaps what."""

    changed = Signal(str)          # a configuration was chosen

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entry = None
        self._loading = False
        self._choosers: dict[str, QComboBox] = {}
        self._build_layout()

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        tiny = QFont(self.hint.font())
        tiny.setPointSizeF(max(7.0, tiny.pointSizeF() - 1.0))
        self.hint.setFont(tiny)
        chrome.mark_hint(self.hint)
        outer.addWidget(self.hint)

        self.choosers_box = QGroupBox("Configuration")
        self.choosers_layout = QVBoxLayout(self.choosers_box)
        self.choosers_layout.setSpacing(4)
        outer.addWidget(self.choosers_box)

        buttons = QHBoxLayout()
        self.first_button = QPushButton("First of each")
        self.first_button.clicked.connect(self._choose_first)
        self.all_button = QPushButton("Show every group at once")
        self.all_button.setToolTip(
            "What the file itself draws. The alternatives overlap, so atoms "
            "appear within a fraction of an angstrom of each other and the "
            "coordination numbers are not those of any real configuration.")
        self.all_button.clicked.connect(self._show_all)
        buttons.addWidget(self.first_button)
        buttons.addWidget(self.all_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        overlap = QGroupBox("How close the alternatives sit")
        box = QVBoxLayout(overlap)
        self.overlaps = QTableWidget(0, 5)
        self.overlaps.setHorizontalHeaderLabels(
            ["atom", "group", "atom", "group", "distance / Å"])
        self.overlaps.verticalHeader().setVisible(False)
        self.overlaps.setEditTriggers(QTableWidget.NoEditTriggers)
        self.overlaps.setAlternatingRowColors(True)
        font = QFont(self.overlaps.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        self.overlaps.setFont(font)
        box.addWidget(self.overlaps)
        self.overlap_note = QLabel("")
        self.overlap_note.setWordWrap(True)
        self.overlap_note.setFont(tiny)
        chrome.mark_hint(self.overlap_note)
        box.addWidget(self.overlap_note)
        outer.addWidget(overlap, 1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setFont(tiny)
        chrome.mark_hint(self.summary)
        outer.addWidget(self.summary)

    # -- content -----------------------------------------------------------
    def set_entry(self, entry) -> None:
        self.entry = entry
        self.reload()

    @property
    def disorder(self):
        return getattr(self.entry, "disorder", None)

    def reload(self) -> None:
        self._loading = True
        try:
            self._clear_choosers()
            disorder = self.disorder
            if self.entry is None or disorder is None or not disorder.present:
                self.hint.setText(
                    "This file declares no disorder groups — every site it "
                    "lists is present in the same unit cell."
                    if self.entry is not None else
                    "Load a structure to see whether it declares disorder.")
                self.first_button.setEnabled(False)
                self.all_button.setEnabled(False)
                self.overlaps.setRowCount(0)
                self.overlap_note.setText("")
                self.summary.setText("")
                return

            self.hint.setText(
                f"The file declares {len(disorder.assemblies)} disorder "
                f"assembl{'ies' if len(disorder.assemblies) != 1 else 'y'}, "
                f"giving {disorder.n_alternatives} possible configurations. "
                "Only one exists in any unit cell. Occupancies are left as the "
                "file gave them and are not renormalised.")
            self.first_button.setEnabled(True)
            self.all_button.setEnabled(True)

            for name, assembly in sorted(disorder.assemblies.items()):
                if assembly.n_groups <= 1:
                    continue
                row = QHBoxLayout()
                label = name or "unnamed"
                row.addWidget(QLabel(f"assembly {label}"))
                chooser = QComboBox()
                chooser.addItem("all groups at once", "")
                for group in assembly.groups:
                    count = len(assembly.sites.get(group, []))
                    chooser.addItem(
                        f"group {group}  ({count} site"
                        f"{'s' if count != 1 else ''})", group)
                chosen = disorder.selected.get(name, "")
                index = chooser.findData(chosen)
                chooser.setCurrentIndex(max(index, 0))
                chooser.currentIndexChanged.connect(
                    lambda _=0, n=name, c=chooser: self._chose(n, c))
                row.addWidget(chooser, 1)
                holder = QWidget()
                holder.setLayout(row)
                self.choosers_layout.addWidget(holder)
                self._choosers[name] = chooser

            self._refresh_overlaps()
            self._refresh_summary()
        finally:
            self._loading = False

    def _clear_choosers(self) -> None:
        self._choosers.clear()
        while self.choosers_layout.count():
            item = self.choosers_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

    def _refresh_overlaps(self) -> None:
        pairs = disorder_mod.close_pairs_between_groups(
            self.entry.source, self.disorder)
        self.overlaps.setRowCount(len(pairs))
        for r, (label_a, group_a, label_b, group_b, distance) in enumerate(pairs):
            values = [label_a, group_a, label_b, group_b, f"{distance:.4f}"]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c == 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.overlaps.setItem(r, c, item)
        self.overlaps.resizeColumnsToContents()
        self.overlaps.horizontalHeader().setSectionResizeMode(
            4, QHeaderView.Stretch)
        if pairs:
            self.overlap_note.setText(
                f"{len(pairs)} pairs of atoms from different groups of the same "
                f"assembly lie within 1.2 Å of each other, the closest at "
                f"{pairs[0][4]:.4f} Å. That is what makes them "
                "alternatives rather than separate sites.")
        else:
            self.overlap_note.setText(
                "No two atoms from different groups of the same assembly lie "
                "within 1.2 Å of each other.")

    def _refresh_summary(self) -> None:
        disorder = self.disorder
        if disorder is None:
            self.summary.setText("")
            return
        shown = len(self.entry.structure.sites)
        total = len(self.entry.source.sites)
        lines = disorder.describe()
        self.summary.setText(
            f"{shown} of the file's {total} sites are in use. "
            + "  ".join(lines))

    # -- choosing ----------------------------------------------------------
    def _chose(self, assembly: str, chooser: QComboBox) -> None:
        if self._loading or self.entry is None:
            return
        group = chooser.currentData() or ""
        self.entry.choose_disorder(assembly, group)
        self._refresh_summary()
        self.changed.emit(
            f"chose group {group} of assembly {assembly or 'unnamed'}"
            if group else
            f"showed every group of assembly {assembly or 'unnamed'}")

    def _choose_first(self) -> None:
        if self.entry is None or self.disorder is None:
            return
        self.disorder.choose_first()
        self.entry.invalidate()
        self.reload()
        self.changed.emit("chose the first group of every assembly")

    def _show_all(self) -> None:
        if self.entry is None or self.disorder is None:
            return
        self.disorder.show_everything()
        self.entry.invalidate()
        self.reload()
        self.changed.emit("showed every disorder group at once")
