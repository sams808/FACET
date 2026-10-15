"""The Environments tab of the Model workspace.

The atoms of one species, grouped by what is around them within a stated
tolerance (``facet.core.md_environments``): one row per group with its
population, its representative and its means; a group highlighted on the
frame, its representative exported as a cluster, the table and the
per-atom rows exported as CSV; and the populations followed over the
frames of the trajectory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import md_environments as env_mod
from . import chrome
from .md_dialogs import _hint

COLUMNS = ("group", "n", "CN", "first shell", "second shell",
           "mean d / Å", "mean angle / °", "spread d", "spread °",
           "mean BVS", "void radius / Å", "representative")


class EnvironmentsPage(QWidget):
    """Controls, the groups table and the actions on a group."""

    runRequested = Signal()
    framesRequested = Signal()
    highlightRequested = Signal(object, str)     # rows, words
    representativeRequested = Signal(int)        # a frame row to export
    exportRequested = Signal(str)                # 'groups' or 'atoms'

    def __init__(self, parent=None, *, theme=None):
        super().__init__(parent)
        self.theme = theme
        self.groups: list = []
        self.envs: list = []
        self.per_atom: dict = {}
        column = QVBoxLayout(self)
        column.setContentsMargins(8, 8, 8, 8)
        column.setSpacing(6)
        lead = _hint("The atoms of one species grouped by their surroundings "
                     "at the strip's threshold: the sorted bond lengths and "
                     "the angles at the centre agree within the tolerances, "
                     "and the shells' compositions agree exactly. The "
                     "tolerances are yours; the counts are reported as they "
                     "come out.")
        column.addWidget(lead)

        row = QHBoxLayout()
        row.addWidget(QLabel("Species"))
        self.species = QComboBox()
        self.species.setToolTip("The element whose environments are grouped")
        row.addWidget(self.species)
        row.addWidget(QLabel("Qn"))
        self.qn = QSpinBox()
        self.qn.setRange(-1, 8)
        self.qn.setValue(-1)
        self.qn.setSpecialValueText("any")
        self.qn.setToolTip("Only the formers with this Qn (any: every atom "
                           "of the species)")
        row.addWidget(self.qn)
        row.addWidget(QLabel("tolerance"))
        self.tol_d = QDoubleSpinBox()
        self.tol_d.setRange(0.001, 2.0)
        self.tol_d.setDecimals(3)
        self.tol_d.setSingleStep(0.01)
        self.tol_d.setValue(0.05)
        self.tol_d.setSuffix(" Å")
        self.tol_d.setToolTip("How far any sorted bond length may sit from "
                              "the group's mean")
        row.addWidget(self.tol_d)
        self.tol_a = QDoubleSpinBox()
        self.tol_a.setRange(0.1, 90.0)
        self.tol_a.setDecimals(1)
        self.tol_a.setSingleStep(1.0)
        self.tol_a.setValue(8.0)
        self.tol_a.setSuffix(" °")
        self.tol_a.setToolTip("How far any sorted angle may sit from the "
                              "group's mean")
        row.addWidget(self.tol_a)
        row.addStretch(1)
        column.addLayout(row)
        row = QHBoxLayout()
        self.second = QCheckBox("same second shell")
        self.second.setToolTip("Require the composition of the next-nearest "
                               "shell to agree exactly as well")
        row.addWidget(self.second)
        row.addStretch(1)
        column.addLayout(row)

        buttons = QHBoxLayout()
        self.run = QPushButton("Group this frame")
        self.run.clicked.connect(self.runRequested)
        buttons.addWidget(self.run)
        self.frames = QPushButton("Populations over the frames…")
        self.frames.setToolTip("Group every frame of the trajectory the same "
                               "way and plot each group's population against "
                               "the frame (the groups are those of this frame)")
        self.frames.clicked.connect(self.framesRequested)
        buttons.addWidget(self.frames)
        buttons.addStretch(1)
        self.export_groups = QPushButton("Groups as CSV…")
        self.export_groups.clicked.connect(lambda: self.exportRequested.emit("groups"))
        buttons.addWidget(self.export_groups)
        self.export_atoms = QPushButton("Atoms as CSV…")
        self.export_atoms.setToolTip("Every atom described: its environment, "
                                     "its group, its bond-valence sum and free "
                                     "volume")
        self.export_atoms.clicked.connect(lambda: self.exportRequested.emit("atoms"))
        buttons.addWidget(self.export_atoms)
        column.addLayout(buttons)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(list(COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setToolTip("Click a group to highlight its members on the "
                              "frame; double-click to export its "
                              "representative as XYZ")
        self.table.itemClicked.connect(self._on_click)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        column.addWidget(self.table, 1)

        self.summary = _hint("No grouping yet.")
        column.addWidget(self.summary)
        for w in (self.export_groups, self.export_atoms, self.frames):
            w.setEnabled(False)

    # -- state ------------------------------------------------------------------
    def set_species(self, species) -> None:
        current = self.species.currentText()
        self.species.blockSignals(True)
        self.species.clear()
        self.species.addItems(list(species))
        i = self.species.findText(current)
        self.species.setCurrentIndex(i if i >= 0 else 0)
        self.species.blockSignals(False)
        chrome.fit_combo(self.species)

    def settings(self) -> dict:
        return {"element": self.species.currentText(),
                "qn": None if self.qn.value() < 0 else int(self.qn.value()),
                "tol_distance_ang": float(self.tol_d.value()),
                "tol_angle_deg": float(self.tol_a.value()),
                "same_second_shell": self.second.isChecked()}

    def show_groups(self, envs, groups, per_atom: dict, words: str) -> None:
        self.envs, self.groups, self.per_atom = list(envs), list(groups), dict(per_atom)
        table = self.table
        table.setRowCount(0)
        for g in self.groups:
            r = table.rowCount()
            table.insertRow(r)
            cells = [
                g.label, str(g.n), str(g.signature[1]),
                " ".join(f"{e}{c}" for e, c in g.signature[3]),
                " ".join(f"{e}{c}" for e, c in g.signature[4]) if len(g.signature) > 4 else "",
                " ".join(f"{d:.3f}" for d in g.mean_distances_ang),
                f"{g.mean_angles_deg.mean():.1f}" if len(g.mean_angles_deg) else "",
                f"{g.spread_distance_ang:.3f}", f"{g.spread_angle_deg:.1f}",
                _fmt(g.scalars.get("bvs")), _fmt(g.scalars.get("void_radius_ang")),
                str(g.representative)]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c in (1, 2, 5, 6, 7, 8, 9, 10, 11):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(r, c, item)
        self.summary.setText(words)
        for w in (self.export_groups, self.export_atoms, self.frames):
            w.setEnabled(bool(self.groups))

    def clear(self, words: str = "No grouping yet.") -> None:
        self.show_groups([], [], {}, words)

    def current_group(self):
        r = self.table.currentRow()
        return self.groups[r] if 0 <= r < len(self.groups) else None

    def _on_click(self, item) -> None:
        g = self.groups[item.row()] if 0 <= item.row() < len(self.groups) else None
        if g is not None:
            self.highlightRequested.emit(list(g.members),
                                         f"{g.label} ({g.n} atoms)")

    def _on_double_click(self, item) -> None:
        g = self.groups[item.row()] if 0 <= item.row() < len(self.groups) else None
        if g is not None:
            self.representativeRequested.emit(int(g.representative))

    def group_rows(self) -> list[dict]:
        return env_mod.group_rows(self.groups)

    def atom_rows(self, frame) -> list[dict]:
        return env_mod.per_atom_rows(frame, self.envs, self.groups, self.per_atom)


def _fmt(value) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if v != v else f"{v:.3f}"
