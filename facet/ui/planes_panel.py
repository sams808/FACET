"""Lattice planes and slabs.

Two panels' worth of controls in one place, because they answer the same
question from opposite directions: a plane says where a layer is, a slab hides
everything that is not in it.

The panel reports what it finds on each plane -- the spacing, how many atoms lie
on it, which elements they are -- and stops there. Whether a family of planes
carrying only one element amounts to "a layer" is a judgement about the
structure, and it is not made here.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import planes as planes_mod
from . import chrome

# Distinct starting colours, so a second plane never lands on the first one's.
PALETTE = [
    (0.36, 0.72, 0.92), (0.96, 0.66, 0.30), (0.56, 0.86, 0.52),
    (0.92, 0.50, 0.62), (0.76, 0.62, 0.94), (0.42, 0.84, 0.84),
]

# Common planes worth one click, as a starting point rather than a suggestion.
PRESETS = [
    ("(1 0 0)", (1, 0, 0)), ("(0 1 0)", (0, 1, 0)), ("(0 0 1)", (0, 0, 1)),
    ("(1 1 0)", (1, 1, 0)), ("(1 0 1)", (1, 0, 1)), ("(0 1 1)", (0, 1, 1)),
    ("(1 1 1)", (1, 1, 1)), ("(1 -1 0)", (1, -1, 0)),
    ("(1 1 -1)", (1, 1, -1)), ("(2 0 0)", (2, 0, 0)),
]


def _index_spin(value: int = 0) -> QSpinBox:
    box = QSpinBox()
    box.setRange(-12, 12)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


def _swatch(button: QPushButton, rgb) -> None:
    colour = QColor(int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    button.setStyleSheet(
        f"background:{colour.name()}; border:1px solid #555; min-width:34px;")


class PlanesPanel(QWidget):
    """Add, edit and remove lattice planes; define one slab."""

    changed = Signal(str)   # rebuild the scene; the text says what changed

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure = None
        self.planes: list[planes_mod.LatticePlane] = []
        self.slab = planes_mod.Slab()
        self._loading = False

        self._build_layout()

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        hint = QLabel(
            "The normal of an (hkl) plane is h a* + k b* + l c*, the "
            "reciprocal-lattice vector. That is not h a + k b + l c except in a "
            "cubic cell — for a monoclinic β of 113° the two are "
            "27° apart, and for hexagonal (1 1 1) they are 55° apart.")
        hint.setWordWrap(True)
        tiny = QFont(hint.font())
        tiny.setPointSizeF(max(7.0, tiny.pointSizeF() - 1.0))
        hint.setFont(tiny)
        chrome.mark_hint(hint)
        outer.addWidget(hint)

        outer.addWidget(self._planes_group(), 1)
        outer.addWidget(self._slab_group())
        outer.addWidget(self._report_group(), 1)

    def _planes_group(self) -> QWidget:
        group = QGroupBox("Lattice planes")
        box = QVBoxLayout(group)
        box.setSpacing(4)

        # Two rows: the preset list and the typed indices together asked for
        # more width than the panel has, and the combo was rendering "choose..."
        # as "cho...".
        self.preset = QComboBox()
        self.preset.addItem("choose a plane…", None)
        for label, hkl in PRESETS:
            self.preset.addItem(label, hkl)
        self.preset.currentIndexChanged.connect(self._add_preset)
        chrome.name_inside(
            self.preset, "preset",
            tip="Add one of the common planes by name, instead of typing its "
                "indices.")
        box.addWidget(self.preset)

        add = QHBoxLayout()
        add.setSpacing(chrome.FIELD_SPACING)
        add.addWidget(QLabel("h"))
        self.new_h = _index_spin(1)
        add.addWidget(self.new_h)
        add.addWidget(QLabel("k"))
        self.new_k = _index_spin(0)
        add.addWidget(self.new_k)
        add.addWidget(QLabel("l"))
        self.new_l = _index_spin(0)
        add.addWidget(self.new_l)
        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self._add_typed)
        add.addWidget(self.add_button)
        add.addStretch(1)
        self.clear_button = QPushButton("Remove all")
        self.clear_button.clicked.connect(self._clear)
        add.addWidget(self.clear_button)
        box.addLayout(add)

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["show", "h k l", "d / Å", "offset / d", "repeat",
             "opacity", "colour", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents)
        font = QFont(self.table.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        self.table.setFont(font)
        self.table.currentCellChanged.connect(lambda *_: self._report())
        box.addWidget(self.table, 1)
        return group

    def _slab_group(self) -> QWidget:
        group = QGroupBox("Slab — show only what lies between two planes")
        # Stacked, not side by side: two form layouts competing for a
        # 398 px panel squeeze each other's label columns to nothing.
        row = QVBoxLayout(group)
        row.setSpacing(12)

        self.slab_on = QCheckBox("enabled")
        self.slab_on.toggled.connect(self._slab_changed)
        row.addWidget(self.slab_on)

        form = QFormLayout()
        form.setSpacing(3)
        indices = QHBoxLayout()
        self.slab_h = _index_spin(0)
        self.slab_k = _index_spin(0)
        self.slab_l = _index_spin(1)
        for label, widget in (("h", self.slab_h), ("k", self.slab_k),
                              ("l", self.slab_l)):
            indices.addWidget(QLabel(label))
            indices.addWidget(widget)
            widget.valueChanged.connect(self._slab_changed)
        indices.addStretch(1)
        holder = QWidget()
        holder.setLayout(indices)
        form.addRow("normal", holder)
        row.addLayout(form)

        second = QFormLayout()
        second.setSpacing(3)
        self.slab_centre = QDoubleSpinBox()
        self.slab_centre.setRange(-20.0, 20.0)
        self.slab_centre.setSingleStep(0.05)
        self.slab_centre.setDecimals(3)
        self.slab_centre.setToolTip(
            "Where the slab sits, in units of the interplanar spacing d, so the "
            "control behaves the same whatever the indices.")
        self.slab_centre.valueChanged.connect(self._slab_changed)
        second.addRow("centre / d", self.slab_centre)

        self.slab_thickness = QDoubleSpinBox()
        self.slab_thickness.setRange(0.10, 200.0)
        self.slab_thickness.setSingleStep(0.25)
        self.slab_thickness.setDecimals(2)
        self.slab_thickness.setValue(4.0)
        self.slab_thickness.setSuffix(" Å")
        self.slab_thickness.valueChanged.connect(self._slab_changed)
        second.addRow("thickness", self.slab_thickness)
        row.addLayout(second)

        self.slab_report = QLabel("")
        self.slab_report.setWordWrap(True)
        small = QFont(self.slab_report.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.0))
        self.slab_report.setFont(small)
        chrome.mark_hint(self.slab_report)
        row.addWidget(self.slab_report, 1)
        return group

    def _report_group(self) -> QWidget:
        group = QGroupBox("What lies on the selected plane")
        box = QVBoxLayout(group)
        box.setSpacing(4)

        tolerance_row = QHBoxLayout()
        tolerance_row.addWidget(QLabel("within"))
        self.tolerance = QDoubleSpinBox()
        self.tolerance.setRange(0.01, 3.0)
        self.tolerance.setSingleStep(0.05)
        self.tolerance.setDecimals(2)
        self.tolerance.setValue(planes_mod.ON_PLANE)
        self.tolerance.setSuffix(" Å")
        self.tolerance.valueChanged.connect(self._report)
        tolerance_row.addWidget(self.tolerance)
        tolerance_row.addWidget(QLabel("of a plane of the family"))
        tolerance_row.addStretch(1)
        box.addLayout(tolerance_row)

        self.contents = QTableWidget(0, 3)
        self.contents.setHorizontalHeaderLabels(
            ["atom", "element", "distance / Å"])
        self.contents.verticalHeader().setVisible(False)
        self.contents.setEditTriggers(QTableWidget.NoEditTriggers)
        self.contents.setAlternatingRowColors(True)
        font = QFont(self.contents.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        self.contents.setFont(font)
        box.addWidget(self.contents, 1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        small = QFont(self.summary.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.0))
        self.summary.setFont(small)
        chrome.mark_hint(self.summary)
        box.addWidget(self.summary)
        return group

    # -- content -----------------------------------------------------------
    def set_structure(self, structure) -> None:
        self.structure = structure
        self._refresh_table()
        self._report()
        self._describe_slab()

    def current_planes(self) -> list[planes_mod.LatticePlane]:
        return list(self.planes)

    def current_slab(self) -> planes_mod.Slab:
        return self.slab

    # -- adding and removing -----------------------------------------------
    def _add_preset(self, index: int) -> None:
        if self._loading or index <= 0:
            return
        hkl = self.preset.itemData(index)
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(0)
        self.preset.blockSignals(False)
        if hkl:
            self.add_plane(*hkl)

    def _add_typed(self) -> None:
        self.add_plane(self.new_h.value(), self.new_k.value(),
                       self.new_l.value())

    def add_plane(self, h: int, k: int, l: int) -> None:
        if (h, k, l) == (0, 0, 0):
            self.summary.setText(
                "(0 0 0) is not a plane — it has no normal.")
            return
        colour = PALETTE[len(self.planes) % len(PALETTE)]
        self.planes.append(planes_mod.LatticePlane(int(h), int(k), int(l),
                                                   color=colour))
        self._refresh_table()
        self.table.setCurrentCell(len(self.planes) - 1, 1)
        self.changed.emit(f"added the ({h} {k} {l}) plane")
        self._report()

    def _remove(self, plane) -> None:
        if plane in self.planes:
            self.planes.remove(plane)
        self._refresh_table()
        self.changed.emit(f"removed the ({plane.hkl}) plane")
        self._report()

    def _clear(self) -> None:
        self.planes = []
        self._refresh_table()
        self.changed.emit("removed every plane")
        self._report()

    # -- the table ---------------------------------------------------------
    def _refresh_table(self) -> None:
        self._loading = True
        try:
            self.table.setRowCount(len(self.planes))
            for row, plane in enumerate(self.planes):
                self._fill_row(row, plane)
            self.table.resizeColumnsToContents()
        finally:
            self._loading = False

    def _fill_row(self, row: int, plane) -> None:
        show = QCheckBox()
        show.setChecked(plane.visible)
        show.toggled.connect(
            lambda on, p=plane: self._set(p, "visible", on))
        self.table.setCellWidget(row, 0, show)

        item = QTableWidgetItem(f"({plane.hkl})")
        item.setTextAlignment(Qt.AlignCenter)
        self.table.setItem(row, 1, item)

        try:
            d = (planes_mod.spacing(self.structure, plane)
                 if self.structure is not None else float("nan"))
        except ValueError:
            d = float("nan")
        spacing_item = QTableWidgetItem("" if d != d else f"{d:.4f}")
        spacing_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.table.setItem(row, 2, spacing_item)

        offset = QDoubleSpinBox()
        offset.setRange(-20.0, 20.0)
        offset.setSingleStep(0.05)
        offset.setDecimals(3)
        offset.setValue(plane.offset)
        offset.setKeyboardTracking(False)
        offset.valueChanged.connect(
            lambda v, p=plane: self._set(p, "offset", float(v)))
        self.table.setCellWidget(row, 3, offset)

        repeat = QSpinBox()
        repeat.setRange(1, 40)
        repeat.setValue(plane.repeat)
        repeat.setKeyboardTracking(False)
        repeat.setToolTip("How many planes of the family to draw, spaced by d.")
        repeat.valueChanged.connect(
            lambda v, p=plane: self._set(p, "repeat", int(v)))
        self.table.setCellWidget(row, 4, repeat)

        opacity = QSlider(Qt.Horizontal)
        opacity.setRange(5, 100)
        opacity.setValue(int(plane.alpha * 100))
        opacity.setMinimumWidth(70)
        opacity.valueChanged.connect(
            lambda v, p=plane: self._set(p, "alpha", v / 100.0))
        self.table.setCellWidget(row, 5, opacity)

        swatch = QPushButton()
        _swatch(swatch, plane.color)
        swatch.clicked.connect(lambda _=False, p=plane: self._pick_color(p))
        self.table.setCellWidget(row, 6, swatch)

        remove = QPushButton("×")
        remove.setMaximumWidth(26)
        remove.setToolTip("Remove this plane")
        remove.clicked.connect(lambda _=False, p=plane: self._remove(p))
        self.table.setCellWidget(row, 7, remove)

    def _set(self, plane, attribute: str, value) -> None:
        if self._loading:
            return
        setattr(plane, attribute, value)
        self.changed.emit(f"changed the {attribute} of ({plane.hkl})")
        self._report()

    def _pick_color(self, plane) -> None:
        current = QColor(int(plane.color[0] * 255), int(plane.color[1] * 255),
                         int(plane.color[2] * 255))
        chosen = QColorDialog.getColor(current, self, "Plane colour")
        if not chosen.isValid():
            return
        plane.color = (chosen.redF(), chosen.greenF(), chosen.blueF())
        self._refresh_table()
        self.changed.emit(f"recoloured the ({plane.hkl}) plane")


    def load_slab_controls(self) -> None:
        """Push the slab model into its controls without emitting changes.

        Used when undo restores a snapshot: the controls have to follow the
        model, and each setValue would otherwise look like the user editing it
        and record another history step.
        """
        self._loading = True
        try:
            self.slab_h.setValue(self.slab.h)
            self.slab_k.setValue(self.slab.k)
            self.slab_l.setValue(self.slab.l)
            self.slab_centre.setValue(self.slab.centre)
            self.slab_thickness.setValue(self.slab.thickness)
            self.slab_on.setChecked(self.slab.enabled)
        finally:
            self._loading = False
        self._describe_slab()

    def reload(self) -> None:
        """Rebuild the plane table and the report from the current model."""
        self._refresh_table()
        self._report()

    # -- the slab ----------------------------------------------------------
    def _slab_changed(self) -> None:
        if self._loading:
            return
        self.slab.enabled = self.slab_on.isChecked()
        self.slab.h = self.slab_h.value()
        self.slab.k = self.slab_k.value()
        self.slab.l = self.slab_l.value()
        self.slab.centre = float(self.slab_centre.value())
        self.slab.thickness = float(self.slab_thickness.value())
        self._describe_slab()
        self.changed.emit(
            f"set the slab to ({self.slab.hkl}), "
            f"{self.slab.thickness:.2f} A"
            + ("" if self.slab.enabled else ", off"))

    def _describe_slab(self) -> None:
        if self.structure is None:
            self.slab_report.setText("")
            return
        if not self.slab.is_valid:
            self.slab_report.setText(
                "(0 0 0) has no normal, so it cannot bound a slab.")
            return
        try:
            _, d = planes_mod.normal_and_spacing(
                self.structure, self.slab.h, self.slab.k, self.slab.l)
        except ValueError as error:
            self.slab_report.setText(str(error))
            return
        inside = len(planes_mod.atoms_in_slab(self.structure, self.slab))
        total = len(self.structure.atoms)
        state = "" if self.slab.enabled else " (not applied)"
        self.slab_report.setText(
            f"d({self.slab.hkl}) = {d:.4f} Å. The slab is "
            f"{self.slab.thickness:.2f} Å thick, centred "
            f"{self.slab.centre * d:+.3f} Å from the origin, and contains "
            f"{inside} of the cell's {total} atoms{state}.")

    # -- what is on the plane ----------------------------------------------
    def _report(self) -> None:
        row = self.table.currentRow()
        if (self.structure is None or not self.planes
                or not 0 <= row < len(self.planes)):
            self.contents.setRowCount(0)
            self.summary.setText(
                "" if self.structure is not None
                else "Load a structure to see what lies on a plane.")
            return
        plane = self.planes[row]
        tolerance = float(self.tolerance.value())
        try:
            normal, d = planes_mod.normal_and_spacing(self.structure, plane.h,
                                                      plane.k, plane.l)
            indices = planes_mod.atoms_on_plane(self.structure, plane,
                                                tolerance)
        except ValueError as error:
            self.contents.setRowCount(0)
            self.summary.setText(str(error))
            return

        import numpy as np

        self.contents.setRowCount(len(indices))
        for r, index in enumerate(indices):
            atom = self.structure.atoms[index]
            along = float(np.dot(atom.cart, normal))
            phase = along / d - plane.offset
            distance = abs(phase - round(phase)) * d
            for c, text in enumerate((atom.label, atom.element,
                                      f"{distance:.4f}")):
                item = QTableWidgetItem(text)
                if c == 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.contents.setItem(r, c, item)
        self.contents.resizeColumnsToContents()

        composition = planes_mod.plane_occupancy(self.structure, plane,
                                                 tolerance)
        pieces = ", ".join(f"{element} {amount:.3g}"
                           for element, amount in sorted(composition.items()))
        self.summary.setText(
            f"({plane.hkl}): d = {d:.4f} Å. {len(indices)} of the cell's "
            f"{len(self.structure.atoms)} atoms lie within {tolerance:.2f} "
            f"Å of a plane of this family"
            + (f" — {pieces}." if pieces else "."))
