"""Volumetric fields: isosurfaces in 3D and sections in 2D.

Three sources of field, and the panel treats them the same way once they exist:

* a **bond-valence sum** map -- what valence a probe cation of a stated element
  and charge would have at every point in the cell. Its level sets are where that
  probe would be correctly bonded, which is the three-dimensional form of the
  argument the rest of the program makes about cutoffs;
* a **bond-valence energy landscape**, the same field turned into an energy, which
  is what a migration path is usually read from;
* a **file**: VASP CHGCAR, LOCPOT, ELFCAR or PARCHG, Gaussian CUBE, or XCrySDen
  XSF.

Either can be shown as an isosurface in the 3D view or cut with a plane and drawn
as a contour map. The plane is specified by Miller indices, so its normal is
h a* + k b* + l c* and it means the same thing in a monoclinic cell as in a cubic
one.

The panel reports the field's statistics and where the surfaces fall. It does not
say whether a basin is connected, or whether a path is a conduction pathway.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..core import bv, elements, planes as planes_mod, volume as volume_mod
from . import chrome
from .section_view import COLORMAPS, SectionView


def _spin(low, high, value, step, decimals=3, suffix=""):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


class VolumePanel(QWidget):
    """Build or load a field, show its isosurface, and cut sections through it."""

    isosurfaceChanged = Signal(object, float)   # grid (or None), level
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure = None
        self.params: bv.ParameterSet = bv.DEFAULT
        self.grid: volume_mod.Grid | None = None
        self._loading = False

        self.section = SectionView()
        self.readout = QLabel("")
        chrome.mark_hint(self.readout)
        self.section.hovered.connect(self._on_hover)

        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        tiny = QFont(self.stats.font())
        tiny.setPointSizeF(max(7.0, tiny.pointSizeF() - 1.0))
        self.stats.setFont(tiny)
        chrome.mark_hint(self.stats)

        self._build_layout()
        chrome.fit_every_combo(self)

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        hint = QLabel(
            "A bond-valence map answers the same question as the cutoff "
            "explorer, in three dimensions: where would a probe cation of this "
            "element and charge be correctly bonded? Its level set at the "
            "probe's own valence is that surface.")
        hint.setWordWrap(True)
        tiny = QFont(hint.font())
        tiny.setPointSizeF(max(7.0, tiny.pointSizeF() - 1.0))
        hint.setFont(tiny)
        chrome.mark_hint(hint)
        outer.addWidget(hint)

        outer.addWidget(self._source_group())
        outer.addWidget(self._display_group())

        split = QSplitter(Qt.Vertical)
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(self.section, 1)
        box.addWidget(self.readout)
        split.addWidget(holder)
        outer.addWidget(split, 1)
        outer.addWidget(self.stats)

    def _source_group(self) -> QWidget:
        group = QGroupBox("Field")
        # Stacked, not side by side: two form layouts competing for a
        # 398 px panel squeeze each other's label columns to nothing.
        row = QVBoxLayout(group)
        row.setSpacing(12)

        form = QFormLayout()
        form.setSpacing(3)
        self.kind = QComboBox()
        self.kind.addItem("bond-valence sum of a probe", "bvs")
        self.kind.addItem("bond-valence energy landscape", "bvel")
        self.kind.addItem("loaded from a file", "file")
        self.kind.currentIndexChanged.connect(self._on_kind)
        chrome.fit_combo(self.kind)
        form.addRow("kind", self.kind)

        probe = QHBoxLayout()
        self.probe_element = QComboBox()
        self.probe_element.setEditable(True)
        for symbol in ("Li", "Na", "K", "Mg", "Ca", "Ag", "Cu", "Bi", "O", "F"):
            self.probe_element.addItem(symbol)
        self.probe_element.setCurrentText("Na")
        chrome.fit_combo(self.probe_element)
        probe.addWidget(self.probe_element)
        self.probe_ox = QSpinBox()
        self.probe_ox.setRange(-4, 8)
        self.probe_ox.setValue(1)
        self.probe_ox.setPrefix("charge ")
        probe.addWidget(self.probe_ox)
        holder = QWidget()
        holder.setLayout(probe)
        form.addRow("probe", holder)
        row.addLayout(form)

        second = QFormLayout()
        second.setSpacing(3)
        self.resolution = _spin(0.05, 1.0, 0.25, 0.05, 2, " Å")
        self.resolution.setToolTip(
            "Grid spacing. A finer grid takes proportionally longer in all "
            "three directions, so halving it is eight times the work.")
        second.addRow("grid spacing", self.resolution)
        self.rmax = _spin(2.0, 12.0, 6.0, 0.5, 1, " Å")
        self.rmax.setToolTip(
            "How far from each anion the probe's valence is accumulated.")
        second.addRow("search radius", self.rmax)
        self.temperature = _spin(1.0, 2000.0, 300.0, 25.0, 0, " K")
        self.temperature.setToolTip(
            "Only used by the energy landscape, which converts a valence "
            "mismatch into an energy at this temperature.")
        second.addRow("temperature", self.temperature)
        row.addLayout(second)

        buttons = QVBoxLayout()
        buttons.setSpacing(3)
        self.compute_button = QPushButton("Compute")
        self.compute_button.clicked.connect(self.compute)
        self.load_button = QPushButton("Load a file…")
        self.load_button.clicked.connect(self._load)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        for button in (self.compute_button, self.load_button,
                       self.clear_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        row.addLayout(buttons)
        row.addStretch(1)
        return group

    def _display_group(self) -> QWidget:
        group = QGroupBox("Isosurface and section")
        # Stacked, not side by side: two form layouts competing for a
        # 398 px panel squeeze each other's label columns to nothing.
        row = QVBoxLayout(group)
        row.setSpacing(12)

        iso = QFormLayout()
        iso.setSpacing(3)
        self.level = _spin(-1e6, 1e6, 1.0, 0.1, 4)
        self.level.valueChanged.connect(self._emit_isosurface)
        iso.addRow("level", self.level)
        self.show_iso = QCheckBox("show in the 3D view")
        self.show_iso.toggled.connect(self._emit_isosurface)
        iso.addRow(self.show_iso)
        self.iso_step = QSpinBox()
        self.iso_step.setRange(1, 4)
        self.iso_step.setValue(1)
        self.iso_step.setToolTip(
            "Take every nth grid point when triangulating. 2 is four times "
            "fewer triangles and a visibly coarser surface.")
        self.iso_step.valueChanged.connect(self._emit_isosurface)
        iso.addRow("coarsen by", self.iso_step)
        row.addLayout(iso)

        plane = QFormLayout()
        plane.setSpacing(3)
        indices = QHBoxLayout()
        self.h = QSpinBox()
        self.k = QSpinBox()
        self.l = QSpinBox()
        for label, widget, default in (("h", self.h, 0), ("k", self.k, 0),
                                       ("l", self.l, 1)):
            widget.setRange(-12, 12)
            widget.setValue(default)
            widget.valueChanged.connect(self.refresh_section)
            indices.addWidget(QLabel(label))
            indices.addWidget(widget)
        indices.addStretch(1)
        holder = QWidget()
        holder.setLayout(indices)
        plane.addRow("plane", holder)

        self.offset = _spin(-20.0, 20.0, 0.0, 0.05, 3)
        self.offset.setToolTip(
            "Where the plane sits, in units of its own interplanar spacing d.")
        self.offset.valueChanged.connect(self.refresh_section)
        plane.addRow("offset / d", self.offset)
        self.size = _spin(2.0, 60.0, 12.0, 1.0, 1, " Å")
        self.size.valueChanged.connect(self.refresh_section)
        plane.addRow("extent", self.size)
        self.samples = QSpinBox()
        self.samples.setRange(32, 800)
        self.samples.setValue(240)
        self.samples.valueChanged.connect(self.refresh_section)
        plane.addRow("samples", self.samples)
        row.addLayout(plane)

        look = QVBoxLayout()
        look.setSpacing(2)
        colours = QHBoxLayout()
        colours.addWidget(QLabel("colours"))
        self.colormap = QComboBox()
        for name in COLORMAPS:
            self.colormap.addItem(name)
        self.colormap.currentTextChanged.connect(self._apply_look)
        colours.addWidget(self.colormap)
        look.addLayout(colours)

        contours = QHBoxLayout()
        contours.addWidget(QLabel("contours"))
        self.n_levels = QSpinBox()
        self.n_levels.setRange(0, 40)
        self.n_levels.setValue(9)
        self.n_levels.valueChanged.connect(self.refresh_section)
        contours.addWidget(self.n_levels)
        look.addLayout(contours)

        self.log_scale = QCheckBox("logarithmic scale")
        self.log_scale.setToolTip(
            "A charge density or a valence sum spans decades; a linear scale "
            "then shows only the peaks.")
        self.log_scale.toggled.connect(self._apply_look)
        look.addWidget(self.log_scale)
        self.show_map = QCheckBox("colour map")
        self.show_map.setChecked(True)
        self.show_map.toggled.connect(self._apply_look)
        look.addWidget(self.show_map)
        self.show_atoms = QCheckBox("mark atoms on the plane")
        self.show_atoms.setChecked(True)
        self.show_atoms.toggled.connect(self.refresh_section)
        look.addWidget(self.show_atoms)
        row.addLayout(look)

        exports = QVBoxLayout()
        exports.setSpacing(3)
        for label, handler in (("Save SVG…", lambda: self._export("svg")),
                               ("Save PDF…", lambda: self._export("pdf")),
                               ("Save the section…", self._export_csv)):
            button = QPushButton(label)
            button.clicked.connect(handler)
            exports.addWidget(button)
        exports.addStretch(1)
        row.addLayout(exports)
        row.addStretch(1)
        return group

    # -- content -----------------------------------------------------------
    def apply_theme(self, theme) -> None:
        self.section.apply_theme(theme)

    def set_context(self, structure, params=None) -> None:
        if structure is not self.structure:
            self.clear()
        self.structure = structure
        self.params = params or bv.DEFAULT
        enabled = structure is not None
        self.compute_button.setEnabled(enabled)
        self._on_kind()

    def _on_kind(self) -> None:
        kind = self.kind.currentData()
        for widget in (self.probe_element, self.probe_ox, self.resolution,
                       self.rmax):
            widget.setEnabled(kind in ("bvs", "bvel"))
        self.temperature.setEnabled(kind == "bvel")
        self.compute_button.setEnabled(self.structure is not None
                                       and kind in ("bvs", "bvel"))

    def clear(self) -> None:
        self.grid = None
        self.section.set_section(None, (-5, 5, -5, 5))
        self.stats.setText("")
        self.readout.setText("")
        self.show_iso.setChecked(False)
        self.isosurfaceChanged.emit(None, 0.0)

    # -- building the field ------------------------------------------------
    def compute(self) -> None:
        if self.structure is None:
            return
        kind = self.kind.currentData()
        element = elements.normalise(self.probe_element.currentText())
        ox = int(self.probe_ox.value())
        if not element:
            QMessageBox.warning(self, "Which probe?",
                                "Give the probe an element symbol.")
            return

        spacing = float(self.resolution.value())
        cell = self.structure.cell
        estimate = 1
        for length in cell.lengths:
            estimate *= max(int(round(length / spacing)), 1)
        if estimate > 4_000_000:
            answer = QMessageBox.question(
                self, "That is a large grid",
                f"{estimate:,} points at {spacing:.2f} Å spacing. "
                "Halving the spacing is eight times the work. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if answer != QMessageBox.Yes:
                return

        progress = QProgressDialog(
            f"Building the field on {estimate:,} points…", "", 0, 0, self)
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        from PySide6.QtWidgets import QApplication

        QApplication.processEvents()
        try:
            grid = volume_mod.bond_valence_grid(
                self.structure, element, ox, resolution=spacing,
                params=self.params, rmax=float(self.rmax.value()))
            if kind == "bvel":
                grid = volume_mod.bond_valence_energy(
                    grid, ox, temperature=float(self.temperature.value()))
        except (ValueError, MemoryError) as error:
            progress.close()
            QMessageBox.warning(self, "Could not build the field", str(error))
            return
        finally:
            progress.close()

        self.grid = grid
        self._after_grid(default_level=(abs(ox) if kind == "bvs" else 1.0))

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load a volumetric file", "",
            volume_mod.VOLUME_FILE_FILTER)
        if not path:
            return
        try:
            grid, structure = volume_mod.read_volume(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not read the file", str(error))
            return
        self.grid = grid
        self.kind.setCurrentIndex(self.kind.findData("file"))
        statistics = grid.statistics()
        middle = 0.5 * (statistics["min"] + statistics["max"])
        self._after_grid(default_level=middle)
        if structure is not None and self.structure is not None:
            same = np.allclose(structure.cell.lengths,
                               self.structure.cell.lengths, atol=0.05)
            if not same:
                self.statusMessage.emit(
                    "the file's cell does not match the loaded structure, so "
                    "the field and the atoms are not on the same lattice")

    def _after_grid(self, default_level: float) -> None:
        self._loading = True
        try:
            statistics = self.grid.statistics()
            low, high = statistics["min"], statistics["max"]
            self.level.setRange(min(low, -1e6), max(high, 1e6))
            self.level.setValue(float(np.clip(default_level, low, high)))
            # A valence sum or a charge density spans decades -- it diverges at
            # the nuclei -- so linear contour levels all land in the empty region
            # close in and none near the value that matters. Switched on rather
            # than assumed: the control shows it and the message says why.
            if low > 0 and high / low > 100.0:
                self.log_scale.setChecked(True)
            # an extent that covers the cell
            span = max(self.structure.cell.lengths) if self.structure else 12.0
            self.size.setValue(min(float(span) * 1.1, self.size.maximum()))
        finally:
            self._loading = False
        self.refresh_section()
        self._emit_isosurface()
        self._describe()
        statistics = self.grid.statistics()
        if self.log_scale.isChecked() and statistics["min"] > 0:
            self.statusMessage.emit(
                f"the field spans {statistics['max'] / statistics['min']:.0f}x, "
                "so the scale and the contour levels were set logarithmic")

    def _describe(self) -> None:
        if self.grid is None:
            self.stats.setText("")
            return
        statistics = self.grid.statistics()
        shape = self.grid.shape
        spacing = self.grid.spacing
        lines = [
            f"{self.grid.name or 'field'}: {shape[0]}×{shape[1]}×"
            f"{shape[2]} points at {spacing[0]:.3f}/{spacing[1]:.3f}/"
            f"{spacing[2]:.3f} Å; range {statistics['min']:.4g} to "
            f"{statistics['max']:.4g} {self.grid.units}, mean "
            f"{statistics['mean']:.4g}"]
        lines.extend(self.grid.notes)
        self.stats.setText("  ·  ".join(lines))

    # -- the section -------------------------------------------------------
    def refresh_section(self) -> None:
        if self._loading or self.grid is None or self.structure is None:
            return
        h, k, l = self.h.value(), self.k.value(), self.l.value()
        if (h, k, l) == (0, 0, 0):
            self.section.set_section(None, (-5, 5, -5, 5))
            self.readout.setText("(0 0 0) is not a plane.")
            return
        try:
            u_axis, v_axis, normal = planes_mod.in_plane_axes(self.structure,
                                                             h, k, l)
            _, d = planes_mod.normal_and_spacing(self.structure, h, k, l)
        except ValueError as error:
            self.readout.setText(str(error))
            return

        origin = normal * float(self.offset.value()) * d
        size = float(self.size.value())
        samples = int(self.samples.value())
        values, extent = volume_mod.section(self.grid, origin, u_axis, v_axis,
                                            size=size, samples=samples)
        levels = volume_mod.nice_levels(values, int(self.n_levels.value()),
                                        log=self.log_scale.isChecked())
        self.section.set_section(
            values, extent, levels, units=self.grid.units,
            title=(f"{self.grid.name or 'field'} on ({h} {k} {l}), "
                   f"offset {self.offset.value():.3f} d = "
                   f"{self.offset.value() * d:+.3f} Å"),
            footnote=(f"plane normal h a* + k b* + l c*, d = {d:.4f} Å; "
                      f"{samples}×{samples} samples over {size:.1f} "
                      f"Å; {len(levels)} contour levels"))
        self.section.markers = (self._atom_markers(origin, u_axis, v_axis, size)
                                if self.show_atoms.isChecked() else [])
        self._apply_look()

    def _atom_markers(self, origin, u_axis, v_axis, size: float):
        """Atoms lying within half a grid spacing of the plane, in plane coordinates."""
        if self.structure is None or not self.structure.atoms:
            return []
        normal = np.cross(u_axis, v_axis)
        normal = normal / (np.linalg.norm(normal) or 1.0)
        tolerance = max(self.grid.spacing) if self.grid is not None else 0.3

        markers = []
        half = size / 2.0
        orth = self.structure.cell.orth
        # include the neighbouring cells, or a plane through the cell edge shows
        # atoms on one side only
        for atom in self.structure.atoms:
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        point = atom.cart + orth @ np.array([i, j, k], float)
                        offset = point - origin
                        if abs(float(np.dot(offset, normal))) > tolerance:
                            continue
                        u = float(np.dot(offset, u_axis))
                        v = float(np.dot(offset, v_axis))
                        if abs(u) <= half and abs(v) <= half:
                            markers.append((u, v, atom.label))
        return markers

    def _apply_look(self) -> None:
        self.section.colormap = self.colormap.currentText()
        self.section.log_scale = self.log_scale.isChecked()
        self.section.show_map = self.show_map.isChecked()
        self.section.show_contours = self.n_levels.value() > 0
        self.section.update()

    def _emit_isosurface(self) -> None:
        if self._loading:
            return
        if self.grid is None or not self.show_iso.isChecked():
            self.isosurfaceChanged.emit(None, 0.0)
            return
        self.isosurfaceChanged.emit(self.grid, float(self.level.value()))

    def iso_step_value(self) -> int:
        return int(self.iso_step.value())

    # -- output ------------------------------------------------------------
    def _on_hover(self, u: float, v: float, value: float) -> None:
        if value != value:
            self.readout.setText("")
            return
        self.readout.setText(
            f"u = {u:.3f} Å, v = {v:.3f} Å   →   "
            f"{value:.5g} {self.grid.units if self.grid else ''}".rstrip())

    def _export(self, kind: str) -> None:
        if self.section.values is None:
            return
        name = (self.structure.name if self.structure else "section")
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save the section as {kind.upper()}",
            f"{name}_section.{kind}", f"{kind.upper()} (*.{kind})")
        if not path:
            return
        try:
            if kind == "svg":
                self.section.save_svg(path)
            else:
                self.section.save_pdf(path)
        except Exception as error:                      # pragma: no cover
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self.statusMessage.emit(f"wrote {path}")

    def _export_csv(self) -> None:
        """The sampled values, so the map can be replotted or fitted elsewhere."""
        if self.section.values is None:
            return
        from pathlib import Path

        name = (self.structure.name if self.structure else "section")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the sampled section", f"{name}_section.csv",
            "CSV (*.csv)")
        if not path:
            return
        values = self.section.values
        umin, umax, vmin, vmax = self.section.extent
        rows, cols = values.shape
        u_axis = np.linspace(umin, umax, cols)
        v_axis = np.linspace(vmin, vmax, rows)
        lines = [f"# {self.section.title}", f"# {self.section.footnote}",
                 f"# units: {self.grid.units if self.grid else ''}",
                 "u_angstrom,v_angstrom,value"]
        for j, v in enumerate(v_axis):
            for i, u in enumerate(u_axis):
                lines.append(f"{u:.5f},{v:.5f},{values[j, i]:.8g}")
        try:
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self.statusMessage.emit(f"wrote {path}")
