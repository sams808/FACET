"""The diffraction panel.

A calculated powder pattern for the loaded structure, optionally with a measured
pattern loaded over it and a difference curve beneath. Everything that shapes
the calculation is exposed: radiation, wavelength, the displacement parameter
used where the file gave none, the profile width and mixing, the zero shift.

There is no goodness-of-fit figure anywhere in this panel, and no statement
about whether a pattern matches. A least-squares scale factor is the only thing
fitted, and it is reported as a number so you can see what it did. Deciding
whether the calculated and measured patterns are the same phase is the
crystallographer's job, and this panel exists to put the two curves in front of
you at the same scale so you can do it.
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
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import diffraction as dif
from . import chrome
from .plot import Plot, Series, Ticks

CALCULATED = (120, 190, 240)
MEASURED = (232, 148, 96)
DIFFERENCE = (150, 200, 140)
TICK = (150, 160, 175)


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


class DiffractionPanel(QWidget):
    """Controls, plot and reflection list for one structure."""

    reflection_selected = Signal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure = None
        self.pattern: dif.Pattern | None = None
        self.measured: tuple[np.ndarray, np.ndarray] | None = None
        self.measured_name = ""
        self._scale_factor = 1.0

        self.plot = Plot()
        self.plot.set_labels(x="2θ / °", y="intensity (relative)")
        self.plot.hovered.connect(self._on_hover)
        self.plot.picked.connect(self._on_pick)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["h k l", "d / Å", "2θ / °", "I", "mult",
             "|F|²", "LP"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setAlternatingRowColors(True)
        font = QFont(self.table.font())
        font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
        self.table.setFont(font)
        self.table.itemSelectionChanged.connect(self._on_row)

        self.readout = QLabel("")
        chrome.mark_hint(self.readout)
        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        small = QFont(self.notes.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.0))
        self.notes.setFont(small)
        chrome.mark_hint(self.notes)

        self._build_layout()
        self._connect()

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        hint = QLabel(
            "A pattern calculated from the structure, with real scattering "
            "factors. Nothing is refined and no fit quality is reported — "
            "the only thing fitted to a measured pattern is one scale factor, "
            "shown below.")
        hint.setWordWrap(True)
        tiny = QFont(hint.font())
        tiny.setPointSizeF(max(7.0, tiny.pointSizeF() - 1.0))
        hint.setFont(tiny)
        chrome.mark_hint(hint)
        outer.addWidget(hint)

        outer.addWidget(self._controls())

        split = QSplitter(Qt.Vertical)
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(2)
        box.addWidget(self.plot, 1)
        box.addWidget(self.readout)
        split.addWidget(holder)
        split.addWidget(self.table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, 1)
        outer.addWidget(self.notes)

    def _controls(self) -> QWidget:
        """Self-labelling fields, two per row, with the diffractometer hidden.

        Three form layouts side by side is what this used to be, and in a 398 px
        panel each column was squeezed until its labels were zero pixels wide.
        The spin boxes now carry their own names; what described the instrument
        rather than the structure sits behind a disclosure, because it is set
        once.
        """
        group = QGroupBox()
        rows = QVBoxLayout(group)
        rows.setContentsMargins(8, 8, 8, 8)
        rows.setSpacing(chrome.FIELD_SPACING)

        # -- the experiment ------------------------------------------------
        self.radiation = QComboBox()
        for name in dif.RADIATIONS:
            self.radiation.addItem(name)
        chrome.name_inside(
            self.radiation, "radiation",
            tip="X-ray uses the International Tables form factors, which fall "
                "off with angle. Neutron uses coherent scattering lengths, "
                "which do not and which can be negative -- that is what lets a "
                "light atom be seen beside a heavy one. Electron uses Peng's "
                "fit.")

        self.wavelength_choice = QComboBox()
        # The item text is asserted by tests/test_plot_diffraction_panel.py,
        # which looks for a line starting "Mo Ka1": leave it alone.
        for name, value in dif.WAVELENGTHS.items():
            self.wavelength_choice.addItem(f"{name}   {value:.5f} Å", value)
        self.wavelength_choice.setCurrentIndex(0)
        chrome.name_inside(self.wavelength_choice, "λ",
                           tip="A common characteristic wavelength. Choosing "
                               "one fills the exact value below.")

        self.wavelength = chrome.name_inside(
            _spin(0.10, 10.0, 1.540598, 0.001, 6), "λ", " Å",
            tip="The wavelength actually used, to six decimals.")
        self.two_theta_min = chrome.name_inside(
            _spin(0.0, 179.0, 5.0, 1.0, 2), "2θ from", "°")
        self.two_theta_max = chrome.name_inside(
            _spin(1.0, 180.0, 90.0, 1.0, 2), "2θ to", "°")

        rows.addLayout(chrome.field_grid([
            self.radiation, self.wavelength_choice,
            (self.wavelength, 2),
            self.two_theta_min, self.two_theta_max,
        ]))

        # -- the diffractometer, which is set once -------------------------
        self.step = chrome.name_inside(
            _spin(0.001, 0.5, 0.020, 0.005, 3), "step", "°",
            tip="The 2-theta grid the profile is sampled on.")
        self.b_iso = chrome.name_inside(
            _spin(0.0, 10.0, dif.DEFAULT_B_ISO, 0.1, 2), "B₀", " Å²",
            tip="Isotropic displacement parameter B used only for atoms whose "
                "site carries no U_iso in the file. Where the file gives one, "
                "it is used instead and the panel says so.")
        self.monochromator = chrome.name_inside(
            _spin(0.0, 90.0, 0.0, 0.5, 2), "mono 2θ", "°",
            tip="2-theta of a monochromator crystal, which changes the "
                "polarisation term. Zero means no monochromator correction.")
        self.zero_shift = chrome.name_inside(
            _spin(-2.0, 2.0, 0.0, 0.01, 3), "zero", "°",
            tip="Shifts the calculated pattern bodily in 2-theta, for a "
                "displaced or misaligned sample. Not a refined parameter.")

        self.u = chrome.name_inside(_spin(-1.0, 1.0, 0.010, 0.001, 4), "U")
        self.v = chrome.name_inside(_spin(-1.0, 1.0, -0.004, 0.001, 4), "V")
        self.w = chrome.name_inside(_spin(0.0, 1.0, 0.006, 0.001, 4), "W")
        for box in (self.u, self.v, self.w):
            box.setToolTip(
                "Caglioti width: FWHM^2 = U tan^2(theta) + V tan(theta) + W. "
                "These describe the diffractometer, not the structure.")
        self.eta = chrome.name_inside(
            _spin(0.0, 1.0, 0.5, 0.05, 2), "η",
            tip="The pseudo-Voigt mixing: 0 is pure Gaussian, 1 is pure "
                "Lorentzian.")

        instrument = QWidget()
        inner = QVBoxLayout(instrument)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addLayout(chrome.field_grid([
            self.step, self.b_iso,
            self.monochromator, self.zero_shift,
            self.u, self.v,
            self.w, self.eta,
        ]))
        rows.addWidget(chrome.disclosure(
            "Peak shape, zero shift and the B fallback", instrument))

        # -- what to show ---------------------------------------------------
        self.show_ticks = QCheckBox("marks")
        self.show_ticks.setChecked(True)
        self.show_ticks.setToolTip("A tick at each reflection's position.")
        self.show_sticks = QCheckBox("lines only")
        self.show_sticks.setToolTip(
            "Draw each reflection as a bare vertical line at its calculated "
            "intensity, with no peak shape at all.")
        self.show_difference = QCheckBox("difference")
        self.show_difference.setChecked(True)
        self.show_difference.setToolTip(
            "The measured pattern minus the scaled calculation, beneath.")
        self.fill = QCheckBox("shade")
        self.fill.setToolTip("Shade the area under the calculated curve.")
        rows.addLayout(chrome.field_grid([
            self.show_ticks, self.show_sticks,
            self.show_difference, self.fill,
        ]))

        # -- actions ---------------------------------------------------------
        self.load_measured = QPushButton("Load measured…")
        self.clear_measured = QPushButton("Clear")
        self.clear_measured.setEnabled(False)
        self.export_svg = QPushButton("Save SVG…")
        self.export_pdf = QPushButton("Save PDF…")
        self.export_csv = QPushButton("Save pattern…")
        rows.addLayout(chrome.field_grid([
            self.load_measured, self.clear_measured,
            self.export_svg, self.export_pdf,
            (self.export_csv, 2),
        ]))
        return group

    def _connect(self) -> None:
        for widget in (self.radiation, self.wavelength_choice):
            widget.currentIndexChanged.connect(self._on_experiment)
        for widget in (self.wavelength, self.two_theta_min, self.two_theta_max,
                       self.b_iso, self.monochromator):
            widget.valueChanged.connect(self.recompute)
        for widget in (self.step, self.u, self.v, self.w, self.eta,
                       self.zero_shift):
            widget.valueChanged.connect(self.redraw)
        for widget in (self.show_ticks, self.show_sticks,
                       self.show_difference, self.fill):
            widget.toggled.connect(self.redraw)
        self.load_measured.clicked.connect(self._choose_measured)
        self.clear_measured.clicked.connect(self._clear_measured)
        self.export_svg.clicked.connect(lambda: self._export("svg"))
        self.export_pdf.clicked.connect(lambda: self._export("pdf"))
        self.export_csv.clicked.connect(self._export_csv)

    def _on_experiment(self) -> None:
        """A named wavelength or a change of radiation fills in the exact value."""
        value = self.wavelength_choice.currentData()
        if value is not None:
            self.wavelength.blockSignals(True)
            self.wavelength.setValue(float(value))
            self.wavelength.blockSignals(False)
        self.recompute()

    # -- content -----------------------------------------------------------
    def apply_theme(self, theme) -> None:
        self.plot.apply_theme(theme)

    def set_structure(self, structure) -> None:
        """Adopt a structure, recomputing only if it is a different one.

        A pattern depends on the cell and its contents and on nothing else --
        not on the bond-valence cutoff, not on which site is selected. The main
        window refreshes its panels on every threshold move, so without this
        guard the whole enumeration would run again for no change in the answer.
        """
        if structure is self.structure:
            return
        self.structure = structure
        self.recompute(keep_view=False)

    def recompute(self, keep_view: bool = True) -> None:
        if self.structure is None:
            self.pattern = None
            self.table.setRowCount(0)
            self.plot.set_series([])
            self.plot.set_ticks([])
            self.notes.setText("")
            return
        try:
            self.pattern = dif.powder_pattern(
                self.structure,
                wavelength=float(self.wavelength.value()),
                two_theta_min=float(self.two_theta_min.value()),
                two_theta_max=float(self.two_theta_max.value()),
                radiation=self.radiation.currentText(),
                b_iso=float(self.b_iso.value()),
                monochromator_two_theta=(float(self.monochromator.value())
                                         or None))
        except ValueError as error:
            self.pattern = None
            self.notes.setText(str(error))
            self.plot.set_series([])
            self.plot.set_ticks([])
            self.table.setRowCount(0)
            return
        self._fill_table()
        self.redraw(keep_view=keep_view)

    def _fill_table(self) -> None:
        pattern = self.pattern
        rows = pattern.reflections if pattern else []
        self.table.setRowCount(len(rows))
        for r, line in enumerate(rows):
            values = [line.hkl, f"{line.d:.4f}", f"{line.two_theta:.3f}",
                      f"{line.intensity:.2f}", str(line.multiplicity),
                      f"{line.f_squared:.3e}",
                      f"{line.lorentz_polarisation:.3f}"]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()

    def redraw(self, keep_view: bool = True) -> None:
        pattern = self.pattern
        if pattern is None or not pattern.reflections:
            self.plot.set_series([])
            self.plot.set_ticks([])
            if pattern is not None:
                self.notes.setText(" · ".join(pattern.notes))
            return

        low = float(self.two_theta_min.value())
        high = float(self.two_theta_max.value())
        series: list[Series] = []

        if self.show_sticks.isChecked():
            x, y = [], []
            for line in pattern.reflections:
                position = line.two_theta + float(self.zero_shift.value())
                x.extend([position, position, position])
                y.extend([0.0, line.intensity, 0.0])
            calculated_x = np.array(x)
            calculated_y = np.array(y)
        else:
            calculated_x, calculated_y = pattern.profile(
                two_theta_min=low, two_theta_max=high,
                step=float(self.step.value()), u=float(self.u.value()),
                v=float(self.v.value()), w=float(self.w.value()),
                eta=float(self.eta.value()),
                zero_shift=float(self.zero_shift.value()))

        plotted_y = calculated_y
        self._scale_factor = 1.0
        if self.measured is not None:
            measured_x, measured_y = self.measured
            plotted_y, self._scale_factor = dif.scale_to_measured(
                calculated_x, calculated_y, measured_x, measured_y)
            series.append(Series(measured_x, measured_y,
                                 label=f"measured — {self.measured_name}",
                                 color=MEASURED, width=1.1))

        series.append(Series(calculated_x, plotted_y,
                             label=f"calculated — {pattern.radiation}",
                             color=CALCULATED, width=1.5,
                             filled=self.fill.isChecked()))

        if self.measured is not None and self.show_difference.isChecked():
            measured_x, measured_y = self.measured
            x, difference, _ = dif.difference_curve(
                calculated_x, calculated_y, measured_x, measured_y)
            span = float(np.max(np.abs(difference))) if len(difference) else 0.0
            baseline = -(span + 0.08 * max(float(np.max(plotted_y)), 1.0))
            series.append(Series(x, difference, label="difference",
                                 color=DIFFERENCE, width=1.0,
                                 offset=baseline))

        self.plot.set_series(series, keep_view=keep_view)

        ticks = []
        if self.show_ticks.isChecked():
            positions = np.array([r.two_theta + float(self.zero_shift.value())
                                  for r in pattern.reflections])
            ticks.append(Ticks(positions, color=TICK, row=0))
        self.plot.set_ticks(ticks)

        self.plot.set_labels(
            x="2θ / °",
            y="intensity (relative)",
            title=(f"{pattern.structure_name or 'structure'} — "
                   f"{pattern.radiation}, λ = {pattern.wavelength:.5f} "
                   "Å"),
            footnote=self._footnote())
        self.notes.setText(" · ".join(pattern.notes))

    def _footnote(self) -> str:
        if self.pattern is None:
            return ""
        parts = [f"{len(self.pattern.reflections)} lines"]
        if self.measured is not None:
            parts.append(f"scale factor {self._scale_factor:.4g} "
                         "(least squares, the only fitted quantity)")
        return " · ".join(parts)

    # -- measured data -----------------------------------------------------
    def _choose_measured(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measured powder pattern", "",
            dif.PATTERN_FILE_FILTER)
        if not path:
            return
        try:
            self.measured = dif.read_pattern(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not read the pattern", str(error))
            return
        from pathlib import Path

        self.measured_name = Path(path).name
        self.clear_measured.setEnabled(True)
        x = self.measured[0]
        # follow the measurement's own range, which is what you want to compare
        self.two_theta_min.blockSignals(True)
        self.two_theta_max.blockSignals(True)
        self.two_theta_min.setValue(max(float(np.min(x)), 0.0))
        self.two_theta_max.setValue(min(float(np.max(x)), 180.0))
        self.two_theta_min.blockSignals(False)
        self.two_theta_max.blockSignals(False)
        self.recompute(keep_view=False)

    def _clear_measured(self) -> None:
        self.measured = None
        self.measured_name = ""
        self.clear_measured.setEnabled(False)
        self.redraw(keep_view=True)

    # -- export ------------------------------------------------------------
    def _export(self, kind: str) -> None:
        if self.pattern is None:
            return
        name = (self.pattern.structure_name or "pattern").replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, f"Save the plot as {kind.upper()}", f"{name}.{kind}",
            f"{kind.upper()} (*.{kind})")
        if not path:
            return
        try:
            if kind == "svg":
                self.plot.save_svg(path)
            else:
                self.plot.save_pdf(path)
        except Exception as error:                      # pragma: no cover
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self.readout.setText(f"wrote {path}")

    def _export_csv(self) -> None:
        if self.pattern is None:
            return
        name = (self.pattern.structure_name or "pattern").replace(" ", "_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the calculated pattern", f"{name}_pattern.csv",
            "CSV (*.csv);;Two-column profile (*.xy)")
        if not path:
            return
        from pathlib import Path

        pattern = self.pattern
        try:
            if str(path).lower().endswith(".xy"):
                x, y = pattern.profile(
                    two_theta_min=float(self.two_theta_min.value()),
                    two_theta_max=float(self.two_theta_max.value()),
                    step=float(self.step.value()), u=float(self.u.value()),
                    v=float(self.v.value()), w=float(self.w.value()),
                    eta=float(self.eta.value()),
                    zero_shift=float(self.zero_shift.value()))
                body = "\n".join(f"{a:.5f} {b:.6f}" for a, b in zip(x, y))
                header = "\n".join(f"# {note}" for note in pattern.notes)
                Path(path).write_text(f"{header}\n{body}\n", encoding="utf-8")
            else:
                lines = ["# " + note for note in pattern.notes]
                lines.append("h,k,l,d,two_theta,intensity,multiplicity,"
                             "f_squared,lorentz_polarisation")
                for r in pattern.reflections:
                    lines.append(
                        f"{r.h},{r.k},{r.l},{r.d:.6f},{r.two_theta:.5f},"
                        f"{r.intensity:.4f},{r.multiplicity},"
                        f"{r.f_squared:.6e},{r.lorentz_polarisation:.6f}")
                Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self.readout.setText(f"wrote {path}")

    # -- interaction -------------------------------------------------------
    def _nearest(self, two_theta: float):
        if self.pattern is None or not self.pattern.reflections:
            return None
        shift = float(self.zero_shift.value())
        return min(self.pattern.reflections,
                   key=lambda r: abs(r.two_theta + shift - two_theta))

    def _on_hover(self, x: float, y: float) -> None:
        line = self._nearest(x)
        if line is None:
            self.readout.setText("")
            return
        shift = float(self.zero_shift.value())
        self.readout.setText(
            f"2θ = {x:.3f}°    nearest line {line.hkl}  "
            f"at {line.two_theta + shift:.3f}°   d = {line.d:.4f} Å   "
            f"I = {line.intensity:.2f}   multiplicity {line.multiplicity}")

    def _on_pick(self, x: float, y: float) -> None:
        line = self._nearest(x)
        if line is None:
            return
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.text() == line.hkl:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                break
        self.reflection_selected.emit(line.h, line.k, line.l)

    def _on_row(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows or self.pattern is None:
            return
        index = rows[0].row()
        if 0 <= index < len(self.pattern.reflections):
            line = self.pattern.reflections[index]
            self.reflection_selected.emit(line.h, line.k, line.l)
