"""The pair distribution function panel.

G(r) calculated from the structure, with a measured G(r) optionally over it and
the difference beneath. Everything that shapes the calculation is on the panel:
the radiation, the displacement parameter used where the file gave none, the
correlation parameters, and the instrument's Q range and resolution.

The panel says which of those came from the structure and which were assumed,
and it puts the pair-weight table on screen, because the single most useful fact
about a PDF of a heavy-element compound is how little of it is the light atoms:
the X-ray PDF of Bi2O3 is three quarters Bi-Bi.

Selecting a range on the plot integrates R(r) over it and reports the
scattering-weighted coordination number. No fit quality is computed anywhere,
and the only thing fitted to measured data is one scale factor, shown as a
number.
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
from ..core import pdf as pdf_mod
from . import chrome
from .plot import Plot, Series

CALCULATED = (120, 190, 240)
MEASURED = (232, 148, 96)
DIFFERENCE = (150, 200, 140)


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


class PDFPanel(QWidget):
    """G(r) for the loaded structure."""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure = None
        self.pdf = None
        self.measured = None
        self.measured_name = ""
        self._scale_factor = 1.0
        self._loading = True

        self.plot = Plot()
        self.plot.hovered.connect(self._on_hover)
        self.plot.picked.connect(self._on_pick)
        self._band: list[float] = []

        self.readout = QLabel("")
        chrome.mark_hint(self.readout)

        self.notes = QLabel("")
        self.notes.setWordWrap(True)
        small = QFont(self.notes.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.0))
        self.notes.setFont(small)
        chrome.mark_hint(self.notes)

        self.weights_table = QTableWidget(0, 3)
        self.weights_table.setHorizontalHeaderLabels(
            ["pair", "weight", "b (this radiation)"])
        self.weights_table.verticalHeader().setVisible(False)
        self.weights_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.weights_table.setAlternatingRowColors(True)
        self.weights_table.setFont(small)

        self._build_layout()
        self._connect()
        self._loading = False

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        hint = QLabel(
            "G(r) = R(r)/r − 4πrρ₀, from every interatomic distance in the "
            "structure. The distances and their degeneracies are exact; the "
            "peak widths use the file's displacement parameters where it gives "
            "them; Qmax, Qdamp and the correlation terms are properties of a "
            "measurement and are yours to set. This is the PDF of the average "
            "crystal — a disordered material's measured PDF differs from it by "
            "design.")
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
        split.addWidget(self.weights_table)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        outer.addWidget(split, 1)
        outer.addWidget(self.notes)

    def _controls(self) -> QWidget:
        """Two rows of what is reached for, and the instrument behind a button.

        Every control names itself -- the spin boxes carry their name in the
        prefix -- because a separate label column costs more width than this
        panel has. See ``chrome.name_inside``.
        """
        group = QGroupBox()
        rows = QVBoxLayout(group)
        rows.setContentsMargins(8, 8, 8, 8)
        rows.setSpacing(chrome.FIELD_SPACING)

        # -- what the structure gives ------------------------------------
        self.radiation = QComboBox()
        for name in dif.RADIATIONS:
            self.radiation.addItem(name)
        chrome.name_inside(
            self.radiation, "radiation",
            tip="The weights w_ij come from this. An X-ray PDF of a "
                "heavy-element compound is dominated by the heavy pairs; a "
                "neutron PDF of the same compound is not. The table below "
                "gives the split.")

        self.which = QComboBox()
        # Short items on purpose: the full name is in the tool tip and on the
        # plot's own y axis, and this combo was measured getting a text area of
        # zero pixels for 156 px of text.
        for label, explanation in (
                ("G(r)", "the reduced PDF, R(r)/r - 4 pi r rho0"),
                ("g(r)", "the pair distribution function, which tends to 1"),
                ("R(r)", "the radial distribution; a peak's area is the "
                         "scattering-weighted coordination number")):
            self.which.addItem(label)
            self.which.setItemData(self.which.count() - 1, explanation,
                                   Qt.ToolTipRole)
        chrome.name_inside(self.which, "show",
                           tip="Which of the three forms to plot. They are one "
                               "curve written three ways.")

        self.r_max = chrome.name_inside(
            _spin(5.0, 100.0, 20.0, 1.0, 1), "r to", " Å",
            tip="How far out to compute. Every distance in the structure is "
                "exact to any radius; this only sets where the plot stops.")
        self.dr = chrome.name_inside(
            _spin(0.001, 0.05, 0.01, 0.002, 3), "dr", " Å",
            tip="The r grid. Nyquist asks for dr <= pi/Qmax (0.10 A at "
                "Qmax = 30), and the Gaussians ask for roughly sigma/5.")

        rows.addLayout(chrome.field_grid([
            self.radiation, self.which,
            self.r_max, self.dr,
        ]))

        # -- what a measurement adds, behind a disclosure -----------------
        self.q_min = chrome.name_inside(
            _spin(0.0, 10.0, 0.0, 0.1, 2), "Q from", " Å⁻¹",
            tip="The low-Q end of the transform. Above zero the kernel's "
                "integral becomes zero rather than one, which removes the mean "
                "of G(r) -- what excluding the small-Q region of a measurement "
                "does.")
        self.q_max = chrome.name_inside(
            _spin(0.0, 60.0, 0.0, 1.0, 2), "Q to", " Å⁻¹",
            tip="0 means no truncation: the ideal, infinite-Q PDF. Set it to a "
                "measurement's Qmax and the amplitude loss and the termination "
                "ripple below the first peak appear, because truncation is a "
                "convolution in r.")
        self.window = QComboBox()
        for name in pdf_mod.WINDOWS:
            self.window.addItem(name)
        chrome.name_inside(self.window, "window",
                           tip="The shape cut out of Q. A boxcar has the "
                               "closed-form kernel above; Lorch is applied by "
                               "transforming, windowing and transforming back.")
        self.q_damp = chrome.name_inside(
            _spin(0.0, 0.2, 0.0, 0.005, 4), "Qdamp", " Å⁻¹",
            tip="Finite Q-resolution: G(r) x exp(-(r Qdamp)^2/2). The dual of "
                "the truncation above -- a convolution in Q is a "
                "multiplication in r.")
        self.delta1 = chrome.name_inside(
            _spin(0.0, 5.0, 0.0, 0.05, 3), "δ₁", " Å",
            tip="Correlated motion, the high-temperature form: "
                "sigma' = sigma sqrt(1 - d1/r). Zero by default because it is "
                "fitted against data and cannot be measured from a structure. "
                "Near neighbours move together, so a measured first peak is "
                "narrower than sigma^2 = U_i + U_j.")
        self.delta2 = chrome.name_inside(
            _spin(0.0, 10.0, 0.0, 0.1, 3), "δ₂", " Å²",
            tip="The same, in the low-temperature form: "
                "sigma' = sigma sqrt(1 - d2/r^2). Also zero by default.")
        self.u_iso = chrome.name_inside(
            _spin(0.0, 0.2, pdf_mod.DEFAULT_U_ISO, 0.001, 5), "U₀", " Å²",
            tip="Used only for atoms whose site gives no U in the file. The "
                "notes say how many that was. Derived from the same constant "
                "the diffraction panel uses, so the two cannot disagree.")

        instrument = QWidget()
        inner = QVBoxLayout(instrument)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addLayout(chrome.field_grid([
            self.q_min, self.q_max,
            self.window, self.q_damp,
            self.delta1, self.delta2,
            self.u_iso,
        ]))
        rows.addWidget(chrome.disclosure(
            "Instrument, correlation and the U fallback", instrument))

        # -- measured data -------------------------------------------------
        self.load_measured = QPushButton("Load measured G(r)…")
        self.clear_measured = QPushButton("Clear")
        self.show_difference = QCheckBox("difference")
        self.show_difference.setChecked(True)
        self.export_csv = QPushButton("Export CSV…")
        rows.addLayout(chrome.field_grid([
            self.load_measured, self.clear_measured,
            self.show_difference, self.export_csv,
        ]))
        return group

    def _connect(self) -> None:
        for widget in (self.r_max, self.dr, self.u_iso, self.q_min, self.q_max,
                       self.q_damp, self.delta1, self.delta2):
            widget.valueChanged.connect(lambda _=0: self.recompute())
        self.radiation.currentIndexChanged.connect(lambda _=0: self.recompute())
        self.window.currentIndexChanged.connect(lambda _=0: self.recompute())
        self.which.currentIndexChanged.connect(lambda _=0: self.redraw())
        self.show_difference.toggled.connect(lambda _=False: self.redraw())
        self.load_measured.clicked.connect(self._choose_measured)
        self.clear_measured.clicked.connect(self._clear_measured)
        self.export_csv.clicked.connect(self._export_csv)

    def apply_theme(self, theme) -> None:
        self.plot.apply_theme(theme)

    # -- computing ---------------------------------------------------------
    def set_structure(self, structure) -> None:
        """Adopt a structure. A PDF depends on the cell and its contents only.

        Not on the bond-valence threshold, and not on which site is selected --
        so the main window's refresh on every threshold move must not recompute
        it.
        """
        if structure is self.structure:
            return
        self.structure = structure
        self._band = []
        self.recompute(keep_view=False)

    def recompute(self, keep_view: bool = True) -> None:
        if self._loading:
            return
        if self.structure is None:
            self.pdf = None
            self.plot.set_series([])
            self.weights_table.setRowCount(0)
            self.notes.setText("")
            return
        try:
            self.pdf = pdf_mod.pair_distribution(
                self.structure,
                r_max=float(self.r_max.value()),
                dr=float(self.dr.value()),
                radiation=self.radiation.currentText(),
                u_iso=float(self.u_iso.value()),
                delta1=float(self.delta1.value()),
                delta2=float(self.delta2.value()),
                q_min=float(self.q_min.value()),
                q_max=float(self.q_max.value()),
                q_damp=float(self.q_damp.value()),
                window=self.window.currentText())
        except ValueError as exc:
            self.pdf = None
            self.plot.set_series([])
            self.notes.setText(str(exc))
            return
        self._fill_weights()
        self.redraw(keep_view=keep_view)

    def _fill_weights(self) -> None:
        weights = self.pdf.weights if self.pdf else None
        if weights is None or not weights.pairs:
            self.weights_table.setRowCount(0)
            return
        rows = sorted(weights.pairs.items(), key=lambda kv: -kv[1])
        self.weights_table.setRowCount(len(rows))
        for row, ((a, b), share) in enumerate(rows):
            self.weights_table.setItem(row, 0, QTableWidgetItem(f"{a}–{b}"))
            item = QTableWidgetItem(f"{100.0 * share:.1f} %")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.weights_table.setItem(row, 1, item)
            text = ", ".join(f"{s} {weights.b.get(s, float('nan')):.3f}"
                             for s in dict.fromkeys((a, b)))
            self.weights_table.setItem(row, 2, QTableWidgetItem(text))
        self.weights_table.resizeColumnsToContents()

    # -- drawing -----------------------------------------------------------
    def _curve(self):
        """The chosen curve, and how to label its axis."""
        which = self.which.currentIndex()
        if which == 1:
            return self.pdf.g, "g(r)"
        if which == 2:
            return self.pdf.R, "R(r) / Å⁻¹"
        return self.pdf.G, "G(r) / Å⁻²"

    def redraw(self, keep_view: bool = True) -> None:
        if self.pdf is None or not len(self.pdf.r):
            self.plot.set_series([])
            return
        y, axis = self._curve()
        series: list[Series] = []
        plotted = y
        self._scale_factor = 1.0

        if self.measured is not None:
            mx, my = self.measured
            plotted, self._scale_factor = dif.scale_to_measured(
                self.pdf.r, y, mx, my)
            series.append(Series(mx, my,
                                 label=f"measured — {self.measured_name}",
                                 color=MEASURED, width=1.1))
        series.append(Series(self.pdf.r, plotted,
                             label=f"calculated — {self.pdf.radiation}",
                             color=CALCULATED, width=1.4))

        if self.measured is not None and self.show_difference.isChecked():
            mx, my = self.measured
            x, difference, _ = dif.difference_curve(self.pdf.r, y, mx, my)
            span = float(np.max(np.abs(difference))) if len(difference) else 0.0
            baseline = -(span + 0.08 * max(float(np.max(np.abs(plotted))), 1.0))
            series.append(Series(x, difference, label="difference",
                                 color=DIFFERENCE, width=1.0,
                                 offset=baseline))

        self.plot.set_series(series, keep_view=keep_view)
        self.plot.set_labels(
            x="r / Å", y=axis,
            title=(f"{self.pdf.structure_name or 'structure'} — "
                   f"{self.pdf.radiation} PDF"
                   + (f", Qmax = {self.pdf.q_max:g} Å⁻¹"
                      if self.pdf.q_max else ", no Qmax")),
            footnote=self._footnote())
        self.notes.setText(" · ".join(self.pdf.notes))

    def _footnote(self) -> str:
        if self.pdf is None:
            return ""
        parts = [f"ρ₀ = {self.pdf.rho0:.5f} Å⁻³",
                 f"{self.pdf.n_cell:g} atoms/cell",
                 f"{self.pdf.n_pairs} ordered pairs"]
        if self.measured is not None:
            parts.append(f"scale factor {self._scale_factor:.4g} "
                         "(least squares, the only fitted quantity)")
        if len(self._band) == 2:
            lo, hi = sorted(self._band)
            area = self.pdf.coordination_in(lo, hi)
            parts.append(f"∫R(r) over {lo:.3f}–{hi:.3f} Å = {area:.3f} "
                         "(scattering-weighted CN)")
        return " · ".join(parts)

    # -- interaction -------------------------------------------------------
    def _on_hover(self, x: float, y: float) -> None:
        if self.pdf is None or not len(self.pdf.r):
            self.readout.setText("")
            return
        index = int(np.clip(np.searchsorted(self.pdf.r, x), 0,
                            len(self.pdf.r) - 1))
        r = float(self.pdf.r[index])
        self.readout.setText(
            f"r = {r:.4f} Å    G = {self.pdf.G[index]:+.4f} Å⁻²    "
            f"g = {self.pdf.g[index]:.4f}    R = {self.pdf.R[index]:.4f} Å⁻¹")

    def _on_pick(self, x: float, y: float) -> None:
        """Two clicks set a band; its integral is the weighted CN of the shell."""
        if self.pdf is None:
            return
        if len(self._band) >= 2:
            self._band = []
        self._band.append(float(x))
        if len(self._band) == 2:
            lo, hi = sorted(self._band)
            area = self.pdf.coordination_in(lo, hi)
            self.statusMessage.emit(
                f"∫R(r) dr over {lo:.3f}–{hi:.3f} Å = {area:.4f}. This is the "
                "scattering-weighted coordination number of that shell, not an "
                "atom count: the pair-weight table gives the split.")
        else:
            self.statusMessage.emit(
                f"Band start r = {self._band[0]:.3f} Å — click again to close "
                "it and integrate R(r).")
        self.redraw()

    # -- measured data -----------------------------------------------------
    def _choose_measured(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a measured G(r)", "", dif.PATTERN_FILE_FILTER)
        if not path:
            return
        try:
            x, y = dif.read_pattern(path)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not read that file", str(exc))
            return
        from pathlib import Path

        self.measured = (x, y)
        self.measured_name = Path(path).name
        self.redraw(keep_view=False)

    def _clear_measured(self) -> None:
        self.measured = None
        self.measured_name = ""
        self.redraw()

    def _export_csv(self) -> None:
        if self.pdf is None:
            QMessageBox.information(self, "Nothing to export",
                                    "No pair distribution function has been "
                                    "computed yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the pair distribution function", "pdf.csv",
            "CSV (*.csv)")
        if not path:
            return
        pdf_mod.to_csv(self.pdf, path)
        self.statusMessage.emit(f"Wrote {path}")
