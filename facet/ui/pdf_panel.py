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
        group = QGroupBox()
        rows = QVBoxLayout(group)
        rows.setSpacing(4)

        top = QHBoxLayout()
        top.setSpacing(12)
        rows.addLayout(top)

        self.radiation = QComboBox()
        for name in dif.RADIATIONS:
            self.radiation.addItem(name)
        self.radiation.setToolTip(
            "The weights w_ij come from this. An X-ray PDF of a heavy-element "
            "compound is dominated by the heavy pairs; a neutron PDF of the "
            "same compound is not. The table below gives the split.")
        top.addWidget(QLabel("Radiation"))
        top.addWidget(self.radiation)

        self.r_max = _spin(5.0, 100.0, 20.0, 1.0, 1, " Å")
        top.addWidget(QLabel("r max"))
        top.addWidget(self.r_max)

        self.dr = _spin(0.001, 0.05, 0.01, 0.002, 3, " Å")
        self.dr.setToolTip(
            "The r grid. Nyquist asks for dr ≤ π/Qmax (0.10 Å at Qmax = 30), "
            "and the Gaussians ask for roughly σ/5.")
        top.addWidget(QLabel("dr"))
        top.addWidget(self.dr)

        self.u_iso = _spin(0.0, 0.2, pdf_mod.DEFAULT_U_ISO, 0.001, 5, " Å²")
        self.u_iso.setToolTip(
            "Used only for atoms whose site gives no U in the file. The notes "
            "say how many that was. Derived from the same constant the "
            "diffraction panel uses, so the two cannot disagree.")
        top.addWidget(QLabel("U if absent"))
        top.addWidget(self.u_iso)

        self.which = QComboBox()
        for label in ("G(r)  reduced", "g(r)  pair distribution",
                      "R(r)  radial distribution"):
            self.which.addItem(label)
        top.addWidget(QLabel("Show"))
        top.addWidget(self.which)
        top.addStretch(1)

        bottom = QHBoxLayout()
        bottom.setSpacing(12)
        rows.addLayout(bottom)

        self.q_min = _spin(0.0, 10.0, 0.0, 0.1, 2, " Å⁻¹")
        bottom.addWidget(QLabel("Q min"))
        bottom.addWidget(self.q_min)

        self.q_max = _spin(0.0, 60.0, 0.0, 1.0, 2, " Å⁻¹")
        self.q_max.setToolTip(
            "0 means no truncation: the ideal, infinite-Q PDF. Set it to a "
            "measurement's Qmax and the amplitude loss and the termination "
            "ripple below the first peak appear, because truncation is a "
            "convolution in r.")
        bottom.addWidget(QLabel("Q max"))
        bottom.addWidget(self.q_max)

        self.window = QComboBox()
        for name in pdf_mod.WINDOWS:
            self.window.addItem(name)
        bottom.addWidget(QLabel("window"))
        bottom.addWidget(self.window)

        self.q_damp = _spin(0.0, 0.2, 0.0, 0.005, 4, " Å⁻¹")
        self.q_damp.setToolTip(
            "Finite Q-resolution: G(r) × exp(−(r·Qdamp)²/2). The dual of the "
            "truncation above — a convolution in Q is a multiplication in r.")
        bottom.addWidget(QLabel("Q damp"))
        bottom.addWidget(self.q_damp)

        self.delta1 = _spin(0.0, 5.0, 0.0, 0.05, 3, " Å")
        self.delta1.setToolTip(
            "Correlated motion, the high-temperature form: σ′ = σ√(1 − δ₁/r). "
            "Zero by default because it is fitted against data and cannot be "
            "measured from a structure. Near neighbours move together, so a "
            "measured first peak is narrower than σ² = U_i + U_j.")
        bottom.addWidget(QLabel("δ₁"))
        bottom.addWidget(self.delta1)

        self.delta2 = _spin(0.0, 10.0, 0.0, 0.1, 3, " Å²")
        self.delta2.setToolTip("The same, in the low-temperature form: σ′ = "
                               "σ√(1 − δ₂/r²). Also zero by default.")
        bottom.addWidget(QLabel("δ₂"))
        bottom.addWidget(self.delta2)
        bottom.addStretch(1)

        third = QHBoxLayout()
        third.setSpacing(10)
        rows.addLayout(third)

        self.load_measured = QPushButton("Load measured G(r)…")
        third.addWidget(self.load_measured)
        self.clear_measured = QPushButton("Clear")
        third.addWidget(self.clear_measured)
        self.show_difference = QCheckBox("difference")
        self.show_difference.setChecked(True)
        third.addWidget(self.show_difference)
        third.addStretch(1)
        self.export_csv = QPushButton("Export CSV…")
        third.addWidget(self.export_csv)
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
