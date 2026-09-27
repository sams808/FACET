"""The EXAFS panel.

Three tabs, and the split between them is the point.

**Shells** is what the structure says: the shell list an Artemis or Larch fit
starts from, with two independently sourced estimates of sigma^2 and the
contributing site labels.

**Resolution** is what a k range can do with that list: dR = pi/(2 dk), the
number of independent points, which shells are closer together than dR, and how
many parameters the list would need against how many the range supports. This is
the coordination-number argument applied to EXAFS -- a shell separation that
survives the resolution and one that does not are different situations, and the
panel states which you have.

**FEFF** is what a real scattering calculation says. FACET writes the input,
does not bundle the program, and runs it only if pointed at an executable. What
comes back -- degeneracies, effective path lengths, amplitude ratios, and the
k-dependent amplitude and phase of every path -- is shown beside the
crystallographic shells and labelled as FEFF's.

chi(k) appears only when FEFF path files are present. FACET does not compute it
from the structure alone: without FEFF's phase shifts the peaks of its Fourier
transform land about 0.4 A from the distances that produced them, which is the
one quantity anyone would read off such a curve.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core import exafs as exafs_mod
from . import chrome
from .plot import Plot, Series

CHI = (120, 190, 240)
FEFF_CHI = (232, 148, 96)


def _table(headers):
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setAlternatingRowColors(True)
    font = QFont(t.font())
    font.setPointSizeF(max(7.5, font.pointSizeF() - 0.5))
    t.setFont(font)
    return t


def _fill(table, rows):
    table.setRowCount(len(rows))
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            if isinstance(value, float):
                text = "" if value != value else f"{value:.4f}"
            else:
                text = "" if value is None else str(value)
            item = QTableWidgetItem(text)
            if isinstance(value, (int, float)):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()


def _spin(low, high, value, step, decimals=2, suffix=""):
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    box.setKeyboardTracking(False)
    return box


class ExafsPanel(QWidget):
    """Shells, resolution, and FEFF read back."""

    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.structure = None
        self.site_index = None
        self.table = None
        self.resolution = None
        self.paths: list = []
        self.feff_directory = None
        self.theme = None
        self._loading = True

        self.shells = _table(["element", "Z", "N", "N (occupancy)", "R / Å",
                              "spread / Å", "σ² file / Å²",
                              "σ² Einstein / Å²", "μ / amu", "sites"])
        self.shell_notes = QLabel("")
        self.shell_notes.setWordWrap(True)
        small = QFont(self.shell_notes.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1.0))
        self.shell_notes.setFont(small)
        chrome.mark_hint(self.shell_notes)

        self.report = QTextBrowser()
        self.report.setFrameShape(QTextBrowser.NoFrame)

        self.path_table = _table(["FEFF file", "kind", "legs", "r_eff / Å",
                                  "degeneracy", "amplitude / %",
                                  "FACET shell", "shell R / Å", "shell N"])
        self.chi_plot = Plot()
        self.chi_notes = QLabel("")
        self.chi_notes.setWordWrap(True)
        self.chi_notes.setFont(small)
        chrome.mark_hint(self.chi_notes)

        self._build_layout()
        self._connect()
        self._loading = False

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        hint = QLabel(
            "The shell list an EXAFS fit starts from, and what a k range can "
            "separate in it. χ(k) is not computed from the structure: without "
            "FEFF's phase shifts its Fourier transform peaks land about 0.4 Å "
            "from the distances that made them. Point the FEFF tab at a "
            "calculation and χ(k) is assembled from FEFF's amplitudes with "
            "these degeneracies and σ².")
        hint.setWordWrap(True)
        tiny = QFont(hint.font())
        tiny.setPointSizeF(max(7.0, tiny.pointSizeF() - 1.0))
        hint.setFont(tiny)
        chrome.mark_hint(hint)
        outer.addWidget(hint)

        outer.addWidget(self._controls())

        self.tabs = QTabWidget()
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.shells, 1)
        box.addWidget(self.shell_notes)
        self.tabs.addTab(holder, "Shells")
        self.tabs.addTab(self.report, "Resolution")

        feff = QWidget()
        column = QVBoxLayout(feff)
        column.setContentsMargins(0, 0, 0, 0)
        column.addWidget(self._feff_controls())
        column.addWidget(self.path_table, 2)
        column.addWidget(self.chi_plot, 3)
        column.addWidget(self.chi_notes)
        self.tabs.addTab(feff, "FEFF")
        outer.addWidget(self.tabs, 1)

    def _controls(self) -> QWidget:
        """Self-labelling controls, two per row, with the knobs behind a button.

        The spin boxes carry their names in their prefixes; a label column of
        its own cost more width than the panel has, and "parameters/shell" was
        rendering as "para".
        """
        group = QGroupBox()
        rows = QVBoxLayout(group)
        rows.setContentsMargins(8, 8, 8, 8)
        rows.setSpacing(chrome.FIELD_SPACING)

        self.absorber = QComboBox()
        chrome.name_inside(self.absorber, "absorber",
                           tip="Which site the absorbing atom is. The shells "
                               "below are its neighbours.")

        self.r_max = chrome.name_inside(
            _spin(3.0, 12.0, 6.0, 0.5, 1), "cluster", " Å",
            tip="How far out to gather neighbours, and the cluster radius "
                "written into a FEFF input.")
        self.tolerance = chrome.name_inside(
            _spin(0.005, 0.5, 0.05, 0.005, 3), "shell", " Å",
            tip="Contacts of one element within this of each other are one "
                "shell. A contact joins only if it is within the tolerance of "
                "every member, so no shell is ever wider than this.")

        self.k_min = chrome.name_inside(
            _spin(0.5, 20.0, 3.0, 0.5, 2), "k from", " Å⁻¹",
            tip="The k range you intend to fit. It sets the resolution "
                "dR = pi/(2 dk) reported on the Resolution tab.")
        self.k_max = chrome.name_inside(
            _spin(2.0, 30.0, 14.0, 0.5, 2), "k to", " Å⁻¹",
            tip="The high-k end of the fitting range.")

        self.r_window_min = chrome.name_inside(
            _spin(0.0, 10.0, 1.0, 0.1, 2), "r from", " Å",
            tip="The r window of the fit, which sets how many independent "
                "points the data holds.")
        self.r_window_max = chrome.name_inside(
            _spin(1.0, 12.0, 4.5, 0.1, 2), "r to", " Å",
            tip="The outer edge of the fitting window.")

        rows.addLayout(chrome.field_grid([
            (self.absorber, 2),
            self.r_max, self.tolerance,
            self.k_min, self.k_max,
            self.r_window_min, self.r_window_max,
        ]))

        self.temperature = chrome.name_inside(
            _spin(0.0, 1500.0, 300.0, 10.0, 0), "T", " K",
            tip="The temperature of the Einstein sigma^2 model. It is a model, "
                "not a measurement of this structure.")
        self.theta_e = chrome.name_inside(
            _spin(50.0, 1200.0, 500.0, 10.0, 0), "θE", " K",
            tip="The Einstein temperature of the sigma^2 model. A knob: at "
                "500 K and 300 K a Bi-O pair gives 0.0048 A^2.")
        self.per_shell = QSpinBox()
        self.per_shell.setRange(1, 6)
        self.per_shell.setValue(3)
        chrome.name_inside(
            self.per_shell, "par/shell",
            tip="Parameters per shell in the count against N_idp. Three is N, "
                "R and sigma^2, with S0^2 and E0 shared between shells.")

        knobs = QWidget()
        inner = QVBoxLayout(knobs)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addLayout(chrome.field_grid([
            self.temperature, self.theta_e,
            (self.per_shell, 2),
        ]))
        rows.addWidget(chrome.disclosure(
            "Temperature, the sigma-squared model, and the parameter count",
            knobs))

        self.export_button = QPushButton("Export shells…")
        rows.addLayout(chrome.field_grid([(self.export_button, 2)]))
        return group

    def _feff_controls(self) -> QWidget:
        group = QGroupBox()
        rows = QVBoxLayout(group)
        rows.setContentsMargins(8, 8, 8, 8)
        rows.setSpacing(chrome.FIELD_SPACING)

        self.write_input = QPushButton("Write feff.inp…")
        self.write_input.setToolTip(
            "A FEFF input for this absorber, with the cluster larger than "
            "RPATH and PRINT set so the path files are written.")
        self.read_output = QPushButton("Read a calculation…")
        self.read_output.setToolTip(
            "Point at a folder holding files.dat and feffNNNN.dat. FACET "
            "reads them; it does not run FEFF and does not ship it.")
        self.run_feff = QPushButton("Run FEFF here…")
        self.run_feff.setToolTip(
            "Runs an feff8l or feff6l executable you already have, in a folder "
            "you choose. Nothing is bundled.")

        self.s02 = chrome.name_inside(
            _spin(0.1, 1.5, 0.9, 0.05, 2), "S₀²",
            tip="The amplitude reduction factor applied to the path sum. Your "
                "value, not a fitted one.")
        self.use_sigma2 = QComboBox()
        for label in ("σ² Einstein", "σ² from the file", "σ² = 0"):
            self.use_sigma2.addItem(label)
        chrome.name_inside(
            self.use_sigma2, "sigma2",
            tip="Which sigma^2 from the shell table to apply to the matching "
                "FEFF paths: the Einstein model, the file's displacement "
                "parameters, or none at all.")

        rows.addLayout(chrome.field_grid([
            self.write_input, self.read_output,
            (self.run_feff, 2),
            self.s02, self.use_sigma2,
        ]))
        return group

    def _connect(self) -> None:
        self.absorber.currentIndexChanged.connect(self._on_absorber)
        for widget in (self.r_max, self.tolerance, self.temperature,
                       self.theta_e):
            widget.valueChanged.connect(lambda _=0: self.recompute())
        for widget in (self.k_min, self.k_max, self.r_window_min,
                       self.r_window_max):
            widget.valueChanged.connect(lambda _=0: self._refresh_resolution())
        self.per_shell.valueChanged.connect(
            lambda _=0: self._refresh_resolution())
        self.export_button.clicked.connect(self._export)
        self.write_input.clicked.connect(self._write_input)
        self.read_output.clicked.connect(self._read_output)
        self.run_feff.clicked.connect(self._run_feff)
        self.s02.valueChanged.connect(lambda _=0: self._refresh_chi())
        self.use_sigma2.currentIndexChanged.connect(
            lambda _=0: self._refresh_chi())

    def apply_theme(self, theme) -> None:
        self.theme = theme
        self.chi_plot.apply_theme(theme)
        if self.table is not None:
            self._refresh_resolution()

    # -- context -----------------------------------------------------------
    def set_context(self, structure, site_index=None) -> None:
        """Adopt a structure, keeping the chosen absorber where possible."""
        changed = structure is not self.structure
        self.structure = structure
        if changed:
            self._loading = True
            self.absorber.clear()
            if structure is not None:
                for index, site in enumerate(structure.sites):
                    self.absorber.addItem(
                        f"{site.label}  ({site.element})", index)
            self._loading = False
        if site_index is not None:
            position = self.absorber.findData(int(site_index))
            if position >= 0 and position != self.absorber.currentIndex():
                self._loading = True
                self.absorber.setCurrentIndex(position)
                self._loading = False
        self.recompute()

    def _on_absorber(self) -> None:
        if not self._loading:
            self.recompute()

    def recompute(self) -> None:
        if self._loading or self.structure is None:
            self.table = None
            self.shells.setRowCount(0)
            self.report.setHtml("")
            return
        site = self.absorber.currentData()
        if site is None:
            return
        self.site_index = int(site)
        self.table = exafs_mod.shell_table(
            self.structure, self.site_index,
            r_max=float(self.r_max.value()),
            tolerance=float(self.tolerance.value()),
            temperature=float(self.temperature.value()),
            einstein_temperature=float(self.theta_e.value()))
        self._fill_shells()
        self._refresh_resolution()
        self._refresh_paths()

    def _fill_shells(self) -> None:
        if self.table is None:
            return
        _fill(self.shells, [
            [s.element, s.z, s.count, s.count_occupancy, s.r_mean, s.spread,
             s.sigma2_uncorrelated, s.sigma2_einstein, s.reduced_mass,
             ", ".join(s.labels)]
            for s in self.table.shells])
        self.shell_notes.setText(" · ".join(self.table.notes))

    def _refresh_resolution(self) -> None:
        if self.table is None:
            return
        muted = chrome.muted_hex(self.theme)
        try:
            self.resolution = exafs_mod.resolution(
                self.table, k_min=float(self.k_min.value()),
                k_max=float(self.k_max.value()),
                r_min=float(self.r_window_min.value()),
                r_max=float(self.r_window_max.value()),
                parameters_per_shell=int(self.per_shell.value()))
        except ValueError as exc:
            self.report.setHtml(f"<p style='color:{muted}'>{exc}</p>")
            return
        res = self.resolution
        rows = "".join(
            f"<tr><td>{self.table.shells[a].element} "
            f"{self.table.shells[a].r_mean:.3f} Å</td>"
            f"<td>{self.table.shells[b].element} "
            f"{self.table.shells[b].r_mean:.3f} Å</td>"
            f"<td align='right'>{gap:.3f} Å</td></tr>"
            for a, b, gap in res.unresolved)
        self.report.setHtml(f"""
        <h3 style='margin-bottom:2px'>Over k = {res.k_min:g}–{res.k_max:g}
        Å<sup>-1</sup></h3>
        <table width='100%' cellpadding='3'>
          <tr><td>Resolution &Delta;R = &pi;/(2&Delta;k)</td>
              <td align='right'><b>{res.delta_r:.4f} Å</b></td></tr>
          <tr><td>Independent points over r = {res.r_min:g}–{res.r_max:g} Å,
              N<sub>idp</sub> = 2&Delta;k&Delta;R/&pi;</td>
              <td align='right'><b>{res.n_independent:.1f}</b></td></tr>
          <tr><td>Parameters for the shells in that window</td>
              <td align='right'><b>{res.n_parameters}</b></td></tr>
        </table>
        <h4 style='margin-bottom:2px'>Shell pairs closer together than
        &Delta;R</h4>
        <table width='100%' cellpadding='2' style='font-size:11px'>
          <tr style='color:{muted}'><th align='left'>shell</th>
          <th align='left'>next shell</th><th align='right'>apart</th></tr>
          {rows or "<tr><td colspan='3'>none</td></tr>"}
        </table>
        <p style='color:{muted};font-size:11px'>
        {"<br>".join(res.notes)}</p>
        """)

    # -- FEFF --------------------------------------------------------------
    def _write_input(self) -> None:
        if self.structure is None or self.site_index is None:
            return
        from ..core import exporters

        path, _ = QFileDialog.getSaveFileName(
            self, "Write a FEFF input", "feff.inp", "FEFF input (*.inp)")
        if not path:
            return
        exporters.write_feff(self.structure, self.site_index, path,
                             rmax=float(self.r_max.value()))
        self.statusMessage.emit(
            f"Wrote {path}. FACET does not run FEFF; the Read button takes the "
            "output folder back.")

    def _read_output(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "A folder holding files.dat and feffNNNN.dat")
        if not directory:
            return
        self._adopt_feff(Path(directory))

    def _adopt_feff(self, directory: Path) -> None:
        try:
            self.paths = exafs_mod.read_feff_directory(directory)
        except OSError as exc:
            QMessageBox.warning(self, "Could not read that folder", str(exc))
            return
        self.feff_directory = directory
        if not self.paths:
            QMessageBox.information(
                self, "Nothing to read",
                "No files.dat or feffNNNN.dat in that folder. FEFF writes the "
                "path files only when asked: PRINT 0 0 0 0 0 3, which is what "
                "FACET's exported input sets.")
            return
        self._refresh_paths()
        self.statusMessage.emit(
            f"Read {len(self.paths)} FEFF paths from {directory}")

    def _run_feff(self) -> None:
        """Run a FEFF the user already has, in a folder they choose."""
        import subprocess

        if self.structure is None or self.site_index is None:
            return
        executable = exafs_mod.find_feff_executable()
        if executable is None:
            executable, _ = QFileDialog.getOpenFileName(
                self, "Where is your feff8l or feff6l?", "",
                "Executables (*.exe *.bat *.cmd);;All files (*)")
            if not executable:
                return
            executable = Path(executable)
        directory = QFileDialog.getExistingDirectory(
            self, "An empty folder to run the calculation in")
        if not directory:
            return
        from ..core import exporters

        target = Path(directory)
        exporters.write_feff(self.structure, self.site_index,
                             target / "feff.inp",
                             rmax=float(self.r_max.value()))
        self.statusMessage.emit(f"Running {executable} in {target}…")
        try:
            result = subprocess.run([str(executable)], cwd=str(target),
                                    capture_output=True, text=True,
                                    timeout=1800)
        except (OSError, subprocess.SubprocessError) as exc:
            QMessageBox.warning(self, "FEFF did not run", str(exc))
            return
        if not (target / "files.dat").is_file():
            tail = "\n".join((result.stdout or "").splitlines()[-12:])
            QMessageBox.warning(
                self, "FEFF ran but wrote no path files",
                f"{executable} finished with code {result.returncode}.\n\n"
                f"{tail}")
            return
        self._adopt_feff(target)

    def _sigma2_map(self) -> dict[int, float]:
        """sigma^2 per FEFF path index, from whichever source is chosen."""
        if self.table is None or not self.paths:
            return {}
        mode = self.use_sigma2.currentIndex()
        if mode == 2:
            return {}
        rows = exafs_mod.compare_to_shells(self.paths, self.table)
        out: dict[int, float] = {}
        for row in rows:
            if row["shell"] is None:
                continue
            shell = self.table.shells[row["shell"]]
            value = (shell.sigma2_einstein if mode == 0
                     else shell.sigma2_uncorrelated)
            if value == value:                  # not NaN
                digits = "".join(c for c in row["feff_file"] if c.isdigit())
                out[int(digits or 0)] = float(value)
        return out

    def _refresh_paths(self) -> None:
        if not self.paths or self.table is None:
            self.path_table.setRowCount(0)
            self.chi_plot.set_series([])
            self.chi_notes.setText("")
            return
        rows = exafs_mod.compare_to_shells(self.paths, self.table)
        _fill(self.path_table, [
            [r["feff_file"], r["kind"], r["n_legs"], r["r_effective"],
             r["degeneracy"], r["amplitude_ratio"],
             "" if r["shell"] is None else r["shell_element"],
             r["shell_r"], r["shell_count"] or ""]
            for r in rows])
        self._refresh_chi()

    def _refresh_chi(self) -> None:
        if not self.paths:
            return
        chi = exafs_mod.chi_from_paths(
            self.paths, s02=float(self.s02.value()),
            sigma2=self._sigma2_map())
        if not len(chi.k):
            self.chi_plot.set_series([])
            self.chi_notes.setText(" · ".join(chi.notes))
            return
        series = [Series(chi.k, chi.k ** 2 * chi.chi,
                         label="k²χ(k) — FEFF amplitudes, FACET σ²",
                         color=CHI, width=1.3)]
        if self.feff_directory is not None:
            own = self.feff_directory / "chi.dat"
            if own.is_file():
                k, value = exafs_mod.read_chi(own)
                if len(k):
                    series.append(Series(k, k ** 2 * value,
                                         label="k²χ(k) — FEFF's own chi.dat",
                                         color=FEFF_CHI, width=1.0,
                                         dashed=True))
        self.chi_plot.set_series(series, keep_view=False)
        self.chi_plot.set_labels(
            x="k / Å⁻¹", y="k²χ(k) / Å⁻²",
            title=f"{len(chi.per_path)} FEFF paths, S₀² = {chi.s02:g}",
            footnote="Amplitudes, phases and mean free paths are FEFF's; the "
                     "degeneracies are FEFF's; σ² is from the shell table.")
        self.chi_notes.setText(" · ".join(chi.notes))

    # -- export ------------------------------------------------------------
    def _export(self) -> None:
        if self.table is None or not self.table.shells:
            QMessageBox.information(self, "Nothing to export",
                                    "No shells have been computed yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export the shell table", "shells.csv", "CSV (*.csv)")
        if not path:
            return
        from ..core.version_info import provenance_lines

        lines = ["# " + line for line in provenance_lines()]
        lines += ["# " + note for note in self.table.notes]
        if self.resolution is not None:
            lines += ["# " + note for note in self.resolution.notes]
        lines.append("# columns in Artemis/Larch path-parameter order")
        lines.append("element,Z,N,N_occupancy,R_A,spread_A,"
                     "sigma2_file_A2,sigma2_einstein_A2,mu_amu,sites")
        for s in self.table.shells:
            lines.append(
                f"{s.element},{s.z},{s.count},{s.count_occupancy:.4f},"
                f"{s.r_mean:.5f},{s.spread:.5f},"
                f"{s.sigma2_uncorrelated:.6f},{s.sigma2_einstein:.6f},"
                f"{s.reduced_mass:.4f},{' '.join(s.labels)}")
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.statusMessage.emit(f"Wrote {path}")
