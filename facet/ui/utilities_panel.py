"""The utilities panel.

What VESTA and CrystalMaker put under Utilities or Analysis, for the selected
structure and site: the cell and its reciprocal, density and composition,
reflections, bond angles, radial shells, polyhedral connectivity, and the
bond-valence quantities that span a whole structure.

Everything here reports numbers. Nothing here says what they mean.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core import cn_methods, utilities


def _table(headers: list[str]) -> QTableWidget:
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


def _fill(table: QTableWidget, rows: list[list]) -> None:
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


class UtilitiesPanel(QWidget):
    """Per-structure and per-site tables, recomputed on demand."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.structure = None
        self.results = None
        self.site_result = None
        self.v_bond = 0.075

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        self.summary = QTextBrowser()
        self.summary.setFrameShape(QTextBrowser.NoFrame)
        self.tabs.addTab(self.summary, "Cell")

        self.angles = _table(["ligand A", "ligand B", "angle / °",
                              "d(A) / Å", "d(B) / Å"])
        self.tabs.addTab(self._wrap(self.angles,
                                    "Ligand–centre–ligand angles for the "
                                    "selected site, bonded set only."),
                         "Angles")

        self.shells = _table(["element", "distance / Å", "count"])
        shell_row = QHBoxLayout()
        shell_row.addWidget(QLabel("radius"))
        self.shell_radius = QDoubleSpinBox()
        self.shell_radius.setRange(2.0, 12.0)
        self.shell_radius.setValue(6.0)
        self.shell_radius.setSuffix(" Å")
        self.shell_radius.valueChanged.connect(self._refresh_shells)
        shell_row.addWidget(self.shell_radius)
        shell_row.addStretch(1)
        self.tabs.addTab(self._wrap(self.shells,
                                    "Neighbours grouped into shells by "
                                    "distance, with degeneracies.", shell_row),
                         "Shells")

        self.reflections = _table(["h", "k", "l", "d / Å", "2θ / °"])
        refl_row = QHBoxLayout()
        refl_row.addWidget(QLabel("λ"))
        self.wavelength = QComboBox()
        for name, value in (("Cu Kα1  1.5406", 1.5406), ("Cu Kα  1.5418", 1.5418),
                            ("Mo Kα1  0.7093", 0.70930), ("Co Kα1  1.7890", 1.78897),
                            ("Cr Kα1  2.2897", 2.28970), ("Ag Kα1  0.5594", 0.55941)):
            self.wavelength.addItem(name, value)
        self.wavelength.currentIndexChanged.connect(self._refresh_reflections)
        refl_row.addWidget(self.wavelength)
        refl_row.addWidget(QLabel("2θ max"))
        self.two_theta_max = QDoubleSpinBox()
        self.two_theta_max.setRange(5.0, 160.0)
        self.two_theta_max.setValue(80.0)
        self.two_theta_max.valueChanged.connect(self._refresh_reflections)
        refl_row.addWidget(self.two_theta_max)
        refl_row.addStretch(1)
        self.tabs.addTab(self._wrap(
            self.reflections,
            "Reflection geometry only — d-spacings and Bragg angles. "
            "Intensities need structure factors and are not computed here.",
            refl_row), "Reflections")

        self.methods = _table(["method", "CN", "parameters", "note"])
        method_row = QHBoxLayout()
        self.include_pymatgen = QCheckBox("include pymatgen strategies")
        self.include_pymatgen.setToolTip(
            "Eight further near-neighbour rules from pymatgen. Off by default "
            "because they rebuild the structure and are much slower, and the "
            "built application ships without pymatgen.")
        self.include_pymatgen.toggled.connect(self._refresh_methods)
        method_row.addWidget(self.include_pymatgen)
        method_row.addStretch(1)
        self.tabs.addTab(self._wrap(
            self.methods,
            "The same site counted every defensible way. No rule is marked "
            "correct; they are shown together.", method_row),
            "CN methods")

        self.connectivity = _table(["site A", "site B", "shared ligands",
                                    "sharing"])
        self.tabs.addTab(self._wrap(
            self.connectivity,
            "Polyhedra sharing one ligand, two, or three or more."),
            "Connectivity")

    def _wrap(self, widget, hint: str, extra=None) -> QWidget:
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(8, 8, 8, 8)
        label = QLabel(hint)
        label.setWordWrap(True)
        font = QFont(label.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        label.setFont(font)
        label.setStyleSheet("color:#8a93a3;")
        box.addWidget(label)
        if extra is not None:
            box.addLayout(extra)
        box.addWidget(widget, 1)
        return holder

    # -- content -----------------------------------------------------------
    def update_for(self, structure, results, site_result, v_bond: float) -> None:
        self.structure = structure
        self.results = results
        self.site_result = site_result
        self.v_bond = v_bond
        if structure is None:
            self.summary.setHtml("")
            for t in (self.angles, self.shells, self.reflections,
                      self.connectivity):
                t.setRowCount(0)
            return
        self._refresh_summary()
        self._refresh_angles()
        self._refresh_shells()
        self._refresh_reflections()
        self._refresh_connectivity()
        self._refresh_methods()

    def _refresh_summary(self) -> None:
        s = self.structure
        cell = s.cell
        rec = utilities.reciprocal_cell(s)
        dens = utilities.density(s)
        comp = utilities.composition_summary(s)
        gii = (utilities.global_instability_index(self.results)
               if self.results else {"gii": float("nan"), "n_sites": 0})
        bsi = utilities.bond_strain_index(self.results) if self.results else float("nan")

        counts = ", ".join(f"{e} {n:g}" for e, n in comp["counts"].items())
        atomic = ", ".join(f"{e} {v:.1f} %"
                           for e, v in comp["atomic_percent"].items())
        weight = ", ".join(f"{e} {v:.1f} %"
                           for e, v in comp["weight_percent"].items())
        charge = ("balanced" if comp["balanced"]
                  else (f"{comp['net_charge']:+.3f}"
                        if comp["net_charge"] is not None else "not determined"))

        self.summary.setHtml(f"""
        <h3 style='margin-bottom:2px'>Unit cell</h3>
        <table width='100%' cellpadding='3'>
          <tr><td>a, b, c</td><td align='right'>
              {cell.a:.5f}, {cell.b:.5f}, {cell.c:.5f} Å</td></tr>
          <tr><td>α, β, γ</td><td align='right'>
              {cell.alpha:.4f}, {cell.beta:.4f}, {cell.gamma:.4f}°</td></tr>
          <tr><td>Volume</td><td align='right'>{cell.volume:.4f} Å<sup>3</sup></td></tr>
          <tr><td>Space group</td><td align='right'>
              {s.spacegroup_hm or '?'} (No. {s.spacegroup_number or '?'})</td></tr>
        </table>

        <h4 style='margin-bottom:2px'>Reciprocal cell</h4>
        <table width='100%' cellpadding='3'>
          <tr><td>a*, b*, c*</td><td align='right'>
              {rec.get('a*', float('nan')):.6f}, {rec.get('b*', float('nan')):.6f},
              {rec.get('c*', float('nan')):.6f} Å<sup>-1</sup></td></tr>
          <tr><td>α*, β*, γ*</td><td align='right'>
              {rec.get('alpha*', float('nan')):.4f},
              {rec.get('beta*', float('nan')):.4f},
              {rec.get('gamma*', float('nan')):.4f}°</td></tr>
        </table>
        <p style='color:#8a93a3;font-size:11px'>Crystallographic convention,
        a* = (b × c)/V, without the 2π factor — the one d-spacings use.</p>

        <h4 style='margin-bottom:2px'>Composition</h4>
        <table width='100%' cellpadding='3'>
          <tr><td>Cell contents</td><td align='right'>{counts}</td></tr>
          <tr><td>Formula weight</td><td align='right'>
              {comp['formula_weight']:.3f} g/mol</td></tr>
          <tr><td>Density</td><td align='right'>
              {dens['density']:.4f} g/cm<sup>3</sup></td></tr>
          <tr><td>Atomic %</td><td align='right'>{atomic}</td></tr>
          <tr><td>Weight %</td><td align='right'>{weight}</td></tr>
          <tr><td>Charge balance</td><td align='right'>{charge}</td></tr>
        </table>

        <h4 style='margin-bottom:2px'>Bond valence, whole structure</h4>
        <table width='100%' cellpadding='3'>
          <tr><td>Global instability index</td><td align='right'>
              {gii['gii']:.4f} v.u. over {gii['n_sites']} sites</td></tr>
          <tr><td>Largest discrepancy</td><td align='right'>
              {gii.get('worst_site') or '—'}
              {('%+.3f v.u.' % gii['worst_discrepancy'])
               if gii.get('worst_discrepancy') is not None else ''}</td></tr>
          <tr><td>Bond strain index</td><td align='right'>{bsi:.4f} v.u.</td></tr>
        </table>
        """)

    def _refresh_angles(self) -> None:
        if self.site_result is None:
            self.angles.setRowCount(0)
            return
        rows = utilities.bond_angles(self.site_result)
        _fill(self.angles, [[r.a, r.b, r.angle, r.d_a, r.d_b] for r in rows])

    def _refresh_shells(self) -> None:
        if self.structure is None or self.site_result is None:
            self.shells.setRowCount(0)
            return
        try:
            rows = utilities.radial_shells(self.structure,
                                           self.site_result.site_index,
                                           rmax=self.shell_radius.value())
        except Exception:
            rows = []
        _fill(self.shells, [[r["element"], r["distance"], r["count"]]
                            for r in rows])

    def _refresh_reflections(self) -> None:
        if self.structure is None:
            self.reflections.setRowCount(0)
            return
        rows = utilities.reflection_list(
            self.structure, wavelength=self.wavelength.currentData(),
            two_theta_max=self.two_theta_max.value(), max_index=6)
        _fill(self.reflections,
              [[r["h"], r["k"], r["l"], r["d"], r["two_theta"]]
               for r in rows[:600]])

    def _refresh_methods(self) -> None:
        if self.site_result is None:
            self.methods.setRowCount(0)
            return
        try:
            found = cn_methods.all_methods(
                self.site_result, self.structure,
                include_pymatgen=self.include_pymatgen.isChecked())
        except Exception:
            self.methods.setRowCount(0)
            return
        rows = []
        for m in found:
            params = ", ".join(f"{k} {v:.3g}" if isinstance(v, float)
                               else f"{k} {v}"
                               for k, v in m.parameters.items())
            rows.append([m.method, m.display, params, m.note])
        summary = cn_methods.spread(found)
        if summary.get("n"):
            rows.append(["— spread —",
                         f"{summary['min']:.2f}–{summary['max']:.2f}",
                         f"{summary['n']} methods",
                         "distinct integers: "
                         + ", ".join(str(v) for v in summary["integer_values"])])
        _fill(self.methods, rows)

    def _refresh_connectivity(self) -> None:
        if self.structure is None or not self.results:
            self.connectivity.setRowCount(0)
            return
        rows = utilities.polyhedral_connectivity(self.structure, self.results,
                                                 self.v_bond)
        _fill(self.connectivity,
              [[r["site_a"], r["site_b"], r["shared_ligands"], r["sharing"]]
               for r in rows])
