"""The utilities panel.

What VESTA and CrystalMaker put under Utilities or Analysis, for the selected
structure and site: the cell and its reciprocal, density and composition,
reflections, bond angles, radial shells, polyhedral connectivity, and the
bond-valence quantities that span a whole structure.

Everything here reports numbers. Nothing here says what they mean.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
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

from ..core import adp, bv_report, cn_methods, network, quality, utilities
from . import chrome


def _escape(text: str) -> str:
    """A file's own strings go into HTML. A reference field with an ampersand
    or an angle bracket in it would otherwise eat the rest of the table."""
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


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

    # the key of the scale of basicity moderating parameters chosen on the
    # Basicity tab, for the Site tab to follow
    basicityScaleChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.structure = None
        self.results = None
        self.site_result = None
        self.v_bond = 0.075
        self.theme = None
        # the analysis covering the anions, as a callable or a list, from
        # the window's project cache; None makes the panel search itself
        self._full_results = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        # Which tabs hold figures from a structure that has since changed. A
        # tab is computed when it is shown and not before; see update_for.
        self._stale: set[str] = set()
        self.tabs.currentChanged.connect(self._on_tab_shown)

        self.summary = QTextBrowser()
        self.summary.setFrameShape(QTextBrowser.NoFrame)
        self.tabs.addTab(self.summary, "Cell")

        # What the file itself looks like: where it came from, and what the
        # health checks measured. Second, not last, because it is the thing to
        # read before trusting any of the numbers on the other tabs.
        self.file_report = QTextBrowser()
        self.file_report.setFrameShape(QTextBrowser.NoFrame)
        self.tabs.addTab(self.file_report, "File")

        self.apriori = _table(["site", "CN", "\u0394 topol", "\u0394 cryst",
                               "a priori / v.u.", "observed / v.u."])
        self.tabs.addTab(self._wrap(
            self.apriori,
            "Bond valences the bond topology alone would give, from the "
            "valence-sum and loop rules. \u0394 topol is the spread of those "
            "about their mean -- variation the connectivity requires. "
            "\u0394 cryst is how far the observed valences sit from them -- "
            "variation it does not account for. Both in valence units, as mean "
            "absolute deviations. Gagn\u00e9 & Hawthorne, IUCrJ 7 (2020) 581."),
            "A priori")

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

        self.cutoffs = _table(["pair", "R0 / Å", "b / Å", "bond ≤ / Å",
                               "listed ≤ / Å", "window / Å", "shortest / Å",
                               "bonds", "listed", "R0 from"])
        self.tabs.addTab(self._wrap(
            self.cutoffs,
            "The distance cutoff each pair gets from the valence thresholds. "
            "A cutoff is not chosen here: d = R0 − b ln(v), so it follows from "
            "the threshold and the pair's own R0. Two pairs with different R0 "
            "get different distances, which is the point."),
            "Cutoffs")

        self.stability = _table(["site", "CN", "stable from", "stable to",
                                 "width / v.u.", "CN over the whole scan"])
        self.tabs.addTab(self._wrap(
            self.stability,
            "How far the threshold can move before a coordination number "
            "changes. Reported as a width; no number is called reliable."),
            "Stability")

        self.valences = _table(["site", "element", "expected", "BVS",
                                "difference", "contacts"])
        self.balance = QLabel("")
        self.balance.setWordWrap(True)
        balance_font = QFont(self.balance.font())
        balance_font.setPointSizeF(max(7.0, balance_font.pointSizeF() - 1.0))
        self.balance.setFont(balance_font)
        chrome.mark_hint(self.balance)
        anion_holder = QWidget()
        anion_box = QVBoxLayout(anion_holder)
        anion_box.setContentsMargins(0, 0, 0, 0)
        anion_box.addWidget(self.valences, 1)
        anion_box.addWidget(self.balance)
        self.tabs.addTab(self._wrap(
            anion_holder,
            "Bond-valence sums for the anions, from a search around each anion "
            "rather than from the cation lists. The two totals below are the "
            "same bonds counted from opposite ends."),
            "Anions")

        # Optical basicity: Duffy and Ingram's Lambda from the composition,
        # the microscopic lambda of every anion site from the bond valences
        # it receives, and the anions about every cation site, on a chosen
        # scale of basicity moderating parameters.
        from ..core import basicity as basicity_mod

        self.basicity_text = QTextBrowser()
        self.basicity_text.setFrameShape(QTextBrowser.NoFrame)
        self.basicity_anions = _table(["anion site", "element", "atoms", "CN",
                                       "Σ s / v.u.", "λ (bond)",
                                       "λ (listed)", "shares of |z|"])
        self.basicity_cations = _table(["cation site", "element", "CN",
                                        "λ̄ about it",
                                        "λ̄ host only", "anions"])
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("scale"))
        self.basicity_scale_box = QComboBox()
        for key, scale in basicity_mod.SCALES.items():
            self.basicity_scale_box.addItem(scale.name, key)
        self.basicity_scale_box.setToolTip(
            "The set of basicity moderating parameters gamma. The first "
            "derives every cation's from Pauling's 1960 electronegativity "
            "through Duffy and Ingram's relation; the others are measured "
            "tables, with that relation behind them for a cation they do "
            "not list. Every value says which it is.")
        self.basicity_scale_box.currentIndexChanged.connect(
            self._on_basicity_scale)
        scale_row.addWidget(self.basicity_scale_box)
        scale_row.addStretch(1)
        basicity_holder = QWidget()
        basicity_box = QVBoxLayout(basicity_holder)
        basicity_box.setContentsMargins(0, 0, 0, 0)
        basicity_box.addWidget(self.basicity_text, 3)
        basicity_box.addWidget(self.basicity_anions, 3)
        basicity_box.addWidget(self.basicity_cations, 2)
        self.tabs.addTab(self._wrap(
            basicity_holder,
            "Optical basicity (Duffy and Ingram): Λ from the composition, "
            "and the microscopic λ of each anion site from the bond "
            "valences it receives, λ = 1 − Σ (s/|z|)(1 − "
            "1/γ) over its bonded contacts. The two agree when every "
            "cation's bonds carry its formal charge; their difference is "
            "the departure of the sums from the charges.", scale_row),
            "Basicity")

    def _refresh_cutoffs(self) -> None:
        """The cutoff table: this program's central claim, as numbers."""
        from ..core import bv

        params = getattr(self, "params", None) or bv.DEFAULT
        try:
            table = bv_report.cutoff_table(
                self.structure, params, v_bond=self.v_bond,
                v_list=getattr(self, "v_list", bv.V_LIST_DEFAULT),
                results=self.results)
        except ValueError:
            self.cutoffs.setRowCount(0)
            return
        rows = []
        for row in table.rows:
            rows.append([row.label, row.r0, row.b, row.d_bond, row.d_list,
                         row.window,
                         row.shortest if row.shortest is not None else None,
                         row.n_bonds, row.n_listed,
                         "fitted" if row.fitted else "estimated"])
        _fill(self.cutoffs, rows)

    def _refresh_stability(self) -> None:
        if not self.results:
            self.stability.setRowCount(0)
            return
        scan = bv_report.threshold_scan(self.results)
        rows = []
        for result in self.results:
            counts = scan.per_site.get(result.label)
            if counts is None:
                continue
            span = scan.stable_range(result.label, self.v_bond)
            low, high = span if span else (float("nan"), float("nan"))
            rows.append([result.label, result.cn_valence, low, high,
                         high - low,
                         f"{int(counts.min())}–{int(counts.max())}"])
        _fill(self.stability, rows)

    def _refresh_anions(self) -> None:
        from ..core import bv

        params = getattr(self, "params", None) or bv.DEFAULT
        try:
            rows = bv_report.anion_sums(
                self.structure, params, v_bond=self.v_bond,
                v_list=getattr(self, "v_list", bv.V_LIST_DEFAULT))
        except Exception:
            self.valences.setRowCount(0)
            self.balance.setText("")
            return
        _fill(self.valences, [
            [row.label, row.element,
             row.expected if row.expected is not None else None,
             row.bvs,
             row.discrepancy if row.discrepancy is not None else None,
             row.n_contacts]
            for row in rows])

        if not self.results:
            self.balance.setText("")
            return
        balance = bv_report.charge_balance(
            self.structure, self.results, params, v_bond=self.v_bond,
            v_list=getattr(self, "v_list", bv.V_LIST_DEFAULT))
        self.balance.setText(
            f"cations distribute {balance['cation_valence']:.4f} v.u. per cell; "
            f"anions receive {balance['anion_valence']:.4f}; "
            f"difference {balance['difference']:+.4f} "
            f"({balance['relative_difference'] * 100:+.4f}%). "
            + balance["note"])

    def _wrap(self, widget, hint: str, extra=None) -> QWidget:
        holder = QWidget()
        box = QVBoxLayout(holder)
        box.setContentsMargins(8, 8, 8, 8)
        label = QLabel(hint)
        label.setWordWrap(True)
        font = QFont(label.font())
        font.setPointSizeF(max(7.0, font.pointSizeF() - 1.0))
        label.setFont(font)
        chrome.mark_hint(label)
        box.addWidget(label)
        if extra is not None:
            box.addLayout(extra)
        box.addWidget(widget, 1)
        return holder

    # -- content -----------------------------------------------------------
    def apply_theme(self, theme) -> None:
        """Adopt a theme, so the hand-built HTML follows it too.

        The tables and labels are coloured by the application style sheet; the
        summary is HTML this panel writes itself, so it has to be rewritten.
        """
        self.theme = theme
        if self.structure is not None:
            self._refresh_summary()
            if "Basicity" not in self._stale:
                self._refresh_basicity()

    def update_for(self, structure, results, site_result, v_bond: float,
                   params=None, v_list: float | None = None,
                   full_results=None) -> None:
        from ..core import bv

        self.structure = structure
        self.results = results
        self.site_result = site_result
        self.v_bond = v_bond
        self.params = params or bv.DEFAULT
        self.v_list = (v_list if v_list is not None else bv.V_LIST_DEFAULT)
        self._full_results = full_results
        if structure is None:
            self.summary.setHtml("")
            self.file_report.setHtml("")
            self.basicity_text.setHtml("")
            self.apriori.setRowCount(0)
            for t in (self.angles, self.shells, self.reflections,
                      self.connectivity, self.cutoffs, self.stability,
                      self.valences, self.basicity_anions,
                      self.basicity_cations):
                t.setRowCount(0)
            self.balance.setText("")
            return
        # Every tab is now out of date; the one on screen is brought up to
        # date at once and the rest when they are shown. Recomputing all ten
        # took 3.8 s on a 484-atom structure and this is called on every move
        # of the threshold, so the window froze for seconds at a time over
        # figures that were behind another tab.
        self._stale = set(self.REFRESHERS)
        self._refresh_current()

    # What computes each tab, by the name on it. A tab with no entry here --
    # there are none today -- would simply never be marked stale.
    REFRESHERS = {
        "Cell": "_refresh_summary",
        "File": "_refresh_file",
        "A priori": "_refresh_apriori",
        "Angles": "_refresh_angles",
        "Shells": "_refresh_shells",
        "Reflections": "_refresh_reflections",
        "CN methods": "_refresh_methods",
        "Connectivity": "_refresh_connectivity",
        "Cutoffs": "_refresh_cutoffs",
        "Stability": "_refresh_stability",
        "Anions": "_refresh_anions",
        "Basicity": "_refresh_basicity",
    }

    # -- optical basicity ------------------------------------------------------
    def basicity_scale(self) -> str:
        """The key of the scale chosen on the Basicity tab."""
        key = self.basicity_scale_box.currentData()
        from ..core import basicity as basicity_mod

        return str(key) if key else basicity_mod.DEFAULT_SCALE

    def set_basicity_scale(self, key: str) -> None:
        at = self.basicity_scale_box.findData(str(key))
        if at >= 0:
            self.basicity_scale_box.setCurrentIndex(at)

    def _on_basicity_scale(self, _index: int) -> None:
        self._stale.add("Basicity")
        self._refresh_current()
        self.basicityScaleChanged.emit(self.basicity_scale())

    def _refresh_basicity(self) -> None:
        from ..core import basicity as basicity_mod

        if self.structure is None:
            self.basicity_text.setHtml("")
            self.basicity_anions.setRowCount(0)
            self.basicity_cations.setRowCount(0)
            return
        muted = chrome.muted_hex(self.theme)
        try:
            sb = basicity_mod.structure_basicity(
                self.structure, self._with_anions(), self.basicity_scale(),
                v_bond=self.v_bond)
        except Exception as error:               # a parameter or reader fault
            self.basicity_text.setHtml(
                f"<p style='color:{muted}'>Optical basicity not computed: "
                f"{_escape(error)}</p>")
            self.basicity_anions.setRowCount(0)
            self.basicity_cations.setRowCount(0)
            return
        scale = basicity_mod.scale_named(sb.scale)
        s = self.structure

        def num(value, decimals=3):
            return "&mdash;" if value != value else f"{value:.{decimals}f}"

        element_rows = "".join(
            f"<tr><td>&lambda; of {element}, {n:g} atom-charges</td>"
            f"<td align='right'>{num(mean)} <span style='color:{muted}'>"
            f"({num(listed)} to the listing cut)</span></td></tr>"
            for element, (mean, listed, n) in sb.by_element.items())
        species_rows = "".join(
            f"<tr><td>{r.moderator.species if r.moderator else r.element}</td>"
            f"<td align='right'>{r.count:g}</td>"
            f"<td align='right'>{num(r.fraction, 4)}</td>"
            f"<td align='right'>{num(r.moderator.gamma) if r.moderator else '&mdash;'}</td>"
            f"<td align='right'>{num(r.moderator.basicity) if r.moderator else '&mdash;'}</td>"
            f"<td style='color:{muted}'>"
            f"{_escape(r.moderator.origin) if r.moderator else 'none on this scale'}"
            f"</td></tr>"
            for r in sb.bulk.rows)
        anion_rows = "".join(
            f"<tr><td>{element}</td><td align='right'>{n:g}</td>"
            f"<td align='right'>{charge:g}</td>"
            f"<td align='right'>&gamma; &times; {factor:g}</td></tr>"
            for element, n, charge, factor in sb.bulk.anions)
        notes = "".join(
            f"<p style='color:{muted};font-size:11px;margin:3px 0'>{_escape(n)}</p>"
            for n in list(sb.bulk.notes) + list(sb.notes))
        self.basicity_text.setHtml(f"""
        <h3 style='margin-bottom:2px'>Optical basicity</h3>
        <table width='100%' cellpadding='3'>
          <tr><td>&Lambda; from the composition (Duffy &amp; Ingram, eq. 4)</td>
              <td align='right'><b>{num(sb.bulk.value)}</b></td></tr>
          <tr><td>Mean &lambda; of the anions, from the bond valences</td>
              <td align='right'><b>{num(sb.mean)}</b> <span style='color:{muted}'>
              ({num(sb.mean_listed)} to the listing cut)</span></td></tr>
          <tr><td>Difference, bonds &minus; composition</td>
              <td align='right'>{'&mdash;' if sb.difference != sb.difference else f'{sb.difference:+.3f}'}</td></tr>
          {element_rows}
        </table>
        <h4 style='margin-bottom:2px'>Moderating parameters, {_escape(scale.name)}</h4>
        <table width='100%' cellspacing='0' cellpadding='2' style='font-size:11px'>
          <tr style='color:{muted}'><th align='left'>species</th>
              <th align='right'>atoms</th><th align='right'>X<sub>A</sub></th>
              <th align='right'>&gamma;</th><th align='right'>&Lambda; oxide</th>
              <th align='left'>&gamma; from</th></tr>
          {species_rows}
        </table>
        <table width='100%' cellspacing='0' cellpadding='2' style='font-size:11px'>
          <tr style='color:{muted}'><th align='left'>anion</th>
              <th align='right'>atoms</th><th align='right'>charge</th>
              <th align='right'>on the scale as</th></tr>
          {anion_rows}
        </table>
        <p style='color:{muted};font-size:11px'>{_escape(scale.source)}.
        &gamma; = 1.36 (&chi; &minus; 0.26) for a cation the scale does not
        list. Bond threshold {self.v_bond:.4f} v.u.</p>
        {notes}
        """)

        def atoms_of(index):
            site = s.sites[index]
            return float(site.occupancy) * (site.multiplicity or 1)

        by_index = {r.site_index: r for r in self._with_anions()}
        _fill(self.basicity_anions, [
            [m.label, m.element, atoms_of(index),
             by_index[index].cn_valence if index in by_index else None,
             m.sum_bonded, m.value, m.value_listed,
             "; ".join(f"{x.label} {x.share:.3f}"
                       + ("" if x.moderator is None or x.moderator.measured
                          else " (chi)")
                       for x in m.shares if x.bonded)
             or ("; ".join(m.notes) if m.notes else "")]
            for index, m in sb.anions.items()])
        _fill(self.basicity_cations, [
            [e.label, e.element,
             by_index[index].cn_valence if index in by_index else None,
             e.value, e.value_host,
             "; ".join(f"{label} {value:.3f}"
                       for label, _, value, _ in e.anions)]
            for index, e in sb.environments.items()])

    def _on_tab_shown(self, index: int) -> None:
        self._refresh_current()

    def _refresh_current(self) -> None:
        """Bring the tab on screen up to date, if it is not already."""
        index = self.tabs.currentIndex()
        if index < 0 or self.structure is None:
            return
        self.refresh_tab(self.tabs.tabText(index))

    def refresh_tab(self, title: str) -> None:
        """Compute one tab now, whether or not it is the one on screen.

        Public because a caller that wants a tab's contents without showing it
        -- an export, a test -- needs a way to ask for them that does not
        depend on which tab happens to be in front.
        """
        if title not in self._stale or self.structure is None:
            return
        self._stale.discard(title)
        getattr(self, self.REFRESHERS[title])()

    def refresh_every_tab(self) -> None:
        """All of them, for a caller that needs the whole panel populated."""
        for title in list(self._stale):
            self.refresh_tab(title)

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
        <p style='color:{chrome.muted_hex(self.theme)};font-size:11px'>Crystallographic convention,
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

    def _refresh_file(self) -> None:
        """Provenance, then every health check, then the displacement table.

        The findings are stated as what was measured and what it was compared
        against. None of them says whether the structure is usable: that
        depends on the question being asked of it, and a file that is useless
        for a bond-valence sum can be perfectly good for indexing a powder
        pattern.
        """
        s = self.structure
        report = quality.check(s, self.results, self.v_bond)
        s.issues = list(report.findings)

        ink = chrome.text_hex(self.theme)
        muted = chrome.muted_hex(self.theme)
        rows = [f"<div style='color:{ink}; font-size:9pt'>"]

        rows.append("<h3 style='margin-bottom:2px'>Where this came from</h3>")
        rows.append("<table cellspacing='0' cellpadding='2'>")
        for name, value in (("file", s.source_path or "—"),
                            ("name", s.name or "—"),
                            ("formula", s.formula or "—"),
                            ("space group", s.spacegroup_hm or "—"),
                            ("database code", s.database_code or "—"),
                            ("reference", s.reference or "—"),
                            ("year", str(s.year) if s.year else "—")):
            rows.append(f"<tr><td style='color:{muted}'>{name}</td>"
                        f"<td>{_escape(str(value))}</td></tr>")
        rows.append("</table>")

        rows.append("<h3 style='margin-bottom:2px'>Checks</h3>")
        if not report.findings:
            rows.append(f"<p style='color:{muted}'>Every check ran and none of "
                        f"them measured anything outside its ordinary "
                        f"range.</p>")
        else:
            rows.append(f"<p style='color:{muted}'>What each check measured, "
                        f"and the value it was compared against. None of these "
                        f"says whether the structure is usable — that depends "
                        f"on what is being asked of it.</p>")
            colours = {quality.Level.IMPOSSIBLE: self._level_colour(2),
                       quality.Level.CHECK: self._level_colour(1),
                       quality.Level.NOTE: muted}
            headings = {quality.Level.IMPOSSIBLE:
                        "No structure has been reported with this",
                        quality.Level.CHECK: "Outside what is usually seen",
                        quality.Level.NOTE: "Worth knowing"}
            for level in (quality.Level.IMPOSSIBLE, quality.Level.CHECK,
                          quality.Level.NOTE):
                here = [f for f in report.findings if f.level == level]
                if not here:
                    continue
                rows.append(f"<h4 style='margin-bottom:2px; "
                            f"color:{colours[level]}'>{headings[level]}"
                            f"</h4><table cellspacing='0' cellpadding='2'>")
                for f in here:
                    where = f"{_escape(f.where)}" if f.where else ""
                    against = ("" if f.reference is None
                               else f"<span style='color:{muted}'> — compared "
                                    f"against {f.reference:g}</span>")
                    rows.append(f"<tr><td style='color:{muted}'>{where}</td>"
                                f"<td>{_escape(f.message)}{against}</td></tr>")
                rows.append("</table>")

        rows.append(self._displacement_html(ink, muted))
        rows.append("</div>")
        self.file_report.setHtml("".join(rows))

    def _refresh_apriori(self) -> None:
        """A priori bond valences and the two indices, site by site.

        A structure whose bond topology does not close has no answer here: the
        valence-sum rule cannot hold at every site of a cell that is not charge
        balanced, and a number produced anyway would look like the others. The
        reason is shown in the table instead of a row of figures.
        """
        if self.structure is None:
            self.apriori.setRowCount(0)
            return
        try:
            rows = network.analyse(self.structure, self._with_anions(),
                                   self.v_bond)
            reason = ""
        except ValueError as error:
            rows, reason = [], str(error)
        except Exception as error:
            rows, reason = [], f"{type(error).__name__}: {error}"

        if reason:
            self.apriori.setRowCount(1)
            self.apriori.setSpan(0, 0, 1, self.apriori.columnCount())
            item = QTableWidgetItem(f"No a priori bond valences: {reason}")
            item.setFlags(Qt.ItemIsEnabled)
            self.apriori.setItem(0, 0, item)
            return
        self.apriori.clearSpans()

        def numbers(values):
            return " \u00b7 ".join(f"{v:.3f}" for v in values)

        _fill(self.apriori, [
            [r.label, r.coordination, f"{r.delta_topol:.3f}",
             f"{r.delta_cryst:.3f}", numbers(r.a_priori), numbers(r.observed)]
            for r in rows])

    def _with_anions(self):
        """An analysis covering the anions, which the topology needs.

        The panel's own `results` may be the cations only; the bond counts have
        to be taken from the anion end as well or there is nothing to check
        them against.
        """
        from ..core import coordination

        if self.results and any(
                self.structure.sites[r.site_index].is_anion
                for r in self.results):
            return self.results
        if self._full_results is not None:
            # the window's own cache: one search, shared with the Site tab
            found = (self._full_results() if callable(self._full_results)
                     else self._full_results)
            if found:
                return found
        return coordination.analyse_structure(
            self.structure, self.params, v_bond=self.v_bond,
            v_list=self.v_list, cations_only=False)

    def _displacement_html(self, ink: str, muted: str) -> str:
        """Per-site displacement parameters, where the file gave a tensor.

        U_eq is comparable with the U_iso a file quotes. The three r.m.s.
        values are the displacements along the principal axes of the ellipsoid,
        which is the part that no single number carries: a site can have an
        ordinary U_eq and still be four times longer than it is wide.
        """
        s = self.structure
        shapes = [(site, adp.for_site(s.cell, site)) for site in s.sites]
        shapes = [(site, e) for site, e in shapes if e is not None]
        if not shapes:
            isotropic = sum(1 for site in s.sites if site.u_iso)
            if isotropic:
                return (f"<h3 style='margin-bottom:2px'>Displacement</h3>"
                        f"<p style='color:{muted}'>The file gives isotropic "
                        f"displacement parameters only, for {isotropic} of "
                        f"{len(s.sites)} sites.</p>")
            return (f"<h3 style='margin-bottom:2px'>Displacement</h3>"
                    f"<p style='color:{muted}'>The file gives no displacement "
                    f"parameters.</p>")

        out = ["<h3 style='margin-bottom:2px'>Displacement</h3>",
               f"<p style='color:{muted}'>Anisotropic parameters, converted to "
               f"Cartesian axes and diagonalised. U<sub>eq</sub> is one third "
               f"of the trace, which is the quantity comparable with a quoted "
               f"U<sub>iso</sub>.</p>",
               "<table cellspacing='0' cellpadding='2'>",
               f"<tr style='color:{muted}'><td>site</td>"
               f"<td>U<sub>eq</sub> / Å²</td>"
               f"<td>r.m.s. along the principal axes / Å</td>"
               f"<td>longest / shortest</td></tr>"]
        for site, e in shapes:
            if e.is_ellipsoid:
                lengths = " · ".join(f"{v:.3f}" for v in e.rms)
                ratio = f"{e.anisotropy:.2f}"
            else:
                lengths = " · ".join(
                    f"{v:+.4f}" for v in e.eigenvalues) + " (eigenvalues, Å²)"
                ratio = "—"
            out.append(f"<tr><td>{_escape(site.label)}</td>"
                       f"<td>{e.u_equivalent:.4f}</td>"
                       f"<td>{lengths}</td><td>{ratio}</td></tr>")
        out.append("</table>")
        return "".join(out)

    def _level_colour(self, level: int) -> str:
        """The levels take the theme's own colours rather than a red chosen
        here, so that they stay legible on every theme including the light
        default. Only the most severe is given the emphasis colour; using it
        for both would make the distinction between them invisible."""
        from . import chrome

        if level >= 2 and self.theme is not None:
            return chrome._hex(self.theme.selection_color)
        return chrome.text_hex(self.theme)

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
