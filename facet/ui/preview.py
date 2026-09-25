"""The Structure workspace.

Three zones, as planned: a site list on the left, the 3D view in the middle with
its analysis panel to the right, and the cutoff explorer along the bottom.

The bottom strip is the spine. Everything above it follows the threshold it
shows, and it follows nothing -- drag it and the coordination number, the
bond-valence sum, phi, the contact table and the drawn bonds all move together,
with no neighbour search and no re-analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QFont, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QDoubleSpinBox,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core import (bv, cif, coordination, exporters, project as project_mod,
                    theme as theme_mod)
from ..gl.labels import AtomLabel, BondLabel, LabelScope, LabelSettings
from ..gl.scene import Style, build_scene, merge_scenes
from ..gl.view import StructureView
from ..version import NAME, __version__
from .cutoff_explorer import CutoffExplorer
from .structure_list import StructureList
from .theme_panel import ThemePanel
from .utilities_panel import UtilitiesPanel


class PolyhedraMode:
    NONE = "none"
    SELECTED = "selected site"
    ALL = "all cations"


class PreviewWindow(QMainWindow):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.setWindowTitle(f"{NAME} {__version__}")
        self.resize(1420, 900)

        self.project = project_mod.Project()
        self.scene = None
        self.theme = theme_mod.Theme()
        self.poly_mode = PolyhedraMode.SELECTED
        self.labels = LabelSettings()

        self.view = StructureView()
        self.view.set_theme(self.theme)
        self.view.sitePicked.connect(self._on_site_picked)
        self.view.measured.connect(lambda t: self.statusBar().showMessage(t, 9000))
        self.view.ready.connect(self._on_ready)
        self.view.labelsChanged.connect(self._on_labels_cycled)

        self.explorer = CutoffExplorer()
        self.explorer.set_theme(self.theme)
        self.explorer.thresholdChanged.connect(self._on_threshold)

        self.analysis = QTextBrowser()
        self.analysis.setOpenExternalLinks(False)
        self.analysis.setFrameShape(QTextBrowser.NoFrame)

        self.theme_panel = ThemePanel(self.theme)
        self.theme_panel.themeChanged.connect(self._on_theme_cosmetic)
        self.theme_panel.rebuildNeeded.connect(self._on_theme_structural)

        self.site_list = QListWidget()
        self.site_list.currentRowChanged.connect(self._on_site_row)

        self.structure_panel = StructureList()
        self.structure_panel.set_project(self.project)
        self.structure_panel.activeChanged.connect(self._on_structure_row)
        self.structure_panel.visibilityChanged.connect(
            lambda: self._rebuild(reframe=True))
        self.structure_panel.overlayChanged.connect(self._on_overlay)
        self.structure_panel.removeRequested.connect(self._on_remove)

        self.utilities = UtilitiesPanel()

        self._build_layout()
        self._build_menu()
        self.setStatusBar(QStatusBar())
        self._set_enabled(False)
        self.setAcceptDrops(True)

        if path:
            self.load(path)

    # -- the active structure ----------------------------------------------
    @property
    def structure(self):
        entry = self.project.current
        return entry.structure if entry else None

    @property
    def results(self):
        entry = self.project.current
        return self.project.results_for(entry) if entry else []

    @property
    def site_index(self):
        entry = self.project.current
        return entry.selected_site if entry else None

    @site_index.setter
    def site_index(self, value):
        entry = self.project.current
        if entry is not None:
            entry.selected_site = value

    @property
    def v_bond(self) -> float:
        return self.project.v_bond

    @v_bond.setter
    def v_bond(self, value: float) -> None:
        self.project.set_bond_threshold(value)

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        structures = QDockWidget("Structures", self)
        structures.setWidget(self.structure_panel)
        structures.setFeatures(QDockWidget.DockWidgetMovable
                               | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, structures)
        structures.setMinimumWidth(210)

        sites = QDockWidget("Sites", self)
        sites.setWidget(self.site_list)
        sites.setFeatures(QDockWidget.DockWidgetMovable
                          | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, sites)
        sites.setMinimumWidth(210)

        tabs = QTabWidget()
        tabs.addTab(self.analysis, "Site")
        tabs.addTab(self.utilities, "Utilities")
        tabs.addTab(self.theme_panel, "Appearance")
        tabs.setMinimumWidth(400)

        centre = QWidget()
        col = QVBoxLayout(centre)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._toolbar_row())
        col.addWidget(self.view, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(centre)
        split.addWidget(tabs)
        split.setStretchFactor(0, 1)
        split.setSizes([1000, 400])

        vertical = QSplitter(Qt.Vertical)
        vertical.addWidget(split)
        vertical.addWidget(self.explorer)
        vertical.setStretchFactor(0, 1)
        vertical.setSizes([640, 230])
        self.setCentralWidget(vertical)

    def _toolbar_row(self) -> QWidget:
        bar = QWidget()
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(10)

        self.style_box = QComboBox()
        for s in (Style.BALL_AND_STICK, Style.SPACE_FILLING,
                  Style.STICK, Style.WIREFRAME):
            self.style_box.addItem(s.value, s)
        self.style_box.activated.connect(lambda _=0: self._rebuild())
        row.addWidget(QLabel("Style"))
        row.addWidget(self.style_box)

        self.poly_box = QComboBox()
        for m in (PolyhedraMode.SELECTED, PolyhedraMode.ALL, PolyhedraMode.NONE):
            self.poly_box.addItem(m, m)
        self.poly_box.activated.connect(self._on_poly_mode)
        row.addWidget(QLabel("Polyhedra"))
        row.addWidget(self.poly_box)

        row.addWidget(QLabel("Cell range"))
        self.range_boxes = []
        for _ in range(3):
            box = QSpinBox()
            box.setRange(1, 6)
            box.setFixedWidth(46)
            box.valueChanged.connect(lambda _=0: self._rebuild())
            self.range_boxes.append(box)
            row.addWidget(box)

        self.cell_check = QCheckBox("Unit cell")
        self.cell_check.setChecked(True)
        self.cell_check.toggled.connect(lambda _=False: self._rebuild())
        row.addWidget(self.cell_check)

        row.addWidget(QLabel("Label atoms"))
        self.atom_label_box = QComboBox()
        for kind in AtomLabel:
            self.atom_label_box.addItem(kind.value, kind)
        self.atom_label_box.activated.connect(self._on_labels)
        row.addWidget(self.atom_label_box)

        row.addWidget(QLabel("bonds"))
        self.bond_label_box = QComboBox()
        for kind in BondLabel:
            self.bond_label_box.addItem(kind.value, kind)
        self.bond_label_box.activated.connect(self._on_labels)
        row.addWidget(self.bond_label_box)

        self.ortho_check = QCheckBox("Orthographic")
        self.ortho_check.toggled.connect(self.view.set_projection)
        row.addWidget(self.ortho_check)

        row.addStretch(1)
        # The search radius, expressed as the valence below which a contact is
        # not tabulated at all. Unlike the bond threshold this re-runs the
        # neighbour search, so it is a spin box rather than something to drag.
        row.addWidget(QLabel("Tabulate above"))
        self.list_threshold = QDoubleSpinBox()
        self.list_threshold.setDecimals(4)
        self.list_threshold.setRange(0.0005, 0.2)
        self.list_threshold.setSingleStep(0.005)
        self.list_threshold.setValue(bv.V_LIST_DEFAULT)
        self.list_threshold.setSuffix(" v.u.")
        self.list_threshold.setToolTip(
            "Contacts weaker than this are not searched for at all.\n"
            "Widening it re-runs the neighbour search; the bond threshold "
            "below only classifies what was found.")
        self.list_threshold.valueChanged.connect(self._on_list_threshold)
        row.addWidget(self.list_threshold)
        return bar

    def _build_menu(self) -> None:
        openf = QAction("&Open structure…", self)
        openf.setShortcut(QKeySequence.Open)
        openf.triggered.connect(self._choose_file)

        save = QAction("&Export image…", self)
        save.setShortcut(QKeySequence.Save)
        save.triggered.connect(self._save_image)

        quit_ = QAction("&Quit", self)
        quit_.setShortcut(QKeySequence.Quit)
        quit_.triggered.connect(self.close)

        openfolder = QAction("Open a &folder of CIFs…", self)
        openfolder.triggered.connect(self._choose_folder)

        m = self.menuBar().addMenu("&File")
        m.addAction(openf)
        m.addAction(openfolder)
        m.addSeparator()
        m.addAction(save)

        export = m.addMenu("&Export")
        for label, handler in (
                ("Site table (&CSV)…", self._export_sites_csv),
                ("Contact table (CSV)…", self._export_contacts_csv),
                ("Both tables (&XLSX)…", self._export_xlsx),
                (None, None),
                ("Structure as CI&F…", self._export_cif),
                ("Structure as &POSCAR…", self._export_poscar),
                ("Structure as X&YZ…", self._export_xyz),
                ("Structure as &VESTA…", self._export_vesta),
                (None, None),
                ("FEFF input for this &site…", self._export_feff)):
            if label is None:
                export.addSeparator()
                continue
            action = QAction(label, self)
            action.triggered.connect(handler)
            export.addAction(action)

        m.addSeparator()
        session_save = QAction("Save sessio&n…", self)
        session_save.triggered.connect(self._save_session)
        session_open = QAction("Open session…", self)
        session_open.triggered.connect(self._open_session)
        m.addAction(session_save)
        m.addAction(session_open)
        m.addSeparator()
        m.addAction(quit_)

        v = self.menuBar().addMenu("&View")
        reset = QAction("&Reset view", self)
        reset.setShortcut("R")
        reset.triggered.connect(self.view.reset_view)
        v.addAction(reset)
        v.addSeparator()

        axes = QActionGroup(self)
        for name, axis in (("Along &a", (1, 0, 0)), ("Along &b", (0, 1, 0)),
                           ("Along &c", (0, 0, 1))):
            act = QAction(name, self)
            act.triggered.connect(lambda _=False, ax=axis: self._view_along(ax))
            axes.addAction(act)
            v.addAction(act)

        h = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self._about)
        h.addAction(about)

    def _set_enabled(self, on: bool) -> None:
        for w in (self.style_box, self.poly_box, self.cell_check,
                  self.atom_label_box, self.bond_label_box,
                  self.ortho_check, self.list_threshold, *self.range_boxes):
            w.setEnabled(on)

    # -- drag and drop -----------------------------------------------------
    def dragEnterEvent(self, event) -> None:
        """Accept a dropped structure file.

        The shortest path from "someone sent me a CIF" to an answer, and the
        one a person who does not write code will reach for first.
        """
        if event.mimeData().hasUrls() and any(
                u.toLocalFile().lower().endswith((".cif", ".mcif"))
                for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".cif", ".mcif")):
                self.load(path)
                event.acceptProposedAction()
                return

    # -- loading -----------------------------------------------------------
    def _choose_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open structures", "",
            "Crystal structures (*.cif);;All files (*)")
        if paths:
            self.load_many(paths)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a folder of CIFs")
        if not folder:
            return
        paths = sorted(str(p) for p in Path(folder).glob("*.cif"))
        if not paths:
            QMessageBox.information(self, "Nothing to open",
                                    f"No .cif files in {folder}")
            return
        self.load_many(paths)

    def load_many(self, paths) -> None:
        """Load a set of files. Failures are listed, not raised.

        A folder of downloaded CIFs reliably contains a few that will not
        parse; the rest still open.
        """
        added, failed = self.project.add_files(paths)
        self.structure_panel.refresh()
        if added:
            self.project.set_active(len(self.project) - len(added))
            self._after_load(reframe=True)
        if failed:
            detail = "\n".join(f"{Path(p).name}: {why}" for p, why in failed)
            QMessageBox.warning(
                self, "Some files could not be read",
                f"{len(added)} loaded, {len(failed)} skipped.\n\n{detail}")

    def load(self, path: str) -> None:
        self.load_many([path])

    def _after_load(self, reframe: bool = False) -> None:
        entry = self.project.current
        if entry is None:
            return
        results = self.project.results_for(entry)
        if entry.selected_site is None and results:
            entry.selected_site = results[0].site_index
        elements = sorted({e for x in self.project for e in
                           x.structure.elements_present})
        self.theme_panel.set_elements(elements)
        self._fill_site_list()
        self.structure_panel.refresh()
        self._set_enabled(True)
        self._rebuild(reframe=reframe)
        self.setWindowTitle(
            f"{NAME} {__version__} — {entry.name}  ·  "
            f"{entry.structure.spacegroup_hm or 'unknown symmetry'}"
            + (f"   [{len(self.project)} structures]" if len(self.project) > 1
               else ""))
        if entry.structure.notes:
            self.statusBar().showMessage(entry.structure.notes[0], 14000)

    def _fill_site_list(self) -> None:
        self.site_list.blockSignals(True)
        self.site_list.clear()
        self._rows: list[int] = []
        for r in self.results:
            item = QListWidgetItem(f"{r.label}   CN {r.cn_valence}")
            item.setToolTip(r.summary())
            self.site_list.addItem(item)
            self._rows.append(r.site_index)
        if self.site_index in self._rows:
            self.site_list.setCurrentRow(self._rows.index(self.site_index))
        self.site_list.blockSignals(False)

    # -- rebuilding --------------------------------------------------------
    def _polyhedron_sites(self) -> list[int]:
        entry = self.project.current
        if entry is None or self.poly_mode == PolyhedraMode.NONE:
            return []
        if self.poly_mode == PolyhedraMode.ALL:
            return [r.site_index for r in self.project.results_for(entry)]
        return [entry.selected_site] if entry.selected_site is not None else []

    def _rebuild(self, reframe: bool = False) -> None:
        entries = self.project.visible_entries
        if not entries:
            return
        style = self.style_box.currentData() or Style.BALL_AND_STICK
        cell_range = tuple(b.value() for b in self.range_boxes)
        active = self.project.current

        scenes = []
        for entry in entries:
            results = self.project.results_for(entry)
            sites = (self._polyhedron_sites() if entry is active
                     else ([r.site_index for r in results]
                           if self.poly_mode == PolyhedraMode.ALL else []))
            scenes.append(build_scene(
                entry.structure, results, style=style,
                v_bond=self.project.v_bond, v_list=self.project.v_list,
                polyhedron_sites=sites,
                show_cell=self.cell_check.isChecked(),
                cell_range=cell_range, theme=self.theme,
                params=self.project.params))

        self.scene = (scenes[0] if len(scenes) == 1
                      else merge_scenes(scenes, [e.offset for e in entries]))
        self.view.set_scene(self.scene, reframe=reframe)
        self.view.set_theme(self.theme)
        self.view.set_label_context(
            active.structure if active else None,
            self.project.results_for(active) if active else None)
        self.explorer.set_result(self._current_result())
        self.explorer.set_threshold(self.project.v_bond)
        self._update_analysis()
        self.utilities.update_for(
            active.structure if active else None,
            self.project.results_for(active) if active else None,
            self._current_result(), self.project.v_bond)

    def _current_result(self):
        entry = self.project.current
        if entry is None or entry.selected_site is None:
            return None
        return next((r for r in self.project.results_for(entry)
                     if r.site_index == entry.selected_site), None)

    # -- events ------------------------------------------------------------
    def _on_ready(self, caps) -> None:
        self.statusBar().showMessage(f"Renderer: {caps.describe()}", 15000)

    def _on_site_picked(self, site_index: int) -> None:
        if site_index < 0 or self.structure is None:
            return
        if not any(r.site_index == site_index for r in self.results):
            self.statusBar().showMessage(
                f"{self.structure.sites[site_index].label} is an anion site — "
                "the coordination analysis is carried on cation sites", 6000)
            return
        self.site_index = site_index
        if site_index in getattr(self, "_rows", []):
            self.site_list.blockSignals(True)
            self.site_list.setCurrentRow(self._rows.index(site_index))
            self.site_list.blockSignals(False)
        self._rebuild()

    def _on_site_row(self, row: int) -> None:
        if 0 <= row < len(getattr(self, "_rows", [])):
            self.site_index = self._rows[row]
            self._rebuild()

    def _on_labels(self) -> None:
        self.labels.atom = self.atom_label_box.currentData()
        self.labels.bond = self.bond_label_box.currentData()
        self.view.set_labels(self.labels)

    def _on_labels_cycled(self, settings) -> None:
        """The viewport cycled the atom label kind with the L key."""
        self.labels = settings
        i = self.atom_label_box.findData(settings.atom)
        if i >= 0:
            self.atom_label_box.setCurrentIndex(i)

    def _on_structure_row(self, row: int) -> None:
        self.project.set_active(row)
        self._after_load(reframe=not self.project.overlay)

    def _on_overlay(self, on: bool, spacing: float) -> None:
        self.project.set_overlay(on, spacing)
        self.structure_panel.refresh()
        self._rebuild(reframe=True)

    def _on_remove(self, row: int) -> None:
        if row < 0:
            return
        self.project.remove(row)
        self.structure_panel.refresh()
        if len(self.project):
            self._after_load(reframe=True)
        else:
            self.scene = None
            self.site_list.clear()
            self._set_enabled(False)
            self.setWindowTitle(f"{NAME} {__version__}")

    def _on_list_threshold(self, value: float) -> None:
        """The tabulation threshold. Unlike the bond threshold this DOES
        re-run the neighbour search, so it is a spin box rather than a drag."""
        self.project.set_list_threshold(value)
        self._rebuild()

    def _on_poly_mode(self) -> None:
        self.poly_mode = self.poly_box.currentData()
        self._rebuild()

    def _on_threshold(self, v: float) -> None:
        """Restyle, do not re-analyse."""
        self.project.set_bond_threshold(float(v))
        self.view.set_threshold(self.project.v_bond)
        self._update_analysis()
        entry = self.project.current
        self.utilities.update_for(
            entry.structure if entry else None,
            self.project.results_for(entry) if entry else None,
            self._current_result(), self.project.v_bond)

    def _on_theme_cosmetic(self, theme) -> None:
        """Background, fog, ambient occlusion: no vertex data changes."""
        self.theme = theme
        self.view.set_theme(theme)
        self.explorer.set_theme(theme)

    def _on_theme_structural(self, theme) -> None:
        """Colour mode, palette, sizes: the vertex arrays must be rebuilt."""
        self.theme = theme
        self.explorer.set_theme(theme)
        self._rebuild()

    def _view_along(self, axis) -> None:
        if self.structure is None:
            return
        # a crystallographic axis, converted to the cartesian frame
        import numpy as np

        direction = self.structure.cell.orth @ np.array(axis, float)
        self.view.view_along(direction)

    # -- analysis panel ----------------------------------------------------
    def _update_analysis(self) -> None:
        r = self._current_result()
        if r is None:
            self.analysis.setHtml(
                "<p style='color:#8a93a3'>Click a cation, or choose a site.</p>")
            return

        v = self.v_bond
        bonded = [c for c in r.contacts if c.has_valence and c.valence > v]
        cn = len(bonded)
        bvs = sum(c.valence * c.occupancy for c in bonded)
        discrepancy = ("" if r.ox is None
                       else f" ({bvs - r.ox:+.2f} against {r.ox:+d})")
        # The R0 uncertainty is systematic: it scales the whole sum rather
        # than averaging out, so a discrepancy smaller than it is not a
        # measurement of anything.
        uncertainty = ("" if r.bvs_uncertainty != r.bvs_uncertainty
                       else f" &plusmn; {r.bvs_uncertainty:.2f}")

        rows = "".join(
            f"<tr><td>{c.label}</td>"
            f"<td align='right'>{c.distance:.4f}</td>"
            f"<td align='right'>{c.valence:.4f}</td>"
            f"<td align='center'>{'&#9679;' if c.valence > v else '&#9675;'}</td>"
            f"<td style='color:#8a93a3'>"
            f"{'' if (c.param and c.param.fitted) else 'est.'}</td></tr>"
            for c in r.contacts if c.has_valence)

        plateau = r.current_plateau
        plateau_row = (
            f"<tr><td>Plateau at this threshold</td><td align='right'>"
            f"{plateau.width_decades:.2f} decades &middot; "
            f"{plateau.width_angstrom:.3f} &Aring;</td></tr>"
            if plateau else
            "<tr><td>Plateau at this threshold</td>"
            "<td align='right'>on a step edge</td></tr>")

        # statements of fact about how the numbers were produced, in the same
        # muted voice as the rest of the provenance
        notes = "".join(
            f"<p style='color:#8a93a3;font-size:11px;margin:3px 0'>{n}</p>"
            for n in r.notes)

        param = next((c.param for c in r.contacts if c.param), None)
        provenance = ""
        if param is not None:
            provenance = (
                f"<p style='color:#8a93a3;font-size:11px;margin-top:10px'>"
                f"R<sub>0</sub>({param.label}) = {param.r0:.3f} Å, "
                f"b = {param.b:.2f} Å &middot; "
                f"{'fitted' if param.fitted else 'estimated'}<br>{param.source}"
                f"</p>")

        self.analysis.setHtml(f"""
        <h2 style='margin-bottom:0'>{r.label}</h2>
        <p style='color:#8a93a3;margin-top:2px'>
          {r.element}{'' if r.ox is None else f'{r.ox:+d}'}
          &middot; Wyckoff {r.wyckoff or '?'}
          &middot; site symmetry {r.site_symmetry or '?'}
          &middot; multiplicity {r.multiplicity or '?'}
          &middot; oxidation state from {r.ox_source}
        </p>
        <table width='100%' cellspacing='0' cellpadding='3'>
          <tr><td>Coordination number</td>
              <td align='right'><b style='font-size:15px'>{cn}</b></td></tr>
          <tr><td>Bond-valence sum</td>
              <td align='right'>{bvs:.2f}{uncertainty} v.u.{discrepancy}</td></tr>
          <tr><td>ECoN (Hoppe)</td><td align='right'>{r.cn_ecoN:.2f}</td></tr>
          <tr><td>Maximum-gap split</td><td align='right'>{r.cn_gap}</td></tr>
          <tr><td>&phi; (stereoactivity)</td><td align='right'>{r.phi:.3f}</td></tr>
          <tr><td>Void cone half-angle</td>
              <td align='right'>{r.void_angle:.1f}&deg;</td></tr>
          <tr><td>Mean bond length</td>
              <td align='right'>{r.shape.get('d_mean') or float('nan'):.4f} Å</td></tr>
          <tr><td>Spread within the polyhedron</td>
              <td align='right'>{r.shape.get('spread') or float('nan'):.4f} Å</td></tr>
          {plateau_row}
        </table>
        {notes}
        <h4 style='margin-bottom:2px'>Contacts</h4>
        <table width='100%' cellspacing='0' cellpadding='2' style='font-size:11px'>
          <tr style='color:#8a93a3'>
            <th align='left'>atom</th><th align='right'>d / Å</th>
            <th align='right'>v / v.u.</th><th></th><th></th></tr>
          {rows}
        </table>
        {provenance}
        """)

    # -- export ------------------------------------------------------------
    def _provenance(self) -> list[str]:
        from ..core.version_info import provenance_lines

        estimated = any(r.uses_estimated_params
                        for e in self.project
                        for r in self.project.results_for(e))
        return provenance_lines(self.project.params, self.project.v_bond,
                                self.project.v_list, estimated)

    def _ask(self, title: str, default: str, filt: str) -> str | None:
        path, _ = QFileDialog.getSaveFileName(self, title, default, filt)
        return path or None

    def _wrote(self, path) -> None:
        self.statusBar().showMessage(f"Wrote {path}", 8000)

    def _export_sites_csv(self) -> None:
        path = self._ask("Export the site table", "facet_sites.csv",
                         "CSV (*.csv)")
        if path:
            exporters.write_csv(self.project.site_table(), path,
                                self._provenance())
            self._wrote(path)

    def _export_contacts_csv(self) -> None:
        path = self._ask("Export the contact table", "facet_contacts.csv",
                         "CSV (*.csv)")
        if path:
            exporters.write_csv(self.project.contact_table(), path,
                                self._provenance())
            self._wrote(path)

    def _export_xlsx(self) -> None:
        path = self._ask("Export both tables", "facet.xlsx", "Excel (*.xlsx)")
        if path:
            out = exporters.write_xlsx({"sites": self.project.site_table(),
                                        "contacts": self.project.contact_table()},
                                       path, self._provenance())
            self._wrote(out)

    def _export_cif(self) -> None:
        entry = self.project.current
        if entry is None:
            return
        path = self._ask("Export as CIF", f"{Path(entry.name).stem}_facet.cif",
                         "CIF (*.cif)")
        if path:
            exporters.write_cif(entry.structure, path,
                                self.project.results_for(entry),
                                self._provenance())
            self._wrote(path)

    def _export_poscar(self) -> None:
        entry = self.project.current
        if entry is None:
            return
        path = self._ask("Export as POSCAR", "POSCAR", "VASP (POSCAR*);;All (*)")
        if path:
            exporters.write_poscar(entry.structure, path)
            self._wrote(path)

    def _export_xyz(self) -> None:
        entry = self.project.current
        if entry is None:
            return
        path = self._ask("Export as XYZ", f"{Path(entry.name).stem}.xyz",
                         "XYZ (*.xyz)")
        if path:
            exporters.write_xyz(entry.structure, path,
                                cell_range=tuple(b.value()
                                                 for b in self.range_boxes))
            self._wrote(path)

    def _export_vesta(self) -> None:
        entry = self.project.current
        if entry is None:
            return
        path = self._ask("Export for VESTA", f"{Path(entry.name).stem}.vesta",
                         "VESTA (*.vesta)")
        if path:
            exporters.write_vesta(entry.structure, path, theme=self.theme,
                                  v_bond=self.project.v_bond)
            self._wrote(path)

    def _export_feff(self) -> None:
        entry = self.project.current
        result = self._current_result()
        if entry is None or result is None:
            QMessageBox.information(self, "No site selected",
                                    "Select a cation site first.")
            return
        path = self._ask(f"FEFF input for {result.label}", "feff.inp",
                         "FEFF input (*.inp)")
        if path:
            exporters.write_feff(entry.structure, result.site_index, path)
            self._wrote(path)

    def _save_session(self) -> None:
        path = self._ask("Save the session", "facet_session.json",
                         "FACET session (*.json)")
        if path:
            exporters.save_session(self.project, path, theme=self.theme,
                                   camera=self.view.camera, labels=self.labels)
            self._wrote(path)

    def _open_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open a session", "",
                                              "FACET session (*.json)")
        if not path:
            return
        try:
            data = exporters.load_session(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not read the session", str(exc))
            return

        paths = [e["path"] for e in data.get("entries", []) if e.get("path")]
        missing = [p for p in paths if not Path(p).exists()]
        self.project.clear()
        self.load_many([p for p in paths if Path(p).exists()])

        self.project.set_bond_threshold(data.get("v_bond",
                                                 self.project.v_bond))
        self.project.set_list_threshold(data.get("v_list",
                                                 self.project.v_list))
        if "theme" in data:
            self.theme = theme_mod.Theme.from_dict(data["theme"])
            self.theme_panel.theme = self.theme
            self.theme_panel._reload()
        if "camera" in data:
            import numpy as np

            cam = data["camera"]
            self.view.camera.target = np.array(cam["target"], float)
            self.view.camera.distance = cam["distance"]
            self.view.camera.orientation = np.array(cam["orientation"], float)
            self.view.camera.fov = cam["fov"]
            self.view.camera.orthographic = cam["orthographic"]
        self.project.set_overlay(data.get("overlay", False),
                                 data.get("overlay_spacing", 0.0))
        self.structure_panel.refresh()
        self._rebuild(reframe="camera" not in data)
        if missing:
            QMessageBox.information(
                self, "Some files have moved",
                "A session records where the files were, not their contents.\n\n"
                + "\n".join(Path(p).name for p in missing))

    # -- output ------------------------------------------------------------
    def _save_image(self) -> None:
        if self.scene is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export image", "structure.png", "PNG image (*.png)")
        if not path:
            return
        self.view.grab_image(2400, 1800, supersample=2).save(path)
        self.statusBar().showMessage(f"Wrote {path}", 8000)

    def _about(self) -> None:
        caps = self.view.caps
        QMessageBox.information(
            self, f"About {NAME}",
            f"<b>{NAME} {__version__}</b><br>"
            "Coordination analysis, cut by bond valence.<br><br>"
            f"Renderer: {caps.describe() if caps else 'not initialised'}")


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtGui import QSurfaceFormat

    from ..gl import caps as caps_mod

    argv = list(sys.argv if argv is None else argv)
    QSurfaceFormat.setDefaultFormat(caps_mod.request_format())
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(NAME)

    window = PreviewWindow(argv[1] if len(argv) > 1 else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
