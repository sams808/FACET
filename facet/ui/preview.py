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

from ..core import (bv, cif, coordination, exporters, history as history_mod,
                    overrides as overrides_mod, planes as planes_mod,
                    project as project_mod, readers, theme as theme_mod)
from ..gl.labels import AtomLabel, BondLabel, LabelScope, LabelSettings
from ..gl.scene import Style, build_scene, merge_scenes
from ..gl.view import StructureView
from ..version import NAME, __version__
from .cutoff_explorer import CutoffExplorer
from .structure_list import StructureList
from .theme_panel import ThemePanel
from .diffraction_panel import DiffractionPanel
from .overrides_panel import OverridesPanel
from .planes_panel import PlanesPanel
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
        self.theme_panel.viewingChanged.connect(self._on_viewing)

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

        self.diffraction = DiffractionPanel()
        self.diffraction.apply_theme(self.theme)
        self.diffraction.reflection_selected.connect(self._on_reflection)

        self.planes_panel = PlanesPanel()
        self.planes_panel.changed.connect(self._on_presentation_change)

        self.overrides_panel = OverridesPanel()
        self.overrides_panel.changed.connect(self._on_presentation_change)
        self.overrides_panel.focusAtom.connect(self._focus_atom)

        self.history = history_mod.History()
        self.view.contextRequested.connect(self._on_context_menu)

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
        tabs.addTab(self.diffraction, "Diffraction")
        tabs.addTab(self.planes_panel, "Planes")
        tabs.addTab(self.overrides_panel, "Overrides")
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

        e = self.menuBar().addMenu("&Edit")
        self.undo_action = QAction("&Undo", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.undo_action.triggered.connect(self._undo)
        self.redo_action = QAction("&Redo", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.redo_action.triggered.connect(self._redo)
        e.addAction(self.undo_action)
        e.addAction(self.redo_action)
        e.addSeparator()
        clear_overrides = QAction("Clear all per-atom and per-site styles", self)
        clear_overrides.triggered.connect(self._clear_overrides)
        e.addAction(clear_overrides)
        self._refresh_history_actions()

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
    _DROPPABLE = (".cif", ".mcif", ".vasp", ".xyz", ".extxyz", ".vesta",
                  ".res", ".ins", ".pdb", ".ent", ".cmtx", ".cmdf")

    def _droppable(self, path: str) -> bool:
        lower = path.lower()
        name = Path(path).name.upper()
        return (lower.endswith(self._DROPPABLE)
                or name.startswith(("POSCAR", "CONTCAR")))

    def dragEnterEvent(self, event) -> None:
        """Accept dropped structure files, of any format FACET reads.

        The shortest path from "someone sent me a file" to an answer, and the
        one a person who does not write code will reach for first.
        """
        if event.mimeData().hasUrls() and any(
                self._droppable(u.toLocalFile())
                for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [u.toLocalFile() for u in event.mimeData().urls()
                 if self._droppable(u.toLocalFile())]
        if paths:
            self.load_many(paths)
            event.acceptProposedAction()

    # -- loading -----------------------------------------------------------
    def _choose_file(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open structures", "", readers.FILE_FILTER)
        if paths:
            self.load_many(paths)

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a folder of CIFs")
        if not folder:
            return
        patterns = ("*.cif", "*.mcif", "POSCAR*", "CONTCAR*", "*.vasp",
                    "*.xyz", "*.extxyz", "*.vesta", "*.res", "*.ins",
                    "*.pdb", "*.cmtx")
        found: list[str] = []
        for pattern in patterns:
            found += [str(p) for p in Path(folder).glob(pattern)]
        paths = sorted(set(found))
        if not paths:
            QMessageBox.information(
                self, "Nothing to open",
                f"No structure files in {folder}. FACET reads CIF, POSCAR, XYZ, "
                "VESTA .vesta, SHELX .res/.ins, PDB and CrystalMaker "
                ".cmtx.")
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
        # An atom override is keyed by index into the drawn scene, so a change of
        # structure or cell range can leave it addressing a different atom. Drop
        # the ones that no longer address anything rather than let them land
        # somewhere arbitrary.
        entry.overrides.prune(len(entry.structure.atoms),
                              [site.label for site in entry.structure.sites])
        self._rebuild(reframe=reframe)
        self.setWindowTitle(
            f"{NAME} {__version__} — {entry.name}  ·  "
            f"{entry.structure.spacegroup_hm or 'unknown symmetry'}"
            + (f"   [{len(self.project)} structures]" if len(self.project) > 1
               else ""))
        # A newly opened structure starts its own history. Undo covers how a
        # structure is drawn, not which structure is open, so there is nothing
        # sensible to step back to across a load.
        self.history.reset(self._snapshot(), f"opened {entry.name}")
        self._refresh_history_actions()
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
                params=self.project.params,
                lattice_planes=(self.planes_panel.current_planes()
                                if entry is active else None),
                slab=(self.planes_panel.current_slab()
                      if entry is active else None),
                overrides=entry.overrides))

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
        self.diffraction.set_structure(active.structure if active else None)
        self.planes_panel.set_structure(active.structure if active else None)
        self.overrides_panel.set_context(
            active.structure if active else None,
            active.overrides if active else None)

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
        self.diffraction.apply_theme(theme)

    def _on_theme_structural(self, theme) -> None:
        """Colour mode, palette, sizes: the vertex arrays must be rebuilt."""
        self.theme = theme
        self.explorer.set_theme(theme)
        self.diffraction.apply_theme(theme)
        self._rebuild()

    def _on_reflection(self, h: int, k: int, l: int) -> None:
        """Look down the normal of the chosen reflection's planes.

        The normal is h a* + k b* + l c*, in the reciprocal basis -- not
        h a + k b + l c, which points somewhere else in any cell that is not
        orthogonal. Choosing a line in the reflection list therefore turns the
        structure to show the planes that produced it edge-on.
        """
        structure = self.structure
        if structure is None:
            return
        import numpy as np

        reciprocal = np.linalg.inv(structure.cell.orth).T
        normal = reciprocal @ np.array([h, k, l], float)
        if np.linalg.norm(normal) > 1e-9:
            self.view.view_along(normal)
            self.statusBar().showMessage(
                f"viewing down the normal of ({h} {k} {l})", 6000)

    # -- presentation state, undo and the context menu ----------------------
    def _snapshot(self) -> dict:
        """Everything undo restores. Deliberately not the structures.

        Undo changes how a structure is drawn, never what it is: loading a file,
        closing one, and anything written to disk are all outside it.
        """
        entry = self.project.current
        slab = self.planes_panel.slab
        return {
            "overrides": (entry.overrides.to_dict() if entry is not None
                          else None),
            "planes": [
                {"h": p.h, "k": p.k, "l": p.l, "offset": p.offset,
                 "color": list(p.color), "alpha": p.alpha,
                 "visible": p.visible, "repeat": p.repeat,
                 "show_edges": p.show_edges}
                for p in self.planes_panel.current_planes()],
            "slab": {"h": slab.h, "k": slab.k, "l": slab.l,
                     "centre": slab.centre, "thickness": slab.thickness,
                     "enabled": slab.enabled},
            "v_bond": self.project.v_bond,
        }

    def _restore(self, snapshot: dict) -> None:
        if not snapshot:
            return
        entry = self.project.current
        if entry is not None and snapshot.get("overrides") is not None:
            entry.overrides = overrides_mod.StyleOverrides.from_dict(
                snapshot["overrides"])

        self.planes_panel.planes = [
            planes_mod.LatticePlane(
                raw["h"], raw["k"], raw["l"], offset=raw["offset"],
                color=tuple(raw["color"]), alpha=raw["alpha"],
                visible=raw["visible"], show_edges=raw.get("show_edges", True),
                repeat=raw.get("repeat", 1))
            for raw in snapshot.get("planes", [])]

        raw = snapshot.get("slab") or {}
        slab = self.planes_panel.slab
        for name in ("h", "k", "l", "centre", "thickness", "enabled"):
            if name in raw:
                setattr(slab, name, raw[name])
        self.planes_panel.load_slab_controls()
        self.planes_panel.reload()

        if "v_bond" in snapshot:
            self.project.set_bond_threshold(float(snapshot["v_bond"]))
            self.explorer.set_threshold(self.project.v_bond)

        self.overrides_panel.set_context(
            entry.structure if entry else None,
            entry.overrides if entry else None)
        self._rebuild()
        self._refresh_history_actions()

    def _on_presentation_change(self, description: str = "") -> None:
        """A panel changed something drawable: rebuild, then record it."""
        self._rebuild()
        self.history.push(description or "changed the presentation",
                          self._snapshot())
        self._refresh_history_actions()

    def _refresh_history_actions(self) -> None:
        undo = getattr(self, "undo_action", None)
        redo = getattr(self, "redo_action", None)
        if undo is None or redo is None:
            return
        undo.setEnabled(self.history.can_undo)
        redo.setEnabled(self.history.can_redo)
        undo.setText(("&Undo " + self.history.undo_description).rstrip()
                     if self.history.can_undo else "&Undo")
        redo.setText(("&Redo " + self.history.redo_description).rstrip()
                     if self.history.can_redo else "&Redo")

    def _undo(self) -> None:
        description = self.history.undo_description
        snapshot = self.history.undo()
        if snapshot is None:
            return
        self._restore(snapshot)
        self.statusBar().showMessage("undid: " + description, 6000)

    def _redo(self) -> None:
        snapshot = self.history.redo()
        if snapshot is None:
            return
        description = self.history.undo_description
        self._restore(snapshot)
        self.statusBar().showMessage("redid: " + description, 6000)

    def _clear_overrides(self) -> None:
        entry = self.project.current
        if entry is None or entry.overrides.is_empty:
            return
        entry.overrides.clear()
        self.overrides_panel.refresh()
        self._on_presentation_change("removed every override")

    def _focus_atom(self, index: int) -> None:
        if self.scene is None or not (0 <= index < self.scene.n_atoms):
            return
        self.view.select_atom(index)
        self.statusBar().showMessage(
            self.scene.atom_label[index] + " ("
            + self.scene.atom_element[index] + ")", 6000)

    def _on_context_menu(self, atom_index: int, global_pos=None) -> None:
        """Raise the right-click menu for whatever is under the pointer."""
        from PySide6.QtGui import QCursor

        menu = self._build_context_menu(atom_index)
        if menu is None:
            return
        menu.exec(global_pos if global_pos is not None else QCursor.pos())

    def _build_context_menu(self, atom_index: int):
        """The right-click menu, built but not shown.

        Kept separate from showing it so the menu's contents can be inspected
        without entering a modal event loop -- the contents are the part worth
        testing, and exec() would block a test forever.
        """
        from PySide6.QtWidgets import QMenu

        entry = self.project.current
        if entry is None:
            return None
        scene = self.scene
        menu = QMenu(self)
        # Submenus are constructed with `menu` as their parent rather than with
        # menu.addMenu("title"): that form returns a submenu owned by Python, so
        # the only reference keeping it alive would be a local in this method
        # and it would be destroyed the moment this returns -- before the menu is
        # shown.

        if scene is not None and 0 <= atom_index < scene.n_atoms:
            label = scene.atom_label[atom_index]
            element = scene.atom_element[atom_index]
            site_index = int(scene.atom_site[atom_index])
            site_label = (entry.structure.sites[site_index].label
                          if 0 <= site_index < len(entry.structure.sites)
                          else label)

            header = menu.addAction(label + "  (" + element + ")")
            header.setEnabled(False)
            menu.addSeparator()

            menu.addAction("Select this site").triggered.connect(
                lambda _=False, i=site_index: self._on_site_picked(i))
            menu.addAction("Show this site's polyhedron").triggered.connect(
                lambda _=False, i=site_index: self._show_polyhedron_for(i))
            menu.addSeparator()

            site_menu = QMenu("Site " + site_label, menu)
            menu.addMenu(site_menu)
            site_menu.addAction("Colour...").triggered.connect(
                lambda _=False: self._pick_override_colour("site", site_label))
            site_menu.addAction("Hide").triggered.connect(
                lambda _=False: self._apply_override("site", site_label,
                                                     "hid", visible=False))
            site_menu.addAction("Show").triggered.connect(
                lambda _=False: self._apply_override("site", site_label,
                                                     "showed", visible=True))
            site_menu.addAction("Bigger").triggered.connect(
                lambda _=False: self._scale_override("site", site_label, 1.25))
            site_menu.addAction("Smaller").triggered.connect(
                lambda _=False: self._scale_override("site", site_label, 0.8))
            site_menu.addSeparator()
            site_menu.addAction("Clear this site").triggered.connect(
                lambda _=False: self._clear_override("site", site_label))

            atom_menu = QMenu("This atom only", menu)
            menu.addMenu(atom_menu)
            atom_menu.addAction("Colour...").triggered.connect(
                lambda _=False: self._pick_override_colour("atom", atom_index))
            atom_menu.addAction("Hide").triggered.connect(
                lambda _=False: self._apply_override("atom", atom_index,
                                                     "hid", visible=False))
            atom_menu.addAction("Bigger").triggered.connect(
                lambda _=False: self._scale_override("atom", atom_index, 1.25))
            atom_menu.addAction("Smaller").triggered.connect(
                lambda _=False: self._scale_override("atom", atom_index, 0.8))
            atom_menu.addSeparator()
            atom_menu.addAction("Clear this atom").triggered.connect(
                lambda _=False: self._clear_override("atom", atom_index))

            element_menu = QMenu("Every " + element, menu)
            menu.addMenu(element_menu)
            element_menu.addAction("Colour...").triggered.connect(
                lambda _=False: self._pick_override_colour("element", element))
            element_menu.addAction("Hide").triggered.connect(
                lambda _=False: self._apply_override("element", element,
                                                     "hid", visible=False))
            element_menu.addSeparator()
            element_menu.addAction("Clear").triggered.connect(
                lambda _=False: self._clear_override("element", element))
            menu.addSeparator()

        menu.addAction("Reset the view").triggered.connect(self.view.reset_view)
        for name, axis in (("Look along a", (1, 0, 0)),
                           ("Look along b", (0, 1, 0)),
                           ("Look along c", (0, 0, 1))):
            menu.addAction(name).triggered.connect(
                lambda _=False, ax=axis: self._view_along(ax))
        menu.addSeparator()
        if not entry.overrides.is_empty:
            menu.addAction("Remove every override").triggered.connect(
                self._clear_overrides)
        if self.history.can_undo:
            menu.addAction("Undo " + self.history.undo_description
                           ).triggered.connect(self._undo)
        return menu

    def _show_polyhedron_for(self, site_index: int) -> None:
        entry = self.project.current
        if entry is None:
            return
        entry.selected_site = site_index
        self._fill_site_list()
        self._rebuild()

    def _apply_override(self, level: str, target, verb: str, **fields) -> None:
        entry = self.project.current
        if entry is None:
            return
        if level == "site":
            entry.overrides.set_site(target, **fields)
        elif level == "atom":
            entry.overrides.set_atom(int(target), **fields)
        else:
            entry.overrides.set_element(target, **fields)
        self.overrides_panel.refresh()
        self._on_presentation_change(verb + " " + level + " " + str(target))

    def _scale_override(self, level: str, target, factor: float) -> None:
        """Multiply the drawn size, compounding with any scale already set."""
        entry = self.project.current
        if entry is None:
            return
        if level == "site":
            existing = entry.overrides.by_site.get(target)
        elif level == "atom":
            existing = entry.overrides.by_atom.get(int(target))
        else:
            existing = entry.overrides.by_element.get(target)
        current = (1.0 if existing is None or existing.radius_scale is None
                   else existing.radius_scale)
        self._apply_override(level, target, "resized",
                             radius_scale=current * factor)

    def _pick_override_colour(self, level: str, target) -> None:
        from PySide6.QtWidgets import QColorDialog

        chosen = QColorDialog.getColor(Qt.white, self, "Colour for "
                                       + str(target))
        if not chosen.isValid():
            return
        self._apply_override(
            level, target, "coloured",
            color=(chosen.redF(), chosen.greenF(), chosen.blueF()))

    def _clear_override(self, level: str, target) -> None:
        entry = self.project.current
        if entry is None:
            return
        if level == "site":
            entry.overrides.clear_site(target)
        elif level == "atom":
            entry.overrides.clear_atom(int(target))
        else:
            entry.overrides.clear_element(target)
        self.overrides_panel.refresh()
        self._on_presentation_change(
            "cleared the override on " + level + " " + str(target))

    def _on_viewing(self, stereo_mode, separation: float, sort: bool) -> None:
        """Stereo and transparency ordering: how you are looking, not the theme."""
        self.view.set_stereo(stereo_mode, separation)
        renderer = getattr(self.view, "_renderer", None)
        if renderer is not None:
            renderer.sort_transparency = bool(sort)
            renderer._sort_key = None          # force one re-sort either way
        fallback = getattr(self.view, "_fallback", None)
        if fallback is not None:
            # the software tier sorts everything by depth already, so there is
            # nothing to switch off there
            pass
        self.view.update()
        from ..gl import stereo as stereo_mod

        if stereo_mode is not stereo_mod.Mode.OFF:
            self.statusBar().showMessage(
                f"stereo: {stereo_mode.value}, eye separation "
                f"{separation:.1f} deg", 8000)

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
                                   camera=self.view.camera, labels=self.labels,
                                   presentation=self._snapshot())
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

        # Per-entry overrides, matched back to their files by path. Matching by
        # path rather than by position means a session whose files have moved
        # loses the overrides for the files that are gone and keeps the rest,
        # instead of applying one structure's overrides to another.
        by_path = {e.get("path"): e for e in data.get("entries", [])
                   if e.get("path")}
        for entry in self.project.entries:
            saved = by_path.get(entry.path)
            if saved and saved.get("overrides"):
                entry.overrides = overrides_mod.StyleOverrides.from_dict(
                    saved["overrides"])
                entry.overrides.prune(
                    len(entry.structure.atoms),
                    [site.label for site in entry.structure.sites])

        presentation = data.get("presentation")
        self.structure_panel.refresh()
        if presentation:
            self._restore(presentation)
        self._rebuild(reframe="camera" not in data)
        active = self.project.current
        self.history.reset(self._snapshot(), "opened the session")
        self._refresh_history_actions()
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
