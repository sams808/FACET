"""The Structure workspace.

Three zones, as planned: a site list on the left, the 3D view in the middle with
its analysis panel to the right, and the cutoff explorer along the bottom.

The bottom strip is the spine. Everything above it follows the threshold it
shows, and it follows nothing -- drag it and the coordination number, the
bond-valence sum, phi, the contact table and the drawn bonds all move together,
with no neighbour search and no re-analysis.
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

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
    QMenu,
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
from . import chrome
from .cutoff_explorer import CutoffExplorer
from .structure_list import StructureList
from .theme_panel import ThemePanel
from .diffraction_panel import DiffractionPanel
from .disorder_panel import DisorderPanel
from .overrides_panel import OverridesPanel
from .exafs_panel import ExafsPanel
from .pdf_panel import PDFPanel
from .planes_panel import PlanesPanel
from .utilities_panel import UtilitiesPanel
from .volume_panel import VolumePanel


class PolyhedraMode:
    NONE = "none"
    SELECTED = "selected site"
    ALL = "all cations"


class _WorkspaceView(StructureView):
    """The crystal window's 3D view, whose empty state also says where an MD
    model goes: a dropped or opened dump, data file or trajectory opens in a
    Model window of its own (:meth:`PreviewWindow.open_md_models`), never in
    this view."""

    MD_HINT = ("MD models and trajectories (LAMMPS dump or data, extended "
               "XYZ, XDATCAR, DL_POLY, DCD, XTC, ...) open in a Model window")

    def _paint_placeholder(self, painter) -> None:
        super()._paint_placeholder(painter)
        # one line under the view's own, in the pen and font it left set
        below = self.rect().adjusted(24, 0, -24, 0)
        below.moveTop(below.top() + int(2.2 * painter.fontMetrics().height()))
        font = QFont(painter.font())
        font.setPointSizeF(max(font.pointSizeF() - 1.5, 6.0))
        painter.setFont(font)
        painter.drawText(below, Qt.AlignCenter | Qt.TextWordWrap, self.MD_HINT)


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
        # Analysis overlays, off until asked for: they are a statement about the
        # bonding rather than a picture of the crystallography, so they should
        # not be what a structure looks like when it is first opened.
        self.show_vectors = False
        self.show_void_cones = False
        self.vector_scale = 0.6
        # The Model windows opened from here (open_md_models): each is a
        # top-level window of its own, never a tab of this one.
        self._model_windows: list = []

        self.view = _WorkspaceView()
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
        # Double-click to turn about a site. The list is the only place that
        # names every site, so it is where a site that cannot be found in the
        # view -- buried, off frame, one of forty -- has to be reachable from.
        self.site_list.itemDoubleClicked.connect(self._on_site_double_click)
        self.site_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.site_list.customContextMenuRequested.connect(self._site_menu)
        self.site_list.setToolTip(
            "Double-click a site to turn the view about it")

        self.anion_check = QCheckBox("Anion sites")
        self.anion_check.setToolTip(
            "Analyse each anion as a site in its own right -- its coordination "
            "number, its contacts and its own bond-valence sum, counted from "
            "the cations around it. Off by default: in a phosphate the anions "
            "outnumber the cations three to one.")
        self.anion_check.toggled.connect(self._on_include_anions)

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

        self.exafs_panel = ExafsPanel()
        self.exafs_panel.apply_theme(self.theme)
        self.exafs_panel.statusMessage.connect(
            lambda text: self.statusBar().showMessage(text, 12000))

        self.pdf_panel = PDFPanel()
        self.pdf_panel.apply_theme(self.theme)
        self.pdf_panel.statusMessage.connect(
            lambda text: self.statusBar().showMessage(text, 12000))

        self.planes_panel = PlanesPanel()
        self.planes_panel.changed.connect(self._on_presentation_change)

        self.disorder_panel = DisorderPanel()
        self.disorder_panel.changed.connect(self._on_disorder)

        self.overrides_panel = OverridesPanel()
        self.overrides_panel.changed.connect(self._on_presentation_change)
        self.overrides_panel.focusAtom.connect(self._focus_atom)

        self.volume_panel = VolumePanel()
        self.volume_panel.apply_theme(self.theme)
        self.volume_panel.isosurfaceChanged.connect(self._on_isosurface)
        self.volume_panel.statusMessage.connect(
            lambda text: self.statusBar().showMessage(text, 9000))

        self.history = history_mod.History()
        self.view.contextRequested.connect(self._on_context_menu)
        self.view.pivotChanged.connect(self._on_pivot_changed)

        self._build_layout()
        self._build_menu()
        self._add_md_menu_items()
        # The panels are built without a theme, and the two that compose their
        # own HTML fall back to near-black text when they have none. That is
        # exactly right for the four light themes -- which is why it has never
        # shown -- and unreadable on the two dark ones. The panels only: the
        # application's own styling is the application's to do, and doing it
        # from here would restyle every other window that happens to be open.
        self._theme_the_panels(self.theme)
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
        site_box = QWidget()
        site_layout = QVBoxLayout(site_box)
        site_layout.setContentsMargins(0, 0, 0, 0)
        site_layout.setSpacing(4)
        site_layout.addWidget(self.site_list)
        site_layout.addWidget(self.anion_check)
        sites.setWidget(site_box)
        sites.setFeatures(QDockWidget.DockWidgetMovable
                          | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, sites)
        sites.setMinimumWidth(210)

        tabs = QTabWidget()
        # Every page scrolls. A page that cannot scroll sets a floor under the
        # window's height: the Volume panel wanted 1130 px of it, which forced
        # a minimum window of 1434 x 1421 on a 1739 x 930 screen -- taller than
        # the display. With setWidgetResizable the wrapper costs nothing while
        # there is room.
        for widget, title in ((self.analysis, "Site"),
                              (self.utilities, "Tools"),
                              (self.diffraction, "Diffraction"),
                              (self.pdf_panel, "PDF"),
                              (self.exafs_panel, "EXAFS"),
                              (self.planes_panel, "Planes"),
                              (self.overrides_panel, "Styles"),
                              (self.volume_panel, "Volume"),
                              (self.disorder_panel, "Disorder"),
                              (self.theme_panel, "Theme")):
            tabs.addTab(chrome.in_scroll_area(widget), title)
        # Kept at 400. The ten tab labels want 476 px of tab bar, so a wider
        # minimum would show them all -- but it also raises the window's own
        # minimum width, measured 1434 -> 1496, and a window that cannot be
        # made small enough is the worse fault of the two. Above about a
        # 1700 px window the splitter gives the column more than 476 anyway;
        # below it the bar scrolls, with the names intact rather than elided.
        tabs.setMinimumWidth(400)
        # Ten tabs need more than 400 px of tab bar, so the bar may still
        # scroll on a small screen; let it elide rather than hide a tab
        # entirely, and keep the arrows for when it does.
        tabs.setUsesScrollButtons(True)
        # Never elide a tab name. Eliding makes every tab unreadable at once,
        # which is worse than scrolling: with the arrows the names that are on
        # screen are at least the real names.
        tabs.tabBar().setElideMode(Qt.ElideNone)

        centre = QWidget()
        col = QVBoxLayout(centre)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(self._toolbar_row())
        col.addWidget(self.view, 1)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(centre)
        split.addWidget(tabs)
        # The panels used to have stretch 0, so a wider window made the 3-D view
        # wider and left them at 400 px for ever -- measured identical at 1400,
        # 1600 and 1920. They now take a quarter of the growth, which is what
        # lets the tab bar show all ten tabs on a large screen.
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        # 500, not 400: the ten tab labels want 496 px of tab bar, and a
        # column narrower than that hides some of them behind scroll arrows
        # from the moment the window opens.
        split.setSizes([980, 500])

        vertical = QSplitter(Qt.Vertical)
        vertical.addWidget(split)
        vertical.addWidget(self.explorer)
        vertical.setStretchFactor(0, 1)
        vertical.setSizes([640, 230])
        self.setCentralWidget(vertical)

    @staticmethod
    def _fit_combo(box: QComboBox) -> None:
        """Stop a combo box and its popup from eliding their own items.

        Call once, after the items are added. Two separate mechanisms are
        needed. ``AdjustToContents`` makes the *popup* size itself from the
        widest item rather than from the (possibly squeezed) closed combo, and
        the popup's view elides in the middle under the Windows styles, which
        is what turned "bond valence" into "bon…nce". ``setMinimumWidth`` is
        then the only thing a layout that has been given less room than it
        asked for still honours, so it is what keeps the *closed* combo
        readable.
        """
        box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        view = box.view()
        if view is not None:
            view.setTextElideMode(Qt.ElideNone)
        box.setMinimumWidth(box.sizeHint().width())

    def _toolbar_row(self) -> QWidget:
        # Two rows, not one. Everything below asks for about 1400 px, and on a
        # laptop screen the centre column gets roughly 900; a QHBoxLayout given
        # less than its minimum shrinks its children *below* their minimum size
        # hint, which is what truncated the label combo boxes.
        bar = QWidget()
        stack = QVBoxLayout(bar)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)

        upper = QWidget()
        row = QHBoxLayout(upper)
        row.setContentsMargins(10, 6, 10, 3)
        row.setSpacing(10)
        stack.addWidget(upper)

        self.style_box = QComboBox()
        for s in (Style.BALL_AND_STICK, Style.SPACE_FILLING,
                  Style.STICK, Style.WIREFRAME, Style.ELLIPSOIDS):
            self.style_box.addItem(s.value, s)
        self.style_box.activated.connect(lambda _=0: self._on_style())
        row.addWidget(QLabel("Style"))
        row.addWidget(self.style_box)

        # Only shown in the ellipsoid style, because it means nothing in the
        # others -- and it is not a display preference: it names the surface
        # being drawn, which is something a figure has to state.
        self.probability_box = QComboBox()
        for fraction in (0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99):
            self.probability_box.addItem(f"{fraction:.0%}", fraction)
        self.probability_box.setCurrentIndex(2)          # 50%, the convention
        self.probability_box.activated.connect(lambda _=0: self._rebuild())
        self.probability_label = QLabel("at")
        self.probability_label.setToolTip(
            "The fraction of the displacement distribution the drawn surface "
            "encloses")
        self.probability_box.setToolTip(self.probability_label.toolTip())
        row.addWidget(self.probability_label)
        row.addWidget(self.probability_box)
        self.probability_label.hide()
        self.probability_box.hide()

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

        self.ortho_check = QCheckBox("Orthographic")
        self.ortho_check.toggled.connect(self.view.set_projection)
        row.addWidget(self.ortho_check)
        row.addStretch(1)

        lower = QWidget()
        row = QHBoxLayout(lower)
        row.setContentsMargins(10, 3, 10, 6)
        row.setSpacing(10)
        stack.addWidget(lower)

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

        for box in (self.style_box, self.poly_box,
                    self.atom_label_box, self.bond_label_box):
            self._fit_combo(box)

        # Says which atom the view is turning about, when it is not the middle
        # of the cell. A camera state the user set and can see no other way:
        # the status bar message that announces it expires after a few seconds,
        # and the centre does not.
        self.pivot_label = QLabel("")
        chrome.mark_hint(self.pivot_label)
        row.addWidget(self.pivot_label)

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
        openf = QAction("&Open…", self)
        openf.setShortcut(QKeySequence.Open)
        openf.triggered.connect(self._choose_file)

        save = QAction("Export &image…", self)
        save.setShortcut(QKeySequence.Save)
        save.triggered.connect(self._save_image)

        vector = QAction("Export &vector…", self)
        vector.setToolTip("SVG or PDF: real vector geometry, not a bitmap in a "
                          "wrapper.")
        vector.triggered.connect(self._save_vector)

        quit_ = QAction("&Quit", self)
        quit_.setShortcut(QKeySequence.Quit)
        quit_.triggered.connect(self.close)

        openfolder = QAction("Open &folder…", self)
        openfolder.setToolTip("Every CIF in a folder, loaded at once.")
        openfolder.triggered.connect(self._choose_folder)

        m = self.menuBar().addMenu("&File")
        m.addAction(openf)
        m.addAction(openfolder)
        m.addSeparator()
        m.addAction(save)
        m.addAction(vector)

        export = m.addMenu("&Export")
        for label, handler in (
                ("&Sites (CSV)…", self._export_sites_csv),
                ("&Contacts (CSV)…", self._export_contacts_csv),
                ("&Both (XLSX)…", self._export_xlsx),
                (None, None),
                ("CI&F…", self._export_cif),
                ("&POSCAR…", self._export_poscar),
                ("X&YZ…", self._export_xyz),
                ("&VESTA…", self._export_vesta),
                (None, None),
                ("F&EFF input…", self._export_feff),
                (None, None),
                ("C&utoff table…", self._export_cutoffs),
                ("&Threshold scan…", self._export_scan)):
            if label is None:
                export.addSeparator()
                continue
            action = QAction(label, self)
            action.triggered.connect(handler)
            export.addAction(action)

        m.addSeparator()
        load_params = QAction("Load &parameters…", self)
        load_params.setToolTip(
            "Read a published parameter set: the IUCr bvparm distribution, a "
            "softBV-style table, or a FACET parameter file. FACET does not ship "
            "the large compilations, because each comes with its own terms.")
        load_params.triggered.connect(self._load_parameters)
        reset_params = QAction("Use the &built-in parameters", self)
        reset_params.triggered.connect(self._reset_parameters)
        save_params = QAction("Save parameter&s…", self)
        save_params.triggered.connect(self._save_parameters)
        m.addAction(load_params)
        m.addAction(save_params)
        m.addAction(reset_params)

        m.addSeparator()
        session_save = QAction("Save sessio&n…", self)
        session_save.setToolTip("The files, the threshold, the styles and the "
                                "view, so the same picture opens again.")
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
        clear_overrides = QAction("&Clear all styles", self)
        clear_overrides.setToolTip(
            "Removes every per-atom, per-site and per-element colour, size and "
            "visibility change. Undoable.")
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

        v.addSeparator()
        # Not "&cell": the View menu already spends c on "Along c", and two
        # items sharing a letter makes Qt move the highlight instead of
        # triggering -- the keyboard route then fails silently.
        cell_centre = QAction("Rotat&e about the cell centre", self)
        cell_centre.setShortcut("C")
        cell_centre.setToolTip(
            "Put the centre of rotation back at the middle of the drawn cell "
            "block. Right-click an atom to turn about that atom instead.")
        cell_centre.triggered.connect(self._center_on_cell)
        v.addAction(cell_centre)

        v.addSeparator()
        self.vector_action = QAction("Show bond-&valence vector", self)
        self.vector_action.setCheckable(True)
        self.vector_action.setToolTip(
            "A lobe on each cation along -V/|V|, where "
            "V = sum of v_i u_i over the bonded contacts. Its length is "
            "phi x the mean bond length x a display scale. This is the vector "
            "sum, which is what can be measured; for an ns2 cation it is the "
            "direction a lone pair is conventionally described as occupying.")
        self.vector_action.toggled.connect(self._on_show_vectors)
        v.addAction(self.vector_action)

        self.cone_action = QAction("Show v&oid cone", self)
        self.cone_action.setCheckable(True)
        self.cone_action.setToolTip(
            "A cone of the measured void half-angle about the void axis, on "
            "the sites whose polyhedra are drawn. Where several equally wide "
            "cones exist, one axis is returned.")
        self.cone_action.toggled.connect(self._on_show_cones)
        v.addAction(self.cone_action)

        scale = QMenu("Vector &length", v)
        scale_group = QActionGroup(self)
        scale_group.setExclusive(True)
        self._scale_actions = {}
        for label, value in (("Short (0.4)", 0.4), ("Medium (0.6)", 0.6),
                             ("Long (0.9)", 0.9), ("Very long (1.3)", 1.3)):
            act = QAction(label, self)
            act.setCheckable(True)
            act.setChecked(abs(value - self.vector_scale) < 1e-9)
            act.triggered.connect(
                lambda _=False, x=value: self._on_vector_scale(x))
            scale_group.addAction(act)
            scale.addAction(act)
            self._scale_actions[value] = act
        v.addMenu(scale)

        v.addSeparator()
        # A QMenu built with addMenu("title") is owned by Python and is
        # destroyed when this method returns; parenting it to the menu keeps it.
        themes = QMenu("&Theme", v)
        group = QActionGroup(self)
        group.setExclusive(True)
        self._theme_actions = {}
        for name in theme_mod.PRESETS:
            act = QAction(name, self)
            act.setCheckable(True)
            act.triggered.connect(lambda _=False, n=name: self._choose_theme(n))
            group.addAction(act)
            themes.addAction(act)
            self._theme_actions[name] = act
        themes.addSeparator()
        more = QAction("More colour settings…", self)
        more.setToolTip("The Appearance tab: element colours, colour modes, "
                        "sizes, and saving a theme to share with a group.")
        more.triggered.connect(self._show_appearance_tab)
        themes.addAction(more)
        v.addMenu(themes)
        self._sync_theme_menu(self.theme)

        h = self.menuBar().addMenu("&Help")
        manual = QAction("&Manual", self)
        manual.setShortcut(QKeySequence.HelpContents)
        manual.triggered.connect(self._show_manual)
        h.addAction(manual)
        shortcuts = QAction("&Keyboard and mouse", self)
        shortcuts.triggered.connect(lambda: self._show_manual("shortcuts"))
        h.addAction(shortcuts)
        h.addSeparator()
        notices = QAction("&Licences", self)
        notices.setToolTip("What the bundled components require of a build "
                           "that is passed on.")
        notices.triggered.connect(lambda: self._show_manual("licences"))
        h.addAction(notices)
        h.addSeparator()
        about = QAction("&About " + NAME, self)
        about.triggered.connect(self._about)
        h.addAction(about)

    def _on_show_vectors(self, on: bool) -> None:
        self.show_vectors = bool(on)
        self._rebuild()
        if on:
            message = ("Bond-valence vector: a lobe along -V/|V|, with "
                       "V = sum of v_i u_i over the bonded contacts; length = "
                       f"phi x mean bond length x {self.vector_scale:g}")
        else:
            message = "Bond-valence vector hidden"
        self.statusBar().showMessage(message, 9000)

    def _on_show_cones(self, on: bool) -> None:
        self.show_void_cones = bool(on)
        self._rebuild()
        if on and self.poly_mode == PolyhedraMode.NONE:
            self.statusBar().showMessage(
                "Void cones are drawn on the sites whose polyhedra are shown, "
                "and polyhedra are set to none.", 9000)

    def _on_vector_scale(self, value: float) -> None:
        self.vector_scale = float(value)
        for scale, action in getattr(self, "_scale_actions", {}).items():
            action.setChecked(abs(scale - self.vector_scale) < 1e-9)
        if self.show_vectors:
            self._rebuild()

    def _center_on_atom(self, atom_index: int, label: str) -> None:
        """Turn the view about one drawn atom.

        The index is resolved to a position here and then thrown away. An index
        into the drawn scene does not survive a rebuild -- widening the
        tabulation threshold renumbers it, a slab renumbers it, the cell range
        renumbers it -- and rebuilds happen on nearly every interaction. The
        point does survive, and it is what the camera needs.
        """
        point = self.view.atom_position(atom_index)
        if point is None:
            return
        self.view.center_on(point, label)
        self.statusBar().showMessage(
            f"Turning about {label} at "
            f"({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}) Å. "
            f"Press C, or Escape, to put the centre back in the middle of the "
            f"cell; panning with the right or middle button moves it too.",
            12000)

    def _cell_centre(self):
        """The middle of the drawn block of cells, in world coordinates.

        Not the centroid of the atoms, which is what the framing uses: for a
        centrosymmetric cell they are the same point, and for one that is not
        they can be nearly 2 Å apart. The user asked for the centre of the
        cell, so this is the cell.
        """
        entry = self.project.current
        if entry is None:
            return None
        import numpy as np

        cells = np.array([b.value() for b in self.range_boxes], float)
        centre = entry.structure.cell.orth @ (0.5 * cells)
        return centre + np.asarray(getattr(entry, "offset", np.zeros(3)), float)

    def _center_on_cell(self) -> None:
        """Put the centre of rotation back at the middle of the cell block."""
        centre = self._cell_centre()
        if centre is None:
            if self.scene is not None:
                self.view.center_on_scene()
            return
        self.view.center_on(centre, "")
        self.statusBar().showMessage(
            "Turning about the middle of the drawn cell block, at "
            f"({centre[0]:.3f}, {centre[1]:.3f}, {centre[2]:.3f}) Å.", 9000)

    def _on_pivot_changed(self, caption) -> None:
        """Say which atom the view is turning about, for as long as it is."""
        if not hasattr(self, "pivot_label"):
            return
        if caption:
            self.pivot_label.setText(f"turning about {caption}")
            self.pivot_label.setToolTip(
                "The view rotates about this atom. Press C or Escape to put "
                "the centre back in the middle of the cell.")
        else:
            self.pivot_label.setText("")
            self.pivot_label.setToolTip("")

    def _show_appearance_tab(self) -> None:
        """Bring the Theme tab forward.

        Walks up from the panel rather than naming an index, because each page
        is wrapped in a scroll area -- ``setCurrentWidget(self.theme_panel)``
        addresses the panel, which is no longer the tab widget's own child.
        """
        tabs = self.theme_panel.parentWidget()
        while tabs is not None and not isinstance(tabs, QTabWidget):
            tabs = tabs.parentWidget()
        if tabs is None:
            return
        for index in range(tabs.count()):
            page = tabs.widget(index)
            if page is self.theme_panel or page.isAncestorOf(self.theme_panel):
                tabs.setCurrentIndex(index)
                return

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
                or name.startswith(("POSCAR", "CONTCAR"))
                or md_droppable(path))

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

        An MD model or trajectory among them (:func:`read_or_route`: a file
        ``readers.read`` refuses with MDModelFile, an XYZ trajectory that
        states no box, a file under an MD format's own extension) opens in a
        Model window of its own (:meth:`open_md_models`) rather than being
        listed as unreadable, and the crystal files of the same set load here
        as they always did. File > Open, a drop, a restored session and the
        command line all come through here.
        """
        added, failed, models = [], [], []
        for path in paths:
            kind, value = read_or_route(path)
            if kind == "structure":
                added.append(self.project.add(value, str(path)))
            elif kind == "md":
                models.append((str(path), value))
            else:
                failed.append((str(path), value))
        self.structure_panel.refresh()
        if added:
            self.project.set_active(len(self.project) - len(added))
            self._after_load(reframe=True)
        if models:
            self.open_md_models(model_groups(models), dropped_with=models)
        if failed:
            detail = "\n".join(f"{Path(p).name}: {why}" for p, why in failed)
            opened = (f", {len(models)} opened in a Model window" if models
                      else "")
            QMessageBox.warning(
                self, "Some files could not be read",
                f"{len(added)} loaded{opened}, {len(failed)} skipped.\n\n"
                f"{detail}")

    def load(self, path: str) -> None:
        self.load_many([path])

    # -- MD models: a Model window each ------------------------------------
    @property
    def model_windows(self) -> list:
        """The Model windows opened from this window that are still open."""
        self._model_windows = [w for w in self._model_windows if _open(w)]
        return list(self._model_windows)

    def open_md_models(self, groups, *, read_options=None,
                       dropped_with=None) -> list:
        """Open each model in a Model window of its own; return the windows.

        ``groups`` holds one entry per model: a path, or a list of paths that
        are one model's files in order (:func:`model_groups`). The window
        (``md_workspace.open_model_window``) reads the model, asks for what
        the file does not state (a type map, a box) and runs the analyses; a
        fault in opening one window is listed with its file, and the others
        still open. ``dropped_with``: the (path, MD format) pairs opened
        together; a LAMMPS data file among them is offered to the windows
        of the other files of its folder (:func:`md_companions`), never
        filled in.
        """
        groups = list(groups)
        if not groups:
            return []
        try:
            from . import md_workspace
        except ImportError as error:
            QMessageBox.warning(
                self, "The Model window is not available",
                "These files are MD models or trajectories, which open in the "
                "Model window, and this build of FACET does not hold it "
                f"({error}). They can be analysed from the command line: "
                "py -3.11 -m facet.md analyse <file>.\n\n"
                + "\n".join(_group_name(g) for g in groups))
            return []
        opened, refused = [], []
        for paths in groups:
            try:
                window = md_workspace.open_model_window(
                    paths, self, read_options=read_options)
            except Exception as error:        # listed below, never silent
                refused.append(f"{_group_name(paths)}: "
                               f"{type(error).__name__}: {error}")
                continue
            if window is not None:
                opened.append(window)
                offered = md_companions(paths, dropped_with or ())
                setter = getattr(window, "set_companions", None)
                if offered and setter is not None:
                    setter(offered)
        # open_model_window may append to _model_windows itself; each once
        kept: list = []
        for w in self._model_windows + opened:
            if _open(w) and not any(w is k for k in kept):
                kept.append(w)
        self._model_windows = kept
        if opened:
            names = ", ".join(_group_name(g) for g in groups[:3]) + (
                f" and {len(groups) - 3} more" if len(groups) > 3 else "")
            self.statusBar().showMessage(
                f"Opened in a Model window: {names}", 12000)
        if refused:
            QMessageBox.warning(self, "A Model window did not open",
                                "\n".join(refused))
        return opened

    def open_md_paths(self, paths, *, series: bool = False) -> list:
        """Open files chosen as MD models (File > Open MD model…).

        A file an MD reader recognises (:func:`md_format_of`) opens in a
        Model window even where the crystal reader would also read it (a
        one-frame extended XYZ, a POSCAR); any other file goes to
        :meth:`load_many`, which loads a crystal here and lists what neither
        reader takes. With ``series`` the MD files are one model, read in
        the natural order of their names (dump.20 before dump.100);
        otherwise :func:`model_groups` decides.
        """
        from ..core import md_readers

        models, crystals = [], []
        for path in _unique(paths):
            found = md_format_of(path)
            if found is None:
                crystals.append(path)
            else:
                models.append((path, found))
        if crystals:
            self.load_many(crystals)
        if not models:
            return []
        if series and len(models) > 1:
            groups = [sorted((p for p, _ in models),
                             key=md_readers.natural_sort_key)]
        else:
            groups = model_groups(models)
        return self.open_md_models(groups)

    def _choose_md_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open MD models", "", md_file_filter())
        if paths:
            self.open_md_paths(paths)

    def _choose_md_series(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open the files of one MD model", "", md_file_filter())
        if paths:
            self.open_md_paths(paths, series=True)

    def _add_md_menu_items(self) -> None:
        """File > Open MD model… and Open MD series…, after Open folder…, and
        Help > MD models.

        Added after the menus are built, by position, so that the menu code
        above stays as it is; an item that is not found puts the new ones at
        the end of their menu instead.
        """
        open_md = QAction("Open MD &model…", self)
        open_md.setToolTip(
            "An MD model or trajectory, each file in a Model window of its "
            "own, which reads every frame and runs the MD analyses. "
            "Dropping the file on this window does the same.")
        open_md.triggered.connect(self._choose_md_files)
        self.open_md_action = open_md
        open_series = QAction("Open MD se&ries as one model…", self)
        open_series.setToolTip(
            "Several files of one run (dump.0.lammpstrj, dump.1000.lammpstrj, "
            "...) read as one trajectory, in the natural order of their names.")
        open_series.triggered.connect(self._choose_md_series)
        self.open_md_series_action = open_series
        _insert_after(self._menu_bar_action("File"), "Open folder…",
                      (open_md, open_series))

        help_md = QAction("MD models and the Model &window", self)
        help_md.triggered.connect(self._show_md_manual)
        self.md_help_action = help_md
        _insert_after(self._menu_bar_action("Help"), "Keyboard and mouse",
                      (help_md,))

    def _show_md_manual(self) -> None:
        from .help import MD_SECTION

        self._show_manual(MD_SECTION)

    def _menu_bar_action(self, title: str):
        """The menu bar's action whose title (without its &) is ``title``.

        The action, not its menu: the Python wrapper QAction.menu() returns
        is invalidated when the action's own wrapper is collected (PySide6
        6.9.1, measured 2026-10-07; the C++ menu lives on), so the menu is
        taken from the action where it is used, with the action held."""
        for action in self.menuBar().actions():
            if action.text().replace("&", "") == title:
                return action
        return None

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
                overrides=entry.overrides,
                show_vectors=self.show_vectors,
                show_void_cones=(self.show_void_cones and entry is active),
                vector_scale=self.vector_scale,
                ellipsoid_probability=self.probability_box.currentData()
                or 0.50))

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
            self._current_result(), self.project.v_bond,
            params=self.project.params, v_list=self.project.v_list)
        self.diffraction.set_structure(active.structure if active else None)
        self.pdf_panel.set_structure(active.structure if active else None)
        self.exafs_panel.set_context(
            active.structure if active else None,
            active.selected_site if active else None)
        self.planes_panel.set_structure(active.structure if active else None)
        self.overrides_panel.set_context(
            active.structure if active else None,
            active.overrides if active else None)
        self.disorder_panel.set_entry(active)
        self.volume_panel.set_context(
            active.structure if active else None, self.project.params)

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

    def _atom_of_site(self, site_index: int) -> int | None:
        """A drawn atom of a site: the one nearest the middle of the cell.

        A site is drawn as many times as the cell range repeats it, and they
        are not interchangeable for this purpose -- turning about a copy at the
        edge of the block puts the rest of the structure off to one side. The
        one nearest the middle is the copy the eye takes as the site.
        """
        import numpy as np

        scene = self.scene
        if scene is None or scene.n_atoms == 0:
            return None
        which = np.flatnonzero(
            np.asarray(scene.atom_site) == int(site_index))
        if not len(which):
            return None
        centre = self._cell_centre()
        if centre is None:
            centre = scene.center
        offsets = np.linalg.norm(
            scene.atom_position[which] - np.asarray(centre, float), axis=1)
        return int(which[int(np.argmin(offsets))])

    def _center_on_site(self, site_index: int) -> None:
        """Turn the view about a site chosen by name rather than by clicking."""
        index = self._atom_of_site(site_index)
        if index is None:
            self.statusBar().showMessage(
                "That site is not drawn at the moment.", 6000)
            return
        self._center_on_atom(index, self.scene.atom_label[index])

    def _on_include_anions(self, on: bool) -> None:
        """Add the anions to the site list, or take them out again.

        The selected site may be one of the anions that is about to vanish, so
        the selection is dropped rather than left pointing at a site index that
        now means a different site.
        """
        self.project.set_include_anions(on)
        self.site_index = None
        self._rebuild()
        # the list is filled on load and on a threshold change, neither of
        # which this is
        self._fill_site_list()
        self.statusBar().showMessage(
            "Anion sites are analysed as well as the cations. An anion's "
            "bond-valence sum counts the cations around it, so it is the other "
            "half of the same check." if on
            else "Cation sites only.", 9000)

    def _on_site_double_click(self, item) -> None:
        row = self.site_list.row(item)
        if 0 <= row < len(getattr(self, "_rows", [])):
            self._center_on_site(self._rows[row])

    def _site_menu(self, position) -> None:
        item = self.site_list.itemAt(position)
        if item is None:
            return
        menu = self._build_site_menu(self.site_list.row(item))
        if menu is not None:
            menu.exec(self.site_list.viewport().mapToGlobal(position))

    def _build_site_menu(self, row: int):
        """The site list's menu, built but not shown.

        Separated for the same reason as the viewport's: the contents are the
        part worth testing, and exec() would block a test forever.
        """
        from PySide6.QtWidgets import QMenu

        if not (0 <= row < len(getattr(self, "_rows", []))):
            return None
        site_index = self._rows[row]
        index = self._atom_of_site(site_index)
        menu = QMenu(self)
        chrome.apply(menu, self.theme)
        turn = menu.addAction(
            f"Rotate about {self.scene.atom_label[index]}"
            if index is not None else "Rotate about this site")
        turn.setEnabled(index is not None)
        turn.triggered.connect(lambda: self._center_on_site(site_index))
        menu.addAction("Rotate about the cell centre").triggered.connect(
            self._center_on_cell)
        return menu

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
            # and tell the viewport, which otherwise keeps drawing the file
            # that has just been closed -- along with its analysis panels, its
            # cutoff staircase and a camera framed on it.
            self.view.set_scene(None, reframe=False)
            self.explorer.set_result(None)
            self._update_analysis()
            self.site_list.clear()
            self._set_enabled(False)
            self.setWindowTitle(f"{NAME} {__version__}")

    def _on_list_threshold(self, value: float) -> None:
        """The tabulation threshold. Unlike the bond threshold this DOES
        re-run the neighbour search, so it is a spin box rather than a drag."""
        self.project.set_list_threshold(value)
        self._rebuild()

    def _on_style(self) -> None:
        """A style change, and the controls that belong to only one style."""
        ellipsoids = self.style_box.currentData() is Style.ELLIPSOIDS
        self.probability_label.setVisible(ellipsoids)
        self.probability_box.setVisible(ellipsoids)
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
            self._current_result(), self.project.v_bond,
            params=self.project.params, v_list=self.project.v_list)

    def _theme_the_panels(self, theme) -> None:
        """Give the theme to everything in this window that draws with it.

        Separate from :meth:`_apply_theme_everywhere`, which also restyles the
        whole application. That is right when the user picks a theme and wrong
        from a constructor: a QApplication style sheet re-polishes every widget
        of every window that exists, so with several open the cost grows with
        the square of their number. Measured, building five windows in turn:
        0.20 s for the first and 19 s for the fifth, against 0.2 s each with
        the application left alone.
        """
        self.theme = theme
        self.explorer.set_theme(theme)
        self.diffraction.apply_theme(theme)
        self.volume_panel.apply_theme(theme)
        self.pdf_panel.apply_theme(theme)
        self.exafs_panel.apply_theme(theme)
        self.utilities.apply_theme(theme)

    def _apply_theme_everywhere(self, theme) -> None:
        """Hand a theme to every part of the window, the frame included.

        The viewport is not the only thing a theme decides. The menus, docks,
        tables and the hand-written HTML take their colours from it too, or a
        white picture would sit in a grey window and the dark preset would draw
        a dark picture inside a light one.
        """
        self._theme_the_panels(theme)
        app = QApplication.instance()
        if app is not None:
            chrome.apply(app, theme)
        # a style sheet changes a combo box's padding, so the widths measured
        # when the boxes were built are no longer the widths they need
        for box in (self.style_box, self.poly_box,
                    self.atom_label_box, self.bond_label_box):
            box.setMinimumWidth(0)
            self._fit_combo(box)
        self._sync_theme_menu(theme)
        self._update_analysis()

    def _on_theme_cosmetic(self, theme) -> None:
        """Background, fog, ambient occlusion: no vertex data changes."""
        self._apply_theme_everywhere(theme)
        self.view.set_theme(theme)

    def _on_theme_structural(self, theme) -> None:
        """Colour mode, palette, sizes: the vertex arrays must be rebuilt."""
        self._apply_theme_everywhere(theme)
        self._rebuild()
        # The Model windows opened from here take it too. Only here, not on
        # every cosmetic step of a slider: a Model window adopting a theme
        # loads its frame again (ModelWindow.apply_theme).
        for window in self.model_windows:
            adopt = getattr(window, "apply_theme", None)
            if adopt is not None:
                adopt(theme)

    def _sync_theme_menu(self, theme) -> None:
        """Tick the View > Theme entry matching the theme now in use.

        A theme edited by hand in the Appearance tab matches no preset, and then
        nothing is ticked -- which is the honest answer.
        """
        for name, action in getattr(self, "_theme_actions", {}).items():
            action.setChecked(name == theme.name)

    def _choose_theme(self, name: str) -> None:
        """Switch to a named preset, from the menu rather than the panel.

        Delegated to the Appearance panel so that there is one implementation:
        the panel keeps the user's own element colours and colour mode across
        the change, and re-reads its own widgets afterwards.
        """
        index = self.theme_panel.preset_box.findText(name)
        if index < 0:
            return
        self.theme_panel.preset_box.setCurrentIndex(index)
        self.theme_panel._apply_preset()

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

    def _on_isosurface(self, grid, level: float) -> None:
        """Put a level set of a volumetric field into the 3D scene, or take it out.

        Triangulated here rather than in the panel because the scene is the
        window's, and because a level set is geometry: it belongs with the atoms
        and the polyhedra, ordered against them for transparency, not drawn as a
        separate layer on top.
        """
        if self.scene is None:
            return
        if grid is None:
            self.scene.set_isosurface(np.zeros((0, 3)), np.zeros((0, 3), int),
                                      np.zeros((0, 3)))
            self.view.set_scene(self.scene, reframe=False)
            return
        from ..core import volume as volume_mod

        try:
            vertices, faces, normals = volume_mod.isosurface(
                grid, level, step=self.volume_panel.iso_step_value())
        except (ValueError, MemoryError) as error:
            self.statusBar().showMessage(str(error), 9000)
            return
        if not len(faces):
            self.statusBar().showMessage(
                f"no surface at {level:g}: the field does not cross that level",
                9000)
        self.scene.set_isosurface(
            vertices, faces, normals,
            color=self.theme.polyhedron_color, alpha=0.55,
            label=f"{grid.name or 'field'} = {level:g} {grid.units}".strip())
        self.view.set_scene(self.scene, reframe=False)
        self.statusBar().showMessage(
            f"{len(faces)} triangles at {grid.name or 'field'} = {level:g} "
            f"{grid.units}".rstrip(), 9000)

    def _on_disorder(self, description: str = "") -> None:
        """A different disorder configuration is a different structure.

        Not a presentation change: the site list, every coordination number, the
        bond valences and the diffraction pattern all change, so the entry is
        re-analysed and the panels refilled rather than the scene merely rebuilt.
        A configuration choice is therefore not undoable -- undo covers how a
        structure is drawn, and this changes which structure it is.
        """
        entry = self.project.current
        if entry is None:
            return
        entry.invalidate()
        if entry.selected_site is not None and entry.selected_site >= len(
                entry.structure.sites):
            entry.selected_site = None
        entry.overrides.prune(len(entry.structure.atoms),
                              [site.label for site in entry.structure.sites])
        self._fill_site_list()
        self._rebuild(reframe=False)
        self.history.reset(self._snapshot(), description or "chose a configuration")
        self._refresh_history_actions()
        self.statusBar().showMessage(description, 9000)

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
        """Show an atom named somewhere else -- an override row, say.

        It used to draw a ring round it and nothing more, which shows nothing
        at all when the atom is behind the structure or outside the frame: the
        two cases where being shown it is the point. It now brings the atom to
        the middle of the view, where a ring round it means something.
        """
        if self.scene is None or not (0 <= index < self.scene.n_atoms):
            return
        self.view.select_atom(index)
        self._center_on_atom(index, self.scene.atom_label[index])

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
            turn = menu.addAction(f"Rotate about {label}")
            turn.setToolTip(
                "Turn the view about this atom instead of about the middle of "
                "the cell. The picture does not jump; dragging with the right "
                "or middle button moves the centre off it again.")
            turn.triggered.connect(
                lambda _=False, i=atom_index, t=label: self._center_on_atom(i, t))
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

        menu.addAction("Rotate about the cell centre").triggered.connect(
            self._center_on_cell)
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
                f"<p style='color:{chrome.muted_hex(self.theme)}'>"
                f"Click a cation, or choose a site.</p>")
            return

        muted = chrome.muted_hex(self.theme)
        # phi over everything down to the listing threshold, beside phi over the
        # bonded set: the gap between them is how much the index depends on
        # where the cut was put, which is the whole argument of the program.
        phi_listed = ("" if r.phi_listed != r.phi_listed
                      else f" <span style='color:{chrome.muted_hex(self.theme)}'>"
                           f"({r.phi_listed:.3f} to the listing cut)</span>")
        v = self.v_bond
        bonded = [c for c in r.contacts if c.has_valence and c.valence > v]
        cn = len(bonded)
        bvs = sum(c.valence * c.occupancy for c in bonded)
        # against the magnitude: a sum of positive terms is compared with the
        # size of the charge, not its sign. An anion reading 1.05 against -1 is
        # 0.05 over, not 2.05.
        discrepancy = ("" if r.ox is None
                       else f" ({bvs - abs(r.ox):+.2f} against {r.ox:+d})")
        # The R0 uncertainty is systematic: it scales the whole sum rather
        # than averaging out, so a discrepancy smaller than it is not a
        # measurement of anything.
        uncertainty = ("" if r.bvs_uncertainty != r.bvs_uncertainty
                       else f" &plusmn; {r.bvs_uncertainty:.2f}")

        # What the bond topology alone would give for each bond. Computed
        # from an analysis that covers the anions, cached on the entry, and
        # only when a panel asks for it -- see Entry.network.
        entry = self.project.current
        apriori_rows, apriori_reason = (
            self.project.network_for(entry) if entry is not None else ({}, ""))
        split = apriori_rows.get(r.site_index)
        a_priori_of = {}
        if split is not None:
            bonded = [c for c in r.contacts
                      if c.has_valence and c.valence >= v]
            for contact, value in zip(bonded, split.a_priori):
                a_priori_of[id(contact)] = value

        def contact_row(c) -> str:
            ideal = a_priori_of.get(id(c))
            ideal_cell = ("&mdash;" if ideal is None else f"{ideal:.4f}")
            return (
                f"<tr><td>{c.label}</td>"
                f"<td align='right'>{c.distance:.4f}</td>"
                f"<td align='right'>{c.valence:.4f}</td>"
                f"<td align='right' style='color:{muted}'>{ideal_cell}</td>"
                f"<td align='center'>"
                f"{'&#9679;' if c.valence > v else '&#9675;'}</td>"
                f"<td style='color:{muted}'>"
                f"{'' if (c.param and c.param.fitted) else 'est.'}</td></tr>")

        rows = "".join(contact_row(c) for c in r.contacts if c.has_valence)

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
            f"<p style='color:{muted};font-size:11px;margin:3px 0'>{n}</p>"
            for n in r.notes)

        # Gagne & Hawthorne's split of bond-length variation: the part the
        # bond topology requires, and the part it does not account for. Their
        # means over the transition-metal oxides are 0.102 and 0.113 v.u.
        if split is not None:
            apriori_block = (
                f"<tr><td>&Delta;<sub>topol</sub> <span style='color:{muted}'>"
                f"(the topology requires)</span></td>"
                f"<td align='right'>{split.delta_topol:.3f} v.u.</td></tr>"
                f"<tr><td>&Delta;<sub>cryst</sub> <span style='color:{muted}'>"
                f"(it does not account for)</span></td>"
                f"<td align='right'>{split.delta_cryst:.3f} v.u.</td></tr>")
        elif apriori_reason:
            reason = (apriori_reason.replace("&", "&amp;")
                      .replace("<", "&lt;").replace(">", "&gt;"))
            apriori_block = (
                f"<tr><td colspan='2' style='color:{muted};font-size:11px'>"
                f"No a priori bond valences: {reason}</td></tr>")
        else:
            apriori_block = ""

        param = next((c.param for c in r.contacts if c.param), None)
        provenance = ""
        if param is not None:
            provenance = (
                f"<p style='color:{muted};font-size:11px;margin-top:10px'>"
                f"R<sub>0</sub>({param.label}) = {param.r0:.3f} Å, "
                f"b = {param.b:.2f} Å &middot; "
                f"{'fitted' if param.fitted else 'estimated'}<br>{param.source}"
                f"</p>")

        self.analysis.setHtml(f"""
        <h2 style='margin-bottom:0'>{r.label}</h2>
        <p style='color:{muted};margin-top:2px'>
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
          <tr><td>&phi; (stereoactivity)</td>
              <td align='right'>{r.phi:.3f}{phi_listed}</td></tr>
          <tr><td>|&Sigma;v<sub>i</sub><b>&ucirc;</b><sub>i</sub>|
              (bond-valence vector)</td>
              <td align='right'>{r.bvv:.3f} v.u.</td></tr>
          <tr><td>Void cone half-angle</td>
              <td align='right'>{r.void_angle:.1f}&deg;</td></tr>
          <tr><td>Mean bond length</td>
              <td align='right'>{r.shape.get('d_mean') or float('nan'):.4f} Å</td></tr>
          <tr><td>Spread within the polyhedron</td>
              <td align='right'>{r.shape.get('spread') or float('nan'):.4f} Å</td></tr>
          {plateau_row}
          {apriori_block}
        </table>
        {notes}
        <h4 style='margin-bottom:2px'>Contacts</h4>
        <table width='100%' cellspacing='0' cellpadding='2' style='font-size:11px'>
          <tr style='color:{muted}'>
            <th align='left'>atom</th><th align='right'>d / Å</th>
            <th align='right'>v / v.u.</th>
            <th align='right'>a priori</th><th></th><th></th></tr>
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
        self.restore_session(data)

    def restore_session(self, data: dict) -> None:
        """Put a session back, without a file dialog in the way.

        Separated so that what the restore does can be driven and checked; the
        dialog is the part that cannot be. The same split as the context menus,
        for the same reason.
        """
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
            # and to everything that draws with it. Without this the restored
            # theme reached the theme panel's own controls and nothing else:
            # the viewport kept the theme it opened with, and the five panels
            # that build HTML by hand kept none at all, which on Slate or Dark
            # is near-black text on a near-black ground.
            self._apply_theme_everywhere(self.theme)
            self.view.set_theme(self.theme)
            self._sync_theme_menu(self.theme)
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
            if saved and saved.get("disorder") and entry.disorder is not None:
                # before the overrides, because the configuration decides how
                # many atoms there are for an atom override to address
                entry.disorder.apply_dict(saved["disorder"])
                entry.invalidate()
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

    # -- bond-valence parameters -------------------------------------------
    def _load_parameters(self) -> None:
        """Read a published parameter set from wherever the user obtained it.

        FACET ships the small Brese and O'Keeffe table it needs and the
        O'Keeffe-Brese estimator, and nothing larger: the IUCr bvparm file,
        Gagne and Hawthorne's tables and the softBV set each come with their own
        terms, and bundling them in an application that gets passed around a
        research group is not something to do casually. Loading one instead keeps
        the user's own copy the source, and every parameter used afterwards
        carries that file's name as its provenance.
        """
        from ..core import bv_files

        path, _ = QFileDialog.getOpenFileName(
            self, "Load bond-valence parameters", "", bv_files.FILE_FILTER)
        if not path:
            return
        try:
            params, report = bv_files.load(path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Could not read the parameters",
                                str(error))
            return

        self.project.set_parameters(params)
        for entry in self.project.entries:
            entry.invalidate()
        self._fill_site_list()
        self._rebuild()
        QMessageBox.information(
            self, "Parameters loaded",
            f"{report.name}\n\n{report.describe()}\n\n"
            "Every bond valence from here on uses these values, and every "
            "table names them as the source.")
        self.statusBar().showMessage(
            f"bond-valence parameters: {report.name} ({report.pairs} pairs)",
            12000)

    def _reset_parameters(self) -> None:
        self.project.set_parameters(bv.DEFAULT)
        for entry in self.project.entries:
            entry.invalidate()
        self._fill_site_list()
        self._rebuild()
        self.statusBar().showMessage(
            f"bond-valence parameters: {bv.DEFAULT.name}", 9000)

    def _save_parameters(self) -> None:
        """Write out the set in use, so an edited one can be passed on as data."""
        from ..core import bv_files

        path = self._ask("Save the parameters in use",
                         "facet_bv_parameters.json",
                         "FACET parameter set (*.json)")
        if not path:
            return
        try:
            bv_files.write_json(path, self.project.params)
        except OSError as error:
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self._wrote(path)

    def _export_cutoffs(self) -> None:
        """The distance cutoffs the current thresholds imply, as a table."""
        from ..core import bv_report

        if self.structure is None:
            return
        path = self._ask("Save the cutoff table",
                         f"{(self.structure.name or 'structure')}_cutoffs.csv",
                         "CSV (*.csv)")
        if not path:
            return
        table = bv_report.cutoff_table(
            self.structure, self.project.params, v_bond=self.project.v_bond,
            v_list=self.project.v_list, results=self.results)
        try:
            Path(path).write_text(table.as_csv(), encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self._wrote(path)

    def _export_scan(self) -> None:
        """Coordination number against the threshold, for every site."""
        from ..core import bv_report

        if not self.results:
            return
        path = self._ask("Save the threshold scan",
                         f"{(self.structure.name or 'structure')}_scan.csv",
                         "CSV (*.csv)")
        if not path:
            return
        scan = bv_report.threshold_scan(self.results)
        try:
            Path(path).write_text(scan.as_csv(), encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Could not save", str(error))
            return
        self._wrote(path)

    # -- output ------------------------------------------------------------
    def _save_vector(self) -> None:
        """A resolution-free figure, drawn as shapes rather than pixels."""
        from ..gl import vector_export

        if self.scene is None:
            return
        name = (self.structure.name or "structure").replace(" ", "_")
        path, chosen = QFileDialog.getSaveFileName(
            self, "Export a vector figure", f"{name}.svg",
            vector_export.FILE_FILTER)
        if not path:
            return
        if not path.lower().endswith((".svg", ".pdf")):
            path += ".pdf" if "PDF" in (chosen or "") else ".svg"

        answer = QMessageBox.question(
            self, "Background",
            "Use a white background for the figure?\n\n"
            "The screen background is kept otherwise. "
            + " ".join(vector_export.describe_differences()),
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes)
        if answer == QMessageBox.Cancel:
            return
        background = (vector_export.Background.WHITE
                      if answer == QMessageBox.Yes
                      else vector_export.Background.THEME)
        try:
            self.view.save_vector(path, background=background,
                                  title=self.structure.name or "")
        except Exception as error:
            QMessageBox.warning(self, "Could not write the figure", str(error))
            return
        self._wrote(path)

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
        """About, with what this session is actually running.

        The renderer tier belongs here rather than only in a log: on a machine
        without a graphics card FACET falls back to drawing in software, and a
        user comparing two machines' output wants to know which they had.
        """
        from .help import AboutDialog

        caps = self.view.caps
        dialog = AboutDialog(
            self, renderer=caps.describe() if caps else "", theme=self.theme)
        self._about_dialog = dialog          # keep it alive while it is open
        dialog.exec()

    def _show_manual(self, section: str = "") -> None:
        """Open the manual, at ``section`` if one is named.

        Held on the window rather than shown modally, so the manual can stay
        open beside the workspace while its instructions are followed.
        """
        from .help import ManualDialog

        existing = getattr(self, "_manual_dialog", None)
        if existing is not None and existing.isVisible():
            existing.show_section(section)
            existing.raise_()
            existing.activateWindow()
            return
        dialog = ManualDialog(self, section=section, theme=self.theme)
        self._manual_dialog = dialog
        dialog.show()


# ---------------------------------------------------------------------------
# MD models: which files go to a Model window
# ---------------------------------------------------------------------------
#
# What a file IS is decided by its content, by readers.read (MDModelFile) and
# md_readers' sniffs. The names below serve the file dialog's list and the
# drag cursor only. The built-in formats of md_readers.MD_FORMATS carry no
# extensions of their own, so they are listed here, keyed by format name; a
# format the registry adds brings its extensions and stems (FormatSpec), and
# the text ones that list none take theirs from _MODULE_PATTERNS.

_BUILTIN_PATTERNS = {
    "lammps-dump": ("LAMMPS dump", ("*.lammpstrj", "*.lammpsdump", "*.dump",
                                    "*.lammpstrj.gz", "*.dump.gz")),
    "lammps-data": ("LAMMPS data", ("*.data", "*.lmp", "*.data.gz")),
    "extxyz": ("Extended XYZ", ("*.extxyz", "*.xyz", "*.extxyz.gz",
                                "*.xyz.gz")),
    "vasp-xdatcar": ("VASP XDATCAR", ("XDATCAR*",)),
    "dlpoly-config": ("DL_POLY CONFIG", ("CONFIG*", "REVCON*", "CFGMIN*")),
    "dlpoly-history": ("DL_POLY HISTORY", ("HISTORY*",)),
}
_MODULE_LABELS = {
    "dcd": "DCD", "lammps-dump-binary": "LAMMPS binary dump",
    "lammps-dump-yaml": "LAMMPS YAML dump", "atomeye-cfg": "AtomEye CFG",
    "gromacs-xtc": "GROMACS XTC", "ase-traj": "ASE trajectory",
    "gsd": "HOOMD-blue GSD", "amber-netcdf": "AMBER NetCDF",
    "castep-md": "CASTEP .md", "gromacs-gro": "GROMACS .gro",
    "xsf": "XCrySDen XSF", "pdb-models": "PDB trajectory", "imd": "IMD",
    "vasp-poscar": "POSCAR series",
}
_MODULE_PATTERNS = {
    "castep-md": ("*.md", "*.geom"), "gromacs-gro": ("*.gro",),
    "xsf": ("*.xsf", "*.axsf"), "pdb-models": ("*.pdb", "*.ent"),
    "imd": ("*.imd",), "vasp-poscar": ("POSCAR*", "CONTCAR*", "*.vasp"),
}
# A file under one of these names opens in a Model window even when the
# crystal reader refuses it outright (an empty or cut dump): the Model window
# then states what the MD reader finds, in the MD reader's words.
_MD_ONLY_SUFFIXES = (".lammpstrj", ".lammpsdump", ".dump")


@lru_cache(maxsize=1)
def md_filter_entries() -> tuple[tuple[str, tuple[str, ...], str], ...]:
    """(label, name patterns, format name) of every MD format FACET reads:
    the built-in ones, then each format module's (``md_readers.
    format_specs``), in md_readers' order."""
    from ..core import md_readers

    out = [(*_BUILTIN_PATTERNS[name], name) for name in md_readers.MD_FORMATS
           if name in _BUILTIN_PATTERNS]
    for spec in md_readers.format_specs():
        patterns = tuple(dict.fromkeys(
            [f"*{e}" for e in spec.extensions]
            + [f"{s}*" for s in spec.stems]
            + list(_MODULE_PATTERNS.get(spec.name, ()))))
        if patterns:
            label = _MODULE_LABELS.get(spec.name) \
                or spec.description.split(" (")[0]
            out.append((label, patterns, spec.name))
    return tuple(out)


def md_file_filter() -> str:
    """The file dialog filter of File > Open MD model…: every MD format
    FACET reads, together and one by one, then All files."""
    entries = md_filter_entries()
    every = dict.fromkeys(p for _label, patterns, _f in entries
                          for p in patterns)
    parts = [f"MD models and trajectories ({' '.join(every)})"]
    parts += [f"{label} ({' '.join(patterns)})"
              for label, patterns, _f in entries]
    parts.append("All files (*)")
    return ";;".join(parts)


def _md_name_format(path) -> str | None:
    """The MD format whose name patterns the file's name matches, or None.
    Extensions compare without regard to case, stems (XDATCAR, CONFIG)
    with it, as VASP and DL_POLY name their files."""
    name = Path(path).name
    lower = name.lower()
    # the built-in names first: they need no import of the format modules,
    # which costs about 0.6 s the first time (measured 2026-10-07), so the
    # first drag of a dump does not wait for it
    def entries():
        for fmt, (label, patterns) in _BUILTIN_PATTERNS.items():
            yield label, patterns, fmt
        yield from md_filter_entries()

    for _label, patterns, file_format in entries():
        for pattern in patterns:
            if pattern.startswith("*."):
                if fnmatch.fnmatchcase(lower, pattern):
                    return file_format
            elif fnmatch.fnmatchcase(name, pattern):
                return file_format
    return None


def md_droppable(path) -> bool:
    """Whether a dragged file may be an MD model: its name (an MD format's
    extension or stem) or its first bytes (``md_readers.sniff_md``,
    ``binary_md_format``) say so. Where it goes once dropped is decided by
    :func:`read_or_route`."""
    if _md_name_format(path) is not None:
        return True
    p = Path(path)
    try:
        if not p.is_file():
            return False
    except OSError:
        return False
    from ..core import md_readers

    return (md_readers.sniff_md(p) is not None
            or md_readers.binary_md_format(p) is not None)


def _md_despite_refusal(path) -> str | None:
    """The MD format of a file ``readers.read`` refused with plain
    UnsupportedFormat that the Model window opens all the same, or None:

    * an XYZ trajectory, or one frame of LAMMPS's 'dump xyz' or CP2K's XMOL
      trajectory, that states no box (no ``Lattice=``): the MD reader reads
      it given the box, which the Model window asks for (lines ending in CR
      alone are left to the crystal reader's refusal, which says to convert
      them);
    * a file under a format module's extension or an MD-only extension
      (:data:`_MD_ONLY_SUFFIXES`) that the crystal reader refused (an empty
      or cut file), whose fault the MD reader then states.
    """
    from ..core import md_readers

    p = Path(path)
    try:
        if not p.is_file():
            return None
    except OSError:
        return None
    if md_readers.sniff_md(p) == "extxyz" \
            and not md_readers.xyz_has_lattice(p) \
            and not md_readers.cr_only_line_endings(p) \
            and (md_readers.is_multiframe_xyz(p)
                 or md_readers.plain_xyz_writer(p) is not None):
        return "extxyz"
    name = p.name.lower()
    suffix = Path(name[:-3] if name.endswith(".gz") else name).suffix
    if not suffix:
        return None
    spec = next((s for s in md_readers.format_specs()
                 if suffix in s.extensions), None)
    if spec is not None:
        return spec.name
    return "lammps-dump" if suffix in _MD_ONLY_SUFFIXES else None


def read_or_route(path) -> tuple[str, object]:
    """Where a file opens: ('structure', Structure) for the crystal window,
    ('md', format name) for a Model window, ('failed', why) otherwise.

    ``readers.read`` decides first, so a file is read once: an MDModelFile
    goes to a Model window with the format it names, and a refusal that
    :func:`_md_despite_refusal` recognises does too. Every other exception
    is a failure with its text, as ``Project.add_files`` records it.
    """
    try:
        return "structure", readers.read(path)
    except readers.MDModelFile as error:
        return "md", error.file_format or "md"
    except readers.UnsupportedFormat as error:
        found = _md_despite_refusal(path)
        return ("md", found) if found is not None else ("failed", str(error))
    except Exception as error:
        return "failed", str(error)


def md_format_of(path) -> str | None:
    """The MD format a file chosen as an MD model is read as, or None.

    By its first bytes (a binary format, then ``md_readers.sniff_md``), then
    a format module's extension or stem on a file that is not empty
    (``md_readers.named_md_format``), then an MD-only extension
    (:data:`_MD_ONLY_SUFFIXES`). Unlike :func:`read_or_route` this does not
    ask the crystal reader: a one-frame extended XYZ or a POSCAR chosen as a
    model opens as a one-frame model.
    """
    from ..core import md_readers

    p = Path(path)
    try:
        if not p.is_file():
            return None
        size = p.stat().st_size
    except OSError:
        return None
    spec = md_readers.binary_md_format(p)
    if spec is not None:
        return spec.name
    found = md_readers.sniff_md(p)
    if found is not None:
        return found
    spec = md_readers.named_md_format(p)
    if spec is not None and size > 0:
        return spec.name
    name = p.name.lower()
    suffix = Path(name[:-3] if name.endswith(".gz") else name).suffix
    return "lammps-dump" if suffix in _MD_ONLY_SUFFIXES else None


def _name_shape(name: str) -> str:
    """A file name with its digit runs (and a .gz suffix) taken out, so the
    files of one series compare equal: dump.150.cfg -> dump.#.cfg."""
    if name.lower().endswith(".gz"):
        name = name[:-3]
    return re.sub(r"\d+", "#", name)


def _unique(paths) -> list[str]:
    """The paths in their order, each once (compared as the OS compares)."""
    seen, out = set(), []
    for path in paths:
        key = os.path.normcase(os.path.abspath(str(path)))
        if key not in seen:
            seen.add(key)
            out.append(str(path))
    return out


def model_groups(models) -> list:
    """One entry per model, from (path, MD format) pairs: a path, or a list
    of the paths of one model in natural order of their names.

    Files are one model only when their format writes one snapshot per file
    (a format module with ``read_series``: LAMMPS 'dump cfg', a POSCAR
    series) and they share a directory and a name up to its digits
    (dump.2000.cfg, dump.2010.cfg, ...). Every other file is a model of its
    own: two dumps named glass_300K and glass_600K are two runs, and reading
    them as one trajectory would average them. File > Open MD series reads
    files as one model when that is what is meant.
    """
    from ..core import md_readers

    per_snapshot = {s.name for s in md_readers.format_specs()
                    if s.read_series is not None}
    groups: dict[tuple, list[str]] = {}
    for path, file_format in models:
        p = Path(path)
        if file_format in per_snapshot:
            key = (os.path.normcase(str(p.parent.resolve())),
                   _name_shape(p.name), file_format)
        else:
            key = (os.path.normcase(os.path.abspath(str(p))),)
        members = groups.setdefault(key, [])
        if str(path) not in members:
            members.append(str(path))
    return [members[0] if len(members) == 1
            else sorted(members, key=md_readers.natural_sort_key)
            for members in groups.values()]


def md_companions(group, models) -> list[str]:
    """The LAMMPS data files among ``models`` ((path, MD format) pairs
    opened together) that sit in the folder of ``group``'s first file and
    are not ``group``'s own: a dump's window offers each as the file whose
    Masses name its types, or as its topology or box."""
    paths = [group] if not isinstance(group, (list, tuple)) else list(group)
    if not paths:
        return []
    folder = os.path.normcase(str(Path(paths[0]).resolve().parent))
    own = {os.path.normcase(os.path.abspath(str(p))) for p in paths}
    out = []
    for path, file_format in models:
        key = os.path.normcase(os.path.abspath(str(path)))
        if file_format != "lammps-data" or key in own:
            continue
        if os.path.normcase(str(Path(path).resolve().parent)) == folder:
            out.append(str(path))
    return out


def _group_name(group) -> str:
    if isinstance(group, (list, tuple)):
        if len(group) == 1:
            return Path(group[0]).name
        return (f"{Path(group[0]).name} .. {Path(group[-1]).name} "
                f"({len(group)} files)")
    return Path(group).name


def _open(window) -> bool:
    """Whether a Model window still exists and is shown."""
    try:
        import shiboken6

        return bool(shiboken6.isValid(window) and window.isVisible())
    except Exception:              # a window deleted under us is not open
        return False


def _insert_after(bar_action, text: str, actions) -> None:
    """Put ``actions`` after the item whose text (without its mnemonic &) is
    ``text`` in the menu of ``bar_action`` (a menu bar action, held while its
    menu is used), in order; at the end when there is none."""
    menu = bar_action.menu() if bar_action is not None else None
    if menu is None:
        return
    items = menu.actions()
    at = next((i for i, a in enumerate(items)
               if a.text().replace("&", "") == text), None)
    before = items[at + 1] if at is not None and at + 1 < len(items) else None
    for action in actions:
        if before is None:
            menu.addAction(action)
        else:
            menu.insertAction(before, action)


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtGui import QSurfaceFormat

    from ..gl import caps as caps_mod

    argv = list(sys.argv if argv is None else argv)
    QSurfaceFormat.setDefaultFormat(caps_mod.request_format())
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName(NAME)
    # the same chrome the real entry point applies, so running this module
    # directly does not give a differently coloured window
    chrome.apply(app, theme_mod.Theme())

    window = PreviewWindow(argv[1] if len(argv) > 1 else None)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
