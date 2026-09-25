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
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..core import bv, cif, coordination, theme as theme_mod
from ..gl.scene import Style, build_scene
from ..gl.view import StructureView
from ..version import NAME, __version__
from .cutoff_explorer import CutoffExplorer
from .theme_panel import ThemePanel


class PolyhedraMode:
    NONE = "none"
    SELECTED = "selected site"
    ALL = "all cations"


class PreviewWindow(QMainWindow):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.setWindowTitle(f"{NAME} {__version__}")
        self.resize(1420, 900)

        self.structure = None
        self.results: list[coordination.SiteResult] = []
        self.scene = None
        self.site_index: int | None = None
        self.v_bond = bv.V_BOND_DEFAULT
        self.theme = theme_mod.Theme()
        self.poly_mode = PolyhedraMode.SELECTED

        self.view = StructureView()
        self.view.set_theme(self.theme)
        self.view.sitePicked.connect(self._on_site_picked)
        self.view.measured.connect(lambda t: self.statusBar().showMessage(t, 9000))
        self.view.ready.connect(self._on_ready)

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

        self._build_layout()
        self._build_menu()
        self.setStatusBar(QStatusBar())
        self._set_enabled(False)
        self.setAcceptDrops(True)

        if path:
            self.load(path)

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        sites = QDockWidget("Sites", self)
        sites.setWidget(self.site_list)
        sites.setFeatures(QDockWidget.DockWidgetMovable
                          | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, sites)
        sites.setMinimumWidth(190)

        tabs = QTabWidget()
        tabs.addTab(self.analysis, "Site")
        tabs.addTab(self.theme_panel, "Appearance")
        tabs.setMinimumWidth(380)

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

        self.label_check = QCheckBox("Labels")
        self.label_check.toggled.connect(
            lambda on: self.view.set_labels(on, elements_only=True))
        row.addWidget(self.label_check)

        self.ortho_check = QCheckBox("Orthographic")
        self.ortho_check.toggled.connect(self.view.set_projection)
        row.addWidget(self.ortho_check)

        row.addStretch(1)
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

        m = self.menuBar().addMenu("&File")
        m.addAction(openf)
        m.addAction(save)
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
                  self.label_check, self.ortho_check, *self.range_boxes):
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
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a structure", "",
            "Crystal structures (*.cif);;All files (*)")
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        try:
            structure = cif.read(path)
            results = coordination.analyse_structure(structure)
        except Exception as exc:
            QMessageBox.warning(self, "Could not read the structure",
                                f"{Path(path).name}\n\n{exc}")
            return

        self.structure = structure
        self.results = results
        self.site_index = results[0].site_index if results else None
        self.theme_panel.set_elements(structure.elements_present)
        self._fill_site_list()
        self._set_enabled(True)
        self._rebuild(reframe=True)
        self.setWindowTitle(
            f"{NAME} {__version__} — {Path(path).name}  ·  "
            f"{structure.spacegroup_hm or 'unknown symmetry'}")
        if structure.notes:
            self.statusBar().showMessage(structure.notes[0], 14000)

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
        if self.poly_mode == PolyhedraMode.NONE:
            return []
        if self.poly_mode == PolyhedraMode.ALL:
            return [r.site_index for r in self.results]
        return [self.site_index] if self.site_index is not None else []

    def _rebuild(self, reframe: bool = False) -> None:
        if self.structure is None:
            return
        style = self.style_box.currentData() or Style.BALL_AND_STICK
        cell_range = tuple(b.value() for b in self.range_boxes)
        self.scene = build_scene(
            self.structure, self.results, style=style, v_bond=self.v_bond,
            polyhedron_sites=self._polyhedron_sites(),
            show_cell=self.cell_check.isChecked(),
            cell_range=cell_range, theme=self.theme)
        self.view.set_scene(self.scene, reframe=reframe)
        self.view.set_theme(self.theme)
        self.explorer.set_result(self._current_result())
        self.explorer.set_threshold(self.v_bond)
        self._update_analysis()

    def _current_result(self):
        if self.site_index is None:
            return None
        return next((r for r in self.results
                     if r.site_index == self.site_index), None)

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

    def _on_poly_mode(self) -> None:
        self.poly_mode = self.poly_box.currentData()
        self._rebuild()

    def _on_threshold(self, v: float) -> None:
        """The whole point: restyle, do not re-analyse."""
        self.v_bond = float(v)
        self.view.set_threshold(self.v_bond)
        self._update_analysis()

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

        rows = "".join(
            f"<tr><td>{c.label}</td>"
            f"<td align='right'>{c.distance:.4f}</td>"
            f"<td align='right'>{c.valence:.4f}</td>"
            f"<td align='center'>{'&#9679;' if c.valence > v else '&#9675;'}</td>"
            f"<td style='color:#8a93a3'>"
            f"{'' if (c.param and c.param.fitted) else 'est.'}</td></tr>"
            for c in r.contacts if c.has_valence)

        warnings = "".join(
            f"<p style='color:#e0a04a;margin:4px 0'>{w}</p>" for w in r.warnings)

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
              <td align='right'>{bvs:.2f} v.u.{discrepancy}</td></tr>
          <tr><td>ECoN (Hoppe)</td><td align='right'>{r.cn_ecoN:.2f}</td></tr>
          <tr><td>Maximum-gap split</td><td align='right'>{r.cn_gap}</td></tr>
          <tr><td>&phi; (stereoactivity)</td><td align='right'>{r.phi:.3f}</td></tr>
          <tr><td>Void cone half-angle</td>
              <td align='right'>{r.void_angle:.1f}&deg;</td></tr>
          <tr><td>Mean bond length</td>
              <td align='right'>{r.shape.get('d_mean') or float('nan'):.4f} Å</td></tr>
          <tr><td>Spread within the polyhedron</td>
              <td align='right'>{r.shape.get('spread') or float('nan'):.4f} Å</td></tr>
        </table>
        {warnings}
        <h4 style='margin-bottom:2px'>Contacts</h4>
        <table width='100%' cellspacing='0' cellpadding='2' style='font-size:11px'>
          <tr style='color:#8a93a3'>
            <th align='left'>atom</th><th align='right'>d / Å</th>
            <th align='right'>v / v.u.</th><th></th><th></th></tr>
          {rows}
        </table>
        {provenance}
        """)

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
