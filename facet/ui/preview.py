"""A runnable preview of the Structure workspace.

Not the finished interface -- it is the vertical slice that proves the parts fit
together, and the thing to put in front of someone to find out whether the
cutoff explorer reads the way it is meant to.

What it demonstrates:

* the 3D view, on whichever tier this machine supports
* clicking an atom to select its site
* **the cutoff slider**: drag it and the coordination number, bond-valence sum,
  phi and the drawn bonds all follow, with no neighbour search and no
  re-analysis. That interaction is the application's whole argument, and it has
  to feel instant or the argument does not land.
* the plateau readout, which says whether the coordination number on screen is
  a property of the structure or of the threshold
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QFont, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..core import bv, cif, coordination
from ..gl.scene import Style, build_scene
from ..gl.view import StructureView
from ..version import NAME, __version__

# The slider works in log-valence, because that is the axis on which a plateau
# width equals a distance gap. A linear slider would spend most of its travel in
# a region where nothing changes.
V_MIN, V_MAX = 0.005, 0.5
STEPS = 1000


def _slider_to_valence(pos: int) -> float:
    t = pos / STEPS
    return float(10 ** (math.log10(V_MIN) + t * (math.log10(V_MAX) - math.log10(V_MIN))))


def _valence_to_slider(v: float) -> int:
    t = ((math.log10(max(v, V_MIN)) - math.log10(V_MIN))
         / (math.log10(V_MAX) - math.log10(V_MIN)))
    return int(round(t * STEPS))


class PreviewWindow(QMainWindow):
    def __init__(self, path: str | None = None):
        super().__init__()
        self.setWindowTitle(f"{NAME} {__version__} — structure preview")
        self.resize(1180, 820)

        self.structure = None
        self.results: list[coordination.SiteResult] = []
        self.scene = None
        self.site_index: int | None = None
        self.v_bond = bv.V_BOND_DEFAULT

        self.view = StructureView()
        self.view.sitePicked.connect(self._on_site_picked)
        self.view.measured.connect(lambda t: self.statusBar().showMessage(t, 8000))
        self.view.ready.connect(self._on_ready)

        self.readout = QLabel("Open a structure to begin.")
        self.readout.setTextFormat(Qt.RichText)
        self.readout.setWordWrap(True)
        self.readout.setMinimumWidth(330)
        self.readout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        f = QFont(self.readout.font())
        f.setPointSizeF(f.pointSizeF() + 0.5)
        self.readout.setFont(f)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, STEPS)
        self.slider.setValue(_valence_to_slider(self.v_bond))
        self.slider.valueChanged.connect(self._on_threshold)
        self.slider.setEnabled(False)

        self.threshold_label = QLabel()
        self.threshold_label.setMinimumWidth(250)

        self._build_layout()
        self._build_menu()
        self.setStatusBar(QStatusBar())

        if path:
            self.load(path)

    # -- layout ------------------------------------------------------------
    def _build_layout(self) -> None:
        centre = QWidget()
        outer = QVBoxLayout(centre)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.view, 1)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.addWidget(self.readout)
        side_layout.addStretch(1)

        self.style_box = QComboBox()
        for s in (Style.BALL_AND_STICK, Style.SPACE_FILLING, Style.STICK):
            self.style_box.addItem(s.value, s)
        self.style_box.currentIndexChanged.connect(self._rebuild_scene)
        side_layout.addWidget(QLabel("Style"))
        side_layout.addWidget(self.style_box)

        self.labels_box = QCheckBox("Atom labels")
        self.labels_box.toggled.connect(
            lambda on: self.view.set_labels(on, elements_only=True))
        side_layout.addWidget(self.labels_box)

        self.ortho_box = QCheckBox("Orthographic")
        self.ortho_box.toggled.connect(self.view.set_projection)
        side_layout.addWidget(self.ortho_box)

        side.setFixedWidth(360)
        top.addWidget(side)
        outer.addLayout(top, 1)

        strip = QWidget()
        strip_layout = QHBoxLayout(strip)
        strip_layout.setContentsMargins(12, 8, 12, 10)
        strip_layout.addWidget(QLabel("Bond threshold"))
        strip_layout.addWidget(self.slider, 1)
        strip_layout.addWidget(self.threshold_label)
        strip.setMaximumHeight(52)
        outer.addWidget(strip)

        self.setCentralWidget(centre)
        self._update_threshold_label()

    def _build_menu(self) -> None:
        openf = QAction("&Open structure…", self)
        openf.setShortcut(QKeySequence.Open)
        openf.triggered.connect(self._choose_file)

        save = QAction("&Save image…", self)
        save.setShortcut(QKeySequence.Save)
        save.triggered.connect(self._save_image)

        reset = QAction("&Reset view", self)
        reset.setShortcut("R")
        reset.triggered.connect(self.view.reset_view)

        about = QAction("&About", self)
        about.triggered.connect(self._about)

        m = self.menuBar().addMenu("&File")
        m.addAction(openf)
        m.addAction(save)
        v = self.menuBar().addMenu("&View")
        v.addAction(reset)
        h = self.menuBar().addMenu("&Help")
        h.addAction(about)

    # -- loading -----------------------------------------------------------
    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open a structure", "", "Crystal structures (*.cif);;All files (*)")
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        try:
            self.structure = cif.read(path)
            self.results = coordination.analyse_structure(self.structure)
        except Exception as exc:
            QMessageBox.warning(self, "Could not read the structure",
                                f"{Path(path).name}\n\n{exc}")
            return

        self.site_index = self.results[0].site_index if self.results else None
        self._rebuild_scene(reframe=True)
        self.slider.setEnabled(True)
        self.setWindowTitle(
            f"{NAME} {__version__} — {Path(path).name}  ·  "
            f"{self.structure.spacegroup_hm or 'unknown symmetry'}")
        for note in self.structure.notes:
            self.statusBar().showMessage(note, 12000)

    def _rebuild_scene(self, *_args, reframe: bool = False) -> None:
        if self.structure is None:
            return
        style = self.style_box.currentData() or Style.BALL_AND_STICK
        self.scene = build_scene(self.structure, self.results, style=style,
                                 v_bond=self.v_bond,
                                 polyhedron_site=self.site_index)
        self.view.set_scene(self.scene, reframe=reframe)
        self._update_readout()

    # -- interaction -------------------------------------------------------
    def _on_ready(self, caps) -> None:
        self.statusBar().showMessage(f"Renderer: {caps.describe()}", 15000)

    def _on_site_picked(self, site_index: int) -> None:
        if site_index < 0 or self.structure is None:
            return
        if not any(r.site_index == site_index for r in self.results):
            # an anion was clicked; nothing is analysed for it yet
            self.statusBar().showMessage(
                f"{self.structure.sites[site_index].label} is an anion site — "
                "cation sites carry the coordination analysis", 5000)
            return
        self.site_index = site_index
        self._rebuild_scene()

    def _on_threshold(self, pos: int) -> None:
        self.v_bond = _slider_to_valence(pos)
        self.view.set_threshold(self.v_bond)     # restyle only: no re-analysis
        self._update_threshold_label()
        self._update_readout()

    def _update_threshold_label(self) -> None:
        text = f"{self.v_bond:.4f} v.u."
        result = self._current_result()
        if result is not None:
            for c in result.contacts:
                if c.param is not None:
                    text += f"   ({c.param.label} ≡ {c.param.distance_for(self.v_bond):.3f} Å)"
                    break
        self.threshold_label.setText(text)

    def _current_result(self):
        if self.site_index is None:
            return None
        return next((r for r in self.results if r.site_index == self.site_index), None)

    # -- readout -----------------------------------------------------------
    def _update_readout(self) -> None:
        r = self._current_result()
        if r is None:
            self.readout.setText("Click a cation to analyse its site.")
            return

        v = self.v_bond
        cn = r.cn_at(v)
        bonded = [c for c in r.contacts if c.has_valence and c.valence > v]
        bvs = sum(c.valence * c.occupancy for c in bonded)
        plateau = next((p for p in r.plateaus if p.contains(v)), None)

        if plateau is None:
            verdict = ("<span style='color:#e0a04a'>on a step edge</span> — "
                       "the smallest change of threshold changes this number")
        elif plateau.width_decades >= 0.5:
            verdict = (f"<span style='color:#6fbf73'>stable</span> over "
                       f"{plateau.width_decades:.2f} decades of threshold "
                       f"(a {plateau.width_angstrom:.3f} Å gap) — a property of "
                       f"the structure")
        else:
            verdict = (f"<span style='color:#e0a04a'>narrow</span>: stable over "
                       f"only {plateau.width_decades:.2f} decades "
                       f"({plateau.width_angstrom:.3f} Å) — this is a choice of "
                       f"cutoff more than a property of the site")

        rows = "".join(
            f"<tr><td>{c.label}</td><td align='right'>{c.distance:.3f}</td>"
            f"<td align='right'>{c.valence:.4f}</td>"
            f"<td>{'●' if c.valence > v else '○'}</td></tr>"
            for c in r.contacts[:14] if c.has_valence)

        alt = "  ".join(f"CN {p.cn}: {p.width_decades:.2f}"
                        for p in r.plateaus if p.width_decades > 0.15)

        html = f"""
        <h3 style='margin-bottom:2px'>{r.label} &nbsp;
           <span style='font-weight:normal;color:#9aa3b2'>{r.element}
           {'' if r.ox is None else f'{r.ox:+d}'} ·
           {r.wyckoff or '?'} · {r.site_symmetry or '?'}</span></h3>
        <table width='100%' cellspacing='0' cellpadding='2'>
          <tr><td>Coordination number</td><td align='right'><b>{cn}</b></td></tr>
          <tr><td>Bond-valence sum</td><td align='right'>{bvs:.2f} v.u.</td></tr>
          <tr><td>ECoN (Hoppe)</td><td align='right'>{r.cn_ecoN:.2f}</td></tr>
          <tr><td>Max-gap split</td><td align='right'>{r.cn_gap}</td></tr>
          <tr><td>φ (stereoactivity)</td><td align='right'>{r.phi:.3f}</td></tr>
          <tr><td>Void cone</td><td align='right'>{r.void_angle:.1f}°</td></tr>
        </table>
        <p style='color:#c8cfdb'>{verdict}</p>
        <p style='color:#9aa3b2;font-size:11px'>Plateau widths (decades): {alt}</p>
        <table width='100%' cellspacing='0' cellpadding='1'
               style='font-size:11px;color:#c8cfdb'>
          <tr><th align='left'>contact</th><th align='right'>d / Å</th>
              <th align='right'>v / v.u.</th><th></th></tr>
          {rows}
        </table>
        """
        self.readout.setText(html)

    # -- output ------------------------------------------------------------
    def _save_image(self) -> None:
        if self.scene is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save image", "structure.png", "PNG image (*.png)")
        if not path:
            return
        image = self.view.grab_image(1800, 1350, supersample=2)
        image.save(path)
        self.statusBar().showMessage(f"Wrote {path}", 6000)

    def _about(self) -> None:
        caps = self.view.caps
        QMessageBox.information(
            self, f"About {NAME}",
            f"<b>{NAME} {__version__}</b><br>"
            f"Coordination analysis, cut by bond valence.<br><br>"
            f"Renderer: {caps.describe() if caps else 'not initialised'}")


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtWidgets import QApplication
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
