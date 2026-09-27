"""Colour and appearance controls.

Everything visible is adjustable, and the whole set saves to a file so a group
can share one look and a set of figures can match. Per-element colours are
edited by clicking a swatch; the rest is presets, ramps and sliders.

The colour *mode* control is the one that is not merely cosmetic: colouring by
bond-valence sum, coordination number or phi puts the analysis onto the
structure, so a stereoactive site is visible in the picture and not only in the
table beside it.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import theme as theme_mod
from . import chrome


def _swatch(rgb, size: int = 18) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)
    c = QColor()
    c.setRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    p.setBrush(c)
    p.setPen(QColor(0, 0, 0, 90))
    p.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)
    p.end()
    return QIcon(pix)


def _to_qcolor(rgb) -> QColor:
    c = QColor()
    c.setRgbF(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return c


def _from_qcolor(c: QColor):
    return (c.redF(), c.greenF(), c.blueF())


class ThemePanel(QWidget):
    """Edits a :class:`~facet.core.theme.Theme` in place."""

    themeChanged = Signal(object)          # cosmetic: no scene rebuild needed
    rebuildNeeded = Signal(object)         # colour mode or scale: rebuild
    viewingChanged = Signal(object, float, bool)   # stereo mode, separation, sort

    def __init__(self, theme: theme_mod.Theme | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.theme = theme or theme_mod.Theme()
        self._elements: list[str] = []
        self._element_buttons: dict[str, QToolButton] = {}
        self._loading = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        inner = QWidget()
        self._layout = QVBoxLayout(inner)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(10)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._build_presets()
        self._build_mode()
        self._build_elements()
        self._build_scene_colors()
        self._build_sizes()
        self._build_quality()
        self._build_viewing()
        self._layout.addStretch(1)
        self._build_io()
        self._reload()
        chrome.fit_every_combo(self)

    # -- sections ----------------------------------------------------------
    def _group(self, title: str) -> QFormLayout:
        box = QGroupBox(title)
        form = QFormLayout(box)
        form.setLabelAlignment(Qt.AlignLeft)
        form.setContentsMargins(10, 8, 10, 8)
        self._layout.addWidget(box)
        return form

    def _build_presets(self) -> None:
        form = self._group("Preset")
        self.preset_box = QComboBox()
        for name in theme_mod.PRESETS:
            self.preset_box.addItem(name)
        self.preset_box.activated.connect(self._apply_preset)
        form.addRow(self.preset_box)

        self.palette_box = QComboBox()
        for name in theme_mod.PALETTES:
            self.palette_box.addItem(name)
        self.palette_box.activated.connect(self._apply_palette)
        form.addRow("Element palette", self.palette_box)

    def _build_mode(self) -> None:
        form = self._group("Colour by")
        self.mode_box = QComboBox()
        for mode in theme_mod.ColorMode:
            self.mode_box.addItem(mode.value, mode)
        self.mode_box.activated.connect(self._apply_mode)
        form.addRow(self.mode_box)

        self.bond_mode_box = QComboBox()
        for mode in theme_mod.BondColorMode:
            self.bond_mode_box.addItem(mode.value, mode)
        self.bond_mode_box.activated.connect(self._apply_bond_mode)
        form.addRow("Bonds", self.bond_mode_box)

        self.autoscale = QCheckBox("Autoscale the colour range")
        self.autoscale.setChecked(True)
        self.autoscale.toggled.connect(self._apply_scale)
        form.addRow(self.autoscale)

        row = QHBoxLayout()
        self.scale_min = QDoubleSpinBox()
        self.scale_max = QDoubleSpinBox()
        for s in (self.scale_min, self.scale_max):
            s.setRange(-99.0, 99.0)
            s.setDecimals(2)
            s.setSingleStep(0.1)
            s.setEnabled(False)
            s.valueChanged.connect(self._apply_scale)
        row.addWidget(self.scale_min)
        row.addWidget(QLabel("to"))
        row.addWidget(self.scale_max)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Range", holder)

    def _build_elements(self) -> None:
        box = QGroupBox("Element colours")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 8, 10, 8)
        hint = QLabel("Click a swatch to change it. "
                      "Right-click to restore the palette default.")
        hint.setWordWrap(True)
        f = QFont(hint.font())
        f.setPointSizeF(max(7.5, f.pointSizeF() - 1.0))
        hint.setFont(f)
        v.addWidget(hint)
        self.element_grid = QGridLayout()
        self.element_grid.setSpacing(4)
        v.addLayout(self.element_grid)
        self._layout.addWidget(box)

    def _build_scene_colors(self) -> None:
        form = self._group("Scene")
        self._scene_buttons: dict[str, QToolButton] = {}
        for attr, label in (("background", "Background"),
                            ("cell_color", "Unit cell"),
                            ("polyhedron_color", "Polyhedra"),
                            ("subthreshold_color", "Sub-threshold contacts"),
                            ("selection_color", "Selection"),
                            ("label_color", "Labels")):
            btn = QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setText("change")
            btn.clicked.connect(lambda _=False, a=attr: self._pick_scene_color(a))
            self._scene_buttons[attr] = btn
            form.addRow(label, btn)

        self.poly_alpha = QSlider(Qt.Horizontal)
        self.poly_alpha.setRange(0, 100)
        self.poly_alpha.valueChanged.connect(self._apply_poly_alpha)
        form.addRow("Polyhedron opacity", self.poly_alpha)

    def _build_sizes(self) -> None:
        form = self._group("Sizes")
        self.atom_scale = QSlider(Qt.Horizontal)
        self.atom_scale.setRange(20, 260)
        self.atom_scale.valueChanged.connect(self._apply_sizes)
        form.addRow("Atom size", self.atom_scale)

        self.bond_scale = QSlider(Qt.Horizontal)
        self.bond_scale.setRange(20, 300)
        self.bond_scale.valueChanged.connect(self._apply_sizes)
        form.addRow("Bond thickness", self.bond_scale)

    def _build_quality(self) -> None:
        form = self._group("Rendering")
        self.fog = QSlider(Qt.Horizontal)
        self.fog.setRange(0, 100)
        self.fog.valueChanged.connect(self._apply_quality)
        form.addRow("Depth cueing", self.fog)

        self.ao = QCheckBox("Ambient occlusion")
        self.ao.toggled.connect(self._apply_quality)
        form.addRow(self.ao)

        self.outline = QCheckBox("Contact outlines")
        self.outline.toggled.connect(self._apply_quality)
        form.addRow(self.outline)

    def _build_viewing(self) -> None:
        """Stereo, and how transparency is ordered.

        Not part of the theme: a theme is a palette and a set of sizes, and it is
        meant to be saved and shared. How you happen to be looking at the screen
        is not. So these emit their own signal and are not written to the theme
        file.
        """
        from ..gl import stereo as stereo_mod

        form = self._group("Viewing")
        self.stereo_box = QComboBox()
        for mode in stereo_mod.Mode:
            self.stereo_box.addItem(mode.value, mode)
        self.stereo_box.setToolTip(
            "Renders the structure twice, once per eye. Red-cyan needs the "
            "glasses; side by side needs a stereoscope or a knack for parallel "
            "viewing, and the crossed pair is the same thing for people who "
            "find crossing easier.")
        self.stereo_box.activated.connect(self._apply_viewing)
        form.addRow("Stereo", self.stereo_box)

        self.stereo_separation = QSlider(Qt.Horizontal)
        self.stereo_separation.setRange(2, 40)          # tenths of a degree
        self.stereo_separation.setValue(12)
        self.stereo_separation.setToolTip(
            "The half-angle between the eyes, in tenths of a degree. An angle "
            "rather than a distance, so the depth impression does not change as "
            "you zoom. About 1.2 degrees suits a screen at arm's length.")
        self.stereo_separation.valueChanged.connect(self._apply_viewing)
        form.addRow("Eye separation", self.stereo_separation)

        self.sort_transparency = QCheckBox("Order transparent surfaces by depth")
        self.sort_transparency.setChecked(True)
        self.sort_transparency.setToolTip(
            "Alpha blending depends on the order it is done in, so without this "
            "one of two overlapping polyhedra looks solid and the other looks "
            "absent depending on which was uploaded last. Sorting costs nothing "
            "while the camera is still.")
        self.sort_transparency.toggled.connect(self._apply_viewing)
        form.addRow(self.sort_transparency)

    def _apply_viewing(self, *_) -> None:
        if self._loading:
            return
        self.viewingChanged.emit(self.stereo_box.currentData(),
                                 self.stereo_separation.value() / 10.0,
                                 self.sort_transparency.isChecked())

    def _build_io(self) -> None:
        row = QHBoxLayout()
        save = QPushButton("Save theme…")
        save.clicked.connect(self._save)
        load = QPushButton("Load theme…")
        load.clicked.connect(self._load)
        reset = QPushButton("Reset colours")
        reset.clicked.connect(self._reset)
        row.addWidget(save)
        row.addWidget(load)
        row.addWidget(reset)
        holder = QWidget()
        holder.setLayout(row)
        self._layout.addWidget(holder)

    # -- element grid ------------------------------------------------------
    def set_elements(self, symbols: list[str]) -> None:
        """Show only the elements the loaded structure actually contains."""
        self._elements = list(dict.fromkeys(symbols))
        while self.element_grid.count():
            item = self.element_grid.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._element_buttons.clear()

        for i, sym in enumerate(self._elements):
            btn = QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            btn.setText(sym)
            btn.setMinimumWidth(66)
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.clicked.connect(lambda _=False, s=sym: self._pick_element(s))
            btn.customContextMenuRequested.connect(
                lambda _p, s=sym: self._reset_element(s))
            self._element_buttons[sym] = btn
            self.element_grid.addWidget(btn, i // 3, i % 3)
        self._refresh_swatches()

    def _refresh_swatches(self) -> None:
        for sym, btn in self._element_buttons.items():
            btn.setIcon(_swatch(self.theme.element_color(sym)))
            overridden = sym in self.theme.overrides
            f = QFont(btn.font())
            f.setBold(overridden)
            btn.setFont(f)
            btn.setToolTip(f"{sym} — "
                           + ("custom" if overridden else "palette default"))
        for attr, btn in self._scene_buttons.items():
            btn.setIcon(_swatch(getattr(self.theme, attr)))

    # -- handlers ----------------------------------------------------------
    def _pick_element(self, symbol: str) -> None:
        c = QColorDialog.getColor(_to_qcolor(self.theme.element_color(symbol)),
                                  self, f"Colour for {symbol}")
        if c.isValid():
            self.theme.set_element_color(symbol, _from_qcolor(c))
            self._refresh_swatches()
            self.rebuildNeeded.emit(self.theme)

    def _reset_element(self, symbol: str) -> None:
        self.theme.clear_element_color(symbol)
        self._refresh_swatches()
        self.rebuildNeeded.emit(self.theme)

    def _pick_scene_color(self, attr: str) -> None:
        c = QColorDialog.getColor(_to_qcolor(getattr(self.theme, attr)), self,
                                  attr.replace("_", " "))
        if not c.isValid():
            return
        setattr(self.theme, attr, _from_qcolor(c))
        self._refresh_swatches()
        # the sub-threshold colour is baked into the bond vertex colours
        if attr in ("subthreshold_color", "polyhedron_color"):
            self.rebuildNeeded.emit(self.theme)
        else:
            self.themeChanged.emit(self.theme)

    def _apply_preset(self) -> None:
        name = self.preset_box.currentText()
        factory = theme_mod.PRESETS.get(name)
        if factory is None:
            return
        overrides = dict(self.theme.overrides)
        mode = self.theme.color_mode
        self.theme = factory()
        self.theme.overrides = overrides          # keep the user's own colours
        self.theme.color_mode = mode
        self._reload()
        self.rebuildNeeded.emit(self.theme)

    def _apply_palette(self) -> None:
        self.theme.palette_name = self.palette_box.currentText()
        self._refresh_swatches()
        self.rebuildNeeded.emit(self.theme)

    def _apply_mode(self) -> None:
        if self._loading:
            return
        self.theme.color_mode = self.mode_box.currentData()
        self.rebuildNeeded.emit(self.theme)

    def _apply_bond_mode(self) -> None:
        if self._loading:
            return
        self.theme.bond_color_mode = self.bond_mode_box.currentData()
        self.rebuildNeeded.emit(self.theme)

    def _apply_scale(self) -> None:
        if self._loading:
            return
        auto = self.autoscale.isChecked()
        self.scale_min.setEnabled(not auto)
        self.scale_max.setEnabled(not auto)
        if auto:
            self.theme.scale_min = self.theme.scale_max = None
        else:
            self.theme.scale_min = self.scale_min.value()
            self.theme.scale_max = self.scale_max.value()
        self.rebuildNeeded.emit(self.theme)

    def _apply_poly_alpha(self, value: int) -> None:
        if self._loading:
            return
        self.theme.polyhedron_alpha = value / 100.0
        self.rebuildNeeded.emit(self.theme)

    def _apply_sizes(self) -> None:
        if self._loading:
            return
        self.theme.atom_scale = self.atom_scale.value() / 100.0
        self.theme.bond_scale = self.bond_scale.value() / 100.0
        self.rebuildNeeded.emit(self.theme)

    def _apply_quality(self) -> None:
        if self._loading:
            return
        self.theme.fog_amount = self.fog.value() / 100.0
        self.theme.ambient_occlusion = self.ao.isChecked()
        self.theme.outlines = self.outline.isChecked()
        self.themeChanged.emit(self.theme)

    def _reset(self) -> None:
        self.theme.reset_overrides()
        self._refresh_swatches()
        self.rebuildNeeded.emit(self.theme)

    def _save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save theme", "facet-theme.json",
                                              "FACET theme (*.json)")
        if path:
            self.theme.save(path)

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load theme", "",
                                              "FACET theme (*.json)")
        if not path:
            return
        try:
            self.theme = theme_mod.Theme.load(Path(path))
        except Exception:
            return
        self._reload()
        self.rebuildNeeded.emit(self.theme)

    # -- state -------------------------------------------------------------
    def _reload(self) -> None:
        self._loading = True
        t = self.theme
        self.palette_box.setCurrentText(t.palette_name)
        i = self.mode_box.findData(t.color_mode)
        if i >= 0:
            self.mode_box.setCurrentIndex(i)
        i = self.bond_mode_box.findData(t.bond_color_mode)
        if i >= 0:
            self.bond_mode_box.setCurrentIndex(i)
        auto = t.scale_min is None or t.scale_max is None
        self.autoscale.setChecked(auto)
        self.scale_min.setEnabled(not auto)
        self.scale_max.setEnabled(not auto)
        if not auto:
            self.scale_min.setValue(t.scale_min)
            self.scale_max.setValue(t.scale_max)
        self.poly_alpha.setValue(int(t.polyhedron_alpha * 100))
        self.atom_scale.setValue(int(t.atom_scale * 100))
        self.bond_scale.setValue(int(t.bond_scale * 100))
        self.fog.setValue(int(t.fog_amount * 100))
        self.ao.setChecked(t.ambient_occlusion)
        self.outline.setChecked(t.outlines)
        self._refresh_swatches()
        self._loading = False
