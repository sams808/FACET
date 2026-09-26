"""A 2D section through a volumetric field, drawn with QPainter.

The map VESTA calls a 2D data display: a plane cut through a charge density, an
ELF, a bond-valence sum or a bond-valence energy landscape, shown as a colour
map with contour lines over it.

Drawn with QPainter for the same reasons as the diffraction plot -- no plotting
library in the executable, works with no graphics card, and SVG and PDF export
come out of the same render path as the screen. The colour map is built as a
QImage and scaled; the contours are drawn as lines, so they stay sharp in a
vector export while the map behind them is a resampled image, which is what a
contour figure wants.

Colour maps are chosen for what they have to do here. Viridis and magma are
perceptually uniform, so an apparent feature in the map is a feature in the data
rather than an artefact of the colours -- which matters when the thing being
looked at is whether a bond-valence basin closes. The diverging map is for a
difference field, where zero has to be visible.
"""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

# Perceptually uniform maps, as a handful of anchor colours interpolated in RGB.
# Not the full 256-entry tables: the anchors reproduce them closely enough that
# the ordering and the lightness ramp are preserved, which is the property that
# matters, and it keeps this file readable.
COLORMAPS: dict[str, tuple] = {
    "viridis": ((0.267, 0.005, 0.329), (0.283, 0.141, 0.458),
                (0.254, 0.265, 0.530), (0.207, 0.372, 0.553),
                (0.164, 0.471, 0.558), (0.128, 0.567, 0.551),
                (0.135, 0.659, 0.518), (0.267, 0.749, 0.441),
                (0.478, 0.821, 0.318), (0.741, 0.873, 0.150),
                (0.993, 0.906, 0.144)),
    "magma": ((0.001, 0.000, 0.014), (0.116, 0.066, 0.276),
              (0.305, 0.071, 0.483), (0.492, 0.109, 0.502),
              (0.681, 0.165, 0.450), (0.855, 0.267, 0.353),
              (0.962, 0.451, 0.325), (0.995, 0.652, 0.462),
              (0.996, 0.831, 0.640), (0.987, 0.991, 0.750)),
    "grey": ((0.02, 0.02, 0.03), (0.98, 0.98, 0.98)),
    "difference": ((0.020, 0.188, 0.380), (0.270, 0.459, 0.706),
                   (0.639, 0.745, 0.859), (0.969, 0.969, 0.969),
                   (0.957, 0.647, 0.510), (0.792, 0.306, 0.243),
                   (0.404, 0.000, 0.121)),
}


def sample_colormap(name: str, t):
    """Colours for values in 0..1. Returns an (n, 3) array of 0..255 integers."""
    anchors = np.array(COLORMAPS.get(name, COLORMAPS["viridis"]), float)
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    positions = np.linspace(0.0, 1.0, len(anchors))
    out = np.empty(t.shape + (3,), float)
    for channel in range(3):
        out[..., channel] = np.interp(t, positions, anchors[:, channel])
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


class SectionView(QWidget):
    """A colour map with contours, axes in angstrom, and a readout."""

    hovered = Signal(float, float, float)      # u, v, value

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.values: np.ndarray | None = None
        self.extent = (-5.0, 5.0, -5.0, 5.0)
        self.levels: np.ndarray = np.zeros(0)
        self.colormap = "viridis"
        self.log_scale = False
        self.show_contours = True
        self.show_map = True
        self.title = ""
        self.units = ""
        self.footnote = ""
        self.u_label = "u / Å"
        self.v_label = "v / Å"
        self.vmin: float | None = None
        self.vmax: float | None = None
        self.markers: list[tuple[float, float, str]] = []

        self.background = QColor(22, 24, 28)
        self.text = QColor(225, 229, 236)
        self.axis = QColor(120, 128, 140)
        self.faint = QColor(70, 76, 86)
        self.contour_color = QColor(255, 255, 255)
        self._cursor = None

    # -- content -----------------------------------------------------------
    def set_section(self, values, extent, levels=None, units: str = "",
                    title: str = "", footnote: str = "") -> None:
        self.values = None if values is None else np.asarray(values, float)
        self.extent = tuple(float(x) for x in extent)
        self.levels = (np.zeros(0) if levels is None
                       else np.atleast_1d(np.asarray(levels, float)))
        self.units = units
        self.title = title
        self.footnote = footnote
        self.update()

    def set_markers(self, markers) -> None:
        """Points to mark on the plane: atoms lying in or near it."""
        self.markers = list(markers or [])
        self.update()

    def apply_theme(self, theme) -> None:
        def to_color(rgb, lighten=0.0):
            r, g, b = (min(1.0, c + lighten) for c in rgb)
            return QColor(int(r * 255), int(g * 255), int(b * 255))

        self.background = to_color(theme.background)
        self.text = to_color(theme.label_color)
        self.axis = to_color(theme.cell_color, 0.10)
        self.faint = to_color(theme.cell_color, -0.10)
        self.update()

    @property
    def data_range(self) -> tuple[float, float]:
        if self.values is None or not self.values.size:
            return 0.0, 1.0
        finite = self.values[np.isfinite(self.values)]
        if not finite.size:
            return 0.0, 1.0
        low = self.vmin if self.vmin is not None else float(finite.min())
        high = self.vmax if self.vmax is not None else float(finite.max())
        if high <= low:
            high = low + 1.0
        return low, high

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self.render_to(painter, self.width(), self.height())
        painter.end()

    def render_to(self, painter: QPainter, width: float, height: float,
                  for_export: bool = False) -> None:
        font = QFont(painter.font())
        if for_export:
            font.setPointSizeF(9.0)
        painter.setFont(font)
        metrics = QFontMetricsF(font)

        ink = QColor(20, 20, 20) if for_export else self.text
        axis = QColor(90, 90, 90) if for_export else self.axis
        ground = QColor(255, 255, 255) if for_export else self.background
        painter.fillRect(QRectF(0, 0, width, height), ground)

        rect, bar = self._layout(width, height, metrics)
        if self.values is None or self.values.size == 0:
            painter.setPen(QPen(self.faint))
            painter.drawText(QRectF(0, 0, width, height), Qt.AlignCenter,
                             "No section to show.")
            return

        if self.title:
            painter.setPen(QPen(ink))
            painter.drawText(QRectF(rect.left(), 2, rect.width(),
                                    metrics.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, self.title)

        if self.show_map:
            painter.drawImage(rect, self._image())
        if self.show_contours and self.levels.size:
            self._draw_contours(painter, rect)
        self._draw_markers(painter, rect, metrics, ink)
        self._draw_axes(painter, rect, metrics, axis, ink)
        self._draw_colorbar(painter, bar, metrics, axis, ink)

        if self.footnote:
            painter.setPen(QPen(self.faint))
            small = QFont(font)
            small.setPointSizeF(max(6.5, font.pointSizeF() - 1.5))
            painter.setFont(small)
            painter.drawText(
                QRectF(rect.left(), height - metrics.height() - 2,
                       width - rect.left(), metrics.height()),
                Qt.AlignLeft | Qt.AlignVCenter, self.footnote)
            painter.setFont(font)

        if self._cursor is not None and not for_export:
            self._draw_crosshair(painter, rect, metrics)

    def _layout(self, width, height, metrics):
        left = metrics.horizontalAdvance("-00.0") + 12
        bar_width = 16.0
        bar_gap = metrics.horizontalAdvance("-0.000e+00") + 14
        top = 8.0 + (metrics.height() + 4 if self.title else 0)
        bottom = metrics.height() * 2 + 12 + (metrics.height() + 2
                                              if self.footnote else 0)
        right = bar_width + bar_gap + 10

        # a section is square in the data, so keep the aspect ratio
        available_w = max(width - left - right, 10.0)
        available_h = max(height - top - bottom, 10.0)
        umin, umax, vmin, vmax = self.extent
        aspect = (vmax - vmin) / max(umax - umin, 1e-9)
        side_w = min(available_w, available_h / max(aspect, 1e-9))
        side_h = side_w * aspect
        rect = QRectF(left, top + (available_h - side_h) / 2.0, side_w, side_h)
        bar = QRectF(rect.right() + 12, rect.top(), bar_width, rect.height())
        return rect, bar

    def _normalised(self) -> np.ndarray:
        low, high = self.data_range
        values = np.array(self.values, float)
        if self.log_scale:
            floor = max(low, 1e-12)
            values = np.log10(np.clip(values, floor, None))
            low, high = math.log10(floor), math.log10(max(high, floor * 10))
        with np.errstate(invalid="ignore"):
            t = (values - low) / (high - low)
        return np.nan_to_num(np.clip(t, 0.0, 1.0), nan=0.0)

    def _image(self) -> QImage:
        rgb = sample_colormap(self.colormap, self._normalised())
        rows, cols = rgb.shape[:2]
        rgba = np.empty((rows, cols, 4), np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = 255
        # v increases upwards in the data and downwards on screen
        rgba = np.ascontiguousarray(rgba[::-1])
        image = QImage(rgba.data, cols, rows, 4 * cols, QImage.Format_RGBA8888)
        return image.copy()

    def _to_device(self, rect: QRectF, u, v):
        umin, umax, vmin, vmax = self.extent
        x = rect.left() + (np.asarray(u, float) - umin) / (umax - umin) * rect.width()
        y = rect.bottom() - (np.asarray(v, float) - vmin) / (vmax - vmin) * rect.height()
        return x, y

    def _to_data(self, rect: QRectF, x: float, y: float):
        umin, umax, vmin, vmax = self.extent
        u = umin + (x - rect.left()) / max(rect.width(), 1e-9) * (umax - umin)
        v = vmin + (rect.bottom() - y) / max(rect.height(), 1e-9) * (vmax - vmin)
        return u, v

    def _draw_contours(self, painter: QPainter, rect: QRectF) -> None:
        from ..core import volume

        lines = volume.contour_lines(self.values, self.extent, self.levels)
        low, high = self.data_range
        painter.save()
        painter.setClipRect(rect)
        for level, segments in lines.items():
            if not len(segments):
                continue
            # A contour is drawn on top of the colour map, and at this level the
            # colour underneath it is known exactly -- it is the map sampled at
            # the same fraction. So take the ink from there rather than from a
            # fixed grey, which disappeared on a pale map and on a white ground.
            fraction = min(max((level - low) / max(high - low, 1e-12), 0.0), 1.0)
            under = sample_colormap(self.colormap, fraction).reshape(3)
            luma = (0.2126 * under[0] + 0.7152 * under[1]
                    + 0.0722 * under[2]) / 255.0
            shade = 255 if luma < 0.5 else 0
            pen = QPen(QColor(shade, shade, shade, 215), 0.9)
            pen.setCosmetic(True)
            painter.setPen(pen)
            x0, y0 = self._to_device(rect, segments[:, 0, 0], segments[:, 0, 1])
            x1, y1 = self._to_device(rect, segments[:, 1, 0], segments[:, 1, 1])
            for a, b, c, d in zip(x0, y0, x1, y1):
                painter.drawLine(QPointF(float(a), float(b)),
                                 QPointF(float(c), float(d)))
        painter.restore()

    def _draw_markers(self, painter, rect, metrics, ink) -> None:
        if not self.markers:
            return
        painter.save()
        painter.setClipRect(rect.adjusted(-30, -30, 30, 30))
        # A marker lands anywhere on the colour map, so neither a light nor a
        # dark ink is safe on its own: draw both, the dark one as an outline
        # around the light one. That reads on every map and on both grounds.
        dark = QColor(0, 0, 0, 190)
        light = QColor(255, 255, 255, 235)
        for u, v, label in self.markers:
            x, y = self._to_device(rect, u, v)
            x, y = float(x), float(y)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(dark, 2.6))
            painter.drawEllipse(QPointF(x, y), 4.0, 4.0)
            painter.setPen(QPen(light, 1.2))
            painter.drawEllipse(QPointF(x, y), 4.0, 4.0)
            if label:
                at = QPointF(x + 6, y - 4)
                painter.setPen(QPen(dark))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    painter.drawText(QPointF(at.x() + dx, at.y() + dy), label)
                painter.setPen(QPen(light))
                painter.drawText(at, label)
        painter.setBrush(Qt.NoBrush)
        painter.restore()

    def _draw_axes(self, painter, rect, metrics, axis, ink) -> None:
        pen = QPen(axis, 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

        umin, umax, vmin, vmax = self.extent
        painter.setPen(QPen(ink))
        for value in np.linspace(umin, umax, 5):
            x, _ = self._to_device(rect, value, vmin)
            painter.drawText(QRectF(float(x) - 30, rect.bottom() + 2, 60,
                                    metrics.height()),
                             Qt.AlignHCenter | Qt.AlignVCenter, f"{value:.1f}")
        for value in np.linspace(vmin, vmax, 5):
            _, y = self._to_device(rect, umin, value)
            painter.drawText(QRectF(0, float(y) - metrics.height() / 2,
                                    rect.left() - 5, metrics.height()),
                             Qt.AlignRight | Qt.AlignVCenter, f"{value:.1f}")
        painter.drawText(
            QRectF(rect.left(), rect.bottom() + 2 + metrics.height(),
                   rect.width(), metrics.height()),
            Qt.AlignHCenter | Qt.AlignVCenter, self.u_label)
        painter.save()
        painter.translate(metrics.height() * 0.3, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QRectF(-rect.height() / 2, -metrics.height() / 2,
                                rect.height(), metrics.height()),
                         Qt.AlignHCenter | Qt.AlignVCenter, self.v_label)
        painter.restore()

    def _draw_colorbar(self, painter, bar: QRectF, metrics, axis, ink) -> None:
        steps = 128
        t = np.linspace(1.0, 0.0, steps)
        rgb = sample_colormap(self.colormap, t)
        rgba = np.empty((steps, 1, 4), np.uint8)
        rgba[:, 0, :3] = rgb
        rgba[..., 3] = 255
        rgba = np.ascontiguousarray(rgba)
        strip = QImage(rgba.data, 1, steps, 4, QImage.Format_RGBA8888).copy()
        painter.drawImage(bar, strip)

        pen = QPen(axis, 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(bar)

        low, high = self.data_range
        painter.setPen(QPen(ink))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            if self.log_scale:
                floor = max(low, 1e-12)
                value = 10.0 ** (math.log10(floor) + fraction
                                 * (math.log10(max(high, floor * 10))
                                    - math.log10(floor)))
            else:
                value = low + fraction * (high - low)
            y = bar.bottom() - fraction * bar.height()
            text = (f"{value:.3g}" if abs(value) < 1e4 else f"{value:.2e}")
            painter.drawText(QRectF(bar.right() + 4, y - metrics.height() / 2,
                                    120, metrics.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, text)
        if self.units:
            painter.drawText(QRectF(bar.left() - 20, bar.top()
                                    - metrics.height() - 2, 160,
                                    metrics.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, self.units)

    def _draw_crosshair(self, painter, rect, metrics) -> None:
        x, y = float(self._cursor.x()), float(self._cursor.y())
        if not rect.contains(QPointF(x, y)):
            return
        pen = QPen(QColor(255, 214, 92), 0.8, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))

        u, v = self._to_data(rect, x, y)
        value = self.value_at(u, v)
        text = f"{u:.2f}, {v:.2f} → {value:.4g} {self.units}".strip()
        width = metrics.horizontalAdvance(text) + 10
        box = QRectF(min(x + 6, rect.right() - width),
                     max(y - metrics.height() - 4, rect.top() + 1),
                     width, metrics.height() + 2)
        plate = QColor(self.background)
        plate.setAlpha(215)
        painter.setBrush(plate)
        painter.setPen(QPen(self.faint, 0.8))
        painter.drawRect(box)
        painter.setPen(QPen(QColor(255, 214, 92)))
        painter.drawText(box, Qt.AlignCenter, text)
        painter.setBrush(Qt.NoBrush)

    # -- reading -----------------------------------------------------------
    def value_at(self, u: float, v: float) -> float:
        """The field at a point on the plane, by bilinear interpolation."""
        if self.values is None or self.values.size == 0:
            return float("nan")
        umin, umax, vmin, vmax = self.extent
        rows, cols = self.values.shape
        fu = (u - umin) / max(umax - umin, 1e-12) * (cols - 1)
        fv = (v - vmin) / max(vmax - vmin, 1e-12) * (rows - 1)
        if not (0 <= fu <= cols - 1 and 0 <= fv <= rows - 1):
            return float("nan")
        i0, j0 = int(math.floor(fu)), int(math.floor(fv))
        i1 = min(i0 + 1, cols - 1)
        j1 = min(j0 + 1, rows - 1)
        du, dv = fu - i0, fv - j0
        return float(
            self.values[j0, i0] * (1 - du) * (1 - dv)
            + self.values[j0, i1] * du * (1 - dv)
            + self.values[j1, i0] * (1 - du) * dv
            + self.values[j1, i1] * du * dv)

    def mouseMoveEvent(self, event) -> None:
        self._cursor = event.position().toPoint()
        from PySide6.QtGui import QFontMetricsF as _FM

        rect, _ = self._layout(self.width(), self.height(),
                               _FM(QFont(self.font())))
        u, v = self._to_data(rect, event.position().x(), event.position().y())
        self.hovered.emit(u, v, self.value_at(u, v))
        self.update()

    def leaveEvent(self, event) -> None:
        self._cursor = None
        self.update()

    # -- export ------------------------------------------------------------
    def save_svg(self, path, width: int = 760, height: int = 620) -> None:
        from PySide6.QtSvg import QSvgGenerator

        generator = QSvgGenerator()
        generator.setFileName(str(path))
        generator.setSize(QSize(width, height))
        generator.setViewBox(QRectF(0, 0, width, height))
        generator.setTitle(self.title or "FACET section")
        painter = QPainter(generator)
        try:
            self.render_to(painter, width, height, for_export=True)
        finally:
            painter.end()

    def save_pdf(self, path, width_mm: float = 150.0,
                 height_mm: float = 125.0) -> None:
        from PySide6.QtCore import QMarginsF, QSizeF
        from PySide6.QtGui import QPageSize, QPdfWriter

        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QSizeF(width_mm, height_mm),
                                     QPageSize.Millimeter))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        writer.setResolution(600)
        writer.setTitle(self.title or "FACET section")
        painter = QPainter(writer)
        try:
            scale = writer.resolution() / 25.4
            self.render_to(painter, width_mm * scale, height_mm * scale,
                           for_export=True)
        finally:
            painter.end()
