"""A line plot drawn with QPainter.

FACET does not bundle a plotting library. That is partly size -- matplotlib is
excluded from the executable along with pandas and PIL -- and partly the
promise that the application runs on a machine with no graphics card: QPainter
is the same software rasteriser the BASIC render tier already falls back to.

Drawing the plot ourselves buys two other things. Vector export is free,
because a QSvgGenerator or a QPdfWriter is just another QPaintDevice: the same
:meth:`Plot.render_to` produces the screen, an SVG and a PDF from one code
path. And the axis logic can be told about crystallography -- reflection ticks
under the axis, a difference curve on its own offset baseline -- without
fighting a general-purpose library.

Nothing here interprets the data. It draws what it is given.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QWidget


@dataclass
class Series:
    """One curve. ``x`` and ``y`` must be the same length."""

    x: np.ndarray
    y: np.ndarray
    label: str = ""
    color: tuple[int, int, int] = (120, 190, 240)
    width: float = 1.4
    dashed: bool = False
    filled: bool = False           # fill down to the baseline
    offset: float = 0.0            # shifted vertically, for a difference curve
    visible: bool = True

    def __post_init__(self):
        self.x = np.asarray(self.x, float)
        self.y = np.asarray(self.y, float)
        if len(self.x) != len(self.y):
            raise ValueError(
                f"series {self.label!r}: {len(self.x)} x values but "
                f"{len(self.y)} y values")


@dataclass
class Ticks:
    """Vertical marks under the axis: reflection positions, for instance."""

    x: np.ndarray
    labels: list[str] = field(default_factory=list)
    color: tuple[int, int, int] = (150, 160, 175)
    row: int = 0                   # which band of ticks, counting downwards
    title: str = ""

    def __post_init__(self):
        self.x = np.asarray(self.x, float)


class Plot(QWidget):
    """An x-y plot with pan, zoom and a crosshair readout.

    Interaction, deliberately close to what a diffractometer's own software
    does: drag to pan, wheel to zoom about the cursor, wheel with Ctrl to zoom
    the intensity axis only, double-click to fit everything, and a crosshair
    that reports the value under the pointer.
    """

    picked = Signal(float, float)          # data coordinates of a click
    hovered = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.series: list[Series] = []
        self.ticks: list[Ticks] = []
        self.x_label = ""
        self.y_label = ""
        self.title = ""
        self.footnote = ""

        # colours, overwritten by apply_theme
        self.background = QColor(22, 24, 28)
        self.panel = QColor(28, 31, 36)
        self.axis = QColor(120, 128, 140)
        self.text = QColor(225, 229, 236)
        self.faint = QColor(70, 76, 86)
        self.crosshair = QColor(255, 214, 92)

        self._x_range: tuple[float, float] | None = None
        self._y_range: tuple[float, float] | None = None
        self._auto = True
        self._cursor: QPoint | None = None
        self._drag: QPoint | None = None
        self._drag_range: tuple[tuple[float, float], tuple[float, float]] | None = None
        self._tick_band = 0.0

    # -- content -----------------------------------------------------------
    def set_series(self, series: list[Series], keep_view: bool = False) -> None:
        self.series = list(series)
        if not keep_view:
            self.fit()
        self.update()

    def set_ticks(self, ticks: list[Ticks]) -> None:
        self.ticks = list(ticks)
        self.update()

    def set_labels(self, x: str = "", y: str = "", title: str = "",
                   footnote: str = "") -> None:
        self.x_label, self.y_label = x, y
        self.title, self.footnote = title, footnote
        self.update()

    def apply_theme(self, theme) -> None:
        """Follow the application's palette, so the plot matches the 3D view."""
        def to_color(rgb, lighten=0.0):
            r, g, b = (min(1.0, c + lighten) for c in rgb)
            return QColor(int(r * 255), int(g * 255), int(b * 255))

        self.background = to_color(theme.background)
        self.panel = to_color(theme.background, 0.035)
        self.text = to_color(theme.label_color)
        self.axis = to_color(theme.cell_color, 0.10)
        self.faint = to_color(theme.cell_color, -0.10)
        self.crosshair = to_color(theme.selection_color)
        self.update()

    # -- the view ----------------------------------------------------------
    def fit(self) -> None:
        """Frame every visible point, with a little headroom."""
        xs, ys = [], []
        for s in self.series:
            if not s.visible or not len(s.x):
                continue
            xs.append((float(np.min(s.x)), float(np.max(s.x))))
            ys.append((float(np.min(s.y + s.offset)),
                       float(np.max(s.y + s.offset))))
        for t in self.ticks:
            if len(t.x):
                xs.append((float(np.min(t.x)), float(np.max(t.x))))
        if not xs:
            self._x_range, self._y_range, self._auto = None, None, True
            return
        x0 = min(a for a, _ in xs)
        x1 = max(b for _, b in xs)
        if ys:
            y0 = min(a for a, _ in ys)
            y1 = max(b for _, b in ys)
        else:
            y0, y1 = 0.0, 1.0
        if x1 <= x0:
            x0, x1 = x0 - 1.0, x1 + 1.0
        span = y1 - y0
        if span <= 0:
            span = max(abs(y1), 1.0)
        self._x_range = (x0, x1)
        self._y_range = (y0 - 0.04 * span, y1 + 0.08 * span)
        self._auto = True
        self.update()

    def set_x_range(self, low: float, high: float) -> None:
        if high > low:
            self._x_range = (float(low), float(high))
            self._auto = False
            self.update()

    def set_y_range(self, low: float, high: float) -> None:
        if high > low:
            self._y_range = (float(low), float(high))
            self._auto = False
            self.update()

    @property
    def x_range(self):
        return self._x_range or (0.0, 1.0)

    @property
    def y_range(self):
        return self._y_range or (0.0, 1.0)

    # -- geometry ----------------------------------------------------------
    def _metrics(self, device_width: float, device_height: float,
                 font: QFont):
        """Margins and the plotting rectangle, in device coordinates."""
        fm = QFontMetricsF(font)
        left = fm.horizontalAdvance("00000") + 14
        right = 10.0
        top = 10.0 + (fm.height() + 4 if self.title else 0)
        bottom = fm.height() + 14
        if self.x_label:
            bottom += fm.height()
        if self.footnote:
            bottom += fm.height() + 2
        rows = max((t.row for t in self.ticks), default=-1) + 1
        self._tick_band = rows * (fm.height() * 0.62 + 3)
        bottom += self._tick_band
        rect = QRectF(left, top,
                      max(device_width - left - right, 10.0),
                      max(device_height - top - bottom, 10.0))
        return rect, fm

    def _to_device(self, rect: QRectF, x, y):
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        sx = rect.width() / (x1 - x0) if x1 > x0 else 1.0
        sy = rect.height() / (y1 - y0) if y1 > y0 else 1.0
        return (rect.left() + (np.asarray(x, float) - x0) * sx,
                rect.bottom() - (np.asarray(y, float) - y0) * sy)

    def _to_data(self, rect: QRectF, px: float, py: float):
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        fx = (px - rect.left()) / rect.width() if rect.width() else 0.0
        fy = (rect.bottom() - py) / rect.height() if rect.height() else 0.0
        return x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)

    # -- painting ----------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        self.render_to(painter, self.width(), self.height())
        painter.end()

    def render_to(self, painter: QPainter, width: float, height: float,
                  for_export: bool = False) -> None:
        """Draw the whole plot onto any paint device.

        The screen, an SVG file and a PDF page all come through here, which is
        the only way an exported figure is guaranteed to be the figure that was
        on screen. ``for_export`` drops the crosshair and uses a light ground,
        because a figure destined for a paper is printed on white.
        """
        font = QFont(painter.font())
        if for_export:
            font.setPointSizeF(9.0)
        painter.setFont(font)
        rect, fm = self._metrics(width, height, font)

        background = QColor(255, 255, 255) if for_export else self.background
        ink = QColor(20, 20, 20) if for_export else self.text
        axis = QColor(90, 90, 90) if for_export else self.axis
        faint = QColor(215, 215, 215) if for_export else self.faint

        painter.fillRect(QRectF(0, 0, width, height), background)
        if not for_export:
            painter.fillRect(rect, self.panel)

        if self.title:
            painter.setPen(QPen(ink))
            painter.drawText(QRectF(rect.left(), 2, rect.width(), fm.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, self.title)

        self._draw_grid(painter, rect, fm, axis, faint, ink)
        painter.save()
        painter.setClipRect(rect)
        for s in self.series:
            if s.visible and len(s.x):
                self._draw_series(painter, rect, s)
        painter.restore()
        self._draw_ticks(painter, rect, fm, ink)
        self._draw_frame(painter, rect, axis)
        self._draw_legend(painter, rect, fm, ink, background)

        if self.footnote:
            painter.setPen(QPen(faint if for_export else self.faint))
            small = QFont(font)
            small.setPointSizeF(max(6.5, font.pointSizeF() - 1.5))
            painter.setFont(small)
            painter.drawText(
                QRectF(rect.left(), height - fm.height() - 2,
                       rect.width(), fm.height()),
                Qt.AlignLeft | Qt.AlignVCenter, self.footnote)
            painter.setFont(font)

        if self._cursor is not None and not for_export and rect.contains(
                QPointF(self._cursor)):
            self._draw_crosshair(painter, rect, fm)

    def _nice_step(self, span: float, target: int) -> float:
        if span <= 0 or target <= 0:
            return 1.0
        raw = span / target
        power = 10.0 ** math.floor(math.log10(raw))
        for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
            if raw <= multiple * power:
                return multiple * power
        return 10.0 * power

    def _draw_grid(self, painter, rect, fm, axis, faint, ink) -> None:
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        step_x = self._nice_step(x1 - x0, max(int(rect.width() / 70), 2))
        step_y = self._nice_step(y1 - y0, max(int(rect.height() / 42), 2))

        painter.setPen(QPen(faint, 0.7, Qt.DotLine))
        value = math.ceil(x0 / step_x) * step_x
        xs = []
        while value <= x1 + 1e-9:
            px = self._to_device(rect, value, 0)[0]
            painter.drawLine(QPointF(float(px), rect.top()),
                             QPointF(float(px), rect.bottom()))
            xs.append((float(px), value))
            value += step_x
        value = math.ceil(y0 / step_y) * step_y
        ys = []
        while value <= y1 + 1e-9:
            py = self._to_device(rect, 0, value)[1]
            painter.drawLine(QPointF(rect.left(), float(py)),
                             QPointF(rect.right(), float(py)))
            ys.append((float(py), value))
            value += step_y

        painter.setPen(QPen(ink))
        decimals_x = max(0, -int(math.floor(math.log10(step_x))))
        decimals_y = max(0, -int(math.floor(math.log10(step_y))))
        for px, value in xs:
            painter.drawText(
                QRectF(px - 40, rect.bottom() + 2 + self._tick_band, 80,
                       fm.height()),
                Qt.AlignHCenter | Qt.AlignVCenter, f"{value:.{decimals_x}f}")
        for py, value in ys:
            painter.drawText(QRectF(0, py - fm.height() / 2,
                                    rect.left() - 6, fm.height()),
                             Qt.AlignRight | Qt.AlignVCenter,
                             f"{value:.{decimals_y}f}")
        if self.x_label:
            painter.drawText(
                QRectF(rect.left(), rect.bottom() + 2 + self._tick_band
                       + fm.height(), rect.width(), fm.height()),
                Qt.AlignHCenter | Qt.AlignVCenter, self.x_label)
        if self.y_label:
            painter.save()
            painter.translate(fm.height() * 0.4, rect.center().y())
            painter.rotate(-90)
            painter.drawText(QRectF(-rect.height() / 2, -fm.height() / 2,
                                    rect.height(), fm.height()),
                             Qt.AlignHCenter | Qt.AlignVCenter, self.y_label)
            painter.restore()

    def _draw_series(self, painter, rect, s: Series) -> None:
        px, py = self._to_device(rect, s.x, s.y + s.offset)
        colour = QColor(*s.color)
        if s.filled:
            polygon = QPolygonF()
            baseline = float(self._to_device(rect, 0, s.offset)[1])
            polygon.append(QPointF(float(px[0]), baseline))
            for a, b in zip(px, py):
                polygon.append(QPointF(float(a), float(b)))
            polygon.append(QPointF(float(px[-1]), baseline))
            fill = QColor(colour)
            fill.setAlpha(58)
            painter.setBrush(fill)
            painter.setPen(Qt.NoPen)
            painter.drawPolygon(polygon)
            painter.setBrush(Qt.NoBrush)

        pen = QPen(colour, s.width)
        pen.setStyle(Qt.DashLine if s.dashed else Qt.SolidLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        path = QPainterPath(QPointF(float(px[0]), float(py[0])))
        for a, b in zip(px[1:], py[1:]):
            path.lineTo(QPointF(float(a), float(b)))
        painter.drawPath(path)

    def _draw_ticks(self, painter, rect, fm, ink) -> None:
        if not self.ticks:
            return
        height = fm.height() * 0.62
        x0, x1 = self.x_range
        for t in self.ticks:
            if not len(t.x):
                continue
            top = rect.bottom() + 2 + t.row * (height + 3)
            pen = QPen(QColor(*t.color), 1.1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            inside = t.x[(t.x >= x0) & (t.x <= x1)]
            px, _ = self._to_device(rect, inside, np.zeros(len(inside)))
            for a in np.atleast_1d(px):
                painter.drawLine(QPointF(float(a), top),
                                 QPointF(float(a), top + height))
            if t.title:
                painter.setPen(QPen(ink))
                painter.drawText(QRectF(rect.right() + 2, top, 60, height),
                                 Qt.AlignLeft | Qt.AlignVCenter, t.title)

    def _draw_frame(self, painter, rect, axis) -> None:
        pen = QPen(axis, 1.0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)

    def _draw_legend(self, painter, rect, fm, ink, background) -> None:
        entries = [s for s in self.series if s.visible and s.label]
        if not entries:
            return
        width = max(fm.horizontalAdvance(s.label) for s in entries) + 30
        height = len(entries) * (fm.height() + 2) + 6
        box = QRectF(rect.right() - width - 8, rect.top() + 8, width, height)
        plate = QColor(background)
        plate.setAlpha(205)
        painter.setBrush(plate)
        painter.setPen(QPen(self.faint, 0.8))
        painter.drawRect(box)
        painter.setBrush(Qt.NoBrush)
        y = box.top() + 3
        for s in entries:
            pen = QPen(QColor(*s.color), max(s.width, 1.6))
            pen.setStyle(Qt.DashLine if s.dashed else Qt.SolidLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            middle = y + fm.height() / 2
            painter.drawLine(QPointF(box.left() + 5, middle),
                             QPointF(box.left() + 22, middle))
            painter.setPen(QPen(ink))
            painter.drawText(QRectF(box.left() + 26, y, width - 30,
                                    fm.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, s.label)
            y += fm.height() + 2

    def _draw_crosshair(self, painter, rect, fm) -> None:
        px, py = float(self._cursor.x()), float(self._cursor.y())
        pen = QPen(self.crosshair, 0.8, Qt.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
        painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))

        x, y = self._to_data(rect, px, py)
        text = f"{x:.3f}, {y:.2f}"
        width = fm.horizontalAdvance(text) + 10
        box = QRectF(min(px + 6, rect.right() - width),
                     max(py - fm.height() - 4, rect.top() + 1),
                     width, fm.height() + 2)
        plate = QColor(self.background)
        plate.setAlpha(215)
        painter.setBrush(plate)
        painter.setPen(QPen(self.faint, 0.8))
        painter.drawRect(box)
        painter.setPen(QPen(self.crosshair))
        painter.drawText(box, Qt.AlignCenter, text)
        painter.setBrush(Qt.NoBrush)

    # -- export ------------------------------------------------------------
    def save_svg(self, path, width: int = 900, height: int = 520) -> None:
        """Write the plot as SVG. Every curve stays a curve, not pixels."""
        from PySide6.QtCore import QSize
        from PySide6.QtSvg import QSvgGenerator

        generator = QSvgGenerator()
        generator.setFileName(str(path))
        generator.setSize(QSize(width, height))
        generator.setViewBox(QRectF(0, 0, width, height))
        generator.setTitle(self.title or "FACET plot")
        generator.setDescription(self.footnote or "")
        painter = QPainter(generator)
        try:
            self.render_to(painter, width, height, for_export=True)
        finally:
            painter.end()

    def save_pdf(self, path, width_mm: float = 180.0,
                 height_mm: float = 110.0) -> None:
        """Write the plot as a single-page PDF at a stated physical size."""
        from PySide6.QtCore import QMarginsF, QSizeF
        from PySide6.QtGui import QPageSize, QPdfWriter

        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QSizeF(width_mm, height_mm),
                                     QPageSize.Millimeter))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        writer.setResolution(600)
        writer.setTitle(self.title or "FACET plot")
        painter = QPainter(writer)
        try:
            scale = writer.resolution() / 25.4
            self.render_to(painter, width_mm * scale, height_mm * scale,
                           for_export=True)
        finally:
            painter.end()

    # -- interaction -------------------------------------------------------
    def mouseMoveEvent(self, event) -> None:
        self._cursor = event.position().toPoint()
        rect, _ = self._metrics(self.width(), self.height(), self.font())
        if self._drag is not None and self._drag_range is not None:
            dx = event.position().x() - self._drag.x()
            dy = event.position().y() - self._drag.y()
            (x0, x1), (y0, y1) = self._drag_range
            span_x = (x1 - x0) * dx / rect.width() if rect.width() else 0.0
            span_y = (y1 - y0) * dy / rect.height() if rect.height() else 0.0
            self._x_range = (x0 - span_x, x1 - span_x)
            self._y_range = (y0 + span_y, y1 + span_y)
            self._auto = False
        else:
            x, y = self._to_data(rect, event.position().x(),
                                 event.position().y())
            self.hovered.emit(x, y)
        self.update()

    def leaveEvent(self, event) -> None:
        self._cursor = None
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag = event.position().toPoint()
            self._drag_range = (self.x_range, self.y_range)
            self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        moved = (self._drag is not None
                 and (event.position().toPoint() - self._drag).manhattanLength() > 3)
        self._drag = None
        self._drag_range = None
        self.unsetCursor()
        if event.button() == Qt.LeftButton and not moved:
            rect, _ = self._metrics(self.width(), self.height(), self.font())
            x, y = self._to_data(rect, event.position().x(),
                                 event.position().y())
            self.picked.emit(x, y)

    def mouseDoubleClickEvent(self, event) -> None:
        self.fit()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        factor = 0.85 ** steps
        rect, _ = self._metrics(self.width(), self.height(), self.font())
        x, y = self._to_data(rect, event.position().x(), event.position().y())
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        if event.modifiers() & Qt.ControlModifier:
            self._y_range = (y + (y0 - y) * factor, y + (y1 - y) * factor)
        else:
            self._x_range = (x + (x0 - x) * factor, x + (x1 - x) * factor)
            self._y_range = (y + (y0 - y) * factor, y + (y1 - y) * factor)
        self._auto = False
        self.update()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_0, Qt.Key_Home):
            self.fit()
        else:
            super().keyPressEvent(event)
