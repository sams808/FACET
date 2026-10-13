"""The shell plot: what sits around one atom, and what each shell adds to its
bond-valence sum.

The cutoff explorer answers "which coordination number does this threshold
give". This plot answers the companion question: "what does each successive
neighbour contribute". For the selected site it draws, against distance or
against bond valence,

* **CN**, the number of contacts counted so far -- a staircase that steps at
  every contact, read on the left axis;
* **the running bond-valence sum** over those same contacts, weighted by
  occupancy -- the integral the staircase is the derivative of, read on the
  right axis, with the size of the site's charge drawn across it for scale;
* **the contacts** as ticks on a rail, **the threshold** as the same draggable
  line the explorer has, and **the pin** where the file was opened.

Everything here is a re-sum of the contacts the analysis already found. No
neighbour search, no new parameter: a contact adds its own valence, and the
plot shows how much of the sum each shell carries.

Drawn with QPainter, like the explorer, so it repaints while the threshold is
dragged and stays crisp on a high-DPI display.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QSizePolicy, QToolButton, QToolTip, QWidget

from ..core import bv
from ..core.coordination import SiteResult

AXIS_DISTANCE = "distance"
AXIS_VALENCE = "valence"


@dataclass(frozen=True)
class Shell:
    """One contact in the order the plot includes it, with the totals after it."""

    index: int            # into result.contacts
    label: str
    element: str
    distance: float
    valence: float
    occupancy: float
    cn_after: int         # contacts counted once this one is included
    sum_after: float      # bond-valence sum, occupancy-weighted, once included


def shell_curve(result: SiteResult | None, axis: str = AXIS_DISTANCE) -> list[Shell]:
    """The contacts of a site in the order the plot includes them, with the
    running count and sum after each one.

    Against distance the contacts are included from the nearest outwards.
    Against valence they are included from the strongest down, which is the
    same order only when every contact uses the same parameter; with two
    anions (oxide and fluoride, say) the two orders differ, and each axis
    shows its own.

    Contacts with no bond-valence parameter are left out: they have no
    valence to add and the staircase of the explorer leaves them out too.
    """
    if result is None:
        return []
    rows = [(i, c) for i, c in enumerate(result.contacts) if c.has_valence]
    if axis == AXIS_VALENCE:
        rows.sort(key=lambda t: -t[1].valence)
    else:
        rows.sort(key=lambda t: t[1].distance)
    out: list[Shell] = []
    total = 0.0
    for k, (i, c) in enumerate(rows, start=1):
        total += float(c.valence) * float(c.occupancy)
        out.append(Shell(index=i, label=c.label, element=c.element,
                         distance=float(c.distance), valence=float(c.valence),
                         occupancy=float(c.occupancy), cn_after=k,
                         sum_after=total))
    return out


def at_threshold(result: SiteResult | None, v_bond: float) -> tuple[int, float, Shell | None]:
    """(CN, bond-valence sum, the next contact outside) at a threshold.

    Bonded means valence above the threshold, as the analysis counts it. The
    next contact is the strongest one left outside, with what it would add.
    """
    if result is None:
        return 0, float("nan"), None
    shells = shell_curve(result, AXIS_VALENCE)
    inside = [s for s in shells if s.valence > v_bond]
    outside = [s for s in shells if s.valence <= v_bond]
    cn = len(inside)
    total = inside[-1].sum_after if inside else 0.0
    return cn, total, (outside[0] if outside else None)


class ShellPlot(QWidget):
    """CN and the running bond-valence sum of one site, shell by shell."""

    thresholdChanged = Signal(float)
    contactHovered = Signal(int)          # index into result.contacts, or -1

    HEADROOM = 16

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(210)
        self.setMinimumWidth(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.SizeHorCursor)

        self.result: SiteResult | None = None
        self.v_bond = bv.V_BOND_DEFAULT
        self.pin_v: float | None = None
        self.axis = AXIS_DISTANCE
        self._theme = None
        self._dragging = False

        self._plot = QRectF()
        self._rail = QRectF()

        # the axis switch sits in the corner of the plot itself, where the
        # axis it changes is
        self.axis_button = QToolButton(self)
        self.axis_button.setAutoRaise(True)
        self.axis_button.setCursor(Qt.ArrowCursor)
        self.axis_button.clicked.connect(self.toggle_axis)
        self._refresh_axis_button()

    # -- api ---------------------------------------------------------------
    def set_result(self, result: SiteResult | None) -> None:
        self.result = result
        if result is not None:
            self.v_bond = result.v_bond
        self.update()

    def set_threshold(self, v: float, emit: bool = False) -> None:
        low, high = self._v_data_range()
        self.v_bond = float(np.clip(v, low, high))
        if emit:
            self.thresholdChanged.emit(self.v_bond)
        self.update()

    def set_pin(self, v: float | None) -> None:
        self.pin_v = None if v is None else float(v)
        self.update()

    def set_theme(self, theme) -> None:
        self._theme = theme
        self.update()

    def set_axis(self, axis: str) -> None:
        self.axis = AXIS_VALENCE if axis == AXIS_VALENCE else AXIS_DISTANCE
        self._refresh_axis_button()
        self.update()

    def toggle_axis(self) -> None:
        self.set_axis(AXIS_VALENCE if self.axis == AXIS_DISTANCE
                      else AXIS_DISTANCE)

    def _refresh_axis_button(self) -> None:
        if self.axis == AXIS_DISTANCE:
            self.axis_button.setText("Å")
            self.axis_button.setToolTip(
                "Horizontal axis: distance. Click to read the same shells "
                "against bond valence.")
        else:
            self.axis_button.setText("v.u.")
            self.axis_button.setToolTip(
                "Horizontal axis: bond valence, logarithmic. Click to read "
                "the same shells against distance.")

    # -- colours -----------------------------------------------------------
    def _light(self) -> bool:
        return bool(self._theme and self._theme.is_light_background)

    def _c(self, dark: str, light: str) -> QColor:
        return QColor(light if self._light() else dark)

    @property
    def _ink(self) -> QColor:
        return self._c("#d8dde6", "#1c1f24")

    @property
    def _muted(self) -> QColor:
        return self._c("#8a93a3", "#6b7280")

    @property
    def _sum_colour(self) -> QColor:
        return self._c("#7fb8a4", "#2a7f62")

    @property
    def _accent(self) -> QColor:
        return self._c("#ffd65c", "#b8860b")

    @property
    def _pin_colour(self) -> QColor:
        return self._c("#c58fd6", "#7b3f99")

    @property
    def _ground(self) -> QColor:
        if self._theme is not None:
            c = QColor()
            bgr, bgg, bgb = self._theme.background
            f = 0.92 if not self._light() else 1.0
            c.setRgbF(min(bgr / f, 1), min(bgg / f, 1), min(bgb / f, 1))
            return c
        return QColor("#14161b")

    # -- data ranges and axis mapping --------------------------------------
    def _dominant_param(self):
        if self.result is None:
            return None
        for c in self.result.contacts:
            if c.param is not None:
                return c.param
        return None

    def _v_data_range(self) -> tuple[float, float]:
        """The valences the site has, with a margin, plus the marks."""
        lo, hi = 0.004, 0.6
        if self.result is not None:
            values = [c.valence for c in self.result.contacts
                      if c.has_valence and c.valence > 0]
            if values:
                hi = max(hi, max(values) * 1.12)
                lo = min(lo, min(values) / 1.12)
        return lo, hi

    def _v_range(self) -> tuple[float, float]:
        lo, hi = self._v_data_range()
        for v in (float(self.v_bond), self.pin_v):
            if v is not None and v > 0:
                lo, hi = min(lo, v / 1.05), max(hi, v * 1.05)
        return lo, hi

    def _d_range(self) -> tuple[float, float]:
        """The distances the site has, with a margin, plus the marks."""
        lo, hi = 1.5, 3.5
        if self.result is not None:
            ds = [c.distance for c in self.result.contacts if c.has_valence]
            if ds:
                lo, hi = min(ds) - 0.15, max(ds) + 0.15
        param = self._dominant_param()
        if param is not None:
            for v in (float(self.v_bond), self.pin_v):
                if v is not None and v > 0:
                    d = float(param.distance_for(v))
                    lo, hi = min(lo, d - 0.05), max(hi, d + 0.05)
        return lo, hi

    def _v_to_x(self, v: float) -> float:
        if self.axis == AXIS_VALENCE:
            v_lo, v_hi = self._v_range()
            lo, hi = math.log10(v_lo), math.log10(v_hi)
            t = (math.log10(min(max(float(v), v_lo), v_hi)) - lo) / (hi - lo)
            return self._plot.left() + t * self._plot.width()
        param = self._dominant_param()
        if param is None:
            return self._plot.left()
        return self._d_to_x(float(param.distance_for(v)))

    def _d_to_x(self, d: float) -> float:
        d_lo, d_hi = self._d_range()
        t = (min(max(float(d), d_lo), d_hi) - d_lo) / max(d_hi - d_lo, 1e-9)
        return self._plot.left() + t * self._plot.width()

    def _x_to_v(self, x: float) -> float:
        t = (x - self._plot.left()) / max(self._plot.width(), 1.0)
        t = float(np.clip(t, 0.0, 1.0))
        if self.axis == AXIS_VALENCE:
            v_lo, v_hi = self._v_range()
            lo, hi = math.log10(v_lo), math.log10(v_hi)
            return float(10 ** (lo + t * (hi - lo)))
        d_lo, d_hi = self._d_range()
        d = d_lo + t * (d_hi - d_lo)
        param = self._dominant_param()
        if param is None:
            return self.v_bond
        return float(param.valence(d))

    def _shell_x(self, s: Shell) -> float:
        return (self._v_to_x(s.valence) if self.axis == AXIS_VALENCE
                else self._d_to_x(s.distance))

    def _cn_max(self, shells: list[Shell]) -> int:
        return max(len(shells), 1)

    def _sum_max(self, shells: list[Shell]) -> float:
        total = shells[-1].sum_after if shells else 0.0
        ox = abs(self.result.ox) if (self.result is not None
                                     and self.result.ox is not None) else 0.0
        return max(total, ox, 1e-6) * 1.08

    def _cn_to_y(self, cn: float, cn_max: int) -> float:
        return self._plot.bottom() - (cn / max(cn_max, 1)) * self._plot.height()

    def _sum_to_y(self, s: float, s_max: float) -> float:
        return self._plot.bottom() - (s / s_max) * self._plot.height()

    # -- painting ----------------------------------------------------------
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._place_button()

    def _place_button(self) -> None:
        hint = self.axis_button.sizeHint()
        self.axis_button.move(self.width() - hint.width() - 4, 2)

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.fillRect(self.rect(), self._ground)

        shells = shell_curve(self.result, self.axis)
        if not shells:
            p.setPen(self._muted)
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Select a site to see its shells")
            p.end()
            return

        self._layout()
        self._paint_grid(p, shells)
        self._paint_charge(p, shells)
        self._paint_staircases(p, shells)
        self._paint_rail(p, shells)
        self._paint_pin(p)
        self._paint_threshold(p)
        self._paint_readout(p)
        p.end()

    def _small_font(self) -> QFont:
        fnt = QFont(self.font())
        fnt.setPointSizeF(max(7.0, fnt.pointSizeF() - 1.5))
        return fnt

    def _layout(self) -> None:
        f = QFontMetricsF(self.font())
        small = QFontMetricsF(self._small_font())
        top = f.height() + 10 + self.HEADROOM
        bottom = 24 + small.height() + f.height() + 6
        self._plot = QRectF(self.rect()).adjusted(46, top, -58, -bottom)
        self._rail = QRectF(self._plot.left(), self._plot.bottom() + 6,
                            self._plot.width(), 16)

    def _paint_grid(self, p: QPainter, shells: list[Shell]) -> None:
        cn_max = self._cn_max(shells)
        s_max = self._sum_max(shells)
        p.setPen(QPen(self._muted, 1.0))
        p.drawLine(QPointF(self._plot.left(), self._plot.top()),
                   QPointF(self._plot.left(), self._plot.bottom()))
        p.drawLine(QPointF(self._plot.left(), self._plot.bottom()),
                   QPointF(self._plot.right(), self._plot.bottom()))
        p.setPen(QPen(self._sum_colour, 1.0))
        p.drawLine(QPointF(self._plot.right(), self._plot.top()),
                   QPointF(self._plot.right(), self._plot.bottom()))

        fnt = self._small_font()
        p.setFont(fnt)
        fm = QFontMetricsF(fnt)
        step = 1 if cn_max <= 8 else 2
        for cn in range(0, cn_max + 1, step):
            y = self._cn_to_y(cn, cn_max)
            p.setPen(QPen(QColor(self._muted.red(), self._muted.green(),
                                 self._muted.blue(), 45), 1.0, Qt.DotLine))
            p.drawLine(QPointF(self._plot.left(), y),
                       QPointF(self._plot.right(), y))
            p.setPen(self._muted)
            p.drawText(QRectF(2, y - 8, 40, 16),
                       Qt.AlignRight | Qt.AlignVCenter, str(cn))
        p.setPen(self._muted)
        p.drawText(QRectF(2, self._plot.top() - self.HEADROOM - 2, 40, 16),
                   Qt.AlignRight | Qt.AlignVCenter, "CN")

        # the right axis: the running sum, a few round ticks
        tick = _round_step(s_max / 4)
        v = 0.0
        p.setPen(self._sum_colour)
        while v <= s_max + 1e-9:
            y = self._sum_to_y(v, s_max)
            p.drawLine(QPointF(self._plot.right(), y),
                       QPointF(self._plot.right() + 4, y))
            p.drawText(QRectF(self._plot.right() + 6, y - 8, 50, 16),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{v:g}")
            v += tick
        p.drawText(QRectF(self._plot.right() + 6,
                          self._plot.top() - self.HEADROOM - 2, 50, 16),
                   Qt.AlignLeft | Qt.AlignVCenter, "Σ v.u.")

        # the horizontal axis caption, in the corner the button does not use
        caption = ("distance, Å" if self.axis == AXIS_DISTANCE
                   else "bond valence, v.u. (log)")
        lo, hi = (self._d_range() if self.axis == AXIS_DISTANCE
                  else self._v_range())
        fmt = "{:.2f}" if self.axis == AXIS_DISTANCE else "{:.3f}"
        p.setPen(self._muted)
        p.drawText(QRectF(self._plot.left(), self._rail.bottom() + 2,
                          80, fm.height()), Qt.AlignLeft, fmt.format(lo))
        p.drawText(QRectF(self._plot.right() - 80, self._rail.bottom() + 2,
                          80, fm.height()), Qt.AlignRight, fmt.format(hi))
        p.drawText(QRectF(self._plot.left(), self._rail.bottom() + 2,
                          self._plot.width(), fm.height()), Qt.AlignCenter,
                   caption)
        p.setFont(self.font())

    def _paint_charge(self, p: QPainter, shells: list[Shell]) -> None:
        """The size of the site's charge, across the sum axis, for scale.

        Drawn as a line and a number, not as a target: where the running sum
        meets it is a fact about the structure and the parameters, and the
        reader draws the conclusion.
        """
        r = self.result
        if r is None or r.ox is None:
            return
        s_max = self._sum_max(shells)
        y = self._sum_to_y(abs(r.ox), s_max)
        col = QColor(self._sum_colour)
        col.setAlpha(150)
        p.setPen(QPen(col, 1.0, Qt.DashLine))
        p.drawLine(QPointF(self._plot.left(), y), QPointF(self._plot.right(), y))
        fnt = self._small_font()
        p.setFont(fnt)
        p.setPen(col)
        # under the line, at the end where the staircases start from zero
        # and so run along the bottom: the left against distance, the right
        # against valence
        at_left = self.axis == AXIS_DISTANCE
        left = self._plot.left() + 4 if at_left else self._plot.right() - 124
        p.drawText(QRectF(left, y + 1, 120, 14),
                   (Qt.AlignLeft if at_left else Qt.AlignRight)
                   | Qt.AlignVCenter, f"|charge| {abs(r.ox)}")
        p.setFont(self.font())

    def _staircase(self, shells: list[Shell], y_of) -> QPainterPath:
        """The step function through the shells, in the order they are
        included: left to right against distance, right to left against
        valence."""
        xs = [self._shell_x(s) for s in shells]
        inward = self.axis == AXIS_VALENCE
        start = self._plot.right() if inward else self._plot.left()
        end = self._plot.left() if inward else self._plot.right()
        path = QPainterPath()
        y = y_of(0)
        path.moveTo(start, y)
        for s, x in zip(shells, xs):
            path.lineTo(x, y)
            y = y_of(s)
            path.lineTo(x, y)
        path.lineTo(end, y)
        return path

    def _paint_staircases(self, p: QPainter, shells: list[Shell]) -> None:
        cn_max = self._cn_max(shells)
        s_max = self._sum_max(shells)

        def cn_y(s):
            return self._cn_to_y(0 if s == 0 else s.cn_after, cn_max)

        def sum_y(s):
            return self._sum_to_y(0.0 if s == 0 else s.sum_after, s_max)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(self._sum_colour, 2.0, Qt.SolidLine, Qt.FlatCap,
                      Qt.MiterJoin))
        p.drawPath(self._staircase(shells, sum_y))
        p.setPen(QPen(self._ink, 2.0, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin))
        p.drawPath(self._staircase(shells, cn_y))

    def _paint_rail(self, p: QPainter, shells: list[Shell]) -> None:
        p.setPen(QPen(self._muted, 1.0))
        p.drawLine(QPointF(self._rail.left(), self._rail.center().y()),
                   QPointF(self._rail.right(), self._rail.center().y()))
        for s in shells:
            x = self._shell_x(s)
            bonded = s.valence > self.v_bond
            h = self._rail.height() * (0.95 if bonded else 0.55)
            p.setPen(QPen(self._ink if bonded else self._muted,
                          2.2 if bonded else 1.2))
            p.drawLine(QPointF(x, self._rail.center().y() - h / 2),
                       QPointF(x, self._rail.center().y() + h / 2))

    def _paint_pin(self, p: QPainter) -> None:
        if self.pin_v is None or self.pin_v <= 0:
            return
        x = self._v_to_x(self.pin_v)
        col = self._pin_colour
        p.setPen(QPen(col, 1.4))
        p.drawLine(QPointF(x, self._plot.top() - 2),
                   QPointF(x, self._rail.bottom()))
        p.setBrush(QBrush(col))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(x, self._plot.top() - 7), 4.0, 4.0)
        p.setBrush(Qt.NoBrush)

    def _paint_threshold(self, p: QPainter) -> None:
        x = self._v_to_x(self.v_bond)
        accent = self._accent
        p.setPen(QPen(accent, 2.0))
        p.drawLine(QPointF(x, self._plot.top() - 4),
                   QPointF(x, self._rail.bottom()))
        handle = QPolygonF([QPointF(x, self._plot.top() - 4),
                            QPointF(x - 5, self._plot.top() - 12),
                            QPointF(x + 5, self._plot.top() - 12)])
        p.setBrush(QBrush(accent))
        p.setPen(Qt.NoPen)
        p.drawPolygon(handle)
        p.setBrush(Qt.NoBrush)

    def _paint_readout(self, p: QPainter) -> None:
        """What the threshold includes, and what the next shell would add."""
        text = self.readout_text()
        fnt = QFont(self.font())
        fnt.setPointSizeF(max(7.5, fnt.pointSizeF() - 0.5))
        p.setFont(fnt)
        fm = QFontMetricsF(fnt)
        p.setPen(self._ink)
        # the whole width of the widget, less a margin: the sum axis's
        # labels sit above this line, not beside it
        p.drawText(QRectF(self._plot.left(), self.height() - fm.height() - 2,
                          self.width() - self._plot.left() - 6, fm.height()),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(text, Qt.ElideRight,
                                 self.width() - self._plot.left() - 6))
        p.setFont(self.font())

    def readout_text(self) -> str:
        """The measurement at the threshold, stated; what the next shell adds."""
        cn, total, nxt = at_threshold(self.result, self.v_bond)
        r = self.result
        if r is None:
            return ""
        text = f"{r.label}: CN {cn} · Σ {total:.3f}"
        if r.bvs_listed == r.bvs_listed:
            text += f" of {r.bvs_listed:.3f} v.u."
        else:
            text += " v.u."
        text += f" at {self.v_bond:.4f}"
        if nxt is not None:
            text += (f"  ·  next {nxt.label} {nxt.distance:.3f} Å "
                     f"+{nxt.valence * nxt.occupancy:.3f}")
        return text

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self.result is None:
            return
        self._dragging = True
        self.set_threshold(self._x_to_v(event.position().x()), emit=True)

    def mouseMoveEvent(self, event) -> None:
        if self.result is None:
            return
        if self._dragging:
            self.set_threshold(self._x_to_v(event.position().x()), emit=True)
            return
        self._maybe_tooltip(event)

    def mouseReleaseEvent(self, _event) -> None:
        self._dragging = False

    def _maybe_tooltip(self, event) -> None:
        shells = shell_curve(self.result, self.axis)
        x = event.position().x()
        best, best_dx = None, 7.0
        for s in shells:
            dx = abs(self._shell_x(s) - x)
            if dx < best_dx:
                best, best_dx = s, dx
        if best is None:
            QToolTip.hideText()
            self.contactHovered.emit(-1)
            return
        state = ("above the threshold" if best.valence > self.v_bond
                 else "below the threshold")
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{self.result.label}–{best.label}\n{best.distance:.4f} Å · "
            f"{best.valence:.4f} v.u. — {state}\n"
            f"with it: CN {best.cn_after}, Σ {best.sum_after:.3f} v.u.", self)
        self.contactHovered.emit(best.index)

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()
        self.contactHovered.emit(-1)


def _round_step(raw: float) -> float:
    """A tick step of 0.1, 0.2, 0.5, 1, 2, 5 ... near the value asked for."""
    if raw <= 0:
        return 1.0
    power = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * power >= raw:
            return m * power
    return 10 * power
