"""The cutoff explorer.

The screen the application exists for. It shows, for one site, the coordination
number as a function of the threshold that produced it -- the whole staircase,
not one step -- so that the question "what is the coordination number?" is
answered with the honest shape of the answer.

Four things are on one axis:

* **the staircase**, CN against threshold, on a logarithmic valence axis
* **the plateaus**, shaded in proportion to their width. The shading encodes
  the measured width and nothing else -- no threshold is marked as good or
  bad, because what counts as a wide plateau depends on the question.
* **the contacts themselves**, each a tick at its own bond valence, so the
  steps are visibly the contacts and not an abstraction
* **reference marks**: the tabulation threshold, the conventional distance
  cutoff, and where the other coordination-number definitions land

The axis is logarithmic in valence because that is the axis on which a plateau
width *is* a distance gap: ln(v_k / v_k+1) = (d_k+1 - d_k) / b. The top scale
reads the same axis in angstrom for the selected cation-anion pair, so the same
picture can be read either way.

Drawn with QPainter rather than matplotlib: it must repaint while a threshold is
being dragged, and it must stay crisp on a high-DPI display.
"""
from __future__ import annotations

import math

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
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from ..core import bv
from ..core.coordination import SiteResult

V_MIN, V_MAX = 0.004, 0.6

# Shading saturates at this width, so the darkest band is not "good" -- it is
# simply the widest that the scale distinguishes.
SHADE_FULL_DECADES = 1.0


class CutoffExplorer(QWidget):
    """CN against threshold for one site, with its plateaus and its contacts."""

    thresholdChanged = Signal(float)
    contactHovered = Signal(int)          # index into result.contacts, or -1

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setCursor(Qt.SizeHorCursor)

        self.result: SiteResult | None = None
        self.v_bond = bv.V_BOND_DEFAULT
        self.reference_distance: float | None = 3.00   # the usual Bi-O convention
        self.show_distance_axis = True

        self._dragging = False
        self._hover_v: float | None = None
        self._theme = None

        # geometry, recomputed on every paint
        self._plot = QRectF()
        self._rail = QRectF()

    # -- api ---------------------------------------------------------------
    def set_result(self, result: SiteResult | None) -> None:
        self.result = result
        if result is not None:
            self.v_bond = result.v_bond
        self.update()

    def set_threshold(self, v: float, emit: bool = False) -> None:
        self.v_bond = float(np.clip(v, V_MIN, V_MAX))
        if emit:
            self.thresholdChanged.emit(self.v_bond)
        self.update()

    def set_theme(self, theme) -> None:
        self._theme = theme
        self.update()

    def set_reference_distance(self, d: float | None) -> None:
        self.reference_distance = d
        self.update()

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
    def _ground(self) -> QColor:
        if self._theme is not None:
            c = QColor()
            bgr, bgg, bgb = self._theme.background
            # a touch lighter than the viewport, so the panel reads as a panel
            f = 0.92 if not self._light() else 1.0
            c.setRgbF(min(bgr / f, 1), min(bgg / f, 1), min(bgb / f, 1))
            return c
        return QColor("#14161b")

    # -- axis mapping ------------------------------------------------------
    def _v_to_x(self, v: float) -> float:
        lo, hi = math.log10(V_MIN), math.log10(V_MAX)
        t = (math.log10(max(float(v), V_MIN)) - lo) / (hi - lo)
        return self._plot.left() + t * self._plot.width()

    def _x_to_v(self, x: float) -> float:
        lo, hi = math.log10(V_MIN), math.log10(V_MAX)
        t = (x - self._plot.left()) / max(self._plot.width(), 1.0)
        return float(10 ** (lo + np.clip(t, 0.0, 1.0) * (hi - lo)))

    def _dominant_param(self):
        """The bond-valence parameter to read the distance axis against."""
        if self.result is None:
            return None
        for c in self.result.contacts:
            if c.param is not None:
                return c.param
        return None

    # -- painting ----------------------------------------------------------
    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.TextAntialiasing, True)
        p.fillRect(self.rect(), self._ground)

        if self.result is None or not self.result.contacts:
            self._paint_empty(p)
            p.end()
            return

        self._layout()
        self._paint_plateaus(p)
        self._paint_grid(p)
        self._paint_staircase(p)
        self._paint_contact_rail(p)
        self._paint_references(p)
        self._paint_threshold(p)
        self._paint_readout(p)
        p.end()

    def _paint_empty(self, p: QPainter) -> None:
        p.setPen(self._muted)
        p.drawText(self.rect(), Qt.AlignCenter,
                   "Select a cation site to explore its coordination")

    def _layout(self) -> None:
        f = QFontMetricsF(self.font())
        top = f.height() + 10 if self.show_distance_axis else 8
        bottom = f.height() * 2 + 26
        self._plot = QRectF(self.rect()).adjusted(46, top, -14, -bottom)
        self._rail = QRectF(self._plot.left(), self._plot.bottom() + 6,
                            self._plot.width(), 16)

    # -- layers ------------------------------------------------------------
    def _paint_plateaus(self, p: QPainter) -> None:
        """Shade each plateau in proportion to its width.

        One neutral colour whose opacity tracks the measured width, rather than
        a green/amber split. A two-colour scheme encodes a verdict -- this
        threshold is sound, that one is not -- and that is the reader's call,
        not the application's.
        """
        r = self.result
        cns = [q.cn for q in r.plateaus] or [0]
        cn_max = max(cns + [1])

        for q in r.plateaus:
            x0, x1 = self._v_to_x(q.v_low), self._v_to_x(q.v_high)
            if x1 - x0 < 0.6:
                continue
            current = q.contains(self.v_bond)
            weight = min(q.width_decades / SHADE_FULL_DECADES, 1.0)
            alpha = int((26 + 54 * weight) * (2.0 if current else 1.0))
            col = QColor(120, 150, 190, min(alpha, 150))
            p.fillRect(QRectF(x0, self._plot.top(), x1 - x0, self._plot.height()),
                       col)

            if current:
                pen = QPen(col.lighter(150), 1.2, Qt.DashLine)
                p.setPen(pen)
                p.drawLine(QPointF(x0, self._plot.top()),
                           QPointF(x0, self._plot.bottom()))
                p.drawLine(QPointF(x1, self._plot.top()),
                           QPointF(x1, self._plot.bottom()))

            # label the wide ones; a narrow band has no room and no claim
            if q.width_decades > 0.28:
                p.setPen(self._ink if current else self._muted)
                fnt = QFont(self.font())
                fnt.setBold(current)
                p.setFont(fnt)
                y = self._cn_to_y(q.cn, cn_max)
                p.drawText(QRectF(x0, y - 20, x1 - x0, 18), Qt.AlignCenter,
                           f"CN {q.cn}   {q.width_decades:.2f} dec")
                p.setFont(self.font())

    def _cn_to_y(self, cn: float, cn_max: int) -> float:
        t = cn / max(cn_max, 1)
        return self._plot.bottom() - t * (self._plot.height() - 16)

    def _paint_grid(self, p: QPainter) -> None:
        r = self.result
        cn_max = max([q.cn for q in r.plateaus] + [1])
        p.setPen(QPen(self._muted, 1.0))
        p.drawLine(QPointF(self._plot.left(), self._plot.top()),
                   QPointF(self._plot.left(), self._plot.bottom()))
        p.drawLine(QPointF(self._plot.left(), self._plot.bottom()),
                   QPointF(self._plot.right(), self._plot.bottom()))

        step = 1 if cn_max <= 8 else 2
        fnt = QFont(self.font())
        fnt.setPointSizeF(max(7.0, fnt.pointSizeF() - 1.5))
        p.setFont(fnt)
        for cn in range(0, cn_max + 1, step):
            y = self._cn_to_y(cn, cn_max)
            p.setPen(QPen(QColor(self._muted.red(), self._muted.green(),
                                 self._muted.blue(), 45), 1.0, Qt.DotLine))
            p.drawLine(QPointF(self._plot.left(), y), QPointF(self._plot.right(), y))
            p.setPen(self._muted)
            p.drawText(QRectF(2, y - 8, 40, 16),
                       Qt.AlignRight | Qt.AlignVCenter, str(cn))
        p.setPen(self._muted)
        p.drawText(QRectF(2, self._plot.top() - 2, 40, 16),
                   Qt.AlignRight | Qt.AlignVCenter, "CN")
        p.setFont(self.font())

    def _paint_staircase(self, p: QPainter) -> None:
        """CN against threshold, as the step function it is."""
        r = self.result
        cn_max = max([q.cn for q in r.plateaus] + [1])
        values = sorted((c.valence for c in r.contacts if c.has_valence),
                        reverse=True)
        if not values:
            return

        path = QPainterPath()
        x = self._v_to_x(V_MIN)
        y = self._cn_to_y(len(values), cn_max)
        path.moveTo(x, y)
        for k, v in enumerate(reversed(values), start=1):
            # walking inwards: at threshold v the count drops from
            # len - k + 1 to len - k
            xv = self._v_to_x(v)
            path.lineTo(xv, y)
            y = self._cn_to_y(len(values) - k, cn_max)
            path.lineTo(xv, y)
        path.lineTo(self._v_to_x(V_MAX), y)

        p.setPen(QPen(self._ink, 2.0, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)

    def _paint_contact_rail(self, p: QPainter) -> None:
        """Every contact as a tick at its own valence.

        The steps of the staircase are these ticks. Showing them makes the
        staircase concrete rather than an abstraction, and makes it obvious when
        a plateau is narrow because two contacts are nearly coincident.
        """
        r = self.result
        p.setPen(QPen(self._muted, 1.0))
        p.drawLine(QPointF(self._rail.left(), self._rail.center().y()),
                   QPointF(self._rail.right(), self._rail.center().y()))

        for c in r.contacts:
            if not c.has_valence:
                continue
            x = self._v_to_x(c.valence)
            bonded = c.valence >= self.v_bond
            h = self._rail.height() * (0.95 if bonded else 0.55)
            col = self._ink if bonded else self._muted
            p.setPen(QPen(col, 2.2 if bonded else 1.2))
            p.drawLine(QPointF(x, self._rail.center().y() - h / 2),
                       QPointF(x, self._rail.center().y() + h / 2))

    def _paint_references(self, p: QPainter) -> None:
        """Marks for the thresholds and rules a reader already knows."""
        r = self.result
        marks: list[tuple[float, str, QColor]] = []

        marks.append((r.v_list, "tabulate", QColor("#6f8cb0")))

        param = self._dominant_param()
        if param is not None and self.reference_distance:
            v = float(param.valence(self.reference_distance))
            if V_MIN < v < V_MAX:
                marks.append((v, f"{self.reference_distance:.2f} Å "
                                 f"convention", QColor("#b08a5a")))

        # where the max-gap rule falls: between the last included contact and
        # the first excluded one
        vals = sorted((c.valence for c in r.contacts if c.has_valence),
                      reverse=True)
        if r.cn_gap and 0 < r.cn_gap < len(vals):
            v = math.sqrt(vals[r.cn_gap - 1] * vals[r.cn_gap])
            marks.append((v, "max gap", QColor("#7f9e79")))

        fnt = QFont(self.font())
        fnt.setPointSizeF(max(7.0, fnt.pointSizeF() - 1.5))
        p.setFont(fnt)
        fm = QFontMetricsF(fnt)

        # Stagger labels that would otherwise collide. The 0.075 v.u. threshold
        # sits at 3.05 A for Bi-O, which is almost exactly the 3.00 A
        # convention -- so these marks land on top of each other precisely in
        # the case the figure is meant to make a point about.
        placed: list[tuple[float, float, int]] = []          # left, right, row

        def overlaps(left: float, right: float, row: int) -> bool:
            return any(r == row and left < pr and right > pl
                       for pl, pr, r in placed)

        for v, label, col in sorted(marks, key=lambda m: m[0]):
            x = self._v_to_x(v)
            w = fm.horizontalAdvance(label) + 8
            left = min(max(x - w / 2, self._plot.left()), self._plot.right() - w)
            row = 0
            while row < 3 and overlaps(left, left + w, row):
                row += 1
            placed.append((left, left + w, row))

            p.setPen(QPen(col, 1.2, Qt.DashLine))
            p.drawLine(QPointF(x, self._plot.top()),
                       QPointF(x, self._plot.bottom()))
            p.setPen(col)
            y = self._rail.bottom() + 2 + row * (fm.height() - 1)
            p.drawText(QRectF(left, y, w, fm.height()), Qt.AlignCenter, label)
        p.setFont(self.font())

    def _paint_threshold(self, p: QPainter) -> None:
        x = self._v_to_x(self.v_bond)
        accent = QColor("#ffd65c") if not self._light() else QColor("#b8860b")
        p.setPen(QPen(accent, 2.0))
        p.drawLine(QPointF(x, self._plot.top() - 4),
                   QPointF(x, self._rail.bottom()))

        # a grab handle, so it is obvious the line is draggable
        handle = QPolygonF([QPointF(x, self._plot.top() - 4),
                            QPointF(x - 5, self._plot.top() - 12),
                            QPointF(x + 5, self._plot.top() - 12)])
        p.setBrush(QBrush(accent))
        p.setPen(Qt.NoPen)
        p.drawPolygon(handle)

        if self.show_distance_axis:
            param = self._dominant_param()
            fnt = QFont(self.font())
            fnt.setPointSizeF(max(7.0, fnt.pointSizeF() - 1.0))
            p.setFont(fnt)
            p.setPen(accent)
            text = f"{self.v_bond:.4f} v.u."
            if param is not None:
                text += f"  ≡  {param.distance_for(self.v_bond):.3f} Å " \
                        f"({param.label})"
            fm = QFontMetricsF(fnt)
            w = fm.horizontalAdvance(text) + 8
            left = min(max(x - w / 2, self._plot.left()), self._plot.right() - w)
            p.drawText(QRectF(left, 1, w, fm.height()), Qt.AlignCenter, text)
            p.setFont(self.font())

    def _paint_readout(self, p: QPainter) -> None:
        """The measurement, stated. No conclusion drawn from it."""
        r = self.result
        plateau = next((q for q in r.plateaus if q.contains(self.v_bond)), None)
        cn = r.cn_at(self.v_bond)

        if plateau is None:
            text = f"CN {cn} at {self.v_bond:.4f} v.u.  ·  on a step edge"
        else:
            text = (f"CN {cn} at {self.v_bond:.4f} v.u.  ·  plateau "
                    f"{plateau.v_low:.4f}–{plateau.v_high:.4f} v.u.  ·  "
                    f"{plateau.width_decades:.2f} decades  ·  "
                    f"{plateau.width_angstrom:.3f} Å gap")
        col = self._ink

        fnt = QFont(self.font())
        fnt.setPointSizeF(max(7.5, fnt.pointSizeF() - 0.5))
        p.setFont(fnt)
        p.setPen(col)
        fm = QFontMetricsF(fnt)
        p.drawText(QRectF(self._plot.left(), self.height() - fm.height() - 2,
                          self._plot.width(), fm.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, text)
        p.setFont(self.font())

    # -- interaction -------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.LeftButton or self.result is None:
            return
        self._dragging = True
        self._set_from_x(event.position().x())

    def mouseMoveEvent(self, event) -> None:
        if self.result is None:
            return
        if self._dragging:
            self._set_from_x(event.position().x())
            return
        self._maybe_tooltip(event)

    def mouseReleaseEvent(self, _event) -> None:
        self._dragging = False

    def mouseDoubleClickEvent(self, event) -> None:
        """Snap the threshold to the centre of the plateau under the pointer.

        A convenience for landing exactly on a plateau midpoint rather than
        near it. It makes no claim about that plateau.
        """
        if self.result is None:
            return
        v = self._x_to_v(event.position().x())
        plateau = next((q for q in self.result.plateaus if q.contains(v)), None)
        if plateau is None:
            return
        centre = math.sqrt(plateau.v_low * plateau.v_high)   # midpoint in log
        self.set_threshold(centre, emit=True)

    def _set_from_x(self, x: float) -> None:
        self.set_threshold(self._x_to_v(x), emit=True)

    def _maybe_tooltip(self, event) -> None:
        r = self.result
        x = event.position().x()
        best, best_dx = None, 7.0
        for i, c in enumerate(r.contacts):
            if not c.has_valence:
                continue
            dx = abs(self._v_to_x(c.valence) - x)
            if dx < best_dx:
                best, best_dx = i, dx
        if best is None:
            QToolTip.hideText()
            self.contactHovered.emit(-1)
            return
        c = r.contacts[best]
        state = ("above the threshold" if c.valence >= self.v_bond
                 else "below the threshold")
        note = "" if (c.param and c.param.fitted) else "  (estimated parameter)"
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{r.label}–{c.label}\n{c.distance:.4f} Å\n"
            f"{c.valence:.4f} v.u. — {state}{note}", self)
        self.contactHovered.emit(best)

    def leaveEvent(self, _event) -> None:
        QToolTip.hideText()
        self.contactHovered.emit(-1)
