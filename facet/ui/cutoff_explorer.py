"""The cutoff explorer.

The screen the application exists for. It shows, for one site, the coordination
number as a function of the threshold that produced it -- the whole staircase,
not one step -- so that the question "what is the coordination number?" is
answered with the honest shape of the answer.

On one horizontal axis:

* **the staircase**, CN against threshold, read on the left axis
* **the running bond-valence sum** over the same contacts, read on the right
  axis: the integral the staircase is the derivative of, with the size of the
  site's charge drawn across it for scale. It shows what each successive
  neighbour adds, and what the next contact outside the cut would add.
* **the plateaus**, shaded in proportion to their width. The shading encodes
  the measured width and nothing else -- no threshold is marked as good or
  bad, because what counts as a wide plateau depends on the question.
* **the contacts themselves**, each a tick at its own bond valence (or its
  own distance), so the steps are visibly the contacts and not an abstraction
* **reference marks**: the tabulation threshold, the conventional distance
  cutoff, where the max-gap rule lands, and a pin at the threshold the file
  was opened with
* **the threshold**, a draggable line, with its value in both units above

The axis is bond valence by default, logarithmic because that is the axis on
which a plateau width *is* a distance gap: ln(v_k / v_k+1) = (d_k+1 - d_k) / b.
The corner button turns the same picture into one against distance, linear,
for the pair the selected site is read against. With one kind of anion the
two are the same curve on two scales; with two kinds the contacts are
included in a different order, and each axis shows its own.

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
from PySide6.QtWidgets import QSizePolicy, QToolButton, QToolTip, QWidget

from ..core import bv
from ..core import shells as shells_mod
from ..core.coordination import SiteResult
from ..core.shells import AXIS_DISTANCE, AXIS_VALENCE, Shell

V_MIN, V_MAX = 0.004, 0.6

# Shading saturates at this width, so the darkest band is not "good" -- it is
# simply the widest that the scale distinguishes.
SHADE_FULL_DECADES = 1.0


class CutoffExplorer(QWidget):
    """CN and the running bond-valence sum of one site against threshold,
    with its plateaus and its contacts."""

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
        # The threshold the structure was first analysed with: a pin, so that
        # once the cut has been dragged the reader can still see where the
        # numbers first shown for this file came from.
        self.pin_v: float | None = None
        # The defaults Sami chose after trying the options: the distance
        # axis, the staircase and the sum in stacked panes with the area
        # under the sum filled and the next contact's step shaded, and one
        # bar per contact under the rail. Each is a View menu option.
        self.axis = AXIS_DISTANCE
        self.fill_sum = True          # area under the running sum
        self.sum_as_fraction = False  # right axis as % of |charge|
        self.stacked = True           # CN above, the sum in its own pane below
        self.show_bars = True         # one bar per contact under the rail
        self.show_band = True         # the R0 uncertainty around the sum
        self.show_econ = True         # ECoN as a line on the CN axis
        # a second site's staircase, overlaid in its own colour
        self.second: SiteResult | None = None
        # which parameter the distance axis is read against: a label such
        # as 'Bi3+-O', or None for the first parameter found
        self.axis_pair: str | None = None
        # named thresholds the cursor can jump to: {name: v}
        self.bookmarks: dict[str, float] = {}
        self.setFocusPolicy(Qt.StrongFocus)

        self._dragging = False
        self._hover_v: float | None = None
        self._theme = None

        # geometry, recomputed on every paint
        self._plot = QRectF()
        self._sum_plot = QRectF()
        self._bars = QRectF()
        self._rail = QRectF()
        self.setMinimumHeight(210 + 80 + 34)

        # the axis switch sits in the corner of the plot, by the axis it changes
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
        # Clamped to what the site has, not to a fixed 0.6: on a silicate
        # every bond lies above that, so the threshold could not be dragged to
        # where the coordination number actually changes.
        low, high = self._data_range()
        self.v_bond = float(np.clip(v, low, high))
        if emit:
            self.thresholdChanged.emit(self.v_bond)
        self.update()

    def set_pin(self, v: float | None) -> None:
        """Mark the threshold the file was first analysed with."""
        self.pin_v = None if v is None else float(v)
        self.update()

    def set_theme(self, theme) -> None:
        self._theme = theme
        self.update()

    def set_reference_distance(self, d: float | None) -> None:
        self.reference_distance = d
        self.update()

    def set_axis(self, axis: str) -> None:
        self.axis = AXIS_DISTANCE if axis == AXIS_DISTANCE else AXIS_VALENCE
        self._refresh_axis_button()
        self.update()

    def set_option(self, name: str, on: bool) -> None:
        """One of fill_sum, sum_as_fraction, stacked, show_bars, show_band,
        show_econ."""
        if name not in ("fill_sum", "sum_as_fraction", "stacked", "show_bars",
                        "show_band", "show_econ"):
            raise ValueError(f"no explorer option named {name!r}")
        setattr(self, name, bool(on))
        # the stacked panes and the bar strip need their own height, or the
        # CN pane shrinks to a few pixels and its ticks print on each other
        self.setMinimumHeight(210 + (80 if self.stacked else 0)
                              + (34 if self.show_bars else 0))
        self.updateGeometry()
        self.update()

    def set_second(self, result: SiteResult | None) -> None:
        """Overlay a second site's staircase, or (None) none."""
        self.second = result
        self.update()

    def pairs(self) -> list[str]:
        """The parameter labels the site's contacts use, first found first."""
        if self.result is None:
            return []
        out = []
        for c in self.result.contacts:
            if c.param is not None and c.param.label not in out:
                out.append(c.param.label)
        return out

    def set_axis_pair(self, label: str | None) -> None:
        """Read the distance axis against the parameter with this label."""
        self.axis_pair = label
        self.update()

    def add_bookmark(self, name: str, v: float | None = None) -> None:
        self.bookmarks[name] = float(self.v_bond if v is None else v)
        self.update()

    def remove_bookmark(self, name: str) -> None:
        self.bookmarks.pop(name, None)
        self.update()

    def jump_to(self, v: float) -> None:
        self.set_threshold(v, emit=True)

    def nudge(self, steps: int) -> None:
        """Move the threshold by ``steps`` of two per cent, in log valence."""
        self.set_threshold(self.v_bond * (1.02 ** steps), emit=True)

    def jump_plateau(self, direction: int) -> None:
        """To the middle of the next plateau up (+1) or down (-1)."""
        r = self.result
        if r is None or not r.plateaus:
            return
        ordered = sorted(r.plateaus, key=lambda q: q.v_low)
        here = next((i for i, q in enumerate(ordered) if q.contains(self.v_bond)),
                    None)
        if here is None:
            target = next((q for q in ordered if q.v_low > self.v_bond), None) \
                if direction > 0 else next(
                (q for q in reversed(ordered) if q.v_high <= self.v_bond), None)
        else:
            k = here + (1 if direction > 0 else -1)
            target = ordered[k] if 0 <= k < len(ordered) else None
        if target is None:
            return
        self.set_threshold(math.sqrt(target.v_low * target.v_high), emit=True)

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_Left:
            self.nudge(-1)
        elif key == Qt.Key_Right:
            self.nudge(+1)
        elif key == Qt.Key_Home and self.pin_v is not None:
            self.jump_to(self.pin_v)
        elif key == Qt.Key_PageUp:
            self.jump_plateau(+1)
        elif key == Qt.Key_PageDown:
            self.jump_plateau(-1)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def sensitivity(self, index: int) -> tuple[float, float]:
        """What crossing the cut would do: (change in the bond-valence sum,
        change in phi) if contact ``index`` moved to the other side."""
        r = self.result
        c = r.contacts[index]
        if not c.has_valence:
            return 0.0, 0.0
        bonded = [x for x in r.contacts if x.has_valence and x.valence > self.v_bond]
        other = [x for x in bonded if x is not c] if c in bonded else bonded + [c]

        def phi_of(rows):
            if not rows:
                return float("nan")
            return bv.phi_index(np.array([x.vector for x in rows]),
                                [x.valence * x.occupancy for x in rows])[1]

        d_sum = -c.valence * c.occupancy if c in bonded else c.valence * c.occupancy
        return d_sum, phi_of(other) - phi_of(bonded)

    def toggle_axis(self) -> None:
        self.set_axis(AXIS_DISTANCE if self.axis == AXIS_VALENCE
                      else AXIS_VALENCE)

    def _refresh_axis_button(self) -> None:
        if self.axis == AXIS_VALENCE:
            self.axis_button.setText("v.u.")
            self.axis_button.setToolTip(
                "Horizontal axis: bond valence, logarithmic. Click to read "
                "the same picture against distance.")
        else:
            self.axis_button.setText("Å")
            self.axis_button.setToolTip(
                "Horizontal axis: distance, for the pair this site is read "
                "against. Click to go back to bond valence.")

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
    def _data_range(self) -> tuple[float, float]:
        """The valences this site actually has, with a margin.

        V_MIN and V_MAX are floors, not the axis. They were chosen for Bi-O,
        whose strongest contact is about 0.5 v.u.; a silicon-oxygen bond is
        about 1.0 and a phosphorus-oxygen bond more. With a fixed top, 81 of
        143 sites in the reference collection had at least one contact mapped
        past the right-hand edge of the plot -- and since `_v_to_x` clamped only
        the low end, the tick, the step and the shading for those contacts were
        painted outside the box and cut off at the widget's edge.

        Keeping the shipped span as a floor means a site that fitted before
        still looks exactly as it did.
        """
        lo, hi = V_MIN, V_MAX
        if self.result is not None:
            values = [c.valence for c in self.result.contacts
                      if c.has_valence and c.valence > 0]
            if values:
                hi = max(hi, max(values) * 1.12)
                lo = min(lo, min(values) / 1.12)
        return lo, hi

    def _v_range(self) -> tuple[float, float]:
        """The axis as drawn: the data, the threshold and the pin wherever
        they sit.

        The threshold is included so that a value carried over from another
        site -- it is one setting for the whole structure -- cannot leave the
        marker off the end of the plot.
        """
        lo, hi = self._data_range()
        extra = [float(self.v_bond), self.pin_v] + list(self.bookmarks.values())
        if self.second is not None:
            extra += [c.valence for c in self.second.contacts
                      if c.has_valence and c.valence > 0]
        for v in extra:
            if v is not None and v > 0:
                lo, hi = min(lo, v / 1.05), max(hi, v * 1.05)
        return lo, hi

    def _d_range(self) -> tuple[float, float]:
        """The distance axis: the contacts, the threshold and the pin read
        through the dominant parameter, with a margin."""
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
        if self.axis == AXIS_DISTANCE:
            param = self._dominant_param()
            if param is None:
                return self._plot.left()
            return self._d_to_x(float(param.distance_for(max(float(v), 1e-9))))
        v_lo, v_hi = self._v_range()
        lo, hi = math.log10(v_lo), math.log10(v_hi)
        t = (math.log10(min(max(float(v), v_lo), v_hi)) - lo) / (hi - lo)
        return self._plot.left() + t * self._plot.width()

    def _d_to_x(self, d: float) -> float:
        d_lo, d_hi = self._d_range()
        t = (min(max(float(d), d_lo), d_hi) - d_lo) / max(d_hi - d_lo, 1e-9)
        return self._plot.left() + t * self._plot.width()

    def _x_to_v(self, x: float) -> float:
        t = (x - self._plot.left()) / max(self._plot.width(), 1.0)
        t = float(np.clip(t, 0.0, 1.0))
        if self.axis == AXIS_DISTANCE:
            d_lo, d_hi = self._d_range()
            param = self._dominant_param()
            if param is None:
                return self.v_bond
            return float(param.valence(d_lo + t * (d_hi - d_lo)))
        v_lo, v_hi = self._v_range()
        lo, hi = math.log10(v_lo), math.log10(v_hi)
        return float(10 ** (lo + t * (hi - lo)))

    def _shell_x(self, s: Shell) -> float:
        """Where one contact sits: at its own valence, or its own distance."""
        return (self._d_to_x(s.distance) if self.axis == AXIS_DISTANCE
                else self._v_to_x(s.valence))

    def _shells(self) -> list[Shell]:
        return shells_mod.shell_curve(self.result, self.axis)

    def estimated_contacts(self) -> set[int]:
        """Indices into the site's contacts whose bond-valence parameter is
        estimated from the O'Keeffe-Brese expression rather than fitted."""
        if self.result is None:
            return set()
        return {i for i, c in enumerate(self.result.contacts)
                if c.has_valence and c.param is not None and not c.param.fitted}

    def _dominant_param(self):
        """The bond-valence parameter to read the distance axis against."""
        if self.result is None:
            return None
        if self.axis_pair:
            for c in self.result.contacts:
                if c.param is not None and c.param.label == self.axis_pair:
                    return c.param
        for c in self.result.contacts:
            if c.param is not None:
                return c.param
        return None

    # -- vertical scales ---------------------------------------------------
    def _cn_max(self) -> int:
        r = self.result
        return max([q.cn for q in r.plateaus] + [len(self._shells()), 1])

    def _cn_to_y(self, cn: float, cn_max: int) -> float:
        """Where a coordination number sits in the plot.

        The plot rectangle is the data box exactly: CN 0 is its bottom edge and
        cn_max its top. The headroom above the top step is reserved by the
        layout instead, so that the shaded plateau bands, the gridlines and the
        staircase all share one boundary -- they did not before, and the
        shading ran 16 px above the highest gridline.
        """
        t = cn / max(cn_max, 1)
        return self._plot.bottom() - t * self._plot.height()

    def _charge(self) -> float:
        r = self.result
        return abs(r.ox) if (r is not None and r.ox is not None) else 0.0

    def _sum_value(self, total: float) -> float:
        """A running sum in the unit of the right axis: v.u., or % of the
        site's charge when that option is on and the charge is known."""
        q = self._charge()
        if self.sum_as_fraction and q > 0:
            return 100.0 * total / q
        return total

    def _sum_max(self) -> float:
        shells = self._shells()
        total = shells[-1].sum_after if shells else 0.0
        return max(self._sum_value(total), self._sum_value(self._charge()),
                   1e-6) * 1.08

    def _sum_to_y(self, s: float, s_max: float) -> float:
        return self._sum_plot.bottom() - (s / s_max) * self._sum_plot.height()

    # -- painting ----------------------------------------------------------
    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        hint = self.axis_button.sizeHint()
        self.axis_button.move(self.width() - hint.width() - 4, 2)

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
        self._paint_charge(p)
        self._paint_sum(p)
        self._paint_staircase(p)
        self._paint_contact_rail(p)
        self._paint_bars(p)
        self._paint_references(p)
        self._paint_threshold(p)
        self._paint_readout(p)
        p.end()

    def _paint_empty(self, p: QPainter) -> None:
        p.setPen(self._muted)
        p.drawText(self.rect(), Qt.AlignCenter,
                   "Select a cation site to explore its coordination")

    # Space above the highest gridline, so the top step of the staircase is
    # inside the plot rather than on its edge, and so the threshold's grab
    # handle has somewhere to sit.
    HEADROOM = 16

    def _small_font(self) -> QFont:
        fnt = QFont(self.font())
        fnt.setPointSizeF(max(7.0, fnt.pointSizeF() - 1.5))
        return fnt

    def _layout(self) -> None:
        """Size the plot, the rail and the space the labels below them need.

        Done in two passes because the two depend on each other only one way
        round: how many rows of reference labels are needed depends on where
        they fall horizontally, which depends on the plot's left edge and width
        -- and those do not depend on the rows. So the first pass fixes the
        horizontal geometry and the second spends the vertical.
        """
        f = QFontMetricsF(self.font())
        small = QFontMetricsF(self._small_font())
        top = (f.height() + 10 if self.show_distance_axis else 8) + self.HEADROOM

        provisional = f.height() * 2 + 26
        self._plot = QRectF(self.rect()).adjusted(46, top, -58, -provisional)
        rows = self._reference_rows(small)

        # rail, the bar strip if asked for, then one line per row of labels,
        # then the readout
        bars = 34 if self.show_bars else 0
        bottom = 24 + bars + rows * (small.height() - 1) + f.height() + 6
        whole = QRectF(self.rect()).adjusted(46, top, -58, -bottom)
        if self.stacked:
            # CN above, the sum below, a gap for the lower pane's headroom
            gap = 14
            h = (whole.height() - gap) / 2
            self._plot = QRectF(whole.left(), whole.top(), whole.width(), h)
            self._sum_plot = QRectF(whole.left(), whole.top() + h + gap,
                                    whole.width(), h)
        else:
            self._plot = whole
            self._sum_plot = QRectF(whole)
        self._rail = QRectF(whole.left(), whole.bottom() + 6, whole.width(), 16)
        self._bars = QRectF(whole.left(), self._rail.bottom() + 2,
                            whole.width(), bars)

    # -- layers ------------------------------------------------------------
    def _paint_plateaus(self, p: QPainter) -> None:
        """Shade each plateau in proportion to its width.

        One neutral colour whose opacity tracks the measured width, rather than
        a green/amber split. A two-colour scheme encodes a verdict -- this
        threshold is sound, that one is not -- and that is the reader's call,
        not the application's.
        """
        r = self.result
        cn_max = self._cn_max()

        for q in r.plateaus:
            x0, x1 = self._v_to_x(q.v_low), self._v_to_x(q.v_high)
            x0, x1 = min(x0, x1), max(x0, x1)
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

    def _paint_grid(self, p: QPainter) -> None:
        cn_max = self._cn_max()
        s_max = self._sum_max()
        p.setPen(QPen(self._muted, 1.0))
        p.drawLine(QPointF(self._plot.left(), self._plot.top()),
                   QPointF(self._plot.left(), self._plot.bottom()))
        p.drawLine(QPointF(self._plot.left(), self._plot.bottom()),
                   QPointF(self._plot.right(), self._plot.bottom()))
        sp = self._sum_plot
        p.setPen(QPen(self._sum_colour, 1.0))
        p.drawLine(QPointF(sp.right(), sp.top()), QPointF(sp.right(), sp.bottom()))
        if self.stacked:
            p.setPen(QPen(self._muted, 1.0))
            p.drawLine(QPointF(sp.left(), sp.top()), QPointF(sp.left(), sp.bottom()))
            p.drawLine(QPointF(sp.left(), sp.bottom()), QPointF(sp.right(), sp.bottom()))

        # ticks at least 12 px apart, whatever the pane's height
        step = max(1 if cn_max <= 8 else 2,
                   math.ceil(12.0 * cn_max / max(self._plot.height(), 1.0)))
        fnt = self._small_font()
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
        # In the headroom, not on the top tick: the highest gridline now sits
        # exactly at the top of the plot, so a caption placed there printed on
        # top of the number beside it.
        p.drawText(QRectF(2, self._plot.top() - self.HEADROOM - 2, 40, 16),
                   Qt.AlignRight | Qt.AlignVCenter, "CN")

        # the right axis: the running sum, a few round ticks
        tick = _round_step(s_max / 4)
        v = 0.0
        p.setPen(self._sum_colour)
        unit = "%" if (self.sum_as_fraction and self._charge() > 0) else ""
        while v <= s_max + 1e-9:
            y = self._sum_to_y(v, s_max)
            p.drawLine(QPointF(sp.right(), y), QPointF(sp.right() + 4, y))
            p.drawText(QRectF(sp.right() + 6, y - 8, 50, 16),
                       Qt.AlignLeft | Qt.AlignVCenter, f"{v:g}{unit}")
            v += tick
        p.drawText(QRectF(sp.right() + 6, sp.top() - self.HEADROOM - 2, 54, 16),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   "Σ / |q|" if unit else "Σ v.u.")
        p.setFont(self.font())

    def _paint_charge(self, p: QPainter) -> None:
        """The size of the site's charge, across the sum axis, for scale.

        Drawn as a line and a number, not as a target: where the running sum
        meets it is a fact about the structure and the parameters, and the
        reader draws the conclusion.
        """
        r = self.result
        if r is None or r.ox is None:
            return
        y = self._sum_to_y(self._sum_value(abs(r.ox)), self._sum_max())
        sp = self._sum_plot
        col = QColor(self._sum_colour)
        col.setAlpha(150)
        p.setPen(QPen(col, 1.0, Qt.DashLine))
        p.drawLine(QPointF(sp.left(), y), QPointF(sp.right(), y))
        # under the line, at the end where the staircases start from zero and
        # so run along the bottom: the left against distance, the right
        # against valence
        at_left = self.axis == AXIS_DISTANCE
        left = sp.left() + 4 if at_left else sp.right() - 124
        p.setFont(self._small_font())
        p.setPen(col)
        p.drawText(QRectF(left, y + 1, 120, 14),
                   (Qt.AlignLeft if at_left else Qt.AlignRight)
                   | Qt.AlignVCenter,
                   "100 % of |charge|" if self.sum_as_fraction
                   else f"|charge| {abs(r.ox)}")
        p.setFont(self.font())

    def _step_path(self, shells: list[Shell], y_of) -> QPainterPath:
        """The step function through the shells, in the order they are
        included: left to right against distance, right to left against
        valence, from zero to the full count."""
        inward = self.axis == AXIS_VALENCE
        start = self._plot.right() if inward else self._plot.left()
        end = self._plot.left() if inward else self._plot.right()
        path = QPainterPath()
        y = y_of(None)
        path.moveTo(start, y)
        for s in shells:
            x = self._shell_x(s)
            path.lineTo(x, y)
            y = y_of(s)
            path.lineTo(x, y)
        path.lineTo(end, y)
        return path

    def _paint_sum(self, p: QPainter) -> None:
        """The running bond-valence sum, on the right axis."""
        shells = self._shells()
        if not shells:
            return
        s_max = self._sum_max()
        sp = self._sum_plot

        def y_of(s):
            return self._sum_to_y(
                self._sum_value(0.0 if s is None else s.sum_after), s_max)

        path = self._step_path(shells, y_of)
        if self.fill_sum:
            # the area under the sum: how much valence the contacts counted
            # so far carry, read as an area rather than a line
            area = QPainterPath(path)
            area.lineTo(path.currentPosition().x(), sp.bottom())
            first = path.elementAt(0)
            area.lineTo(first.x, sp.bottom())
            area.closeSubpath()
            fill = QColor(self._sum_colour)
            fill.setAlpha(42)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(fill))
            p.drawPath(area)
            # the step the next contact outside the cut would add: the slice
            # between the threshold and that contact, from the sum now to the
            # sum with it, with the increment written in it
            cn, total, nxt = shells_mod.at_threshold(self.result, self.v_bond)
            if nxt is not None:
                x0 = self._v_to_x(self.v_bond)
                x1 = self._shell_x(nxt)
                y0 = self._sum_to_y(self._sum_value(total), s_max)
                y1 = self._sum_to_y(self._sum_value(nxt.sum_after), s_max)
                slab = QColor(self._sum_colour)
                slab.setAlpha(110)
                p.setBrush(QBrush(slab))
                rect = QRectF(min(x0, x1), min(y0, y1), abs(x1 - x0),
                              max(abs(y1 - y0), 2.0))
                p.drawRect(rect)
                p.setFont(self._small_font())
                p.setPen(self._sum_colour)
                add = nxt.valence * nxt.occupancy
                text = f"+{add:.3f}"
                fm = QFontMetricsF(self._small_font())
                w = fm.horizontalAdvance(text) + 4
                p.drawText(QRectF(rect.center().x() - w / 2,
                                  rect.top() - fm.height() - 1, w, fm.height()),
                           Qt.AlignCenter, text)
                p.setFont(self.font())
        if self.show_band:
            # The R0 uncertainty is systematic: it scales the whole sum, by
            # r0_error / b (0.02 Å / 0.37 Å for a fitted parameter, 0.06 for
            # an estimated one). Drawn as a band so a gap to the charge
            # smaller than it can be seen for what it is.
            param = self._dominant_param()
            if param is not None:
                rel = (0.02 if param.fitted else 0.06) / max(float(param.b), 1e-6)
                hi = self._step_path(shells, lambda s: self._sum_to_y(
                    self._sum_value((0.0 if s is None else s.sum_after) * (1 + rel)),
                    s_max))
                lo = self._step_path(shells, lambda s: self._sum_to_y(
                    self._sum_value((0.0 if s is None else s.sum_after) * (1 - rel)),
                    s_max))
                band = QPainterPath(hi)
                band.connectPath(lo.toReversed())
                band.closeSubpath()
                fill = QColor(self._sum_colour)
                fill.setAlpha(60)
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(fill))
                p.drawPath(band)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(self._sum_colour, 2.0, Qt.SolidLine, Qt.FlatCap,
                      Qt.MiterJoin))
        p.drawPath(path)

    def _paint_staircase(self, p: QPainter) -> None:
        """CN against threshold, as the step function it is."""
        shells = self._shells()
        if not shells:
            return
        cn_max = self._cn_max()
        r = self.result
        if self.show_econ and r is not None and r.cn_ecoN == r.cn_ecoN:
            # ECoN as a line on the CN axis: where Hoppe's count sits against
            # the staircase, which it never equals exactly
            y = self._cn_to_y(min(float(r.cn_ecoN), cn_max), cn_max)
            col = QColor(self._muted)
            col.setAlpha(160)
            p.setPen(QPen(col, 1.0, Qt.DashDotLine))
            p.drawLine(QPointF(self._plot.left(), y),
                       QPointF(self._plot.right(), y))
            p.setFont(self._small_font())
            p.setPen(col)
            p.drawText(QRectF(self._plot.right() - 90, y - 14, 86, 13),
                       Qt.AlignRight | Qt.AlignVCenter, f"ECoN {r.cn_ecoN:.2f}")
            p.setFont(self.font())
        if self.second is not None:
            other = shells_mod.shell_curve(self.second, self.axis)
            if other:
                col = self._c("#e0a0c8", "#a03070")
                p.setPen(QPen(col, 1.6, Qt.DashLine, Qt.FlatCap, Qt.MiterJoin))
                p.setBrush(Qt.NoBrush)
                p.drawPath(self._step_path(
                    other, lambda s: self._cn_to_y(
                        0 if s is None else s.cn_after, cn_max)))
                p.setFont(self._small_font())
                p.setPen(col)
                p.drawText(QRectF(self._plot.left() + 4, self._plot.top() + 2,
                                  200, 13), Qt.AlignLeft | Qt.AlignVCenter,
                           f"overlay: {self.second.label}")
                p.setFont(self.font())
        p.setPen(QPen(self._ink, 2.0, Qt.SolidLine, Qt.FlatCap, Qt.MiterJoin))
        p.setBrush(Qt.NoBrush)
        p.drawPath(self._step_path(
            shells, lambda s: self._cn_to_y(0 if s is None else s.cn_after,
                                            cn_max)))

    def _paint_contact_rail(self, p: QPainter) -> None:
        """Every contact as a tick at its own valence or distance.

        The steps of the staircase are these ticks. Showing them makes the
        staircase concrete rather than an abstraction, and makes it obvious when
        a plateau is narrow because two contacts are nearly coincident.
        """
        p.setPen(QPen(self._muted, 1.0))
        p.drawLine(QPointF(self._rail.left(), self._rail.center().y()),
                   QPointF(self._rail.right(), self._rail.center().y()))

        estimated = self.estimated_contacts()
        for s in self._shells():
            x = self._shell_x(s)
            bonded = s.valence >= self.v_bond
            h = self._rail.height() * (0.95 if bonded else 0.55)
            col = self._ink if bonded else self._muted
            pen = QPen(col, 2.2 if bonded else 1.2)
            if s.index in estimated:
                # an estimated parameter: the tick is dashed, and an "est."
                # sits above it, the mark the Site tab and the report use
                pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(QPointF(x, self._rail.center().y() - h / 2),
                       QPointF(x, self._rail.center().y() + h / 2))
            if s.index in estimated:
                fnt = self._small_font()
                p.setFont(fnt)
                fm = QFontMetricsF(fnt)
                p.drawText(QRectF(x - 14, self._rail.top() - fm.height() - 1,
                                  28, fm.height()), Qt.AlignHCenter, "est.")

    def _paint_bars(self, p: QPainter) -> None:
        """One bar per contact under the rail: its own contribution to the
        sum, in the order of inclusion, with the running sum above each bar
        where there is room. The integral's slope made visible term by term."""
        if not self.show_bars or self._bars.height() <= 0:
            return
        shells = self._shells()
        if not shells:
            return
        top = max(s.valence * s.occupancy for s in shells)
        if top <= 0:
            return
        fnt = self._small_font()
        fm = QFontMetricsF(fnt)
        p.setFont(fnt)
        base = self._bars.bottom() - 2
        usable = self._bars.height() - fm.height() - 4
        xs = [self._shell_x(s) for s in shells]
        gaps = [abs(b - a) for a, b in zip(xs, xs[1:])] or [1e9]
        room = min(gaps) > 30
        for s, x in zip(shells, xs):
            h = usable * (s.valence * s.occupancy) / top
            bonded = s.valence >= self.v_bond
            col = QColor(self._sum_colour)
            col.setAlpha(200 if bonded else 90)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(QRectF(x - 3, base - h, 6, h))
            if room:
                p.setPen(self._sum_colour if bonded else self._muted)
                p.drawText(QRectF(x - 24, base - h - fm.height() - 1, 48,
                                  fm.height()), Qt.AlignCenter,
                           f"{s.sum_after:.2f}")
        p.setBrush(Qt.NoBrush)
        p.setFont(self.font())

    def _reference_marks(self) -> list:
        """The thresholds and rules a reader already knows, as (v, label, col)."""
        r = self.result
        if r is None:
            return []
        marks = [(r.v_list, "tabulate", QColor("#6f8cb0"))]

        param = self._dominant_param()
        if param is not None and self.reference_distance:
            v = float(param.valence(self.reference_distance))
            v_lo, v_hi = self._v_range()
            if v_lo < v < v_hi:
                marks.append((v, f"{self.reference_distance:.2f} Å "
                                 f"convention", QColor("#b08a5a")))

        # where the max-gap rule falls: between the last included contact and
        # the first excluded one
        vals = sorted((c.valence for c in r.contacts if c.has_valence),
                      reverse=True)
        if r.cn_gap and 0 < r.cn_gap < len(vals):
            v = math.sqrt(vals[r.cn_gap - 1] * vals[r.cn_gap])
            marks.append((v, "max gap", QColor("#7f9e79")))
        # the authors' bonds: a CIF's own _geom_bond loop, read as the
        # threshold their list implies for this site (the weakest listed
        # contact), so the published coordination can be seen against the
        # staircase
        authors = self.authors_threshold()
        if authors is not None:
            marks.append((authors, "authors' bonds", self._c("#d9a066", "#8a5a1e")))
        # the pin: where the threshold stood when the file was opened, so the
        # numbers first shown for it can be found again after a drag
        if self.pin_v is not None and self.pin_v > 0:
            marks.append((self.pin_v, self._pin_label(), self._pin_colour()))
        for name, v in self.bookmarks.items():
            if v > 0:
                marks.append((v, f"⚑ {name}", self._c("#9fc5e8", "#2b6cb0")))
        return sorted(marks, key=lambda m: m[0])

    def authors_threshold(self) -> float | None:
        """The threshold a CIF's own bond list implies for this site: the
        smallest valence among the listed contacts the file bonds to it,
        matched by site label and distance; None without such a list."""
        r = self.result
        loops = getattr(self, "bond_loops", None)
        if r is None or not loops:
            return None
        listed = []
        taken: set[int] = set()
        for a, b, d in loops:
            other = b if a == r.label else a if b == r.label else None
            if other is None:
                continue
            # the nearest contact of that label not yet matched: a loop
            # lists each bond once, and two O at 1.605 and 1.611 are two
            best, best_gap = None, 0.02
            for i, c in enumerate(r.contacts):
                if i in taken or not c.has_valence or c.label != other:
                    continue
                gap = abs(c.distance - d)
                if gap < best_gap:
                    best, best_gap = i, gap
            if best is not None:
                taken.add(best)
                listed.append(r.contacts[best].valence)
        if not listed:
            return None
        return float(min(listed)) * 0.999

    def set_bond_loops(self, loops) -> None:
        self.bond_loops = list(loops or [])
        self.update()

    def _pin_label(self) -> str:
        return f"opened at {self.pin_v:.4f}"

    def _pin_colour(self) -> QColor:
        return self._c("#c58fd6", "#7b3f99")

    def _axis_end_labels(self) -> tuple[str, str]:
        """The two ends of the horizontal axis, in its own unit."""
        if self.axis == AXIS_DISTANCE:
            lo, hi = self._d_range()
            return f"{lo:.2f} Å", f"{hi:.2f} Å"
        lo, hi = self._v_range()
        return f"{lo:.3f}", f"{hi:.3f} v.u."

    def _place_references(self, fm: QFontMetricsF) -> list:
        """Lay the mark labels out, staggering the ones that collide.

        The 0.075 v.u. threshold sits at 3.05 A for Bi-O, which is almost
        exactly the 3.00 A convention, so these marks land on top of each other
        precisely in the case the figure is meant to make a point about.

        Returns (v, label, colour, left, width, row) per mark. Shared with the
        layout, which needs the row count to know how much room to leave --
        without that the second row was printed over the readout. The axis
        end labels occupy the ends of the first row, so marks avoid them.
        """
        placed: list = []
        lo_text, hi_text = self._axis_end_labels()
        ends = [(self._plot.left(), fm.horizontalAdvance(lo_text) + 6, 0),
                (self._plot.right() - fm.horizontalAdvance(hi_text) - 6,
                 fm.horizontalAdvance(hi_text) + 6, 0)]

        def overlaps(left: float, right: float, row: int) -> bool:
            boxes = [(q[3], q[3] + q[4], q[5]) for q in placed] + \
                    [(l, l + w, rw) for l, w, rw in ends]
            return any(other == row and left < pr and right > pl
                       for pl, pr, other in boxes)

        for v, label, col in self._reference_marks():
            x = self._v_to_x(v)
            w = fm.horizontalAdvance(label) + 8
            left = min(max(x - w / 2, self._plot.left()),
                       self._plot.right() - w)
            row = 0
            while row < 3 and overlaps(left, left + w, row):
                row += 1
            placed.append((v, label, col, left, w, row))
        return placed

    def _reference_rows(self, fm: QFontMetricsF) -> int:
        """How many rows of mark labels this site needs. At least one."""
        placed = self._place_references(fm)
        return max((q[5] for q in placed), default=0) + 1

    def _paint_references(self, p: QPainter) -> None:
        """Marks for the thresholds and rules a reader already knows, and the
        two ends of the axis in its unit."""
        if self.result is None:
            return
        fnt = self._small_font()
        p.setFont(fnt)
        fm = QFontMetricsF(fnt)

        lo_text, hi_text = self._axis_end_labels()
        y0 = self._bars.bottom() + 2 if self.show_bars else self._rail.bottom() + 2
        p.setPen(self._muted)
        p.drawText(QRectF(self._plot.left(), y0, 120, fm.height()),
                   Qt.AlignLeft | Qt.AlignVCenter, lo_text)
        p.drawText(QRectF(self._plot.right() - 120, y0, 120, fm.height()),
                   Qt.AlignRight | Qt.AlignVCenter, hi_text)

        for v, label, col, left, w, row in self._place_references(fm):
            x = self._v_to_x(v)
            is_pin = self.pin_v is not None and v == self.pin_v \
                and label == self._pin_label()
            if is_pin:
                # a pin: a thin solid shaft with a round head above the plot,
                # drawn differently from the dashed rules so it reads as a
                # placed marker rather than a convention
                p.setPen(QPen(col, 1.4, Qt.SolidLine))
                p.drawLine(QPointF(x, self._plot.top() - 2),
                           QPointF(x, self._plot.bottom()))
                p.setBrush(QBrush(col))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(x, self._plot.top() - 7), 4.0, 4.0)
                p.setBrush(Qt.NoBrush)
            else:
                p.setPen(QPen(col, 1.2, Qt.DashLine))
                p.drawLine(QPointF(x, self._plot.top()),
                           QPointF(x, self._plot.bottom()))
            p.setPen(col)
            y = y0 + row * (fm.height() - 1)
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
        p.setBrush(Qt.NoBrush)

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

    def readout_text(self) -> str:
        """The measurement, stated. No conclusion drawn from it."""
        r = self.result
        if r is None:
            return ""
        plateau = next((q for q in r.plateaus if q.contains(self.v_bond)), None)
        cn, total, nxt = shells_mod.at_threshold(r, self.v_bond)
        text = f"CN {cn} · Σ {total:.3f}"
        if r.bvs_listed == r.bvs_listed:
            text += f" of {r.bvs_listed:.3f} v.u."
        else:
            text += " v.u."
        text += f" at {self.v_bond:.4f}"
        if plateau is None:
            text += "  ·  on a step edge"
        else:
            text += (f"  ·  plateau {plateau.v_low:.4f}–{plateau.v_high:.4f}, "
                     f"{plateau.width_decades:.2f} dec, "
                     f"{plateau.width_angstrom:.3f} Å gap")
        if nxt is not None:
            text += (f"  ·  next {nxt.label} {nxt.distance:.3f} Å "
                     f"+{nxt.valence * nxt.occupancy:.3f}")
        return text

    def _paint_readout(self, p: QPainter) -> None:
        text = self.readout_text()
        fnt = QFont(self.font())
        fnt.setPointSizeF(max(7.5, fnt.pointSizeF() - 0.5))
        p.setFont(fnt)
        p.setPen(self._ink)
        fm = QFontMetricsF(fnt)
        width = self.width() - self._plot.left() - 6
        p.drawText(QRectF(self._plot.left(), self.height() - fm.height() - 2,
                          width, fm.height()),
                   Qt.AlignLeft | Qt.AlignVCenter,
                   fm.elidedText(text, Qt.ElideRight, width))
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
        for s in self._shells():
            dx = abs(self._shell_x(s) - x)
            if dx < best_dx:
                best, best_dx = s, dx
        if best is None:
            QToolTip.hideText()
            self.contactHovered.emit(-1)
            return
        c = r.contacts[best.index]
        state = ("above the threshold" if c.valence >= self.v_bond
                 else "below the threshold")
        note = "" if (c.param and c.param.fitted) else "  (estimated parameter)"
        d_sum, d_phi = self.sensitivity(best.index)
        crossing = (f"across the cut: Σ {d_sum:+.3f} v.u."
                    + (f", φ {d_phi:+.3f}" if d_phi == d_phi else ""))
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{r.label}–{c.label}\n{c.distance:.4f} Å\n"
            f"{c.valence:.4f} v.u. — {state}{note}\n"
            f"with it: CN {best.cn_after}, Σ {best.sum_after:.3f} v.u.\n"
            f"{crossing}", self)
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
