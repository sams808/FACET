"""The Overview page of an MD run: each question group as a card of
headline measurements, in plain sentences with units.

:func:`facet.core.md_analysis.analyse` returns hundreds of descriptors; the
Results tree lists every one, and the headline numbers drown among them.
This page puts first what the run was asked for: one card per question
group of the Setup page (:data:`~facet.ui.md_names.QUESTION_GROUPS`), each
card naming its analyses' headline measurements as sentences ('Coordination
of Si: 4.01 ± 0.001 (bond valence) · 4.00 (distance)'), a small fraction
bar for a Qⁿ or speciation distribution, and a link into the tree
positioned on that group. An analysis that produced nothing shows the
engine's reason.

Everything here is read from the ModelResult's containers (md_stats
Distribution/Histogram/Series/Scalar and the Tables); nothing is computed
anew and nothing is judged: the sentences state what the run measured,
with units, and leave the reading to the reader.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFontMetricsF, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.md_stats import Distribution, Scalar, Series
from .md_names import (
    analysis_title,
    criterion_text,
    display_name,
    group_of,
    group_titles,
    ion_text,
)
from .md_plot import style_for

__all__ = ["OverviewPage", "Headline", "analysis_headlines", "run_header",
           "FractionBar"]


# ---------------------------------------------------------------------------
# headline sentences, from the containers
# ---------------------------------------------------------------------------

@dataclass
class Headline:
    """One sentence of the overview, with an optional fraction bar
    (label, fraction) drawn beside it."""

    text: str
    bar: tuple[tuple[str, float], ...] = field(default_factory=tuple)


def _g(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}g}"


def _pm(scalar: Scalar, digits: int = 3) -> str:
    """'4.01 ± 0.001 unit' from a Scalar ('' when its mean is NaN)."""
    if not math.isfinite(scalar.mean):
        return ""
    unit = "" if scalar.unit in ("", "1") else f" {scalar.unit}"
    if math.isfinite(scalar.std) and scalar.std > 0:
        return f"{_g(scalar.mean, digits)} ± {_g(scalar.std, 2)}{unit}"
    return f"{_g(scalar.mean, digits)}{unit}"


def _pct(x: float) -> str:
    value = 100.0 * float(x)
    return f"{value:.1f} %" if 0 < value < 9.95 else f"{value:.0f} %"


def _scalar(tables, name: str) -> Scalar | None:
    c = tables.get(name)
    return c if isinstance(c, Scalar) and math.isfinite(c.mean) else None


def _fractions(dist: Distribution, label=str) -> list[tuple[str, float]]:
    """(label, mean fraction) of each category worth drawing (>= 0.5 %)."""
    total = float(np.nansum(dist.mean)) or 1.0
    out = []
    for key, mean in zip(dist.keys, dist.mean):
        share = float(mean) / total if dist.kind == "count" else float(mean)
        if math.isfinite(share) and share >= 0.005:
            out.append((label(key), share))
    return out


def _glass_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    elements = []
    for name in tables:
        match = re.fullmatch(r"mean CN ([A-Z][a-z]?) \((BV|distance)\)", name)
        if match and match[1] not in elements:
            elements.append(match[1])
    for element in elements:
        bv = _scalar(tables, f"mean CN {element} (BV)")
        dist = _scalar(tables, f"mean CN {element} (distance)")
        parts = []
        if bv is not None:
            parts.append(f"{_pm(bv)} (bond valence)")
        if dist is not None:
            parts.append(f"{_g(dist.mean)} (distance)")
        if parts:
            lines.append(Headline(f"Coordination of {element}: "
                                  + " · ".join(parts)))
    for name, container in tables.items():
        match = re.fullmatch(r"Qn ([A-Z][a-z]?) \(BV\)", name)
        if match and isinstance(container, Distribution):
            bars = _fractions(container, lambda k: f"Q{k}")
            if bars:
                top = max(bars, key=lambda b: b[1])
                lines.append(Headline(
                    f"Qⁿ of {match[1]}: mostly {top[0]} ({_pct(top[1])}; "
                    "bond-valence cut)", tuple(bars)))
    for name, container in tables.items():
        match = re.fullmatch(r"([A-Z][a-z]?) speciation \(BV\)", name)
        if match and isinstance(container, Distribution):
            bars = _fractions(container, str)
            if bars:
                text = " · ".join(f"{_pct(f)} {k}" for k, f in bars)
                lines.append(Headline(f"{match[1]} speciation: {text} "
                                      "(bond-valence cut)", tuple(bars)))
    density = _scalar(tables, "density")
    if density is not None:
        lines.append(Headline(f"Density {_pm(density, 4)}"))
    return lines


def _rings_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    modal = None
    for name, container in tables.items():
        match = re.fullmatch(r"ring sizes \(nodes\), (\w+)", name)
        if match and isinstance(container, Distribution) \
                and np.isfinite(container.mean).any():
            k = int(np.nanargmax(container.mean))
            modal = container.keys[k]
            lines.append(Headline(
                f"Most common ring size: {modal} nodes "
                f"({_pct(container.mean[k])} of the rings; "
                f"{criterion_text(match[1])})"))
    for name, container in tables.items():
        if name.startswith("R_C(n)") and isinstance(container, Series):
            if modal is not None and modal in container.axis:
                at = float(container.mean[list(container.axis).index(modal)])
                lines.append(Headline(
                    f"Rings per node R_C({modal}) = {_g(at)}"))
            break
    return lines


def _components_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    pieces = _scalar(tables, "number of components")
    largest = _scalar(tables, "largest piece, fraction of nodes")
    if pieces is not None:
        n = pieces.mean
        text = (f"The former network is {_g(n)} connected piece"
                f"{'' if n == 1 else 's (frame mean)'}")
        if largest is not None:
            text += (f"; the largest holds {_pct(largest.mean)} of the "
                     "nodes")
        lines.append(Headline(text))
    dim = tables.get("nodes by component dimensionality")
    if isinstance(dim, Distribution) and np.isfinite(dim.mean).any():
        k = int(np.nanargmax(dim.mean))
        lines.append(Headline(
            f"{_pct(dim.mean[k])} of the nodes sit in a piece of "
            f"dimensionality {dim.keys[k]}"))
    return lines


def _channels_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    for name, container in tables.items():
        match = re.fullmatch(r"percolation threshold of (\S+), any axis",
                             name)
        if match and isinstance(container, Scalar) \
                and math.isfinite(container.mean):
            ion = ion_text(match[1])
            axes = []
            for axis in ("a", "b", "c"):
                s = _scalar(tables,
                            f"percolation threshold of {match[1]} along "
                            f"{axis}")
                if s is not None:
                    axes.append(f"{axis} {_g(s.mean)}")
            tail = f" (along {' · '.join(axes)} v.u.)" if axes else ""
            lines.append(Headline(
                f"{ion} paths cross the box at mismatch "
                f"≥ {_g(container.mean)} v.u.{tail}"))
    for name, container in tables.items():
        match = re.fullmatch(
            r"accessible volume fraction of (\S+) at Delta (.+)", name)
        if match and isinstance(container, Scalar) \
                and math.isfinite(container.mean):
            lines.append(Headline(
                f"Volume open to {ion_text(match[1])} at "
                f"Δ = {match[2]}: {_pct(container.mean)} of the box"))
    low = next((c for n, c in tables.items()
                if n.startswith("lowest mismatch") and isinstance(c, Scalar)
                and math.isfinite(c.mean)), None)
    if low is not None:
        lines.append(Headline("Lowest mismatch on the grid: "
                              f"{_g(low.mean)} v.u."))
    return lines


def _modifier_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    for name, container in tables.items():
        match = re.fullmatch(r"modifier-rich anion fraction \(k_rich (\d+)\)",
                             name)
        if match and isinstance(container, Scalar) \
                and math.isfinite(container.mean):
            lines.append(Headline(
                f"Modifier-rich anions (≥ {match[1]} modifiers): "
                f"{_pct(container.mean)} of the anions"))
    clustered = _scalar(tables, "fraction of modifier atoms in a cluster")
    if clustered is not None:
        spans = []
        for axis in ("a", "b", "c"):
            s = _scalar(tables,
                        f"modifier clusters span {axis} (1 yes, 0 no)")
            if s is not None and s.mean >= 1:
                spans.append(axis)
        text = (f"{_pct(clustered.mean)} of the modifier atoms sit in a "
                "cluster")
        if spans:
            text += f"; the clusters span {', '.join(spans)}"
        lines.append(Headline(text))
    return lines


def _voids_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    fraction = _scalar(tables, "void fraction (union of the void spheres)")
    regions = _scalar(tables, "number of void regions")
    largest = _scalar(tables, "largest void region, union volume")
    if fraction is not None:
        text = f"Void volume: {_pct(fraction.mean)} of the box"
        if regions is not None:
            text += f", in {_g(regions.mean)} regions (frame mean)"
        if largest is not None:
            text += f"; the largest {_g(largest.mean)} {largest.unit}"
        lines.append(Headline(text))
    spanning = _scalar(tables, "void regions spanning an axis")
    if spanning is not None:
        lines.append(Headline(
            f"Void regions spanning an axis: {_g(spanning.mean)}"))
    return lines


def _msd_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    for name, container in tables.items():
        match = re.fullmatch(r"D ([A-Z][a-z]?)", name)
        if match and isinstance(container, Scalar) \
                and math.isfinite(container.mean):
            lines.append(Headline(
                f"D({match[1]}) = {_pm(container)} (from the MSD fit "
                "window)"))
    if lines:
        return lines
    parts, unit, t_last = [], "", None
    for name, container in tables.items():
        match = re.fullmatch(r"MSD ([A-Z][a-z]?)", name)
        if match and isinstance(container, Series) and container.axis.size:
            t_last = float(container.axis[-1])
            unit = container.value_unit
            parts.append(f"{match[1]} {_g(container.mean[-1])}")
    if parts:
        axis_unit = next((c.axis_unit for c in tables.values()
                          if isinstance(c, Series)), "")
        lines.append(Headline(
            f"MSD at the last lag ({_g(t_last)} {axis_unit}): "
            + " · ".join(parts) + f" {unit}"))
        lines.append(Headline("No fit window was given, so no diffusion "
                              "coefficient is listed"))
    return lines


def _scattering_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    for name, container in tables.items():
        match = re.fullmatch(r"FSDP position (.+)", name)
        if match and isinstance(container, Scalar) \
                and math.isfinite(container.mean):
            text = (f"First sharp diffraction peak ({match[1]}): "
                    f"Q = {_pm(container)}")
            width = _scalar(tables, f"FSDP FWHM {match[1]}")
            if width is not None:
                text += f", FWHM {_pm(width)}"
            lines.append(Headline(text))
    density = _scalar(tables, "number density")
    if density is not None:
        lines.append(Headline(f"Number density {_pm(density)}"))
    return lines


def _comparison_headlines(tables) -> list[Headline]:
    lines: list[Headline] = []
    for name, container in tables.items():
        if not name.endswith("(R_chi)"):
            continue
        for row in getattr(container, "rows", ()):
            r = row.get("R_chi")
            if r is None:
                continue
            text = f"{row.get('quantity', name)}: R_χ = {_g(r)}"
            if row.get("scale") is not None:
                text += f" (scale {_g(row['scale'])})"
            lines.append(Headline(text))
    return lines


_BUILDERS = {
    "glass": _glass_headlines,
    "rings": _rings_headlines,
    "components": _components_headlines,
    "channels": _channels_headlines,
    "modifier-density": _modifier_headlines,
    "void-regions": _voids_headlines,
    "msd": _msd_headlines,
    "scattering": _scattering_headlines,
    "scattering-comparison": _comparison_headlines,
}


def _generic_headlines(tables) -> list[Headline]:
    lines = []
    for name, container in tables.items():
        if isinstance(container, Scalar) and math.isfinite(container.mean):
            lines.append(Headline(f"{display_name(name)}: {_pm(container)}"))
        if len(lines) == 3:
            break
    return lines


def analysis_headlines(analysis: str, output) -> list[Headline]:
    """The headline sentences of one analysis, from its containers alone.
    An analysis that produced nothing gives the engine's reason; one whose
    builder finds nothing to say gives its descriptor count."""
    if output.error is not None:
        return [Headline(f"Not computed: {output.error}")]
    builder = _BUILDERS.get(analysis)
    lines = builder(output.tables) if builder is not None else []
    if not lines:
        lines = _generic_headlines(output.tables)
    if not lines:
        lines = [Headline(f"{len(output.tables)} descriptors; the tree "
                          "lists every one")]
    return lines


def run_header(result) -> str:
    """One compact line naming the run: file · frames · v_bond · formers."""
    prov = result.provenance
    frames = prov.frames_used
    parts = [Path(prov.source_path).name or str(prov.source_path),
             f"{len(frames)} frame{'s' if len(frames) != 1 else ''}"]
    if prov.v_bond_vu is not None:
        parts.append(f"v_bond {prov.v_bond_vu:g} v.u.")
    parts.append("formers " + (", ".join(sorted(prov.formers))
                               if prov.formers else "none named"))
    return " · ".join(parts)


# ---------------------------------------------------------------------------
# the widgets
# ---------------------------------------------------------------------------

class FractionBar(QWidget):
    """A one-line stacked bar of fractions, drawn with the figure palette
    (:func:`facet.ui.md_plot.style_for`). The labels are drawn inside the
    segments that fit them; every (label, fraction) is in the tooltip, so
    colour never carries the numbers alone."""

    HEIGHT = 22

    def __init__(self, fractions, parent=None):
        super().__init__(parent)
        self.fractions = [(str(k), max(float(f), 0.0)) for k, f in fractions]
        self.setFixedHeight(self.HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(" · ".join(f"{k} {100 * f:.1f} %"
                                   for k, f in self.fractions))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        width = self.width() - 1
        total = sum(f for _, f in self.fractions) or 1.0
        x = 0.0
        fm = QFontMetricsF(self.font())
        for i, (key, fraction) in enumerate(self.fractions):
            w = width * fraction / total
            colour = QColor(*style_for(i).colour)
            painter.fillRect(int(x), 2, max(int(w), 1), self.HEIGHT - 4,
                             colour)
            text = f"{key} {100 * fraction / total:.0f} %"
            if fm.horizontalAdvance(text) + 6 <= w:
                painter.setPen(Qt.white if colour.lightness() < 128
                               else Qt.black)
                painter.drawText(
                    int(x) + 3, 2, int(w) - 6, self.HEIGHT - 4,
                    Qt.AlignVCenter | Qt.AlignLeft, text)
            x += w
        painter.setPen(QPen(self.palette().mid().color()))
        painter.drawRect(0, 2, width, self.HEIGHT - 5)
        painter.end()


class _Card(QFrame):
    """One question group: its title, each analysis's headlines, and a
    link into the tree positioned on the group."""

    def __init__(self, group: str, parent=None):
        super().__init__(parent)
        self.group = group
        self.setFrameShape(QFrame.StyledPanel)
        self.sentences: list[str] = []
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(4)
        title = QLabel(group)
        title.setTextFormat(Qt.PlainText)
        font = title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        title.setFont(font)
        self._layout.addWidget(title)

    def add_analysis(self, title: str, lines: list[Headline]) -> None:
        caption = QLabel(title)
        caption.setTextFormat(Qt.PlainText)
        caption.setWordWrap(True)
        font = caption.font()
        font.setBold(True)
        caption.setFont(font)
        palette = caption.palette()
        palette.setColor(caption.foregroundRole(),
                         palette.placeholderText().color())
        caption.setPalette(palette)
        self._layout.addWidget(caption)
        for line in lines:
            label = QLabel(line.text)
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._layout.addWidget(label)
            self.sentences.append(line.text)
            if line.bar:
                self._layout.addWidget(FractionBar(line.bar))

    def finish(self, n_descriptors: int, on_link) -> None:
        link = QLabel(f'<a href="{self.group}">{n_descriptors} '
                      'descriptors →</a>')
        link.setTextFormat(Qt.RichText)
        link.setToolTip("Open this group in the descriptor tree.")
        link.linkActivated.connect(on_link)
        self._layout.addWidget(link)


class OverviewPage(QWidget):
    """The run's headline measurements, one card per question group.

    ``set_result`` fills it from a ModelResult (None clears it);
    ``groupRequested(title)`` fires when a card's descriptor link is
    clicked, so the browser can put the tree on that group."""

    groupRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self._scroll)
        self._body = QWidget()
        self._column = QVBoxLayout(self._body)
        self._column.setContentsMargins(10, 10, 10, 10)
        self._column.setSpacing(8)
        self._scroll.setWidget(self._body)
        self.header = QLabel()
        self.header.setTextFormat(Qt.PlainText)
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._column.addWidget(self.header)
        self._cards: dict[str, _Card] = {}
        self._column.addStretch(1)

    def set_result(self, result) -> None:
        for card in self._cards.values():
            card.setParent(None)
            card.deleteLater()
        self._cards = {}
        if result is None:
            self.header.setText("")
            return
        self.header.setText(run_header(result))
        by_group: dict[str, list[str]] = {}
        for analysis in result.outputs:
            by_group.setdefault(group_of(analysis), []).append(analysis)
        for group in group_titles():
            analyses = by_group.get(group)
            if not analyses:
                continue
            card = _Card(group)
            count = 0
            for analysis in analyses:
                output = result.outputs[analysis]
                count += len(output.tables)
                card.add_analysis(analysis_title(analysis),
                                  analysis_headlines(analysis, output))
            card.finish(count, self.groupRequested.emit)
            # before the trailing stretch
            self._column.insertWidget(self._column.count() - 1, card)
            self._cards[group] = card

    def card_groups(self) -> list[str]:
        """The group titles shown, in order."""
        return list(self._cards)

    def card_sentences(self, group: str) -> list[str]:
        """The headline sentences of one card."""
        card = self._cards.get(group)
        return list(card.sentences) if card is not None else []

    def visible_texts(self) -> list[str]:
        """Every text this page shows, for a reader checking the wording."""
        texts = [self.header.text()]
        for card in self._cards.values():
            texts.append(card.group)
            texts.extend(card.sentences)
        for widget in self.findChildren(QWidget):
            texts.append(widget.toolTip())
            text = getattr(widget, "text", None)
            if callable(text):
                value = text()
                if isinstance(value, str):
                    texts.append(value)
        return [t for t in texts if t]
