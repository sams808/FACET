"""Plots of frame-averaged MD descriptors, drawn with QPainter.

An MD descriptor is a mean over frames with a spread (``md_stats``): a
:class:`~facet.core.md_stats.Distribution` (categories), a
:class:`~facet.core.md_stats.Histogram` (bins), a
:class:`~facet.core.md_stats.Series` (a function of r, Q, t...) or a
:class:`~facet.core.md_stats.Scalar`. :class:`StatPlot` draws them with the
spread visible: bars with whiskers, a step line or a curve with a band of
mean +- std. The band is the sample standard deviation across frames
(ddof = 1, ``md_stats.STD_DDOF``), not a standard error, and every figure
that draws one says so under the axes.

It extends :class:`facet.ui.plot.Plot` (pan, wheel zoom, double-click to
fit, the theme colours, the tick step rule) and adds what that line plot
does not have: logarithmic axes, categorical axes, bands, bars, whiskers,
markers, a legend outside the data, and exports at a stated physical size:
SVG and PDF as vectors, PNG at 600 dpi. On screen one drawing unit is one
pixel; in an export one unit is one point (1/72 in), so a curve drawn
1.1 units wide is 0.39 mm wide on paper whatever the resolution, and a pen
is never a device hairline (``Plot.save_pdf`` uses cosmetic pens, which
come out 1 device pixel wide at 600 dpi).

Colour follows Okabe & Ito (2008), *Color Universal Design*, the palette
FACET's "High contrast" element palette is built from
(``theme.PALETTES['High contrast']``). Colour never carries a meaning alone:
each series also has its own marker shape and dash pattern, and each bar
series its own hatch, so a figure printed in grey or read with a
colour-vision deficiency still tells the series apart. Yellow is kept last
and out of the line cycle: on white it has a contrast ratio near 1.2:1.

Nothing here interprets a value. It draws what it is given, with its unit.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QApplication

from ..core.md_stats import STD_DDOF, Distribution, Histogram, Scalar, Series
from .plot import Plot

__all__ = [
    "OKABE_ITO", "LINE_COLOURS", "MARKERS", "DASHES", "HATCHES",
    "SeriesStyle", "style_for", "StatSeries", "Figure", "StatPlot",
    "axis_label", "unit_text", "axis_symbol", "category_label",
    "is_log_spaced", "figure_for", "figure_for_family",
    "figure_for_comparison", "figure_for_fractions", "figure_for_table",
    "SPREAD_ANNOTATION", "EXPORT_WIDTH_MM", "EXPORT_HEIGHT_MM", "EXPORT_DPI",
    "SAVE_FIGURE_TEXT", "FIGURE_FILE_FILTER", "file_stem", "ask_figure_path",
]

# ---------------------------------------------------------------------------
# palette and series styles
# ---------------------------------------------------------------------------

# Okabe & Ito (2008), Color Universal Design (CUD), https://jfly.uni-koeln.de/
# color/ : the eight colours with their published RGB values. theme.py's
# "High contrast" element palette holds the first six and yellow (N, O, Cl,
# Bi, P, F, S there) to two decimals, within 4/255 of these values;
# tests/test_md_views.py checks it.
OKABE_ITO: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("blue", (0, 114, 178)),
    ("vermillion", (213, 94, 0)),
    ("bluish green", (0, 158, 115)),
    ("reddish purple", (204, 121, 167)),
    ("orange", (230, 159, 0)),
    ("sky blue", (86, 180, 233)),
    ("black", (0, 0, 0)),
    ("yellow", (240, 228, 66)),
)
# The cycle lines and bars take: every colour but yellow.
LINE_COLOURS = OKABE_ITO[:7]
# Seven markers and five dashes: 7 and 5 share no factor, so the first 35
# series all differ in (marker, dash), and the marker changes with the colour.
MARKERS = ("circle", "square", "triangle", "diamond", "triangle down",
           "cross", "plus")
DASHES = ("solid", "dash", "dot", "dash dot", "dash dot dot")
HATCHES = ("solid", "diagonal", "back diagonal", "cross", "horizontal",
           "vertical")
_DASH_STYLE = {"solid": Qt.SolidLine, "dash": Qt.DashLine, "dot": Qt.DotLine,
               "dash dot": Qt.DashDotLine,
               "dash dot dot": Qt.DashDotDotLine}

# The page an export is drawn on unless the caller states another one: a
# two-column journal figure, 180 mm x 110 mm, and 600 dpi for a raster.
EXPORT_WIDTH_MM = 180.0
EXPORT_HEIGHT_MM = 110.0
EXPORT_DPI = 600

SAVE_FIGURE_TEXT = "Save figure\u2026 (SVG, PDF, 600 dpi PNG)"
FIGURE_FILE_FILTER = ("SVG, vector (*.svg);;PDF, vector (*.pdf);;"
                      "PNG, 600 dpi (*.png)")


def file_stem(text: str, fallback: str = "figure") -> str:
    """A file name made of a title: the characters a file name takes."""
    return re.sub(r"[^A-Za-z0-9.+=()-]+", "_", str(text)).strip("_") \
        or fallback


def ask_figure_path(parent, title: str, stem: str) -> str | None:
    """The save dialog of a figure: SVG, PDF or a 600 dpi PNG. Returns the
    path with its suffix (added from the filter chosen when typed without
    one), or None when the dialog was closed."""
    from PySide6.QtWidgets import QFileDialog

    path, chosen = QFileDialog.getSaveFileName(parent, title, f"{stem}.svg",
                                               FIGURE_FILE_FILTER)
    if not path:
        return None
    if Path(path).suffix.lower() not in (".svg", ".pdf", ".png"):
        path += {"PDF": ".pdf", "PNG": ".png"}.get(str(chosen)[:3], ".svg")
    return path


SPREAD_ANNOTATION = (f"Bands and whiskers: mean \u00b1 sample standard deviation "
                     f"across frames (ddof = {STD_DDOF}); a spread, not a "
                     f"standard error, since the frames of one trajectory "
                     f"are correlated in time.")


@dataclass(frozen=True)
class SeriesStyle:
    """How one series is drawn: colour, marker, dash and hatch together."""

    colour: tuple[int, int, int]
    colour_name: str
    marker: str
    dash: str
    hatch: str


def style_for(index: int) -> SeriesStyle:
    """The style of the ``index``-th series of a figure.

    Colour and marker cycle together over seven, the dash over five and the
    hatch over six, so two series of one figure never share both their
    marker and their dash (up to 35 series)."""
    i = int(index)
    name, rgb = LINE_COLOURS[i % len(LINE_COLOURS)]
    return SeriesStyle(rgb, name, MARKERS[i % len(MARKERS)],
                       DASHES[i % len(DASHES)], HATCHES[i % len(HATCHES)])


# ---------------------------------------------------------------------------
# what a figure holds
# ---------------------------------------------------------------------------

KINDS = ("line", "step", "bars", "points")


@dataclass
class StatSeries:
    """One mean with its spread.

    ``kind``: 'line' (a curve through the points), 'step' (a histogram:
    ``edges`` hold n + 1 bin edges for n means), 'bars' (categories at
    ``x`` = 0, 1, ...) or 'points' (markers, no line). ``std`` None or NaN:
    no band, no whisker there.
    """

    label: str
    x: np.ndarray
    mean: np.ndarray
    std: np.ndarray | None = None
    kind: str = "line"
    edges: np.ndarray | None = None
    style: SeriesStyle | None = None
    band: bool = True
    whiskers: bool = True
    markers: bool = True
    visible: bool = True

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"series {self.label!r}: kind {self.kind!r} is "
                             f"not one of {KINDS}")
        self.x = np.asarray(self.x, dtype=np.float64).ravel()
        self.mean = np.asarray(self.mean, dtype=np.float64).ravel()
        if self.std is not None:
            self.std = np.asarray(self.std, dtype=np.float64).ravel()
            if self.std.shape != self.mean.shape:
                raise ValueError(f"series {self.label!r}: {self.std.size} "
                                 f"std values for {self.mean.size} means")
        if self.kind == "step":
            if self.edges is None:
                raise ValueError(f"series {self.label!r}: a step series "
                                 "needs its bin edges")
            self.edges = np.asarray(self.edges, dtype=np.float64).ravel()
            if self.edges.size != self.mean.size + 1:
                raise ValueError(f"series {self.label!r}: {self.edges.size} "
                                 f"edges for {self.mean.size} bins")
        elif self.x.shape != self.mean.shape:
            raise ValueError(f"series {self.label!r}: {self.x.size} x values "
                             f"for {self.mean.size} means")

    @property
    def has_spread(self) -> bool:
        return self.std is not None and bool(np.isfinite(self.std).any())


@dataclass
class Figure:
    """Everything one :class:`StatPlot` draws.

    Every axis has a name and a unit; ``axis_label`` joins them. A
    categorical x axis (``categories``) has the unit 'category'.
    ``annotations`` are drawn under the axes (a definition, a presentation
    choice); ``notes`` are not drawn, the browser lists them. ``vlines``:
    (x, label, kind) vertical marks, kind 'current' (solid) or 'reference'
    (dashed).
    """

    series: list[StatSeries]
    x_name: str
    x_unit: str
    y_name: str
    y_unit: str
    title: str = ""
    categories: tuple[str, ...] | None = None
    log_x: bool = False
    log_y: bool = False
    annotations: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    vlines: tuple[tuple[float, str, str], ...] = ()

    def __post_init__(self) -> None:
        for what, unit in (("x", self.x_unit), ("y", self.y_unit)):
            if not isinstance(unit, str) or not unit.strip():
                raise ValueError(f"the {what} axis needs a unit ('1' when "
                                 "dimensionless, 'category' for categories)")
        for k, s in enumerate(self.series):
            if s.style is None:
                s.style = style_for(k)

    @property
    def x_label(self) -> str:
        return axis_label(self.x_name, self.x_unit)

    @property
    def y_label(self) -> str:
        return axis_label(self.y_name, self.y_unit)


def unit_text(unit: str) -> str:
    """'1' and '' read 'dimensionless'; any other unit is shown as given."""
    unit = (unit or "").strip()
    return "dimensionless" if unit in ("", "1") else unit


def axis_label(name: str, unit: str) -> str:
    """'name (unit)', the form every axis of these plots takes."""
    return f"{name} ({unit_text(unit)})"


# axis names of md_stats.Series (the variable with its unit, FACET style) and
# the symbol an axis label shows for them
_AXIS_SYMBOLS = {
    "r_ang": "r", "q_inv_ang": "Q", "k_inv_ang": "k", "t_ps": "t",
    "t_fs": "t", "freq_thz": "frequency", "wavenumber_inv_cm": "wavenumber",
    "delta_ppm": "δ", "n_ring_nodes": "ring size n",
    "k_shell": "shell k", "energy_ev": "E",
}
_UNIT_SUFFIXES = {"ang", "ps", "fs", "thz", "ppm", "deg", "cm", "ev", "inv",
                  "ang2", "ang3", "vu", "nm"}


def axis_symbol(axis_name: str) -> str:
    """The symbol shown for a Series axis ('r_ang' -> 'r')."""
    if axis_name in _AXIS_SYMBOLS:
        return _AXIS_SYMBOLS[axis_name]
    parts = axis_name.split("_")
    while len(parts) > 1 and parts[-1].lower() in _UNIT_SUFFIXES:
        parts.pop()
    return " ".join(parts)


def category_label(key) -> str:
    """A Distribution key as text: tuples joined with '-' (md_stats' rule)."""
    if isinstance(key, tuple):
        return "-".join(category_label(k) for k in key)
    return str(key)


def is_log_spaced(axis) -> bool:
    """True for a positive axis whose points are spaced by a constant ratio
    over at least 1.5 decades (log-spaced lag times, for instance): such an
    axis is drawn logarithmic. A presentation choice; the plot can switch."""
    values = np.asarray(axis, dtype=np.float64)
    if values.size < 8 or not np.all(np.isfinite(values)) or values[0] <= 0:
        return False
    ratios = np.log(values[1:] / values[:-1])
    if np.any(ratios <= 0):
        return False
    if math.log10(values[-1] / values[0]) < 1.5:
        return False
    return float(np.std(ratios) / np.mean(ratios)) < 0.25


# names whose curves are conventionally read on log-log axes (a ballistic
# then diffusive regime is a slope change there); a presentation default
_LOG_LOG_PREFIXES = ("MSD", "<dr^2>", "<dr^4>", "charge-displacement MSD")


# ---------------------------------------------------------------------------
# figures from md_stats containers
# ---------------------------------------------------------------------------

MAX_CATEGORIES = 40


def _spread_note(series: Sequence[StatSeries]) -> tuple[str, ...]:
    return (SPREAD_ANNOTATION,) if any(s.has_spread for s in series) else ()


def _bars_from_distributions(dists: Sequence[Distribution],
                             labels: Sequence[str]):
    """Grouped bars over the union of the keys (missing key: mean 0)."""
    keys: list = []
    seen = set()
    for d in dists:
        for k in d.keys:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    try:
        keys.sort(key=_key_rank)
    except TypeError:
        pass
    means = np.zeros((len(dists), len(keys)))
    stds = np.full((len(dists), len(keys)), np.nan)
    for r, d in enumerate(dists):
        index = {k: c for c, k in enumerate(d.keys)}
        for c, k in enumerate(keys):
            if k in index:
                means[r, c] = d.mean[index[k]]
                stds[r, c] = d.std[index[k]]
    annotations = []
    if len(keys) > MAX_CATEGORIES:
        largest = np.nanmax(means, axis=0)
        order = np.argsort(-np.nan_to_num(largest, nan=-np.inf),
                           kind="stable")[:MAX_CATEGORIES]
        order = np.sort(order)
        annotations.append(f"The {MAX_CATEGORIES} of {len(keys)} categories "
                           "with the largest mean are drawn; the table lists "
                           "every one.")
        keys = [keys[i] for i in order]
        means, stds = means[:, order], stds[:, order]
    x = np.arange(len(keys), dtype=np.float64)
    series = [StatSeries(label, x, means[r], stds[r], kind="bars")
              for r, label in enumerate(labels)]
    return series, tuple(category_label(k) for k in keys), annotations


def _key_rank(key):
    if isinstance(key, (bool, int, float, np.integer, np.floating)):
        return (0, float(key), "")
    if isinstance(key, str):
        return (1, 0.0, key)
    if isinstance(key, tuple):
        return (2, 0.0, repr(tuple(_key_rank(k) for k in key)))
    return (3, 0.0, repr(key))


def _distribution_axes(dist: Distribution) -> tuple[str, str]:
    if dist.kind == "fraction":
        return "fraction", "1"
    return f"number per {dist.row_kind}", "count"


def _histogram_axes(hist: Histogram) -> tuple[str, str]:
    if hist.density:
        return "probability density", hist.value_unit
    return f"samples per bin, per {hist.row_kind}", hist.value_unit


def figure_for(container, *, title: str | None = None) -> Figure:
    """The figure of one md_stats container.

    Distribution: bars with +- std whiskers. Histogram: a step line with a
    +- std band. Series: a line with a +- std band, on log axes when its
    axis is log-spaced or its name is an MSD. Scalar: the value in each
    frame, with the mean +- std band over them. Anything else: ValueError.
    """
    name = title or container.name
    if isinstance(container, Distribution):
        series, categories, extra = _bars_from_distributions(
            [container], [name])
        y_name, y_unit = _distribution_axes(container)
        return Figure(series, name, "category", y_name, y_unit,
                      title=name, categories=categories,
                      annotations=tuple(extra) + _spread_note(series),
                      notes=tuple(container.notes))
    if isinstance(container, Histogram):
        s = StatSeries(name, container.centres, container.mean,
                       container.std, kind="step", edges=container.edges)
        y_name, y_unit = _histogram_axes(container)
        return Figure([s], name, container.unit or "1", y_name,
                      y_unit, title=name,
                      annotations=_spread_note([s]),
                      notes=tuple(container.notes))
    if isinstance(container, Series):
        s = StatSeries(name, container.axis, container.mean,
                       container.std, kind="line",
                       markers=container.axis.size <= 60)
        log_x, log_y = _log_defaults([container])
        return Figure([s], axis_symbol(container.axis_name),
                      container.axis_unit or "1", name,
                      container.value_unit or "1",
                      title=name, log_x=log_x,
                      log_y=log_y, annotations=_spread_note([s]),
                      notes=tuple(container.notes))
    if isinstance(container, Scalar):
        frames = np.asarray(container.frames, dtype=np.float64)
        values = np.asarray(container.per_frame, dtype=np.float64)
        points = StatSeries(f"{name}, each {container.row_kind}",
                            frames, values, None, kind="points")
        span = (frames.min(), frames.max()) if frames.size > 1 else \
            (frames[0] - 0.5, frames[0] + 0.5)
        std = container.std if math.isfinite(container.std) else float("nan")
        mean = StatSeries(f"mean \u00b1 std over {container.n_frames} "
                          f"{container.row_kind}s",
                          np.array(span), np.array([container.mean] * 2),
                          np.array([std] * 2), kind="line", markers=False)
        series = [points, mean]
        series[0].style = style_for(0)
        series[1].style = replace(style_for(1), dash="dash")
        return Figure(series, container.row_kind, "index", name,
                      container.unit or "1", title=name,
                      annotations=_spread_note(series),
                      notes=tuple(container.notes))
    raise ValueError(f"no figure is drawn for {type(container).__name__}")


def _log_defaults(series: Sequence[Series]) -> tuple[bool, bool]:
    first = series[0]
    # an MSD starts at t = 0, which a log axis leaves out (and counts under
    # the plot): what decides is that the axis runs over positive times
    axis = np.asarray(first.axis, dtype=np.float64)
    log_log = all(s.name.startswith(_LOG_LOG_PREFIXES) for s in series) and \
        first.axis_name.startswith("t_") and bool((axis > 0).any()) \
        and float(np.nanmin(axis)) >= 0
    log_x = log_log or all(is_log_spaced(s.axis) for s in series)
    return log_x, log_log


def figure_for_family(containers: Sequence, *, title: str,
                      y_name: str | None = None,
                      labels: Sequence[str] | None = None) -> Figure | None:
    """One figure overlaying several containers of one kind and unit
    (the partial g(r), the CN of each element...); None when they cannot
    share axes. Scalars give one bar per descriptor with its whisker."""
    items = list(containers)
    if len(items) < 2:
        return figure_for(items[0], title=title) if items else None
    kinds = {type(c) for c in items}
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    labels = list(labels) if labels is not None else [c.name for c in items]
    if len(labels) != len(items):
        raise ValueError("one label per container is needed")
    if kind is Series:
        if len({(c.axis_name, c.axis_unit, c.value_unit) for c in items}) != 1:
            return None
        series = [StatSeries(label, c.axis, c.mean, c.std, kind="line",
                             markers=True) for label, c in zip(labels, items)]
        log_x, log_y = _log_defaults(items)
        first = items[0]
        return Figure(series, axis_symbol(first.axis_name),
                      first.axis_unit or "1", y_name or title,
                      first.value_unit or "1", title=title, log_x=log_x,
                      log_y=log_y, annotations=_spread_note(series))
    if kind is Histogram:
        if len({(c.unit, c.value_unit, c.density) for c in items}) != 1:
            return None
        series = [StatSeries(label, c.centres, c.mean, c.std, kind="step",
                             edges=c.edges) for label, c in zip(labels, items)]
        y, unit = _histogram_axes(items[0])
        return Figure(series, title, items[0].unit or "1", y, unit,
                      title=title, annotations=_spread_note(series))
    if kind is Distribution:
        if len({c.kind for c in items}) != 1:
            return None
        series, categories, extra = _bars_from_distributions(items, labels)
        y, unit = _distribution_axes(items[0])
        return Figure(series, title, "category", y, unit, title=title,
                      categories=categories,
                      annotations=tuple(extra) + _spread_note(series))
    if kind is Scalar:
        if len({c.unit for c in items}) != 1:
            return None
        x = np.arange(len(items), dtype=np.float64)
        s = StatSeries(title, x, [c.mean for c in items],
                       [c.std for c in items], kind="bars")
        return Figure([s], "descriptor", "category", y_name or title,
                      items[0].unit or "1", title=title,
                      categories=tuple(labels),
                      annotations=_spread_note([s]))
    return None


def _rows(table) -> list[dict]:
    if hasattr(table, "as_rows"):
        return list(table.as_rows())
    return [dict(r) for r in table]


def figure_for_comparison(table, summary=None) -> Figure | None:
    """A measured curve with the scaled model curve and their difference
    (md_analysis scattering-comparison rows), and the R-factor with its
    definition under the axes. None when the rows are not such a table."""
    rows = _rows(table)
    if not rows:
        return None
    keys = list(rows[0])
    # the columns as md_analysis names them, with or without their unit:
    # 'measured (1)' or 'measured', 'model, scaled (1)' or 'model (scaled)'
    measured_key = next((k for k in keys if k == "measured"
                         or k.startswith("measured (")), None)
    model_key = next((k for k in keys if k == "model (scaled)"
                      or k.startswith("model, scaled")), None)
    axis = next((k for k in ("q_inv_ang", "r_ang") if k in rows[0]), None)
    if measured_key is None or model_key is None or axis is None:
        return None
    unit_match = re.match(r"^measured \(([^()]+)\)$", measured_key)
    x = np.array([r[axis] for r in rows], dtype=np.float64)
    measured = np.array([r[measured_key] for r in rows], dtype=np.float64)
    model = np.array([r[model_key] for r in rows], dtype=np.float64)
    order = np.argsort(x, kind="stable")
    x, measured, model = x[order], measured[order], model[order]
    quantity = str(rows[0].get("quantity", "curve"))
    series = [StatSeries("measured", x, measured, kind="points",
                         markers=True),
              StatSeries("model (scaled)", x, model, kind="line",
                         markers=False),
              StatSeries("measured - model", x, measured - model,
                         kind="line", markers=False)]
    series[0].style = style_for(6)          # black, the data
    series[1].style = replace(style_for(1), dash="solid")
    series[2].style = replace(style_for(0), dash="dash")
    annotations = []
    for row in (_rows(summary) if summary is not None else []):
        r_chi = row.get("R_chi")
        if r_chi is None:
            continue
        unit = "1/Å" if axis == "q_inv_ang" else "Å"
        low = next((v for k, v in row.items() if k.startswith("range low")),
                   None)
        high = next((v for k, v in row.items() if k.startswith("range high")),
                    None)
        text = f"R_chi = {_number(r_chi)}"
        if row.get("points compared") is not None:
            text += f" over {row['points compared']} measured points"
        if low is not None and high is not None:
            text += f", {_number(low)} to {_number(high)} {unit}"
        if row.get("scale") is not None:
            text += (f"; model scale {_number(row['scale'])} "
                     f"({row.get('scale source', 'stated')})")
        annotations.append(text + ".")
        if row.get("R_chi definition"):
            annotations.append(f"Definition: {row['R_chi definition']}")
        if row.get("measured file"):
            annotations.append(f"Measured file: {row['measured file']}")
    symbol = "Q" if axis == "q_inv_ang" else "r"
    unit = "1/Å" if axis == "q_inv_ang" else "Å"
    if unit_match is not None:
        y_unit = unit_match.group(1)
    else:
        y_unit = "1" if quantity.startswith("S(Q)") else "as measured"
    return Figure(series, symbol, unit, quantity, y_unit,
                  title=f"{quantity}: measured and model",
                  annotations=tuple(annotations))


def figure_for_fractions(table) -> Figure | None:
    """Model fractions (mean +- std across frames) beside measured ones
    (+- the stated uncertainty), from md_analysis nmr-comparison rows."""
    rows = _rows(table)
    need = {"key", "model mean", "measured"}
    if not rows or not need <= set(rows[0]):
        return None
    keys = [str(r["key"]) for r in rows]
    x = np.arange(len(rows), dtype=np.float64)
    model = StatSeries(str(rows[0].get("model", "model")), x,
                       [r["model mean"] for r in rows],
                       [r.get("model std (across frames)", np.nan)
                        for r in rows], kind="bars")
    measured = StatSeries("measured, \u00b1 stated uncertainty", x,
                          [r["measured"] for r in rows],
                          [r.get("measured uncertainty", np.nan)
                           for r in rows], kind="bars")
    descriptor = str(rows[0].get("descriptor", "fractions"))
    notes = []
    if rows[0].get("measured source"):
        notes.append(f"Measured: {rows[0]['measured source']}.")
    notes.append("Model whiskers: \u00b1 sample std across frames (ddof = "
                 f"{STD_DDOF}); measured whiskers: the uncertainty stated "
                 "with the measurement.")
    return Figure([model, measured], descriptor, "category", "fraction", "1",
                  title=f"{descriptor}: model and measured",
                  categories=tuple(keys), annotations=tuple(notes))


def figure_for_table(table, summary=None) -> Figure | None:
    """A figure for a Table when its columns are a known comparison."""
    return figure_for_comparison(table, summary) or figure_for_fractions(table)


def _number(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.4g}"


# ---------------------------------------------------------------------------
# the plot widget
# ---------------------------------------------------------------------------

@dataclass
class _Metrics:
    """Sizes in drawing units: pixels on screen, points in an export."""

    line: float
    thin: float
    marker: float
    cap: float
    tick: float
    gap: float
    band_alpha: int
    hatch_step: float


_SCREEN = _Metrics(line=1.6, thin=0.8, marker=6.5, cap=4.0, tick=4.0, gap=6.0,
                   band_alpha=58, hatch_step=5.0)
_PRINT = _Metrics(line=1.1, thin=0.45, marker=4.2, cap=2.6, tick=2.8,
                  gap=4.0, band_alpha=48, hatch_step=3.2)
_EXPORT_FONT_PT = 8
# the most of the figure height a rotated category label may take
_CATEGORY_SHARE = 0.28


@dataclass
class _Layout:
    rect: QRectF
    x_ticks: list
    y_ticks: list
    x_minor: list
    y_minor: list
    rotate_categories: bool
    category_height: float
    x_title_top: float
    legend: QRectF | None
    legend_rows: list
    annotation_lines: list
    footnote_lines: list
    title_height: float
    font: QFont
    fm: QFontMetricsF
    # the y-axis title, on one line or, when longer than the axis, two
    y_label_lines: list = field(default_factory=list)


def _split_title(text: str, fm: QFontMetricsF, room: float) -> list[str]:
    """An axis title as one line, or as the two lines of most even width
    when it is longer than ``room``: the unit ends the title, and an
    elided title lost it (the threshold panel's 'fraction of the element's
    atoms per bin (dimensionless)' showed as 'fraction of the element's
    atoms per...' on a 230 px axis)."""
    text = str(text or "")
    if not text or fm.horizontalAdvance(text) <= room:
        return [text] if text else []
    words = text.split()
    if len(words) < 2:
        return [text]
    best = None
    for k in range(1, len(words)):
        first, second = " ".join(words[:k]), " ".join(words[k:])
        widest = max(fm.horizontalAdvance(first),
                     fm.horizontalAdvance(second))
        if best is None or widest < best[0]:
            best = (widest, [first, second])
    return best[1]


def _legend_labels(labels: Sequence[str], title: str) -> list[str]:
    """The legend's texts: each label without the words every label shares
    at its start and its end when the title states them, so that what
    differs is what the legend shows. 'S(Q) Na-O (Faber-Ziman)' under the
    title 'S(Q) *-* (Faber-Ziman)' showed as 'S(Q) Na-O (Faber-Zi...'; it
    is 'Na-O' now. The labels themselves, which the rows and the exports
    carry, are unchanged; any other set of labels is returned as given."""
    labels = [str(t) for t in labels]
    if len(labels) < 2:
        return labels
    first = labels[0]
    head = 0
    while head < len(first) and all(len(t) > head and t[head] == first[head]
                                    for t in labels):
        head += 1
    tail = 0
    while tail < len(first) and all(len(t) > tail and
                                    t[-1 - tail] == first[-1 - tail]
                                    for t in labels):
        tail += 1
    # whole words only: the shared start up to its last space, the shared
    # end from its first space
    start = first[:head]
    start = start[:start.rfind(" ") + 1] if " " in start else ""
    end = first[len(first) - tail:] if tail else ""
    end = end[end.find(" "):] if " " in end else ""
    if not (start.strip() or end.strip()):
        return labels
    if (start.strip() and start.strip() not in title) or \
            (end.strip() and end.strip() not in title):
        return labels
    shown = [t[len(start):len(t) - len(end)].strip() for t in labels]
    if any(not t for t in shown) or len(set(shown)) != len(shown):
        return labels
    return shown


class StatPlot(Plot):
    """A :class:`Figure` on screen and in SVG, PDF and 600 dpi PNG files.

    Interaction as :class:`~facet.ui.plot.Plot`: drag pans, the wheel zooms
    (with Ctrl, the y axis only), a double-click fits. With ``pick_mode``
    a click or drag emits ``picked`` (data coordinates) instead of panning,
    for a plot that sets a value, such as a threshold.
    """

    # what a save from the context menu wrote, or why it did not
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.figure: Figure | None = None
        self.log_x = False
        self.log_y = False
        self.pick_mode = False
        self.empty_text = "Nothing to draw."
        self.file_stem = ""
        self._category_room = 200.0
        self._picking = False
        self._dropped: dict[str, int] = {"x": 0, "y": 0}
        # every plot, wherever it sits (the result browser, the threshold
        # panel), saves itself from its context menu
        self.save_figure_action = QAction(SAVE_FIGURE_TEXT, self)
        self.save_figure_action.setToolTip(
            "Write this figure as SVG or PDF (vector) or as a 600 dpi PNG, "
            f"{EXPORT_WIDTH_MM:g} mm x {EXPORT_HEIGHT_MM:g} mm.")
        self.save_figure_action.setEnabled(False)
        self.save_figure_action.triggered.connect(self.choose_save)
        self.addAction(self.save_figure_action)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

    # -- content -----------------------------------------------------------
    def set_figure(self, figure: Figure | None, keep_view: bool = False
                   ) -> None:
        self.figure = figure
        self.save_figure_action.setEnabled(figure is not None)
        if figure is not None:
            self.log_x, self.log_y = figure.log_x, figure.log_y
            self.x_label, self.y_label = figure.x_label, figure.y_label
            self.title = figure.title
            self.series = []
        if not keep_view:
            self.fit()
        self.update()

    def set_log(self, x: bool | None = None, y: bool | None = None) -> None:
        """Switch either axis between linear and logarithmic, and refit."""
        if x is not None:
            self.log_x = bool(x)
        if y is not None:
            self.log_y = bool(y)
        if self.figure is not None and self.figure.categories is not None:
            self.log_x = False
        self.fit()
        self.update()

    def set_vlines(self, vlines) -> None:
        """Vertical marks (x, label, 'current' | 'reference'), in data x."""
        if self.figure is not None:
            self.figure.vlines = tuple(vlines)
        self.update()

    def set_empty_text(self, text: str) -> None:
        self.empty_text = text
        self.update()

    def axis_labels(self) -> tuple[str, str]:
        if self.figure is None:
            return "", ""
        return self.figure.x_label, self.figure.y_label

    # -- data <-> axis space -------------------------------------------------
    def _tx(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if not self.log_x:
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(np.where(values > 0, values,
                                                          1.0)), np.nan)

    def _ty(self, values) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if not self.log_y:
            return values
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(np.where(values > 0, values,
                                                          1.0)), np.nan)

    def _x_data(self, value: float) -> float:
        return 10.0 ** value if self.log_x else value

    def _y_data(self, value: float) -> float:
        return 10.0 ** value if self.log_y else value

    def _y_floor(self) -> float:
        """Axis-space y where bars start: 0, or below the view on log y."""
        if self.log_y:
            return self.y_range[0] - 1.0
        return 0.0

    def _visible_series(self) -> list[StatSeries]:
        if self.figure is None:
            return []
        return [s for s in self.figure.series if s.visible and s.mean.size]

    # -- view ------------------------------------------------------------------
    def fit(self) -> None:
        """Frame every visible mean and its spread, with headroom."""
        self._fit_ranges()
        self.update()

    def _fit_ranges(self) -> None:
        series = self._visible_series()
        if not series:
            self._x_range, self._y_range, self._auto = None, None, True
            return
        xs, ys = [], []
        categorical = self.figure.categories is not None
        any_bars = False
        for s in series:
            if s.kind == "step":
                xs.append(self._tx(s.edges))
            else:
                xs.append(self._tx(s.x))
            spread = np.where(np.isfinite(s.std), s.std, 0.0) \
                if s.std is not None and (s.band or s.whiskers) else 0.0
            ys.append(self._ty(s.mean))
            ys.append(self._ty(s.mean + spread))
            ys.append(self._ty(s.mean - spread))
            any_bars |= s.kind == "bars"
        x_all = np.concatenate([a[np.isfinite(a)] for a in xs]) \
            if xs else np.array([])
        y_all = np.concatenate([a[np.isfinite(a)] for a in ys]) \
            if ys else np.array([])
        if categorical:
            n = len(self.figure.categories)
            x0, x1 = -0.6, max(n - 0.4, 0.6)
        elif x_all.size:
            x0, x1 = float(x_all.min()), float(x_all.max())
            if x1 <= x0:
                x0, x1 = x0 - 0.5, x1 + 0.5
            pad = 0.02 * (x1 - x0) if not self.log_x else 0.03 * (x1 - x0)
            if any(s.kind in ("points", "bars") for s in series):
                x0, x1 = x0 - pad, x1 + pad
        else:
            x0, x1 = 0.0, 1.0
        if y_all.size:
            y0, y1 = float(y_all.min()), float(y_all.max())
            if (any_bars or any(s.kind == "step" for s in series)) \
                    and not self.log_y:
                y0 = min(y0, 0.0)
                y1 = max(y1, 0.0)
        else:
            y0, y1 = 0.0, 1.0
        if y1 <= y0:
            # one value (a constant, an NVT density): 5 % of it either side
            pad = 0.05 * abs(y1) if y1 else 0.5
            y0, y1 = y0 - pad, y1 + pad
        span = y1 - y0
        bottom = y0 if (any_bars and not self.log_y and y0 == 0.0) \
            else y0 - 0.04 * span
        self._x_range = (x0, x1)
        self._y_range = (bottom, y1 + 0.07 * span)
        self._auto = True

    # -- ticks -----------------------------------------------------------------
    def _linear_ticks(self, lo: float, hi: float, target: int):
        step = self._nice_step(hi - lo, max(target, 2))
        first = math.ceil(lo / step - 1e-9)
        values = []
        k = first
        while k * step <= hi + step * 1e-9 and len(values) < 200:
            values.append(k * step)
            k += 1
        return [(v, _tick_text(v, step)) for v in values], []

    def _log_ticks(self, lo: float, hi: float, target: int):
        """Decades (labelled) and 2..9 times them (unlabelled), in axis space."""
        if hi - lo < 0.9:
            ticks, _ = self._linear_ticks(10 ** lo, 10 ** hi, target)
            return [(math.log10(v), text) for v, text in ticks if v > 0], []
        stride = max(1, int(math.ceil((hi - lo) / max(target, 2))))
        # 2 and 5 times each decade are labelled too where there is room, and
        # always on an axis of less than 1.5 decades, which holds at most one
        # decade: an MSD from 2 to 18 ps had the single label '10'
        labelled = (1, 2, 5) if hi - lo <= 1.5 or (
            hi - lo <= 2.5 and target >= 6) else (1,)
        major, minor = [], []
        for k in range(int(math.floor(lo)), int(math.ceil(hi)) + 1):
            if lo - 1e-9 <= k <= hi + 1e-9 and k % stride == 0:
                major.append((float(k), _decade_text(k)))
            if stride == 1 and hi - lo <= 7:
                for m in range(2, 10):
                    v = k + math.log10(m)
                    if not lo <= v <= hi:
                        continue
                    if m in labelled:
                        major.append((v, f"{m * 10.0 ** k:g}"))
                    else:
                        minor.append(v)
        major.sort()
        return major, minor

    # -- layout ------------------------------------------------------------------
    def _font(self, for_export: bool) -> QFont:
        if for_export:
            font = QFont(QApplication.font().family())
            font.setPixelSize(_EXPORT_FONT_PT)
            return font
        return QFont(self.font())

    def _layout(self, width: float, height: float, font: QFont,
                fm: QFontMetricsF, m: _Metrics) -> _Layout:
        figure = self.figure
        line_h = fm.height()
        title_h = line_h + m.gap if figure is not None and figure.title else 0
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        # first guess of the plot rectangle, refined once the tick labels
        # and the legend are measured
        rect = QRectF(60, title_h + m.gap, max(width - 80, 10),
                      max(height - title_h - 70, 10))
        if self.log_y:
            y_ticks, y_minor = self._log_ticks(y0, y1, int(rect.height() / 38))
        else:
            y_ticks, y_minor = self._linear_ticks(y0, y1,
                                                  int(rect.height() / 38))
        label_w = max((fm.horizontalAdvance(t) for _, t in y_ticks),
                      default=fm.horizontalAdvance("0"))
        left = m.gap + line_h + m.gap + label_w + m.tick + 2

        legend_rows = []
        legend = None
        series = [s for s in self._visible_series() if s.label]
        if len(series) == 1 and figure is not None and \
                series[0].label == figure.title:
            series = []             # the title already names it
        legend_w = 0.0
        if series:
            sample = 30.0 if m is _SCREEN else 20.0
            # a narrow figure (a Model window's, rows under it: ~530 px)
            # gives its legend a little more of its width: at 0.30 a
            # scalar's 'mean ± std over 10 frames' was cut to '... 10 fr...'
            cap = width * (0.30 if width >= 800 else 0.38)
            labels = [s.label for s in series]
            if any(fm.horizontalAdvance(t) > cap - sample - 8
                   for t in labels):
                labels = _legend_labels(labels, figure.title
                                        if figure is not None else "")
            texts = [fm.elidedText(t, Qt.ElideRight, cap - sample - 8)
                     for t in labels]
            legend_w = max(fm.horizontalAdvance(t) for t in texts) \
                + sample + 3 * m.gap
            legend_rows = list(zip(series, texts))
        right = m.gap + legend_w + (m.gap if legend_w else 0)

        plot_w = max(width - left - right, 20)
        categorical = figure is not None and figure.categories is not None
        rotate = False
        cat_h = line_h
        if categorical and figure.categories:
            slot = plot_w / max(x1 - x0, 1e-9)
            widest = max(fm.horizontalAdvance(c) for c in figure.categories)
            if widest > slot * 0.92:
                rotate = True
                cat_h = min(widest, height * _CATEGORY_SHARE) * 0.66 \
                    + line_h * 0.8
        x_ticks, x_minor = ([], [])
        if not categorical:
            # A tick label is centred on its tick, so one at the right end of
            # the axis overhangs the plot by half its width; with no legend
            # there was only a gap's room for it, and the MSD plot's last
            # label, '2.0', was cut to '2.C' at the widget's edge. The plot
            # gives up what the label needs.
            for _ in range(3):
                if self.log_x:
                    x_ticks, x_minor = self._log_ticks(x0, x1,
                                                       int(plot_w / 80))
                else:
                    x_ticks, x_minor = self._linear_ticks(x0, x1,
                                                          int(plot_w / 80))
                over = max((left + (value - x0) / (x1 - x0) * plot_w
                            + fm.horizontalAdvance(text) / 2 - (width - 1)
                            for value, text in x_ticks if x1 > x0),
                           default=0.0)
                if over <= 0.5:
                    break
                right += over
                plot_w = max(width - left - right, 20)
        annotations = list(figure.annotations) if figure is not None else []
        dropped = [f"{n} point(s) at or below 0 are not drawn on the log "
                   f"{axis} axis." for axis, n in self._dropped.items() if n]
        annotation_lines = _wrap(annotations + dropped, fm, width - 2 * m.gap)
        footnote_lines = _wrap([self.footnote] if self.footnote else [], fm,
                               width - 2 * m.gap)
        bottom = (m.tick + 2 + cat_h + m.gap / 2 + line_h + m.gap
                  + len(annotation_lines) * line_h
                  + len(footnote_lines) * line_h + m.gap / 2)
        top = title_h + m.gap
        rect = QRectF(left, top, plot_w, max(height - top - bottom, 20))
        y_lines = _split_title(figure.y_label if figure is not None else "",
                               fm, rect.height())
        if len(y_lines) > 1:
            # a second line of y title: the plot moves right by one line
            rect.setLeft(rect.left() + min(line_h, max(rect.width() - 20,
                                                       0.0)))
        x_title_top = rect.bottom() + m.tick + 2 + cat_h + m.gap / 2
        if legend_rows:
            rows_fit = max(1, int((rect.height() - m.gap) // (line_h + 2)))
            if len(legend_rows) > rows_fit:
                extra = len(legend_rows) - rows_fit + 1
                legend_rows = legend_rows[:rows_fit - 1] + \
                    [(None, f"+ {extra} more (in the table)")]
            legend_h = len(legend_rows) * (line_h + 2) + m.gap
            legend = QRectF(rect.right() + m.gap, rect.top(),
                            legend_w, legend_h)
        return _Layout(rect, x_ticks, y_ticks, x_minor, y_minor, rotate,
                       cat_h, x_title_top, legend, legend_rows,
                       annotation_lines,
                       footnote_lines, title_h, font, fm, y_lines)

    def _metrics(self, device_width, device_height, font):
        """The plot rectangle Plot's pan and zoom handlers use."""
        fm = QFontMetricsF(font, self)
        layout = self._layout(device_width, device_height, font, fm, _SCREEN)
        return layout.rect, fm

    # -- mapping -------------------------------------------------------------------
    def _map(self, rect: QRectF, x, y):
        """Data values to device coordinates, through the log transforms."""
        return self._to_device(rect, self._tx(x), self._ty(y))

    # -- painting ------------------------------------------------------------------
    def render_to(self, painter: QPainter, width: float, height: float,
                  for_export: bool = False) -> None:
        """Draw the figure on any paint device, in drawing units: device
        pixels on screen, points in an export (the save_* methods scale the
        painter so)."""
        m = _PRINT if for_export else _SCREEN
        font = self._font(for_export)
        painter.save()
        painter.setFont(font)
        fm = QFontMetricsF(font) if for_export else QFontMetricsF(font, self)
        if (self._x_range is None or self._y_range is None) \
                and self._visible_series():
            self._fit_ranges()
        self._count_dropped()
        self._category_room = height * _CATEGORY_SHARE
        layout = self._layout(width, height, font, fm, m)
        rect = layout.rect

        background = QColor(255, 255, 255) if for_export else self.background
        ink = QColor(20, 20, 20) if for_export else self.text
        axis = QColor(80, 80, 80) if for_export else self.axis
        faint = QColor(222, 222, 222) if for_export else self.faint
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(QRectF(0, 0, width, height), background)
        if not for_export:
            painter.fillRect(rect, self.panel)

        if self.figure is None or not self._visible_series():
            painter.setPen(QPen(ink))
            painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap,
                             self.empty_text)
            painter.restore()
            return

        if self.figure.title:
            painter.setPen(QPen(ink))
            bold = QFont(font)
            bold.setBold(True)
            painter.setFont(bold)
            painter.drawText(QRectF(rect.left(), m.gap / 2, width - rect.left()
                                    - m.gap, layout.title_height),
                             Qt.AlignLeft | Qt.AlignVCenter,
                             QFontMetricsF(bold).elidedText(
                                 self.figure.title, Qt.ElideRight,
                                 width - rect.left() - m.gap))
            painter.setFont(font)

        self._draw_axes_grid(painter, layout, m, axis, faint, ink, for_export)
        painter.save()
        painter.setClipRect(rect)
        for index, s in enumerate(self._visible_series()):
            self._draw_stat_series(painter, rect, s, index, m, ink,
                                   background, for_export)
        self._draw_vlines(painter, rect, m, ink, for_export, layout.fm)
        painter.restore()
        frame_pen = QPen(axis, m.thin * 1.4)
        painter.setPen(frame_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(rect)
        self._draw_legend_box(painter, layout, m, ink, background, for_export)
        self._draw_texts(painter, layout, m, ink, width, for_export)
        if (self._cursor is not None and not for_export
                and rect.contains(QPointF(self._cursor))):
            self._draw_crosshair(painter, rect, fm)
        painter.restore()

    def _count_dropped(self) -> None:
        dropped = {"x": 0, "y": 0}
        for s in self._visible_series():
            if self.log_x:
                xs = s.edges if s.kind == "step" else s.x
                dropped["x"] += int(np.sum(np.isfinite(xs) & (xs <= 0)))
            if self.log_y:
                dropped["y"] += int(np.sum(np.isfinite(s.mean)
                                           & (s.mean <= 0)))
        self._dropped = dropped

    def _draw_axes_grid(self, painter, layout: _Layout, m: _Metrics, axis,
                        faint, ink, for_export) -> None:
        rect, fm = layout.rect, layout.fm
        grid = QPen(faint, m.thin, Qt.DotLine if not for_export
                    else Qt.SolidLine)
        tick_pen = QPen(axis, m.thin * 1.4)
        figure = self.figure
        # y
        for value, text in layout.y_ticks:
            py = float(self._to_device(rect, 0.0, value)[1])
            if not rect.top() - 0.5 <= py <= rect.bottom() + 0.5:
                continue
            painter.setPen(grid)
            painter.drawLine(QPointF(rect.left(), py),
                             QPointF(rect.right(), py))
            painter.setPen(tick_pen)
            painter.drawLine(QPointF(rect.left() - m.tick, py),
                             QPointF(rect.left(), py))
            painter.setPen(QPen(ink))
            painter.drawText(QRectF(0, py - fm.height() / 2,
                                    rect.left() - m.tick - 2, fm.height()),
                             Qt.AlignRight | Qt.AlignVCenter, text)
        painter.setPen(tick_pen)
        for value in layout.y_minor:
            py = float(self._to_device(rect, 0.0, value)[1])
            if rect.top() <= py <= rect.bottom():
                painter.drawLine(QPointF(rect.left() - m.tick * 0.55, py),
                                 QPointF(rect.left(), py))
        # x
        label_top = rect.bottom() + m.tick + 2
        if figure.categories is not None:
            painter.setPen(QPen(ink))
            for c, text in enumerate(figure.categories):
                px = float(self._to_device(rect, float(c), 0.0)[0])
                if not rect.left() - 1 <= px <= rect.right() + 1:
                    continue
                painter.setPen(tick_pen)
                painter.drawLine(QPointF(px, rect.bottom()),
                                 QPointF(px, rect.bottom() + m.tick))
                painter.setPen(QPen(ink))
                if layout.rotate_categories:
                    painter.save()
                    painter.translate(px + fm.height() * 0.3, label_top)
                    painter.rotate(-40)
                    shown = fm.elidedText(text, Qt.ElideRight,
                                          self._category_room)
                    painter.drawText(QRectF(-fm.horizontalAdvance(shown) - 2,
                                            -fm.height() / 2,
                                            fm.horizontalAdvance(shown) + 2,
                                            fm.height()),
                                     Qt.AlignRight | Qt.AlignVCenter, shown)
                    painter.restore()
                else:
                    painter.drawText(QRectF(px - 60, label_top, 120,
                                            fm.height()),
                                     Qt.AlignHCenter | Qt.AlignTop, text)
        else:
            for value, text in layout.x_ticks:
                px = float(self._to_device(rect, value, 0.0)[0])
                if not rect.left() - 0.5 <= px <= rect.right() + 0.5:
                    continue
                painter.setPen(grid)
                painter.drawLine(QPointF(px, rect.top()),
                                 QPointF(px, rect.bottom()))
                painter.setPen(tick_pen)
                painter.drawLine(QPointF(px, rect.bottom()),
                                 QPointF(px, rect.bottom() + m.tick))
                painter.setPen(QPen(ink))
                painter.drawText(QRectF(px - 50, label_top, 100,
                                        fm.height()),
                                 Qt.AlignHCenter | Qt.AlignTop, text)
            painter.setPen(tick_pen)
            for value in layout.x_minor:
                px = float(self._to_device(rect, value, 0.0)[0])
                if rect.left() <= px <= rect.right():
                    painter.drawLine(QPointF(px, rect.bottom()),
                                     QPointF(px, rect.bottom()
                                             + m.tick * 0.55))
        # axis titles
        painter.setPen(QPen(ink))
        x_title_top = layout.x_title_top
        painter.drawText(QRectF(rect.left(), x_title_top, rect.width(),
                                fm.height()),
                         Qt.AlignHCenter | Qt.AlignVCenter,
                         fm.elidedText(figure.x_label, Qt.ElideRight,
                                       rect.width()))
        lines = layout.y_label_lines or [figure.y_label]
        for i, line in enumerate(lines):
            painter.save()
            painter.translate(m.gap + fm.height() * (i + 0.5),
                              rect.center().y())
            painter.rotate(-90)
            painter.drawText(QRectF(-rect.height() / 2, -fm.height() / 2,
                                    rect.height(), fm.height()),
                             Qt.AlignHCenter | Qt.AlignVCenter,
                             fm.elidedText(line, Qt.ElideRight,
                                           rect.height()))
            painter.restore()

    def _series_colour(self, s: StatSeries, ink, background,
                       for_export) -> QColor:
        colour = QColor(*s.style.colour)
        if not for_export and s.style.colour == (0, 0, 0) and \
                background.lightnessF() < 0.5:
            return QColor(ink)
        return colour

    def _draw_stat_series(self, painter, rect, s: StatSeries, index: int,
                          m: _Metrics, ink, background, for_export) -> None:
        colour = self._series_colour(s, ink, background, for_export)
        if s.kind == "bars":
            self._draw_bars(painter, rect, s, index, m, colour, ink,
                            background)
            return
        if s.kind == "step":
            xs = np.repeat(s.edges, 2)[1:-1]
            ys = np.repeat(s.mean, 2)
            std = np.repeat(s.std, 2) if s.std is not None else None
        else:
            xs, ys, std = s.x, s.mean, s.std
        px, py = self._map(rect, xs, ys)
        if s.band and std is not None and np.isfinite(std).any():
            self._draw_band(painter, rect, xs, ys, std, colour, m)
        if s.kind in ("line", "step"):
            pen = QPen(colour, m.line)
            pen.setStyle(_DASH_STYLE[s.style.dash])
            pen.setCapStyle(Qt.FlatCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for polyline in _polyline(px, py):
                painter.drawPolyline(polyline)
        if s.kind == "points" and s.whiskers and std is not None:
            self._draw_whiskers(painter, rect, s.x, s.mean, std, colour, m)
        if s.markers or s.kind == "points":
            if s.kind == "step":
                mx, my = self._map(rect, s.x, s.mean)
            else:
                mx, my = px, py
            n = mx.size
            every = 1 if s.kind == "points" or n <= 40 else \
                max(1, int(math.ceil(n / 14)))
            start = (index * every // 3) % every if every > 1 else 0
            painter.setPen(QPen(background, m.thin * 0.8))
            painter.setBrush(QBrush(colour))
            for k in range(start, n, every):
                if np.isfinite(mx[k]) and np.isfinite(my[k]):
                    _draw_marker(painter, s.style.marker, float(mx[k]),
                                 float(my[k]), m.marker, colour, m)

    def _draw_band(self, painter, rect, xs, ys, std, colour, m) -> None:
        defined = np.isfinite(ys) & np.isfinite(std)
        upper = np.where(defined, ys + std, np.nan)
        lower = np.where(defined, ys - std, np.nan)
        if self.log_y:
            floor = 10.0 ** (self.y_range[0] - 1.0)
            lower = np.where(defined & (lower <= 0), floor, lower)
        ux, uy = self._map(rect, xs, upper)
        lx, ly = self._map(rect, xs, lower)
        fill = QColor(colour)
        fill.setAlpha(m.band_alpha)
        painter.setPen(Qt.NoPen)
        painter.setBrush(fill)
        finite = np.isfinite(ux) & np.isfinite(uy) & np.isfinite(ly)
        for start, stop in _runs(finite):
            outline = _points(ux[start:stop], uy[start:stop]) + \
                _points(lx[start:stop][::-1], ly[start:stop][::-1])
            painter.drawPolygon(QPolygonF(outline))
        painter.setBrush(Qt.NoBrush)

    def _draw_whiskers(self, painter, rect, x, mean, std, colour, m,
                       offset: float = 0.0) -> None:
        pen = QPen(colour, m.thin * 1.6)
        pen.setCapStyle(Qt.FlatCap)
        painter.setPen(pen)
        for k in range(mean.size):
            if not (np.isfinite(mean[k]) and np.isfinite(std[k])) \
                    or std[k] <= 0:
                continue
            hi = mean[k] + std[k]
            lo = mean[k] - std[k]
            if self.log_y and lo <= 0:
                lo = 10.0 ** (self.y_range[0] - 1.0)
            px, (py_hi, py_lo) = self._map(rect, [x[k], x[k]], [hi, lo])
            cx = float(px[0]) + offset
            if not (np.isfinite(py_hi) and np.isfinite(py_lo)):
                continue
            painter.drawLine(QPointF(cx, float(py_hi)),
                             QPointF(cx, float(py_lo)))
            painter.drawLine(QPointF(cx - m.cap, float(py_hi)),
                             QPointF(cx + m.cap, float(py_hi)))
            painter.drawLine(QPointF(cx - m.cap, float(py_lo)),
                             QPointF(cx + m.cap, float(py_lo)))

    def _draw_bars(self, painter, rect, s: StatSeries, index, m, colour, ink,
                   background) -> None:
        bars = [b for b in self._visible_series() if b.kind == "bars"]
        n_groups = max(len(bars), 1)
        slot = bars.index(s) if s in bars else 0
        width = 0.8 / n_groups
        base_y = float(self._to_device(rect, 0.0, self._y_floor())[1])
        for k in range(s.mean.size):
            value = s.mean[k]
            if not np.isfinite(value):
                continue
            left = s.x[k] - 0.4 + slot * width
            xs, _ = self._to_device(rect, np.array([left, left + width]),
                                    np.zeros(2))
            _, tops = self._map(rect, np.array([s.x[k]]), np.array([value]))
            x0, x1, top = float(xs[0]), float(xs[1]), float(tops[0])
            if not np.isfinite(top):
                continue
            bar = QRectF(QPointF(float(x0), min(top, base_y)),
                         QPointF(float(x1), max(top, base_y)))
            _fill_hatched(painter, bar, s.style.hatch, colour, m)
            outline = QPen(colour.darker(130), m.thin * 1.2)
            painter.setPen(outline)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(bar)
        if s.whiskers and s.std is not None:
            centres = s.x - 0.4 + (slot + 0.5) * width
            self._draw_whiskers(painter, rect, centres, s.mean, s.std,
                                QColor(ink), m)

    def _draw_vlines(self, painter, rect, m, ink, for_export, fm) -> None:
        if self.figure is None:
            return
        for x, label, kind in self.figure.vlines:
            px = self._map(rect, [x], [0.0])[0][0]
            if not np.isfinite(px):
                continue
            colour = QColor(ink) if kind == "current" else \
                QColor(*OKABE_ITO[1][1])
            pen = QPen(colour, m.line * (1.0 if kind == "current" else 0.8))
            pen.setStyle(Qt.SolidLine if kind == "current" else Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(QPointF(float(px), rect.top()),
                             QPointF(float(px), rect.bottom()))
            if label:
                width = fm.horizontalAdvance(label) + 6
                row = 0 if kind == "current" else 1
                left = float(px) + 3 if float(px) + width < rect.right() \
                    else float(px) - width - 3
                box = QRectF(left, rect.top() + 2 + row * (fm.height() + 2),
                             width, fm.height())
                plate = QColor(255, 255, 255) if for_export else \
                    QColor(self.background)
                plate.setAlpha(210)
                painter.fillRect(box, plate)
                painter.setPen(QPen(colour))
                painter.drawText(box, Qt.AlignCenter, label)

    def _draw_legend_box(self, painter, layout: _Layout, m, ink, background,
                         for_export) -> None:
        if layout.legend is None:
            return
        fm = layout.fm
        box = layout.legend
        y = box.top() + m.gap / 2
        sample = 30.0 if m is _SCREEN else 20.0
        for s, text in layout.legend_rows:
            middle = y + fm.height() / 2
            if s is not None:
                colour = self._series_colour(s, ink, background, for_export)
                left = box.left() + m.gap / 2
                if s.kind == "bars":
                    patch = QRectF(left + 2, middle - fm.height() * 0.32,
                                   sample - 4, fm.height() * 0.64)
                    _fill_hatched(painter, patch, s.style.hatch, colour, m)
                    painter.setPen(QPen(colour.darker(130), m.thin * 1.2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(patch)
                else:
                    if s.band and s.has_spread:
                        fill = QColor(colour)
                        fill.setAlpha(m.band_alpha)
                        painter.fillRect(QRectF(left, middle - fm.height()
                                                * 0.28, sample,
                                                fm.height() * 0.56), fill)
                    if s.kind in ("line", "step"):
                        pen = QPen(colour, m.line)
                        pen.setStyle(_DASH_STYLE[s.style.dash])
                        painter.setPen(pen)
                        painter.drawLine(QPointF(left, middle),
                                         QPointF(left + sample, middle))
                    if s.markers or s.kind == "points":
                        painter.setPen(QPen(background, m.thin * 0.8))
                        painter.setBrush(QBrush(colour))
                        _draw_marker(painter, s.style.marker,
                                     left + sample / 2, middle, m.marker,
                                     colour, m)
                        painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(ink))
            text_left = box.left() + m.gap / 2 + sample + m.gap
            painter.drawText(QRectF(text_left, y,
                                    box.right() - text_left, fm.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, text)
            y += fm.height() + 2

    def _draw_texts(self, painter, layout: _Layout, m, ink, width,
                    for_export) -> None:
        fm = layout.fm
        y = layout.x_title_top + fm.height() + m.gap / 2
        painter.setPen(QPen(ink))
        for line in layout.annotation_lines:
            painter.drawText(QRectF(m.gap, y, width - 2 * m.gap, fm.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, line)
            y += fm.height()
        if layout.footnote_lines:
            muted = QColor(110, 110, 110) if for_export else QColor(self.axis)
            painter.setPen(QPen(muted))
            for line in layout.footnote_lines:
                painter.drawText(QRectF(m.gap, y, width - 2 * m.gap,
                                        fm.height()),
                                 Qt.AlignLeft | Qt.AlignVCenter, line)
                y += fm.height()

    def _draw_crosshair(self, painter, rect, fm) -> None:
        px, py = float(self._cursor.x()), float(self._cursor.y())
        pen = QPen(self.crosshair, 0.8, Qt.DashLine)
        painter.setPen(pen)
        painter.drawLine(QPointF(px, rect.top()), QPointF(px, rect.bottom()))
        painter.drawLine(QPointF(rect.left(), py), QPointF(rect.right(), py))
        ax, ay = self._to_data(rect, px, py)
        x, y = self._x_data(ax), self._y_data(ay)
        figure = self.figure
        if figure is not None and figure.categories is not None:
            k = int(round(x))
            x_text = figure.categories[k] if 0 <= k < len(
                figure.categories) else ""
        else:
            x_text = f"{x:.4g} {unit_text(figure.x_unit) if figure else ''}"
        y_text = f"{y:.4g} {unit_text(figure.y_unit) if figure else ''}"
        text = f"{x_text.strip()}, {y_text.strip()}"
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

    # -- export ------------------------------------------------------------------
    @staticmethod
    def _points(width_mm: float, height_mm: float) -> tuple[float, float]:
        if not (width_mm > 0 and height_mm > 0):
            raise ValueError("a figure needs a positive width and height")
        return width_mm / 25.4 * 72.0, height_mm / 25.4 * 72.0

    def save_svg(self, path, width_mm: float = EXPORT_WIDTH_MM,
                 height_mm: float = EXPORT_HEIGHT_MM) -> Path:
        """Vector SVG at the stated size (1 user unit = 1 pt)."""
        from PySide6.QtSvg import QSvgGenerator

        w, h = self._points(width_mm, height_mm)
        generator = QSvgGenerator()
        generator.setFileName(str(path))
        # the size is a whole number of pixels: at 254 dpi a pixel is
        # 0.1 mm, so the width and height written are the ones asked for
        generator.setResolution(254)
        generator.setSize(QSize(int(round(width_mm * 10)),
                                int(round(height_mm * 10))))
        generator.setViewBox(QRectF(0, 0, w, h))
        generator.setTitle(self.title or "FACET figure")
        generator.setDescription(self.footnote or "")
        painter = QPainter(generator)
        try:
            self.render_to(painter, w, h, for_export=True)
        finally:
            painter.end()
        return Path(path)

    def save_pdf(self, path, width_mm: float = EXPORT_WIDTH_MM,
                 height_mm: float = EXPORT_HEIGHT_MM) -> Path:
        """A one-page vector PDF at the stated size."""
        from PySide6.QtCore import QMarginsF, QSizeF
        from PySide6.QtGui import QPageSize, QPdfWriter

        w, h = self._points(width_mm, height_mm)
        writer = QPdfWriter(str(path))
        writer.setPageSize(QPageSize(QSizeF(width_mm, height_mm),
                                     QPageSize.Millimeter))
        writer.setPageMargins(QMarginsF(0, 0, 0, 0))
        writer.setResolution(EXPORT_DPI)
        writer.setTitle(self.title or "FACET figure")
        painter = QPainter(writer)
        try:
            scale = writer.resolution() / 72.0
            painter.scale(scale, scale)
            self.render_to(painter, w, h, for_export=True)
        finally:
            painter.end()
        return Path(path)

    def render_image(self, width_mm: float = EXPORT_WIDTH_MM,
                     height_mm: float = EXPORT_HEIGHT_MM,
                     dpi: int = EXPORT_DPI) -> QImage:
        """The export drawn into an image of the stated size and resolution,
        with the resolution written into it (dots per metre)."""
        w, h = self._points(width_mm, height_mm)
        if not dpi > 0:
            raise ValueError("dpi needs a positive number")
        width_px = int(round(width_mm / 25.4 * dpi))
        height_px = int(round(height_mm / 25.4 * dpi))
        image = QImage(width_px, height_px, QImage.Format_ARGB32)
        image.fill(QColor(255, 255, 255))
        per_metre = int(round(dpi / 0.0254))
        image.setDotsPerMeterX(per_metre)
        image.setDotsPerMeterY(per_metre)
        painter = QPainter(image)
        try:
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setRenderHint(QPainter.TextAntialiasing, True)
            painter.scale(width_px / w, height_px / h)
            self.render_to(painter, w, h, for_export=True)
        finally:
            painter.end()
        return image

    def save_png(self, path, width_mm: float = EXPORT_WIDTH_MM,
                 height_mm: float = EXPORT_HEIGHT_MM,
                 dpi: int = EXPORT_DPI) -> Path:
        """A PNG at ``dpi`` (600 by default), its resolution in the file."""
        image = self.render_image(width_mm, height_mm, dpi)
        if not image.save(str(path), "PNG"):
            raise OSError(f"{path}: the PNG could not be written")
        return Path(path)

    def save(self, path, width_mm: float = EXPORT_WIDTH_MM,
             height_mm: float = EXPORT_HEIGHT_MM) -> Path:
        """By suffix: .svg, .pdf or .png (600 dpi)."""
        suffix = Path(path).suffix.lower()
        if suffix == ".svg":
            return self.save_svg(path, width_mm, height_mm)
        if suffix == ".pdf":
            return self.save_pdf(path, width_mm, height_mm)
        if suffix == ".png":
            return self.save_png(path, width_mm, height_mm)
        raise ValueError(f"{Path(path).name}: a figure is written as .svg, "
                         ".pdf or .png")

    @Slot()
    def choose_save(self) -> Path | None:
        """Ask where to write the figure shown (SVG, PDF or 600 dpi PNG)
        and write it; ``statusMessage`` says what was written, or why
        not."""
        if self.figure is None:
            self.statusMessage.emit("The plot holds no figure to save.")
            return None
        stem = file_stem(self.file_stem or self.title or "", "figure")
        path = ask_figure_path(self, "Save the figure", stem)
        if path is None:
            return None
        try:
            written = self.save(path)
        except (OSError, ValueError) as error:
            self.statusMessage.emit(f"The figure was not written: {error}")
            return None
        self.statusMessage.emit(f"Wrote {written}")
        return written

    # -- interaction -------------------------------------------------------------
    def data_at(self, px: float, py: float) -> tuple[float, float]:
        """Data coordinates under a widget position."""
        rect, _ = self._metrics(self.width(), self.height(), self.font())
        ax, ay = self._to_data(rect, px, py)
        return self._x_data(ax), self._y_data(ay)

    def mousePressEvent(self, event) -> None:
        if self.pick_mode and event.button() == Qt.LeftButton:
            self._picking = True
            x, y = self.data_at(event.position().x(), event.position().y())
            self.picked.emit(x, y)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._picking:
            self._cursor = event.position().toPoint()
            x, y = self.data_at(event.position().x(), event.position().y())
            self.picked.emit(x, y)
            self.update()
            return
        if self._drag is None:
            self._cursor = event.position().toPoint()
            x, y = self.data_at(event.position().x(), event.position().y())
            self.hovered.emit(x, y)
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._picking:
            self._picking = False
            return
        moved = (self._drag is not None and
                 (event.position().toPoint() - self._drag).manhattanLength()
                 > 3)
        self._drag = None
        self._drag_range = None
        self.unsetCursor()
        if event.button() == Qt.LeftButton and not moved:
            x, y = self.data_at(event.position().x(), event.position().y())
            self.picked.emit(x, y)


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------

def _tick_text(value: float, step: float) -> str:
    if abs(value) < step * 1e-6:
        return "0"
    if 1e-4 <= step < 1e6 and abs(value) < 1e7:
        decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)))
        if abs(round(step, decimals) - step) > step * 1e-6:
            decimals += 1
        return f"{value:.{decimals}f}"
    return f"{value:.3g}"


def _decade_text(k: int) -> str:
    if -3 <= k <= 4:
        return f"{10.0 ** k:g}"
    return f"1e{k}"


def _runs(mask: np.ndarray):
    """(start, stop) of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(
        ([0], np.asarray(mask, dtype=np.int8), [0]))))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def _points(xs: np.ndarray, ys: np.ndarray) -> list:
    return [QPointF(a, b) for a, b in zip(xs.tolist(), ys.tolist())]


def _polyline(px: np.ndarray, py: np.ndarray) -> list[QPolygonF]:
    """Polylines through the points, broken where a point is not finite."""
    finite = np.isfinite(px) & np.isfinite(py)
    return [QPolygonF(_points(px[a:b], py[a:b])) for a, b in _runs(finite)]


def _draw_marker(painter: QPainter, shape: str, x: float, y: float,
                 size: float, colour: QColor, m: _Metrics) -> None:
    r = size / 2.0
    if shape == "circle":
        painter.drawEllipse(QPointF(x, y), r * 0.92, r * 0.92)
    elif shape == "square":
        s = r * 0.82
        painter.drawRect(QRectF(x - s, y - s, 2 * s, 2 * s))
    elif shape in ("triangle", "triangle down"):
        sign = -1.0 if shape == "triangle" else 1.0
        painter.drawPolygon(QPolygonF([QPointF(x, y + sign * r * 1.05),
                                       QPointF(x - r, y - sign * r * 0.75),
                                       QPointF(x + r, y - sign * r * 0.75)]))
    elif shape == "diamond":
        painter.drawPolygon(QPolygonF([QPointF(x, y - r * 1.1),
                                       QPointF(x + r * 0.85, y),
                                       QPointF(x, y + r * 1.1),
                                       QPointF(x - r * 0.85, y)]))
    else:
        pen = QPen(colour, m.line * 1.05)
        pen.setCapStyle(Qt.FlatCap)
        painter.save()
        painter.setPen(pen)
        if shape == "cross":
            painter.drawLine(QPointF(x - r, y - r), QPointF(x + r, y + r))
            painter.drawLine(QPointF(x - r, y + r), QPointF(x + r, y - r))
        else:
            painter.drawLine(QPointF(x - r * 1.1, y), QPointF(x + r * 1.1, y))
            painter.drawLine(QPointF(x, y - r * 1.1), QPointF(x, y + r * 1.1))
        painter.restore()


def _fill_hatched(painter: QPainter, rect: QRectF, hatch: str,
                  colour: QColor, m: _Metrics) -> None:
    """Fill a bar: solid, or a light tint with hatch lines drawn as vectors
    (a Qt pattern brush is drawn in device pixels, which vanish at 600 dpi)."""
    if rect.width() <= 0 or rect.height() <= 0:
        return
    fill = QColor(colour)
    if hatch == "solid":
        fill.setAlpha(205)
        painter.fillRect(rect, fill)
        return
    fill.setAlpha(55)
    painter.fillRect(rect, fill)
    painter.save()
    painter.setClipRect(rect, Qt.IntersectClip)
    pen = QPen(colour, m.thin * 1.3)
    painter.setPen(pen)
    step = m.hatch_step
    left, top, right, bottom = rect.left(), rect.top(), rect.right(), \
        rect.bottom()
    span = rect.width() + rect.height()
    if hatch in ("diagonal", "cross"):
        t = 0.0
        while t <= span:
            painter.drawLine(QPointF(left + t, bottom),
                             QPointF(left + t - rect.height(), top))
            t += step
    if hatch in ("back diagonal", "cross"):
        t = 0.0
        while t <= span:
            painter.drawLine(QPointF(right - t, bottom),
                             QPointF(right - t + rect.height(), top))
            t += step
    if hatch == "horizontal":
        y = bottom
        while y >= top:
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            y -= step
    if hatch == "vertical":
        x = left + step / 2
        while x <= right:
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            x += step
    painter.restore()


def _wrap(paragraphs: Sequence[str], fm: QFontMetricsF, width: float
          ) -> list[str]:
    """Paragraphs broken into lines no wider than ``width``."""
    lines: list[str] = []
    for paragraph in paragraphs:
        words = str(paragraph).split()
        current = ""
        for word in words:
            trial = f"{current} {word}" if current else word
            if fm.horizontalAdvance(trial) <= width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines
