"""The Compare workspace: structures side by side, in the main window's
centre.

Two kinds of comparison live here, both built from numbers the rest of the
application already measures:

* **Site to site** -- a comparison basket of references: a crystal site
  (one :class:`~facet.core.coordination.SiteResult`) or an element of an MD
  model (the per-atom numbers of the frame shown, from its
  ``bulk.ValenceTable``). The basket is listed in the right column, overlaid
  as one staircase figure (CN against v_bond on the log axis every staircase
  in FACET uses), and, when a crystal site and a model element share an
  element, drawn as the model's per-atom distribution with the crystal
  site's value as a labelled vertical line -- the reference crystal against
  the glass.
* **Models side by side** -- two or more ticked model results: their
  headline measurements aligned row by row with a difference column
  (Overview), any descriptor at full width with the models overlaid or
  their difference drawn (Figure), every descriptor as one table with a
  column per model (Table), and the frames the models have on show laid
  along x in one 3D scene (3D). A statement names, in plain words, where
  the runs' parameters differ (v_bond, formers, frames); it states the
  difference and nothing more.

LAYOUT
------
:class:`CompareWorkspace` owns four widgets the window puts in its stacked
regions the way a model's controller does (``widgets``: the right column,
the Elements dock page, the toolbar, the centre canvas). The right column
is slim: the ticked models (each with its legend swatch, the same colour
and marker wherever that model appears), the site basket, the descriptor
tree grouped by question with plain names and a filter, and the
parameter-difference statement. The centre takes the comparison itself,
on four pages the toolbar switches: Overview | Figure | Table | 3D, with
Save figure…, Save rows… and Export comparison…. Back to the structure
restores the selected entry's own column and centre; the basket is kept.

The basket lives on the window, not on an entry, and is **not saved in
sessions**: a session restores files and settings, and the basket is a
working selection over them.

Crystal values are computed from the site's contact list at the project's
current v_bond; model values are those of the frame named in the row, at
that model's own v_bond. Every number here can be recomputed from the
SiteResult, the ValenceTable or the ModelResult container it came from.

A difference is always the model named minus the first ticked model (A),
written "B − A": a scalar as the difference with, where A is not zero and
the row is not itself a percentage, the relative change; a curve point by
point on the common axis (refused, in words, when the axes differ); a
distribution key by key, a key absent from one model counting 0 there.

THE 3D SIDE BY SIDE, MEASURED
-----------------------------
Each ticked model contributes the Scene its controller built for the frame
on show (``md_scene.build_md_scene`` on a worker: bulk-built, atoms only
above ``ATOMS_ONLY_ABOVE`` on the QPainter tier); :func:`side_by_side_scene`
lays the blocks along x with a gap and merges them with
``facet.gl.scene.merge_scenes``, so the merge is the only work done here on
the GUI thread. Measured 2026-10-13 on this machine (two 15 552-atom quartz
supercell frames, 31 104 atoms, ``scripts`` in the task's scratchpad):
merge 2 scenes with bonds (65 k bonds) 0.21 s; merge atoms only 0.05 s;
``StructureView.set_scene`` upload on the OpenGL tier 0.05 s; one QPainter
render of the merged scene at 980 x 640: 4.2 s with bonds, 1.6 s atoms only.
The painter tier therefore draws the merged scene atoms only above
``ATOMS_ONLY_ABOVE`` merged atoms (the per-model rule, applied to the sum),
and the note under the view says so. (Figures updated by the measuring
script's output; see its log.)
"""
from __future__ import annotations

import copy
import csv
import math
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QIcon, QPainter, QPen, QPixmap, \
    QPolygonF
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import bulk, bv, coordination, exporters
from ..core.md_stats import Distribution, Histogram, Scalar, Series
from ..gl.scene import Scene, merge_scenes
from . import chrome
from .md_names import analysis_title, display_name, group_of, group_titles, \
    kind_badge
from .md_plot import (
    Figure,
    SeriesStyle,
    StatSeries,
    axis_symbol,
    category_label,
    figure_for_family,
    style_for,
)
from .md_results_overview import FractionBar, analysis_measures, run_header
from .md_threshold import BVS_BIN_VU, PHI_BIN, staircase as model_staircase, \
    threshold_range
from .md_views import DescriptorView, RowTable, _NameDelegate, _RAW_ID

__all__ = ["CompareWorkspace", "CompareElements", "crystal_values",
           "model_values", "per_atom_values", "common_descriptors",
           "all_descriptors", "parameter_differences", "overview_rows",
           "difference", "difference_text", "series_difference",
           "distribution_difference", "histogram_difference",
           "difference_figure", "comparison_rows", "comparison_header",
           "write_comparison", "side_by_side_scene", "emphasise_element",
           "atoms_only", "marker_glyph", "swatch_pixmap", "VALUE_KINDS",
           "PAGES", "DEFAULT_GAP_ANG"]

# what the crystal-vs-glass selector offers: (key, axis name, axis unit)
VALUE_KINDS = (("CN", "CN", "1"),
               ("BVS", "bond-valence sum", "v.u."),
               ("phi", "phi", "1"))
_BIN = {"CN": 1.0, "BVS": BVS_BIN_VU, "phi": PHI_BIN}

STAIRCASE_POINTS = 240

# the centre's pages, in toolbar order
PAGES = (("overview", "Overview"), ("figure", "Figure"), ("table", "Table"),
         ("3d", "3D"))

# the gap between two frames in the 3D row when the Structures dock's
# spacing is 0 (which superimposes crystals): a drawing choice, in the
# note under the view and adjustable there
DEFAULT_GAP_ANG = 6.0

_MARKER_GLYPHS = {"circle": "●", "square": "■", "triangle": "▲",
                  "diamond": "◆", "triangle down": "▼", "cross": "✕",
                  "plus": "+"}
_DASH = "—"
_NAN = float("nan")
_ROLE = Qt.UserRole + 1


# ---------------------------------------------------------------------------
# value extraction (no widgets)
# ---------------------------------------------------------------------------

def crystal_values(result: coordination.SiteResult, v_bond: float) -> dict:
    """One crystal site at ``v_bond``, from its contact list.

    CN, BVS and phi are recomputed over the contacts with v > v_bond, so a
    threshold moved after the analysis still gives the numbers of the
    threshold asked for; the plateau is the one of ``result.plateaus`` the
    threshold sits on (NaN on a step edge); the mean bond length is the
    mean distance of those same contacts.
    """
    v = float(v_bond)
    bonded = [c for c in result.contacts
              if c.has_valence and c.valence > v]
    bvs = phi = d_mean = float("nan")
    if bonded:
        weights = [c.valence * c.occupancy for c in bonded]
        bvs = bv.bvs(weights)
        _mag, phi, _vec = bv.phi_index(
            np.array([c.vector for c in bonded]), weights)
        d_mean = float(np.mean([c.distance for c in bonded]))
    plateau = next((p for p in result.plateaus if p.contains(v)), None)
    return {
        "element": result.element,
        "site": result.label,
        "cn": len(bonded),
        "bvs_vu": float(bvs),
        "phi": float(phi),
        "plateau_decades": (plateau.width_decades if plateau
                            else float("nan")),
        "d_mean_ang": d_mean,
        "v_bond_vu": v,
    }


def per_atom_values(table: bulk.ValenceTable, element: str, v_bond: float,
                    which: str = "CN") -> np.ndarray:
    """The per-atom values of one element of a frame at ``v_bond``:
    'CN', 'BVS' (v.u.) or 'phi' (an atom with no bond has CN 0 and NaN for
    the other two), from ``bulk.at_threshold`` -- the same lookup the
    threshold strip reads."""
    results = bulk.at_threshold(table, float(v_bond))
    sel = np.asarray(table.elements) == str(element)
    arrays = {"CN": results.cn, "BVS": results.bvs_vu, "phi": results.phi}
    if which not in arrays:
        raise ValueError(f"{which!r} is not one of {tuple(arrays)}")
    return np.asarray(arrays[which], dtype=np.float64)[sel]


def model_values(table: bulk.ValenceTable, element: str, v_bond: float
                 ) -> dict:
    """One element of one frame at ``v_bond``: the mean and spread (standard
    deviation over the element's atoms, ddof = 0) of CN, BVS, phi, the
    plateau width and the bond length, with the atom counts behind each.

    BVS, phi and the plateau are averaged over the atoms where they exist
    (an atom with no bond above v_bond has none); the row says how many
    that is. Bond lengths are the element's listed contacts with
    v > v_bond, read from the atom's own row of the table.
    """
    v = float(v_bond)
    results = bulk.at_threshold(table, v)
    sel = np.asarray(table.elements) == str(element)
    n_atoms = int(sel.sum())

    def spread(values: np.ndarray) -> tuple[float, float, int]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan"), float("nan"), 0
        return (float(finite.mean()), float(finite.std(ddof=0)),
                int(finite.size))

    cn_mean, cn_std, _ = spread(
        np.asarray(results.cn, dtype=np.float64)[sel])
    bvs_mean, bvs_std, n_bonded = spread(
        np.asarray(results.bvs_vu, dtype=np.float64)[sel])
    phi_mean, phi_std, _ = spread(
        np.asarray(results.phi, dtype=np.float64)[sel])
    pl_mean, pl_std, _ = spread(
        np.asarray(results.plateau_decades, dtype=np.float64)[sel])
    valid = (np.arange(table.v_vu.shape[1])[None, :]
             < np.asarray(table.n_listed)[:, None])
    mask = valid[sel] & (table.v_vu[sel] > v)
    d = np.asarray(table.d_ang)[sel][mask]
    d_mean, d_std, _ = spread(np.asarray(d, dtype=np.float64))
    return {
        "element": str(element),
        "atoms": n_atoms,
        "atoms_bonded": n_bonded,
        "cn": cn_mean, "cn_std": cn_std,
        "bvs_vu": bvs_mean, "bvs_std": bvs_std,
        "phi": phi_mean, "phi_std": phi_std,
        "plateau_decades": pl_mean, "plateau_std": pl_std,
        "d_mean_ang": d_mean, "d_std_ang": d_std,
        "v_bond_vu": v,
    }


# ---------------------------------------------------------------------------
# alignment of model results (no widgets)
# ---------------------------------------------------------------------------

def _descriptor_order(results: Sequence, at_least: int
                      ) -> list[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []
    for result in results:
        for analysis, name, _container in result.descriptors():
            key = (str(analysis), str(name))
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1
    kept = [key for key in order if counts[key] >= at_least]
    titles = group_titles()
    kept.sort(key=lambda key: (titles.index(group_of(key[0]))
                               if group_of(key[0]) in titles
                               else len(titles), order.index(key)))
    return kept


def common_descriptors(results: Sequence) -> list[tuple[str, str]]:
    """The ``(analysis, descriptor)`` pairs present in at least two of the
    given ModelResults, in question-group order then as the runs list
    them."""
    return _descriptor_order(results, 2)


def all_descriptors(results: Sequence) -> list[tuple[str, str]]:
    """Every ``(analysis, descriptor)`` pair any of the given ModelResults
    holds, in question-group order then as the runs list them."""
    return _descriptor_order(results, 1)


def _compact_frames(frames: Sequence[int]) -> str:
    values = sorted(int(f) for f in frames)
    if not values:
        return "none"
    parts, start, last = [], values[0], values[0]
    for v in values[1:]:
        if v == last + 1:
            last = v
            continue
        parts.append(f"{start}" if start == last else f"{start}-{last}")
        start = last = v
    parts.append(f"{start}" if start == last else f"{start}-{last}")
    noun = "frame" if len(values) == 1 else "frames"
    return f"{len(values)} {noun} ({', '.join(parts)})"


def parameter_differences(labelled: Sequence[tuple[str, object]]
                          ) -> list[str]:
    """Plain statements of where the runs' parameters differ, from each
    result's run provenance: v_bond, the formers and the frames. A
    statement of fact, never a judgement; an empty list when they match."""
    out: list[str] = []
    if len(labelled) < 2:
        return out

    def stated(describe) -> list[tuple[str, str]]:
        # a list, not a mapping: two models of one file share a label, and
        # their difference still has to be seen
        return [(label, describe(result.provenance))
                for label, result in labelled]

    v_bond = stated(lambda p: "not stated" if p.v_bond_vu is None
                    else f"{p.v_bond_vu:g} v.u.")
    if len({text for _l, text in v_bond}) > 1:
        out.append("v_bond differs: " + "; ".join(
            f"{label}: {text}" for label, text in v_bond) + ".")
    formers = stated(lambda p: ", ".join(sorted(p.formers))
                     if p.formers else "none named")
    if len({text for _l, text in formers}) > 1:
        out.append("The formers differ: " + "; ".join(
            f"{label}: {text}" for label, text in formers) + ".")
    frames = stated(lambda p: _compact_frames(p.frames_used))
    if len({text for _l, text in frames}) > 1:
        out.append("The frames analysed differ: " + "; ".join(
            f"{label}: {text}" for label, text in frames) + ".")
    return out


def _entry_title(entry) -> str:
    return str(entry.name).splitlines()[0]


def _pm(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return _DASH
    if not math.isfinite(std) or std == 0:
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} ± {std:.2g}"


def marker_glyph(style: SeriesStyle) -> str:
    """The text glyph of a style's marker, so a header or a note carries
    the marker beside the colour."""
    return _MARKER_GLYPHS.get(style.marker, "●")


# ---------------------------------------------------------------------------
# differences (no widgets)
# ---------------------------------------------------------------------------

def difference(a: float, b: float, *, percent: bool = False
               ) -> tuple[float, float]:
    """``(b − a, 100 (b − a) / |a|)``. Both NaN where a side lacks a value;
    the relative part NaN when ``a`` is 0 or the row is itself a
    percentage (there the difference is in points)."""
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return _NAN, _NAN
    if not (math.isfinite(a) and math.isfinite(b)):
        return _NAN, _NAN
    d = b - a
    rel = _NAN if percent or a == 0 else 100.0 * d / abs(a)
    return d, rel


def difference_text(d: float, rel: float, unit: str = "1",
                    percent: bool = False) -> str:
    """A difference as a cell: '+0.12 Å (+5.3 %)', '-2.1 points' for a
    percentage row, '—' where there is none."""
    if not math.isfinite(d):
        return _DASH
    if percent:
        return f"{100.0 * d:+.1f} points"
    text = f"{d:+.4g}"
    if unit and unit not in ("1", "category"):
        text += f" {unit}"
    if math.isfinite(rel):
        text += f" ({rel:+.1f} %)"
    return text


def _axis_text(series: Series) -> str:
    axis = np.asarray(series.axis, dtype=np.float64)
    if axis.size == 0:
        return "no points"
    return (f"{axis.size} points from {axis[0]:.4g} to {axis[-1]:.4g} "
            f"{series.axis_unit}")


def series_difference(a: Series, b: Series) -> tuple[np.ndarray, np.ndarray]:
    """``b − a`` point by point on their common axis: (axis, difference).
    Raises ValueError, in words, when the axes or the units differ."""
    if (a.axis_name, a.axis_unit, a.value_unit) != \
            (b.axis_name, b.axis_unit, b.value_unit):
        raise ValueError(
            f"The axes differ: {a.axis_name} ({a.axis_unit}) in "
            f"{a.value_unit} against {b.axis_name} ({b.axis_unit}) in "
            f"{b.value_unit}; no difference is drawn.")
    xa = np.asarray(a.axis, dtype=np.float64)
    xb = np.asarray(b.axis, dtype=np.float64)
    if xa.shape != xb.shape or not np.allclose(xa, xb, rtol=1e-6,
                                                atol=1e-9):
        raise ValueError(
            f"The axes differ: {_axis_text(a)} against {_axis_text(b)}; "
            "no difference is drawn.")
    return xa.copy(), (np.asarray(b.mean, dtype=np.float64)
                       - np.asarray(a.mean, dtype=np.float64))


def _union_keys(dists: Sequence[Distribution]) -> list:
    from .md_plot import _key_rank

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
    return keys


def distribution_difference(a: Distribution, b: Distribution
                            ) -> tuple[list, np.ndarray]:
    """``b − a`` key by key over the union of the keys (a key absent from a
    model counts 0 there): (keys, difference). Raises ValueError when one
    holds fractions and the other counts."""
    if a.kind != b.kind:
        raise ValueError(
            f"The distributions differ in kind ({a.kind} against {b.kind}); "
            "no difference is drawn.")
    keys = _union_keys([a, b])
    da = dict(zip(a.keys, np.asarray(a.mean, dtype=np.float64)))
    db = dict(zip(b.keys, np.asarray(b.mean, dtype=np.float64)))
    return keys, np.array([db.get(k, 0.0) - da.get(k, 0.0) for k in keys],
                          dtype=np.float64)


def histogram_difference(a: Histogram, b: Histogram
                         ) -> tuple[np.ndarray, np.ndarray]:
    """``b − a`` bin by bin on common edges: (edges, difference). Raises
    ValueError, in words, when the edges, the unit or the normalisation
    differ."""
    if (a.unit, a.density) != (b.unit, b.density):
        raise ValueError(
            f"The histograms differ: {a.unit}, "
            f"{'density' if a.density else 'counts'} against {b.unit}, "
            f"{'density' if b.density else 'counts'}; no difference is "
            "drawn.")
    ea = np.asarray(a.edges, dtype=np.float64)
    eb = np.asarray(b.edges, dtype=np.float64)
    if ea.shape != eb.shape or not np.allclose(ea, eb, rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"The bin edges differ: {ea.size - 1} bins from {ea[0]:.4g} to "
            f"{ea[-1]:.4g} against {eb.size - 1} bins from {eb[0]:.4g} to "
            f"{eb[-1]:.4g} {a.unit}; no difference is drawn.")
    return ea.copy(), (np.asarray(b.mean, dtype=np.float64)
                       - np.asarray(a.mean, dtype=np.float64))


def difference_figure(pairs: Sequence[tuple[str, object]], *, title: str,
                      styles: Sequence[SeriesStyle] | None = None) -> Figure:
    """The difference figure of one descriptor across models: one series
    per model after the first, each the model named minus the first model
    ('B − A'). Series point by point, Histogram bin by bin, Distribution
    key by key, Scalar as bars of the differences. Raises ValueError, in
    words, when a pair cannot be subtracted (:func:`series_difference` and
    the others)."""
    if len(pairs) < 2:
        raise ValueError("Two models are needed for a difference.")
    label_a, a = pairs[0]
    kind = type(a)
    if any(type(c) is not kind for _l, c in pairs):
        raise ValueError("The descriptors differ in kind across the models; "
                         "no difference is drawn.")

    def style_of(k: int) -> SeriesStyle:
        return styles[k] if styles is not None and k < len(styles) \
            else style_for(k)

    series: list[StatSeries] = []
    if kind is Series:
        for k, (label_b, b) in enumerate(pairs[1:], start=1):
            axis, diff = series_difference(a, b)
            s = StatSeries(f"{label_b} − {label_a}", axis, diff, None,
                           kind="line", markers=axis.size <= 60,
                           band=False)
            s.style = style_of(k)
            series.append(s)
        return Figure(series, axis_symbol(a.axis_name), a.axis_unit or "1",
                      f"difference in {title}", a.value_unit or "1",
                      title=f"{title}: difference",
                      annotations=(f"Each curve is the model named minus "
                                   f"{label_a}, point by point on the "
                                   "common axis.",))
    if kind is Histogram:
        for k, (label_b, b) in enumerate(pairs[1:], start=1):
            edges, diff = histogram_difference(a, b)
            s = StatSeries(f"{label_b} − {label_a}",
                           0.5 * (edges[:-1] + edges[1:]), diff, None,
                           kind="step", edges=edges, band=False,
                           markers=False)
            s.style = style_of(k)
            series.append(s)
        y_name = ("difference in probability density" if a.density
                  else "difference in samples per bin")
        return Figure(series, title, a.unit or "1", y_name,
                      a.value_unit or "1", title=f"{title}: difference",
                      annotations=(f"Each step is the model named minus "
                                   f"{label_a}, bin by bin on the common "
                                   "edges; "
                                   + ("densities as the runs state them."
                                      if a.density else
                                      "counts as the runs state them."),))
    if kind is Distribution:
        keys = _union_keys([c for _l, c in pairs])
        x = np.arange(len(keys), dtype=np.float64)
        da = dict(zip(a.keys, np.asarray(a.mean, dtype=np.float64)))
        for k, (label_b, b) in enumerate(pairs[1:], start=1):
            distribution_difference(a, b)           # the kind check
            db = dict(zip(b.keys, np.asarray(b.mean, dtype=np.float64)))
            diff = np.array([db.get(key, 0.0) - da.get(key, 0.0)
                             for key in keys])
            s = StatSeries(f"{label_b} − {label_a}", x, diff, None,
                           kind="bars")
            s.style = style_of(k)
            series.append(s)
        y_name, y_unit = (("difference in fraction", "1")
                          if a.kind == "fraction"
                          else (f"difference in number per {a.row_kind}",
                                "count"))
        return Figure(series, title, "category", y_name, y_unit,
                      title=f"{title}: difference",
                      categories=tuple(category_label(k) for k in keys),
                      annotations=(f"Each bar is the model named minus "
                                   f"{label_a}, key by key; a key absent "
                                   "from a model counts 0 there. "
                                   + ("Fractions as the runs state them."
                                      if a.kind == "fraction" else
                                      "Counts as the runs state them."),))
    if kind is Scalar:
        if len({c.unit for _l, c in pairs}) != 1:
            raise ValueError("The units differ across the models; no "
                             "difference is drawn.")
        labels = [f"{label_b} − {label_a}" for label_b, _b in pairs[1:]]
        diffs = [float(b.mean) - float(a.mean) for _l, b in pairs[1:]]
        s = StatSeries(f"difference from {label_a}",
                       np.arange(len(diffs), dtype=np.float64), diffs,
                       None, kind="bars")
        s.style = style_of(1)
        return Figure([s], "model", "category", f"difference in {title}",
                      a.unit or "1", title=f"{title}: difference",
                      categories=tuple(labels),
                      annotations=(f"Each bar is the model named minus "
                                   f"{label_a} (means over the frames of "
                                   "each run).",))
    raise ValueError("Only curves, histograms, distributions and numbers "
                     "are subtracted; this descriptor holds rows.")


# ---------------------------------------------------------------------------
# the side-by-side overview and the comparison table (no widgets)
# ---------------------------------------------------------------------------

def overview_rows(labelled: Sequence[tuple[str, object]]) -> list[dict]:
    """The headline measurements of every ticked run, aligned row by row.

    One dict per measurement key met in any run, in question-group order
    then as first met: ``group``, ``analysis``, ``key``, ``label``,
    ``unit``, ``percent``, and per model (in the order given) ``present``,
    ``values``, ``stds``, ``texts`` (the cell), ``bars`` (aligned (label,
    fraction) pairs over the union of the bar labels, () where the model
    lacks the row). ``differences`` holds, per model after the first,
    ``(b − a, relative %)`` from :func:`difference`; ``difference_texts``
    the cells, a bar row's as per-key points.
    """
    n = len(labelled)
    rows: dict[tuple, dict] = {}
    order: list[tuple] = []
    for i, (_label, result) in enumerate(labelled):
        for analysis, output in result.outputs.items():
            group = group_of(analysis)
            for measure in analysis_measures(analysis, output):
                key = (group, analysis, measure.key)
                row = rows.get(key)
                if row is None:
                    row = {"group": group, "analysis": analysis,
                           "key": measure.key, "label": measure.label,
                           "unit": measure.unit, "percent": measure.percent,
                           "present": [False] * n, "values": [_NAN] * n,
                           "stds": [_NAN] * n, "texts": [""] * n,
                           "bars": [()] * n}
                    rows[key] = row
                    order.append(key)
                row["present"][i] = True
                row["values"][i] = float(measure.value)
                row["stds"][i] = float(measure.std)
                row["texts"][i] = measure.text
                row["bars"][i] = tuple(measure.bar)
    titles = group_titles()
    first_seen = {key: i for i, key in enumerate(order)}
    order = sorted(order, key=lambda key: (
        titles.index(key[0]) if key[0] in titles else len(titles),
        first_seen[key]))
    out = []
    for key in order:
        row = rows[key]
        if any(row["bars"]):
            labels: list[str] = []
            for bar in row["bars"]:
                for k, _f in bar:
                    if k not in labels:
                        labels.append(k)
            row["bar_labels"] = labels
            aligned = []
            for bar, present in zip(row["bars"], row["present"]):
                table = dict(bar)
                aligned.append(tuple((k, float(table.get(k, 0.0)))
                                     for k in labels) if present else ())
            row["bars"] = aligned
        row["differences"] = [difference(row["values"][0], v,
                                         percent=row["percent"])
                              for v in row["values"][1:]]
        texts = []
        for j, (d, rel) in enumerate(row["differences"], start=1):
            if row["bars"][0] and row["bars"][j]:
                first, other = dict(row["bars"][0]), dict(row["bars"][j])
                parts = [f"{k} {100.0 * (other.get(k, 0.0) - first.get(k, 0.0)):+.1f}"
                         for k in row.get("bar_labels", [])]
                texts.append(" · ".join(parts) + " points" if parts
                             else _DASH)
            else:
                texts.append(difference_text(d, rel, row["unit"],
                                             row["percent"]))
        row["difference_texts"] = texts
        out.append(row)
    return out


def _in_scope(scope, analysis: str, name: str) -> bool:
    if not scope or scope[0] == "all":
        return True
    if scope[0] == "group":
        return group_of(analysis) == scope[1]
    if scope[0] == "analysis":
        return analysis == scope[1]
    if scope[0] == "item":
        return analysis == scope[1] and name == scope[2]
    return True


def comparison_rows(labelled: Sequence[tuple[str, object]], *,
                    scope: tuple | None = None, filter_text: str = ""
                    ) -> list[dict]:
    """Every descriptor any ticked run holds, as one table: question,
    analysis, descriptor (readable name), id, kind, key (a distribution's
    category, '' otherwise), unit, one column per model (mean ± std over
    the frames; the shape of a curve or a histogram; '—' where the model
    lacks it), the difference column(s) 'B − A', then the plain numbers
    '<model> mean' / '<model> std' per model for a reader of the file.

    ``scope``: ('all',), ('group', title), ('analysis', name) or ('item',
    analysis, descriptor) narrows the rows; ``filter_text`` keeps the rows
    whose descriptor, id or analysis holds the text.
    """
    labels = [label for label, _r in labelled]
    results = [result for _l, result in labelled]
    needle = str(filter_text or "").strip().lower()
    out: list[dict] = []

    def base(analysis: str, name: str, badge: str, unit: str, key: str
             ) -> dict:
        return {"question": group_of(analysis),
                "analysis": analysis_title(analysis),
                "descriptor": display_name(name), "id": name, "kind": badge,
                "key": key, "unit": unit}

    def finish(row: dict, means: list[float], stds: list[float],
               texts: list[str], unit: str, percent: bool = False) -> dict:
        for label, text in zip(labels, texts):
            row[label] = text
        for label_b, mean in zip(labels[1:], means[1:]):
            d, rel = difference(means[0], mean, percent=percent)
            row[f"{label_b} − {labels[0]}"] = difference_text(d, rel, unit,
                                                               percent)
        for label, mean, std in zip(labels, means, stds):
            row[f"{label} mean"] = mean if math.isfinite(mean) else None
            row[f"{label} std"] = std if math.isfinite(std) else None
        return row

    for analysis, name in all_descriptors(results):
        if not _in_scope(scope, analysis, name):
            continue
        if needle and not any(needle in text.lower() for text in (
                display_name(name), name, analysis_title(analysis),
                analysis)):
            continue
        containers = []
        for result in results:
            output = result.outputs.get(analysis)
            containers.append(output.tables.get(name)
                              if output is not None else None)
        present = [c for c in containers if c is not None]
        kinds = {type(c) for c in present}
        badge = kind_badge(present[0]) if present else "table"
        if len(kinds) == 1 and kinds == {Scalar}:
            unit = present[0].unit or "1"
            means = [float(c.mean) if c is not None else _NAN
                     for c in containers]
            stds = [float(c.std) if c is not None else _NAN
                    for c in containers]
            texts = [_pm(m, s) if c is not None else _DASH
                     for c, m, s in zip(containers, means, stds)]
            out.append(finish(base(analysis, name, badge, unit, ""),
                              means, stds, texts, unit))
        elif len(kinds) == 1 and kinds == {Distribution}:
            unit = "1" if present[0].kind == "fraction" else "count"
            keys = _union_keys(present)
            lookup = [dict(zip(c.keys, zip(np.asarray(c.mean, float),
                                           np.asarray(c.std, float))))
                      if c is not None else None for c in containers]
            for key in keys:
                means, stds, texts = [], [], []
                for table in lookup:
                    if table is None:
                        means.append(_NAN)
                        stds.append(_NAN)
                        texts.append(_DASH)
                        continue
                    mean, std = table.get(key, (0.0, _NAN))
                    means.append(float(mean))
                    stds.append(float(std))
                    texts.append(_pm(float(mean), float(std)))
                out.append(finish(base(analysis, name, badge, unit,
                                       category_label(key)),
                                  means, stds, texts, unit,
                                  percent=(unit == "1")))
        else:
            texts = []
            for c in containers:
                if c is None:
                    texts.append(_DASH)
                elif isinstance(c, Series):
                    texts.append(f"curve, {_axis_text(c)}")
                elif isinstance(c, Histogram):
                    texts.append(f"histogram, {c.n_bins} bins of "
                                 f"{c.unit or '1'}")
                elif isinstance(c, Scalar):
                    texts.append(_pm(float(c.mean), float(c.std)))
                elif isinstance(c, Distribution):
                    texts.append(f"{len(c.keys)} categories")
                else:
                    rows_of = getattr(c, "rows", None)
                    texts.append(f"{len(rows_of)} rows"
                                 if rows_of is not None else "rows")
            unit = ""
            if present and isinstance(present[0], Series):
                unit = present[0].value_unit or "1"
            elif present and isinstance(present[0], Histogram):
                unit = present[0].value_unit or "1"
            row = base(analysis, name, badge, unit, "")
            for label, text in zip(labels, texts):
                row[label] = text
            for label_b in labels[1:]:
                row[f"{label_b} − {labels[0]}"] = _DASH
            for label in labels:
                row[f"{label} mean"] = None
                row[f"{label} std"] = None
            out.append(row)
    return out


def comparison_header(labelled: Sequence[tuple[str, object]],
                      extra: Sequence[str] = ()) -> list[str]:
    """The provenance lines every saved comparison carries: each model with
    its file, frames, v_bond and formers, then where the runs' parameters
    differ, then ``extra``."""
    try:
        from ..version import NAME, __version__
        lines = [f"{NAME} {__version__} comparison of {len(labelled)} "
                 "models"]
    except ImportError:                       # pragma: no cover - packaging
        lines = [f"FACET comparison of {len(labelled)} models"]
    for label, result in labelled:
        lines.append(f"{label}: {run_header(result)}")
    lines.extend(parameter_differences(labelled))
    lines.extend(str(line) for line in extra)
    return [" ".join(line.split()) for line in lines]


def _csv_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return ""
        return repr(number)
    return str(value)


def _flat_overview(rows: Sequence[dict], labels: Sequence[str]
                   ) -> list[dict]:
    out = []
    for row in rows:
        flat = {"question": row["group"],
                "analysis": analysis_title(row["analysis"]),
                "measurement": row["label"], "key": row["key"],
                "unit": row["unit"]}
        for label, text, value, std in zip(labels, row["texts"],
                                           row["values"], row["stds"]):
            flat[label] = text or _DASH
            flat[f"{label} value"] = value if math.isfinite(value) else None
            flat[f"{label} std"] = std if math.isfinite(std) else None
        for label_b, text in zip(labels[1:], row["difference_texts"]):
            flat[f"{label_b} − {labels[0]}"] = text
        out.append(flat)
    return out


def write_comparison(path, labelled: Sequence[tuple[str, object]],
                     *, table_rows: Sequence[dict] | None = None,
                     overview: Sequence[dict] | None = None,
                     extra_header: Sequence[str] = ()) -> Path:
    """Write the comparison: an ``.xlsx`` workbook (sheets provenance,
    overview, table) when the path ends so, else one CSV of the table with
    the provenance as '#' lines. Every float in full."""
    labels = [label for label, _r in labelled]
    table = list(table_rows if table_rows is not None
                 else comparison_rows(labelled))
    flat = _flat_overview(overview if overview is not None
                          else overview_rows(labelled), labels)
    header = comparison_header(labelled, extra_header)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".xlsx":
        sheets = {"overview": flat or [{"note": "no measurements"}],
                  "table": table or [{"note": "no descriptors"}]}
        return exporters.write_xlsx(sheets, target, provenance=header)
    columns: list[str] = []
    for row in table:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(target, "w", newline="", encoding="utf-8") as handle:
        for line in header:
            handle.write(f"# {line}\n")
        writer = csv.DictWriter(handle, fieldnames=columns, restval="")
        writer.writeheader()
        for row in table:
            writer.writerow({k: _csv_cell(row.get(k)) for k in columns})
    return target


# ---------------------------------------------------------------------------
# the 3D row (no widgets)
# ---------------------------------------------------------------------------

def _x_extent(scene: Scene) -> tuple[float, float]:
    x = np.asarray(scene.atom_position[:, 0], dtype=np.float64)
    r = np.asarray(scene.atom_radius, dtype=np.float64)
    lo, hi = float((x - r).min()), float((x + r).max())
    if len(scene.cell_segments):
        cx = np.asarray(scene.cell_segments, dtype=np.float64)[..., 0]
        lo, hi = min(lo, float(cx.min())), max(hi, float(cx.max()))
    return lo, hi


def side_by_side_scene(scenes: Sequence[Scene | None], gap_ang: float
                       ) -> tuple[Scene, list[np.ndarray]]:
    """One Scene of several, laid along x: each block begins ``gap_ang``
    after the previous block ends (its atoms with their radii and its box
    edges), the row centred on x = 0; y and z are left as they are.
    Returns the merged scene (``facet.gl.scene.merge_scenes``, so
    ``structure_of`` names each atom's block) and the offsets applied, one
    per scene given, zeros for a scene left out (None or empty)."""
    gap = float(gap_ang)
    offsets: list[np.ndarray] = []
    kept: list[Scene] = []
    kept_offsets: list[np.ndarray] = []
    x = 0.0
    for scene in scenes:
        if scene is None or not scene.n_atoms:
            offsets.append(np.zeros(3))
            continue
        lo, hi = _x_extent(scene)
        offset = np.array([x - lo, 0.0, 0.0])
        offsets.append(offset)
        kept.append(scene)
        kept_offsets.append(offset)
        x += (hi - lo) + gap
    span = x - gap if kept else 0.0
    shift = np.array([0.5 * span, 0.0, 0.0])
    offsets = [o - shift if s is not None and s.n_atoms else o
               for o, s in zip(offsets, scenes)]
    kept_offsets = [o - shift for o in kept_offsets]
    if not kept:
        return Scene(), offsets
    if len(kept) == 1:
        merged = merge_scenes(kept, kept_offsets)
    else:
        merged = merge_scenes(kept, kept_offsets)
    return merged, offsets


def atoms_only(scene: Scene) -> Scene:
    """A shallow copy of a scene with its bonds left out (the QPainter
    tier above the atoms-only limit)."""
    out = copy.copy(scene)
    out.bond_a = np.zeros((0, 3), np.float32)
    out.bond_b = np.zeros((0, 3), np.float32)
    out.bond_radius = np.zeros(0, np.float32)
    out.bond_color_a = np.zeros((0, 3), np.float32)
    out.bond_color_b = np.zeros((0, 3), np.float32)
    out.bond_valence = np.zeros(0, np.float32)
    out.bond_distance = np.zeros(0, np.float32)
    out.bond_atoms = np.zeros((0, 2), np.int32)
    out.bond_cation = np.zeros(0, np.int8)
    out.bond_occupancy = np.ones(0, np.float32)
    out._bond_base_a = np.zeros((0, 3), np.float32)
    out._bond_base_b = np.zeros((0, 3), np.float32)
    out.overlay_meshes = []
    return out


def emphasise_element(scene: Scene, symbol: str, theme=None) -> Scene:
    """A shallow copy of a merged scene with every atom that is not
    ``symbol`` paled and shrunk, and the bonds to none of its atoms with
    them -- what a model's Elements dock does to its own frame
    (``md_highlight._dim``'s constants), here by element name since a
    merged scene's rows come from several frames."""
    from ..core import theme as theme_mod
    from .md_highlight import DIM_ATOM_RADIUS, DIM_BOND_RADIUS, DIM_KEEP

    out = copy.copy(scene)
    keep = np.array([e == symbol for e in scene.atom_element], dtype=bool)
    if keep.size != scene.n_atoms or not symbol:
        return out
    theme = theme or getattr(scene, "theme", None)
    bg = np.array(getattr(theme, "background", None)
                  or theme_mod.FALLBACK_BACKGROUND, np.float32)
    pale = (1 - DIM_KEEP) * bg + DIM_KEEP * scene.atom_color
    out.atom_color = np.where(keep[:, None], scene.atom_color,
                              pale).astype(np.float32)
    out.atom_radius = np.where(keep, scene.atom_radius,
                               scene.atom_radius * DIM_ATOM_RADIUS
                               ).astype(np.float32)
    if scene.n_bonds:
        ends = np.asarray(scene.bond_atoms, int)
        bond_keep = keep[ends[:, 0]] | keep[ends[:, 1]]
        for name in ("bond_color_a", "bond_color_b"):
            c = getattr(scene, name)
            setattr(out, name, np.where(bond_keep[:, None], c,
                                        (1 - DIM_KEEP) * bg + DIM_KEEP * c)
                    .astype(np.float32))
        out.bond_radius = np.where(bond_keep, scene.bond_radius,
                                   scene.bond_radius * DIM_BOND_RADIUS
                                   ).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# swatches (QPainter primitives)
# ---------------------------------------------------------------------------

def swatch_pixmap(style: SeriesStyle, size: int = 16) -> QPixmap:
    """A model's legend swatch: its line through its marker, in its
    colour, so a list row or a column header carries both."""
    size = int(size)
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    colour = QColor(*style.colour)
    painter.setPen(QPen(colour, 2.0))
    painter.drawLine(QPointF(1, size / 2), QPointF(size - 1, size / 2))
    painter.setBrush(QBrush(colour))
    x = y = size / 2.0
    r = size * 0.26
    shape = style.marker
    if shape == "circle":
        painter.drawEllipse(QPointF(x, y), r, r)
    elif shape == "square":
        painter.drawRect(QRectF(x - r, y - r, 2 * r, 2 * r))
    elif shape in ("triangle", "triangle down"):
        sign = -1.0 if shape == "triangle" else 1.0
        painter.drawPolygon(QPolygonF([QPointF(x, y + sign * r * 1.1),
                                       QPointF(x - r * 1.05, y - sign * r
                                               * 0.8),
                                       QPointF(x + r * 1.05, y - sign * r
                                               * 0.8)]))
    elif shape == "diamond":
        painter.drawPolygon(QPolygonF([QPointF(x, y - r * 1.2),
                                       QPointF(x + r * 0.9, y),
                                       QPointF(x, y + r * 1.2),
                                       QPointF(x - r * 0.9, y)]))
    else:
        painter.setPen(QPen(colour, 2.2))
        if shape == "cross":
            painter.drawLine(QPointF(x - r, y - r), QPointF(x + r, y + r))
            painter.drawLine(QPointF(x - r, y + r), QPointF(x + r, y - r))
        else:
            painter.drawLine(QPointF(x - r * 1.2, y), QPointF(x + r * 1.2, y))
            painter.drawLine(QPointF(x, y - r * 1.2), QPointF(x, y + r * 1.2))
    painter.end()
    return pixmap


# ---------------------------------------------------------------------------
# the Elements dock page
# ---------------------------------------------------------------------------

class CompareElements(QWidget):
    """The Elements dock while comparing: one row per element any ticked
    model holds, with that element's atom count and mean CN in each model
    (at that model's strip v_bond, on its frame shown). A click emphasises
    the element in the 3D side by side (``elementChosen``; '' clears)."""

    elementChosen = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chosen = ""
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self.table = QTableWidget(0, 1)
        self.table.setHorizontalHeaderLabels(["element"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setToolTip(
            "Each ticked model's atoms and mean CN of the element, on the "
            "frame it has on show at its own v_bond. Click an element to "
            "emphasise its atoms in the 3D side by side (click again to "
            "show every atom alike).")
        self.table.itemClicked.connect(self._on_click)
        column.addWidget(self.table, 1)
        self.note = QLabel("Tick models with a frame drawn.")
        self.note.setWordWrap(True)
        chrome.mark_hint(self.note)
        column.addWidget(self.note)

    def set_models(self, models: Sequence[dict]) -> None:
        """``models``: dicts with ``label``, ``style``, ``v_bond`` and
        ``rows`` (element -> (atoms, mean CN))."""
        table = self.table
        table.blockSignals(True)
        table.clear()
        table.setRowCount(0)
        headers = ["element"]
        for model in models:
            glyph = marker_glyph(model["style"])
            headers += [f"atoms {glyph}", f"CN {glyph}"]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        for i, model in enumerate(models):
            colour = QColor(*model["style"].colour)
            for c in (1 + 2 * i, 2 + 2 * i):
                item = table.horizontalHeaderItem(c)
                item.setForeground(QBrush(colour))
                item.setToolTip(f"{model['label']}: "
                                + ("atoms of the element"
                                   if c % 2 else
                                   f"mean CN at v_bond {model['v_bond']:.4g} "
                                   "v.u."))
        elements: list[str] = []
        for model in models:
            for element in model["rows"]:
                if element not in elements:
                    elements.append(element)
        for element in elements:
            r = table.rowCount()
            table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(element))
            for i, model in enumerate(models):
                found = model["rows"].get(element)
                atoms = f"{int(found[0]):,}".replace(",", " ") if found \
                    else _DASH
                cn = (f"{found[1]:.2f}" if found and math.isfinite(found[1])
                      else _DASH)
                for c, text in ((1 + 2 * i, atoms), (2 + 2 * i, cn)):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    table.setItem(r, c, item)
            if element == self.chosen:
                table.selectRow(r)
        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.blockSignals(False)
        if models:
            self.note.setText(
                "mean CN on the frame shown, at each model's v_bond: "
                + "; ".join(f"{marker_glyph(m['style'])} {m['label']} "
                            f"{m['v_bond']:.4g} v.u." for m in models))
        else:
            self.note.setText("Tick models with a frame drawn.")

    def elements(self) -> list[str]:
        return [self.table.item(r, 0).text()
                for r in range(self.table.rowCount())]

    def choose(self, symbol: str) -> None:
        symbol = str(symbol or "")
        if symbol == self.chosen:
            return
        self.chosen = symbol
        self.table.clearSelection()
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).text() == symbol:
                self.table.selectRow(r)
        self.elementChosen.emit(self.chosen)

    def _on_click(self, item: QTableWidgetItem) -> None:
        symbol = self.table.item(item.row(), 0).text()
        if symbol == self.chosen:
            self.chosen = ""
            self.table.clearSelection()
            self.elementChosen.emit("")
        else:
            self.choose(symbol)


# ---------------------------------------------------------------------------
# the side-by-side overview table
# ---------------------------------------------------------------------------

class _OverviewTable(QTableWidget):
    """The Overview page's table: a group row, an analysis row, then the
    measurements, one column per model with its swatch and the difference
    column(s); a distribution row draws its aligned fraction bars."""

    LABEL_WIDTH = 330
    MODEL_WIDTH = 170
    DIFF_WIDTH = 190

    def __init__(self, parent=None):
        super().__init__(0, 0, parent)
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setHighlightSections(False)
        self.setToolTip("The headline measurements of each ticked run, "
                        "aligned row by row; '—' where a run lacks the "
                        "row. The last column(s) are the model named minus "
                        "the first ticked model.")
        self.sentences: list[str] = []

    def fill(self, rows: Sequence[dict], labels: Sequence[str],
             styles: Sequence[SeriesStyle]) -> None:
        self.clear()
        self.setRowCount(0)
        self.sentences = []
        n = len(labels)
        columns = ["Measurement", "unit"] + list(labels) + [
            f"{label} − {labels[0]}" for label in labels[1:]] if labels \
            else ["Measurement", "unit"]
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        for i, (label, style) in enumerate(zip(labels, styles)):
            item = self.horizontalHeaderItem(2 + i)
            item.setIcon(QIcon(swatch_pixmap(style)))
            item.setToolTip(f"{label}: mean ± standard deviation over the "
                            "frames of its run")
        for j, label in enumerate(labels[1:]):
            item = self.horizontalHeaderItem(2 + n + j)
            item.setToolTip(f"{label} minus {labels[0]}: the difference, "
                            "and the relative change where the first "
                            "value is not 0; a percentage row in points")
        bold = self.font()
        bold.setBold(True)
        muted = self.palette().placeholderText().color()
        last_group = last_analysis = None
        for row in rows:
            if row["group"] != last_group:
                self._spanning(row["group"], bold, None)
                last_group, last_analysis = row["group"], None
            if row["analysis"] != last_analysis:
                self._spanning(analysis_title(row["analysis"]), None, muted)
                last_analysis = row["analysis"]
            r = self.rowCount()
            self.insertRow(r)
            label = QTableWidgetItem(row["label"])
            label.setToolTip(f"{row['label']}\n{row['analysis']}: "
                             f"{row['key']}")
            self.setItem(r, 0, label)
            self.setItem(r, 1, QTableWidgetItem(
                "%" if row["percent"] else (row["unit"] or "")))
            for i in range(n):
                if row["bars"][i]:
                    bar = FractionBar(row["bars"][i])
                    self.setCellWidget(r, 2 + i, bar)
                    text = row["texts"][i]
                else:
                    text = row["texts"][i] if row["present"][i] else _DASH
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setToolTip(text)
                    self.setItem(r, 2 + i, item)
                self.sentences.append(f"{row['label']} ({labels[i]}): "
                                      f"{text or _DASH}")
            for j, text in enumerate(row["difference_texts"]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                item.setToolTip(text)
                self.setItem(r, 2 + n + j, item)
                self.sentences.append(f"{row['label']} ({columns[2 + n + j]})"
                                      f": {text}")
        self.setColumnWidth(0, self.LABEL_WIDTH)
        self.setColumnWidth(1, 64)
        for i in range(n):
            self.setColumnWidth(2 + i, self.MODEL_WIDTH)
        for j in range(max(n - 1, 0)):
            self.setColumnWidth(2 + n + j, self.DIFF_WIDTH)
        self.resizeRowsToContents()

    def _spanning(self, text: str, font, colour) -> None:
        r = self.rowCount()
        self.insertRow(r)
        item = QTableWidgetItem(text)
        if font is not None:
            item.setFont(font)
        if colour is not None:
            item.setForeground(QBrush(colour))
        self.setItem(r, 0, item)
        self.setSpan(r, 0, 1, max(self.columnCount(), 1))
        self.sentences.append(text)


# ---------------------------------------------------------------------------
# the workspace
# ---------------------------------------------------------------------------

class _Ref:
    """One basket row: a crystal site or a model element."""

    def __init__(self, kind: str, entry, key):
        self.kind = kind            # 'crystal' | 'model'
        self.entry = entry
        self.key = key              # site index | element symbol

    def matches(self, other: "_Ref") -> bool:
        return (self.kind == other.kind and self.entry is other.entry
                and self.key == other.key)


class CompareWorkspace(QObject):
    """The Compare workspace: four widgets for the window's stacked regions
    (``widgets``: the right column, the Elements dock page, the toolbar,
    the centre canvas) and the state behind them.

    The basket: ``add_crystal_site``, ``add_model_element``,
    ``remove_ref``, ``clear``, ``basket_rows``. The models:
    ``set_model_candidates`` (every model entry; those with a result can
    be ticked), ``ticked_results``. The centre: ``show_page`` /
    ``page_shown`` ('overview', 'figure', 'table', '3d'),
    ``select_descriptor``. Figures as data: ``staircase_figure``,
    ``crystal_vs_glass_figure``. Files: ``export_comparison``.

    Signals: ``closeRequested`` (Back), ``statusMessage(str)``.
    """

    closeRequested = Signal()
    statusMessage = Signal(str)

    BASKET_COLUMNS = ("structure", "site", "CN (1)", "frame", "")

    def __init__(self, project, window=None, *, theme=None):
        super().__init__(window if isinstance(window, QObject) else None)
        self.project = project
        self.window = window
        self.theme = theme
        self._refs: list[_Ref] = []
        self._candidates: list = []        # ModelEntry with a result
        self._models: list = []            # every model entry seen
        self._stale: set[str] = set()
        self._element = ""
        self._last_3d_s = 0.0
        self._build()
        if theme is not None:
            self.apply_theme(theme)
        self._refresh_all()

    # -- the widgets ---------------------------------------------------------------
    @property
    def widgets(self) -> tuple:
        """(column, elements, toolbar, canvas): the window's first four
        stacked regions, in the order of ``ModelController.widgets``."""
        return (self.column, self.elements, self.toolbar, self.canvas)

    def _build(self) -> None:
        self._build_column()
        self.elements = CompareElements()
        self.elements.elementChosen.connect(self._on_element_chosen)
        self._build_toolbar()
        self._build_canvas()

    def _build_column(self) -> None:
        self.column = QWidget()
        column = QVBoxLayout(self.column)
        column.setContentsMargins(6, 6, 6, 6)
        column.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Compare structures")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch(1)
        self.back_button = QPushButton("Back to the structure")
        self.back_button.setToolTip(
            "Leave the comparison: the selected structure's own column "
            "and centre come back. The basket and the ticks are kept.")
        self.back_button.clicked.connect(self.closeRequested)
        header.addWidget(self.back_button)
        column.addLayout(header)

        models_title = QLabel("Models (tick to compare)")
        chrome.mark_hint(models_title)
        column.addWidget(models_title)
        self.models_list = QListWidget()
        self.models_list.setMaximumHeight(112)
        self.models_list.setToolTip(
            "Every MD model of the project with a finished run. The swatch "
            "is the model's colour and marker in every figure, header and "
            "legend of the comparison.")
        self.models_list.itemChanged.connect(
            lambda _item: self._on_models_changed())
        column.addWidget(self.models_list)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        self.warning.setTextInteractionFlags(Qt.TextSelectableByMouse)
        chrome.mark_hint(self.warning)
        self.warning.hide()
        column.addWidget(self.warning)

        basket_row = QHBoxLayout()
        basket_title = QLabel("Site basket")
        chrome.mark_hint(basket_title)
        basket_row.addWidget(basket_title)
        basket_row.addStretch(1)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Empty the basket.")
        self.clear_button.clicked.connect(self.clear)
        basket_row.addWidget(self.clear_button)
        column.addLayout(basket_row)
        self.basket_table = QTableWidget(0, len(self.BASKET_COLUMNS))
        self.basket_table.setHorizontalHeaderLabels(
            list(self.BASKET_COLUMNS))
        self.basket_table.verticalHeader().setVisible(False)
        self.basket_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.basket_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.basket_table.setMinimumHeight(72)
        self.basket_table.setMaximumHeight(150)
        self.basket_table.setToolTip(
            "References added from a right-click: a row of the Sites list "
            "or an atom in the 3D view (a crystal site, read at the "
            "project's v_bond), or a row of a model's Elements dock (that "
            "element, the per-atom numbers of the frame named at the "
            "model's v_bond). The full numbers are the rows of the "
            "staircase figure (Figure page); × removes a row.")
        self.basket_table.itemClicked.connect(self._on_basket_click)
        column.addWidget(self.basket_table)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter the descriptors…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        column.addWidget(self.filter_edit)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Descriptor", "Kind", "Models"])
        self.tree.setUniformRowHeights(True)
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.setColumnWidth(1, 56)
        self.tree.setColumnWidth(2, 60)
        self.tree.setItemDelegateForColumn(0, _NameDelegate(self.tree))
        self.tree.setToolTip(
            "The site-to-site figures, then every descriptor of the ticked "
            "runs by question; 'Models' counts the runs that hold it. "
            "Choose a row to draw it on the Figure page; a question or an "
            "analysis narrows the Table page.")
        self.tree.currentItemChanged.connect(
            lambda current, _previous=None: self._on_tree_current(current))
        self.tree.itemClicked.connect(
            lambda item, _column=0: self._on_tree_current(item, True))
        column.addWidget(self.tree, 1)

    def _build_toolbar(self) -> None:
        self.toolbar = QWidget()
        row = QHBoxLayout(self.toolbar)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)
        title = QLabel("Compare")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        row.addWidget(title)
        self._page_group = QButtonGroup(self)
        self._page_group.setExclusive(True)
        self.page_buttons: dict[str, QToolButton] = {}
        tips = {"overview": "The headline measurements of each ticked run, "
                            "side by side with a difference column.",
                "figure": "The descriptor chosen in the tree at full width, "
                          "the models overlaid or their difference.",
                "table": "Every descriptor as one table: a column per model "
                         "and the difference.",
                "3d": "The frames the models have on show, laid along x in "
                      "one scene."}
        for key, label in PAGES:
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setToolTip(tips[key])
            button.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self._page_group.addButton(button)
            row.addWidget(button)
            self.page_buttons[key] = button
        row.addStretch(1)
        self.save_figure_button = QPushButton("Save figure…")
        self.save_figure_button.setToolTip(
            "Write the figure shown as SVG, PDF or a 600 dpi PNG.")
        self.save_figure_button.clicked.connect(self.save_figure)
        row.addWidget(self.save_figure_button)
        self.save_rows_button = QPushButton("Save rows…")
        self.save_rows_button.setToolTip(
            "Write the rows of the page shown as CSV, the provenance as "
            "'#' lines.")
        self.save_rows_button.clicked.connect(self.save_rows)
        row.addWidget(self.save_rows_button)
        self.export_button = QPushButton("Export comparison…")
        self.export_button.setToolTip(
            "Write the whole comparison: the overview and the table, as "
            "CSV or an XLSX workbook, with a provenance header naming every "
            "model, its file, frames, v_bond and formers.")
        self.export_button.clicked.connect(self.choose_export)
        row.addWidget(self.export_button)

    def _build_canvas(self) -> None:
        self.canvas = QStackedWidget()

        # overview
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(6)
        self.overview_header = QLabel("")
        self.overview_header.setWordWrap(True)
        self.overview_header.setTextInteractionFlags(
            Qt.TextSelectableByMouse)
        col.addWidget(self.overview_header)
        self.overview_table = _OverviewTable()
        col.addWidget(self.overview_table, 1)
        self.overview_empty = QLabel(
            "Tick two or more models with finished runs: their headline "
            "measurements are aligned here, row by row, with the "
            "difference between them.")
        self.overview_empty.setWordWrap(True)
        self.overview_empty.setAlignment(Qt.AlignCenter)
        chrome.mark_hint(self.overview_empty)
        col.addWidget(self.overview_empty)
        self.overview_page = page
        self.canvas.addWidget(page)

        # figure
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(6, 6, 6, 0)
        self.difference_check = QCheckBox("Difference (B − A)")
        self.difference_check.setToolTip(
            "Draw each model minus the first ticked model instead of the "
            "overlay: a curve point by point on the common axis, a "
            "distribution key by key, a histogram bin by bin. Refused in "
            "words when the axes differ.")
        self.difference_check.toggled.connect(
            lambda _on: self._refresh_figure())
        row.addWidget(self.difference_check)
        self.difference_note = QLabel("")
        self.difference_note.setWordWrap(True)
        chrome.mark_hint(self.difference_note)
        row.addWidget(self.difference_note, 1)
        col.addWidget(controls)
        self.view = DescriptorView()
        self.view.figure_button.clicked.connect(self.save_figure)
        self.view.rows_button.clicked.connect(self.save_rows)
        self.view.plot.statusMessage.connect(self.statusMessage)
        col.addWidget(self.view, 1)
        self.figure_page = page
        self.canvas.addWidget(page)

        # table
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(8, 8, 8, 8)
        col.setSpacing(6)
        self.table_scope = QLabel("")
        self.table_scope.setWordWrap(True)
        self.table_scope.setTextInteractionFlags(Qt.TextSelectableByMouse)
        col.addWidget(self.table_scope)
        self.table = RowTable()
        self.table.setToolTip(
            "Every descriptor of the ticked runs: mean ± standard deviation "
            "over the frames of each run, and the model named minus the "
            "first. A curve or a histogram is described by its shape; the "
            "Figure page draws it.")
        col.addWidget(self.table, 1)
        self.table_page = page
        self.canvas.addWidget(page)

        # 3D
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        controls = QWidget()
        row = QHBoxLayout(controls)
        row.setContentsMargins(6, 6, 6, 4)
        row.setSpacing(8)
        gap_label = QLabel("Gap between frames")
        row.addWidget(gap_label)
        self.gap_box = QDoubleSpinBox()
        self.gap_box.setRange(0.0, 200.0)
        self.gap_box.setSingleStep(2.0)
        self.gap_box.setDecimals(1)
        self.gap_box.setSuffix(" Å")
        spacing = float(getattr(self.project, "overlay_spacing", 0.0) or 0.0)
        self.gap_box.setValue(spacing if spacing > 0 else DEFAULT_GAP_ANG)
        self.gap_box.setToolTip(
            "The space along x between one frame's block (its atoms and "
            "box) and the next; the Structures dock's spacing when it is "
            f"set, else {DEFAULT_GAP_ANG:g} Å.")
        self.gap_box.valueChanged.connect(lambda _v: self._refresh_3d())
        row.addWidget(self.gap_box)
        self.reframe_button = QPushButton("Reframe")
        self.reframe_button.setToolTip("Fit the whole row in the view.")
        self.reframe_button.clicked.connect(self._reframe)
        row.addWidget(self.reframe_button)
        self.scene_note = QLabel("")
        self.scene_note.setWordWrap(True)
        self.scene_note.setTextInteractionFlags(Qt.TextSelectableByMouse)
        chrome.mark_hint(self.scene_note)
        row.addWidget(self.scene_note, 1)
        col.addWidget(controls)
        from ..gl.view import StructureView

        self.view3d = StructureView()
        if self.theme is not None:
            self.view3d.set_theme(self.theme)
        self.view3d.ready.connect(lambda _caps: self._invalidate("3d"))
        self.view3d.measured.connect(self.statusMessage)
        col.addWidget(self.view3d, 1)
        self.scene_page = page
        self.canvas.addWidget(page)

        self._pages = {"overview": self.overview_page,
                       "figure": self.figure_page, "table": self.table_page,
                       "3d": self.scene_page}
        self.show_page("overview")

    # -- pages ------------------------------------------------------------------------
    def show_page(self, name: str) -> None:
        """Put one of 'overview', 'figure', 'table', '3d' in the centre
        and refresh it when it is stale."""
        if name not in self._pages:
            raise ValueError(f"{name!r} is not one of {tuple(self._pages)}")
        self.canvas.setCurrentWidget(self._pages[name])
        button = self.page_buttons[name]
        if not button.isChecked():
            button.setChecked(True)
        if name in self._stale:
            self._refresh_page(name)
        self.save_figure_button.setEnabled(name == "figure"
                                           and self.view.figure is not None)
        self.save_rows_button.setEnabled(name != "3d")

    def page_shown(self) -> str:
        current = self.canvas.currentWidget()
        for name, page in self._pages.items():
            if page is current:
                return name
        return ""

    def _invalidate(self, *names: str) -> None:
        self._stale.update(names)
        shown = self.page_shown()
        if shown in self._stale:
            self._refresh_page(shown)

    def _refresh_page(self, name: str) -> None:
        self._stale.discard(name)
        if name == "overview":
            self._refresh_overview()
        elif name == "figure":
            self._refresh_figure()
        elif name == "table":
            self._refresh_table()
        elif name == "3d":
            self._refresh_3d()

    # -- the basket ----------------------------------------------------------------------
    def refs(self) -> list[tuple[str, object, object]]:
        return [(r.kind, r.entry, r.key) for r in self._refs]

    def add_crystal_site(self, entry, site_index: int) -> bool:
        """A crystal site into the basket (once). True when added."""
        ref = _Ref("crystal", entry, int(site_index))
        if any(ref.matches(r) for r in self._refs):
            return False
        self._refs.append(ref)
        self._refresh_all()
        return True

    def add_model_element(self, entry, element: str) -> bool:
        """An element of an MD model into the basket (once)."""
        ref = _Ref("model", entry, str(element))
        if any(ref.matches(r) for r in self._refs):
            return False
        self._refs.append(ref)
        self._refresh_all()
        return True

    def remove_ref(self, index: int) -> None:
        if 0 <= index < len(self._refs):
            del self._refs[index]
            self._refresh_all()

    def clear(self) -> None:
        self._refs.clear()
        self._refresh_all()

    def drop_entry(self, entry) -> None:
        """Forget a removed entry: its basket rows and its tick."""
        self._refs = [r for r in self._refs if r.entry is not entry]
        self._candidates = [e for e in self._candidates if e is not entry]
        self._models = [e for e in self._models if e is not entry]
        self._fill_models_list()
        self._refresh_all()

    def set_v_bond(self, _v: float | None = None) -> None:
        """The project's bond threshold moved: re-read the crystal rows and
        the marks (the project itself already holds the value)."""
        self._refresh_all()

    def frame_changed(self, entry) -> None:
        """A model drew another frame (or the same at another threshold):
        its basket rows, its Elements rows and the 3D row follow."""
        if entry in self._candidates or any(r.entry is entry
                                            for r in self._refs):
            self._fill_basket(self.basket_rows())
            self._refresh_elements()
            self._invalidate("figure", "3d")

    def _site_result(self, ref: _Ref):
        for result in self.project.results_for(ref.entry):
            if result.site_index == ref.key:
                return result
        return None

    @staticmethod
    def _model_frame(ref_or_entry):
        """(ValenceTable, v_bond, frame index) of a model's frame shown,
        or None before a frame is drawn with its table."""
        entry = getattr(ref_or_entry, "entry", ref_or_entry)
        view = getattr(entry, "frame_view", None)
        table = getattr(view, "table", None)
        if table is None:
            return None
        v_bond = getattr(view, "v_bond_vu", bv.V_BOND_DEFAULT)
        controller = getattr(entry, "controller", None)
        strip = getattr(controller, "strip", None)
        if strip is not None and getattr(strip, "table", None) is not None:
            v_bond = float(strip.v_bond)
        return table, float(v_bond), int(getattr(view, "k", 0))

    def basket_rows(self) -> list[dict]:
        """The basket as numbers: one dict per row, the keys of
        :func:`crystal_values` / :func:`model_values` plus ``structure``,
        ``kind`` and ``frame`` (None for a crystal)."""
        rows = []
        for ref in self._refs:
            row = {"structure": _entry_title(ref.entry), "kind": ref.kind,
                   "frame": None}
            if ref.kind == "crystal":
                result = self._site_result(ref)
                if result is None:
                    row.update({"site": f"site {ref.key}",
                                "missing": "the site is not analysed"})
                else:
                    row.update(crystal_values(result,
                                              float(self.project.v_bond)))
            else:
                found = self._model_frame(ref)
                if found is None:
                    row.update({"site": str(ref.key),
                                "missing": "no frame is drawn yet"})
                else:
                    table, v_bond, k = found
                    row.update(model_values(table, ref.key, v_bond))
                    row["site"] = str(ref.key)
                    row["frame"] = k
            rows.append(row)
        return rows

    def _fill_basket(self, rows: list[dict]) -> None:
        table = self.basket_table
        table.setRowCount(0)
        for row in rows:
            i = table.rowCount()
            table.insertRow(i)
            if "missing" in row:
                cells = (row["structure"], f"{row['site']} "
                         f"({row['missing']})", "", "")
            elif row["kind"] == "crystal":
                cells = (row["structure"], row["site"], str(row["cn"]),
                         _DASH)
            else:
                cells = (row["structure"],
                         f"{row['site']} ({row['atoms']} atoms)",
                         _pm(row["cn"], row["cn_std"]), str(row["frame"]))
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c >= 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, c, item)
            remove = QTableWidgetItem("×")
            remove.setTextAlignment(Qt.AlignCenter)
            remove.setToolTip("Remove this row from the basket")
            table.setItem(i, len(self.BASKET_COLUMNS) - 1, remove)
        table.resizeColumnsToContents()
        self.clear_button.setEnabled(bool(rows))

    def _on_basket_click(self, item: QTableWidgetItem) -> None:
        if item.column() == len(self.BASKET_COLUMNS) - 1:
            self.remove_ref(item.row())

    # -- the models -----------------------------------------------------------------------
    def set_model_candidates(self, entries) -> None:
        """The model entries of the project; those with a result can be
        ticked."""
        self._models = list(entries)
        kept = [id(e) for e in self._candidates]
        self._candidates = [e for e in entries
                            if getattr(e, "result", None) is not None]
        if [id(e) for e in self._candidates] != kept:
            self._fill_models_list()
        self._on_models_changed()

    def _model_labels(self) -> list[str]:
        """One label per candidate, numbered where two files share a name,
        so two runs of one file stay apart in every legend and header."""
        titles = [_entry_title(e) for e in self._candidates]
        labels = []
        for i, title in enumerate(titles):
            if titles.count(title) > 1:
                labels.append(f"{title} ({titles[:i + 1].count(title)})")
            else:
                labels.append(title)
        return labels

    def style_of(self, entry) -> SeriesStyle:
        """The legend style of a model: its position among the models
        with a result, so a model keeps its colour and marker wherever it
        appears while it is listed."""
        for i, candidate in enumerate(self._candidates):
            if candidate is entry:
                return style_for(i)
        return style_for(len(self._candidates))

    def _fill_models_list(self) -> None:
        checked = {id(self.models_list.item(i).data(Qt.UserRole))
                   for i in range(self.models_list.count())
                   if self.models_list.item(i).checkState() == Qt.Checked}
        had_rows = self.models_list.count() > 0
        self.models_list.blockSignals(True)
        self.models_list.clear()
        for entry, label in zip(self._candidates, self._model_labels()):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            tick = (id(entry) in checked or not had_rows) \
                and len(self._candidates) > 1
            item.setCheckState(Qt.Checked if tick else Qt.Unchecked)
            item.setData(Qt.UserRole, entry)
            style = self.style_of(entry)
            item.setIcon(QIcon(swatch_pixmap(style)))
            item.setToolTip(f"{label}: {style.colour_name}, "
                            f"{style.marker} marker; "
                            f"{run_header(entry.result)}")
            self.models_list.addItem(item)
        self.models_list.blockSignals(False)

    def ticked_entries(self) -> list:
        out = []
        for i in range(self.models_list.count()):
            item = self.models_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            entry = item.data(Qt.UserRole)
            if getattr(entry, "result", None) is not None:
                out.append((item.text(), entry))
        return out

    def ticked_results(self) -> list[tuple[str, object]]:
        """(label, ModelResult) of every ticked model."""
        return [(label, entry.result) for label, entry
                in self.ticked_entries()]

    def ticked_styles(self) -> list[SeriesStyle]:
        return [self.style_of(entry) for _l, entry in self.ticked_entries()]

    def _on_models_changed(self) -> None:
        ticked = self.ticked_results()
        differences = parameter_differences(ticked)
        self.warning.setText(
            "The runs' parameters differ. " + " ".join(differences)
            if differences else "")
        self.warning.setVisible(bool(differences))
        self._fill_tree()
        self._refresh_elements()
        self._invalidate("overview", "figure", "table", "3d")

    # -- the tree ----------------------------------------------------------------------------
    def _fill_tree(self) -> None:
        current = self.current_role()
        self.tree.blockSignals(True)
        self.tree.clear()
        to_select = None
        if self._refs:
            head = QTreeWidgetItem(["Site to site", "", f"{len(self._refs)} "
                                    "rows"])
            head.setData(0, _ROLE, ("group", "Site to site"))
            head.setToolTip(0, "The basket's references against each other.")
            self.tree.addTopLevelItem(head)
            leaves = [(("site", "stairs"), "Staircases of the basket rows",
                       "curve", "CN against v_bond on the log axis, one "
                       "curve per basket row")]
            for key, name, unit in VALUE_KINDS:
                leaves.append((("site", "hist", key),
                               f"Crystal against glass — {name}",
                               "distr", f"The model's per-atom {name} "
                               f"({unit}) binned, the crystal site's value "
                               "as a line"))
            for role, text, tag, tip in leaves:
                leaf = QTreeWidgetItem([text, tag, ""])
                leaf.setData(0, _ROLE, role)
                leaf.setToolTip(0, tip)
                head.addChild(leaf)
                if role == current:
                    to_select = leaf
            head.setExpanded(True)
        ticked = self.ticked_results()
        results = [r for _l, r in ticked]
        if results:
            by_group: dict[str, dict[str, list[str]]] = {}
            for analysis, name in all_descriptors(results):
                by_group.setdefault(group_of(analysis), {}).setdefault(
                    analysis, []).append(name)
            for group in group_titles():
                analyses = by_group.get(group)
                if not analyses:
                    continue
                count = sum(len(v) for v in analyses.values())
                head = QTreeWidgetItem([group, "", f"{count} descriptors"])
                head.setData(0, _ROLE, ("group", group))
                head.setToolTip(0, f"{group}: " + ", ".join(
                    analysis_title(a) for a in analyses))
                self.tree.addTopLevelItem(head)
                for analysis, names in analyses.items():
                    title = analysis_title(analysis)
                    node = QTreeWidgetItem([title, "",
                                            f"{len(names)} descriptors"])
                    node.setData(0, _ROLE, ("analysis", analysis))
                    node.setData(0, _RAW_ID,
                                 analysis if title != analysis else "")
                    node.setToolTip(0, analysis)
                    head.addChild(node)
                    for name in names:
                        holders = [r for r in results
                                   if r.outputs.get(analysis) is not None
                                   and name in r.outputs[analysis].tables]
                        container = holders[0].outputs[analysis].tables[name]
                        badge = kind_badge(container)
                        label = display_name(name)
                        leaf = QTreeWidgetItem([
                            label, {"distribution": "distr"}.get(badge,
                                                                 badge),
                            f"{len(holders)} of {len(results)}"])
                        leaf.setData(0, _ROLE, ("item", analysis, name))
                        leaf.setData(0, _RAW_ID, name if label != name
                                     else "")
                        leaf.setToolTip(0, f"{label}\n{name} ({badge}); in "
                                        f"{len(holders)} of {len(results)} "
                                        "ticked models")
                        leaf.setTextAlignment(2, Qt.AlignRight
                                              | Qt.AlignVCenter)
                        node.addChild(leaf)
                        if ("item", analysis, name) == current:
                            to_select = leaf
                    node.setExpanded(True)
                head.setExpanded(True)
        self.tree.blockSignals(False)
        if to_select is not None:
            self.tree.setCurrentItem(to_select)
        self._apply_filter(self.filter_edit.text())

    def _walk(self):
        stack = [self.tree.topLevelItem(i)
                 for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            yield item
            stack[0:0] = [item.child(k) for k in range(item.childCount())]

    def tree_roles(self) -> list[tuple]:
        """The role of every tree row, in tree order: ('group', title),
        ('analysis', name), ('item', analysis, descriptor), ('site',
        'stairs') or ('site', 'hist', kind)."""
        return [tuple(item.data(0, _ROLE)) for item in self._walk()
                if item.data(0, _ROLE)]

    def current_role(self) -> tuple:
        item = self.tree.currentItem()
        role = item.data(0, _ROLE) if item is not None else None
        return tuple(role) if role else ()

    def select_descriptor(self, analysis: str, name: str | None = None
                          ) -> bool:
        """Put the tree on a descriptor (or an analysis), and so the Figure
        page on it. False when the tree lacks it."""
        wanted = ("item", analysis, name) if name is not None \
            else ("analysis", analysis)
        for item in self._walk():
            role = item.data(0, _ROLE)
            if role and tuple(role) == wanted:
                self.tree.setCurrentItem(item)
                self._on_tree_current(item, True)
                return True
        return False

    def select_site_figure(self, which: str, kind: str = "CN") -> bool:
        """Put the tree on 'stairs' or the crystal-vs-glass 'hist' of one
        value kind."""
        wanted = ("site", "stairs") if which == "stairs" \
            else ("site", "hist", kind)
        for item in self._walk():
            role = item.data(0, _ROLE)
            if role and tuple(role) == wanted:
                self.tree.setCurrentItem(item)
                self._on_tree_current(item, True)
                return True
        return False

    def _on_tree_current(self, item, clicked: bool = False) -> None:
        role = tuple(item.data(0, _ROLE) or ()) if item is not None else ()
        if not role:
            return
        if role[0] in ("item", "site"):
            self._stale.discard("figure")
            self._refresh_figure()
            if clicked or self.page_shown() in ("overview", "table"):
                self.show_page("figure")
        self._invalidate("table")

    def _apply_filter(self, text: str) -> None:
        needle = str(text).strip().lower()
        for i in range(self.tree.topLevelItemCount()):
            self._filter_item(self.tree.topLevelItem(i), needle, False)
        self._invalidate("table")

    def _filter_item(self, item, needle: str, inherited: bool) -> bool:
        role = tuple(item.data(0, _ROLE) or ())
        raw = ""
        if role and role[0] == "item":
            raw = str(role[2])
        elif role and role[0] == "analysis":
            raw = str(role[1])
        own = (not needle or needle in item.text(0).lower()
               or needle in raw.lower())
        cascade = inherited or bool(needle) and bool(raw) \
            and needle in raw.lower()
        shown = False
        for k in range(item.childCount()):
            if self._filter_item(item.child(k), needle, cascade):
                shown = True
        if item.childCount() == 0:
            shown = own or inherited
        elif not shown:
            shown = own and not needle
        item.setHidden(not shown)
        return shown

    def visible_descriptors(self) -> list[str]:
        """The engine ids of the descriptor leaves the filter leaves
        shown."""
        out = []
        for item in self._walk():
            role = tuple(item.data(0, _ROLE) or ())
            if not role or role[0] != "item" or item.isHidden():
                continue
            parent, shown = item.parent(), True
            while parent is not None:
                if parent.isHidden():
                    shown = False
                    break
                parent = parent.parent()
            if shown:
                out.append(role[2])
        return out

    # -- the overview page ---------------------------------------------------------------------
    def _refresh_overview(self) -> None:
        ticked = self.ticked_entries()
        labels = [label for label, _e in ticked]
        results = [(label, entry.result) for label, entry in ticked]
        if len(results) < 2:
            self.overview_table.fill([], [], [])
            self.overview_table.hide()
            self.overview_empty.show()
            self.overview_header.setText(
                f"{len(results)} model ticked." if results
                else "No model ticked.")
            return
        rows = overview_rows(results)
        self.overview_table.fill(rows, labels, self.ticked_styles())
        self.overview_table.show()
        self.overview_empty.hide()
        lines = [f"{marker_glyph(style)} {label}: {run_header(result)}"
                 for (label, result), style in zip(results,
                                                   self.ticked_styles())]
        lines += parameter_differences(results)
        lines.append(f"Differences are the model named minus {labels[0]}; "
                     "'—' where a run lacks the row.")
        self.overview_header.setText("\n".join(lines))

    def overview_sentences(self) -> list[str]:
        """Every text the Overview page shows, for a reader checking the
        wording."""
        return [self.overview_header.text()] + list(
            self.overview_table.sentences)

    # -- the figure page -------------------------------------------------------------------------
    def _refresh_figure(self) -> None:
        role = self.current_role()
        if not role or role[0] not in ("item", "site"):
            self.difference_check.setEnabled(False)
            self.difference_note.setText("")
            self.view.show_item(
                "", "Choose a descriptor in the tree, or a site-to-site "
                    "figure; with two or more models ticked the models are "
                    "overlaid.", None, lambda _pf: [], [], [])
            self.save_figure_button.setEnabled(False)
            return
        if role[0] == "site":
            self._show_site_figure(role)
        else:
            self._show_descriptor(role[1], role[2])
        self.save_figure_button.setEnabled(
            self.view.figure is not None and self.page_shown() == "figure")

    def _basket_header(self) -> list[str]:
        return [
            "FACET site-to-site comparison",
            f"crystal rows at the project's v_bond = "
            f"{float(self.project.v_bond):g} v.u.",
            "model rows: mean and standard deviation (ddof = 0) over the "
            "element's atoms on the frame named, at that model's v_bond",
        ]

    def _show_site_figure(self, role: tuple) -> None:
        self.difference_check.setEnabled(False)
        self.difference_note.setText("")
        rows = self.basket_rows()
        header = self._basket_header()
        if role[1] == "stairs":
            figure = self.staircase_figure()
            title = "CN against v_bond, every basket row"
            subtitle = (f"{len(rows)} basket rows · a crystal row is its "
                        "site's staircase; a model row the mean CN of the "
                        "element's atoms on the frame named")
            empty = "The staircases of the basket rows are overlaid here."
        else:
            which = role[2]
            figure = self.crystal_vs_glass_figure(which)
            name = next(n for k, n, _u in VALUE_KINDS if k == which)
            title = f"{name} of the model against the crystal site"
            subtitle = ("the model's per-atom values binned, the crystal "
                        "site's value as a labelled line; both elements "
                        "have to match")
            empty = ("Add a crystal site and a model element of the same "
                     "element: the model's per-atom values are binned here "
                     "with the site's value as a line.")
        self.view.plot.set_empty_text(empty)
        self.view.show_item(title, subtitle, figure, lambda _pf: rows,
                            header, [], per_frame_available=False,
                            footnote="FACET site-to-site comparison")
        self.view.plot.setVisible(True)

    def _containers(self, analysis: str, name: str
                    ) -> list[tuple[str, object, SeriesStyle]]:
        out = []
        for (label, entry), style in zip(self.ticked_entries(),
                                         self.ticked_styles()):
            output = entry.result.outputs.get(analysis)
            if output is None:
                continue
            container = output.tables.get(name)
            if container is not None:
                out.append((label, container, style))
        return out

    def _show_descriptor(self, analysis: str, name: str) -> None:
        found = self._containers(analysis, name)
        labels = [label for label, _c, _s in found]
        items = [c for _l, c, _s in found]
        styles = [s for _l, _c, s in found]
        stats = all(isinstance(c, (Distribution, Histogram, Series, Scalar))
                    for c in items)
        self.difference_check.setEnabled(len(items) >= 2 and stats)
        want_difference = (self.difference_check.isChecked()
                           and self.difference_check.isEnabled())
        title = f"{display_name(name)} — {len(items)} model" \
                f"{'s' if len(items) != 1 else ''}"
        note = ""
        figure = None
        if want_difference:
            try:
                figure = difference_figure(
                    [(label, c) for label, c, _s in found],
                    title=display_name(name), styles=styles)
            except ValueError as error:
                note = str(error)
                figure = None
        if figure is None:
            if len(items) == 1 and stats:
                from .md_plot import figure_for

                try:
                    figure = figure_for(items[0], title=display_name(name))
                    for s in figure.series:
                        s.style = styles[0]
                except (ValueError, TypeError):
                    figure = None
            elif stats and all(isinstance(c, Scalar) for c in items) \
                    and len({c.unit for c in items}) == 1:
                x = np.arange(len(items), dtype=np.float64)
                s = StatSeries(display_name(name), x,
                               np.array([c.mean for c in items]),
                               np.array([c.std for c in items]),
                               kind="points")
                s.style = styles[0]
                figure = Figure([s], "model", "category",
                                display_name(name), items[0].unit or "1",
                                title=display_name(name),
                                categories=tuple(labels),
                                annotations=("One point per model: the mean "
                                             "over its frames, the whisker "
                                             "its standard deviation.",))
            elif stats:
                try:
                    figure = figure_for_family(items,
                                               title=display_name(name),
                                               labels=labels)
                except (ValueError, TypeError):
                    figure = None
                if figure is not None and len(figure.series) == len(styles):
                    for s, style in zip(figure.series, styles):
                        s.style = style
        if want_difference and not note and figure is not None:
            note = (f"Difference: each series is the model named minus "
                    f"{labels[0]}.")
        self.difference_note.setText(note)
        header = comparison_header(self.ticked_results(), [
            f"analysis: {analysis}", f"descriptor: {name}"])
        notes: list[str] = []
        for label, c, _s in found:
            notes += [f"{label}: {n}" for n in getattr(c, "notes", ())]
        kinds = {kind_badge(c) for c in items}
        subtitle = (f"{analysis_title(analysis)} · "
                    + ", ".join(sorted(kinds))
                    + (" · one series per model" if len(items) > 1
                       else ""))
        if found:
            subtitle += " · " + ", ".join(
                f"{marker_glyph(s)} {label}" for label, _c, s in found)

        def rows_for(per_frame: bool) -> list[dict]:
            rows: list[dict] = []
            for label, c, _s in found:
                try:
                    part = c.as_rows(per_frame=per_frame)
                except TypeError:
                    part = c.as_rows()
                for r in part:
                    rows.append({"model": label, **r})
            return rows

        self.view.plot.set_empty_text("No figure for this descriptor; the "
                                      "rows are under it.")
        self.view.show_item(title, subtitle, figure, rows_for, header,
                            notes, per_frame_available=stats,
                            footnote=" · ".join(
                                f"{label}: {run_header(entry.result)}"
                                for label, entry in self.ticked_entries()))

    # -- the site-to-site figures (data) -------------------------------------------------------
    def _basket_style(self, ref: _Ref, index: int) -> SeriesStyle:
        if ref.kind == "model" and ref.entry in self._candidates:
            return self.style_of(ref.entry)
        return style_for(len(self._candidates) + index)

    def staircase_figure(self) -> Figure | None:
        """CN against v_bond for every basket row: a crystal row its
        site's staircase, a model row the mean CN of the element's atoms on
        the frame named; the current v_bond and the models' v_bond marked.
        None while the basket draws nothing."""
        series: list[StatSeries] = []
        vlines: list[tuple[float, str, str]] = []
        model_v: dict[float, None] = {}
        for index, ref in enumerate(self._refs):
            label = f"{_entry_title(ref.entry)} · "
            if ref.kind == "crystal":
                result = self._site_result(ref)
                if result is None:
                    continue
                grid, cn = coordination.staircase(result,
                                                  n=STAIRCASE_POINTS)
                s = StatSeries(label + result.label, grid,
                               np.asarray(cn, dtype=np.float64), None,
                               kind="line", markers=False, band=False)
            else:
                found = self._model_frame(ref)
                if found is None:
                    continue
                table, v_bond, _k = found
                low, high = threshold_range(table)
                if not (high > low):
                    continue
                grid = np.logspace(math.log10(low), math.log10(high),
                                   STAIRCASE_POINTS)
                stairs = model_staircase(table, grid)
                if str(ref.key) not in stairs:
                    continue
                s = StatSeries(label + f"{ref.key} (mean)", grid,
                               stairs[str(ref.key)], None, kind="line",
                               markers=False, band=False)
                model_v.setdefault(round(v_bond, 9))
            s.style = self._basket_style(ref, index)
            series.append(s)
        if not series:
            return None
        current = float(self.project.v_bond)
        vlines.append((current, f"v_bond {current:.4g}", "current"))
        for v in model_v:
            if not math.isclose(v, current, rel_tol=1e-9):
                vlines.append((v, f"model v_bond {v:.4g}", "reference"))
        return Figure(
            series, "v_bond", "v.u.", "CN", "1",
            title="CN against v_bond, every basket row",
            log_x=True, vlines=tuple(vlines),
            annotations=("A crystal row is its site's staircase; a model "
                         "row is the mean CN of the element's atoms on the "
                         "frame named in the basket.",))

    def crystal_vs_glass_figure(self, which: str = "CN") -> Figure | None:
        """The model elements' per-atom values ('CN', 'BVS' or 'phi')
        binned, for the elements a basket crystal site shares, with each
        such site's value as a labelled vertical line. None when no pair
        of rows shares an element."""
        name, unit = next((n, u) for key, n, u in VALUE_KINDS
                          if key == which)
        series: list[StatSeries] = []
        vlines: list[tuple[float, str, str]] = []
        shown_elements: set[str] = set()
        crystal_rows = []
        for ref in self._refs:
            if ref.kind != "crystal":
                continue
            result = self._site_result(ref)
            if result is not None:
                crystal_rows.append((ref, result))
        crystal_elements = {r.element for _ref, r in crystal_rows}
        v = float(self.project.v_bond)
        lines: dict[str, list[tuple[str, float]]] = {}
        for _ref, result in crystal_rows:
            row = crystal_values(result, v)
            value = {"CN": float(row["cn"]), "BVS": row["bvs_vu"],
                     "phi": row["phi"]}[which]
            if math.isfinite(value):
                lines.setdefault(result.element, []).append(
                    (result.label, value))
        for index, ref in enumerate(self._refs):
            if ref.kind != "model" or ref.key not in crystal_elements:
                continue
            found = self._model_frame(ref)
            if found is None:
                continue
            table, v_bond, k = found
            values = per_atom_values(table, ref.key, v_bond, which)
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            width = _BIN[which]
            marked = [value for _label, value
                      in lines.get(str(ref.key), [])]
            low = math.floor(min([float(finite.min())] + marked)
                             / width) * width
            high = math.ceil(max([float(finite.max())] + marked)
                             / width) * width
            if which == "CN":
                edges = np.arange(low - 0.5, high + 1.5, 1.0)
            else:
                n_bins = max(int(round((high - low) / width)), 1)
                edges = low + width * np.arange(n_bins + 1)
            hist, _ = np.histogram(finite, bins=edges)
            s = StatSeries(
                f"{_entry_title(ref.entry)} · {ref.key}, frame {k}",
                0.5 * (edges[:-1] + edges[1:]),
                hist / finite.size, None, kind="step", edges=edges,
                band=False, markers=False)
            s.style = self._basket_style(ref, index)
            series.append(s)
            shown_elements.add(str(ref.key))
        for element, marks in lines.items():
            if element not in shown_elements:
                continue
            for label, value in marks:
                vlines.append((value, f"{label} {value:.3g}", "reference"))
        if not series:
            return None
        bin_note = ("Bins of 1 in CN." if which == "CN"
                    else f"Bins of {_BIN[which]:g} {unit}.")
        return Figure(
            series, name, unit, "fraction of the element's atoms per bin",
            "1", title=f"{name} of the model against the crystal site",
            vlines=tuple(vlines),
            annotations=(bin_note + " The vertical lines are the crystal "
                         "sites' values at the project's v_bond "
                         f"({v:g} v.u.); the bars are the model's atoms at "
                         "the model's v_bond.",))

    # -- the table page ----------------------------------------------------------------------------
    def table_scope_role(self) -> tuple:
        """What the Table page is narrowed to: the tree's current group,
        analysis or descriptor, or ('all',)."""
        role = self.current_role()
        if role and role[0] in ("group", "analysis", "item") \
                and role != ("group", "Site to site"):
            return role
        return ("all",)

    def _refresh_table(self) -> None:
        ticked = self.ticked_results()
        labels = [label for label, _r in ticked]
        scope = self.table_scope_role()
        rows = comparison_rows(ticked, scope=scope,
                               filter_text=self.filter_edit.text())
        header = comparison_header(ticked)
        self.table.set_rows(rows, header)
        if scope[0] == "group":
            where = f"the question '{scope[1]}'"
        elif scope[0] == "analysis":
            where = f"the analysis '{analysis_title(scope[1])}'"
        elif scope[0] == "item":
            where = f"the descriptor '{display_name(scope[2])}'"
        else:
            where = "every descriptor"
        needle = self.filter_edit.text().strip()
        text = (f"{len(rows)} rows: {where}"
                + (f", filtered by '{needle}'" if needle else "")
                + f"; {len(labels)} model{'s' if len(labels) != 1 else ''}"
                + (" (" + ", ".join(labels) + ")" if labels else "")
                + ". Cells are mean ± standard deviation over the frames "
                "of each run; the difference column is the model named "
                f"minus {labels[0]}." if labels else
                "Tick models with results to list their descriptors here.")
        self.table_scope.setText(text)

    def table_rows(self) -> list[dict]:
        return self.table.rows()

    # -- the 3D page ----------------------------------------------------------------------------------
    def painter_tier(self) -> bool:
        from ..gl import caps as caps_mod

        caps = getattr(self.view3d, "caps", None)
        return caps is None or caps.tier is caps_mod.Tier.BASIC \
            or getattr(self.view3d, "_fallback", None) is not None

    def _crystal_scene(self, entry):
        """The window's drawing of a basket crystal, when it is the one the
        window has drawn alone (no rebuild here: a crystal scene costs
        seconds)."""
        window = self.window
        scene = getattr(window, "scene", None)
        if scene is None or window is None:
            return None
        current = getattr(window, "project", None)
        if current is None or current.current is not entry:
            return None
        if getattr(current, "overlay", False) or len(
                scene.structure_of) and len(set(scene.structure_of)) > 1:
            return None
        return scene

    def scene_blocks(self) -> list[tuple[str, Scene | None, str]]:
        """(label, scene or None, note) per block of the 3D row: each
        ticked model's frame on show, then each basket crystal the window
        has drawn."""
        blocks = []
        for label, entry in self.ticked_entries():
            fv = getattr(entry, "frame_view", None)
            scene = getattr(fv, "scene", None)
            if scene is None:
                blocks.append((label, None, f"{label}: no frame drawn yet"))
                continue
            note = (f"{label}: frame {fv.k}, {scene.n_cell_atoms} atoms"
                    + (" (atoms only)" if fv.table is None else
                       f", {scene.n_bonds} contacts drawn"))
            blocks.append((label, scene, note))
        seen = set()
        for ref in self._refs:
            if ref.kind != "crystal" or id(ref.entry) in seen:
                continue
            seen.add(id(ref.entry))
            title = _entry_title(ref.entry)
            scene = self._crystal_scene(ref.entry)
            if scene is None:
                blocks.append((title, None,
                               f"{title}: compared as numbers (select it in "
                               "the Structures dock to draw it here)"))
            else:
                blocks.append((title, scene,
                               f"{title}: the window's drawing, "
                               f"{scene.n_atoms} atoms"))
        return blocks

    def _refresh_3d(self) -> None:
        try:
            from .md_workspace import ATOMS_ONLY_ABOVE
        except ImportError:                 # pragma: no cover - packaging
            ATOMS_ONLY_ABOVE = 5000
        clock = time.perf_counter()
        blocks = self.scene_blocks()
        scenes = [scene for _l, scene, _n in blocks]
        merged, _offsets = side_by_side_scene(scenes, self.gap_box.value())
        notes = [note for _l, _s, note in blocks]
        tier_note = ""
        if merged.n_atoms and self.painter_tier() \
                and merged.n_atoms > ATOMS_ONLY_ABOVE and merged.n_bonds:
            merged = atoms_only(merged)
            tier_note = (f" The QPainter renderer (no OpenGL) draws the "
                         f"{merged.n_atoms} merged atoms without bonds "
                         f"(above {ATOMS_ONLY_ABOVE}).")
        if self._element and merged.n_atoms:
            merged = emphasise_element(merged, self._element, self.theme)
        drawn = sum(1 for s in scenes if s is not None and s.n_atoms)
        if merged.n_atoms:
            self.view3d.set_scene(merged, reframe=True)
        else:
            self.view3d.set_scene(None, reframe=False)
        self._last_3d_s = time.perf_counter() - clock
        head = (f"{drawn} block{'s' if drawn != 1 else ''} side by side, "
                f"gap {self.gap_box.value():g} Å, merged in "
                f"{1000 * self._last_3d_s:.0f} ms"
                + (f"; {self._element} emphasised" if self._element else "")
                + ". " if drawn else "Nothing to draw: ")
        self.scene_note.setText(head + "; ".join(notes) + tier_note
                                if notes else
                                "Tick models with a frame drawn, or add a "
                                "crystal the window has drawn to the "
                                "basket.")

    def _reframe(self) -> None:
        if self.view3d.scene is not None:
            self.view3d.set_scene(self.view3d.scene, reframe=True)

    def _on_element_chosen(self, symbol: str) -> None:
        self._element = str(symbol or "")
        self._invalidate("3d")

    # -- the Elements page ------------------------------------------------------------------------------
    def _element_rows(self, entry) -> tuple[dict, float] | None:
        controller = getattr(entry, "controller", None)
        strip = getattr(controller, "strip", None)
        if strip is not None and getattr(strip, "results", None) is not None:
            rows = {str(r["element"]): (int(r["atoms"]),
                                        float(r.get("mean CN", _NAN)))
                    for r in strip.element_rows()}
            return rows, float(strip.v_bond)
        found = self._model_frame(entry)
        if found is None:
            return None
        table, v_bond, _k = found
        results = bulk.at_threshold(table, v_bond)
        elements = np.asarray(table.elements)
        rows = {}
        for element in dict.fromkeys(str(e) for e in elements):
            sel = elements == element
            cn = np.asarray(results.cn, dtype=np.float64)[sel]
            rows[element] = (int(sel.sum()),
                             float(cn.mean()) if cn.size else _NAN)
        return rows, v_bond

    def _refresh_elements(self) -> None:
        models = []
        for (label, entry), style in zip(self.ticked_entries(),
                                         self.ticked_styles()):
            found = self._element_rows(entry)
            if found is None:
                continue
            rows, v_bond = found
            models.append({"label": label, "style": style,
                           "v_bond": v_bond, "rows": rows})
        self.elements.set_models(models)

    # -- everything ----------------------------------------------------------------------------------------
    def _refresh_all(self) -> None:
        self._fill_basket(self.basket_rows())
        self._fill_tree()
        self._refresh_elements()
        self._invalidate("overview", "figure", "table", "3d")

    # -- saving ---------------------------------------------------------------------------------------------
    def _dialog_parent(self):
        from PySide6.QtWidgets import QWidget as _QWidget

        return self.window if isinstance(self.window, _QWidget) else None

    def save_figure(self) -> None:
        if self.page_shown() != "figure" or self.view.figure is None:
            self.statusMessage.emit(
                "No figure to save on this page: the Figure page draws the "
                "descriptor chosen in the tree.")
            return
        self.view.plot.choose_save()

    def rows_shown(self) -> tuple[list[dict], list[str], str]:
        """(rows, header lines, file stem) of the page shown: the Figure
        page's rows, the Table page's rows or the Overview's rows."""
        page = self.page_shown()
        ticked = self.ticked_results()
        if page == "figure":
            return self.view.rows(), self.view.header, "comparison_rows"
        if page == "table":
            return self.table.rows(), comparison_header(ticked), \
                "comparison_table"
        if page == "overview":
            labels = [label for label, _r in ticked]
            return _flat_overview(overview_rows(ticked), labels), \
                comparison_header(ticked), "comparison_overview"
        return [], [], "comparison"

    def save_rows(self) -> None:
        rows, header, stem = self.rows_shown()
        if not rows:
            self.statusMessage.emit("No rows to save on this page.")
            return
        path, _f = QFileDialog.getSaveFileName(
            self._dialog_parent(), "Save the rows shown", f"{stem}.csv",
            "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            for line in header:
                handle.write(f"# {line}\n")
            writer = csv.DictWriter(handle, fieldnames=columns, restval="")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_cell(row.get(k)) for k in columns})
        self.statusMessage.emit(f"Wrote {Path(path).name}")

    def export_comparison(self, path) -> Path:
        """Write the comparison of the ticked models (the overview and the
        table, with the provenance header) as CSV or XLSX by suffix."""
        ticked = self.ticked_results()
        if len(ticked) < 2:
            raise ValueError("Tick two or more models with results to export "
                             "a comparison.")
        return write_comparison(path, ticked,
                                extra_header=self._basket_lines())

    def _basket_lines(self) -> list[str]:
        lines = []
        for row in self.basket_rows():
            if "missing" in row:
                continue
            where = (f"frame {row['frame']}" if row["kind"] == "model"
                     else f"project v_bond {row['v_bond_vu']:g} v.u.")
            lines.append(f"basket: {row['structure']} · {row['site']} · "
                         f"CN {_pm(float(row['cn']), float(row.get('cn_std', _NAN)))} "
                         f"({where})")
        return lines

    def choose_export(self) -> None:
        if len(self.ticked_results()) < 2:
            self.statusMessage.emit("Tick two or more models with results to "
                                    "export a comparison.")
            return
        path, chosen = QFileDialog.getSaveFileName(
            self._dialog_parent(), "Export the comparison",
            "comparison.xlsx",
            "Excel workbook (*.xlsx);;CSV (*.csv)")
        if not path:
            return
        if Path(path).suffix.lower() not in (".xlsx", ".csv"):
            path += ".csv" if "CSV" in str(chosen) else ".xlsx"
        try:
            written = self.export_comparison(path)
        except (OSError, ValueError) as error:
            self.statusMessage.emit(f"The comparison was not written: "
                                    f"{error}")
        else:
            self.statusMessage.emit(f"Wrote {Path(written).name}")

    # -- theme ----------------------------------------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        self.theme = theme
        self.view.apply_theme(theme)
        self.view3d.set_theme(theme)
        self._invalidate("3d")

    def visible_texts(self) -> list[str]:
        """Every text the four widgets show, for a reader checking the
        wording."""
        texts = list(self.overview_sentences())
        texts += [self.warning.text(), self.table_scope.text(),
                  self.scene_note.text(), self.difference_note.text(),
                  self.view.title.text(), self.view.subtitle.text(),
                  self.elements.note.text()]
        for widget in (self.column, self.elements, self.toolbar,
                       self.canvas):
            for child in widget.findChildren(QWidget):
                texts.append(child.toolTip())
                text = getattr(child, "text", None)
                if callable(text):
                    value = text()
                    if isinstance(value, str):
                        texts.append(value)
        for item in self._walk():
            texts += [item.text(c) for c in range(3)] + [item.toolTip(0)]
        return [t for t in texts if t]
