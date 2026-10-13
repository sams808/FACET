"""The Compare workspace: structures side by side, site to site.

Two kinds of comparison live here, both built from numbers the rest of the
application already measures:

* **Site to site** -- a comparison basket of references: a crystal site
  (one :class:`~facet.core.coordination.SiteResult`) or an element of an MD
  model (the per-atom numbers of the frame shown, from its
  ``bulk.ValenceTable``). The basket is listed as a table, overlaid as one
  staircase figure (CN against v_bond on the log axis every staircase in
  FACET uses), and, when a crystal site and a model element share an
  element, drawn as the model's per-atom distribution with the crystal
  site's value as a labelled vertical line -- the reference crystal against
  the glass.
* **Same calculations across models** -- the descriptors present in two or
  more ticked model results, shown side by side: curves overlaid, bars
  grouped, scalars tabulated. A header states, in plain words, where the
  runs' parameters differ (v_bond, formers, frames); it states the
  difference and nothing more.

The basket lives on the main window, not on an entry, and is **not saved
in sessions**: a session restores files and settings, and the basket is a
working selection over them.

Crystal values are computed from the site's contact list at the project's
current v_bond; model values are those of the frame named in the row, at
that model's own v_bond. Every number here can be recomputed from the
SiteResult or the ValenceTable it came from.
"""
from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import bulk, bv, coordination
from ..core.md_stats import Scalar
from . import chrome
from .md_names import analysis_title, display_name, group_of, kind_badge
from .md_plot import (
    Figure,
    StatPlot,
    StatSeries,
    figure_for_family,
    style_for,
)
from .md_threshold import BVS_BIN_VU, PHI_BIN, staircase as model_staircase, \
    threshold_range
from .md_views import DescriptorView

__all__ = ["ComparePage", "crystal_values", "model_values",
           "per_atom_values", "common_descriptors", "parameter_differences",
           "VALUE_KINDS"]

# what the crystal-vs-glass selector offers: (key, axis name, axis unit)
VALUE_KINDS = (("CN", "CN", "1"),
               ("BVS", "bond-valence sum", "v.u."),
               ("phi", "phi", "1"))
_BIN = {"CN": 1.0, "BVS": BVS_BIN_VU, "phi": PHI_BIN}

STAIRCASE_POINTS = 240


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

def common_descriptors(results: Sequence) -> list[tuple[str, str]]:
    """The ``(analysis, descriptor)`` pairs present in at least two of the
    given ModelResults, in question-group order then as the runs list
    them."""
    counts: dict[tuple[str, str], int] = {}
    order: list[tuple[str, str]] = []
    for result in results:
        for analysis, name, _container in result.descriptors():
            key = (str(analysis), str(name))
            if key not in counts:
                counts[key] = 0
                order.append(key)
            counts[key] += 1
    shared = [key for key in order if counts[key] >= 2]
    shared.sort(key=lambda key: (group_of(key[0]),
                                 order.index(key)))
    return shared


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
        return "–"
    if not math.isfinite(std):
        return f"{mean:.{digits}g}"
    return f"{mean:.{digits}g} ± {std:.2g}"


# ---------------------------------------------------------------------------
# the page
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


class ComparePage(QWidget):
    """The Compare workspace, shown in the window's right column.

    Holds the comparison basket (``add_crystal_site``,
    ``add_model_element``, ``remove_ref``, ``clear``) and the
    across-models view (``set_model_candidates``). The basket lives on the
    window for the life of the window and is not saved in sessions.

    Signals: ``closeRequested`` (the Back button), ``overlayToggled(bool)``
    (the side-by-side check, wired by the window to the Structures dock's
    existing overlay), ``statusMessage(str)``.
    """

    closeRequested = Signal()
    overlayToggled = Signal(bool)
    statusMessage = Signal(str)

    BASKET_COLUMNS = ("structure", "site", "CN (1)", "BVS (v.u.)",
                      "phi (1)", "plateau (decades)", "mean d (Å)",
                      "frame", "")

    def __init__(self, project, parent=None, *, theme=None):
        super().__init__(parent)
        self.project = project
        self.theme = theme
        self._refs: list[_Ref] = []
        self._candidates: list = []        # ModelEntry with a result
        self._build()
        if theme is not None:
            self.apply_theme(theme)
        self._refresh_all()

    # -- layout -------------------------------------------------------------
    def _build(self) -> None:
        column = QVBoxLayout(self)
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
            "Leave the comparison and show the selected structure's own "
            "column again. The basket is kept.")
        self.back_button.clicked.connect(self.closeRequested)
        header.addWidget(self.back_button)
        column.addLayout(header)

        self.overlay_check = QCheckBox("Show all side by side (3D view)")
        self.overlay_check.setToolTip(
            "The Structures dock's \"Show all at once\": every ticked "
            "crystal drawn together, superimposed or spaced in a row.")
        self.overlay_check.toggled.connect(self._on_overlay_check)
        column.addWidget(self.overlay_check)
        self.overlay_note = QLabel(
            "Model frames are not drawn in the overlay; a model takes part "
            "in the comparison as numbers.")
        self.overlay_note.setWordWrap(True)
        chrome.mark_hint(self.overlay_note)
        column.addWidget(self.overlay_note)

        self.tabs = QTabWidget()
        self.tabs.addTab(chrome.in_scroll_area(self._build_sites_tab()),
                         "Site to site")
        self.tabs.addTab(self._build_models_tab(), "Across models")
        column.addWidget(self.tabs, 1)

    def _build_sites_tab(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        hint = QLabel(
            "Add references from a right-click: a row of the Sites list or "
            "an atom in the 3D view (a crystal site), or a row of a "
            "model's Elements dock (that element). Crystal rows are read "
            "at the project's v_bond; model rows are the per-atom numbers "
            "of the frame named, at that model's v_bond.")
        hint.setWordWrap(True)
        chrome.mark_hint(hint)
        col.addWidget(hint)

        self.basket_table = QTableWidget(0, len(self.BASKET_COLUMNS))
        self.basket_table.setHorizontalHeaderLabels(
            list(self.BASKET_COLUMNS))
        self.basket_table.verticalHeader().setVisible(False)
        self.basket_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.basket_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.basket_table.setMinimumHeight(110)
        self.basket_table.setToolTip(
            "The comparison basket. A model cell is mean ± standard "
            "deviation over that element's atoms; × removes the row.")
        self.basket_table.itemClicked.connect(self._on_basket_click)
        col.addWidget(self.basket_table)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Empty the basket.")
        self.clear_button.clicked.connect(self.clear)
        buttons.addWidget(self.clear_button)
        col.addLayout(buttons)

        self.stairs_plot = StatPlot()
        self.stairs_plot.setMinimumHeight(190)
        self.stairs_plot.file_stem = "compare_staircases"
        self.stairs_plot.set_empty_text(
            "The staircases of the basket rows are overlaid here.")
        col.addWidget(self.stairs_plot, 1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Crystal against glass:"))
        self.hist_kind = QComboBox()
        for key, name, unit in VALUE_KINDS:
            self.hist_kind.addItem(f"{name} ({unit})", key)
        self.hist_kind.setToolTip(
            "Which per-atom value of the model is binned; the matching "
            "crystal site's value is the vertical line.")
        self.hist_kind.activated.connect(lambda _=0: self._refresh_hist())
        row.addWidget(self.hist_kind)
        row.addStretch(1)
        self.save_sites_figure = QPushButton("Save figure…")
        self.save_sites_figure.setToolTip(
            "Write the staircase overlay as SVG, PDF or a 600 dpi PNG.")
        self.save_sites_figure.clicked.connect(self.stairs_plot.choose_save)
        row.addWidget(self.save_sites_figure)
        self.save_basket_button = QPushButton("Save rows…")
        self.save_basket_button.setToolTip(
            "Write the basket as CSV, the provenance as '#' lines.")
        self.save_basket_button.clicked.connect(self._save_basket)
        row.addWidget(self.save_basket_button)
        col.addLayout(row)

        self.hist_plot = StatPlot()
        self.hist_plot.setMinimumHeight(190)
        self.hist_plot.file_stem = "crystal_vs_glass"
        self.hist_plot.set_empty_text(
            "Add a crystal site and a model element of the same element: "
            "the model's per-atom values are binned here with the site's "
            "value as a line.")
        col.addWidget(self.hist_plot, 1)
        return page

    def _build_models_tab(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        hint = QLabel("Tick two or more models that have results; the "
                      "descriptors present in at least two of them are "
                      "listed below.")
        hint.setWordWrap(True)
        chrome.mark_hint(hint)
        col.addWidget(hint)

        self.models_list = QListWidget()
        self.models_list.setMaximumHeight(96)
        self.models_list.setToolTip(
            "Every MD model of the project with a finished run.")
        self.models_list.itemChanged.connect(
            lambda _item: self._refresh_descriptors())
        col.addWidget(self.models_list)

        self.warning = QLabel("")
        self.warning.setWordWrap(True)
        chrome.mark_hint(self.warning)
        self.warning.hide()
        col.addWidget(self.warning)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter the descriptors…")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        col.addWidget(self.filter_edit)

        split = QSplitter(Qt.Vertical)
        self.descriptor_list = QListWidget()
        self.descriptor_list.setToolTip(
            "The descriptors two or more ticked results hold, grouped by "
            "question. Choose one to see the models side by side.")
        self.descriptor_list.currentItemChanged.connect(
            lambda _c, _p=None: self._show_descriptor())
        split.addWidget(self.descriptor_list)
        self.view = DescriptorView()
        self.view.figure_button.clicked.connect(self.view.plot.choose_save)
        self.view.rows_button.clicked.connect(self._save_view_rows)
        split.addWidget(self.view)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 3)
        col.addWidget(split, 1)
        return page

    # -- the basket ----------------------------------------------------------
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
        self._fill_models_list()
        self._refresh_all()

    def set_v_bond(self, _v: float | None = None) -> None:
        """The project's bond threshold moved: re-read the crystal rows and
        the marks (the project itself already holds the value)."""
        self._refresh_all()

    def set_overlay_state(self, on: bool) -> None:
        """Mirror the Structures dock's overlay without re-emitting."""
        self.overlay_check.blockSignals(True)
        self.overlay_check.setChecked(bool(on))
        self.overlay_check.blockSignals(False)

    def _on_overlay_check(self, on: bool) -> None:
        self.overlayToggled.emit(bool(on))

    def _site_result(self, ref: _Ref):
        for result in self.project.results_for(ref.entry):
            if result.site_index == ref.key:
                return result
        return None

    @staticmethod
    def _model_frame(ref: _Ref):
        """(ValenceTable, v_bond, frame index) of a model ref's frame shown,
        or None before a frame is drawn with its table."""
        view = getattr(ref.entry, "frame_view", None)
        table = getattr(view, "table", None)
        if table is None:
            return None
        v_bond = getattr(view, "v_bond_vu", bv.V_BOND_DEFAULT)
        controller = getattr(ref.entry, "controller", None)
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

    def _refresh_all(self) -> None:
        rows = self.basket_rows()
        self._fill_basket(rows)
        self._refresh_staircases()
        self._refresh_hist()
        self._update_enabled()

    def _fill_basket(self, rows: list[dict]) -> None:
        table = self.basket_table
        table.setRowCount(0)
        for row in rows:
            i = table.rowCount()
            table.insertRow(i)
            if "missing" in row:
                cells = (row["structure"], f"{row['site']} "
                         f"({row['missing']})", "", "", "", "", "", "")
            elif row["kind"] == "crystal":
                cells = (row["structure"], row["site"],
                         str(row["cn"]), f"{row['bvs_vu']:.3f}",
                         f"{row['phi']:.3f}",
                         self._plateau_text(row["plateau_decades"]),
                         f"{row['d_mean_ang']:.3f}", "–")
            else:
                cells = (row["structure"],
                         f"{row['site']} ({row['atoms']} atoms)",
                         _pm(row["cn"], row["cn_std"]),
                         _pm(row["bvs_vu"], row["bvs_std"]),
                         _pm(row["phi"], row["phi_std"]),
                         _pm(row["plateau_decades"], row["plateau_std"]),
                         _pm(row["d_mean_ang"], row["d_std_ang"]),
                         str(row["frame"]))
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

    @staticmethod
    def _plateau_text(value: float) -> str:
        if not math.isfinite(value):
            return "on a step edge"
        return f"{value:.3f}"

    def _on_basket_click(self, item: QTableWidgetItem) -> None:
        if item.column() == len(self.BASKET_COLUMNS) - 1:
            self.remove_ref(item.row())

    # -- the staircase overlay -----------------------------------------------
    def _staircase_series(self) -> tuple[list[StatSeries], list]:
        series: list[StatSeries] = []
        vlines: list[tuple[float, str, str]] = []
        model_v: dict[float, None] = {}
        for ref in self._refs:
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
            s.style = style_for(len(series))
            series.append(s)
        current = float(self.project.v_bond)
        vlines.append((current, f"v_bond {current:.4g}", "current"))
        for v in model_v:
            if not math.isclose(v, current, rel_tol=1e-9):
                vlines.append((v, f"model v_bond {v:.4g}", "reference"))
        return series, vlines

    def _refresh_staircases(self) -> None:
        series, vlines = self._staircase_series()
        if not series:
            self.stairs_plot.set_figure(None)
            return
        figure = Figure(
            series, "v_bond", "v.u.", "CN", "1",
            title="CN against v_bond, every basket row",
            log_x=True, vlines=tuple(vlines),
            annotations=("A crystal row is its site's staircase; a model "
                         "row is the mean CN of the element's atoms on the "
                         "frame named in the basket.",))
        self.stairs_plot.set_figure(figure)

    # -- crystal against glass -------------------------------------------------
    def _refresh_hist(self) -> None:
        which = self.hist_kind.currentData() or "CN"
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
        # the crystal sites' values, by element, so the binned range always
        # holds the vertical lines they become
        lines: dict[str, list[tuple[str, float]]] = {}
        for _ref, result in crystal_rows:
            row = crystal_values(result, v)
            value = {"CN": float(row["cn"]), "BVS": row["bvs_vu"],
                     "phi": row["phi"]}[which]
            if math.isfinite(value):
                lines.setdefault(result.element, []).append(
                    (result.label, value))
        for ref in self._refs:
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
            s.style = style_for(len(series))
            series.append(s)
            shown_elements.add(str(ref.key))
        for element, marks in lines.items():
            if element not in shown_elements:
                continue
            for label, value in marks:
                vlines.append((value, f"{label} {value:.3g}", "reference"))
        if not series:
            self.hist_plot.set_figure(None)
            return
        bin_note = ("Bins of 1 in CN." if which == "CN"
                    else f"Bins of {_BIN[which]:g} {unit}.")
        figure = Figure(
            series, name, unit, "fraction of the element's atoms per bin",
            "1", title=f"{name} of the model against the crystal site",
            vlines=tuple(vlines),
            annotations=(bin_note + " The vertical lines are the crystal "
                         "sites' values at the project's v_bond "
                         f"({v:g} v.u.); the bars are the model's atoms at "
                         "the model's v_bond.",))
        self.hist_plot.set_figure(figure)

    def _update_enabled(self) -> None:
        has_rows = bool(self._refs)
        self.clear_button.setEnabled(has_rows)
        self.save_basket_button.setEnabled(has_rows)
        self.save_sites_figure.setEnabled(
            self.stairs_plot.figure is not None)

    # -- across models ---------------------------------------------------------
    def set_model_candidates(self, entries) -> None:
        """The model entries that can be ticked: those with a result."""
        kept = {id(e) for e in self._candidates}
        self._candidates = [e for e in entries
                            if getattr(e, "result", None) is not None]
        if {id(e) for e in self._candidates} != kept:
            self._fill_models_list()
        self._refresh_descriptors()

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

    def _fill_models_list(self) -> None:
        checked = {self.models_list.item(i).text()
                   for i in range(self.models_list.count())
                   if self.models_list.item(i).checkState() == Qt.Checked}
        self.models_list.blockSignals(True)
        self.models_list.clear()
        for entry, label in zip(self._candidates, self._model_labels()):
            item = QListWidgetItem(label)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(
                Qt.Checked if (label in checked or not checked)
                and len(self._candidates) > 1 else Qt.Unchecked)
            item.setData(Qt.UserRole, entry)
            self.models_list.addItem(item)
        self.models_list.blockSignals(False)

    def ticked_results(self) -> list[tuple[str, object]]:
        """(label, ModelResult) of every ticked model."""
        out = []
        for i in range(self.models_list.count()):
            item = self.models_list.item(i)
            if item.checkState() != Qt.Checked:
                continue
            entry = item.data(Qt.UserRole)
            result = getattr(entry, "result", None)
            if result is not None:
                out.append((item.text(), result))
        return out

    def _refresh_descriptors(self) -> None:
        ticked = self.ticked_results()
        differences = parameter_differences(ticked)
        self.warning.setText(
            "The runs' parameters differ. " + " ".join(differences)
            if differences else "")
        self.warning.setVisible(bool(differences))
        current = None
        if self.descriptor_list.currentItem() is not None:
            current = self.descriptor_list.currentItem().data(Qt.UserRole)
        self.descriptor_list.blockSignals(True)
        self.descriptor_list.clear()
        if len(ticked) >= 2:
            shared = common_descriptors([r for _l, r in ticked])
            for analysis, name in shared:
                text = (f"{group_of(analysis)} · "
                        f"{analysis_title(analysis)} · "
                        f"{display_name(name)}")
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, (analysis, name))
                self.descriptor_list.addItem(item)
                if (analysis, name) == current:
                    self.descriptor_list.setCurrentItem(item)
        self.descriptor_list.blockSignals(False)
        self._apply_filter(self.filter_edit.text())
        self._show_descriptor()

    def _apply_filter(self, text: str) -> None:
        needle = str(text).strip().lower()
        for i in range(self.descriptor_list.count()):
            item = self.descriptor_list.item(i)
            item.setHidden(bool(needle)
                           and needle not in item.text().lower())

    def _containers(self) -> tuple[list[tuple[str, object]], tuple]:
        item = self.descriptor_list.currentItem()
        ticked = self.ticked_results()
        if item is None or len(ticked) < 2:
            return [], ("", "")
        analysis, name = item.data(Qt.UserRole)
        out = []
        for label, result in ticked:
            output = result.outputs.get(analysis)
            if output is None:
                continue
            container = output.tables.get(name)
            if container is not None:
                out.append((label, container))
        return out, (analysis, name)

    def _show_descriptor(self) -> None:
        containers, (analysis, name) = self._containers()
        if len(containers) < 2:
            self.view.show_item(
                "", "Tick two or more models with results and choose a "
                    "descriptor.", None, lambda _pf: [], [], [])
            return
        labels = [label for label, _c in containers]
        items = [c for _l, c in containers]
        title = f"{display_name(name)} — {len(items)} models"
        if all(isinstance(c, Scalar) for c in items) \
                and len({c.unit for c in items}) == 1:
            # one dot per model, its spread as the whisker, beside the table
            x = np.arange(len(items), dtype=np.float64)
            s = StatSeries(display_name(name), x,
                           np.array([c.mean for c in items]),
                           np.array([c.std for c in items]), kind="points")
            figure = Figure([s], "model", "category", display_name(name),
                            items[0].unit or "1", title=display_name(name),
                            categories=tuple(labels))
        else:
            try:
                figure = figure_for_family(items, title=display_name(name),
                                           labels=labels)
            except (ValueError, TypeError):
                figure = None
        header = [f"FACET comparison: {analysis_title(analysis)} / {name}",
                  "models: " + "; ".join(labels)]
        header += parameter_differences(self.ticked_results())
        notes: list[str] = []
        for label, c in containers:
            notes += [f"{label}: {n}" for n in getattr(c, "notes", ())]
        kinds = {kind_badge(c) for c in items}
        subtitle = (f"{analysis_title(analysis)} · "
                    + ", ".join(sorted(kinds))
                    + " · one series per model")

        def rows_for(per_frame: bool) -> list[dict]:
            rows: list[dict] = []
            for label, c in containers:
                try:
                    part = c.as_rows(per_frame=per_frame)
                except TypeError:
                    part = c.as_rows()
                for r in part:
                    rows.append({"model": label, **r})
            return rows

        self.view.show_item(title, subtitle, figure, rows_for, header,
                            notes)

    # -- saving ---------------------------------------------------------------
    def _save_basket(self) -> None:
        rows = self.basket_rows()
        if not rows:
            return
        path, _f = QFileDialog.getSaveFileName(
            self, "Save the basket rows", "comparison_basket.csv",
            "CSV (*.csv)")
        if not path:
            return
        header = [
            "FACET site-to-site comparison",
            f"crystal rows at the project's v_bond = "
            f"{float(self.project.v_bond):g} v.u.",
            "model rows: mean and standard deviation (ddof = 0) over the "
            "element's atoms on the frame named, at that model's v_bond",
        ]
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            for line in header:
                handle.write(f"# {line}\n")
            writer = csv.DictWriter(handle, fieldnames=columns,
                                    restval="")
            writer.writeheader()
            writer.writerows(rows)
        self.statusMessage.emit(f"Wrote {Path(path).name}")

    def _save_view_rows(self) -> None:
        rows = self.view.rows()
        if not rows:
            return
        path, _f = QFileDialog.getSaveFileName(
            self, "Save the rows shown", "comparison_rows.csv",
            "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as handle:
            for line in self.view.header:
                handle.write(f"# {line}\n")
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]),
                                    restval="")
            writer.writeheader()
            writer.writerows(rows)
        self.statusMessage.emit(f"Wrote {Path(path).name}")

    # -- theme -----------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        self.theme = theme
        for plot in (self.stairs_plot, self.hist_plot):
            plot.apply_theme(theme)
        self.view.apply_theme(theme)
