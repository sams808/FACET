"""The bond threshold of one MD frame, moved without a new search.

The bulk engine searches a frame once (``bulk.analyse_frame``) and keeps,
per atom, every counter-ion contact above the tabulation threshold v_list,
sorted by valence, with running sums (``bulk.ValenceTable``). Any bond
threshold v_bond is then a lookup: ``bulk.at_threshold(table, v_bond)``
gives every atom's CN, bond-valence sum and phi at that threshold, exactly
as a new search at that threshold would (bulk.py, "WHAT IS COMPUTED"), in
1-4 ms for 3 000-12 000 atoms instead of a search of 0.1-0.6 s (measured on
the 2 880-atom and 11 520-atom models of the Model-workspace scouting,
2026-10-07). :class:`ThresholdPanel` puts that lookup under a slider:

* the slider runs on a logarithmic scale from v_list to the largest valence
  of the frame, the axis on which a plateau width is a distance gap
  (cutoff_explorer.py);
* each move re-reads the table (``bulk.at_threshold`` only, no search) and
  redraws the CN of each element, and the distributions of its
  bond-valence sum and of phi;
* the staircase, the mean CN of each element against v_bond, is drawn once
  per frame from the sorted listed valences: at any v it is the number of
  listed contacts with v_i > v (the strict rule of the CN) over the atoms of
  the element; a click on it sets the threshold there, as the crystal
  window's cutoff explorer does for one site.

One frame is shown, never an average: its numbers are those of that frame
at that threshold. The frame-averaged results of an analysis run stay those
of the threshold the run used; the panel names both thresholds whenever
they differ. The distance-cut descriptors do not depend on v_bond.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import bulk, bv
from .md_plot import Figure, StatPlot, StatSeries, style_for
from .md_views import start_worker

__all__ = ["ThresholdPanel", "staircase", "threshold_range", "SLIDER_STEPS",
           "STAIRCASE_POINTS", "BVS_BIN_VU", "PHI_BIN", "BACKGROUND_ABOVE"]

# positions of the slider between v_list and the largest valence (log scale)
SLIDER_STEPS = 1000
# points of the staircase drawing (each point exact; the steps between two
# points are drawn as a slope, the readout and the bars are exact)
STAIRCASE_POINTS = 600
# bin widths of the two distributions: presentation choices, stated under
# the plots. 0.02 in phi is glass.py's default bin (md_analysis.GlassOptions).
BVS_BIN_VU = 0.05
PHI_BIN = 0.02
# set_frame searches on a worker thread above this many atoms unless told
# otherwise: about 0.2 s of search (bulk.py's table, 0.5-0.6 s at 10 000
# atoms), the point where a frozen window becomes noticeable
BACKGROUND_ABOVE = 4000


def threshold_range(table: bulk.ValenceTable) -> tuple[float, float]:
    """(lowest, highest) threshold the slider offers: v_list (or the
    smallest listed valence when v_list is 0) and the largest listed
    valence. Equal when the frame lists no contact."""
    valid = np.arange(table.v_vu.shape[1])[None, :] < table.n_listed[:, None]
    values = table.v_vu[valid]
    if values.size == 0:
        low = max(float(table.v_list_vu), 1e-6)
        return low, low
    high = float(values.max())
    low = float(table.v_list_vu) if table.v_list_vu > 0 else \
        float(values[values > 0].min()) if (values > 0).any() else 1e-6
    return min(low, high), high


def staircase(table: bulk.ValenceTable, v_grid) -> dict[str, np.ndarray]:
    """Mean CN of each element at each threshold of ``v_grid``: listed
    contacts with v > threshold (the CN's strict rule) over the element's
    atoms. Equal, point for point, to the mean of
    ``bulk.at_threshold(table, v).cn`` over those atoms."""
    grid = np.asarray(v_grid, dtype=np.float64)
    valid = np.arange(table.v_vu.shape[1])[None, :] < table.n_listed[:, None]
    out = {}
    for element in sorted({str(e) for e in table.elements}):
        rows = table.elements == element
        n_atoms = int(rows.sum())
        values = np.sort(table.v_vu[rows][valid[rows]])
        above = values.size - np.searchsorted(values, grid, side="right")
        out[element] = above / n_atoms
    return out


class _FrameWorker(QObject):
    """bulk.analyse_frame on a worker thread; results go to the panel's
    slots through queued connections."""

    finished = Signal(object)
    failed = Signal(object)

    def __init__(self, generation: int, frame, ox_atom, params,
                 v_list_vu: float):
        super().__init__()
        self._job = (generation, frame, ox_atom, params, v_list_vu)

    @Slot()
    def run(self) -> None:
        job, self._job = self._job, None
        generation, frame, ox_atom, params, v_list = job
        try:
            table, _ = bulk.analyse_frame(frame, ox_atom, params,
                                          v_list_vu=v_list)
        except ValueError as error:
            self.failed.emit((generation, str(error)))
        except BaseException as error:  # noqa: BLE001 - the thread must end
            self.failed.emit((generation,
                              f"{type(error).__name__}: {error}"))
        else:
            self.finished.emit((generation, table))
        finally:
            # whatever was raised, the thread ends (an exception other than
            # ValueError used to leave it running and the panel 'searching')
            job = frame = table = None    # noqa: F841
            QThread.currentThread().quit()


class ThresholdPanel(QWidget):
    """v_bond on one frame: a log slider, the CN staircase and the CN,
    bond-valence-sum and phi distributions of each element at the threshold.

    ``set_frame(frame, ox_atom, params)`` searches the frame once (on a
    worker thread with ``background=True``); ``set_table`` takes a valence
    table already built. Signals: ``thresholdChanged(v_bond)`` on every
    move, ``frameReady()`` once a table is shown, ``failed(text)`` when the
    engine refuses the frame.
    """

    thresholdChanged = Signal(float)
    frameReady = Signal()
    failed = Signal(str)
    # what a figure saved from a plot's context menu wrote, or why not
    statusMessage = Signal(str)

    def __init__(self, parent=None, *, theme=None):
        super().__init__(parent)
        self.table: bulk.ValenceTable | None = None
        self.results: bulk.AtomResults | None = None
        self.v_bond: float = bv.V_BOND_DEFAULT
        self.analysis_v_bond: float | None = None
        self.label = ""
        self._range = (bv.V_LIST_DEFAULT, 1.0)
        self._elements: list[str] = []
        self._grid = np.array([])
        self._stairs: dict[str, np.ndarray] = {}
        self._bvs_edges = np.arange(0.0, 1.0 + BVS_BIN_VU, BVS_BIN_VU)
        self._phi_edges = np.arange(0.0, 1.0 + PHI_BIN / 2, PHI_BIN)
        self._generation = 0
        self._thread: QThread | None = None
        self._worker: _FrameWorker | None = None

        self.heading = QLabel("No frame.")
        self.heading.setTextFormat(Qt.PlainText)
        self.heading.setWordWrap(True)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)
        self.slider.setToolTip(
            "v_bond on a logarithmic scale from v_list to the largest "
            "valence of the frame; each position re-reads the contacts of "
            "the one search, with no new search.")
        self.spin = QDoubleSpinBox()
        self.spin.setDecimals(4)
        self.spin.setSuffix(" v.u.")
        self.spin.setKeyboardTracking(False)
        self.spin.setToolTip("v_bond typed in, in valence units.")
        self.reset_button = QPushButton("Analysis threshold")
        self.reset_button.setToolTip("Put v_bond back where the analysis "
                                     "run had it.")
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QLabel("v_bond"))
        row.addWidget(self.slider, 1)
        row.addWidget(self.spin)
        row.addWidget(self.reset_button)
        self.banner = QLabel()
        self.banner.setTextFormat(Qt.PlainText)
        self.banner.setWordWrap(True)

        self.stairs_plot = StatPlot()
        self.stairs_plot.pick_mode = True
        self.stairs_plot.setToolTip("Click or drag to set v_bond.")
        self.cn_plot = StatPlot()
        self.bvs_plot = StatPlot()
        self.phi_plot = StatPlot()
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.addWidget(self.stairs_plot, 0, 0)
        grid.addWidget(self.cn_plot, 0, 1)
        grid.addWidget(self.bvs_plot, 1, 0)
        grid.addWidget(self.phi_plot, 1, 1)

        self.element_table = QTableWidget()
        self.element_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.element_table.verticalHeader().setVisible(False)
        self.element_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self.element_table.setWordWrap(False)
        row_height = int(self.element_table.fontMetrics().height() * 1.35) + 2
        self.element_table.verticalHeader().setDefaultSectionSize(row_height)
        self.element_table.setMaximumHeight(row_height * 7)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.heading)
        layout.addLayout(row)
        layout.addWidget(self.banner)
        layout.addLayout(grid, 1)
        layout.addWidget(self.element_table)

        self.slider.valueChanged.connect(self._on_slider)
        self.spin.valueChanged.connect(self._on_spin)
        self.reset_button.clicked.connect(self.reset_threshold)
        self.stairs_plot.picked.connect(self._on_pick)
        for plot, stem in zip(self.plots(), ("mean_CN_against_v_bond",
                                             "CN_at_v_bond", "BVS_at_v_bond",
                                             "phi_at_v_bond")):
            plot.set_empty_text("No frame is loaded.")
            # each plot saves itself (SVG, PDF, 600 dpi PNG) from its
            # context menu; what it wrote reaches the window's status bar
            plot.file_stem = stem
            plot.statusMessage.connect(self.statusMessage)
        if theme is not None:
            self.apply_theme(theme)
        self._set_enabled(False)

    # -- loading -----------------------------------------------------------
    def plots(self) -> tuple[StatPlot, ...]:
        return (self.stairs_plot, self.cn_plot, self.bvs_plot, self.phi_plot)

    def set_frame(self, frame, ox_atom, params: bv.ParameterSet | None = None,
                  *, v_list_vu: float = bv.V_LIST_DEFAULT,
                  v_bond_vu: float | None = None,
                  analysis_v_bond_vu: float | None = None,
                  label: str = "", table: bulk.ValenceTable | None = None,
                  background: bool | None = None) -> None:
        """Search ``frame`` once and show it. ``ox_atom``: one oxidation
        state per atom (``ModelOxidation.per_atom``). ``v_bond_vu``: where
        the slider starts (the analysis threshold, or bv.V_BOND_DEFAULT).
        ``table``: the frame's valence table when the caller already built
        it (no search here). ``background``: search on a worker thread,
        then ``frameReady`` or ``failed``; None does so above
        :data:`BACKGROUND_ABOVE` atoms. A search on the calling thread
        raises ValueError when the engine refuses the frame."""
        self._generation += 1
        self._pending = (v_bond_vu, analysis_v_bond_vu, label)
        if table is not None:
            if table.n_atoms != frame.n_atoms:
                raise ValueError(f"the table holds {table.n_atoms} atoms and "
                                 f"the frame {frame.n_atoms}")
            self.set_table(table, v_bond_vu=v_bond_vu,
                           analysis_v_bond_vu=analysis_v_bond_vu, label=label)
            return
        if background is None:
            background = frame.n_atoms > BACKGROUND_ABOVE
        if not background:
            table, _ = bulk.analyse_frame(frame, ox_atom, params,
                                          v_list_vu=v_list_vu)
            self.set_table(table, v_bond_vu=v_bond_vu,
                           analysis_v_bond_vu=analysis_v_bond_vu,
                           label=label)
            return
        self.heading.setText(f"{label or 'Frame'}: searching for contacts"
                             "…")
        self._set_enabled(False)
        worker = _FrameWorker(self._generation, frame, ox_atom, params,
                              v_list_vu)
        self._worker = worker
        self._thread = start_worker(worker, self._on_worker_finished,
                                    self._on_worker_failed,
                                    self._on_thread_done)

    def is_busy(self) -> bool:
        return self._thread is not None

    def shutdown(self, timeout_ms: int = 30000) -> bool:
        """Wait for a search in progress; True when none is left running."""
        thread = self._thread
        if thread is None:
            return True
        done = thread.wait(timeout_ms)
        if done:
            self._on_thread_done()
        return done

    @Slot(object)
    def _on_worker_finished(self, payload) -> None:
        generation, table = payload
        if generation != self._generation:
            return
        v_bond, analysis_v_bond, label = self._pending
        self.set_table(table, v_bond_vu=v_bond,
                       analysis_v_bond_vu=analysis_v_bond, label=label)

    @Slot(object)
    def _on_worker_failed(self, payload) -> None:
        generation, text = payload
        if generation != self._generation:
            return
        self.heading.setText(f"The frame was not analysed: {text}")
        self.failed.emit(text)

    @Slot()
    def _on_thread_done(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return              # an earlier search ended; a newer one runs
        self._thread = None
        self._worker = None

    def set_table(self, table: bulk.ValenceTable, *,
                  v_bond_vu: float | None = None,
                  analysis_v_bond_vu: float | None = None,
                  label: str = "") -> None:
        """Show a valence table already built (no search here)."""
        if not isinstance(table, bulk.ValenceTable):
            raise ValueError("a bulk.ValenceTable is needed")
        self.table = table
        self.label = label
        self.analysis_v_bond = analysis_v_bond_vu
        self._elements = sorted({str(e) for e in table.elements})
        self._range = threshold_range(table)
        low, high = self._range
        self._grid = np.geomspace(low, high, STAIRCASE_POINTS) \
            if high > low else np.array([low])
        self._stairs = staircase(table, self._grid)
        valid = np.arange(table.v_vu.shape[1])[None, :] < \
            table.n_listed[:, None]
        listed_sum = np.where(valid, table.v_vu, 0.0).sum(axis=1)
        top = float(np.max(listed_sum)) if listed_sum.size else 1.0
        n_bins = max(1, int(math.ceil(top / BVS_BIN_VU + 1e-9)))
        self._bvs_edges = BVS_BIN_VU * np.arange(n_bins + 1, dtype=np.float64)
        n_listed = int(table.n_listed.sum())
        heading = [label or "Frame",
                   f"{table.n_atoms} atoms",
                   f"one bond-valence search to {table.r_search_ang:g} Å",
                   f"{n_listed} listed contact ends (v > "
                   f"{table.v_list_vu:g} v.u.)"]
        self.heading.setText(" · ".join(heading))
        self.heading.setToolTip("\n".join(str(n) for n in table.notes))
        self.spin.blockSignals(True)
        self.spin.setRange(low, high)
        self.spin.setSingleStep(max((high - low) / 200.0, 1e-4))
        self.spin.blockSignals(False)
        self._set_enabled(high > low)
        start = v_bond_vu if v_bond_vu is not None else (
            analysis_v_bond_vu if analysis_v_bond_vu is not None
            else bv.V_BOND_DEFAULT)
        self.reset_button.setEnabled(analysis_v_bond_vu is not None
                                     and high > low)
        self._draw_staircase()
        self.set_threshold(start, emit=False)
        self.frameReady.emit()

    def clear(self) -> None:
        self.table = None
        self.results = None
        self.heading.setText("No frame.")
        self.banner.clear()
        self.element_table.clear()
        self.element_table.setRowCount(0)
        for plot in self.plots():
            plot.set_figure(None)
        self._set_enabled(False)

    def _set_enabled(self, on: bool) -> None:
        for widget in (self.slider, self.spin, self.reset_button):
            widget.setEnabled(on)

    # -- the threshold -------------------------------------------------------
    def value_at(self, position: int) -> float:
        """The threshold at a slider position (log scale)."""
        low, high = self._range
        if high <= low:
            return low
        fraction = min(max(int(position), 0), SLIDER_STEPS) / SLIDER_STEPS
        return float(low * (high / low) ** fraction)

    def position_for(self, v: float) -> int:
        low, high = self._range
        if high <= low or v <= low:
            return 0
        if v >= high:
            return SLIDER_STEPS
        return int(round(SLIDER_STEPS * math.log(v / low)
                         / math.log(high / low)))

    def set_threshold(self, v_bond_vu: float, emit: bool = True) -> None:
        """Re-read the table at ``v_bond_vu`` (clamped to the slider's
        range) and redraw; no search."""
        if self.table is None:
            return
        low, high = self._range
        value = float(v_bond_vu)
        if not math.isfinite(value):
            return
        value = min(max(value, low), high)
        self.v_bond = value
        self.results = bulk.at_threshold(self.table, value)
        self.slider.blockSignals(True)
        self.slider.setValue(self.position_for(value))
        self.slider.blockSignals(False)
        self.spin.blockSignals(True)
        self.spin.setValue(value)
        self.spin.blockSignals(False)
        self._redraw()
        if emit:
            self.thresholdChanged.emit(value)

    @Slot()
    def reset_threshold(self) -> None:
        if self.analysis_v_bond is not None:
            self.set_threshold(self.analysis_v_bond)

    @Slot(int)
    def _on_slider(self, position: int) -> None:
        self.set_threshold(self.value_at(position))

    @Slot(float)
    def _on_spin(self, value: float) -> None:
        self.set_threshold(value)

    @Slot(float, float)
    def _on_pick(self, x: float, _y: float) -> None:
        if self.table is not None and x > 0:
            self.set_threshold(x)

    # -- what the panel shows ------------------------------------------------
    def cn_counts(self) -> dict[str, dict[int, int]]:
        """CN value -> atoms with it, per element, at the current v_bond."""
        if self.results is None:
            return {}
        out = {}
        for element in self._elements:
            rows = self.table.elements == element
            values, counts = np.unique(self.results.cn[rows],
                                       return_counts=True)
            out[element] = {int(v): int(c) for v, c in zip(values, counts)}
        return out

    def element_rows(self) -> list[dict]:
        """Per element at the current v_bond: atoms, mean CN, CN range,
        atoms with no bond, mean bond-valence sum, mean phi (NaN-free
        means over the atoms with a bond)."""
        if self.results is None:
            return []
        rows = []
        for element in self._elements:
            sel = self.table.elements == element
            cn = self.results.cn[sel]
            bvs = self.results.bvs_vu[sel]
            phi = self.results.phi[sel]
            bonded = np.isfinite(bvs)
            rows.append({
                "element": element,
                "atoms": int(sel.sum()),
                "mean CN": float(cn.mean()) if cn.size else float("nan"),
                "CN min": int(cn.min()) if cn.size else 0,
                "CN max": int(cn.max()) if cn.size else 0,
                "atoms with CN 0": int((cn == 0).sum()),
                "mean BVS (v.u.)": float(bvs[bonded].mean())
                if bonded.any() else float("nan"),
                "mean phi (1)": float(np.nanmean(phi))
                if np.isfinite(phi).any() else float("nan"),
            })
        return rows

    def current_bonds(self) -> bulk.Bonds | None:
        """The bonds the CN counts at the current v_bond (bulk.bonds_at)."""
        if self.table is None:
            return None
        return bulk.bonds_at(self.table, self.v_bond)

    def _style(self, element: str):
        return style_for(self._elements.index(element))

    def _draw_staircase(self) -> None:
        series = []
        for element in self._elements:
            s = StatSeries(element, self._grid, self._stairs[element],
                           None, kind="line", markers=True, band=False)
            s.style = self._style(element)
            series.append(s)
        low, high = self._range
        figure = Figure(series, "v_bond", "v.u.", "mean CN", "1",
                        title="Mean CN of each element against v_bond",
                        log_x=high > low * 1.0001,
                        annotations=("Mean over the atoms of each element of "
                                     "the contacts with v > v_bond (one "
                                     "frame). Click to set v_bond.",))
        self.stairs_plot.set_figure(figure)

    def _vlines(self):
        lines = [(self.v_bond, f"v_bond {self.v_bond:.4g}", "current")]
        if self.analysis_v_bond is not None and \
                not math.isclose(self.analysis_v_bond, self.v_bond,
                                 rel_tol=1e-12):
            lines.append((self.analysis_v_bond,
                          f"analysis {self.analysis_v_bond:.4g}",
                          "reference"))
        return tuple(lines)

    def _redraw(self) -> None:
        self.stairs_plot.set_vlines(self._vlines())
        results, table = self.results, self.table
        at = f"at v_bond = {self.v_bond:.4g} v.u."
        # CN of each element
        counts = self.cn_counts()
        cn_max = max((max(c) for c in counts.values() if c), default=0)
        keys = list(range(0, cn_max + 1))
        x = np.arange(len(keys), dtype=np.float64)
        bars = []
        for element in self._elements:
            n = sum(counts[element].values())
            frac = np.array([counts[element].get(k, 0) / n for k in keys])
            s = StatSeries(element, x, frac, None, kind="bars",
                           whiskers=False)
            s.style = self._style(element)
            bars.append(s)
        self.cn_plot.set_figure(Figure(
            bars, "CN", "category", "fraction of the element's atoms", "1",
            title=f"CN of each element {at}",
            categories=tuple(str(k) for k in keys)))
        # BVS and phi distributions
        unbonded = []
        bvs_series, phi_series = [], []
        for element in self._elements:
            sel = table.elements == element
            n = int(sel.sum())
            bvs = results.bvs_vu[sel]
            phi = results.phi[sel]
            missing = int((~np.isfinite(bvs)).sum())
            if missing:
                unbonded.append(f"{element} {missing}")
            for values, edges, out in ((bvs, self._bvs_edges, bvs_series),
                                       (phi, self._phi_edges, phi_series)):
                finite = values[np.isfinite(values)]
                hist, _ = np.histogram(finite, bins=edges)
                above = int((finite > edges[-1]).sum())
                if above:
                    hist[-1] += above
                s = StatSeries(element, 0.5 * (edges[:-1] + edges[1:]),
                               hist / n, None, kind="step", edges=edges,
                               band=False, markers=True)
                s.style = self._style(element)
                out.append(s)
        left_out = []
        if unbonded:
            left_out.append("Atoms with no contact above v_bond have no BVS "
                            "and no phi and are not binned: "
                            + ", ".join(unbonded) + ".")
        self.bvs_plot.set_figure(Figure(
            bvs_series, "bond-valence sum", "v.u.",
            "fraction of the element's atoms per bin", "1",
            title=f"Bond-valence sum {at}",
            annotations=tuple(left_out) + (f"Bins of {BVS_BIN_VU:g} v.u.",)),
            keep_view=False)
        self.phi_plot.set_figure(Figure(
            phi_series, "phi", "1", "fraction of the element's atoms per bin",
            "1", title=f"phi {at}",
            annotations=tuple(left_out) + (f"Bins of {PHI_BIN:g}.",)))
        self._fill_element_table()
        self._update_banner()

    def _fill_element_table(self) -> None:
        rows = self.element_rows()
        columns = list(rows[0]) if rows else []
        table = self.element_table
        table.setUpdatesEnabled(False)
        try:
            table.clear()
            table.setColumnCount(len(columns))
            table.setRowCount(len(rows))
            table.setHorizontalHeaderLabels(columns)
            for r, row in enumerate(rows):
                for c, column in enumerate(columns):
                    value = row[column]
                    if isinstance(value, float):
                        text = "NaN" if math.isnan(value) else f"{value:.4f}"
                    else:
                        text = str(value)
                    item = QTableWidgetItem(text)
                    if not isinstance(value, str):
                        item.setTextAlignment(Qt.AlignRight
                                              | Qt.AlignVCenter)
                    table.setItem(r, c, item)
        finally:
            table.setUpdatesEnabled(True)

    def _update_banner(self) -> None:
        if self.table is None:
            self.banner.clear()
            return
        text = [f"One frame at v_bond = {self.v_bond:.4f} v.u."]
        if self.analysis_v_bond is not None:
            if math.isclose(self.analysis_v_bond, self.v_bond,
                            rel_tol=1e-12):
                text.append("This is the threshold of the analysis run.")
            else:
                text.append(f"The analysis run used {self.analysis_v_bond:.4f}"
                            " v.u.; its frame-averaged results are at that "
                            "threshold.")
        text.append("The distance-cut results do not depend on v_bond.")
        self.banner.setText(" ".join(text))

    # -- theme -------------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        for plot in self.plots():
            plot.apply_theme(theme)

    def visible_texts(self) -> list[str]:
        """Every text the panel shows or can show as a tooltip."""
        texts = [self.heading.text(), self.heading.toolTip(),
                 self.banner.text(), self.spin.suffix()]
        for widget in self.findChildren(QWidget):
            texts.append(widget.toolTip())
            method = getattr(widget, "text", None)
            if callable(method):
                try:
                    value = method()
                except TypeError:
                    continue
                if isinstance(value, str):
                    texts.append(value)
        for plot in self.plots():
            figure = plot.figure
            texts.append(plot.empty_text)
            if figure is not None:
                texts += [figure.title, figure.x_label, figure.y_label,
                          *figure.annotations,
                          *(s.label for s in figure.series),
                          *(label for _, label, _ in figure.vlines)]
        for c in range(self.element_table.columnCount()):
            header = self.element_table.horizontalHeaderItem(c)
            if header is not None:
                texts.append(header.text())
        return [t for t in texts if t]
