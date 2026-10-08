"""An MD analysis run, browsed: every descriptor with its figure, rows and provenance.

:func:`facet.core.md_analysis.analyse` returns a
:class:`~facet.core.md_analysis.ModelResult`: for each analysis that ran, a
set of descriptors (md_stats containers and tables) with the provenance of
the run. :class:`ResultBrowser` lays it out as a tree, analysis then
descriptor, with the descriptors that differ only by element gathered under
one node ('g *-*' holds every partial g(r)), and shows the item chosen as a
figure beside its rows (:class:`DescriptorView`). The provenance header of
the run (file, frames, type map, oxidation states, bond-valence set and
thresholds, formers, cutoffs, method parameters, FACET version) is shown
above the tree and written into every export:

* the whole result as one XLSX workbook or a folder of CSV files
  (:mod:`facet.core.md_export`, on a worker thread, since a workbook of a
  large run takes tens of seconds to write);
* the rows shown, as CSV with that header as '#' lines and every float in
  full (the shortest text that reads back as the same float);
* the figure shown, as SVG, PDF or a 600 dpi PNG (:mod:`.md_plot`), with a
  footnote naming the file, the frames, v_bond and the FACET version.

An analysis that produced nothing is listed with the reason the engine gave.
Nothing here computes or judges a value: it shows what the run measured.
"""
from __future__ import annotations

import atexit
import html
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QAbstractTableModel,
    QCoreApplication,
    QModelIndex,
    QObject,
    QThread,
    Qt,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTableView,
    QTextBrowser,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import exporters, md_export
from ..core.md_analysis import ANALYSIS_SUMMARIES, ModelResult, Table
from ..core.md_stats import Distribution, Histogram, Scalar, Series
from .md_plot import (
    Figure,
    StatPlot,
    ask_figure_path,
    figure_for,
    figure_for_family,
    figure_for_table,
    file_stem,
    unit_text,
)

__all__ = ["ResultBrowser", "DescriptorView", "RowTable", "family_key",
           "summary_line", "value_text", "start_worker", "KIND_NAMES",
           "com_drift_annotation",
           "MAX_OVERLAY"]

KIND_NAMES = {Distribution: "categories", Histogram: "histogram",
              Series: "curve", Scalar: "value", Table: "table"}
# the most curves or bar series one family figure overlays; the table beside
# it lists every member
MAX_OVERLAY = 12
_ROLE = Qt.UserRole + 1
_RAW = Qt.UserRole + 2


def _kind_name(container) -> str:
    for cls, name in KIND_NAMES.items():
        if isinstance(container, cls):
            return name
    return "rows" if hasattr(container, "as_rows") else "object"


def value_text(container) -> str:
    """A Scalar as 'mean ± std unit'; '' for any other container."""
    if not isinstance(container, Scalar):
        return ""
    unit = "" if container.unit in ("", "1") else f" {container.unit}"
    if math.isfinite(container.std):
        return f"{container.mean:.6g} ± {container.std:.2g}{unit}"
    return f"{container.mean:.6g}{unit}"


def _model_elements(result: ModelResult) -> set[str]:
    prov = result.provenance
    found: set[str] = set()
    if prov.ox is not None:
        found |= set(prov.ox.ox)
    found |= {str(v) for v in prov.type_map.values()}
    return {e for e in found if e and e[0].isupper() and len(e) <= 3}


def family_key(name: str, elements: Iterable[str]) -> str | None:
    """The descriptor name with each element symbol of the model put as '*'
    ('g Na-O' -> 'g *-*'), or None when it names no element. A symbol counts
    when no letter precedes or follows it, and not in S(Q) or F(Q), which
    name functions."""
    symbols = sorted({str(e) for e in elements}, key=len, reverse=True)
    if not symbols:
        return None
    # no letter of either case after it: the B of '(BV)' and the N of 'NBO'
    # are parts of a word, not boron or nitrogen
    pattern = re.compile(r"(?<![A-Za-z])(" + "|".join(map(re.escape,
                                                           symbols))
                         + r")(?![A-Za-z])(?!\(Q\))")
    key, count = pattern.subn("*", name)
    return key if count else None


# md_dynamics.msd's note when the centre-of-mass drift was not removed: the
# centre of mass's own MSD at the last lag, and its fraction of each
# element's MSD there
_COM_NOTE = re.compile(r"at the last lag \((?P<t>[^)]*)\) the centre of "
                       r"mass's own MSD is (?P<v>[^ ]+) Å\^2")


def com_drift_annotation(name: str, notes: Iterable[str]) -> str | None:
    """The line an MSD figure carries under its axes when the analysis notes
    say the centre-of-mass drift was not removed: the centre of mass's own
    MSD at the last lag and, for one element, its fraction of that
    element's MSD there (the engine's numbers, quoted). None for any other
    figure."""
    if not str(name).startswith("MSD") or "centre of mass" in str(name):
        return None
    for note in notes:
        match = _COM_NOTE.search(str(note))
        if match is None:
            continue
        text = (f"Centre-of-mass drift not removed: at the last lag "
                f"({match['t']}) the centre of mass's own MSD is "
                f"{match['v']} Å²")
        element = str(name)[3:].strip()
        tail = str(note).split("fraction of each element's MSD there:")
        fraction = None
        if element and "*" not in element and len(tail) == 2:
            found = re.search(rf"(?<![A-Za-z]){re.escape(element)} "
                              r"([0-9.eE+-]+)", tail[1])
            fraction = found.group(1) if found else None
        if fraction is not None:
            text += f", {fraction} of {name} there"
        elif len(tail) == 2:
            text += f"; as a fraction of each element's MSD:{tail[1]}"
        return text + (". The Dynamics option 'remove the centre-of-mass "
                       "drift' subtracts it.")
    return None


def summary_line(result: ModelResult) -> str:
    """One line naming the run: file, frames, thresholds, formers, version."""
    prov = result.provenance
    frames = prov.frames_used
    frames_text = (f"{len(frames)} frame{'s' if len(frames) != 1 else ''}"
                   + (f" ({_compact(frames)})" if frames else ""))
    parts = [Path(prov.source_path).name or prov.source_path,
             prov.file_format, frames_text]
    if prov.v_bond_vu is not None:
        parts.append(f"v_bond {prov.v_bond_vu:g} v.u.")
    if prov.v_list_vu is not None:
        parts.append(f"v_list {prov.v_list_vu:g} v.u.")
    parts.append("formers " + (", ".join(sorted(prov.formers))
                               if prov.formers else "none named"))
    parts.append(f"FACET {prov.facet_version}")
    return " · ".join(str(p) for p in parts if p)


def _compact(values: Sequence[int]) -> str:
    values = sorted(int(v) for v in values)
    if not values:
        return ""
    runs, start, prev = [], values[0], values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        runs.append((start, prev))
        start = prev = v
    runs.append((start, prev))
    text = ", ".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)
    if len(text) > 40 and len(values) > 2:
        steps = np.diff(values)
        if steps.size and (steps == steps[0]).all():
            return f"{values[0]}-{values[-1]} step {int(steps[0])}"
        return f"{values[0]}-{values[-1]}, {len(values)} chosen"
    return text


def _footnote(result: ModelResult) -> str:
    prov = result.provenance
    bits = [f"file {Path(prov.source_path).name}",
            f"frames {_compact(prov.frames_used)}"
            f" ({len(prov.frames_used)})"]
    if prov.v_bond_vu is not None:
        bits.append(f"v_bond {prov.v_bond_vu:g} v.u.")
    bits.append(f"FACET {prov.facet_version}")
    if result.cancelled:
        bits.append("run cancelled, partial results")
    return "; ".join(bits)


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def _uniform(rows: Iterable[Mapping]) -> list[dict]:
    rows = [dict(r) for r in rows]
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{c: row.get(c) for c in columns} for row in rows]


def _rows_of(container, per_frame: bool = False) -> list[dict]:
    try:
        return md_export.descriptor_rows(container, per_frame=per_frame)
    except ValueError:
        return [{"value": repr(container)}]


def _display(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        return f"{number:.6g}"
    if isinstance(value, (tuple, list)):
        return "; ".join(_display(v) for v in value)
    return str(value)


def _full(value) -> str:
    """Every digit: the shortest text that reads back as the same float."""
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
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return repr(number)
    if isinstance(value, (tuple, list)):
        return "; ".join(_full(v) for v in value)
    return str(value)


_ROLES = frozenset((Qt.DisplayRole, _RAW, Qt.TextAlignmentRole,
                    Qt.ToolTipRole))


class _RowModel(QAbstractTableModel):
    """Rows of dicts for a QTableView, formatted only when a cell is drawn,
    so a 10 000-row descriptor costs nothing until it is scrolled to."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list[dict] = []
        self.columns: list[str] = []

    def set_rows(self, rows: list[dict], columns: list[str]) -> None:
        self.beginResetModel()
        self.rows, self.columns = rows, columns
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def value(self, row: int, column: int):
        return self.rows[row].get(self.columns[column])

    def data(self, index, role=Qt.DisplayRole):
        if role not in _ROLES or not index.isValid():
            return None
        value = self.value(index.row(), index.column())
        if role == Qt.DisplayRole:
            return _display(value)
        if role == _RAW:
            return _full(value)
        if role == Qt.TextAlignmentRole:
            if isinstance(value, (int, float, np.integer, np.floating)) \
                    and not isinstance(value, (bool, np.bool_)):
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return int(Qt.AlignLeft | Qt.AlignVCenter)
        if role == Qt.ToolTipRole:
            text = _display(value)
            return text if len(text) > 32 else None
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal \
                and 0 <= section < len(self.columns):
            return self.columns[section]
        return None


class RowTable(QTableView):
    """Rows of a descriptor, column names with their units.

    Ctrl+C copies the selected cells as tab-separated text with the column
    names; numbers are copied in full, not as displayed. The context menu
    also copies every row with the provenance header. Cells are formatted
    when drawn, so a long descriptor is listed whole.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = _RowModel(self)
        self.setModel(self._model)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.ElideRight)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(
            int(self.fontMetrics().height() * 1.35) + 2)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setResizeContentsPrecision(60)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        self._rows: list[dict] = []
        self._columns: list[str] = []
        self._header: list[str] = []
        self._constant: dict[str, object] = {}

    def set_rows(self, rows: Sequence[Mapping], header: Sequence[str] = ()
                 ) -> None:
        rows = _uniform(rows)
        self._rows = rows
        self._header = list(header)
        self._columns = list(rows[0]) if rows else []
        self._model.set_rows(rows, self._columns)
        # A column that holds one value in every row (the descriptor's name,
        # its kind, the frames used) is hidden on screen and named once
        # above the table (constant_text), so the columns that vary (the
        # key, the mean, the std) fit beside the figure. Copies and saved
        # rows keep every column.
        self._constant = {}
        if len(rows) >= 2:
            for column in self._columns:
                first = _full(rows[0].get(column))
                if all(_full(r.get(column)) == first for r in rows[1:]):
                    self._constant[column] = rows[0].get(column)
        # widths from the column name and the first rows, measured here:
        # resizeColumnsToContents asks the model for every role of every
        # sampled cell, which on a 1 000-row descriptor took 0.2 s
        fm = self.fontMetrics()
        sample = rows[:40]
        for c, column in enumerate(self._columns):
            self.setColumnHidden(c, column in self._constant)
            widest = max([fm.horizontalAdvance(column)]
                         + [fm.horizontalAdvance(_display(r.get(column)))
                            for r in sample])
            self.setColumnWidth(c, min(widest + 16, 260))

    def shown_columns(self) -> list[str]:
        """The columns drawn on screen (those that vary between rows)."""
        return [c for i, c in enumerate(self._columns)
                if not self.isColumnHidden(i)]

    def constant_text(self) -> str:
        """The columns hidden on screen, each with its one value."""
        if not self._constant:
            return ""
        return "Same in every row: " + "; ".join(
            f"{column} = {_display(value) or '(empty)'}"
            for column, value in self._constant.items())

    def rowCount(self) -> int:
        return self._model.rowCount()

    def columnCount(self) -> int:
        return self._model.columnCount()

    def text_at(self, row: int, column: int) -> str:
        """The cell as displayed."""
        return _display(self._model.value(row, column))

    def rows(self) -> list[dict]:
        return [dict(r) for r in self._rows]

    def columns(self) -> list[str]:
        return list(self._columns)

    def selection_text(self) -> str:
        indexes = self.selectionModel().selectedIndexes() \
            if self.selectionModel() is not None else []
        if not indexes:
            return ""
        chosen = {(i.row(), i.column()) for i in indexes}
        rows = sorted({r for r, _ in chosen})
        cols = sorted({c for _, c in chosen})
        lines = ["\t".join(self._columns[c] for c in cols)]
        for r in rows:
            lines.append("\t".join(_full(self._model.value(r, c))
                                   if (r, c) in chosen else ""
                                   for c in cols))
        return "\n".join(lines)

    def all_text(self, with_header: bool = True) -> str:
        lines = [f"# {line}" for line in self._header] if with_header else []
        lines.append("\t".join(self._columns))
        for row in self._rows:
            lines.append("\t".join(_full(row.get(c)) for c in self._columns))
        return "\n".join(lines)

    def copy_selection(self) -> str:
        text = self.selection_text() or self.all_text(with_header=False)
        QGuiApplication.clipboard().setText(text)
        return text

    def copy_all(self) -> str:
        text = self.all_text(with_header=True)
        QGuiApplication.clipboard().setText(text)
        return text

    def keyPressEvent(self, event) -> None:
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            return
        super().keyPressEvent(event)

    def _menu(self, position) -> None:
        menu = QMenu(self)
        menu.addAction("Copy the selected cells", self.copy_selection)
        menu.addAction("Copy every row with the provenance header",
                       self.copy_all)
        menu.exec(self.viewport().mapToGlobal(position))


# ---------------------------------------------------------------------------
# one item: figure | rows, notes under
# ---------------------------------------------------------------------------

class DescriptorView(QWidget):
    """A figure beside its rows (above them in a narrow view), the notes
    under both.

    ``show_item`` takes the figure (None: rows only), a function giving the
    rows with or without one column per frame, the provenance header those
    rows are copied and exported with, and the notes."""

    # The least width at which the rows sit beside the figure; narrower, they
    # sit under it. In a Model window at 1440 px the view is about 550 px
    # wide, where a table beside a 360 px figure showed one or two of its
    # columns and cut the numbers of the next (the final check, 2026-10-07).
    SIDE_BY_SIDE_MIN = 900

    def __init__(self, parent=None):
        super().__init__(parent)
        self.title = QLabel()
        self.title.setTextFormat(Qt.PlainText)
        self.title.setWordWrap(True)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        self.subtitle = QLabel()
        self.subtitle.setWordWrap(True)
        self.subtitle.setTextFormat(Qt.PlainText)

        self.log_x = QCheckBox("Log x")
        self.log_x.setToolTip("Draw the x axis on a logarithmic scale.")
        self.log_y = QCheckBox("Log y")
        self.log_y.setToolTip("Draw the y axis on a logarithmic scale; "
                              "values at or below 0 are left out of the "
                              "drawing and counted under it.")
        self.per_frame = QCheckBox("One column per frame")
        self.per_frame.setToolTip("Add each frame's value to the rows, "
                                  "beside the mean and the spread.")
        self.copy_button = QPushButton("Copy rows")
        self.copy_button.setToolTip("Copy every row, with the provenance "
                                    "header, as tab-separated text.")
        self.figure_button = QPushButton("Save figure…")
        self.figure_button.setToolTip("Write this figure as SVG, PDF or a "
                                      "600 dpi PNG.")
        self.rows_button = QPushButton("Save rows…")
        self.rows_button.setToolTip("Write these rows as CSV, the "
                                    "provenance header as '#' lines.")
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        for w in (self.log_x, self.log_y, self.per_frame):
            controls.addWidget(w)
        controls.addStretch(1)
        for w in (self.copy_button, self.figure_button, self.rows_button):
            controls.addWidget(w)

        self.plot = StatPlot()
        self.plot.setMinimumWidth(360)
        self.table = RowTable()
        self.constants = QLabel()
        self.constants.setWordWrap(True)
        self.constants.setTextFormat(Qt.PlainText)
        self.constants.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rows_side = QWidget()
        rows_column = QVBoxLayout(rows_side)
        rows_column.setContentsMargins(0, 0, 0, 0)
        rows_column.addWidget(self.constants)
        rows_column.addWidget(self.table, 1)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.plot)
        self.splitter.addWidget(rows_side)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setChildrenCollapsible(False)

        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(96)
        self.notes.setPlaceholderText("No notes.")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addLayout(controls)
        layout.addWidget(self.splitter, 1)
        layout.addWidget(self.notes)

        self._rows_for: Callable[[bool], list[dict]] | None = None
        self._header: list[str] = []
        self.log_x.toggled.connect(self._log_changed)
        self.log_y.toggled.connect(self._log_changed)
        self.per_frame.toggled.connect(self._refill_rows)
        self.copy_button.clicked.connect(self.table.copy_all)

    @property
    def figure(self) -> Figure | None:
        return self.plot.figure

    @property
    def header(self) -> list[str]:
        return list(self._header)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_orientation()

    def _fit_orientation(self) -> None:
        """Rows beside the figure in a wide view, under it in a narrow one."""
        wanted = (Qt.Horizontal if self.width() >= self.SIDE_BY_SIDE_MIN
                  else Qt.Vertical)
        if self.splitter.orientation() == wanted:
            return
        self.splitter.setOrientation(wanted)
        total = (self.splitter.width() if wanted == Qt.Horizontal
                 else self.splitter.height())
        total = max(int(total), 500)
        # the figure carries its notes under its axes (an MSD's drift, the
        # points a log axis leaves out): it takes the larger share
        share = int(total * (0.6 if wanted == Qt.Horizontal else 0.66))
        self.splitter.setSizes([share, total - share])

    def show_item(self, title: str, subtitle: str, figure: Figure | None,
                  rows_for: Callable[[bool], list[dict]],
                  header: Sequence[str], notes: Sequence[str], *,
                  per_frame_available: bool = True,
                  footnote: str = "") -> None:
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self._rows_for = rows_for
        self._header = list(header)
        self.plot.footnote = footnote
        self.plot.file_stem = title
        self.plot.set_figure(figure)
        has_figure = figure is not None
        self.plot.setVisible(has_figure)
        self.figure_button.setEnabled(has_figure)
        categorical = has_figure and figure.categories is not None
        for box, value, usable in (
                (self.log_x, has_figure and figure.log_x,
                 has_figure and not categorical),
                (self.log_y, has_figure and figure.log_y, has_figure)):
            box.blockSignals(True)
            box.setChecked(bool(value))
            box.setEnabled(bool(usable))
            box.blockSignals(False)
        self.per_frame.blockSignals(True)
        self.per_frame.setChecked(False)
        self.per_frame.setEnabled(per_frame_available)
        self.per_frame.blockSignals(False)
        self._refill_rows()
        lines = [str(n) for n in notes if str(n).strip()]
        self.notes.setPlainText("\n".join(lines))

    def rows(self) -> list[dict]:
        return self.table.rows()

    @Slot()
    def _refill_rows(self) -> None:
        rows = self._rows_for(self.per_frame.isChecked()) \
            if self._rows_for else []
        self.table.set_rows(rows, self._header)
        text = self.table.constant_text()
        self.constants.setText(text)
        self.constants.setVisible(bool(text))

    @Slot()
    def _log_changed(self) -> None:
        self.plot.set_log(self.log_x.isChecked(), self.log_y.isChecked())

    def apply_theme(self, theme) -> None:
        self.plot.apply_theme(theme)


# ---------------------------------------------------------------------------
# worker threads
# ---------------------------------------------------------------------------

class _Keeper(QObject):
    """Holds every worker thread until it has finished, on the GUI thread.

    A QThread destroyed while it runs ends the process. A thread owned by a
    widget would be destroyed with the widget, so a window closed during a
    search or an export would take the thread with it. The threads started
    here have no parent: this object keeps them, and releases each one in a
    slot that runs on the GUI thread (queued) once the thread has
    finished."""

    # How long the application waits, at quit, for a search or an export
    # still running (neither can be cancelled; an export is left to finish
    # its file).
    QUIT_WAIT_MS = 60_000

    def __init__(self):
        super().__init__()
        self.running: dict = {}
        self._hooked = False
        # a QThread still running when the interpreter ends aborts the
        # process: wait for them at exit too (a script without app.exec())
        atexit.register(_wait_at_exit, self)

    def hook(self) -> None:
        if self._hooked:
            return
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self.wait_all)
            self._hooked = True

    def keep(self, thread: QThread, worker: QObject) -> None:
        self.hook()
        self.running[thread] = worker
        thread.finished.connect(self._release, Qt.QueuedConnection)

    @Slot()
    def wait_all(self) -> None:
        for thread in list(self.running):
            try:
                thread.wait(self.QUIT_WAIT_MS)
            except RuntimeError:        # already deleted
                pass

    @Slot()
    def _release(self) -> None:
        thread = self.sender()
        # The thread first, then its worker: the worker is deleted here, on
        # the GUI thread, once its thread has ended (md_jobs.delete_worker).
        # It used to be deleted on its own thread as that thread ended, and
        # this reference dropped before the wait: the Python wrapper then
        # deleted it here while the worker thread was deleting it too
        # ('Fatal Python error: Aborted' in
        # test_a_finished_worker_frees_what_it_held, under load), and a
        # delete on the worker thread could deadlock with a connect here.
        if isinstance(thread, QThread):
            thread.wait()
        worker = self.running.pop(thread, None)
        from .md_jobs import delete_worker

        delete_worker(worker)
        if isinstance(thread, QThread):
            thread.deleteLater()


def _wait_at_exit(keeper: _Keeper) -> None:
    try:
        keeper.wait_all()
    except RuntimeError:
        pass


_KEEPER: _Keeper | None = None


def start_worker(worker: QObject, finished_slot, failed_slot,
                 done_slot=None) -> QThread:
    """Run ``worker.run`` on a new thread. ``worker`` has the signals
    ``finished`` and ``failed``; they reach ``finished_slot`` and
    ``failed_slot`` (bound @Slot methods of QObjects living on the GUI
    thread) through queued connections, and ``done_slot`` follows the
    thread's end. Call from the GUI thread."""
    global _KEEPER
    if _KEEPER is None:
        _KEEPER = _Keeper()
    thread = QThread()
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    # the worker is deleted on the GUI thread once the thread has ended
    # (_Keeper._release), never on its own thread: see md_jobs.delete_worker
    worker.finished.connect(finished_slot, Qt.QueuedConnection)
    worker.failed.connect(failed_slot, Qt.QueuedConnection)
    # quit() is thread-safe and is called on the worker thread: queued to
    # the GUI thread (the QThread object lives there), it would wait behind
    # a GUI thread blocked in QThread.wait(), and the two would deadlock
    worker.finished.connect(thread.quit, Qt.DirectConnection)
    worker.failed.connect(thread.quit, Qt.DirectConnection)
    if done_slot is not None:
        thread.finished.connect(done_slot, Qt.QueuedConnection)
    _KEEPER.keep(thread, worker)
    thread.start()
    return thread


# ---------------------------------------------------------------------------
# export on a worker thread
# ---------------------------------------------------------------------------

class _ExportWorker(QObject):
    """md_export.export on its own thread. Its signals reach the browser's
    slots, which live on the GUI thread, through queued connections."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, result: ModelResult, path: str):
        super().__init__()
        self._result = result
        self._path = path

    @Slot()
    def run(self) -> None:
        result, self._result = self._result, None
        name = Path(self._path).name
        try:
            paths = md_export.export(result, self._path)
        except (OSError, ValueError, RuntimeError) as error:
            self.failed.emit(f"{name}: {error}")
        except BaseException as error:  # noqa: BLE001 - the thread must end
            self.failed.emit(f"{name}: {type(error).__name__}: {error}")
        else:
            self.finished.emit([str(p) for p in paths])
        finally:
            # whatever was raised, the thread ends: an exception no clause
            # above names used to leave the thread's event loop running, the
            # browser at its busy text, and a closing window waiting
            result = None                 # noqa: F841
            QThread.currentThread().quit()


# ---------------------------------------------------------------------------
# the browser
# ---------------------------------------------------------------------------

class ResultBrowser(QWidget):
    """A ModelResult as a tree beside a view of the item chosen.

    Signals: ``itemShown(analysis, descriptor)`` (descriptor '' for an
    analysis, a family or the run), ``exportFinished(list of paths)``,
    ``exportFailed(text)``, ``statusMessage(text)``. Actions for a window's
    menus: ``export_workbook_action``, ``export_csv_action``,
    ``export_figure_action``, ``export_rows_action``. An export runs on a
    thread no widget owns (:func:`start_worker`), so closing the window
    while a workbook is written lets the file be finished; :meth:`shutdown`
    waits for it.
    """

    itemShown = Signal(str, str)
    exportFinished = Signal(object)
    exportFailed = Signal(str)
    statusMessage = Signal(str)

    # The width of the tree's name column: the longest names two levels
    # down ('connectivity all formers (distance)') in a 9 pt Segoe UI.
    NAME_WIDTH = 240

    def __init__(self, parent=None, *, theme=None):
        super().__init__(parent)
        self.result: ModelResult | None = None
        self.extra_header: list[str] = []
        self._elements: set[str] = set()
        self._current: tuple = ()
        self._export_thread: QThread | None = None
        self._export_worker: _ExportWorker | None = None

        self.summary = QLabel("No analysis result.")
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.provenance_button = QToolButton()
        self.provenance_button.setText("Provenance")
        self.provenance_button.setCheckable(True)
        self.provenance_button.setToolTip(
            "Show the header every export of this run carries: the file, "
            "frames, type map, oxidation states, bond-valence set and "
            "thresholds, formers, cutoffs and method parameters.")
        self.export_all_button = QPushButton("Export all…")
        self.export_all_button.setToolTip(
            "Write every descriptor of the run: one XLSX workbook, or a "
            "folder of CSV files, each with the provenance header.")
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self.summary, 1)
        top.addWidget(self.provenance_button)
        top.addWidget(self.export_all_button)
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.banner.setTextFormat(Qt.PlainText)
        self.banner.setVisible(False)
        self.banner.setFrameShape(QFrame.StyledPanel)
        self.provenance_text = QPlainTextEdit()
        self.provenance_text.setReadOnly(True)
        self.provenance_text.setMaximumHeight(170)
        self.provenance_text.setVisible(False)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["Descriptor", "Kind", "Mean ± std"])
        self.tree.setUniformRowHeights(True)
        self.tree.header().setStretchLastSection(True)
        self.tree.setMinimumWidth(300)
        self.tree.setColumnWidth(0, self.NAME_WIDTH)
        self.tree.setColumnWidth(1, 76)
        # The kind (curve, histogram, table...) is named under the title of
        # the item shown and in each row's tool tip; as a column it pushed
        # 'Mean ± std' out of a Model window's tree, which showed one or two
        # letters of 'Kind' at its edge (the final check, 2026-10-07).
        self.tree.setColumnHidden(1, True)
        self.view = DescriptorView()
        self.page = QTextBrowser()
        self.page.setOpenLinks(False)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.page)
        self.stack.addWidget(self.view)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.tree)
        self.splitter.addWidget(self.stack)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setChildrenCollapsible(False)
        # the tree wide enough for its names and their mean ± std
        self.splitter.setSizes([410, 560])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addLayout(top)
        layout.addWidget(self.banner)
        layout.addWidget(self.provenance_text)
        layout.addWidget(self.splitter, 1)

        self.export_workbook_action = QAction(
            "Export every descriptor as a &workbook…", self)
        self.export_csv_action = QAction(
            "Export every descriptor as &CSV files…", self)
        self.export_figure_action = QAction("Save the &figure shown…",
                                            self)
        self.export_rows_action = QAction("Save the &rows shown…", self)
        self.export_workbook_action.triggered.connect(
            self.choose_export_workbook)
        self.export_csv_action.triggered.connect(self.choose_export_csv)
        self.export_figure_action.triggered.connect(self.choose_export_figure)
        self.export_rows_action.triggered.connect(self.choose_export_rows)
        self.export_all_button.clicked.connect(self._export_menu)
        self.provenance_button.toggled.connect(self.provenance_text.setVisible)
        self.tree.currentItemChanged.connect(self._on_current)
        self.view.figure_button.clicked.connect(self.choose_export_figure)
        self.view.rows_button.clicked.connect(self.choose_export_rows)
        # a figure saved from the plot's own context menu says so here
        self.view.plot.statusMessage.connect(self.statusMessage)
        self.empty_text = ""
        if theme is not None:
            self.apply_theme(theme)
        self._update_actions()

    # -- content -----------------------------------------------------------
    def set_empty_text(self, text: str) -> None:
        """What the browser shows while it holds no run (the window that
        holds it says where a run starts)."""
        self.empty_text = str(text)
        if self.result is None:
            self._show_empty()

    def _show_empty(self) -> None:
        if self.empty_text:
            self.page.setHtml(f"<p>{html.escape(self.empty_text)}</p>")
        else:
            self.page.clear()

    def set_result(self, result: ModelResult | None) -> None:
        """Show a run (None clears the browser)."""
        if result is not None and not isinstance(result, ModelResult):
            raise ValueError("a ModelResult (md_analysis.analyse) is needed")
        self.result = result
        self.tree.clear()
        self._current = ()
        if result is None:
            self.summary.setText("No analysis result.")
            self.banner.setVisible(False)
            self.provenance_text.clear()
            self._show_empty()
            self.stack.setCurrentWidget(self.page)
            self._update_actions()
            return
        self._elements = _model_elements(result)
        self.summary.setText(summary_line(result))
        self.summary.setToolTip(result.provenance.source_path)
        banner = []
        if result.cancelled:
            banner.append("The run was cancelled: each analysis holds the "
                          "frames it had finished, listed in its "
                          "provenance.")
        failed = result.failed
        if failed:
            banner.append(f"{len(failed)} of {len(result.outputs)} analyses "
                          "produced no result; each is listed with the "
                          "reason.")
        self.banner.setText(" ".join(banner))
        self.banner.setVisible(bool(banner))
        self.provenance_text.setPlainText("\n".join(self.run_header()))
        self._fill_tree(result)
        self._update_actions()
        first = self._first_leaf()
        self.tree.setCurrentItem(first if first is not None
                                 else self.tree.topLevelItem(0))

    def clear(self) -> None:
        self.set_result(None)

    def run_header(self) -> list[str]:
        if self.result is None:
            return []
        return md_export.header_lines(self.result) + list(self.extra_header)

    def set_extra_header(self, lines: Sequence[str]) -> None:
        """Lines added to every header this browser writes or copies (the
        window's own record, such as a threshold moved after the run)."""
        self.extra_header = [str(line) for line in lines]
        if self.result is not None:
            self.provenance_text.setPlainText("\n".join(self.run_header()))
            item = self.tree.currentItem()
            if item is not None:
                self._show(item)

    def _fill_tree(self, result: ModelResult) -> None:
        run = QTreeWidgetItem(["Run: provenance and notes", "", ""])
        run.setData(0, _ROLE, ("run",))
        self.tree.addTopLevelItem(run)
        for analysis, output in result.outputs.items():
            if output.error is not None:
                node = QTreeWidgetItem([analysis, "not computed", ""])
                node.setToolTip(0, output.error)
                node.setData(0, _ROLE, ("analysis", analysis))
                reason = QTreeWidgetItem([f"reason: {output.error}", "", ""])
                reason.setToolTip(0, output.error)
                reason.setData(0, _ROLE, ("analysis", analysis))
                node.addChild(reason)
                self.tree.addTopLevelItem(node)
                continue
            node = QTreeWidgetItem([analysis, f"{len(output.tables)} "
                                    "descriptors", ""])
            node.setToolTip(0, ANALYSIS_SUMMARIES.get(analysis, analysis))
            node.setData(0, _ROLE, ("analysis", analysis))
            self.tree.addTopLevelItem(node)
            for entry in self._grouped(output.tables):
                if entry[0] == "family":
                    _, key, names = entry
                    kind = type(output.tables[names[0]])
                    group = QTreeWidgetItem([f"{key}  ({len(names)})",
                                             KIND_NAMES.get(kind, "rows"),
                                             ""])
                    group.setData(0, _ROLE, ("family", analysis, key,
                                             kind.__name__))
                    group.setToolTip(0, ", ".join(names))
                    node.addChild(group)
                    for name in names:
                        group.addChild(self._leaf(analysis, name,
                                                  output.tables[name]))
                else:
                    node.addChild(self._leaf(analysis, entry[1],
                                             output.tables[entry[1]]))
        self.tree.expandToDepth(0)
        self.tree.setColumnWidth(0, self.NAME_WIDTH)
        self.tree.setColumnWidth(1, 76)

    def _leaf(self, analysis: str, name: str, container) -> QTreeWidgetItem:
        kind = _kind_name(container)
        value = value_text(container)
        item = QTreeWidgetItem([name, kind, value])
        item.setData(0, _ROLE, ("item", analysis, name))
        item.setToolTip(0, f"{name} ({kind})")
        if value:
            item.setToolTip(2, value)
        return item

    def _grouped(self, tables: Mapping[str, object]):
        """('family', key, names) for two or more descriptors of one kind
        that differ only by element; ('item', name) for the others; in the
        order of the first member."""
        keys: dict[str, list[str]] = {}
        for name, container in tables.items():
            key = family_key(name, self._elements)
            if key is not None:
                keys.setdefault(f"{type(container).__name__}|{key}",
                                []).append(name)
        emitted: set[str] = set()
        for name, container in tables.items():
            key = family_key(name, self._elements)
            slot = f"{type(container).__name__}|{key}" if key else None
            members = keys.get(slot, []) if slot else []
            if len(members) >= 2:
                if slot not in emitted:
                    emitted.add(slot)
                    yield ("family", key, members)
            else:
                yield ("item", name)

    def _first_leaf(self) -> QTreeWidgetItem | None:
        for i in range(self.tree.topLevelItemCount()):
            node = self.tree.topLevelItem(i)
            stack = [node]
            while stack:
                item = stack.pop(0)
                role = item.data(0, _ROLE)
                if role and role[0] == "item":
                    return item
                stack[0:0] = [item.child(k) for k in range(item.childCount())]
        return None

    # -- what the tree lists ----------------------------------------------------
    def items(self) -> list[tuple[str, str]]:
        """(analysis, descriptor) of every descriptor leaf, in tree order."""
        out = []
        for item in self._walk():
            role = item.data(0, _ROLE)
            if role and role[0] == "item":
                out.append((role[1], role[2]))
        return out

    def _walk(self):
        stack = [self.tree.topLevelItem(i)
                 for i in range(self.tree.topLevelItemCount())]
        while stack:
            item = stack.pop(0)
            yield item
            stack[0:0] = [item.child(k) for k in range(item.childCount())]

    def find_item(self, analysis: str, descriptor: str | None = None
                  ) -> QTreeWidgetItem | None:
        for item in self._walk():
            role = item.data(0, _ROLE)
            if not role:
                continue
            if descriptor is None and role == ("analysis", analysis):
                return item
            if descriptor is not None and role[0] in ("item", "family") \
                    and role[1] == analysis and role[2] == descriptor:
                return item
        return None

    def select(self, analysis: str, descriptor: str | None = None) -> bool:
        """Show an analysis, a descriptor or a family (by its key)."""
        item = self.find_item(analysis, descriptor)
        if item is None:
            return False
        self.tree.setCurrentItem(item)
        return True

    def current(self) -> tuple:
        """('item', analysis, name) / ('family', analysis, key, kind) /
        ('analysis', analysis) / ('run',) or ()."""
        return self._current

    # -- showing an item -----------------------------------------------------------
    @Slot(QTreeWidgetItem, QTreeWidgetItem)
    def _on_current(self, item, _previous=None) -> None:
        if item is not None:
            self._show(item)

    def _show(self, item: QTreeWidgetItem) -> None:
        result = self.result
        role = item.data(0, _ROLE)
        if result is None or not role:
            return
        self._current = tuple(role)
        if role[0] == "run":
            self._show_run()
            self.itemShown.emit("", "")
        elif role[0] == "analysis":
            self._show_analysis(role[1])
            self.itemShown.emit(role[1], "")
        elif role[0] == "family":
            self._show_family(role[1], role[2], role[3])
            self.itemShown.emit(role[1], "")
        else:
            self._show_descriptor(role[1], role[2])
            self.itemShown.emit(role[1], role[2])
        self._update_actions()

    def _descriptor_header(self, analysis: str, name: str | None,
                           container=None) -> list[str]:
        return md_export.header_lines(self.result, analysis, name,
                                      container) + list(self.extra_header)

    def _show_descriptor(self, analysis: str, name: str) -> None:
        output = self.result.outputs[analysis]
        container = output.tables[name]
        figure = None
        if isinstance(container, (Distribution, Histogram, Series, Scalar)):
            figure = figure_for(container, title=name)
        elif isinstance(container, Table) or hasattr(container, "as_rows"):
            summary = output.tables.get(f"{name} (R_chi)")
            base = name[:-len(" (R_chi)")] if name.endswith(" (R_chi)") \
                else None
            if base is not None and base in output.tables:
                figure = figure_for_table(output.tables[base], container)
            else:
                figure = figure_for_table(container, summary)
        stats = isinstance(container, (Distribution, Histogram, Series,
                                       Scalar))
        if figure is not None:
            drift = com_drift_annotation(name, output.notes)
            if drift:
                figure.annotations = (drift,) + tuple(figure.annotations)
        notes = list(getattr(container, "notes", ()) or ())
        if figure is not None:
            notes = [n for n in notes if n not in figure.annotations]
        notes += [f"analysis note: {n}" for n in output.notes]
        subtitle = [analysis, _kind_name(container)]
        n_frames = getattr(container, "n_frames", None)
        if n_frames is not None:
            row_kind = getattr(container, "row_kind", "frame")
            subtitle.append(f"{n_frames} {row_kind}{'s' if n_frames != 1 else ''}")
        if isinstance(container, Scalar):
            subtitle.append(value_text(container))
        self.view.show_item(
            name, " · ".join(subtitle), figure,
            lambda per_frame, c=container: _rows_of(c, per_frame),
            self._descriptor_header(analysis, name, container), notes,
            per_frame_available=stats, footnote=_footnote(self.result))
        self.stack.setCurrentWidget(self.view)

    def _show_family(self, analysis: str, key: str, kind: str) -> None:
        output = self.result.outputs[analysis]
        names = [n for n, c in output.tables.items()
                 if family_key(n, self._elements) == key
                 and type(c).__name__ == kind]
        containers = [output.tables[n] for n in names]
        drawn = containers[:MAX_OVERLAY]
        first_token = key.split()[0] if key.split() else key
        y_name = first_token if kind == "Series" and "*" not in first_token \
            else key
        figure = figure_for_family(drawn, title=key, y_name=y_name,
                                   labels=names[:MAX_OVERLAY])
        if figure is not None and len(containers) > len(drawn):
            figure.annotations = (f"{len(drawn)} of {len(containers)} "
                                  "descriptors are drawn; the rows list "
                                  "every one.",) + tuple(figure.annotations)
        if figure is not None and kind == "Series":
            drift = com_drift_annotation(key, output.notes)
            if drift:
                figure.annotations = (drift,) + tuple(figure.annotations)
        notes = [f"{n}: {note}" for n in names
                 for note in (getattr(output.tables[n], "notes", ()) or ())]
        notes += [f"analysis note: {n}" for n in output.notes]
        stats = all(isinstance(c, (Distribution, Histogram, Series, Scalar))
                    for c in containers)
        header = self._descriptor_header(analysis, None) + \
            [f"descriptors: {', '.join(names)}"]
        self.view.show_item(
            key, f"{analysis} · {len(names)} descriptors that differ "
            "by element", figure,
            lambda per_frame, cs=containers: [
                row for c in cs for row in _rows_of(c, per_frame)],
            header, notes, per_frame_available=stats,
            footnote=_footnote(self.result))
        self.stack.setCurrentWidget(self.view)

    def _show_analysis(self, analysis: str) -> None:
        output = self.result.outputs[analysis]
        parts = [f"<h3>{html.escape(analysis)}</h3>",
                 f"<p>{html.escape(ANALYSIS_SUMMARIES.get(analysis, ''))}"
                 "</p>"]
        if output.error is not None:
            parts.append("<p><b>Not computed.</b> The engine gave this "
                         f"reason:</p><p>{html.escape(output.error)}</p>")
        else:
            parts.append(f"<p>{len(output.tables)} descriptors, "
                         f"{output.seconds:.2f} s.</p>")
        if output.notes:
            parts.append("<p><b>Notes</b></p><ul>" + "".join(
                f"<li>{html.escape(str(n))}</li>" for n in output.notes)
                + "</ul>")
        if output.provenance is not None:
            parts.append("<p><b>Provenance of this analysis</b></p><pre>"
                         + html.escape("\n".join(
                             output.provenance.as_lines())) + "</pre>")
        self.page.setHtml("".join(parts))
        self.stack.setCurrentWidget(self.page)

    def _show_run(self) -> None:
        result = self.result
        parts = ["<h3>Run</h3>",
                 f"<p>{html.escape(summary_line(result))}</p>"]
        if result.cancelled:
            parts.append("<p><b>Cancelled.</b> The analyses hold the frames "
                         "finished before the run stopped.</p>")
        rows = []
        for analysis, output in result.outputs.items():
            status = (f"not computed: {output.error}" if output.error
                      else f"{len(output.tables)} descriptors")
            rows.append(f"<tr><td>{html.escape(analysis)}</td>"
                        f"<td>{html.escape(status)}</td>"
                        f"<td align='right'>{output.seconds:.2f} s</td></tr>")
        parts.append("<table cellspacing='0' cellpadding='3'>"
                     "<tr><th align='left'>analysis</th>"
                     "<th align='left'>result</th><th>time</th></tr>"
                     + "".join(rows) + "</table>")
        if result.notes:
            parts.append("<p><b>Notes</b></p><ul>" + "".join(
                f"<li>{html.escape(str(n))}</li>" for n in result.notes)
                + "</ul>")
        parts.append("<p><b>Provenance</b></p><pre>" + html.escape(
            "\n".join(self.run_header())) + "</pre>")
        self.page.setHtml("".join(parts))
        self.stack.setCurrentWidget(self.page)

    # -- exports -----------------------------------------------------------------
    def _update_actions(self) -> None:
        has = self.result is not None
        busy = self._export_thread is not None
        showing_view = self.stack.currentWidget() is self.view
        self.export_workbook_action.setEnabled(has and not busy)
        self.export_csv_action.setEnabled(has and not busy)
        self.export_all_button.setEnabled(has and not busy)
        self.export_figure_action.setEnabled(
            has and showing_view and self.view.figure is not None)
        self.export_rows_action.setEnabled(has and showing_view)

    def export_all(self, path) -> list[Path]:
        """Every descriptor, now: ``path`` ending .xlsx gives one workbook,
        any other path a folder of CSV files (md_export.export)."""
        if self.result is None:
            raise ValueError("no analysis result to export")
        return list(md_export.export(self.result, path))

    def start_export_all(self, path) -> bool:
        """:meth:`export_all` on a worker thread; ``exportFinished`` or
        ``exportFailed`` follows. False while another export runs."""
        if self.result is None or self._export_thread is not None:
            return False
        worker = _ExportWorker(self.result, str(path))
        self._export_worker = worker
        self.statusMessage.emit(f"Writing {Path(str(path)).name}…")
        self._export_thread = start_worker(
            worker, self._on_export_finished, self._on_export_failed,
            self._on_export_thread_done)
        self._update_actions()
        return True

    def is_exporting(self) -> bool:
        return self._export_thread is not None

    def shutdown(self, timeout_ms: int = 60000) -> bool:
        """Wait for an export in progress; True when none is left running."""
        thread = self._export_thread
        if thread is None:
            return True
        done = thread.wait(timeout_ms)
        if done:
            self._on_export_thread_done()
        return done

    @Slot(object)
    def _on_export_finished(self, paths) -> None:
        self.statusMessage.emit(f"Wrote {len(paths)} file(s): "
                                f"{Path(paths[0]).parent if paths else ''}")
        self.exportFinished.emit(list(paths))

    @Slot(str)
    def _on_export_failed(self, text: str) -> None:
        self.statusMessage.emit(f"Export stopped: {text}")
        self.exportFailed.emit(text)

    @Slot()
    def _on_export_thread_done(self) -> None:
        self._export_thread = None
        self._export_worker = None
        self._update_actions()

    def export_figure(self, path, width_mm: float | None = None,
                      height_mm: float | None = None) -> Path:
        """The figure shown, by suffix: .svg, .pdf or .png (600 dpi)."""
        if self.view.figure is None or self.stack.currentWidget() \
                is not self.view:
            raise ValueError("the item shown has no figure")
        kwargs = {}
        if width_mm is not None:
            kwargs["width_mm"] = width_mm
        if height_mm is not None:
            kwargs["height_mm"] = height_mm
        return self.view.plot.save(path, **kwargs)

    def export_rows(self, path, *, per_frame: bool | None = None) -> Path:
        """The rows shown, as CSV with the provenance header, every float
        in full. ``per_frame`` None follows the view's check box."""
        if self.result is None or self.stack.currentWidget() is not self.view:
            raise ValueError("no descriptor is shown")
        rows = self.view.table.rows()
        if per_frame is not None and per_frame != \
                self.view.per_frame.isChecked() and self.view._rows_for:
            rows = self.view._rows_for(per_frame)
        rows = _uniform(rows) or [{"note": "no rows"}]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        return exporters.write_csv(
            [{k: _full(v) for k, v in row.items()} for row in rows], target,
            provenance=[" ".join(str(line).split())
                        for line in self.view.header])

    # dialogs
    def choose_export_workbook(self) -> None:
        if self.result is None:
            return
        stem = Path(self.result.provenance.source_path).stem or "model"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export every descriptor as a workbook",
            f"{stem}_analysis.xlsx", "Excel workbook (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.start_export_all(path)

    def choose_export_csv(self) -> None:
        if self.result is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Export every descriptor as CSV files into a folder")
        if folder:
            self.start_export_all(folder)

    def choose_export_figure(self) -> None:
        if self.view.figure is None or \
                self.stack.currentWidget() is not self.view:
            # never silent: the menu item can be reached by its shortcut
            self.statusMessage.emit("No figure to save: the item shown has "
                                    "none (choose a descriptor in the tree)")
            return
        path = ask_figure_path(self, "Save the figure shown",
                               file_stem(self.view.title.text(), "figure"))
        if not path:
            return
        try:
            written = self.export_figure(path)
        except (OSError, ValueError) as error:
            self.statusMessage.emit(f"The figure was not written: {error}")
        else:
            self.statusMessage.emit(f"Wrote {written}")

    def choose_export_rows(self) -> None:
        if self.stack.currentWidget() is not self.view:
            self.statusMessage.emit("No rows to save: no descriptor is shown "
                                    "(choose one in the tree)")
            return
        stem = re.sub(r"[^A-Za-z0-9.+=()-]+", "_",
                      self.view.title.text()).strip("_") or "rows"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save the rows shown", f"{stem}.csv", "CSV (*.csv)")
        if not path:
            return
        if not path.lower().endswith(".csv"):
            path += ".csv"
        try:
            written = self.export_rows(path)
        except (OSError, ValueError) as error:
            self.statusMessage.emit(f"The rows were not written: {error}")
        else:
            self.statusMessage.emit(f"Wrote {written}")

    @Slot()
    def _export_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction(self.export_workbook_action)
        menu.addAction(self.export_csv_action)
        menu.exec(self.export_all_button.mapToGlobal(
            self.export_all_button.rect().bottomLeft()))

    # -- theme -------------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        self.view.apply_theme(theme)

    def visible_texts(self) -> list[str]:
        """Every text this browser shows or can show as a tooltip, for a
        reader that checks the wording."""
        texts = [self.summary.text(), self.banner.text(),
                 self.provenance_text.toPlainText(), self.page.toPlainText(),
                 self.view.title.text(), self.view.subtitle.text(),
                 self.view.notes.toPlainText()]
        for widget in self.findChildren(QWidget):
            texts.append(widget.toolTip())
            for getter in ("text", "placeholderText"):
                method = getattr(widget, getter, None)
                if callable(method):
                    try:
                        value = method()
                    except TypeError:
                        continue
                    if isinstance(value, str):
                        texts.append(value)
        for action in (self.export_workbook_action, self.export_csv_action,
                       self.export_figure_action, self.export_rows_action):
            texts += [action.text(), action.toolTip()]
        for item in self._walk():
            for column in range(3):
                texts += [item.text(column), item.toolTip(column)]
        table = self.view.table
        texts += table.columns()
        figure = self.view.figure
        if figure is not None:
            texts += [figure.title, figure.x_label, figure.y_label,
                      *figure.annotations, *(s.label for s in figure.series)]
        return [t for t in texts if t]
