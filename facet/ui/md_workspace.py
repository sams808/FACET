"""The Model workspace: an MD model opened, set up, analysed and exported,
inside the main window.

A model is an entry of the project (:class:`facet.core.project.ModelEntry`)
listed in the Structures dock beside the crystals. Selecting it switches the
window's five stacked regions to the widgets of this module's
:class:`ModelController`, one per entry:

* the Sites dock becomes an **Elements** panel (element, atoms, oxidation
  state, mean CN at v_bond on the frame shown; a click emphasises the
  element's atoms in the view);
* the toolbar rows become the **model toolbar** (3D view | Figure, the frame
  stepper, the drawing style, Highlight…; the labels, the box, the
  projection, and a note with the frame's timestep and bond count);
* the centre shows the frame in the existing ``StructureView``, the figure
  and rows of the descriptor chosen in Results (the result browser's view,
  re-parented here), or, before the file reads, the reader's questions
  (:class:`~.md_dialogs.ReadOptionsPanel`);
* the bottom strip is a :class:`ThresholdStrip`: the threshold panel's
  slider row and the mean-CN-against-v_bond staircase of each element;
* the right column holds four tabs, **Setup** (:class:`~.md_setup.
  SetupPage`), **Results** (the browser's tree and summary), **Highlight**
  (:class:`~.md_highlight.HighlightPage`) and **Notes**.

The flow is the one the former Model window kept:

1. **Reading** (:class:`~.md_jobs.Job` running :func:`~.md_jobs.open_model`
   on a worker thread). A refusal for want of an input only the user has (a
   type map, a topology, a box, units) shows the read panel with the
   reader's message and fields for exactly those inputs; nothing is filled
   in. ``trajectory`` is None until the file reads.
2. **Setup**: the page's preset, formers, frames, thresholds and the
   analyses by question, over the real ``SetupPanel`` the page owns.
3. **Run** (:class:`~.md_jobs.AnalysisJob`): ``md_analysis.analyse`` on a
   worker thread; the page's footer shows the stage, a progress bar and
   Cancel, the status bar the stage. A cancelled run keeps what was
   finished, and never replaces the last complete run.
4. **Results**: the browser lists the descriptors; the figure of the one
   chosen is drawn in the centre, and the toolbar's 3D view brings the
   frame back.
5. **The frame**: loaded on a worker thread (:func:`~.md_jobs.
   load_frame_view`, the bulk engine's one search), then the highlight
   rules are applied to a scene built from it on another worker (the
   by-charge channels take seconds on a 1 Å grid), and the scene is handed
   to the view on the GUI thread. The QPainter tier draws every rule.
6. **Export**: CSV files or an XLSX workbook on a worker thread, the figure
   or rows shown, and the request as a TOML file the command line reads.

THE THREADING RULE: no widget is touched from a worker thread. Every job's
signals reach ``@Slot`` methods of the controller, a ``QObject`` living on
the GUI thread (:mod:`.md_jobs`). Closing the window during a run cancels
the jobs and waits at most :data:`CLOSE_WAIT_MS` for them; work that takes
longer to notice the cancel finishes in the background, held by
``md_jobs`` so that no running thread is destroyed.
"""
from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core import theme as theme_mod
from ..gl.labels import AtomLabel, BondLabel, LabelSettings
from ..gl.scene import Style
from . import chrome, md_jobs
from .md_dialogs import ReadOptionsPanel, _escape, _hint
from .md_plot import Figure, StatSeries
from .md_threshold import ThresholdPanel

__all__ = ["ModelController", "ThresholdStrip", "ElementsPanel",
           "ModelToolbar", "ResultsPage", "md_file_filter",
           "ATOMS_ONLY_ABOVE", "CLOSE_WAIT_MS", "view_class"]

# Above this many atoms the QPainter tier (no OpenGL) draws the atoms of a
# frame without bonds. A choice from the UI scouting's measurements of
# 2026-10-07: the QPainter tier took 0.64 s per frame at 3 000 atoms and
# 1.9-6.4 s at 9 000-24 000, bonds being 69 % of it.
ATOMS_ONLY_ABOVE = 5000

# How long a closing window waits for a model's work to stop before leaving
# it to finish in the background. A choice: long enough for a cancel between
# two frames of a small model, short enough that closing never feels stuck.
CLOSE_WAIT_MS = 3000

# How long the highlight rules wait after an edit (or a move of the
# threshold strip) before the scene is rebuilt: a slider dragged across the
# strip then costs one rebuild, not one per pixel.
REDRAW_DELAY_MS = 150

# The drawing styles the model toolbar offers (no ellipsoids: an MD frame
# has no displacement parameters, md_scene draws that style as balls).
STYLES = (Style.BALL_AND_STICK, Style.SPACE_FILLING, Style.STICK,
          Style.WIREFRAME)
# The labels a frame can carry without a Structure: md_scene names each atom
# by element and file id; CN and BVS labels read a crystal's site results.
ATOM_LABELS = (AtomLabel.NONE, AtomLabel.ELEMENT, AtomLabel.SITE,
               AtomLabel.ELEMENT_INDEX)
BOND_LABELS = (BondLabel.NONE, BondLabel.DISTANCE, BondLabel.VALENCE,
               BondLabel.BOTH)

# What the result browser says while it holds no run.
EMPTY_TEXT = ("No run yet. Tick analyses in Setup and press Run analyses "
              "(Ctrl+R); the results are listed here.")
RUNNING_TEXT = ("A run is in progress (its stage is under the Run button "
                "and in the status bar); its results are listed here when it "
                "ends.")


def view_class():
    """The 3D view the workspace draws a frame in: the window's own
    ``StructureView``. A module attribute (``VIEW_CLASS``) so a script
    without an OpenGL context can put a QPainter-only widget of the same
    interface in its place."""
    global VIEW_CLASS
    if VIEW_CLASS is None:
        from ..gl.view import StructureView

        VIEW_CLASS = StructureView
    return VIEW_CLASS


VIEW_CLASS = None


def md_file_filter() -> str:
    """A file-dialog filter of the MD formats read_trajectory reads: the
    main window's own (``preview.md_file_filter``), so every file dialog
    lists the same names; a list of the names when that module is not in
    this build."""
    try:
        from .preview import md_file_filter as window_filter

        return window_filter()
    except Exception:              # noqa: BLE001 - the filter is a convenience
        pass
    from ..core import md_readers

    patterns = ["*.lammpstrj", "*.lammpstrj.gz", "*.dump", "*.dump.gz",
                "*.data", "*.lmp", "*.extxyz", "*.xyz", "XDATCAR*",
                "*.vasp", "CONFIG*", "REVCON*", "HISTORY*", "*.config",
                "*.history"]
    try:
        for spec in md_readers.format_specs():
            patterns.extend(f"*{e}" for e in spec.extensions)
            patterns.extend(f"{s}*" for s in spec.stems)
    except Exception:              # noqa: BLE001 - the filter is a convenience
        pass
    unique = list(dict.fromkeys(patterns))
    return (f"MD models ({' '.join(unique)});;All files (*)")


def _normalise_source(paths):
    if isinstance(paths, (list, tuple)):
        items = [str(p) for p in paths]
        return items[0] if len(items) == 1 else items
    return str(paths)


def _thousands(n: int) -> str:
    return f"{int(n):,}".replace(",", " ")


# ---------------------------------------------------------------------------
# the threshold strip
# ---------------------------------------------------------------------------

class ThresholdStrip(ThresholdPanel):
    """The threshold panel trimmed to its v_bond row and the staircase of
    each element (mean CN against v_bond on the frame shown), in the place
    and the look of the crystal's cutoff explorer. Everything the panel
    measures (``element_rows``, ``results``, ``current_bonds``) stays
    available; its other plots and its table are hidden, not removed."""

    def __init__(self, parent=None, *, theme=None):
        super().__init__(parent, theme=theme)
        for widget in (self.cn_plot, self.bvs_plot, self.phi_plot,
                       self.element_table, self.banner, self.heading):
            widget.hide()
        self.stairs_plot.setMinimumHeight(120)
        self.stairs_plot.setMinimumWidth(240)
        self.layout().setContentsMargins(6, 2, 6, 2)
        self.setToolTip("The threshold strip: the mean CN of each element "
                        "against v_bond on the frame shown. Click or drag "
                        "to move v_bond; the view and the Elements dock "
                        "follow, with no new neighbour search.")
        self.stairs_plot.set_empty_text("No frame is loaded yet.")

    def _draw_staircase(self) -> None:
        series = []
        for element in self._elements:
            s = StatSeries(element, self._grid, self._stairs[element], None,
                           kind="line", markers=True, band=False)
            s.style = self._style(element)
            series.append(s)
        low, high = self._range
        self.stairs_plot.set_figure(Figure(series, "v_bond", "v.u.",
                                           "mean CN", "1",
                                           log_x=high > low * 1.0001))

    def _redraw(self) -> None:
        # the hidden plots and table are not refilled on every move
        self.stairs_plot.set_vlines(self._vlines())


# ---------------------------------------------------------------------------
# the Elements panel (what the Sites dock shows for a model)
# ---------------------------------------------------------------------------

class ElementsPanel(QWidget):
    """Per element of the frame shown: atoms, oxidation state (a model
    input) and mean CN at the strip's v_bond. ``elementChosen(symbol)`` on
    a click ('' when the chosen element is clicked again, or the selection
    cleared): the window emphasises that element's atoms in the view."""

    elementChosen = Signal(str)
    COLUMNS = ("element", "atoms", "state", "CN")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.chosen = ""
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setToolTip("The frame shown at the strip's v_bond. Click "
                              "an element to emphasise its atoms in the view "
                              "(click it again to show every atom alike); "
                              "the oxidation state is a model input "
                              "(Setup > More…).")
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.itemClicked.connect(self._on_click)
        column.addWidget(self.table, 1)
        self.note = _hint("No frame is loaded yet.")
        column.addWidget(self.note)

    def set_rows(self, rows: Sequence[Mapping], states: Mapping[str, int],
                 v_bond: float) -> None:
        table = self.table
        table.blockSignals(True)
        table.setRowCount(0)
        for row in rows:
            i = table.rowCount()
            table.insertRow(i)
            element = str(row["element"])
            state = states.get(element)
            mean = row.get("mean CN", float("nan"))
            cells = (element, _thousands(row["atoms"]),
                     "" if state is None else f"{int(state):+d}",
                     "" if mean != mean else f"{mean:.2f}")
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                table.setItem(i, c, item)
            if element == self.chosen:
                table.selectRow(i)
        table.blockSignals(False)
        self.note.setText(f"mean CN at v_bond {v_bond:.4g} v.u.; the states "
                          "are model inputs")

    def clear(self) -> None:
        self.table.setRowCount(0)
        self.chosen = ""
        self.note.setText("No frame is loaded yet.")

    def elements(self) -> list[str]:
        return [self.table.item(r, 0).text()
                for r in range(self.table.rowCount())]

    def choose(self, symbol: str) -> None:
        """Emphasise ``symbol`` ('' for none); the same as a click."""
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
# the model toolbar
# ---------------------------------------------------------------------------

class ModelToolbar(QWidget):
    """The two toolbar rows of a model: the canvas switch, the frame
    stepper, the style, Highlight…; the labels, the box, the projection
    and the frame's note."""

    canvasRequested = Signal(str)        # '3d' or 'figure'
    frameRequested = Signal(int)
    styleChanged = Signal()
    highlightRequested = Signal()
    labelsChanged = Signal(object)       # LabelSettings
    boxToggled = Signal(bool)
    orthographicToggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        stack = QVBoxLayout(self)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)

        upper = QWidget()
        row = QHBoxLayout(upper)
        row.setContentsMargins(10, 6, 10, 3)
        row.setSpacing(10)
        stack.addWidget(upper)

        self.show_3d = QPushButton("3D view")
        self.show_figure = QPushButton("Figure")
        self.canvas_group = QButtonGroup(self)
        for button in (self.show_3d, self.show_figure):
            button.setCheckable(True)
            self.canvas_group.addButton(button)
        self.show_3d.setChecked(True)
        self.show_3d.setToolTip("The frame in the 3D view")
        self.show_figure.setToolTip("The figure and rows of the descriptor "
                                    "chosen in Results")
        self.show_3d.clicked.connect(lambda: self.canvasRequested.emit("3d"))
        self.show_figure.clicked.connect(
            lambda: self.canvasRequested.emit("figure"))
        switch = QHBoxLayout()
        switch.setSpacing(0)
        switch.addWidget(self.show_3d)
        switch.addWidget(self.show_figure)
        row.addLayout(switch)

        self.prev_button = QToolButton()
        self.prev_button.setText("◀")
        self.prev_button.setToolTip("Previous frame (Page Up)")
        self.prev_button.clicked.connect(lambda: self.step(-1))
        self.frame_box = QSpinBox()
        self.frame_box.setRange(0, 0)
        self.frame_box.setPrefix("Frame ")
        self.frame_box.setSuffix(" of 0")
        self.frame_box.setKeyboardTracking(False)
        self.frame_box.setToolTip("The frame drawn and measured in the strip "
                                  "below; a run uses the frames chosen in "
                                  "Setup.")
        self.frame_box.valueChanged.connect(self.frameRequested)
        self.next_button = QToolButton()
        self.next_button.setText("▶")
        self.next_button.setToolTip("Next frame (Page Down)")
        self.next_button.clicked.connect(lambda: self.step(1))
        row.addWidget(self.prev_button)
        row.addWidget(self.frame_box)
        row.addWidget(self.next_button)

        self.style_box = QComboBox()
        for style in STYLES:
            self.style_box.addItem(style.value, style)
        self.style_box.setToolTip("How the atoms and bonds are drawn")
        self.style_box.activated.connect(lambda _i: self.styleChanged.emit())
        row.addWidget(self.style_box)
        self.highlight_button = QPushButton("Highlight…")
        self.highlight_button.setToolTip("Rules that colour, filter and "
                                         "annotate the view (the Highlight "
                                         "tab; H)")
        self.highlight_button.clicked.connect(self.highlightRequested)
        row.addWidget(self.highlight_button)
        row.addStretch(1)

        lower = QWidget()
        row = QHBoxLayout(lower)
        row.setContentsMargins(10, 3, 10, 6)
        row.setSpacing(10)
        stack.addWidget(lower)
        row.addWidget(QLabel("Label atoms"))
        self.atom_label_box = QComboBox()
        for kind in ATOM_LABELS:
            self.atom_label_box.addItem(kind.value, kind)
        self.atom_label_box.setToolTip("What each atom is labelled with: its "
                                       "element, its element and id in the "
                                       "file, or a number per element")
        self.atom_label_box.activated.connect(lambda _i: self._on_labels())
        row.addWidget(self.atom_label_box)
        row.addWidget(QLabel("bonds"))
        self.bond_label_box = QComboBox()
        for kind in BOND_LABELS:
            self.bond_label_box.addItem(kind.value, kind)
        self.bond_label_box.setToolTip("What each drawn bond is labelled "
                                       "with: its length in Å, its valence "
                                       "in v.u., or both")
        self.bond_label_box.activated.connect(lambda _i: self._on_labels())
        row.addWidget(self.bond_label_box)
        self.box_check = QCheckBox("Box")
        self.box_check.setChecked(True)
        self.box_check.setToolTip("Draw the periodic box of the frame")
        self.box_check.toggled.connect(self.boxToggled)
        row.addWidget(self.box_check)
        self.ortho_check = QCheckBox("Orthographic")
        self.ortho_check.setToolTip("An orthographic projection: parallel "
                                    "lines stay parallel, as in a figure")
        self.ortho_check.toggled.connect(self.orthographicToggled)
        row.addWidget(self.ortho_check)
        self.frame_note = _hint("")
        self.frame_note.setWordWrap(False)
        # clipped before it widens the window: a note, not a control
        self.frame_note.setSizePolicy(QSizePolicy.Ignored,
                                      QSizePolicy.Preferred)
        row.addWidget(self.frame_note, 1)
        self.fit_combos()
        self.set_enabled(False)

    def fit_combos(self) -> None:
        for box in (self.style_box, self.atom_label_box, self.bond_label_box):
            box.setMinimumWidth(0)
            chrome.fit_combo(box)

    def set_enabled(self, on: bool) -> None:
        for widget in (self.show_3d, self.show_figure, self.prev_button,
                       self.frame_box, self.next_button, self.style_box,
                       self.highlight_button, self.atom_label_box,
                       self.bond_label_box, self.box_check, self.ortho_check):
            widget.setEnabled(bool(on))

    def set_model(self, n_frames: int) -> None:
        n = max(int(n_frames), 1)
        self.frame_box.blockSignals(True)
        self.frame_box.setRange(0, n - 1)
        self.frame_box.setSuffix(f" of {n}")
        self.frame_box.setValue(0)
        self.frame_box.blockSignals(False)
        self.set_enabled(True)
        self.prev_button.setEnabled(n > 1)
        self.next_button.setEnabled(n > 1)
        self.frame_box.setEnabled(n > 1)

    def set_frame(self, k: int, note: str = "", tip: str = "") -> None:
        self.frame_box.blockSignals(True)
        self.frame_box.setValue(int(k))
        self.frame_box.blockSignals(False)
        self.frame_note.setText(note)
        self.frame_note.setToolTip(tip or note)

    def set_canvas(self, which: str) -> None:
        (self.show_figure if which == "figure" else self.show_3d).setChecked(
            True)

    def step(self, by: int) -> None:
        """Ask for the frame ``by`` steps away (clamped to the range)."""
        wanted = min(max(self.frame_box.value() + int(by),
                         self.frame_box.minimum()), self.frame_box.maximum())
        if wanted != self.frame_box.value():
            self.frame_box.setValue(wanted)       # emits frameRequested

    def style(self) -> Style:
        return self.style_box.currentData() or Style.BALL_AND_STICK

    def labels(self) -> LabelSettings:
        settings = LabelSettings()
        settings.atom = self.atom_label_box.currentData() or AtomLabel.NONE
        settings.bond = self.bond_label_box.currentData() or BondLabel.NONE
        return settings

    def _on_labels(self) -> None:
        self.labelsChanged.emit(self.labels())


# ---------------------------------------------------------------------------
# the Results tab: the browser's tree in the column, its figure in the centre
# ---------------------------------------------------------------------------

class ResultsPage(QWidget):
    """The result browser's summary row, a filter field and its tree in the
    column; the browser's figure/rows stack is handed to the centre
    (``canvas``), as the Setup page hands its footer to the window."""

    def __init__(self, browser, parent=None):
        super().__init__(parent)
        self.browser = browser
        column = QVBoxLayout(self)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(4)
        tree, stack = browser.tree, browser.stack
        browser.splitter.hide()
        tree.setParent(None)
        stack.setParent(None)
        self.canvas = stack
        browser.layout().setContentsMargins(0, 0, 0, 0)
        column.addWidget(browser)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter descriptors…  (name, element, "
                                       "analysis)")
        self.search.setClearButtonEnabled(True)
        self.search.setToolTip("Hide the descriptors whose label and engine "
                               "id lack this text; an analysis stays when "
                               "one of its descriptors matches.")
        self.search.textChanged.connect(self.filter)
        column.addWidget(self.search)
        # the kind chips sit with the search field, over the tree
        chips = browser.kind_chips
        if chips.parentWidget() is not None \
                and chips.parentWidget().layout() is not None:
            chips.parentWidget().layout().removeWidget(chips)
        chips.setParent(None)
        column.addWidget(chips)
        tree.setMinimumWidth(200)
        column.addWidget(tree, 1)
        self.note = _hint("Click a descriptor: its figure and rows are shown "
                          "in the centre; the 3D view comes back from the "
                          "toolbar.")
        column.addWidget(self.note)

    def filter(self, text: str) -> None:
        """The browser hides the rows whose label and engine id lack this
        text, combined with the kind chips."""
        self.browser.set_text_filter(text)

    def visible_descriptors(self) -> list[str]:
        """The engine ids of the descriptor leaves the filters leave
        shown."""
        return self.browser.visible_descriptors()


# ---------------------------------------------------------------------------
# a field the Setup page does not show, revealed in a small dialog
# ---------------------------------------------------------------------------

class _FieldDialog(QDialog):
    """One option group of the hidden SetupPanel, shown non-modally so a
    field a reason names can be filled; the group goes back to the panel
    when the dialog closes."""

    def __init__(self, group, panel, where: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{group.title()} inputs")
        self.setModal(False)
        self.group = group
        self.panel = panel
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.scroll = chrome.in_scroll_area(group)
        layout.addWidget(self.scroll, 1)
        note = _hint("Every value typed here is part of the request; the "
                     "Setup tab's tree shows what each analysis runs with.")
        layout.addWidget(note)
        row = QHBoxLayout()
        row.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        row.addWidget(close)
        layout.addLayout(row)
        self.resize(560, 420)
        group.show()
        panel.focus_field(where)

    def closeEvent(self, event) -> None:
        group = self.group
        self.scroll.takeWidget()
        group.setParent(self.panel)
        layout = self.panel.layout()
        if layout is not None:
            layout.addWidget(group)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------

class ModelController(QObject):
    """One MD model in the main window: reading, setup, run, results, the
    frame and its highlight rules, export.

    ``entry`` is the project's :class:`~facet.core.project.ModelEntry`; the
    controller fills its trajectory, summary, request, result and frame as
    it goes. ``window`` is the main window (the parent of the dialogs, the
    source of the theme); it adds the five widgets below to its stacked
    regions and shows them when the entry is selected:

    ``column`` (the right column: a pending note, then the four tabs),
    ``elements`` (the Sites dock's page), ``toolbar``, ``canvas`` (the
    centre: the reading note, the read panel, the 3D ``view``, the figure)
    and ``strip`` (the threshold strip).

    Signals: ``opened(ModelSummary | ReadProblem)`` once the file reads or
    the reader refuses; ``finished(ModelResult | JobFailure)`` after each
    run; ``frameShown(k)``; ``statusMessage(text)`` for the status bar;
    ``stateChanged()`` whenever the menus' enabled state may differ.
    """

    opened = Signal(object)
    finished = Signal(object)
    frameShown = Signal(int)
    statusMessage = Signal(str)
    stateChanged = Signal()

    def __init__(self, entry, window=None, *, theme=None, read_options=None):
        super().__init__(window if isinstance(window, QObject) else None)
        self.entry = entry
        self.window = window
        theme = theme if theme is not None else getattr(window, "theme", None)
        self.theme = theme if theme is not None else theme_mod.Theme()
        self.source = _normalise_source(entry.path)
        if read_options is not None:
            entry.read_options = dict(read_options)
        self.read_options = dict(entry.read_options or {})
        self.companions: tuple = tuple(entry.companions or ())
        self.trajectory = None
        self.summary = None
        self.problem = None
        self.result = None
        self.complete_result = None
        self.partial_result = None
        self.last_failure = None
        self.last_request = None
        self.page = None                 # SetupPage
        self.highlight = None            # HighlightPage
        self.browser = None              # ResultBrowser
        self.results_page = None
        self.notes_text = None
        self.tabs = None
        self.setup_tab = None
        self.frame_view = None
        self.frame_data = None
        self._frame_data_key = None
        self.element = ""                # the Elements dock's emphasis
        self._open_job = None
        self._run_job = None
        self._frame_job = None
        self._highlight_job = None
        self._export_job = None
        self._links: dict = {}
        self._closing = False
        self._pending_frame = None
        self._pending_redraw = False
        self._reframe_next = True
        self._model_generation = 0
        self._pending_request_spec = (dict(entry.request_spec)
                                      if entry.request_spec else None)
        self._field_dialogs: list = []
        self.last_draw_s = 0.0
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(REDRAW_DELAY_MS)
        self._redraw_timer.timeout.connect(self.redraw)
        entry.controller = self
        self._build_widgets()
        self.open_model()

    # -- widgets ---------------------------------------------------------------
    def _build_widgets(self) -> None:
        # the right column: a note until the file reads, then the tabs
        self.column = QStackedWidget()
        pending = QWidget()
        pending_col = QVBoxLayout(pending)
        pending_col.addStretch(1)
        self.pending_label = QLabel()
        self.pending_label.setWordWrap(True)
        self.pending_label.setAlignment(Qt.AlignCenter)
        pending_col.addWidget(self.pending_label)
        pending_col.addStretch(2)
        self.column.addWidget(pending)
        self.column.setMinimumWidth(400)

        self.elements = ElementsPanel()
        self.elements.elementChosen.connect(self._on_element_chosen)

        self.toolbar = ModelToolbar()
        self.toolbar.canvasRequested.connect(self.show_canvas)
        self.toolbar.frameRequested.connect(self._on_frame_box)
        self.toolbar.styleChanged.connect(self._schedule_redraw)
        self.toolbar.highlightRequested.connect(self.show_highlight)
        self.toolbar.labelsChanged.connect(self._on_labels)
        self.toolbar.boxToggled.connect(lambda _on: self._schedule_redraw())
        self.toolbar.orthographicToggled.connect(self._on_orthographic)

        # the centre: reading, the read panel, the frame, the figure
        self.canvas = QStackedWidget()
        self.reading_page = QWidget()
        reading_col = QVBoxLayout(self.reading_page)
        reading_col.addStretch(1)
        self.reading_label = QLabel()
        self.reading_label.setAlignment(Qt.AlignCenter)
        self.reading_label.setWordWrap(True)
        busy = QProgressBar()
        busy.setRange(0, 0)
        busy.setMaximumWidth(360)
        busy_row = QHBoxLayout()
        busy_row.addStretch(1)
        busy_row.addWidget(busy)
        busy_row.addStretch(1)
        reading_col.addWidget(self.reading_label)
        reading_col.addLayout(busy_row)
        reading_col.addStretch(2)
        self.canvas.addWidget(self.reading_page)
        self.read_panel = ReadOptionsPanel(theme=self.theme)
        self.read_panel.setMaximumWidth(1100)
        self.read_panel.readRequested.connect(self._on_read_again)
        self.read_panel.openOtherRequested.connect(self.choose_md_file)
        self.read_page = chrome.in_scroll_area(self.read_panel)
        self.canvas.addWidget(self.read_page)
        self.view = view_class()()
        self.view.set_theme(self.theme)
        self.view.ready.connect(self._on_view_ready)
        self.view.measured.connect(self.statusMessage)
        self.canvas.addWidget(self.view)
        self.figure_page = None

        self.strip = ThresholdStrip(theme=self.theme)
        self.strip.thresholdChanged.connect(self._on_strip_threshold)
        self.strip.statusMessage.connect(self._relay_status)

    @property
    def widgets(self) -> tuple:
        """The five widgets the window puts in its stacked regions."""
        return (self.column, self.elements, self.toolbar, self.canvas,
                self.strip)

    def _build_workspace(self) -> None:
        """The four tabs, built once the file has read (and again after a
        read with other options)."""
        from .md_highlight import HighlightPage
        from .md_setup import SetupPage
        from .md_views import ResultBrowser

        self._drop_workspace()
        self.page = SetupPage(self.summary, self.trajectory, theme=self.theme)
        self.page.runRequested.connect(self._run_from_page)
        self.page.cancelRequested.connect(self.cancel)
        self.page.rereadRequested.connect(self.reread)
        self.page.fieldRequested.connect(self._on_field_requested)
        self.page.changed.connect(self._on_page_changed)
        self.setup_tab = QWidget()
        setup_col = QVBoxLayout(self.setup_tab)
        setup_col.setContentsMargins(0, 0, 0, 0)
        setup_col.setSpacing(0)
        self.setup_scroll = chrome.in_scroll_area(self.page)
        setup_col.addWidget(self.setup_scroll, 1)
        setup_col.addWidget(self.page.detach_footer())

        self.browser = ResultBrowser(theme=self.theme)
        self.browser.set_empty_text(EMPTY_TEXT)
        self.browser.statusMessage.connect(self._relay_status)
        self.browser.tree.currentItemChanged.connect(self._on_descriptor)
        # a click on the descriptor already current brings its figure back
        # from the 3D view as well
        self.browser.tree.itemClicked.connect(self._on_descriptor)
        for name in ("export_figure_action", "export_rows_action"):
            getattr(self.browser, name).changed.connect(self.stateChanged)
        self.results_page = ResultsPage(self.browser)
        # in a scroll area: the figure's rows and notes want 350 px of
        # height, and a stacked centre takes the largest page's minimum,
        # which put the window's minimum height above a 768 px screen
        self.figure_page = chrome.in_scroll_area(self.results_page.canvas)
        self.canvas.addWidget(self.figure_page)

        self.highlight = HighlightPage(theme=self.theme)
        self.highlight.rulesChanged.connect(self._schedule_redraw)
        self.highlight_scroll = chrome.in_scroll_area(self.highlight)

        self.notes_text = QPlainTextEdit()
        self.notes_text.setReadOnly(True)
        self.notes_text.setLineWrapMode(QPlainTextEdit.WidgetWidth)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.setup_tab, "Setup")
        self.tabs.addTab(self.results_page, "Results")
        self.tabs.addTab(self.highlight_scroll, "Highlight")
        self.tabs.addTab(self.notes_text, "Notes")
        self.tabs.setMinimumWidth(400)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.currentChanged.connect(self._on_tab)
        self.column.addWidget(self.tabs)
        self.column.setCurrentWidget(self.tabs)
        self.toolbar.set_model(self.summary.n_frames)
        self.elements.clear()
        self.strip.clear()
        self._show_model_notes()
        self.show_canvas("3d")

    def _drop_workspace(self) -> None:
        for dialog in self._field_dialogs:
            try:
                dialog.close()
            except RuntimeError:
                pass
        self._field_dialogs = []
        for widget in (self.tabs, self.figure_page):
            if widget is None:
                continue
            holder = widget.parentWidget()
            stack = (holder if isinstance(holder, QStackedWidget)
                     else (self.column if widget is self.tabs else self.canvas))
            if stack.indexOf(widget) >= 0:
                stack.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
        self.tabs = self.figure_page = None
        self.page = self.highlight = self.browser = None
        self.results_page = self.notes_text = None

    # -- reading ---------------------------------------------------------------
    def open_model(self, read_options=None) -> None:
        """Read the source (again) on a worker thread."""
        if self._open_job is not None and self._open_job.pending:
            self.statusMessage.emit("The model file is being read")
            return
        if read_options is not None:
            self.read_options = dict(read_options)
            self.entry.read_options = dict(read_options)
        source = self.source
        options = dict(self.read_options)
        name = _escape(md_jobs.source_label(source))
        self.reading_label.setText(f"Reading {name} …")
        self.pending_label.setText(f"Reading {name}: the setup is built once "
                                   "the file has read.")
        self.canvas.setCurrentWidget(self.reading_page)
        self.column.setCurrentIndex(0)
        self.statusMessage.emit(f"Reading {md_jobs.source_label(source)}")

        def work(progress, cancelled):
            return md_jobs.open_model(source, options)

        job = md_jobs.Job(work, name="open")
        self._connect(job, succeeded=self._on_open_done,
                      failed=self._on_open_failed)
        self._open_job = job
        job.start()
        self.stateChanged.emit()

    @Slot(object)
    def _on_open_done(self, outcome) -> None:
        if self._closing:
            return
        entry = self.entry
        if isinstance(outcome, md_jobs.Opened):
            previous = self.trajectory
            self.trajectory = outcome.trajectory
            self.summary = outcome.summary
            self.problem = None
            entry.trajectory, entry.summary, entry.problem = (
                self.trajectory, self.summary, None)
            entry.read_options = dict(self.summary.read_options)
            self.read_options = dict(self.summary.read_options)
            # a model read again (other read options) is a new model: its
            # results and frame are those of the new read
            self.result = self.complete_result = self.partial_result = None
            entry.result = None
            self.frame_view = self.frame_data = None
            entry.frame_view = None
            self._reframe_next = True
            self._model_generation += 1
            if previous is not None and previous is not outcome.trajectory \
                    and not self._busy():
                close = getattr(previous, "close", None)
                if close is not None:
                    close()
            self._build_workspace()
            self.statusMessage.emit(
                f"Read {self.summary.label}: {self.summary.n_atoms} atoms × "
                f"{self.summary.n_frames} frame(s) ({self.summary.file_format})")
            if self._pending_request_spec:
                spec, self._pending_request_spec = self._pending_request_spec, None
                try:
                    self.apply_request_spec(spec)
                except (ValueError, KeyError) as error:
                    self.statusMessage.emit(f"The session's request was not "
                                            f"applied in full: {error}")
            self.stateChanged.emit()
            self.opened.emit(self.summary)
            QTimer.singleShot(0, self._first_frame)
            return
        self.trajectory = None
        self.problem = outcome
        entry.problem = outcome
        if self.companions:
            self.read_panel.companions = tuple(self.companions)
        self.read_panel.set_problem(outcome)
        self.canvas.setCurrentWidget(self.read_page)
        self.pending_label.setText(
            "The reader asks for more information: answer it in the centre, "
            "then Read again." if outcome.can_supply
            else "The file was not read; the reader's reason is in the "
                 "centre.")
        self.column.setCurrentIndex(0)
        self.statusMessage.emit(
            "The reader asks for more information" if outcome.can_supply
            else "The file was not read")
        self.stateChanged.emit()
        self.opened.emit(outcome)

    @Slot(object)
    def _on_open_failed(self, failure) -> None:
        """An exception the read did not sort into a refusal: shown in
        full, and the fields of the last refusal stay (the options it asked
        for, the options the format takes), so the inputs can be typed
        again rather than the workspace dead-ending."""
        if self._closing:
            return
        previous = self.problem
        if previous is not None:
            problem = md_jobs.ReadProblem(
                source=self.source, read_options=dict(self.read_options),
                message=f"{failure.message}\n\n(The read before this one "
                        f"reported: {previous.message})",
                error_type="error", file_format=previous.file_format,
                options_taken=previous.options_taken, needs=previous.needs,
                type_keys=previous.type_keys,
                keys_complete=previous.keys_complete,
                evidence=dict(previous.evidence),
                counts=dict(previous.counts),
                counts_note=previous.counts_note,
                suggested=dict(previous.suggested))
        else:
            problem = md_jobs.ReadProblem(
                source=self.source, read_options=dict(self.read_options),
                message=failure.message, error_type="error",
                file_format=None, options_taken=frozenset(), needs=())
        self._on_open_done(problem)

    @Slot(object)
    def _on_read_again(self, options) -> None:
        self.open_model(options)

    @Slot(object)
    def reread(self, options) -> bool:
        """Read the model file again with ``options`` (the setup's Read
        options box): the setup is built anew for the model read. False,
        with the reason in the status bar, while a run is in progress."""
        if self.is_running():
            self.statusMessage.emit("A run is in progress: cancel it, or "
                                    "wait for it, before the file is read "
                                    "again")
            return False
        self.open_model(dict(options or {}))
        return True

    def set_companions(self, paths) -> None:
        """Files opened together with this model (a LAMMPS data file dropped
        with its dump): the read panel offers each beside the fields it can
        answer (masses, topology, box), with a button; none is filled in."""
        self.companions = tuple(str(p) for p in paths or ())
        self.entry.companions = self.companions
        self.read_panel.set_companions(self.companions)

    @Slot()
    def choose_md_file(self) -> None:
        """Open another MD file (the read panel's button): the window lists
        it as a model of its own."""
        opener = getattr(self.window, "_choose_md_files", None)
        if opener is not None:
            opener()

    # -- the run -------------------------------------------------------------
    @property
    def request(self):
        """The setup's AnalysisRequest (None while it cannot run)."""
        if self.page is None:
            return None
        return self.page.panel.request()

    def can_run(self) -> tuple[bool, str]:
        if self.page is None:
            return False, "No model is read yet"
        if self.is_running():
            return False, "A run is in progress"
        return self.page.can_run()

    def run_analysis(self, request=None) -> bool:
        """Start a run of ``request`` (the setup's when None); False when
        nothing was started, with the reason in the status bar."""
        from ..core import md_analysis as ma

        if self.trajectory is None or self.page is None:
            self.statusMessage.emit("No model is read yet")
            return False
        if self.is_running():
            self.statusMessage.emit("A run is in progress")
            return False
        if request is None or isinstance(request, bool):
            try:
                request = self.page.request()
            except ma.RequestError as error:
                self.statusMessage.emit(str(error))
                return False
        self.last_request = request
        self.entry.request = request
        job = md_jobs.AnalysisJob(self.trajectory, request)
        self._connect(job, progressed=self._on_run_progress,
                      succeeded=self._on_run_finished,
                      failed=self._on_run_failed, ended=self._on_run_ended)
        self._run_job = job
        job.start()
        self._set_running(True)
        self.page.set_progress(0, 1, "starting")
        self.statusMessage.emit("Starting: " + ", ".join(request.analyses))
        return True

    @Slot()
    def _run_from_page(self) -> None:
        self.run_analysis()

    @Slot()
    def cancel(self) -> None:
        """Ask the run to stop after the frame in progress."""
        job = self._run_job
        if job is None or not job.pending:
            return
        job.cancel()
        if self.page is not None:
            self.page.cancel_button.setEnabled(False)
        self.statusMessage.emit("Cancelling: the run stops after the frame "
                                "or analysis in progress, and keeps what is "
                                "finished")
        self.stateChanged.emit()

    def is_running(self) -> bool:
        """True from Run until the run's result (or failure) arrives."""
        return self._run_job is not None and self._run_job.pending

    @Slot(int, int, str)
    def _on_run_progress(self, done: int, total: int, stage: str) -> None:
        if self._closing or self.page is None:
            return
        self.page.set_progress(done, total, stage)
        elapsed = self._run_job.elapsed_s() if self._run_job else 0.0
        prefix = "Cancelling; " if self._run_job and \
            self._run_job.cancel_requested else ""
        self.statusMessage.emit(f"{prefix}{stage} ({elapsed:.0f} s)")

    @Slot(object)
    def _on_run_finished(self, result) -> None:
        if self._closing:
            return
        self.last_failure = None
        self._set_running(False)
        elapsed = self._run_job.elapsed_s() if self._run_job else 0.0
        failed = result.failed
        extra = f"; {len(failed)} analysis(es) produced nothing" if failed \
            else ""
        if result.cancelled and self.complete_result is not None:
            # a cancel never takes the last complete run away: it stays on
            # show, and the partial result is one menu item away
            self.partial_result = result
            self.stateChanged.emit()
            self.statusMessage.emit(
                f"Run cancelled after {elapsed:.1f} s{extra}; the last "
                "complete run stays on show (Model > Show the cancelled "
                "run's partial result shows what was finished)")
            self.finished.emit(result)
            return
        if result.cancelled:
            self.partial_result = result
        else:
            self.complete_result = result
            self.partial_result = None
        self._show_stored(result)
        state = "cancelled; the finished part is shown" if result.cancelled \
            else "finished"
        self.statusMessage.emit(f"Run {state} in {elapsed:.1f} s{extra}")
        self.finished.emit(result)

    def _show_stored(self, result) -> None:
        self.result = result
        self.entry.result = result
        self._show_result(result)
        self._frame_to_strip()
        # the formers of the run feed the highlight descriptors (Qn, former
        # CN) when none are ticked
        self.frame_data = None
        if self.highlight is not None and (
                any(r.enabled for r in self.highlight.rules())
                or self.highlight.colour_by() != "element"):
            self._schedule_redraw()
        self.stateChanged.emit()

    @Slot()
    def show_complete_result(self) -> bool:
        """Show the last run that finished uncancelled."""
        if self.complete_result is None:
            return False
        self._show_stored(self.complete_result)
        self.statusMessage.emit("Showing the last complete run")
        return True

    @Slot()
    def show_partial_result(self) -> bool:
        """Show what the cancelled run finished."""
        if self.partial_result is None:
            return False
        self._show_stored(self.partial_result)
        self.statusMessage.emit("Showing the cancelled run's partial result "
                                "(the frames it finished)")
        return True

    @Slot(object)
    def _on_run_failed(self, failure) -> None:
        if self._closing:
            return
        self.last_failure = failure
        self._set_running(False)
        tracks = [m for m in failure.missing
                  if f": {md_jobs.TRACKS_INPUT}: " in m] \
            if failure.kind == "request" else []
        if failure.kind == "cancelled":
            text = ("The run was cancelled before any frame was analysed; "
                    "nothing to show")
        elif tracks:
            request = getattr(self._run_job, "request", None)
            reason = tracks[0].split(f": {md_jobs.TRACKS_INPUT}: ", 1)[1]
            if self.page is not None and request is not None:
                self.page.panel.note_refusal(request, reason)
            text = ("The run did not start: the tracks the dynamics analyses "
                    f"read could not be collected ({reason}). Those analyses "
                    "now say so in Setup; Run analyses runs the others")
        elif failure.kind == "request":
            text = "The run needs: " + "; ".join(failure.missing)
        else:
            text = f"The run stopped: {failure.message}"
        self.statusMessage.emit(text)
        notes = text + ("\n\n" + failure.detail if failure.detail else "")
        if self.result is not None:
            notes += ("\n\nThe results on show are those of an earlier run, "
                      "whose notes follow.\n\n"
                      + "\n".join(self.notes_lines(self.result)))
        if self.notes_text is not None:
            self.notes_text.setPlainText(notes)
        self.finished.emit(failure)

    @Slot()
    def _on_run_ended(self) -> None:
        if self._closing:
            return
        self._set_running(self.is_running())

    def _set_running(self, running: bool) -> None:
        if self.page is not None:
            self.page.set_running(running)
        if self.browser is not None:
            self.browser.set_empty_text(RUNNING_TEXT if running
                                        else EMPTY_TEXT)
        self.stateChanged.emit()

    @Slot()
    def _on_page_changed(self) -> None:
        self.stateChanged.emit()

    @Slot(str)
    def _relay_status(self, text: str) -> None:
        """A panel's own status line (a figure saved from a plot's menu, an
        export written) goes to the window's status bar."""
        if not self._closing:
            self.statusMessage.emit(str(text))

    # -- results -------------------------------------------------------------
    def _show_result(self, result) -> None:
        if self.browser is None:
            return
        self.browser.set_result(result)
        self.notes_text.setPlainText("\n".join(self.notes_lines(result)))
        self.tabs.setCurrentWidget(self.results_page)
        self.show_canvas("figure")

    def notes_lines(self, result=None) -> list[str]:
        """Every note of the run and the model: the run header, each
        analysis's provenance, notes and reason, the timings, then the
        load summary."""
        from ..core import md_export

        result = self.result if result is None else result
        lines: list[str] = []
        if result is not None:
            lines.append("== the run ==")
            if result.cancelled:
                lines.append("cancelled: the analyses hold the frames they "
                             "received")
            lines.extend(md_export.header_lines(result))
            lines.extend(f"note: {n}" for n in result.notes)
            for name, output in result.outputs.items():
                lines.append("")
                lines.append(f"== {name} ==")
                if output.error:
                    lines.append(f"not computed: {output.error}")
                section = [f"note: {n}" for n in output.notes]
                if output.provenance is not None:
                    section.extend(output.provenance.as_lines())
                lines.extend(dict.fromkeys(section))
                lines.append(f"time: {output.seconds:.3g} s")
            if result.timings_s:
                lines.append("")
                lines.append("== timings ==")
                lines.extend(f"{k}: {v:.3g} s"
                             for k, v in result.timings_s.items())
            lines.append("")
        fv = self.frame_view
        if fv is not None:
            lines.append("== the frame shown ==")
            lines.append(f"frame {fv.k}: {fv.note}")
            if self.highlight is not None and self.highlight.result:
                lines.append(f"highlight: {self.highlight.result.note}")
            lines.append("")
        if self.summary is not None:
            lines.append("== the model, as read ==")
            lines.extend(self.summary.describe_lines)
        return lines

    def _show_model_notes(self) -> None:
        if self.notes_text is not None:
            self.notes_text.setPlainText("\n".join(self.notes_lines(None)))

    @Slot(object, object)
    def _on_descriptor(self, item, _previous=None) -> None:
        if item is not None and self.result is not None and \
                self.browser is not None and self.tabs is not None:
            self.show_canvas("figure")
        self.stateChanged.emit()

    @Slot(int)
    def _on_tab(self, index: int) -> None:
        if self.tabs is None:
            return
        page = self.tabs.widget(index)
        if page is self.results_page and self.result is not None:
            self.show_canvas("figure")
        else:
            self.show_canvas("3d")

    @Slot(str)
    def show_canvas(self, which: str) -> None:
        """The centre: '3d' (the frame) or 'figure' (the descriptor chosen
        in Results, when a run is on show)."""
        if self.trajectory is None:
            return
        figure = which == "figure" and self.figure_page is not None \
            and self.result is not None
        self.canvas.setCurrentWidget(self.figure_page if figure else self.view)
        self.toolbar.set_canvas("figure" if figure else "3d")
        self.stateChanged.emit()           # Save the figure / rows shown

    def canvas_shown(self) -> str:
        """'3d', 'figure', 'reading' or 'read-options'."""
        current = self.canvas.currentWidget()
        if current is self.view:
            return "3d"
        if current is self.figure_page:
            return "figure"
        if current is self.read_page:
            return "read-options"
        return "reading"

    @Slot()
    def show_highlight(self) -> None:
        if self.tabs is not None:
            self.tabs.setCurrentWidget(self.highlight_scroll)

    def show_tab(self, name: str) -> None:
        """Bring up a tab, and put the centre where that tab belongs.

        The canvas rule is applied even when the tab asked for is already the
        current one: choosing a descriptor from the Results tree moves the
        centre to the figure wherever the tabs stand, so without this, asking
        again for the tab that shows the frame would leave the figure there.
        """
        if self.tabs is None:
            return
        for i in range(self.tabs.count()):
            if self.tabs.tabText(i) == name:
                if self.tabs.currentIndex() == i:
                    self._on_tab(i)
                else:
                    self.tabs.setCurrentIndex(i)
                return

    def current_tab(self) -> str:
        if self.tabs is None:
            return ""
        return self.tabs.tabText(self.tabs.currentIndex())

    # -- the setup's fields ------------------------------------------------------
    @Slot(str)
    def _on_field_requested(self, where: str) -> None:
        """A reason link or a double-click asks for a field: the Setup tab
        comes forward; a field the page does not show is revealed in a
        small dialog holding its option group."""
        if self.page is None:
            return
        self.show_tab("Setup")
        if where in ("frames", "formers", "v_bond_vu", "timestep_fs",
                     "frame_interval_ps", "params", "v_list_vu"):
            return
        group, _, name = where.rpartition(".")
        group = group or None
        panel = self.page.panel
        module = next((m for m in panel.groups.values()
                       if (group, name) in m.fields
                       or where in getattr(m, "extra", {})), None)
        if module is None:
            self.statusMessage.emit(f"No field named {where} in the setup")
            return
        for dialog in self._field_dialogs:
            try:
                if dialog.group is module and dialog.isVisible():
                    dialog.raise_()
                    panel.focus_field(where)
                    return
            except RuntimeError:
                continue
        self._field_dialogs = [d for d in self._field_dialogs
                               if _alive(d) and d.isVisible()]
        dialog = _FieldDialog(module, panel, where,
                              self.window if isinstance(self.window, QWidget)
                              else None)
        self._field_dialogs.append(dialog)
        dialog.show()

    # -- the frame ---------------------------------------------------------------
    def painter_tier(self) -> bool:
        """True on the QPainter tier, or while the view has not said."""
        from ..gl import caps as caps_mod

        caps = getattr(self.view, "caps", None)
        return caps is None or caps.tier is caps_mod.Tier.BASIC \
            or getattr(self.view, "_fallback", None) is not None

    def with_bonds(self) -> bool:
        n = int(self.summary.n_atoms) if self.summary is not None else 0
        return not (self.painter_tier() and n > ATOMS_ONLY_ABOVE)

    def renderer_text(self) -> str:
        caps = getattr(self.view, "caps", None)
        return caps.describe() if caps is not None else "not initialised"

    def _first_frame(self) -> None:
        if self.summary is None or self._closing:
            return
        self.show_frame(int(self.entry.frame_k or 0))

    @Slot(int)
    def _on_frame_box(self, k: int) -> None:
        self.show_frame(int(k), self._strip_v_bond())

    def step_frame(self, by: int) -> None:
        self.toolbar.step(by)

    def go_to_frame(self, k: int) -> None:
        if self.summary is None:
            return
        k = min(max(int(k), 0), int(self.summary.n_frames) - 1)
        if k != self.toolbar.frame_box.value():
            self.toolbar.frame_box.setValue(k)        # emits frameRequested
        else:
            self.show_frame(k, self._strip_v_bond())

    def _strip_v_bond(self) -> float | None:
        return float(self.strip.v_bond) if self.strip.table is not None \
            else None

    def show_frame(self, k: int, v_bond_vu: float | None = None) -> None:
        """Load frame ``k`` (and its bonds, unless only the atoms are drawn)
        on a worker thread, then draw it and hand it to the strip."""
        if self.trajectory is None or self.page is None:
            return
        if self._frame_job is not None and self._frame_job.pending:
            self._pending_frame = (k, v_bond_vu)
            return
        from ..core import bv

        page = self.page
        overrides = page.panel.oxidation.overrides()
        if page.panel.oxidation.problems():
            self.toolbar.set_frame(k, "The frame is drawn once every element "
                                      "has an oxidation state (Setup > "
                                      "More…).")
            return
        params = None
        try:
            params = page.panel.params_field.value()
        except ValueError:
            params = None
        if params is page.panel.params_field.UNSET:
            params = None
        v_bond = float(v_bond_vu if v_bond_vu is not None else
                       page.v_bond.value() if page.v_bond is not None
                       else bv.V_BOND_DEFAULT)
        v_list = float(page.v_list.value())
        with_bonds = self.with_bonds()
        note = "" if with_bonds else (
            f"{self.summary.n_atoms} atoms on the QPainter renderer (no "
            f"OpenGL): above {ATOMS_ONLY_ABOVE} atoms the atoms are drawn "
            "without bonds; the OpenGL tiers draw the bonds")
        trajectory = self.trajectory
        theme = self.theme.copy() if hasattr(self.theme, "copy") \
            else self.theme

        def work(progress, cancelled):
            return md_jobs.load_frame_view(trajectory, k, overrides, params,
                                           v_bond, v_list,
                                           with_bonds=with_bonds,
                                           theme=theme, atoms_only_note=note)

        job = md_jobs.Job(work, name="frame")
        job.generation = self._model_generation
        self._connect(job, succeeded=self._on_frame_loaded,
                      failed=self._on_frame_failed,
                      ended=self._on_frame_ended)
        self._frame_job = job
        self.toolbar.set_frame(k, f"loading frame {k} …")
        self.statusMessage.emit(f"Loading frame {k} …")
        job.start()

    @Slot(object)
    def _on_frame_loaded(self, fv) -> None:
        if self._closing or self.page is None:
            return
        job = self.sender()
        if getattr(job, "generation", self._model_generation) != \
                self._model_generation:
            return                  # a frame of the model as read before
        self.frame_view = fv
        self.frame_data = None
        self.entry.frame_view = fv
        self.entry.frame_k = int(fv.k)
        timestep = self.summary.timesteps[fv.k] if self.summary is not None \
            and fv.k < len(self.summary.timesteps) else None
        from ..core.md_model import NO_TIMESTEP

        parts = []
        if timestep is not None and int(timestep) != NO_TIMESTEP:
            parts.append(f"timestep {_thousands(int(timestep))}")
        if fv.n_bonds is not None:
            parts.append(f"{_thousands(fv.n_bonds)} bonds")
        else:
            parts.append("atoms only")
        self.toolbar.set_frame(fv.k, " · ".join(parts),
                               f"frame {fv.k}: {fv.note} ({fv.seconds:.2f} s)")
        self._frame_to_strip()
        self.redraw()
        self.frameShown.emit(int(fv.k))

    def _frame_to_strip(self) -> None:
        """Hand the frame on show to the strip: with its valence table (no
        second search), the threshold it was drawn at, and the threshold of
        the last run, so the strip can name both."""
        fv = self.frame_view
        if fv is None or fv.table is None:
            if fv is not None:
                self.strip.clear()
                self.elements.clear()
            return
        run_v = float(self.result.request.v_bond_vu) \
            if self.result is not None else None
        # the strip keeps its position across frames and runs; a frame
        # loaded while it holds none starts at the threshold it was drawn at
        start = self._strip_v_bond()
        start = fv.v_bond_vu if start is None else start
        try:
            self.strip.set_frame(fv.frame, fv.ox_atom, fv.params,
                                 v_list_vu=fv.v_list_vu, v_bond_vu=start,
                                 analysis_v_bond_vu=run_v,
                                 label=f"frame {fv.k}", table=fv.table)
        except Exception as error:         # noqa: BLE001 - shown, not raised
            self.statusMessage.emit(f"The threshold strip could not take "
                                    f"frame {fv.k}: {error}")
            return
        self._fill_elements()

    def _fill_elements(self) -> None:
        fv = self.frame_view
        if fv is None or self.strip.results is None:
            return
        symbols = np.asarray(fv.frame.elements)
        states = {}
        for symbol in fv.frame.species:
            rows = symbols == symbol
            if rows.any():
                states[str(symbol)] = int(np.asarray(fv.ox_atom)[rows][0])
        self.elements.set_rows(self.strip.element_rows(), states,
                               float(self.strip.v_bond))

    @Slot(object)
    def _on_frame_failed(self, failure) -> None:
        if self._closing:
            return
        self.toolbar.set_frame(self.toolbar.frame_box.value(),
                               f"The frame was not loaded: {failure.message}")
        self.statusMessage.emit(f"The frame was not loaded: {failure.message}")

    @Slot()
    def _on_frame_ended(self) -> None:
        if self._closing:
            return
        pending, self._pending_frame = self._pending_frame, None
        if pending is not None:
            self.show_frame(*pending)

    @Slot(object)
    def _on_view_ready(self, caps) -> None:
        """The view knows its tier now: a frame drawn without bonds while
        the tier was unknown is drawn again with them when the tier allows."""
        fv = self.frame_view
        if fv is not None and fv.bonds is None and self.with_bonds():
            self.show_frame(fv.k, fv.v_bond_vu)
        self.statusMessage.emit(self.status_line())
        self.stateChanged.emit()

    @Slot(float)
    def _on_strip_threshold(self, v_bond_vu: float) -> None:
        """The strip moved v_bond: the drawn bonds restyle at once (no
        search), the Elements rows follow, and the rules are applied again
        after a short delay."""
        fv = self.frame_view
        if fv is not None and fv.table is not None:
            self.view.set_threshold(float(v_bond_vu))
            self._fill_elements()
            self._schedule_redraw()

    @Slot(str)
    def _on_element_chosen(self, symbol: str) -> None:
        self.element = str(symbol or "")
        self._schedule_redraw()

    @Slot(object)
    def _on_labels(self, settings) -> None:
        self.view.set_labels(settings)

    @Slot(bool)
    def _on_orthographic(self, on: bool) -> None:
        self.view.set_projection(bool(on))

    # -- the highlight rules, applied on a worker ----------------------------------
    @Slot()
    def _schedule_redraw(self) -> None:
        if self.frame_view is not None:
            self._redraw_timer.start()

    def _frame_data_at(self, v_bond: float):
        """The FrameData of the frame shown at ``v_bond`` (the strip's),
        with the frame's cached spheres and landscapes carried over from
        the one built before at another threshold."""
        from .md_highlight import FrameData

        fv = self.frame_view
        if fv is None or fv.table is None:
            # a frame drawn without bonds carries no valence table, so there
            # is no FrameData to build; redraw() says so to the user before
            # it gets here, and a caller reaching this directly gets None
            # rather than FrameData.from_frame's ValueError
            return None
        formers = self.page.formers() if self.page is not None else frozenset()
        key = (id(fv), round(float(v_bond), 9), formers)
        if self.frame_data is not None and self._frame_data_key == key:
            return self.frame_data
        glass = self.result.provenance if self.result is not None else None
        fd = FrameData.from_frame(fv.frame, fv.table, k=fv.k, v_bond=v_bond,
                                  formers=formers, ox_atom=fv.ox_atom,
                                  params=fv.params, glass=glass)
        old = self.frame_data
        if old is not None and old.frame is fv.frame:
            fd._spheres = old._spheres
            fd._landscapes = old._landscapes
            fd._void_regions = old._void_regions
        self.frame_data = fd
        self._frame_data_key = key
        return fd

    @Slot()
    def redraw(self) -> None:
        """Build the frame's scene with the rules applied (a worker thread;
        the void spheres and the channel landscapes take seconds) and hand
        it to the view."""
        fv = self.frame_view
        if fv is None or fv.scene is None or self._closing:
            return
        if fv.table is None:
            # atoms only: no table to read the rules from
            if self.highlight is not None:
                self.highlight.set_frame_data(None)
                self.highlight.summary.setText(
                    "The frame is drawn without bonds (QPainter renderer "
                    f"above {ATOMS_ONLY_ABOVE} atoms): the rules need its "
                    "bonds.")
            self.view.set_scene(fv.scene, reframe=self._reframe_next)
            self._reframe_next = False
            self.last_draw_s = fv.seconds
            self.statusMessage.emit(self.status_line())
            return
        if self._highlight_job is not None and self._highlight_job.pending:
            self._pending_redraw = True
            return
        v_bond = self._strip_v_bond()
        v_bond = fv.v_bond_vu if v_bond is None else v_bond
        fd = self._frame_data_at(v_bond)
        if self.highlight is not None:
            self.highlight.set_frame_data(fd)
            rules = self.highlight.rules()
            colour_by = self.highlight.colour_by()
            dim_rest = self.highlight.dim_rest()
        else:
            rules, colour_by, dim_rest = [], "element", True
        element = self.element
        style = self.toolbar.style()
        show_cell = self.toolbar.box_check.isChecked()
        theme = self.theme.copy() if hasattr(self.theme, "copy") \
            else self.theme
        frame, table, params, v_list = fv.frame, fv.table, fv.params, \
            fv.v_list_vu
        active = [r for r in rules if r.enabled]

        def work(progress, cancelled):
            from .md_highlight import _dim, apply_rules
            from .md_scene import build_md_scene

            clock = time.perf_counter()
            scene = build_md_scene(frame, table, theme=theme, params=params,
                                   v_bond=v_bond, v_list=v_list, style=style,
                                   show_cell=show_cell)
            outcome = apply_rules(scene, fd, rules, dim_rest=dim_rest,
                                  colour_by=colour_by, theme=theme)
            if element:
                keep = np.asarray(fd.elements) == element
                _dim(scene, keep, theme)
            return scene, outcome, time.perf_counter() - clock

        job = md_jobs.Job(work, name="highlight")
        job.generation = self._model_generation
        self._connect(job, succeeded=self._on_redrawn,
                      failed=self._on_redraw_failed,
                      ended=self._on_redraw_ended)
        self._highlight_job = job
        if active:
            self.statusMessage.emit(
                f"Applying {len(active)} highlight rule(s) to frame {fv.k} …")
        job.start()

    @Slot(object)
    def _on_redrawn(self, payload) -> None:
        if self._closing:
            return
        job = self.sender()
        if getattr(job, "generation", self._model_generation) != \
                self._model_generation:
            return
        scene, outcome, seconds = payload
        self.view.set_scene(scene, reframe=self._reframe_next)
        self._reframe_next = False
        self.last_draw_s = float(seconds)
        if self.highlight is not None:
            self.highlight.show_result(outcome)
        self.statusMessage.emit(self.status_line())

    @Slot(object)
    def _on_redraw_failed(self, failure) -> None:
        if self._closing:
            return
        fv = self.frame_view
        if fv is not None and fv.scene is not None:
            self.view.set_scene(fv.scene, reframe=self._reframe_next)
            self._reframe_next = False
        if self.highlight is not None:
            self.highlight.summary.setText(f"The rules were not applied: "
                                           f"{failure.message}")
        self.statusMessage.emit(f"The rules were not applied: "
                                f"{failure.message}")

    @Slot()
    def _on_redraw_ended(self) -> None:
        if self._closing:
            return
        if self._pending_redraw:
            self._pending_redraw = False
            self.redraw()

    def status_line(self) -> str:
        """The status bar's line for this model: the renderer tier and the
        frame's draw time, atoms and bonds at v_bond."""
        fv = self.frame_view
        parts = [f"Renderer: {self.renderer_text()}"]
        if fv is None:
            if self.summary is not None:
                parts.append(f"{self.summary.label}: no frame drawn yet")
            return " · ".join(parts)
        seconds = fv.seconds + self.last_draw_s
        parts.append(f"frame {fv.k} drawn in {seconds:.2f} s from the bulk "
                     "engine's table")
        v = self._strip_v_bond()
        v = fv.v_bond_vu if v is None else v
        atoms = _thousands(fv.frame.n_atoms)
        if fv.table is not None and self.strip.results is not None:
            bonds = self.strip.current_bonds()
            n_bonds = len(bonds) if bonds is not None else fv.n_bonds
            parts.append(f"{atoms} atoms, {_thousands(n_bonds)} cation-anion "
                         f"bonds at v_bond {v:g} v.u.")
        else:
            parts.append(f"{atoms} atoms, drawn without bonds")
        return " · ".join(parts)

    # -- export --------------------------------------------------------------
    def export_results(self, path) -> md_jobs.Job | None:
        """Write the last result: a path ending in .xlsx is one workbook,
        any other path a directory of CSV files (md_export.export). Runs on
        a worker thread; returns the job (None when there is no result)."""
        if self.result is None:
            self.statusMessage.emit("Nothing to export: no run has finished")
            return None
        if self._export_job is not None and self._export_job.pending:
            self.statusMessage.emit("An export is in progress")
            return None
        from ..core import md_export

        result = self.result
        target = str(path)

        def work(progress, cancelled):
            return md_export.export(result, target)

        job = md_jobs.Job(work, name="export")
        self._connect(job, succeeded=self._on_export_done,
                      failed=self._on_export_failed)
        self._export_job = job
        self.statusMessage.emit(f"Writing {target} …")
        job.start()
        return job

    @Slot(object)
    def _on_export_done(self, written) -> None:
        if self._closing:
            return
        paths = list(written or [])
        where = paths[0].parent if len(paths) > 1 else (paths[0] if paths
                                                         else "")
        self.statusMessage.emit(f"Wrote {len(paths)} file(s) to {where}")

    @Slot(object)
    def _on_export_failed(self, failure) -> None:
        if self._closing:
            return
        self.statusMessage.emit(f"The export stopped: {failure.message}")

    def _dialog_parent(self):
        return self.window if isinstance(self.window, QWidget) else None

    @Slot()
    def choose_export_csv(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self._dialog_parent(),
            "Export results as CSV files: choose a folder")
        if folder:
            stem = Path(md_jobs.source_label(self.source)).stem
            self.export_results(Path(folder) / f"{stem}_facet_md")

    @Slot()
    def choose_export_xlsx(self) -> None:
        stem = Path(md_jobs.source_label(self.source)).stem
        path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(), "Export results as an XLSX workbook",
            f"{stem}_facet_md.xlsx", "Excel workbook (*.xlsx)")
        if path:
            if not path.lower().endswith(".xlsx"):
                path += ".xlsx"
            self.export_results(path)

    @Slot()
    def save_figure_shown(self) -> None:
        if self.browser is None:
            self.statusMessage.emit("No model is read yet")
            return
        self.browser.choose_export_figure()

    @Slot()
    def save_rows_shown(self) -> None:
        if self.browser is None:
            self.statusMessage.emit("No model is read yet")
            return
        self.browser.choose_export_rows()

    def figure_shown(self) -> bool:
        return (self.browser is not None and self.result is not None
                and self.canvas_shown() == "figure"
                and self.browser.export_figure_action.isEnabled())

    def rows_shown(self) -> bool:
        return (self.browser is not None and self.result is not None
                and self.canvas_shown() == "figure"
                and self.browser.export_rows_action.isEnabled())

    # -- the request as a file ----------------------------------------------
    def request_spec(self) -> dict:
        """The setup as a request file holds it (``SetupPanel.
        request_spec``); ValueError while no model is read."""
        if self.page is None:
            raise ValueError("no model is read yet")
        spec = self.page.panel.request_spec()
        self.entry.request_spec = spec
        return spec

    def save_request(self, path) -> Path:
        """Write the setup as a request file (TOML) that ``py -3.11 -m
        facet.md analyse FILE --request`` reads; the first line names the
        command. ValueError while there is no setup (no model read)."""
        from .md_dialogs import request_toml

        spec = self.request_spec()
        source = self.source[0] if isinstance(self.source, list) \
            else str(self.source)
        text = request_toml(spec, source=source)
        target = Path(path)
        target.write_text(text, encoding="utf-8")
        return target

    @Slot()
    def choose_save_request(self) -> None:
        if self.page is None:
            self.statusMessage.emit("No model is read yet: no request to save")
            return
        stem = Path(md_jobs.source_label(self.source)).stem
        path, _ = QFileDialog.getSaveFileName(
            self._dialog_parent(), "Save the request for the command line",
            f"{stem}_request.toml", "Request file (*.toml)")
        if not path:
            return
        if not path.lower().endswith(".toml"):
            path += ".toml"
        try:
            written = self.save_request(path)
        except (OSError, ValueError) as error:
            self.statusMessage.emit(f"The request was not written: {error}")
        else:
            self.statusMessage.emit(
                f"Wrote {written}: py -3.11 -m facet.md analyse <the model "
                f"file> --request {written.name} repeats the run")

    def load_request(self, path) -> dict:
        """Fill the setup from a request file (TOML or JSON) as the command
        line reads it (``facet.md.cli.request_from_mapping`` checks it
        first; ValueError names what it refuses). Returns the mapping."""
        from ..md import cli

        try:
            spec = cli._load_mapping(str(path), "the request file")
        except Exception as error:         # noqa: BLE001 - said, not raised
            raise ValueError(f"{Path(str(path)).name}: {error}") from None
        checked = {k: v for k, v in spec.items()
                   if k not in ("type_map", "read_options")}
        try:
            cli.request_from_mapping(checked)
        except (cli.UsageError, ValueError) as error:
            raise ValueError(f"{Path(str(path)).name}: {error}") from None
        self.apply_request_spec(spec)
        return spec

    def apply_request_spec(self, spec: Mapping) -> None:
        """Put a request mapping into the Setup page: the analyses ticked,
        the formers, frames, thresholds, oxidation states and every option
        field named, as text (the page checks each as it is typed). Keys
        the page has no field for (a type map, read options, measured
        curves) are left to the status bar."""
        from ..core import md_analysis as ma

        if self.page is None:
            raise ValueError("no model is read yet")
        page = self.page
        panel = page.panel
        skipped: list[str] = []
        self.page.preset.blockSignals(True)
        try:
            for key, value in spec.items():
                if key in ("analyses",):
                    continue
                if key == "formers":
                    if value == ma.NO_FORMERS:
                        page.none_box.setChecked(True)
                    elif isinstance(value, (list, tuple, set, frozenset)):
                        page.none_box.setChecked(False)
                        panel.formers.set_formers(set(str(s) for s in value))
                    continue
                if key == "frames":
                    first, last, stride = _frames_of(value,
                                                     self.summary.n_frames)
                    page.frame_range.set_range(first, last, stride)
                    continue
                if key in ("v_bond_vu", "v_bond"):
                    page.v_bond.setValue(float(value))
                    continue
                if key in ("v_list_vu", "v_list"):
                    page.v_list.setValue(float(value))
                    continue
                if key in ("ox", "ox_overrides"):
                    for symbol, state in dict(value).items():
                        try:
                            panel.oxidation.set_state(str(symbol), int(state))
                        except (KeyError, ValueError):
                            skipped.append(f"ox.{symbol}")
                    continue
                if key in ("type_map", "read_options"):
                    continue
                if isinstance(value, Mapping):
                    for name, item in value.items():
                        try:
                            panel.field(key, name).set_text(_field_text(item))
                        except KeyError:
                            skipped.append(f"{key}.{name}")
                    continue
                try:
                    panel.field(None, key).set_text(_field_text(value))
                except KeyError:
                    skipped.append(key)
            names = [str(a) for a in spec.get("analyses", ())]
            panel.set_analyses(names)
        finally:
            self.page.preset.blockSignals(False)
        page.preset.setCurrentText("Custom")
        panel.refresh()
        if skipped:
            self.statusMessage.emit("Request fields with no setup field, "
                                    "left as they were: " + ", ".join(skipped))
        self.stateChanged.emit()

    @Slot()
    def choose_load_request(self) -> None:
        if self.page is None:
            self.statusMessage.emit("No model is read yet: a request fills "
                                    "the setup of a model")
            return
        path, _ = QFileDialog.getOpenFileName(
            self._dialog_parent(), "Load a request file", "",
            "Request file (*.toml *.json);;All files (*)")
        if not path:
            return
        try:
            self.load_request(path)
        except ValueError as error:
            self.statusMessage.emit(f"The request was not loaded: {error}")
        else:
            self.statusMessage.emit(f"Setup filled from {Path(path).name}")

    # -- theme ---------------------------------------------------------------
    def apply_theme(self, theme) -> None:
        """Adopt ``theme`` in this model's views; the frame is drawn again
        with it."""
        self.theme = theme
        self.view.set_theme(theme)
        self.toolbar.fit_combos()
        for widget in (self.page, self.highlight, self.browser, self.strip,
                       self.read_panel):
            if widget is None:
                continue
            for name in ("apply_theme", "set_theme"):
                method = getattr(widget, name, None)
                if method is not None:
                    method(theme)
                    break
        if self.frame_view is not None:
            self.show_frame(self.frame_view.k, self._strip_v_bond())

    # -- jobs and closing ------------------------------------------------------
    def _connect(self, job, **slots) -> None:
        links = []
        for signal_name, slot in slots.items():
            signal = getattr(job, signal_name)
            signal.connect(slot)
            links.append((signal, slot))
        job.ended.connect(self._forget_job)
        links.append((job.ended, self._forget_job))
        self._links[job] = links

    @Slot()
    def _forget_job(self) -> None:
        job = self.sender()
        if job is not None:
            self._links.pop(job, None)

    def _jobs(self) -> list:
        return [j for j in (self._open_job, self._run_job, self._frame_job,
                            self._highlight_job, self._export_job)
                if j is not None and not j.done]

    def _busy(self) -> bool:
        """True while any job of this model has not delivered its result."""
        return any(j is not None and j.pending for j in (
            self._open_job, self._run_job, self._frame_job,
            self._highlight_job, self._export_job))

    @property
    def closing(self) -> bool:
        return self._closing

    def shutdown(self, wait_ms: int | None = None) -> bool:
        """Cancel this model's work and wait at most ``wait_ms``
        (:data:`CLOSE_WAIT_MS`) for it; work that takes longer finishes in
        the background, held by md_jobs. Returns True when nothing is left
        running. Called when the entry is removed or the window closes."""
        if self._closing:
            return not self._jobs()
        self._closing = True
        self._redraw_timer.stop()
        wait_ms = CLOSE_WAIT_MS if wait_ms is None else int(wait_ms)
        jobs = self._jobs()
        for job in jobs:
            job.cancel()
        deadline = time.perf_counter() + wait_ms / 1000.0

        def left_ms() -> int:
            return int(max(0.0, deadline - time.perf_counter()) * 1000)

        for job in jobs:
            job.wait(left_ms())
        for panel in (self.browser, self.strip):
            shutdown = getattr(panel, "shutdown", None)
            if shutdown is not None:
                try:
                    shutdown(left_ms())
                except Exception:          # noqa: BLE001 - closing regardless
                    pass
        for job, links in list(self._links.items()):
            for signal, slot in links:
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self._links.clear()
        for dialog in self._field_dialogs:
            try:
                dialog.close()
            except RuntimeError:
                pass
        self._field_dialogs = []
        still = [j for j in jobs if j.is_running()]
        if not still and self.trajectory is not None:
            close = getattr(self.trajectory, "close", None)
            if close is not None:
                close()
        return not still

    def dispose(self) -> None:
        """Free the widgets (after :meth:`shutdown`, and once the window
        has taken them out of its stacks)."""
        for widget in self.widgets:
            try:
                widget.setParent(None)
                widget.deleteLater()
            except RuntimeError:
                pass
        if self.entry.controller is self:
            self.entry.controller = None
        self.deleteLater()


def _alive(obj) -> bool:
    try:
        import shiboken6

        return bool(shiboken6.isValid(obj))
    except Exception:              # noqa: BLE001 - not a Qt object
        return obj is not None


def _frames_of(value, n_frames: int) -> tuple[int, int, int]:
    """(first, last, stride) of a request's frames: 'a:b:c' text, a
    slice, or a list of indices (its first, last and common step)."""
    n = max(int(n_frames), 1)
    if isinstance(value, slice):
        start = 0 if value.start is None else int(value.start)
        stop = n if value.stop is None else int(value.stop)
        step = 1 if value.step is None else int(value.step)
        return start, max(start, stop - 1), max(step, 1)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(":")]
        start = int(parts[0]) if parts and parts[0] else 0
        stop = int(parts[1]) if len(parts) > 1 and parts[1] else n
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
        return start, max(start, stop - 1), max(step, 1)
    indices = sorted(int(k) for k in value)
    if not indices:
        return 0, n - 1, 1
    steps = {b - a for a, b in zip(indices, indices[1:])}
    step = steps.pop() if len(steps) == 1 else 1
    return indices[0], indices[-1], max(step, 1)


def _field_text(value) -> str:
    """A request value as the text an option field takes."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(_field_text(v) for v in value)
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_field_text(v)}" for k, v in value.items())
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:g}"
    return str(value)
